//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection_cc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! variables in the orbital advection step. Data is shifted by the appropriate offset
//! during the recv/unpack step, so these functions both communicate the data and perform
//! the shift.

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/cell_locations.hpp"
#include "shearing_box.hpp"
#include "orbital_advection.hpp"
#include "mhd/mhd.hpp"
#include "remap_fluxes.hpp"
#include "mesh/prolongation.hpp"   // PROLONG_CC_SLOPE

//----------------------------------------------------------------------------------------
// OrbitalAdvectionCC derived class constructor:

OrbitalAdvectionCC::OrbitalAdvectionCC(MeshBlockPack *pp, ParameterInput *pin, int nv) :
    OrbitalAdvection(pp, pin) {
  // Initialize boundary buffers
  int nmb = std::max((pp->nmb_thispack), (pp->pmesh->nmb_maxperrank));
  auto &indcs = pp->pmesh->mb_indcs;
  int ncells3 = indcs.nx3;
  int ncells2 = indcs.ng + maxjshift;
  int ncells1 = indcs.nx1;
  for (int n=0; n<2; ++n) {
    Kokkos::realloc(sendbuf[n].vars,nmb,nv,ncells3,ncells2,ncells1);
    Kokkos::realloc(recvbuf[n].vars,nmb,nv,ncells3,ncells2,ncells1);
  }
  // x2 faces with a level change
  nvar_oa = nv;
  BuildMixedLists();
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::PackAndSendCC()
//! \brief Pack cell-centered variables into boundary buffers and send to neighbors for
//! the orbital advection step.  Only ghost zones on the x2-faces (Y-faces) are passed.
//! Communication of coarse arrays is not needed since variable resolution in x2 is not
//! allowed in this shearing box implementation.
//!
//! Input arrays must be 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus OrbitalAdvectionCC::PackAndSendCC(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR

  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  auto &maxjshift_ = maxjshift;
  auto &xlev_ = xlev;

  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*2*nvar;  // only consider 2 neighbors (x2-faces)
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("oa-pack", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(2*nvar);
    const int n = (tmember.league_rank() - m*(2*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(2*nvar) - n*nvar);

    // indices of x2-face buffers in nghbr view
    int nnghbr;
    if (n==0) {nnghbr=8;} else {nnghbr=12;}

    // only load buffers when neighbor exists (and is on the same level)
    if (nghbr.d_view(m,nnghbr).gid >= 0 && xlev_.d_view(m,n) == 0) {
      // neighbor must always be at same level, so use same indices to pack buffer
      // Note j-range of indices extended by shear
      int il = is;
      int iu = ie;
      int jl, ju;
      if (n==0) {
        jl = js;
        ju = js + (ng + maxjshift_ - 1);;
      } else {
        jl = je - (ng + maxjshift_ - 1);
        ju = je;
      }
      int kl = ks;
      int ku = ke;
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nji = nj*ni;
      int nkji = nk*nj*ni;

      // index of recv'ing (destination) MB and buffer [0,1]: MB IDs are stored
      // sequentially in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,nnghbr).gid - mbgid.d_view(0);
      int dn = (n+1) % 2;

      // Middle loop over k,j,i
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkji), [&](const int idx) {
        int k = (idx)/nji;
        int j = (idx - k*nji)/ni;
        int i = (idx - k*nji - j*ni) + il;
        k += kl;
        j += jl;

        // copy directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,nnghbr).rank == my_rank) {
          rbuf[dn].vars(dm,v,(k-kl),(j-jl),(i-il)) = a(m,v,k,j,i);

        // else copy into send buffer for MPI communication below
        } else {
          sbuf[n].vars(m,v,(k-kl),(j-jl),(i-il)) = a(m,v,k,j,i);
        }
      });
    } // end if-neighbor-exists block
  }); // end par_for_outer

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if (nghbr.h_view(m,nnghbr).gid >= 0 && xlev.h_view(m,n) == 0) {
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,nnghbr).dest;
        int drank = nghbr.h_view(m,nnghbr).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,nnghbr).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);

          // get ptr to send buffer when neighbor is at coarser/same/fine level
          using Kokkos::ALL;
          auto send_ptr = Kokkos::subview(sbuf[n].vars, m, ALL, ALL, ALL, ALL);
          int data_size = send_ptr.size();

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_orb_advect, &(sbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  // rows for x2 neighbours on another level
  if (any_mixed) PackMixedCC(a);
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \!fn void OrbitalAdvectionCC::RecvAndUnpackCC()
//! \brief Receive and unpack boundary buffers for CC variables with orbital advection,
//! and apply shift in x2- (y-) direction across entire MeshBlock by applying both an
//! integer shift and a fractional offset to input array a.

TaskStatus OrbitalAdvectionCC::RecvAndUnpackCC(DvceArray5D<Real> &a,
                                               ReconstructionMethod rcon) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  auto &nghbr = pmy_pack->pmb->nghbr;
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if (nghbr.h_view(m,nnghbr).gid >= 0 && xlev.h_view(m,n) == 0) {
        if (nghbr.h_view(m,nnghbr).rank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[m]), &test, MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {
            bflag = true;
          }
        }
      }
    }
  }
  // the mixed-level rows (null requests test complete)
  if (any_mixed) {
    for (int n=0; n<2; ++n) {
      for (int r=0; r<4*nmb; ++r) {
        int test;
        if (MPI_Test(&(mrreq[n][r]), &test, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
          no_errors = false;
        }
        if (!(static_cast<bool>(test))) bflag = true;
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (bflag) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: buffers have all completed, so unpack and apply shift

  int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L of in array must be NVAR
  // rows from finer neighbours on other ranks into their quarter of the buffer
  if (any_mixed) UnpackMixedCC();

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &is = indcs.is, &ie = indcs.ie;
  auto &js = indcs.js, &je = indcs.je;
  auto &ks = indcs.ks, &ke = indcs.ke;
  auto &ng = indcs.ng;
  int jfs = ng + maxjshift;
  int jfe = jfs + indcs.nx2 - 1;
  int nfx = indcs.nx2 + 2*(ng + maxjshift);

  auto &mbsize = pmy_pack->pmb->mb_size;
  Real &dt = pmy_pack->pmesh->dt;
  Real qo = qshear*omega0;

  auto &xlev_ = xlev;
  auto &fown_ = fown;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nfx) * 2;
  par_for_outer("oa-unpk",DevExeSpace(),scr_size,scr_lvl,0,(nmb-1),0,(nvar-1),ks,ke,is,ie,
  KOKKOS_LAMBDA(TeamMember_t member, const int m, const int n, const int k, const int i) {
    ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nfx); // 1D slice of data
    ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nfx); // "flux" at faces

    Real &x1min = mbsize.d_view(m).x1min;
    Real &x1max = mbsize.d_view(m).x1max;
    int nx1 = indcs.nx1;
    Real x1v = CellCenterX(i-is, nx1, x1min, x1max);

    Real yshear = -(qo)*x1v*dt;
    int joffset = static_cast<int>(yshear/(mbsize.d_view(m).dx2));

    // Load scratch array with no shift
    par_for_inner(member, 0, (nfx-1), [&](const int jf) {
      if (jf < jfs) {
        // Load from L boundary buffer
        a_(jf) = rbuf[0].vars(m,n,(k-ks),jf,(i-is));
      } else if (jf <= jfe) {
        // Load from conserved variables themselves (addressed with j=jf-jfs+js)
        a_(jf) = a(m,n,k,jf-jfs+js,i);
      } else {
        // Load from R boundary buffer
        a_(jf) = rbuf[1].vars(m,n,(k-ks),jf-(jfe+1),(i-is));
      }
    });
    member.team_barrier();

    // Compute "fluxes" at shifted cell faces
    Real epsi = fmod(yshear,(mbsize.d_view(m).dx2))/(mbsize.d_view(m).dx2);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
      case ReconstructionMethod::wenoz:
        PPMX_RemapFlx(member, (jfs-joffset), (jfe+1-joffset), epsi, a_, flx);
        break;
      default:
        break;
    }
    member.team_barrier();

    // the remap flux through an x2 face with a level change: the content that
    // crosses face J in +x2 (whole cells of the integer shift plus the fractional flux)
    if (xlev_.d_view(m,0) != 0 || xlev_.d_view(m,1) != 0) {
      Kokkos::single(Kokkos::PerTeam(member), [&]() {
        for (int f=0; f<2; ++f) {
          if (xlev_.d_view(m,f) == 0) continue;
          const int jface = (f == 0) ? jfs : (jfe + 1);
          Real fl = flx(jface - joffset);
          if (joffset > 0) {
            for (int l=jface-joffset; l<jface; ++l) fl += a_(l);
          } else if (joffset < 0) {
            for (int l=jface; l<jface-joffset; ++l) fl -= a_(l);
          }
          fown_(m,n,f,k-ks,i-is) = fl;
        }
      });
    }

    // Update CC variables with both integer shift (from a_) and a conservative remap
    // for the remaining fraction of a cell using upwind "fluxes"
    par_for_inner(member, js, je, [&](const int j) {
      int jf = j-js + jfs;
      a(m,n,k,j,i) = a_(jf-joffset) - (flx(jf+1-joffset) - flx(jf-joffset));
    });
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::PackMixedCC()
//! \brief The rows for x2 neighbours on another level (stage S2), at the
//! receiver's resolution and in the layout of its receive buffer (row r of the lower buffer
//! [0] is adjacent to the face at r = nrow-1, of the upper buffer [1] at r = 0).
//!  * A finer block restricts 2x2x2 cells into its quarter (x1 half fx, x3 half fz) of the
//!    coarser neighbour's buffer.
//!  * A coarser block prolongates into the whole buffer of each of its four finer neighbours,
//!    limited linear in all three directions as ProlongCC (slopes from the block's own cells,
//!    ghost cells included: those of the last boundary exchange).
//! Same rank: written into the receive buffer directly; else into msend and sent.

void OrbitalAdvectionCC::PackMixedCC(DvceArray5D<Real> &a) {
  const int nmb = pmy_pack->nmb_thispack;
  const int nvar = a.extent_int(1);
  const int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, je = indcs.je, ks = indcs.ks;
  const int nk = indcs.nx3, ni = indcs.nx1, nk2 = nk/2, ni2 = ni/2;
  const int R = nrow;
  auto &xlev_ = xlev;
  auto &xslot_ = xslot;
  auto &rbuf = recvbuf;
  for (int n=0; n<2; ++n) {
    const int dn = (n + 1) % 2;
    const int base = (n == 0) ? 8 : 12;
    auto sbm = msend[n];
    // finer block: restriction into its quarter of the coarser neighbour's buffer
    par_for("oa-mix-r", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, nk2-1, 0, R-1, 0, ni2-1,
    KOKKOS_LAMBDA(const int m, const int v, const int kc, const int r, const int ic) {
      if (xlev_.d_view(m,n) != 1) return;
      const int slot = xslot_.d_view(m,n);
      const int sf = slot - base, fx = sf & 1, fz = sf >> 1;
      const int j0 = (n == 0) ? (js + 2*r) : (je - 2*(R - 1 - r) - 1);
      const int i0 = is + 2*ic, k0 = ks + 2*kc;
      const Real val = 0.125*(a(m,v,k0  ,j0  ,i0) + a(m,v,k0  ,j0  ,i0+1)
                            + a(m,v,k0  ,j0+1,i0) + a(m,v,k0  ,j0+1,i0+1)
                            + a(m,v,k0+1,j0  ,i0) + a(m,v,k0+1,j0  ,i0+1)
                            + a(m,v,k0+1,j0+1,i0) + a(m,v,k0+1,j0+1,i0+1));
      if (nghbr.d_view(m,slot).rank == my_rank) {
        const int dm = nghbr.d_view(m,slot).gid - mbgid.d_view(0);
        rbuf[dn].vars(dm, v, fz*nk2 + kc, r, fx*ni2 + ic) = val;
      } else {
        sbm(m, 0, ((v*nk2 + kc)*R + r)*ni2 + ic) = val;
      }
    });
    // coarser block: prolongation into the buffers of its four finer neighbours
    par_for("oa-mix-p", DevExeSpace(), 0, nmb-1, 0, 4*nvar-1, 0, nk-1, 0, R-1, 0, ni-1,
    KOKKOS_LAMBDA(const int m, const int vs, const int kf, const int rf, const int ifn) {
      if (xlev_.d_view(m,n) != 2) return;
      const int sf = vs / nvar, v = vs - sf*nvar;
      const int fx = sf & 1, fz = sf >> 1;
      const int slot = base + sf;
      const int ii = is + fx*ni2 + ifn/2;
      const int kk = ks + fz*nk2 + kf/2;
      const int d = (n == 0) ? rf : (R - 1 - rf);       // fine rows from the face
      const int jj = (n == 0) ? (js + d/2) : (je - d/2);
      // the child next to the face: lower (n=0, face below) or upper (n=1) half of jj
      const Real cy = ((d % 2 == 0) == (n == 0)) ? -1.0 : 1.0;
      const Real cx = (ifn % 2 == 0) ? -1.0 : 1.0;
      const Real cz = (kf % 2 == 0) ? -1.0 : 1.0;
      const Real u0 = a(m,v,kk,jj,ii);
      Real dl = u0 - a(m,v,kk,jj,ii-1), dr = a(m,v,kk,jj,ii+1) - u0;
      const Real dv1 = PROLONG_CC_SLOPE(dl, dr);
      dl = u0 - a(m,v,kk,jj-1,ii); dr = a(m,v,kk,jj+1,ii) - u0;
      const Real dv2 = PROLONG_CC_SLOPE(dl, dr);
      dl = u0 - a(m,v,kk-1,jj,ii); dr = a(m,v,kk+1,jj,ii) - u0;
      const Real dv3 = PROLONG_CC_SLOPE(dl, dr);
      const Real val = u0 + cx*dv1 + cy*dv2 + cz*dv3;
      if (nghbr.d_view(m,slot).rank == my_rank) {
        const int dm = nghbr.d_view(m,slot).gid - mbgid.d_view(0);
        rbuf[dn].vars(dm, v, kf, rf, ifn) = val;
      } else {
        sbm(m, sf, ((v*nk + kf)*R + rf)*ni + ifn) = val;
      }
    });
  }
#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  bool no_errors = true;
  const int scf = nvar*nk*R*ni, sfc = nvar*nk2*R*ni2;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      const int base = (n == 0) ? 8 : 12;
      for (int sf=0; sf<4; ++sf) {
        int slot = -1, size = 0, req = 4*m + sf;
        if (xlev.h_view(m,n) == 1 && sf == 0) {
          slot = xslot.h_view(m,n); size = sfc;
        } else if (xlev.h_view(m,n) == 2) {
          slot = base + sf; size = scf;
        }
        if (slot < 0) continue;
        const int drank = nghbr.h_view(m,slot).rank;
        if (drank == my_rank) continue;
        const int lid = nghbr.h_view(m,slot).gid - pmy_pack->pmesh->gids_eachrank[drank];
        const int tag = CreateBvals_MPI_Tag(lid, nghbr.h_view(m,slot).dest);
        auto ptr = Kokkos::subview(msend[n], m, (xlev.h_view(m,n) == 1) ? 0 : sf, Kokkos::ALL);
        if (MPI_Isend(ptr.data(), size, MPI_ATHENA_REAL, drank, tag, comm_orb_advect,
                      &(msreq[n][req])) != MPI_SUCCESS) {
          no_errors = false;
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting sends (orbital advection)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvectionCC::UnpackMixedCC()
//! \brief The restricted rows that finer neighbours on other ranks sent: into their
//! quarter of the receive buffer

void OrbitalAdvectionCC::UnpackMixedCC() {
#if MPI_PARALLEL_ENABLED
  const int nmb = pmy_pack->nmb_thispack;
  const int nvar = nvar_oa;
  const int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nk2 = indcs.nx3/2, ni2 = indcs.nx1/2;
  const int R = nrow;
  auto &xlev_ = xlev;
  auto &rbuf = recvbuf;
  for (int n=0; n<2; ++n) {
    const int base = (n == 0) ? 8 : 12;
    auto rbm = mrecv[n];
    par_for("oa-mix-u", DevExeSpace(), 0, nmb-1, 0, 4*nvar-1, 0, nk2-1, 0, R-1, 0, ni2-1,
    KOKKOS_LAMBDA(const int m, const int vs, const int kc, const int r, const int ic) {
      if (xlev_.d_view(m,n) != 2) return;
      const int sf = vs / nvar, v = vs - sf*nvar;
      if (nghbr.d_view(m,base+sf).rank == my_rank) return;
      const int fx = sf & 1, fz = sf >> 1;
      rbuf[n].vars(m, v, fz*nk2 + kc, r, fx*ni2 + ic) =
          rbm(m, sf, ((v*nk2 + kc)*R + r)*ni2 + ic);
    });
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus OrbitalAdvectionCC::SendFluxCC()
//! \brief A block with a coarser x2 neighbour sends its remap fluxes through that
//! face, summed over 2x2 cells and divided by 8 (the coarse cell's content units: a coarse
//! column has 4 fine columns of half the width in x2), to the coarser block

TaskStatus OrbitalAdvectionCC::SendFluxCC() {
  if (!any_mixed || !oa_flux_correct) return TaskStatus::complete;
  const int nmb = pmy_pack->nmb_thispack;
  const int nvar = nvar_oa;
  const int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nk2 = indcs.nx3/2, ni2 = indcs.nx1/2;
  auto &xlev_ = xlev;
  auto &xslot_ = xslot;
  auto &fown_ = fown;
  for (int n=0; n<2; ++n) {
    const int dn = (n + 1) % 2;
    const int base = (n == 0) ? 8 : 12;
    auto fsb = fsend[n];
    auto frb = frecv[dn];
    par_for("oa-flx-s", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, nk2-1, 0, ni2-1,
    KOKKOS_LAMBDA(const int m, const int v, const int kc, const int ic) {
      if (xlev_.d_view(m,n) != 1) return;
      const int slot = xslot_.d_view(m,n);
      const int sf = slot - base;
      const Real f = 0.125*(fown_(m,v,n,2*kc  ,2*ic) + fown_(m,v,n,2*kc  ,2*ic+1)
                          + fown_(m,v,n,2*kc+1,2*ic) + fown_(m,v,n,2*kc+1,2*ic+1));
      const int idx = (v*nk2 + kc)*ni2 + ic;
      if (nghbr.d_view(m,slot).rank == my_rank) {
        const int dm = nghbr.d_view(m,slot).gid - mbgid.d_view(0);
        frb(dm, sf, idx) = f;
      } else {
        fsb(m, 0, idx) = f;
      }
    });
  }
#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  bool no_errors = true;
  const int sff = nvar*nk2*ni2;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      if (xlev.h_view(m,n) != 1) continue;
      const int slot = xslot.h_view(m,n);
      const int drank = nghbr.h_view(m,slot).rank;
      if (drank == my_rank) continue;
      const int lid = nghbr.h_view(m,slot).gid - pmy_pack->pmesh->gids_eachrank[drank];
      const int tag = CreateBvals_MPI_Tag(lid, 16 + nghbr.h_view(m,slot).dest);
      auto ptr = Kokkos::subview(fsend[n], m, 0, Kokkos::ALL);
      if (MPI_Isend(ptr.data(), sff, MPI_ATHENA_REAL, drank, tag, comm_orb_advect,
                    &(fsreq[n][4*m])) != MPI_SUCCESS) {
        no_errors = false;
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting flux sends (orbital advection)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus OrbitalAdvectionCC::RecvFluxCC()
//! \brief A block with finer x2 neighbours replaces its remap flux through that
//! face by theirs: the cell next to the face changes by the difference

TaskStatus OrbitalAdvectionCC::RecvFluxCC(DvceArray5D<Real> &a) {
  if (!any_mixed || !oa_flux_correct) return TaskStatus::complete;
  const int nmb = pmy_pack->nmb_thispack;
#if MPI_PARALLEL_ENABLED
  bool done = true, no_errors = true;
  for (int n=0; n<2; ++n) {
    for (int r=0; r<4*nmb; ++r) {
      int test;
      if (MPI_Test(&(frreq[n][r]), &test, MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
      if (!(static_cast<bool>(test))) done = false;
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing flux receives (orbital advection)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!done) return TaskStatus::incomplete;
#endif
  const int nvar = nvar_oa;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, je = indcs.je, ks = indcs.ks;
  const int nk = indcs.nx3, ni = indcs.nx1, nk2 = nk/2, ni2 = ni/2;
  auto &xlev_ = xlev;
  auto &fown_ = fown;
  for (int n=0; n<2; ++n) {
    auto frb = frecv[n];
    par_for("oa-flx-r", DevExeSpace(), 0, nmb-1, 0, nvar-1, 0, nk-1, 0, ni-1,
    KOKKOS_LAMBDA(const int m, const int v, const int kk, const int ii) {
      if (xlev_.d_view(m,n) != 2) return;
      const int fx = (ii >= ni2) ? 1 : 0, fz = (kk >= nk2) ? 1 : 0;
      const int sf = fx + 2*fz;
      const Real df = frb(m, sf, (v*nk2 + (kk - fz*nk2))*ni2 + (ii - fx*ni2))
                    - fown_(m,v,n,kk,ii);
      if (n == 0) {
        a(m,v,ks+kk,js,is+ii) += df;
      } else {
        a(m,v,ks+kk,je,is+ii) -= df;
      }
    });
  }
  return TaskStatus::complete;
}
