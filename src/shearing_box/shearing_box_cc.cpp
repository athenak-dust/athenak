//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box_cc.cpp
//! \brief functions to pack/send and recv/unpack boundary values for cell-centered (CC)
//! variables with shearing box boundaries.
//! One implementation (S1') for the MeshBlock arrays and, with mesh
//! refinement, the coarse arrays (PackAndSendCoarseCC / RecvAndUnpackCoarseCC).

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
// ShearingBoxCC derived class constructor:

ShearingBoxCC::ShearingBoxCC(MeshBlockPack *pp, ParameterInput *pin, int nvar) :
    ShearingBox(pp, pin), nvar_(nvar) {
  AllocateBuffers();
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxCC::AllocateBuffers()
//! \brief Buffers for the current lists of x1-face MeshBlocks; with coarse_on also for the
//! coarse arrays.  Called by the constructor and after BuildBoundaryLists.

void ShearingBoxCC::AllocateBuffers() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells3 = indcs.nx3 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  int ncells1 = indcs.ng;
  for (int n=0; n<2; ++n) {
    int nmb = std::max(1,nmb_x1bndry(n));
    Kokkos::realloc(sendbuf[n].vars,nmb,ncells2,nvar_,ncells3,ncells1);
    Kokkos::realloc(recvbuf[n].vars,nmb,ncells2,nvar_,ncells3,ncells1);
    if (coarse_on) {
      int cncells3 = (indcs.cnx3 > 1) ? indcs.cnx3 + 2*indcs.ng : 1;
      int cncells2 = indcs.cnx2 + 2*indcs.ng;
      Kokkos::realloc(csendbuf[n].vars,nmb,cncells2,nvar_,cncells3,ncells1);
      Kokkos::realloc(crecvbuf[n].vars,nmb,cncells2,nvar_,cncells3,ncells1);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::PackAndSendCC()
//! \brief Apply shearing sheet BCs to cell-centered variables, including MPI
//! MPI communications. Both the inner_x1 and outer_x1 boundaries are updated.
//! Called on the physics_bcs task after purely periodic BC communication is finished.

TaskStatus ShearingBoxCC::PackAndSendCC(DvceArray5D<Real> &a, ReconstructionMethod rcon) {
  return PackAndSend(a, rcon, false);
}

TaskStatus ShearingBoxCC::PackAndSendCoarseCC(DvceArray5D<Real> &ca,
                                              ReconstructionMethod rcon) {
  return PackAndSend(ca, rcon, true);
}

TaskStatus ShearingBoxCC::PackAndSend(DvceArray5D<Real> &a, ReconstructionMethod rcon,
                                      const bool coarse) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const auto &ng = indcs.ng;
  const ShearSet st = Set(coarse);
  const int ie = st.ie, js = st.js, je = st.je;

  // copy ghost zones at x1-faces into send buffer view
  // apply fractional cell offset to data in send buffers using conservative remap
  const int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L must be NVAR
  const auto &mbsize = pmy_pack->pmb->mb_size;
  int kl = st.kl, ku = st.ku;
  int nj = st.nj;
  const Real dxfac = st.dxfac;
  const int &gids_ = pmy_pack->gids;
  const Real &yshear_ = yshear;
  const auto &x1bndry_mbgid_ = x1bndry_mbgid;
  ShearingBoxBoundaryBuffer *sbufs = coarse ? csendbuf : sendbuf;
  ShearingBoxBoundaryBuffer *rbufs = coarse ? crecvbuf : recvbuf;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj) * 2;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    auto sbuf = sbufs[n].vars;
    par_for_outer("shrcc",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,(nvar-1),kl,ku,0,(ng-1),
    KOKKOS_LAMBDA(TeamMember_t member,const int m,const int v,const int k,const int i) {
      ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nj); // 1D slice of data
      ScrArray1D<Real> flx(member.team_scratch(scr_lvl), nj); // "flux" at faces
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;

      // Load scratch array
      if (n==0) {
        par_for_inner(member, 0, nj-1, [&](const int j) {
          a_(j) = a(mm,v,k,j,i);
        });
      } else {
        par_for_inner(member, 0, nj-1, [&](const int j) {
          a_(j) = a(mm,v,k,j,(ie+1)+i);
        });
      }
      member.team_barrier();

      // compute fractional offset
      const Real dy = dxfac*mbsize.d_view(mm).dx2;
      Real eps = fmod(yshear_,dy)/dy;
      if (n == 1) {eps *= -1.0;}

      // Compute "fluxes" at shifted cell faces
      switch (rcon) {
        case ReconstructionMethod::dc:
          DC_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        case ReconstructionMethod::plm:
          PLM_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        case ReconstructionMethod::ppm4:
        case ReconstructionMethod::ppmx:
        case ReconstructionMethod::wenoz:
          PPMX_RemapFlx(member, js, (je+1), eps, a_, flx);
          break;
        default:
          break;
      }
      member.team_barrier();

      // update data in send buffer with fractional shift
      par_for_inner(member, js, je, [&](const int j) {
        sbuf(m,j,v,k,i) = a_(j) - (flx(j+1) - flx(j));
      });
    });
  }

  // shift data at x1 boundaries by integer number of cells (the pieces of ShearingBox::Pieces).
  //  * Case1 and case3 are when the integer shift (jr<ng), so that the sending MB
  //    overlaps the ghost cells of the two neighbors, and so requires copy/send
  //    to three separate target MBs.
  //  * Case2 is when the sending MB straddles the boundary between MBs, and so requires
  //    copy/send to only two target MBs.
  // Use deep copy if target MB on same rank, or MPI sends if not
  Kokkos::fence();
  bool no_errors=true;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      std::pair<int,int> jsrc[3], jdst[3];
      int jsh[3];
      const int np = Pieces(n, m, st, jsrc, jdst, jsh);
      for (int l=0; l<np; ++l) {
        int tgid, trank;
        FindTargetMB(gid,jsh[l],tgid,trank);
        if (trank == global_variable::my_rank) {
          int tm = TargetIndex(n,tgid);
          using Kokkos::ALL;
          auto src = subview(sbufs[n].vars,m, jsrc[l],ALL,ALL,ALL);
          auto dst = subview(rbufs[n].vars,tm,jdst[l],ALL,ALL,ALL);
          deep_copy(DevExeSpace(), dst, src);
#if MPI_PARALLEL_ENABLED
        } else {
          using Kokkos::ALL;
          auto send_ptr = subview(sbufs[n].vars,m,jsrc[l],ALL,ALL,ALL);
          // create tag using GID of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(tgid, (st.tagbase | (n<<2) | l));
          int data_size = send_ptr.size();
          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, trank, tag,
                               comm_sbox, &(sbufs[n].vars_req[3*m + l]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
#endif
        }
      }
    }
  }
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \!fn void ShearingBoxCC::RecvAndUnpackCC()
//! \brief Check MPI communication of boundary buffers for CC variables have finished,
//! then copy buffers into ghost zones. Shift has already been performed in
//! PackAndSendCC() function

TaskStatus ShearingBoxCC::RecvAndUnpackCC(DvceArray5D<Real> &a) {
  return RecvAndUnpack(a, false);
}

TaskStatus ShearingBoxCC::RecvAndUnpackCoarseCC(DvceArray5D<Real> &ca) {
  return RecvAndUnpack(ca, true);
}

TaskStatus ShearingBoxCC::RecvAndUnpack(DvceArray5D<Real> &a, const bool coarse) {
  // create local references for variables in kernel
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int &ng = indcs.ng;
  const ShearSet st = Set(coarse);
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed
  bool done = true;
  if (!TestRecvs(coarse, done)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if recv boundary buffer communications have not completed
  if (!done) {return TaskStatus::incomplete;}
#endif

  //----- STEP 2: communications have all completed, so unpack and apply shift
  // copy recv buffer view into ghost zones at x1-faces
  const int nvar = a.extent_int(1);  // TODO(@user): 2nd index from L must be NVAR
  const int ie = st.ie;
  int kl = st.kl, ku = st.ku;
  int nj = st.nj;
  const int &gids_ = pmy_pack->gids;
  const auto &x1bndry_mbgid_ = x1bndry_mbgid;
  ShearingBoxBoundaryBuffer *rbufs = coarse ? crecvbuf : recvbuf;
  int scr_lvl=0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nj) * 3;
  for (int n=0; n<2; ++n) {
    int nmb1 = nmb_x1bndry(n) - 1;
    auto rbuf = rbufs[n].vars;
    par_for_outer("shrcc",DevExeSpace(),scr_size,scr_lvl,0,nmb1,0,(nvar-1),kl,ku,0,(ng-1),
    KOKKOS_LAMBDA(TeamMember_t member,const int m,const int v,const int k,const int i) {
      int mm = x1bndry_mbgid_.d_view(n,m) - gids_;
      if (n==0) {
        par_for_inner(member, 0, nj-1, [&](const int j) {
          a(mm,v,k,j,i) = rbuf(m,j,v,k,i);
        });
        member.team_barrier();
      } else {
        par_for_inner(member, 0, nj-1, [&](const int j) {
          a(mm,v,k,j,(ie+1)+i) = rbuf(m,j,v,k,i);
        });
        member.team_barrier();
      }
    });
  }

  return TaskStatus::complete;
}
