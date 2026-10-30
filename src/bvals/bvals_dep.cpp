//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_dep.cpp
//! \brief Additive ghost-zone ("deposit") exchange for cell-centered fields built by
//! particle-mesh deposition. A particle near a MeshBlock boundary deposits part of its
//! weight into ghost cells of its own MeshBlock; those contributions physically belong
//! to the *active* cells of the neighboring MeshBlock and must be added there. Thus the
//! data flow is the reverse of the ordinary copy exchange: the sender packs its GHOST
//! regions and the receiver *sums* the buffers into its active-zone edge strips.
//!
//! Because the receive strips of different neighbor buffers overlap (a face strip and
//! an edge strip share cells near active-zone corners), the receiver must accumulate
//! buffers with a scalar loop over neighbors, exactly as in
//! MeshBoundaryValuesFC::SumBoundaryFluxes().
//!
//! The production exchange remains same-level only. An explicitly enabled,
//! single-rank, static-2-D-SMR test path maps every deposited ghost-cell density to the
//! active leaf cells that geometrically overlap it. It preserves the integrated
//! deposited quantity by applying overlap_volume/receiving_volume. This path is a
//! diagnostic owner-level transfer, not the exact transpose of AthenaK prolongation.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"
#include "dust/dust_input.hpp"

//----------------------------------------------------------------------------------------
// MeshBoundaryValuesDep constructor:

MeshBoundaryValuesDep::MeshBoundaryValuesDep(MeshBlockPack *pp, ParameterInput *pin) :
  MeshBoundaryValues(pp, pin, false) {
#if MPI_PARALLEL_ENABLED
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_dep_);   // the general path
#endif
  if (!pp->pmesh->multilevel) return;
  const dust::AmrPic pic = dust::ResolveAmrPic(pin, true);
  // a 3-D mesh with refinement and instant_switch uses the general
  // remap (DepRemapPlan, bvals_dep_general.cpp), which the dust module sets up; it checks the rest
  if (pp->pmesh->three_d && pic == dust::AmrPic::instant_switch) return;
  // refinable clouds deposit into the leaves directly (2-D and 3-D), and several ranks
  // are supported only for them (through the dust halo): this exchange is not used then
  const bool rc = (pic == dust::AmrPic::refinable_cloud);
  const bool clouds3d = rc && pp->pmesh->three_d;
  // (the leaf traversal handles the sheared image and the physical x3 faces)
  const bool shear3d = clouds3d &&
      (pp->pmesh->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic ||
       pp->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic);
  if (!((pp->pmesh->two_d || clouds3d) && (pp->pmesh->strictly_periodic || shear3d) &&
        (global_variable::nranks == 1 || rc))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Additive deposit exchange does not support this "
              << "SMR/AMR configuration" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (rc) {
    // The composite dust kernels deposit directly into active leaf cells.
    local_smr_direct_ = true;
  } else if (pic == dust::AmrPic::instant_switch) {
    local_smr_remap_ = true;
    BuildLocalSMRMap();
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dust on a refined mesh needs <dust>/amr_pic = refinable_cloud "
              << "or instant_switch" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::BuildLocalSMRMap()
//! \brief Precompute the active-leaf destinations of every local ghost cell.  This runs
//! on the device, and only the block itself and its neighbours (each once) are
//! searched instead of every block, so that a dynamic mesh can rebuild the map after each
//! regrid; per target block only the cells near the ghost cell are tested.  Blocks narrower
//! than two ghost depths (the coarse ghost zone then reaches past the finer neighbours) search
//! every block of the pack, as before.  The overlaps, the
//! tests and the weights are those of the former full search, so the map is the same (the
//! order of the targets of a ghost cell may differ, which does not change any sum).

void MeshBoundaryValuesDep::BuildLocalSMRMap() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int ncells1 = indcs.nx1 + 2*indcs.ng;
  const int ncells2 = indcs.nx2 + 2*indcs.ng;
  const int ncells3 = 1;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  if (smr_target_count_.extent_int(0) != nmb) {
    smr_target_count_ = DvceArray4D<int>("dep_smr_count", nmb, ncells3, ncells2, ncells1);
    smr_target_index_ = DvceArray5D<int>("dep_smr_index", nmb, ncells3, ncells2, ncells1,
                                         max_smr_targets_);
    smr_target_weight_ = DvceArray5D<Real>("dep_smr_weight", nmb, ncells3, ncells2, ncells1,
                                           max_smr_targets_);
  }
  auto count = smr_target_count_;
  auto index = smr_target_index_;
  auto weight = smr_target_weight_;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const int nnghbr = pmy_pack->pmb->nnghbr;
  const int gids = pmy_pack->gids;
  auto &meshsize = pmy_pack->pmesh->mesh_size;
  const Real x1min = meshsize.x1min, x1max = meshsize.x1max;
  const Real x2min = meshsize.x2min, x2max = meshsize.x2max;
  const Real lxmesh = x1max - x1min, lymesh = x2max - x2min;
  constexpr int ntmax = max_smr_targets_;
  // the ghost zone of a coarse block is 2 ng cells of its finer
  // neighbours; with blocks narrower than that (e.g. 4^2 cells, ng = 4) it reaches past the
  // neighbours, so every block of the pack is searched then
  const bool full = (2*indcs.ng > nx1) || (2*indcs.ng > nx2);
  const int ncand = full ? nmb : nnghbr;
  int nmany = 0, nuncovered = 0, nremote = 0;
  Kokkos::parallel_reduce("DepSMRMap",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*ncells2*ncells1),
  KOKKOS_LAMBDA(const int idx, int &many, int &uncovered, int &remote) {
    const int m = idx/(ncells2*ncells1);
    const int r = idx - m*(ncells2*ncells1);
    const int j = r/ncells1, i = r - j*ncells1;
    if (i >= is && i <= ie && j >= js && j <= je) {count(m,0,j,i) = 0; return;}
    const Real sdx1 = mbsize.d_view(m).dx1, sdx2 = mbsize.d_view(m).dx2;
    Real xlo = mbsize.d_view(m).x1min + static_cast<Real>(i - is)*sdx1;
    Real xhi = xlo + sdx1;
    Real ylo = mbsize.d_view(m).x2min + static_cast<Real>(j - js)*sdx2;
    Real yhi = ylo + sdx2;
    while (0.5*(xlo + xhi) < x1min) {xlo += lxmesh; xhi += lxmesh;}
    while (0.5*(xlo + xhi) >= x1max) {xlo -= lxmesh; xhi -= lxmesh;}
    while (0.5*(ylo + yhi) < x2min) {ylo += lymesh; yhi += lymesh;}
    while (0.5*(ylo + yhi) >= x2max) {ylo -= lymesh; yhi -= lymesh;}
    int ntarget = 0;
    Real overlap_sum = 0.0;
    for (int n=-1; n<ncand; ++n) {   // the block itself, then each neighbour (or block) once
      int tm = m;
      if (n >= 0 && full) {
        if (n == m) continue;
        tm = n;
      } else if (n >= 0) {
        const int g = nghbr.d_view(m,n).gid;
        if (g < 0 || g == gids + m) continue;
        bool seen = false;
        for (int q=0; q<n; ++q) {
          if (nghbr.d_view(m,q).gid == g) {seen = true; break;}
        }
        if (seen) continue;
        tm = g - gids;
        if (tm < 0 || tm >= nmb) {++remote; continue;}   // on another rank: not supported
      }
      const Real dx1min = mbsize.d_view(tm).x1min, dx1max = mbsize.d_view(tm).x1max;
      const Real dx2min = mbsize.d_view(tm).x2min, dx2max = mbsize.d_view(tm).x2max;
      const Real ddx1 = mbsize.d_view(tm).dx1, ddx2 = mbsize.d_view(tm).dx2;
      if (xhi <= dx1min || xlo >= dx1max || yhi <= dx2min || ylo >= dx2max) continue;
      // the target cells near the ghost cell (one cell of margin; the tests below decide)
      const int tj0 = Kokkos::max(0, static_cast<int>(Kokkos::floor((ylo - dx2min)/ddx2)) - 1);
      const int tj1 = Kokkos::min(nx2 - 1, static_cast<int>(Kokkos::floor((yhi - dx2min)/ddx2)) + 1);
      const int ti0 = Kokkos::max(0, static_cast<int>(Kokkos::floor((xlo - dx1min)/ddx1)) - 1);
      const int ti1 = Kokkos::min(nx1 - 1, static_cast<int>(Kokkos::floor((xhi - dx1min)/ddx1)) + 1);
      for (int tj=tj0; tj<=tj1; ++tj) {
        Real tylo = dx2min + static_cast<Real>(tj)*ddx2;
        Real tyhi = tylo + ddx2;
        Real oy = Kokkos::fmax(0.0, Kokkos::fmin(yhi, tyhi) - Kokkos::fmax(ylo, tylo));
        if (oy == 0.0) continue;
        for (int ti=ti0; ti<=ti1; ++ti) {
          Real txlo = dx1min + static_cast<Real>(ti)*ddx1;
          Real txhi = txlo + ddx1;
          Real ox = Kokkos::fmax(0.0, Kokkos::fmin(xhi, txhi) - Kokkos::fmax(xlo, txlo));
          Real overlap = ox*oy;
          Real overlap_tol = 1.0e-12*Kokkos::fmin(sdx1*sdx2, ddx1*ddx2);
          if (overlap <= overlap_tol) continue;
          if (ntarget >= ntmax) {++many; continue;}
          index(m,0,j,i,ntarget) = (((tm*ncells3 + ks)*ncells2 + tj + js)*ncells1 + ti + is);
          weight(m,0,j,i,ntarget) = overlap/(ddx1*ddx2);
          overlap_sum += overlap;
          ++ntarget;
        }
      }
    }
    const Real source_volume = sdx1*sdx2;
    if (Kokkos::fabs(overlap_sum - source_volume) > 1.0e-10*source_volume) ++uncovered;
    count(m,0,j,i) = ntarget;
  }, Kokkos::Sum<int>(nmany), Kokkos::Sum<int>(nuncovered), Kokkos::Sum<int>(nremote));
  if (nmany > 0 || nuncovered > 0 || nremote > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Static-SMR deposit map: " << nmany << " ghost cells overlap too many leaves, "
              << nuncovered << " are not covered by the block and its neighbours, " << nremote
              << " neighbours lie on another rank" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesDep::RemapLocalSMRDeposit()
//! \brief Add integrated ghost deposits to overlapping active leaf cells.

TaskStatus MeshBoundaryValuesDep::RemapLocalSMRDeposit(DvceArray5D<Real> &a) {
  int nmb = pmy_pack->nmb_thispack;
  int nvar = a.extent_int(1);
  int ncells3 = a.extent_int(2);
  int ncells2 = a.extent_int(3);
  int ncells1 = a.extent_int(4);
  int ncell = nmb*ncells3*ncells2*ncells1;
  auto counts = smr_target_count_;
  auto targets = smr_target_index_;
  auto weights = smr_target_weight_;

  Kokkos::parallel_for("DepSMRRemap", Kokkos::RangePolicy<>(DevExeSpace(),0,ncell),
  KOKKOS_LAMBDA(const int idx) {
    int m = idx/(ncells3*ncells2*ncells1);
    int remainder = idx - m*(ncells3*ncells2*ncells1);
    int k = remainder/(ncells2*ncells1);
    remainder -= k*(ncells2*ncells1);
    int j = remainder/ncells1;
    int i = remainder - j*ncells1;
    int count = counts(m,k,j,i);
    for (int t=0; t<count; ++t) {
      int encoded = targets(m,k,j,i,t);
      int ti = encoded % ncells1;
      encoded /= ncells1;
      int tj = encoded % ncells2;
      encoded /= ncells2;
      int tk = encoded % ncells3;
      int tm = encoded/ncells3;
      Real weight = weights(m,k,j,i,t);
      for (int v=0; v<nvar; ++v) {
        Kokkos::atomic_add(&a(tm,v,tk,tj,ti), weight*a(m,v,k,j,i));
      }
    }
  });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::InitSendIndices
//! \brief Calculates indices of GHOST cells packed into send buffers. These are the
//! mirror image of the receive ("unpack into ghosts") indices of the copy exchange in
//! MeshBoundaryValuesCC::InitRecvIndices. Only same-level indices are used.

void MeshBoundaryValuesDep::InitSendIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs = pmy_pack->pmesh->mb_indcs;
  int ng = mb_indcs.ng;
  if ((f1 != 0) || (f2 != 0)) {return;}  // only same-level buffers used

  auto &isame = buf.isame[0];
  if (ox1 == 0) {
    isame.bis = mb_indcs.is;          isame.bie = mb_indcs.ie;
  } else if (ox1 > 0) {
    isame.bis = mb_indcs.ie + 1;      isame.bie = mb_indcs.ie + ng;
  } else {
    isame.bis = mb_indcs.is - ng;     isame.bie = mb_indcs.is - 1;
  }

  if (ox2 == 0) {
    isame.bjs = mb_indcs.js;          isame.bje = mb_indcs.je;
  } else if (ox2 > 0) {
    isame.bjs = mb_indcs.je + 1;      isame.bje = mb_indcs.je + ng;
  } else {
    isame.bjs = mb_indcs.js - ng;     isame.bje = mb_indcs.js - 1;
  }

  if (ox3 == 0) {
    isame.bks = mb_indcs.ks;          isame.bke = mb_indcs.ke;
  } else if (ox3 > 0) {
    isame.bks = mb_indcs.ke + 1;      isame.bke = mb_indcs.ke + ng;
  } else {
    isame.bks = mb_indcs.ks - ng;     isame.bke = mb_indcs.ks - 1;
  }
  buf.isame_ndat = (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*
                   (isame.bke - isame.bks + 1);
  // Keep these per-axis extents identical to the partner ranges constructed by
  // InitRecvIndices(). Rank-packed MPI sends and receives must agree on payload size.
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::InitRecvIndices
//! \brief Calculates indices of ACTIVE cells into which receive buffers are summed.
//! These are the mirror image of the send ("pack from active") indices of the copy
//! exchange in MeshBoundaryValuesCC::InitSendIndices.

void MeshBoundaryValuesDep::InitRecvIndices(MeshBoundaryBuffer &buf,
                                            int ox1, int ox2, int ox3, int f1, int f2) {
  auto &mb_indcs = pmy_pack->pmesh->mb_indcs;
  int ng1 = mb_indcs.ng - 1;
  if ((f1 != 0) || (f2 != 0)) {return;}  // only same-level buffers used

  auto &isame = buf.isame[0];
  isame.bis = (ox1 > 0) ? (mb_indcs.ie - ng1) : mb_indcs.is;
  isame.bie = (ox1 < 0) ? (mb_indcs.is + ng1) : mb_indcs.ie;
  isame.bjs = (ox2 > 0) ? (mb_indcs.je - ng1) : mb_indcs.js;
  isame.bje = (ox2 < 0) ? (mb_indcs.js + ng1) : mb_indcs.je;
  isame.bks = (ox3 > 0) ? (mb_indcs.ke - ng1) : mb_indcs.ks;
  isame.bke = (ox3 < 0) ? (mb_indcs.ks + ng1) : mb_indcs.ke;
  buf.isame_ndat = (isame.bie - isame.bis + 1)*(isame.bje - isame.bjs + 1)*
                   (isame.bke - isame.bks + 1);
  // Keep these per-axis extents identical to the partner ranges constructed by
  // InitSendIndices(). Rank-packed MPI sends and receives must agree on payload size.
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesDep::PackAndSendDeposit()
//! \brief Pack ghost-region deposits into boundary buffers and send to neighbors.
//! Adapted from MeshBoundaryValuesCC::PackAndSendCC with all coarse/fine/z4c branches
//! removed (uniform grid only). Same-rank neighbors are written directly into the
//! destination receive buffer; the task graph guarantees the receiver does not start
//! summing until all local packs are complete.
//!
//! Input array must be a 5D Kokkos View dimensioned (nmb, nvar, nx3, nx2, nx1)

TaskStatus MeshBoundaryValuesDep::PackAndSendDeposit(DvceArray5D<Real> &a) {
  if (General()) {   // pack the source ghost cells of both parts, send them
    for (int p=0; p<2; ++p) GeneralPack(a, (p == 0) ? plan_->stat : plan_->shr, gsbuf_[p]);
#if MPI_PARALLEL_ENABLED
    Kokkos::fence();
    const int nvar = a.extent_int(1);
    bool no_errors = true;
    for (int p=0; p<2; ++p) {
      const DepRemapMap &map = (p == 0) ? plan_->stat : plan_->shr;
      for (std::size_t q=0; q<map.send_rank.size(); ++q) {
        MPI_Request req;
        int ierr = MPI_Isend(gsbuf_[p].data() + map.send_off[q]*nvar, map.send_cnt[q]*nvar,
                             MPI_ATHENA_REAL, map.send_rank[q], 31 + p, comm_dep_, &req);
        if (ierr != MPI_SUCCESS) no_errors = false;
        gsreq_.push_back(req);
      }
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "MPI error in posting the deposit sends" << std::endl;
      std::exit(EXIT_FAILURE);
    }
#endif
    return TaskStatus::complete;
  }
  if (local_smr_direct_) return TaskStatus::complete;
  if (local_smr_remap_) return RemapLocalSMRDeposit(a);

  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  // Build (or refresh) the rank-packed metadata before the kernel writes off-rank
  // payloads directly into the aggregate send buffer.
  if (rank_packed_bvals_nvars_ != nvar ||
      rank_packed_mesh_seq_ != pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq()) {
    BuildRankPackedVarMetadata(nvar);
  }
  auto aggsbuf = rank_sendbuf_vars_;
  auto sendoff = send_agg_offset_;
#endif
  // Outer loop over (# of MeshBlocks)*(# of buffers)*(# of variables)
  int nmnv = nmb*nnghbr*nvar;
  Kokkos::TeamPolicy<> policy(DevExeSpace(), nmnv, Kokkos::AUTO);
  Kokkos::parallel_for("DepSendBuff", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/(nnghbr*nvar);
    const int n = (tmember.league_rank() - m*(nnghbr*nvar))/nvar;
    const int v = (tmember.league_rank() - m*(nnghbr*nvar) - n*nvar);

    // only load buffers when neighbor exists
    if (nghbr.d_view(m,n).gid >= 0) {
      int il = sbuf[n].isame[0].bis;
      int iu = sbuf[n].isame[0].bie;
      int jl = sbuf[n].isame[0].bjs;
      int ju = sbuf[n].isame[0].bje;
      int kl = sbuf[n].isame[0].bks;
      int ku = sbuf[n].isame[0].bke;
      int ni = iu - il + 1;
      int nj = ju - jl + 1;
      int nk = ku - kl + 1;
      int nkj  = nk*nj;

      // indices of recv'ing (destination) MB and buffer: MB IDs are stored sequentially
      // in MeshBlockPacks, so array index equals (target_id - first_id)
      int dm = nghbr.d_view(m,n).gid - mbgid.d_view(0);
      int dn = nghbr.d_view(m,n).dest;

      // Middle loop over k,j
      Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
        int k = idx / nj;
        int j = (idx - k * nj) + jl;
        k += kl;

        // Inner (vector) loop over i
        // copy directly into recv buffer if MeshBlocks on same rank
        if (nghbr.d_view(m,n).rank == my_rank) {
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            rbuf[dn].vars(dm, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
          });
        // else copy into send buffer for MPI communication below
        } else {
#if MPI_PARALLEL_ENABLED
          int base = sendoff(m*nnghbr + n);
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            aggsbuf(base + (i-il + ni*(j-jl + nj*(k-kl + nk*v)))) = a(m,v,k,j,i);
          });
#else
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
          });
#endif
        }
      });
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  // Send one aggregate boundary payload to each neighboring MPI rank. The packing
  // kernel wrote off-rank entries directly into rank_sendbuf_vars_.
  Kokkos::fence();
  bool no_errors = true;
  std::fill(send_var_reqs_.begin(), send_var_reqs_.end(), MPI_REQUEST_NULL);
  for (std::size_t i = 0; i < send_var_msgs_.size(); ++i) {
    const auto &msg = send_var_msgs_[i];
    int ierr = MPI_Isend(rank_sendbuf_vars_.data() + msg.offset, msg.data_size,
                         MPI_ATHENA_REAL, msg.rank, 1, comm_vars, &send_var_reqs_[i]);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesDep::RecvAndSumDeposit()
//! \brief Check receives are complete, then SUM buffers into active-zone edge strips.
//! The loop over neighbor buffers is intentionally scalar (sequential within each team):
//! receive strips of different buffers overlap near active-zone edges/corners, so
//! concurrent accumulation over buffers would race (cf. SumBoundaryFluxes).

TaskStatus MeshBoundaryValuesDep::RecvAndSumDeposit(DvceArray5D<Real> &a) {
  if (General()) {   // all receives complete, then the two parts in order
#if MPI_PARALLEL_ENABLED
    bool done = true;
    for (auto &r : grreq_) {
      int test = 1;
      MPI_Test(&r, &test, MPI_STATUS_IGNORE);
      if (!test) done = false;
    }
    if (!done) return TaskStatus::incomplete;
#endif
    GeneralDropRecord(a, plan_->stat);   // dropped ghost deposits beyond an x3 face
    GeneralApply(a, plan_->stat, grbuf_[0]);
    GeneralApply(a, plan_->shr, grbuf_[1]);
    return TaskStatus::complete;
  }
  if (local_smr_direct_) return TaskStatus::complete;
  if (local_smr_remap_) return TaskStatus::complete;

  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (std::size_t i = 0; i < recv_var_reqs_.size(); ++i) {
    int test;
    int ierr = MPI_Test(&recv_var_reqs_[i], &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
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

  //----- STEP 2: buffers have all completed, so sum into active cells

  int nvar = a.extent_int(1);
#if MPI_PARALLEL_ENABLED
  auto aggrbuf = rank_recvbuf_vars_;
  auto recvoff = recv_agg_offset_;
#endif

  // Outer loop over (# of MeshBlocks)*(# of variables); loop over buffers is scalar
  Kokkos::TeamPolicy<> policy(DevExeSpace(), (nmb*nvar), Kokkos::AUTO);
  Kokkos::parallel_for("DepRecvSum", policy, KOKKOS_LAMBDA(TeamMember_t tmember) {
    const int m = (tmember.league_rank())/nvar;
    const int v = (tmember.league_rank() - m*nvar);

    for (int n=0; n<nnghbr; ++n) {
      // only sum buffers when neighbor exists
      if (nghbr.d_view(m,n).gid >= 0) {
        int il = rbuf[n].isame[0].bis;
        int iu = rbuf[n].isame[0].bie;
        int jl = rbuf[n].isame[0].bjs;
        int ju = rbuf[n].isame[0].bje;
        int kl = rbuf[n].isame[0].bks;
        int ku = rbuf[n].isame[0].bke;
        int ni = iu - il + 1;
        int nj = ju - jl + 1;
        int nk = ku - kl + 1;
        int nkj  = nk*nj;
#if MPI_PARALLEL_ENABLED
        // A non-negative aggregate offset identifies an off-rank payload. Same-rank
        // payloads remain in their per-neighbor receive buffers.
        const int base = recvoff(m*nnghbr + n);
#endif

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // Inner (vector) loop over i
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            const int bi = (i-il + ni*(j-jl + nj*(k-kl + nk*v)));
#if MPI_PARALLEL_ENABLED
            a(m,v,k,j,i) += (base >= 0) ? aggrbuf(base + bi) : rbuf[n].vars(m, bi);
#else
            a(m,v,k,j,i) += rbuf[n].vars(m, bi);
#endif
          });
        });
      }
      // barrier so no thread starts summing the next (overlapping) buffer early
      tmember.team_barrier();
    }
  });  // end par_for_outer

  return TaskStatus::complete;
}
