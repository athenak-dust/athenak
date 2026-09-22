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

//----------------------------------------------------------------------------------------
// MeshBoundaryValuesDep constructor:

MeshBoundaryValuesDep::MeshBoundaryValuesDep(MeshBlockPack *pp, ParameterInput *pin) :
  MeshBoundaryValues(pp, pin, false) {
  if (pp->pmesh->multilevel) {
    bool experimental_smr = pin->GetOrAddBoolean(
        "dust", "experimental_static_smr", false);
    if (!(experimental_smr && !(pp->pmesh->adaptive) && pp->pmesh->two_d &&
          pp->pmesh->strictly_periodic && global_variable::nranks == 1)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Additive deposit exchange does not support this "
                << "SMR/AMR configuration" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::string transfer = pin->GetOrAddString(
        "dust", "experimental_smr_transfer", "smooth_p2");
    if (transfer.compare("smooth_p2") == 0) {
      // The composite dust kernels deposit directly into active leaf cells.
      local_smr_direct_ = true;
    } else if (transfer.compare("owner_overlap") == 0) {
      local_smr_remap_ = true;
      BuildLocalSMRMap();
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Unknown experimental static-SMR deposit transfer"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValuesDep::BuildLocalSMRMap()
//! \brief Precompute the active-leaf destinations of every local ghost cell.

void MeshBoundaryValuesDep::BuildLocalSMRMap() {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = pmy_pack->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  int ncells3 = 1;
  smr_target_count_ = DualArray4D<int>("dep_smr_count", nmb, ncells3,
                                       ncells2, ncells1);
  smr_target_index_ = DualArray5D<int>("dep_smr_index", nmb, ncells3,
                                       ncells2, ncells1, max_smr_targets_);
  smr_target_weight_ = DualArray5D<Real>("dep_smr_weight", nmb, ncells3,
                                         ncells2, ncells1, max_smr_targets_);
  Kokkos::deep_copy(smr_target_count_.h_view, 0);
  Kokkos::deep_copy(smr_target_index_.h_view, -1);
  Kokkos::deep_copy(smr_target_weight_.h_view, 0.0);

  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &meshsize = pmy_pack->pmesh->mesh_size;
  Real lxmesh = meshsize.x1max - meshsize.x1min;
  Real lymesh = meshsize.x2max - meshsize.x2min;

  for (int m=0; m<nmb; ++m) {
    const auto &src = mbsize.h_view(m);
    for (int j=0; j<ncells2; ++j) {
      for (int i=0; i<ncells1; ++i) {
        bool active = (i >= indcs.is && i <= indcs.ie &&
                       j >= indcs.js && j <= indcs.je);
        if (active) continue;

        Real xlo = src.x1min + static_cast<Real>(i - indcs.is)*src.dx1;
        Real xhi = xlo + src.dx1;
        Real ylo = src.x2min + static_cast<Real>(j - indcs.js)*src.dx2;
        Real yhi = ylo + src.dx2;
        while (0.5*(xlo + xhi) < meshsize.x1min) {xlo += lxmesh; xhi += lxmesh;}
        while (0.5*(xlo + xhi) >= meshsize.x1max) {xlo -= lxmesh; xhi -= lxmesh;}
        while (0.5*(ylo + yhi) < meshsize.x2min) {ylo += lymesh; yhi += lymesh;}
        while (0.5*(ylo + yhi) >= meshsize.x2max) {ylo -= lymesh; yhi -= lymesh;}

        int ntarget = 0;
        Real overlap_sum = 0.0;
        for (int tm=0; tm<nmb; ++tm) {
          const auto &dst = mbsize.h_view(tm);
          if (xhi <= dst.x1min || xlo >= dst.x1max ||
              yhi <= dst.x2min || ylo >= dst.x2max) continue;
          for (int tj=0; tj<indcs.nx2; ++tj) {
            Real tylo = dst.x2min + static_cast<Real>(tj)*dst.dx2;
            Real tyhi = tylo + dst.dx2;
            Real oy = std::max(0.0, std::min(yhi, tyhi) - std::max(ylo, tylo));
            if (oy == 0.0) continue;
            for (int ti=0; ti<indcs.nx1; ++ti) {
              Real txlo = dst.x1min + static_cast<Real>(ti)*dst.dx1;
              Real txhi = txlo + dst.dx1;
              Real ox = std::max(0.0, std::min(xhi, txhi) - std::max(xlo, txlo));
              Real overlap = ox*oy;
              Real overlap_tol = 1.0e-12*std::min(src.dx1*src.dx2,
                                                  dst.dx1*dst.dx2);
              if (overlap <= overlap_tol) continue;
              if (ntarget >= max_smr_targets_) {
                std::cout << "### FATAL ERROR in " << __FILE__ << " at line "
                          << __LINE__ << std::endl
                          << "Static-SMR deposit cell overlaps too many leaves" << std::endl;
                std::exit(EXIT_FAILURE);
              }
              int tk = indcs.ks;
              int tj_full = tj + indcs.js;
              int ti_full = ti + indcs.is;
              int encoded = (((tm*ncells3 + tk)*ncells2 + tj_full)*ncells1 + ti_full);
              smr_target_index_.h_view(m,0,j,i,ntarget) = encoded;
              smr_target_weight_.h_view(m,0,j,i,ntarget) =
                  overlap/(dst.dx1*dst.dx2);
              overlap_sum += overlap;
              ++ntarget;
            }
          }
        }
        Real source_volume = src.dx1*src.dx2;
        if (std::abs(overlap_sum - source_volume) > 1.0e-10*source_volume) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Static-SMR deposit map does not cover ghost cell: "
                    << "overlap/source=" << overlap_sum/source_volume << std::endl;
          std::exit(EXIT_FAILURE);
        }
        smr_target_count_.h_view(m,0,j,i) = ntarget;
      }
    }
  }

  smr_target_count_.template modify<HostMemSpace>();
  smr_target_count_.template sync<DevExeSpace>();
  smr_target_index_.template modify<HostMemSpace>();
  smr_target_index_.template sync<DevExeSpace>();
  smr_target_weight_.template modify<HostMemSpace>();
  smr_target_weight_.template sync<DevExeSpace>();
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
  auto &counts = smr_target_count_;
  auto &targets = smr_target_index_;
  auto &weights = smr_target_weight_;

  Kokkos::parallel_for("DepSMRRemap", Kokkos::RangePolicy<>(DevExeSpace(),0,ncell),
  KOKKOS_LAMBDA(const int idx) {
    int m = idx/(ncells3*ncells2*ncells1);
    int remainder = idx - m*(ncells3*ncells2*ncells1);
    int k = remainder/(ncells2*ncells1);
    remainder -= k*(ncells2*ncells1);
    int j = remainder/ncells1;
    int i = remainder - j*ncells1;
    int count = counts.d_view(m,k,j,i);
    for (int t=0; t<count; ++t) {
      int encoded = targets.d_view(m,k,j,i,t);
      int ti = encoded % ncells1;
      encoded /= ncells1;
      int tj = encoded % ncells2;
      encoded /= ncells2;
      int tk = encoded % ncells3;
      int tm = encoded/ncells3;
      Real weight = weights.d_view(m,k,j,i,t);
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
