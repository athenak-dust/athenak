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
//! Only same-level (uniform grid) exchanges are supported.

#include <cstdlib>
#include <iostream>
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
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Additive deposit exchange does not support SMR/AMR"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
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
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  int nvar = a.extent_int(1);

  {int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &mbgid = pmy_pack->pmb->mb_gid;
  auto &sbuf = sendbuf;
  auto &rbuf = recvbuf;
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
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            sbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) ) = a(m,v,k,j,i);
          });
        }
      });
    } // end if-neighbor-exists block
    tmember.team_barrier();
  }); // end par_for_outer
  }

#if MPI_PARALLEL_ENABLED
  // Send boundary buffer to neighboring MeshBlocks using MPI
  Kokkos::fence();
  int my_rank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) {  // neighbor exists and not a physical boundary
        // index and rank of destination Neighbor
        int dn = nghbr.h_view(m,n).dest;
        int drank = nghbr.h_view(m,n).rank;
        if (drank != my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int lid = nghbr.h_view(m,n).gid - pmy_pack->pmesh->gids_eachrank[drank];
          int tag = CreateBvals_MPI_Tag(lid, dn);
          int data_size = nvar*(sendbuf[n].isame_ndat);
          auto send_ptr = Kokkos::subview(sendbuf[n].vars, m, Kokkos::ALL);

          int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                               comm_vars, &(sendbuf[n].vars_req[m]));
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
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus MeshBoundaryValuesDep::RecvAndSumDeposit()
//! \brief Check receives are complete, then SUM buffers into active-zone edge strips.
//! The loop over neighbor buffers is intentionally scalar (sequential within each team):
//! receive strips of different buffers overlap near active-zone edges/corners, so
//! concurrent accumulation over buffers would race (cf. SumBoundaryFluxes).

TaskStatus MeshBoundaryValuesDep::RecvAndSumDeposit(DvceArray5D<Real> &a) {
  // create local references for variables in kernel
  int nmb = pmy_pack->nmb_thispack;
  int nnghbr = pmy_pack->pmb->nnghbr;
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &rbuf = recvbuf;
#if MPI_PARALLEL_ENABLED
  //----- STEP 1: check that recv boundary buffer communications have all completed

  bool bflag = false;
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<nnghbr; ++n) {
      if (nghbr.h_view(m,n).gid >= 0) { // neighbor exists and not a physical boundary
        if (nghbr.h_view(m,n).rank != global_variable::my_rank) {
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

        // Middle loop over k,j
        Kokkos::parallel_for(Kokkos::TeamThreadRange<>(tmember, nkj), [&](const int idx) {
          int k = idx / nj;
          int j = (idx - k * nj) + jl;
          k += kl;

          // Inner (vector) loop over i
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(tmember,il,iu+1),
          [&](const int i) {
            a(m,v,k,j,i) += rbuf[n].vars(m, (i-il + ni*(j-jl + nj*(k-kl + nk*v))) );
          });
        });
      }
      // barrier so no thread starts summing the next (overlapping) buffer early
      tmember.team_barrier();
    }
  });  // end par_for_outer

  return TaskStatus::complete;
}
