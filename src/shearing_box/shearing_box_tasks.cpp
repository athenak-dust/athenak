//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shearing_box_tasks.cpp
//! \brief functions included in task lists to post/clear non-blocking MPI calls for
//! shearing box, and flux correction steps with shearing box boundaries.
//! The pieces of the integer shift are computed once
//! (ShearingBox::Pieces) for the MeshBlock arrays and, with mesh refinement, for the coarse
//! arrays (coarse_on); the MPI posts and waits below loop over both.

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
//! \fn bool ShearingBox::PostRecvs(), TestRecvs(), WaitRecvs(), WaitSends()
//! \brief MPI receives and sends of one exchange (fine or coarse).  Return false on an MPI
//! error; TestRecvs sets done = false while a receive is outstanding.

bool ShearingBox::PostRecvs(const bool coarse) {
  bool no_errors = true;
#if MPI_PARALLEL_ENABLED
  const ShearSet st = Set(coarse);
  ShearingBoxBoundaryBuffer *rbuf = coarse ? crecvbuf : recvbuf;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      std::pair<int,int> jsrc[3], jdst[3];
      int jsh[3];
      const int np = Pieces(n, m, st, jsrc, jdst, jsh);
      for (int l=0; l<np; ++l) {
        int sgid, srank;
        FindTargetMB(gid,-jsh[l],sgid,srank);   // the sender of this piece
        if (srank != global_variable::my_rank) {
          // create tag using GID of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(gid, (st.tagbase | (n<<2) | l));
          using Kokkos::ALL;
          auto recv_ptr = Kokkos::subview(rbuf[n].vars, m, jdst[l], ALL, ALL, ALL);
          int data_size = recv_ptr.size();
          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, srank, tag,
                               comm_sbox, &(rbuf[n].vars_req[3*m + l]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
#endif
  return no_errors;
}

bool ShearingBox::TestRecvs(const bool coarse, bool &done) {
  bool no_errors = true;
  done = true;
#if MPI_PARALLEL_ENABLED
  const ShearSet st = Set(coarse);
  ShearingBoxBoundaryBuffer *rbuf = coarse ? crecvbuf : recvbuf;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      std::pair<int,int> jsrc[3], jdst[3];
      int jsh[3];
      const int np = Pieces(n, m, st, jsrc, jdst, jsh);
      for (int l=0; l<np; ++l) {
        int sgid, srank;
        FindTargetMB(gid,-jsh[l],sgid,srank);
        if (srank != global_variable::my_rank) {
          int test;
          int ierr = MPI_Test(&(rbuf[n].vars_req[3*m + l]),&test,MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
          if (!(static_cast<bool>(test))) {done = false;}
        }
      }
    }
  }
#endif
  return no_errors;
}

bool ShearingBox::WaitRecvs(const bool coarse) {
  bool no_errors = true;
#if MPI_PARALLEL_ENABLED
  const ShearSet st = Set(coarse);
  ShearingBoxBoundaryBuffer *rbuf = coarse ? crecvbuf : recvbuf;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      std::pair<int,int> jsrc[3], jdst[3];
      int jsh[3];
      const int np = Pieces(n, m, st, jsrc, jdst, jsh);
      for (int l=0; l<np; ++l) {
        int sgid, srank;
        FindTargetMB(gid,-jsh[l],sgid,srank);
        if (srank != global_variable::my_rank) {
          int ierr = MPI_Wait(&(rbuf[n].vars_req[3*m + l]), MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
#endif
  return no_errors;
}

bool ShearingBox::WaitSends(const bool coarse) {
  bool no_errors = true;
#if MPI_PARALLEL_ENABLED
  const ShearSet st = Set(coarse);
  ShearingBoxBoundaryBuffer *sbuf = coarse ? csendbuf : sendbuf;
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      int gid = x1bndry_mbgid.h_view(n,m);
      std::pair<int,int> jsrc[3], jdst[3];
      int jsh[3];
      const int np = Pieces(n, m, st, jsrc, jdst, jsh);
      for (int l=0; l<np; ++l) {
        int tgid, trank;
        FindTargetMB(gid,jsh[l],tgid,trank);
        if (trank != global_variable::my_rank) {
          int ierr = MPI_Wait(&(sbuf[n].vars_req[3*m + l]), MPI_STATUS_IGNORE);
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
#endif
  return no_errors;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::InitRecv
//! \brief Calculates x2-distance that x1-boundaries have sheared.  With MPI, posts
//! non-blocking receives for boundary communications for shearing box boundaries

TaskStatus ShearingBox::InitRecv(Real time) {
  // figure out distance boundaries are sheared
  const auto &mesh_size = pmy_pack->pmesh->mesh_size;
  Real lx = (mesh_size.x1max - mesh_size.x1min);
  yshear = (qshear*omega0)*lx*time;

  // x1-face blocks on several levels (face_levels): collected ghost zones; plan with
  // this yshear and post their receives
  if (face_levels) {
    PlanPull();
    return TaskStatus::complete;
  }
  // post non-blocking receives
  bool no_errors = PostRecvs(false);
  if (coarse_on) {no_errors = PostRecvs(true) && no_errors;}
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void ShearingBox::ClearRecv
//! \brief Waits for all MPI receives associated with communcation for shearing box
//! boundaries to complete before allowing execution to continue

TaskStatus ShearingBox::ClearRecv() {
  if (face_levels) {
    if (!WaitPull()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in clearing (face levels)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return TaskStatus::complete;
  }
  bool no_errors = WaitRecvs(false);
  if (coarse_on) {no_errors = WaitRecvs(true) && no_errors;}
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void ShearingBox::ClearSend
//! \brief Waits for all MPI sends associated with communcation for shearing box
//! boundaries to complete before allowing execution to continue

TaskStatus ShearingBox::ClearSend() {
  if (face_levels) {
    if (!WaitPull()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MPI error in clearing (face levels)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return TaskStatus::complete;
  }
  bool no_errors = WaitSends(false);
  if (coarse_on) {no_errors = WaitSends(true) && no_errors;}
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return TaskStatus::complete;
}
