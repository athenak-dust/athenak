//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection_tasks.cpp
//! \brief functions included in task lists to post/clear non-blocking MPI calls for
//! orbital advection

#include <cstdlib>
#include <iostream>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "shearing_box.hpp"
#include "orbital_advection.hpp"

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::InitRecv
//! \brief Posts non-blocking receives (with MPI) for boundary communications with
//! orbital advection

TaskStatus OrbitalAdvection::InitRecv() {
#if MPI_PARALLEL_ENABLED
  const int &nmb = pmy_pack->nmb_thispack;
  const auto &nghbr = pmy_pack->pmb->nghbr;

  // Initialize communications of variables
  bool no_errors=true;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if (nghbr.h_view(m,nnghbr).gid >= 0 && xlev.h_view(m,n) == 0) {
        // rank of neighboring MeshBlock sending data
        int srank = nghbr.h_view(m,nnghbr).rank;

        // post non-blocking receive if neighboring MeshBlock on a different rank
        if (srank != global_variable::my_rank) {
          // create tag using local ID and buffer index of *receiving* MeshBlock
          int tag = CreateBvals_MPI_Tag(m, nnghbr);

          // get pointer to variables
          using Kokkos::ALL;
          auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, ALL, ALL, ALL, ALL);
          int data_size = recv_ptr.size();

          // Post non-blocking receive for this buffer on this MeshBlock
          int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, srank, tag,
                               comm_orb_advect, &(recvbuf[n].vars_req[m]));
          if (ierr != MPI_SUCCESS) {no_errors=false;}
        }
      }
    }
  }
  // x2 faces with a level change: the rows from a coarser neighbour (into the
  // receive buffer itself), from four finer neighbours (one message each) and their fluxes
  if (any_mixed) {
    const int my_rank = global_variable::my_rank;
    const int nk = pmy_pack->pmesh->mb_indcs.nx3, ni = pmy_pack->pmesh->mb_indcs.nx1;
    const int sfc = nvar_oa*(nk/2)*nrow*(ni/2), sff = nvar_oa*(nk/2)*(ni/2);
    for (int m=0; m<nmb; ++m) {
      for (int n=0; n<2; ++n) {
        const int base = (n == 0) ? 8 : 12;
        if (xlev.h_view(m,n) == 1) {
          const int slot = xslot.h_view(m,n);
          const int srank = nghbr.h_view(m,slot).rank;
          if (srank != my_rank) {
            using Kokkos::ALL;
            auto recv_ptr = Kokkos::subview(recvbuf[n].vars, m, ALL, ALL, ALL, ALL);
            int ierr = MPI_Irecv(recv_ptr.data(), recv_ptr.size(), MPI_ATHENA_REAL, srank,
                                 CreateBvals_MPI_Tag(m, slot), comm_orb_advect,
                                 &(mrreq[n][4*m]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        } else if (xlev.h_view(m,n) == 2) {
          for (int sf=0; sf<4; ++sf) {
            const int srank = nghbr.h_view(m,base+sf).rank;
            if (srank == my_rank) continue;
            int ierr = MPI_Irecv(Kokkos::subview(mrecv[n], m, sf, Kokkos::ALL).data(), sfc,
                                 MPI_ATHENA_REAL, srank, CreateBvals_MPI_Tag(m, base+sf),
                                 comm_orb_advect, &(mrreq[n][4*m+sf]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
            if (!oa_flux_correct) continue;
            ierr = MPI_Irecv(Kokkos::subview(frecv[n], m, sf, Kokkos::ALL).data(), sff,
                             MPI_ATHENA_REAL, srank, CreateBvals_MPI_Tag(m, 16+base+sf),
                             comm_orb_advect, &(frreq[n][4*m+sf]));
            if (ierr != MPI_SUCCESS) {no_errors=false;}
          }
        }
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void OrbitalAdvection::ClearRecv
//! \brief Waits for all MPI receives associated with communcation with orbital
//! advection to complete before allowing execution to continue

TaskStatus OrbitalAdvection::ClearRecv() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  int &nmb = pmy_pack->nmb_thispack;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking receives for vars to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if ( (nghbr.h_view(m,nnghbr).gid >= 0) && xlev.h_view(m,n) == 0 &&
           (nghbr.h_view(m,nnghbr).rank != global_variable::my_rank) ) {
        int ierr = MPI_Wait(&(recvbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  // the mixed-level receives (null requests return at once)
  if (any_mixed) {
    for (int n=0; n<2; ++n) {
      for (int r=0; r<4*nmb; ++r) {
        if (MPI_Wait(&(mrreq[n][r]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
        if (MPI_Wait(&(frreq[n][r]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn  void OrbitalAdvection::ClearSend
//! \brief Waits for all MPI sends associated with communcation of boundary variables
//! to complete before allowing execution to continue

TaskStatus OrbitalAdvection::ClearSend() {
#if MPI_PARALLEL_ENABLED
  bool no_errors=true;
  int &nmb = pmy_pack->nmb_thispack;
  auto &nghbr = pmy_pack->pmb->nghbr;

  // wait for all non-blocking sends for vars to finish before continuing
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      // indices of x2-face buffers in nghbr view
      int nnghbr;
      if (n==0) {nnghbr=8;} else {nnghbr=12;}
      if ( (nghbr.h_view(m,nnghbr).gid >= 0) && xlev.h_view(m,n) == 0 &&
           (nghbr.h_view(m,nnghbr).rank != global_variable::my_rank) ) {
        int ierr = MPI_Wait(&(sendbuf[n].vars_req[m]), MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {no_errors=false;}
      }
    }
  }
  // the mixed-level sends
  if (any_mixed) {
    for (int n=0; n<2; ++n) {
      for (int r=0; r<4*nmb; ++r) {
        if (MPI_Wait(&(msreq[n][r]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
        if (MPI_Wait(&(fsreq[n][r]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}
