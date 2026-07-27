//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_part.cpp
//! \brief

#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <Kokkos_Core.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/nghbr_index.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::UpdateGID()
//! \brief Updates GID of particles that cross boundary of their parent MeshBlock.  If
//! the new GID is on a different rank, store the particle index and destination rank in
//! the send list.

KOKKOS_INLINE_FUNCTION
void UpdateGID(int &newgid, NeighborBlock nghbr, int myrank, int *pcounter,
               DvceArray1D<ParticleLocationData> slist, int p) {
  newgid = nghbr.gid;
#if MPI_PARALLEL_ENABLED
  if (nghbr.rank != myrank) {
    int index = Kokkos::atomic_fetch_add(pcounter,1);
    slist(index).prtcl_indx = p;
    slist(index).dest_rank  = nghbr.rank;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::SetNewGID()
//! \brief

TaskStatus ParticlesBoundaryValues::SetNewPrtclGID() {
  // create local references for variables in kernel
  auto gids = pmy_part->pmy_pack->gids;
  auto &pr = pmy_part->prtcl_rdata;
  auto &pi = pmy_part->prtcl_idata;
  int npart = pmy_part->nprtcl_thispack;
  auto &mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto &mblev = pmy_part->pmy_pack->pmb->mb_lev;
  auto &meshsize = pmy_part->pmy_pack->pmesh->mesh_size;
  auto myrank = global_variable::my_rank;
  auto &nghbr = pmy_part->pmy_pack->pmb->nghbr;
  int *pcounter = nullptr;
  bool &multi_d = pmy_part->pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_part->pmy_pack->pmesh->three_d;
  // dust particles carry RK position registers (IPX1/IPY1/IPZ1) that must be shifted by
  // the same box length as the positions when a particle wraps at a periodic boundary
  const bool has_reg = (pmy_part->particle_type == ParticleType::dust);

  // shear-periodic x1 boundaries: particles crossing the radial mesh boundaries are
  // shifted azimuthally by the (folded) shear offset (positions only; velocities are
  // shear-relative), and routed to the boundary MeshBlock found from the shear maps
  const bool shear_x1 = shear_periodic_x1;
  Real ysh = 0.0;
  if (shear_x1) {
    Real lx = (meshsize.x1max - meshsize.x1min);
    Real ly = (meshsize.x2max - meshsize.x2min);
    ysh = fmod(qshear_sp*omega0_sp*lx*(pmy_part->pmy_pack->pmesh->time), ly);
  }
  const int nmbx2 = nmb_sp_x2, nmbx3 = nmb_sp_x3;
  auto &sgid = sgid_map;
  auto &srnk = srank_map;

#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    // Keep a worst-case-capacity send list, but allocate only when the local particle
    // high-water mark grows. nprtcl_send below is the logical valid-prefix length.  The
    // persistent allocation avoids the previous grow-to-npart/shrink-to-nsend churn.
    int required_capacity = std::max(npart, 1);
    if (static_cast<int>(sendlist.extent(0)) < required_capacity) {
      Kokkos::realloc(sendlist, required_capacity);
    }
    Kokkos::deep_copy(send_count.d_view, 0);
    pcounter = send_count.d_view.data();
  }
#endif
  // Capture only the CUDA-accessible view in the kernel. Capturing the DualView (or
  // accessing the sendlist member through `this`) leaves a host pointer in a CUDA
  // lambda and fails as soon as an inter-rank particle is appended or packed.
  auto psendl = sendlist.d_view;
  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int mylevel = mblev.d_view(m);
    Real x1 = pr(IPX,p);
    Real x2 = pr(IPY,p);
    Real x3 = pr(IPZ,p);

    // length of MeshBlock in each direction
    Real lx = (mbsize.d_view(m).x1max - mbsize.d_view(m).x1min);
    Real ly = (mbsize.d_view(m).x2max - mbsize.d_view(m).x2min);
    Real lz = (mbsize.d_view(m).x3max - mbsize.d_view(m).x3min);

    // integer offset of particle relative to center of MeshBlock (-1,0,+1)
    int ix = static_cast<int>((x1 - mbsize.d_view(m).x1min + lx)/lx) - 1;
    int iy = static_cast<int>((x2 - mbsize.d_view(m).x2min + ly)/ly) - 1;
    int iz = static_cast<int>((x3 - mbsize.d_view(m).x3min + lz)/lz) - 1;

    // sublock indices for faces and edges with S/AMR
    int fx = (x1 < 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max))? 0 : 1;
    int fy = (x2 < 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max))? 0 : 1;
    int fz = (x3 < 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max))? 0 : 1;
    fy = multi_d ? fy : 0;
    fz = three_d ? fz : 0;

    // only update particle GID if it has crossed MeshBlock boundary
    if ((abs(ix) + abs(iy) + abs(iz)) != 0) {
      bool cross_in  = shear_x1 && (x1 < meshsize.x1min);
      bool cross_out = shear_x1 && (x1 > meshsize.x1max);
      if (cross_in || cross_out) {
        // shear-periodic radial crossing: wrap x, shift y azimuthally by the (folded)
        // shear offset with no velocity change (velocities are shear-relative), wrap
        // y/z periodically, then route to the boundary MeshBlock at the new (y,z).
        // The RK position registers are shifted identically.
        Real lxm = (meshsize.x1max - meshsize.x1min);
        pr(IPX,p) += cross_in ? lxm : -lxm;
        if (has_reg) {pr(IPX1,p) += cross_in ? lxm : -lxm;}
        Real lym = (meshsize.x2max - meshsize.x2min);
        Real ynew = pr(IPY,p) + (cross_in ? -ysh : ysh);
        if (ynew < meshsize.x2min) {ynew += lym;}
        if (ynew >= meshsize.x2max) {ynew -= lym;}
        Real dy = ynew - pr(IPY,p);
        pr(IPY,p) = ynew;
        if (has_reg) {pr(IPY1,p) += dy;}
        Real lzm = (meshsize.x3max - meshsize.x3min);
        if (x3 < meshsize.x3min) {
          pr(IPZ,p) += lzm;
          if (has_reg) {pr(IPZ1,p) += lzm;}
        } else if (x3 > meshsize.x3max) {
          pr(IPZ,p) -= lzm;
          if (has_reg) {pr(IPZ1,p) -= lzm;}
        }
        // destination MeshBlock on the opposite radial side at the new (y,z)
        int b2 = static_cast<int>(((pr(IPY,p) - meshsize.x2min)/lym)
                                  *static_cast<Real>(nmbx2));
        b2 = (b2 < 0) ? 0 : ((b2 > nmbx2-1) ? (nmbx2-1) : b2);
        int b3 = 0;
        if (three_d) {
          b3 = static_cast<int>(((pr(IPZ,p) - meshsize.x3min)/lzm)
                                *static_cast<Real>(nmbx3));
          b3 = (b3 < 0) ? 0 : ((b3 > nmbx3-1) ? (nmbx3-1) : b3);
        }
        NeighborBlock nb;
        nb.gid = sgid.d_view((cross_in ? 1 : 0), b3, b2);
        nb.lev = mylevel;
        nb.rank = srnk.d_view((cross_in ? 1 : 0), b3, b2);
        nb.dest = 0;
        UpdateGID(pi(PGID,p), nb, myrank, pcounter, psendl, p);
      } else {
      if (iz == 0) {
        if (iy == 0) {
          // x1 face
          int indx = NeighborIndex(ix,0,0,0,0);           // neighbor at same level
          if (nghbr.d_view(m,indx).lev > mylevel) {       // neighbor at finer level
            indx = NeighborIndex(ix,0,0,fy,fz);
          }
          while (nghbr.d_view(m,indx).gid < 0) {indx++;}  // neighbor at coarser level
          UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);
        } else if (ix == 0) {
          // x2 face
          int indx = NeighborIndex(0,iy,0,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(0,iy,0,fx,fz);
          }
          while (nghbr.d_view(m,indx).gid < 0) {indx++;}
          UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);
        } else {
          // x1x2 edge
          int indx = NeighborIndex(ix,iy,0,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(ix,iy,0,fz,0);
          }
          while (nghbr.d_view(m,indx).gid < 0) {indx++;}
          UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);
        }
      } else if (iy == 0) {
        if (ix == 0) {
          // x3 face
          int indx = NeighborIndex(0,0,iz,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(0,0,iz,fx,fy);
          }
          while (nghbr.d_view(m,indx).gid < 0) {indx++;}
          UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);
        } else {
          // x3x1 edge
          int indx = NeighborIndex(ix,0,iz,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(ix,0,iz,fy,0);
          }
          while (nghbr.d_view(m,indx).gid < 0) {indx++;}
          UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);
        }
      } else {
        if (ix == 0) {
          // x2x3 edge
          int indx = NeighborIndex(0,iy,iz,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(0,iy,iz,fx,0);
          }
          while (nghbr.d_view(m,indx).gid < 0) {indx++;}
          UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);
        } else {
          // corners
          int indx = NeighborIndex(ix,iy,iz,0,0);
          UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);
        }
      }

      // reset x,y,z positions if particle crosses Mesh boundary using periodic BCs
      // RK position registers must be shifted with the position so the low-storage
      // combination (gam0*x + gam1*x1) stays in a single periodic image
      if (x1 < meshsize.x1min) {
        pr(IPX,p) += (meshsize.x1max - meshsize.x1min);
        if (has_reg) {pr(IPX1,p) += (meshsize.x1max - meshsize.x1min);}
      } else if (x1 > meshsize.x1max) {
        pr(IPX,p) -= (meshsize.x1max - meshsize.x1min);
        if (has_reg) {pr(IPX1,p) -= (meshsize.x1max - meshsize.x1min);}
      }
      if (x2 < meshsize.x2min) {
        pr(IPY,p) += (meshsize.x2max - meshsize.x2min);
        if (has_reg) {pr(IPY1,p) += (meshsize.x2max - meshsize.x2min);}
      } else if (x2 > meshsize.x2max) {
        pr(IPY,p) -= (meshsize.x2max - meshsize.x2min);
        if (has_reg) {pr(IPY1,p) -= (meshsize.x2max - meshsize.x2min);}
      }
      if (x3 < meshsize.x3min) {
        pr(IPZ,p) += (meshsize.x3max - meshsize.x3min);
        if (has_reg) {pr(IPZ1,p) += (meshsize.x3max - meshsize.x3min);}
      } else if (x3 > meshsize.x3max) {
        pr(IPZ,p) -= (meshsize.x3max - meshsize.x3min);
        if (has_reg) {pr(IPZ1,p) -= (meshsize.x3max - meshsize.x3min);}
      }
      }  // end shear/standard crossing branch
    }
  });
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    // This deep copy also synchronizes completion of the device append kernel.  Copy only
    // the valid prefix to the host; DualView::sync() would copy the full capacity.
    Kokkos::deep_copy(send_count.h_view, send_count.d_view);
    nprtcl_send = send_count.h_view(0);
    if (nprtcl_send < 0 || nprtcl_send > static_cast<int>(sendlist.extent(0))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Invalid particle send count " << nprtcl_send
                << " on rank " << global_variable::my_rank << " (capacity "
                << sendlist.extent(0) << ")" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (nprtcl_send > 0) {
      auto d_prefix = Kokkos::subview(sendlist.d_view,
                                      std::make_pair(0, nprtcl_send));
      auto h_prefix = Kokkos::subview(sendlist.h_view,
                                      std::make_pair(0, nprtcl_send));
      Kokkos::deep_copy(h_prefix, d_prefix);
      for (int n=0; n<nprtcl_send; ++n) {
        int p = sendlist.h_view(n).prtcl_indx;
        int r = sendlist.h_view(n).dest_rank;
        if (p < 0 || p >= npart || r < 0 || r >= global_variable::nranks) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Invalid particle send-list entry " << n
                    << " on rank " << global_variable::my_rank << ": particle=" << p
                    << ", destination rank=" << r << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }
  } else {
    nprtcl_send = 0;
  }
#else
  nprtcl_send = 0;
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::CountSendsAndRecvs()
//! \brief

TaskStatus ParticlesBoundaryValues::CountSendsAndRecvs() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) {
    nprtcl_send = 0;
    nprtcl_recv = 0;
    nsends = 0;
    nrecvs = 0;
    sends_thisrank.clear();
    recvs_thisrank.clear();
    sends_allranks.clear();
    return TaskStatus::complete;
  }

  // Sort sendlist on host by destrank.
  std::sort(sendlist.h_view.data(), sendlist.h_view.data() + nprtcl_send, SortByRank);
  // Copy only the sorted valid prefix back to the device.
  if (nprtcl_send > 0) {
    auto h_prefix = Kokkos::subview(sendlist.h_view,
                                    std::make_pair(0, nprtcl_send));
    auto d_prefix = Kokkos::subview(sendlist.d_view,
                                    std::make_pair(0, nprtcl_send));
    Kokkos::deep_copy(d_prefix, h_prefix);
  }

  // load STL::vector of ParticleMessageData with <sendrank, recvrank, nprtcls> for sends
  // from this rank. Length will be nsends; initially this length is unknown
  sends_thisrank.clear();
  if (nprtcl_send > 0) {
    int &myrank = global_variable::my_rank;
    int rank = sendlist.h_view(0).dest_rank;
    int nprtcl = 1;

    for (int n=1; n<nprtcl_send; ++n) {
      if (sendlist.h_view(n).dest_rank == rank) {
        ++nprtcl;
      } else {
        sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
        rank = sendlist.h_view(n).dest_rank;
        nprtcl = 1;
      }
    }
    sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
  }
  nsends = sends_thisrank.size();

  // Share number of ranks to send to amongst all ranks
  nsends_eachrank[global_variable::my_rank] = nsends;
  MPI_Allgather(&nsends, 1, MPI_INT, nsends_eachrank.data(), 1, MPI_INT, mpi_comm_part);

  // Now share ParticleMessageData amongst all ranks
  // First create vector of starting indices in full vector
  std::vector<int> nsends_displ;
  nsends_displ.resize(global_variable::nranks);
  nsends_displ[0] = 0;
  for (int n=1; n<(global_variable::nranks); ++n) {
    nsends_displ[n] = nsends_displ[n-1] + nsends_eachrank[n-1];
  }
  int nsends_allranks = nsends_displ[global_variable::nranks - 1] +
                        nsends_eachrank[global_variable::nranks - 1];
  // Load ParticleMessageData on this rank into full vector
  sends_allranks.resize(nsends_allranks, ParticleMessageData(0,0,0));
  for (int n=0; n<nsends_eachrank[global_variable::my_rank]; ++n) {
    sends_allranks[n + nsends_displ[global_variable::my_rank]] = sends_thisrank[n];
  }

  // Share tuples using MPI derived data type for tuple of 3*int
  MPI_Datatype mpi_ituple;
  MPI_Type_contiguous(3, MPI_INT, &mpi_ituple);
  MPI_Type_commit(&mpi_ituple);
  MPI_Allgatherv(MPI_IN_PLACE, nsends_eachrank[global_variable::my_rank],
                   mpi_ituple, sends_allranks.data(), nsends_eachrank.data(),
                   nsends_displ.data(), mpi_ituple, mpi_comm_part);
  MPI_Type_free(&mpi_ituple);
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::InitPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::InitPrtclRecv() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) {return TaskStatus::complete;}

  // load STL::vector of ParticleMessageData with <sendrank,recvrank,nprtcl_recv> for
  // receives // on this rank. Length will be nrecvs, initially this length is unknown
  recvs_thisrank.clear();

  int nsends_allranks = sends_allranks.size();
  for (int n=0; n<nsends_allranks; ++n) {
    if (sends_allranks[n].recvrank == global_variable::my_rank) {
      recvs_thisrank.emplace_back(sends_allranks[n]);
    }
  }
  nrecvs = recvs_thisrank.size();

  // Figure out how many particles will be received from all ranks
  nprtcl_recv=0;
  for (int n=0; n<nrecvs; ++n) {
    nprtcl_recv += recvs_thisrank[n].nprtcls;
  }

  // Retain high-water capacities across migration stages.  MPI uses only the valid
  // prefixes described by nprtcl_recv, so no buffer contents need to be preserved.
  std::size_t rrecv_required = static_cast<std::size_t>(pmy_part->nrdata)*nprtcl_recv;
  std::size_t irecv_required = static_cast<std::size_t>(pmy_part->nidata)*nprtcl_recv;
  if (prtcl_rrecvbuf.extent(0) < rrecv_required) {
    Kokkos::realloc(prtcl_rrecvbuf, rrecv_required);
  }
  if (prtcl_irecvbuf.extent(0) < irecv_required) {
    Kokkos::realloc(prtcl_irecvbuf, irecv_required);
  }

  // Post non-blocking receives
  bool no_errors=true;
  rrecv_req.clear();
  irecv_req.clear();
  for (int n=0; n<nrecvs; ++n) {
    rrecv_req.emplace_back(MPI_REQUEST_NULL);
    irecv_req.emplace_back(MPI_REQUEST_NULL);
  }

  // Init receives for Reals
  int data_start=0;
  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = (pmy_part->nrdata)*(recvs_thisrank[n].nprtcls);
    int data_end = data_start + data_size;
    auto recv_ptr = Kokkos::subview(prtcl_rrecvbuf, std::make_pair(data_start, data_end));
    int drank = recvs_thisrank[n].sendrank;
    int tag = 0; // 0 for Reals, 1 for ints

    // Post non-blocking receive
    int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                         mpi_comm_part, &(rrecv_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    data_start += data_size;
  }
  // Init receives for ints
  data_start=0;
  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed, get pointer to variables
    int data_size = (pmy_part->nidata)*(recvs_thisrank[n].nprtcls);
    int data_end = data_start + data_size;
    auto recv_ptr = Kokkos::subview(prtcl_irecvbuf, std::make_pair(data_start, data_end));
    int drank = recvs_thisrank[n].sendrank;
    int tag = 1; // 0 for Reals, 1 for ints

    // Post non-blocking receive
    int ierr = MPI_Irecv(recv_ptr.data(), data_size, MPI_INT, drank, tag,
                         mpi_comm_part, &(irecv_req[n]));
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    data_start += data_size;
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
//! \fn void ParticlesBoundaryValues::PackAndSendPrtcls()
//! \brief

TaskStatus ParticlesBoundaryValues::PackAndSendPrtcls() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) {return TaskStatus::complete;}

  // Figure out how many particles will be sent from this ranks
  nprtcl_send=0;
  for (int n=0; n<nsends; ++n) {
    nprtcl_send += sends_thisrank[n].nprtcls;
  }

  bool no_errors=true;
  if (nprtcl_send > 0) {
    // Retain high-water capacities across migration stages.  Only the current valid
    // prefixes are packed and passed to MPI.
    std::size_t rsend_required = static_cast<std::size_t>(pmy_part->nrdata)*nprtcl_send;
    std::size_t isend_required = static_cast<std::size_t>(pmy_part->nidata)*nprtcl_send;
    if (prtcl_rsendbuf.extent(0) < rsend_required) {
      Kokkos::realloc(prtcl_rsendbuf, rsend_required);
    }
    if (prtcl_isendbuf.extent(0) < isend_required) {
      Kokkos::realloc(prtcl_isendbuf, isend_required);
    }

    // sendlist on device is already sorted by destrank in CountSendAndRecvs()
    // Use sendlist on device to load particles into send buffer ordered by dest_rank
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rsendbuf = prtcl_rsendbuf;
    auto &isendbuf = prtcl_isendbuf;
    auto sendlist_d = sendlist.d_view;
    par_for("ppack",DevExeSpace(),0,(nprtcl_send-1), KOKKOS_LAMBDA(const int n) {
      int p = sendlist_d(n).prtcl_indx;
      for (int i=0; i<nidata; ++i) {
        isendbuf(nidata*n + i) = pi(i,p);
      }
      for (int i=0; i<nrdata; ++i) {
        rsendbuf(nrdata*n + i) = pr(i,p);
      }
    });

    // Post non-blocking sends
    Kokkos::fence();
    rsend_req.clear();
    isend_req.clear();
    for (int n=0; n<nsends; ++n) {
      rsend_req.emplace_back(MPI_REQUEST_NULL);
      isend_req.emplace_back(MPI_REQUEST_NULL);
    }

    // Send Reals
    int data_start=0;
    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed, get pointer to variables
      int data_size = nrdata*(sends_thisrank[n].nprtcls);
      int data_end = data_start + data_size;
      auto send_ptr = Kokkos::subview(prtcl_rsendbuf,std::make_pair(data_start,data_end));
      int drank = sends_thisrank[n].recvrank;
      int tag = 0; // 0 for Reals, 1 for ints

      // Post non-blocking sends
      int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_ATHENA_REAL, drank, tag,
                           mpi_comm_part, &(rsend_req[n]));
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      data_start += data_size;
    }
    // Send ints
    data_start=0;
    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed, get pointer to variables
      int data_size = nidata*(sends_thisrank[n].nprtcls);
      int data_end = data_start + data_size;
      auto send_ptr = Kokkos::subview(prtcl_isendbuf,std::make_pair(data_start,data_end));
      int drank = sends_thisrank[n].recvrank;
      int tag = 1; // 0 for Reals, 1 for ints

      // Post non-blocking sends
      int ierr = MPI_Isend(send_ptr.data(), data_size, MPI_INT, drank, tag,
                           mpi_comm_part, &(isend_req[n]));
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      data_start += data_size;
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
//! \fn void ParticlesBoundaryValues::RecvAndUnpackPrtcls()
//! \brief

TaskStatus ParticlesBoundaryValues::RecvAndUnpackPrtcls() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) {return TaskStatus::complete;}

  // check that particle communications have all completed
  bool bflag = false;
  bool no_errors=true;
  for (int n=0; n<nrecvs; ++n) {
    int test;
    int ierr = MPI_Test(&(rrecv_req[n]), &test, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    if (!(static_cast<bool>(test))) {
      bflag = true;
    }
    ierr = MPI_Test(&(irecv_req[n]), &test, MPI_STATUS_IGNORE);
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
  // exit if particle communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  // Sort the valid send-list prefix by particle index.  The device copy is used both to
  // place received particles in the first holes and to pack the compaction move list.
  std::sort(sendlist.h_view.data(), sendlist.h_view.data() + nprtcl_send, SortByIndex);
  if (nprtcl_send > 0) {
    auto h_prefix = Kokkos::subview(sendlist.h_view,
                                    std::make_pair(0, nprtcl_send));
    auto d_prefix = Kokkos::subview(sendlist.d_view,
                                    std::make_pair(0, nprtcl_send));
    Kokkos::deep_copy(d_prefix, h_prefix);
  }

  const int old_npart = pmy_part->nprtcl_thispack;
  const int new_npart = old_npart + (nprtcl_recv - nprtcl_send);
  if (new_npart < 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Particle migration produced a negative local count "
              << new_npart << " on rank " << global_variable::my_rank << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // The logical particle count is independent of array capacity.  Grow capacity by 1.5x
  // when needed and retain the high-water allocation; migration never shrinks the views.
  std::size_t rcapacity = pmy_part->prtcl_rdata.extent(1);
  std::size_t icapacity = pmy_part->prtcl_idata.extent(1);
  if (static_cast<std::size_t>(new_npart) > rcapacity ||
      static_cast<std::size_t>(new_npart) > icapacity) {
    std::size_t old_capacity = std::max(rcapacity, icapacity);
    std::size_t grown_capacity =
        old_capacity + std::max<std::size_t>(old_capacity/2, 1);
    std::size_t new_capacity =
        std::max<std::size_t>(static_cast<std::size_t>(new_npart), grown_capacity);
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, new_capacity);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, new_capacity);
  }

  // unpack particles into positions of sent particles
  if (nprtcl_recv > 0) {
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rrecvbuf = prtcl_rrecvbuf;
    auto &irecvbuf = prtcl_irecvbuf;
    auto sendlist_d = sendlist.d_view;
    int nsend = nprtcl_send;
    int npart = old_npart;
    par_for("punpack",DevExeSpace(),0,(nprtcl_recv-1), KOKKOS_LAMBDA(const int n) {
      int p;
      if (n < nsend) {
        p = sendlist_d(n).prtcl_indx; // place particles in holes created by sends
      } else {
        p = npart + (n - nsend);           // place particle at end of arrays
      }
      for (int i=0; i<nidata; ++i) {
        pi(i,p) = irecvbuf(nidata*n + i);
      }
      for (int i=0; i<nrdata; ++i) {
        pr(i,p) = rrecvbuf(nrdata*n + i);
      }
    });
  }

  // At this point have filled npart_recv holes in particle arrays from sends
  // If (nprtcl_recv < nprtcl_send), have to move particles from end of arrays to fill
  // remaining holes
  int nremain = nprtcl_send - nprtcl_recv;
  if (nremain > 0) {
    // Build the source/destination mapping on the host.  Sources are live particles in
    // [new_npart, old_npart), while destinations are unfilled holes below new_npart, so
    // every move is independent and can run in one device kernel.
    std::size_t old_move_capacity = compactlist.extent(0);
    if (static_cast<std::size_t>(nremain) > old_move_capacity) {
      std::size_t grown_capacity =
          old_move_capacity + std::max<std::size_t>(old_move_capacity/2, 1);
      std::size_t new_capacity =
          std::max<std::size_t>(static_cast<std::size_t>(nremain), grown_capacity);
      Kokkos::realloc(compactlist, new_capacity);
    }

    int nmove = 0;
    int i_last_hole = nprtcl_send-1;
    int i_next_hole = nprtcl_recv;
    for (int n=1; n<=nremain; ++n) {
      int nend = old_npart-n;
      if (nend > sendlist.h_view(i_last_hole).prtcl_indx) {
        int next_hole = sendlist.h_view(i_next_hole).prtcl_indx;
        if (next_hole >= new_npart || nend < new_npart) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "Invalid particle compaction move " << nend << " -> "
                    << next_hole << " on rank " << global_variable::my_rank << std::endl;
          std::exit(EXIT_FAILURE);
        }
        compactlist.h_view(nmove).src = nend;
        compactlist.h_view(nmove).dst = next_hole;
        ++nmove;
        i_next_hole += 1;
      } else {
        // this index contains a hole, so do nothing except find new index of last hole
        i_last_hole -= 1;
      }
    }

    if (nmove > 0) {
      auto h_prefix = Kokkos::subview(compactlist.h_view, std::make_pair(0, nmove));
      auto d_prefix = Kokkos::subview(compactlist.d_view, std::make_pair(0, nmove));
      Kokkos::deep_copy(d_prefix, h_prefix);

      int nrdata = pmy_part->nrdata;
      int nidata = pmy_part->nidata;
      auto &pr = pmy_part->prtcl_rdata;
      auto &pi = pmy_part->prtcl_idata;
      auto moves = compactlist.d_view;
      par_for("pcompact",DevExeSpace(),0,(nmove-1), KOKKOS_LAMBDA(const int n) {
        int src = moves(n).src;
        int dst = moves(n).dst;
        for (int i=0; i<nidata; ++i) {
          pi(i,dst) = pi(i,src);
        }
        for (int i=0; i<nrdata; ++i) {
          pr(i,dst) = pr(i,src);
        }
      });
    }
  }

  // Update only the local logical count.  Per-rank counts are synchronized when a
  // particle output needs them rather than on every migration stage.
  pmy_part->nprtcl_thispack = new_npart;
  pmy_part->pmy_pack->pmesh->nprtcl_thisrank = new_npart;
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearPrtclSend()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearPrtclSend() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) {
    nsends = 0;
    return TaskStatus::complete;
  }

  bool no_errors=true;
  // wait for all non-blocking sends for vars to finish before continuing
  for (int n=0; n<nsends; ++n) {
    int ierr = MPI_Wait(&(rsend_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    ierr = MPI_Wait(&(isend_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rsend_req.clear();
  isend_req.clear();
#endif
  nsends=0;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearPrtclRecv() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) {
    nrecvs = 0;
    return TaskStatus::complete;
  }

  bool no_errors=true;
  // wait for all non-blocking receives to finish before continuing
  for (int n=0; n<nrecvs; ++n) {
    int ierr = MPI_Wait(&(rrecv_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
    ierr = MPI_Wait(&(irecv_req[n]), MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {no_errors=false;}
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rrecv_req.clear();
  irecv_req.clear();
#endif
  nrecvs=0;
  return TaskStatus::complete;
}

} // namespace particles
