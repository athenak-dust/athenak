//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals.cpp
//! \brief constructors and initializers for both particle and Mesh variable boundary
//! classes.

#include <cstdlib>
#include <iostream>
#include <utility>
#include <algorithm> // max

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/nghbr_index.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"

//----------------------------------------------------------------------------------------
// MeshBoundaryValues constructor:

MeshBoundaryValues::MeshBoundaryValues(MeshBlockPack *pp, ParameterInput *pin, bool z4c) :
  u_in("uin",1,1),
  b_in("bin",1,1),
  i_in("iin",1,1),
  pmy_pack(pp),
  is_z4c_(z4c) {
  // allocate vector of status flags and MPI requests (if needed)
  int nnghbr = pmy_pack->pmb->nnghbr;

#if MPI_PARALLEL_ENABLED
  // Initialize all 56 MPI request pointers to nullptr first
  for (int n=0; n<56; ++n) {
    sendbuf[n].vars_req = nullptr;
    sendbuf[n].flux_req = nullptr;
    recvbuf[n].vars_req = nullptr;
    recvbuf[n].flux_req = nullptr;
  }
#endif

  // sendbuf and recvbuf are fixed-length [56-element] arrays
  // Initialize some of the data in appropriate elements based on dimensionality of
  // problem (indicated by value of nnghbr)
  for (int n=0; n<nnghbr; ++n) {
#if MPI_PARALLEL_ENABLED
    // allocate vector of MPI requests (if needed)
    int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
    sendbuf[n].vars_req = new MPI_Request[nmb];
    sendbuf[n].flux_req = new MPI_Request[nmb];
    recvbuf[n].vars_req = new MPI_Request[nmb];
    recvbuf[n].flux_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      sendbuf[n].flux_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].flux_req[m] = MPI_REQUEST_NULL;
    }
#endif
    // initialize data sizes in each send/recv buffer to zero
    sendbuf[n].isame_ndat = 0;
    sendbuf[n].isame_z4c_ndat = 0;
    sendbuf[n].icoar_ndat = 0;
    sendbuf[n].ifine_ndat = 0;
    sendbuf[n].iflxs_ndat = 0;
    sendbuf[n].iflxc_ndat = 0;
    recvbuf[n].isame_ndat = 0;
    recvbuf[n].isame_z4c_ndat = 0;
    recvbuf[n].icoar_ndat = 0;
    recvbuf[n].ifine_ndat = 0;
    recvbuf[n].iflxs_ndat = 0;
    recvbuf[n].iflxc_ndat = 0;
  }

#if MPI_PARALLEL_ENABLED
  // create unique communicators for variables and fluxes in this BoundaryValues object
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_vars);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_flux);
#endif
}

//----------------------------------------------------------------------------------------
// MeshBoundaryValues destructor

MeshBoundaryValues::~MeshBoundaryValues() {
#if MPI_PARALLEL_ENABLED
  int nnghbr = pmy_pack->pmb->nnghbr;
  for (int n=0; n<nnghbr; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] sendbuf[n].flux_req;
    delete [] recvbuf[n].vars_req;
    delete [] recvbuf[n].flux_req;
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBoundaryValues::InitializeBuffers
//! \brief initialize each element of send/recv MeshBoundaryBuffers fixed-length arrays
//!
//! NOTE: order of vector elements is crucial and cannot be changed.  It must match
//! order of boundaries in nghbr vector
//! NOTE2: work here cannot be done in MeshBoundaryValues constructor since it calls pure
//! virtual functions that only get instantiated when the derived classes are constructed

void MeshBoundaryValues::InitializeBuffers(const int nvar) {
  // allocate memory for inflow BCs (but only if domain not strictly periodic)
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    Kokkos::realloc(u_in, nvar, 6);
    Kokkos::realloc(b_in, 3, 6);   // always 3 components of face-fields
    Kokkos::realloc(i_in, nvar, 6);
  }

  // set number of subblocks in x2- and x3-dirs
  int nfx = 1, nfy = 1, nfz = 1;
  if (pmy_pack->pmesh->multilevel) {
    nfx = 2;
    if (pmy_pack->pmesh->multi_d) nfy = 2;
    if (pmy_pack->pmesh->three_d) nfz = 2;
  }

  // initialize buffers used for uniform grid and SMR/AMR calculations

  // x1 faces; NeighborIndex = [0,...,7]
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  for (int n=-1; n<=1; n+=2) {
    for (int fz=0; fz<nfz; fz++) {
      for (int fy = 0; fy<nfy; fy++) {
        int indx = NeighborIndex(n,0,0,fy,fz);
        InitSendIndices(sendbuf[indx],n, 0, 0, fy, fz);
        InitRecvIndices(recvbuf[indx],n, 0, 0, fy, fz);
        sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        indx++;
      }
    }
  }

  // add more buffers in 2D
  if (pmy_pack->pmesh->multi_d) {
    // x2 faces; NeighborIndex = [8,...,15]
    for (int m=-1; m<=1; m+=2) {
      for (int fz=0; fz<nfz; fz++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,m,0,fx,fz);
          InitSendIndices(sendbuf[indx],0, m, 0, fx, fz);
          InitRecvIndices(recvbuf[indx],0, m, 0, fx, fz);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x1x2 edges; NeighborIndex = [16,...,23]
    for (int m=-1; m<=1; m+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fz=0; fz<nfz; fz++) {
          int indx = NeighborIndex(n,m,0,fz,0);
          InitSendIndices(sendbuf[indx],n, m, 0, fz, 0);
          InitRecvIndices(recvbuf[indx],n, m, 0, fz, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }
  }

  // add more buffers in 3D
  if (pmy_pack->pmesh->three_d) {
    // x3 faces; NeighborIndex = [24,...,31]
    for (int l=-1; l<=1; l+=2) {
      for (int fy=0; fy<nfy; fy++) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,0,l,fx,fy);
          InitSendIndices(sendbuf[indx],0, 0, l, fx, fy);
          InitRecvIndices(recvbuf[indx],0, 0, l, fx, fy);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x3x1 edges; NeighborIndex = [32,...,39]
    for (int l=-1; l<=1; l+=2) {
      for (int n=-1; n<=1; n+=2) {
        for (int fy=0; fy<nfy; fy++) {
          int indx = NeighborIndex(n,0,l,fy,0);
          InitSendIndices(sendbuf[indx],n, 0, l, fy, 0);
          InitRecvIndices(recvbuf[indx],n, 0, l, fy, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // x2x3 edges; NeighborIndex = [40,...,47]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int fx=0; fx<nfx; fx++) {
          int indx = NeighborIndex(0,m,l,fx,0);
          InitSendIndices(sendbuf[indx],0, m, l, fx, 0);
          InitRecvIndices(recvbuf[indx],0, m, l, fx, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          indx++;
        }
      }
    }

    // corners; NeighborIndex = [48,...,55]
    for (int l=-1; l<=1; l+=2) {
      for (int m=-1; m<=1; m+=2) {
        for (int n=-1; n<=1; n+=2) {
          int indx = NeighborIndex(n,m,l,0,0);
          InitSendIndices(sendbuf[indx],n, m, l, 0, 0);
          InitRecvIndices(recvbuf[indx],n, m, l, 0, 0);
          sendbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
          recvbuf[indx].AllocateBuffers(nmb, nvar, is_z4c_);
        }
      }
    }
  }

  return;
}

//----------------------------------------------------------------------------------------
// ParticlesBoundaryValues constructor:

particles::ParticlesBoundaryValues::ParticlesBoundaryValues(
  particles::Particles *pp, ParameterInput *pin) :
    sendlist("sendlist",1),
    sgid_map("sgid_map",1,1,1),
    srank_map("srank_map",1,1,1),
#if MPI_PARALLEL_ENABLED
    prtcl_rsendbuf("rsend",1),
    prtcl_rrecvbuf("rrecv",1),
    prtcl_isendbuf("isend",1),
    prtcl_irecvbuf("irecv",1),
#endif
    pmy_part(pp) {
#if MPI_PARALLEL_ENABLED
  //resize vectors over number of ranks
  nsends_eachrank.resize(global_variable::nranks);

  // create unique communicator for particles
  MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm_part);
#endif

  // With shear-periodic x1 boundaries, build the (side, lx3, lx2) -> gid/rank maps of
  // the MeshBlocks adjacent to the two radial boundaries, used to route particles that
  // cross them (their azimuthal shear shift changes the destination MeshBlock).
  Mesh *pmesh = pp->pmy_pack->pmesh;
  shear_periodic_x1 =
      (pmesh->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic);
  if (shear_periodic_x1) {
    if (pmesh->multilevel) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particles with shear-periodic boundaries do not "
                << "support SMR/AMR" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    qshear_sp = pin->GetReal("shearing_box","qshear");
    omega0_sp = pin->GetReal("shearing_box","omega0");
    auto &mindcs = pmesh->mesh_indcs;
    auto &bindcs = pmesh->mb_indcs;
    int nmbx1 = mindcs.nx1/bindcs.nx1;
    nmb_sp_x2 = std::max(mindcs.nx2/bindcs.nx2, 1);
    nmb_sp_x3 = std::max(mindcs.nx3/bindcs.nx3, 1);
    sgid_map = DualArray3D<int>("sgid_map", 2, nmb_sp_x3, nmb_sp_x2);
    srank_map = DualArray3D<int>("srank_map", 2, nmb_sp_x3, nmb_sp_x2);
    for (int m=0; m<(pmesh->nmb_total); ++m) {
      auto &ll = pmesh->lloc_eachmb[m];
      if (ll.lx1 == 0) {                // MeshBlocks at the inner x1 boundary
        sgid_map.h_view(0,ll.lx3,ll.lx2) = m;
        srank_map.h_view(0,ll.lx3,ll.lx2) = pmesh->rank_eachmb[m];
      }
      if (ll.lx1 == (nmbx1-1)) {        // MeshBlocks at the outer x1 boundary
        sgid_map.h_view(1,ll.lx3,ll.lx2) = m;
        srank_map.h_view(1,ll.lx3,ll.lx2) = pmesh->rank_eachmb[m];
      }
    }
    Kokkos::deep_copy(sgid_map.d_view, sgid_map.h_view);
    Kokkos::deep_copy(srank_map.d_view, srank_map.h_view);
  }
}

//----------------------------------------------------------------------------------------
// destructor

particles::ParticlesBoundaryValues::~ParticlesBoundaryValues() {
}
