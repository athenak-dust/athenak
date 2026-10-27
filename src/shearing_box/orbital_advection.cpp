//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.cpp
//! \brief constructor for OrbitalAdvection abstract base class.

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
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
//! OrbitalAdvection base class constructor
//! Called by Hydro and MHD constructors, so cannot access any data inside Hydro/MHD
//! classes as it may not be properly allocated yet.

OrbitalAdvection::OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin) :
    maxjshift(1),
    shearing_box_r_phi(false),     // 2D r-phi not yet implemented
    pmy_pack(ppack) {
  // Read shear rate and orbital frequency
  qshear = pin->GetReal("shearing_box","qshear");
  omega0 = pin->GetReal("shearing_box","omega0");

  // estimate maximum integer shift in x2-direction for orbital advection
  Real xmin = fabs(ppack->pmesh->mesh_size.x1min);
  Real xmax = fabs(ppack->pmesh->mesh_size.x1max);
  maxjshift = static_cast<int>((ppack->pmesh->cfl_no)*std::max(xmin,xmax)) + 1;

  // x2 neighbours on another level are allowed (BuildMixedLists,
  // called by the derived classes, which know the number of variables)
  any_mixed = false;
  nvar_oa = 0;
  nrow = ppack->pmesh->mb_indcs.ng + maxjshift;
#if MPI_PARALLEL_ENABLED
  nreq_mixed = 0;
  for (int n=0; n<2; ++n) {
    msreq[n] = nullptr; mrreq[n] = nullptr; fsreq[n] = nullptr; frreq[n] = nullptr;
  }
#endif

#if MPI_PARALLEL_ENABLED
  // For orbital advection, communication is only with x2-face neighbors
  // initialize vectors of MPI request in 2 elements of fixed length arrays
  for (int n=0; n<2; ++n) {
    int nmb = std::max((ppack->nmb_thispack), (ppack->pmesh->nmb_maxperrank));
    sendbuf[n].vars_req = new MPI_Request[nmb];
    recvbuf[n].vars_req = new MPI_Request[nmb];
    for (int m=0; m<nmb; ++m) {
      sendbuf[n].vars_req[m] = MPI_REQUEST_NULL;
      recvbuf[n].vars_req[m] = MPI_REQUEST_NULL;
    }
  }
  // create unique communicators for shearing box
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_orb_advect);
#endif
}

//----------------------------------------------------------------------------------------
// OrbitalAdvection base class destructor

OrbitalAdvection::~OrbitalAdvection() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    delete [] sendbuf[n].vars_req;
    delete [] recvbuf[n].vars_req;
    delete [] msreq[n]; delete [] mrreq[n]; delete [] fsreq[n]; delete [] frreq[n];
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn void OrbitalAdvection::BuildMixedLists()
//! \brief which x2 faces of the blocks of this rank have a
//! neighbour on another level (xlev), the slot of a coarser neighbour, and the buffers and MPI
//! requests of the mixed-level exchange.  Called by the derived constructors and after every
//! regrid (the neighbour lists must be current).

void OrbitalAdvection::BuildMixedLists() {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  nrow = indcs.ng + maxjshift;
  Kokkos::realloc(xlev, std::max(1, nmb), 2);
  Kokkos::realloc(xslot, std::max(1, nmb), 2);
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &lev = pmy_pack->pmb->mb_lev;
  any_mixed = false;
  bool bad = false;
  for (int m=0; m<nmb; ++m) {
    for (int n=0; n<2; ++n) {
      const int base = (n == 0) ? 8 : 12;
      int x = 0, slot = -1, nfiner = 0;
      if (pm->multilevel && pm->three_d) {
        for (int s=0; s<4; ++s) {
          const auto &g = nghbr.h_view(m, base+s);
          if (g.gid < 0) continue;
          if (g.lev > lev.h_view(m)) {
            nfiner++;
          } else if (g.lev < lev.h_view(m)) {
            x = 1; slot = base + s;
          }
        }
        if (nfiner > 0) {
          if (nfiner != 4 || x != 0) bad = true;
          x = 2;
        }
      }
      xlev.h_view(m,n) = x;
      xslot.h_view(m,n) = slot;
      if (x != 0) any_mixed = true;
    }
  }
  xlev.template modify<HostMemSpace>();
  xlev.template sync<DevExeSpace>();
  xslot.template modify<HostMemSpace>();
  xslot.template sync<DevExeSpace>();
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Orbital advection: an x2 face with finer neighbours must have all four"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (any_mixed && nvar_oa <= 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Orbital advection of face-centred fields with x2 neighbours on another "
              << "level is not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (any_mixed && 2*nrow > indcs.nx2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Orbital advection with x2 neighbours on another level: meshblock/nx2 = "
              << indcs.nx2 << " must be at least 2*(nghost + maxjshift) = " << 2*nrow << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nk = indcs.nx3, ni = indcs.nx1;
  const int nv = std::max(1, nvar_oa);
  const int scf = any_mixed ? nv*nk*nrow*ni : 1;
  const int sfc = any_mixed ? nv*(nk/2)*nrow*(ni/2) : 1;
  const int sff = any_mixed ? nv*(nk/2)*(ni/2) : 1;
  const int nmbm = any_mixed ? nmb : 1;
  for (int n=0; n<2; ++n) {
    Kokkos::realloc(msend[n], nmbm, 4, scf);
    Kokkos::realloc(mrecv[n], nmbm, 4, sfc);
    Kokkos::realloc(fsend[n], nmbm, 1, sff);
    Kokkos::realloc(frecv[n], nmbm, 4, sff);
  }
  Kokkos::realloc(fown, nmbm, nv, 2, any_mixed ? nk : 1, any_mixed ? ni : 1);
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    delete [] msreq[n]; delete [] mrreq[n]; delete [] fsreq[n]; delete [] frreq[n];
    msreq[n] = new MPI_Request[4*nmbm];
    mrreq[n] = new MPI_Request[4*nmbm];
    fsreq[n] = new MPI_Request[4*nmbm];
    frreq[n] = new MPI_Request[4*nmbm];
    for (int r=0; r<4*nmbm; ++r) {
      msreq[n][r] = MPI_REQUEST_NULL; mrreq[n][r] = MPI_REQUEST_NULL;
      fsreq[n][r] = MPI_REQUEST_NULL; frreq[n][r] = MPI_REQUEST_NULL;
    }
  }
  nreq_mixed = 4*nmbm;
#endif
}
