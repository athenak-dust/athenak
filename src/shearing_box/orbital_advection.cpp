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

  // S1: the remap exchanges with the x2-face neighbours at the same level;
  // with mesh refinement check that every neighbour in x2 is on the level of its block
  // (only in 3-D, where the orbital advection runs; the 2-D r-z dust runs never call it)
  if (ppack->pmesh->multilevel && ppack->pmesh->three_d) {
    auto &nghbr = ppack->pmb->nghbr;
    auto &lev = ppack->pmb->mb_lev;
    bool ok = true;
    for (int m=0; m<ppack->nmb_thispack; ++m) {
      for (int n : {8, 12}) {
        if (nghbr.h_view(m,n).gid >= 0 && nghbr.h_view(m,n).lev != lev.h_view(m)) ok = false;
      }
    }
    if (!ok) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Orbital advection with mesh refinement: every MeshBlock's x2 neighbours "
                << "must be on its level (refine in bands of x1)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

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
  }
#endif
}
