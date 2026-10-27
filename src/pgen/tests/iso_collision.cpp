//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file iso_collision.cpp
//! \brief Symmetric isothermal colliding flows along x1: rho0 moving at +U
//! for x < xc and -U for x > xc.  The exact solution is a central state at rest with
//! density rho* = rho0 (U + s)/s between two shocks moving outward at speed s, where
//! s (s + U) = cs^2.  The initial condition is the exact solution at time t0 (t0 = 0 gives
//! the bare collision), set as exact cell averages of density and momentum.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

void ProblemGenerator::IsoCollision(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->phydro->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "iso_collision requires isothermal hydro" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real cs = pmbp->phydro->peos->eos_data.iso_cs;
  const Real u = pin->GetReal("problem", "U");
  const Real xc = pin->GetReal("problem", "xc");
  const Real t0 = pin->GetOrAddReal("problem", "t0", 0.0);
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const Real s = 0.5*(std::sqrt(u*u + 4.0*cs*cs) - u);
  const Real rhos = rho0*(u + s)/s;
  const Real xa = xc - s*t0, xb = xc + s*t0;    // plateau [xa, xb]
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto &u0 = pmbp->phydro->u0;
  par_for("iso_collision", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), 0, (nx3 - 1),
          0, (nx2 - 1), 0, (nx1 - 1), KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const Real dx = size.d_view(m).dx1;
    const Real xl = size.d_view(m).x1min + i*dx, xr = xl + dx;
    // fractions of the cell left of the plateau, in it, and right of it
    const Real fl = fmax(0.0, fmin(xr, xa) - xl)/dx;
    const Real fr = fmax(0.0, xr - fmax(xl, xb))/dx;
    const Real fp = fmax(0.0, 1.0 - fl - fr);
    u0(m,IDN,k+ks,j+js,i+is) = rho0*(fl + fr) + rhos*fp;
    u0(m,IM1,k+ks,j+js,i+is) = rho0*u*(fl - fr);
    u0(m,IM2,k+ks,j+js,i+is) = 0.0;
    u0(m,IM3,k+ks,j+js,i+is) = 0.0;
  });
  std::cout << "iso_collision: s = " << s << ", rho* = " << rhos << ", plateau ["
            << xa << ", " << xb << "] at t0 = " << t0 << std::endl;
}
