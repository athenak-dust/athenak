//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file oblique_shock.cpp
//! \brief Isothermal planar shock travelling at an angle to the grid (an
//! oblique-shock robustness test).  Upstream rho1 at rest on the side n.x > x0 + vs t,
//! with n = (cos theta, sin theta); shock speed vs = mach*cs; downstream rho2 = mach^2 rho1
//! moving with u2 = vs (1 - 1/mach^2) along n.  Cell averages of rho and M by 4x4
//! sub-sampling of the step.  Use outflow boundaries.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

void ProblemGenerator::ObliqueShock(ParameterInput *pin, const bool restart) {
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->phydro->peos->eos_data.is_ideal ||
      !pmy_mesh_->two_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "oblique_shock requires 2-D isothermal hydro" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real cs = pmbp->phydro->peos->eos_data.iso_cs;
  const Real mach = pin->GetReal("problem", "mach");
  const Real th = pin->GetReal("problem", "theta_deg")*M_PI/180.0;
  const Real x0 = pin->GetReal("problem", "front0");
  const Real rho1 = pin->GetOrAddReal("problem", "rho1", 1.0);
  const Real rho2 = rho1*mach*mach, u2 = mach*cs*(1.0 - 1.0/(mach*mach));
  const Real nx = std::cos(th), ny = std::sin(th);
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  auto &u0 = pmbp->phydro->u0;
  par_for("oblique_shock", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), 0, (nx2 - 1),
          0, (nx1 - 1), KOKKOS_LAMBDA(int m, int j, int i) {
    const Real dx = size.d_view(m).dx1, dy = size.d_view(m).dx2;
    const Real xc = CellCenterX(i, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real yc = CellCenterX(j, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real fd = 0.0;   // downstream fraction of the cell
    for (int a=0; a<4; ++a) {
      for (int b=0; b<4; ++b) {
        const Real x = xc + dx*((a + 0.5)/4.0 - 0.5), y = yc + dy*((b + 0.5)/4.0 - 0.5);
        if (nx*x + ny*y < x0) fd += 1.0/16.0;
      }
    }
    const Real d = fd*rho2 + (1.0 - fd)*rho1;
    u0(m,IDN,ks,j+js,i+is) = d;
    u0(m,IM1,ks,j+js,i+is) = fd*rho2*u2*nx;
    u0(m,IM2,ks,j+js,i+is) = fd*rho2*u2*ny;
    u0(m,IM3,ks,j+js,i+is) = 0.0;
  });
}
