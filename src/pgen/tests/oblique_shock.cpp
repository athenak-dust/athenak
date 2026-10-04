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
//! sub-sampling of the step.  With ix1/ox1/ix2/ox2_bc = user the ghost cells are set to the
//! exact moving planar shock (front at x0 + vs t_bc, t_bc = time + dt/2, the mid-point of
//! the step, since the boundary function does not know the RK stage), sub-sampled 4x4, so
//! each boundary cell is upstream or downstream according to the analytic solution.

#include <cmath>
#include <iostream>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

namespace {
Real os_mach, os_th, os_x0, os_rho1, os_cs;
void ObliqueShockBCs(Mesh *pm);

// sub-sampled conserved state of the exact moving shock in a cell (centre xc, yc)
KOKKOS_INLINE_FUNCTION
void ShockCellState(Real xc, Real yc, Real dx, Real dy, Real front, Real nx, Real ny,
                    Real rho1, Real rho2, Real u2, Real &d, Real &m1, Real &m2) {
  Real fd = 0.0;
  for (int a=0; a<4; ++a) {
    for (int b=0; b<4; ++b) {
      const Real x = xc + dx*((a + 0.5)/4.0 - 0.5), y = yc + dy*((b + 0.5)/4.0 - 0.5);
      if (nx*x + ny*y < front) fd += 1.0/16.0;
    }
  }
  d = fd*rho2 + (1.0 - fd)*rho1; m1 = fd*rho2*u2*nx; m2 = fd*rho2*u2*ny;
}
}  // namespace

void ProblemGenerator::ObliqueShock(ParameterInput *pin, const bool restart) {
  user_bcs_func = ObliqueShockBCs;
  os_mach = pin->GetReal("problem", "mach");
  os_th = pin->GetReal("problem", "theta_deg")*M_PI/180.0;
  os_x0 = pin->GetReal("problem", "front0");
  os_rho1 = pin->GetOrAddReal("problem", "rho1", 1.0);
  if (pmy_mesh_->pmb_pack->phydro != nullptr) {
    os_cs = pmy_mesh_->pmb_pack->phydro->peos->eos_data.iso_cs;
  }
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

namespace {
void ObliqueShockBCs(Mesh *pm) {
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng, nx1 = indcs.nx1, nx2 = indcs.nx2;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks;
  const int n1 = nx1 + 2*ng;
  auto &mb_bcs = pm->pmb_pack->pmb->mb_bcs;
  auto &size = pm->pmb_pack->pmb->mb_size;
  auto &u0 = pm->pmb_pack->phydro->u0;
  const int nmb = pm->pmb_pack->nmb_thispack;
  const Real rho1 = os_rho1, rho2 = os_rho1*os_mach*os_mach;
  const Real vs = os_mach*os_cs, u2 = vs*(1.0 - 1.0/(os_mach*os_mach));
  const Real nx = std::cos(os_th), ny = std::sin(os_th);
  const Real front = os_x0 + vs*(pm->time + 0.5*pm->dt);
  auto fill = [=] KOKKOS_FUNCTION (int m, int j, int i) {
    const Real dx = size.d_view(m).dx1, dy = size.d_view(m).dx2;
    const Real xc = CellCenterX(i-is, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real yc = CellCenterX(j-js, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real d, m1, m2;
    ShockCellState(xc, yc, dx, dy, front, nx, ny, rho1, rho2, u2, d, m1, m2);
    u0(m,IDN,ks,j,i) = d; u0(m,IM1,ks,j,i) = m1; u0(m,IM2,ks,j,i) = m2;
    u0(m,IM3,ks,j,i) = 0.0;
  };
  par_for("osbc_x1", DevExeSpace(), 0, (nmb-1), 0, (nx2 + 2*ng - 1), 0, (ng-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x1) == BoundaryFlag::user) fill(m, j, i);
    if (mb_bcs.d_view(m,BoundaryFace::outer_x1) == BoundaryFlag::user) fill(m, j, ie+i+1);
  });
  par_for("osbc_x2", DevExeSpace(), 0, (nmb-1), 0, (ng-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    if (mb_bcs.d_view(m,BoundaryFace::inner_x2) == BoundaryFlag::user) fill(m, j, i);
    if (mb_bcs.d_view(m,BoundaryFace::outer_x2) == BoundaryFlag::user) fill(m, je+j+1, i);
  });
}
}  // namespace
