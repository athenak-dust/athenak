//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file simple_wave.cpp
//! \brief Isothermal finite-amplitude simple wave: the
//! right-moving Riemann-invariant state u_par - cs ln(rho/rho0) = 0 with
//! u_par = amp cs sin(k.x), rho = rho0 exp(u_par/cs), velocity along k plus an optional
//! uniform frame velocity (boost_v1, boost_v2) for the boost check.  k = 2 pi (kx_mode/Lx,
//! kz_mode/Ly) on a 2-D periodic box.  Cell averages of rho and M from 3x3 Gauss points.
//! Each characteristic value u_par moves with u_par + cs along k (shock at T/(2 pi amp)).
//! At the end of the run a double-precision dump <basename>.final.swv is written (single
//! rank): header "simple_wave v1", line "time ncell", then ncell x {x, y, dx, dy, rho, M1,
//! M2, M3}.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "pgen/pgen.hpp"

namespace {
std::string swv_basename;
void SimpleWaveFinal(ParameterInput *pin, Mesh *pm);
}

void ProblemGenerator::SimpleWave(ParameterInput *pin, const bool restart) {
  pgen_final_func = SimpleWaveFinal;
  swv_basename = pin->GetString("job", "basename");
  if (restart) return;
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->phydro->peos->eos_data.is_ideal ||
      !pmy_mesh_->two_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "simple_wave requires 2-D isothermal hydro" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real cs = pmbp->phydro->peos->eos_data.iso_cs;
  const Real amp = pin->GetReal("problem", "amp");
  const Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  const int mx = pin->GetOrAddInteger("problem", "kx_mode", 1);
  const int mz = pin->GetOrAddInteger("problem", "kz_mode", 0);
  const Real bv1 = pin->GetOrAddReal("problem", "boost_v1", 0.0);
  const Real bv2 = pin->GetOrAddReal("problem", "boost_v2", 0.0);
  auto &size = pmbp->pmb->mb_size;
  const Real lx = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
  const Real ly = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
  const Real kx = 2.0*M_PI*mx/lx, ky = 2.0*M_PI*mz/ly;
  const Real kk = std::sqrt(kx*kx + ky*ky);
  const Real nx = kx/kk, ny = ky/kk;
  auto &indcs = pmy_mesh_->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  auto &u0 = pmbp->phydro->u0;
  const Real gq[3] = {-std::sqrt(0.6), 0.0, std::sqrt(0.6)};
  const Real gw[3] = {5.0/9.0, 8.0/9.0, 5.0/9.0};
  par_for("swv", DevExeSpace(), 0, (pmbp->nmb_thispack - 1), 0, (nx2 - 1), 0, (nx1 - 1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    const Real dx = size.d_view(m).dx1, dy = size.d_view(m).dx2;
    const Real xc = CellCenterX(i, nx1, size.d_view(m).x1min, size.d_view(m).x1max);
    const Real yc = CellCenterX(j, nx2, size.d_view(m).x2min, size.d_view(m).x2max);
    Real d = 0.0, m1 = 0.0, m2 = 0.0;
    for (int a=0; a<3; ++a) {
      for (int b=0; b<3; ++b) {
        const Real x = xc + 0.5*dx*gq[a], y = yc + 0.5*dy*gq[b];
        const Real up = amp*cs*sin(kx*x + ky*y);
        const Real rho = rho0*exp(up/cs);
        const Real w = 0.25*gw[a]*gw[b];
        d += w*rho; m1 += w*rho*(up*nx + bv1); m2 += w*rho*(up*ny + bv2);
      }
    }
    u0(m,IDN,ks,j+js,i+is) = d;
    u0(m,IM1,ks,j+js,i+is) = m1;
    u0(m,IM2,ks,j+js,i+is) = m2;
    u0(m,IM3,ks,j+js,i+is) = 0.0;
  });
}

namespace {
void SimpleWaveFinal(ParameterInput *pin, Mesh *pm) {
  if (global_variable::nranks > 1) return;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->phydro->u0);
  auto &sz = pmbp->pmb->mb_size;
  auto &ind = pm->mb_indcs;
  std::vector<double> buf;
  for (int m=0; m<pmbp->nmb_thispack; ++m) {
    const auto &s = sz.h_view(m);
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i) {
        buf.insert(buf.end(), {s.x1min + (i - ind.is + 0.5)*s.dx1,
                               s.x2min + (j - ind.js + 0.5)*s.dx2, s.dx1, s.dx2,
                               u(m,IDN,ind.ks,j,i), u(m,IM1,ind.ks,j,i),
                               u(m,IM2,ind.ks,j,i), u(m,IM3,ind.ks,j,i)});
      }
    }
  }
  std::string fn = swv_basename + ".final.swv";
  FILE *f = std::fopen(fn.c_str(), "wb");
  if (f == nullptr) {std::cerr << "cannot open " << fn << std::endl; return;}
  std::fprintf(f, "simple_wave v1\n%.17g %zu\n", pm->time, buf.size()/8);
  std::fwrite(buf.data(), sizeof(double), buf.size(), f);
  std::fclose(f);
}
}  // namespace
