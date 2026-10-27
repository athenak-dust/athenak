//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file prolong_amp_unit.cpp
//! \brief Unit test of ProlongCCAmplitudeAware: the compiled device function is called
//! on prescribed coarse stencils and the
//! children are checked for (1) the density lower bound 0.1 rho after the positivity
//! scaling, (2) conservation of the coarse average of every variable, (3) bitwise equality
//! with the central slope when every weight is 1.  Cases: a 3-D example (centre 1,
//! neighbours 0.1 and 4 along every axis: MC corner child -0.35), its 2-D version, a 3-D
//! case with momenta, and NRAND random strong 3-D stencils (rho in [1e-3, 10], u in [-5, 5]),
//! for which the number of negative children of plain MC and plain central slopes is also
//! reported.  Prints "PROLONG_AMP_UNIT ..." lines; run with nlim = 0.

#include <cmath>
#include <iostream>
#include <random>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/prolongation.hpp"
#include "pgen/pgen.hpp"

void ProblemGenerator::ProlongAmpUnit(ParameterInput *pin, const bool restart) {
  if (restart) return;
  const int nrand = pin->GetOrAddInteger("problem", "nrand", 100000);
  const Real eps = pin->GetOrAddReal("problem", "eps", 1.0e-3);
  const Real cs = 1.0;
  const int nfix = 4, n = nfix + nrand;
  DvceArray5D<Real> ca("unit_ca", n, 4, 3, 3, 3), a("unit_a", n, 4, 2, 2, 2);
  DvceArray1D<int64_t> cnt("unit_cnt", 30);
  auto ca_h = Kokkos::create_mirror_view(ca);
  Kokkos::deep_copy(ca_h, 0.0);
  // background: rho = 1, M = 0 everywhere
  for (int m=0; m<n; ++m) for (int k=0; k<3; ++k) for (int j=0; j<3; ++j)
    for (int i=0; i<3; ++i) ca_h(m,IDN,k,j,i) = 1.0;
  auto axis = [&](int m, int v, Real lo, Real c, Real hi, int dir) {
    if (dir == 0) {ca_h(m,v,1,1,0) = lo; ca_h(m,v,1,1,1) = c; ca_h(m,v,1,1,2) = hi;}
    if (dir == 1) {ca_h(m,v,1,0,1) = lo; ca_h(m,v,1,1,1) = c; ca_h(m,v,1,2,1) = hi;}
    if (dir == 2) {ca_h(m,v,0,1,1) = lo; ca_h(m,v,1,1,1) = c; ca_h(m,v,2,1,1) = hi;}
  };
  // case 0: the 3-D example; case 1: same stencil, used in 2-D (three_d = false)
  for (int m=0; m<2; ++m) for (int d=0; d<3; ++d) axis(m, IDN, 0.1, 1.0, 4.0, d);
  // case 2: 3-D example with momenta M = rho u, u = (2, -1, 0.5) on the low side, 0 centre,
  // (-3, 1, 0) on the high side along every axis
  for (int d=0; d<3; ++d) {
    axis(2, IDN, 0.1, 1.0, 4.0, d);
    axis(2, IM1, 0.2, 0.0, -12.0, d); axis(2, IM2, -0.1, 0.0, 4.0, d);
    axis(2, IM3, 0.05, 0.0, 0.0, d);
  }
  // case 3: tiny perturbation (every weight 1): must equal the central slope bitwise
  for (int d=0; d<3; ++d) {axis(3, IDN, 1.0 - 1e-6, 1.0, 1.0 + 3e-6, d);
                           axis(3, IM1, -2e-7, 0.0, 5e-7, d);}
  std::mt19937 gen(12345);
  std::uniform_real_distribution<Real> lr(std::log(1e-3), std::log(10.0)), uu(-5.0, 5.0);
  for (int m=nfix; m<n; ++m) {
    for (int k=0; k<3; ++k) for (int j=0; j<3; ++j) for (int i=0; i<3; ++i) {
      const Real r = std::exp(lr(gen));
      ca_h(m,IDN,k,j,i) = r;
      for (int v=IM1; v<=IM3; ++v) ca_h(m,v,k,j,i) = r*uu(gen);
    }
  }
  Kokkos::deep_copy(ca, ca_h);
  par_for("prolamp_unit", DevExeSpace(), 0, n-1, 0, 3, KOKKOS_LAMBDA(int m, int v) {
    const bool three_d = (m != 1);
    ProlongCCAmplitudeAware(m, v, 1, 1, 1, 0, 0, 0, true, three_d, ca, a, eps, cs, cnt);
  });
  auto a_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), a);
  auto c_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), cnt);

  // checks
  auto children = [&](int m, int v, bool three_d, Real *out) {
    int nc = 0;
    for (int k=0; k<(three_d ? 2 : 1); ++k) for (int j=0; j<2; ++j) for (int i=0; i<2; ++i)
      out[nc++] = a_h(m,v,k,j,i);
    return nc;
  };
  // plain slopes for comparison (host): child = c + sum_d s_d dvar_d with dvar = slope/4
  auto plain_min_child = [&](int m, int kind) {
    Real dv[3];
    for (int d=0; d<3; ++d) {
      Real lo = (d == 0) ? ca_h(m,IDN,1,1,0) : (d == 1) ? ca_h(m,IDN,1,0,1) : ca_h(m,IDN,0,1,1);
      Real hi = (d == 0) ? ca_h(m,IDN,1,1,2) : (d == 1) ? ca_h(m,IDN,1,2,1) : ca_h(m,IDN,2,1,1);
      Real c = ca_h(m,IDN,1,1,1), dl = c - lo, dr = hi - c;
      dv[d] = (kind == 0) ? 0.125*(dl + dr) : PROLONG_CC_MC(dl, dr);
    }
    return ca_h(m,IDN,1,1,1) - std::fabs(dv[0]) - std::fabs(dv[1]) - std::fabs(dv[2]);
  };
  int bad_bound = 0, bad_cons = 0, neg_central = 0, neg_mc = 0, scaled = 0;
  Real worst_cons = 0.0, worst_ratio = 1e30;
  for (int m=0; m<n; ++m) {
    const bool three_d = (m != 1);
    Real ch[8];
    int nc = children(m, IDN, three_d, ch);
    Real mn = 1e30, mean = 0.0;
    for (int q=0; q<nc; ++q) {mn = std::fmin(mn, ch[q]); mean += ch[q]/nc;}
    const Real c = ca_h(m,IDN,1,1,1);
    worst_ratio = std::fmin(worst_ratio, mn/c);
    if (mn < 0.1*c*(1.0 - 1e-12)) bad_bound++;
    if (mn < 0.1*c*(1.0 + 1e-12) && mn > 0.1*c*(1.0 - 1e-12)) scaled++;
    for (int v=0; v<4; ++v) {
      Real chv[8]; nc = children(m, v, three_d, chv); Real mv = 0.0;
      for (int q=0; q<nc; ++q) mv += chv[q]/nc;
      const Real err = std::fabs(mv - ca_h(m,v,1,1,1))/std::fmax(std::fabs(ca_h(m,v,1,1,1)), 1.0);
      worst_cons = std::fmax(worst_cons, err);
      if (err > 1e-13) bad_cons++;
    }
    if (m >= nfix && three_d) {
      if (plain_min_child(m, 0) < 0.0) neg_central++;
      if (plain_min_child(m, 1) < 0.0) neg_mc++;
    }
  }
  // case 3: bitwise central
  bool central_ok = true;
  for (int v=0; v<4; ++v) {
    for (int k=0; k<2; ++k) for (int j=0; j<2; ++j) for (int i=0; i<2; ++i) {
      Real d1 = 0.125*((ca_h(3,v,1,1,1) - ca_h(3,v,1,1,0)) + (ca_h(3,v,1,1,2) - ca_h(3,v,1,1,1)));
      Real d2 = 0.125*((ca_h(3,v,1,1,1) - ca_h(3,v,1,0,1)) + (ca_h(3,v,1,2,1) - ca_h(3,v,1,1,1)));
      Real d3 = 0.125*((ca_h(3,v,1,1,1) - ca_h(3,v,0,1,1)) + (ca_h(3,v,2,1,1) - ca_h(3,v,1,1,1)));
      // operation order of ProlongCC: c -/+ d1 -/+ d2 -/+ d3
      Real r = ca_h(3,v,1,1,1);
      r = i ? r + d1 : r - d1; r = j ? r + d2 : r - d2; r = k ? r + d3 : r - d3;
      if (a_h(3,v,k,j,i) != r) central_ok = false;
    }
  }
  Real ch0[8]; children(0, IDN, true, ch0);
  Real mn0 = 1e30; for (int q=0; q<8; ++q) mn0 = std::fmin(mn0, ch0[q]);
  Real ch1[8]; int n1 = children(1, IDN, false, ch1);
  Real mn1 = 1e30; for (int q=0; q<n1; ++q) mn1 = std::fmin(mn1, ch1[q]);
  std::cout << "PROLONG_AMP_UNIT case0 (3-D, 0.1|1|4 on every axis): plain-MC corner child "
            << plain_min_child(0, 1) << ", candidate min child " << mn0 << std::endl;
  std::cout << "PROLONG_AMP_UNIT case1 (same stencil in 2-D): candidate min child " << mn1
            << std::endl;
  std::cout << "PROLONG_AMP_UNIT case3 (tiny perturbation): bitwise central "
            << (central_ok ? "yes" : "NO") << std::endl;
  std::cout << "PROLONG_AMP_UNIT all " << n << " stencils: children below 0.1 rho " << bad_bound
            << ", min child/rho " << worst_ratio << ", stencils at the 0.1 bound (scaled) "
            << scaled << ", positivity counter " << c_h(8)
            << " | conservation violations (> 1e-13) " << bad_cons << ", worst relative "
            << worst_cons << " | random 3-D stencils with a negative child: plain central "
            << neg_central << ", plain MC " << neg_mc << " of " << nrand << std::endl;
}
