//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_gas_shwave.cpp
//! \brief Shearing wave of gas and dust particles coupled by mutual drag, without
//! gravity: Youdin & Johansen (2007, ApJ 662, 613) and Yang & Johansen (2016, ApJS 224,
//! 39) sec. 4.1, Figs. 5-8.
//!
//! Physics: isothermal gas of density rho0 and dust of density rho_p0 = eps_1 rho0
//! (stopping time taus_1) in a 3-D shearing box, back reaction on, no radial pressure
//! gradient.  A wave Re[f_hat exp(i (kx(t) x + ky y))] with kx(t) = kx0 + q Omega ky t
//! (uniform in z) obeys, to linear order (Yang & Johansen eqs. 45a-f with a general q),
//!   d rho_g/dt = -rho0 i (kx ux + ky uy),
//!   d ux/dt    =  2 Omega uy     - eps0 (ux - vx)/ts - i kx cs^2 rho_g/rho0,
//!   d uy/dt    = -(2-q) Omega ux - eps0 (uy - vy)/ts - i ky cs^2 rho_g/rho0,
//!   d rho_p/dt = -rho_p0 i (kx vx + ky vy),
//!   d vx/dt    =  2 Omega vy     - (vx - ux)/ts,
//!   d vy/dt    = -(2-q) Omega vx - (vy - uy)/ts,
//! with u and v the gas and dust velocities relative to the Keplerian shear.  The problem
//! generator integrates these (RK4, steps that resolve ts/(1 + eps0)) next to the run.
//!
//! Initial state: uniform gas with the cell averages of the six real amplitudes
//! <problem>/ysw_rhog, ysw_ux, ysw_uy (times cos(k.x)); the quiet root lattice of
//! dust_setup displaced by xi = -(A/k^2) k sin(k.x0), A = ysw_rhop/rho_p0, with the
//! velocities (ysw_vx, ysw_vy) cos(k.x0); then the split to the leaves.  Yang & Johansen:
//! box L x L, kx0 = -2pi/L, ky = 2pi/L, uy = 1e-3 cs and all other amplitudes zero; their
//! figures match L = 2pi cs/Omega.  kx0 = ysw_nx 2pi/Lx, ky = ysw_ny 2pi/Ly (integers:
//! shear-periodic waves). <problem>/ysw_dt0 > 0 sets the first time step (the code then
//! lets dt grow by at most 2x per cycle), to resolve the initial transient of stiff cases
//! (their Fig. 8).
//!
//! Diagnostics every ysw_dt (and at the end), rank 0, file yswave.dat: the complex
//! amplitudes of rho_g, ux, uy (mesh cell averages, sinc-corrected), of rho_p (particle
//! sums with the refinable-cloud Fourier factor) and vx, vy (mass-weighted), and the
//! linear solution.
//!
//! Crest-following refinement (adaptive mesh, <problem>/ysw_crest_refine = true): every
//! MeshBlock with a cell centre within ysw_crest_halfwidth wavelengths (default 1/8) of
//! the crest of the dust wave, k(t).x + arg(rho_hat_p(t)) = 0 (mod 2 pi), is refined to
//! level ysw_crest_level (default 1) above the root, every other block is derefined.  The
//! crest comes from the linear solution, so all ranks agree without communication; while
//! rho_hat_p is still zero (t = 0) it is the crest that the initial velocities will make,
//! arg(-i k.(v_hat or u_hat)).  With ky = 1 there is one crest per period in y: the
//! stripes of a map are one crest line wrapping through the periodic y boundary.

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "dust/dust.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"
#include "pgen/dust_amr/dust_setup.hpp"

namespace {
using Cplx = std::complex<double>;
using State = std::array<Cplx, 6>;   // rho_g, ux, uy, rho_p, vx, vy

struct YSWave {
  Real kx0 = 0.0, ky = 0.0, qomega = 1.5, omega = 1.0, ts = 1.0, cs = 1.0;
  Real rho0 = 1.0, rhop0 = 1.0, volume = 1.0, dt = 0.02, dt0 = 0.0;
  State amp0 = {};   // initial amplitudes (real)
  State lin = {};    // the linear solution at lin_t
  Real lin_t = 0.0;
  int next = 0;
  bool header = false;
  Real crest_hw = 0.125;   // crest refinement: half-width in wavelengths
  int crest_level = 1;     // and the level above the root
} ysw;

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust_gas_shwave" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

Real KX(Real t) {return ysw.kx0 + ysw.qomega*ysw.ky*t;}

State RHS(Real t, const State &y) {
  const Cplx I(0.0, 1.0);
  const Real om = ysw.omega, q = ysw.qomega/om, ky = ysw.ky, kx = KX(t);
  const Real eps0 = ysw.rhop0/ysw.rho0, cs2 = ysw.cs*ysw.cs, ts = ysw.ts;
  return State{-ysw.rho0*I*(kx*y[1] + ky*y[2]),
      2.0*om*y[2] - eps0*(y[1] - y[4])/ts - I*kx*cs2*y[0]/ysw.rho0,
      -(2.0 - q)*om*y[1] - eps0*(y[2] - y[5])/ts - I*ky*cs2*y[0]/ysw.rho0,
      -ysw.rhop0*I*(kx*y[4] + ky*y[5]),
      2.0*om*y[5] - (y[4] - y[1])/ts,
      -(2.0 - q)*om*y[4] - (y[5] - y[2])/ts};
}

// advance the linear solution from lin_t to t (RK4, h <= ts/(1 + eps0)/20 and 1e-3/Omega)
void AdvanceLinear(Real t) {
  if (t < ysw.lin_t) {ysw.lin = ysw.amp0; ysw.lin_t = 0.0;}
  const Real hmax = std::min(1.0e-3/ysw.omega, 0.05*ysw.ts/(1.0 + ysw.rhop0/ysw.rho0));
  const Real span = t - ysw.lin_t;
  if (span <= 0.0) return;
  const int nstep = static_cast<int>(std::ceil(span/hmax));
  const Real h = span/nstep;
  auto add = [](const State &a, const State &b, Real c) {
    State r; for (int n=0; n<6; ++n) r[n] = a[n] + c*b[n]; return r;
  };
  State y = ysw.lin;
  for (int s=0; s<nstep; ++s) {
    const Real t0 = ysw.lin_t + s*h;
    const State k1 = RHS(t0, y);
    const State k2 = RHS(t0 + 0.5*h, add(y, k1, 0.5*h));
    const State k3 = RHS(t0 + 0.5*h, add(y, k2, 0.5*h));
    const State k4 = RHS(t0 + h, add(y, k3, h));
    for (int n=0; n<6; ++n) y[n] += h/6.0*(k1[n] + 2.0*k2[n] + 2.0*k3[n] + k4[n]);
  }
  ysw.lin = y;
  ysw.lin_t = t;
}

// the crest-following refinement request (see the file header)
void CrestRefinement(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  const Real t = pm->time;
  AdvanceLinear(t);
  const Cplx I(0.0, 1.0);
  const Real kx = KX(t), ky = ysw.ky, kn = std::sqrt(kx*kx + ky*ky);
  Cplx a = ysw.lin[3];
  if (std::abs(a) == 0.0) {   // rho_p not yet perturbed: the crest the velocities make
    const State &y = ysw.lin;
    a = -I*(kx*y[4] + ky*y[5]);
    if (std::abs(a) == 0.0) a = -I*(kx*y[1] + ky*y[2]);
  }
  const Real ph = std::abs(a) > 0.0 ? std::arg(a) : 0.0;
  const Real halfwidth = ysw.crest_hw*2.0*M_PI/kn;
  auto &refine_flag = pm->pmr->refine_flag;
  auto &size = pmbp->pmb->mb_size;
  const auto &ind = pm->mb_indcs;
  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  for (int m=0; m<pmbp->nmb_thispack; ++m) {
    const auto &sz = size.h_view(m);
    bool near = false;
    for (int j=0; j<ind.nx2 && !near; ++j) for (int i=0; i<ind.nx1 && !near; ++i) {
      const Real x = sz.x1min + (i + 0.5)*sz.dx1, y = sz.x2min + (j + 0.5)*sz.dx2;
      // distance to the crest
      const Real d = std::remainder(kx*x + ky*y + ph, 2.0*M_PI)/kn;
      near = std::abs(d) < halfwidth;
    }
    const int want = near ? ysw.crest_level : 0;
    const int lev = pm->lloc_eachmb[m + mbs].level - pm->root_level;
    refine_flag.h_view(m + mbs) = (want > lev) ? 1 : ((want < lev) ? -1 : 0);
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

// the Fourier factor F_c(k) of a particle's mass profile (rho_hat contribution m F_c
// e^{-ik.x_c}): one for point particles; a refinable cloud of half-width ell = root
// dx/2^level is the P1 hat sinc^2(k ell/2) per direction, times (2/3 + 1/3 e^{+-ik ell})
// along a transition lobe (lower, upper) or the hat of half-width 2 ell (both lobes); cf.
// SMRCompositeTransfer::CloudWeights1D
Cplx CloudFactor(bool clouds, int code, Real kx, Real ky, Real rdx1, Real rdx2) {
  if (!clouds) return Cplx(1.0, 0.0);
  using T = SMRCompositeTransfer;
  auto f1 = [](Real k, Real ell, int lobe) {
    auto sinc = [](Real a) {return std::abs(a) < 1.0e-12 ? 1.0 : std::sin(a)/a;};
    if (lobe == 3) {const Real s = sinc(k*ell); return Cplx(s*s, 0.0);}
    const Real s = sinc(0.5*k*ell);
    if (lobe == 0) return Cplx(s*s, 0.0);
    const Cplx I(0.0, 1.0);
    return s*s*(2.0/3.0 + 1.0/3.0*std::exp((lobe == 1 ? 1.0 : -1.0)*I*k*ell));
  };
  const Real scale = 1.0/static_cast<Real>(1 << T::CloudLevel(code));
  return f1(kx, rdx1*scale, T::CloudLobe1(code))*f1(ky, rdx2*scale, T::CloudLobe2(code));
}

void SumAll(double *a, int n) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, a, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}

void Observe(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto *part = pack->ppart;
  const Real t = pm->time;
  auto gas = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pack->phydro->u0);
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_idata);
  const auto ind = pm->mb_indcs;
  const Real kx = KX(t), ky = ysw.ky;
  const Cplx I(0.0, 1.0);
  const bool clouds = pack->pdust->Clouds3D();
  const Real rdx1 = (pm->mesh_size.x1max - pm->mesh_size.x1min)/pm->mesh_indcs.nx1;
  const Real rdx2 = (pm->mesh_size.x2max - pm->mesh_size.x2min)/pm->mesh_indcs.nx2;

  // s[0..5] gas rho, ux, uy (re, im); s[6..11] dust rho, m vx, m vy (re, im);
  // s[12] dust mass, s[13] particles, s[14] gas mass, s[15] volume above the root level
  double s[16] = {0.0};
  for (int b=0; b<pack->nmb_thispack; ++b) {
    const auto sz = pack->pmb->mb_size.h_view(b);
    const Real dv = sz.dx1*sz.dx2*sz.dx3;
    if (pack->pmb->mb_lev.h_view(b) > pm->root_level) s[15] += dv*ind.nx1*ind.nx2*ind.nx3;
    // the cell average of exp(-i th) is exp(-i th_c) sinc(kx dx/2) sinc(ky dy/2)
    auto sinc = [](Real a) {return std::abs(a) < 1.0e-12 ? 1.0 : std::sin(a)/a;};
    const Real win = sinc(0.5*kx*sz.dx1)*sinc(0.5*ky*sz.dx2);
    for (int k=ind.ks; k<=ind.ke; ++k) for (int j=ind.js; j<=ind.je; ++j)
      for (int i=ind.is; i<=ind.ie; ++i) {
        const Real x = sz.x1min + (i - ind.is + 0.5)*sz.dx1;
        const Real y = sz.x2min + (j - ind.js + 0.5)*sz.dx2;
        const Cplx e = std::exp(-I*(kx*x + ky*y))*dv/win;
        const Real d = gas(b,IDN,k,j,i);
        const Cplx cr = (d - ysw.rho0)*e;
        const Cplx cx = gas(b,IM1,k,j,i)/d*e, cy = gas(b,IM2,k,j,i)/d*e;
        s[0] += cr.real(); s[1] += cr.imag(); s[2] += cx.real(); s[3] += cx.imag();
        s[4] += cy.real(); s[5] += cy.imag(); s[14] += d*dv;
      }
  }
  for (int p=0; p<part->nprtcl_thispack; ++p) {
    const Real m = pr(IPM,p);
    const Cplx w = m*CloudFactor(clouds, pi(PLEV,p), kx, ky, rdx1, rdx2)*
                   std::exp(-I*(kx*pr(IPX,p) + ky*pr(IPY,p)));
    s[6] += w.real(); s[7] += w.imag();
    s[8] += pr(IPVX,p)*w.real(); s[9] += pr(IPVX,p)*w.imag();
    s[10] += pr(IPVY,p)*w.real(); s[11] += pr(IPVY,p)*w.imag();
    s[12] += m; s[13] += 1.0;
  }
  SumAll(s, 16);
  const Real vol = ysw.volume;
  const State num = {2.0/vol*Cplx(s[0], s[1]), 2.0/vol*Cplx(s[2], s[3]),
                     2.0/vol*Cplx(s[4], s[5]), 2.0/vol*Cplx(s[6], s[7]),
                     2.0/s[12]*Cplx(s[8], s[9]), 2.0/s[12]*Cplx(s[10], s[11])};
  AdvanceLinear(t);
  if (global_variable::my_rank == 0) {
    std::ofstream o("yswave.dat", ysw.header ? std::ios::app : std::ios::out);
    if (!ysw.header) {
      o << "# dust_gas_shwave: kx0=" << ysw.kx0 << " ky=" << ysw.ky
        << " qOmega=" << ysw.qomega << " Omega=" << ysw.omega << " ts=" << ysw.ts
        << " cs=" << ysw.cs << " rho0=" << ysw.rho0
        << " rho_p0=" << ysw.rhop0 << " V=" << vol << "\n"
        << "# [1]time [2]kx; numerical (re, im): [3,4]rho_g [5,6]ux [7,8]uy"
        << " [9,10]rho_p [11,12]vx [13,14]vy; linear (re, im): [15,16]rho_g [17,18]ux"
        << " [19,20]uy [21,22]rho_p [23,24]vx [25,26]vy; [27]dt [28]gas_mass"
        << " [29]dust_mass [30]particles [31]refined_volume_fraction\n";
      ysw.header = true;
    }
    o << std::setprecision(10) << std::scientific << t << ' ' << kx;
    for (int n=0; n<6; ++n) o << ' ' << num[n].real() << ' ' << num[n].imag();
    for (int n=0; n<6; ++n) o << ' ' << ysw.lin[n].real() << ' ' << ysw.lin[n].imag();
    o << ' ' << pm->dt << ' ' << s[14] << ' ' << s[12] << ' ' << s[13] << ' '
      << s[15]/vol << "\n";
  }
}

void Work(Mesh *pm) {
  if (pm->time + 1.0e-12 >= ysw.next*ysw.dt) {
    Observe(pm);
    ysw.next = static_cast<int>(std::floor(pm->time/ysw.dt + 1.0e-9)) + 1;
  }
}

void Final(ParameterInput *, Mesh *pm) {Observe(pm);}

void YSWaveHistory(HistoryData *pdata, Mesh *pm) {dust_setup::SpeciesHistory(pdata, pm);}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustGasShwave()

void ProblemGenerator::DustGasShwave(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!dust_setup::DustActive(pmy_mesh_)) Fatal("needs <particles> and <dust>");
  if (!pmy_mesh_->three_d || pmbp->phydro == nullptr ||
      !pin->DoesBlockExist("shearing_box") || pmbp->pdust->nspecies != 1 ||
      pmbp->phydro->peos->eos_data.is_ideal) {
    Fatal("needs a 3-D shearing box with isothermal <hydro> and one dust species");
  }
  if (pmbp->pgrav != nullptr) {
    Fatal("no <gravity> (dust_sg_shwave is the self-gravitating wave)");
  }
  if (pin->GetOrAddReal("problem", "etavk", 0.0) != 0.0 ||
      pin->GetOrAddBoolean("hydro_srcterms", "const_accel", false)) {
    Fatal("needs etavk = 0 and no const_accel (no radial drift)");
  }
  dust_setup::RejectKeys(pin, "dust_gas_shwave", {
      {"dust_test", "none", "dust_damping_amr or dusty_wave_amr"},
      {"stratified_dust", "false", "dust_si_strat"},
      {"nsh_vertical", "false", "dust_si_strat"},
      {"vertical_bc", "none", "dust_si_strat"}});
  dust_setup::SetupHistory(pin);
  user_hist_func = YSWaveHistory;
  dust_setup::EnrollRefinement(pin, pmy_mesh_, this);
  if (pin->GetOrAddBoolean("problem", "ysw_crest_refine", false)) {
    if (!pmy_mesh_->adaptive) {
      Fatal("ysw_crest_refine needs <mesh_refinement>/refinement = adaptive");
    }
    ysw.crest_hw = pin->GetOrAddReal("problem", "ysw_crest_halfwidth", 0.125);
    ysw.crest_level = pin->GetOrAddInteger("problem", "ysw_crest_level", 1);
    user_ref_func = CrestRefinement;
  }

  std::vector<Real> eps, taus;
  dust_setup::ReadSpecies(pin, pmbp, false, eps, taus);
  const auto &box = pmy_mesh_->mesh_size;
  const Real lx = box.x1max - box.x1min, ly = box.x2max - box.x2min;
  ysw.kx0 = pin->GetOrAddInteger("problem", "ysw_nx", -1)*2.0*M_PI/lx;
  ysw.ky = pin->GetOrAddInteger("problem", "ysw_ny", 1)*2.0*M_PI/ly;
  ysw.omega = pin->GetReal("shearing_box", "omega0");
  ysw.qomega = pin->GetReal("shearing_box", "qshear")*ysw.omega;
  ysw.ts = taus[0];
  ysw.cs = pmbp->phydro->peos->eos_data.iso_cs;
  ysw.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  ysw.rhop0 = eps[0]*ysw.rho0;
  ysw.volume = lx*ly*(box.x3max - box.x3min);
  ysw.dt = pin->GetOrAddReal("problem", "ysw_dt", 0.02);
  ysw.dt0 = pin->GetOrAddReal("problem", "ysw_dt0", 0.0);
  const char *keys[6] = {"ysw_rhog", "ysw_ux", "ysw_uy", "ysw_rhop", "ysw_vx", "ysw_vy"};
  const Real defaults[6] = {0.0, 0.0, 1.0e-3, 0.0, 0.0, 0.0};
  for (int n=0; n<6; ++n) {
    ysw.amp0[n] = Cplx(pin->GetOrAddReal("problem", keys[n], defaults[n]), 0.0);
  }
  ysw.lin = ysw.amp0;
  ysw.lin_t = 0.0;
  if (ysw.ky == 0.0 || !(ysw.dt > 0.0) || !(ysw.ts > 0.0)) {
    Fatal("invalid ysw_* or taus_1");
  }
  user_work_in_loop_func = Work;
  user_work_keeps_particles = true;   // Observe only reads the particles
  pgen_final_func = Final;
  if (restart) {
    ysw.header = true;
    ysw.next = static_cast<int>(std::floor(pmy_mesh_->time/ysw.dt + 1.0e-9)) + 1;
    dust_setup::LogRestartParticles(pmy_mesh_);
    return;
  }

  // the gas: uniform rho0 plus the cell averages of the wave
  const Real kx0 = ysw.kx0, ky = ysw.ky, rho0 = ysw.rho0;
  const Real ag = ysw.amp0[0].real(), aux = ysw.amp0[1].real(), auy = ysw.amp0[2].real();
  {
    const auto ind = pmy_mesh_->mb_indcs;
    auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->phydro->u0);
    for (int b=0; b<pmbp->nmb_thispack; ++b) {
      const auto sz = pmbp->pmb->mb_size.h_view(b);
      auto sinc = [](Real a) {return std::abs(a) < 1.0e-12 ? 1.0 : std::sin(a)/a;};
      const Real win = sinc(0.5*kx0*sz.dx1)*sinc(0.5*ky*sz.dx2);
      for (int k=0; k<u.extent_int(2); ++k) for (int j=0; j<u.extent_int(3); ++j)
        for (int i=0; i<u.extent_int(4); ++i) {
          const Real x = sz.x1min + (i - ind.is + 0.5)*sz.dx1;
          const Real y = sz.x2min + (j - ind.js + 0.5)*sz.dx2;
          const Real c = win*std::cos(kx0*x + ky*y);
          // linear: rho u = rho0 u' + O(amp^2)
          u(b,IDN,k,j,i) = rho0 + ag*c;
          u(b,IM1,k,j,i) = rho0*aux*c;
          u(b,IM2,k,j,i) = rho0*auy*c;
          u(b,IM3,k,j,i) = 0.0;
        }
    }
    Kokkos::deep_copy(pmbp->phydro->u0, u);
  }

  // the dust: the quiet lattice at rest in the shearing frame, without the split to the
  // leaves
  dust_setup::DustInit d;
  d.rho0 = rho0;
  d.eps = eps;
  d.vx.assign(1, 0.0);
  d.vp.assign(1, 0.0);
  d.placement = "lattice";
  const bool splitting = pmbp->pdust->cloud_split_enable;
  pmbp->pdust->cloud_split_enable = false;
  dust_setup::PlaceParticles(pin, pmy_mesh_, d);
  pmbp->pdust->cloud_split_enable = splitting;
  // the displacement xi = -(A/k^2) k sin(k.x0) and the velocities (vx, vy) cos(k.x0)
  auto *part = pmbp->ppart;
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_rdata);
  const Real k2 = kx0*kx0 + ky*ky, a = ysw.amp0[3].real()/ysw.rhop0;
  const Real avx = ysw.amp0[4].real(), avy = ysw.amp0[5].real();
  for (int p=0; p<part->nprtcl_thispack; ++p) {
    const Real th = kx0*pr(IPX,p) + ky*pr(IPY,p);
    const Real x = pr(IPX,p) - a*kx0/k2*std::sin(th);
    const Real y = pr(IPY,p) - a*ky/k2*std::sin(th);
    pr(IPX,p) = pr(IPX1,p) = x;
    pr(IPY,p) = pr(IPY1,p) = y;
    pr(IPVX,p) = pr(IPVX1,p) = avx*std::cos(th);
    pr(IPVY,p) = pr(IPVY1,p) = avy*std::cos(th);
    pr(IPVZ,p) = pr(IPVZ1,p) = 0.0;
  }
  Kokkos::deep_copy(part->prtcl_rdata, pr);
  if (pmbp->pdust->Clouds3D()) {
    pmbp->pdust->ResetPGIDFromPositions("particle-gas shearing-wave imprint");
    pmbp->pdust->SplitToLeaves();
  }
  std::vector<Real> vx(1, std::abs(avx)), vp(1, std::abs(avy));
  dust_setup::SetParticleDt(pmy_mesh_, vx, vp, 0.0);
  if (ysw.dt0 > 0.0) part->dtnew = std::min(part->dtnew, ysw.dt0);
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(10) << "# DUST_GAS_SHWAVE kx0=" << kx0 << " ky=" << ky
              << " qOmega=" << ysw.qomega << " ts=" << ysw.ts
              << " eps0=" << ysw.rhop0/rho0
              << " cs=" << ysw.cs << " amplitudes(rho_g,ux,uy,rho_p,vx,vy)=";
    for (int n=0; n<6; ++n) std::cout << (n ? "," : "") << ysw.amp0[n].real();
    std::cout << " dt0=" << ysw.dt0 << std::endl;
  }
}
