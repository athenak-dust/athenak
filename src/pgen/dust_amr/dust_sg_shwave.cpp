//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_sg_shwave.cpp
//! \brief Self-gravitating shearing wave of dust particles with gas drag: Johansen et al.
//! (2007, Nature 448, 1022) Supplementary Information sec. 1.3.1, and Simon et al. (2016,
//! ApJ 822, 55) sec. 2.2.2.
//!
//! Physics: uniform dust of density rho_p0 = eps_1 rho0 in a 3-D shearing box, gravity of
//! the dust on the dust only (<gravity>/source = kick = dust), drag on the dust towards
//! gas that stays at rest in the shearing frame (back_reaction = false, etavk = 0).  A
//! non-axisymmetric density wave Re[rho_hat exp(i (kx(t) x + ky y))] with kx(t) = kx0 + q
//! Omega ky t (uniform in z) obeys, to linear order (Johansen et al. eqs. 26-28 with a
//! general q),
//!   d rho_hat/dt = -rho_p0 i (kx wx + ky wy),
//!   d wx/dt      =  2 Omega wy     + i 4piG kx rho_hat/k^2 - wx/t_s,
//!   d wy/dt      = -(2-q) Omega wx + i 4piG ky rho_hat/k^2 - wy/t_s,
//! with w the velocity relative to the Keplerian shear.  The problem generator integrates
//! these (RK4, fine steps) next to the run, so the run carries its own linear reference.
//!
//! Initial state: the quiet root lattice of dust_setup (<problem>/cloud_root_random =
//! true, root_placement = lattice, or point particles on a lattice), displaced by xi =
//! -(A/k^2) k sin(k.x0) so that rho_p = rho_p0 (1 + A cos(k.x0)) + O(A^2), at rest in the
//! shearing frame (A = sgw_amp, wx = wy = 0 as in Johansen et al.); then the usual split
//! to the leaves.  kx0 = sgw_nx 2pi/Lx and ky = sgw_ny 2pi/Ly (integers, so that the wave
//! is shear-periodic at every time).
//!
//! Diagnostics every sgw_dt (and at the end), rank 0, file sgwave.dat:
//!  - the particle estimators of the complex amplitudes, rho_hat = (2/V) sum m exp(-i
//!    theta_p), w_hat = 2 sum m v exp(-i theta_p) / sum m, and the linear solution at the
//!    same time;
//!  - the mesh estimator of rho_hat from the gravity deposit (cell averages,
//!    sinc-corrected);
//!  - the force check: the gathered self-gravity acceleration of every particle against
//!    the acceleration of the measured mode, g = Re[i k 4piG rho_hat/k^2 exp(i theta_p)]:
//!    the projected response g_num/g_mode (TSC smoothing and the discrete operators make
//!    it a few 1e-3 below one); the mass-weighted relative L2 and the max of the residual
//!    against the mode (g_err) and against the fitted mode response*g_mode (g_fit:
//!    without the uniform smoothing, so what is left is harmonics and local errors, e.g.
//!    at level jumps), the latter also per refinement level of the particle's MeshBlock;
//!  - the gas departure from rest (max |rho - rho0|, max |rho v|/rho), the dust mass, the
//!    particle count and the multigrid residual;
//!  - the departure of the gravity deposit from the represented wave, max and rms of
//!    rho_d - rho_p0 - Re[rho_hat e^{i theta}] (cell-averaged) over rho_p0: a defect of
//!    the representation of the uniform background (zero to round-off on a uniform
//!    lattice).
//! With sgw_dump_dt > 0 every rank also writes its particles (x, y, level, g, g_mode) at
//! those times to sgw_force.<n>.r<rank>.csv, for maps of the force error near level
//! jumps.

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "dust/dust.hpp"
#include "globals.hpp"
#include "gravity/self_gravity.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"
#include "pgen/dust_amr/dust_setup.hpp"

namespace {
using Cplx = std::complex<double>;
constexpr int kMaxLev = 8;   // per-level force columns (levels above the root)

struct SGWave {
  Real amp = 1.0e-4, kx0 = 0.0, ky = 0.0, qomega = 1.5, omega = 1.0, ts = 1.0;
  Real fourpig = 1.0, rhop0 = 1.0, rho0 = 1.0, volume = 1.0;
  Real dt = 0.02, dump_dt = 0.0;
  int next = 0, next_dump = 0, nlev = 1;
  bool header = false;
} sgw;

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust_sg_shwave" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

Real KX(Real t) {return sgw.kx0 + sgw.qomega*sgw.ky*t;}

// the linear solution (rho_hat, wx, wy) at time t from rho_hat = A, w = 0 at t = 0
std::array<Cplx, 3> Linear(Real t) {
  const Cplx I(0.0, 1.0);
  const Real om = sgw.omega, q = sgw.qomega/om, ky = sgw.ky;
  auto rhs = [&](Real s, const std::array<Cplx, 3> &y) {
    const Real kx = KX(s), k2 = kx*kx + ky*ky;
    const Cplx g = I*sgw.fourpig*y[0]/k2;
    return std::array<Cplx, 3>{-sgw.rhop0*I*(kx*y[1] + ky*y[2]),
        2.0*om*y[2] + kx*g - y[1]/sgw.ts,
        -(2.0 - q)*om*y[1] + ky*g - y[2]/sgw.ts};
  };
  std::array<Cplx, 3> y = {Cplx(sgw.amp, 0.0), Cplx(0.0), Cplx(0.0)};
  const int nstep = std::max(1, static_cast<int>(std::ceil(t/1.0e-4)));
  const Real h = t/nstep;
  for (int n=0; n<nstep && t > 0.0; ++n) {
    const Real s = n*h;
    auto add = [](const std::array<Cplx, 3> &a, const std::array<Cplx, 3> &b, Real c) {
      return std::array<Cplx, 3>{a[0] + c*b[0], a[1] + c*b[1], a[2] + c*b[2]};
    };
    const auto k1 = rhs(s, y);
    const auto k2 = rhs(s + 0.5*h, add(y, k1, 0.5*h));
    const auto k3 = rhs(s + 0.5*h, add(y, k2, 0.5*h));
    const auto k4 = rhs(s + h, add(y, k3, h));
    for (int c=0; c<3; ++c) y[c] += h/6.0*(k1[c] + 2.0*k2[c] + 2.0*k3[c] + k4[c]);
  }
  return y;
}

void SumAll(double *a, int n) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, a, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
}
void MaxAll(double *a, int n) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, a, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
}

// The Fourier factor F_c(k) of a particle's mass profile, rho_hat contribution m F_c
// e^{-ik.x_c}: one for point particles; for a refinable cloud of half-width ell = root
// dx/2^level the P1 hat sinc^2(k ell/2) per direction, times (2/3 + 1/3 e^{+-ik ell})
// along a transition lobe (lower, upper) or the hat of half-width 2 ell (both lobes), cf.
// SMRCompositeTransfer::CloudWeights1D.
Cplx CloudFactor(bool clouds, int code, Real kx, Real ky, Real rdx1, Real rdx2) {
  if (!clouds) return Cplx(1.0, 0.0);
  using T = SMRCompositeTransfer;
  const int lev = T::CloudLevel(code);
  auto f1 = [](Real k, Real ell, int lobe) {
    auto sinc = [](Real a) {return std::abs(a) < 1.0e-12 ? 1.0 : std::sin(a)/a;};
    if (lobe == 3) {const Real s = sinc(k*ell); return Cplx(s*s, 0.0);}
    const Real s = sinc(0.5*k*ell);
    if (lobe == 0) return Cplx(s*s, 0.0);
    const Cplx I(0.0, 1.0);
    return s*s*(2.0/3.0 + 1.0/3.0*std::exp((lobe == 1 ? 1.0 : -1.0)*I*k*ell));
  };
  const Real scale = 1.0/static_cast<Real>(1 << lev);
  return f1(kx, rdx1*scale, T::CloudLobe1(code))*f1(ky, rdx2*scale, T::CloudLobe2(code));
}

void Observe(Mesh *pm, bool dump) {
  auto *pack = pm->pmb_pack;
  auto *dust = pack->pdust;
  auto *grav = pack->pgrav;
  auto *part = pack->ppart;
  const Real t = pm->time;
  if (grav->last_time != t) grav->Solve(t);
  DvceArray2D<Real> acc;
  dust->GatherGravityAcceleration(t, grav->acceleration, acc);
  DvceArray5D<Real> rho;
  dust->DepositGravityDensity(t, rho);
  auto ah = Kokkos::create_mirror_view_and_copy(HostMemSpace(), acc);
  auto rh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rho);
  auto gas = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pack->phydro->u0);
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_idata);
  auto lev = pack->pmb->mb_lev.h_view;
  const auto ind = pm->mb_indcs;
  const int root = pm->root_level;
  const Real kx = KX(t), ky = sgw.ky, k2 = kx*kx + ky*ky, kn = std::sqrt(k2);
  const Cplx I(0.0, 1.0);
  const bool clouds = dust->Clouds3D();
  const Real rdx1 = (pm->mesh_size.x1max - pm->mesh_size.x1min)/pm->mesh_indcs.nx1;
  const Real rdx2 = (pm->mesh_size.x2max - pm->mesh_size.x2min)/pm->mesh_indcs.nx2;
  const int np = part->nprtcl_thispack;
  std::vector<Cplx> fc(np), ep(np);
  for (int p=0; p<np; ++p) {
    fc[p] = CloudFactor(clouds, pi(PLEV,p), kx, ky, rdx1, rdx2);
    ep[p] = std::exp(I*(kx*pr(IPX,p) + ky*pr(IPY,p)));
  }

  // pass 1: amplitudes of the represented dust (particle sums with the cloud factors),
  // the mesh amplitude of the gravity deposit, the gas departure from rest s[0] mass,
  // s[1..2] sum m F e^{-i th}, s[3..4] m vx F e^{-i th}, s[5..6] m vy F e^{-i th},
  // s[7..8] mesh rho_d e^{-i th} dV, s[9] particles
  double s[10] = {0.0};
  for (int p=0; p<np; ++p) {
    const Real m = pr(IPM,p);
    const Cplx w = m*fc[p]*std::conj(ep[p]);
    s[0] += m; s[1] += w.real(); s[2] += w.imag();
    s[3] += pr(IPVX,p)*w.real(); s[4] += pr(IPVX,p)*w.imag();
    s[5] += pr(IPVY,p)*w.real(); s[6] += pr(IPVY,p)*w.imag();
    s[9] += 1.0;
  }
  SumAll(s, 10);
  const Real vol = sgw.volume;
  const Cplx rhat = 2.0/vol*Cplx(s[1], s[2]);
  // the mesh pass: s[7..8] mesh amplitude; dep[0] max |rho_d - rho_p0 - mode| (the
  // departure of the gravity deposit from the represented wave: a defect of the
  // background representation), dep[1] its sum of squares times dV; gmax: max |rho -
  // rho0| and max |v| of the gas
  double gmax[3] = {0.0, 0.0, 0.0}, dep[2] = {0.0, 0.0};
  for (int b=0; b<pack->nmb_thispack; ++b) {
    const auto sz = pack->pmb->mb_size.h_view(b);
    const Real dv = sz.dx1*sz.dx2*sz.dx3;
    // the cell average of exp(-i th) is exp(-i th_c) sinc(kx dx/2) sinc(ky dy/2)
    auto sinc = [](Real a) {return std::abs(a) < 1.0e-12 ? 1.0 : std::sin(a)/a;};
    const Real win = sinc(0.5*kx*sz.dx1)*sinc(0.5*ky*sz.dx2);
    for (int k=ind.ks; k<=ind.ke; ++k) for (int j=ind.js; j<=ind.je; ++j)
      for (int i=ind.is; i<=ind.ie; ++i) {
        const Real x = sz.x1min + (i - ind.is + 0.5)*sz.dx1;
        const Real y = sz.x2min + (j - ind.js + 0.5)*sz.dx2;
        const Real th = kx*x + ky*y;
        s[7] += rh(b,0,k,j,i)*std::cos(th)*dv/win;
        s[8] -= rh(b,0,k,j,i)*std::sin(th)*dv/win;
        const Real dd = rh(b,0,k,j,i) - sgw.rhop0 - win*std::real(rhat*std::exp(I*th));
        gmax[2] = std::max(gmax[2], std::abs(dd));
        dep[1] += dd*dd*dv;
        const Real d = gas(b,IDN,k,j,i);
        gmax[0] = std::max(gmax[0], std::abs(d - sgw.rho0));
        const Real mv = std::sqrt(gas(b,IM1,k,j,i)*gas(b,IM1,k,j,i) +
            gas(b,IM2,k,j,i)*gas(b,IM2,k,j,i) + gas(b,IM3,k,j,i)*gas(b,IM3,k,j,i));
        gmax[1] = std::max(gmax[1], mv/d);
      }
  }
  {double m78[2] = {s[7], s[8]}; SumAll(m78, 2); s[7] = m78[0]; s[8] = m78[1];}
  SumAll(&dep[1], 1);
  MaxAll(gmax, 3);
  dep[0] = gmax[2];
  const Cplx wx = 2.0/s[0]*Cplx(s[3], s[4]), wy = 2.0/s[0]*Cplx(s[5], s[6]);
  const Cplx rmesh = 2.0/vol*Cplx(s[7], s[8]);
  // the acceleration of the measured mode, g_hat = i k 4piG rho_hat/k^2, averaged over
  // the cloud: particle p feels Re[g_hat conj(F_p) e^{i th_p}]
  const Cplx ghx = I*kx*sgw.fourpig*rhat/k2, ghy = I*ky*sgw.fourpig*rhat/k2;
  const Cplx ghk = I*kn*sgw.fourpig*rhat/k2;
  const Real gamp = std::sqrt(std::norm(ghx) + std::norm(ghy));

  // pass 2: least-squares response r of the numerical force along khat to the mode,
  // r = 2 sum m g_k conj(G_p) / sum m |G_p|^2 with G_p = g_hat_k conj(F_p) e^{i th_p}
  double fp[3] = {0.0, 0.0, 0.0};
  for (int p=0; p<np; ++p) {
    const Real m = pr(IPM,p);
    const Real gk = (ah(0,p)*kx + ah(1,p)*ky)/kn;
    const Cplx G = ghk*std::conj(fc[p])*ep[p];
    const Cplx c = m*gk*std::conj(G);
    fp[0] += c.real(); fp[1] += c.imag(); fp[2] += m*std::norm(G);
  }
  SumAll(fp, 3);
  const Cplx resp = 2.0*Cplx(fp[0], fp[1])/std::max(fp[2], 1.0e-300);

  // pass 3: residuals against the mode (e: smoothing + harmonics + local errors) and
  // against the fitted mode resp*mode (r: without the uniform smoothing), the latter per
  // level. f[0] sum m |e|^2, f[1] sum m |g_mode|^2; per level l: f[2+3l] sum m |r|^2,
  // f[3+3l] sum m |g_mode|^2, f[4+3l] particles; fmax[l] max |r| on level l, fmax[nl] max
  // |e|, fmax[nl+1] max |g_z|
  const int nl = sgw.nlev;
  std::vector<double> f(2 + 3*nl, 0.0), fmax(nl + 2, 0.0);
  std::ofstream out;
  if (dump) {
    char fn[64];
    std::snprintf(fn, sizeof(fn), "sgw_force.%05d.r%d.csv", sgw.next_dump,
                  global_variable::my_rank);
    out.open(fn);
    out << std::setprecision(10) << "# t=" << t << " kx=" << kx << " ky=" << ky
        << " resp_re=" << resp.real() << " resp_im=" << resp.imag() << "\n"
        << "x,y,z,level,code,mass,gx,gy,gz,gx_mode,gy_mode,gx_fit,gy_fit\n";
  }
  for (int p=0; p<np; ++p) {
    const Real m = pr(IPM,p);
    const Cplx e = std::conj(fc[p])*ep[p];
    const Real gx0 = std::real(ghx*e), gy0 = std::real(ghy*e);
    const Real gx1 = std::real(resp*ghx*e), gy1 = std::real(resp*ghy*e);
    const Real gx = ah(0,p), gy = ah(1,p), gz = ah(2,p);
    const int b = pi(PGID,p) - pack->gids;
    const int l = (b >= 0 && b < pack->nmb_thispack) ?
        std::min(nl - 1, std::max(0, lev(b) - root)) : 0;
    const Real ex = gx - gx0, ey = gy - gy0, rx = gx - gx1, ry = gy - gy1;
    f[0] += m*(ex*ex + ey*ey);
    f[1] += m*(gx0*gx0 + gy0*gy0);
    f[2 + 3*l] += m*(rx*rx + ry*ry);
    f[3 + 3*l] += m*(gx0*gx0 + gy0*gy0);
    f[4 + 3*l] += 1.0;
    fmax[l] = std::max(fmax[l], std::sqrt(rx*rx + ry*ry));
    fmax[nl] = std::max(fmax[nl], std::sqrt(ex*ex + ey*ey));
    fmax[nl + 1] = std::max(fmax[nl + 1], std::abs(gz));
    if (dump) {
      out << pr(IPX,p) << ',' << pr(IPY,p) << ',' << pr(IPZ,p) << ',' << l << ','
          << pi(PLEV,p) << ',' << m << ',' << gx << ',' << gy << ',' << gz << ','
          << gx0 << ',' << gy0 << ',' << gx1 << ',' << gy1 << '\n';
    }
  }
  if (dump) out.close();
  SumAll(f.data(), static_cast<int>(f.size()));
  MaxAll(fmax.data(), static_cast<int>(fmax.size()));
  double r2 = 0.0, rmax = 0.0;
  for (int l=0; l<nl; ++l) {r2 += f[2 + 3*l]; rmax = std::max(rmax, fmax[l]);}
  const Real tiny = 1.0e-300;

  const auto lin = Linear(t);
  if (global_variable::my_rank == 0) {
    std::ofstream o("sgwave.dat", sgw.header ? std::ios::app : std::ios::out);
    if (!sgw.header) {
      o << "# dust_sg_shwave: A=" << sgw.amp << " kx0=" << sgw.kx0 << " ky=" << sgw.ky
        << " qOmega=" << sgw.qomega << " Omega=" << sgw.omega << " ts=" << sgw.ts
        << " 4piG=" << sgw.fourpig << " rho_p0=" << sgw.rhop0 << " V=" << vol << "\n";
      o << "# [1]time [2]kx [3]rho_re [4]rho_im [5]wx_re [6]wx_im [7]wy_re [8]wy_im"
        << " [9]lin_rho_re [10]lin_rho_im [11]lin_wx_re [12]lin_wx_im [13]lin_wy_re"
        << " [14]lin_wy_im [15]mesh_rho_re [16]mesh_rho_im [17]g_resp_re [18]g_resp_im"
        << " [19]g_err_l2 [20]g_err_max [21]g_fit_l2 [22]g_fit_max [23]g_mode_amp"
        << " [24]gz_max [25]gas_drho_max [26]gas_v_max [27]dust_mass [28]particles"
        << " [29]mg_residual [30]dep_dev_max [31]dep_dev_rms";
      for (int l=0; l<nl; ++l) {
        o << " [" << 32 + 3*l << "]fit_l2_lev" << l
          << " [" << 33 + 3*l << "]fit_max_lev" << l
          << " [" << 34 + 3*l << "]np_lev" << l;
      }
      o << "\n";
      sgw.header = true;
    }
    o << std::setprecision(10) << std::scientific
      << t << ' ' << kx << ' ' << rhat.real() << ' ' << rhat.imag() << ' '
      << wx.real() << ' ' << wx.imag() << ' ' << wy.real() << ' ' << wy.imag() << ' '
      << lin[0].real() << ' ' << lin[0].imag() << ' '
      << lin[1].real() << ' ' << lin[1].imag() << ' '
      << lin[2].real() << ' ' << lin[2].imag() << ' '
      << rmesh.real() << ' ' << rmesh.imag() << ' '
      << resp.real() << ' ' << resp.imag() << ' '
      << std::sqrt(f[0]/std::max(f[1], tiny)) << ' '
      << fmax[nl]/std::max(gamp, tiny) << ' '
      << std::sqrt(r2/std::max(f[1], tiny)) << ' ' << rmax/std::max(gamp, tiny) << ' '
      << gamp << ' ' << fmax[nl + 1]/std::max(gamp, tiny) << ' '
      << gmax[0] << ' ' << gmax[1] << ' ' << s[0] << ' ' << s[9] << ' '
      << grav->last_solve.relative_residual << ' ' << dep[0]/sgw.rhop0 << ' '
      << std::sqrt(dep[1]/vol)/sgw.rhop0;
    for (int l=0; l<nl; ++l) {
      o << ' ' << std::sqrt(f[2 + 3*l]/std::max(f[3 + 3*l], tiny)) << ' '
        << fmax[l]/std::max(gamp, tiny) << ' ' << f[4 + 3*l];
    }
    o << "\n";
    std::cout << std::setprecision(6) << "# SGWAVE t=" << t << " |rho|=" << std::abs(rhat)
              << " lin=" << std::abs(lin[0])
              << " g_fit_l2=" << std::sqrt(r2/std::max(f[1], tiny))
              << std::endl;
  }
}

void Work(Mesh *pm) {
  const bool due = pm->time + 1.0e-12 >= sgw.next*sgw.dt;
  const bool dump = sgw.dump_dt > 0.0 && pm->time + 1.0e-12 >= sgw.next_dump*sgw.dump_dt;
  if (due || dump) {
    Observe(pm, dump);
    sgw.next = static_cast<int>(std::floor(pm->time/sgw.dt + 1.0e-9)) + 1;
    if (dump) {
      sgw.next_dump = static_cast<int>(std::floor(pm->time/sgw.dump_dt + 1.0e-9)) + 1;
    }
  }
}

void Final(ParameterInput *, Mesh *pm) {Observe(pm, false);}

void SGWaveHistory(HistoryData *pdata, Mesh *pm) {dust_setup::SpeciesHistory(pdata, pm);}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustSGShwave()

void ProblemGenerator::DustSGShwave(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!dust_setup::DustActive(pmy_mesh_)) Fatal("needs <particles> and <dust>");
  if (!pmy_mesh_->three_d || pmbp->pgrav == nullptr || pmbp->phydro == nullptr ||
      !pin->DoesBlockExist("shearing_box") || pmbp->pdust->nspecies != 1) {
    Fatal("needs a 3-D shearing box with <hydro>, <gravity> and one dust species");
  }
  if (pin->GetString("gravity", "source") != "dust" ||
      pin->GetString("gravity", "kick") != "dust") {
    Fatal("needs <gravity>/source = kick = dust");
  }
  if (pmbp->pdust->back_reaction) {
    Fatal("needs <dust>/back_reaction = false (gas at rest)");
  }
  if (pin->GetOrAddReal("problem", "etavk", 0.0) != 0.0 ||
      pin->GetOrAddBoolean("hydro_srcterms", "const_accel", false)) {
    Fatal("needs etavk = 0 and no const_accel (no radial drift)");
  }
  dust_setup::RejectKeys(pin, "dust_sg_shwave", {
      {"dust_test", "none", "dust_damping_amr or dusty_wave_amr"},
      {"stratified_dust", "false", "dust_si_strat"},
      {"nsh_vertical", "false", "dust_si_strat"},
      {"vertical_bc", "none", "dust_si_strat"}});
  dust_setup::SetupHistory(pin);
  user_hist_func = SGWaveHistory;
  dust_setup::EnrollRefinement(pin, pmy_mesh_, this);

  std::vector<Real> eps, taus;
  dust_setup::ReadSpecies(pin, pmbp, false, eps, taus);
  const auto &box = pmy_mesh_->mesh_size;
  const Real lx = box.x1max - box.x1min, ly = box.x2max - box.x2min;
  sgw.amp = pin->GetOrAddReal("problem", "sgw_amp", 1.0e-4);
  sgw.kx0 = pin->GetOrAddInteger("problem", "sgw_nx", -1)*2.0*M_PI/lx;
  sgw.ky = pin->GetOrAddInteger("problem", "sgw_ny", 1)*2.0*M_PI/ly;
  sgw.omega = pin->GetReal("shearing_box", "omega0");
  sgw.qomega = pin->GetReal("shearing_box", "qshear")*sgw.omega;
  sgw.ts = taus[0];
  sgw.fourpig = pin->GetReal("gravity", "four_pi_G");
  sgw.rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  sgw.rhop0 = eps[0]*sgw.rho0;
  sgw.volume = lx*ly*(box.x3max - box.x3min);
  sgw.dt = pin->GetOrAddReal("problem", "sgw_dt", 0.02);
  sgw.dump_dt = pin->GetOrAddReal("problem", "sgw_dump_dt", 0.0);
  sgw.nlev = std::min(kMaxLev,
                      std::max(1, pmy_mesh_->max_level - pmy_mesh_->root_level + 1));
  if (sgw.ky == 0.0 || !(sgw.dt > 0.0) || !(sgw.ts > 0.0)) {
    Fatal("invalid sgw_* or taus_1");
  }
  user_work_in_loop_func = Work;
  user_work_keeps_particles = true;   // Observe only reads the particles
  pgen_final_func = Final;
  if (restart) {
    sgw.header = true;
    sgw.next = static_cast<int>(std::floor(pmy_mesh_->time/sgw.dt + 1.0e-9)) + 1;
    if (sgw.dump_dt > 0.0) {
      sgw.next_dump =
          static_cast<int>(std::floor(pmy_mesh_->time/sgw.dump_dt + 1.0e-9)) + 1;
    }
    dust_setup::LogRestartParticles(pmy_mesh_);
    return;
  }

  // the quiet lattice at rest in the shearing frame, without the split to the leaves
  dust_setup::InitUniformGas(pmy_mesh_, sgw.rho0, 0.0, 0.0);
  dust_setup::DustInit d;
  d.rho0 = sgw.rho0;
  d.eps = eps;
  d.vx.assign(1, 0.0);
  d.vp.assign(1, 0.0);
  d.placement = "lattice";
  const bool splitting = pmbp->pdust->cloud_split_enable;
  pmbp->pdust->cloud_split_enable = false;
  dust_setup::PlaceParticles(pin, pmy_mesh_, d);
  pmbp->pdust->cloud_split_enable = splitting;

  // the displacement xi = -(A/k^2) k sin(k.x0)
  auto *part = pmbp->ppart;
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_rdata);
  const Real k2 = sgw.kx0*sgw.kx0 + sgw.ky*sgw.ky;
  for (int p=0; p<part->nprtcl_thispack; ++p) {
    const Real s = std::sin(sgw.kx0*pr(IPX,p) + sgw.ky*pr(IPY,p));
    const Real x = pr(IPX,p) - sgw.amp*sgw.kx0/k2*s;
    const Real y = pr(IPY,p) - sgw.amp*sgw.ky/k2*s;
    pr(IPX,p) = pr(IPX1,p) = x;
    pr(IPY,p) = pr(IPY1,p) = y;
    pr(IPVX,p) = pr(IPVX1,p) = 0.0;
    pr(IPVY,p) = pr(IPVY1,p) = 0.0;
    pr(IPVZ,p) = pr(IPVZ1,p) = 0.0;
  }
  Kokkos::deep_copy(part->prtcl_rdata, pr);
  if (pmbp->pdust->Clouds3D()) {
    pmbp->pdust->ResetPGIDFromPositions("sg shearing-wave imprint");
    pmbp->pdust->SplitToLeaves();
  }
  std::vector<Real> vx(1, 0.0), vp(1, 0.0);
  dust_setup::SetParticleDt(pmy_mesh_, vx, vp, 0.0);
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(10) << "# DUST_SG_SHWAVE A=" << sgw.amp
              << " kx0=" << sgw.kx0 << " ky=" << sgw.ky << " qOmega=" << sgw.qomega
              << " ts=" << sgw.ts
              << " 4piG=" << sgw.fourpig << " rho_p0=" << sgw.rhop0 << std::endl;
  }
}
