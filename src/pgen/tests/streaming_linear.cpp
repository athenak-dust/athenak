//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file streaming_linear.cpp
//! \brief Problem generator for linear streaming-instability growth-rate tests in the
//! axisymmetric (2D r-z) shearing box (Youdin & Goodman 2005; Youdin & Johansen 2007;
//! Yang & Johansen 2016 Sec. 4.2; Bai & Stone 2010 Sec. 4.3). A single dust species and
//! gas are initialized at the NSH drift equilibrium plus an eigenmode perturbation of
//! the linearized system. The eigenfunctions are standing waves in the vertical
//! direction and traveling in the radial direction (YJ16 Eqs. 46-47):
//!   even fields (rho_g, u_x, u_phi, rho_p, v_x, v_phi):
//!     f(x,z,t) = [Re(f~)cos(kx x - wR t) - Im(f~)sin(kx x - wR t)] e^{st} cos(kz z)
//!   odd fields (u_z, v_z):
//!     f(x,z,t) = -[Re(f~)sin(kx x - wR t) + Im(f~)cos(kx x - wR t)] e^{st} sin(kz z)
//! The complex amplitudes f~ (in code units), the wavenumbers, and the complex
//! frequency are supplied through <problem> parameters (computed by an external
//! eigensolver; see the phase-2 validation notebook). The particle density wave is
//! imprinted by radial lattice displacements proportional to cos(kz z) (YJ07 App. C).
//!
//! With <problem>/user_hist = true, the history records the complex Fourier projection
//! of the particle mass distribution and of the gas density/radial-velocity fields onto
//! the (kx, kz) mode, from which the growth rate is measured by linear regression.
//! In the 2D r-z geometry: radial = x1, vertical = x2 (IM2/IPVY), azimuthal = IM3/IPVZ.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs/outputs.hpp"

// user history: Fourier projections onto the seeded mode
void StreamingLinearHistory(HistoryData *pdata, Mesh *pm);

namespace {

// mode parameters cached for the history function
Real kx_ = 0.0, kz_ = 0.0, rho0_ = 1.0;

//----------------------------------------------------------------------------------------
//! \fn SolveNSH2
//! \brief Single-species NSH drift equilibrium (specialization of the solver in
//! dust_nsh.cpp): with tau = taus*omega0 dimensionless and kappa2 = 2*(2-q),
//! the closed-form solution of the 4x4 linear system.

void SolveNSH2(const Real omega0, const Real qshear, const Real ax,
               const Real eps, const Real taus,
               Real &ugx, Real &ugp, Real &vx, Real &vp) {
  // solve by direct 4x4 Gaussian elimination for robustness
  const int n = 4;
  Real a[4][5] = {};
  Real it = 1.0/taus;
  // unknowns [u_x, u_phi, v_x, v_phi]
  a[0][0] = -eps*it; a[0][1] = 2.0*omega0; a[0][2] = eps*it; a[0][4] = -ax;
  a[1][0] = -(2.0-qshear)*omega0; a[1][1] = -eps*it; a[1][3] = eps*it;
  a[2][0] = it; a[2][2] = -it; a[2][3] = 2.0*omega0;
  a[3][1] = it; a[3][2] = -(2.0-qshear)*omega0; a[3][3] = -it;
  for (int c=0; c<n; ++c) {
    int piv = c;
    for (int r=c+1; r<n; ++r) {
      if (fabs(a[r][c]) > fabs(a[piv][c])) {piv = r;}
    }
    for (int cc=0; cc<=n; ++cc) {std::swap(a[c][cc], a[piv][cc]);}
    for (int r=c+1; r<n; ++r) {
      Real f = a[r][c]/a[c][c];
      for (int cc=c; cc<=n; ++cc) {a[r][cc] -= f*a[c][cc];}
    }
  }
  Real x[4];
  for (int r=n-1; r>=0; --r) {
    Real sum = a[r][4];
    for (int cc=r+1; cc<n; ++cc) {sum -= a[r][cc]*x[cc];}
    x[r] = sum/a[r][r];
  }
  ugx = x[0]; ugp = x[1]; vx = x[2]; vp = x[3];
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::StreamingLinear()
//! \brief Problem Generator for linear streaming-instability eigenmodes

void ProblemGenerator::StreamingLinear(ParameterInput *pin, const bool restart) {
  user_hist_func = StreamingLinearHistory;
  // cache mode parameters for the history function (also needed on restart)
  kx_ = pin->GetReal("problem","kx");
  kz_ = pin->GetReal("problem","kz");
  rho0_ = pin->GetOrAddReal("problem","rho0",1.0);
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr ||
      !(pin->DoesBlockExist("shearing_box"))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Streaming test requires <hydro>, <particles>, <dust>, <shearing_box>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_mesh_->three_d || pmbp->pdust->nspecies != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Streaming test requires 2D r-z geometry and a single dust species"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // mode parameters (from the external eigensolver)
  Real kx = kx_, kz = kz_, rho0 = rho0_;
  Real omega0 = pin->GetReal("shearing_box","omega0");
  Real qshear = pin->GetReal("shearing_box","qshear");
  Real etavk = pin->GetReal("problem","etavk");
  Real eps  = pin->GetReal("problem","eps");
  Real taus = pmbp->pdust->taus.h_view(0);
  Real wr = pin->GetReal("problem","omega_re");   // (unused at t=0, kept for reference)
  (void) wr;
  // complex amplitudes (code units): gas density, gas velocities, dust density (real
  // normalization), dust velocities
  Real Agr = pin->GetReal("problem","drhog_re"), Agi = pin->GetReal("problem","drhog_im");
  Real Uxr = pin->GetReal("problem","dugx_re"),  Uxi = pin->GetReal("problem","dugx_im");
  Real Upr = pin->GetReal("problem","dugp_re"),  Upi = pin->GetReal("problem","dugp_im");
  Real Uzr = pin->GetReal("problem","dugz_re"),  Uzi = pin->GetReal("problem","dugz_im");
  Real Apr = pin->GetReal("problem","drhop_re");
  Real Vxr = pin->GetReal("problem","dvpx_re"),  Vxi = pin->GetReal("problem","dvpx_im");
  Real Vpr = pin->GetReal("problem","dvpp_re"),  Vpi = pin->GetReal("problem","dvpp_im");
  Real Vzr = pin->GetReal("problem","dvpz_re"),  Vzi = pin->GetReal("problem","dvpz_im");

  // radial forcing consistency check (const_accel on the gas)
  Real ax = 2.0*omega0*etavk;
  {
    bool ok = pin->GetOrAddBoolean("hydro_srcterms","const_accel",false);
    Real val = ok ? pin->GetReal("hydro_srcterms","const_accel_val") : 0.0;
    int dir = ok ? pin->GetInteger("hydro_srcterms","const_accel_dir") : 0;
    if (!ok || dir != 1 || fabs(val - ax) > 1.0e-12*fabs(ax)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Streaming test requires <hydro_srcterms> const_accel "
                << "with const_accel_dir = 1 and const_accel_val = 2*omega0*etavk = "
                << ax << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // NSH background drift
  Real ugx, ugp, vnx, vnp;
  SolveNSH2(omega0, qshear, ax, eps, taus, ugx, ugp, vnx, vnp);

  // initialize gas: NSH background + eigenmode at t=0
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int nx1 = indcs.nx1, nx2 = indcs.nx2;
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;
  par_for("stream_gas", DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    Real z = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
    Real ce = cos(kx*x), se = sin(kx*x);
    Real cz = cos(kz*z), sz = sin(kz*z);
    Real rho = rho0 + (Agr*ce - Agi*se)*cz;
    Real ux  = ugx  + (Uxr*ce - Uxi*se)*cz;
    Real up  = ugp  + (Upr*ce - Upi*se)*cz;
    Real uz  =      - (Uzr*se + Uzi*ce)*sz;
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*ux;
    u0(m,IM2,k,j,i) = rho*uz;   // vertical (x2) component
    u0(m,IM3,k,j,i) = rho*up;   // azimuthal component
  });

  // initialize particles: quiet-start lattice displaced radially to imprint the dust
  // density wave, with velocities = NSH + eigenmode evaluated at the unperturbed
  // positions. For ppc > 1, the ppc = Npar^2 particles of each cell are placed on an
  // Npar x Npar subcell lattice (as in classic Athena's streaming2d, parnumcell) so
  // they are genuinely distinct phase-space samples, not coincident copies.
  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto gids = pmbp->gids;
  int npart_permb = npart/nmb;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  int ppc_int = npart_permb/ncells;
  int npar1d = static_cast<int>(std::round(std::sqrt(static_cast<Real>(ppc_int))));
  if ((npart_permb != ppc_int*ncells) || (npar1d*npar1d != ppc_int)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Streaming test requires <particles>/ppc to be a perfect square "
              << "(1, 4, 9, 16, ...) for the quiet-start subcell lattice" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);
  Real rhop0 = eps*rho0;
  Real taus0 = taus;
  int lnx1 = indcs.nx1, lnx2 = indcs.nx2;
  par_for("stream_part", DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = p/npart_permb;
    if (m > (nmb-1)) {m = nmb-1;}
    pi(PGID,p) = gids + m;
    int q = p - m*npart_permb;
    int c = q/ppc_int;
    int sub = q - c*ppc_int;
    int ip = sub % npar1d;
    int jp = sub / npar1d;
    int i = c % lnx1;
    int j = (c/lnx1) % lnx2;
    Real dx1 = mbsize.d_view(m).dx1;
    Real dx2 = mbsize.d_view(m).dx2;
    // subcell quiet-start position: cell left edge + (ip+1/2) dx/Npar
    Real x0 = mbsize.d_view(m).x1min + (static_cast<Real>(i)
              + (static_cast<Real>(ip) + 0.5)/static_cast<Real>(npar1d))*dx1;
    Real z0 = mbsize.d_view(m).x2min + (static_cast<Real>(j)
              + (static_cast<Real>(jp) + 0.5)/static_cast<Real>(npar1d))*dx2;
    Real ce = cos(kx*x0), se = sin(kx*x0);
    Real cz = cos(kz*z0), sz = sin(kz*z0);
    // radial displacement imprinting drho_p = Apr*cos(kx x)*cos(kz z)
    pr(IPX,p) = x0 - (Apr/(kx*rhop0))*se*cz;
    pr(IPY,p) = z0;
    pr(IPZ,p) = 0.0;
    pi(PSP,p) = 0;
    pr(IPTS,p) = taus0;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    pr(IPM,p) = eps*rho0*vol/ppc;
    pr(IPVX,p) = vnx + (Vxr*ce - Vxi*se)*cz;
    pr(IPVY,p) =     - (Vzr*se + Vzi*ce)*sz;   // vertical (x2) component
    pr(IPVZ,p) = vnp + (Vpr*ce - Vpi*se)*cz;   // azimuthal component
    pr(IPRX,p) = 0.0;
    pr(IPRY,p) = 0.0;
    pr(IPRZ,p) = 0.0;
  });

  // particle timestep from drift speeds (perturbations are negligible)
  Real vmax = std::max({fabs(vnx), fabs(vnp), 1.0e-30});
  Real dtnew = std::numeric_limits<float>::max();
  auto &msize = pmbp->pmb->mb_size;
  dtnew = std::min(dtnew, msize.h_view(0).dx1/vmax);
  dtnew = std::min(dtnew, msize.h_view(0).dx2/vmax);
  ppar->dtnew = dtnew;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void StreamingLinearHistory()
//! \brief User history: complex Fourier projections onto the seeded (kx,kz) mode.
//! Records Re/Im of  sum_p m_p e^{-i kx x_p} cos(kz z_p)   (particle mass distribution)
//! and of  sum_cells (rho_g - rho0) e^{-i kx x} cos(kz z) * Vcell  and the same
//! projection of the gas radial momentum. |projection| grows as e^{st}.

void StreamingLinearHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  pdata->nhist = 6;
  pdata->label[0] = "rhop_re";
  pdata->label[1] = "rhop_im";
  pdata->label[2] = "rhog_re";
  pdata->label[3] = "rhog_im";
  pdata->label[4] = "mgx_re";
  pdata->label[5] = "mgx_im";

  Real kx = kx_, kz = kz_, rho0 = rho0_;

  // particle projection
  auto &pr = ppar->prtcl_rdata;
  int npart = ppar->nprtcl_thispack;
  Real pre = 0.0, pim = 0.0;
  Kokkos::parallel_reduce("stream_hp",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int &p, Real &re_, Real &im_) {
    Real w = pr(IPM,p)*cos(kz*pr(IPY,p));
    re_ += w*cos(kx*pr(IPX,p));
    im_ -= w*sin(kx*pr(IPX,p));
  }, Kokkos::Sum<Real>(pre), Kokkos::Sum<Real>(pim));
  pdata->hdata[0] = pre;
  pdata->hdata[1] = pim;

  // gas projections over active cells
  auto &indcs = pm->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;
  Real gre = 0.0, gim = 0.0, mre = 0.0, mim = 0.0;
  Kokkos::parallel_reduce("stream_hg",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &gr_, Real &gi_, Real &mr_, Real &mi_) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real x = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    Real z = CellCenterX(j-js, nx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    Real wc = vol*cos(kz*z);
    Real drho = u0(m,IDN,k,j,i) - rho0;
    gr_ += drho*wc*cos(kx*x);
    gi_ -= drho*wc*sin(kx*x);
    mr_ += u0(m,IM1,k,j,i)*wc*cos(kx*x);
    mi_ -= u0(m,IM1,k,j,i)*wc*sin(kx*x);
  }, Kokkos::Sum<Real>(gre), Kokkos::Sum<Real>(gim),
     Kokkos::Sum<Real>(mre), Kokkos::Sum<Real>(mim));
  pdata->hdata[2] = gre;
  pdata->hdata[3] = gim;
  pdata->hdata[4] = mre;
  pdata->hdata[5] = mim;
  return;
}
