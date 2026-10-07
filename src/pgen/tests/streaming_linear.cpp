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
#include <map>
#include <string>
#include <tuple>
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
bool snapshots_ = false;
int snapshot_number_ = 0;
// refinable clouds: the particle projection is weighted by the cloud's Fourier
// transform sinc^2(k ell/2) per direction, which is invariant under the two-scale split
bool cloud_mode_ = false;
Real cloud_root_dx1_ = 0.0, cloud_root_dx2_ = 0.0;
std::string snapshot_basename_;
int snapshot_every_ = 1, snapshot_calls_ = 0;   // write every N-th history call

// Opt-in, single-rank diagnostic output at the history cadence. Native binary and
// particle-VTK output round to float, which erases the small LinA perturbations.
// Header: ASCII magic, then time/cycle/ncell/nparticle. Payload: native-endian f64
// cell rows [x,z,dx,dz,rho,mx,mz,mphi], then particle rows
// [x,z,m,vx,vz,vphi,tag,level] (v2; v1 had no level column).
void WriteStreamingSnapshot(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto &ind = pm->mb_indcs;
  auto gas = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pack->phydro->u0);
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                               pack->ppart->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                               pack->ppart->prtcl_idata);
  int ncell = pack->nmb_thispack*ind.nx1*ind.nx2;
  int npart = pack->ppart->nprtcl_thispack;
  std::vector<double> cells;
  std::vector<double> particles;
  cells.reserve(8*ncell);
  particles.reserve(8*npart);
  for (int m=0; m<pack->nmb_thispack; ++m) {
    const auto &size = pack->pmb->mb_size.h_view(m);
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i) {
        cells.insert(cells.end(), {size.x1min + (i-ind.is+0.5)*size.dx1,
            size.x2min + (j-ind.js+0.5)*size.dx2, size.dx1, size.dx2,
            gas(m,IDN,ind.ks,j,i), gas(m,IM1,ind.ks,j,i),
            gas(m,IM2,ind.ks,j,i), gas(m,IM3,ind.ks,j,i)});
      }
    }
  }
  for (int p=0; p<npart; ++p) {
    particles.insert(particles.end(), {pr(IPX,p), pr(IPY,p), pr(IPM,p),
        pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p), static_cast<double>(pi(PTAG,p)),
        static_cast<double>(pi(PLEV,p))});
  }
  char number[32];
  std::snprintf(number, sizeof(number), ".%05d.lina", snapshot_number_++);
  std::string filename = snapshot_basename_ + number;
  FILE *file = std::fopen(filename.c_str(), "wb");
  if (file == nullptr) {
    std::cerr << "Cannot open LinA snapshot " << filename << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool ok = std::fprintf(file, "AthenaK LinA snapshot v2 f64-native\n"
      "%.17g %d %d %d\n", pm->time, pm->ncycle, ncell, npart) > 0;
  ok = (std::fwrite(cells.data(), sizeof(double), cells.size(), file)
        == cells.size()) && ok;
  ok = (std::fwrite(particles.data(), sizeof(double), particles.size(), file)
        == particles.size()) && ok;
  ok = (std::fclose(file) == 0) && ok;
  if (!ok) {
    std::cerr << "Failed writing LinA snapshot " << filename << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

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
  snapshots_ = pin->GetOrAddBoolean("problem", "double_precision_snapshots", false);
  if (snapshots_) {
    if (restart || global_variable::nranks != 1 ||
        !pin->GetOrAddBoolean("problem", "user_hist", false)) {
      std::cerr << "LinA snapshots require a fresh, single-rank run with user_hist=true"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    snapshot_basename_ = pin->GetString("job", "basename");
    snapshot_number_ = 0;
    snapshot_every_ = std::max(1, pin->GetOrAddInteger("problem", "snapshot_every", 1));
    snapshot_calls_ = 0;
  }
  if (restart && pin->GetOrAddBoolean("problem", "prescribed_gas", false)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/prescribed_gas (diagnostic) requires a fresh run" << std::endl;
    std::exit(EXIT_FAILURE);
  }
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

  // Optional radial Galilean boost W of the whole equilibrium (gas and dust).  In the
  // axisymmetric shearing box a uniform radial velocity changes only the azimuthal
  // Coriolis balance, by -(2-q) Omega W, which a constant azimuthal acceleration
  // (2-q) Omega W on both species restores exactly; the perturbation equations are then
  // those of the unboosted problem seen from a frame moving with -W, so the growth rate
  // is unchanged and the frequency is Doppler shifted by kx W.  Used to hold the dust,
  // or the eigenmode pattern, at rest relative to the mesh.
  Real boost = pin->GetOrAddReal("problem", "boost_vx", 0.0);
  if (boost != 0.0) {
    Real aphi = (2.0 - qshear)*omega0*boost;
    bool ok = pin->GetOrAddBoolean("hydro_srcterms", "const_accel2", false);
    Real gval = ok ? pin->GetReal("hydro_srcterms", "const_accel2_val") : 0.0;
    int gdir = ok ? pin->GetInteger("hydro_srcterms", "const_accel2_dir") : 0;
    Real dval = pmbp->pdust->const_accel_phi;
    if (!ok || gdir != 3 || fabs(gval - aphi) > 1.0e-12*fabs(aphi) ||
        fabs(dval - aphi) > 1.0e-12*fabs(aphi)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "boost_vx = " << boost << " requires <hydro_srcterms> "
                << "const_accel2 with const_accel2_dir = 3 and <dust>/const_accel_phi, both "
                << "equal to (2-q)*omega0*boost_vx = " << aphi << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ugx += boost;
    vnx += boost;
    if (global_variable::my_rank == 0) {
      std::cout << "# streaming_linear: radial boost W = " << boost << ", gas vx = " << ugx
                << ", dust vx = " << vnx << ", azimuthal accel = " << aphi << std::endl;
    }
  }

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

  // Diagnostic prescribed gas: the gas is the analytic
  // eigenmode at all times, as cell averages over every cell (ghosts included),
  //   f(x,z,t) = [Re f~ cos(kx x - wR t) - Im f~ sin(kx x - wR t)] e^{st} cos(kz z)
  // (odd fields with sin(kz z)); products rho*u are formed from the cell averages (the
  // neglected sub-cell covariance is second order in the perturbation).  The dust is
  // advanced by the usual PC2 gather and kick against this field and does not act on
  // the gas (back_reaction = false).  Requires coupling = pc2 and no boost.
  if (pin->GetOrAddBoolean("problem", "prescribed_gas", false)) {
    auto *pdust = pmbp->pdust;
    if (pdust->back_reaction || pdust->coupling != DustCoupling::hybrid ||
        pdust->hybrid_force_mode != HybridForceMode::pc2 || boost != 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<problem>/prescribed_gas requires <dust>/back_reaction = "
                << "false, <dust>/coupling = pc2 and boost_vx = 0" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Real srate = pin->GetReal("problem", "growth_rate");
    Real wrate = pin->GetReal("problem", "omega_re");
    int nmb_ = nmb;
    int is_ = is, js_ = js;
    auto mbsize_ = mbsize;
    pdust->prescribed_gas = [=](Real t, DvceArray5D<Real> &u) {
      Real amp = std::exp(srate*t), phase = wrate*t;
      int n2 = u.extent_int(2), n3 = u.extent_int(3), n4 = u.extent_int(4);
      par_for("prescribed_gas", DevExeSpace(), 0, nmb_-1, 0, n2-1, 0, n3-1, 0, n4-1,
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        Real dx = mbsize_.d_view(m).dx1, dz = mbsize_.d_view(m).dx2;
        Real xc = mbsize_.d_view(m).x1min + (i - is_ + 0.5)*dx;
        Real zc = mbsize_.d_view(m).x2min + (j - js_ + 0.5)*dz;
        // cell average of cos/sin(k x + c) = cos/sin(k x_c + c) sin(k dx/2)/(k dx/2)
        Real fx = sin(0.5*kx*dx)/(0.5*kx*dx), fz = sin(0.5*kz*dz)/(0.5*kz*dz);
        Real ce = amp*fx*cos(kx*xc - phase), se = amp*fx*sin(kx*xc - phase);
        Real cz = fz*cos(kz*zc), sz = fz*sin(kz*zc);
        Real rho = rho0 + (Agr*ce - Agi*se)*cz;
        Real ux  = ugx  + (Uxr*ce - Uxi*se)*cz;
        Real up  = ugp  + (Upr*ce - Upi*se)*cz;
        Real uz  =      - (Uzr*se + Uzi*ce)*sz;
        u(m,IDN,k,j,i) = rho;
        u(m,IM1,k,j,i) = rho*ux;
        u(m,IM2,k,j,i) = rho*uz;
        u(m,IM3,k,j,i) = rho*up;
      });
    };
    if (global_variable::my_rank == 0) {
      std::cout << "# streaming_linear: prescribed analytic gas (s = " << srate
                << ", wR = " << wrate << "), dust back-reaction off" << std::endl;
    }
  }

  // Diagnostic position oracle: with the prescribed gas the linear
  // dust velocity is the eigenmode's at all times, so the exact dust displacement of a
  // particle whose unperturbed position is X (X0 = X - vnx t at t = 0) is
  //   xi(X,t) = xi_eig(X,t) + [xi_seed(X0) - xi_eig(X0,0)],
  //   xi_eig = Re[v~/(s - i Omega) e^{i(kx X - wR t)}] e^{st} (cos kz z; odd field with
  //   i v~ and sin kz z), Omega = wR - kx vnx,  xi_seed = -(Apr/(kx rho_p0)) sin(kx X0) cos(kz z)
  // (the seed imprints a radial displacement only).  The dust code uses it to place the
  // pieces of in-run splits (DustGasDrag::SplitCloudsP1).
  if (pin->GetOrAddBoolean("problem", "split_oracle_velocity", false) &&
      !pin->GetOrAddBoolean("problem", "split_oracle", false)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/split_oracle_velocity requires split_oracle" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pin->GetOrAddBoolean("problem", "split_oracle_node", false) &&
      !pin->GetOrAddBoolean("problem", "split_oracle", false)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/split_oracle_node requires split_oracle" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // The same displacement is exact for the two-way linear problem as well (2026-10-03):
  // the eigenmode with dust velocity v~ solves the coupled linear equations, and the seed
  // offset xi_seed - xi_eig(X0,0) is solenoidal (same dust density), so it changes neither
  // the dust density nor the drag at linear order; the oracle then gives the exact linear
  // state, not the numerical two-way solution.  No boost (Omega uses the unboosted drift).
  if (pin->GetOrAddBoolean("problem", "split_oracle", false)) {
    if (!pin->GetOrAddBoolean("problem", "seed_after_split", false) || boost != 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<problem>/split_oracle requires seed_after_split and "
                << "boost_vx = 0" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real srate = pin->GetReal("problem", "growth_rate");
    const Real wrate = pin->GetReal("problem", "omega_re");
    const Real om = wrate - kx*vnx;
    const Real rp0 = eps*rho0;
    // c = v~/(s - i Omega) for x and i v~/(s - i Omega) for z
    const Real den = srate*srate + om*om;
    const Real cxr = (Vxr*srate - Vxi*om)/den, cxi = (Vxi*srate + Vxr*om)/den;
    const Real vzr = (Vzr*srate - Vzi*om)/den, vzi = (Vzi*srate + Vzr*om)/den;
    const Real czr = -vzi, czi = vzr;
    const Real kxx = kx, kzz = kz, vd = vnx, ap = Apr;
    pmbp->pdust->split_oracle_node =
        pin->GetOrAddBoolean("problem", "split_oracle_node", false);
    pmbp->pdust->split_oracle_node_meanfree =
        pin->GetOrAddInteger("problem", "split_oracle_node_meanfree", 0);
    if (pmbp->pdust->split_oracle_node_meanfree != 0 &&
        (!pmbp->pdust->split_oracle_node || pmbp->pdust->split_oracle_node_meanfree < 0 ||
         pmbp->pdust->split_oracle_node_meanfree > 2)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/split_oracle_node_meanfree must be 0, 1 or 2 and requires "
                << "split_oracle_node" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pmbp->pdust->split_oracle = [=](Real t, Real x, Real z, Real &dx, Real &dz) {
      const Real x0 = x - vd*t;
      const Real g = std::exp(srate*t), ph = kxx*x - wrate*t;
      const Real c1 = std::cos(ph), s1 = std::sin(ph), c0 = std::cos(kxx*x0), s0 = std::sin(kxx*x0);
      const Real cz = std::cos(kzz*z), sz = std::sin(kzz*z);
      dx = (g*(cxr*c1 - cxi*s1) - (cxr*c0 - cxi*s0) - (ap/(kxx*rp0))*s0)*cz;
      dz = (g*(czr*c1 - czi*s1) - (czr*c0 - czi*s0))*sz;
    };
    // optional velocity part: the exact dust velocity perturbation of the eigenmode,
    //   dv = Re[v~ e^{i(kx X - wR t)}] e^{st} cos kz z (x, phi), Re[i v~ ...] sin kz z (z)
    if (pin->GetOrAddBoolean("problem", "split_oracle_velocity", false)) {
      const Real vxr = Vxr, vxi = Vxi, vzr2 = Vzr, vzi2 = Vzi, vpr = Vpr, vpi = Vpi;
      pmbp->pdust->split_oracle_velocity = [=](Real t, Real x, Real z, Real &dvx, Real &dvz,
                                               Real &dvp) {
        const Real g = std::exp(srate*t), ph = kxx*x - wrate*t;
        const Real c1 = std::cos(ph), s1 = std::sin(ph);
        const Real cz = std::cos(kzz*z), sz = std::sin(kzz*z);
        dvx = g*(vxr*c1 - vxi*s1)*cz;
        dvz = -g*(vzr2*s1 + vzi2*c1)*sz;
        dvp = g*(vpr*c1 - vpi*s1)*cz;
      };
    }
    if (global_variable::my_rank == 0) {
      std::cout << "# streaming_linear: split position oracle on (Omega = " << om << ")"
                << (pmbp->pdust->split_oracle_velocity ? ", with velocities" : "")
                << (pmbp->pdust->split_oracle_node ? ", node convention" : "")
                << (pmbp->pdust->split_oracle_node_meanfree == 1 ? ", mean-free velocity" :
                    (pmbp->pdust->split_oracle_node_meanfree == 2 ?
                     ", mean-free velocity and position" : ""))
                << std::endl;
    }
  }

  // Diagnostic analytic gather: the dust sees the analytic gas
  // velocity at t+dt/2 and the predicted midpoint position with the cloud response of
  // the cloud_transition = 2 gather, instead of the gather from the gas cells
  // (DustGasDrag::AnalyticGather).  The prescribed gas is kept for everything else.
  if (pin->GetOrAddBoolean("problem", "analytic_gather", false)) {
    if (!pin->GetOrAddBoolean("problem", "prescribed_gas", false)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<problem>/analytic_gather requires prescribed_gas"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto &ag = pmbp->pdust->analytic_gather;
    ag.on = true;
    ag.point_all = pin->GetOrAddBoolean("problem", "analytic_gather_point", false);
    ag.centroid_lobes = pin->GetOrAddBoolean("problem", "analytic_gather_centroid", false);
    ag.kx = kx; ag.kz = kz;
    ag.srate = pin->GetReal("problem", "growth_rate");
    ag.wrate = pin->GetReal("problem", "omega_re");
    ag.ux0 = ugx; ag.up0 = ugp;
    ag.uxr = Uxr; ag.uxi = Uxi; ag.uzr = Uzr; ag.uzi = Uzi; ag.upr = Upr; ag.upi = Upi;
    if (global_variable::my_rank == 0) {
      std::cout << "# streaming_linear: analytic gather on"
                << (ag.point_all ? " (point value for every cloud)" : "")
                << (ag.centroid_lobes ? " (lobed directions at the lobe centroid)" : "")
                << std::endl;
    }
  }

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
  // optional radial (and vertical) shift of the quiet-start lattice, in units of the
  // (root) cell: keeps a lattice held at rest by the boost off the cell faces
  Real lshift = pin->GetOrAddReal("problem", "lattice_shift", 0.0);
  Real lshiftz = pin->GetOrAddReal("problem", "lattice_shift_z", 0.0);
  Real rhop0 = eps*rho0;
  Real taus0 = taus;
  int lnx1 = indcs.nx1, lnx2 = indcs.nx2;

  // Refinable clouds: the quiet-start lattice is the ROOT lattice everywhere (one
  // root-level cloud per root cell, ppc sub-lattice inside it), independent of the
  // local refinement.  Clouds lying inside the fine region are then split by the exact
  // two-scale rule, exactly as they will be at run time, so the refined lattice is
  // built by the same rule as the dynamics.  By default every particle keeps its
  // analytic seed (displacement and velocity) from its parent's lattice site, so a finer
  // region starts with the seed sampled at the root spacing.  With
  // <problem>/seed_after_split = true the unperturbed root lattice is split first and
  // every particle then receives the seed at its own lattice site (diagnostic).
  cloud_mode_ = (pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud);
  if (cloud_mode_) {
    auto &transfer = pmbp->pdust->smr_transfer;
    const bool seed_after = pin->GetOrAddBoolean("problem", "seed_after_split", false);
    // analytic LinA seed of the particle at lattice site (x0, z0)
    auto imprint = [&](auto &h, int p, Real x0, Real z0) {
      Real ce = std::cos(kx*x0), se = std::sin(kx*x0);
      Real cz = std::cos(kz*z0), sz = std::sin(kz*z0);
      h(IPX,p) = x0 - (Apr/(kx*rhop0))*se*cz;
      h(IPY,p) = z0;
      h(IPX1,p) = h(IPX,p);
      h(IPY1,p) = z0;
      h(IPVX,p) = vnx + (Vxr*ce - Vxi*se)*cz;
      h(IPVY,p) =     - (Vzr*se + Vzi*ce)*sz;
      h(IPVZ,p) = vnp + (Vpr*ce - Vpi*se)*cz;
      h(IPVX1,p) = h(IPVX,p);
      h(IPVY1,p) = h(IPVY,p);
      h(IPVZ1,p) = h(IPVZ,p);
      if (ppar->nrdata > IPGZY) {   // deformable clouds: analytic velocity gradient
        h(IPGXX,p) = -kx*(Vxr*se + Vxi*ce)*cz;
        h(IPGXY,p) = -kz*(Vxr*ce - Vxi*se)*sz;
        h(IPGYX,p) = -kx*(Vzr*ce - Vzi*se)*sz;
        h(IPGYY,p) = -kz*(Vzr*se + Vzi*ce)*cz;
        h(IPGZX,p) = -kx*(Vpr*se + Vpi*ce)*cz;
        h(IPGZY,p) = -kz*(Vpr*ce - Vpi*se)*sz;
      }
    };
    cloud_root_dx1_ = transfer.root_dx1;
    cloud_root_dx2_ = transfer.root_dx2;
    Real hx = transfer.root_dx1, hz = transfer.root_dx2;
    auto &mblev = pmbp->pmb->mb_lev;
    std::vector<int> first(nmb + 1, 0);
    std::vector<int> nroot1(nmb, 0), nroot2(nmb, 0);
    for (int m=0; m<nmb; ++m) {
      int ratio = 1 << (mblev.h_view(m) - pmy_mesh_->root_level);
      if ((lnx1 % ratio) != 0 || (lnx2 % ratio) != 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "refinable_cloud lattice requires MeshBlocks that "
                  << "contain whole root cells" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      nroot1[m] = lnx1/ratio;
      nroot2[m] = lnx2/ratio;
      first[m+1] = first[m] + ppc_int*nroot1[m]*nroot2[m];
    }
    int nlattice = first[nmb];
    Kokkos::realloc(pr, ppar->nrdata, nlattice);
    Kokkos::realloc(pi, ppar->nidata, nlattice);
    auto hr = Kokkos::create_mirror_view(pr);
    auto hi = Kokkos::create_mirror_view(pi);
    for (int m=0; m<nmb; ++m) {
      const auto &size = mbsize.h_view(m);
      for (int q=0; q<first[m+1]-first[m]; ++q) {
        int p = first[m] + q;
        int c = q/ppc_int;
        int sub = q - c*ppc_int;
        int ip = sub % npar1d;
        int jp = sub / npar1d;
        int i = c % nroot1[m];
        int j = c / nroot1[m];
        Real x0 = size.x1min + (static_cast<Real>(i) + lshift
                  + (static_cast<Real>(ip) + 0.5)/static_cast<Real>(npar1d))*hx;
        Real z0 = size.x2min + (static_cast<Real>(j) + lshiftz
                  + (static_cast<Real>(jp) + 0.5)/static_cast<Real>(npar1d))*hz;
        hi(PGID,p) = gids + m;
        hi(PTAG,p) = p;
        hi(PSP,p) = 0;
        hi(PLEV,p) = 0;
        for (int col=0; col<ppar->nrdata; ++col) hr(col,p) = 0.0;
        hr(IPTS,p) = taus0;
        hr(IPM,p) = eps*rho0*hx*hz/ppc;
        if (seed_after) {   // unperturbed lattice; the seed is imprinted after the split
          hr(IPX,p) = x0;
          hr(IPY,p) = z0;
          hr(IPX1,p) = x0;
          hr(IPY1,p) = z0;
          hr(IPVX,p) = vnx;
          hr(IPVZ,p) = vnp;
          hr(IPVX1,p) = vnx;
          hr(IPVZ1,p) = vnp;
        } else {
          imprint(hr, p, x0, z0);
        }
      }
    }
    Kokkos::deep_copy(pr, hr);
    Kokkos::deep_copy(pi, hi);
    ppar->nprtcl_thispack = nlattice;
    pmy_mesh_->nprtcl_thisrank = nlattice;
    pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = nlattice;
    pmy_mesh_->nprtcl_total = nlattice;
    // <problem>/graded_split_layers = D >= 0 (diagnostic): instead of
    // the run-time split, build a graded normal transition of a STATIC refined region:
    // the equal-weight superposition of the D+1 static P1 + overlap split sets whose
    // refined region is dilated by k = 0..D root cells (each set is P1-consistent, so the
    // sum is), coincident pieces of equal code combined.  One refinement level, ppc = 1,
    // seed_after_split, no merging, no lobe release; outside the refined region nothing
    // is eligible at run time, so the graded pieces are never split again.
    const int glayers = pin->GetOrAddInteger("problem", "graded_split_layers", -1);
    int ntrans = 0;
    if (glayers >= 0) {
      auto *pd = pmbp->pdust;
      if (transfer.cloud_nlev != 1 || ppc_int != 1 || !seed_after || !pd->cloud_p1_split ||
          pd->cloud_merge || pd->cloud_p1_release_exit || !transfer.cloud_split_overlap ||
          transfer.cloud_split_everywhere || global_variable::nranks != 1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<problem>/graded_split_layers requires one refinement "
                  << "level, ppc = 1, seed_after_split, cloud_p1_split with overlap, no "
                  << "merging, no lobe release, no split everywhere, one rank" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const int nf1 = transfer.nfine1, nf2 = transfer.nfine2;
      const Real fd1 = transfer.fine_dx1, fd2 = transfer.fine_dx2;
      const Real gx1min = transfer.x1min, gx2min = transfer.x2min;
      const Real glen1 = transfer.x1max - gx1min, glen2 = transfer.x2max - gx2min;
      auto lev_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_level);
      // refined region dilated by k root cells (2k finest cells), k = 0..D
      std::vector<std::vector<char>> dil(glayers + 1,
          std::vector<char>(static_cast<std::size_t>(nf1)*nf2, 0));
      for (int fj=0; fj<nf2; ++fj) {
        for (int fi=0; fi<nf1; ++fi) dil[0][static_cast<std::size_t>(fj)*nf1 + fi] =
            lev_h(fj, fi) >= 1 ? 1 : 0;
      }
      for (int k=1; k<=glayers; ++k) {
        for (int fj=0; fj<nf2; ++fj) {
          for (int fi=0; fi<nf1; ++fi) {
            char v = 0;
            for (int dj=-2; dj<=2 && !v; ++dj) {
              const int wj = SMRCompositeTransfer::WrapIndex(fj + dj, nf2);
              for (int di=-2; di<=2 && !v; ++di) {
                v = dil[k-1][static_cast<std::size_t>(wj)*nf1 +
                             SMRCompositeTransfer::WrapIndex(fi + di, nf1)];
              }
            }
            dil[k][static_cast<std::size_t>(fj)*nf1 + fi] = v;
          }
        }
      }
      auto gwrap = [](Real x, Real xmin, Real len) {
        while (x < xmin) x += len;
        while (x >= xmin + len) x -= len;
        return x;
      };
      // run-time eligibility of a root cloud (overlap rule) against the dilated region
      const Real slack = pd->cloud_split_slack;
      auto elig = [&](int k, Real x, Real y) {
        x = gwrap(x, gx1min, glen1); y = gwrap(y, gx2min, glen2);
        const Real r1 = (1.0 - slack)*transfer.CloudHalfWidth1(0);
        const Real r2 = (1.0 - slack)*transfer.CloudHalfWidth2(0);
        const int ia = static_cast<int>(std::floor((x - r1 - gx1min)/fd1));
        const int ib = static_cast<int>(std::ceil((x + r1 - gx1min)/fd1)) - 1;
        const int ja = static_cast<int>(std::floor((y - r2 - gx2min)/fd2));
        const int jb = static_cast<int>(std::ceil((y + r2 - gx2min)/fd2)) - 1;
        for (int fj=ja; fj<=jb; ++fj) {
          const int wj = SMRCompositeTransfer::WrapIndex(fj, nf2);
          for (int fi=ia; fi<=ib; ++fi) {
            if (dil[k][static_cast<std::size_t>(wj)*nf1 +
                       SMRCompositeTransfer::WrapIndex(fi, nf1)]) return true;
          }
        }
        return false;
      };
      struct GP {Real x, y, m; int code;};
      std::map<std::tuple<long long, long long, int>, GP> acc;
      auto add = [&](Real x, Real y, Real m, int code) {
        x = gwrap(x, gx1min, glen1); y = gwrap(y, gx2min, glen2);
        auto key = std::make_tuple(std::llround(x/(1.0e-6*fd1)), std::llround(y/(1.0e-6*fd2)),
                                   code);
        auto it = acc.find(key);
        if (it == acc.end()) acc[key] = GP{x, y, m, code};
        else it->second.m += m;
      };
      const Real e1 = transfer.CloudHalfWidth1(1), e2 = transfer.CloudHalfWidth2(1);
      const Real h1 = transfer.CloudHalfWidth1(0), h2 = transfer.CloudHalfWidth2(0);
      const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
      const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
      for (int p=0; p<nlattice; ++p) {
        const Real x = hr(IPX,p), y = hr(IPY,p), m = hr(IPM,p)/static_cast<Real>(glayers + 1);
        for (int k=0; k<=glayers; ++k) {
          if (!elig(k, x, y)) {add(x, y, m, 0); continue;}
          // pieces along one direction: (offset in child half-widths, fraction, lobe)
          auto dir = [](bool lo_sep, bool hi_sep, Real off[3], Real fr[3], int &lobe) {
            int n = 0;
            lobe = (lo_sep ? 0 : 1) | (hi_sep ? 0 : 2);
            if (lo_sep) {off[n] = -1.0; fr[n++] = 0.25;}
            off[n] = 0.0; fr[n++] = 0.5 + (lo_sep ? 0.0 : 0.25) + (hi_sep ? 0.0 : 0.25);
            if (hi_sep) {off[n] = 1.0; fr[n++] = 0.25;}
            return n;
          };
          Real o1[3], f1[3], o2[3], f2[3]; int l1, l2;
          const int n1 = dir(elig(k, x - h1, y), elig(k, x + h1, y), o1, f1, l1);
          const int n2 = dir(elig(k, x, y - h2), elig(k, x, y + h2), o2, f2, l2);
          for (int b=0; b<n2; ++b) {
            for (int a=0; a<n1; ++a) {
              const int c1 = (o1[a] == 0.0) ? l1 : 0, c2 = (o2[b] == 0.0) ? l2 : 0;
              add(x + o1[a]*e1, y + o2[b]*e2, m*f1[a]*f2[b], 1 | (c1 << sh1) | (c2 << sh2));
            }
          }
        }
      }
      const int ng = static_cast<int>(acc.size());
      Kokkos::realloc(pr, ppar->nrdata, ng);
      Kokkos::realloc(pi, ppar->nidata, ng);
      auto gr = Kokkos::create_mirror_view(pr);
      auto gi = Kokkos::create_mirror_view(pi);
      int s = 0;
      for (const auto &kv : acc) {
        const GP &g = kv.second;
        for (int col=0; col<ppar->nrdata; ++col) gr(col,s) = 0.0;
        gr(IPX,s) = g.x; gr(IPX1,s) = g.x; gr(IPY,s) = g.y; gr(IPY1,s) = g.y;
        gr(IPM,s) = g.m; gr(IPTS,s) = taus0;
        gr(IPVX,s) = vnx; gr(IPVZ,s) = vnp; gr(IPVX1,s) = vnx; gr(IPVZ1,s) = vnp;
        int owner = 0;
        for (int mb=0; mb<nmb; ++mb) {
          const auto &size = mbsize.h_view(mb);
          if (g.x >= size.x1min && g.x < size.x1max && g.y >= size.x2min && g.y < size.x2max) {
            owner = mb; break;
          }
        }
        gi(PGID,s) = gids + owner; gi(PTAG,s) = s; gi(PSP,s) = 0; gi(PLEV,s) = g.code;
        ++s;
      }
      Kokkos::deep_copy(pr, gr);
      Kokkos::deep_copy(pi, gi);
      ppar->nprtcl_thispack = ng;
      pmy_mesh_->nprtcl_thisrank = ng;
      pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = ng;
      pmy_mesh_->nprtcl_total = ng;
      if (global_variable::my_rank == 0) {
        std::cout << "# graded_split_layers = " << glayers << ": " << nlattice
                  << " root clouds -> " << ng << " combined pieces" << std::endl;
      }
    } else {
      // one pass per refinement level: children of the first pass may lie inside a
      // still finer region
      while (pmbp->pdust->SplitAndMergeClouds() > 0) {}
      ntrans = pmbp->pdust->MakeTransitionClouds();
    }
    if (seed_after) {
      auto h2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
      auto i2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pi);
      const auto &msz = pmy_mesh_->mesh_size;
      const Real len1 = msz.x1max - msz.x1min, len2 = msz.x2max - msz.x2min;
      for (int p=0; p<ppar->nprtcl_thispack; ++p) {
        imprint(h2, p, h2(IPX,p), h2(IPY,p));
        Real x = h2(IPX,p), z = h2(IPY,p);
        if (x < msz.x1min) x += len1;
        if (x >= msz.x1max) x -= len1;
        h2(IPX,p) = x;
        h2(IPX1,p) = x;
        int owner = -1;
        for (int m=0; m<nmb && owner<0; ++m) {
          const auto &size = mbsize.h_view(m);
          if (x >= size.x1min && x < size.x1max && z >= size.x2min && z < size.x2max) {
            owner = m;
          }
        }
        if (owner < 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "seed_after_split: particle outside every MeshBlock"
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
        i2(PGID,p) = gids + owner;
      }
      Kokkos::deep_copy(pr, h2);
      Kokkos::deep_copy(pi, i2);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "# refinable_cloud lattice: " << nlattice << " root clouds -> "
                << ppar->nprtcl_thispack << " particles after the initial split"
                << " (" << ntrans << " edge children merged into transition clouds)"
                << (seed_after ? ", seed imprinted after the split" : "") << std::endl;
    }
  } else {
  int nrdata = ppar->nrdata;
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
    Real x0 = mbsize.d_view(m).x1min + (static_cast<Real>(i) + lshift
              + (static_cast<Real>(ip) + 0.5)/static_cast<Real>(npar1d))*dx1;
    Real z0 = mbsize.d_view(m).x2min + (static_cast<Real>(j) + lshiftz
              + (static_cast<Real>(jp) + 0.5)/static_cast<Real>(npar1d))*dx2;
    Real ce = cos(kx*x0), se = sin(kx*x0);
    Real cz = cos(kz*z0), sz = sin(kz*z0);
    // radial displacement imprinting drho_p = Apr*cos(kx x)*cos(kz z)
    pr(IPX,p) = x0 - (Apr/(kx*rhop0))*se*cz;
    pr(IPY,p) = z0;
    pr(IPZ,p) = 0.0;
    pi(PSP,p) = 0;
    pi(PLEV,p) = 0;
    pr(IPTS,p) = taus0;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    pr(IPM,p) = eps*rho0*vol/ppc;
    pr(IPVX,p) = vnx + (Vxr*ce - Vxi*se)*cz;
    pr(IPVY,p) =     - (Vzr*se + Vzi*ce)*sz;   // vertical (x2) component
    pr(IPVZ,p) = vnp + (Vpr*ce - Vpi*se)*cz;   // azimuthal component
    pr(IPRX,p) = 0.0;
    pr(IPRY,p) = 0.0;
    pr(IPRZ,p) = 0.0;
    if (nrdata > IPGZY) {   // deformable clouds: analytic velocity gradient
      pr(IPGXX,p) = -kx*(Vxr*se + Vxi*ce)*cz;
      pr(IPGXY,p) = -kz*(Vxr*ce - Vxi*se)*sz;
      pr(IPGYX,p) = -kx*(Vzr*ce - Vzi*se)*sz;
      pr(IPGYY,p) = -kz*(Vzr*se + Vzi*ce)*cz;
      pr(IPGZX,p) = -kx*(Vpr*se + Vpi*ce)*cz;
      pr(IPGZY,p) = -kz*(Vpr*ce - Vpi*se)*sz;
    }
  });
  }

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
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  bool cloud = cloud_mode_;
  Real hx = cloud_root_dx1_, hz = cloud_root_dx2_;
  Real pre = 0.0, pim = 0.0;
  Kokkos::parallel_reduce("stream_hp",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int &p, Real &re_, Real &im_) {
    Real w = pr(IPM,p)*cos(kz*pr(IPY,p));
    int lobe1 = 0;
    Real cxr = 0.0, cxi = 0.0;
    if (cloud) {
      // hat cloud of half-width ell: Fourier transform sinc^2(k ell/2) per direction;
      // a transition cloud adds 1/3 of its hat displaced by -/+ ell
      int code = pi(PLEV,p);
      Real scale = 1.0/static_cast<Real>(1 << SMRCompositeTransfer::CloudLevel(code));
      Real ax = 0.5*kx*hx*scale, az = 0.5*kz*hz*scale;
      Real sx = (ax == 0.0) ? 1.0 : sin(ax)/ax;
      Real sz = (az == 0.0) ? 1.0 : sin(az)/az;
      int l1 = SMRCompositeTransfer::CloudLobe1(code);
      int l2 = SMRCompositeTransfer::CloudLobe2(code);
      // (lobe 3, both edge children attached: 1/2 at the particle, 1/4 at -/+ ell)
      if (l2 == 3) {
        Real d = hz*scale, z = pr(IPY,p);
        w = pr(IPM,p)*(0.5*cos(kz*z) + 0.25*cos(kz*(z - d)) + 0.25*cos(kz*(z + d)));
      } else if (l2) {
        Real d = ((l2 == 1) ? -hz : hz)*scale;
        w = pr(IPM,p)*((2.0/3.0)*cos(kz*pr(IPY,p)) + (1.0/3.0)*cos(kz*(pr(IPY,p) + d)));
      }
      if (l1 == 3) {
        Real d = hx*scale, x = pr(IPX,p);
        cxr = 0.5*cos(kx*x) + 0.25*cos(kx*(x - d)) + 0.25*cos(kx*(x + d));
        cxi = 0.5*sin(kx*x) + 0.25*sin(kx*(x - d)) + 0.25*sin(kx*(x + d));
        lobe1 = l1;
      } else if (l1) {
        Real d = ((l1 == 1) ? -hx : hx)*scale;
        cxr = (2.0/3.0)*cos(kx*pr(IPX,p)) + (1.0/3.0)*cos(kx*(pr(IPX,p) + d));
        cxi = (2.0/3.0)*sin(kx*pr(IPX,p)) + (1.0/3.0)*sin(kx*(pr(IPX,p) + d));
        lobe1 = l1;
      }
      w *= SQR(sx)*SQR(sz);
    }
    if (lobe1) {
      re_ += w*cxr;
      im_ -= w*cxi;
    } else {
      re_ += w*cos(kx*pr(IPX,p));
      im_ -= w*sin(kx*pr(IPX,p));
    }
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
  if (snapshots_ && (snapshot_calls_++ % snapshot_every_) == 0) WriteStreamingSnapshot(pm);
  return;
}
