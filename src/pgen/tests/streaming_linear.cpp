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
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <random>
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
#include "dust/dust_species.hpp"
#include "outputs/outputs.hpp"

// user history: Fourier projections onto the seeded mode
void StreamingLinearHistory(HistoryData *pdata, Mesh *pm);
void StreamingLinearHistory3D(HistoryData *pdata, Mesh *pm);   // 3-D refinable clouds
void StreamingKx0Filter(Mesh *pm, const Real bdt);
void RunMergeReplay(ParameterInput *pin, Mesh *pm, const Real taus0);
void RunAuditReplay(ParameterInput *pin, Mesh *pm);
bool kx0_filter_phi = false;  // also filter the azimuthal momentum (IM3 in the 2-D r-z box)
Real kx0_filter_tstart = 0.0;  // the filter acts from this time on

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
// native restart: the snapshot counters are kept in the input block
// <problem> (rst_snapshot_number, rst_snapshot_calls), which every restart file carries
ParameterInput *lina_pin_ = nullptr;

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
        pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p), static_cast<double>(GetTag(pi,p)),
        static_cast<double>(pi(PLEV,p))});
  }
  char number[32];
  std::snprintf(number, sizeof(number), ".%05d.lina", snapshot_number_++);
  if (lina_pin_ != nullptr) lina_pin_->SetInteger("problem", "rst_snapshot_number", snapshot_number_);
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
  dust::RequireLegacySpeciesInit(pin, "streaming_linear");  // legacy species keys only
  // in 3-D the mode lies in the x1-x3 plane (x2 the azimuth, the state uniform in x2)
  // (StreamingLinearHistory also projects the 3-D lattice with x3 vertical)
  user_hist_func = (pmy_mesh_->three_d && pmy_mesh_->pmb_pack->pdust != nullptr &&
                    pmy_mesh_->pmb_pack->pdust->Clouds3D()) ? StreamingLinearHistory3D
                                                           : StreamingLinearHistory;
  // experimental dynamic mesh: scheduled refinement request
  if (pmy_mesh_->adaptive && pin->DoesBlockExist("dust")) {
    dust::SetupScheduledRefinement(pin);
    user_ref_func = dust::ScheduledRefinement;
    // the dust-density request instead of the schedule
    if (pin->GetOrAddBoolean("dust", "amr_density_trigger", false)) {
      dust::SetupDensityTrigger(pin);
      user_ref_func = dust::DensityRefinement;
    }
    if (pin->GetOrAddBoolean("problem", "prescribed_gas", false)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/prescribed_gas keeps a copy of the mesh layout and is not "
                << "available with adaptive refinement" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // <problem>/kx0_filter (diagnostic): with <problem>/user_srcs = true,
  // remove the z-dependent k_x = 0 part of the gas radial momentum every stage
  if (pin->GetOrAddBoolean("problem", "kx0_filter", false)) user_srcs_func = StreamingKx0Filter;
  kx0_filter_phi = pin->GetOrAddBoolean("problem", "kx0_filter_phi", false);
  kx0_filter_tstart = pin->GetOrAddReal("problem", "kx0_filter_tstart", 0.0);
  // cache mode parameters for the history function (also needed on restart)
  kx_ = pin->GetReal("problem","kx");
  kz_ = pin->GetReal("problem","kz");
  rho0_ = pin->GetOrAddReal("problem","rho0",1.0);
  snapshots_ = pin->GetOrAddBoolean("problem", "double_precision_snapshots", false);
  if (snapshots_) {
    if (global_variable::nranks != 1 ||
        !pin->GetOrAddBoolean("problem", "user_hist", false)) {
      std::cerr << "LinA snapshots require a single-rank run with user_hist=true"
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
  lina_pin_ = pin;
  if (restart) {
    // native restart: gas, particles and the dust counters come from the
    // file; here only what the history function needs.  The split oracle keeps functions of
    // the initial state and is not available after a restart.
    if (pin->GetOrAddBoolean("problem", "split_oracle", false)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/split_oracle (diagnostic) requires a fresh run" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (snapshots_) {
      snapshot_number_ = pin->GetOrAddInteger("problem", "rst_snapshot_number", 0);
      snapshot_calls_ = pin->GetOrAddInteger("problem", "rst_snapshot_calls", 0);
    }
    MeshBlockPack *pr = pmy_mesh_->pmb_pack;
    if (pr->pdust != nullptr) {
      cloud_mode_ = (pr->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud);
      if (cloud_mode_) {
        cloud_root_dx1_ = pr->pdust->smr_transfer.root_dx1;
        cloud_root_dx2_ = pmy_mesh_->three_d ? pr->pdust->leaf_traversal3d.root_dx3
                                             : pr->pdust->smr_transfer.root_dx2;
      }
    }
    return;
  }

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr ||
      !(pin->DoesBlockExist("shearing_box"))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Streaming test requires <hydro>, <particles>, <dust>, <shearing_box>"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // also a 3-D shearing box (x1 radial, x2 azimuthal, x3 vertical),
  // the mode independent of x2, with the lattice (point particles; none of the 2-D diagnostics
  // below).  Refinable clouds in 3-D need <dust>/experimental_3d; their setup is
  // the 3-D branch below
  const bool lina3d = pmy_mesh_->three_d;
  if (pmbp->pdust->nspecies != 1 ||
      (lina3d && ((pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud &&
                   !pmbp->pdust->Clouds3D()) ||
                  pin->GetOrAddBoolean("problem", "prescribed_gas", false) ||
                  pin->GetOrAddBoolean("problem", "split_oracle", false) ||
                  pin->GetOrAddBoolean("problem", "analytic_gather", false) ||
                  pin->GetOrAddBoolean("problem", "double_precision_snapshots", false) ||
                  pin->GetOrAddBoolean("problem", "kx0_filter", false) ||
                  pin->GetOrAddReal("problem", "boost_vx", 0.0) != 0.0))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Streaming test requires a single dust species; in 3-D the lattice or 3-D "
              << "refinable clouds, without the 2-D diagnostics" << std::endl;
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

  // the 3-D LinA (x1 radial, x2 azimuthal, x3 vertical; the state uniform in x2):
  // refinable clouds on the root lattice (npar1d^2 per root cell in x1-x3, one per root cell in
  // x2), split to the leaves, the seed imprinted after the split with seed_after_split
  if (pmy_mesh_->three_d && pmbp->pdust->Clouds3D()) {
    for (const char *key : {"prescribed_gas", "split_oracle", "split_oracle_velocity",
                            "split_oracle_node", "analytic_gather", "kx0_filter",
                            "double_precision_snapshots"}) {
      if (pin->GetOrAddBoolean("problem", key, false)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "<problem>/" << key << " is not available in 3-D" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    if (boost != 0.0 || !pin->GetOrAddString("problem", "merge_replay", "").empty() ||
        !pin->GetOrAddString("problem", "merge_audit_replay", "").empty()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "boost_vx and the replays are not available in 3-D" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto &ind3 = pmy_mesh_->mb_indcs;
    const int is3 = ind3.is, ie3 = ind3.ie, js3 = ind3.js, je3 = ind3.je;
    const int ks3 = ind3.ks, ke3 = ind3.ke, n1 = ind3.nx1, n3 = ind3.nx3;
    const int nmb3 = pmbp->nmb_thispack;
    auto &ug = pmbp->phydro->u0;
    auto &msz3 = pmbp->pmb->mb_size;
    par_for("stream_gas3d", DevExeSpace(),0,(nmb3-1),ks3,ke3,js3,je3,is3,ie3,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      Real x = CellCenterX(i-is3, n1, msz3.d_view(m).x1min, msz3.d_view(m).x1max);
      Real z = CellCenterX(k-ks3, n3, msz3.d_view(m).x3min, msz3.d_view(m).x3max);
      Real ce = cos(kx*x), se = sin(kx*x);
      Real cz = cos(kz*z), sz = sin(kz*z);
      Real rho = rho0 + (Agr*ce - Agi*se)*cz;
      Real ux  = ugx  + (Uxr*ce - Uxi*se)*cz;
      Real up  = ugp  + (Upr*ce - Upi*se)*cz;
      Real uz  =      - (Uzr*se + Uzi*ce)*sz;
      ug(m,IDN,k,j,i) = rho;
      ug(m,IM1,k,j,i) = rho*ux;
      ug(m,IM2,k,j,i) = rho*up;   // azimuthal (x2) component
      ug(m,IM3,k,j,i) = rho*uz;   // vertical (x3) component
    });
    particles::Particles *ppar = pmbp->ppart;
    auto &pr = ppar->prtcl_rdata;
    auto &pi = ppar->prtcl_idata;
    const Real ppc = pin->GetOrAddReal("particles", "ppc", 1.0);
    const int npar1d = static_cast<int>(std::lround(std::sqrt(ppc)));
    if (npar1d*npar1d != static_cast<int>(std::lround(ppc)) ||
        pmbp->pdust->smr_transfer_mode != DustSMRTransfer::refinable_cloud) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "3-D streaming test: refinable clouds and <particles>/ppc a perfect square "
                << "(the x1-x3 sub-lattice of each root cell)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const int ppc_int = npar1d*npar1d;
    const Real lshift = pin->GetOrAddReal("problem", "lattice_shift", 0.0);
    const Real lshiftz = pin->GetOrAddReal("problem", "lattice_shift_z", 0.0);
    const bool seed_after = pin->GetOrAddBoolean("problem", "seed_after_split", false);
    const Real rhop0 = eps*rho0;
    const auto &lt3 = pmbp->pdust->leaf_traversal3d;
    const Real hx = lt3.root_dx1, hy = lt3.root_dx2, hz = lt3.root_dx3;
    cloud_mode_ = true;
    cloud_root_dx1_ = hx;
    cloud_root_dx2_ = hz;
    auto imprint = [&](auto &h, int p, Real x0, Real y0, Real z0) {
      Real ce = std::cos(kx*x0), se = std::sin(kx*x0);
      Real cz = std::cos(kz*z0), sz = std::sin(kz*z0);
      h(IPX,p) = x0 - (Apr/(kx*rhop0))*se*cz;
      h(IPY,p) = y0;
      h(IPZ,p) = z0;
      h(IPX1,p) = h(IPX,p); h(IPY1,p) = y0; h(IPZ1,p) = z0;
      h(IPVX,p) = vnx + (Vxr*ce - Vxi*se)*cz;
      h(IPVY,p) = vnp + (Vpr*ce - Vpi*se)*cz;   // azimuthal
      h(IPVZ,p) =     - (Vzr*se + Vzi*ce)*sz;   // vertical
      h(IPVX1,p) = h(IPVX,p); h(IPVY1,p) = h(IPVY,p); h(IPVZ1,p) = h(IPVZ,p);
    };
    auto &mblev = pmbp->pmb->mb_lev;
    std::vector<int> first(nmb3 + 1, 0), nr1(nmb3), nr2(nmb3), nr3(nmb3);
    for (int m=0; m<nmb3; ++m) {
      const int ratio = 1 << (mblev.h_view(m) - pmy_mesh_->root_level);
      if ((ind3.nx1 % ratio) != 0 || (ind3.nx2 % ratio) != 0 || (ind3.nx3 % ratio) != 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "refinable_cloud lattice requires MeshBlocks that contain whole root cells"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      nr1[m] = ind3.nx1/ratio; nr2[m] = ind3.nx2/ratio; nr3[m] = ind3.nx3/ratio;
      first[m+1] = first[m] + ppc_int*nr1[m]*nr2[m]*nr3[m];
    }
    const int nlattice = first[nmb3];
    Kokkos::realloc(pr, ppar->nrdata, nlattice);
    Kokkos::realloc(pi, ppar->nidata, nlattice);
    auto hr = Kokkos::create_mirror_view(pr);
    auto hi = Kokkos::create_mirror_view(pi);
    const Real taus0 = taus;
    for (int m=0; m<nmb3; ++m) {
      const auto &size = msz3.h_view(m);
      for (int q=0; q<first[m+1]-first[m]; ++q) {
        const int p = first[m] + q;
        const int c = q/ppc_int, sub = q - c*ppc_int;
        const int ip = sub % npar1d, kp = sub/npar1d;
        const int i = c % nr1[m], j = (c/nr1[m]) % nr2[m], kk = c/(nr1[m]*nr2[m]);
        const Real x0 = size.x1min + (i + lshift + (ip + 0.5)/static_cast<Real>(npar1d))*hx;
        const Real y0 = size.x2min + (j + 0.5)*hy;
        const Real z0 = size.x3min + (kk + lshiftz + (kp + 0.5)/static_cast<Real>(npar1d))*hz;
        hi(PGID,p) = pmbp->gids + m;
        SetTag(hi, p, static_cast<std::uint64_t>(p)*global_variable::nranks +
                      global_variable::my_rank);   // unique over ranks
        hi(PSP,p) = 0;
        hi(PLEV,p) = 0;
        for (int col=0; col<ppar->nrdata; ++col) hr(col,p) = 0.0;
        hr(IPTS,p) = taus0;
        hr(IPM,p) = eps*rho0*hx*hy*hz/ppc_int;
        if (seed_after) {
          hr(IPX,p) = x0; hr(IPY,p) = y0; hr(IPZ,p) = z0;
          hr(IPX1,p) = x0; hr(IPY1,p) = y0; hr(IPZ1,p) = z0;
          hr(IPVX,p) = vnx; hr(IPVY,p) = vnp;
          hr(IPVX1,p) = vnx; hr(IPVY1,p) = vnp;
        } else {
          imprint(hr, p, x0, y0, z0);
        }
      }
    }
    Kokkos::deep_copy(pr, hr);
    Kokkos::deep_copy(pi, hi);
    ppar->nprtcl_thispack = nlattice;
    pmy_mesh_->nprtcl_thisrank = nlattice;
    pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = nlattice;
    pmy_mesh_->nprtcl_total = nlattice;
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) {
      MPI_Allgather(&nlattice, 1, MPI_INT, pmy_mesh_->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
      int tot = 0;
      for (int r=0; r<global_variable::nranks; ++r) tot += pmy_mesh_->nprtcl_eachrank[r];
      pmy_mesh_->nprtcl_total = tot;
    }
#endif
    pmbp->pdust->EnsureNextTag();
    pmbp->pdust->SplitToLeaves();
    if (seed_after) {
      auto h2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_rdata);
      auto i2 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_idata);
      const auto &msz = pmy_mesh_->mesh_size;
      const Real len1 = msz.x1max - msz.x1min, len3 = msz.x3max - msz.x3min;
      for (int p=0; p<ppar->nprtcl_thispack; ++p) {
        imprint(h2, p, h2(IPX,p), h2(IPY,p), h2(IPZ,p));
        Real x = h2(IPX,p), y = h2(IPY,p), z = h2(IPZ,p);
        if (x < msz.x1min) x += len1;
        if (x >= msz.x1max) x -= len1;
        if (z < msz.x3min) z += len3;
        if (z >= msz.x3max) z -= len3;
        h2(IPX,p) = x; h2(IPX1,p) = x; h2(IPZ,p) = z; h2(IPZ1,p) = z;
        // several ranks: the block (and rank) from the position below (RouteToOwnerBlocks)
        if (global_variable::nranks > 1) continue;
        int owner = -1;
        for (int m=0; m<nmb3 && owner<0; ++m) {
          const auto &size = msz3.h_view(m);
          if (x >= size.x1min && x < size.x1max && y >= size.x2min && y < size.x2max &&
              z >= size.x3min && z < size.x3max) owner = m;
        }
        if (owner < 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "seed_after_split: particle outside every MeshBlock"
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
        i2(PGID,p) = pmbp->gids + owner;
      }
      Kokkos::deep_copy(ppar->prtcl_rdata, h2);
      Kokkos::deep_copy(ppar->prtcl_idata, i2);
      if (global_variable::nranks > 1) pmbp->pdust->ResetPGIDFromPositions("seed_after_split");
    }
    long long nroot = nlattice, nsplit = ppar->nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &nroot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nsplit, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "# refinable_cloud lattice (3-D): " << nroot << " root clouds -> "
                << nsplit << " particles after the initial split"
                << (seed_after ? ", seed imprinted after the split" : "") << std::endl;
    }
    const Real vmax3 = std::max({fabs(vnx), fabs(vnp), 1.0e-30});
    ppar->dtnew = std::min({msz3.h_view(0).dx1, msz3.h_view(0).dx2, msz3.h_view(0).dx3})/vmax3;
    return;
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
  const int nx3g = indcs.nx3;
  par_for("stream_gas", DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    Real z = lina3d ? CellCenterX(k-ks, nx3g, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max)
                    : CellCenterX(j-js, nx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
    Real ce = cos(kx*x), se = sin(kx*x);
    Real cz = cos(kz*z), sz = sin(kz*z);
    Real rho = rho0 + (Agr*ce - Agi*se)*cz;
    Real ux  = ugx  + (Uxr*ce - Uxi*se)*cz;
    Real up  = ugp  + (Upr*ce - Upi*se)*cz;
    Real uz  =      - (Uzr*se + Uzi*ce)*sz;
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*ux;
    u0(m,IM2,k,j,i) = lina3d ? rho*up : rho*uz;   // 2-D: vertical (x2); 3-D: azimuthal
    u0(m,IM3,k,j,i) = lina3d ? rho*uz : rho*up;
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
        // unique over the ranks (one rank: p)
        SetTag(hi, p, static_cast<std::uint64_t>(p)*global_variable::nranks + global_variable::my_rank);
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
    // <problem>/graded_split_layers (the graded normal transition of
    // a static refined region, a diagnostic) is gone
    dust::RemovedKey(pin, "graded_split_layers",
                     !pin->DoesParameterExist("problem", "graded_split_layers") ||
                     pin->GetInteger("problem", "graded_split_layers") < 0,
                     "the graded static split transition", "problem");
    {
      // one pass per refinement level: children of the first pass may lie inside a
      // still finer region (the tag counter is set collectively first)
      pmbp->pdust->EnsureNextTag();
      pmbp->pdust->SplitToLeaves();
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
        // a sub-lattice with lattice_shift_z < 0 can start below the box (ppc = 25 with
        // lattice_shift_z = -0.125 puts its first sub-row at z < x2min)
        if (z < msz.x2min) z += len2;
        if (z >= msz.x2max) z -= len2;
        h2(IPY,p) = z;
        h2(IPY1,p) = z;
        int owner = -1;
        for (int m=0; m<nmb && owner<0; ++m) {
          const auto &size = mbsize.h_view(m);
          if (x >= size.x1min && x < size.x1max && z >= size.x2min && z < size.x2max) {
            owner = m;
          }
        }
        if (owner < 0) {
          // several ranks: a piece in another rank's block keeps its block;
          // the cloud migration of the first cycle moves it
          if (global_variable::nranks > 1) continue;
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
                << (seed_after ? ", seed imprinted after the split" : "") << std::endl;
    }
    // <problem>/merge_replay = snapshot (diagnostic): replace the
    // particles by a LinA snapshot and replay split and merge with frozen velocities
    if (!pin->GetOrAddString("problem", "merge_replay", "").empty()) {
      RunMergeReplay(pin, pmy_mesh_, taus0);
    }
    // <problem>/merge_audit_replay = audit file (diagnostic): one merge pass
    // from the entry state of a merge-pass audit
    if (!pin->GetOrAddString("problem", "merge_audit_replay", "").empty()) {
      RunAuditReplay(pin, pmy_mesh_);
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
    int k = c/(lnx1*lnx2);   // (3-D) the vertical index; the sub-lattice is in x1-x3
    Real dx1 = mbsize.d_view(m).dx1;
    Real dx2 = mbsize.d_view(m).dx2;
    // subcell quiet-start position: cell left edge + (ip+1/2) dx/Npar
    Real x0 = mbsize.d_view(m).x1min + (static_cast<Real>(i) + lshift
              + (static_cast<Real>(ip) + 0.5)/static_cast<Real>(npar1d))*dx1;
    Real z0 = mbsize.d_view(m).x2min + (static_cast<Real>(j) + lshiftz
              + (static_cast<Real>(jp) + 0.5)/static_cast<Real>(npar1d))*dx2;
    Real y0 = 0.0;
    if (lina3d) {
      y0 = mbsize.d_view(m).x2min + (static_cast<Real>(j) + 0.5)*dx2;
      z0 = mbsize.d_view(m).x3min + (static_cast<Real>(k) + lshiftz
           + (static_cast<Real>(jp) + 0.5)/static_cast<Real>(npar1d))*mbsize.d_view(m).dx3;
    }
    Real ce = cos(kx*x0), se = sin(kx*x0);
    Real cz = cos(kz*z0), sz = sin(kz*z0);
    // radial displacement imprinting drho_p = Apr*cos(kx x)*cos(kz z)
    pr(IPX,p) = x0 - (Apr/(kx*rhop0))*se*cz;
    pr(IPY,p) = lina3d ? y0 : z0;
    pr(IPZ,p) = lina3d ? z0 : 0.0;
    pi(PSP,p) = 0;
    pi(PLEV,p) = 0;
    pr(IPTS,p) = taus0;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (lina3d) vol *= mbsize.d_view(m).dx3;
    pr(IPM,p) = eps*rho0*vol/ppc;
    pr(IPVX,p) = vnx + (Vxr*ce - Vxi*se)*cz;
    const Real vvert = -(Vzr*se + Vzi*ce)*sz, vazim = vnp + (Vpr*ce - Vpi*se)*cz;
    pr(IPVY,p) = lina3d ? vazim : vvert;   // 2-D: vertical (x2); 3-D: azimuthal (x2)
    pr(IPVZ,p) = lina3d ? vvert : vazim;
    pr(IPRX,p) = 0.0;
    pr(IPRY,p) = 0.0;
    pr(IPRZ,p) = 0.0;
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
  const bool h3d = pm->three_d;   // the vertical direction is x3 in 3-D
  Kokkos::parallel_reduce("stream_hp",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int &p, Real &re_, Real &im_) {
    Real w = pr(IPM,p)*cos(kz*(h3d ? pr(IPZ,p) : pr(IPY,p)));
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
    Real z = h3d ? CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max)
                 : CellCenterX(j-js, nx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (h3d) vol *= mbsize.d_view(m).dx3;
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
  if (snapshots_) {
    const bool due = (snapshot_calls_++ % snapshot_every_) == 0;
    if (lina_pin_ != nullptr) lina_pin_->SetInteger("problem", "rst_snapshot_calls", snapshot_calls_);
    if (due) WriteStreamingSnapshot(pm);
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void StreamingLinearHistory3D()
//! \brief StreamingLinearHistory in 3-D (the mode in x1-x3; x3 = vertical): the same
//! projections, summed over x2; the cloud's Fourier factor along x1 and x3 (lobes 1 and 3), its
//! x2 shape integrates out (k_y = 0).

void StreamingLinearHistory3D(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  pdata->nhist = 6;
  pdata->label[0] = "rhop_re";
  pdata->label[1] = "rhop_im";
  pdata->label[2] = "rhog_re";
  pdata->label[3] = "rhog_im";
  pdata->label[4] = "mgx_re";
  pdata->label[5] = "mgx_im";
  const Real kx = kx_, kz = kz_, rho0 = rho0_;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const Real hx = cloud_root_dx1_, hz = cloud_root_dx2_;
  Real pre = 0.0, pim = 0.0;
  Kokkos::parallel_reduce("stream_hp3d",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int &p, Real &re_, Real &im_) {
    using T = SMRCompositeTransfer;
    const int code = pi(PLEV,p);
    const Real scale = 1.0/static_cast<Real>(1 << T::CloudLevel(code));
    const Real ax = 0.5*kx*hx*scale, az = 0.5*kz*hz*scale;
    const Real sx = (ax == 0.0) ? 1.0 : sin(ax)/ax;
    const Real sz = (az == 0.0) ? 1.0 : sin(az)/az;
    const int l1 = T::CloudLobe1(code), l3 = T::CloudLobe3(code);
    const Real x = pr(IPX,p), z = pr(IPZ,p);
    Real w = pr(IPM,p)*cos(kz*z);
    if (l3 == 3) {
      const Real d = hz*scale;
      w = pr(IPM,p)*(0.5*cos(kz*z) + 0.25*cos(kz*(z - d)) + 0.25*cos(kz*(z + d)));
    } else if (l3) {
      const Real d = ((l3 == 1) ? -hz : hz)*scale;
      w = pr(IPM,p)*((2.0/3.0)*cos(kz*z) + (1.0/3.0)*cos(kz*(z + d)));
    }
    Real cxr = cos(kx*x), cxi = sin(kx*x);
    if (l1 == 3) {
      const Real d = hx*scale;
      cxr = 0.5*cos(kx*x) + 0.25*cos(kx*(x - d)) + 0.25*cos(kx*(x + d));
      cxi = 0.5*sin(kx*x) + 0.25*sin(kx*(x - d)) + 0.25*sin(kx*(x + d));
    } else if (l1) {
      const Real d = ((l1 == 1) ? -hx : hx)*scale;
      cxr = (2.0/3.0)*cos(kx*x) + (1.0/3.0)*cos(kx*(x + d));
      cxi = (2.0/3.0)*sin(kx*x) + (1.0/3.0)*sin(kx*(x + d));
    }
    w *= SQR(sx)*SQR(sz);
    re_ += w*cxr;
    im_ -= w*cxi;
  }, Kokkos::Sum<Real>(pre), Kokkos::Sum<Real>(pim));
  pdata->hdata[0] = pre;
  pdata->hdata[1] = pim;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1;
  const int nmkji = pmbp->nmb_thispack*nkji;
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;
  Real gre = 0.0, gim = 0.0, mre = 0.0, mim = 0.0;
  Kokkos::parallel_reduce("stream_hg3d",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &gr_, Real &gi_, Real &mr_, Real &mi_) {
    const int m = idx/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    const int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    const Real x = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    const Real z = CellCenterX(k-ks, nx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
    const Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
    const Real wc = vol*cos(kz*z);
    const Real drho = u0(m,IDN,k,j,i) - rho0;
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
}

//----------------------------------------------------------------------------------------
//! \fn void StreamingKx0Filter()
//! \brief Diagnostic user source: remove the z-dependent k_x = 0 component of
//! the gas radial momentum, i.e. per row of the finest virtual grid the x-average of m_x
//! minus its domain mean (the NSH background, uniform, and every k_x != 0 mode are kept);
//! with <problem>/kx0_filter_phi the same for the azimuthal momentum; from
//! <problem>/kx0_filter_tstart on.
//! A coarse cell covering several finest rows takes the mean of their corrections.
void StreamingKx0Filter(Mesh *pm, const Real bdt) {
  (void) bdt;
  if (pm->time < kx0_filter_tstart) return;
  MeshBlockPack *pmbp = pm->pmb_pack;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int nmb = pmbp->nmb_thispack;
  const auto &t = pmbp->pdust->smr_transfer;
  const int nf2 = t.nfine2;
  const Real fd2 = t.fine_dx2, x2min = t.x2min;
  const Real len1 = pm->mesh_size.x1max - pm->mesh_size.x1min;
  for (int n : {static_cast<int>(IM1), static_cast<int>(IM3)}) {
    if (n == IM3 && !kx0_filter_phi) continue;
    DvceArray1D<Real> row("kx0_row", nf2);
    Kokkos::parallel_for("kx0_sum", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExeSpace(),
        {0, js, is}, {nmb, je + 1, ie + 1}), KOKKOS_LAMBDA(const int m, const int j, const int i) {
      const Real dx = size.d_view(m).dx1, dz = size.d_view(m).dx2;
      const Real z0 = size.d_view(m).x2min + (j - js)*dz;
      const int r = static_cast<int>(Kokkos::round(dz/fd2));
      const int j0 = static_cast<int>(Kokkos::round((z0 - x2min)/fd2));
      for (int q=0; q<r; ++q) Kokkos::atomic_add(&row(j0 + q), u0(m,n,0,j,i)*dx/len1);
    });
    Real mean = 0.0;
    Kokkos::parallel_reduce("kx0_mean", Kokkos::RangePolicy<>(DevExeSpace(), 0, nf2),
    KOKKOS_LAMBDA(const int k, Real &sum) {sum += row(k);}, mean);
    mean /= static_cast<Real>(nf2);
    Kokkos::parallel_for("kx0_apply", Kokkos::MDRangePolicy<Kokkos::Rank<3>>(DevExeSpace(),
        {0, js, is}, {nmb, je + 1, ie + 1}), KOKKOS_LAMBDA(const int m, const int j, const int i) {
      const Real dz = size.d_view(m).dx2;
      const Real z0 = size.d_view(m).x2min + (j - js)*dz;
      const int r = static_cast<int>(Kokkos::round(dz/fd2));
      const int j0 = static_cast<int>(Kokkos::round((z0 - x2min)/fd2));
      Real c = 0.0;
      for (int q=0; q<r; ++q) c += row(j0 + q) - mean;
      u0(m,n,0,j,i) -= c/static_cast<Real>(r);
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn void RunMergeReplay()
//! \brief Diagnostic: local replay of the refinable-cloud split and
//! merge without hydro.  The particles of a LinA snapshot (positions, velocities, masses,
//! tags, codes) replace the lattice; then, per cycle as RefineClouds does it, the split
//! check, a merge pass every cloud_merge_interval cycles (shifted by
//! merge_replay_pass_offset), and a drift with the frozen velocities.  Variants: a random x
//! offset (merge_replay_jitter, in root cells, for fine pieces with x in
//! [merge_replay_jx0, merge_replay_jx1) root cells), a permuted storage order
//! (merge_replay_shuffle = seed > 0).  Snapshots <basename>.replay.NNNNN.lina every
//! merge_replay_snap_every cycles; the merge log goes to cloud_merge_log.txt.  Exits.
void RunMergeReplay(ParameterInput *pin, Mesh *pm, const Real taus0) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  auto *pdust = pmbp->pdust;
  const std::string fname = pin->GetString("problem", "merge_replay");
  const int ncyc = pin->GetOrAddInteger("problem", "merge_replay_ncycles", 21300);
  const Real dt = pin->GetOrAddReal("problem", "merge_replay_dt", 9.18667e-06);
  const int snap_every = pin->GetOrAddInteger("problem", "merge_replay_snap_every", 1775);
  const Real jit = pin->GetOrAddReal("problem", "merge_replay_jitter", 0.0);
  const Real jx0 = pin->GetOrAddReal("problem", "merge_replay_jx0", -1.0e30);
  const Real jx1 = pin->GetOrAddReal("problem", "merge_replay_jx1", 1.0e30);
  const int seed = pin->GetOrAddInteger("problem", "merge_replay_seed", 1);
  const int shuffle = pin->GetOrAddInteger("problem", "merge_replay_shuffle", 0);
  const int offset = pin->GetOrAddInteger("problem", "merge_replay_pass_offset", 0);
  const auto &msz = pm->mesh_size;
  const Real len1 = msz.x1max - msz.x1min, len2 = msz.x2max - msz.x2min;
  const Real hroot = len1/static_cast<Real>(pin->GetInteger("mesh", "nx1"));
  // read the snapshot
  FILE *f = std::fopen(fname.c_str(), "rb");
  char line[256];
  double t0 = 0.0; int cyc0 = 0, nc = 0, np = 0;
  if (f == nullptr || std::fgets(line, sizeof(line), f) == nullptr ||
      std::string(line) != "AthenaK LinA snapshot v2 f64-native\n" ||
      std::fgets(line, sizeof(line), f) == nullptr ||
      std::sscanf(line, "%lf %d %d %d", &t0, &cyc0, &nc, &np) != 4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "merge_replay: cannot read snapshot " << fname << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::vector<double> raw(8*static_cast<size_t>(np));
  if (std::fseek(f, 8L*8L*static_cast<long>(nc), SEEK_CUR) != 0 ||
      std::fread(raw.data(), sizeof(double), raw.size(), f) != raw.size()) {
    std::cout << "### FATAL ERROR: merge_replay: truncated snapshot " << fname << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::fclose(f);
  std::vector<int> order(np);
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(seed);
  if (shuffle > 0) {
    std::mt19937_64 rs(shuffle);
    std::shuffle(order.begin(), order.end(), rs);
  }
  std::uniform_real_distribution<double> uni(-1.0, 1.0);
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  Kokkos::realloc(pr, ppar->nrdata, np);
  Kokkos::realloc(pi, ppar->nidata, np);
  auto hr = Kokkos::create_mirror_view(pr);
  auto hi = Kokkos::create_mirror_view(pi);
  auto &mbsize = pmbp->pmb->mb_size;
  const int nmb = pmbp->nmb_thispack, gids = pmbp->gids;
  std::uint64_t tagbound = 0;   // 1 + the largest tag
  int njit = 0;
  for (int s=0; s<np; ++s) {
    const double *q = &raw[8*static_cast<size_t>(order[s])];
    for (int col=0; col<ppar->nrdata; ++col) hr(col,s) = 0.0;
    Real x = q[0], y = q[1];
    const int code = static_cast<int>(q[7]);
    if (jit != 0.0 && (code & 15) == 1) {
      const Real xr = (x - msz.x1min)/hroot;
      if (xr >= jx0 && xr < jx1) {x += jit*hroot*uni(rng); ++njit;}
    }
    while (x < msz.x1min) x += len1;
    while (x >= msz.x1max) x -= len1;
    hr(IPX,s) = x; hr(IPX1,s) = x; hr(IPY,s) = y; hr(IPY1,s) = y;
    hr(IPM,s) = q[2]; hr(IPTS,s) = taus0;
    hr(IPVX,s) = q[3]; hr(IPVY,s) = q[4]; hr(IPVZ,s) = q[5];
    hr(IPVX1,s) = q[3]; hr(IPVY1,s) = q[4]; hr(IPVZ1,s) = q[5];
    int owner = 0;
    for (int mb=0; mb<nmb; ++mb) {
      const auto &size = mbsize.h_view(mb);
      if (x >= size.x1min && x < size.x1max && y >= size.x2min && y < size.x2max) {
        owner = mb; break;
      }
    }
    hi(PGID,s) = gids + owner; hi(PSP,s) = 0;
    SetTag(hi, s, static_cast<std::uint64_t>(q[6]));
    hi(PLEV,s) = code;
    tagbound = std::max(tagbound, GetTag(hi,s) + 1);
  }
  Kokkos::deep_copy(pr, hr);
  Kokkos::deep_copy(pi, hi);
  ppar->nprtcl_thispack = np;
  pm->nprtcl_thisrank = np;
  pm->nprtcl_eachrank[global_variable::my_rank] = np;
  pm->nprtcl_total = np;
  pdust->cloud_next_tag = tagbound;
  pdust->cloud_next_tag_set = true;
  pm->time = t0;
  pm->ncycle = cyc0;
  std::cout << "# merge_replay: " << fname << " t=" << t0 << " cycle=" << cyc0 << " particles="
            << np << " jittered=" << njit << " (amplitude " << jit << " root cells)"
            << " shuffle=" << shuffle << " pass offset=" << offset << std::endl;
  snapshot_basename_ = pin->GetString("job", "basename") + ".replay";
  snapshot_number_ = 0;
  WriteStreamingSnapshot(pm);
  const Real x1min = msz.x1min, x2min = msz.x2min;
  for (int n=0; n<ncyc; ++n) {
    pdust->SplitAndMergeClouds();
    if (pdust->cloud_merge && (pm->ncycle + offset) % pdust->cloud_merge_interval == 0) {
      pdust->MergeClouds();
    }
    auto &dr = ppar->prtcl_rdata;
    auto &di = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    par_for("replay_drift", DevExeSpace(), 0, npart - 1, KOKKOS_LAMBDA(const int p) {
      Real x = dr(IPX,p) + dr(IPVX,p)*dt, y = dr(IPY,p) + dr(IPVY,p)*dt;
      if (x < x1min) x += len1;
      if (x >= x1min + len1) x -= len1;
      if (y < x2min) y += len2;
      if (y >= x2min + len2) y -= len2;
      dr(IPX,p) = x; dr(IPX1,p) = x; dr(IPY,p) = y; dr(IPY1,p) = y;
      for (int mb=0; mb<nmb; ++mb) {
        if (x >= mbsize.d_view(mb).x1min && x < mbsize.d_view(mb).x1max &&
            y >= mbsize.d_view(mb).x2min && y < mbsize.d_view(mb).x2max) {
          di(PGID,p) = gids + mb; break;
        }
      }
    });
    pm->time += dt;
    pm->ncycle += 1;
    if ((n + 1) % snap_every == 0) WriteStreamingSnapshot(pm);
  }
  std::cout << "# merge_replay done: t=" << pm->time << " cycle=" << pm->ncycle
            << " particles=" << ppar->nprtcl_thispack << std::endl;
  std::exit(EXIT_SUCCESS);
}

//----------------------------------------------------------------------------------------
//! \fn void RunAuditReplay()
//! \brief Diagnostic: replay one merge pass from the entry state of a
//! merge-pass audit file (sections meta, a0_pr, a0_pi, gas).  Restores every particle
//! column, the gas of the active cells, time, cycle, and the hidden merge state (pass
//! counter, next tag, root mass); runs MergeClouds() once without split, drift or hydro;
//! exits.  The pass writes its own audit when cloud_audit_t0/t1 cover the time.
void RunAuditReplay(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  auto *pdust = pmbp->pdust;
  const std::string fname = pin->GetString("problem", "merge_audit_replay");
  std::FILE *f = std::fopen(fname.c_str(), "rb");
  char line[4096];
  if (f == nullptr || std::fgets(line, sizeof(line), f) == nullptr ||
      std::string(line).rfind("AthenaK merge audit v1", 0) != 0) {
    std::cout << "### FATAL ERROR: merge_audit_replay: cannot read " << fname << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string meta;
  std::vector<double> rpr, gas, gasfull;
  std::vector<int> ipi;
  long long n0r = 0, n1r = 0, n0i = 0, n1i = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    char name[64], type[4];
    int nd = 0, used = 0;
    if (std::sscanf(line, "S %63s %3s %d%n", name, type, &nd, &used) != 3) break;
    long long dims[4] = {0, 0, 0, 0}, nbytes = 0;
    const char *q = line + used;
    for (int k=0; k<nd; ++k) {int u = 0; std::sscanf(q, "%lld%n", &dims[k], &u); q += u;}
    std::sscanf(q, "%lld", &nbytes);
    std::vector<char> buf(static_cast<std::size_t>(nbytes));
    if (nbytes > 0 && std::fread(buf.data(), 1, buf.size(), f) != buf.size()) break;
    const std::string nm(name);
    if (nm == "meta") {
      meta.assign(buf.begin(), buf.end());
    } else if (nm == "a0_pr") {
      rpr.resize(buf.size()/sizeof(double)); std::memcpy(rpr.data(), buf.data(), buf.size());
      n0r = dims[0]; n1r = dims[1];
    } else if (nm == "a0_pi") {
      ipi.resize(buf.size()/sizeof(int)); std::memcpy(ipi.data(), buf.data(), buf.size());
      n0i = dims[0]; n1i = dims[1];
    } else if (nm == "gas") {
      gas.resize(buf.size()/sizeof(double)); std::memcpy(gas.data(), buf.data(), buf.size());
    } else if (nm == "gas_full") {
      gasfull.resize(buf.size()/sizeof(double)); std::memcpy(gasfull.data(), buf.data(), buf.size());
    }
  }
  std::fclose(f);
  auto get = [&](const std::string &key) {
    const auto at = meta.find(key + "=");
    if (at == std::string::npos) {
      std::cout << "### FATAL ERROR: merge_audit_replay: no " << key << " in meta" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return std::strtod(meta.c_str() + at + key.size() + 1, nullptr);
  };
  const int np = static_cast<int>(n1r);
  // (an audit file written before the 64-bit tags has no PTAGH row: high words 0)
  const bool no_hi = (n0i == PTAGH && ppar->nidata == PTAGH + 1);
  if (n0r != ppar->nrdata || (n0i != ppar->nidata && !no_hi) || n1i != n1r || np <= 0 ||
      gas.empty()) {
    std::cout << "### FATAL ERROR: merge_audit_replay: incompatible sections in " << fname << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  Kokkos::realloc(pr, ppar->nrdata, np);
  Kokkos::realloc(pi, ppar->nidata, np);
  auto hr = Kokkos::create_mirror_view(pr);
  auto hi = Kokkos::create_mirror_view(pi);
  for (int c=0; c<ppar->nrdata; ++c)
    for (int p=0; p<np; ++p) hr(c,p) = rpr[static_cast<std::size_t>(c)*np + p];
  for (int c=0; c<ppar->nidata; ++c)
    for (int p=0; p<np; ++p) hi(c,p) = (c < n0i) ? ipi[static_cast<std::size_t>(c)*np + p] : 0;
  Kokkos::deep_copy(pr, hr);
  Kokkos::deep_copy(pi, hi);
  ppar->nprtcl_thispack = np;
  pm->nprtcl_thisrank = np;
  pm->nprtcl_eachrank[global_variable::my_rank] = np;
  pm->nprtcl_total = np;
  // gas of the active cells, in the audit writer's order (blocks, rows, columns)
  auto &ind = pm->mb_indcs;
  auto u0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmbp->phydro->u0);
  // the whole array with ghost cells when the audit kept it (the linear gather reads ghosts)
  const bool ghosts = gasfull.size() == u0.span();
  if (ghosts) std::memcpy(u0.data(), gasfull.data(), gasfull.size()*sizeof(double));
  std::size_t k = 0;
  double maxdx = 0.0;
  for (int m=0; m<pmbp->nmb_thispack; ++m) {
    const auto &size = pmbp->pmb->mb_size.h_view(m);
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i, ++k) {
        const double *c = &gas[8*k];
        maxdx = std::max(maxdx, std::abs(c[0] - (size.x1min + (i-ind.is+0.5)*size.dx1)) +
                                std::abs(c[1] - (size.x2min + (j-ind.js+0.5)*size.dx2)));
        u0(m,IDN,ind.ks,j,i) = c[4]; u0(m,IM1,ind.ks,j,i) = c[5];
        u0(m,IM2,ind.ks,j,i) = c[6]; u0(m,IM3,ind.ks,j,i) = c[7];
      }
    }
  }
  if (8*k != gas.size() || maxdx > 1.0e-12) {
    std::cout << "### FATAL ERROR: merge_audit_replay: gas cells do not match the mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::deep_copy(pmbp->phydro->u0, u0);
  pm->time = get("time");
  pm->dt = get("dt");   // the frozen feedback diagnostic scales with dt
  pm->ncycle = static_cast<int>(get("ncycle"));
  pdust->cloud_merge_pass = static_cast<int>(get("merge_pass"));
  {
    const double nt = get("next_tag");   // -1: not set
    pdust->cloud_next_tag_set = (nt >= 0.0);
    pdust->cloud_next_tag = (nt >= 0.0) ? static_cast<std::uint64_t>(nt) : 0;
  }
  const Real mroot = get("mroot0");
  pdust->cloud_merge_mroot.clear();
  if (mroot > 0.0) pdust->cloud_merge_mroot.push_back(mroot);
  std::cout << "# merge_audit_replay: " << fname << " t=" << pm->time << " cycle=" << pm->ncycle
            << " particles=" << np << " pass=" << pdust->cloud_merge_pass << " next_tag="
            << pdust->cloud_next_tag << " mroot=" << mroot << std::endl;
  // <problem>/merge_audit_restart (diagnostic): no replay pass and no exit, the run
  // continues from the restored state; the driver fills the ghost cells before the first cycle
  if (pin->GetOrAddBoolean("problem", "merge_audit_restart", false)) {
    std::cout << "# merge_audit_restart: continuing from the restored state" << std::endl;
    return;
  }
  // <problem>/merge_audit_replay_passes consecutive passes on the frozen state (no drift, no split)
  const int npass = pin->GetOrAddInteger("problem", "merge_audit_replay_passes", 1);
  const Real dtm = get("dt");
  std::cout << "# merge_audit_replay: ghosts restored=" << (ghosts ? 1 : 0) << " passes=" << npass << std::endl;
  for (int n=0; n<npass; ++n) {
    const int nev = pdust->MergeClouds();
    std::cout << "# merge_audit_replay pass " << n << " cycle=" << pm->ncycle << " events=" << nev
              << " particles=" << ppar->nprtcl_thispack << std::endl;
    pm->ncycle += pdust->cloud_merge_interval;
    pm->time += pdust->cloud_merge_interval*dtm;
  }
  std::cout << "# merge_audit_replay done: particles=" << ppar->nprtcl_thispack << std::endl;
  std::exit(EXIT_SUCCESS);
}

