//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_nsh.cpp
//! \brief Problem generator for the multi-species Nakagawa-Sei-Hayashi (NSH) drift
//! equilibrium test in an unstratified shearing box (Nakagawa et al. 1986;
//! Benitez-Llambay et al. 2019 Sec. 3.4; Krapp et al. 2024 Sec. 3.6). The gas feels a
//! constant radial acceleration a_x = 2*Omega*etavk (the pressure-gradient mimic,
//! applied through the standard <hydro_srcterms> const_accel machinery so that dust
//! particles do not feel it), and gas + N dust species settle into uniform steady drift
//! velocities that solve the linear system
//!     0 =  2*Omega*u_phi + a_x + sum_s eps_s*(v_x,s - u_x)/tau_s
//!     0 = -(2-q)*Omega*u_x     + sum_s eps_s*(v_phi,s - u_phi)/tau_s
//!     0 =  2*Omega*v_phi,s - (v_x,s - u_x)/tau_s
//!     0 = -(2-q)*Omega*v_x,s - (v_phi,s - u_phi)/tau_s
//! (velocities relative to the background shear). By default the problem uses a quiet
//! particle lattice, initializes this equilibrium exactly, and verifies it is HELD TO
//! ROUND-OFF: because the
//! rotation/shear/forcing kicks and the implicit drag solve are composed unsplit inside
//! the IMEX stages, the discrete update has the continuum equilibrium as an exact fixed
//! point (any Strang-like splitting error would appear as secular drift).
//! <problem>/particle_placement=random instead gives a reproducible warm start at the
//! same NSH velocities for nonlinear streaming-instability calculations.
//! A physical dust-free equilibrium is obtained by omitting both <particles> and <dust>.
//! In that mode there are no dust species or stopping times, so the solution reduces to
//! u_x=0 and u_phi=-etavk.
//!
//! Works in the 2D r-z shearing box (azimuthal components in IM3/IPVZ, plain periodic
//! boundaries) and in 3D (azimuthal in IM2/IPVY, shear-periodic x1 boundaries).
//! Errors written to "<basename>-errs.dat"; per-species drift history available with
//! <problem>/user_hist = true.
//!
//! Dust on refined meshes: with <problem>/cloud_root_random = true the particles start as
//! root parents (<problem>/root_placement = lattice | random | generic) that split exactly to
//! the leaves as refinable clouds (see pgen/dust_amr/dust_setup.hpp).  Perturbations of the
//! equilibrium for the tests of refined meshes: <problem>/two_stream_v (two cold streams
//! along x2) and blob_amp (a heavier dust disc), 2-D point particles; gas_vx_amp (a
//! divergence-free u_x(y)) in 3-D; dust_ymode_amp (dust masses modulated along y, with error
//! columns in the history) for 3-D root clouds.  <problem>/ba_* adds the 2-D diagnostic
//! outputs of pgen/dust_amr/dust_snapshots.cpp.
//!
//! The related problems are separate problem generators in pgen/dust_amr: the unstratified
//! streaming instability (dust_si_unstrat), the stratified 3-D one (dust_si_strat), and the
//! drag tests without rotation (dust_damping_amr, dusty_wave_amr).

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdio>     // fopen(), fprintf()
#include <iostream>
#include <string>
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "dust/dust_species.hpp"
#include "outputs/outputs.hpp"
#include "pgen/dust_amr/dust_setup.hpp"
#include "pgen/dust_amr/dust_snapshots.hpp"

// functions for error output and user history
void DustNSHErrors(ParameterInput *pin, Mesh *pm);
void DustNSHHistory(HistoryData *pdata, Mesh *pm);

namespace {

// the NSH equilibrium of 3-D refinable clouds: the error columns of the history
dust_setup::DustTest3D dtest3d_;

// the midplane ratios of the new species keys as the problem generator resolved them;
// empty before (a restarted run: epsilon_N directly)
std::vector<Real> nsh_eps_resolved;

//----------------------------------------------------------------------------------------
//! \fn ReadNSHParams
//! \brief Reads the shearing-box and active dust-mixture parameters.

void ReadNSHParams(ParameterInput *pin, MeshBlockPack *pmbp,
                   Real &omega0, Real &qshear, Real &etavk,
                   std::vector<Real> &eps, std::vector<Real> &taus) {
  omega0 = pin->GetReal("shearing_box","omega0");
  qshear = pin->GetReal("shearing_box","qshear");
  etavk  = pin->GetReal("problem","etavk");
  dust_setup::ReadSpecies(pin, pmbp, false, eps, taus);
  if (pmbp->pdust != nullptr && dust::Species(pin).new_init &&
      nsh_eps_resolved.size() == eps.size()) {
    eps = nsh_eps_resolved;
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustNSH()
//! \brief Problem Generator for the multi-species NSH drift equilibrium

void ProblemGenerator::DustNSH(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool dust_active = dust_setup::DustActive(pmy_mesh_);
  dust_setup::RejectKeys(pin, "dust_nsh", {
      {"dust_test", "none", "dust_damping_amr or dusty_wave_amr"},
      {"dust_test_vz0", "0", "dust_damping_amr"},
      {"stratified_dust", "false", "dust_si_strat"},
      {"stratified_dust_sampling", "weight", "dust_si_strat"},
      {"dust_zcut", "0", "dust_si_strat"},
      {"nsh_vertical", "false", "dust_si_strat"},
      {"vertical_bc", "none", "dust_si_strat"}});
  dust_snapshots::Setup(pin, pmy_mesh_, dust_active);
  dust_setup::SetupHistory(pin);
  const bool random_placement = dust_active &&
      pin->GetOrAddString("problem", "particle_placement", "lattice") == "random";
  bool check_equilibrium =
      pin->GetOrAddBoolean("problem","check_equilibrium",!random_placement);
  pgen_final_func = check_equilibrium ? DustNSHErrors : nullptr;
  // Keep this enrolled in dust-free runs as well so an otherwise unchanged input
  // with user_hist=true remains valid; the callback emits no dust columns in that mode.
  user_hist_func = DustNSHHistory;
  dust_setup::EnrollRefinement(pin, pmy_mesh_, this);
  if (restart) {
    dust_snapshots::Restart(pin, pmy_mesh_);
    dust_setup::LogRestartParticles(pmy_mesh_);
    return;
  }

  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH test requires a <hydro> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pin->DoesBlockExist("shearing_box"))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH test requires a <shearing_box> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real omega0 = 0.0, qshear = 0.0, etavk = 0.0;
  std::vector<Real> eps, taus;
  ReadNSHParams(pin, pmbp, omega0, qshear, etavk, eps, taus);
  int nspec = static_cast<int>(eps.size());
  Real ax = 2.0*omega0*etavk;
  if (dust_active && dust::Species(pin).new_init) nsh_eps_resolved = eps;   // DustNSHErrors
  dust_setup::CheckRadialForcing(pin, ax);

  // equilibrium drift velocities
  Real ugx = 0.0, ugp = 0.0;
  std::vector<Real> vx(nspec, 0.0), vp(nspec, 0.0);
  dust_setup::SolveNSH(omega0, qshear, ax, eps, taus, ugx, ugp, vx, vp);
  if (global_variable::my_rank == 0) {
    if (dust_active) {
      std::cout << "# NSH equilibrium: u_gx=" << ugx << " u_gphi=" << ugp << std::endl;
      for (int s=0; s<nspec; ++s) {
        std::cout << "#   species " << s+1 << ": v_x=" << vx[s]
                  << " v_phi=" << vp[s] << std::endl;
      }
    } else {
      std::cout << "# Dust-free NSH equilibrium: u_gx=" << ugx
                << " u_gphi=" << ugp << std::endl;
    }
  }

  // initialize uniform gas at the equilibrium
  Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  bool three_d = pmy_mesh_->three_d;
  dust_setup::InitUniformGas(pmy_mesh_, rho0, ugx, ugp);
  // <problem>/gas_vx_amp, gas_vx_ny: a non-axisymmetric gas velocity u_x += A sin(k_y y)
  // (gas_vx_ny wavelengths along x2; divergence free) for the check of the frame in which the drag
  // samples the gas inside a step of the 3-D shearing box (the cell averages of the sine)
  {
    const Real gamp = pin->GetOrAddReal("problem", "gas_vx_amp", 0.0);
    if (gamp != 0.0 && three_d) {
      auto &indcs = pmy_mesh_->mb_indcs;
      const int ng = indcs.ng, nmb = pmbp->nmb_thispack;
      const int n1 = indcs.nx1 + 2*ng, n2 = indcs.nx2 + 2*ng, n3 = indcs.nx3 + 2*ng;
      auto &u0 = pmbp->phydro->u0;
      const auto &ms = pmy_mesh_->mesh_size;
      const Real ky = 2.0*M_PI*pin->GetOrAddReal("problem", "gas_vx_ny", 1.0)/(ms.x2max - ms.x2min);
      auto &mbs = pmbp->pmb->mb_size;
      const int js0 = indcs.js, nx2 = indcs.nx2;
      par_for("nsh_gas_vx_ky", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real dy = mbs.d_view(m).dx2;
        const Real y = CellCenterX(j - js0, nx2, mbs.d_view(m).x2min, mbs.d_view(m).x2max);
        const Real avg = (cos(ky*(y - 0.5*dy)) - cos(ky*(y + 0.5*dy)))/(ky*dy);   // <sin> over the cell
        u0(m,IM1,k,j,i) += u0(m,IDN,k,j,i)*gamp*avg;
      });
    }
  }

  // There are no particle or dust modules to initialize in the dust-free equilibrium.
  if (!dust_active) return;

  dust_setup::DustInit d;
  d.rho0 = rho0;
  d.eps = eps;
  d.vx = vx;
  d.vp = vp;
  d.placement = "lattice";
  const bool cloud_root_random = pin->GetOrAddBoolean("problem", "cloud_root_random", false);
  // the NSH equilibrium of 3-D refinable clouds in the shearing box, with the error
  // columns of the drag tests (gas against its equilibrium, dust density against eps rho0)
  if (three_d && nspec == 1 && cloud_root_random && pmbp->pdust->Clouds3D()) {
    dtest3d_.on = true;
    dtest3d_.nsh = true;
    dtest3d_.rho0 = rho0; dtest3d_.eps = eps[0]; dtest3d_.ts = taus[0];
    dtest3d_.ux0 = ugx; dtest3d_.uy0 = ugp;
    // <problem>/dust_ymode_amp, dust_ymode_n: the root parent masses times
    // 1 + amp cos(ky y), advected with the drift and the shear
    const Real yamp = pin->GetOrAddReal("problem", "dust_ymode_amp", 0.0);
    if (yamp != 0.0) {
      const auto &ms = pmy_mesh_->mesh_size;
      dtest3d_.ymode = true;
      dtest3d_.yamp = yamp;
      dtest3d_.ky = 2.0*M_PI*pin->GetOrAddReal("problem", "dust_ymode_n", 1.0)/(ms.x2max - ms.x2min);
      dtest3d_.yvx = vx[0]; dtest3d_.yvy = vp[0]; dtest3d_.qom = qshear*omega0;
      dtest3d_.hface = 2.0*(ms.x1max - ms.x1min)/pmy_mesh_->mesh_indcs.nx1;
      dtest3d_.x1min = ms.x1min; dtest3d_.x1max = ms.x1max;
      d.yamp = dtest3d_.yamp;
      d.ky = dtest3d_.ky;
    }
  }
  dust_setup::PlaceParticles(pin, pmy_mesh_, d);
  particles::Particles *ppar = pmbp->ppart;

  // two cold streams along x2 for the multi-stream regrid test; the
  // particles below the midplane get +two_stream_v added to their x2 velocity, the others
  // two_stream_v_upper (default -two_stream_v) (2-D lattice or random placement)
  const Real two_stream_v = pin->GetOrAddReal("problem", "two_stream_v", 0.0);
  const Real two_stream_vu = pin->GetOrAddReal("problem", "two_stream_v_upper", -two_stream_v);
  if (two_stream_v != 0.0 || two_stream_vu != 0.0) {
    if (three_d || cloud_root_random) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/two_stream_v needs a 2-D run without cloud_root_random" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto &prs = ppar->prtcl_rdata;
    par_for("nsh_two_stream", DevExeSpace(), 0, ppar->nprtcl_thispack-1, KOKKOS_LAMBDA(const int p) {
      prs(IPVY,p) += (prs(IPY,p) < 0.0) ? two_stream_v : two_stream_vu;
    });
  }

  // a dust blob; the particles within blob_r of (blob_x1, blob_x2)
  // (periodic distance) carry (1 + blob_amp) times their mass (2-D lattice or random placement)
  const Real blob_amp = pin->GetOrAddReal("problem", "blob_amp", 0.0);
  if (blob_amp != 0.0) {
    if (three_d || cloud_root_random) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/blob_amp needs a 2-D run without cloud_root_random" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real bx = pin->GetReal("problem", "blob_x1"), by = pin->GetReal("problem", "blob_x2");
    const Real br = pin->GetReal("problem", "blob_r");
    const Real lx = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
    const Real ly = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
    auto &prb = ppar->prtcl_rdata;
    par_for("nsh_blob", DevExeSpace(), 0, ppar->nprtcl_thispack-1, KOKKOS_LAMBDA(const int p) {
      Real dx = prb(IPX,p) - bx, dy = prb(IPY,p) - by;
      dx -= lx*std::round(dx/lx); dy -= ly*std::round(dy/ly);
      if (dx*dx + dy*dy < br*br) prb(IPM,p) *= (1.0 + blob_amp);
    });
  }

  // particle timestep from the largest drift speed (transport is radial drift here)
  dust_setup::SetParticleDt(pmy_mesh_, vx, vp,
                            std::max(std::fabs(two_stream_v), std::fabs(two_stream_vu)));
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustNSHErrors()
//! \brief Measures the deviation of the mean gas and per-species dust drift velocities
//! from the NSH equilibrium at the end of the run and appends the errors to
//! "<basename>-errs.dat". Both the coupled and dust-free equilibria are fixed points,
//! so all errors should be at round-off level.

void DustNSHErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  int nspec = (pmbp->pdust != nullptr) ? pmbp->pdust->nspecies : 0;
  bool three_d = pm->three_d;

  Real omega0, qshear, etavk;
  std::vector<Real> eps, taus;
  ReadNSHParams(pin, pmbp, omega0, qshear, etavk, eps, taus);
  Real ugx, ugp;
  std::vector<Real> vx, vp;
  dust_setup::SolveNSH(omega0, qshear, 2.0*omega0*etavk, eps, taus, ugx, ugp, vx, vp);

  // mean gas velocities over active cells
  auto &indcs = pm->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  int iazim = three_d ? IM2 : IM3;
  Real gmx = 0.0, gmp = 0.0, gm = 0.0;
  Kokkos::parallel_reduce("nsh_gsum",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mx, Real &mp, Real &mass) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    mx += u0(m,IM1,k,j,i);
    mp += u0(m,iazim,k,j,i);
    mass += u0(m,IDN,k,j,i);
  }, Kokkos::Sum<Real>(gmx), Kokkos::Sum<Real>(gmp), Kokkos::Sum<Real>(gm));

  // per-species mean dust velocities (absent in the dust-free equilibrium)
  std::vector<Real> pvx(nspec), pvp(nspec), pn(nspec);
  if (nspec > 0) {
    auto &pr = ppar->prtcl_rdata;
    auto &pi = ppar->prtcl_idata;
    int npart = ppar->nprtcl_thispack;
    int ivazim = three_d ? IPVY : IPVZ;
    for (int s=0; s<nspec; ++s) {
      Real sx = 0.0, sp = 0.0, sn = 0.0;
      Kokkos::parallel_reduce("nsh_psum",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
      KOKKOS_LAMBDA(const int &p, Real &x_, Real &p_, Real &n_) {
        if (pi(PSP,p) == s) {
          x_ += pr(IPVX,p);
          p_ += pr(ivazim,p);
          n_ += 1.0;
        }
      }, Kokkos::Sum<Real>(sx), Kokkos::Sum<Real>(sp), Kokkos::Sum<Real>(sn));
      pvx[s] = sx;
      pvp[s] = sp;
      pn[s] = sn;
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &gmx, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gmp, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gm,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (nspec > 0) {
    MPI_Allreduce(MPI_IN_PLACE, pvx.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pvp.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pn.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif

  Real err_ugx = fabs(gmx/gm - ugx);
  Real err_ugp = fabs(gmp/gm - ugp);

  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      std::fclose(pfile);
      pfile = std::fopen(fname.c_str(), "a");
    } else {
      pfile = std::fopen(fname.c_str(), "w");
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  err_ugx  err_ugphi");
      if (nspec > 0) {std::fprintf(pfile, "  err_vx_s  err_vphi_s ...");}
      std::fprintf(pfile, "\n");
    }
    std::fprintf(pfile, "%04d  %04d  %04d  %05d  %e  %e", pm->mesh_indcs.nx1,
                 pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle, err_ugx, err_ugp);
    for (int s=0; s<nspec; ++s) {
      std::fprintf(pfile, "  %e  %e", fabs(pvx[s]/pn[s] - vx[s]),
                   fabs(pvp[s]/pn[s] - vp[s]));
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void DustNSHHistory()
//! \brief User history output: the per-species drift sums of dust_setup, the ba_history
//! columns, and for 3-D refinable clouds the error columns of the equilibrium

void DustNSHHistory(HistoryData *pdata, Mesh *pm) {
  if (!dust_setup::SpeciesHistory(pdata, pm)) return;
  dust_snapshots::HistoryColumns(pdata, pm);
  dust_setup::DustTest3DHistory(pdata, pm, dtest3d_);
  dust_snapshots::Write(pm);
}
