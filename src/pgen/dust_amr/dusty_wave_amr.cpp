//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dusty_wave_amr.cpp
//! \brief Problem generator for the damped dusty sound wave on refined meshes, without
//! rotation (no <shearing_box>): the right-travelling eigenmode of gas and N dust species
//! (at most 4) with <problem>/wave_n wavelengths along x1 and the gas velocity amplitude
//! <problem>/amp (in units of cs).  The dust starts at its eigenmode displacement and velocity.
//! As dusty_wave, but with the particle set-up of dust_setup: point particles on a lattice
//! (in 3-D with <problem>/lattice_sub on an n^3 sub-lattice of every cell), or with
//! <problem>/cloud_root_random = true root parents (<problem>/root_placement = lattice |
//! random | generic) that split exactly to the leaves of a refined mesh as refinable clouds.
//! In 3-D with one species of root parents the history carries the L1 errors against the
//! eigenmode (e_rhog, e_ux, e_rhod over their amplitudes, e_uperp, the non-1-D parts y_rhog,
//! y_rhod, and the leaf mass error dmass).  <problem>/ba_* adds the 2-D diagnostic outputs of
//! dust_snapshots.cpp.

#include <complex>
#include <iomanip>
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
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "pgen/dust_amr/dust_setup.hpp"
#include "pgen/dust_amr/dust_snapshots.hpp"

namespace {
dust_setup::DustTest3D dtest3d_;

void WaveHistory(HistoryData *pdata, Mesh *pm) {
  if (!dust_setup::SpeciesHistory(pdata, pm)) return;
  dust_snapshots::HistoryColumns(pdata, pm);
  dust_setup::DustTest3DHistory(pdata, pm, dtest3d_);
  dust_snapshots::Write(pm);
}

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dusty_wave_amr" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn DustTestOmega
//! \brief dust_test = wave: the right-travelling damped root of the dusty-wave dispersion
//! relation omega^2 (1 + sum_s eps_s/(1 - i omega t_s)) = k^2 cs^2 (as dusty_wave.cpp).

std::complex<Real> DustTestOmega(const Real kwv, const Real cs, const std::vector<Real> &eps,
                                 const std::vector<Real> &ts) {
  const std::complex<Real> iunit(0.0, 1.0);
  Real epstot = 0.0;
  for (const Real e : eps) epstot += e;
  std::complex<Real> omega(kwv*cs/std::sqrt(1.0 + epstot), 0.0);
  for (int it=0; it<100; ++it) {
    std::complex<Real> ssum(0.0, 0.0), dsum(0.0, 0.0);
    for (std::size_t s=0; s<eps.size(); ++s) {
      const std::complex<Real> den = 1.0 - iunit*omega*ts[s];
      ssum += eps[s]/den;
      dsum += eps[s]*iunit*ts[s]/(den*den);
    }
    const std::complex<Real> f = omega*omega*(1.0 + ssum) - SQR(kwv*cs);
    const std::complex<Real> fp = 2.0*omega*(1.0 + ssum) + omega*omega*dsum;
    const std::complex<Real> d = f/fp;
    omega -= d;
    if (std::abs(d) < 1.0e-15*std::abs(omega)) break;
  }
  return omega;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustyWaveAMR()

void ProblemGenerator::DustyWaveAMR(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!dust_setup::DustActive(pmy_mesh_)) Fatal("needs <particles> and <dust>");
  if (pin->GetOrAddString("problem", "dust_test", "wave") != "wave") {
    Fatal("<problem>/dust_test must be wave (dust_damping_amr is the damping test)");
  }
  if (pin->DoesBlockExist("shearing_box")) Fatal("a drag test without rotation: no <shearing_box>");
  dust_setup::RejectKeys(pin, "dusty_wave_amr", {
      {"dust_test_vz0", "0", "dust_damping_amr"},
      {"stratified_dust", "false", "dust_si_strat"},
      {"stratified_dust_sampling", "weight", "dust_si_strat"},
      {"dust_zcut", "0", "dust_si_strat"},
      {"nsh_vertical", "false", "dust_si_strat"},
      {"vertical_bc", "none", "dust_si_strat"},
      {"two_stream_v", "0", "dust_nsh"},
      {"two_stream_v_upper", "0", "dust_nsh"},
      {"blob_amp", "0", "dust_nsh"},
      {"gas_vx_amp", "0", "dust_nsh"},
      {"dust_ymode_amp", "0", "dust_nsh"}});
  dust_snapshots::Setup(pin, pmy_mesh_, true);
  dust_setup::SetupHistory(pin);
  user_hist_func = WaveHistory;
  dust_setup::EnrollRefinement(pin, pmy_mesh_, this);
  if (restart) {
    dust_snapshots::Restart(pin, pmy_mesh_);
    dust_setup::LogRestartParticles(pmy_mesh_);
    return;
  }
  if (pmbp->phydro == nullptr) Fatal("needs a <hydro> block");

  std::vector<Real> eps, taus;
  dust_setup::ReadSpecies(pin, pmbp, false, eps, taus);
  const int nspec = static_cast<int>(eps.size());
  if (nspec > 4) Fatal("at most 4 species");
  const Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  const bool three_d = pmy_mesh_->three_d;

  // the eigenmode (velocity normalisation as dusty_wave.cpp)
  dust_setup::DustInit d;
  const Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-6);
  const int nwave = pin->GetOrAddInteger("problem", "wave_n", 1);
  const Real cs = pmbp->phydro->peos->eos_data.iso_cs;
  const Real kwv = 2.0*M_PI*nwave/(pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);
  const std::complex<Real> iunit(0.0, 1.0);
  const std::complex<Real> omega = DustTestOmega(kwv, cs, eps, taus);
  const std::complex<Real> du = amp*cs, drho = (kwv/omega)*rho0*du;
  d.kwv = kwv;
  for (int s=0; s<nspec; ++s) {
    const std::complex<Real> dv = du/(1.0 - iunit*omega*taus[s]);
    const std::complex<Real> xi = iunit*dv/omega;   // displacement: d(xi)/dt = dv
    d.xir[s] = xi.real(); d.xii[s] = xi.imag();
    d.dvr[s] = dv.real(); d.dvi[s] = dv.imag();
  }
  if (three_d && nspec == 1) {   // the exact solution for the 3-D history errors
    const std::complex<Real> dv = du/(1.0 - iunit*omega*taus[0]);
    const std::complex<Real> drd = (kwv/omega)*eps[0]*rho0*dv;
    dtest3d_.wave = true;
    dtest3d_.kwv = kwv; dtest3d_.wr = omega.real(); dtest3d_.wi = omega.imag();
    dtest3d_.dgr = drho.real(); dtest3d_.dgi = drho.imag();
    dtest3d_.dur = du.real(); dtest3d_.dui = du.imag();
    dtest3d_.ddr = drd.real(); dtest3d_.ddi = drd.imag();
  }
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(15) << "# dusty wave: k = " << kwv << ", omega = "
              << omega.real() << " " << omega.imag() << " i, |drho| = " << std::abs(drho)
              << std::endl;
  }
  // the gas part of the eigenmode (cell-centre values)
  {
    auto &indcs = pmy_mesh_->mb_indcs;
    const int ng = indcs.ng, nmb = pmbp->nmb_thispack;
    const int n1 = indcs.nx1 + 2*ng;
    const int n2 = (indcs.nx2 > 1) ? (indcs.nx2 + 2*ng) : 1;
    const int n3 = (indcs.nx3 > 1) ? (indcs.nx3 + 2*ng) : 1;
    auto &u0 = pmbp->phydro->u0;
    const Real drr = drho.real(), dri = drho.imag(), dur = du.real(), dui = du.imag();
    const int is0 = indcs.is, nx1 = indcs.nx1;
    auto &mbs = pmbp->pmb->mb_size;
    par_for("dust_test_gas", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x = CellCenterX(i - is0, nx1, mbs.d_view(m).x1min, mbs.d_view(m).x1max);
      const Real c = cos(kwv*x), sn = sin(kwv*x);
      const Real rho = rho0 + drr*c - dri*sn, v = dur*c - dui*sn;
      u0(m,IDN,k,j,i) = rho;
      u0(m,IM1,k,j,i) = rho*v;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
    });
  }

  const bool cloud_root_random = pin->GetOrAddBoolean("problem", "cloud_root_random", false);
  if (three_d && nspec == 1 && cloud_root_random) {   // the 3-D history errors
    dtest3d_.on = true;
    dtest3d_.rho0 = rho0; dtest3d_.eps = eps[0]; dtest3d_.ts = taus[0];
    dtest3d_.v0 = 0.0;
  }
  std::vector<Real> vx(nspec, 0.0), vp(nspec, 0.0);
  d.rho0 = rho0;
  d.eps = eps;
  d.vx = vx;
  d.vp = vp;
  d.placement = "lattice";
  dust_setup::PlaceParticles(pin, pmy_mesh_, d);
  dust_setup::SetParticleDt(pmy_mesh_, vx, vp, 0.0);
}
