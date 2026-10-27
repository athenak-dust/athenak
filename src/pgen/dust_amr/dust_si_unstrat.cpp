//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_si_unstrat.cpp
//! \brief Problem generator for the streaming instability in an unstratified shearing box
//! (Youdin & Goodman 2005; nonlinear runs as Johansen & Youdin 2007 and Bai & Stone 2010,
//! e.g. run BA: tau_s = 1, eps = 0.2).  The start is the multi-species NSH drift equilibrium
//! of dust_nsh (radial forcing 2 Omega etavk on the gas through <hydro_srcterms>/const_accel)
//! with the dust at random positions, so that the instability grows from the particle noise.
//!
//! The particles: <problem>/cloud_root_random = true places root parents over the whole box
//! (<problem>/root_placement = random by default, or lattice | generic) that split exactly to
//! the leaves of a refined mesh as refinable clouds (on a uniform mesh, or with
//! <dust>/amr_pic = owner_overlap, the same parents as point particles); otherwise point
//! particles with <problem>/particle_placement = random (default) or lattice.  <problem>/ba_*
//! adds the 2-D diagnostic outputs of dust_snapshots.cpp.  Works in the 2D r-z shearing box
//! (azimuthal components in IM3/IPVZ) and in 3D (azimuthal in IM2/IPVY, shear-periodic x1).

#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "pgen/dust_amr/dust_setup.hpp"
#include "pgen/dust_amr/dust_snapshots.hpp"

namespace {
void SIUnstratHistory(HistoryData *pdata, Mesh *pm) {
  if (!dust_setup::SpeciesHistory(pdata, pm)) return;
  dust_snapshots::HistoryColumns(pdata, pm);
  dust_snapshots::Write(pm);
}

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust_si_unstrat" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustSIUnstrat()

void ProblemGenerator::DustSIUnstrat(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!dust_setup::DustActive(pmy_mesh_)) Fatal("needs <particles> and <dust>");
  dust_setup::RejectKeys(pin, "dust_si_unstrat", {
      {"dust_test", "none", "dust_damping_amr or dusty_wave_amr"},
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
  user_hist_func = SIUnstratHistory;
  dust_setup::EnrollRefinement(pin, pmy_mesh_, this);
  if (restart) {
    dust_snapshots::Restart(pin, pmy_mesh_);
    dust_setup::LogRestartParticles(pmy_mesh_);
    return;
  }
  if (pmbp->phydro == nullptr || !pin->DoesBlockExist("shearing_box")) {
    Fatal("needs a <hydro> and a <shearing_box> block");
  }

  // the NSH drift equilibrium of gas and dust
  const Real omega0 = pin->GetReal("shearing_box","omega0");
  const Real qshear = pin->GetReal("shearing_box","qshear");
  const Real etavk = pin->GetReal("problem","etavk");
  std::vector<Real> eps, taus;
  dust_setup::ReadSpecies(pin, pmbp, false, eps, taus);
  const int nspec = static_cast<int>(eps.size());
  const Real ax = 2.0*omega0*etavk;
  dust_setup::CheckRadialForcing(pin, ax);
  Real ugx = 0.0, ugp = 0.0;
  std::vector<Real> vx(nspec, 0.0), vp(nspec, 0.0);
  dust_setup::SolveNSH(omega0, qshear, ax, eps, taus, ugx, ugp, vx, vp);
  if (global_variable::my_rank == 0) {
    std::cout << "# NSH equilibrium: u_gx=" << ugx << " u_gphi=" << ugp << std::endl;
    for (int s=0; s<nspec; ++s) {
      std::cout << "#   species " << s+1 << ": v_x=" << vx[s] << " v_phi=" << vp[s] << std::endl;
    }
  }

  const Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  dust_setup::InitUniformGas(pmy_mesh_, rho0, ugx, ugp);
  dust_setup::DustInit d;
  d.rho0 = rho0;
  d.eps = eps;
  d.vx = vx;
  d.vp = vp;
  d.placement = "random";
  dust_setup::PlaceParticles(pin, pmy_mesh_, d);
  dust_setup::SetParticleDt(pmy_mesh_, vx, vp, 0.0);
}
