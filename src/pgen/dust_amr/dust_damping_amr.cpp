//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_damping_amr.cpp
//! \brief Problem generator for the drag damping test on refined meshes: uniform dust
//! moving at <problem>/dust_test_v0 (default 0.1) along x1 through uniform gas at rest, without
//! rotation (no <shearing_box>).  The centre-of-mass velocity is conserved and the relative
//! velocity decays as exp(-(1 + eps) t/t_s), so the gas velocity is
//!     u(t) = eps v0/(1 + eps) (1 - exp(-(1 + eps) t/t_s))
//! (one species).  As dust_damping, but with the particle set-up of dust_setup: point
//! particles on a lattice (or random), or with <problem>/cloud_root_random = true root parents
//! (<problem>/root_placement = lattice | random | generic) that split exactly to the leaves
//! of a refined mesh as refinable clouds.  In 3-D with one species of root parents the
//! history carries the errors against the exact solution (e_rhog, e_ux, e_rhod, e_uperp, the
//! non-1-D parts y_rhog, y_rhod, and the leaf mass error dmass).  <problem>/dust_test_vz0 adds a
//! uniform vz to the 3-D root parents (with dust_test_vz_mirror its sign follows z).
//! <problem>/ba_* adds the 2-D diagnostic outputs of dust_snapshots.cpp.
//!
//! With dust_test_v0 = 0 this is also the quiet set-up of uniform dust at rest that
//! dust_freefall and dust_jeans start from.

#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
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
dust_setup::DustTest3D dtest3d_;

void DampingHistory(HistoryData *pdata, Mesh *pm) {
  if (!dust_setup::SpeciesHistory(pdata, pm)) return;
  dust_snapshots::HistoryColumns(pdata, pm);
  dust_setup::DustTest3DHistory(pdata, pm, dtest3d_);
  dust_snapshots::Write(pm);
}

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust_damping_amr" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustDampingAMR()

void ProblemGenerator::DustDampingAMR(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!dust_setup::DustActive(pmy_mesh_)) Fatal("needs <particles> and <dust>");
  if (pin->GetOrAddString("problem", "dust_test", "damping") != "damping") {
    Fatal("<problem>/dust_test must be damping (dusty_wave_amr is the wave)");
  }
  if (pin->DoesBlockExist("shearing_box")) Fatal("a drag test without rotation: no <shearing_box>");
  dust_setup::RejectKeys(pin, "dust_damping_amr", {
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
  user_hist_func = DampingHistory;
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
  const Real v0 = pin->GetOrAddReal("problem", "dust_test_v0", 0.1);
  std::vector<Real> vx(nspec, v0), vp(nspec, 0.0);
  const Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  dust_setup::InitUniformGas(pmy_mesh_, rho0, 0.0, 0.0);

  const bool cloud_root_random = pin->GetOrAddBoolean("problem", "cloud_root_random", false);
  if (pmy_mesh_->three_d && nspec == 1 && cloud_root_random) {   // the 3-D history errors
    dtest3d_.on = true;
    dtest3d_.rho0 = rho0; dtest3d_.eps = eps[0]; dtest3d_.ts = taus[0];
    dtest3d_.v0 = v0;
  }
  dust_setup::DustInit d;
  d.rho0 = rho0;
  d.eps = eps;
  d.vx = vx;
  d.vp = vp;
  d.placement = "lattice";
  d.vz_add = pin->GetOrAddReal("problem", "dust_test_vz0", 0.0);
  d.vz_mirror = pin->GetOrAddBoolean("problem", "dust_test_vz_mirror", false);
  dust_setup::PlaceParticles(pin, pmy_mesh_, d);
  dust_setup::SetParticleDt(pmy_mesh_, vx, vp, 0.0);
}
