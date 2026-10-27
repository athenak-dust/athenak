//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust.cpp
//! \brief implementation of DustGasDrag class constructor and support functions

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "shearing_box/shearing_box.hpp"
#include "dust.hpp"
#include "dust_species.hpp"
#include "dust_resample.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
// constructor: parses input, validates configuration, allocates deposit fields and
// boundary communication objects

DustGasDrag::DustGasDrag(MeshBlockPack *ppack, ParameterInput *pin) :
    taus("dust_taus",1),
    qdep("qdep",1,1,1,1,1),
    ustar("ustar",1,1,1,1,1),
    dmom("dmom",1,1,1,1,1),
    cdummy("cdum",1,1,1,1,1),
    solver_r("dust_solver_r",1,1,1,1,1),
    solver_p("dust_solver_p",1,1,1,1,1),
    solver_ap("dust_solver_ap",1,1,1,1,1),
    pmy_pack(ppack) {
  // The compile-time choices of the dust module, once at start
  // (DUST_CLOUD_LAUNCH_BOUNDS: dust.hpp; DUST_R_QR_COLUMNS, DUST_R_DIAG: dust_resample.hpp)
  {
    static bool told = false;
    if (!told && global_variable::my_rank == 0) {
      std::cout << "# DUST_BUILD launch_bounds=" << DUST_CLOUD_LAUNCH_BOUNDS
                << " r_qr_columns=" << DUST_R_QR_COLUMNS << " r_diag=" << DUST_R_DIAG << std::endl;
    }
    told = true;
  }
  // (1) validate configuration ----------------------------------------------------------
  hydro::Hydro *phyd = pmy_pack->phydro;
  particles::Particles *ppar = pmy_pack->ppart;
  if (phyd == nullptr || ppar == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust> block requires both <hydro> and <particles> blocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmhd != nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust drag currently only couples to Hydro, but <mhd> block detected"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (ppar->particle_type != ParticleType::dust) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust> block requires <particles>/particle_type = dust" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Driver is constructed after physics modules, so read the coupling and integrator
  // directly from the input while validating the module configuration.
  bool pc2_only = false;
  {
    std::string method = pin->GetOrAddString("dust", "coupling", "imex");
    if (method.compare("imex") == 0) {
      coupling = DustCoupling::imex;
    } else if (method.compare("hybrid") == 0) {
      coupling = DustCoupling::hybrid;
    } else if (method.compare("pc2") == 0) {
      // PC2 uses the hybrid task path, but this direct spelling disables the
      // automatic split-BE fallback and selects PC2 for every cycle.
      coupling = DustCoupling::hybrid;
      pc2_only = true;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/coupling = '" << method
                << "' not recognized (must be imex, pc2, or hybrid)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  std::string integrator = pin->GetOrAddString("time", "integrator", "rk2");
  bool bad_integrator = ((coupling == DustCoupling::imex &&
                          integrator.compare("imex2+") != 0) ||
                         (coupling == DustCoupling::hybrid &&
                          integrator.compare("rk2") != 0));
  if (bad_integrator) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust coupling=imex requires integrator=imex2+, while coupling=pc2 "
              << "or hybrid requires integrator=rk2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // keys of the development branches that are no longer read (one NOTE each; the run goes on).
  // Refined meshes, adaptive refinement, 3-D and several MPI ranks need no opt-in (the supported
  // configurations are checked below, in CheckMultiRank and in CheckClouds3D).
  if (global_variable::my_rank == 0) {
    NoteDroppedKey(pin, "dust", "experimental_static_smr", "refined meshes need no opt-in");
    NoteDroppedKey(pin, "dust", "experimental_amr", "adaptive meshes need no opt-in");
    NoteDroppedKey(pin, "dust", "experimental_3d", "3-D refinable clouds need no opt-in");
    NoteDroppedKey(pin, "dust", "experimental_mpi", "several MPI ranks need no opt-in");
    NoteDroppedKey(pin, "dust", "experimental_3d_shear_plain",
                   "test option removed: 3-D clouds take the sheared image across shear-periodic x1");
    NoteDroppedKey(pin, "dust", "experimental_smr_measure_correction",
                   "acted only with the removed smooth_p2 transfer");
    NoteDroppedKey(pin, "dust", "experimental_smr_blend_width_cells",
                   "acted only with the removed smooth_p2 transfer");
    if (pin->DoesParameterExist("dust", "experimental_smr_transfer")) {
      std::cout << "# NOTE: <dust>/experimental_smr_transfer = "
                << pin->GetString("dust", "experimental_smr_transfer")
                << " is read as <dust>/amr_pic (replace the old key)" << std::endl;
    }
  }
  // <dust>/amr_pic (dust_input.hpp): refinable_cloud (default on a refined mesh) or
  // owner_overlap; on a uniform mesh the point particles (none) unless refinable_cloud is given
  {
    const bool ml = pmy_pack->pmesh->multilevel;
    const AmrPic pic = ResolveAmrPic(pin, ml);
    smr_transfer_mode = (pic == AmrPic::refinable_cloud) ? DustSMRTransfer::refinable_cloud :
                        (pic == AmrPic::owner_overlap) ? DustSMRTransfer::owner_overlap :
                                                         DustSMRTransfer::none;
    if (!ml && pin->DoesParameterExist("dust", "amr_pic") && global_variable::my_rank == 0) {
      std::cout << "# NOTE (dust): amr_pic = " << pin->GetString("dust", "amr_pic")
                << " on a uniform mesh: "
                << ((pic == AmrPic::refinable_cloud) ?
                    "refinable clouds (exact TSC clouds, the cloud path of the refined runs)" :
                    "point particles with TSC (owner_overlap acts only on a refined mesh)")
                << std::endl;
    }
  }
  // S3b: refinable clouds on a 3-D mesh (dust_cloud3d.cpp)
  clouds3d = pmy_pack->pmesh->three_d && smr_transfer_mode == DustSMRTransfer::refinable_cloud;
  // Stage 4: shear-periodic x1 in the leaf traversal (the cloud supports across
  // the radial boundaries on the sheared image)
  shear3d = clouds3d &&
      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic;
  // (test) false: particles crossing an x1 face after the push take the shear offset of the start
  // of the step, as the routing of point particles does
  if (shear3d) shear3d_route_fix = pin->GetOrAddBoolean("dust", "cloud_shear_route_time", true);
  // Rixin: default on (history bitwise equal)
  if (shear3d) shear3d_list_cache = pin->GetOrAddBoolean("dust", "cloud_shear_list_cache", true);
  // 3-D clouds at physical x3 faces fold their supports back (the drop mode has
  // no cloud version)
  if (clouds3d && pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic &&
      pin->GetOrAddString("dust", "x3_ghost_deposit", "fold") != "fold") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D refinable clouds at physical x3 faces: x3_ghost_deposit = fold only "
              << "(the supports fold back)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // Rixin: set the default to 4 (the balance of clouds per rank, fastest of 1 / 2 / 4); 0 gives
  // every new block the cost 1 as AthenaK does; one rank never uses it
  lb_cloud_weight = pin->GetOrAddReal("dust", "load_balance_cloud_weight", 4.0);
  if (pin->GetOrAddBoolean("dust", "amr_density_trigger", false)) {   // (block map: the only one)
    halo_trigger_cells = std::max(0, pin->GetOrAddInteger("dust", "amr_rho_buffer", 2));
  }
  if (pmy_pack->pmesh->multilevel) {
    // The fenced phase profile (DUST_AMR_PROFILE) only on request; its fences make the
    // host wait for the device several times per cycle
    ProfOn() = pin->GetOrAddBoolean("dust", "amr_profile", false);
    // static or adaptive refinement (dust_amr.cpp: the dust state is rebuilt
    // after every regrid): a 2-D periodic mesh, or a 3-D mesh with periodic or shear-periodic x1
    // (for owner_overlap and refinable clouds), periodic x2 and periodic or physical x3
    // faces; the boundaries are checked below
    if (!((pmy_pack->pmesh->two_d && pmy_pack->pmesh->strictly_periodic) ||
          pmy_pack->pmesh->three_d)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust on a refined mesh needs a 2-D mesh with periodic "
                << "boundaries or a 3-D mesh" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  bool shear_x1 = (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] ==
                   BoundaryFlag::shear_periodic);
  // Physical x3 faces in 3-D (a stratified box, S3c): particles reflect or leave
  // (<dust>/particle_x3_bc), ghost deposits beyond the faces fold back (or are dropped,
  // <dust>/x3_ghost_deposit); the PC2 gather reads the gas ghost cells of the gas boundary
  // conditions, imex and the automatic hybrid the u* ghost cells.
  const bool x3_phys = pmy_pack->pmesh->three_d &&
      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic;
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    // exceptions: 3D shearing box with shear-periodic x1 (and periodic x2), 3-D with physical
    // x3 faces (x1 periodic or shear-periodic, x2 periodic)
    const auto &mbc = pmy_pack->pmesh->mesh_bcs;
    const bool x1ok = shear_x1 || mbc[BoundaryFace::inner_x1] == BoundaryFlag::periodic;
    const bool x2ok = mbc[BoundaryFace::inner_x2] == BoundaryFlag::periodic;
    if (!(pmy_pack->pmesh->three_d && x1ok && x2ok && (shear_x1 || x3_phys))) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust drag requires periodic boundaries in all "
                << "directions (or shear-periodic x1 and / or physical x3 faces in 3D)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // imex and the automatic hybrid read u*, which gets x3 boundary values
    // (DustGasDrag::UstarX3BCs); every coupling is allowed
    // S3a: the ghost deposits across the shear-periodic faces are moved
    // to the opposite face with the shear offset (DepRemapPlan); the u* ghost fill is remapped
  }
  // AthenaK's time step is cfl_number times the smallest dx/(|v| + c)
  // over the directions; the unsplit RK2 + PLM gas update is stable in 3-D only up to about 1/3
  // (the oblique hydro linear wave breaks down at 0.4).  The dust deposits seed the grid-scale
  // diagonal modes, so a 3-D dust run above 1/3 blows up within a fraction of a sound crossing.
  if (pmy_pack->pmesh->three_d &&
      pin->GetReal("time", "cfl_number") > 1.0/3.0 + 1.0e-12 &&
      !pin->GetOrAddBoolean("dust", "allow_cfl_above_third_3d", false)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D dust: <time>/cfl_number = " << pin->GetReal("time", "cfl_number")
              << " exceeds 1/3, the stability limit of the 3-D RK2 + PLM gas update"
              << "; use cfl_number <= 0.3 (or <dust>/allow_cfl_above_third_3d = true for an "
              << "integrator that allows more)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmesh->mb_indcs.ng < 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust drag requires at least 2 ghost zones" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (2) read parameters ----------------------------------------------------------------
  back_reaction = pin->GetOrAddBoolean("dust","back_reaction",true);
  gamma_switch  = pin->GetOrAddBoolean("dust","gamma_switch",false);
  if (coupling == DustCoupling::hybrid && gamma_switch) {
    if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust): <dust>/gamma_switch=true has no effect for "
                << "coupling=" << (pc2_only ? "pc2" : "hybrid")
                << "; gamma_switch only modifies the imex2+ tableau and will be ignored."
                << std::endl;
    }
    gamma_switch = false;
  }
  stopping_times_initialized = false;
  dt_cfl        = pin->GetOrAddReal("dust","dt_cfl",0.5);
  // Dust API: <dust>/dust_to_gas is the legacy total (to be deprecated); with
  // the new abundance keys it is not read (mixing is an error, dust_species.cpp) and holds the
  // sum of epsilon_N / Z_N for the default masses
  {
    const SpeciesTable &st = Species(pin);
    dust_to_gas = st.new_init ? st.total : pin->GetOrAddReal("dust","dust_to_gas",0.01);
  }

  hybrid_enter_zeta = pin->GetOrAddReal("dust", "hybrid_enter_zeta", 0.5);
  hybrid_enter_chi  = pin->GetOrAddReal("dust", "hybrid_enter_chi", 0.5);
  hybrid_exit_zeta  = pin->GetOrAddReal("dust", "hybrid_exit_zeta", 0.25);
  hybrid_exit_chi   = pin->GetOrAddReal("dust", "hybrid_exit_chi", 0.25);
  // PC2 feedback stiffness monitor, optional dt limit and predictor damping.  Rixin:
  // the damping (as Athena-C) on by default, the dt limit left to the user (off)
  stiff_thresh = pin->GetOrAddReal("dust", "stiff_thresh", -1.0);
  // The IMEX gas gets the lumped solve's change plus its mass share of the residual (3-D
  // clouds; false = the gas takes the clouds' whole impulse, as before)
  imex_consistent_gas = pin->GetOrAddBoolean("dust", "imex_consistent_gas", true);
  pc2_feedback_damping = pin->GetOrAddBoolean("dust", "pc2_feedback_damping", true);
  if (!(hybrid_enter_zeta > hybrid_exit_zeta && hybrid_exit_zeta >= 0.0 &&
        hybrid_enter_chi > hybrid_exit_chi && hybrid_exit_chi >= 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Hybrid enter thresholds must be positive and exceed exit thresholds"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pc2_only) {
    hybrid_force_mode = HybridForceMode::pc2;
    hybrid_mode = HybridMode::pc2;
  } else {
    std::string forced = pin->GetOrAddString("dust", "hybrid_force_mode", "auto");
    if (forced.compare("auto") == 0) {
      hybrid_force_mode = HybridForceMode::automatic;
      hybrid_mode = HybridMode::pc2;
    } else if (forced.compare("pc2") == 0) {
      hybrid_force_mode = HybridForceMode::pc2;
      hybrid_mode = HybridMode::pc2;
    } else if (forced.compare("split_be") == 0) {
      hybrid_force_mode = HybridForceMode::split_be;
      hybrid_mode = HybridMode::split_be;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/hybrid_force_mode = '" << forced
                << "' not recognized (must be auto, pc2, or split_be)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  {
    std::string solver = pin->GetOrAddString("dust", "drag_solver", "local");
    if (solver.compare("local") == 0) {
      drag_solver = DustDragSolver::local;
    } else if (solver.compare("applya") == 0) {
      drag_solver = DustDragSolver::applya;
    } else if (solver.compare("dc1") == 0) {
      drag_solver = DustDragSolver::dc1;
    } else if (solver.compare("dc2") == 0) {
      drag_solver = DustDragSolver::dc2;
    } else if (solver.compare("pcg") == 0) {
      drag_solver = DustDragSolver::pcg;
    } else if (solver.compare("adaptive") == 0) {
      drag_solver = DustDragSolver::adaptive;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/drag_solver = '" << solver
                << "' not recognized (must be local, applya, dc1, dc2, pcg, or adaptive)"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (coupling == DustCoupling::hybrid &&
      drag_solver != DustDragSolver::local && drag_solver != DustDragSolver::dc1 &&
      drag_solver != DustDragSolver::pcg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "The first hybrid split-BE implementation supports only "
              << "drag_solver=local, dc1, or pcg" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmesh->multilevel && drag_solver != DustDragSolver::local) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dust on a refined mesh supports only drag_solver = local "
              << "(the coupled solvers have no composite transfer)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  drag_rtol = pin->GetOrAddReal("dust", "drag_rtol", 1.0e-11);
  drag_atol = pin->GetOrAddReal("dust", "drag_atol", 1.0e-14);
  drag_iter_max = pin->GetOrAddInteger("dust", "drag_iter_max", 200);
  drag_diagnostic_interval = pin->GetOrAddInteger("dust", "drag_diagnostic_interval", 0);
  adaptive_order_c = pin->GetOrAddReal("dust", "drag_adaptive_order_c", 0.25);
  adaptive_rtol_max = pin->GetOrAddReal("dust", "drag_adaptive_rtol_max", 1.0e-3);
  adaptive_tref = pin->GetOrAddReal("dust", "drag_adaptive_tref", 1.0);
  adaptive_state_floor = pin->GetOrAddReal("dust", "drag_adaptive_state_floor", 1.0);
  if (drag_rtol <= 0.0 || drag_atol < 0.0 || drag_iter_max < 1 ||
      drag_diagnostic_interval < 0 || adaptive_order_c <= 0.0 ||
      adaptive_rtol_max <= 0.0 || adaptive_tref <= 0.0 ||
      adaptive_state_floor <= 0.0) {
    // (drag_fail_policy, whose only value was abort, is no longer read)
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid coupled drag-solver tolerance/iteration policy"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0 &&
      (drag_solver == DustDragSolver::pcg || drag_solver == DustDragSolver::adaptive)) {
    solver_pcg_iteration_hist.assign(static_cast<std::size_t>(drag_iter_max) + 1, 0);
  }

  {
    std::string mode = pin->GetOrAddString("dust","stopping_time_mode","species_fixed");
    if (mode.compare("species_fixed") == 0) {
      stopping_time_mode = DustStoppingTimeMode::species_fixed;
    } else if (mode.compare("particle_static") == 0) {
      stopping_time_mode = DustStoppingTimeMode::particle_static;
    } else if (mode.compare("dynamic") == 0) {
      stopping_time_mode = DustStoppingTimeMode::dynamic;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/stopping_time_mode = '" << mode
                << "' not recognized (must be species_fixed, particle_static, or "
                << "dynamic)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // drag work is not deposited into the gas energy equation, so back-reaction requires
  // an isothermal EOS for a consistent energy budget
  if (back_reaction && phyd->peos->eos_data.is_ideal) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust back-reaction requires an isothermal EOS (drag heating is not "
              << "implemented)" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // per-species stopping times from the resolved species table (dust_species.cpp):
  // tau_s_N, stokes_N or the power law; the legacy taus_N (to be deprecated) as before
  {
    const SpeciesTable &st = Species(pin);
    nspecies = st.nspecies;
    species_record = st.Record();
    taus = DualArray1D<Real>("dust_taus", nspecies);
    taus_max = 0.0;
    taus_min = std::numeric_limits<Real>::max();
    for (int s=0; s<nspecies; ++s) {
      const Real ts = static_cast<Real>(st.tau_s[s]);
      taus.h_view(s) = ts;
      taus_max = std::max(taus_max, ts);
      taus_min = std::min(taus_min, ts);
    }
  }
  taus.template modify<HostMemSpace>();
  taus.template sync<DevExeSpace>();

  // deposit scheme
  {
    std::string dep = pin->GetOrAddString("dust","deposit","tsc");
    if (dep.compare("ngp") == 0) {
      deposit = DustDeposit::ngp;
    } else if (dep.compare("cic") == 0) {
      deposit = DustDeposit::cic;
    } else if (dep.compare("tsc") == 0) {
      deposit = DustDeposit::tsc;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/deposit = '" << dep << "' not recognized "
                << "(must be ngp, cic, or tsc)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  {
    // The refinable-cloud path may also be selected on a uniform mesh (amr_pic =
    // refinable_cloud): its clouds are then exact TSC and the gather/deposit options apply
    // unchanged, which gives uniform reference runs with the same coupling as the
    // mixed-resolution runs.
    bool multilevel = pmy_pack->pmesh->multilevel;
    if (smr_transfer_mode == DustSMRTransfer::none && multilevel) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "dust on a refined mesh needs <dust>/amr_pic = refinable_cloud or "
                << "owner_overlap" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
      // <dust>/refinable_mode: the validated parameter sets of the refinable
      // clouds (RefinableMode, dust.hpp).  A preset key given explicitly (input or restart file)
      // overrides the preset value; PrintRefinablePreset lists the resolved values.
      {
        const bool given = pin->DoesParameterExist("dust", "refinable_mode");
        std::string mode = given ? pin->GetString("dust", "refinable_mode") : "default";
        // the switch of the development branches maps onto careful_merge
        if (pin->DoesParameterExist("dust", "cloud_quiet_protection")) {
          const bool q = pin->GetBoolean("dust", "cloud_quiet_protection");
          if (q && given && mode != "careful_merge") {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                      << "<dust>/cloud_quiet_protection = true contradicts refinable_mode = "
                      << mode << " (drop the old key; careful_merge holds the quiet gates)"
                      << std::endl;
            std::exit(EXIT_FAILURE);
          }
          if (q) mode = "careful_merge";
          if (global_variable::my_rank == 0) {
            std::cout << "# NOTE: <dust>/cloud_quiet_protection = " << (q ? "true" : "false")
                      << (q ? " is read as refinable_mode = careful_merge" : " is ignored")
                      << " (replace the old key)" << std::endl;
          }
        }
        mode = pin->GetOrAddString("dust", "refinable_mode", mode);
        if (mode == "default") {
          refinable_mode = RefinableMode::standard;
        } else if (mode == "careful_merge") {
          refinable_mode = RefinableMode::careful_merge;
        } else if (mode == "split_only") {
          refinable_mode = RefinableMode::split_only;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << "<dust>/refinable_mode = '" << mode << "' not recognized (default, "
                    << "careful_merge or split_only)" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
      // the keys that differ between the presets: the given value, else the preset's.  They are
      // not added to the parameter list (nor to the restart header), so that a restart with
      // another refinable_mode takes that mode's values
      auto preset_bool = [pin](const char *key, bool def) {
        return pin->DoesParameterExist("dust", key) ? pin->GetBoolean("dust", key) : def;
      };
      auto preset_real = [pin](const char *key, Real def) {
        return pin->DoesParameterExist("dust", key) ? pin->GetReal("dust", key) : def;
      };
      const bool careful = (refinable_mode == RefinableMode::careful_merge);
      const bool split_only = (refinable_mode == RefinableMode::split_only);
      {
        static const char *const keys[] = {
            "cloud_merge", "cloud_merge_stencil", "cloud_merge_budget_kappa", "cloud_merge_alpha",
            "cloud_merge_alpha_leaf", "cloud_merge_budget", "cloud_merge_schedule",
            "cloud_merge_dt_omega", "cloud_merge_interval", "cloud_merge_mode", "cloud_resample",
            "cloud_resample_eta", "cloud_resample_cold_alpha", "cloud_resample_schedule",
            "cloud_resample_dt", "cloud_resample_dt_omega", "cloud_resample_interval",
            "cloud_resample_after_regrid"};
        refinable_given.clear();
        for (const char *k : keys) {
          if (pin->DoesParameterExist("dust", k)) {
            refinable_given += std::string(" ") + k + "=" + pin->GetString("dust", k);
          }
        }
      }
      const bool x3phys3d = clouds3d &&
          pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic;
      if (!multilevel && !((pmy_pack->pmesh->two_d || clouds3d) &&
                           (pmy_pack->pmesh->strictly_periodic || shear3d || x3phys3d) &&
                           (global_variable::nranks == 1 || clouds3d) &&
                           drag_solver == DustDragSolver::local)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "refinable_cloud on a uniform mesh requires supported "
                  << "2-D/3-D cloud geometry and drag_solver=local; several ranks require "
                  << "3-D clouds"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      cloud_split_slack = pin->GetOrAddReal("dust", "cloud_split_slack", 0.02);
      cloud_gather_order = pin->GetOrAddInteger("dust", "cloud_gather_order", 2);
      cloud_transition = pin->GetOrAddInteger("dust", "cloud_transition",
                                              cloud_gather_order == 2 ? 2 : 1);
      if (cloud_transition < 0 || cloud_transition > 2 ||
          (cloud_transition == 2 && cloud_gather_order != 2)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_transition must be 0, 1 or 2, and 2 "
                  << "requires cloud_gather_order=2" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // The Kokkos P1 split on every backend (Codex review; it runs on CPUs
      // too); the host split needs the finite fine-grid table (few levels, one rank) and stays a
      // reference, selected explicitly
      cloud_split_device = pin->GetOrAddBoolean("dust", "cloud_split_device", true);
      // Test switch: false keeps every cloud at its level whatever the mesh
      cloud_split_enable = pin->GetOrAddBoolean("dust", "cloud_split_enable", true);
      // G (quantum merge under the pass budget): on in default and careful_merge (a full AMR
      // needs a coarsening channel), off in split_only
      cloud_merge = preset_bool("cloud_merge", !split_only);
      cloud_merge_buffer = pin->GetOrAddInteger("dust", "cloud_merge_buffer", 2);
      // Rixin: production alpha
      // 0.2 in 3-D (0.05 in 2-D); alpha H / tau lets the merged cloud's
      // velocity error move its mass by at most alpha cells within one damping time
      cloud_merge_alpha = pin->GetOrAddReal("dust", "cloud_merge_alpha",
                                            pmy_pack->pmesh->three_d ? 0.2 : 0.05);
      // The 3-D G velocity tolerance on the cell size of the leaf
      // that holds the merge cell (alpha 2^gap H / tau when the leaf is gap levels coarser);
      // Rixin: on by default
      cloud_merge_alpha_leaf = pin->GetOrAddBoolean("dust", "cloud_merge_alpha_leaf", true);
      cloud_merge_interval = pin->GetOrAddInteger("dust", "cloud_merge_interval", 10);
      cloud_merge_min_mass = pin->GetOrAddReal("dust", "cloud_merge_min_mass", 0.5);
      cloud_merge_nc = pin->GetOrAddBoolean("dust", "cloud_merge_nc", true);
      cloud_merge_lobe_wait = pin->GetOrAddInteger("dust", "cloud_merge_lobe_wait", 0);
      // Quantum G and several ranks exist only in the Kokkos merge
      cloud_merge_device = pin->GetOrAddBoolean("dust", "cloud_merge_device", true);
      // Merge schedule in physical time, burst after derefinement
      {
        Real om = 0.0;
        if (pin->DoesBlockExist("shearing_box") && pin->DoesParameterExist("shearing_box", "omega0")) {
          om = pin->GetReal("shearing_box", "omega0");
        }
        // Rixin: physical time (every cloud_merge_dt_omega = 0.25
        // / omega0) whenever omega0 is given, cycles otherwise
        const std::string ms = pin->GetOrAddString("dust", "cloud_merge_schedule",
                                                   (om > 0.0) ? "time" : "cycle");
        auto fatal = [](const std::string &msg) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << msg << std::endl;
          std::exit(EXIT_FAILURE);
        };
        if (ms == "time") {
          cloud_merge_by_time = true;
          if (!(om > 0.0)) fatal("cloud_merge_schedule = time needs <shearing_box>/omega0 > 0");
          cloud_merge_dt = pin->GetOrAddReal("dust", "cloud_merge_dt_omega", 0.25)/om;
          if (!(cloud_merge_dt > 0.0)) fatal("cloud_merge_dt_omega must be > 0");
          cloud_merge_t0 = pin->GetOrAddReal("dust", "cloud_merge_t0", 0.0);
        } else if (ms != "cycle") {
          fatal("cloud_merge_schedule must be cycle or time");
        }
        const Real bdt = pin->GetOrAddReal("dust", "cloud_merge_burst_dt_omega", 0.0);
        if (bdt > 0.0) {
          if (!(om > 0.0)) fatal("cloud_merge_burst_dt_omega needs <shearing_box>/omega0 > 0");
          cloud_merge_burst_dt = bdt/om;
        }
        cloud_merge_burst_interval = pin->GetOrAddInteger("dust", "cloud_merge_burst_interval", 50);
        if (cloud_merge_burst_interval < 1) fatal("cloud_merge_burst_interval must be >= 1");
      }
      {
        // Quantum G by default; tent stays a research control
        std::string mode = pin->GetOrAddString("dust", "cloud_merge_mode", "quantum");
        if (mode == "tent") cloud_merge_mode = 0;
        else if (mode == "quantum") cloud_merge_mode = 1;
        else {
          std::cout << "### FATAL ERROR: <dust>/cloud_merge_mode must be tent or quantum" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
      // stage X (the exact stencil re-merge before G in quantum mode): careful_merge only
      cloud_merge_stencil = preset_bool("cloud_merge_stencil", careful);
      // Rixin: no X in a 3-D shearing box: the background shear pulls the
      // members of a P1 family apart in x2 (q Omega l per unit time), so no family is ever complete
      // again.  G with its budget (beta, kappa) and R with the cold gate stay available there.
      if (shear_x1 && pmy_pack->pmesh->three_d && cloud_merge_stencil) {
        if (pin->DoesParameterExist("dust", "cloud_merge_stencil")) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << "<dust>/cloud_merge_stencil (stage X) is off in a 3-D shearing box: the "
                    << "shear separates the members of a family" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        cloud_merge_stencil = false;
        if (global_variable::my_rank == 0) {
          std::cout << "# NOTE (dust): refinable_mode = careful_merge in the 3-D shearing box: "
                    << "stage X off (the shear separates the members of a family); "
                    << "the cold gate of R and kappa stay" << std::endl;
        }
      }
      cloud_merge_stencil_tol = pin->GetOrAddReal("dust", "cloud_merge_stencil_tol", 0.02);
      cloud_merge_smax = pin->GetOrAddReal("dust", "cloud_merge_smax", 1.0);
      cloud_merge_K = pin->GetOrAddInteger("dust", "cloud_merge_K", 4);
      // Merged clouds keep their members' common ts (default); false reproduces the
      // runs before 2026-10-16, where ts was mass-averaged with the velocities (last-bit drift)
      cloud_merge_exact_ts = pin->GetOrAddBoolean("dust", "cloud_merge_exact_ts", true);
      cloud_merge_g_lag = pin->GetOrAddInteger("dust", "cloud_merge_g_lag", 2);
      cloud_merge_xprotect = pin->GetOrAddBoolean("dust", "cloud_merge_xprotect", false);
      // the mass floor of the pass budget (named after the absorption budget)
      cloud_absorb_floor = pin->GetOrAddReal("dust", "cloud_absorb_floor", 0.1);
      // Rixin: the production pass budget beta = 0.3 (was -1,
      // no budget); cloud_merge_budget = -1 switches it off
      cloud_merge_budget = pin->GetOrAddReal("dust", "cloud_merge_budget", 0.3);
      // The structure-relative budget (careful_merge: 1; 0 = off)
      cloud_merge_budget_kappa = preset_real("cloud_merge_budget_kappa", careful ? 1.0 : 0.0);
      cloud_merge_budget_evals = pin->GetOrAddInteger("dust", "cloud_merge_budget_evals", 4);
      cloud_merge_budget_thin = pin->GetOrAddBoolean("dust", "cloud_merge_budget_thin", false);
      cloud_merge_budget_x = pin->GetOrAddBoolean("dust", "cloud_merge_budget_x", false);
      // R (resampling) on in default and careful_merge, with the production clock below; off in
      // split_only
      cloud_resample = preset_bool("cloud_resample", !split_only);
      // <dust>/cloud_resample_schedule = cycle (default: every
      // cloud_resample_interval cycles, as before) or time (event k at t0 + k dt, dt from
      // cloud_resample_dt in code time or cloud_resample_dt_omega in units of 1/omega0).  Only
      // the chosen mode triggers; keys of the other mode are ignored with a warning (the input
      // kept in a restart file still carries them when a run switches modes at a restart).
      {
        auto fatal = [](const std::string &msg) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << msg << std::endl;
          std::exit(EXIT_FAILURE);
        };
        const bool g_int = pin->DoesParameterExist("dust", "cloud_resample_interval");
        const bool g_dt = pin->DoesParameterExist("dust", "cloud_resample_dt");
        bool g_dto = pin->DoesParameterExist("dust", "cloud_resample_dt_omega");
        const bool g_t0 = pin->DoesParameterExist("dust", "cloud_resample_t0");
        const bool g_rst = pin->DoesParameterExist("dust", "cloud_resample_schedule_reset");
        if (pin->DoesBlockExist("shearing_box") &&
            pin->DoesParameterExist("shearing_box", "omega0")) {
          cloud_resample_omega = pin->GetReal("shearing_box", "omega0");
        }
        // Rixin: physical time whenever omega0 is given (every
        // cloud_resample_dt_omega = 1 / omega0 unless cloud_resample_dt is given), cycles otherwise
        const std::string sched = pin->GetOrAddString("dust", "cloud_resample_schedule",
                                                      (cloud_resample_omega > 0.0) ? "time" : "cycle");
        // Rixin: every 0.25 / omega0 (was 1), with no resampling after the
        // regrids: G and R on their schedules only
        if (sched == "time" && !g_dt && !g_dto && cloud_resample_omega > 0.0) {
          pin->GetOrAddReal("dust", "cloud_resample_dt_omega", 0.25);
          g_dto = true;
        }
        auto ignored = [](const std::string &keys, const std::string &mode) {
          if (global_variable::my_rank == 0) {
            std::cout << "# WARNING: " << keys << " ignored: cloud_resample_schedule = " << mode
                      << std::endl;
          }
        };
        if (sched == "time") {
          cloud_resample_by_time = true;
          if (g_int) ignored("cloud_resample_interval", "time");
          if (g_dt == g_dto) {
            fatal("cloud_resample_schedule = time needs exactly one of cloud_resample_dt and "
                  "cloud_resample_dt_omega");
          }
          if (g_dto) {
            if (!(cloud_resample_omega > 0.0)) {
              fatal("cloud_resample_dt_omega needs <shearing_box>/omega0 > 0");
            }
            cloud_resample_dt = pin->GetReal("dust", "cloud_resample_dt_omega")/cloud_resample_omega;
          } else {
            cloud_resample_dt = pin->GetReal("dust", "cloud_resample_dt");
          }
          if (!(cloud_resample_dt > 0.0)) fatal("the resampling interval must be > 0");
          cloud_resample_t0 = pin->GetOrAddReal("dust", "cloud_resample_t0", 0.0);
          cloud_resample_schedule_reset =
              pin->GetOrAddBoolean("dust", "cloud_resample_schedule_reset", false);
        } else if (sched == "cycle") {
          if (g_dt || g_dto || g_t0 || g_rst) {
            ignored("cloud_resample_dt, cloud_resample_dt_omega, cloud_resample_t0 and "
                    "cloud_resample_schedule_reset", "cycle");
          }
        } else {
          fatal("cloud_resample_schedule must be cycle or time");
        }
      }
      if (!cloud_resample_by_time) {
        cloud_resample_interval = pin->GetOrAddInteger("dust", "cloud_resample_interval", 625);
      }
      // Rixin: the production cap factor (was 2.0)
      cloud_resample_eta = pin->GetOrAddReal("dust", "cloud_resample_eta", 1.5);
      // Resampling leaves cold patches alone (careful_merge: 0.05; 0 = off)
      cloud_resample_cold_alpha = preset_real("cloud_resample_cold_alpha", careful ? 0.05 : 0.0);
      cloud_resample_patch = pin->GetOrAddInteger("dust", "cloud_resample_patch", 2);
      // Patches inside one MeshBlock of min(cloud_resample_patch 2^level,
      // cloud_resample_patch_cells)^2 of its cells
      cloud_resample_patch_cells = pin->GetOrAddInteger("dust", "cloud_resample_patch_cells", 4);   // cap
      // A block patch, or a quadrant of one, holding more than this many pool clouds
      // (of one species) is split into its four quadrants, recursively down to single cells;
      // 0 = never.  Only for block patches whose widths min(cloud_resample_patch 2^level,
      // cloud_resample_patch_cells) are powers of two (Codex review: the halving loses the
      // odd row and column otherwise) and at most 16.  Rixin: default 1024 where allowed;
      // 0 reproduces the results before 6f9512a9
      auto pow2 = [](const int n) {return n >= 1 && (n & (n - 1)) == 0;};
      const bool split_ok = cloud_resample_patch_cells <= 16 &&
                            pow2(cloud_resample_patch) && pow2(cloud_resample_patch_cells);
      cloud_resample_patch_split = pin->GetOrAddInteger("dust", "cloud_resample_patch_split",
                                                        split_ok ? 1024 : 0);
      cloud_resample_threads = pin->GetOrAddInteger("dust", "cloud_resample_threads", 1);
      // For 3-D refinable clouds only (2-D ignores it): the device solve leaves out the
      // patches of n clouds on nl leaves with n <= 7 nl + 9 (the nominal rows: no null vector is
      // certain there).  false: every patch is solved (the reference path)
      cloud_resample_rank_screen = pin->GetOrAddBoolean("dust", "cloud_resample_rank_screen", true);
      // Rixin: the per-patch solves on the device by default (same patches and algorithm;
      // false keeps the host solves)
      // Rixin: the default follows the build: the device solver when the
      // default execution space is not a host space (GPU builds), the threaded host solver
      // otherwise (on a CPU build the host solver was 3.3x faster than the device code)
      // (3-D refinable clouds have only the device solver, which also runs on a CPU)
      cloud_resample_device = pin->GetOrAddBoolean("dust", "cloud_resample_device",
          !std::is_same<DevExeSpace, Kokkos::DefaultHostExecutionSpace>::value || clouds3d);
      cloud_resample_device_mem_mb = pin->GetOrAddInteger("dust", "cloud_resample_device_mem_mb", 2048);
      // A middle bucket (65 .. cloud_resample_device_mid clouds; 0: off) of the
      // device solves with narrower teams, several per SM
      cloud_resample_device_mid = pin->GetOrAddInteger("dust", "cloud_resample_device_mid", 0);
      cloud_resample_device_mid_team = pin->GetOrAddInteger("dust", "cloud_resample_device_mid_team", 128);
      // Default 1024 since the patch split (GB200, three jobs: the resampling 10-11% and
      // the pilot segment about 3% faster than with 512; before the split 1024 was 3% slower)
      cloud_resample_device_wide_team = pin->GetOrAddInteger("dust", "cloud_resample_device_wide_team", 1024);
      // The device QR update shares the rows among the threads as well.  Its sums run
      // in another order than the host's, so the device solve agrees with the host solve only to
      // round-off (the kept clouds can differ; both satisfy the constraints to round-off); false:
      // one thread per column in the host's order (bitwise comparable).  Default true since GB200
      // (QR -35%, resampling -10%, no fallback)
      cloud_resample_qr_rows = pin->GetOrAddBoolean("dust", "cloud_resample_qr_rows", true);
      // Default <256, 2> (at most 128 registers): on GB200 the two production
      // cloud kernels are 8-16% faster than without a bound (BA 128/256 and the dynamic mesh).
      // Rixin: a compile-time choice (DUST_CLOUD_LAUNCH_BOUNDS, dust.hpp);
      // an input or restart may only ask for the compiled value
      if (pin->GetOrAddInteger("dust", "cloud_launch_bounds", DUST_CLOUD_LAUNCH_BOUNDS) !=
          DUST_CLOUD_LAUNCH_BOUNDS) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "<dust>/cloud_launch_bounds = " << pin->GetString("dust", "cloud_launch_bounds")
                  << ", but this binary was built with DUST_CLOUD_LAUNCH_BOUNDS = "
                  << DUST_CLOUD_LAUNCH_BOUNDS << " (the launch bound of the cloud kernels is a "
                  << "compile-time choice since 2026-10-20: rebuild with "
                  << "-DDUST_CLOUD_LAUNCH_BOUNDS=<n> in CMAKE_CXX_FLAGS, or drop the key)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // Rixin: the column-by-column device QR (qr_rows = false) exists only
      // in the reference build (-DDUST_R_QR_COLUMNS=1, dust_resample.hpp); without it a run whose
      // device R solver would take that path stops instead of silently running the row update
      if (!cloud_resample_qr_rows && !DUST_R_QR_COLUMNS && cloud_resample &&
          cloud_resample_device) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "<dust>/cloud_resample_qr_rows = false (the column-by-column QR of the device "
                  << "resampling solve) is not compiled in this binary (DUST_R_QR_COLUMNS = 0): "
                  << "rebuild with -DDUST_R_QR_COLUMNS=1 in CMAKE_CXX_FLAGS (the reference "
                  << "build), or use the row-parallel update (true, the default)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // The order in which a patch takes its candidates: array (default, as before)
      // or tag (PTAG order, independent of the storage order)
      {
        const std::string ro = pin->GetOrAddString("dust", "cloud_resample_order", "array");
        if (ro != "array" && ro != "tag" && ro != "position") {
          std::cout << "### FATAL ERROR: <dust>/cloud_resample_order must be array, tag or position" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        cloud_resample_tag_order = (ro != "array");   // reordered pool
        cloud_resample_pos_order = (ro == "position");
      }
      cloud_resample_m0 = pin->GetOrAddReal("dust", "cloud_resample_m0", -1.0);
      if (cloud_resample_m0 > 0.0 && nspecies > 1 && global_variable::my_rank == 0) {
        std::cout << "# NOTE (dust): cloud_resample_m0 is the root cloud mass of every species "
                  << "(leave it unset for the per-species masses of the initial clouds)" << std::endl;
      }
      // Resampling after every regrid.  Rixin: default
      // false: no G or R after the regrids, the scheduled ones (G and R every 0.25 / omega0) cover
      // them; a regrid every ~40 cycles made the per-regrid R latency-bound
      cloud_resample_after_regrid = pin->GetOrAddBoolean("dust", "cloud_resample_after_regrid",
                                                         false);
      // The resampling after a regrid only on the blocks the regrid made and their
      // neighbours (the scheduled resampling stays global)
      // Rixin: default true (false reproduces the earlier runs).  The
      // scheduled resampling (every cloud_resample_dt_omega) stays global, also when a regrid event
      // falls on its cycle
      cloud_resample_event_local = pin->GetOrAddBoolean("dust", "cloud_resample_event_local", true);
      {
        // 2026-10-21 (scaling): regrid events handled together, cloud_event_dt_omega / omega0 after
        // the first regrid (0: every regrid at the next cycle)
        const Real edt = pin->GetOrAddReal("dust", "cloud_event_dt_omega", 0.0);
        if (edt > 0.0) {
          Real om = 0.0;
          if (pin->DoesBlockExist("shearing_box") && pin->DoesParameterExist("shearing_box", "omega0")) {
            om = pin->GetReal("shearing_box", "omega0");
          }
          if (!(om > 0.0)) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                      << "cloud_event_dt_omega needs <shearing_box>/omega0 > 0" << std::endl;
            std::exit(EXIT_FAILURE);
          }
          cloud_event_dt = edt/om;
        }
      }
      // Rixin: a regrid event that only deleted blocks runs G but no R,
      // unless the scheduled resampling falls on that cycle
      cloud_resample_after_derefine = pin->GetOrAddBoolean("dust",
                                         "cloud_resample_after_derefine", false);
      if (cloud_resample && ((!cloud_resample_by_time && cloud_resample_interval < 1) ||
                             cloud_resample_patch < 1 || cloud_resample_patch_split < 0 ||
                             (cloud_resample_patch_split > 0 && !split_ok) ||
                             cloud_gather_order != 2)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "cloud_resample requires cloud_resample_interval >= 1, cloud_resample_patch >= 1,"
                  << " cloud_resample_patch_split >= 0 (> 0 only with block patches whose"
                  << " cloud_resample_patch and cloud_resample_patch_cells <= 16 are powers of two),"
                  << " cloud_gather_order = 2" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // (2026-10-20: quantum is the default mode now, so its requirements apply only when G is on;
      // cloud_merge = false with the default mode is a run without G, not an error)
      if (cloud_merge && cloud_merge_mode == 1 &&
          (!cloud_merge_device || cloud_merge_K < 1 || cloud_merge_K > 8 ||
           cloud_merge_smax <= 0.0)) {
        std::cout << "### FATAL ERROR: <dust>/cloud_merge_mode = quantum needs "
                  << "cloud_merge_device, 1 <= cloud_merge_K <= 8 and cloud_merge_smax > 0"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_merge && (cloud_merge_buffer < 2 ||
                          cloud_merge_alpha <= 0.0 || cloud_merge_interval < 1)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_merge "
                  << "needs cloud_merge_buffer >= 2, "
                  << "cloud_merge_alpha > 0"
                  << " and cloud_merge_interval >= 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_transition == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "the P1 split requires cloud_transition=1 or 2 "
                  << "(the gather used for transition clouds)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_gather_order != 1 && cloud_gather_order != 2) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_gather_order must be 1 or 2" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      PrintRefinablePreset();
    }
    if (UsesCompositeTransfer() && deposit != DustDeposit::tsc) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/amr_pic = refinable_cloud requires <dust>/deposit = tsc"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // shearing box parameters (must match those read by the ShearingBox constructor)
  is_shearing_box = pin->DoesBlockExist("shearing_box");
  if (is_shearing_box) {
    qshear = pin->GetReal("shearing_box","qshear");
    omega0 = pin->GetReal("shearing_box","omega0");
    is_stratified = pin->GetOrAddBoolean("shearing_box","stratified",false);
  } else {
    qshear = 0.0;
    omega0 = 0.0;
    is_stratified = false;
  }
  // Point particles crossing a shear-periodic x1 face take the shear offset of the
  // time of their position (<dust>/shear_route_time, default true; 3-D, where the push advects y
  // with the background shear).  The 3-D clouds keep their own key (cloud_shear_route_time).
  if (is_shearing_box && pmy_pack->pmesh->three_d && !clouds3d &&
      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic) {
    shear_route_point = pin->GetOrAddBoolean("dust", "shear_route_time", true);
  }
  // constant azimuthal acceleration on the particles (with the gas's const_accel2 it keeps
  // a radially boosted 2-D r-z shearing-box equilibrium, see the streaming_linear pgen)
  const_accel_phi = pin->GetOrAddReal("dust", "const_accel_phi", 0.0);
  if (const_accel_phi != 0.0 &&
      !(is_shearing_box && !pmy_pack->pmesh->three_d && pc2_only)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<dust>/const_accel_phi requires a 2-D r-z shearing box and "
              << "coupling=pc2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (drag_solver != DustDragSolver::local && shear_x1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Coupled drag solvers require ordinary periodic boundaries; "
              << "the exact-transpose shear deposit is not implemented" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // (3) allocate deposited fields (with ghost zones) ------------------------------------
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int nmb = std::max((pmy_pack->nmb_thispack), (pmy_pack->pmesh->nmb_maxperrank));
  int ncells1 = indcs.nx1 + 2*(indcs.ng);
  int ncells2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*(indcs.ng)) : 1;
  int ncells3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*(indcs.ng)) : 1;
  Kokkos::realloc(qdep,  nmb, 5, ncells3, ncells2, ncells1);
  Kokkos::realloc(ustar, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(dmom,  nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(dmom, 0.0);  // read as R_g=0 in stage 2 if back_reaction is off
  if (drag_solver == DustDragSolver::pcg || drag_solver == DustDragSolver::adaptive) {
    Kokkos::realloc(solver_r, nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(solver_p, nmb, 3, ncells3, ncells2, ncells1);
  }
  // The old owner_overlap transfer also runs on a dynamic mesh (one rank)
  // In 3-D also on several ranks (the general deposit remap)
  if (pmy_pack->pmesh->adaptive && smr_transfer_mode != DustSMRTransfer::refinable_cloud &&
      !(smr_transfer_mode == DustSMRTransfer::owner_overlap &&
        (global_variable::nranks == 1 || pmy_pack->pmesh->three_d))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust on an adaptive mesh needs <dust>/amr_pic = refinable_cloud (or "
              << "owner_overlap: one rank in 2-D)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
    BuildRefinableCloudTransfer(pin);
    // the linear gather of the implicit stages reads u* in the ghost cells (its central
    // differences): with several levels the u* exchange restricts and prolongates through a
    // coarse buffer (PC2 forms its midpoint velocity from gas states that include ghosts)
    const bool implicit_stages = coupling == DustCoupling::imex ||
        (coupling == DustCoupling::hybrid && hybrid_force_mode != HybridForceMode::pc2);
    if (pmy_pack->pmesh->multilevel && implicit_stages && UstarGhosts()) {
      int nccells1 = indcs.cnx1 + 2*(indcs.ng);
      int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
      int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(cdummy, nmb, 3, nccells3, nccells2, nccells1);
    }
    CheckMultiRank();
    CheckClouds3D();   // the 3-D restrictions
  } else if (pmy_pack->pmesh->multilevel) {
    if (smr_transfer_mode == DustSMRTransfer::owner_overlap) {
      if (global_variable::my_rank == 0) {
        std::cout << "# DUST_PIC amr_pic = owner_overlap (point particles, TSC on the block that "
                  << "owns them, ghost deposits remapped to the leaves)" << std::endl;
      }
      int nccells1 = indcs.cnx1 + 2*(indcs.ng);
      int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
      int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(cdummy, nmb, 3, nccells3, nccells2, nccells1);
    }
  }
  if (drag_solver != DustDragSolver::local) {
    Kokkos::realloc(solver_ap, nmb, 3, ncells3, ncells2, ncells1);
  }

  // (4) allocate boundary communication objects -----------------------------------------
  pbval_qp = new MeshBoundaryValuesDep(pmy_pack, pin);
  pbval_qp->InitializeBuffers(5);
  pbval_dm = new MeshBoundaryValuesDep(pmy_pack, pin);
  pbval_dm->InitializeBuffers(3);
  pbval_us = new MeshBoundaryValuesCC(pmy_pack, pin, false);
  pbval_us->InitializeBuffers(3);
  if (drag_solver != DustDragSolver::local) {
    // These objects own distinct communicators/requests and are reinitialized for every
    // matrix-free matvec; they must not alias the one-shot stage exchanges above.
    pbval_solver_copy = new MeshBoundaryValuesCC(pmy_pack, pin, false);
    pbval_solver_copy->InitializeBuffers(3);
    pbval_solver_add = new MeshBoundaryValuesDep(pmy_pack, pin);
    pbval_solver_add->InitializeBuffers(3);
  }
  // shear-periodic remap of the u* radial ghost zones (3D shearing box only)
  if (shear_x1 && pmy_pack->pmesh->three_d) {
    psbox_us = new ShearingBoxCC(pmy_pack, pin, 3);
    // Collected u* ghost zones (S2c): the u* prolongation stays out of the slots
    // across the shear-periodic faces, as for the gas
    if (psbox_us->face_levels) pbval_us->skip_shear_x1_prol = true;
    // With refinement also the coarse u* array (cdummy), before its prolongation
    if (pmy_pack->pmesh->multilevel) {
      psbox_us->coarse_on = true;
      psbox_us->AllocateBuffers();
    }
  }
  // S3a: in 3-D with refinement or with shear-periodic x1 faces the ghost
  // deposits go through the general remap (any levels, several ranks, the shear offset)
  // Not for 3-D refinable clouds: they deposit into the leaves directly
  if (pmy_pack->pmesh->three_d && !clouds3d &&
      (pmy_pack->pmesh->multilevel || shear_x1 || x3_phys)) {
    if (drag_solver != DustDragSolver::local ||
        !(smr_transfer_mode == DustSMRTransfer::owner_overlap ||
          smr_transfer_mode == DustSMRTransfer::none)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "3-D dust with refinement or a shearing box: drag_solver = local and "
                << "point particles (amr_pic = owner_overlap on a refined mesh)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const auto &id = pmy_pack->pmesh->mb_indcs;
    if (pmy_pack->pmesh->multilevel &&
        (id.nx1 < 2*id.ng || id.nx2 < 2*id.ng || id.nx3 < 2*id.ng)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "3-D dust with refinement: MeshBlocks need at least 2*nghost cells in each "
                << "direction (the ghost zone of a coarser block reaches 2*nghost finer cells)"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // Ghost deposits beyond physical x3 faces: fold back (default) or drop
    int x3mode = 0;
    if (x3_phys) {
      const std::string xd = pin->GetOrAddString("dust", "x3_ghost_deposit", "fold");
      if (xd != "fold" && xd != "drop") {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "<dust>/x3_ghost_deposit must be fold or drop" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      x3mode = (xd == "fold") ? 1 : 2;
    }
    dep_plan = new DepRemapPlan(pmy_pack, shear_x1, qshear, omega0, x3mode);
    pbval_qp->SetGeneralPlan(dep_plan);
    pbval_dm->SetGeneralPlan(dep_plan);
    if (x3mode == 2) pbval_dm->dropped_sum.assign(3, 0.0);   // the record from the start
  }
  // owner_overlap on a dynamic mesh: the block boxes and the block table of the
  // initial mesh (the exchanges above built their remaps); 2-D only
  if (pmy_pack->pmesh->adaptive && smr_transfer_mode == DustSMRTransfer::owner_overlap &&
      !pmy_pack->pmesh->three_d) {
    BuildOwnerOverlapMesh(false);
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::PrintRefinablePreset
//! \brief The resolved parameter set of the refinable clouds (<dust>/refinable_mode), once on
//! rank 0: split, G (merge) and R (resampling) with their schedules, and the preset keys that
//! the input or the restart file gives explicitly.

void DustGasDrag::PrintRefinablePreset() const {
  if (global_variable::my_rank != 0) return;
  const char *mode = (refinable_mode == RefinableMode::careful_merge) ? "careful_merge" :
                     (refinable_mode == RefinableMode::split_only) ? "split_only" : "default";
  const Real om = cloud_resample_omega;   // <shearing_box>/omega0 (0 without a shearing box)
  auto sched = [om](bool by_time, Real dt, int interval) {
    std::ostringstream o;
    if (by_time) {
      o << "every " << dt*om << "/omega0";
    } else {
      o << "every " << interval << " cycles";
    }
    return o.str();
  };
  std::ostringstream g, r;
  g << "G " << (cloud_merge ? "on" : "off");
  if (cloud_merge) {
    g << " (" << (cloud_merge_mode == 1 ? "quantum" : "tent") << ", stage X "
      << (cloud_merge_stencil ? "on" : "off") << ", alpha " << cloud_merge_alpha;
    if (clouds3d) g << ", alpha_leaf " << (cloud_merge_alpha_leaf ? "on" : "off");
    g << ", budget beta " << cloud_merge_budget << " kappa " << cloud_merge_budget_kappa << ", "
      << sched(cloud_merge_by_time, cloud_merge_dt, cloud_merge_interval) << ")";
  }
  r << "R " << (cloud_resample ? "on" : "off");
  if (cloud_resample) {
    r << " (eta " << cloud_resample_eta << ", cold gate alpha " << cloud_resample_cold_alpha
      << ", " << sched(cloud_resample_by_time, cloud_resample_dt, cloud_resample_interval)
      << ", after regrids " << (cloud_resample_after_regrid ? "on" : "off") << ")";
  }
  std::cout << "# DUST_PIC amr_pic = refinable_cloud, refinable_mode = " << mode << std::endl
            << "#   split: moving P1 (" << (cloud_split_enable ? "on" : "off") << "), gather order "
            << cloud_gather_order << ", transition " << cloud_transition << "; " << g.str()
            << std::endl
            << "#   " << r.str() << std::endl;
  if (!refinable_given.empty()) {
    std::cout << "#   given explicitly:" << refinable_given << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckMultiRank
//! \brief With several MPI ranks, refinable clouds run only the production path
//! that the dust halo covers (PC2 linear gather, quantum G with or without the pass budget,
//! device or host R); every other option stops here until it has its exchange.

void DustGasDrag::CheckMultiRank() {
  if (global_variable::nranks == 1) return;
  std::vector<std::string> bad;
  // PC2, or imex2+ (2026-10-17: its deposit and gather-kick kernels use the halo); not the
  // automatic hybrid, whose split-BE preparation pass has no halo exchange yet
  if (!(coupling == DustCoupling::imex ||
        (coupling == DustCoupling::hybrid && hybrid_force_mode == HybridForceMode::pc2))) {
    bad.push_back("a coupling other than PC2 or IMEX (coupling = pc2 / imex, or "
                  "hybrid_force_mode = pc2)");
  }
  if (drag_solver != DustDragSolver::local) bad.push_back("drag_solver other than local");
  if (cloud_gather_order != 2 || analytic_gather.on) {
    bad.push_back("gather variants (cloud_gather_order != 2, analytic_gather)");
  }
  // Stage X and the structure-relative budget on several ranks (X merges
  // the families whose nine members are on this rank; the budget's neighbour leaves come from
  // the halo); X under the pass budget (cloud_merge_budget_x) has no exchange yet
  if (cloud_merge && (cloud_merge_mode != 1 || !cloud_merge_device)) {
    bad.push_back("merging other than the device quantum G (tent)");
  }
  if (cloud_merge && cloud_merge_stencil && cloud_merge_budget_x) {
    bad.push_back("stage X under the pass budget (cloud_merge_budget_x)");
  }
  if (!bad.empty()) {
    if (global_variable::my_rank == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Refinable clouds on " << global_variable::nranks << " MPI ranks "
                << "do not support yet:" << std::endl;
      for (const auto &b : bad) std::cout << "  - " << b << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
// destructor

DustGasDrag::~DustGasDrag() {
  if (global_variable::my_rank == 0 && coupling == DustCoupling::hybrid) {
    unsigned long long total = hybrid_pc2_cycles + hybrid_split_be_cycles;
    double pc2_fraction = (total > 0) ?
        static_cast<double>(hybrid_pc2_cycles)/static_cast<double>(total) : 0.0;
    std::cout << std::setprecision(14)
              << "# DUST_HYBRID_SUMMARY pc2_cycles=" << hybrid_pc2_cycles
              << " split_be_cycles=" << hybrid_split_be_cycles
              << " pc2_fraction=" << pc2_fraction;
    // With a fixed mode the switching metric is not computed (not a zero)
    if (NeedsHybridMetric()) {
      std::cout << " last_zeta=" << hybrid_last_zeta << " last_chi=" << hybrid_last_chi;
    } else {
      std::cout << " metric=not_computed";
    }
    std::cout << std::endl;
  }
  if (global_variable::my_rank == 0 && StiffnessActive()) {
    std::cout << std::setprecision(6) << "# DUST_STIFFNESS max_chi=" << stiff_chi_run_max
              << " last_chi=" << stiff_chi_last << " stiff_thresh=" << stiff_thresh
              << " pc2_feedback_damping=" << (pc2_feedback_damping ? 1 : 0) << std::endl;
  }
  if (global_variable::my_rank == 0 && (prof_count[kProfAmr] > 0 || prof_count[kProfRefineClouds] > 0)) {
    // Rank 0's fenced phase times (s) and calls
    const char *names[kProfN] = {"trigger", "amr_total", "regrid_mesh", "regrid_dust_rebuild", "refine_clouds",
                                 "split", "regrid_event_GR", "scheduled_R", "scheduled_G", "sort", "cell_order",
                                 "trigger_deposit", "trigger_copy", "trigger_blocks", "trigger_log"};
    std::cout << "# DUST_AMR_PROFILE";
    for (int k=0; k<kProfN; ++k) {
      std::cout << " " << names[k] << "=" << std::setprecision(6) << prof_time[k] << "(" << prof_count[k] << ")";
    }
    std::cout << std::endl;
  }
  // Codex review: printed once a profiled R has run (the patch count; the last
  // entry, the negative basic solutions, can stay 0)
  if (global_variable::my_rank == 0 && ProfOn() && !DUST_R_DIAG && cloud_resample && cloud_resample_device) {
    std::cout << "# DUST_R_STAGES not compiled (build with -DDUST_R_DIAG=1)" << std::endl;
  }
  if (global_variable::my_rank == 0 && resample::StageTotals()[resample::kStagePatches] > 0) {
    // Device clock cycles of the patch solves by stage, summed over the teams of
    // rank 0 (CUDA only; 0 on other backends)
    const unsigned long long *t = resample::StageTotals();
    std::cout << "# DUST_R_STAGES (rank 0, device clock cycles, summed over the teams) rows=" << t[0]
              << " qr=" << t[1] << " null=" << t[2] << " steps=" << t[3] << " stream=" << t[4]
              << " stream_rest=" << (t[4] - t[1] - t[2] - t[3]) << " check=" << t[5]
              << " patches=" << t[6] << std::endl;
    // The basic solution on the pivot columns
    std::cout << "# DUST_R_BASIS (QR blocks with a null space) blocks=" << t[7] << " feasible=" << t[8]
              << " cap_only=" << t[9] << " negative=" << t[10] << std::endl;
  }
  if (global_variable::my_rank == 0 &&
      smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
    std::cout << "# DUST_CLOUD_SUMMARY splits=" << cloud_split_count
              << " merges=" << cloud_merge_count
              << (cloud_resample_cold_alpha > 0.0 ? " resample_cold_skips=" +
                  std::to_string(cloud_resample_cold_skips) : std::string())
              << " lobe_splits=" << cloud_lobe_split_count
              << (cloud_merge ? " coarsen_events=" + std::to_string(cloud_coarsen_events) +
                  " coarsen_in=" + std::to_string(cloud_coarsen_in) +
                  " coarsen_reject=" + std::to_string(cloud_coarsen_reject) +
                  " far_releases=" + std::to_string(cloud_far_release) +
                  " dK_grad=" + std::to_string(cloud_coarsen_dk_grad) +
                  " dK_res=" + std::to_string(cloud_coarsen_dk_res) : std::string())
              // with several ranks the counters are rank 0's, the particles all ranks'
              << " particles=" << ((global_variable::nranks > 1) ? pmy_pack->pmesh->nprtcl_total
                                                                  : pmy_pack->ppart->nprtcl_thispack)
              << ((global_variable::nranks > 1) ? " counters=rank0" : "") << std::endl;
  }
  if (global_variable::my_rank == 0 && shear3d) {   // (Codex perf review item 1)
    std::cout << "# DUST_SHEAR_LISTS rank0 ysh_updates=" << shear3d_updates << " list_rebuilds="
              << shear3d_rebuilds << " cache=" << (shear3d_list_cache ? "on" : "off") << std::endl;
  }
  if (global_variable::my_rank == 0 && global_variable::nranks > 1 && ProfOn() &&
      smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
    // Rank 0's seconds in the multi-rank parts of the dust (with amr_profile)
    const double *t = halo_.tsec;
    std::cout << "# DUST_MPI_TIMING rank0 migrate_refine=" << mpi_time[0] << " (" << mpi_count[0]
              << ") migrate_push=" << mpi_time[1] << " (" << mpi_count[1] << ") kernels_before="
              << mpi_time[2] << " | halo forward (" << halo_.nexch[0] << "): kernels_before="
              << t[0] << " pack=" << t[1] << " mpi=" << t[2] << " unpack=" << t[3]
              << " | halo reverse (" << halo_.nexch[1] << "): kernels_before=" << t[4] << " pack="
              << t[5] << " mpi=" << t[6] << " add=" << t[7] << " | halo widenings " << halo_widenings_
              << ", halo blocks " << halo_.hslot1 - halo_.hslot0 << ", cells sent "
              << (halo_.soff.empty() ? 0 : halo_.soff.back()) << " received "
              << (halo_.roff.empty() ? 0 : halo_.roff.back())
              << (cloud_halo_strips ? " (strips)" : " (whole blocks)") << std::endl;
  }
  if (global_variable::my_rank == 0 && solver_stage_count > 0) {
    auto quantile = [&](double fraction) {
      if (solver_pcg_stage_count == 0) return 0;
      unsigned long long order = (fraction <= 0.0) ? 1 :
          static_cast<unsigned long long>(
              std::ceil(fraction*static_cast<double>(solver_pcg_stage_count)));
      unsigned long long cumulative = 0;
      for (std::size_t iteration=0; iteration<solver_pcg_iteration_hist.size();
           ++iteration) {
        cumulative += solver_pcg_iteration_hist[iteration];
        if (cumulative >= order) return static_cast<int>(iteration);
      }
      // The histogram and stage count are updated together; reaching this return would
      // indicate internal diagnostic corruption rather than a numerical solver failure.
      return drag_iter_max;
    };
    const char *name = (drag_solver == DustDragSolver::applya) ? "applya" :
                       (drag_solver == DustDragSolver::dc1) ? "dc1" :
                       (drag_solver == DustDragSolver::dc2) ? "dc2" :
                       (drag_solver == DustDragSolver::pcg) ? "pcg" :
                       (drag_solver == DustDragSolver::adaptive) ? "adaptive" : "local";
    std::cout << std::setprecision(14)
              << "# DUST_SOLVER_SUMMARY mode=" << name
              << " stages=" << solver_stage_count
              << " applya=" << solver_applya_count
              << " halos=" << solver_halo_count
              << " reductions=" << solver_reduction_count
              << " fast_accept=" << solver_fast_accept_count
              << " pcg_stages=" << solver_pcg_stage_count
              << " pcg_iter_min=" << quantile(0.0)
              << " pcg_iter_median=" << quantile(0.5)
              << " pcg_iter_p95=" << quantile(0.95)
              << " pcg_iter_max=" << quantile(1.0)
              << " solve_seconds=" << solver_wall_seconds << std::endl;
  }
  delete pbval_qp;
  delete pbval_dm;
  delete pbval_us;
  if (pbval_solver_copy != nullptr) delete pbval_solver_copy;
  if (pbval_solver_add != nullptr) delete pbval_solver_add;
  if (psbox_us != nullptr) {delete psbox_us;}
}

//----------------------------------------------------------------------------------------
//! \fn bool DustGasDrag::ActiveStage
//! \brief Returns false on stages where all dust work is dormant. For imex2+ the final
//! explicit stage is a trivial assembly step (beta=0, gam0=1, gam1=0, and the implicit
//! solve is skipped since estage == nexp_stages), so every dust task no-ops there.

bool DustGasDrag::ActiveStage(Driver *pdrive, int stage) const {
  if (coupling == DustCoupling::hybrid) return true;
  return !((pdrive->integrator.compare("imex2+") == 0) && (stage == pdrive->nexp_stages));
}

bool DustGasDrag::FirstMigrationActive(Driver *pdrive, int stage) const {
  if (coupling == DustCoupling::imex) return ActiveStage(pdrive, stage);
  return (hybrid_mode == HybridMode::pc2 && stage == 2);
}

bool DustGasDrag::SecondMigrationActive(Driver *pdrive, int stage) const {
  return (coupling == DustCoupling::hybrid &&
          hybrid_mode == HybridMode::split_be && stage == 2);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ShearRouteTime, ShiftShearCrossings
//! \brief AthenaK's particle routing (SetNewPrtclGID) moves a particle across a
//! shear-periodic x1 face by the shear offset of pm->time = t^n, but the push has already moved it
//! on.  PC2 and hybrid migrate once, after the full push: the move needs the offset of t^n + dt.
//! imex migrates after every moving stage; a particle wrapped at stage s keeps moving in the
//! following stages with the wrapped x, whose shear advection -q Omega x adds q Omega Lx w dt to its
//! y (w_k = gam0_k w_(k-1) + beta_k over those stages, gam0_k + gam1_k = 1, w_s = 0), so the wrap
//! itself needs the offset of t^n + (1 - w) dt (imex2+: t^n + dt/sqrt(2) at stage 1, t^n + dt at
//! stage 2).  ShiftShearCrossings adds q Omega Lx (tau - t^n) to the y (and the RK register) of the
//! particles the routing will move across an x1 face (the routing's own block-offset test).

Real DustGasDrag::ShearRouteTime(Driver *pdrive, int stage) const {
  const Real t = pmy_pack->pmesh->time, dt = pmy_pack->pmesh->dt;
  if (coupling != DustCoupling::imex) return t + dt;
  Real w = 0.0;
  for (int k = stage + 1; k <= pdrive->nexp_stages; ++k) {
    w = pdrive->gam0[k-1]*w + pdrive->beta[k-1];
  }
  return t + (1.0 - w)*dt;
}

void DustGasDrag::ShiftShearCrossings(const Real tau) {
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  const auto &ms = pmy_pack->pmesh->mesh_size;
  // the difference of the offsets taken unfolded (small), not of the folded fmod values
  const Real dsh = qshear*omega0*(ms.x1max - ms.x1min)*(tau - pmy_pack->pmesh->time);
  if (npart == 0 || dsh == 0.0) return;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int gids = pmy_pack->gids;
  const Real xa = ms.x1min, xb = ms.x1max;
  par_for("dust_shear_route_time", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    const int m = pi(PGID,p) - gids;
    const Real b0 = mbsize.d_view(m).x1min, b1 = mbsize.d_view(m).x1max, lx = b1 - b0;
    const int ix = static_cast<int>((pr(IPX,p) - b0 + lx)/lx) - 1;   // as SetNewPrtclGID
    const bool lo1 = (ix < 0) && (b0 < xa + 0.5*lx);
    const bool hi1 = (ix > 0) && (b1 > xb - 0.5*lx);
    if (lo1 || hi1) {
      const Real d = hi1 ? dsh : -dsh;
      pr(IPY,p) += d;
      pr(IPY1,p) += d;
    }
  });
}

Real DustGasDrag::DragStep(Driver *pdrive) const {
  if (coupling == DustCoupling::hybrid) return pmy_pack->pmesh->dt;
  return (pdrive->a_impl)*(pmy_pack->pmesh->dt);
}

//----------------------------------------------------------------------------------------
//! \fn void DustGasDrag::SetDefaultMasses
//! \brief Default per-particle species/stopping-time/mass initialization: species are
//! striped over the particle index, and masses are normalized so the total dust mass is
//! dust_to_gas times the gas mass for a uniform gas of density <problem>/rho0.
//! Problem generators may overwrite PSP/IPTS/IPM after calling this.

void DustGasDrag::SetDefaultMasses(ParameterInput *pin) {
  particles::Particles *ppar = pmy_pack->ppart;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) return;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  auto &mbsize = pmy_pack->pmb->mb_size;
  bool three_d = pmy_pack->pmesh->three_d;
  Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  Real ppc = static_cast<Real>(RootPpc(pin));   // with the new keys the sum of ppc_N

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &taus_ = taus;
  auto gids = pmy_pack->gids;
  int nspec = nspecies;
  Real d2g = dust_to_gas;
  // With the new keys, every run of ppc_total consecutive particles of a block holds
  // ppc_N of species N (interleaved for equal ppc_N, else in blocks), mass
  // epsilon_N rho0 V_cell/ppc_N; the legacy striping otherwise
  const SpeciesTable &st = Species(pin);
  if (st.new_init) {
    if (st.abundance != "epsilon") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "the default dust masses need epsilon_N (Z_N needs a stratified problem "
                << "generator)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const int ptot = st.ppc_total;
    DualArray1D<int> slot_s("dust_slot_species", ptot);
    DualArray1D<Real> mfac("dust_species_mfac", nspec);
    for (int k=0; k<ptot; ++k) {int sp, kk; st.SlotSpecies(k, sp, kk); slot_s.h_view(k) = sp;}
    for (int sp=0; sp<nspec; ++sp) {
      mfac.h_view(sp) = (st.ppc[sp] > 0) ? st.value[sp]*rho0/static_cast<Real>(st.ppc[sp]) : 0.0;
    }
    slot_s.template modify<HostMemSpace>(); slot_s.template sync<DevExeSpace>();
    mfac.template modify<HostMemSpace>(); mfac.template sync<DevExeSpace>();
    auto ss = slot_s.d_view;
    auto mf = mfac.d_view;
    const int nmb = pmy_pack->nmb_thispack;
    const int npmb = npart/std::max(1, nmb);
    par_for("dust_masses_species",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID,p) - gids;
      Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
      if (three_d) {vol *= mbsize.d_view(m).dx3;}
      const int q = (npmb > 0) ? p % npmb : p;
      const int s = ss(q % ptot);
      pi(PSP,p) = s;
      pr(IPTS,p) = taus_.d_view(s);
      pr(IPM,p) = mf(s)*vol;
    });
    return;
  }
  par_for("dust_masses",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    int s = p % nspec;
    pi(PSP,p) = s;
    pr(IPTS,p) = taus_.d_view(s);
    pr(IPM,p) = d2g*rho0*vol/ppc;
  });
}

} // namespace dust
