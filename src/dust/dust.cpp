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
  if (pmy_pack->pmesh->multilevel) {
    bool experimental_smr = pin->GetOrAddBoolean(
        "dust", "experimental_static_smr", false);
    if (!(experimental_smr && !(pmy_pack->pmesh->adaptive) &&
          pmy_pack->pmesh->two_d && pmy_pack->pmesh->strictly_periodic &&
          global_variable::nranks == 1)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust drag does not support SMR/AMR. The current "
                << "experimental exception requires explicit opt-in, static 2-D SMR, "
                << "ordinary periodic boundaries, and one MPI rank" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  bool shear_x1 = (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] ==
                   BoundaryFlag::shear_periodic);
  if (!(pmy_pack->pmesh->strictly_periodic)) {
    // only exception: 3D shearing box with shear-periodic x1 (and periodic x2/x3)
    if (!(shear_x1 && pmy_pack->pmesh->three_d)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust drag requires periodic boundaries in all "
                << "directions (or shear-periodic x1 in 3D)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust): shear-periodic x1 boundaries active. The "
                << "conservative azimuthal remap of ghost DEPOSITS across the radial "
                << "boundaries is not yet implemented (deposits are exchanged with the "
                << "plain-periodic pattern, exact only for azimuthally uniform states "
                << "such as the NSH drift equilibrium). The u* ghost fill IS remapped."
                << std::endl;
    }
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
  dust_to_gas   = pin->GetOrAddReal("dust","dust_to_gas",0.01);

  hybrid_enter_zeta = pin->GetOrAddReal("dust", "hybrid_enter_zeta", 0.5);
  hybrid_enter_chi  = pin->GetOrAddReal("dust", "hybrid_enter_chi", 0.5);
  hybrid_exit_zeta  = pin->GetOrAddReal("dust", "hybrid_exit_zeta", 0.25);
  hybrid_exit_chi   = pin->GetOrAddReal("dust", "hybrid_exit_chi", 0.25);
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
              << std::endl << "Experimental static-SMR dust currently supports only "
              << "drag_solver=local; coupled solvers require a certified adjoint "
              << "composite transfer" << std::endl;
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
  std::string fail_policy = pin->GetOrAddString("dust", "drag_fail_policy", "abort");
  if (drag_rtol <= 0.0 || drag_atol < 0.0 || drag_iter_max < 1 ||
      drag_diagnostic_interval < 0 || adaptive_order_c <= 0.0 ||
      adaptive_rtol_max <= 0.0 || adaptive_tref <= 0.0 ||
      adaptive_state_floor <= 0.0 || fail_policy.compare("abort") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid coupled drag-solver tolerance/iteration policy; "
              << "only drag_fail_policy=abort is supported" << std::endl;
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

  // per-species stopping times: <dust>/nspecies and taus_1, taus_2, ...
  nspecies = pin->GetOrAddInteger("dust","nspecies",1);
  if (nspecies < 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/nspecies must be at least 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  taus = DualArray1D<Real>("dust_taus", nspecies);
  taus_max = 0.0;
  taus_min = std::numeric_limits<Real>::max();
  for (int s=0; s<nspecies; ++s) {
    Real ts = pin->GetReal("dust", "taus_" + std::to_string(s+1));
    if (ts <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust stopping times must be positive" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    taus.h_view(s) = ts;
    taus_max = std::max(taus_max, ts);
    taus_min = std::min(taus_min, ts);
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
    // The refinable-cloud path may also be selected on a uniform mesh: its clouds are
    // then exact TSC and the gather/deposit options apply unchanged, which gives uniform
    // reference runs with the same coupling as the mixed-resolution runs.
    bool multilevel = pmy_pack->pmesh->multilevel;
    std::string transfer = pin->GetOrAddString(
        "dust", "experimental_smr_transfer", multilevel ? "smooth_p2" : "none");
    if (transfer.compare("none") == 0 && !multilevel) {
      smr_transfer_mode = DustSMRTransfer::none;
    } else if (transfer.compare("owner_overlap") == 0 && multilevel) {
      smr_transfer_mode = DustSMRTransfer::owner_overlap;
    } else if (transfer.compare("smooth_p2") == 0 && multilevel) {
      smr_transfer_mode = DustSMRTransfer::smooth_p2;
    } else if (transfer.compare("refinable_cloud") == 0) {
      smr_transfer_mode = DustSMRTransfer::refinable_cloud;
      if (!multilevel && !(pmy_pack->pmesh->two_d && pmy_pack->pmesh->strictly_periodic &&
                           global_variable::nranks == 1 &&
                           drag_solver == DustDragSolver::local)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "refinable_cloud on a uniform mesh requires a 2-D, "
                  << "strictly periodic, single-rank mesh and drag_solver=local"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      cloud_split_slack = pin->GetOrAddReal("dust", "cloud_split_slack", 0.02);
      cloud_merge_tolerance = pin->GetOrAddReal("dust", "cloud_merge_tolerance", 0.05);
      cloud_deform = pin->GetOrAddBoolean("dust", "cloud_deform", false);
      cloud_gather_order = pin->GetOrAddInteger("dust", "cloud_gather_order",
                                                cloud_deform ? 2 : 1);
      cloud_force_local = pin->GetOrAddBoolean("dust", "cloud_force_local", false);
      cloud_force_children = pin->GetOrAddBoolean("dust", "cloud_force_children", false);
      cloud_gather_level = pin->GetOrAddInteger("dust", "cloud_gather_level", -1);
      cloud_transition = pin->GetOrAddInteger("dust", "cloud_transition", 0);
      if (cloud_transition < 0 || cloud_transition > 2 ||
          (cloud_transition > 0 && (cloud_deform || cloud_force_local ||
                                    cloud_force_children || cloud_gather_level >= 0)) ||
          (cloud_transition == 2 && cloud_gather_order != 2)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_transition must be 0, 1 or 2; it excludes "
                  << "deform/force_local/force_children/gather_level, and 2 requires "
                  << "cloud_gather_order=2" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      cloud_p1_split = pin->GetOrAddBoolean("dust", "cloud_p1_split", false);
      cloud_p1_release_exit = pin->GetOrAddBoolean("dust", "cloud_p1_release_exit", false);
      cloud_split_log = pin->GetOrAddBoolean("dust", "cloud_split_log", false);
      cloud_split_device = pin->GetOrAddBoolean("dust", "cloud_split_device", false);
      cloud_split_log_leaves = pin->GetOrAddBoolean("dust", "cloud_split_log_leaves", false);
      cloud_split_interp = pin->GetOrAddBoolean("dust", "cloud_split_interp", false);
      cloud_merge = pin->GetOrAddBoolean("dust", "cloud_merge", false);
      cloud_merge_buffer = pin->GetOrAddInteger("dust", "cloud_merge_buffer", 2);
      cloud_merge_alpha = pin->GetOrAddReal("dust", "cloud_merge_alpha", 0.2);
      cloud_merge_interval = pin->GetOrAddInteger("dust", "cloud_merge_interval", 10);
      cloud_merge_min_mass = pin->GetOrAddReal("dust", "cloud_merge_min_mass", 0.5);
      cloud_merge_nc = pin->GetOrAddBoolean("dust", "cloud_merge_nc", true);
      cloud_merge_lobe_wait = pin->GetOrAddInteger("dust", "cloud_merge_lobe_wait", 0);
      cloud_merge_absorb = pin->GetOrAddBoolean("dust", "cloud_merge_absorb", false);
      cloud_merge_pending_log = pin->GetOrAddBoolean("dust", "cloud_merge_pending_log", false);
      cloud_merge_d3 = pin->GetOrAddBoolean("dust", "cloud_merge_d3", false);
      cloud_merge_d3_iter = pin->GetOrAddInteger("dust", "cloud_merge_d3_iter", 30);
      cloud_merge_device = pin->GetOrAddBoolean("dust", "cloud_merge_device", false);
      cloud_merge_tstop = pin->GetOrAddReal("dust", "cloud_merge_tstop", -1.0);
      {
        std::string mode = pin->GetOrAddString("dust", "cloud_merge_mode", "tent");
        if (mode == "tent") cloud_merge_mode = 0;
        else if (mode == "quantum") cloud_merge_mode = 1;
        else {
          std::cout << "### FATAL ERROR: <dust>/cloud_merge_mode must be tent or quantum" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
      cloud_merge_stencil = pin->GetOrAddBoolean("dust", "cloud_merge_stencil", true);
      cloud_merge_stencil_tol = pin->GetOrAddReal("dust", "cloud_merge_stencil_tol", 0.02);
      cloud_merge_smax = pin->GetOrAddReal("dust", "cloud_merge_smax", 1.0);
      cloud_merge_K = pin->GetOrAddInteger("dust", "cloud_merge_K", 4);
      cloud_merge_g_lag = pin->GetOrAddInteger("dust", "cloud_merge_g_lag", 2);
      cloud_audit_t0 = pin->GetOrAddReal("dust", "cloud_audit_t0", -1.0);
      cloud_audit_t1 = pin->GetOrAddReal("dust", "cloud_audit_t1", -1.0);
      cloud_audit_compact = pin->GetOrAddBoolean("dust", "cloud_audit_compact", true);
      cloud_merge_xprotect = pin->GetOrAddBoolean("dust", "cloud_merge_xprotect", false);
      cloud_absorb_budget = pin->GetOrAddReal("dust", "cloud_absorb_budget", -1.0);
      cloud_absorb_floor = pin->GetOrAddReal("dust", "cloud_absorb_floor", 0.1);
      cloud_compress = pin->GetOrAddBoolean("dust", "cloud_compress", false);
      cloud_compress_smax = pin->GetOrAddReal("dust", "cloud_compress_smax", 0.5);
      cloud_compress_ds = pin->GetOrAddReal("dust", "cloud_compress_ds", 0.1);
      cloud_compress_buffer = pin->GetOrAddInteger("dust", "cloud_compress_buffer", 4);
      if ((cloud_merge_mode == 1 || cloud_compress) &&
          (!cloud_merge || !cloud_merge_device || cloud_merge_mode != 1 || cloud_merge_K < 1 ||
           cloud_merge_K > 8 || cloud_merge_smax <= 0.0 || cloud_compress_smax <= 0.0 ||
           cloud_compress_buffer < 1 || cloud_merge_d3)) {
        std::cout << "### FATAL ERROR: <dust>/cloud_merge_mode = quantum (and cloud_compress) "
                  << "need cloud_merge, cloud_merge_device, 1 <= cloud_merge_K <= 8, positive "
                  << "extents, cloud_compress_buffer >= 1 and no cloud_merge_d3" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_merge && (!cloud_p1_split || cloud_split_interp || cloud_deform ||
                          cloud_merge_buffer < 2 ||
                          cloud_merge_alpha <= 0.0 || cloud_merge_interval < 1)) {
        // lattice-Ai looks nodes up on the root lattice, which merged centroids leave;
        // deformable clouds carry gradients that a merge would have to combine
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_merge needs cloud_p1_split, excludes "
                  << "cloud_split_interp and cloud_deform, and needs cloud_merge_buffer >= 2, "
                  << "cloud_merge_alpha > 0"
                  << " and cloud_merge_interval >= 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      cloud_presplit_x1 = pin->GetOrAddBoolean("dust", "cloud_presplit_x1", false);
      if (cloud_presplit_x1 && !cloud_p1_split) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_presplit_x1 (diagnostic) requires "
                  << "cloud_p1_split" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_split_interp && (!cloud_p1_split || global_variable::nranks > 1)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_split_interp requires cloud_p1_split and a "
                  << "single rank" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_split_log_leaves && !cloud_split_log) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_split_log_leaves requires cloud_split_log"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_split_log && (!cloud_p1_split || global_variable::nranks > 1)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_split_log (diagnostic) requires "
                  << "cloud_p1_split and a single rank" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_p1_split && cloud_transition == 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_p1_split requires cloud_transition=1 or 2 "
                  << "(the gather used for transition clouds)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_gather_level >= 0 && (cloud_deform || cloud_force_local ||
                                      cloud_force_children)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_gather_level (diagnostic) requires the "
                  << "cloud-averaged deposit (no deform/force_local/force_children)"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_gather_order != 1 && cloud_gather_order != 2) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_gather_order must be 1 or 2" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      if (cloud_deform && (cloud_gather_order != 2 || cloud_force_local ||
                           cloud_force_children)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_deform requires cloud_gather_order=2 and "
                  << "excludes cloud_force_local and cloud_force_children" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/experimental_smr_transfer = '" << transfer
                << "' not recognized (must be owner_overlap, smooth_p2, or "
                << "refinable_cloud on a refined mesh; none or refinable_cloud on a "
                << "uniform mesh)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (UsesCompositeTransfer() && deposit != DustDeposit::tsc) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "experimental_smr_transfer=" << transfer
                << " currently requires <dust>/deposit=tsc" << std::endl;
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
  if (smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
    BuildRefinableCloudTransfer(pin);
    if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust): experimental static-SMR refinable_cloud transfer "
                << "is enabled (hat clouds, exact leaf integrals, two-scale split; "
                << "single rank, 2-D, static mesh). SI convergence is not certified"
                << std::endl;
    }
  } else if (pmy_pack->pmesh->multilevel) {
    if (smr_transfer_mode == DustSMRTransfer::owner_overlap) {
      int nccells1 = indcs.cnx1 + 2*(indcs.ng);
      int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
      int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(cdummy, nmb, 3, nccells3, nccells2, nccells1);
      if (global_variable::my_rank == 0) {
        std::cout << "# WARNING (dust): experimental static-SMR owner_overlap transfer "
                  << "is enabled only as a negative control; it is not a certified "
                  << "gather/deposit adjoint" << std::endl;
      }
    } else {
      BuildSMRCompositeTransfer(pin);
      if (global_variable::my_rank == 0) {
        std::cout << "# WARNING (dust): experimental static-SMR smooth_p2 transfer is "
                  << "enabled. Gather and deposition share one composite leaf stencil. ";
        if (smr_transfer.measure_correction) {
          std::cout << "The construction-time uniform-loading audit passed, but SI "
                    << "convergence is not certified" << std::endl;
        } else {
          std::cout << "The raw blend has a known uniform-loading defect and is for "
                    << "diagnostics only" << std::endl;
        }
      }
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
              << " pc2_fraction=" << pc2_fraction
              << " last_zeta=" << hybrid_last_zeta
              << " last_chi=" << hybrid_last_chi << std::endl;
  }
  if (global_variable::my_rank == 0 &&
      smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
    std::cout << "# DUST_CLOUD_SUMMARY splits=" << cloud_split_count
              << " merges=" << cloud_merge_count
              << (cloud_p1_split ? " lobe_splits=" + std::to_string(cloud_lobe_split_count)
                                 : std::string())
              << (cloud_merge ? " coarsen_events=" + std::to_string(cloud_coarsen_events) +
                  " coarsen_in=" + std::to_string(cloud_coarsen_in) +
                  " coarsen_reject=" + std::to_string(cloud_coarsen_reject) +
                  " far_releases=" + std::to_string(cloud_far_release) +
                  " dK_grad=" + std::to_string(cloud_coarsen_dk_grad) +
                  " dK_res=" + std::to_string(cloud_coarsen_dk_res) : std::string())
              << " particles=" << pmy_pack->ppart->nprtcl_thispack << std::endl;
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
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &taus_ = taus;
  auto gids = pmy_pack->gids;
  int nspec = nspecies;
  Real d2g = dust_to_gas;
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
