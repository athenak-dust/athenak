//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust.cpp
//! \brief implementation of DustGasDrag class constructor and support functions

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
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

bool ParseRealList(const std::string &text, std::vector<Real> &out) {
  std::string s = text;
  for (char &c : s) if (c == ',') c = ' ';
  std::istringstream in(s);
  std::string tok;
  out.clear();
  while (in >> tok) {
    char *end = nullptr;
    errno = 0;
    const double v = std::strtod(tok.c_str(), &end);
    if (end == tok.c_str() || *end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
    out.push_back(static_cast<Real>(v));
  }
  return true;
}

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
  // print the compile-time choices of the dust module once at start
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
  CheckRetiredCoupling(pin);   // retired-compat (coupling = hybrid, dust_input.hpp)
  // Driver is constructed after physics modules, so read the coupling and integrator
  // directly from the input while validating the module configuration.
  {
    std::string method = pin->GetOrAddString("dust", "coupling", "imex");
    if (method.compare("imex") == 0) {
      coupling = DustCoupling::imex;
    } else if (method.compare("pc2") == 0) {
      coupling = DustCoupling::pc2;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/coupling = '" << method
                << "' not recognized (must be imex or pc2)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  std::string integrator = pin->GetOrAddString("time", "integrator", "rk2");
  bool bad_integrator = ((coupling == DustCoupling::imex &&
                          integrator.compare("imex2+") != 0) ||
                         (coupling == DustCoupling::pc2 &&
                          integrator.compare("rk2") != 0));
  if (bad_integrator) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust coupling=imex requires integrator=imex2+, while coupling=pc2 "
              << "requires integrator=rk2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // keys of earlier versions that are no longer read (one NOTE each; the run goes on).
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
  // instant_switch; on a uniform mesh the point particles (none) unless refinable_cloud is given
  {
    const bool ml = pmy_pack->pmesh->multilevel;
    const AmrPic pic = ResolveAmrPic(pin, ml);
    smr_transfer_mode = (pic == AmrPic::refinable_cloud) ? DustSMRTransfer::refinable_cloud :
                        (pic == AmrPic::instant_switch) ? DustSMRTransfer::instant_switch :
                                                         DustSMRTransfer::none;
    if (!ml && pin->DoesParameterExist("dust", "amr_pic") && global_variable::my_rank == 0) {
      std::cout << "# NOTE (dust): amr_pic = " << pin->GetString("dust", "amr_pic")
                << " on a uniform mesh: "
                << ((pic == AmrPic::refinable_cloud) ?
                    "refinable clouds (exact TSC clouds, the cloud path of the refined runs)" :
                    "point particles with TSC (instant_switch acts only on a refined mesh)")
                << std::endl;
    }
  }
  // refinable clouds on a 3-D mesh (dust_cloud3d.cpp): 3; on a 2-D mesh: 2; none: 0
  cloud_dim = (smr_transfer_mode != DustSMRTransfer::refinable_cloud) ? 0 :
              (pmy_pack->pmesh->three_d ? 3 : 2);
  // the prototype keys test_cloud_dc1 / test_cloud_dc_sweeps of the clouds' dc1 are replaced by
  // drag_solver (below).  A restart file of the prototype carries them, and a restart can change a
  // key but not remove it: test_cloud_dc1 = false is ignored (NOTE), true stops with the way to
  // continue (the rst of the prototype holds drag_solver = local, so the solver must be given)
  if (pin->DoesParameterExist("dust", "test_cloud_dc1") &&
      pin->GetBoolean("dust", "test_cloud_dc1")) {
    const bool two = pin->DoesParameterExist("dust", "test_cloud_dc_sweeps") &&
                     pin->GetInteger("dust", "test_cloud_dc_sweeps") == 2;
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/test_cloud_dc1 = true (prototype key) is replaced by <dust>/drag_solver. "
              << "To continue a run of the prototype give both dust/test_cloud_dc1=false and "
              << "dust/drag_solver=" << (two ? "dc2" : "dc1") << " (its scheme"
              << (two ? ", test_cloud_dc_sweeps = 2" : "") << "; or dc1 / dc2 / local) on the "
              << "command line, or in the -i file two lines in its <dust> block: "
              << "test_cloud_dc1 = false and drag_solver = " << (two ? "dc2" : "dc1")
              << " (keys with a block/ prefix are not read there, and the rst's true would stay)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    NoteDroppedKey(pin, "dust", "test_cloud_dc1",
                   "prototype key, replaced by <dust>/drag_solver");
    NoteDroppedKey(pin, "dust", "test_cloud_dc_sweeps",
                   "prototype key, replaced by <dust>/drag_solver = dc1 / dc2");
  }
  // <dust>/point_sort_interval = N (default 50; 0: never): point particles (no refinable clouds)
  // are reordered in their arrays by (MeshBlock, k, j, i) of their cell every N cycles, at the
  // start of the cycle (RefineClouds -> SortParticles, the cloud path's sort).  The TSC
  // deposits and gathers then touch neighboring cells from neighboring threads (a fresh
  // dust_si_strat run stores its particles in random order; sorted, a large uniform run steps
  // several times faster).  Only the order
  // of the deposit sums changes (round-off; GPU atomics have no fixed order anyway): every
  // column moves with its particle, tags included; no point-particle state is kept per
  // array slot across the start of a cycle (stage registers are refilled at stage 1, the
  // gravity kick of point particles regathers in full).
  if (smr_transfer_mode != DustSMRTransfer::refinable_cloud) {
    // default 50: the step time stays flat over the 50 cycles between sorts, and the sort costs
    // little per cycle at this interval; 0 turns it off
    point_sort_interval = std::max(0, pin->GetOrAddInteger("dust", "point_sort_interval", 50));
  }
  // shear-periodic x1 in the leaf traversal (the cloud supports across
  // the radial boundaries on the sheared image)
  shear3d = cloud_dim == 3 &&
      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic;
  // (test) false: particles crossing an x1 face after the push take the shear offset of the start
  // of the step, as the routing of point particles does
  if (shear3d) shear3d_route_fix = pin->GetOrAddBoolean("dust", "cloud_shear_route_time", true);
  // (the shear-image lists are cached, always: bitwise the same results as
  // without the cache)
  // 3-D clouds at physical x3 faces fold their supports back (dropping the ghost deposits has no
  // cloud version)
  if (cloud_dim == 3 && pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic &&
      pin->GetOrAddString("dust", "x3_ghost_deposit", "fold") != "fold") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D refinable clouds at physical x3 faces: x3_ghost_deposit = fold only "
              << "(the supports fold back)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // default 4 (the best balance of clouds per rank, fastest of 1 / 2 / 4); 0 gives
  // every new block the cost 1 as AthenaK does; one rank never uses it
  lb_cloud_weight = pin->GetOrAddReal("dust", "load_balance_cloud_weight", 4.0);
  lb_log_interval = TestKeyInt(pin, "dust", "test_load_balance_log", 0);
  // the partition with the smallest largest rank cost (Mesh::LoadBalance); the mesh has read
  // the same key before its first partition (Mesh::SetLoadBalanceStrategy)
  pmy_pack->pmesh->lb_minmax = TestKeyBool(pin, "dust", "test_load_balance_minmax", true);
  if (GetAmrDensityTrigger(pin)) {   // (block map: the only one)
    halo_trigger_cells = std::max(0, pin->GetOrAddInteger("dust", "amr_rho_buffer", 2));
    // the derefinement test of a block of level l >= 1 reaches 2 amr_rho_buffer cells of its level
    // (= amr_rho_buffer cells of the parent's level): the siblings together see the region of the
    // parent's refinement test, so a group derefines only when that whole region fell below the
    // threshold (the shorter reach derefined groups at the first allowed check and refined them
    // again at the next: a periodic flip-flop).  Hidden switch, read only when given (not added
    // to the parameters, so a restart of an older file takes the default):
    // <dust>/test_derefine_parent_buffer = false restores the reach of amr_rho_buffer cells of
    // the block's level, bitwise; diag keeps those old flags and counts the derefinements the
    // default test would keep (ParentBufferDiag).  0 old reach, 1 default, 2 diag
    trigger_parent_buffer = 1;
    if (pin->DoesParameterExist("dust", "test_derefine_parent_buffer")) {
      const std::string v = pin->GetString("dust", "test_derefine_parent_buffer");
      if (v == "true" || v == "1") {
        trigger_parent_buffer = 1;
      } else if (v == "false" || v == "0") {
        trigger_parent_buffer = 0;
      } else if (v == "diag") {
        trigger_parent_buffer = 2;
      } else {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "<dust>/test_derefine_parent_buffer must be true, false or diag (1, 0 or "
                  << "diag)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    halo_trigger_deref_cells = (trigger_parent_buffer > 0) ? 2*halo_trigger_cells : 0;
    if (halo_trigger_cells > 0 && global_variable::my_rank == 0) {
      if (trigger_parent_buffer == 1) {
        std::cout << "# dust density trigger: a block of level l >= 1 derefines only below the "
                  << "threshold over 2 amr_rho_buffer cells of its level (amr_rho_buffer cells of "
                  << "the parent's level)" << std::endl;
      } else {
        std::cout << "# TEST (dust): "
                  << (trigger_parent_buffer == 0 ?
                      "the derefinement test reaches amr_rho_buffer cells of the block's level (the "
                      "former default)" :
                      "diag: the density trigger keeps the derefinement flags of amr_rho_buffer cells "
                      "of the block's level and counts those a test over 2 amr_rho_buffer cells would "
                      "keep")
                  << " (test_derefine_parent_buffer)" << std::endl;
      }
    }
  }
  if (pmy_pack->pmesh->multilevel) {
    // the fenced phase profile (DUST_AMR_PROFILE) only on request; its fences make the
    // host wait for the device several times per cycle
    ProfOn() = TestKeyBool(pin, "dust", "test_amr_profile", false);
    // static or adaptive refinement (dust_amr.cpp: the dust state is rebuilt
    // after every regrid): a 2-D periodic mesh, or a 3-D mesh with periodic or shear-periodic x1
    // (instant_switch or refinable clouds), periodic x2 and periodic or physical x3
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
  // physical x3 faces in 3-D (a stratified box): particles reflect or leave
  // (<dust>/particle_x3_bc), ghost deposits beyond the faces fold back (or are dropped,
  // <dust>/x3_ghost_deposit); the PC2 gather reads the gas ghost cells of the gas boundary
  // conditions, imex the u* ghost cells.
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
    // imex reads u*, which gets x3 boundary values
    // (DustGasDrag::UstarX3BCs); every coupling is allowed
    // the ghost deposits across the shear-periodic faces are moved
    // to the opposite face with the shear offset (DepRemapPlan); the u* ghost fill is remapped
  }
  // AthenaK's time step is cfl_number times the smallest dx/(|v| + c)
  // over the directions; the unsplit RK2 + PLM gas update is stable in 3-D only up to about 1/3
  // (the oblique hydro linear wave breaks down at 0.4).  The dust deposits seed the grid-scale
  // diagonal modes, so a 3-D dust run above 1/3 blows up within a fraction of a sound crossing.
  if (pmy_pack->pmesh->three_d &&
      pin->GetReal("time", "cfl_number") > 1.0/3.0 + 1.0e-12 &&
      !TestKeyBool(pin, "dust", "test_allow_cfl_above_third_3d", false)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D dust: <time>/cfl_number = " << pin->GetReal("time", "cfl_number")
              << " exceeds 1/3, the stability limit of the 3-D RK2 + PLM gas update; "
              << "use cfl_number <= 0.3 (or <dust>/test_allow_cfl_above_third_3d = true for an "
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
  if (coupling == DustCoupling::pc2 && gamma_switch) {
    if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust): <dust>/gamma_switch=true has no effect for "
                << "coupling=pc2"
                << "; gamma_switch only modifies the imex2+ tableau and will be ignored."
                << std::endl;
    }
    gamma_switch = false;
  }
  stopping_times_initialized = false;
  dt_cfl        = pin->GetOrAddReal("dust","dt_cfl",0.5);
  // <dust>/dust_to_gas is the legacy total (to be deprecated); with
  // the new abundance keys it is not read (mixing is an error, dust_species.cpp) and holds the
  // sum of epsilon_N / Z_N for the default masses
  {
    const SpeciesTable &st = Species(pin);
    dust_to_gas = st.new_init ? st.total : pin->GetOrAddReal("dust","dust_to_gas",0.01);
  }

  // PC2 feedback stiffness monitor, optional dt limit and predictor damping: the damping (as
  // Athena-C) on by default, the dt limit left to the user (off)
  stiff_thresh = pin->GetOrAddReal("dust", "stiff_thresh", -1.0);
  // the IMEX gas gets the lumped solve's change plus its mass share of the residual (3-D
  // clouds; false = the gas takes the clouds' whole impulse)
  imex_consistent_gas = TestKeyBool(pin, "dust", "test_imex_consistent_gas", true);
  // hidden key, on by default: the clouds' share of the lumped IMEX solve's residual goes to the
  // clouds whose center lies in the cell, up to the kick of one nominal cloud, and the u* gather
  // blends to the deposit's adjoint where the dust dominates (dust_imex3d.cpp).  false = the split
  // by the overlap weights and the plain gather.  It applies to 3-D refinable clouds with
  // coupling = imex, back-reaction and imex_consistent_gas and is skipped
  // silently elsewhere (a restart stores the default-added key, so "given in the input" cannot be
  // told from the default); the switches are set below for refinable clouds.
  {
    const bool on = TestKeyBool(pin, "dust", "test_imex_residual_to_center_upto_nominal", true);
    imex_res_want = on && cloud_dim == 3 && coupling == DustCoupling::imex && back_reaction &&
                    imex_consistent_gas;
  }
  // hidden diagnostic key: the dt threshold below which every cycle reports what limits dt
  dt_diag_thr = pin->DoesParameterExist("dust", "test_dt_diag") ?
                pin->GetReal("dust", "test_dt_diag") : 0.0;
  if (dt_diag_thr > 0.0 && global_variable::my_rank == 0) {
    std::cout << "# NOTE (dust): test_dt_diag = " << dt_diag_thr << ": cycles with a smaller dt "
              << "print '# DT_DIAG' lines (the limiting cell / cloud of each rank)" << std::endl;
  }
  pc2_feedback_damping = TestKeyBool(pin, "dust", "test_pc2_feedback_damping", true);
  {
    // default: dc1 for point particles on a uniform mesh with coupling = imex
    // (every geometry the point path supports: periodic, and in 3-D shear-periodic x1 and/or
    // physical x3 faces), local otherwise (PC2, refinable clouds, instant_switch on
    // a refined mesh).  The value is written to the restart file (the default too, as every
    // GetOrAdd key): a run started with drag_solver = local, or before this default, keeps local
    // when it continues from its rst unless drag_solver is given again
    // (also dc1 for 3-D refinable clouds with coupling = imex;
    // it falls back to local only with the lagged x slope of the shear pull on a refined mesh, see
    // the cloud set-up below; dust_imex3d_dc.cpp)
    // (also dc1 for 2-D refinable clouds with coupling = imex, the same operator,
    // dust_imex3d_dc.cpp)
    const bool point_uniform = smr_transfer_mode == DustSMRTransfer::none &&
                               !pmy_pack->pmesh->multilevel;
    const std::string def_solver =
        ((point_uniform || UsesRefinableClouds()) && coupling == DustCoupling::imex) ? "dc1" :
                                                                                     "local";
    solver_given_ = pin->DoesParameterExist("dust", "drag_solver");
    // where a given value comes from (the NOTEs below): a restart file stores every key, a
    // default with the comment "# Default value added at run time", which a change on the command
    // line keeps and a value in an input file replaces
    if (solver_given_) {
      std::string comment;
      for (auto &blk : pin->block) {
        if (blk.block_name != "dust") continue;
        const InputLine *pl = blk.GetPtrToLine("drag_solver");
        if (pl != nullptr) comment = pl->param_comment;
        break;
      }
      solver_src_ = (comment.find("Default value added at run time") != std::string::npos) ?
          "from the restart file, the default of the run that wrote it (or the same value given "
          "again on the command line, which keeps the file's comment)" :
          "given in the input (or in the input of the run that wrote the restart file)";
    }
    // (point particles in a 2-D shearing box: the coupled solvers stop below, no dc1)
    solver_default_dc1_ = (def_solver == "dc1") && !(shear_x1 && !pmy_pack->pmesh->three_d);
    std::string solver = pin->GetOrAddString("dust", "drag_solver", def_solver);
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
    // a coupling without an implicit drag solve, PC2 (coupling = pc2), does not use drag_solver: a value other than local (a restart
    // file of an IMEX run stores dc1, its default) is ignored (NOTE) and the run takes local
    // inside.  The parameter keeps its value, so that a later restart with coupling = imex
    // continues with it.
    if (coupling == DustCoupling::pc2 && drag_solver != DustDragSolver::local) {
      if (global_variable::my_rank == 0) {
        std::cout << "# NOTE (dust): drag_solver = " << solver << " (" << solver_src_
                  << ") is not used with coupling = pc2 (no implicit drag solve): ignored, the run "
                  << "uses local; the parameter keeps " << solver << " for a later restart with "
                  << "coupling = imex" << std::endl;
      }
      drag_solver = DustDragSolver::local;
      solver_default_dc1_ = false;
    }
    // coupling = imex with local where dc1 is the default (point particles here; refinable
    // clouds once their conditions for dc1 are known, below): a NOTE with the value's origin, for
    // a restart of a PC2 run keeps the stored local
    if (drag_solver == DustDragSolver::local && solver_default_dc1_ && !UsesRefinableClouds()) {
      NoteLocalNotDefault();
    }
  }
  // (refinable clouds, 2-D and 3-D: local, dc1 or dc2, checked in their set-up below)
  const bool cloud_dc_req = UsesRefinableClouds() &&
      (drag_solver == DustDragSolver::dc1 || drag_solver == DustDragSolver::dc2);
  if (pmy_pack->pmesh->multilevel && drag_solver != DustDragSolver::local && !cloud_dc_req) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "dust on a refined mesh supports only drag_solver = local "
              << "(the coupled solvers have no composite transfer; drag_solver: " << solver_src_
              << "): give dust/drag_solver=local" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  drag_rtol = pin->GetOrAddReal("dust", "drag_rtol", 1.0e-11);
  drag_atol = pin->GetOrAddReal("dust", "drag_atol", 1.0e-14);
  drag_iter_max = pin->GetOrAddInteger("dust", "drag_iter_max", 200);
  drag_diagnostic_interval = TestKeyInt(pin, "dust", "test_drag_diagnostic_interval", 0);
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
    // species_distribution = continuous: every particle has its own stopping time
    // (particle_static by default; species_fixed is refused)
    const bool cont = Species(pin).continuous;
    std::string mode = pin->GetOrAddString("dust","stopping_time_mode",
                                           cont ? "particle_static" : "species_fixed");
    if (cont && mode.compare("species_fixed") == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<dust>/species_distribution = continuous draws a stopping time "
                << "for every particle: stopping_time_mode = particle_static (the default) or "
                << "dynamic, not species_fixed" << std::endl;
      std::exit(EXIT_FAILURE);
    }
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
  // tau_s_N, stokes_N or the power law, or the legacy taus_N (to be deprecated)
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
      // continuous: the range of the distribution (the bin edges), before the particles refresh it
      taus_max = std::max(taus_max, st.continuous ? static_cast<Real>(st.tau_hi[s]) : ts);
      taus_min = std::min(taus_min, st.continuous ? static_cast<Real>(st.tau_lo[s]) : ts);
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
                << "instant_switch" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
      // <dust>/refinable_mode: the validated parameter sets of the refinable
      // clouds (RefinableMode, dust.hpp).  A preset key given explicitly (input or restart file)
      // overrides the preset value; PrintRefinablePreset lists the resolved values.
      {
        const bool given = pin->DoesParameterExist("dust", "refinable_mode");
        std::string mode = given ? pin->GetString("dust", "refinable_mode") : "default";
        // the old key cloud_quiet_protection maps onto careful_merge
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
            "test_cloud_merge_alpha_leaf", "cloud_merge_budget", "cloud_merge_schedule",
            "cloud_merge_dt_omega", "cloud_merge_interval", "cloud_resample",
            "cloud_resample_eta", "cloud_resample_cold_alpha", "cloud_resample_schedule",
            "cloud_resample_dt", "cloud_resample_dt_omega", "cloud_resample_interval",
            "test_resample_after_regrid"};
        refinable_given.clear();
        for (const char *k : keys) {
          if (pin->DoesParameterExist("dust", k)) {
            refinable_given += std::string(" ") + k + "=" + pin->GetString("dust", k);
          }
        }
      }
      const bool x3phys3d = cloud_dim == 3 &&
          pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic;
      // (several ranks, 2-D and 3-D: the dust halo as on a refined mesh; the options it does
      // not cover stop in CheckMultiRank.  2-D clouds on a uniform mesh were limited to one rank
      // in earlier versions)
      if (!multilevel && !((pmy_pack->pmesh->two_d || cloud_dim == 3) &&
                           (pmy_pack->pmesh->strictly_periodic || shear3d || x3phys3d) &&
                           (drag_solver == DustDragSolver::local ||
                            drag_solver == DustDragSolver::dc1 ||
                            drag_solver == DustDragSolver::dc2))) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "refinable_cloud on a uniform mesh requires supported "
                  << "2-D/3-D cloud geometry and drag_solver = local, dc1 or dc2"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // (the clouds gather the linear reconstruction of the leaves; cloud_gather_order = 1 was
      // retired)  // retired-compat
      // the cloud residual rule (test_imex_residual_to_center_upto_nominal): its gather blend
      // needs a gather with the deposit's own weights (the transition clouds' mode 2, the only one)
      {
        const bool on = imex_res_want;
        imex_resid_center = imex_gather_blend = imex_resid_center_floor = on;
        // (its NOTE follows the dc1 settings below, which turn the gather blend off)
      }
      // drag_solver = dc1 / dc2 (the default with coupling = imex; dust_imex3d_dc.cpp): one / two
      // defect-correction sweeps of the lumped IMEX solve with the clouds' own gather and deposit.
      // Needs coupling = imex (otherwise the default is local).
      // The drag_solver of the rst (also the default, written at run time) is kept on a restart: a
      // run started with local, or before dc1 was the default, continues with local unless
      // drag_solver is given again.  The gather blend is off on this path (in A and in the kick:
      // its dt dependence alone keeps the scheme first order; test_cloud_dc1_blend, which kept it,
      // retired).
      if (cloud_dc_req) {
        const bool ok = coupling == DustCoupling::imex;
        if (!ok && solver_given_) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << "<dust>/drag_solver = " << (drag_solver == DustDragSolver::dc2 ? "dc2" : "dc1")
                    << " (" << solver_src_ << ") cannot run here: the clouds' dc1 / dc2 needs "
                    << "coupling = imex; give dust/drag_solver=local" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (!ok) {
          drag_solver = DustDragSolver::local;
          pin->SetString("dust", "drag_solver", "local");
        }
        cloud_dc1 = ok && back_reaction;
        cloud_dc_sweeps = (drag_solver == DustDragSolver::dc2) ? 2 : 1;
      } else if (drag_solver == DustDragSolver::local && solver_default_dc1_) {
        // (local where dc1 would be the default)
        if (coupling == DustCoupling::imex) {
          NoteLocalNotDefault();
        }
      }
      if (cloud_dc1) {
        cloud_dc1_opcheck = pin->DoesParameterExist("dust", "test_cloud_dc1_opcheck") &&
                            pin->GetBoolean("dust", "test_cloud_dc1_opcheck");
        // one blend value for the correction's operator, the kick and the operator check: off
        imex_gather_blend = false;
        if (global_variable::my_rank == 0) {
          std::cout << "# NOTE (dust): " << (cloud_dim == 3 ? "3-D" : "2-D") << " clouds, drag_solver = dc"
                    << cloud_dc_sweeps << ": "
                    << cloud_dc_sweeps
                    << " defect-correction sweep(s) of the lumped IMEX solve per working stage "
                    << "(the clouds' gather and deposit), gather blend "
                    << "off"
                    << (cloud_dc1_opcheck ? ", operator check every stage" : "") << std::endl;
        }
      }
      // 2-D clouds: the IMEX residual is split as for the points and the 3-D clouds
      // (CloudImplicitConsistency2D, overlap weights) when imex_consistent_gas; earlier
      // versions gave the 2-D clouds the mirror whatever the key (and the rst holds it true by
      // default)
      if (cloud_dim != 3 && coupling == DustCoupling::imex && back_reaction &&
          global_variable::my_rank == 0) {
        std::cout << "# NOTE (dust): 2-D clouds, IMEX residual "
                  << (imex_consistent_gas ?
                      "split between gas and clouds (overlap weights); "
                      "the 2-D clouds took the mirror in earlier versions, so a run continued from "
                      "an older restart file changes; test_imex_consistent_gas = false gives the "
                      "mirror" :
                      "to the gas as the impulse deposit (mirror, test_imex_consistent_gas = false)")
                  << std::endl;
      }
      if (imex_resid_center && global_variable::my_rank == 0) {
        std::cout << "# NOTE (dust): test_imex_residual_to_center_upto_nominal = true: the IMEX "
                  << "residual's dust share of a leaf cell goes to the clouds whose center lies "
                  << "in it (at most the kick of one nominal cloud; the rest, and that of a cell "
                  << "without a centered cloud, by the overlap weights)"
                  << (imex_gather_blend ? ", and the slope part of the clouds' u* gather fades "
                                          "with the dust share Q/(rho + Q)" :
                                          "; the gather blend is off (drag_solver = dc1 / dc2)")
                  << std::endl;
      }
      ReadSplitVelocityKeys(pin);   // <dust>/cloud_split_velocity (dust_split_velocity.cpp)
      // test switch; false keeps every cloud at its level whatever the mesh
      cloud_split_enable = TestKeyBool(pin, "dust", "test_cloud_split_enable", true);
      // G (quantum merge under the pass budget): on in default and careful_merge (a full AMR
      // needs a coarsening channel), off in split_only
      cloud_merge = preset_bool("cloud_merge", !split_only);
      cloud_merge_buffer = pin->GetOrAddInteger("dust", "cloud_merge_buffer", 2);
      // production alpha 0.2 in 3-D (0.05 in 2-D); alpha H / tau lets the merged cloud's
      // velocity error move its mass by at most alpha cells within one damping time
      cloud_merge_alpha = pin->GetOrAddReal("dust", "cloud_merge_alpha",
                                            pmy_pack->pmesh->three_d ? 0.2 : 0.05);
      // the 3-D G velocity tolerance on the cell size of the leaf
      // that holds the merge cell (alpha 2^gap H / tau when the leaf is gap levels coarser);
      // on by default
      cloud_merge_alpha_leaf = TestKeyBool(pin, "dust", "test_cloud_merge_alpha_leaf", true);
      cloud_merge_interval = pin->GetOrAddInteger("dust", "cloud_merge_interval", 10);
      // merge schedule in physical time, burst after derefinement
      {
        Real om = 0.0;
        if (pin->DoesBlockExist("shearing_box") && pin->DoesParameterExist("shearing_box", "omega0")) {
          om = pin->GetReal("shearing_box", "omega0");
        }
        // physical time (every cloud_merge_dt_omega = 0.25
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
        } else if (ms != "cycle") {
          fatal("cloud_merge_schedule must be cycle or time");
        }
        // the merge bursts after a derefinement: <dust>/test_merge_after_derefine (hidden; off
        // or "dt [interval]", MergeAfterDerefine in dust_input.hpp)
        const MergeAfterDerefine mb = MergeAfterDerefine::Parse(
            TestKeyString(pin, "dust", "test_merge_after_derefine", "off"));
        if (mb.dt_omega > 0.0) {
          if (!(om > 0.0)) fatal("test_merge_after_derefine needs <shearing_box>/omega0 > 0");
          cloud_merge_burst_dt = mb.dt_omega/om;
        }
        cloud_merge_burst_interval = mb.interval;
      }
      // stage X (the exact stencil re-merge before G): careful_merge only
      cloud_merge_stencil = preset_bool("cloud_merge_stencil", careful);
      // no X in a 3-D shearing box: the background shear pulls the
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
      // (cloud_merge_smax 1, cloud_merge_K 4: constants; merged clouds keep their members' common
      // ts: always)
      cloud_merge_g_lag = pin->GetOrAddInteger("dust", "cloud_merge_g_lag", 2);
      // the production pass budget beta = 0.3; cloud_merge_budget = -1 switches it off
      cloud_merge_budget = pin->GetOrAddReal("dust", "cloud_merge_budget", 0.3);
      // the structure-relative budget (careful_merge: 1; 0 = off)
      cloud_merge_budget_kappa = preset_real("cloud_merge_budget_kappa", careful ? 1.0 : 0.0);
      // R (resampling) on in default and careful_merge, with the production clock below; off in
      // split_only
      cloud_resample = preset_bool("cloud_resample", !split_only);
      // <dust>/cloud_resample_schedule = cycle (every
      // cloud_resample_interval cycles) or time (event k at t0 + k dt, dt from
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
        const bool g_rst = pin->DoesParameterExist("dust", "cloud_resample_schedule_reset");
        if (pin->DoesBlockExist("shearing_box") &&
            pin->DoesParameterExist("shearing_box", "omega0")) {
          cloud_resample_omega = pin->GetReal("shearing_box", "omega0");
        }
        // physical time whenever omega0 is given (every
        // cloud_resample_dt_omega = 1 / omega0 unless cloud_resample_dt is given), cycles otherwise
        const std::string sched = pin->GetOrAddString("dust", "cloud_resample_schedule",
                                                      (cloud_resample_omega > 0.0) ? "time" : "cycle");
        // every 0.25 / omega0 by default, with no resampling after the
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
          cloud_resample_schedule_reset =
              pin->GetOrAddBoolean("dust", "cloud_resample_schedule_reset", false);
        } else if (sched == "cycle") {
          if (g_dt || g_dto || g_rst) {
            ignored("cloud_resample_dt, cloud_resample_dt_omega and "
                    "cloud_resample_schedule_reset", "cycle");
          }
        } else {
          fatal("cloud_resample_schedule must be cycle or time");
        }
      }
      if (!cloud_resample_by_time) {
        cloud_resample_interval = pin->GetOrAddInteger("dust", "cloud_resample_interval", 625);
      }
      // the production cap factor
      cloud_resample_eta = pin->GetOrAddReal("dust", "cloud_resample_eta", 1.5);
      {
        // per-level eta_L: without <dust>/cloud_resample_eta_levels the same eta on every level;
        // "auto": growth with the level up to a ceiling (in 3-D eta = 8 would give a cloud of
        // level L the nominal mass of level L - 1); a list of numbers: eta_L as given
        const bool three = pmy_pack->pmesh->three_d;
        std::vector<Real> given;
        bool grow = false;
        if (pin->DoesParameterExist("dust", "cloud_resample_eta_levels")) {
          const std::string lv = pin->GetString("dust", "cloud_resample_eta_levels");
          std::string word;
          for (char c : lv) if (!std::isspace(static_cast<unsigned char>(c))) word += std::tolower(c);
          if (word == "auto") {
            grow = true;
          } else if (!ParseRealList(lv, given) || given.empty()) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                      << "<dust>/cloud_resample_eta_levels = '" << lv << "' must be auto or one or "
                      << "more numbers" << std::endl;
            std::exit(EXIT_FAILURE);
          }
        }
        const int nlev = ResampleCapTable::kLevels;
        cloud_resample_eta_lev.assign(nlev, 0.0);
        for (int l=0; l<nlev; ++l) {
          Real e;
          if (!given.empty()) {
            e = given[std::min(l, static_cast<int>(given.size()) - 1)];
          } else if (!grow || cloud_resample_eta <= 0.0) {
            e = cloud_resample_eta;
          } else {
            const Real top = three ? std::max<Real>(4.0, cloud_resample_eta)
                                   : std::max<Real>(2.0, cloud_resample_eta);
            e = std::min(cloud_resample_eta*std::exp2(three ? 0.5*l : 0.25*l), top);
          }
          cloud_resample_eta_lev[l] = e;
          // eta_L / 2^(d L) exactly (a power of two)
          cloud_resample_cap.f[l] = (e > 0.0) ? std::ldexp(e, -(three ? 3 : 2)*l) : -1.0;
        }
      }
      // resampling leaves cold patches alone (careful_merge: 0.05; 0 = off)
      cloud_resample_cold_alpha = preset_real("cloud_resample_cold_alpha", careful ? 0.05 : 0.0);
      // patches inside one MeshBlock of min(cloud_resample_patch 2^level,
      // cloud_resample_patch_cells)^2 of its cells (constants 2 and 4)
      // a block patch, or a quadrant of one, holding more than this many pool clouds
      // (of one species) is split into its four quadrants, recursively down to single cells;
      // 0 = never.  Only for block patches whose widths min(cloud_resample_patch 2^level,
      // cloud_resample_patch_cells) are powers of two (the halving loses the
      // odd row and column otherwise) and at most 16.  Default 1024 where allowed
      auto pow2 = [](const int n) {return n >= 1 && (n & (n - 1)) == 0;};
      const bool split_ok = cloud_resample_patch_cells <= 16 &&
                            pow2(cloud_resample_patch) && pow2(cloud_resample_patch_cells);
      cloud_resample_patch_split = TestKeyInt(pin, "dust", "test_cloud_resample_patch_split",
                                                        split_ok ? 1024 : 0);
      // (3-D) a patch with more pool clouds than this (one cell of a dense core can hold 1e5 to
      // 1e6) is solved as sub-patches of consecutive pool clouds, each keeping its own share of
      // every row, so that the solves stay small and run in parallel (constant 4096)
      // (3-D) the (cloud, leaf cell) pairs of the pool are held in tables with 32-bit offsets: the
      // block patches of a rank are resampled in consecutive batches of about this many pairs
      // (patches are independent, so the result is that of one pass).  A target: a base patch with
      // more pairs is a batch of its own; the hard limit is 2^31 - 1 pairs per batch
      cloud_resample_batch_pairs = static_cast<long long>(pin->GetOrAddReal("dust",
          "cloud_resample_batch_pairs", 268435456.0));   // 2^28 (2^30 in earlier versions)
      cloud_resample_threads = pin->GetOrAddInteger("dust", "cloud_resample_threads", 1);
      // (3-D refinable clouds only; 2-D ignores it): the device solve leaves out the
      // patches of n clouds on nl leaves with n <= 7 nl + 9 (+ the cross-moment rows in a flagged
      // patch; the nominal rows: no null vector is certain there).  false: every patch is solved
      // (the reference path)
      cloud_resample_rank_screen = TestKeyBool(pin, "dust", "test_cloud_resample_rank_screen", true);
      // (3-D refinable clouds) rows of the position-velocity cross moments sum m (x - c)_a (v - u)_b
      // per patch (c the patch center, u the patch mean velocity before R), only in the flagged
      // patches (dust_resample.hpp: several cloud levels, a cloud finer than a footprint leaf, a
      // physical x3 fold or an ambiguous periodic unwrapping; elsewhere the existing rows already
      // fix them and those patches solve bitwise as with none): angmom (default in 3-D; the 3
      // components of sum m (x - c) x (v - u): with the patch mass, center of mass and momentum
      // kept, the patch angular momentum about its center of mass; in a shearing box + 3 rows of
      // spatial second moments, so that the inertial J is kept too), full (all 9 cross moments,
      // + the same 3 in a shearing box), none (no extra rows anywhere).  2-D has no such rows: the
      // key, read there only when present, must be none
      {
        std::string xv;
        if (cloud_dim == 3) {
          xv = TestKeyString(pin, "dust", "test_cloud_resample_xv_rows", "angmom");
        } else {
          xv = pin->DoesParameterExist("dust", "test_cloud_resample_xv_rows")
             ? pin->GetString("dust", "test_cloud_resample_xv_rows") : "none";
        }
        if (xv == "none") {
          cloud_resample_xv_rows = 0;
        } else if (xv == "angmom") {
          cloud_resample_xv_rows = 1;
        } else if (xv == "full") {
          cloud_resample_xv_rows = 2;
        } else {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << "<dust>/test_cloud_resample_xv_rows must be none, angmom or full" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        if (cloud_resample_xv_rows != 0 && cloud_dim != 3) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << "<dust>/test_cloud_resample_xv_rows = " << xv << " exists only for 3-D "
                    << "refinable clouds (the 2-D resampling has no cross-moment rows)" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        // diagnostic: per R pass the changes of J about the center of mass of all clouds, of the
        // per-patch moments and of the leaf masses and momenta (3-D, extra kernels only)
        cloud_resample_xv_audit = pin->DoesParameterExist("dust", "test_cloud_resample_xv_audit") &&
                                  pin->GetBoolean("dust", "test_cloud_resample_xv_audit");
      }
      // the per-patch solves on the device (same patches and algorithm; false keeps the host
      // solves).  The default follows the build: the device solver when the
      // default execution space is not a host space (GPU builds), the threaded host solver
      // otherwise (on a CPU build the host solver is much faster than the device code)
      // (3-D refinable clouds have only the device solver, which also runs on a CPU)
      cloud_resample_device = pin->GetOrAddBoolean("dust", "cloud_resample_device",
          !std::is_same<DevExeSpace, Kokkos::DefaultHostExecutionSpace>::value || cloud_dim == 3);
      cloud_resample_device_mem_mb = pin->GetOrAddInteger("dust", "cloud_resample_device_mem_mb", 2048);
      // default 1024: with the patch split, the resampling was faster than with 512
      cloud_resample_device_wide_team = pin->GetOrAddInteger("dust", "cloud_resample_device_wide_team", 1024);
      // (the device QR update shares the rows among the threads as well: its sums run in another
      // order than the host's, so the device solve agrees with the host solve only to round-off;
      // the reference build -DDUST_R_QR_COLUMNS=1 takes one thread per column in the host's order,
      // dust_resample.hpp.  The launch bound of the cloud kernels is the compile-time
      // DUST_CLOUD_LAUNCH_BOUNDS, dust.hpp; cloud_resample_qr_rows and cloud_launch_bounds
      // retired)
      // the order in which a patch takes its candidates: array (default)
      // or tag (PTAG order, independent of the storage order)
      {
        // (<dust>/test_canonical_order = resample_tag, hidden; cloud_resample_order = position,
        // retired: CheckRetiredKeys stops where R runs, else it is ignored)  // retired-compat
        cloud_resample_tag_order = CanonicalOrder::Parse(
            TestKeyString(pin, "dust", "test_canonical_order", "off")).resample_tag;
      }
      cloud_resample_m0 = pin->GetOrAddReal("dust", "cloud_resample_m0", -1.0);
      if (cloud_resample_m0 > 0.0 && nspecies > 1 && global_variable::my_rank == 0) {
        std::cout << "# NOTE (dust): cloud_resample_m0 is the root cloud mass of every species "
                  << "(leave it unset for the per-species masses of the initial clouds)" << std::endl;
      }
      // resampling after every regrid.  Default false: no G or R after the regrids, the scheduled
      // ones (G and R every 0.25 / omega0) cover them; with a regrid every ~40 cycles the
      // per-regrid R was latency-bound
      // <dust>/test_resample_after_regrid (hidden; ResampleAfterRegrid in dust_input.hpp):
      // off (default), or local / global [derefine] [dt]: the resampling after the regrids, on the
      // blocks the regrid made and their neighbors (local) or on every block (global; the
      // scheduled resampling stays global, also when a regrid event falls on its cycle), also
      // after a regrid that only deleted blocks (derefine; else such an event runs G but no R,
      // unless the scheduled resampling falls on that cycle), with the regrid events of dt /
      // omega0 after the first one handled together (0: every regrid at the next cycle).
      {
        const ResampleAfterRegrid ra = ResampleAfterRegrid::Parse(
            TestKeyString(pin, "dust", "test_resample_after_regrid", "off"));
        cloud_resample_after_regrid = ra.on;
        cloud_resample_event_local = ra.local;
        cloud_resample_after_derefine = ra.derefine;
        if (ra.on && ra.dt_omega > 0.0) {
          Real om = 0.0;
          if (pin->DoesBlockExist("shearing_box") && pin->DoesParameterExist("shearing_box", "omega0")) {
            om = pin->GetReal("shearing_box", "omega0");
          }
          if (!(om > 0.0)) {
            std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                      << "test_resample_after_regrid with dt needs <shearing_box>/omega0 > 0"
                      << std::endl;
            std::exit(EXIT_FAILURE);
          }
          cloud_event_dt = ra.dt_omega/om;
        }
      }
      // the split budget (most clouds one rank's split pass creates per cycle, 0: no limit; off
      // by default) and the local R right after a split pass that held events back or in which
      // some rank with at least the mean clouds per rank (after the split) created at least
      // cloud_split_resample_frac of its clouds (SplitResampleWanted; on by default for 3-D
      // refinable clouds with R; it fires on large splits only: a new level
      // opening, a restart of a file with capped cloud levels); 3-D refinable clouds only.  The
      // default is written to the parameters: a restart file written before this default carries
      // cloud_split_resample = false and keeps it
      cloud_split_budget = pin->GetOrAddInteger("dust", "cloud_split_budget", 0);
      cloud_split_resample = TestKeyBool(pin, "dust", "test_cloud_split_resample",
                                                  cloud_dim == 3 && cloud_resample);
      // (cloud_split_resample_frac: the constant 0.05)
      if (cloud_split_budget < 0 ||
          ((cloud_split_budget > 0 || cloud_split_resample) && cloud_dim != 3)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "<dust>/cloud_split_budget (>= 0) and test_cloud_split_resample "
                  << "are for 3-D refinable clouds" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // without R there is no R after a split: off, also when the parameters say true (a restart
      // file of a run with R carries the default true; a restart may switch R off through
      // refinable_mode = split_only or cloud_resample = false)
      if (cloud_split_resample && !cloud_resample) {
        cloud_split_resample = false;
        if (global_variable::my_rank == 0) {
          std::cout << "# NOTE (dust): test_cloud_split_resample = true ignored: cloud_resample is off"
                    << std::endl;
        }
      }
      split_blk_mark = cloud_split_resample;
      if (cloud_resample && ((!cloud_resample_by_time && cloud_resample_interval < 1) ||
                             cloud_resample_patch_split < 0 ||
                             cloud_resample_batch_pairs < 1 ||
                             cloud_resample_batch_pairs > std::numeric_limits<int>::max() ||
                             (cloud_resample_patch_split > 0 && !split_ok))) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "cloud_resample requires cloud_resample_interval >= 1,"
                  << " test_cloud_resample_patch_split >= 0, 1 <= cloud_resample_batch_pairs < 2^31"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // (the requirements of G apply only when G is on)
      if (cloud_merge && (cloud_merge_buffer < 2 ||
                          cloud_merge_alpha <= 0.0 || cloud_merge_interval < 1)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "<dust>/cloud_merge "
                  << "needs cloud_merge_buffer >= 2, "
                  << "cloud_merge_alpha > 0"
                  << " and cloud_merge_interval >= 1" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      PrintRefinablePreset();
    }
    if (UsesRefinableClouds() && deposit != DustDeposit::tsc) {
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
  // point particles crossing a shear-periodic x1 face take the shear offset of the
  // time of their position (<dust>/shear_route_time, default true; 3-D, where the push advects y
  // with the background shear).  The 3-D clouds keep their own key (cloud_shear_route_time).
  if (is_shearing_box && pmy_pack->pmesh->three_d && cloud_dim != 3 &&
      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic) {
    shear_route_point = pin->GetOrAddBoolean("dust", "shear_route_time", true);
  }
  // constant azimuthal acceleration on the particles (with the gas's const_accel2 it keeps
  // a radially boosted 2-D r-z shearing-box equilibrium, see the streaming_linear pgen)
  const_accel_phi = TestKeyReal(pin, "dust", "test_const_accel_phi", 0.0);
  if (const_accel_phi != 0.0 &&
      !(is_shearing_box && !pmy_pack->pmesh->three_d && coupling == DustCoupling::pc2)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<dust>/test_const_accel_phi requires a 2-D r-z shearing box and "
              << "coupling=pc2" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // point particles in a 3-D shearing box: dc1 / dc2 only (coupling = imex), whose operator is the
  // kick's own gather and impulse deposit (the stage's u* exchange and deposit remap); PCG and the
  // adaptive solver need D = G^T, which the shear-collected ghost gather and the remapped ghost
  // deposit do not give
  point_dc_general = pmy_pack->pmesh->three_d && smr_transfer_mode == DustSMRTransfer::none &&
                     !pmy_pack->pmesh->multilevel && coupling == DustCoupling::imex &&
                     (drag_solver == DustDragSolver::dc1 || drag_solver == DustDragSolver::dc2) &&
                     (shear_x1 || x3_phys);
  // hidden <dust>/test_qrho_diag = N (default 0, off): the QRHO_DIAG line every N cycles
  qrho_diag_ = pin->DoesParameterExist("dust", "test_qrho_diag") ?
               pin->GetInteger("dust", "test_qrho_diag") : 0;
  // hidden <dust>/test_point_dc_opcheck: the operator check of the dc1 / dc2 correction of point
  // particles (coupling = imex) at every working stage (dust_imex3d_dc.cpp, CloudDcOpCheck*)
  point_dc_opcheck = pin->DoesParameterExist("dust", "test_point_dc_opcheck") &&
                     pin->GetBoolean("dust", "test_point_dc_opcheck") &&
                     smr_transfer_mode == DustSMRTransfer::none && coupling == DustCoupling::imex &&
                     back_reaction &&
                     (drag_solver == DustDragSolver::dc1 || drag_solver == DustDragSolver::dc2);
  if (drag_solver != DustDragSolver::local && shear_x1 && !point_dc_general && cloud_dim != 3) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Coupled drag solvers require ordinary periodic boundaries; "
              << "the exact-transpose shear deposit is not implemented (drag_solver: "
              << solver_src_ << "): give dust/drag_solver=local" << std::endl;
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
  // the instant_switch transfer also runs on a dynamic mesh (one rank), in 3-D also on several
  // ranks (the general deposit remap)
  if (pmy_pack->pmesh->adaptive && smr_transfer_mode != DustSMRTransfer::refinable_cloud &&
      !(smr_transfer_mode == DustSMRTransfer::instant_switch &&
        (global_variable::nranks == 1 || pmy_pack->pmesh->three_d))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust on an adaptive mesh needs <dust>/amr_pic = refinable_cloud (or "
              << "instant_switch: one rank in 2-D)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
    BuildRefinableCloudTransfer(pin);
    // the linear gather of the implicit stages reads u* in the ghost cells (its central
    // differences): with several levels the u* exchange restricts and prolongates through a
    // coarse buffer (PC2 forms its midpoint velocity from gas states that include ghosts)
    const bool implicit_stages = coupling == DustCoupling::imex;
    if (pmy_pack->pmesh->multilevel && implicit_stages) {
      int nccells1 = indcs.cnx1 + 2*(indcs.ng);
      int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
      int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(cdummy, nmb, 3, nccells3, nccells2, nccells1);
    }
    CheckMultiRank();
    CheckClouds3D();   // the 3-D restrictions
  } else if (pmy_pack->pmesh->multilevel) {
    if (smr_transfer_mode == DustSMRTransfer::instant_switch) {
      if (global_variable::my_rank == 0) {
        std::cout << "# DUST_PIC amr_pic = instant_switch (point particles, TSC on the block that "
                  << "owns them, ghost deposits remapped to the leaves)" << std::endl;
      }
      int nccells1 = indcs.cnx1 + 2*(indcs.ng);
      int nccells2 = (indcs.cnx2 > 1)? (indcs.cnx2 + 2*(indcs.ng)) : 1;
      int nccells3 = (indcs.cnx3 > 1)? (indcs.cnx3 + 2*(indcs.ng)) : 1;
      Kokkos::realloc(cdummy, nmb, 3, nccells3, nccells2, nccells1);
    }
  }
  // retired-compat: the keys of the retired options, judged now that the
  // representation, G, R and the coupling are known (dust_input.hpp)
  {
    RetiredContext rc;
    rc.clouds = UsesRefinableClouds();
    rc.split = rc.clouds && cloud_split_enable;
    rc.g_on = rc.clouds && cloud_merge;
    rc.r_on = rc.clouds && cloud_resample;
    rc.stencil = cloud_merge_stencil;
    rc.budget = cloud_merge_budget > 0.0;
    rc.dim = cloud_dim;
    // the second round's context  // retired-compat
    rc.merge_by_time = rc.g_on && cloud_merge_by_time;   // retired-compat
    rc.resample_by_time = rc.r_on && cloud_resample_by_time;   // retired-compat
    rc.r_device = rc.r_on && cloud_resample_device;   // retired-compat
    rc.split_resample = rc.clouds && cloud_split_resample;   // retired-compat
    rc.split_mg = rc.split && cloud_split_velocity == 1;   // retired-compat
    rc.split_range = rc.split_mg && split_gb_mode == 1;   // retired-compat
    rc.shear3d = rc.clouds && shear3d;   // retired-compat
    rc.multirank = global_variable::nranks > 1;   // retired-compat
    rc.lb_weight = lb_cloud_weight > 0.0;   // retired-compat
    rc.dc1 = rc.clouds && cloud_dc1;   // retired-compat
    rc.launch_bounds = DUST_CLOUD_LAUNCH_BOUNDS;   // retired-compat
    rc.qr_columns = (DUST_R_QR_COLUMNS != 0);   // retired-compat
    // the analytic gather was read only by streaming_linear (a built-in pgen)  // retired-compat
#if !USER_PROBLEM_ENABLED
    rc.streaming_linear = pin->DoesParameterExist("problem", "pgen_name") &&
                          pin->GetString("problem", "pgen_name") == "streaming_linear";
#endif
    CheckRetiredKeys(pin, rc);
  }
  // the coupled solve of point particles (SolveCoupledStage) runs only with back-reaction and a
  // coupling whose stages reach it (imex): only then its work fields
  // (solver_ap; solver_r / solver_p for pcg / adaptive) and exchange objects (passive dust gets
  // none, also with a given pcg / adaptive).  The clouds' dc1 needs none of these
  // (dust_imex3d_dc.cpp: cloud_dc1, also only with back-reaction)
  const bool point_solve = drag_solver != DustDragSolver::local && !UsesRefinableClouds() &&
      back_reaction && coupling == DustCoupling::imex;
  if (point_solve) {
    Kokkos::realloc(solver_ap, nmb, 3, ncells3, ncells2, ncells1);
    if (drag_solver == DustDragSolver::pcg || drag_solver == DustDragSolver::adaptive) {
      Kokkos::realloc(solver_r, nmb, 3, ncells3, ncells2, ncells1);
      Kokkos::realloc(solver_p, nmb, 3, ncells3, ncells2, ncells1);
    }
  }

  // (4) allocate boundary communication objects -----------------------------------------
  pbval_qp = new MeshBoundaryValuesDep(pmy_pack, pin);
  pbval_qp->InitializeBuffers(5);
  pbval_dm = new MeshBoundaryValuesDep(pmy_pack, pin);
  pbval_dm->InitializeBuffers(3);
  pbval_us = new MeshBoundaryValuesCC(pmy_pack, pin, false);
  pbval_us->InitializeBuffers(3);
  if (point_solve) {
    // These objects own distinct communicators/requests and are reinitialized for every
    // matrix-free matvec; they must not alias the one-shot stage exchanges above.  The copy
    // exchange: this plain one on a periodic mesh, the blocking u* exchange (pbval_dc below) in a
    // 3-D shearing box or with physical x3 faces (PointCopyExchange); one of the two only
    if (!point_dc_general) {
      pbval_solver_copy = new MeshBoundaryValuesCC(pmy_pack, pin, false);
      pbval_solver_copy->InitializeBuffers(3);
    }
    pbval_solver_add = new MeshBoundaryValuesDep(pmy_pack, pin);
    pbval_solver_add->InitializeBuffers(3);
  }
  // shear-periodic remap of the u* radial ghost zones (3D shearing box only)
  if (shear_x1 && pmy_pack->pmesh->three_d) {
    psbox_us = new ShearingBoxCC(pmy_pack, pin, 3);
    // collected u* ghost zones: the u* prolongation stays out of the slots
    // across the shear-periodic faces, as for the gas
    if (psbox_us->face_levels) pbval_us->skip_shear_x1_prol = true;
    // with refinement also the coarse u* array (cdummy), before its prolongation
    if (pmy_pack->pmesh->multilevel) {
      psbox_us->coarse_on = true;
      psbox_us->AllocateBuffers();
    }
  }
  // the clouds' dc1 (and the point solve in a shearing box / with physical x3 faces): the
  // blocking u* exchange of the correction has its own objects (own communicators, requests and
  // buffers; made in the same order on every rank) and coarse buffer, set up as those of u*
  const bool dc_exchange = cloud_dc1 || (point_solve && point_dc_general);
  if (dc_exchange) {
    pbval_dc = new MeshBoundaryValuesCC(pmy_pack, pin, false);
    pbval_dc->InitializeBuffers(3);
    pbval_dc->skip_shear_x1_prol = pbval_us->skip_shear_x1_prol;
    if (psbox_us != nullptr) {
      psbox_dc = new ShearingBoxCC(pmy_pack, pin, 3);
      if (psbox_us->coarse_on) {
        psbox_dc->coarse_on = true;
        psbox_dc->AllocateBuffers();
      }
    }
    dc_cfield = DvceArray5D<Real>("dust_dc_cfield", cdummy.extent(0), cdummy.extent(1),
                                  cdummy.extent(2), cdummy.extent(3), cdummy.extent(4));
  }
  // in 3-D with refinement or with shear-periodic x1 faces the ghost
  // deposits go through the general remap (any levels, several ranks, the shear offset)
  // not for 3-D refinable clouds: they deposit into the leaves directly
  if (pmy_pack->pmesh->three_d && cloud_dim != 3 &&
      (pmy_pack->pmesh->multilevel || shear_x1 || x3_phys)) {
    if ((drag_solver != DustDragSolver::local && !point_dc_general) ||
        !(smr_transfer_mode == DustSMRTransfer::instant_switch ||
          smr_transfer_mode == DustSMRTransfer::none)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "3-D dust with refinement or a shearing box: drag_solver = local and "
                << "point particles (amr_pic = instant_switch on a refined mesh)" << std::endl;
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
    // ghost deposits beyond physical x3 faces: fold back (default) or drop
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
    // (point dc1 / dc2: the operator's additive exchange remaps as the deposits do)
    if (pbval_solver_add != nullptr) {
      pbval_solver_add->SetGeneralPlan(dep_plan);
      if (x3mode == 2) pbval_solver_add->dropped_sum.assign(3, 0.0);
    }
    if (x3mode == 2) pbval_dm->dropped_sum.assign(3, 0.0);   // the record from the start
  }
  // instant_switch on a dynamic mesh: the block boxes and the block table of the
  // initial mesh (the exchanges above built their remaps); 2-D only
  if (pmy_pack->pmesh->adaptive && smr_transfer_mode == DustSMRTransfer::instant_switch &&
      !pmy_pack->pmesh->three_d) {
    BuildOwnerOverlapMesh(false);
  }
  // the storage of the coupled solve / the dc1 correction (rank 0's share), once at start-up
  if (drag_solver != DustDragSolver::local && global_variable::my_rank == 0) {
    const double mib = 1.0/(1024.0*1024.0);
    std::ostringstream os;
    os << std::setprecision(4) << "# NOTE (dust): drag_solver storage on rank 0: ";
    if (!point_solve && !dc_exchange) {
      os << "none (" << (back_reaction ? "the coupled solve does not run in this coupling" :
                                         "back_reaction = false: no coupled solve") << ")";
    } else {
      // (solver_ap, and solver_r / solver_p of pcg / adaptive; 1-element placeholders otherwise)
      const double work = point_solve ? (solver_ap.span() +
          ((drag_solver == DustDragSolver::pcg || drag_solver == DustDragSolver::adaptive) ?
           solver_r.span() + solver_p.span() : 0))*sizeof(Real)*mib : 0.0;
      os << "work fields " << work << " MiB, copy exchange "
         << (pbval_solver_copy != nullptr ? "plain" :
             (pbval_dc != nullptr ? "blocking u*" : "none"))
         << ", additive exchange " << (pbval_solver_add != nullptr ? "yes" : "no")
         << ", blocking u* exchange " << (pbval_dc != nullptr ? "yes" : "no")
         << (psbox_dc != nullptr ? " (+ shear)" : "") << " (coarse buffer "
         << dc_cfield.span()*sizeof(Real)*mib << " MiB)";
    }
    std::cout << os.str() << std::endl;
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
    g << " (quantum, stage X "  // retired-compat
      << (cloud_merge_stencil ? "on" : "off") << ", alpha " << cloud_merge_alpha;
    if (cloud_dim == 3) g << ", alpha_leaf " << (cloud_merge_alpha_leaf ? "on" : "off");
    g << ", budget beta " << cloud_merge_budget << " kappa " << cloud_merge_budget_kappa << ", "
      << sched(cloud_merge_by_time, cloud_merge_dt, cloud_merge_interval) << ")";
  }
  r << "R " << (cloud_resample ? "on" : "off");
  if (cloud_resample) {
    // eta_L of the levels of this mesh
    const int nshow = std::min<int>(ResampleCapTable::kLevels,
                                    pmy_pack->pmesh->max_level - pmy_pack->pmesh->root_level + 1);
    std::ostringstream el;
    for (int l=0; l<nshow; ++l) el << (l ? " " : "") << std::setprecision(4) << cloud_resample_eta_lev[l];
    r << " (eta " << cloud_resample_eta << ", per level " << el.str() << ", cold gate alpha "
      << cloud_resample_cold_alpha
      << ", " << sched(cloud_resample_by_time, cloud_resample_dt, cloud_resample_interval)
      << ", after regrids " << (cloud_resample_after_regrid ? "on" : "off") << ")";
  }
  std::cout << "# DUST_PIC amr_pic = refinable_cloud, refinable_mode = " << mode << std::endl
            << "#   split: moving P1 (" << (cloud_split_enable ? "on" : "off") << "), gather order "
            << 2 << ", transition " << 2 << "; " << g.str()  // retired-compat (log format)
            << std::endl
            << "#   " << r.str() << std::endl;
  if (cloud_split_budget > 0 || cloud_split_resample) {
    std::cout << "#   split budget " << cloud_split_budget << " clouds per rank and cycle"
              << (cloud_split_budget > 0 ? "" : " (no limit)") << "; R after a split "
              << (cloud_split_resample ? "on" : "off");
    if (cloud_split_resample) {
      std::cout << " (when a rank with at least the mean clouds per rank created "
                << cloud_split_resample_frac << " of its clouds, or held events back)";
    }
    std::cout << std::endl;
  }
  if (!refinable_given.empty()) {
    std::cout << "#   given explicitly:" << refinable_given << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::NoteLocalNotDefault
//! \brief coupling = imex with drag_solver = local where dc1 is the default of the configuration:
//! rank 0 says where local came from (a restart keeps the stored solver: a PC2 run, or one
//! started before dc1 was the default, stored local) and how to switch.

void DustGasDrag::NoteLocalNotDefault() const {
  if (global_variable::my_rank != 0) return;
  std::cout << "# NOTE (dust): coupling = imex with drag_solver = local (" << solver_src_
            << "): the lumped IMEX drag without defect correction (first order with "
            << "back-reaction), while this configuration's default is dc1; a restart keeps the "
            << "stored solver, give dust/drag_solver=dc1 to switch" << std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckMultiRank
//! \brief With several MPI ranks, refinable clouds run only the production path
//! that the dust halo covers (PC2 linear gather, quantum G with or without the pass budget,
//! device or host R); every other option stops here until it has its exchange.

void DustGasDrag::CheckMultiRank() {
  if (global_variable::nranks == 1) return;
  std::vector<std::string> bad;
  if (drag_solver != DustDragSolver::local && drag_solver != DustDragSolver::dc1 &&
      drag_solver != DustDragSolver::dc2) {
    bad.push_back("drag_solver other than local, dc1 or dc2");
  }
  // stage X and the structure-relative budget run on several ranks (X merges
  // the families whose nine members are on this rank; the budget's neighbor leaves come from
  // the halo)
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
  if (global_variable::my_rank == 0 && point_sort_interval > 0) {
    std::cout << std::setprecision(6) << "# DUST_POINT_SORT interval=" << point_sort_interval
              << " sorts=" << point_sort_count << " seconds=" << point_sort_seconds
              << " (rank 0, fenced)" << std::endl;
  }
  if (global_variable::my_rank == 0 && StiffnessActive()) {
    std::cout << std::setprecision(6) << "# DUST_STIFFNESS max_chi=" << stiff_chi_run_max
              << " last_chi=" << stiff_chi_last << " stiff_thresh=" << stiff_thresh
              << " pc2_feedback_damping=" << (pc2_feedback_damping ? 1 : 0) << std::endl;
  }
  if (global_variable::my_rank == 0 && (prof_count[kProfAmr] > 0 || prof_count[kProfRefineClouds] > 0)) {
    // rank 0's fenced phase times (s) and calls
    const char *names[kProfN] = {"trigger", "amr_total", "regrid_mesh", "regrid_dust_rebuild", "refine_clouds",
                                 "split", "regrid_event_GR", "scheduled_R", "scheduled_G", "sort", "cell_order",
                                 "trigger_deposit", "trigger_copy", "trigger_blocks", "trigger_log",
                                 "split_R"};
    std::cout << "# DUST_AMR_PROFILE";
    for (int k=0; k<kProfN; ++k) {
      std::cout << " " << names[k] << "=" << std::setprecision(6) << prof_time[k] << "(" << prof_count[k] << ")";
    }
    std::cout << std::endl;
  }
  // printed once a profiled R has run (the patch count; the last
  // entry, the negative basic solutions, can stay 0)
  if (global_variable::my_rank == 0 && ProfOn() && !DUST_R_DIAG && cloud_resample && cloud_resample_device) {
    std::cout << "# DUST_R_STAGES not compiled (build with -DDUST_R_DIAG=1)" << std::endl;
  }
  if (global_variable::my_rank == 0 && resample::StageTotals()[resample::kStagePatches] > 0) {
    // device clock cycles of the patch solves by stage, summed over the teams of
    // rank 0 (CUDA only; 0 on other backends)
    const unsigned long long *t = resample::StageTotals();
    std::cout << "# DUST_R_STAGES (rank 0, device clock cycles, summed over the teams) rows=" << t[0]
              << " qr=" << t[1] << " null=" << t[2] << " steps=" << t[3] << " stream=" << t[4]
              << " stream_rest=" << (t[4] - t[1] - t[2] - t[3]) << " check=" << t[5]
              << " patches=" << t[6] << std::endl;
    // the basic solution on the pivot columns
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
              << (cloud_split_max_level >= 0 ? " cloud_instant_switches=" +
                                                  std::to_string(cloud_instant_switch_count)
                                          : std::string())
              << (cloud_split_max_level >= 0 ? " root_mass=" + [this] {
                    // the root-cloud masses R and G used (recorded before the first switch)
                    std::ostringstream o;
                    o << std::setprecision(17);
                    for (std::size_t s=0; s<cloud_merge_mroot.size(); ++s) {
                      o << (s ? "," : "") << cloud_merge_mroot[s];
                    }
                    return o.str();
                  }() : std::string())
              << (cloud_split_budget > 0 ? " split_held=" + std::to_string(cloud_split_held)
                                         : std::string())
              << (cloud_split_resample ? " split_resamples=" + std::to_string(split_resample_calls)
                                       : std::string())
              << (cloud_merge ? " coarsen_events=" + std::to_string(cloud_coarsen_events) +
                  " coarsen_in=" + std::to_string(cloud_coarsen_in) +
                  " coarsen_reject=" + std::to_string(cloud_coarsen_reject) +  // retired-compat
                  " far_releases=" + std::to_string(cloud_far_release) : std::string())
              // several ranks: the counters are rank 0's, the particles all ranks'
              << " particles=" << ((global_variable::nranks > 1) ? pmy_pack->pmesh->nprtcl_total
                                                                  : pmy_pack->ppart->nprtcl_thispack)
              << ((global_variable::nranks > 1) ? " counters=rank0" : "") << std::endl;
  }
  SplitVelocitySummary();   // (cloud_split_velocity = mesh_gradient) one line per rank
  if (smr_transfer_mode == DustSMRTransfer::refinable_cloud &&
      (cloud_order_interval > 0 || cloud_sort_interval > 0)) {
    // every rank: launches of the cloud kernels in the index order and in storage order (the
    // index longer than the particles after a migration: none, the index is
    // kept; keep=1 printed for the log parsers), index builds and sorts
    std::ostringstream os;
    os << "# DUST_CLOUD_ORDER rank=" << global_variable::my_rank << " ordered=" << cloud_ord_used
       << " storage_order=" << cloud_ord_fallback << " index_builds=" << cloud_ord_builds
       << " sorts=" << cloud_sort_count << " keep=" << 1
       << " mean_particles=" << ord_stat_npart/std::max(1LL, ord_stat_cycles)
       << " mean_blocks=" << ord_stat_nmb/std::max(1LL, ord_stat_cycles) << "\n";
    std::cout << os.str() << std::flush;
  }
  if (global_variable::my_rank == 0 && shear3d) {   // shear-image list counters
    std::cout << "# DUST_SHEAR_LISTS rank0 ysh_updates=" << shear3d_updates << " list_rebuilds="
              << shear3d_rebuilds << " cache=on" << std::endl;
  }
  if (global_variable::my_rank == 0 && global_variable::nranks > 1 && ProfOn() &&
      smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
    // rank 0's seconds in the multi-rank parts of the dust (with amr_profile)
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
              << " (strips)" << std::endl;
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
  if (pbval_dc != nullptr) delete pbval_dc;
  if (global_variable::my_rank == 0 && cloud_dc1 && dc_prof_n_[0] > 0) {
    // (<dust>/amr_profile only: fenced phases of the dc1 correction on rank 0)
    std::cout << "# DC1_PROFILE rank 0: exchange " << dc_prof_t_[0] << " s (" << dc_prof_n_[0]
              << "), slope pack + forward " << dc_prof_t_[1] << " s, operator kernel "
              << dc_prof_t_[2] << " s, reverse add " << dc_prof_t_[3] << " s, correction "
              << dc_prof_t_[4] << " s" << std::endl;
  }
  if (psbox_dc != nullptr) delete psbox_dc;
  // the general deposit remap (after the exchanges that point to it); its destructor joins a
  // shear prefetch thread that may still run
  if (dep_plan != nullptr) {delete dep_plan; dep_plan = nullptr;}
}

//----------------------------------------------------------------------------------------
//! \fn bool DustGasDrag::ActiveStage
//! \brief Returns false on stages where all dust work is dormant. For imex2+ the final
//! explicit stage is a trivial assembly step (beta=0, gam0=1, gam1=0, and the implicit
//! solve is skipped since estage == nexp_stages), so every dust task no-ops there.

bool DustGasDrag::ActiveStage(Driver *pdrive, int stage) const {
  if (coupling == DustCoupling::pc2) return true;
  return !((pdrive->integrator.compare("imex2+") == 0) && (stage == pdrive->nexp_stages));
}

bool DustGasDrag::MigrationActive(Driver *pdrive, int stage) const {
  if (coupling == DustCoupling::imex) return ActiveStage(pdrive, stage);
  return stage == 2;   // (PC2: once per cycle, after the advance)
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ShearRouteTime, ShiftShearCrossings
//! \brief AthenaK's particle routing (SetNewPrtclGID) moves a particle across a
//! shear-periodic x1 face by the shear offset of pm->time = t^n, but the push has already moved it
//! on.  PC2 migrates once, after the full push: the move needs the offset of t^n + dt.
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
  if (coupling == DustCoupling::pc2) return pmy_pack->pmesh->dt;
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
  Real ppc = static_cast<Real>(RootPpc(pin));   // per-species keys: sum of ppc_N

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &taus_ = taus;
  auto gids = pmy_pack->gids;
  int nspec = nspecies;
  Real d2g = dust_to_gas;
  // the per-species keys: every run of ppc_total consecutive particles of a block holds
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
