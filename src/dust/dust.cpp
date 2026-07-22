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
  // Driver is constructed after physics modules, so read the integrator from the input
  std::string integrator = pin->GetOrAddString("time", "integrator", "rk2");
  if (integrator.compare("imex2+") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust drag requires <time>/integrator = imex2+ (imex2/imex3 would need "
              << "implicit pre-stages that are not implemented)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmy_pack->pmesh->multilevel) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust drag does not support SMR/AMR" << std::endl;
    std::exit(EXIT_FAILURE);
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
  stopping_times_initialized = false;
  dt_cfl        = pin->GetOrAddReal("dust","dt_cfl",0.5);
  dust_to_gas   = pin->GetOrAddReal("dust","dust_to_gas",0.01);

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
  for (int s=0; s<nspecies; ++s) {
    Real ts = pin->GetReal("dust", "taus_" + std::to_string(s+1));
    if (ts <= 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Dust stopping times must be positive" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    taus.h_view(s) = ts;
    taus_max = std::max(taus_max, ts);
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
  Kokkos::realloc(qdep,  nmb, 4, ncells3, ncells2, ncells1);
  Kokkos::realloc(ustar, nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::realloc(dmom,  nmb, 3, ncells3, ncells2, ncells1);
  Kokkos::deep_copy(dmom, 0.0);  // read as R_g=0 in stage 2 if back_reaction is off
  if (drag_solver != DustDragSolver::local) {
    Kokkos::realloc(solver_r,  nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(solver_p,  nmb, 3, ncells3, ncells2, ncells1);
    Kokkos::realloc(solver_ap, nmb, 3, ncells3, ncells2, ncells1);
  }

  // (4) allocate boundary communication objects -----------------------------------------
  pbval_qp = new MeshBoundaryValuesDep(pmy_pack, pin);
  pbval_qp->InitializeBuffers(4);
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
  if (global_variable::my_rank == 0 && solver_stage_count > 0) {
    std::vector<int> iterations = solver_pcg_iterations;
    std::sort(iterations.begin(), iterations.end());
    auto quantile = [&](double fraction) {
      if (iterations.empty()) return 0;
      if (fraction <= 0.0) return iterations.front();
      std::size_t index = static_cast<std::size_t>(
          std::ceil(fraction*static_cast<double>(iterations.size())) - 1.0);
      index = std::min(index, iterations.size() - 1);
      return iterations[index];
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
  return !((pdrive->integrator.compare("imex2+") == 0) && (stage == pdrive->nexp_stages));
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
