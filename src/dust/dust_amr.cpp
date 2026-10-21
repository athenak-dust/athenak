//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_amr.cpp
//! \brief Experimental dynamic mesh for the refinable-cloud dust (AMR
//! stage A: the particles are kept as they are, only the mesh changes).  2-D, one MPI
//! rank, ordinary periodic boundaries; opt-in with <dust>/experimental_amr = true.
//!
//! AthenaK regrids at the end of a cycle, after every stage, the time increment and the
//! outputs (Driver::Execute), so a regrid never falls between the predictor and the
//! corrector of the PC2 drag coupling.  The gas is restricted / prolongated and the
//! MeshBlocks are renumbered by MeshRefinement; nothing there knows about particles.
//! RebuildAfterRegrid() is what the dust module does right afterwards:
//!   - the maps from the virtual finest grid to the active leaves are rebuilt;
//!   - (the merge masks are evaluated on the actual leaves at their use: nothing
//!     cached to mark stale);
//!   - every particle's MeshBlock id (PGID) is set again from its position;
//!   - the particle time step is computed again with the new cell sizes;
//!   - the regrid is marked as an event (blocks created / deleted): the next cycle runs its
//!     split, then a merge pass if one is due or blocks were deleted, then one resampling
//!     (HandleRegridEvent in dust_cloud.cpp).
//! Tag, mass, position, velocity, cloud level and lobes of a particle are not touched: a
//! coarser mesh does not make a cloud wider, a finer mesh does not split it (that is the
//! split's decision at the next cycle, with the new maps).
//!
//! ScheduledRefinement() is a refinement request for tests: rectangles with a time window
//! in which their blocks are wanted at a given level.
//!
//! DensityRefinement() is the minimal dust-density request: at the
//! check (end of the cycle, after the outputs, before the regrid) the dust mass of the
//! particles at their new positions is deposited on the current leaves with the exact cloud
//! integrals (the transfer of the coupling and of the field outputs).  amr_rho_map = block
//! (as the AthenaK criteria): a block is judged on the maximum of
//! rho_d / mean over its own leaf cells, widened by amr_rho_buffer cells of its own level
//! (periodic; the cells of neighbouring blocks, from the halo on several ranks).  (The root-grid
//! map, amr_rho_map = root, was removed.)  A block whose value
//! reaches amr_rho_refine is flagged for
//! refinement, one that stays below amr_rho_derefine (< amr_rho_refine, hysteresis) for
//! derefinement.
//! The sibling, 2:1 and refinement_interval rules of MeshRefinement apply as usual.  Each
//! check writes a line to dust_trigger_log.txt.
//!
//! For the comparison with the old transfer, the old owner_overlap
//! transfer also runs on the dynamic mesh, one rank.  Point particles, TSC on the block that
//! holds them, ghost deposits added by volume to the leaves they overlap (the remap of
//! MeshBoundaryValuesDep, rebuilt after every regrid); after a regrid the block of every
//! particle is set from its position (BuildOwnerOverlapMesh, ResetPGIDOwnerOverlap).  The
//! density trigger uses that deposit (OwnerOverlapDeposit) with the same block values.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/mesh_refinement.hpp"
#include "driver/driver.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"

namespace dust {

namespace {
// the refinement hook of the problem generator is a plain function pointer, so the
// schedule is kept here (set once from the input by SetupScheduledRefinement)
struct SchedPatch {
  Real t0, t1;                          // wanted for t0 <= t < t1
  Real x1min, x1max, x2min, x2max;      // blocks whose centre lies inside
  int level;                            // levels above the root
};
std::vector<SchedPatch> sched_;

// the dust-density request (SetupDensityTrigger)
struct DensityTrigger {
  Real rho_refine = 0.0, rho_derefine = 0.0;   // in units of the mean dust density
  int buffer = 0;                               // cells of the block's level
  // The thresholds grow by level_factor per level (amr_rho_level_factor): a
  // block of level l is refined at rho_refine f^l and derefined below rho_derefine f^(l - 1); 4 is
  // a fixed dust mass per cell (2-D, the old amr_rho_level_scaling = mass), 1 the same thresholds
  // on every level
  Real level_factor = 1.0;
  // With map = block, the block values on the device (default) or on the host; check:
  // both, every value compared (diagnostic)
  bool device = true, device_check = false;
};
DensityTrigger trig_;
// Per-block audit of the trigger
bool audit_ = false;
int audit_max_ = 40, audit_dumps_ = 0;
std::vector<long long> audit_prev_;   // (level, lx1, lx2, applied flag) of the previous check, flattened

// Order-independent 64-bit sum over the particles of a hash of every real column and of every
// integer column except PGID: equal before and after a regrid exactly when no particle was
// lost, duplicated or changed in any bit (up to hash collisions).
unsigned long long ParticleChecksum(particles::Particles *ppar) {
  static_assert(sizeof(Real) == sizeof(unsigned long long), "checksum assumes 64-bit Real");
  const int npart = ppar->nprtcl_thispack;
  if (npart == 0) return 0ULL;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  unsigned long long sum = 0ULL;
  Kokkos::parallel_reduce("dust_amr_checksum", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, unsigned long long &s) {
    union {Real r; unsigned long long u;} bits;
    unsigned long long h = 1469598103934665603ULL;          // FNV-1a over 64-bit words
    for (int q=0; q<nrdata; ++q) {
      bits.r = pr(q,p);
      h = (h ^ bits.u)*1099511628211ULL;
    }
    for (int q=0; q<nidata; ++q) {
      if (q == PGID) continue;
      h = (h ^ static_cast<unsigned long long>(static_cast<unsigned int>(pi(q,p))))*1099511628211ULL;
    }
    s += h;
  }, Kokkos::Sum<unsigned long long>(sum));
  return sum;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn dust::SetupScheduledRefinement
//! \brief Read <dust>/amr_sched_n and, for k = 1..n, amr_schedK_t0, _t1, _x1min, _x1max,
//! _x2min, _x2max, _level (default 1).  Returns the number of patches.

int SetupScheduledRefinement(ParameterInput *pin) {
  sched_.clear();
  const int n = pin->GetOrAddInteger("dust", "amr_sched_n", 0);
  for (int k=1; k<=n; ++k) {
    const std::string b = "amr_sched" + std::to_string(k) + "_";
    SchedPatch s;
    s.t0 = pin->GetReal("dust", b + "t0");
    s.t1 = pin->GetReal("dust", b + "t1");
    s.x1min = pin->GetReal("dust", b + "x1min");
    s.x1max = pin->GetReal("dust", b + "x1max");
    s.x2min = pin->GetReal("dust", b + "x2min");
    s.x2max = pin->GetReal("dust", b + "x2max");
    s.level = pin->GetOrAddInteger("dust", b + "level", 1);
    sched_.push_back(s);
  }
  return n;
}

//----------------------------------------------------------------------------------------
//! \fn dust::ScheduledRefinement
//! \brief Refinement request (<amr_criterionN>/method = user): a block is flagged for
//! refinement when a patch that is active at the current time and contains its centre
//! wants a finer level, for derefinement when every such patch (or none) wants a coarser
//! one.  The request is a function of time and place only, so it is repeated every check
//! until the mesh has it; MeshRefinement applies its own 2:1 and sibling rules.

void ScheduledRefinement(MeshBlockPack *pmbp) {
  if (sched_.empty()) return;   // no request: the flags stay zero
  Mesh *pm = pmbp->pmesh;
  auto &refine_flag = pm->pmr->refine_flag;
  auto &indcs = pm->mb_indcs;
  auto &size = pmbp->pmb->mb_size;
  const int nmb = pmbp->nmb_thispack;
  const int mbs = pm->gids_eachrank[global_variable::my_rank];
  for (int m=0; m<nmb; ++m) {
    const auto &sz = size.h_view(m);
    const Real xc = sz.x1min + 0.5*indcs.nx1*sz.dx1;
    const Real yc = sz.x2min + 0.5*indcs.nx2*sz.dx2;
    int want = 0;
    for (const auto &s : sched_) {
      if (pm->time >= s.t0 && pm->time < s.t1 && xc > s.x1min && xc < s.x1max &&
          yc > s.x2min && yc < s.x2max) {
        want = std::max(want, s.level);
      }
    }
    const int lev = pm->lloc_eachmb[m + mbs].level - pm->root_level;
    refine_flag.h_view(m + mbs) = (want > lev) ? 1 : ((want < lev) ? -1 : 0);
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
}

//----------------------------------------------------------------------------------------
//! \fn dust::SetupDensityTrigger
//! \brief Read <dust>/amr_rho_refine, amr_rho_derefine (< amr_rho_refine) and
//! amr_rho_buffer (root cells, default 2) for DensityRefinement.

void SetupDensityTrigger(ParameterInput *pin) {
  auto fatal = [](const std::string &msg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << msg << std::endl;
    std::exit(EXIT_FAILURE);
  };
  if (!sched_.empty()) fatal("<dust>/amr_density_trigger and amr_sched_n > 0 exclude each other");
  trig_.rho_refine = pin->GetReal("dust", "amr_rho_refine");
  trig_.rho_derefine = pin->GetReal("dust", "amr_rho_derefine");
  trig_.buffer = pin->GetOrAddInteger("dust", "amr_rho_buffer", 2);
  {
    // Rixin: each block on its own leaf cells, as the AthenaK criteria.
    // The root-grid map (root) is gone
    RemovedKey(pin, "amr_rho_map",
               !pin->DoesParameterExist("dust", "amr_rho_map") ||
               pin->GetString("dust", "amr_rho_map") != "root",
               "the root-grid map of the density trigger");
    const std::string mp = pin->GetOrAddString("dust", "amr_rho_map", "block");
    if (mp != "block") fatal("<dust>/amr_rho_map must be block");
  }
  {
    // The thresholds grow by amr_rho_level_factor per level (1:
    // the same on every level; 4: a fixed dust mass per cell in 2-D; 2 follows the growth of the
    // peak density with resolution in the 2048^2 run).  The older
    // amr_rho_level_scaling (none / mass) is folded into it: none is factor 1, mass factor 4 (2-D,
    // the trigger is 2-D only), the same threshold rule.  An input with the old key only gets that
    // factor (one note).  With both keys the factor wins, as before the fold (Codex review of the
    // cleanup: mass next to factor 2 was a valid input); a note says the alias is ignored.
    Real fsc = 1.0;
    const bool has_sc = pin->DoesParameterExist("dust", "amr_rho_level_scaling");
    const bool has_f = pin->DoesParameterExist("dust", "amr_rho_level_factor");
    if (has_sc) {
      const std::string sc = pin->GetString("dust", "amr_rho_level_scaling");
      if (sc == "mass") {
        fsc = 4.0;
      } else if (sc != "none") {
        fatal("<dust>/amr_rho_level_scaling (folded into amr_rho_level_factor) must be none or mass");
      }
      if (has_f && pin->GetReal("dust", "amr_rho_level_factor") != fsc &&
          global_variable::my_rank == 0) {
        std::cout << "# NOTE (dust): amr_rho_level_scaling = " << sc << " is ignored, the given "
                  << "amr_rho_level_factor = " << pin->GetString("dust", "amr_rho_level_factor")
                  << " is used (as before the key was folded into it on 2026-10-20)" << std::endl;
      }
      if (!has_f && global_variable::my_rank == 0) {
        std::cout << "# NOTE (dust): amr_rho_level_scaling = " << sc << " is read as "
                  << "amr_rho_level_factor = " << fsc << " (the key was folded into "
                  << "amr_rho_level_factor on 2026-10-20)" << std::endl;
      }
    }
    trig_.level_factor = pin->GetOrAddReal("dust", "amr_rho_level_factor", fsc);
    if (!(trig_.level_factor >= 1.0)) fatal("<dust>/amr_rho_level_factor must be >= 1");
  }
  trig_.device = pin->GetOrAddBoolean("dust", "amr_rho_device", true);
  trig_.device_check = pin->GetOrAddBoolean("dust", "amr_rho_device_check", false);
  audit_ = pin->GetOrAddBoolean("dust", "amr_trigger_audit", false);
  audit_max_ = pin->GetOrAddInteger("dust", "amr_trigger_audit_max", 40);
  if (!(trig_.rho_derefine >= 0.0 && trig_.rho_derefine < trig_.rho_refine) || trig_.buffer < 0) {
    fatal("amr_density_trigger needs 0 <= amr_rho_derefine < amr_rho_refine and amr_rho_buffer >= 0");
  }
  // several ranks: the buffer is part of the reach of the halo (DustGasDrag::HaloReach), so the
  // strips hold the buffer cells; BlockValue checks that the received cells cover it
  if (global_variable::my_rank == 0) {
    std::cout << "# dust density trigger: refine at rho_d/mean >= " << trig_.rho_refine
              << ", derefine below " << trig_.rho_derefine << ", buffer " << trig_.buffer
              << " cells of the block's level, each block on its own leaf cells"
              << "; thresholds x" << trig_.level_factor << " per level" << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn dust::DensityRefinement
//! \brief Refinement request from the deposited dust density (see the file header).

void DensityRefinement(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  DustGasDrag *pdust = pmbp->pdust;
  double dummy_t = 0.0; long long dummy_c = 0;   // for the profile timer
  ProfTimer prof_timer(pdust ? pdust->prof_time[DustGasDrag::kProfTrigger] : dummy_t,
                       pdust ? pdust->prof_count[DustGasDrag::kProfTrigger] : dummy_c);
  // The old owner_overlap transfer has its own deposit (TSC on the owner blocks, the
  // ghost parts added to the leaves they overlap); the block values below are the same
  const bool oo = (pdust != nullptr) &&
                  pdust->smr_transfer_mode == DustSMRTransfer::owner_overlap;
  if (pdust == nullptr || !(pdust->UsesCompositeTransfer() || oo) || pm->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "amr_density_trigger needs 2-D refinable-cloud or owner_overlap dust" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // several ranks (Codex review): the deposit below traverses the supports, so
  // the reach of the halo is checked first, before the leaf traversal is copied: after the
  // push (particles moved and may have changed owner) and after a restart whose pending mesh
  // check runs before any cycle
  pdust->CheckHaloReach("density trigger");
  // Fenced sub-phases (deposit with the halo sums and totals, copy to the host, the
  // per-block decisions on the host)
  using PClock = std::chrono::steady_clock;
  if (ProfOn()) Kokkos::fence();
  auto tlap = PClock::now();
  auto lap = [&](const int k) {   // (no fence and no time unless amr_profile)
    if (!ProfOn()) return;
    Kokkos::fence();
    const auto now = PClock::now();
    pdust->prof_time[k] += std::chrono::duration<double>(now - tlap).count();
    ++pdust->prof_count[k];
    tlap = now;
  };
  particles::Particles *ppar = pmbp->ppart;
  auto &refine_flag = pm->pmr->refine_flag;
  auto &ind = pm->mb_indcs;
  const int nmb = pmbp->nmb_thispack, mbs = pm->gids_eachrank[global_variable::my_rank];
  const int is = ind.is, js = ind.js, nx1 = ind.nx1, nx2 = ind.nx2;
  const int ncells1 = nx1 + 2*ind.ng, ncells2 = nx2 + 2*ind.ng;
  auto &mbsize = pmbp->pmb->mb_size;
  const int npart = ppar->nprtcl_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  auto transfer = pdust->smr_transfer;
  auto lt = pdust->leaf_traversal;   // the actual leaves
  // dust density on the current leaves from the particles at their new positions; several
  // ranks: leaves of the halo blocks too, added back to their owners
  {
    const int nf = std::max(1, lt.nfld*ncells2*ncells1);
    if (pdust->trig_f.extent_int(1) != nf) {
      pdust->trig_f = DvceArray2D<Real>("amr_rho", 1, nf);   // zero on allocation
    } else {
      Kokkos::deep_copy(pdust->trig_f, 0.0);
    }
  }
  auto f = pdust->trig_f;
  if (oo) {   // one rank, no halo (nfld = nmb): the leaf densities in the same flat layout
    pdust->OwnerOverlapDeposit(pdust->oo_rho, false);
    auto rho = pdust->oo_rho;
    const int ks = ind.ks;
    par_for("amr_rho_oo_copy", DevExeSpace(), 0, nmb*ncells2*ncells1 - 1, KOKKOS_LAMBDA(const int k) {
      const int m = k/(ncells2*ncells1), r = k % (ncells2*ncells1);
      f(0,k) = rho(m,0,ks,r/ncells1,r % ncells1);
    });
  } else {
  par_for("amr_rho_deposit", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    const Real mass = pr(IPM,p);
    ForEachCompositeWeight(transfer, lt, pr(IPX,p), pr(IPY,p), pi(PLEV,p), lt.Anchor(pi(PGID,p)),
                           [&](const int target, const Real w) {
      if (w == 0.0) return;
      const int tm = target/(ncells1*ncells2);
      DepositAdd(&f(0,target), mass*w/(lt.box(tm,2)*lt.box(tm,3)));
    });
  });
  pdust->HaloReverseAddFlat(f, 1);
  // map = block: the buffer cells of a block may lie on a halo block; the owners' sums there
  // (the strips hold every cell within the reach of the local blocks, at least 3 cells of the
  // root level and 6 of any finer level, checked below)
  if (trig_.buffer > 0) pdust->HaloForwardFlat(f, 1);
  }
  Real mtot = 0.0;
  int nlev[3] = {0, 0, 0};   // clouds at the root level, finer plain, lobed
  Kokkos::parallel_reduce("amr_rho_totals", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, Real &m, int &n0, int &n1, int &nl) {
    m += pr(IPM,p);
    const int code = pi(PLEV,p);
    if (code > 15) ++nl; else if ((code & 15) == 0) ++n0; else ++n1;
  }, mtot, nlev[0], nlev[1], nlev[2]);
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &mtot, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, nlev, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  lap(DustGasDrag::kProfTrigDeposit);
  // On the device the field stays there (the host copy only for the host path and
  // the comparison)
  const bool dev_blocks = trig_.device;
  const bool host_f = !dev_blocks || trig_.device_check;
  decltype(Kokkos::create_mirror_view_and_copy(HostMemSpace(), f)) fhf;
  if (host_f) fhf = Kokkos::create_mirror_view_and_copy(HostMemSpace(), f);
  lap(DustGasDrag::kProfTrigCopy);
  auto fh = [&](const int m, const int j, const int i) {return fhf(0, (m*ncells2 + j)*ncells1 + i);};
  const int nr1 = pm->mesh_indcs.nx1, nr2 = pm->mesh_indcs.nx2;
  const Real x0 = pm->mesh_size.x1min, y0 = pm->mesh_size.x2min;
  const Real hx = (pm->mesh_size.x1max - x0)/nr1, hy = (pm->mesh_size.x2max - y0)/nr2;
  const Real mean = mtot/((pm->mesh_size.x1max - x0)*(pm->mesh_size.x2max - y0));
  Real qmax = 0.0, mfine = 0.0, vfine = 0.0;
  if (!host_f && nmb > 0) {   // the largest value and the mass on finer blocks, on the device
    if (pdust->trig_fine.extent_int(0) != nmb) pdust->trig_fine = DvceArray1D<int>("amr_rho_fine", nmb);
    auto fine_d = pdust->trig_fine;
    auto fine_h = Kokkos::create_mirror_view(fine_d);
    for (int m=0; m<nmb; ++m) fine_h(m) = (pm->lloc_eachmb[m + mbs].level > pm->root_level) ? 1 : 0;
    Kokkos::deep_copy(fine_d, fine_h);
    auto mbsz = mbsize;
    Real qm = 0.0, mf = 0.0;
    Kokkos::parallel_reduce("amr_rho_qmax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nx2*nx1),
    KOKKOS_LAMBDA(const int k, Real &qq, Real &mm) {
      const int m = k/(nx1*nx2), r = k % (nx1*nx2), j = r/nx1, i = r % nx1;
      const Real v = f(0, (m*ncells2 + js + j)*ncells1 + is + i);
      qq = Kokkos::fmax(qq, v/mean);
      if (fine_d(m)) mm += v*mbsz.d_view(m).dx1*mbsz.d_view(m).dx2;
    }, Kokkos::Max<Real>(qm), mf);
    qmax = std::max(0.0, qm); mfine = mf;
  }
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    const bool fine = pm->lloc_eachmb[m + mbs].level > pm->root_level;
    if (!host_f) {
      if (fine) vfine += nx1*sz.dx1*nx2*sz.dx2;
      continue;
    }
    for (int j=0; j<nx2; ++j) {
      for (int i=0; i<nx1; ++i) {
        const Real q = fh(m, js+j, is+i)/mean;
        const int ri = std::min(nr1-1, static_cast<int>((sz.x1min + (i + 0.5)*sz.dx1 - x0)/hx));
        const int rj = std::min(nr2-1, static_cast<int>((sz.x2min + (j + 0.5)*sz.dx2 - y0)/hy));
        qmax = std::max(qmax, q);
        if (fine) mfine += fh(m, js+j, is+i)*sz.dx1*sz.dx2;
      }
    }
    if (fine) vfine += nx1*sz.dx1*nx2*sz.dx2;
  }
#if MPI_PARALLEL_ENABLED
  // the totals over all ranks
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &qmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    Real fv[2] = {mfine, vfine};
    MPI_Allreduce(MPI_IN_PLACE, fv, 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    mfine = fv[0]; vfine = fv[1];
  }
#endif
  // the maximum of rho_d / mean over the leaf cells (local and halo slots) that
  // overlap the block widened by the buffer, periodic images included; the widened box must be
  // covered by those slots (a buffer cell outside the halo would make the result depend on the
  // ranks)
  const Real lx = pm->mesh_size.x1max - x0, ly = pm->mesh_size.x2max - y0;
  const int nslot = lt.nfld;   // local and halo slots: the deposit f covers these
  decltype(Kokkos::create_mirror_view_and_copy(HostMemSpace(), lt.box)) boxh;
  if (host_f) boxh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lt.box);
  // a halo slot holds only the cells it receives (its strip, or the whole block)
  auto SlotRect = [&](const int s, int r[4]) {
    r[0] = 0; r[1] = nx1; r[2] = 0; r[3] = nx2;
    if (s >= nmb) {
      const auto &rc = pdust->halo_.rect[s - nmb];
      r[0] = rc[0]; r[1] = rc[1]; r[2] = rc[2]; r[3] = rc[3];
    }
  };
  // The slots binned by the root cells their received cells touch (generous ranges,
  // wrapped), so that a block looks only at the slots near its widened box instead of all of
  // them (blocks x slots x 9 images was most of the trigger's time); every slot that overlaps the
  // widened box through some periodic image is in one of the bins the box touches
  std::vector<int> bptr, bslot, stamp, cand;
  auto RootRange = [&](const Real a, const Real c, const Real org, const Real h, const int n,
                       int &k0, int &k1) {   // root cells [k0, k1] touched by [a, c], unwrapped
    k0 = static_cast<int>(std::floor((a - org)/h - 1.0e-9));
    k1 = static_cast<int>(std::ceil((c - org)/h + 1.0e-9)) - 1;
    if (k1 - k0 >= n) k1 = k0 + n - 1;   // wider than the domain: every cell once
  };
  if (host_f) {
    std::vector<int> cnt(nr1*nr2 + 1, 0);
    for (int pass=0; pass<2; ++pass) {
      if (pass == 1) {
        bptr.assign(nr1*nr2 + 1, 0);
        for (int c=0; c<nr1*nr2; ++c) bptr[c + 1] = bptr[c] + cnt[c];
        bslot.assign(bptr[nr1*nr2], 0);
        std::fill(cnt.begin(), cnt.end(), 0);
      }
      for (int s=0; s<nslot; ++s) {
        int r[4]; SlotRect(s, r);
        if (r[1] <= r[0] || r[3] <= r[2]) continue;
        int i0, i1, j0, j1;
        RootRange(boxh(s,0) + r[0]*boxh(s,2), boxh(s,0) + r[1]*boxh(s,2), x0, hx, nr1, i0, i1);
        RootRange(boxh(s,1) + r[2]*boxh(s,3), boxh(s,1) + r[3]*boxh(s,3), y0, hy, nr2, j0, j1);
        for (int j=j0; j<=j1; ++j) {
          for (int i=i0; i<=i1; ++i) {
            const int c = ((j % nr2 + nr2) % nr2)*nr1 + (i % nr1 + nr1) % nr1;
            if (pass == 1) bslot[bptr[c] + cnt[c]] = s;
            ++cnt[c];
          }
        }
      }
    }
    stamp.assign(nslot, -1);
  }
  auto BlockValue = [&](const int m, const Real bx0, const Real bx1, const Real by0, const Real by1,
                        const Real bdx, const Real bdy) {
    const Real ex0 = bx0 - trig_.buffer*bdx, ex1 = bx1 + trig_.buffer*bdx;
    const Real ey0 = by0 - trig_.buffer*bdy, ey1 = by1 + trig_.buffer*bdy;
    const Real tol = 1.0e-9*std::min(bdx, bdy);
    Real q = 0.0, area = 0.0;
    int nbad = 0;
    // the candidate slots: those binned in the root cells the widened box touches, each once
    cand.clear();
    {
      int i0, i1, j0, j1;
      RootRange(ex0, ex1, x0, hx, nr1, i0, i1);
      RootRange(ey0, ey1, y0, hy, nr2, j0, j1);
      for (int j=j0; j<=j1; ++j) {
        for (int i=i0; i<=i1; ++i) {
          const int c = ((j % nr2 + nr2) % nr2)*nr1 + (i % nr1 + nr1) % nr1;
          for (int e=bptr[c]; e<bptr[c + 1]; ++e) {
            const int s = bslot[e];
            if (stamp[s] != m) {stamp[s] = m; cand.push_back(s);}
          }
        }
      }
    }
    for (const int s : cand) {
      const Real sdx = boxh(s,2), sdy = boxh(s,3);
      int rr[4]; SlotRect(s, rr);
      const int r0 = rr[0], r1 = rr[1], r2 = rr[2], r3 = rr[3];
      for (int oy=-1; oy<=1; ++oy) {
        for (int ox=-1; ox<=1; ++ox) {
          const Real b0 = boxh(s,0) + ox*lx, d0 = boxh(s,1) + oy*ly;
          const Real a0 = b0 + r0*sdx, a1 = b0 + r1*sdx;   // the received cells
          const Real c0 = d0 + r2*sdy, c1 = d0 + r3*sdy;
          if (a1 <= ex0 + tol || a0 >= ex1 - tol || c1 <= ey0 + tol || c0 >= ey1 - tol) continue;
          area += (std::min(a1, ex1) - std::max(a0, ex0))*(std::min(c1, ey1) - std::max(c0, ey0));
          const int i0 = std::max(r0, static_cast<int>(std::floor((ex0 - b0)/sdx + 1.0e-9)));
          const int i1 = std::min(r1, static_cast<int>(std::ceil((ex1 - b0)/sdx - 1.0e-9)));
          const int j0 = std::max(r2, static_cast<int>(std::floor((ey0 - d0)/sdy + 1.0e-9)));
          const int j1 = std::min(r3, static_cast<int>(std::ceil((ey1 - d0)/sdy - 1.0e-9)));
          for (int j=j0; j<j1; ++j) {
            for (int i=i0; i<i1; ++i) {
              const Real v = fh(s, js+j, is+i);
              if (!std::isfinite(v)) {++nbad; continue;}   // std::max would drop a NaN
              q = std::max(q, v/mean);
            }
          }
        }
      }
    }
    const Real want = (ex1 - ex0)*(ey1 - ey0);
    if (std::fabs(area - want) > 1.0e-9*want || nbad > 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "amr_rho_map = block: the buffer of block " << m + mbs << " (" << trig_.buffer
                << " cells) is not covered by the local and halo blocks (covered area "
                << area << " of " << want << ", " << nbad << " non-finite cells)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return q;
  };
  // The block values on the device: one thread per (block, slot) pair, the same
  // tests and cells as BlockValue; per block the largest value (atomic max), the covered area and
  // the non-finite cells, then only these three numbers per block go to the host
  decltype(Kokkos::create_mirror_view(DvceArray1D<Real>())) qd_h, ad_h;
  decltype(Kokkos::create_mirror_view(DvceArray1D<int>())) bd_h;
  if (dev_blocks) {
    const int nh = nslot - nmb;
    if (pdust->trig_rect.extent_int(0) != std::max(1, nh)) {
      pdust->trig_rect = DvceArray2D<int>("amr_rho_rect", std::max(1, nh), 4);
    }
    auto rect_d = pdust->trig_rect;
    auto rect_h = Kokkos::create_mirror_view(rect_d);
    for (int h=0; h<nh; ++h) {
      for (int c=0; c<4; ++c) rect_h(h, c) = pdust->halo_.rect[h][c];
    }
    Kokkos::deep_copy(rect_d, rect_h);
    if (pdust->trig_q.extent_int(0) != std::max(1, nmb)) {
      pdust->trig_q = DvceArray1D<Real>("amr_rho_q", std::max(1, nmb));
      pdust->trig_area = DvceArray1D<Real>("amr_rho_area", std::max(1, nmb));
      pdust->trig_bad = DvceArray1D<int>("amr_rho_bad", std::max(1, nmb));
    } else {
      Kokkos::deep_copy(pdust->trig_q, 0.0);
      Kokkos::deep_copy(pdust->trig_area, 0.0);
      Kokkos::deep_copy(pdust->trig_bad, 0);
    }
    auto qd = pdust->trig_q, ad = pdust->trig_area;
    auto bd = pdust->trig_bad;
    auto box = lt.box;
    auto mbsz = mbsize;
    const int bb = trig_.buffer;
    // (block, slot) as a 2-D range: nmb*nslot overflows an int on large meshes (Codex review)
    Kokkos::parallel_for("amr_rho_blocks",
    Kokkos::MDRangePolicy<DevExeSpace, Kokkos::Rank<2>>({0, 0}, {nmb, nslot}),
    KOKKOS_LAMBDA(const int m, const int s) {
      const Real bdx = mbsz.d_view(m).dx1, bdy = mbsz.d_view(m).dx2;
      const Real ex0 = mbsz.d_view(m).x1min - bb*bdx, ex1 = mbsz.d_view(m).x1max + bb*bdx;
      const Real ey0 = mbsz.d_view(m).x2min - bb*bdy, ey1 = mbsz.d_view(m).x2max + bb*bdy;
      const Real tol = 1.0e-9*Kokkos::fmin(bdx, bdy);
      const Real sdx = box(s,2), sdy = box(s,3);
      int r0 = 0, r1 = nx1, r2 = 0, r3 = nx2;   // a halo slot: the cells it receives
      if (s >= nmb) {r0 = rect_d(s-nmb,0); r1 = rect_d(s-nmb,1); r2 = rect_d(s-nmb,2); r3 = rect_d(s-nmb,3);}
      Real q = 0.0, area = 0.0;
      int nbad = 0;
      for (int oy=-1; oy<=1; ++oy) {
        for (int ox=-1; ox<=1; ++ox) {
          const Real b0 = box(s,0) + ox*lx, d0 = box(s,1) + oy*ly;
          const Real a0 = b0 + r0*sdx, a1 = b0 + r1*sdx;
          const Real c0 = d0 + r2*sdy, c1 = d0 + r3*sdy;
          if (a1 <= ex0 + tol || a0 >= ex1 - tol || c1 <= ey0 + tol || c0 >= ey1 - tol) continue;
          area += (Kokkos::fmin(a1, ex1) - Kokkos::fmax(a0, ex0))*
                  (Kokkos::fmin(c1, ey1) - Kokkos::fmax(c0, ey0));
          const int i0 = Kokkos::max(r0, static_cast<int>(Kokkos::floor((ex0 - b0)/sdx + 1.0e-9)));
          const int i1 = Kokkos::min(r1, static_cast<int>(Kokkos::ceil((ex1 - b0)/sdx - 1.0e-9)));
          const int j0 = Kokkos::max(r2, static_cast<int>(Kokkos::floor((ey0 - d0)/sdy + 1.0e-9)));
          const int j1 = Kokkos::min(r3, static_cast<int>(Kokkos::ceil((ey1 - d0)/sdy - 1.0e-9)));
          for (int j=j0; j<j1; ++j) {
            for (int i=i0; i<i1; ++i) {
              const Real v = f(0, (s*ncells2 + js + j)*ncells1 + is + i);
              if (!Kokkos::isfinite(v)) {++nbad; continue;}
              q = Kokkos::fmax(q, v/mean);
            }
          }
        }
      }
      if (area > 0.0) Kokkos::atomic_add(&ad(m), area);
      if (q > 0.0) Kokkos::atomic_max(&qd(m), q);
      if (nbad > 0) Kokkos::atomic_add(&bd(m), nbad);
    });
    qd_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), qd);
    ad_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ad);
    bd_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), bd);
  }
  int nref = 0, nder = 0, nmis = 0;
  std::vector<Real> qblk(nmb, 0.0);
  std::vector<int> rawf(nmb, 0), appf(nmb, 0);
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    Real q = 0.0;
    if (dev_blocks) {
      const Real ex0 = sz.x1min - trig_.buffer*sz.dx1, ex1 = sz.x1max + trig_.buffer*sz.dx1;
      const Real ey0 = sz.x2min - trig_.buffer*sz.dx2, ey1 = sz.x2max + trig_.buffer*sz.dx2;
      const Real want = (ex1 - ex0)*(ey1 - ey0);   // as in BlockValue
      if (std::fabs(ad_h(m) - want) > 1.0e-9*want || bd_h(m) > 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "amr_rho_map = block: the buffer of block " << m + mbs << " (" << trig_.buffer
                  << " cells) is not covered by the local and halo blocks (covered area "
                  << ad_h(m) << " of " << want << ", " << bd_h(m) << " non-finite cells)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      q = qd_h(m);
      if (trig_.device_check) {
        const Real qh = BlockValue(m, sz.x1min, sz.x1max, sz.x2min, sz.x2max, sz.dx1, sz.dx2);
        if (qh != q) {
          if (nmis < 5) {
            std::cout << "# amr_rho_device_check: block " << m + mbs << " device " << q << " host "
                      << qh << std::endl;
          }
          ++nmis;
        }
      }
    } else {
      q = BlockValue(m, sz.x1min, sz.x1max, sz.x2min, sz.x2max, sz.dx1, sz.dx2);
    }
    const int lev = pm->lloc_eachmb[m + mbs].level;
    int flag = 0;
    // the thresholds of this block's level (amr_rho_level_factor)
    const int l = lev - pm->root_level;
    const Real fr = std::pow(trig_.level_factor, l);
    const Real fd = (l > 0) ? std::pow(trig_.level_factor, l - 1) : 1.0;
    if (q >= trig_.rho_refine*fr && lev < pm->max_level) flag = 1;
    else if (q < trig_.rho_derefine*fd && lev > pm->root_level) flag = -1;
    refine_flag.h_view(m + mbs) = flag;
    if (flag > 0) ++nref;
    if (flag < 0) ++nder;
    // what CheckForRefinement keeps after this request: no change within refinement_interval
    // cycles of the block's last change (its age was incremented before this call)
    qblk[m] = q; rawf[m] = flag;
    appf[m] = (pm->pmr->ncyc_since_ref(m + mbs) < pm->pmr->refinement_interval) ? 0 : flag;
  }
  lap(DustGasDrag::kProfTrigBlocks);
  if (nmis > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "amr_rho_device_check: " << nmis << " block values differ between device and host"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  // Audit: per-block lines every check; the pre-regrid state (blocks and particles)
  // at the checks whose applied flags differ from those of the previous check
  if (audit_ && global_variable::my_rank == 0) {
    std::vector<long long> cur;
    for (int m=0; m<nmb; ++m) {
      const auto &ll = pm->lloc_eachmb[m + mbs];
      cur.insert(cur.end(), {ll.level, ll.lx1, ll.lx2, appf[m]});
    }
    std::FILE *fb = std::fopen("dust_trigger_blocks.txt", "a");
    if (fb != nullptr) {
      std::fseek(fb, 0, SEEK_END);
      if (std::ftell(fb) == 0) {
        std::fprintf(fb, "# ncycle time level lx1 lx2 q raw_flag applied_flag age\n");
      }
      for (int m=0; m<nmb; ++m) {
        const auto &ll = pm->lloc_eachmb[m + mbs];
        std::fprintf(fb, "%d %.17g %d %d %d %.17g %d %d %d\n", pm->ncycle, pm->time, ll.level, ll.lx1,
                     ll.lx2, qblk[m], rawf[m], appf[m], pm->pmr->ncyc_since_ref(m + mbs));
      }
      std::fclose(fb);
    }
    if (cur != audit_prev_ && audit_dumps_ < audit_max_) {
      ++audit_dumps_;
      char name[64];
      std::snprintf(name, sizeof(name), "trigger_audit.%08d.bin", pm->ncycle);
      std::FILE *fd = std::fopen(name, "wb");
      if (fd != nullptr) {
        auto prh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
        auto pih = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pi);
        std::fprintf(fd, "AthenaK trigger audit v1\n%.17g %d %d %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g %d %d\n",
                     pm->time, pm->ncycle, nmb, npart, nr1, nr2, x0, pm->mesh_size.x1max, y0,
                     pm->mesh_size.x2max, trig_.rho_refine, trig_.rho_derefine, trig_.buffer,
                     pm->pmr->refinement_interval);
        std::vector<double> row(13);
        for (int m=0; m<nmb; ++m) {
          const auto &ll = pm->lloc_eachmb[m + mbs];
          const auto &sz = mbsize.h_view(m);
          row = {static_cast<double>(ll.level), static_cast<double>(ll.lx1), static_cast<double>(ll.lx2),
                 sz.x1min, sz.x1max, sz.x2min, sz.x2max, static_cast<double>(nx1), static_cast<double>(nx2),
                 qblk[m], static_cast<double>(rawf[m]), static_cast<double>(appf[m]),
                 static_cast<double>(pm->pmr->ncyc_since_ref(m + mbs))};
          std::fwrite(row.data(), sizeof(double), row.size(), fd);
        }
        for (int p=0; p<npart; ++p) {
          const double r4[4] = {prh(IPX,p), prh(IPY,p), prh(IPM,p), static_cast<double>(pih(PLEV,p))};
          std::fwrite(r4, sizeof(double), 4, fd);
        }
        std::fclose(fd);
      }
    }
    audit_prev_ = cur;
  }
  // several ranks (Codex review): the log holds totals over the ranks
  int nblk = nmb, nflag[2] = {nref, nder};
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    nblk = pm->nmb_total;
    MPI_Allreduce(MPI_IN_PLACE, nflag, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank == 0) {
    std::FILE *fp = AppendLog("dust_trigger_log.txt");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# ncycle time max_rho/mean blocks flag_refine flag_derefine fine_volume_share "
                     "fine_dust_mass_share clouds_root clouds_finer_plain clouds_lobed "
                     "blocks_per_level(0/1/...)\n");
      }
      const Real vbox = (pm->mesh_size.x1max - x0)*(pm->mesh_size.x2max - y0);
      // the blocks of every level, from the global block list
      std::vector<int> perlev(pm->max_level - pm->root_level + 1, 0);
      for (int g=0; g<pm->nmb_total; ++g) ++perlev[pm->lloc_eachmb[g].level - pm->root_level];
      std::string pl;
      for (std::size_t l=0; l<perlev.size(); ++l) pl += (l ? "/" : "") + std::to_string(perlev[l]);
      std::fprintf(fp, "%d %.17g %.6e %d %d %d %.6e %.6e %d %d %d %s\n", pm->ncycle, pm->time, qmax, nblk,
                   nflag[0], nflag[1], vfine/vbox, mfine/mtot, nlev[0], nlev[1], nlev[2], pl.c_str());
      std::fflush(fp);
    }
  }
  lap(DustGasDrag::kProfTrigLog);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ResumeAmrAfterRestart
//! \brief Called by Driver::Initialize of a restarted run before the
//! first time step.  Restores the cooldown age of every block and, when the file was written
//! among the outputs of a cycle, carries out the mesh check that followed those outputs in
//! the run that wrote it.  A file without the record (older code) keeps the old behaviour:
//! ages start at zero and no check is repeated.  First, on any mesh, the
//! MeshBlock id of every refinable cloud is set from its position (files written before the
//! routing fix 9807158d can name a wrong block; the composite weights never read it).

void DustGasDrag::ResumeAmrAfterRestart(Driver *pdrive, ParameterInput *pin) {
  Mesh *pm = pmy_pack->pmesh;
  if (smr_transfer_mode == DustSMRTransfer::refinable_cloud) ResetPGIDFromPositions("restart");
  if (!experimental_amr || !pm->adaptive || pm->pmr == nullptr) return;
  if (!amr_restart_state_found) {
    if (global_variable::my_rank == 0) {
      std::cout << "# dust AMR restart: the file has no amr_state record: cooldown ages "
                << "start at zero, the mesh check of the restart cycle is not repeated" << std::endl;
    }
    return;
  }
  if (static_cast<int>(amr_restart_ages.size()) != pm->nmb_total) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "amr_state of the restart file has " << amr_restart_ages.size()
              << " blocks, the mesh " << pm->nmb_total << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int m=0; m<pm->nmb_total; ++m) pm->pmr->ncyc_since_ref(m) = amr_restart_ages[m];
  if (global_variable::my_rank == 0) {
    std::cout << "# dust AMR restart: cooldown ages of " << pm->nmb_total
              << " blocks restored; pending mesh check of cycle " << pm->ncycle << ": "
              << (amr_restart_pending ? "done now" : "none") << std::endl;
  }
  if (amr_restart_pending) pm->pmr->AdaptiveMeshRefinement(pdrive, pin);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ResetPGIDFromPositions
//! \brief The MeshBlock id of every particle from the leaf that holds
//! its position (refinable clouds: the leaf maps).  Returns how many ids changed and prints
//! it; stops if a position has no owning block of this pack.

int DustGasDrag::ResetPGIDFromPositions(const char *when) {
  if (global_variable::nranks > 1) return RouteToOwnerBlocks(when);
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const auto lt = leaf_traversal;   // the actual leaves
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ncell12 = (indcs.nx1 + 2*indcs.ng)*(indcs.nx2 + 2*indcs.ng);
  const int nmb = pmy_pack->nmb_thispack;
  const int gids = pmy_pack->gids;
  int nchanged = 0, nbad = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_pgid_reset", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &changed, int &bad) {
      const int leaf = lt.LeafOwner(pr(IPX,p), pr(IPY,p), lt.Anchor(pi(PGID,p)));
      const int owner = (leaf < 0) ? -1 : leaf/ncell12;
      if (owner < 0 || owner >= nmb) {
        ++bad;
      } else if (pi(PGID,p) != gids + owner) {
        pi(PGID,p) = gids + owner;
        ++changed;
      }
    }, Kokkos::Sum<int>(nchanged), Kokkos::Sum<int>(nbad));
  }
  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << nbad << " particles have no owning MeshBlock (" << when << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "# dust: MeshBlock ids set from the positions at the " << when << ": "
              << nchanged << " of " << npart << " changed" << std::endl;
  }
  return nchanged;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RouteToOwnerBlocks
//! \brief Several ranks: every particle to the block that holds its position
//! (the leaf traversal of the current mesh, all blocks of the mesh): its MeshBlock id, and the
//! particles of blocks of other ranks (after a regrid with load balancing: anywhere) through
//! the AthenaK particle exchange, with a send list filled here.  Returns how many ids of this
//! rank's particles changed.

int DustGasDrag::RouteToOwnerBlocks(const char *when) {
  particles::Particles *ppar = pmy_pack->ppart;
  auto *pb = ppar->pbval_part;
  const int npart = ppar->nprtcl_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const auto lt = leaf_traversal;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ncell12 = (indcs.nx1 + 2*indcs.ng)*(indcs.nx2 + 2*indcs.ng);
  const int nmb = pmy_pack->nmb_thispack, gids = pmy_pack->gids;
  auto sgid = leaf_slot_gid_d;
  auto srank = leaf_slot_rank_d;
  int nchanged = 0, nbad = 0;
#if MPI_PARALLEL_ENABLED
  if (pb->sendlist.extent_int(0) < std::max(npart, 1)) Kokkos::realloc(pb->sendlist, std::max(npart, 1));
  Kokkos::deep_copy(pb->send_count.d_view, 0);
  auto psendl = pb->sendlist.d_view;
  int *pcounter = pb->send_count.d_view.data();
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_route_owner", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &changed, int &bad) {
      const int leaf = lt.LeafOwner(pr(IPX,p), pr(IPY,p), lt.Anchor(pi(PGID,p)));
      const int slot = (leaf < 0) ? -1 : leaf/ncell12;
      if (slot < 0) {++bad; return;}
      const int g = (slot < nmb) ? gids + slot : sgid(slot);
      if (pi(PGID,p) != g) {pi(PGID,p) = g; ++changed;}
      if (slot >= nmb) {
        const int k = Kokkos::atomic_fetch_add(pcounter, 1);
        psendl(k).prtcl_indx = p;
        psendl(k).dest_gid = g;
        psendl(k).dest_rank = srank(slot);
      }
    }, Kokkos::Sum<int>(nchanged), Kokkos::Sum<int>(nbad));
  }
  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << nbad << " particles have no owning MeshBlock (" << (when ? when : "split") << ")"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::deep_copy(pb->send_count.h_view, pb->send_count.d_view);
  pb->nprtcl_send = pb->send_count.h_view(0);
  if (pb->nprtcl_send > 0) {
    auto d = Kokkos::subview(pb->sendlist.d_view, std::make_pair(0, pb->nprtcl_send));
    auto h = Kokkos::subview(pb->sendlist.h_view, std::make_pair(0, pb->nprtcl_send));
    Kokkos::deep_copy(h, d);
  }
  const int nsent = pb->nprtcl_send;
  pb->CountSendsAndRecvs();
  pb->InitPrtclRecv();
  pb->PackAndSendPrtcls();
  while (pb->RecvAndUnpackPrtcls() != TaskStatus::complete) {}
  pb->ClearPrtclSend();
  pb->ClearPrtclRecv();
  Mesh *pm = pmy_pack->pmesh;
  long long tot = 0;
  for (int n=0; n<global_variable::nranks; ++n) tot += pm->nprtcl_eachrank[n];
  pm->nprtcl_total = static_cast<int>(tot);
  long long sums[2] = {nchanged, nsent};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (global_variable::my_rank == 0 && when != nullptr) {   // nullptr: quiet (SplitToLeaves)
    std::cout << "# dust: MeshBlock ids set from the positions at the " << when << ": "
              << sums[0] << " of " << tot << " changed, " << sums[1] << " sent to other ranks"
              << std::endl;
  }
#else
  (void)pb; (void)npart; (void)pr; (void)pi; (void)ncell12; (void)nmb; (void)gids;
  (void)sgid; (void)srank; (void)nbad; (void)when;
#endif
  return nchanged;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BeforeRegrid
//! \brief Called by MeshRefinement::AdaptiveMeshRefinement before the mesh changes: the
//! particle count and checksum that RebuildAfterRegrid must find again.

void DustGasDrag::BeforeRegrid(int nnew, int ndel) {
  amr_npart_before = pmy_pack->ppart->nprtcl_thispack;
  amr_regrid_nnew = nnew;
  amr_regrid_ndel = ndel;
  amr_checksum = ParticleChecksum(pmy_pack->ppart);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RebuildAfterRegrid
//! \brief Called by MeshRefinement::AdaptiveMeshRefinement right after the mesh changed
//! and the gas ghost cells were filled again.  See the file header.  The particle arrays
//! are only read, except PGID.  Self-check on the new maps: the weights of every cloud
//! over the leaves add up to one and point into existing blocks.

void DustGasDrag::RebuildAfterRegrid(Driver *pdrive, ParameterInput *pin) {
  const bool oo = (smr_transfer_mode == DustSMRTransfer::owner_overlap);
  if (smr_transfer_mode != DustSMRTransfer::refinable_cloud && !oo) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust on a dynamic mesh needs experimental_smr_transfer = refinable_cloud "
              << "(or owner_overlap)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Mesh *pmesh = pmy_pack->pmesh;
  particles::Particles *ppar = pmy_pack->ppart;
  Real dwmax = 0.0;
  double msum = 0.0;
  const int nmb = pmy_pack->nmb_thispack;
  int npart = ppar->nprtcl_thispack;
  if (oo) {
    // The old owner_overlap path: point particles, nothing to split or merge; the boxes,
    // the block table, the deposit remaps and the block of every particle on the new mesh
    BuildOwnerOverlapMesh(true);
    ResetPGIDOwnerOverlap(nullptr);
    auto pr = ppar->prtcl_rdata;
    if (npart > 0) {
      Kokkos::parallel_reduce("dust_oo_regrid_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, double &ms) {ms += static_cast<double>(pr(IPM,p));},
      Kokkos::Sum<double>(msum));
    }
  } else {
  BuildRefinableCloudTransfer(pin);
  // Several ranks: the particles follow their blocks to the ranks that hold
  // them on the new mesh (load balancing may move blocks anywhere)
  if (global_variable::nranks > 1) {
    RouteToOwnerBlocks("regrid");
    CheckHaloReach("regrid");   // clouds one level behind the new mesh
  }
  npart = ppar->nprtcl_thispack;
  // Every regrid is an event, handled at the start of the next cycle after
  // its split (HandleRegridEvent); refine / derefine from the blocks actually created and
  // deleted, not from the net change of the block count.  A regrid that deleted blocks also
  // starts a burst of merge passes.  Two regrids before one handling (possible
  // only around a restart) make one event with the summed counts.
  amr_event_pending = true;
  amr_event_nnew += amr_regrid_nnew;
  amr_event_ndel += amr_regrid_ndel;
  amr_event_regrid = amr_regrids + 1;   // the number this regrid gets below
  if (amr_regrid_ndel > 0 && cloud_merge_burst_dt > 0.0) {
    cloud_merge_burst_until = pmesh->time + cloud_merge_burst_dt;
  }

  const auto lt = leaf_traversal;   // the actual leaves
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  auto &indcs = pmesh->mb_indcs;
  const int ncell12 = (indcs.nx1 + 2*indcs.ng)*(indcs.nx2 + 2*indcs.ng);
  const int gids = pmy_pack->gids;
  int nbad = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_amr_regrid", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &dw, int &bad, double &ms) {
      const Real xp = pr(IPX,p), yp = pr(IPY,p);
      const int leaf = lt.LeafOwner(xp, yp, lt.Anchor(pi(PGID,p)));   // the old id: a guess only
      const int owner = (leaf < 0) ? -1 : leaf/ncell12;
      pi(PGID,p) = gids + owner;
      Real sw = 0.0;
      int b = (owner < 0 || owner >= nmb) ? 1 : 0;
      lt.ForEachCloudWeight(xp, yp, pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int target, const Real w) {
        sw += w;
        if (target < 0 || target/ncell12 >= lt.nfld) ++b;   // halo blocks included
      });
      dw = Kokkos::fmax(dw, Kokkos::fabs(sw - 1.0));
      bad += b;
      ms += static_cast<double>(pr(IPM,p));
    }, Kokkos::Max<Real>(dwmax), Kokkos::Sum<int>(nbad), Kokkos::Sum<double>(msum));
  }
  if (nbad > 0 || dwmax > 1.0e-10) {
    // the first particles without an owning block or with a weight outside the blocks
    DvceArray2D<Real> ex("dust_amr_bad", 8, 6);
    DvceArray1D<int> nex("dust_amr_nbad", 1);
    Kokkos::parallel_for("dust_amr_bad", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      const Real xp = pr(IPX,p), yp = pr(IPY,p);
      const int leaf = lt.LeafOwner(xp, yp, lt.Anchor(pi(PGID,p)));
      int bad = (leaf < 0) ? 1 : 0, nout = 0;
      lt.ForEachCloudWeight(xp, yp, pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int target, const Real) {
        if (target < 0 || target/ncell12 >= lt.nfld) ++nout;
      });
      if (bad == 0 && nout == 0) return;
      const int k = Kokkos::atomic_fetch_add(&nex(0), 1);
      if (k >= 8) return;
      ex(k, 0) = xp; ex(k, 1) = yp; ex(k, 2) = pi(PGID,p) - gids; ex(k, 3) = leaf;
      ex(k, 4) = nout; ex(k, 5) = pi(PLEV,p);
    });
    auto exh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ex);
    auto nexh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), nex);
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust after a regrid: " << nbad << " cloud weights outside the mesh blocks, "
              << "largest |sum of weights - 1| = " << dwmax << "; first of " << nexh(0)
              << " particles (x, y, block, owner leaf, weights outside, code):" << std::endl;
    for (int k=0; k<std::min(8, nexh(0)); ++k) {
      std::cout << "  " << exh(k, 0) << " " << exh(k, 1) << " " << exh(k, 2) << " " << exh(k, 3)
                << " " << exh(k, 4) << " " << exh(k, 5) << std::endl;
    }
    std::exit(EXIT_FAILURE);
  }
  }   // refinable clouds

  // nothing but PGID may have changed since BeforeRegrid (several ranks: over all ranks; the
  // checksum is a sum over the particles)
  unsigned long long check = ParticleChecksum(ppar);
  long long nb4 = amr_npart_before, naft = npart;
  unsigned long long ck4 = amr_checksum;
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &nb4, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &naft, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ck4, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &check, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  if (nb4 != naft || check != ck4) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust after a regrid: particles " << amr_npart_before << " -> " << npart
              << ", checksum " << amr_checksum << " -> " << check << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // the particle time step with the new cell sizes (the gas step is set by the caller)
  (void) ComputeNewTimeStep(pdrive, pdrive->nexp_stages);

  ++amr_regrids;
  int nlev1 = 0;
  {
    auto &mblev = pmy_pack->pmb->mb_lev;
    for (int m=0; m<nmb; ++m) {
      if (mblev.h_view(m) > pmesh->root_level) ++nlev1;
    }
  }
  // several ranks (Codex review): the log holds totals over the ranks (sums;
  // the largest weight error; the smallest particle step)
  int nblk = nmb;
  long long nprt = npart;
  Real dtp = ppar->dtnew;
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    nblk = pmesh->nmb_total;
    MPI_Allreduce(MPI_IN_PLACE, &nlev1, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &nprt, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &msum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &dwmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &dtp, 1, MPI_ATHENA_REAL, MPI_MIN, MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank == 0) {
    std::FILE *fp = AppendLog("dust_amr_log.txt");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# regrid ncycle time blocks blocks_above_root particles particle_mass "
                     "max|sum_w-1| dt_particles checksum\n");
      }
      std::fprintf(fp, "%d %d %.17g %d %d %lld %.17g %.3e %.9e %llu\n", amr_regrids, pmesh->ncycle,
                   pmesh->time, nblk, nlev1, nprt, msum, dwmax, dtp, check);
      std::fflush(fp);
    }
    std::cout << "# dust regrid " << amr_regrids << " cycle=" << pmesh->ncycle << " time="
              << pmesh->time << " blocks=" << nblk << " (above root " << nlev1 << ", created "
              << amr_regrid_nnew << ", deleted " << amr_regrid_ndel << ") particles="
              << nprt << " max|sum w - 1|=" << dwmax << std::endl;
  }
  // The leaf traversal also right after a regrid (old clouds on the new mesh)
  if (cloud_leaf_check > 0) CheckLeafTraversal();
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildOwnerOverlapMesh
//! \brief The old owner_overlap transfer on a dynamic mesh, one rank.
//! For the current mesh (after the constructor and after every regrid): the block boxes in
//! leaf_traversal (boxes only: nfld = nall = nmb; the density trigger reads them), the table of
//! the block of every finest-level block cell (the block of a position), and, with remap, the
//! ghost-to-leaf maps of the deposit exchanges again (MeshBoundaryValuesDep::BuildLocalSMRMap).

void DustGasDrag::BuildOwnerOverlapMesh(const bool remap) {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int gids = pmy_pack->gids;
  auto &mbsize = pmy_pack->pmb->mb_size;
  if (global_variable::nranks != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "owner_overlap on a dynamic mesh: one MPI rank only" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &lt = leaf_traversal;
  lt.nmb = nmb; lt.gids = gids; lt.nfld = nmb; lt.nall = nmb;
  lt.nx1 = indcs.nx1; lt.nx2 = indcs.nx2; lt.is = indcs.is; lt.js = indcs.js;
  lt.ncells1 = indcs.nx1 + 2*indcs.ng; lt.ncells2 = indcs.nx2 + 2*indcs.ng;
  lt.box = DvceArray2D<Real>("dust_oo_box", std::max(1, nmb), 6);
  auto box = Kokkos::create_mirror_view(lt.box);
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    box(m,0) = sz.x1min; box(m,1) = sz.x2min; box(m,2) = sz.dx1; box(m,3) = sz.dx2;
    box(m,4) = 1.0/sz.dx1; box(m,5) = 1.0/sz.dx2;
  }
  Kokkos::deep_copy(lt.box, box);
  halo_.rect.clear();
  // the finest-level block grid: every block covers 2^(nl - its level) cells per direction
  const int nl = pm->max_level - pm->root_level;
  const int nbx = pm->nmb_rootx1 << nl, nby = pm->nmb_rootx2 << nl;
  if (oo_block_map.extent_int(0) != nby || oo_block_map.extent_int(1) != nbx) {
    oo_block_map = DvceArray2D<int>("dust_oo_block_map", nby, nbx);
  }
  auto map = Kokkos::create_mirror_view(oo_block_map);
  Kokkos::deep_copy(map, -1);
  int nbad = 0;
  for (int m=0; m<nmb; ++m) {
    const LogicalLocation &ll = pm->lloc_eachmb[gids + m];
    const int sh = nl - (ll.level - pm->root_level);
    if (sh < 0) {++nbad; continue;}
    const int a0 = static_cast<int>(ll.lx1) << sh, b0 = static_cast<int>(ll.lx2) << sh;
    for (int b=b0; b<b0 + (1 << sh); ++b) {
      for (int a=a0; a<a0 + (1 << sh); ++a) {
        if (a < 0 || a >= nbx || b < 0 || b >= nby || map(b,a) >= 0) {++nbad; continue;}
        map(b,a) = m;
      }
    }
  }
  for (int b=0; b<nby; ++b) for (int a=0; a<nbx; ++a) if (map(b,a) < 0) ++nbad;
  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "owner_overlap block table: " << nbad << " cells without exactly one block"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::deep_copy(oo_block_map, map);
  if (remap) {
    pbval_qp->BuildLocalSMRMap();
    pbval_dm->BuildLocalSMRMap();
    if (pbval_solver_add != nullptr) pbval_solver_add->BuildLocalSMRMap();
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ResetPGIDOwnerOverlap
//! \brief The MeshBlock id of every particle from the block table (BuildOwnerOverlapMesh)
//! at its position (periodic image in the box; the position itself is not changed).  Returns
//! how many ids changed; prints it unless when is nullptr.

int DustGasDrag::ResetPGIDOwnerOverlap(const char *when) {
  Mesh *pm = pmy_pack->pmesh;
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  auto map = oo_block_map;
  const int nbx = map.extent_int(1), nby = map.extent_int(0);
  const Real x1min = pm->mesh_size.x1min, x2min = pm->mesh_size.x2min;
  const Real lx = pm->mesh_size.x1max - x1min, ly = pm->mesh_size.x2max - x2min;
  const int gids = pmy_pack->gids, nmb = pmy_pack->nmb_thispack;
  int nchanged = 0, nbad = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_oo_pgid", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &changed, int &bad) {
      Real x = pr(IPX,p) - x1min, y = pr(IPY,p) - x2min;
      x -= lx*Kokkos::floor(x/lx); y -= ly*Kokkos::floor(y/ly);
      const int ix = Kokkos::min(nbx - 1, Kokkos::max(0, static_cast<int>(x*(nbx/lx))));
      const int iy = Kokkos::min(nby - 1, Kokkos::max(0, static_cast<int>(y*(nby/ly))));
      const int m = map(iy, ix);
      if (m < 0 || m >= nmb) {++bad; return;}
      if (pi(PGID,p) != gids + m) {pi(PGID,p) = gids + m; ++changed;}
    }, Kokkos::Sum<int>(nchanged), Kokkos::Sum<int>(nbad));
  }
  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << nbad << " particles have no owning MeshBlock (owner_overlap)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (when != nullptr && global_variable::my_rank == 0) {
    std::cout << "# dust (owner_overlap): MeshBlock ids set from the positions at the " << when << ": "
              << nchanged << " of " << npart << " changed" << std::endl;
  }
  return nchanged;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::OwnerOverlapDeposit
//! \brief The leaf fields of the old owner_overlap transfer.  Every particle deposits
//! with the coupling's stencil (PMWeights, <dust>/deposit) on its own block, ghost cells
//! included; the ghost parts are then added to the leaf cells they overlap, by volume, with the
//! remap of the coupling's deposit exchange (MeshBoundaryValuesDep::RemapLocalSMRDeposit).
//! Component 0: the dust density; with moments also 1-3 the momentum density and 4-6 the raw
//! second moments sum(rho v_c^2) (the central moments are formed by the caller).  The ghost
//! cells keep their own deposits (not read).  Stops if the dust mass on the leaves differs from
//! the particle mass (relative 1e-9): checks the remap of the current mesh at every use.

void DustGasDrag::OwnerOverlapDeposit(DvceArray5D<Real> &a, const bool moments) {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int nvar = moments ? 7 : 1;
  const int ncells1 = indcs.nx1 + 2*indcs.ng, ncells2 = indcs.nx2 + 2*indcs.ng;
  if (!pbval_dm->LocalSMRRemap()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "owner_overlap leaf deposit needs the static/dynamic SMR remap" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (a.extent_int(0) != nmb || a.extent_int(1) != nvar || a.extent_int(3) != ncells2 ||
      a.extent_int(4) != ncells1) {
    a = DvceArray5D<Real>("dust_oo_deposit", nmb, nvar, 1, ncells2, ncells1);   // zero
  } else {
    Kokkos::deep_copy(DevExeSpace(), a, 0.0);
  }
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks, nx1 = indcs.nx1, nx2 = indcs.nx2;
  const int gids = pmy_pack->gids;
  const int scheme = static_cast<int>(deposit);
  auto a_ = a;
  int nbad = 0;
  double mpart = 0.0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_oo_deposit", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &bad, double &mp) {
      const int m = pi(PGID,p) - gids;
      if (m < 0 || m >= nmb) {++bad; return;}
      int ip, jp;
      Real wx[3], wy[3];
      PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is, scheme, ip, wx);
      PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js, scheme, jp, wy);
      if (ip < 1 || ip > ncells1 - 2 || jp < 1 || jp > ncells2 - 2) {++bad; return;}
      const Real d = pr(IPM,p)/(mbsize.d_view(m).dx1*mbsize.d_view(m).dx2);
      const Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
      mp += static_cast<double>(pr(IPM,p));
      for (int b=0; b<3; ++b) {
        for (int c=0; c<3; ++c) {
          const Real w = wy[b]*wx[c];
          if (w == 0.0) continue;
          const int jj = jp + b - 1, ii = ip + c - 1;
          DepositAdd(&a_(m,0,ks,jj,ii), w*d);
          if (moments) {
            for (int q=0; q<3; ++q) {
              DepositAdd(&a_(m,1+q,ks,jj,ii), w*d*v[q]);
              DepositAdd(&a_(m,4+q,ks,jj,ii), w*d*v[q]*v[q]);
            }
          }
        }
      }
    }, Kokkos::Sum<int>(nbad), Kokkos::Sum<double>(mpart));
  }
  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "owner_overlap leaf deposit: " << nbad << " particles outside their block's "
              << "stencil" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pbval_dm->RemapLocalSMRDeposit(a);
  double mleaf = 0.0;
  Kokkos::parallel_reduce("dust_oo_deposit_mass",
  Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*nx2*nx1),
  KOKKOS_LAMBDA(const int k, double &ml) {
    const int m = k/(nx1*nx2), r = k % (nx1*nx2);
    ml += static_cast<double>(a_(m,0,ks,js + r/nx1,is + r % nx1))*
          mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
  }, Kokkos::Sum<double>(mleaf));
  if (!(std::fabs(mleaf - mpart) <= 1.0e-9*mpart)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "owner_overlap leaf deposit: dust mass on the leaves " << mleaf
              << " != particle mass " << mpart << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace dust
