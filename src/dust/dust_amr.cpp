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
//!   - the cached merge / compress masks are marked stale (rebuilt at their next use);
//!   - every particle's MeshBlock id (PGID) is set again from its position;
//!   - the particle time step is computed again with the new cell sizes.
//! Tag, mass, position, velocity, cloud level and lobes of a particle are not touched: a
//! coarser mesh does not make a cloud wider, a finer mesh does not split it (that is the
//! split's decision at the next cycle, with the new maps).
//!
//! ScheduledRefinement() is a refinement request for tests: rectangles with a time window
//! in which their blocks are wanted at a given level.

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
//! \fn DustGasDrag::BeforeRegrid
//! \brief Called by MeshRefinement::AdaptiveMeshRefinement before the mesh changes: the
//! particle count and checksum that RebuildAfterRegrid must find again.

void DustGasDrag::BeforeRegrid() {
  amr_npart_before = pmy_pack->ppart->nprtcl_thispack;
  amr_checksum = ParticleChecksum(pmy_pack->ppart);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RebuildAfterRegrid
//! \brief Called by MeshRefinement::AdaptiveMeshRefinement right after the mesh changed
//! and the gas ghost cells were filled again.  See the file header.  The particle arrays
//! are only read, except PGID.  Self-check on the new maps: the weights of every cloud
//! over the leaves add up to one and point into existing blocks.

void DustGasDrag::RebuildAfterRegrid(Driver *pdrive, ParameterInput *pin) {
  if (smr_transfer_mode != DustSMRTransfer::refinable_cloud) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust on a dynamic mesh needs experimental_smr_transfer = refinable_cloud"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Mesh *pmesh = pmy_pack->pmesh;
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;

  BuildRefinableCloudTransfer(pin);
  merge_mask_built = false;       // built once per layout, at the next merge pass
  compress_mask_built = false;

  const auto transfer = smr_transfer;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  auto &indcs = pmesh->mb_indcs;
  const int ncell12 = (indcs.nx1 + 2*indcs.ng)*(indcs.nx2 + 2*indcs.ng);
  const int nmb = pmy_pack->nmb_thispack;
  const int gids = pmy_pack->gids;
  Real dwmax = 0.0;
  int nbad = 0;
  double msum = 0.0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_amr_regrid", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &dw, int &bad, double &ms) {
      const Real xp = pr(IPX,p), yp = pr(IPY,p);
      const int owner = transfer.LeafOwner(xp, yp)/ncell12;
      pi(PGID,p) = gids + owner;
      Real sw = 0.0;
      int b = (owner < 0 || owner >= nmb) ? 1 : 0;
      transfer.ForEachCloudWeight(xp, yp, pi(PLEV,p), [&](const int target, const Real w) {
        sw += w;
        if (target < 0 || target/ncell12 >= nmb) ++b;
      });
      dw = Kokkos::fmax(dw, Kokkos::fabs(sw - 1.0));
      bad += b;
      ms += static_cast<double>(pr(IPM,p));
    }, Kokkos::Max<Real>(dwmax), Kokkos::Sum<int>(nbad), Kokkos::Sum<double>(msum));
  }
  if (nbad > 0 || dwmax > 1.0e-10) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust after a regrid: " << nbad << " cloud weights outside the mesh blocks, "
              << "largest |sum of weights - 1| = " << dwmax << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // nothing but PGID may have changed since BeforeRegrid
  const unsigned long long check = ParticleChecksum(ppar);
  if (amr_npart_before != npart || check != amr_checksum) {
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
  if (global_variable::my_rank == 0) {
    std::FILE *fp = std::fopen("dust_amr_log.txt", "a");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# regrid ncycle time blocks blocks_above_root particles particle_mass "
                     "max|sum_w-1| dt_particles checksum\n");
      }
      std::fprintf(fp, "%d %d %.17g %d %d %d %.17g %.3e %.9e %llu\n", amr_regrids, pmesh->ncycle,
                   pmesh->time, nmb, nlev1, npart, msum, dwmax, ppar->dtnew, check);
      std::fclose(fp);
    }
    std::cout << "# dust regrid " << amr_regrids << " cycle=" << pmesh->ncycle << " time="
              << pmesh->time << " blocks=" << nmb << " (above root " << nlev1 << ") particles="
              << npart << " max|sum w - 1|=" << dwmax << std::endl;
  }
}

}  // namespace dust
