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
//!
//! DensityRefinement() is the minimal dust-density request: at the
//! check (end of the cycle, after the outputs, before the regrid) the dust mass of the
//! particles at their new positions is deposited on the current leaves with the exact cloud
//! integrals (the transfer of the coupling and of the field outputs); rho_d / mean is taken
//! as a maximum onto the root grid, the map is widened by amr_rho_buffer root cells
//! (periodic), and a block whose cover reaches amr_rho_refine is flagged for refinement,
//! one that stays below amr_rho_derefine (< amr_rho_refine, hysteresis) for derefinement.
//! The sibling, 2:1 and refinement_interval rules of MeshRefinement apply as usual.  Each
//! check writes a line to dust_trigger_log.txt.

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
  int buffer = 0;                               // root cells
};
DensityTrigger trig_;

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
  if (!(trig_.rho_derefine >= 0.0 && trig_.rho_derefine < trig_.rho_refine) || trig_.buffer < 0) {
    fatal("amr_density_trigger needs 0 <= amr_rho_derefine < amr_rho_refine and amr_rho_buffer >= 0");
  }
  if (global_variable::my_rank == 0) {
    std::cout << "# dust density trigger: refine at rho_d/mean >= " << trig_.rho_refine
              << ", derefine below " << trig_.rho_derefine << ", buffer " << trig_.buffer
              << " root cells" << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn dust::DensityRefinement
//! \brief Refinement request from the deposited dust density (see the file header).

void DensityRefinement(MeshBlockPack *pmbp) {
  Mesh *pm = pmbp->pmesh;
  DustGasDrag *pdust = pmbp->pdust;
  if (pdust == nullptr || !pdust->UsesCompositeTransfer() || pm->three_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "amr_density_trigger needs 2-D refinable-cloud dust" << std::endl;
    std::exit(EXIT_FAILURE);
  }
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
  // dust density on the current leaves from the particles at their new positions
  DvceArray3D<Real> f("amr_rho", nmb, ncells2, ncells1);
  Kokkos::deep_copy(DevExeSpace(), f, 0.0);
  par_for("amr_rho_deposit", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    const Real mass = pr(IPM,p);
    transfer.ForEachWeight(pr(IPX,p), pr(IPY,p), pi(PLEV,p), [&](const int target, const Real w) {
      if (w == 0.0) return;
      const int ti = target % ncells1, tj = (target/ncells1) % ncells2, tm = target/(ncells1*ncells2);
      DepositAdd(&f(tm,tj,ti), mass*w/(mbsize.d_view(tm).dx1*mbsize.d_view(tm).dx2));
    });
  });
  Real mtot = 0.0;
  int nlev[3] = {0, 0, 0};   // clouds at the root level, finer plain, lobed
  Kokkos::parallel_reduce("amr_rho_totals", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, Real &m, int &n0, int &n1, int &nl) {
    m += pr(IPM,p);
    const int code = pi(PLEV,p);
    if (code > 15) ++nl; else if ((code & 15) == 0) ++n0; else ++n1;
  }, mtot, nlev[0], nlev[1], nlev[2]);
  auto fh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), f);
  const int nr1 = pm->mesh_indcs.nx1, nr2 = pm->mesh_indcs.nx2;
  const Real x0 = pm->mesh_size.x1min, y0 = pm->mesh_size.x2min;
  const Real hx = (pm->mesh_size.x1max - x0)/nr1, hy = (pm->mesh_size.x2max - y0)/nr2;
  const Real mean = mtot/((pm->mesh_size.x1max - x0)*(pm->mesh_size.x2max - y0));
  // maximum of rho_d / mean per root cell, then widened by the buffer (periodic)
  std::vector<Real> rmax(nr1*nr2, 0.0), wide(nr1*nr2, 0.0);
  Real qmax = 0.0, mfine = 0.0, vfine = 0.0;
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    const bool fine = pm->lloc_eachmb[m + mbs].level > pm->root_level;
    for (int j=0; j<nx2; ++j) {
      for (int i=0; i<nx1; ++i) {
        const Real q = fh(m, js+j, is+i)/mean;
        const int ri = std::min(nr1-1, static_cast<int>((sz.x1min + (i + 0.5)*sz.dx1 - x0)/hx));
        const int rj = std::min(nr2-1, static_cast<int>((sz.x2min + (j + 0.5)*sz.dx2 - y0)/hy));
        rmax[rj*nr1 + ri] = std::max(rmax[rj*nr1 + ri], q);
        qmax = std::max(qmax, q);
        if (fine) mfine += fh(m, js+j, is+i)*sz.dx1*sz.dx2;
      }
    }
    if (fine) vfine += nx1*sz.dx1*nx2*sz.dx2;
  }
  const int b = trig_.buffer;
  for (int rj=0; rj<nr2; ++rj) {
    for (int ri=0; ri<nr1; ++ri) {
      Real w = 0.0;
      for (int dj=-b; dj<=b; ++dj) {
        for (int di=-b; di<=b; ++di) {
          w = std::max(w, rmax[((rj + dj + nr2) % nr2)*nr1 + (ri + di + nr1) % nr1]);
        }
      }
      wide[rj*nr1 + ri] = w;
    }
  }
  int nref = 0, nder = 0;
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    const int i0 = static_cast<int>(std::floor((sz.x1min - x0)/hx + 1.0e-9));
    const int i1 = static_cast<int>(std::ceil((sz.x1max - x0)/hx - 1.0e-9));
    const int j0 = static_cast<int>(std::floor((sz.x2min - y0)/hy + 1.0e-9));
    const int j1 = static_cast<int>(std::ceil((sz.x2max - y0)/hy - 1.0e-9));
    Real q = 0.0;
    for (int rj=j0; rj<j1; ++rj) {
      for (int ri=i0; ri<i1; ++ri) q = std::max(q, wide[rj*nr1 + ri]);
    }
    const int lev = pm->lloc_eachmb[m + mbs].level;
    int flag = 0;
    if (q >= trig_.rho_refine && lev < pm->max_level) flag = 1;
    else if (q < trig_.rho_derefine && lev > pm->root_level) flag = -1;
    refine_flag.h_view(m + mbs) = flag;
    if (flag > 0) ++nref;
    if (flag < 0) ++nder;
  }
  refine_flag.template modify<HostMemSpace>();
  refine_flag.template sync<DevExeSpace>();
  if (global_variable::my_rank == 0) {
    std::FILE *fp = std::fopen("dust_trigger_log.txt", "a");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# ncycle time max_rho/mean blocks flag_refine flag_derefine fine_volume_share "
                     "fine_dust_mass_share clouds_root clouds_finer_plain clouds_lobed\n");
      }
      const Real vbox = (pm->mesh_size.x1max - x0)*(pm->mesh_size.x2max - y0);
      std::fprintf(fp, "%d %.17g %.6e %d %d %d %.6e %.6e %d %d %d\n", pm->ncycle, pm->time, qmax, nmb,
                   nref, nder, vfine/vbox, mfine/mtot, nlev[0], nlev[1], nlev[2]);
      std::fclose(fp);
    }
  }
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
