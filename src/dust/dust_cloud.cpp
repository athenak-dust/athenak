//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud.cpp
//! \brief Refinable Lagrangian hat clouds on a single-rank,
//! two-dimensional, static 2:1 mesh: the finest-level virtual grid that maps positions
//! to active leaves, and the exact two-scale split with merging of coincident children.
//!
//! Every particle carries a cloud level PLEV. Its hat cloud has half-width
//! root_dx/2^PLEV per direction, and its weight on an active leaf is the exact integral
//! of the cloud over that leaf (SMRCompositeTransfer::ForEachCloudWeight).  The
//! deposited fields therefore depend only on the continuous cloud density
//! sum_p m_p S_p(x - x_p), which the mesh merely integrates over its leaves.  A cloud
//! changes size only through the identity
//!   S_ell(x) = 1/4 S_{ell/2}(x + ell/2) + 1/2 S_{ell/2}(x) + 1/4 S_{ell/2}(x - ell/2),
//! applied per direction, which leaves that continuous density unchanged.  Children
//! that coincide with existing same-level particles (the shared children of adjacent
//! lattice parents) are merged by centre of mass, so a quiet-start lattice refines to
//! 2^d equal children per parent.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <unordered_map>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust {
namespace {

int EncodeCloudLeaf(const int m, const int j, const int i,
                    const int ncells2, const int ncells1) {
  return (m*ncells2 + j)*ncells1 + i;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildRefinableCloudTransfer
//! \brief Map every cell of the finest-level virtual grid to the active leaf that
//! contains it, and record that leaf's level.

void DustGasDrag::BuildRefinableCloudTransfer(ParameterInput *pin) {
  Mesh *pmesh = pmy_pack->pmesh;
  auto &indcs = pmesh->mb_indcs;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int nmb = pmy_pack->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  // <dust>/cloud_virtual_levels (diagnostic control, default 0): extra levels of the
  // virtual grid below the finest mesh level, so clouds can be finer than every gas leaf
  // (e.g. a uniform gas mesh with the fine clouds and sampling of a refined region; the
  // fine virtual cells map to their coarse leaf exactly as in the coarse part of a mixed
  // mesh)
  const int nlev_mesh = pmesh->max_level - pmesh->root_level;
  const int nvirt = pin->GetOrAddInteger("dust", "cloud_virtual_levels", 0);
  int nlev = nlev_mesh + nvirt;

  // The virtual finest-grid table (cloud_target, cloud_level) is read
  // only by the host paths and the checks; the production device paths use the actual leaves
  // (LeafTraversal).  <dust>/cloud_table = auto (default: built when one of its users is set),
  // true or false.  The table supports the root and two levels of refinement; without it the
  // levels are limited by the cloud code (4 bits) only.
  // The level-native traversal of the table (already off by default; the
  // production kernels use the actual leaves) and its check are gone; the table
  // itself stays (host split, host merge, leaf check, other controls)
  RemovedKey(pin, "cloud_native_transfer", KeyFalse(pin, "cloud_native_transfer"),
             "the level-native traversal");
  RemovedKey(pin, "cloud_native_check",
             !pin->DoesParameterExist("dust", "cloud_native_check") ||
             pin->GetInteger("dust", "cloud_native_check") <= 0,
             "the check of the level-native traversal");
  // cloud_halo_coarser (test): levels a cloud may be coarser than the block of its centre, for the
  // reach of the halo (-1: from the split rule); the per-cycle check widens it where needed
  cloud_halo_coarser = pin->GetOrAddInteger("dust", "cloud_halo_coarser", -1);
  // cloud_halo_strips: exchange only the cells of each halo block that this rank's clouds can reach
  // (false: whole blocks)
  cloud_halo_strips = pin->GetOrAddBoolean("dust", "cloud_halo_strips", true);   // Rixin: default true
  // Every cloud_sort_interval cycles (0: never) the particle arrays are reordered
  // by block, cell in the block (row by row) and old position, after
  // the split, resampling and merge pass of the cycle (SortParticles).
  // Following a Codex review, every 64 cycles by default on the configuration where it was timed and
  // its effect on G checked (production PC2 path, device G and R, cloud_merge_exact_ts, stopping
  // times fixed per species: stopping_time_mode = species_fixed); 0 is the control
  {
    const bool prod = coupling == DustCoupling::hybrid &&
                      hybrid_force_mode != HybridForceMode::split_be && cloud_gather_order == 2;
    // Not in 3-D yet (the sort keys are 2-D cells)
    const bool sortdef = prod && !clouds3d && (!cloud_merge || cloud_merge_device) &&
                         (!cloud_resample || cloud_resample_device) && cloud_merge_exact_ts &&
                         stopping_time_mode == DustStoppingTimeMode::species_fixed;
    cloud_sort_interval = pin->GetOrAddInteger("dust", "cloud_sort_interval", sortdef ? 64 : 0);
  }
  {
    // The Morton sort order is gone; linear (default) and
    // random (a test tool) stay
    // (Codex review of the cleanup: without sorting, cloud_sort_interval = 0, the order never
    // acted; morton is then accepted)
    RemovedKey(pin, "cloud_sort_order", cloud_sort_interval <= 0 ||
               !pin->DoesParameterExist("dust", "cloud_sort_order") ||
               pin->GetString("dust", "cloud_sort_order") != "morton",
               "the Morton order of the particle sort");
    const std::string so = pin->GetOrAddString("dust", "cloud_sort_order", "linear");
    if (so != "linear" && so != "random" && !(so == "morton" && cloud_sort_interval <= 0)) {
      std::cout << "### FATAL ERROR: <dust>/cloud_sort_order must be linear or random" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    cloud_sort_random = (so == "random");   // diagnostic: a random permutation
    // The diagnostic sort before the split, R and merge of the cycle
    // is gone; the sort runs after them
    RemovedKey(pin, "cloud_sort_at_start", cloud_sort_interval <= 0 || KeyFalse(pin, "cloud_sort_at_start"),
               "the sort at the start of the cycle");
    cloud_merge_position_order = pin->GetOrAddBoolean("dust", "cloud_merge_position_order", false);
    // The cell-sorted index is on by default (every 8 cycles) where the two
    // production cloud kernels run (hybrid / PC2 coupling, linear gather); 0 turns it off (the
    // control, and the runs before ce92ed4b).  Rixin: default on also in 3-D (keys
    // (block, k, j, i) since 063823b1; GB200 c640 -2.7%)
    {
      const bool prod = coupling == DustCoupling::hybrid &&
                        hybrid_force_mode != HybridForceMode::split_be && cloud_gather_order == 2;
      cloud_order_interval = pin->GetOrAddInteger("dust", "cloud_order_interval", prod ? 8 : 0);
    }
  }
  std::string users;
  auto user = [&](const bool on, const char *name) {if (on) users += std::string(" ") + name;};
  user(!cloud_split_device, "cloud_split_device=false (host split)");
  user(cloud_merge && !cloud_merge_device, "cloud_merge_device=false (host merge)");
  const std::string mode = pin->GetOrAddString("dust", "cloud_table", "auto");
  bool table = !users.empty();
  if (mode == "true") {
    table = true;
  } else if (mode == "false") {
    if (!users.empty()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dust>/cloud_table = false, but these need the table:" << users << std::endl;
      std::exit(EXIT_FAILURE);
    }
    table = false;
  } else if (mode != "auto") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/cloud_table must be auto, true or false" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // No finest-grid table in 3-D (CheckClouds3D stops the paths that need it)
  if (clouds3d && mode == "true") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/cloud_table = true: no finest-grid table in 3-D" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (clouds3d) table = false;
  if (nlev_mesh < 0 || nvirt < 0 || (table && nlev > 2) ||
      nlev > SMRCompositeTransfer::kCloudLevelMask) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "refinable_cloud requires a uniform mesh or 2:1 static refinement"
              << " (virtual levels included) with at most "
              << (table ? "two levels when the table is built (cloud_table; users:" + users + ")"
                        : std::to_string(SMRCompositeTransfer::kCloudLevelMask) + " levels")
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &t = smr_transfer;
  t.cloud_table = table;
  t.cloud_nlev = nlev;
  t.ncoarse1 = pmesh->mesh_indcs.nx1;
  t.ncoarse2 = pmesh->mesh_indcs.nx2;
  t.nfine1 = t.ncoarse1 << nlev;
  t.nfine2 = t.ncoarse2 << nlev;
  t.x1min = pmesh->mesh_size.x1min;
  t.x1max = pmesh->mesh_size.x1max;
  t.x2min = pmesh->mesh_size.x2min;
  t.x2max = pmesh->mesh_size.x2max;
  t.root_dx1 = (t.x1max - t.x1min)/static_cast<Real>(t.ncoarse1);
  t.root_dx2 = (t.x2max - t.x2min)/static_cast<Real>(t.ncoarse2);
  t.fine_dx1 = t.root_dx1/static_cast<Real>(1 << nlev);
  t.fine_dx2 = t.root_dx2/static_cast<Real>(1 << nlev);
  // controls: cloud_max_level=0 keeps every cloud at root size (translation-invariant
  // dust on the mixed gas mesh); cloud_split_everywhere refines the whole lattice
  t.cloud_max_level = pin->GetOrAddInteger("dust", "cloud_max_level", nlev);
  t.cloud_split_everywhere = pin->GetOrAddBoolean("dust", "cloud_split_everywhere", false);
  // Rixin: split on the first overlap with finer leaves by default (the production
  // rule since the P1 + overlap round; false is the older rule, kept for comparison)
  t.cloud_split_overlap = pin->GetOrAddBoolean("dust", "cloud_split_overlap", true);

  // every block aligned with the finest virtual grid (the actual-leaf traversal relies on it)
  for (int m=0; m<nmb; ++m) {
    const auto &size = mbsize.h_view(m);
    int lev = mblev.h_view(m) - pmesh->root_level;
    int ratio = 1 << (nlev - lev);
    Real off1 = (size.x1min - t.x1min)/t.fine_dx1;
    Real off2 = (size.x2min - t.x2min)/t.fine_dx2;
    int fi0 = static_cast<int>(std::lround(off1));
    int fj0 = static_cast<int>(std::lround(off2));
    // The third direction in 3-D
    bool bad3 = false;
    if (clouds3d) {
      const Real fdx3 = (pmesh->mesh_size.x3max - pmesh->mesh_size.x3min)/
                        static_cast<Real>(pmesh->mesh_indcs.nx3 << nlev);
      const Real off3 = (size.x3min - pmesh->mesh_size.x3min)/fdx3;
      bad3 = std::abs(off3 - std::lround(off3)) > 1.0e-8 ||
             std::abs(size.dx3 - ratio*fdx3) > 1.0e-12*size.dx3;
    }
    if (lev < 0 || lev > nlev || std::abs(off1 - fi0) > 1.0e-8 || std::abs(off2 - fj0) > 1.0e-8 ||
        std::abs(size.dx1 - ratio*t.fine_dx1) > 1.0e-12*size.dx1 ||
        std::abs(size.dx2 - ratio*t.fine_dx2) > 1.0e-12*size.dx2 || bad3) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MeshBlock " << m << " is not aligned with the finest "
                << "virtual grid" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // <dust>/cloud_split_buffer (default 0): the refined dust region
  // reaches this many cells of the next-coarser level beyond the refined gas leaves
  // (periodic Chebyshev dilation of each level's region), so the dust size transition
  // sits on uniform coarse gas.  Only the level map of split/merge eligibility changes;
  // deposit and gather keep the leaf owners.
  cloud_split_buffer = pin->GetOrAddInteger("dust", "cloud_split_buffer", 0);
  if (cloud_split_buffer < 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/cloud_split_buffer must be >= 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (table) {
    t.cloud_target = DvceArray2D<int>("dust_cloud_target", t.nfine2, t.nfine1);
    t.cloud_level = DvceArray2D<int>("dust_cloud_level", t.nfine2, t.nfine1);
    auto target = Kokkos::create_mirror_view(t.cloud_target);
    auto level = Kokkos::create_mirror_view(t.cloud_level);
    Kokkos::deep_copy(target, -1);
    Kokkos::deep_copy(level, -1);
    for (int m=0; m<nmb; ++m) {
      const auto &size = mbsize.h_view(m);
      int lev = mblev.h_view(m) - pmesh->root_level;
      int ratio = 1 << (nlev - lev);
      int fi0 = static_cast<int>(std::lround((size.x1min - t.x1min)/t.fine_dx1));
      int fj0 = static_cast<int>(std::lround((size.x2min - t.x2min)/t.fine_dx2));
      for (int j=0; j<indcs.nx2; ++j) {
        for (int i=0; i<indcs.nx1; ++i) {
          int leaf = EncodeCloudLeaf(m, j + indcs.js, i + indcs.is, ncells2, ncells1);
          for (int sj=0; sj<ratio; ++sj) {
            for (int si=0; si<ratio; ++si) {
              int fj = fj0 + j*ratio + sj;
              int fi = fi0 + i*ratio + si;
              if (fj < 0 || fj >= t.nfine2 || fi < 0 || fi >= t.nfine1 ||
                  target(fj,fi) >= 0) {
                std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                          << std::endl << "virtual fine cell outside the mesh or owned "
                          << "twice" << std::endl;
                std::exit(EXIT_FAILURE);
              }
              target(fj,fi) = leaf;
              level(fj,fi) = lev;
            }
          }
        }
      }
    }
    for (int fj=0; fj<t.nfine2; ++fj) {
      for (int fi=0; fi<t.nfine1; ++fi) {
        if (target(fj,fi) < 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "virtual fine cell has no active leaf owner"
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }
    if (cloud_split_buffer > 0) {
      // a plain host copy: on a host backend the mirror view aliases the map itself
      std::vector<int> mesh_level(static_cast<std::size_t>(t.nfine1)*t.nfine2);
      for (int fj=0; fj<t.nfine2; ++fj) {
        for (int fi=0; fi<t.nfine1; ++fi) {
          mesh_level[static_cast<std::size_t>(fj)*t.nfine1 + fi] = level(fj,fi);
        }
      }
      for (int lev=1; lev<=nlev; ++lev) {
        const int r = cloud_split_buffer << (nlev - lev + 1);   // finest cells per buffer
        for (int fj=0; fj<t.nfine2; ++fj) {
          for (int fi=0; fi<t.nfine1; ++fi) {
            if (mesh_level[static_cast<std::size_t>(fj)*t.nfine1 + fi] < lev) continue;
            for (int dj=-r; dj<=r; ++dj) {
              const int wj = SMRCompositeTransfer::WrapIndex(fj + dj, t.nfine2);
              for (int di=-r; di<=r; ++di) {
                int &l = level(wj, SMRCompositeTransfer::WrapIndex(fi + di, t.nfine1));
                l = std::max(l, lev);
              }
            }
          }
        }
      }
    }
    Kokkos::deep_copy(t.cloud_target, target);
    Kokkos::deep_copy(t.cloud_level, level);
  }

  if (global_variable::my_rank == 0 && !cloud_map_announced) {
    cloud_map_announced = true;
    std::cout << "# refinable_cloud map: levels=" << nlev
              << (nvirt > 0 ? " (" + std::to_string(nvirt) + " virtual)" : "")
              << (table ? ", finest-grid table " : ", no finest-grid table (actual leaves), finest level ")
              << t.nfine1 << "x" << t.nfine2 << ", split slack=" << cloud_split_slack
              << ", max cloud level=" << t.cloud_max_level
              << ", gather order=" << cloud_gather_order
              << (cloud_transition > 0 ? ", P1 transition clouds (mode " +
                  std::to_string(cloud_transition) + ")" : "")
              << ", moving P1 split without merging"
              << (cloud_split_buffer > 0 ? ", dust split buffer " +
                  std::to_string(cloud_split_buffer) + " coarse cells" : "")
              << (t.cloud_split_everywhere ? ", split everywhere" : "")
              << (t.cloud_split_overlap ? ", split on first overlap with finer leaves" : "")
              << std::endl;
  }
  if (clouds3d) {BuildLeafTraversal3D();} else {BuildLeafTraversal();}
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RequireCloudTable
//! \brief Stop when a path that reads the virtual finest-grid table runs without
//! it (<dust>/cloud_table = auto builds it for the paths known at the start; a problem
//! generator's split oracle, for example, is set later).

void DustGasDrag::RequireCloudTable(const char *what) const {
  if (smr_transfer.cloud_table) return;
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << what << " reads the virtual finest-grid table, which was not built: set "
            << "<dust>/cloud_table = true (at most two levels of refinement)" << std::endl;
  std::exit(EXIT_FAILURE);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::EnsureNextTag
//! \brief The first tag of created particles, once: the largest tag of all particles + 1.
//! It used to be the particle count at the first creation, which repeats tags when
//! particles were removed before (a resampling at t = 0 before the first split of a run that
//! starts on the root grid: D6 had 251 tags twice).  Called at the start of every cycle, so
//! that it is set before anything removes particles; restarts carry the counter.

void DustGasDrag::EnsureNextTag() {
  if (cloud_next_tag_set) return;
  // Each rank hands out base + rank, then steps by the number of ranks;
  // base = 1 + the largest live tag (as before), at least the bound of the restart file this run
  // segment started from (tags of deleted particles are not issued again)
  std::uint64_t base = std::max(LiveTagBound(), cloud_tag_floor);
  const std::uint64_t rank = static_cast<std::uint64_t>(global_variable::my_rank);
  if (base > UINT64_MAX - rank) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust particle tags: the 64-bit tag space is exhausted (base " << base
              << ", rank " << rank << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  cloud_next_tag = base + rank;
  cloud_next_tag_set = true;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::IssueTags
//! \brief The first of n new tags of this rank (first, first + nranks, ...); the
//! counter moves on by n*nranks.  Checked: a counter that would pass 2^64 - 1 stops the run.

std::uint64_t DustGasDrag::IssueTags(const std::uint64_t n) {
  if (!cloud_next_tag_set) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust particle tags issued before the counter was set (EnsureNextTag)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const std::uint64_t stride = static_cast<std::uint64_t>(global_variable::nranks);
  const std::uint64_t first = cloud_next_tag;
  if (n > 0 && (n > UINT64_MAX/stride || n*stride > UINT64_MAX - first)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust particle tags: the 64-bit tag space is exhausted (next tag " << first
              << ", " << n << " new tags, stride " << stride << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  cloud_next_tag = first + n*stride;
  return first;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::LiveTagBound, DustGasDrag::IssuedTagBound
//! \brief Both collective: 1 + the largest tag of the particles of all ranks (0 if
//! there is none); a bound above every tag issued so far: the largest of LiveTagBound, the
//! counters of all ranks and the bound this run segment started from.

std::uint64_t DustGasDrag::LiveTagBound() const {
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  std::uint64_t mx = 0;
  Kokkos::parallel_reduce("dust_max_tag", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, std::uint64_t &m) {
    const std::uint64_t t = GetTag(pi,p);
    if (t > m) m = t;
  }, Kokkos::Max<std::uint64_t>(mx));
  // (2^64 - 1 is never issued: IssueTags stops before the counter passes it)
  if (npart > 0 && mx == UINT64_MAX) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dust particle tags: a particle has the tag 2^64 - 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::uint64_t b = (npart > 0) ? mx + 1 : 0;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &b, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
#endif
  return b;
}

std::uint64_t DustGasDrag::IssuedTagBound() const {
  std::uint64_t b = std::max(cloud_tag_floor, cloud_next_tag_set ? cloud_next_tag : 0);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &b, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
#endif
  return std::max(b, LiveTagBound());
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MigrateClouds
//! \brief Several ranks only.  A cloud created or moved in RefineClouds whose
//! block is on another rank keeps its parent's block (NewCloudBlock); the AthenaK particle
//! exchange moves it to the block that holds it (at most one block away).  One rank: the
//! blocks are set already, nothing to do.

void DustGasDrag::MigrateClouds() {
  if (global_variable::nranks == 1) return;
  {
    const auto tw = std::chrono::steady_clock::now();
    Kokkos::fence();   // the kernels before (their time, not the migration's)
    mpi_time[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count();
  }
  const auto t0 = std::chrono::steady_clock::now();
  FrameToBlock();
  auto *pb = pmy_pack->ppart->pbval_part;
  pb->SetNewPrtclGID();
  pb->CountSendsAndRecvs();
  pb->InitPrtclRecv();
  pb->PackAndSendPrtcls();
  while (pb->RecvAndUnpackPrtcls() != TaskStatus::complete) {}
  pb->ClearPrtclSend();
  pb->ClearPrtclRecv();
  // the counts of all ranks are known now (RecvAndUnpackPrtcls): the total for the outputs
  Mesh *pm = pmy_pack->pmesh;
  int tot = 0;
  for (int n=0; n<global_variable::nranks; ++n) tot += pm->nprtcl_eachrank[n];
  pm->nprtcl_total = tot;
  mpi_time[0] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++mpi_count[0];
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitToLeaves
//! \brief The initial split of a problem generator: split passes until no cloud anywhere can
//! be split (one level per pass).  Several ranks: after every pass the pieces go
//! to the rank of the block that holds them, by position (RouteToOwnerBlocks: a root cloud
//! over a deep mesh has children several small blocks away, further than the neighbour step
//! of the particle exchange), and the loop ends when no rank split anything.  One rank: the
//! passes alone, as before.  Returns the number of particles split.

int DustGasDrag::SplitToLeaves() {
  long long total = 0;
  while (true) {
    long long n = SplitAndMergeClouds();
    total += n;
    if (global_variable::nranks > 1) RouteToOwnerBlocks(nullptr);
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) {
      MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }
#endif
    if (n == 0) break;
  }
  return static_cast<int>(total);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckHaloReach
//! \brief Several ranks: the actual supports against the reach of the halo.  For
//! every cloud, how far its support (two cells of its level on a lobed side, one otherwise)
//! extends beyond the block that holds it; where that exceeds the block's allowance, the
//! block's reach is widened (until the mesh changes) and every rank rebuilds the leaf traversal
//! and the halo.  The level rule (HaloReach) is then only an estimate: a case it misses costs a
//! rebuild, not a lost deposit.  Returns whether the halo was rebuilt.

bool DustGasDrag::CheckHaloReach(const char *when) {
  if (global_variable::nranks == 1) return false;
  if (clouds3d) return CheckHaloReach3D(when);
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  const int nmb = pmy_pack->nmb_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const auto lt = leaf_traversal;
  auto allow = halo_allow_d_;
  DvceArray1D<std::int64_t> need("dust_halo_need", std::max(nmb, 1));
  int nover = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_halo_reach", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &cnt) {
      const int m = lt.Anchor(pi(PGID,p));
      Real x = pr(IPX,p), y = pr(IPY,p);
      lt.Frame(m, x, y);
      const int code = pi(PLEV,p);
      const int lev = SMRCompositeTransfer::CloudLevel(code);
      const Real e1 = (SMRCompositeTransfer::CloudLobe1(code) ? 2.0 : 1.0)*lt.CloudHalfWidth1(lev);
      const Real e2 = (SMRCompositeTransfer::CloudLobe2(code) ? 2.0 : 1.0)*lt.CloudHalfWidth2(lev);
      const Real bx0 = lt.box(m,0), bx1 = bx0 + lt.nx1*lt.box(m,2);
      const Real by0 = lt.box(m,1), by1 = by0 + lt.nx2*lt.box(m,3);
      const Real o1 = Kokkos::fmax(0.0, Kokkos::fmax(bx0 - (x - e1), (x + e1) - bx1));
      const Real o2 = Kokkos::fmax(0.0, Kokkos::fmax(by0 - (y - e2), (y + e2) - by1));
      const std::int64_t d = static_cast<std::int64_t>(Kokkos::fmax(Kokkos::ceil(o1/lt.fine_dx1),
                                                                     Kokkos::ceil(o2/lt.fine_dx2)));
      if (d > allow(m)) {
        const std::int64_t margin = static_cast<std::int64_t>(1) << (lt.nlev - lt.blev(m));
        Kokkos::atomic_max(&need(m), d + margin);
        ++cnt;
      }
    }, Kokkos::Sum<int>(nover));
  }
  long long tot = nover;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &tot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (tot == 0) return false;
  auto nh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), need);
  int nwide = 0;
  for (int m=0; m<nmb; ++m) {
    if (nh(m) > halo_need_[m]) {halo_need_[m] = nh(m); ++nwide;}
  }
  halo_widenings_ += static_cast<unsigned long long>(nwide);
  const int nhalo_before = leaf_traversal.nfld - leaf_traversal.nmb;
  BuildLeafTraversal();
  if (global_variable::my_rank == 0) {
    std::cout << "# dust halo: " << tot << " clouds reached beyond the halo ("
              << when << ", cycle " << pmy_pack->pmesh->ncycle << "); rank 0 widened " << nwide
              << " blocks, halo blocks " << nhalo_before << " -> "
              << leaf_traversal.nfld - leaf_traversal.nmb << std::endl;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::FrameToBlock
//! \brief Several ranks: a new cloud's position is wrapped into the domain, but
//! the block it keeps may be across a periodic boundary.  Put such particles in the image next
//! to their block (positions and the RK position registers together), so that the particle
//! exchange sees a step of one block at most and wraps them again.

void DustGasDrag::FrameToBlock(const Real tpos) {
  if (clouds3d && (global_variable::nranks > 1 || shear3d)) {   // with shear3d also on one rank
    FrameToBlock3D(tpos);
    return;
  }
  if (global_variable::nranks == 1) return;
  {
    auto pr = pmy_pack->ppart->prtcl_rdata;
    auto pi = pmy_pack->ppart->prtcl_idata;
    const auto lt = leaf_traversal;
    const int npart = pmy_pack->ppart->nprtcl_thispack;
    Kokkos::parallel_for("dust_migrate_frame", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      const int m = lt.Anchor(pi(PGID,p));
      const Real cx = lt.box(m,0) + 0.5*lt.nx1*lt.box(m,2), cy = lt.box(m,1) + 0.5*lt.nx2*lt.box(m,3);
      const Real sx = lt.lx*round((pr(IPX,p) - cx)*lt.ilx), sy = lt.ly*round((pr(IPY,p) - cy)*lt.ily);
      if (sx != 0.0) {pr(IPX,p) -= sx; pr(IPX1,p) -= sx;}
      if (sy != 0.0) {pr(IPY,p) -= sy; pr(IPY1,p) -= sy;}
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RefineClouds
//! \brief Task wrapper: split eligible clouds at the start of every cycle.

TaskStatus DustGasDrag::RefineClouds(Driver *pdrive, int stage) {
  if (smr_transfer_mode != DustSMRTransfer::refinable_cloud) return TaskStatus::complete;
  ProfTimer prof_all(prof_time[kProfRefineClouds], prof_count[kProfRefineClouds]);
  UpdateShear3D(pmy_pack->pmesh->time);   // the positions at the start of the step
  EnsureNextTag();
  {
    ProfTimer pt(prof_time[kProfSplit], prof_count[kProfSplit]);
    SplitAndMergeClouds();
  }
  // the merge clock is read once per cycle (MergeDue advances it)
  const bool merge_due = cloud_merge && MergeDue();
  if (amr_event_pending && !(cloud_resample && cloud_resample_after_regrid)) {
    amr_event_pending = false;          // no resampling after regrids: nothing to do
    amr_event_nnew = amr_event_ndel = 0;
  }
  // 2026-10-21 (scaling): the event may wait cloud_event_dt after its first regrid
  bool event_now = amr_event_pending;
  if (event_now && cloud_event_dt > 0.0) {
    const double t = pmy_pack->pmesh->time;
    event_now = t >= amr_event_first_time + cloud_event_dt - 1.0e-12*std::fmax(1.0, std::fabs(t));
  }
  if (event_now) {
    // The cycle after a regrid: merge pass first, then one resampling
    ProfTimer pt(prof_time[kProfEvent], prof_count[kProfEvent]);
    HandleRegridEvent(merge_due);
  } else {
    // Resampling before the merge pass, so that an audit-only pass of the same cycle
    // writes the resampled state; it is due every cloud_resample_interval cycles, or at the
    // planned physical times
    const bool resample_due = cloud_resample && ResampleDue();
    // Several ranks: the split pieces go to the rank of their block before R and G
    // (other cycles leave them to the migration after the push; until then they keep a block
    // of this rank as their anchor, which the leaf traversal allows)
    if (resample_due || merge_due) {
      MigrateClouds();
      CheckHaloReach("split");   // R and G traverse the supports too
    }
    if (resample_due) {
      ProfTimer pt(prof_time[kProfSchedR], prof_count[kProfSchedR]);
      ResampleClouds();
    }
    if (merge_due) {
      ProfTimer pt(prof_time[kProfSchedG], prof_count[kProfSchedG]);
      MergeClouds();
    }
  }
#if MPI_PARALLEL_ENABLED
  // the counts of all ranks for the outputs (the split, R and G changed them)
  if (global_variable::nranks > 1) {
    Mesh *pm = pmy_pack->pmesh;
    int n = pmy_pack->ppart->nprtcl_thispack;
    MPI_Allgather(&n, 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
    int tot = 0;
    for (int r=0; r<global_variable::nranks; ++r) tot += pm->nprtcl_eachrank[r];
    pm->nprtcl_total = tot;
    pm->nprtcl_thisrank = n;
  }
#endif
  CheckHaloReach("refine");   // before this cycle's gathers and deposits
  if (cloud_sort_interval > 0 &&
      pmy_pack->pmesh->ncycle % cloud_sort_interval == 0) {
    ProfTimer pt(prof_time[kProfSort], prof_count[kProfSort]);
    SortParticles();
  }
  if (cloud_order_interval > 0) {
    ProfTimer pt(prof_time[kProfOrder], prof_count[kProfOrder]);
    UpdateCloudOrder();
  }
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::HandleRegridEvent
//! \brief The first cycle after a regrid.  The split of this cycle
//! has run (the particle refinement of the regrid).  With merging on, a merge pass runs next
//! when one is due anyway or when the regrid deleted blocks (the first pass of the burst,
//! whatever the phase of cloud_merge_burst_interval).  Then one resampling: a scheduled
//! resampling of this cycle is the same one (its event number advances as usual); otherwise
//! the schedule is not touched.  The event is done even if nothing could be resampled.  One
//! line per event in the log.

void DustGasDrag::HandleRegridEvent(const bool merge_due) {
  Mesh *pmesh = pmy_pack->pmesh;
  particles::Particles *ppar = pmy_pack->ppart;
  const int nnew = amr_event_nnew, ndel = amr_event_ndel;
  const char *kind = (nnew > 0 && ndel > 0) ? "mixed" : ((ndel > 0) ? "derefine" : "refine");
  const bool merge_ok = cloud_merge;
  const bool do_merge = merge_due || (merge_ok && ndel > 0);
  const char *why = !do_merge ? "none"
                  : ((merge_due && ndel > 0) ? "due+derefine" : (merge_due ? "due" : "derefine"));
  MigrateClouds();   // every cloud in its block before G and R
  CheckHaloReach("regrid event");
  const int n0 = ppar->nprtcl_thispack;
  Kokkos::Timer timer;
  if (do_merge) {
    MergeClouds();
    // several ranks: a merged cloud whose block is on another rank keeps its parent's block
    // (NewCloudBlock); R bins every cloud by the leaf that holds it, so it moves first.  The
    // cycle path runs R before G and leaves this to the push migration.  (GB200 AMR pilot,
    // 2026-10-17: the first derefinement on 4 ranks aborted in R.)
    MigrateClouds();
    CheckHaloReach("regrid event merge");
  }
  Kokkos::fence();
  const double t_merge = timer.seconds();
  const int n1 = ppar->nprtcl_thispack;
  const bool scheduled = ResampleDue();
  // Rixin: no extra resampling after a pure derefinement; G has run on
  // the coarsened region, the scheduled resampling still reaches everything
  const bool skip_r = !scheduled && nnew == 0 && ndel > 0 && !cloud_resample_after_derefine;
  timer.reset();
  // the blocks changed by every regrid of this event are fresh for the local R
  amr_event_max_age = (amr_event_first_ncycle >= 0)
                    ? static_cast<int>(pmesh->ncycle - amr_event_first_ncycle) : 0;
  const int removed = skip_r ? 0 : ResampleClouds(cloud_resample_event_local && !scheduled);
  amr_event_max_age = 0;
  Kokkos::fence();
  const double t_resample = timer.seconds();
  const int n2 = ppar->nprtcl_thispack;
  ++amr_event_handled;
  if (scheduled) ++amr_event_merged;
  // several ranks: totals over the ranks (Codex review), the slowest rank's times
  long long nn[4] = {n0, n1, n2, removed};
  double tt[2] = {t_merge, t_resample};
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, nn, 4, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, tt, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank == 0) {
    std::cout << "# regrid event " << amr_event_handled << ": regrid "
              << amr_event_regrid << ", " << kind << " (created " << nnew << ", deleted " << ndel
              << "), cycle " << pmesh->ncycle << ", time " << pmesh->time << ": merge pass "
              << why << ", resampling " << (scheduled ? "(the scheduled one of this cycle)"
                                                      : (skip_r ? "skipped (derefinement)"
                                                                : "(extra)"))
              << " removed " << nn[3] << " of " << nn[1] << std::endl;
  }
  amr_event_pending = false;
  amr_event_nnew = amr_event_ndel = 0;
  amr_event_first_time = -1.0e300;
  amr_event_first_ncycle = -1;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitAndMergeClouds
//! \brief The split of the cycle: the moving P1 split (SplitCloudsP1, on the device by
//! default).

int DustGasDrag::SplitAndMergeClouds() {
  if (!cloud_split_enable) return 0;   // test switch
  return SplitCloudsP1();
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitCloudsP1
//! \brief Moving-particle P1 split (<dust>/cloud_p1_split), called by
//! SplitAndMergeClouds.  A plain level-l cloud whose support lies on finer leaves is
//! split, but along each direction an edge child is separated only if the lattice
//! neighbour on that side (the virtual level-l cloud one parent spacing H away) would
//! be split too; otherwise the edge child stays attached to the centre child as a
//! transition lobe.  A lobed cloud separates a lobe once that neighbour becomes
//! eligible.  Along a direction, the transition band of eligible parents with an
//! ineligible neighbour is exactly H wide, so every lattice row holds one transition
//! cloud whatever its phase and the particle basis stays P1 while the lattice drifts
//! through the split surface.  Nothing is ever merged: every particle keeps its own
//! attributes, so independent streams and stopping times are preserved.

int DustGasDrag::SplitCloudsP1() {
  if (cloud_split_device) {
    return clouds3d ? SplitCloudsP1Device3D() : SplitCloudsP1Device();
  }
  RequireCloudTable("the host P1 split (cloud_split_device = false)");
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  auto transfer = smr_transfer;
  Real slack = cloud_split_slack;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;

  // eligible(x, y, l): a plain level-l cloud at (x, y) would be split
  DvceArray1D<int> flag("dust_cloud_flag", npart);
  int nflag = 0;
  Kokkos::parallel_reduce("dust_cloud_p1_flag",
  Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int p, int &sum) {
    const int code = pi(PLEV,p);
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    const int l1 = SMRCompositeTransfer::CloudLobe1(code);
    const int l2 = SMRCompositeTransfer::CloudLobe2(code);
    const Real x = pr(IPX,p), y = pr(IPY,p);
    int f = 0;
    if (l1 == 0 && l2 == 0) {
      f = transfer.CloudInsideFiner(x, y, lev, slack) ? 1 : 0;
    } else if (lev > 0 && transfer.CloudInsideFiner(x, y, lev - 1, slack)) {
      const Real h1 = transfer.CloudHalfWidth1(lev - 1);
      const Real h2 = transfer.CloudHalfWidth2(lev - 1);
      if (((l1 & 1) && transfer.CloudInsideFiner(x - h1, y, lev - 1, slack)) ||
          ((l1 & 2) && transfer.CloudInsideFiner(x + h1, y, lev - 1, slack)) ||
          ((l2 & 1) && transfer.CloudInsideFiner(x, y - h2, lev - 1, slack)) ||
          ((l2 & 2) && transfer.CloudInsideFiner(x, y + h2, lev - 1, slack))) f = 1;
    }
    flag(p) = f;
    sum += f;
  }, Kokkos::Sum<int>(nflag));
  if (nflag == 0) return 0;

  auto pr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
  auto pi_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pi);
  auto flag_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), flag);
  auto target_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                      transfer.cloud_target);
  // host copy of the transfer for the eligibility tests (views point to host mirrors)
  auto level_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_level);
  EnsureNextTag();
  const Real x1min = transfer.x1min, x2min = transfer.x2min;
  const Real len1 = transfer.x1max - transfer.x1min;
  const Real len2 = transfer.x2max - transfer.x2min;
  const int nfine1 = transfer.nfine1, nfine2 = transfer.nfine2;
  const Real fdx1 = transfer.fine_dx1, fdx2 = transfer.fine_dx2;
  const int gids = pmy_pack->gids;
  const int ncells1 = qdep.extent_int(4);
  const int ncells2 = qdep.extent_int(3);
  auto wrap = [](Real x, Real xmin, Real len) {
    while (x < xmin) x += len;
    while (x >= xmin + len) x -= len;
    return x;
  };
  auto owner_gid = [&](Real x, Real y) {
    int fi = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((x - x1min)/fdx1)), nfine1);
    int fj = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((y - x2min)/fdx2)), nfine2);
    return gids + target_h(fj, fi)/(ncells1*ncells2);
  };
  // host version of SMRCompositeTransfer::CloudInsideFiner for a plain level-l cloud
  auto eligible = [&](Real x, Real y, int lev) {
    if (lev >= transfer.cloud_nlev || lev >= transfer.cloud_max_level) return false;
    if (transfer.cloud_split_everywhere) return true;
    x = wrap(x, x1min, len1);
    y = wrap(y, x2min, len2);
    Real r1 = (1.0 - slack)*transfer.CloudHalfWidth1(lev);
    Real r2 = (1.0 - slack)*transfer.CloudHalfWidth2(lev);
    int ia = static_cast<int>(std::floor((x - r1 - x1min)/fdx1));
    int ib = static_cast<int>(std::ceil((x + r1 - x1min)/fdx1)) - 1;
    int ja = static_cast<int>(std::floor((y - r2 - x2min)/fdx2));
    int jb = static_cast<int>(std::ceil((y + r2 - x2min)/fdx2)) - 1;
    if (transfer.cloud_split_overlap) {
      for (int fj=ja; fj<=jb; ++fj) {
        int wj = SMRCompositeTransfer::WrapIndex(fj, nfine2);
        for (int fi=ia; fi<=ib; ++fi) {
          if (level_h(wj, SMRCompositeTransfer::WrapIndex(fi, nfine1)) > lev) return true;
        }
      }
      return false;
    }
    for (int fj=ja; fj<=jb; ++fj) {
      int wj = SMRCompositeTransfer::WrapIndex(fj, nfine2);
      for (int fi=ia; fi<=ib; ++fi) {
        if (level_h(wj, SMRCompositeTransfer::WrapIndex(fi, nfine1)) <= lev) return false;
      }
    }
    return true;
  };

  // pieces of a cloud along one direction: offset (in child half-widths), fraction of
  // the cloud's mass, lobe code.  lobe_old is the cloud's lobe set along the direction
  // (3 for a plain parent, whose shape is the centre child with both edge children
  // attached), sep_lo/sep_hi say which attached edge children are separated now.
  struct Piece {Real off; Real frac; int lobe;};
  auto pieces = [](int lobe_old, bool sep_lo, bool sep_hi, Piece out[3]) {
    int n = 0;
    Real total = 0.5 + 0.25*((lobe_old & 1) ? 1 : 0) + 0.25*((lobe_old & 2) ? 1 : 0);
    bool lo = (lobe_old & 1) && sep_lo, hi = (lobe_old & 2) && sep_hi;
    int lobe_new = lobe_old & ~((lo ? 1 : 0) | (hi ? 2 : 0));
    Real keep = 0.5 + 0.25*((lobe_new & 1) ? 1 : 0) + 0.25*((lobe_new & 2) ? 1 : 0);
    out[n++] = {0.0, keep/total, lobe_new};
    if (lo) out[n++] = {-1.0, 0.25/total, 0};
    if (hi) out[n++] = {1.0, 0.25/total, 0};
    return n;
  };

  const int capacity = npart - nflag + 9*nflag;
  std::vector<Real> R(static_cast<std::size_t>(nrdata)*capacity, 0.0);
  std::vector<int> I(static_cast<std::size_t>(nidata)*capacity, 0);
  auto rr = [&](int c, int s) -> Real& { return R[static_cast<std::size_t>(s)*nrdata + c]; };
  auto ii = [&](int c, int s) -> int& { return I[static_cast<std::size_t>(s)*nidata + c]; };
  int nslot = 0;
  for (int p=0; p<npart; ++p) {
    if (flag_h(p)) continue;
    for (int c=0; c<nrdata; ++c) rr(c,nslot) = pr_h(c,p);
    for (int c=0; c<nidata; ++c) ii(c,nslot) = pi_h(c,p);
    ++nslot;
  }

  // call-level leaf deposit of all flagged clouds (side 0) and all their pieces (side 1)

  unsigned long long nparent = 0, nlobe = 0;
  for (int p=0; p<npart; ++p) {
    if (!flag_h(p)) continue;
    const int code = pi_h(PLEV,p);
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    const Real x = pr_h(IPX,p), y = pr_h(IPY,p);
    Piece p1[3], p2[3];
    int n1, n2, clev;
    if (SMRCompositeTransfer::CloudLobe1(code) == 0 &&
        SMRCompositeTransfer::CloudLobe2(code) == 0) {
      // plain parent: children at level lev+1, edges separated where the neighbour
      // parent is eligible as well
      const Real h1 = transfer.CloudHalfWidth1(lev), h2 = transfer.CloudHalfWidth2(lev);
      n1 = pieces(3, eligible(x - h1, y, lev), eligible(x + h1, y, lev), p1);
      n2 = pieces(3, eligible(x, y - h2, lev), eligible(x, y + h2, lev), p2);
      clev = lev + 1;
      ++nparent;
    } else {
      // lobed cloud of level lev (parent level lev-1): separate the lobes whose
      // neighbour parent has become eligible
      const Real h1 = transfer.CloudHalfWidth1(lev - 1);
      const Real h2 = transfer.CloudHalfWidth2(lev - 1);
      n1 = pieces(SMRCompositeTransfer::CloudLobe1(code), eligible(x - h1, y, lev - 1),
                  eligible(x + h1, y, lev - 1), p1);
      n2 = pieces(SMRCompositeTransfer::CloudLobe2(code), eligible(x, y - h2, lev - 1),
                  eligible(x, y + h2, lev - 1), p2);
      clev = lev;
      ++nlobe;
    }
    const Real e1 = transfer.CloudHalfWidth1(clev), e2 = transfer.CloudHalfWidth2(clev);
    for (int b=0; b<n2; ++b) {
      for (int a=0; a<n1; ++a) {
        int s = nslot++;
        Real cx = wrap(x + p1[a].off*e1, x1min, len1);
        Real cy = wrap(y + p2[b].off*e2, x2min, len2);
        for (int c=0; c<nrdata; ++c) rr(c,s) = pr_h(c,p);
        for (int c=0; c<nidata; ++c) ii(c,s) = pi_h(c,p);
        rr(IPX,s) = cx; rr(IPX1,s) = cx;
        rr(IPY,s) = cy; rr(IPY1,s) = cy;
        rr(IPM,s) = pr_h(IPM,p)*p1[a].frac*p2[b].frac;
        ii(PLEV,s) = clev | (p1[a].lobe << sh1) | (p2[b].lobe << sh2);
        ii(PGID,s) = owner_gid(cx, cy);
        if (a > 0 || b > 0) {   // the centre keeps the tag
          const std::uint64_t t = IssueTags(1);
          ii(PTAG,s) = TagLo(t); ii(PTAGH,s) = TagHi(t);
        }
      }
    }
  }
  cloud_split_count += nparent;
  cloud_lobe_split_count += nlobe;

  int nnew = nslot;
  Kokkos::realloc(pr, nrdata, nnew);
  Kokkos::realloc(pi, nidata, nnew);
  auto pr_new = Kokkos::create_mirror_view(pr);
  auto pi_new = Kokkos::create_mirror_view(pi);
  for (int s=0; s<nnew; ++s) {
    for (int c=0; c<nrdata; ++c) pr_new(c,s) = rr(c,s);
    for (int c=0; c<nidata; ++c) pi_new(c,s) = ii(c,s);
  }
  Kokkos::deep_copy(pr, pr_new);
  Kokkos::deep_copy(pi, pi_new);
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;
  return nflag;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MergeClouds
//! \brief Coarsening of refinable clouds (<dust>/cloud_merge).  A
//! cell of the level-L grid is a merge cell when no leaf finer than L lies within
//! cloud_merge_buffer level-L cells of it (periodic, Chebyshev distance).  In merge cells
//!  (1) lobed level-(L+1) clouds release all lobes into plain pieces (deposit, mass and
//!      momentum unchanged; the transition-2 point gather becomes the cloud integral);
//!  (2) plain level-(L+1) pieces are merged by the P1 tent merge described in the body:
//!      the exact inverse of the two-scale split on a uniform lattice and at a P1 split
//!      edge, and a second-order coarse graining of perturbations; for irregular particles
//!      the tent shares are approximate and what is left of a piece waits for a later
//!      centre.
//! Mass and momentum are conserved exactly; the kinetic energy removed is split into the
//! part of the fitted velocity gradient and the residual part.  Host prototype for a
//! single rank, like SplitCloudsP1; the algorithm only looks at a centre's tent support.
int DustGasDrag::MergeClouds() {
  ++gravity_mass_epoch;   // G changes the deposited mass (gravity field reuse)
  if (clouds3d) return MergeCloudsDevice3D();
  if (cloud_merge_device) return MergeCloudsDevice();
  RequireCloudTable("the host merge pass (cloud_merge_device = false)");
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart = ppar->nprtcl_thispack;
  const auto &transfer = smr_transfer;
  const int nlev = transfer.cloud_nlev;
  if (npart == 0 || nlev == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
  auto pr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
  auto pi_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pi);
  auto level_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_level);
  auto target_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_target);
  EnsureNextTag();

  const Real x1min = transfer.x1min, x2min = transfer.x2min;
  const Real len1 = transfer.x1max - transfer.x1min;
  const Real len2 = transfer.x2max - transfer.x2min;
  const int nfine1 = transfer.nfine1, nfine2 = transfer.nfine2;
  const Real fdx1 = transfer.fine_dx1, fdx2 = transfer.fine_dx2;
  const int gids = pmy_pack->gids;
  const int ncells1 = qdep.extent_int(4);
  const int ncells2 = qdep.extent_int(3);
  const int B = cloud_merge_buffer;
  auto wrap = [](Real x, Real xmin, Real len) {
    while (x < xmin) x += len;
    while (x >= xmin + len) x -= len;
    return x;
  };
  auto owner_gid = [&](Real x, Real y) {
    int fi = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((x - x1min)/fdx1)), nfine1);
    int fj = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((y - x2min)/fdx2)), nfine2);
    return gids + target_h(fj, fi)/(ncells1*ncells2);
  };

  // merge-cell masks of the level-L grids, L = 0 .. nlev-1
  std::vector<std::vector<char>> merge_cell(nlev);
  std::vector<int> ng1(nlev), ng2(nlev);
  for (int L=0; L<nlev; ++L) {
    const int r = 1 << (nlev - L);
    const int n1 = nfine1/r, n2 = nfine2/r;
    ng1[L] = n1; ng2[L] = n2;
    std::vector<int> maxlev(static_cast<std::size_t>(n1)*n2, -1);
    for (int fj=0; fj<nfine2; ++fj) {
      for (int fi=0; fi<nfine1; ++fi) {
        int &m = maxlev[static_cast<std::size_t>(fj/r)*n1 + fi/r];
        m = std::max(m, static_cast<int>(level_h(fj, fi)));
      }
    }
    merge_cell[L].assign(static_cast<std::size_t>(n1)*n2, 0);
    for (int J=0; J<n2; ++J) {
      for (int I=0; I<n1; ++I) {
        bool ok = true;
        for (int dj=-B; dj<=B && ok; ++dj) {
          const int wj = SMRCompositeTransfer::WrapIndex(J + dj, n2);
          for (int di=-B; di<=B; ++di) {
            if (maxlev[static_cast<std::size_t>(wj)*n1 +
                       SMRCompositeTransfer::WrapIndex(I + di, n1)] > L) {ok = false; break;}
          }
        }
        merge_cell[L][static_cast<std::size_t>(J)*n1 + I] = ok ? 1 : 0;
      }
    }
  }
  // merge cell (index into merge_cell[L]) of a position, or -1
  auto merge_index = [&](Real x, Real y, int L) {
    const Real dx1 = transfer.root_dx1/static_cast<Real>(1 << L);
    const Real dx2 = transfer.root_dx2/static_cast<Real>(1 << L);
    const int I = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((wrap(x, x1min, len1) - x1min)/dx1)), ng1[L]);
    const int J = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((wrap(y, x2min, len2) - x2min)/dx2)), ng2[L]);
    const int c = J*ng1[L] + I;
    return merge_cell[L][c] ? c : -1;
  };

  // working list: every particle, plus the pieces of released lobes
  std::vector<std::vector<Real>> R;
  std::vector<std::vector<int>> I;
  R.reserve(npart + npart/2);
  I.reserve(npart + npart/2);
  std::vector<Real> rrow(nrdata);
  std::vector<int> irow(nidata);
  unsigned long long nrelease = 0;
  struct Piece {Real off; Real frac; int lobe;};
  auto release = [](int lobe_old, Piece out[3]) {   // separate every attached lobe
    int n = 0;
    Real total = 0.5 + 0.25*((lobe_old & 1) ? 1 : 0) + 0.25*((lobe_old & 2) ? 1 : 0);
    out[n++] = {0.0, 0.5/total, 0};
    if (lobe_old & 1) out[n++] = {-1.0, 0.25/total, 0};
    if (lobe_old & 2) out[n++] = {1.0, 0.25/total, 0};
    return n;
  };
  for (int p=0; p<npart; ++p) {
    for (int c=0; c<nrdata; ++c) rrow[c] = pr_h(c,p);
    for (int c=0; c<nidata; ++c) irow[c] = pi_h(c,p);
    const int code = irow[PLEV];
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    const int l1 = SMRCompositeTransfer::CloudLobe1(code);
    const int l2 = SMRCompositeTransfer::CloudLobe2(code);
    if ((l1 == 0 && l2 == 0) || lev == 0 || merge_index(rrow[IPX], rrow[IPY], lev - 1) < 0) {
      R.push_back(rrow); I.push_back(irow);
      continue;
    }
    Piece p1[3], p2[3];
    const int n1 = (l1 == 0) ? 1 : release(l1, p1);
    const int n2 = (l2 == 0) ? 1 : release(l2, p2);
    if (l1 == 0) p1[0] = {0.0, 1.0, 0};
    if (l2 == 0) p2[0] = {0.0, 1.0, 0};
    const Real e1 = transfer.CloudHalfWidth1(lev), e2 = transfer.CloudHalfWidth2(lev);
    for (int b=0; b<n2; ++b) {
      for (int a=0; a<n1; ++a) {
        std::vector<Real> r = rrow;
        std::vector<int> i = irow;
        const Real cx = wrap(rrow[IPX] + p1[a].off*e1, x1min, len1);
        const Real cy = wrap(rrow[IPY] + p2[b].off*e2, x2min, len2);
        r[IPX] = cx; r[IPX1] = cx; r[IPY] = cy; r[IPY1] = cy;
        r[IPM] = rrow[IPM]*p1[a].frac*p2[b].frac;
        i[PLEV] = lev;
        i[PGID] = owner_gid(cx, cy);
        if (a > 0 || b > 0) {
          const std::uint64_t t = IssueTags(1);
          i[PTAG] = TagLo(t); i[PTAGH] = TagHi(t);
        }
        R.push_back(r); I.push_back(i);
      }
    }
    ++nrelease;
  }

  // P1 tent merge, per fine level l >= 1 (target L = l-1) and species.
  // m_ref(L) = m_root(s)/4^L is the mass of a level-L cloud in the initial sampling and
  // m_node = m_ref/4 the expected mass of one level-l node of a uniform fine lattice.  A
  // centre is a zone piece with no level-L cloud and no other centre within its tent support
  // (|dx| < H_L, |dy| < H_L).  It commits only when every level-l piece within that support
  // is in the merge zone; it then takes its own mass and, from each neighbour, the share
  // w*m_node (w the tent weight, at most what the neighbour has left), and becomes one plain
  // level-L cloud at the centroid of what it took.  The deposit is then unchanged exactly on
  // a uniform lattice and at a P1 split edge; neighbours keep the rest for the next centre.
  // Contributions whose velocity residual about the mass-weighted fit exceeds
  // alpha*H_L/tau_c are left out; a commit lighter than cloud_merge_min_mass*m_ref waits.
  const int nw = static_cast<int>(R.size());
  if (cloud_merge_mroot.empty()) {
    int nsp = 1;
    for (int s=0; s<nw; ++s) nsp = std::max(nsp, I[s][PSP] + 1);
    cloud_merge_mroot.assign(nsp, 0.0);
    std::vector<Real> m1(nsp, 0.0);
    for (int s=0; s<nw; ++s) {
      const int code = I[s][PLEV], sp = I[s][PSP];
      if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) {
        continue;
      }
      if (SMRCompositeTransfer::CloudLevel(code) == 0) {
        cloud_merge_mroot[sp] = std::max(cloud_merge_mroot[sp], R[s][IPM]);
      } else if (SMRCompositeTransfer::CloudLevel(code) == 1) {
        m1[sp] = std::max(m1[sp], R[s][IPM]);
      }
    }
    // without root-level clouds, a level-1 centre piece carries 1/4 of its parent
    for (int sp=0; sp<nsp; ++sp) {
      if (cloud_merge_mroot[sp] <= 0.0) cloud_merge_mroot[sp] = 4.0*m1[sp];
    }
  }
  std::vector<char> removed(nw, 0);
  std::vector<std::vector<Real>> Rm;   // merged clouds
  std::vector<std::vector<int>> Im;
  unsigned long long nevents = 0, nin = 0, nreject = 0, npending = 0;
  unsigned long long nc_shift = 0, nc_skip = 0;
  Real dk_grad = 0.0, dk_res = 0.0;
  const Real omega = is_shearing_box ? omega0 : 0.0;
  for (int lev=1; lev<=nlev; ++lev) {
    const int L = lev - 1;
    const Real H1 = transfer.root_dx1/static_cast<Real>(1 << L);
    const Real H2 = transfer.root_dx2/static_cast<Real>(1 << L);
    const Real H = std::min(H1, H2);
    const int n1 = ng1[L], n2 = ng2[L];
    auto cell = [&](int s) {
      const int ci = SMRCompositeTransfer::WrapIndex(static_cast<int>(
          std::floor((wrap(R[s][IPX], x1min, len1) - x1min)/H1)), n1);
      const int cj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
          std::floor((wrap(R[s][IPY], x2min, len2) - x2min)/H2)), n2);
      return std::make_pair(ci, cj);
    };
    std::map<int, std::vector<int>> pieces_sp, coarse_sp;
    for (int s=0; s<nw; ++s) {
      const int code = I[s][PLEV];
      if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) {
        continue;
      }
      const int cl = SMRCompositeTransfer::CloudLevel(code);
      if (cl == lev) pieces_sp[I[s][PSP]].push_back(s);
      if (cl == L) coarse_sp[I[s][PSP]].push_back(s);
    }
    for (auto &kv : pieces_sp) {
      const int sp = kv.first;
      const Real mref = cloud_merge_mroot[std::min<int>(sp, cloud_merge_mroot.size() - 1)]/
                        static_cast<Real>(1 << (2*L));
      const Real mnode = 0.25*mref;
      std::unordered_map<int, std::vector<int>> hp, hc, hz;
      for (int s : kv.second) {auto c = cell(s); hp[c.second*n1 + c.first].push_back(s);}
      for (int s : coarse_sp[sp]) {auto c = cell(s); hc[c.second*n1 + c.first].push_back(s);}
      auto within = [&](int a, int b) {
        return std::fabs(transfer.MinImage1(R[a][IPX] - R[b][IPX])) < H1*(1.0 - 1.0e-12) &&
               std::fabs(transfer.MinImage2(R[a][IPY] - R[b][IPY])) < H2*(1.0 - 1.0e-12);
      };
      const Real ell1 = transfer.CloudHalfWidth1(lev), ell2 = transfer.CloudHalfWidth2(lev);
      // an offset within 0.1 ell of a multiple of ell is taken as that lattice offset, so that
      // tent weights of a (displaced) lattice are the exact P1 values; irregular offsets keep
      // the continuous tent weight
      auto snap = [](Real d, Real ell) {
        const Real r = std::round(d/ell);
        return (std::fabs(d - r*ell) < 0.1*ell) ? r*ell : d;
      };
      auto tent = [&](Real dx, Real dy) {
        return std::max<Real>(0.0, 1.0 - std::fabs(snap(dx, ell1))/H1)*
               std::max<Real>(0.0, 1.0 - std::fabs(snap(dy, ell2))/H2);
      };
      // a centre or level-L cloud blocks a candidate only within H - ell/2, so that the next
      // centre of a displaced lattice (about H away) is not blocked
      auto blocks = [&](int a, int b) {
        return std::fabs(transfer.MinImage1(R[a][IPX] - R[b][IPX])) < H1 - 0.5*ell1 &&
               std::fabs(transfer.MinImage2(R[a][IPY] - R[b][IPY])) < H2 - 0.5*ell2;
      };
      // members of hash h within the tent support of s
      auto near = [&](int s, std::unordered_map<int, std::vector<int>> &h,
                      std::vector<int> &out) {
        out.clear();
        auto c = cell(s);
        for (int dj=-1; dj<=1; ++dj) {
          for (int di=-1; di<=1; ++di) {
            auto it = h.find(SMRCompositeTransfer::WrapIndex(c.second + dj, n2)*n1 +
                             SMRCompositeTransfer::WrapIndex(c.first + di, n1));
            if (it == h.end()) continue;
            for (int t : it->second) if (t != s && !removed[t] && within(s, t)) out.push_back(t);
          }
        }
      };
      auto in_zone = [&](int s) {return merge_index(R[s][IPX], R[s][IPY], L) >= 0;};
      // designate centres heaviest first (P1 centre pieces carry 1/4 of their parent and sit
      // on the root lattice), then by position
      std::vector<int> zone;
      for (int s : kv.second) if (in_zone(s)) zone.push_back(s);
      std::sort(zone.begin(), zone.end(), [&](int a, int b) {
        if (R[a][IPM] != R[b][IPM]) return R[a][IPM] > R[b][IPM];
        return (R[a][IPX] < R[b][IPX]) || (R[a][IPX] == R[b][IPX] && R[a][IPY] < R[b][IPY]);
      });
      std::vector<int> centres, tmp;
      // level-L clouds committed in this pass (x, y, m), by cell, for the implicit node mass
      std::vector<std::array<Real,3>> newc;
      std::unordered_map<int, std::vector<int>> hn;
      // implicit content of the fine node at (xn, yn): the mass 1/4 m tent(d) that each level-L
      // cloud within H holds of that node in its two-scale decomposition, and its first moment
      // relative to the node (the child sits at the cloud position plus the snapped offset)
      struct Imp {Real m, mx, my;};
      auto implicit = [&](Real xn, Real yn) {
        Imp sum = {0.0, 0.0, 0.0};
        const int ci = SMRCompositeTransfer::WrapIndex(static_cast<int>(
            std::floor((wrap(xn, x1min, len1) - x1min)/H1)), n1);
        const int cj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
            std::floor((wrap(yn, x2min, len2) - x2min)/H2)), n2);
        for (int dj=-1; dj<=1; ++dj) {
          for (int di=-1; di<=1; ++di) {
            const int key = SMRCompositeTransfer::WrapIndex(cj + dj, n2)*n1 +
                            SMRCompositeTransfer::WrapIndex(ci + di, n1);
            auto add = [&](Real cx, Real cy, Real cm) {
              const Real dx = transfer.MinImage1(xn - cx), dy = transfer.MinImage2(yn - cy);
              const Real w = 0.25*cm*tent(dx, dy);
              if (w <= 0.0) return;
              sum.m += w; sum.mx += w*(snap(dx, ell1) - dx); sum.my += w*(snap(dy, ell2) - dy);
            };
            auto it = hc.find(key);
            if (it != hc.end()) for (int t : it->second) add(R[t][IPX], R[t][IPY], R[t][IPM]);
            auto jt = hn.find(key);
            if (jt != hn.end()) for (int k : jt->second) add(newc[k][0], newc[k][1], newc[k][2]);
          }
        }
        return sum;
      };
      for (int s : zone) {
        bool blocked = false;
        near(s, hc, tmp);
        for (int t : tmp) if (blocks(s, t)) {blocked = true; break;}
        if (blocked) continue;
        near(s, hz, tmp);
        for (int t : tmp) if (blocks(s, t)) {blocked = true; break;}
        if (blocked) continue;
        centres.push_back(s);
        auto c = cell(s);
        hz[c.second*n1 + c.first].push_back(s);
      }
      for (int c : centres) {
        if (removed[c] || R[c][IPM] <= 0.0) continue;
        std::vector<int> nb, nb0;
        near(c, hp, nb0);
        for (int t : nb0) {   // pieces with a positive tent weight only
          // with cloud_merge_exact_ts only pieces with the centre's ts (bitwise): the new cloud
          // takes that ts (Codex review)
          if (cloud_merge_exact_ts && R[t][IPTS] != R[c][IPTS]) continue;
          if (tent(transfer.MinImage1(R[t][IPX] - R[c][IPX]),
                   transfer.MinImage2(R[t][IPY] - R[c][IPY])) > 0.0) nb.push_back(t);
        }
        bool ready = true;
        for (int t : nb) if (!in_zone(t)) {ready = false; break;}
        if (!ready) {++npending; continue;}
        // nodes: coincident pieces (offsets within 0.1 ell of a multiple of ell); the centre's
        // own node includes the centre.  A node gives min(E, w*(E + I)), shared over its pieces
        // by mass, so that on a uniform lattice every node is used up by the centres around it.
        std::map<std::pair<long long, long long>, std::vector<int>> nodes;
        nodes[{0, 0}].push_back(c);
        for (int t : nb) {
          const Real dx = transfer.MinImage1(R[t][IPX] - R[c][IPX]);
          const Real dy = transfer.MinImage2(R[t][IPY] - R[c][IPY]);
          const long long r1 = std::llround(dx/ell1), r2 = std::llround(dy/ell2);
          if (std::fabs(dx - r1*ell1) < 0.1*ell1 && std::fabs(dy - r2*ell2) < 0.1*ell2) {
            nodes[{r1, r2}].push_back(t);
          } else {
            nodes[{(1LL << 40) + t, 0}].push_back(t);   // irregular piece: its own node
          }
        }
        // node convention (cloud_merge_nc): a share enters the centroid at the node's effective
        // position, (implicit + explicit first moment)/(implicit + explicit mass), and after the
        // commit the node's remaining pieces are shifted so that this effective position is
        // unchanged; a partly used node at the merge front then still carries its own
        // displacement, and the next centre takes the rest at the same effective position
        struct Node {std::vector<int> p; int q0; Real E, Ex, Ey, I, Ix, Iy, ex, ey, share;};
        std::vector<Node> nd;
        struct Con {int s; Real m, ox, oy; int node;};   // ox, oy: offset from the centre
        std::vector<Con> con;
        for (auto &kv2 : nodes) {
          Node n;
          n.p = kv2.second; n.q0 = n.p[0];
          const Real ox0 = transfer.MinImage1(R[n.q0][IPX] - R[c][IPX]);
          const Real oy0 = transfer.MinImage2(R[n.q0][IPY] - R[c][IPY]);
          const Real w = tent(ox0, oy0);
          n.E = n.Ex = n.Ey = 0.0;
          for (int t : n.p) {
            n.E += R[t][IPM];
            n.Ex += R[t][IPM]*transfer.MinImage1(R[t][IPX] - R[n.q0][IPX]);
            n.Ey += R[t][IPM]*transfer.MinImage2(R[t][IPY] - R[n.q0][IPY]);
          }
          if (n.E <= 0.0) continue;
          Imp im = implicit(R[n.q0][IPX], R[n.q0][IPY]);
          n.I = im.m; n.Ix = im.mx; n.Iy = im.my;
          n.share = std::min(n.E, w*(n.E + n.I));
          if (n.E - n.share < 1.0e-6*mnode) n.share = n.E;   // no dust crumbs
          if (cloud_merge_nc) {
            n.ex = (n.Ix + n.Ex)/(n.I + n.E); n.ey = (n.Iy + n.Ey)/(n.I + n.E);
          } else {
            n.ex = n.Ex/n.E; n.ey = n.Ey/n.E;
          }
          const int k = static_cast<int>(nd.size());
          for (int t : n.p) {
            const Real s = n.share*R[t][IPM]/n.E;
            if (s > 0.0) con.push_back({t, s, ox0 + n.ex, oy0 + n.ey, k});
          }
          nd.push_back(n);
        }
        // velocity tolerance: drop the worst contribution until all residuals fit
        Real M = 0.0, xc = 0.0, yc = 0.0, vm[3] = {0.0, 0.0, 0.0}, G[3][2];
        bool ok = false;
        while (!con.empty()) {
          M = 0.0; Real mx = 0.0, my = 0.0, mts = 0.0;
          for (int k=0; k<3; ++k) vm[k] = 0.0;
          for (auto &e : con) {
            const int s = e.s; const Real m = e.m;
            M += m; mx += m*e.ox; my += m*e.oy;
            vm[0] += m*R[s][IPVX]; vm[1] += m*R[s][IPVY]; vm[2] += m*R[s][IPVZ];
            mts += m*R[s][IPTS];
          }
          xc = mx/M; yc = my/M;   // relative to the centre
          for (int k=0; k<3; ++k) vm[k] /= M;
          const Real ts = mts/M;
          const Real tau = (omega > 0.0) ? std::min(ts, 1.0/omega) : ts;
          const Real tol = cloud_merge_alpha*H/tau;
          for (int k=0; k<3; ++k) {G[k][0] = 0.0; G[k][1] = 0.0;}
          if (con.size() >= 5) {
            Real cxx = 0.0, cxy = 0.0, cyy = 0.0, bx[3] = {0.0, 0.0, 0.0}, by[3] = {0.0, 0.0, 0.0};
            for (auto &e : con) {
              const int s = e.s; const Real m = e.m;
              const Real dx = e.ox - xc, dy = e.oy - yc;
              const Real v[3] = {R[s][IPVX] - vm[0], R[s][IPVY] - vm[1], R[s][IPVZ] - vm[2]};
              cxx += m*dx*dx; cxy += m*dx*dy; cyy += m*dy*dy;
              for (int k=0; k<3; ++k) {bx[k] += m*dx*v[k]; by[k] += m*dy*v[k];}
            }
            const Real det = cxx*cyy - cxy*cxy;
            if (det > 1.0e-6*M*M*H*H*H*H) {
              for (int k=0; k<3; ++k) {
                G[k][0] = (cyy*bx[k] - cxy*by[k])/det;
                G[k][1] = (cxx*by[k] - cxy*bx[k])/det;
              }
            }
          }
          int worst = -1;
          Real rmax = 0.0;
          for (int k=0; k<static_cast<int>(con.size()); ++k) {
            const int s = con[k].s;
            const Real dx = con[k].ox - xc, dy = con[k].oy - yc;
            const Real v[3] = {R[s][IPVX], R[s][IPVY], R[s][IPVZ]};
            Real r2 = 0.0;
            for (int q=0; q<3; ++q) {
              const Real d = v[q] - vm[q] - G[q][0]*dx - G[q][1]*dy;
              r2 += d*d;
            }
            if (r2 > rmax) {rmax = r2; worst = k;}
          }
          if (rmax <= tol*tol) {ok = true; break;}
          if (con[worst].s == c) break;   // the centre itself does not fit: no commit
          con.erase(con.begin() + worst);
        }
        if (!ok || M < cloud_merge_min_mass*mref) {++nreject; continue;}
        // commit: one plain level-L cloud with the mass-weighted mean of every attribute
        std::vector<Real> r(nrdata, 0.0);
        for (auto &e : con) {
          const int s = e.s; const Real m = e.m;
          for (int k=0; k<nrdata; ++k) r[k] += m*R[s][k];
          const Real dx = e.ox - xc, dy = e.oy - yc;
          const Real v[3] = {R[s][IPVX] - vm[0], R[s][IPVY] - vm[1], R[s][IPVZ] - vm[2]};
          for (int q=0; q<3; ++q) {
            const Real g = G[q][0]*dx + G[q][1]*dy;
            dk_grad += 0.5*m*g*g;
            dk_res += 0.5*m*(v[q] - g)*(v[q] - g);
          }
        }
        for (int k=0; k<nrdata; ++k) r[k] /= M;
        const Real px = wrap(R[c][IPX] + xc, x1min, len1), py = wrap(R[c][IPY] + yc, x2min, len2);
        r[IPM] = M;
        r[IPX] = px; r[IPX1] = px; r[IPY] = py; r[IPY1] = py;
        r[IPVX] = vm[0]; r[IPVY] = vm[1]; r[IPVZ] = vm[2];
        if (cloud_merge_exact_ts) r[IPTS] = R[c][IPTS];   // the common ts of the pieces
        std::vector<int> i = I[c];
        i[PLEV] = L;
        i[PGID] = owner_gid(px, py);
        Rm.push_back(r); Im.push_back(i);
        {
          const int ci = SMRCompositeTransfer::WrapIndex(static_cast<int>(
              std::floor((px - x1min)/H1)), n1);
          const int cj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
              std::floor((py - x2min)/H2)), n2);
          hn[cj*n1 + ci].push_back(static_cast<int>(newc.size()));
          newc.push_back({px, py, M});
        }
        std::vector<Real> taken(nd.size(), 0.0);
        for (auto &e : con) {
          taken[e.node] += e.m;
          R[e.s][IPM] -= e.m;
          if (e.s == c || R[e.s][IPM] <= 0.0) {R[e.s][IPM] = 0.0; removed[e.s] = 1;}
        }
        if (cloud_merge_nc) {
          for (std::size_t k=0; k<nd.size(); ++k) {
            const Node &n = nd[k];
            const Real rem = n.E - taken[k];
            if (rem <= 1.0e-6*mnode || taken[k] <= 0.0) continue;
            const Real dx = transfer.MinImage1(R[n.q0][IPX] - px);
            const Real dy = transfer.MinImage2(R[n.q0][IPY] - py);
            const Real In = 0.25*M*tent(dx, dy);
            const Real xr = (n.ex*(n.I + In + rem) - n.Ix - In*(snap(dx, ell1) - dx))/rem;
            const Real yr = (n.ey*(n.I + In + rem) - n.Iy - In*(snap(dy, ell2) - dy))/rem;
            const Real sx = xr - n.Ex/n.E, sy = yr - n.Ey/n.E;
            if (std::fabs(sx) >= 0.1*ell1 || std::fabs(sy) >= 0.1*ell2) {++nc_skip; continue;}
            for (int t : n.p) {
              if (removed[t]) continue;
              const Real nx = wrap(R[t][IPX] + sx, x1min, len1);
              const Real ny = wrap(R[t][IPY] + sy, x2min, len2);
              R[t][IPX] = nx; R[t][IPX1] = nx; R[t][IPY] = ny; R[t][IPY1] = ny;
              I[t][PGID] = owner_gid(nx, ny);
            }
            ++nc_shift;
          }
        }
        ++nevents;
        nin += con.size();
      }
    }
  }

  // totals before (original particles) and after (kept + merged), for the log
  Real tb[4] = {0.0, 0.0, 0.0, 0.0}, ta[4] = {0.0, 0.0, 0.0, 0.0};
  for (int p=0; p<npart; ++p) {
    tb[0] += pr_h(IPM,p); tb[1] += pr_h(IPM,p)*pr_h(IPVX,p);
    tb[2] += pr_h(IPM,p)*pr_h(IPVY,p); tb[3] += pr_h(IPM,p)*pr_h(IPVZ,p);
  }
  int nnew = 0;
  for (int s=0; s<nw; ++s) if (!removed[s]) ++nnew;
  nnew += static_cast<int>(Rm.size());
  Kokkos::realloc(pr, nrdata, nnew);
  Kokkos::realloc(pi, nidata, nnew);
  auto pr_new = Kokkos::create_mirror_view(pr);
  auto pi_new = Kokkos::create_mirror_view(pi);
  Real mlev[3] = {0.0, 0.0, 0.0}, mlobe = 0.0;
  int q = 0;
  auto put = [&](const std::vector<Real> &r, const std::vector<int> &i) {
    for (int c=0; c<nrdata; ++c) pr_new(c,q) = r[c];
    for (int c=0; c<nidata; ++c) pi_new(c,q) = i[c];
    ta[0] += r[IPM]; ta[1] += r[IPM]*r[IPVX]; ta[2] += r[IPM]*r[IPVY]; ta[3] += r[IPM]*r[IPVZ];
    const int code = i[PLEV];
    mlev[std::min(2, SMRCompositeTransfer::CloudLevel(code))] += r[IPM];
    if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) {
      mlobe += r[IPM];
    }
    ++q;
  };
  for (int s=0; s<nw; ++s) if (!removed[s]) put(R[s], I[s]);
  for (std::size_t k=0; k<Rm.size(); ++k) put(Rm[k], Im[k]);
  Kokkos::deep_copy(pr, pr_new);
  Kokkos::deep_copy(pi, pi_new);
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;

  cloud_coarsen_events += nevents;
  if (phst_track) phst_g_events += nevents;
  cloud_coarsen_in += nin;
  cloud_coarsen_reject += nreject;
  cloud_far_release += nrelease;
  cloud_coarsen_dk_grad += dk_grad;
  cloud_coarsen_dk_res += dk_res;
  return static_cast<int>(nevents);
}

} // namespace dust
