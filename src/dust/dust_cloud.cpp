//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud.cpp
//! \brief Refinable Lagrangian hat clouds (2-D and 3-D): their geometry, the schedule of the
//! split (P1), resampling (R) and merge (G) passes, and the cloud instant switch.
//!
//! Every particle carries a cloud level PLEV. Its hat cloud has half-width
//! root_dx/2^PLEV per direction, and its weight on an active leaf is the exact integral
//! of the cloud over that leaf (LeafTraversal::ForEachCloudWeight, block-local).  The
//! deposited fields therefore depend only on the continuous cloud density
//! sum_p m_p S_p(x - x_p), which the mesh merely integrates over its leaves.  A cloud
//! changes size only through the identity
//!   S_ell(x) = 1/4 S_{ell/2}(x + ell/2) + 1/2 S_{ell/2}(x) + 1/4 S_{ell/2}(x - ell/2),
//! applied per direction, which leaves that continuous density unchanged.  Children
//! that coincide with existing same-level particles (the shared children of adjacent
//! lattice parents) are merged by center of mass, so a quiet-start lattice refines to
//! 2^d equal children per parent.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
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
#include "particles/particle_counts.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildRefinableCloudTransfer
//! \brief The geometry of the refinable clouds (the finest-level virtual grid that every
//! MeshBlock aligns with, the split keys) and the leaf traversal of the clouds' dimension.

void DustGasDrag::BuildRefinableCloudTransfer(ParameterInput *pin) {
  Mesh *pmesh = pmy_pack->pmesh;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int nmb = pmy_pack->nmb_thispack;
  // <dust>/cloud_virtual_levels (diagnostic control, default 0): extra levels of the
  // virtual grid below the finest mesh level, so clouds can be finer than every gas leaf
  // (e.g. a uniform gas mesh with the fine clouds and sampling of a refined region; the
  // fine virtual cells map to their coarse leaf exactly as in the coarse part of a mixed
  // mesh)
  const int nlev_mesh = pmesh->max_level - pmesh->root_level;
  const int nvirt = TestKeyInt(pin, "dust", "test_cloud_virtual_levels", 0);
  int nlev = nlev_mesh + nvirt;

  // test option: levels a cloud may be coarser than the block of its center, for the
  // reach of the halo (-1: from the split rule); the per-cycle check widens it where needed
  cloud_halo_coarser = pin->GetOrAddInteger("dust", "cloud_halo_coarser", -1);
  // (the halo exchanges only the cells of each halo block that this rank's clouds can reach:
  // always, cloud_halo_strips retired)
  // every cloud_sort_interval cycles (0: never) the particle arrays are reordered
  // by block, cell in the block (row by row; 3-D clouds: (k, j, i)) and old position, after
  // the split, resampling and merge pass of the cycle (SortParticles).  By default every 64
  // cycles on the 2-D configuration where it was timed and
  // its effect on G checked (production PC2 path, device G and R, stopping
  // times fixed per species: stopping_time_mode = species_fixed); 0 is the control.  3-D clouds:
  // every 8 cycles by default, any coupling (much faster in a multi-level 3-D self-gravity run,
  // a little more with the index kept after migrations).  The order changes the R and
  // G decisions, so a sorted run is not bitwise equal to an unsorted one; conservation is
  // unaffected.  (Restart files carry the value of the run that wrote them.)
  {
    const bool prod = coupling == DustCoupling::pc2;
    // (retired-compat: an input of the old code with cloud_merge_exact_ts = false, without effect
    // with G off, had sort interval 0 by default here: kept)
    const bool exact_false = pin->DoesParameterExist("dust", "cloud_merge_exact_ts") &&
                             !pin->GetBoolean("dust", "cloud_merge_exact_ts");   // retired-compat
    const bool sortdef = prod && cloud_dim != 3 &&
                         (!cloud_resample || cloud_resample_device) && !exact_false &&
                         stopping_time_mode == DustStoppingTimeMode::species_fixed;
    cloud_sort_interval = pin->GetOrAddInteger("dust", "cloud_sort_interval",
                                               cloud_dim == 3 ? 8 : (sortdef ? 64 : 0));
  }
  {
    // diagnostic: a random permutation (<dust>/test_canonical_order = sort_random, hidden)
    cloud_sort_random = CanonicalOrder::Parse(
        TestKeyString(pin, "dust", "test_canonical_order", "off")).sort_random;
    // the cell-sorted index is on by default (every 8 cycles) where the two production cloud
    // kernels run (PC2 coupling, linear gather) and for 3-D clouds with any coupling
    // (IMEX included; keys (block, k, j, i)); 0 turns it off
    {
      const bool prod = coupling == DustCoupling::pc2;
      cloud_order_interval = pin->GetOrAddInteger("dust", "cloud_order_interval",
                                                  (prod || cloud_dim == 3) ? 8 : 0);
    }
    // (the kernels keep the index when the migration of a stage leaves fewer particles than it
    // holds, skipping its slots past the end: always, cloud_order_keep retired)
    // (hidden test key, dust.hpp; once given it stays in the restart header: -1 removes it)
    if (pin->DoesParameterExist("dust", "test_cloud_ord_budget")) {
      test_cloud_ord_budget = pin->GetInteger("dust", "test_cloud_ord_budget");
    }
  }
  if (nlev_mesh < 0 || nvirt < 0 || nlev > CloudGeometry::kCloudLevelMask) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "refinable_cloud requires a uniform mesh or 2:1 static refinement"
              << " (virtual levels included) with at most "
              << std::to_string(CloudGeometry::kCloudLevelMask) + " levels"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &t = cloud_geometry;
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
  // By default the clouds follow the mesh down to its finest level.  <dust>/test_cloud_max_level
  // (hidden, tests only; 0 keeps every cloud at root size): above it the clouds neither split
  // nor change size.  Neither key is ever added to the parameters as a default.  The old key
  // cloud_max_level was added with the run's level count as its default (it is in many restart
  // files): it is ignored.
  t.cloud_max_level = nlev;
  const bool test_cap = pin->DoesParameterExist("dust", "test_cloud_max_level");
  if (test_cap) {
    t.cloud_max_level = pin->GetInteger("dust", "test_cloud_max_level");
  }
  if (pin->DoesParameterExist("dust", "cloud_max_level") && global_variable::my_rank == 0) {
    std::cout << "# NOTE (dust): <dust>/cloud_max_level = "
              << pin->GetString("dust", "cloud_max_level") << " is ignored (an old key; the "
              << "level cap of the split is <dust>/cloud_split_max_level)" << std::endl;
  }
  // <dust>/cloud_split_max_level = L (unset or negative: no limit): the clouds split up to level
  // L; above it they keep the mass of level L while their kernel follows the mesh (the cloud
  // instant switch, CloudInstantSwitch); R caps them as level-L clouds, G stops at level L.
  cloud_split_max_level = -1;
  if (pin->DoesParameterExist("dust", "cloud_split_max_level")) {
    const int lcap = pin->GetInteger("dust", "cloud_split_max_level");
    if (lcap >= 0) {
      if (test_cap) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "<dust>/cloud_split_max_level and the test control "
                  << "<dust>/test_cloud_max_level exclude each other" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      cloud_split_max_level = lcap;
      t.cloud_max_level = std::min(nlev, lcap);
      // R: a cloud above L has the cap of level L, eta_L m0 / 2^(d L) (the table is built)
      for (int l=lcap+1; l<ResampleCapTable::kLevels; ++l) {
        cloud_resample_cap.f[l] = cloud_resample_cap.f[lcap];
      }
    }
    if (global_variable::my_rank == 0) {
      if (lcap < 0) {
        std::cout << "# DUST cloud_split_max_level = " << lcap << ": no limit (the clouds split "
                  << "down to the finest level of the mesh)" << std::endl;
      } else {
        std::cout << "# DUST cloud_split_max_level = " << lcap << ": the clouds split up to level "
                  << lcap << " (mesh levels " << nlev << "); above it every plain cloud keeps its "
                  << "mass and takes the level of the leaf that holds its center at the start of "
                  << "each cycle (cloud instant switch); R caps them as level-" << lcap
                  << " clouds, G merges levels <= " << lcap << " only"
                  << (lcap >= nlev ? " (no level above it on this mesh)" : "") << std::endl;
      }
    }
  }
  // <dust>/amr_rho_kernel_above_split_max_level = leaf | split_max_level (default leaf, added
  // to the parameters): with cloud_split_max_level = L, the kernel with which the density trigger
  // deposits the switched clouds (above L: the mass of level L, the kernel of the leaf that holds
  // them).  leaf: their own leaf kernel, the deposit of the coupling and the diagnostics
  // (DepositCloudDensity3D; the default).  split_max_level: the kernel of level L
  // (DepositTriggerDensityCapped3D, TriggerCloudCode; the former default, bitwise), which
  // smeared the clump cores and stopped the mesh about 3 levels above L.  A restart of an older
  // file (no key) takes leaf.  Without cloud_split_max_level every cloud has its own kernel
  // either way.
  {
    const std::string k = TestKeyString(pin, "dust", "test_amr_rho_kernel_above_split_max_level",
                                              "leaf");
    if (k == "leaf") {
      trigger_leaf_kernel = true;
    } else if (k == "split_max_level") {
      trigger_leaf_kernel = false;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dust>/test_amr_rho_kernel_above_split_max_level = " << k << ": must be leaf or "
                << "split_max_level" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  {
    static bool noted = false;   // (the transfer is rebuilt at every regrid)
    if (!noted) {
      noted = true;
      if (global_variable::my_rank == 0) {
        if (cloud_split_max_level >= 0) {
          std::cout << "# dust density trigger: the clouds above cloud_split_max_level = "
                    << cloud_split_max_level << " are deposited with "
                    << (trigger_leaf_kernel ? "their leaf kernel (as in the coupling)"
                                            : "the kernel of that level (the former default)")
                    << " (test_amr_rho_kernel_above_split_max_level = "
                    << (trigger_leaf_kernel ? "leaf" : "split_max_level") << ")" << std::endl;
        } else if (!trigger_leaf_kernel) {
          std::cout << "# NOTE (dust): test_amr_rho_kernel_above_split_max_level = split_max_level has "
                    << "no effect without cloud_split_max_level" << std::endl;
        }
      }
    }
  }
  // cloud_split_everywhere refines the whole lattice
  t.cloud_split_everywhere = TestKeyBool(pin, "dust", "test_cloud_split_everywhere", false);
  // split on the first overlap with finer leaves by default (the production rule; false is
  // the older rule, kept for comparison)
  t.cloud_split_overlap = TestKeyBool(pin, "dust", "test_cloud_split_overlap", true);

  // every block aligned with the finest virtual grid (the actual-leaf traversal relies on it)
  for (int m=0; m<nmb; ++m) {
    const auto &size = mbsize.h_view(m);
    int lev = mblev.h_view(m) - pmesh->root_level;
    int ratio = 1 << (nlev - lev);
    Real off1 = (size.x1min - t.x1min)/t.fine_dx1;
    Real off2 = (size.x2min - t.x2min)/t.fine_dx2;
    int fi0 = static_cast<int>(std::lround(off1));
    int fj0 = static_cast<int>(std::lround(off2));
    // the third direction in 3-D
    bool bad3 = false;
    if (cloud_dim == 3) {
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
  cloud_split_buffer = TestKeyInt(pin, "dust", "test_cloud_split_buffer", 0);
  if (cloud_split_buffer < 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/test_cloud_split_buffer must be >= 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (global_variable::my_rank == 0 && !cloud_map_announced) {
    cloud_map_announced = true;
    std::cout << "# refinable_cloud map: levels=" << nlev
              << (nvirt > 0 ? " (" + std::to_string(nvirt) + " virtual)" : "")
              << ", actual leaves, finest level "
              << t.nfine1 << "x" << t.nfine2 << ", split slack=" << cloud_split_slack
              << ", max cloud level=" << t.cloud_max_level
              << ", gather order=" << 2  // retired-compat
              << ", P1 transition clouds (mode 2)"
              << ", moving P1 split without merging"
              << (cloud_split_buffer > 0 ? ", dust split buffer " +
                  std::to_string(cloud_split_buffer) + " coarse cells" : "")
              << (t.cloud_split_everywhere ? ", split everywhere" : "")
              << (t.cloud_split_overlap ? ", split on first overlap with finer leaves" : "")
              << std::endl;
  }
  if (cloud_dim == 3) {BuildLeafTraversal3D();} else {BuildLeafTraversal();}
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::EnsureNextTag
//! \brief The first tag of created particles, once: the largest tag of all particles + 1.
//! (Not the particle count at the first creation: that repeats tags when particles were
//! removed before, e.g. by a resampling at t = 0 before the first split.)  Called at the
//! start of every cycle, so that it is set before anything removes particles; restarts carry
//! the counter.

void DustGasDrag::EnsureNextTag() {
  if (cloud_next_tag_set) return;
  // each rank hands out base + rank, then steps by the number of ranks; base = 1 + the
  // largest live tag, at least the bound of the restart file this run
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
  pm->nprtcl_total = particles::TotalCount(pm->nprtcl_eachrank, global_variable::nranks);
  mpi_time[0] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++mpi_count[0];
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitToLeaves
//! \brief The initial split of a problem generator: split passes until no cloud anywhere can
//! be split (one level per pass).  Several ranks: after every pass the pieces go
//! to the rank of the block that holds them, by position (RouteToOwnerBlocks: a root cloud
//! over a deep mesh has children several small blocks away, further than the neighbor step
//! of the particle exchange), and the loop ends when no rank split anything.  One rank: the
//! passes alone.  Returns the number of particles split.

long long DustGasDrag::SplitToLeaves() {
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
  return total;
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
  return cloud_dim == 3 ? CheckHaloReachT(leaf_traversal3d, when) : CheckHaloReachT(leaf_traversal, when);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckHaloReachT
//! \brief CheckHaloReach on either traversal (the 2-D check and CheckHaloReach3D
//! written once, the same operations).

template <class LTT>
bool DustGasDrag::CheckHaloReachT(const LTT &lt, const char *when) {
  using A = LeafApi<LTT>;
  constexpr int kD = A::kDim;
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  const int nmb = pmy_pack->nmb_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  auto allow = halo_allow_d_;
  DvceArray1D<std::int64_t> need(std::string(kD == 3 ? "dust_halo_need3d" : "dust_halo_need"),
                                 std::max(nmb, 1));
  int nover = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce(kD == 3 ? "dust_halo_reach3d" : "dust_halo_reach",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &cnt) {
      const int m = lt.Anchor(pi(PGID,p));
      Real x[3];
      A::Load(pr, p, x);
      A::Frame(lt, m, x);
      const int code = pi(PLEV,p);
      const int lev = CloudGeometry::CloudLevel(code);
      // per direction: how far the support (two cells of its level on a lobed side) reaches beyond
      // the block, in finest cells; the largest, max(c0, max(c1, c2)) as before
      Real c[3] = {0.0, 0.0, 0.0};
      for (int d=0; d<kD; ++d) {
        const Real e = (A::Lobe(code, d) ? 2.0 : 1.0)*A::HalfWidth(lt, lev, d);
        const Real b0 = A::Lo(lt, m, d), b1 = b0 + A::Nx(lt, d)*A::Dx(lt, m, d);
        const Real o = Kokkos::fmax(0.0, Kokkos::fmax(b0 - (x[d] - e), (x[d] + e) - b1));
        c[d] = Kokkos::ceil(o/A::FineDx(lt, d));
      }
      Real cm = c[kD - 1];
      for (int d=kD-2; d>=0; --d) cm = Kokkos::fmax(c[d], cm);
      const std::int64_t dd = static_cast<std::int64_t>(cm);
      if (dd > allow(m)) {
        const std::int64_t margin = static_cast<std::int64_t>(1) << (lt.nlev - lt.blev(m));
        Kokkos::atomic_max(&need(m), dd + margin);
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
  const int nhalo_before = lt.nfld - lt.nmb;
  if (kD == 3) {BuildLeafTraversal3D();} else {BuildLeafTraversal();}
  const int nhalo_after = (kD == 3) ? leaf_traversal3d.nfld - leaf_traversal3d.nmb
                                    : leaf_traversal.nfld - leaf_traversal.nmb;
  if (global_variable::my_rank == 0) {
    std::cout << (kD == 3 ? "# dust halo (3-D): " : "# dust halo: ") << tot
              << " clouds reached beyond the halo (" << when << ", cycle "
              << pmy_pack->pmesh->ncycle << "); rank 0 widened " << nwide
              << " blocks, halo blocks " << nhalo_before << " -> " << nhalo_after << std::endl;
  }
  return true;
}
template bool DustGasDrag::CheckHaloReachT<LeafTraversal>(const LeafTraversal &, const char *);
template bool DustGasDrag::CheckHaloReachT<LeafTraversal3D>(const LeafTraversal3D &, const char *);

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::FrameToBlock
//! \brief Several ranks: a new cloud's position is wrapped into the domain, but
//! the block it keeps may be across a periodic boundary.  Put such particles in the image next
//! to their block (positions and the RK position registers together), so that the particle
//! exchange sees a step of one block at most and wraps them again.

void DustGasDrag::FrameToBlock(const Real tpos) {
  if (cloud_dim == 3) {FrameToBlockT(leaf_traversal3d, tpos); return;}
  FrameToBlockT(leaf_traversal, tpos);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::FrameToBlockT
//! \brief FrameToBlock on either traversal (the 2-D kernel and FrameToBlock3D written
//! once, the same operations): several ranks, every particle into the periodic image next to its
//! block (a step across a shear-periodic x1 face moves y by n ysp, ysp the shear offset at the time
//! of the positions; physical x3 faces: no image in x3); 3-D shearing box (also on one rank), the
//! positions at tpos != pm->time beyond an x1 face get the difference of the shear offsets first.

namespace {
// FrameToBlockT's x3 step: the periodic image in x3 next to block m (3-D with periodic x3);
// nothing in 2-D.  (Overloads instead of if constexpr inside the device lambda, which nvcc
// refused.)
template <class LTT, class PR>
KOKKOS_INLINE_FUNCTION
void FrameZ(const LTT &, const int, const PR &, const int, const bool) {}
template <class PR>
KOKKOS_INLINE_FUNCTION
void FrameZ(const LeafTraversal3D &lt, const int m, const PR &pr, const int p, const bool zper) {
  const Real cz = lt.box(m,2) + 0.5*lt.nx3*lt.box(m,5);
  const Real sz = zper ? lt.lz*round((pr(IPZ,p) - cz)*lt.ilz) : 0.0;
  if (sz != 0.0) {pr(IPZ,p) -= sz; pr(IPZ1,p) -= sz;}
}
}  // namespace

template <class LTT>
void DustGasDrag::FrameToBlockT(const LTT &lt, const Real tpos) {
  using A = LeafApi<LTT>;
  constexpr bool k3 = (A::kDim == 3);
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  if (npart == 0) return;
  const bool frame = global_variable::nranks > 1;
  // the particle routing (bvals_part.cpp) moves a particle across a shear-periodic x1
  // face by ysh at pm->time; positions at tpos != pm->time (after the push) get the difference
  // first, so that the move is the one of their time (3-D)
  Real dsh = 0.0, ysp = 0.0;   // ysp: the shear offset at the time of the positions
  bool sh = false, zper = true;
  if constexpr (k3) {
    ysp = lt.ysh;
    sh = lt.shear;
    zper = lt.x3per;
    if (shear3d && shear3d_route_fix && tpos >= 0.0) {
      const Real t0 = pmy_pack->pmesh->time;
      ysp = std::fmod(qshear*omega0*lt.lx*tpos, lt.ly);
      // the difference unfolded: the folded values jump by ly when q Omega lx t passes a
      // multiple of ly, which moved the crossing particles by a whole box in y (routing abort)
      dsh = qshear*omega0*lt.lx*(tpos - t0);
    }
  }
  if (!frame && dsh == 0.0) return;
  Kokkos::parallel_for(k3 ? "dust_migrate_frame3d" : "dust_migrate_frame",
                       Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (frame) {
      const int m = lt.Anchor(pi(PGID,p));
      const Real cx = A::Lo(lt, m, 0) + 0.5*A::Nx(lt, 0)*A::Dx(lt, m, 0);
      const Real cy = A::Lo(lt, m, 1) + 0.5*A::Nx(lt, 1)*A::Dx(lt, m, 1);
      const Real n1 = round((pr(IPX,p) - cx)*lt.ilx);
      const Real sx = lt.lx*n1;
      if (sx != 0.0) {
        pr(IPX,p) -= sx; pr(IPX1,p) -= sx;
        if (sh) {pr(IPY,p) += n1*ysp; pr(IPY1,p) += n1*ysp;}
      }
      const Real sy = lt.ly*round((pr(IPY,p) - cy)*lt.ily);
      if (sy != 0.0) {pr(IPY,p) -= sy; pr(IPY1,p) -= sy;}
      FrameZ(lt, m, pr, p, zper);   // (3-D: the periodic x3 image; 2-D: nothing)
    }
    if (dsh != 0.0) {
      const Real x = pr(IPX,p);
      const Real c = (x >= lt.x1max) ? 1.0 : ((x < lt.x1min) ? -1.0 : 0.0);
      if (c != 0.0) {pr(IPY,p) += c*dsh; pr(IPY1,p) += c*dsh;}
    }
  });
}
template void DustGasDrag::FrameToBlockT<LeafTraversal>(const LeafTraversal &, const Real);
template void DustGasDrag::FrameToBlockT<LeafTraversal3D>(const LeafTraversal3D &, const Real);

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RefineClouds
//! \brief Task wrapper: split eligible clouds at the start of every cycle.

TaskStatus DustGasDrag::RefineClouds(Driver *pdrive, int stage) {
  if (lb_log_interval > 0 && pmy_pack->pmesh->ncycle % lb_log_interval == 0) LogRankClouds();
  if (smr_transfer_mode != DustSMRTransfer::refinable_cloud) {
    // point particles: the cell sort of <dust>/point_sort_interval (after the x3 outflow removal
    // of this cycle, before every gather and deposit; no send list is pending here)
    if (point_sort_interval > 0 && pmy_pack->pmesh->ncycle % point_sort_interval == 0) {
      Kokkos::fence();
      Kokkos::Timer t;
      SortParticles();
      point_sort_seconds += t.seconds();   // SortParticles ends with a fence
      ++point_sort_count;
    }
    return TaskStatus::complete;
  }
  ProfTimer prof_all(prof_time[kProfRefineClouds], prof_count[kProfRefineClouds]);
  UpdateShear3D(pmy_pack->pmesh->time);   // the positions at the start of the step
  EnsureNextTag();
  {
    ProfTimer pt(prof_time[kProfSplit], prof_count[kProfSplit]);
    SplitAndMergeClouds();
  }
  // the split budget's log line and the local R after a large split (collective; nothing with
  // both keys off)
  const bool split_r = (cloud_split_budget > 0 || cloud_split_resample) && SplitResampleWanted();
  // the split budget, several ranks: the pieces of a held event can lie several blocks from
  // their parent's block, beyond the neighbor step of the particle exchange; the new pieces
  // of other ranks' blocks go there now, by position (as in SplitToLeaves; collective)
  if (split_route_due) RouteToOwnerBlocksT(leaf_traversal3d, nullptr, split_last_before);
  // the merge clock is read once per cycle (MergeDue advances it)
  const bool merge_due = cloud_merge && MergeDue();
  if (amr_event_pending && !(cloud_resample && cloud_resample_after_regrid)) {
    amr_event_pending = false;          // no resampling after regrids: nothing to do
    amr_event_nnew = amr_event_ndel = 0;
  }
  // the event may wait cloud_event_dt after its first regrid (later regrids join it)
  bool event_now = amr_event_pending;
  if (event_now && cloud_event_dt > 0.0) {
    const double t = pmy_pack->pmesh->time;
    event_now = t >= amr_event_first_time + cloud_event_dt - 1.0e-12*std::fmax(1.0, std::fabs(t));
  }
  if (event_now) {
    // the cycle after a regrid: merge pass first, then one resampling
    ProfTimer pt(prof_time[kProfEvent], prof_count[kProfEvent]);
    HandleRegridEvent(merge_due, split_r);
  } else {
    // resampling before the merge pass, so that an audit-only pass of the same cycle writes
    // the resampled state; every cloud_resample_interval cycles, or at the planned physical
    // times
    const bool resample_due = cloud_resample && ResampleDue();
    // the local R after a large split, unless the scheduled (global) one runs anyway
    const bool split_local = split_r && !resample_due;
    // several ranks: the split pieces to the rank of their block before R and G
    // (other cycles leave them to the migration after the push; until then they keep a block
    // of this rank as their anchor, which the leaf traversal allows)
    if (resample_due || merge_due || split_local) {
      MigrateClouds();
      CheckHaloReach("split");   // R and G traverse the supports too
    }
    if (resample_due) {
      ProfTimer pt(prof_time[kProfSchedR], prof_count[kProfSchedR]);
      ResampleClouds();
    } else if (split_local) {
      ProfTimer pt(prof_time[kProfSplitR], prof_count[kProfSplitR]);
      ResampleAfterSplit();
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
    pm->nprtcl_total = particles::TotalCount(pm->nprtcl_eachrank, global_variable::nranks);
    pm->nprtcl_thisrank = n;
  }
#endif
  CheckHaloReach("refine");   // before this cycle's gathers and deposits
  ++ord_stat_cycles;
  ord_stat_npart += pmy_pack->ppart->nprtcl_thispack;
  ord_stat_nmb += pmy_pack->nmb_thispack;
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

void DustGasDrag::HandleRegridEvent(const bool merge_due, const bool split_r) {
  Mesh *pmesh = pmy_pack->pmesh;
  particles::Particles *ppar = pmy_pack->ppart;
  const int nnew = amr_event_nnew, ndel = amr_event_ndel;
  const char *kind = (nnew > 0 && ndel > 0) ? "mixed" : ((ndel > 0) ? "derefine" : "refine");
  const bool merge_ok = cloud_merge;
  const bool do_merge = merge_due || (merge_ok && ndel > 0);
  const char *why = !do_merge ? "none"
                  : ((merge_due && ndel > 0) ? "due+derefine" : (merge_due ? "due" : "derefine"));
  MigrateClouds();   // several ranks: every cloud in its block before G and R
  CheckHaloReach("regrid event");
  const int n0 = ppar->nprtcl_thispack;
  Kokkos::Timer timer;
  if (do_merge) {
    MergeClouds();
    // several ranks: a merged cloud whose block is on another rank keeps its parent's block
    // (NewCloudBlock); R bins every cloud by the leaf that holds it, so it moves first.  The
    // cycle path runs R before G and leaves this to the push migration.  (Without this, a
    // derefinement on several ranks aborts in R.)
    MigrateClouds();
    CheckHaloReach("regrid event merge");
  }
  Kokkos::fence();
  const double t_merge = timer.seconds();
  const int n1 = ppar->nprtcl_thispack;
  const bool scheduled = ResampleDue();
  // no extra resampling after a pure derefinement (unless cloud_resample_after_derefine); G
  // has run on the coarsened region, the scheduled resampling still reaches everything
  // (a large split of this cycle with cloud_split_resample: R on its blocks too)
  const bool skip_r = !scheduled && nnew == 0 && ndel > 0 && !cloud_resample_after_derefine &&
                      !split_r;
  timer.reset();
  // the blocks changed by every regrid of this event are fresh for the local R
  amr_event_max_age = (amr_event_first_ncycle >= 0)
                    ? static_cast<int>(pmesh->ncycle - amr_event_first_ncycle) : 0;
  const bool local = cloud_resample_event_local && !scheduled;
  if (local && split_r && !skip_r) {
    PrepareSplitBlocks();
    r_split_blocks = true;
    ++split_resample_calls;
  }
  const int removed = skip_r ? 0 : ResampleClouds(local);
  r_split_blocks = false;
  amr_event_max_age = 0;
  Kokkos::fence();
  const double t_resample = timer.seconds();
  const int n2 = ppar->nprtcl_thispack;
  ++amr_event_handled;
  if (scheduled) ++amr_event_merged;
  // several ranks: totals over the ranks, the slowest rank's times
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
//! \fn DustGasDrag::SplitResampleWanted
//! \brief After the split of the cycle (cloud_split_budget or cloud_split_resample on): the
//! counts of every rank (one MPI_Allgather), one log line when the budget held events back or
//! the local R runs.  The local R after the split is due when the split held events back, or
//! when some rank r that holds many clouds created n_r >= cloud_split_resample_frac * b_r
//! clouds (b_r the clouds it had before the split): many = b_r + n_r at least the mean of b + n
//! over the ranks.  Per rank, because the R's memory transient is per GPU; a rank with fewer
//! clouds than the mean never triggers it, however large its split is relative to its own
//! clouds.  Every rank evaluates the same rule on the same gathered counts, so the decision is
//! the same on all ranks (the R that follows is collective).  One rank: n >= frac * b, the
//! former rule.  Collective.

bool DustGasDrag::SplitResampleWanted() {
  const int nr = global_variable::nranks;
  // {clouds before the split, new clouds, held events, new pieces of other ranks' blocks}
  const long long mine[4] = {split_last_before, split_last_new, split_last_held,
                             split_last_offrank};
  std::vector<long long> all(mine, mine + 4);
#if MPI_PARALLEL_ENABLED
  if (nr > 1) {
    all.resize(4*static_cast<std::size_t>(nr));
    MPI_Allgather(mine, 4, MPI_LONG_LONG, all.data(), 4, MPI_LONG_LONG, MPI_COMM_WORLD);
  }
#endif
  const int ng = static_cast<int>(all.size()/4);   // the ranks gathered (1 without MPI)
  long long v[4] = {0, 0, 0, 0};   // the sums over the ranks
  for (int r=0; r<ng; ++r) {
    for (int k=0; k<4; ++k) v[k] += all[4*r + k];
  }
  // the mean clouds per rank after the split
  const double mean = static_cast<double>(v[0] + v[1])/static_cast<double>(ng);
  // rank r's split is large: a rank with at least the mean clouds (after the split) that
  // created n_r >= frac * b_r
  auto large = [&](const int r) {
    const long long b = all[4*r], n = all[4*r + 1];
    return n > 0 && static_cast<double>(b + n) >= mean &&
           static_cast<double>(n) >= cloud_split_resample_frac*static_cast<double>(b);
  };
  int nlarge = 0;
  if (cloud_split_resample) {
    for (int r=0; r<ng; ++r) nlarge += large(r) ? 1 : 0;
  }
  const bool want = cloud_split_resample && v[1] > 0 && (v[2] > 0 || nlarge > 0);
  // (the budget, several ranks) new pieces of other ranks' blocks: routed by position now
  split_route_due = cloud_split_budget > 0 && nr > 1 && v[3] > 0;
  if (global_variable::my_rank == 0 && (v[2] > 0 || want)) {
    std::cout << "# split: cycle " << pmy_pack->pmesh->ncycle << ", time "
              << pmy_pack->pmesh->time << ": " << v[1] << " clouds created from " << v[0]
              << ", " << v[2] << " split events held back by the budget";
    if (cloud_split_budget > 0 && nr > 1) {
      std::cout << ", " << v[3] << " new clouds routed to the ranks of their blocks";
    }
    std::cout << (want ? "; R after the split" : "");
    if (want && ng > 1 && nlarge > 0) {
      // the ranks whose split triggered it (the first four), and the mean per rank
      std::cout << " (" << nlarge << " of " << ng << " ranks with at least the mean " << mean
                << " clouds after the split created at least " << cloud_split_resample_frac
                << " of their clouds:";
      int shown = 0;
      for (int r=0; r<ng && shown<4; ++r) {
        if (!large(r)) continue;
        std::cout << (shown > 0 ? "," : "") << " rank " << r << " " << all[4*r + 1]
                  << " from " << all[4*r];
        ++shown;
      }
      std::cout << (nlarge > shown ? ", ...)" : ")");
    }
    std::cout << std::endl;
  }
  return want;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::PrepareSplitBlocks
//! \brief split_gid_flag: the blocks (global ids) of the split events of this cycle on any
//! rank, as bits: 1 the block of a split parent, 2 a block a piece landed in (the pieces of a
//! split lie in the block of their parent, in a neighbor, in a block shifted along x2 across a
//! shear-periodic x1 face, or after a hold by the budget several blocks away).  Collective.

void DustGasDrag::PrepareSplitBlocks() {
  const int ntot = pmy_pack->pmesh->nmb_total;
  split_gid_flag.assign(ntot, 0);
  for (std::size_t q=0; q+1<split_gid_mark.size(); q+=2) {
    const int g = split_gid_mark[q];
    if (g >= 0 && g < ntot) split_gid_flag[g] |= split_gid_mark[q + 1];
  }
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, split_gid_flag.data(), ntot, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ResampleAfterSplit
//! \brief The local R right after a large split: the clouds in the blocks of this cycle's
//! split events (the blocks of the parents and the blocks the pieces landed in; the split
//! pieces have been migrated).  One log line, with the number of those blocks that are neither
//! a parent's block nor one of its MeshBlock neighbors (nghbr: across a shear-periodic x1
//! face these are the unshifted blocks), and the number of pieces outside their own parent's
//! block and its neighbors (across a shear-periodic face, after a hold by the budget).
//! Collective.  Returns the number of clouds removed on this rank.

int DustGasDrag::ResampleAfterSplit() {
  PrepareSplitBlocks();
  int nsb = 0;
  for (const int f : split_gid_flag) nsb += (f != 0) ? 1 : 0;
  // (the log) piece blocks of this rank that are neither a parent's block nor its neighbor
  long long nfar = 0;
  {
    const int nmb = pmy_pack->nmb_thispack, gids = pmy_pack->gids;
    const int ntot = static_cast<int>(split_gid_flag.size());
    const auto &nb = pmy_pack->pmb->nghbr;
    for (int m=0; m<nmb; ++m) {
      if (gids + m >= ntot || split_gid_flag[gids + m] != 2) continue;
      bool near = false;
      for (int n=0; n<pmy_pack->pmb->nnghbr && !near; ++n) {
        const int g = nb.h_view(m,n).gid;
        if (g >= 0 && g < ntot && (split_gid_flag[g] & 1)) near = true;
      }
      if (!near) ++nfar;
    }
  }
  // mass and momentum of this rank's clouds before and after (the log line checks them)
  auto moments = [this](double s[4]) {
    auto pr = pmy_pack->ppart->prtcl_rdata;
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
    Kokkos::parallel_reduce("dust_split_r_sums",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, pmy_pack->ppart->nprtcl_thispack),
    KOKKOS_LAMBDA(const int p, double &sa, double &sb, double &sc, double &sd) {
      const double m = pr(IPM,p);
      sa += m; sb += m*pr(IPVX,p); sc += m*pr(IPVY,p); sd += m*pr(IPVZ,p);
    }, a, b, c, d);
    s[0] = a; s[1] = b; s[2] = c; s[3] = d;
  };
  double sums[9];
  moments(sums);
  const int n0 = pmy_pack->ppart->nprtcl_thispack;
  Kokkos::Timer timer;
  r_split_blocks = true;
  r_fresh_blocks = false;
  const int removed = ResampleClouds(true);
  r_split_blocks = false;
  r_fresh_blocks = true;
  Kokkos::fence();
  double t = timer.seconds();
  moments(sums + 4);
  ++split_resample_calls;
  long long nn[4] = {n0, removed, nfar, split_last_far};
  // |the mass| summed: the scale of the momentum differences
  sums[8] = 0.0;
  {
    auto pr = pmy_pack->ppart->prtcl_rdata;
    double sv = 0.0;
    Kokkos::parallel_reduce("dust_split_r_scale",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, pmy_pack->ppart->nprtcl_thispack),
    KOKKOS_LAMBDA(const int p, double &s) {
      s += Kokkos::fabs(pr(IPM,p))*(Kokkos::fabs(pr(IPVX,p)) + Kokkos::fabs(pr(IPVY,p)) +
                                    Kokkos::fabs(pr(IPVZ,p)));
    }, sv);
    sums[8] = sv;
  }
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, nn, 4, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, sums, 9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank == 0) {
    const double dm = (sums[4] - sums[0])/std::fmax(std::fabs(sums[0]), 1.0e-300);
    const double sc = std::fmax(sums[8], 1.0e-300);
    const double dp = std::fmax(std::fabs(sums[5] - sums[1]),
                                std::fmax(std::fabs(sums[6] - sums[2]), std::fabs(sums[7] - sums[3])))/sc;
    std::cout << "# R after the split " << split_resample_calls << ": cycle "
              << pmy_pack->pmesh->ncycle << ", " << nsb << " split blocks (" << nn[2]
              << " of them neither a parent's block nor its neighbor; " << nn[3]
              << " pieces outside their parent block's neighbors), "
              << "removed " << nn[1] << " of " << nn[0] << " clouds, " << t << " s; relative "
              << "change of the mass " << dm << ", of the momentum " << dp << std::endl;
  }
  return removed;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitAndMergeClouds
//! \brief The split of the cycle: the moving P1 split (SplitCloudsP1, on the device by
//! default).

int DustGasDrag::SplitAndMergeClouds() {
  const int nsplit = cloud_split_enable ? SplitCloudsP1() : 0;   // (false: test switch)
  if (cloud_split_max_level >= 0) CloudInstantSwitch();
  return nsplit;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudInstantSwitch
//! \brief <dust>/cloud_split_max_level = L: the instant switch of the cloud framework above the
//! cap (the former cloud_instant_switch, now for the clouds above L only; not the
//! point particles of amr_pic = instant_switch).  Every plain cloud of level >= L takes the level
//! max(L, level of the leaf that holds its center): its hat has the cell width of that leaf, at
//! once, when it crosses into a leaf of another level or the mesh changes under it.  It never
//! splits (the split stops at L) and keeps its mass, position, velocity and block.  The deposit
//! stays the exact leaf integrals of the switched hat, whose weights sum to one: the deposited
//! mass is exact for any width.  The drag deposits the impulse actually applied to each cloud
//! with those weights, so the total momentum of gas and dust is conserved (to round-off) for
//! any kernel; this does not need the gather to be the adjoint of the deposit, and the
//! production linear gather (cell values plus slopes) is not.  The level is fixed for the whole
//! step, so the gather and the deposit of a step use the same kernel.  Lobed clouds (level-L
//! transition clouds of the split lattice) keep their level; so does a cloud whose leaf is not
//! found.  Unlike a split, a switch changes the deposit: the gravity field of the last solve is
//! not reused.  Runs after the split of every cycle and of every pass of the initial split.
//! Returns the clouds whose level changed (this rank).
//!
//! The root-cloud mass of every species (cloud_merge_mroot: R's mass caps, G's quantum) is
//! inferred from the plain level-0 / level-1 clouds (EnsureRootCloudMass), assuming the masses
//! of the split lattice (a level-1 cloud carries 1/4 of a root cloud in 2-D, 1/8 in 3-D).  The
//! switch breaks that (L = 0: a root cloud of mass M at level 1 or 2; any L: a level-L cloud at
//! a finer level), so the value is recorded here once, collectively, before the first switch
//! changes any level, and the device G, the host G and R all read this one value (a cached
//! value would be wrong after root clouds switch to level 1, and a 2-D start that switches every
//! root cloud to level 2 would leave zero).  A restart reads it from the file.

namespace {
// CloudInstantSwitch on either traversal: every plain cloud of level >= L takes the level
// max(L, level of the leaf that holds its center); lobed clouds and clouds whose leaf is not found keep
// theirs.  Returns the clouds whose level changed (this rank).
template <class LTT>
int InstantSwitchLevels(const LTT &lt, const int L, particles::Particles *ppar) {
  using T = CloudGeometry;
  using A = LeafApi<LTT>;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  int nchange = 0;
  Kokkos::parallel_reduce(A::kDim == 3 ? "dust_cloud_instant_switch3d" : "dust_cloud_instant_switch2d",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &nc) {
    const int code = pi(PLEV,p);
    if ((code & ~T::kCloudLevelMask) != 0 || code < L) return;   // lobed, or below the cap
    int b = -1;
    Real x[3];
    A::Load(pr, p, x);
    A::Owner(lt, x, lt.Anchor(pi(PGID,p)), &b);
    if (b < 0) return;
    const int lev = (lt.blev(b) > L) ? lt.blev(b) : L;
    if (lev != code) {pi(PLEV,p) = lev; ++nc;}
  }, Kokkos::Sum<int>(nchange));
  return nchange;
}
}  // namespace

int DustGasDrag::CloudInstantSwitch() {
  const int L = cloud_split_max_level;
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  // the root-cloud mass before any level changes (collective; once, then a no-op)
  if (L >= 0 && L < cloud_geometry.cloud_nlev && cloud_merge_mroot.empty()) {
    EnsureRootCloudMass();
    if (global_variable::my_rank == 0) {
      std::cout << "# DUST cloud_split_max_level: root cloud mass per species (before the first "
                << "instant switch, cycle " << pmy_pack->pmesh->ncycle << ") ="
                << std::setprecision(17);
      for (const Real m : cloud_merge_mroot) std::cout << " " << m;
      std::cout << std::setprecision(6) << std::endl;
    }
  }
  // (no leaf above L on this mesh: every plain cloud of level >= L is at L already)
  if (L < 0 || L >= cloud_geometry.cloud_nlev || npart == 0) return 0;
  int nchange = 0;
  // (one kernel on the traversal interface, LeafApi; the 2-D and 3-D kernels before were the same
  // rule)
  nchange = cloud_dim == 3 ? InstantSwitchLevels(leaf_traversal3d, L, ppar)
                     : InstantSwitchLevels(leaf_traversal, L, ppar);
  if (nchange > 0) {
    cloud_instant_switch_count += static_cast<unsigned long long>(nchange);
    ++gravity_mass_epoch;   // the deposit changed (no reuse of the gravity field)
  }
  return nchange;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositTriggerDensityCapped3D
//! \brief <dust>/cloud_split_max_level = L with amr_rho_kernel_above_split_max_level =
//! split_max_level (the former default): the dust density the 3-D density trigger
//! reads, as
//! DepositCloudDensity3D(f, -1, true) but with every plain cloud above L deposited with the
//! kernel of its mass level L (TriggerCloudCode): the switched (finer) kernels sample the density
//! with fewer clouds than cells, and their grainy deposit would drive the refinement (a positive
//! feedback with close level thresholds).  Only the trigger reads it; the dynamics and the
//! diagnostics keep the switched kernel.

void DustGasDrag::DepositTriggerDensityCapped3D(DvceArray5D<Real> &f) {
  if (f.extent_int(0) != qdep.extent_int(0) || f.extent_int(1) != 4 ||
      f.extent_int(2) != qdep.extent_int(2) || f.extent_int(3) != qdep.extent_int(3) ||
      f.extent_int(4) != qdep.extent_int(4)) {
    Kokkos::realloc(f, qdep.extent(0), 4, qdep.extent(2), qdep.extent(3), qdep.extent(4));
  }
  Kokkos::deep_copy(f, 0.0);
  UpdateShear3D(pmy_pack->pmesh->time);
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const auto lt = leaf_traversal3d;
  const int lcap = cloud_split_max_level;
  auto ff = f;
  // (written once, CloudFieldDepositT)
  CloudFieldDepositT<LeafTraversal3D, false>("dust_cloud3d_trigger_density_cap", lt, pr, pi,
                                             npart, -1, false, 1, ff, CloudCodeCapped{lcap});
  // several ranks: the owners hold every rank's clouds
  HaloReverseAdd(f, 0, 1);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitCloudsP1
//! \brief Moving-particle P1 split, called by
//! SplitAndMergeClouds.  A plain level-l cloud whose support lies on finer leaves is
//! split, but along each direction an edge child is separated only if the lattice
//! neighbor on that side (the virtual level-l cloud one parent spacing H away) would
//! be split too; otherwise the edge child stays attached to the center child as a
//! transition lobe.  A lobed cloud separates a lobe once that neighbor becomes
//! eligible.  Along a direction, the transition band of eligible parents with an
//! ineligible neighbor is exactly H wide, so every lattice row holds one transition
//! cloud whatever its phase and the particle basis stays P1 while the lattice drifts
//! through the split surface.  Nothing is ever merged: every particle keeps its own
//! attributes, so independent streams and stopping times are preserved.  Kokkos on every
//! backend (SplitCloudsP1DeviceT, dust_cloud_device.cpp); the host split was retired.  // retired-compat

int DustGasDrag::SplitCloudsP1() {
  // (mesh_gradient, several ranks) the supports within the halo before the moments' deposit,
  // including the coarser support of the fit's retry (CheckSplitVelocityReach)
  if (cloud_split_velocity != 0) CheckSplitVelocityReach();
  return cloud_dim == 3 ? SplitCloudsP1DeviceT(leaf_traversal3d) : SplitCloudsP1DeviceT(leaf_traversal);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MergeClouds
//! \brief Coarsening of refinable clouds (<dust>/cloud_merge).  A
//! cell of the level-L grid is a merge cell when no leaf finer than L lies within
//! cloud_merge_buffer level-L cells of it (periodic, Chebyshev distance).  In merge cells
//!  (1) lobed level-(L+1) clouds release all lobes into plain pieces (deposit, mass and
//!      momentum unchanged; the transition-2 point gather becomes the cloud integral);
//!  (2) plain level-(L+1) pieces are merged by the quantum pass (MergeQuantumT,
//!      dust_cloud_device.cpp).
//! Mass and momentum are conserved exactly.  Kokkos on every backend; the host merge and the
//! 2-D tent pass were retired.  // retired-compat
int DustGasDrag::MergeClouds() {
  ++gravity_mass_epoch;   // G changes the deposited mass (no reuse of the gravity field)
  return cloud_dim == 3 ? MergeQuantumT(leaf_traversal3d) : MergeQuantumT(leaf_traversal);
}

} // namespace dust
