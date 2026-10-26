//========================================================================================
// AthenaK: composite AMR self-gravity. BSD-3-Clause; see LICENSE.
//========================================================================================
#include "gravity/self_gravity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <sys/resource.h>

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "gravity/composite.hpp"
#include "gravity/geometric_mg.hpp"
#include "gravity/slab_boundary.hpp"
#include "gravity/vacuum_boundary.hpp"
#include "hydro/hydro.hpp"
#include "dust/dust.hpp"
#include "parameter_input.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"

namespace gravity {
namespace {
[[noreturn]] void Fail(const std::string &message) {
  std::cerr << "### FATAL ERROR in self-gravity: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
Real Sum(Real value) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return value;
}
double ProfileTime(bool enabled) {
  if (!enabled) return 0;
  Kokkos::fence();
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
long PeakRssKiB() {
  struct rusage usage;
  return getrusage(RUSAGE_SELF, &usage) == 0 ? usage.ru_maxrss : -1;
}
std::uint64_t CsrBytes(const HostCsr &a) {
  return (a.rowptr.capacity()+a.colidx.capacity())*sizeof(int)
      +a.values.capacity()*sizeof(Real);
}
struct ExactSourceDifference {
  DvceArray1D<Real> current, previous;
  KOKKOS_INLINE_FUNCTION void operator()(const int q, int &count) const {
    count += current(q) != previous(q);
  }
};
}  // namespace

SelfGravity::SelfGravity(MeshBlockPack *pack, ParameterInput *pin) : pack_(pack) {
  auto *pm = pack->pmesh;
  if (pack->phydro == nullptr || pack->pmhd != nullptr ||
      (pack->ppart != nullptr && pack->pdust == nullptr) ||
      pack->prad != nullptr || pack->pcoord->is_special_relativistic ||
      pack->pcoord->is_general_relativistic || pack->pcoord->is_dynamical_relativistic ||
      pm->nmb_packs_thisrank != 1 || pm->one_d) {
    Fail("requires Newtonian 2D/3D hydro, one pack per rank; standalone particles "
         "without the dust module are not supported");
  }
  coupled_dust_ = pack->pdust != nullptr;
  if (coupled_dust_ && !pack->pdust->GravityTransferSupported())
    Fail("dust gravity currently requires supported 3D refinable clouds");
  if (pin->GetOrAddString("gravity", "solver", "multigrid") != "multigrid") {
    Fail("solver must be multigrid");
  }
  const auto evolution = pin->GetString("time", "evolution");
  if (evolution != "dynamic" && evolution != "static") {
    Fail("requires dynamic or static evolution");
  }
  if (evolution == "dynamic") {
    const auto integrator = pin->GetOrAddString("time", "integrator", "rk2");
    const bool cloud_imex = coupled_dust_ && integrator == "imex2+" &&
        pack->pdust->Clouds3D() && pack->pdust->coupling == DustCoupling::imex &&
        pack->pdust->drag_solver == DustDragSolver::local;
    if (integrator != "rk1" && integrator != "rk2" && integrator != "rk3" && !cloud_imex) {
      Fail("gravity half-kicks support rk1/rk2/rk3, or imex2+ with supported "
           "3D refinable-cloud local dust drag");
    }
  }
  auto *orbital = pack->phydro->porb_u;
  if (orbital != nullptr) {
    if (pack->phydro->psbox_u == nullptr ||
        (!pm->three_d && orbital->shearing_box_r_phi)) {
      Fail("gravity orbital splitting requires a supported 3D shearing box");
    }
    // The existing 2D r-z box has Coriolis terms but no orbital remap.
    // Coupled dust retains the existing complete gas+dust transport between the
    // common gravity kicks. Never half-remap only gas while dust still drifts
    // its whole background shear inside the original transport update.
    orbital_split_ = pm->three_d && evolution == "dynamic" && !coupled_dust_;
  }
  four_pi_G = pin->GetReal("gravity", "four_pi_G");
  tolerance_ = pin->GetOrAddReal("gravity", "rtol", 1.0e-10);
  gravity_cfl_ = pin->GetOrAddReal("gravity", "cfl", 0.25);
  max_iterations_ = pin->GetOrAddInteger("gravity", "max_iterations", 150);
  restart_ = pin->GetOrAddInteger("gravity", "gmres_restart", 24);
  report_each_solve_ = pin->GetOrAddBoolean("gravity", "report_each_solve", false);
  profile_ = pin->GetOrAddBoolean("gravity", "profile", false);
  phase_threads_ = pin->GetOrAddInteger("gravity", "phase_threads", 0);
  if (phase_threads_ < 0) Fail("gravity/phase_threads must be nonnegative");
#if MPI_PARALLEL_ENABLED
  int mpi_thread_level = MPI_THREAD_SINGLE, mpi_main_thread = 0;
  MPI_Query_thread(&mpi_thread_level);
  MPI_Is_thread_main(&mpi_main_thread);
  if (!mpi_main_thread) Fail("SelfGravity must be initialized on the MPI main thread");
  if (mpi_thread_level < MPI_THREAD_FUNNELED) {
    if (phase_threads_ > 1)
      Fail("gravity/phase_threads>1 requires MPI_THREAD_FUNNELED or stronger support");
    phase_serial_only_ = true;
    if (phase_threads_ == 0 && global_variable::my_rank == 0)
      std::cout << "# SG_PHASE_THREADS auto_capped=1 mpi_thread_level=" << mpi_thread_level << '\n';
  }
#endif
  setup_ = pin->GetOrAddString("gravity", "setup", "distributed");
  if (setup_ != "reference" && setup_ != "distributed") {
    Fail("gravity/setup must be reference or distributed");
  }
  if (setup_ == "distributed") {
    const int cells[3] = {pm->mb_indcs.nx1,pm->mb_indcs.nx2,pm->mb_indcs.nx3};
    for (int d = 0; d < (pm->three_d ? 3 : 2); ++d) {
      if (cells[d] <= 0 || (cells[d] & (cells[d]-1)) != 0) {
        Fail("distributed gravity requires power-of-two active MeshBlock dimensions; "
             "use a supported block shape or explicitly select setup=reference");
      }
    }
  }
  vertical_bc_ = pin->GetOrAddString("gravity", "vertical_bc", "periodic");
  isolated_method_ = pin->GetOrAddString("gravity", "isolated_method",
      setup_ == "reference" ? "direct" : "vacuum");
  if (isolated_method_ != "direct" && isolated_method_ != "vacuum") {
    Fail("gravity/isolated_method must be direct or vacuum");
  }
  if (vertical_bc_ == "isolated" && isolated_method_ == "vacuum"
      && setup_ != "distributed") Fail("vacuum isolation requires distributed setup");
  if (!std::isfinite(four_pi_G) || four_pi_G <= 0 || !std::isfinite(tolerance_) ||
      tolerance_ <= 0 || tolerance_ >= 1 || !std::isfinite(gravity_cfl_) ||
      gravity_cfl_ <= 0 || gravity_cfl_ > 0.5 || max_iterations_ < 1 || restart_ < 2) {
    Fail("invalid gravitational constant, tolerance, iteration count, or timestep factor");
  }
  if (vertical_bc_ != "periodic" && vertical_bc_ != "isolated") {
    Fail("gravity/vertical_bc must be periodic or isolated");
  }
  if (vertical_bc_ == "isolated") {
    if (!pm->three_d) Fail("vertical isolated gravity requires 3D (x3 is vertical)");
    slab_modes_x_ = pin->GetOrAddInteger("gravity", "slab_modes_x", 0);
    slab_modes_y_ = pin->GetOrAddInteger("gravity", "slab_modes_y", 0);
    if (slab_modes_x_ < 0 || slab_modes_y_ < 0) Fail("negative slab mode cutoff");
  }
  for (int d = 0; d < (pm->three_d ? 3 : 2); ++d) {
    if (d == 2 && vertical_bc_ == "isolated") continue;
    const auto left = pm->mesh_bcs[2*d], right = pm->mesh_bcs[2*d+1];
    if (!pm->three_d && (left == BoundaryFlag::shear_periodic ||
                        right == BoundaryFlag::shear_periodic)) {
      Fail("shear-periodic gravity requires 3D hydro; 2D r-z boxes use periodic x1");
    }
    const bool periodic = left == BoundaryFlag::periodic && right == BoundaryFlag::periodic;
    const bool shear = d == 0 && pack->phydro->psbox_u != nullptr &&
        left == BoundaryFlag::shear_periodic && right == BoundaryFlag::shear_periodic;
    if (!periodic && !shear) Fail("unsupported gravity boundary in direction " +
                                std::to_string(d+1));
  }
  dtnew = std::numeric_limits<Real>::max();
}

SelfGravity::~SelfGravity() {
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(16) << "# SELF_GRAVITY_SUMMARY solves=" << solve_count
              << " cache_hits=" << cache_hit_count
              << " linear_solves=" << solve_count-cache_hit_count
              << " linear_attempts=" << solve_count-cache_hit_count+preconditioner_refresh_count
              << " phase_updates=" << phase_update_count
              << " preconditioner_refreshes=" << preconditioner_refresh_count
              << " rebuilds=" << rebuild_count << " iterations=" << last_solve.iterations
              << " relative_residual=" << last_solve.relative_residual
              << " source_mean=" << last_source_mean << " time=" << last_time << '\n';
  }
}

void SelfGravity::Invalidate() {
  topology_ = -1;
  phase_valid_ = false;
  field_cache_valid_ = false;
}

void SelfGravity::SetHydroBoundaryTime(Real time) {
  if (!std::isfinite(time)) Fail("nonfinite hydro boundary time");
  hydro_boundary_time_ = time;
  hydro_boundary_time_valid_ = true;
}

void SelfGravity::OrbitalStep(Real dt) {
  if (!UsesOrbitalSplit() || !std::isfinite(dt) || dt <= 0) {
    Fail("invalid gravity orbital halfstep");
  }
  auto *hydro = pack_->phydro;
  auto *orbital = hydro->porb_u;
  auto *pm = pack_->pmesh;
  const Real full_dt = pm->dt;
  // The public remap reads pm->dt, but never pm->time. Restrict this temporary
  // change to the completed remap and restore it before residual RK stages.
  pm->dt = dt;
  auto finish = [](auto operation) {
    TaskStatus status;
    do { status = operation(); } while (status == TaskStatus::incomplete);
    if (status != TaskStatus::complete) Fail("orbital halfstep task failed");
  };
  finish([&]() { return orbital->InitRecv(); });
  finish([&]() { return orbital->PackAndSendCC(hydro->u0); });
  finish([&]() { return orbital->RecvAndUnpackCC(hydro->u0, hydro->recon_method); });
  // Both halfsteps retain the native conservative correction at mixed x2 faces.
  finish([&]() { return orbital->SendFluxCC(); });
  finish([&]() { return orbital->RecvFluxCC(hydro->u0); });
  finish([&]() { return orbital->ClearSend(); });
  finish([&]() { return orbital->ClearRecv(); });
  Kokkos::fence();
  pm->dt = full_dt;
}

void SelfGravity::Rebuild(Real shear_offset) {
  if (setup_ == "distributed") {
    RebuildDistributed(shear_offset);
    return;
  }
  const double t0 = ProfileTime(profile_);
  double t_geometry = 0, t_assembly = 0, t_operators = 0, t_slab = 0;
  auto *pm = pack_->pmesh;
  const auto ind = pm->mb_indcs;
  const int nc = ind.nx1*ind.nx2*ind.nx3;
  if (pm->nmb_total > std::numeric_limits<int>::max()/nc) {
    Fail("composite cell indexing exceeds the current 32-bit CSR capacity");
  }
  const int n = pm->nmb_total*nc;
  CompositeDomain domain;
  domain.ndim = pm->three_d ? 3 : 2;
  domain.lo = {pm->mesh_size.x1min, pm->mesh_size.x2min, pm->mesh_size.x3min};
  domain.hi = {pm->mesh_size.x1max, pm->mesh_size.x2max, pm->mesh_size.x3max};
  domain.root_counts = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  domain.shear_offset = shear_offset;
  domain.isolated_z = vertical_bc_ == "isolated";
  domain.shear_phase_updates = pm->mesh_bcs[0] == BoundaryFlag::shear_periodic;
  const int mbnc[3] = {ind.nx1, ind.nx2, ind.nx3};
  std::vector<CompositeCell> cells(n);
  for (int m = 0; m < pm->nmb_total; ++m) {
    const auto loc = pm->lloc_eachmb[m];
    const int ell = loc.level-pm->root_level;
    const int block[3] = {loc.lx1, loc.lx2, loc.lx3};
    for (int k = 0; k < ind.nx3; ++k) {
      for (int j = 0; j < ind.nx2; ++j) {
        for (int i = 0; i < ind.nx1; ++i) {
          const int q = m*nc + (k*ind.nx2+j)*ind.nx1+i;
          const int ijk[3] = {i,j,k};
          for (int d = 0; d < 3; ++d) {
            if (d >= domain.ndim) {
              cells[q].width[d] = domain.hi[d]-domain.lo[d];
              cells[q].center[d] = 0.5*(domain.lo[d]+domain.hi[d]);
            } else {
              const Real h = std::ldexp((domain.hi[d]-domain.lo[d]) /
                                       domain.root_counts[d], -ell);
              cells[q].width[d] = h;
              cells[q].center[d] = domain.lo[d]+(block[d]*mbnc[d]+ijk[d]+0.5)*h;
            }
          }
        }
      }
    }
  }
  partition_.resize(global_variable::nranks+1);
  for (int r = 0; r < global_variable::nranks; ++r) {
    partition_[r] = pm->gids_eachrank[r]*nc;
  }
  partition_.back() = n;
  const std::vector<int> legacy_partition(partition_.begin(),partition_.end());
  const int begin = partition_[global_variable::my_rank];
  const int local = pack_->nmb_thispack*nc;
  // A shear phase changes the matrix, but not row ownership or the physical
  // interpretation of a local potential entry. The CURRENT operator's true
  // residual decides whether this previous field is already adequate.
  const bool keep_guess = phase_valid_ && topology_ == pm->GetAMRLoadBalanceUpdateSeq()
                          && solution_.extent_int(0) == local;
  for (int m = 0; m < pack_->nmb_thispack; ++m) {
    if (pack_->pmb->mb_gid.h_view(m)*nc != begin+m*nc) {
      Fail("local block layout does not match the global contiguous cell partition");
    }
  }
  try {
    t_geometry = ProfileTime(profile_);
    system_ = std::make_unique<CompositeSystem>(
        BuildCompositeSystem(cells, domain, legacy_partition));
    t_assembly = ProfileTime(profile_);
    mg_ = std::make_unique<SparseMultigrid>(system_->hierarchy);
    for (int d = 0; d < domain.ndim; ++d) {
      force_[d] = std::make_unique<DistributedOperator>(system_->acceleration[d],
                                                       legacy_partition, legacy_partition);
    }
    t_operators = ProfileTime(profile_);
    if (domain.isolated_z) {
      const int nb = system_->boundary_samples.size();
      std::vector<int> bp(global_variable::nranks+1);
      for (int r = 0; r <= global_variable::nranks; ++r) {
        bp[r] = static_cast<long long>(nb)*r/global_variable::nranks;
      }
      slab_ = std::make_unique<SlabBoundary>(cells,domain,system_->boundary_samples,
                                            legacy_partition,bp,slab_modes_x_,slab_modes_y_);
      boundary_values_ = DvceArray1D<Real>("sg_slab_boundary",
          bp[global_variable::my_rank+1]-bp[global_variable::my_rank]);
      boundary_rhs_op_ = std::make_unique<DistributedOperator>(system_->boundary_rhs,
                                                               legacy_partition,bp);
      for (int d = 0; d < 3; ++d) {
        boundary_force_[d] = std::make_unique<DistributedOperator>(
            system_->boundary_acceleration[d],legacy_partition,bp);
      }
    }
    t_slab = ProfileTime(profile_);
  } catch (const std::exception &e) {
    Fail(e.what());
  }
  std::vector<Real> local_volume(local);
  std::vector<int> physical_rows(local);
  for (int q = 0; q < local; ++q) {
    local_volume[q] = system_->volume[begin+q];
    physical_rows[q] = q;
  }
  AllocateFields(local_volume,physical_rows,keep_guess);
  topology_ = pm->GetAMRLoadBalanceUpdateSeq();
  phase_ = shear_offset;
  phase_valid_ = true;
  ++rebuild_count;
  if (profile_) {
    StorageStats storage = mg_->Storage();
    std::uint64_t host_csr = 0, host_nnz = 0;
    for (const auto &a : system_->hierarchy.matrices) {
      host_csr += CsrBytes(a); host_nnz += a.values.size();
    }
    for (const auto &p : system_->hierarchy.prolongation) host_csr += CsrBytes(p);
    for (int d = 0; d < domain.ndim; ++d) {
      host_csr += CsrBytes(system_->acceleration[d]);
      const auto s = force_[d]->Storage();
      storage.csr_bytes += s.csr_bytes; storage.halo_bytes += s.halo_bytes;
    }
    if (slab_) {
      host_csr += CsrBytes(system_->boundary_rhs);
      const auto s = boundary_rhs_op_->Storage();
      storage.csr_bytes += s.csr_bytes; storage.halo_bytes += s.halo_bytes;
      for (int d = 0; d < 3; ++d) {
        host_csr += CsrBytes(system_->boundary_acceleration[d]);
        const auto f = boundary_force_[d]->Storage();
        storage.csr_bytes += f.csr_bytes; storage.halo_bytes += f.halo_bytes;
      }
    }
    std::cout << std::setprecision(10) << "# SG_PROFILE_SETUP rank="
              << global_variable::my_rank << " rebuild=" << rebuild_count
              << " cells=" << n << " local_cells=" << local
              << " geometry_s=" << t_geometry-t0
              << " assembly_s=" << t_assembly-t_geometry
              << " operators_s=" << t_operators-t_assembly
              << " slab_setup_s=" << t_slab-t_operators
              << " vectors_s=" << ProfileTime(true)-t_slab
              << " global_a_nnz=" << host_nnz << " local_a_nnz=" << storage.a_nnz
              << " local_p_nnz=" << storage.p_nnz << " local_r_nnz=" << storage.r_nnz
              << " input_host_csr_bytes=" << host_csr
              << " retained_host_csr_bytes=" << storage.terminal_host_csr_bytes
              << " local_csr_bytes=" << storage.csr_bytes
              << " local_halo_bytes=" << storage.halo_bytes
              << " mg_vector_bytes=" << storage.vector_bytes
              << " field_bytes=" << (rhs_.size()+solution_.size()+work_.size()
                  +volume_.size()+cached_source_.size()+phi.size()+acceleration.size()+extra_density_.size()
                  +boundary_values_.size())*sizeof(Real)
              << " peak_rss_kib=" << PeakRssKiB() << '\n';
  }
  // Every consumer above owns its local device data; no iteration path reads the
  // replicated host assembly. Release it before another phase builds a new one.
  system_.reset();
}

void SelfGravity::RebuildDistributed(Real shear_offset) {
  const double t0 = ProfileTime(profile_);
  auto *pm = pack_->pmesh;
  const auto ind = pm->mb_indcs;
  const int rank = global_variable::my_rank, nranks = global_variable::nranks;
  const GlobalIndex nc64 = GlobalIndex(ind.nx1)*ind.nx2*ind.nx3;
  if (nc64*pack_->nmb_thispack > std::numeric_limits<int>::max()) {
    Fail("owned hydro cells exceed the local device index capacity");
  }
  const int nc = static_cast<int>(nc64), physical_n = pack_->nmb_thispack*nc;
  CompositeDomain domain;
  domain.ndim = pm->three_d ? 3 : 2;
  domain.lo = {pm->mesh_size.x1min,pm->mesh_size.x2min,pm->mesh_size.x3min};
  domain.hi = {pm->mesh_size.x1max,pm->mesh_size.x2max,pm->mesh_size.x3max};
  domain.root_counts = {pm->mesh_indcs.nx1,pm->mesh_indcs.nx2,pm->mesh_indcs.nx3};
  domain.shear_offset = shear_offset;
  domain.isolated_z = vertical_bc_ == "isolated";
  domain.shear_phase_updates = pm->mesh_bcs[0] == BoundaryFlag::shear_periodic;
  std::vector<GravityPatch> patches(pm->nmb_total);
  const int counts[3] = {ind.nx1,ind.nx2,ind.nx3};
  for (int m = 0; m < pm->nmb_total; ++m) {
    const auto loc = pm->lloc_eachmb[m];
    const int ell = loc.level-pm->root_level;
    const int block[3] = {loc.lx1,loc.lx2,loc.lx3};
    auto &p = patches[m];
    p.owner = std::upper_bound(pm->gids_eachrank,pm->gids_eachrank+nranks,m)
              -pm->gids_eachrank-1;
    p.gid = m;
    p.cells = {ind.nx1,ind.nx2,ind.nx3};
    for (int d = 0; d < 3; ++d) {
      if (d >= domain.ndim) {
        p.lo[d] = domain.lo[d]; p.hi[d] = domain.hi[d];
      } else {
        const Real h = std::ldexp((domain.hi[d]-domain.lo[d])/domain.root_counts[d],-ell);
        p.lo[d] = domain.lo[d]+block[d]*counts[d]*h;
        p.hi[d] = p.lo[d]+counts[d]*h;
      }
    }
  }
  std::unique_ptr<VacuumMesh> vacuum_mesh;
  solver_neumann_z_ = false;
  if (domain.isolated_z && isolated_method_ == "vacuum") {
    try {
      vacuum_mesh = std::make_unique<VacuumMesh>(BuildVacuumMesh(patches,domain,nranks));
      patches = vacuum_mesh->patches;
      domain = vacuum_mesh->domain;
      solver_neumann_z_ = vacuum_mesh->neumann_z;
    } catch (const std::exception &e) { Fail(e.what()); }
  }
  const double t_geometry = ProfileTime(profile_);
  try {
    auto setup = BuildGeometricHierarchy(std::move(patches),domain,rank,nranks,solver_neumann_z_);
    const double t_assembly = ProfileTime(profile_);
    const auto old_partition = partition_;
    partition_ = setup.fine.row_partition;
    const int local = setup.hierarchy.matrices[0].LocalRows();
    const bool keep_guess = phase_valid_ && topology_ == pm->GetAMRLoadBalanceUpdateSeq()
        && old_partition == partition_ && solution_.extent_int(0) == local;
    std::vector<int> physical_rows(physical_n,-1);
    for (const auto &p : setup.patches[0]) {
      if (p.owner != rank || p.gid < 0 || p.gid >= pm->nmb_total) continue;
      const int m = p.gid-pm->gids_eachrank[rank];
      if (m < 0 || m >= pack_->nmb_thispack || pack_->pmb->mb_gid.h_view(m) != p.gid
          || p.cells != std::array<int,3>{ind.nx1,ind.nx2,ind.nx3}) {
        Fail("distributed gravity physical patch does not match the hydro pack");
      }
      const GlobalIndex start = p.row_begin-partition_[rank];
      if (start < 0 || start+nc > local) Fail("invalid physical-to-solver row mapping");
      for (int q = 0; q < nc; ++q) physical_rows[m*nc+q] = static_cast<int>(start)+q;
    }
    if (std::find(physical_rows.begin(),physical_rows.end(),-1) != physical_rows.end()) {
      Fail("distributed gravity omitted an owned hydro source cell");
    }
    mg_ = std::make_unique<SparseMultigrid>(setup.hierarchy);
    for (int d = 0; d < domain.ndim; ++d) {
      force_[d] = std::make_unique<DistributedOperator>(setup.fine.acceleration[d],
                                                       partition_,partition_);
    }
    const double t_operators = ProfileTime(profile_);
    if (domain.isolated_z) {
      if (vacuum_mesh) {
        vacuum_ = std::make_unique<VacuumBoundary>(*vacuum_mesh,setup.fine,rank);
        slab_.reset();
      } else {
        std::array<Real,2> minimum{{std::numeric_limits<Real>::max(),
                                    std::numeric_limits<Real>::max()}};
        for (const auto &p : setup.patches[0]) for (int d = 0; d < 2; ++d) {
          minimum[d] = std::min(minimum[d],(p.hi[d]-p.lo[d])/p.cells[d]);
        }
        slab_ = std::make_unique<SlabBoundary>(setup.fine.cells,domain,
            setup.fine.boundary_samples,minimum,slab_modes_x_,slab_modes_y_);
        vacuum_.reset();
      }
      const auto &bp = setup.fine.sample_partition;
      boundary_values_ = DvceArray1D<Real>("sg_slab_boundary",bp[rank+1]-bp[rank]);
      boundary_rhs_op_ = std::make_unique<DistributedOperator>(setup.fine.boundary_rhs,
                                                               partition_,bp);
      for (int d = 0; d < 3; ++d) boundary_force_[d] =
          std::make_unique<DistributedOperator>(setup.fine.boundary_acceleration[d],
                                                partition_,bp);
    }
    const double t_slab = ProfileTime(profile_);
    AllocateFields(setup.fine.volume,physical_rows,keep_guess);
    solver_domain_ = domain;
    solver_patches_ = setup.patches[0];
    geometry_ = std::move(setup.fine_geometry);
    geometry_->SetPhaseThreads(phase_serial_only_ ? 1 : phase_threads_);
    coarse_geometry_ = std::move(setup.coarse_geometry);
    prolongation_phase_rows_ = std::move(setup.prolongation_phase_rows);
    for (auto &coarse : coarse_geometry_) {
      const int threads = coarse->PhaseRows().size() < 256 || phase_serial_only_ ? 1 : phase_threads_;
      coarse->SetPhaseThreads(threads);
    }
    if (pm->mesh_bcs[0] == BoundaryFlag::shear_periodic) {
      const auto &rows = geometry_->PhaseRows();
      std::vector<std::vector<int>> a_rows{rows};
      for (const auto &coarse : coarse_geometry_) a_rows.push_back(coarse->PhaseRows());
      mg_->ConfigureHierarchyPhaseRows(a_rows,prolongation_phase_rows_);
      for (int d = 0; d < domain.ndim; ++d) force_[d]->ConfigurePhaseRows(rows);
      if (slab_ || vacuum_) {
        boundary_rhs_op_->ConfigurePhaseRows(rows);
        for (int d = 0; d < 3; ++d) boundary_force_[d]->ConfigurePhaseRows(rows);
      }
    }
    hierarchy_fresh_ = true;
    topology_ = pm->GetAMRLoadBalanceUpdateSeq();
    phase_ = shear_offset;
    phase_valid_ = true;
    ++rebuild_count;
    system_.reset();
    if (profile_) {
      auto storage = mg_->Storage();
      std::uint64_t global_nnz = storage.a_nnz;
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE,&global_nnz,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
#endif
      for (int d = 0; d < domain.ndim; ++d) {
        const auto s = force_[d]->Storage();
        storage.csr_bytes += s.csr_bytes; storage.halo_bytes += s.halo_bytes;
      }
      if (slab_ || vacuum_) {
        const auto s = boundary_rhs_op_->Storage();
        storage.csr_bytes += s.csr_bytes; storage.halo_bytes += s.halo_bytes;
        for (int d = 0; d < 3; ++d) {
          const auto f = boundary_force_[d]->Storage();
          storage.csr_bytes += f.csr_bytes; storage.halo_bytes += f.halo_bytes;
        }
      }
      std::cout << std::setprecision(10) << "# SG_PROFILE_SETUP rank=" << rank
                << " setup=distributed rebuild=" << rebuild_count
                << " cells=" << partition_.back() << " local_cells=" << local
                << " physical_cells=" << physical_n << " patches=" << pm->nmb_total
                << " physical_cells_global=" << GlobalIndex(pm->nmb_total)*nc
                << " boundary_backend=" << (vacuum_ ? "vacuum" : slab_ ? "direct" : "periodic")
                << " vacuum_cells=" << (vacuum_mesh ? vacuum_mesh->vacuum_unknowns : 0)
                << " vacuum_padding=" << (vacuum_mesh ? vacuum_mesh->padding : 0)
                << " unknown_overhead=" << (vacuum_mesh ? vacuum_mesh->unknown_overhead : 1)
                << " vacuum_metadata_bytes=" << (vacuum_mesh ? vacuum_mesh->metadata_bytes : 0)
                << " vacuum_boundary_bytes=" << (vacuum_ ? vacuum_->DeviceMetadataBytes() : 0)
                << " local_quadratic_interface_fallbacks="
                << std::accumulate(setup.quadratic_interface_fallbacks.begin(),
                    setup.quadratic_interface_fallbacks.end(),std::size_t(0))
                << " fine_quadratic_interface_fallbacks=" << setup.fine.quadratic_interface_fallbacks
                << " geometry_s=" << t_geometry-t0 << " assembly_s=" << t_assembly-t_geometry
                << " operators_s=" << t_operators-t_assembly
                << " slab_setup_s=" << t_slab-t_operators
                << " boundary_setup_s=" << t_slab-t_operators
                << " vectors_s=" << ProfileTime(true)-t_slab
                << " global_a_nnz=" << global_nnz << " local_a_nnz=" << storage.a_nnz
                << " local_p_nnz=" << storage.p_nnz << " local_r_nnz=" << storage.r_nnz
                << " retained_global_host_csr_bytes=0 local_csr_bytes=" << storage.csr_bytes
                << " terminal_owned_host_csr_bytes=" << storage.terminal_host_csr_bytes
                << " local_halo_bytes=" << storage.halo_bytes
                << " host_pattern_bytes=" << storage.host_pattern_bytes
                << " mg_vector_bytes=" << storage.vector_bytes
                << " peak_rss_kib=" << PeakRssKiB() << '\n';
    }
  } catch (const std::exception &e) {
    Fail(e.what());
  }
}

void SelfGravity::UpdatePhase(Real shear_offset) {
  const double begin = ProfileTime(profile_);
  try {
    auto update = geometry_->BuildPhaseUpdate(shear_offset);
    const double fine_assembled = ProfileTime(profile_);
    std::vector<CsrRowUpdate> a;
    a.reserve(coarse_geometry_.size()+1); a.push_back(std::move(update.matrix));
    for (auto &coarse : coarse_geometry_)
      a.push_back(std::move(coarse->BuildPhaseUpdate(shear_offset,false).matrix));
    const double coarse_assembled = ProfileTime(profile_);
    std::vector<CsrRowUpdate> p;
    p.reserve(coarse_geometry_.size());
    auto current_domain = solver_domain_;
    current_domain.shear_offset = shear_offset;
    for (size_t l = 0; l < coarse_geometry_.size(); ++l) {
      const auto &fine = l == 0 ? *geometry_ : *coarse_geometry_[l-1];
      p.push_back(BuildProlongationPhaseUpdate(fine,*coarse_geometry_[l],current_domain,
          global_variable::my_rank,prolongation_phase_rows_[l]));
    }
    const double assembled = ProfileTime(profile_);
    mg_->UpdateHierarchyPhaseRows(a,p);
    for (int d = 0; d < solver_domain_.ndim; ++d) {
      force_[d]->UpdatePhaseRows(update.acceleration[d]);
    }
    if (slab_ || vacuum_) {
      boundary_rhs_op_->UpdatePhaseRows(update.boundary_rhs);
      for (int d = 0; d < 3; ++d) boundary_force_[d]->UpdatePhaseRows(update.boundary_acceleration[d]);
      if (slab_) slab_->SetShearOffset(shear_offset);
    }
    solver_domain_.shear_offset = shear_offset;
    phase_ = shear_offset;
    field_cache_valid_ = false;
    ++phase_update_count;
    if (profile_) {
      const auto counters = mg_->FineUpdates();
      const auto hierarchy = mg_->HierarchyPhaseUpdates();
      std::cout << std::setprecision(10) << "# SG_PROFILE_PHASE rank="
                << global_variable::my_rank << " phase=" << shear_offset
                << " phase_updates=" << phase_update_count
                << " assembly_workers=" << geometry_->PhaseThreads()
                << " phase_threads_config=" << phase_threads_
                << " assembly_s=" << assembled-begin
                << " fine_assembly_s=" << fine_assembled-begin
                << " coarse_assembly_s=" << coarse_assembled-fine_assembled
                << " prolongation_assembly_s=" << assembled-coarse_assembled
                << " operators_s=" << ProfileTime(true)-assembled
                << " value_updates=" << counters.value_updates
                << " structure_rebuilds=" << counters.structure_rebuilds
                << " fine_halo_rebuilds=" << counters.halo_rebuilds
                << " fine_halo_reuses=" << counters.halo_reuses
                << " hierarchy_halo_rebuilds=" << hierarchy.halo_rebuilds
                << " hierarchy_halo_reuses=" << hierarchy.halo_reuses
                << " hierarchy_phase_map_payload_bytes=" << hierarchy.host_map_peak_bytes
                << '\n';
    }
  } catch (const std::exception &e) { Fail(e.what()); }
}

void SelfGravity::RefreshHierarchy() {
  auto refreshed = BuildGeometricHierarchy(solver_patches_,solver_domain_,
      global_variable::my_rank,global_variable::nranks,solver_neumann_z_,false);
  if (refreshed.hierarchy.row_partitions[0] != partition_
      || refreshed.hierarchy.matrices[0].LocalRows() != solution_.extent_int(0)) {
    Fail("hierarchy-only refresh changed fine ownership; full rebuild required");
  }
  auto solver = std::make_unique<SparseMultigrid>(refreshed.hierarchy);
  if (pack_->pmesh->mesh_bcs[0] == BoundaryFlag::shear_periodic) {
    std::vector<std::vector<int>> a_rows{geometry_->PhaseRows()};
    for (const auto &coarse : coarse_geometry_) a_rows.push_back(coarse->PhaseRows());
    solver->ConfigureHierarchyPhaseRows(a_rows,prolongation_phase_rows_);
  }
  mg_ = std::move(solver);
  ++preconditioner_refresh_count;
  hierarchy_fresh_ = true;
  // The current force/boundary operators, physical mapping and geometry remain.
}

void SelfGravity::AllocateFields(const std::vector<Real> &local_volume,
                                 const std::vector<int> &physical_rows, bool keep_guess) {
  const int local = local_volume.size();
  rhs_ = DvceArray1D<Real>("sg_rhs",local);
  if (!keep_guess || solution_.extent_int(0) != local) {
    solution_ = DvceArray1D<Real>("sg_phi",local);
    Kokkos::deep_copy(solution_,0.0);
  }
  work_ = DvceArray1D<Real>("sg_work",local);
  volume_ = DvceArray1D<Real>("sg_volume",local);
  physical_source_ = DvceArray1D<Real>("sg_physical_source",physical_rows.size());
  physical_rows_ = DvceArray1D<int>("sg_physical_rows",physical_rows.size());
  auto vol = Kokkos::create_mirror_view(volume_);
  auto rows = Kokkos::create_mirror_view(physical_rows_);
  for (int q = 0; q < local; ++q) vol(q) = local_volume[q];
  for (size_t q = 0; q < physical_rows.size(); ++q) rows(q) = physical_rows[q];
  Kokkos::deep_copy(volume_,vol);
  Kokkos::deep_copy(physical_rows_,rows);
  field_cache_valid_ = false;
  const auto ind = pack_->pmesh->mb_indcs;
  const int ni = ind.nx1+2*ind.ng, nj = ind.nx2+2*ind.ng;
  const int nk = pack_->pmesh->three_d ? ind.nx3+2*ind.ng : 1;
  phi = DvceArray5D<Real>("sg_phi_pack",pack_->nmb_thispack,1,nk,nj,ni);
  acceleration = DvceArray5D<Real>("sg_acc_pack",pack_->nmb_thispack,3,nk,nj,ni);
  extra_density_ = DvceArray5D<Real>("sg_extra_density",pack_->nmb_thispack,1,nk,nj,ni);
}

bool SelfGravity::CachedSourceUnchanged(const DvceArray1D<Real> &physical_source) {
  const int n = physical_source.extent_int(0);
  int changed = 1;
  if (field_cache_valid_ && cached_source_.extent_int(0) == n
      && phase_valid_ && phase_ == cached_phase_ && topology_ == cached_topology_
      && four_pi_G == cached_four_pi_G_
      && mg_->FineUpdates().updates == cached_operator_updates_) {
    auto previous = cached_source_;
    int different = 0;
    Kokkos::parallel_reduce("sg_exact_source_comparison",Kokkos::RangePolicy<DevExeSpace>(0,n),
                           ExactSourceDifference{physical_source,previous},different);
    changed = different != 0;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&changed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  return changed == 0;
}

void SelfGravity::RememberSource(const DvceArray1D<Real> &physical_source) {
  if (cached_source_.extent(0) != physical_source.extent(0)) {
    cached_source_ = DvceArray1D<Real>("sg_cached_source",physical_source.extent(0));
  }
  Kokkos::deep_copy(cached_source_,physical_source);
}

void SelfGravity::UpdateDustField(Real time) {
  if (coupled_dust_)
    pack_->pdust->GatherGravityAcceleration(time,acceleration,particle_acceleration_);
}

void SelfGravity::Solve(Real time) {
  const double t0 = ProfileTime(profile_);
  auto *pm = pack_->pmesh;
  Real shift = 0;
  if (pm->mesh_bcs[0] == BoundaryFlag::shear_periodic) {
    const auto *ps = pack_->phydro->psbox_u;
    const Real ly = pm->mesh_size.x2max-pm->mesh_size.x2min;
    shift = std::fmod(-ps->qshear*ps->omega0*
                     (pm->mesh_size.x1max-pm->mesh_size.x1min)*time, ly);
  }
  if (!phase_valid_ || topology_ != pm->GetAMRLoadBalanceUpdateSeq()) {
    Rebuild(shift);
  } else if (shift != phase_) {
    if (setup_ == "distributed") UpdatePhase(shift);
    else Rebuild(shift);
  }
  const double t_setup = ProfileTime(profile_);
  const auto ind = pm->mb_indcs;
  const int n = pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  const int solver_n = solution_.extent_int(0);
  auto rows = physical_rows_;
  auto physical_source = physical_source_;
  auto u = pack_->phydro->u0;
  auto extra = extra_density_;
  Kokkos::deep_copy(extra,0.0);
  if (extra_source_) extra_source_(time,extra);
  if (coupled_dust_) {
    auto *dust=pack_->pdust;
    if (!dust->stopping_times_initialized ||
        dust->stopping_time_mode==DustStoppingTimeMode::dynamic)
      dust->RefreshStoppingTimeMaximum();
    dust->DepositGravityDensity(time,dust_density_);
  }
  if (extra.extent(0) < static_cast<std::size_t>(pack_->nmb_thispack)
      || extra.extent(1) < 1 || extra.extent(2) <= static_cast<std::size_t>(ind.ke)
      || extra.extent(3) <= static_cast<std::size_t>(ind.je)
      || extra.extent(4) <= static_cast<std::size_t>(ind.ie)) {
    Fail("extra density callback View does not cover the active hydro indices");
  }
  auto volume = volume_;
  auto b = rhs_;
  const bool dust_source=coupled_dust_;
  auto dust_density=dust_density_;
  Real local_mass = 0;
  Kokkos::parallel_reduce("sg_source_mass",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q, Real &sum) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    physical_source(q) = u(t,IDN,k,j,i)+extra(t,0,k,j,i)
                        +(dust_source ? dust_density(t,0,k,j,i) : Real(0));
    sum += physical_source(q)*volume(rows(q));
  },local_mass);
  Real total_volume = (pm->mesh_size.x1max-pm->mesh_size.x1min)*
                      (pm->mesh_size.x2max-pm->mesh_size.x2min);
  if (pm->three_d) total_volume *= pm->mesh_size.x3max-pm->mesh_size.x3min;
  const Real global_physical_mass = Sum(local_mass);
  last_source_mean = global_physical_mass/total_volume;
  const Real mean = vertical_bc_ == "isolated" ? 0 : last_source_mean;
  const Real fourpig = four_pi_G;
  auto work = work_;
  const bool unchanged_source = CachedSourceUnchanged(physical_source_);
  double t_source = ProfileTime(profile_);
  if (unchanged_source) {
    // Every physical input to the field is identical, including the source
    // callback's CURRENT output. Preserve the actual accepted residual as cached.
    ++cache_hit_count;
    ++solve_count;
    last_time = time;
    last_solve_was_cached = true;
    // A resampling/split can preserve the complete leaf source while changing
    // particle locations, shape codes or count. Reuse phi, not stale gathered g.
    if (coupled_dust_) {UpdateDustField(time); UpdateTimeStep();}
    if (profile_) {
      std::cout << std::setprecision(10) << "# SG_PROFILE_SOLVE rank="
                << global_variable::my_rank << " solve=" << solve_count
                << " time=" << time << " phase=" << shift << " rebuilds=" << rebuild_count
                << " setup_s=" << t_setup-t0 << " source_s=" << t_source-t_setup
                << " boundary_s=0 solve_s=0 force_s=0 iterations=0 matvecs=0 vcycles=0"
                << " residual=" << last_solve.relative_residual
                << " cached=1 residual_cached=1 cache_hits=" << cache_hit_count
                << " physical_mass=" << global_physical_mass
                << " physical_source_cells=" << n << " solver_cells=" << solver_n
                << " boundary_backend=" << (vacuum_ ? "vacuum" : slab_ ? "direct" : "periodic")
                << " outer_flux=" << (vacuum_ ? vacuum_->Flux() : 0)
                << " peak_rss_kib=" << PeakRssKiB() << '\n';
    }
    if (report_each_solve_ && global_variable::my_rank == 0) {
      std::cout << std::setprecision(16) << "# SELF_GRAVITY time=" << time
                << " cells=" << partition_.back() << " levels=" << mg_->Levels()
                << " iterations=" << last_solve.iterations << " residual="
                << last_solve.relative_residual << " cached=1 residual_cached=1\n";
    }
    return;
  }
  field_cache_valid_ = false;
  last_solve_was_cached = false;
  RememberSource(physical_source_);
  Kokkos::deep_copy(b,0.0);
  Kokkos::deep_copy(work_,0.0);
  Kokkos::parallel_for("sg_deposit_physical_source",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q) {
    const int row = rows(q);
    work(row) = physical_source(q);
    b(row) = -fourpig*(physical_source(q)-mean)*volume(row);
  });
  t_source = ProfileTime(profile_);
  if (slab_ || vacuum_) {
    if (vacuum_) vacuum_->GPUFill(four_pi_G,global_physical_mass,boundary_values_);
    else slab_->Compute(work_,four_pi_G,boundary_values_);
    boundary_rhs_op_->Apply(boundary_values_,work_);
    Kokkos::parallel_for("sg_add_slab_rhs",Kokkos::RangePolicy<DevExeSpace>(0,solver_n),
    KOKKOS_LAMBDA(const int q) { b(q) += work(q); });
  }
  const double t_boundary = ProfileTime(profile_);
  const bool cached_preconditioner = setup_ == "distributed" && !hierarchy_fresh_;
  const int attempt_cap = cached_preconditioner ? std::min(24,max_iterations_) : max_iterations_;
  SolveStats cached_attempt;
  double cached_attempt_s = 0, hierarchy_refresh_s = 0, fresh_attempt_s = 0;
  bool refreshed = false;
  try {
    const double attempt_begin = ProfileTime(true);
    last_solve = mg_->Solve(rhs_,solution_,tolerance_,attempt_cap,restart_);
    const double attempt_seconds = ProfileTime(true)-attempt_begin;
    if (cached_preconditioner) {
      cached_attempt = last_solve;
      cached_attempt_s = attempt_seconds;
    } else { fresh_attempt_s = attempt_seconds; }
    if (cached_preconditioner && (!last_solve.converged
                                 || !std::isfinite(last_solve.relative_residual))) {
      const std::string reason = last_solve.preconditioner_breakdown ? "true_residual_gap" :
          !std::isfinite(last_solve.relative_residual)
          ? "nonfinite_residual" : (last_solve.iterations >= attempt_cap
          ? "iteration_cap" : "cached_breakdown");
      auto candidate = solution_;
      const Real largest = std::numeric_limits<Real>::max();
      int nonfinite = 0;
      Kokkos::parallel_reduce("sg_check_partial_phi",Kokkos::RangePolicy<DevExeSpace>(0,solver_n),
      KOKKOS_LAMBDA(const int q, int &count) { count += !(fabs(candidate(q)) <= largest); },nonfinite);
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE,&nonfinite,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
      if (nonfinite) Kokkos::deep_copy(solution_,0.0);
      const double refresh_begin = ProfileTime(true);
      RefreshHierarchy();
      hierarchy_refresh_s = ProfileTime(true)-refresh_begin;
      const double retry_begin = ProfileTime(true);
      last_solve = mg_->Solve(rhs_,solution_,tolerance_,max_iterations_,restart_);
      fresh_attempt_s = ProfileTime(true)-retry_begin;
      refreshed = true;
      if (global_variable::my_rank == 0) {
        std::cout << std::setprecision(16) << "# SELF_GRAVITY_PRECONDITIONER_REFRESH reason="
                  << reason << " count=" << preconditioner_refresh_count
                  << " cached_iterations=" << cached_attempt.iterations
                  << " cached_relative_residual=" << cached_attempt.relative_residual
                  << " cached_attempt_s=" << cached_attempt_s
                  << " hierarchy_refresh_s=" << hierarchy_refresh_s
                  << " fresh_attempt_s=" << fresh_attempt_s
                  << " fresh_iterations=" << last_solve.iterations
                  << " fresh_relative_residual=" << last_solve.relative_residual
                  << " zeroed_nonfinite_phi=" << (nonfinite != 0) << '\n';
      }
    }
  } catch (const std::exception &e) {
    Fail(e.what());
  }
  if (!last_solve.converged || !std::isfinite(last_solve.relative_residual)) {
    if (last_solve.preconditioner_breakdown)
      Fail("multigrid preconditioner breakdown: true residual greatly exceeds Arnoldi "
           "estimate; refusing a stale gravity field");
    Fail("multigrid did not reach the requested residual; refusing a stale gravity field");
  }
  hierarchy_fresh_ = false;
  const double t_solve = ProfileTime(profile_);
  auto sol = solution_;
  Real gauge = 0;
  if (vacuum_) gauge = vacuum_->GaugeShift(solution_);
  else if (vertical_bc_ != "isolated") {
    Real local_phi = 0;
    Kokkos::parallel_reduce("sg_volume_gauge",Kokkos::RangePolicy<DevExeSpace>(0,solver_n),
    KOKKOS_LAMBDA(const int q, Real &sum) { sum += sol(q)*volume(q); },local_phi);
    gauge = Sum(local_phi)/total_volume;
  }
  Kokkos::parallel_for("sg_apply_gauge",Kokkos::RangePolicy<DevExeSpace>(0,solver_n),
  KOKKOS_LAMBDA(const int q) { sol(q) -= gauge; });
  auto potential = phi;
  Kokkos::parallel_for("sg_store_phi",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    potential(t,0,k,j,i) = sol(rows(q));
  });
  auto acc = acceleration;
  Kokkos::deep_copy(acc,0.0);
  for (int d = 0; d < (pm->three_d ? 3 : 2); ++d) {
    force_[d]->Apply(solution_,work_);
    Kokkos::parallel_for("sg_store_acceleration",Kokkos::RangePolicy<DevExeSpace>(0,n),
    KOKKOS_LAMBDA(const int q) {
      int t = q;
      const int i = t%ind.nx1+ind.is; t /= ind.nx1;
      const int j = t%ind.nx2+ind.js; t /= ind.nx2;
      const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
      acc(t,d,k,j,i) = work(rows(q));
    });
    if (slab_ || vacuum_) {
      boundary_force_[d]->Apply(boundary_values_,work_);
      Kokkos::parallel_for("sg_add_slab_force",Kokkos::RangePolicy<DevExeSpace>(0,n),
      KOKKOS_LAMBDA(const int q) {
        int t = q;
        const int i = t%ind.nx1+ind.is; t /= ind.nx1;
        const int j = t%ind.nx2+ind.js; t /= ind.nx2;
        const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
        acc(t,d,k,j,i) += work(rows(q));
      });
    }
  }
  ++solve_count;
  last_time = time;
  UpdateDustField(time);
  UpdateTimeStep();
  cached_phase_ = phase_;
  cached_topology_ = topology_;
  cached_four_pi_G_ = four_pi_G;
  cached_operator_updates_ = mg_->FineUpdates().updates;
  field_cache_valid_ = true;
  if (profile_) {
    const auto storage = mg_->Storage();
    const auto updates = mg_->FineUpdates();
    std::cout << std::setprecision(10) << "# SG_PROFILE_SOLVE rank="
              << global_variable::my_rank << " solve=" << solve_count
              << " time=" << time << " phase=" << shift << " rebuilds=" << rebuild_count
              << " setup_s=" << t_setup-t0 << " source_s=" << t_source-t_setup
              << " boundary_s=" << t_boundary-t_source << " solve_s=" << t_solve-t_boundary
              << " force_s=" << ProfileTime(true)-t_solve
              << " iterations=" << last_solve.iterations
              << " matvecs=" << last_solve.matrix_applications
              << " vcycles=" << last_solve.vcycles
              << " residual=" << last_solve.relative_residual
              << " cached=0 residual_cached=0 cache_hits=" << cache_hit_count
              << " physical_mass=" << global_physical_mass
              << " physical_source_cells=" << n << " solver_cells=" << solver_n
              << " boundary_backend=" << (vacuum_ ? "vacuum" : slab_ ? "direct" : "periodic")
              << " outer_flux=" << (vacuum_ ? vacuum_->Flux() : 0)
              << " gauge_shift=" << gauge
              << " removed_rhs_mean=" << last_solve.removed_rhs_mean
              << " mg_vector_bytes=" << storage.vector_bytes
              << " current_a_nnz=" << storage.current_a_nnz
              << " host_pattern_bytes=" << storage.host_pattern_bytes
              << " phase_rows=" << storage.phase_rows << " phase_nnz=" << storage.phase_nnz
              << " phase_map_bytes=" << storage.phase_map_bytes
              << " fine_updates=" << updates.updates
              << " fine_value_updates=" << updates.value_updates
              << " fine_structure_rebuilds=" << updates.structure_rebuilds
              << " cached_preconditioner=" << cached_preconditioner
              << " cached_attempt_iterations=" << cached_attempt.iterations
              << " cached_attempt_matvecs=" << cached_attempt.matrix_applications
              << " total_iterations=" << last_solve.iterations
                   +(refreshed ? cached_attempt.iterations : 0)
              << " cached_attempt_s=" << cached_attempt_s
              << " hierarchy_refreshed=" << refreshed
              << " hierarchy_refresh_s=" << hierarchy_refresh_s
              << " fresh_attempt_s=" << fresh_attempt_s
              << " preconditioner_refreshes=" << preconditioner_refresh_count
              << " peak_rss_kib=" << PeakRssKiB() << '\n';
  }
  if (report_each_solve_ && global_variable::my_rank == 0) {
    std::cout << std::setprecision(16) << "# SELF_GRAVITY time=" << time
              << " cells=" << partition_.back() << " levels=" << mg_->Levels()
              << " iterations=" << last_solve.iterations << " residual="
              << last_solve.relative_residual << '\n';
  }
}

void SelfGravity::KickHydro(Real dt) {
  const auto ind = pack_->pmesh->mb_indcs;
  const int n = pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  auto u = pack_->phydro->u0;
  auto acc = acceleration;
  const bool energy = pack_->phydro->peos->eos_data.is_ideal;
  Kokkos::parallel_for("sg_hydro_kick",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    const Real rho = u(t,IDN,k,j,i);
    Real de = 0;
    for (int d = 0; d < 3; ++d) {
      const Real dv = dt*acc(t,d,k,j,i);
      de += dv*(u(t,IM1+d,k,j,i)+0.5*rho*dv);
      u(t,IM1+d,k,j,i) += rho*dv;
    }
    if (energy) u(t,IEN,k,j,i) += de;
  });
}

void SelfGravity::KickMatter(Real dt) {
  if (!std::isfinite(dt)) Fail("nonfinite matter kick");
  KickHydro(dt);
  if (coupled_dust_) pack_->pdust->KickGravity(dt,particle_acceleration_);
}

void SelfGravity::UpdateTimeStep() {
  const auto ind = pack_->pmesh->mb_indcs;
  const int ndim = pack_->pmesh->three_d ? 3 : 2;
  const int nc = ind.nx1*ind.nx2*ind.nx3;
  const int n = pack_->nmb_thispack*nc;
  // Match the accepted field's actual gas+callback source. A callback may
  // replace its output View handle, leaving extra_density_ as the old buffer.
  auto source = physical_source_;
  auto acc = acceleration;
  auto size = pack_->pmb->mb_size.d_view;
  const Real fourpig = four_pi_G;
  Real limit = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("sg_timestep",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q, Real &minimum) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    const Real density = fabs(source(q));
    if (density > 0) minimum = fmin(minimum,1.0/sqrt(fourpig*density));
    const Real dx[3] = {size(t).dx1,size(t).dx2,size(t).dx3};
    for (int d = 0; d < ndim; ++d) {
      const Real a = fabs(acc(t,d,k,j,i));
      if (a > 0) minimum = fmin(minimum,sqrt(dx[d]/a));
    }
  },Kokkos::Min<Real>(limit));
  if (coupled_dust_)
    limit=std::min(limit,pack_->pdust->GravityAccelerationTimeStep(particle_acceleration_));
  dtnew = gravity_cfl_*limit;
  if (coupled_dust_ && pack_->pdust->coupling==DustCoupling::hybrid &&
      pack_->pdust->hybrid_force_mode==HybridForceMode::pc2) {
    // Forced PC2 is a resolved-drag method: its gas feedback predictor is not
    // unconditionally stable for dt/t_s or high dust loading. A conservative
    // bound on the local relaxation rate is (1+rho_d/rho_g)/min(t_s).
    // IMEX has no such explicit restriction. Keep legacy gravity-off behavior.
    const Real stopping=pack_->pdust->taus_min;
    auto dust_density=dust_density_;
    auto gas=pack_->phydro->u0;
    Real drag_limit=std::numeric_limits<Real>::max();
    Kokkos::parallel_reduce("gravity_resolved_pc2_timestep",
        Kokkos::RangePolicy<DevExeSpace>(0,n),KOKKOS_LAMBDA(int q,Real &minimum) {
      int t=q;
      const int i=t%ind.nx1+ind.is;t/=ind.nx1;
      const int j=t%ind.nx2+ind.js;t/=ind.nx2;
      const int k=t%ind.nx3+ind.ks;t/=ind.nx3;
      const Real eps=dust_density(t,0,k,j,i)/gas(t,IDN,k,j,i);
      minimum=fmin(minimum,stopping/(1+eps));
    },Kokkos::Min<Real>(drag_limit));
    dtnew=std::min(dtnew,Real(.25)*drag_limit);
  }
}
}  // namespace gravity
