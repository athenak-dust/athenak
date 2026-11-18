//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_GEOMETRIC_MG_HPP_
#define GRAVITY_GEOMETRIC_MG_HPP_

#include <array>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>
#include "gravity/mesh_geometry.hpp"

namespace gravity {

struct TransferDiagnostics {
  Real constant_defect = 0;
  Real interior_linear_defect = 0;
  Real restriction_volume_defect = 0;
  GlobalIndex interior_linear_rows = 0;
};

struct ProlongationCacheStats {
  std::uint64_t queries = 0, hits = 0, misses = 0;
  std::uint64_t bypasses = 0, oversized_bypasses = 0, entries = 0;
  // Sum of worker arena peaks, excluding allocator overhead and uncached query
  // working storage. Each lazily allocated arena is capped at16MiB.
  std::uint64_t worker_peak_bytes = 0;
  double virtual_average_seconds = 0;  // measured only for an explicit stats request
};

struct GeometricSystem {
  DistributedHierarchy hierarchy;
  // Retains patch metadata and the initialized stable phase-row list only.
  // The adapter can refresh the current fine operator without another volume scan.
  std::unique_ptr<MeshGeometry> fine_geometry;
  // Retained geometry metadata only; no host leaf matrices or vectors survive
  // upload. Entry l-1 is hierarchy level l.
  std::vector<std::unique_ptr<MeshGeometry>> coarse_geometry;
  // Independently traced through P's recursive VirtualAverage x-image visits.
  std::vector<std::vector<int>> prolongation_phase_rows;
  // matrix has been moved to hierarchy.matrices[0]. Other finest fields remain.
  LocalCompositeSystem fine;
  // Patch metadata only, one vector per level. No global cell or CSR arrays.
  std::vector<std::vector<GravityPatch>> patches;
  // Local maxima/counts; callers may reduce them for collective diagnostics.
  std::vector<TransferDiagnostics> transfer_checks;
  std::vector<ProlongationCacheStats> initial_prolongation_cache;
  std::vector<std::size_t> quadratic_interface_fallbacks;
  // split (GpuSplit): the systems (fine, hierarchy.matrices, quadratic_interface_fallbacks)
  // are placeholders until FinishSplitHierarchy
  bool split_pending = false, split_fine_forces = true;
};

// Host wall-clock seconds of the hierarchy construction phases since the last reset;
// for <gravity>/profile output only.
struct HierarchyBuildTimes {
  double geometry = 0, fine_system = 0, transfers = 0, coarse_system = 0;
  // (profile detail) the fine system alone, the wait for the coarse chain after it, the fine
  // BuildSystem's row loop and join, and per coarse level l >= 1: transfers l-1 -> l, system l
  double fine_alone = 0, chain_wait = 0, fine_rows = 0, fine_join = 0;
  std::array<double,4> fine_class{};   // thread seconds / rows: inner, outer cells
  std::vector<double> level_transfers, level_coarse;
  std::vector<long long> level_rows;
  // split build (<gravity>/test_gpu_assembly): host tables + classification of all levels, the
  // device thread's wait for them and its device seconds, FinishSplitHierarchy (joins, checks),
  // device / all owned rows of all levels, rows that differ (check), fine-level device stats
  double split_prepare = 0, split_wait = 0, split_device = 0, split_finish = 0, split_join = 0;
  double split_device_wait = 0;   // finish_early: the hierarchy thread's wait for the device
  double split_device_rows = 0, split_rows = 0;
  long long split_differ = 0;
  std::array<double,6> fine_device{};
};
HierarchyBuildTimes &BuildTimes();

// Rediscretized FV A, tensor-linear P, and explicit integrated-residual child-sum
// R. This is R=P0^T, not P^T. Coarsening halves even patch counts, then merges
// complete geometric sibling patch families. Setup stops at <=128 global cells.
// A non-reducible larger grid produces a clear error, never a global fallback.
// Isolated z currently preserves the reference fixed-zero exterior correction;
// its raw-cycle convergence is not asserted by this construction. With internal
// neumann_z=true instead, exterior-center potentials are phi_adjacent+h*q,
// boundary columns contain outward derivatives q, correction fills reflect, and
// the constant nullspace is retained. This requires domain.isolated_z=true.
// threads: host workers of the setup (BuildSystem, transfers) of every geometry
// (MeshGeometry::SetSetupThreads semantics; 1 = serial). The coefficients do not depend on it.
// coarse_quantum > 0 (shear-periodic domains): every coarse level l >= 1 takes the phase
// CoarsePhase(domain.shear_offset, coarse_quantum*MinCellWidth(level patches, y)) instead of
// domain.shear_offset (<gravity>/coarse_phase_quantum); the finest level keeps the exact phase.
// split (test key <gravity>/test_gpu_assembly): the systems of all levels as MeshGeometry split
// builds: after the geometries, PrepareSplit of every level and GpuSplit::Publish; then the host
// rows of all levels (BuildHostRows) on 2/3 of the workers and the transfers on the rest, while
// the thread that owns the device runs GpuSplit::RunDevice; FinishSplitHierarchy after both.
class GpuSplit;
GeometricSystem BuildGeometricHierarchy(std::vector<GravityPatch> patches,
    const CompositeDomain &domain, int rank, int nranks, bool neumann_z = false,
    bool fine_forces = true, int threads = 1, Real coarse_quantum = 0, GpuSplit *split = nullptr);

// The hand-over between the hierarchy thread and the thread that owns the device.
class GpuSplit {
 public:
  int chain_workers = 0;   // transfer workers of the split schedule (0: the default rule)
  GpuSplitOptions options;   // MeshGeometry::PrepareSplit
  // the hierarchy thread joins the device rows itself (FinishSplit of every level after its host
  // rows, while the transfers go on); otherwise FinishSplitHierarchy on the device thread
  bool finish_early = false;
  double wait_s = 0;       // RunDevice: the wait for Publish
  // (hierarchy thread) the next prepared level (fine first); last: no level follows
  void Publish(MeshGeometry *prepared, bool last);
  // (hierarchy thread, on failure, while the geometries still exist) no device pass after
  // this; waits for one that runs
  void Abandon();
  // (device thread) MeshGeometry::RunDevice of every level as it is published; its seconds
  // (returns early after Abandon)
  double RunDevice();
  // (device thread) the device pass will not run (a discarded assembly): WaitDevice returns
  void Cancel();
  // (hierarchy thread) waits for the device pass of level l (true) or Cancel / a failure (false)
  bool WaitDevice(int level);
  // (hierarchy thread) waits for the offsets of level l on the host (true) or Cancel / a failure
  bool WaitCounted(int level);

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool complete_ = false, abandoned_ = false, running_ = false, canceled_ = false;
  int done_ = 0, counted_ = 0;   // levels through RunDevice / with their offsets on the host
  std::vector<MeshGeometry *> levels_;
};
// (device thread, after RunDevice and the hierarchy thread's end) the systems of a split build
// (unless finish_early did it); check != 0: every level also on the host, compared bitwise
// (SG_GPU_ASSEMBLY_CHECK lines)
void FinishSplitHierarchy(GeometricSystem &system, int check, int rank);

// The shear phase of a coarse multigrid level: shift rounded to the nearest multiple of width
// (width <= 0: shift itself). A function of the shift alone, so a rebuilt or restarted
// hierarchy has the coarse operators a continuously updated one has at the same shift.
Real CoarsePhase(Real shift, Real width);
// the smallest cell width along axis d over the patches
Real MinCellWidth(const std::vector<GravityPatch> &patches, int d);

// Both geometries must already have the current phase. Visits only the fixed
// local P dependency list discovered during the original transfer construction.
CsrRowUpdate BuildProlongationPhaseUpdate(const MeshGeometry &fine,
    const MeshGeometry &coarse, const CompositeDomain &domain, int rank,
    const std::vector<int> &rows, ProlongationCacheStats *statistics = nullptr);

}  // namespace gravity
#endif  // GRAVITY_GEOMETRIC_MG_HPP_
