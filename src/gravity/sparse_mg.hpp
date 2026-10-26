//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_SPARSE_MG_HPP_
#define GRAVITY_SPARSE_MG_HPP_

#include <cstdint>
#include <memory>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace gravity {

using GlobalIndex = std::int64_t;

// Legacy reference input with global host setup data. Iteration retains only local
// rows/vectors. The distributed production interface below avoids this global input.
struct HostCsr {
  int rows = 0, cols = 0;
  std::vector<int> rowptr, colidx;
  std::vector<Real> values;
};

struct HostHierarchy {
  std::vector<HostCsr> matrices;       // fine to coarse, including tiny coarse matrix
  std::vector<HostCsr> prolongation;   // P[l]: level l+1 -> l; restriction is its transpose
  // Each supplied partition has nranks+1 nondecreasing offsets. Empty partitions use
  // balanced contiguous ranges. Zero-row ranks are supported on every level.
  std::vector<std::vector<int>> row_partitions;
  bool constant_nullspace = false;    // both left and right constant nullspaces
};

// Contiguous OWNED rows only, with 64-bit GLOBAL column IDs. A zero-row rank
// supplies rowptr={0}. rowptr offsets and locally remapped device columns are
// 32-bit; neither global cell count nor global row/column IDs have that limit.
struct LocalCsr {
  GlobalIndex global_rows = 0, global_cols = 0, row_begin = 0;
  std::vector<int> rowptr{0};
  std::vector<GlobalIndex> colidx;
  std::vector<Real> values;
  int LocalRows() const { return static_cast<int>(rowptr.size())-1; }
};

// Complete replacements for a fixed sorted/unique list of LOCAL output rows.
// rowptr indexes the compact rows array; column IDs remain global and 64-bit.
struct CsrRowUpdate {
  GlobalIndex global_rows = 0, global_cols = 0, row_begin = 0;
  std::vector<int> rows;
  std::vector<int> rowptr{0};
  std::vector<GlobalIndex> colidx;
  std::vector<Real> values;
};

struct DistributedHierarchy {
  std::vector<LocalCsr> matrices;       // fine to coarse, owned A rows only
  std::vector<LocalCsr> prolongation;   // P[l]: coarse -> fine
  std::vector<LocalCsr> restriction;    // R[l]: fine -> coarse; explicit production input
  std::vector<std::vector<GlobalIndex>> row_partitions;
  bool constant_nullspace = false;
};

struct SolveStats {
  bool converged = false;
  int iterations = 0;
  int vcycles = 0;
  int matrix_applications = 0;
  Real relative_residual = 0;
  Real removed_rhs_mean = 0;
};

// Local storage, including retained host halo buffers but excluding input setup data.
struct StorageStats {
  std::uint64_t a_nnz = 0, p_nnz = 0, r_nnz = 0;
  std::uint64_t csr_bytes = 0, halo_bytes = 0, vector_bytes = 0;
  std::uint64_t current_a_nnz = 0, host_pattern_bytes = 0;
  std::uint64_t phase_nnz = 0, phase_rows = 0, phase_map_bytes = 0;
  std::uint64_t terminal_host_csr_bytes = 0;
  // Retained phase ghost-map buckets and value payload; excludes allocator/node overhead.
  std::uint64_t phase_ghost_map_bytes = 0;
};

struct FineOperatorStats {
  std::uint64_t updates = 0, value_updates = 0, structure_rebuilds = 0;
  // Actual local phase operations in the current base-operator epoch. Halo
  // rebuilds are collective; local CSR changes need not occur on every rank.
  std::uint64_t local_pattern_changes = 0, halo_rebuilds = 0, halo_reuses = 0;
  std::uint64_t local_ghost_changes = 0, global_ghost_changes = 0;
  // Last map size/bucket count and peak bucket+value payload bytes, excluding
  // implementation-specific node/allocator overhead (not a process RSS estimate).
  std::uint64_t host_map_entries = 0, host_map_buckets = 0, host_map_peak_bytes = 0;
};

// Restarted flexible GMRES preconditioned by the supplied multigrid V-cycle. The
// production path accepts independent A/P/R; the legacy adapter uses Galerkin R=P^T.
// Supports nonsymmetric conservative FV operators without assuming CG symmetry.
// Calls are collective, and an instance must not be used concurrently by multiple threads.
class SparseMultigrid {
 public:
  // NVCC requires the enclosing type of extended device lambdas to be accessible.
  // Its definition and all implementation storage remain in the .cpp file.
  struct Impl;
#if MPI_PARALLEL_ENABLED
  explicit SparseMultigrid(const HostHierarchy &hierarchy, MPI_Comm comm = MPI_COMM_WORLD);
  explicit SparseMultigrid(const DistributedHierarchy &hierarchy,
                          MPI_Comm comm = MPI_COMM_WORLD);
#else
  explicit SparseMultigrid(const HostHierarchy &hierarchy);
  explicit SparseMultigrid(const DistributedHierarchy &hierarchy);
#endif
  ~SparseMultigrid();
  SparseMultigrid(const SparseMultigrid &) = delete;
  SparseMultigrid &operator=(const SparseMultigrid &) = delete;

  int Levels() const;
  int LocalSize(int level = 0) const;
  GlobalIndex LocalBegin(int level = 0) const;
  GlobalIndex LocalEnd(int level = 0) const;
  StorageStats Storage() const;
  // Refresh only the CURRENT outer fine matrix. The cached MG A/P/R and coarse
  // solve remain unchanged. Ownership and nullspace type must remain unchanged;
  // otherwise construct a fresh hierarchy. Empty partition uses existing ownership.
  void UpdateFineOperator(const LocalCsr &matrix,
                          const std::vector<GlobalIndex> &partition = {});
  FineOperatorStats FineUpdates() const;
  // Sum current fine/coarse A/P phase counters (R is fixed); map peaks are summed
  // per operator, not a simultaneous transient process-memory measurement.
  FineOperatorStats HierarchyPhaseUpdates() const;
  void ConfigureFinePhaseRows(const std::vector<int> &rows);
  // Refresh the outer and V-cycle finest operator together, including selected
  // smoother diagonal entries. They share one CSR/halo; coarse A/P/R stay cached.
  // Re-entering compact mode after a full outer-only update validates/refreshes
  // the entire effective fine diagonal once. A single-level hierarchy has no
  // smoother and retains its cached terminal LU.
  void UpdateFinePhaseRows(const CsrRowUpdate &matrix);
  // Complete current-phase V-cycle: independent compact A and P dependency
  // lists, unchanged physical child-sum R, and current terminal factorization.
  // Terminal ownership must contain <=128 global rows. No fine host CSR retained.
  void ConfigureHierarchyPhaseRows(const std::vector<std::vector<int>> &a_rows,
                                  const std::vector<std::vector<int>> &p_rows);
  void UpdateHierarchyPhaseRows(const std::vector<CsrRowUpdate> &a,
                               const std::vector<CsrRowUpdate> &p);
  SolveStats Solve(const DvceArray1D<Real> &rhs, const DvceArray1D<Real> &x,
                   Real tolerance, int max_iterations, int restart = 24);
  // Collective diagnostics; arguments contain owned rows only. Level zero uses
  // CURRENT outer A; higher levels use the preconditioner matrices, updated by
  // UpdateHierarchyPhaseRows when that interface is configured.
  void ApplyMatrix(int level, const DvceArray1D<Real> &x,
                   const DvceArray1D<Real> &result);
  void ApplyProlongation(int level, const DvceArray1D<Real> &x,
                         const DvceArray1D<Real> &result);
  void ApplyRestriction(int level, const DvceArray1D<Real> &x,
                        const DvceArray1D<Real> &result);
  // Uses the current finest phase operator/diagonal after UpdateFinePhaseRows;
  // all other levels and transfers remain cached for the fine-only interface.
  // UpdateHierarchyPhaseRows instead refreshes every A/P and terminal LU.
  void ApplyVCycle(const DvceArray1D<Real> &rhs, const DvceArray1D<Real> &result);

 private:
  friend class DistributedOperator;
  std::unique_ptr<Impl> impl_;
};

// Standalone rectangular distributed CSR, using the same reusable sparse halo as MG.
// This also serves device evaluation of reconstructed acceleration components.
class DistributedOperator {
 public:
#if MPI_PARALLEL_ENABLED
  DistributedOperator(const HostCsr &matrix, const std::vector<int> &row_partition,
                      const std::vector<int> &col_partition,
                      MPI_Comm comm = MPI_COMM_WORLD);
  DistributedOperator(const LocalCsr &matrix,
                      const std::vector<GlobalIndex> &row_partition,
                      const std::vector<GlobalIndex> &col_partition,
                      MPI_Comm comm = MPI_COMM_WORLD);
#else
  DistributedOperator(const HostCsr &matrix, const std::vector<int> &row_partition,
                      const std::vector<int> &col_partition);
  DistributedOperator(const LocalCsr &matrix,
                      const std::vector<GlobalIndex> &row_partition,
                      const std::vector<GlobalIndex> &col_partition);
#endif
  ~DistributedOperator();
  DistributedOperator(const DistributedOperator &) = delete;
  DistributedOperator &operator=(const DistributedOperator &) = delete;
  void Apply(const DvceArray1D<Real> &x, const DvceArray1D<Real> &result);
  StorageStats Storage() const;
  void ConfigurePhaseRows(const std::vector<int> &rows);
  void UpdatePhaseRows(const CsrRowUpdate &matrix);
  FineOperatorStats PhaseUpdates() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gravity
#endif  // GRAVITY_SPARSE_MG_HPP_
