//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_SPARSE_MG_HPP_
#define GRAVITY_SPARSE_MG_HPP_

#include <memory>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace gravity {

// Global host setup data. Iteration retains only locally owned matrix rows and vectors
// on the device. Global geometry/operator assembly is currently a setup limitation.
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

struct SolveStats {
  bool converged = false;
  int iterations = 0;
  int vcycles = 0;
  int matrix_applications = 0;
  Real relative_residual = 0;
  Real removed_rhs_mean = 0;
};

// Restarted flexible GMRES preconditioned by a genuine Galerkin V-cycle. This supports
// nonsymmetric conservative FV operators; it deliberately does not assume CG symmetry.
// Calls are collective, and an instance must not be used concurrently by multiple threads.
class SparseMultigrid {
 public:
  // NVCC requires the enclosing type of extended device lambdas to be accessible.
  // Its definition and all implementation storage remain in the .cpp file.
  struct Impl;
#if MPI_PARALLEL_ENABLED
  explicit SparseMultigrid(const HostHierarchy &hierarchy, MPI_Comm comm = MPI_COMM_WORLD);
#else
  explicit SparseMultigrid(const HostHierarchy &hierarchy);
#endif
  ~SparseMultigrid();
  SparseMultigrid(const SparseMultigrid &) = delete;
  SparseMultigrid &operator=(const SparseMultigrid &) = delete;

  int Levels() const;
  int LocalSize(int level = 0) const;
  int LocalBegin(int level = 0) const;
  int LocalEnd(int level = 0) const;
  SolveStats Solve(const DvceArray1D<Real> &rhs, const DvceArray1D<Real> &x,
                   Real tolerance, int max_iterations, int restart = 24);
  // Collective diagnostic/useful building blocks; all arguments contain local rows only.
  void ApplyMatrix(int level, const DvceArray1D<Real> &x,
                   const DvceArray1D<Real> &result);
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
#else
  DistributedOperator(const HostCsr &matrix, const std::vector<int> &row_partition,
                      const std::vector<int> &col_partition);
#endif
  ~DistributedOperator();
  DistributedOperator(const DistributedOperator &) = delete;
  DistributedOperator &operator=(const DistributedOperator &) = delete;
  void Apply(const DvceArray1D<Real> &x, const DvceArray1D<Real> &result);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gravity
#endif  // GRAVITY_SPARSE_MG_HPP_
