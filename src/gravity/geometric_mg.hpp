//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_GEOMETRIC_MG_HPP_
#define GRAVITY_GEOMETRIC_MG_HPP_

#include <cstdint>
#include <memory>
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
};

// Rediscretized FV A, tensor-linear P, and explicit integrated-residual child-sum
// R. This is R=P0^T, not P^T. Coarsening halves even patch counts, then merges
// complete geometric sibling patch families. Setup stops at <=128 global cells.
// A non-reducible larger grid produces a clear error, never a global fallback.
// Isolated z currently preserves the reference fixed-zero exterior correction;
// its raw-cycle convergence is not asserted by this construction. With internal
// neumann_z=true instead, exterior-center potentials are phi_adjacent+h*q,
// boundary columns contain outward derivatives q, correction fills reflect, and
// the constant nullspace is retained. This requires domain.isolated_z=true.
GeometricSystem BuildGeometricHierarchy(std::vector<GravityPatch> patches,
    const CompositeDomain &domain, int rank, int nranks, bool neumann_z = false,
    bool fine_forces = true);

// Both geometries must already have the current phase. Visits only the fixed
// local P dependency list discovered during the original transfer construction.
CsrRowUpdate BuildProlongationPhaseUpdate(const MeshGeometry &fine,
    const MeshGeometry &coarse, const CompositeDomain &domain, int rank,
    const std::vector<int> &rows, ProlongationCacheStats *statistics = nullptr);

}  // namespace gravity
#endif  // GRAVITY_GEOMETRIC_MG_HPP_
