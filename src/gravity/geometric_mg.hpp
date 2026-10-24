//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_GEOMETRIC_MG_HPP_
#define GRAVITY_GEOMETRIC_MG_HPP_

#include <memory>
#include "gravity/mesh_geometry.hpp"

namespace gravity {

struct TransferDiagnostics {
  Real constant_defect = 0;
  Real interior_linear_defect = 0;
  Real restriction_volume_defect = 0;
  GlobalIndex interior_linear_rows = 0;
};

struct GeometricSystem {
  DistributedHierarchy hierarchy;
  // Retains patch metadata and the initialized stable phase-row list only.
  // The adapter can refresh the current fine operator without another volume scan.
  std::unique_ptr<MeshGeometry> fine_geometry;
  // matrix has been moved to hierarchy.matrices[0]. Other finest fields remain.
  LocalCompositeSystem fine;
  // Patch metadata only, one vector per level. No global cell or CSR arrays.
  std::vector<std::vector<GravityPatch>> patches;
  // Local maxima/counts; callers may reduce them for collective diagnostics.
  std::vector<TransferDiagnostics> transfer_checks;
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

}  // namespace gravity
#endif  // GRAVITY_GEOMETRIC_MG_HPP_
