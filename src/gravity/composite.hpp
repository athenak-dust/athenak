//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_COMPOSITE_HPP_
#define GRAVITY_COMPOSITE_HPP_

#include <array>
#include <vector>

#include "gravity/sparse_mg.hpp"

namespace gravity {

struct CompositeCell {
  std::array<Real, 3> center, width;
};

struct CompositeDomain {
  std::array<Real, 3> lo{{0, 0, 0}}, hi{{1, 1, 1}};
  int ndim = 3;
  // Root-level CELL counts, rather than MeshBlock counts. Rectangular cells allowed.
  std::array<int, 3> root_counts{{1, 1, 1}};
  // phi(x+Lx,y) = phi(x,y-shear_offset), reduced modulo Ly during setup.
  Real shear_offset = 0;
  bool isolated_z = false;
};

struct BoundarySample {
  std::array<Real, 3> center;
  int side;  // -1 lower z, +1 upper z; sample is one cell-center spacing outside
};

struct CompositeSystem {
  HostHierarchy hierarchy;                // A = -integrated FV Laplacian
  std::array<HostCsr, 3> acceleration;      // MINUS averaged shared normal derivatives
  std::vector<Real> volume;
  std::vector<unsigned char> interface;    // cell touches a different refinement level
  // Caller supplies potential at these exterior CELL CENTERS, not at the wall face.
  // They are fixed data, rather than additional solver unknowns. For isolated_z:
  // A*phi = integrated_physical_rhs + boundary_rhs*phi_boundary;
  // acceleration = G*phi + boundary_acceleration*phi_boundary.
  std::vector<BoundarySample> boundary_samples;
  HostCsr boundary_rhs;
  std::array<HostCsr, 3> boundary_acceleration;
};

// Global host setup reference: cells use caller-supplied global row order. An AABB tree
// finds neighboring face fragments without quadratic searches on whole face planes.
// Dyadic leaf boxes must completely tile the rectangular domain. All active directions
// are periodic; x faces may have a physical y shift. In 3-D isolated_z replaces z
// wrapping by fixed exterior center samples. Computing physical isolated boundary
// values from a source distribution is the caller's responsibility.
// Setup is not distributed, and this API does not transfer fields during AMR rebuilds.
// Coarsening uses dyadic parents at every scale, smoothed aggregate interpolation and
// Galerkin R=P^T. Periodic smoothed P preserves constants, without an exact-linearity
// claim. Isolated coarse correction is homogeneous and has no constant nullspace.
CompositeSystem BuildCompositeSystem(const std::vector<CompositeCell> &cells,
    const CompositeDomain &domain, const std::vector<int> &fine_row_partition = {});

}  // namespace gravity
#endif  // GRAVITY_COMPOSITE_HPP_
