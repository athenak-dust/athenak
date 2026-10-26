//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_VACUUM_MESH_HPP_
#define GRAVITY_VACUUM_MESH_HPP_

#include <cstddef>
#include <vector>

#include "gravity/mesh_geometry.hpp"

namespace gravity {

// Gravity-only metadata; no global cells, source arrays, or matrices are built.
// MeshGeometry(..., neumann_z) expands only the patches owned by its rank.
struct VacuumMesh {
  std::vector<GravityPatch> patches;
  CompositeDomain domain;
  bool neumann_z = true;
  // patches are owner-ordered. Within each owner the unchanged physical patches
  // retain their input order and precede all vacuum patches. Thus physical source
  // rows occupy [row_partition[r], row_partition[r]+physical_cells_per_rank[r]).
  std::vector<GlobalIndex> row_partition, physical_cells_per_rank;
  std::vector<GlobalIndex> total_cells_per_rank;
  // Both maps use PATCH indices. A vacuum entry has original_patch=-1 and MUST
  // receive zero source. Physical gid/geometry/cell ordering are unchanged.
  std::vector<int> original_patch, input_to_output_patch;
  GlobalIndex physical_unknowns = 0, vacuum_unknowns = 0;
  std::size_t physical_patch_count = 0, vacuum_patch_count = 0;
  std::size_t metadata_bytes = 0;
  Real padding = 0, unknown_overhead = 1;
  int balance_passes = 0;
};

// Extend both z faces by >=2*max(Lx,Ly), rounded UP to coarse-tile alignment.
// Actual hydro/dust patches are never refined, clipped, or replaced. Exterior
// patches contain 8^3 cells and satisfy h(d)<=h_boundary*exp(kmin*d/2), using the
// finest LOCAL boundary spacing under their footprint, kmin=2pi/max(Lx,Ly).
// Patch metadata is refined spatially and balanced2:1 (faces/edges/corners and
// UNSHIFTED horizontal periodic images); no finest boundary plane exists. Layout
// is independent of instantaneous shear_offset. Shifted-seam ratios are handled
// by the composite operator, rather than rebuilding the exterior with shear phase.
//
// Initial explicit production constraints (violations throw): ndim=3, isolated_z;
// physical root counts powers-of-two >=8; physical patch counts powers-of-two >=8;
// common root-cell aspect ratio, isotropic dyadic refinement, and patch bounds
// aligned to 8-cell tiles at that patch's resolution. Physical neighbors must
// already be2:1 in unshifted geometry, because source refinement is forbidden.
VacuumMesh BuildVacuumMesh(const std::vector<GravityPatch> &physical_patches,
                          const CompositeDomain &physical_domain, int nranks);

}  // namespace gravity
#endif  // GRAVITY_VACUUM_MESH_HPP_
