//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_MESH_GEOMETRY_HPP_
#define GRAVITY_MESH_GEOMETRY_HPP_

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "gravity/composite.hpp"

namespace gravity {

// Global metadata is one record per patch, never one record per cell. The builder
// assigns row_begin in owner order, preserving input patch order within an owner.
// Cell ordering inside each patch is x1-fastest, then x2, then x3. gid is opaque
// source-adapter metadata and is preserved. Inactive cell counts must equal one.
struct GravityPatch {
  std::array<Real, 3> lo{{0, 0, 0}}, hi{{1, 1, 1}};
  std::array<int, 3> cells{{1, 1, 1}};
  int owner = 0, gid = -1;
  GlobalIndex row_begin = 0;
};

struct LocalCompositeSystem {
  // For isolated z, false means supplied exterior-center potentials; true means
  // supplied OUTWARD NORMAL DERIVATIVES at the corresponding wall faces.
  bool neumann_z = false;
  LocalCsr matrix;
  std::array<LocalCsr, 3> acceleration;
  LocalCsr boundary_rhs;
  std::array<LocalCsr, 3> boundary_acceleration;
  std::vector<Real> volume;
  std::vector<CompositeCell> cells;
  std::vector<unsigned char> interface;
  // Owned samples only, in sample_partition[rank] order. Within a patch the
  // lower z face precedes the upper face; each face has x1-fastest ordering.
  // Coordinates retain exterior-center locations even in Neumann mode; its
  // derivative datum belongs to the corresponding physical wall face.
  std::vector<BoundarySample> boundary_samples;
  // Same owned-sample order: adjacent real solver row, physical face area and
  // adjacent cell's normal width. The adjacent row is owned by this rank.
  std::vector<GlobalIndex> boundary_cells;
  std::vector<Real> boundary_area, boundary_width;
  std::vector<GlobalIndex> row_partition, sample_partition;
  // Unique local coarse fits that lacked a full-rank one-sided quadratic and
  // used a two-sided fit. This fallback is forbidden on the physical fine mesh.
  std::size_t quadratic_interface_fallbacks = 0;
};

// Compact replacements for the same stable local row list in every component.
// Geometry, ownership, volumes and boundary-sample metadata remain unchanged.
struct LocalCompositeUpdate {
  CsrRowUpdate matrix;
  std::array<CsrRowUpdate, 3> acceleration;
  CsrRowUpdate boundary_rhs;
  std::array<CsrRowUpdate, 3> boundary_acceleration;
};

class MeshGeometry {
 public:
  MeshGeometry(std::vector<GravityPatch> patches, const CompositeDomain &domain,
               int rank, int nranks, bool neumann_z = false,
               bool coarse_reconstruction = false);
  ~MeshGeometry();
  MeshGeometry(MeshGeometry &&) noexcept;
  MeshGeometry &operator=(MeshGeometry &&) noexcept;
  MeshGeometry(const MeshGeometry &) = delete;
  MeshGeometry &operator=(const MeshGeometry &) = delete;

  // Coarse setup requests only A, avoiding coordinate/force/boundary CSR storage.
  LocalCompositeSystem BuildSystem(bool forces = true);
  // BuildSystem records this dependency list during its initial row pass. A
  // stand-alone call discovers it once without constructing matrices.
  const std::vector<int> &PhaseRows() const;
  // 1 keeps serial assembly; positive counts are capped by CPU affinity. Zero
  // selects at most eight workers in GPU builds and one in CPU builds. Workers
  // only assemble host rows; callers must not concurrently use this geometry.
  void SetPhaseThreads(int threads);
  int PhaseThreads() const;  // actual workers, including the calling thread
  // Requires initial BuildSystem or PhaseRows. Only PhaseRows are visited;
  // physical topology and sample ownership stay fixed across arbitrary phases.
  LocalCompositeUpdate BuildPhaseUpdate(Real phase);
  const std::vector<GravityPatch> &Patches() const;
  const std::vector<GlobalIndex> &RowPartition() const;
  const std::vector<GlobalIndex> &SamplePartition() const;
  GlobalIndex GlobalRows() const;
  // Point lookup with periodic/shear images; upper endpoints are identified periodically
  // except at isolated z walls, where they select the adjacent interior cell.
  GlobalIndex Locate(const std::array<Real, 3> &point) const;
  CompositeCell Cell(GlobalIndex row) const;
  // Fundamental-domain open-volume intersection, not a global cell expansion.
  std::vector<GlobalIndex> QueryBox(const std::array<Real, 3> &lo,
                                  const std::array<Real, 3> &hi) const;
  // Average over a virtual rectangular cell. Fine leaves contribute overlap
  // volume averages; a larger covering leaf uses centered linear reconstruction.
  // Periodic/shear boxes are split exactly into physical images. Isolated-z
  // exterior correction values are zero, or reflected for neumann_z. Optional
  // images includes recursive fills.
  std::vector<std::pair<GlobalIndex, Real>> VirtualAverage(
      const std::array<Real, 3> &lo, const std::array<Real, 3> &hi,
      bool *images = nullptr) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gravity
#endif  // GRAVITY_MESH_GEOMETRY_HPP_
