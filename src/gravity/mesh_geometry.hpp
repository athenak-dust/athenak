//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_MESH_GEOMETRY_HPP_
#define GRAVITY_MESH_GEOMETRY_HPP_

#include <array>
#include <functional>
#include <string>
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

// The split build's choices (SelfGravity: hidden <gravity>/test_gpu_* keys): the irregular rows
// on the device, their distance in cell widths from the x boundary and isolated-z walls, the
// smallest level (owned rows) on the device
struct GpuSplitOptions {
  bool irregular = true;
  double margin = 12.;
  long long min_rows = 200000;
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
  std::size_t quadratic_x_image_fallbacks = 0;
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
  // selects min(32, affinity CPUs - 2) workers in GPU builds and one in CPU builds. Workers
  // only assemble host rows; callers must not concurrently use this geometry.
  void SetPhaseThreads(int threads);
  int PhaseThreads() const;  // actual workers, including the calling thread
  // Host workers of a full setup (BuildSystem, transfers; default 1). 0 selects every
  // affinity CPU in device builds (the host is otherwise idle during a rebuild), one in CPU
  // builds. The coefficients do not depend on it.
  void SetSetupThreads(int threads);
  // the worker count SetSetupThreads(threads) gives
  static int ResolveSetupThreads(int threads);
  int SetupThreads() const;
  // Requires initial BuildSystem or PhaseRows. Only PhaseRows are visited;
  // physical topology and sample ownership stay fixed across arbitrary phases.
  // false updates only A, without boundary/force CSR allocation or assembly.
  LocalCompositeUpdate BuildPhaseUpdate(Real phase, bool forces = true);
  // host seconds of the last BuildPhaseUpdate: {row assembly, join of the worker chunks}
  std::array<double,2> LastPhaseUpdateSeconds() const;
  // host seconds of the last BuildSystem: {row assembly, join of the worker chunks} and (with
  // SetupTimesOn) summed thread seconds and rows of cells >= 2 cells inside their patch / others
  std::array<double,6> LastBuildSeconds() const;
  // Test key <gravity>/test_gpu_assembly: BuildSystem(forces) in steps, the regular rows
  // (gpu_assembly.hpp) on the device concurrently with the other rows on the host.
  // PrepareSplit (any thread): host tables and the host classification of the owned rows; then
  // BuildHostRows (any thread, the setup workers) and RunDevice (only on the thread that owns
  // the device) in any order or concurrently; then FinishSplit: the BuildSystem(forces) system.
  void PrepareSplit(bool forces, const GpuSplitOptions &options = GpuSplitOptions{});
  void ClassifySplit();   // after PrepareSplit, before BuildHostRows (any thread)
  void BuildHostRows();
  // slot: page-locked download buffers (one per level); counted (optional): called once the
  // offsets are on the host (FinishSplitPrepare may run from then on)
  void RunDevice(int slot = 0, const std::function<void()> &counted = {});
  LocalCompositeSystem FinishSplit();   // = FinishSplitPrepare + FinishSplitCopy
  // FinishSplit in two steps: Prepare after BuildHostRows and the counted call of RunDevice
  // (the irregular rows the device left to the host, the output sizes), Copy after RunDevice
  void FinishSplitPrepare();
  LocalCompositeSystem FinishSplitCopy();
  // (check) BuildSystem(forces) on the host, compared bitwise with a FinishSplit result: the
  // number of rows (and metadata) that differ, the first difference in *first_difference
  long long CheckSplit(const LocalCompositeSystem &split, bool forces,
                       std::string *first_difference = nullptr);
  // after PrepareSplit: {rows the device will assemble, owned rows}
  std::array<long long,2> SplitRows() const;
  // the last FinishSplit: {device rows, owned rows, host tables + classification s, upload +
  // kernels s, download s, join s}
  std::array<double,6> LastDeviceStats() const;
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
  // images includes recursive fills. shear_images reports x-image traversals
  // only, including those first introduced by a recursive reconstruction fill.
  std::vector<std::pair<GlobalIndex, Real>> VirtualAverage(
      const std::array<Real, 3> &lo, const std::array<Real, 3> &hi,
      bool *images = nullptr, bool *shear_images = nullptr) const;

 private:
  LocalCompositeSystem BuildSystemRows(bool forces, bool split);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gravity
#endif  // GRAVITY_MESH_GEOMETRY_HPP_
