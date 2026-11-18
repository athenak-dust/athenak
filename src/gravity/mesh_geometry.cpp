//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/mesh_geometry.hpp"
#include "gravity/gpu_assembly.hpp"
#include "gravity/gpu_irregular.hpp"
#include "gravity/host_workers.hpp"
#include "gravity/row_index.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#if defined(__linux__)
#include <sched.h>
#endif
#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

namespace gravity {
namespace {
using Vec = std::array<double, 3>;
using Entry = std::pair<GlobalIndex, double>;
using Row = std::vector<Entry>;
using Gradient = std::vector<std::pair<GlobalIndex, Vec>>;
using GradientCache = std::unordered_map<GlobalIndex, Gradient>;
using Modes = std::array<double,9>;
struct Quadratic {
  std::vector<std::pair<GlobalIndex,Modes>> coefficients;
  bool phase_dependent = false;
  bool interface_fallback = false;
  bool x_image_fallback = false;
};
using QuadraticCache = std::unordered_map<GlobalIndex,Quadratic>;
struct Box { Vec lo{}, hi{}; };
struct Node { Box box; int begin, end, left = -1, right = -1; };
struct Face {
  GlobalIndex left, right;
  int normal, side;
  double area = 1., distance;
  Vec delta{}, ti{}, tj{}, width{};
};
struct ReconstructionFaces {
  std::unordered_map<GlobalIndex,std::vector<Face>> rows;
  size_t bytes = 0;
  static size_t Bytes(const std::vector<Face> &faces) {
    return faces.capacity()*sizeof(Face)+sizeof(std::vector<Face>)+
           sizeof(GlobalIndex)+4*sizeof(void *);
  }
  void Clear() { rows.clear(); bytes = 0; }
};
// Long rows (and fit sample/coefficient lists) are indexed: see row_index.hpp.
using AccRow = IndexedRow<double>;

void Add(Row &row, GlobalIndex column, double value) {
  // Most rows contain only seven entries; a short contiguous scan avoids a
  // heap node per coefficient and avoids gradient maps on aligned interiors.
  if (value == 0.) return;
  for (auto &v : row) if (v.first == column) { v.second += value; return; }
  row.emplace_back(column, value);
}

// As Add(Row &): the same entries and sums, with the index on long rows.
void Add(AccRow &row, GlobalIndex column, double value) { row.Add(column, value); }

template<typename Csr>
void Append(Csr &matrix, AccRow &accumulated, GlobalIndex split, bool exterior,
            double sign = 1.) {
  auto &row = accumulated.Release();   // sorted below
  std::sort(row.begin(), row.end(), [](const Entry &a, const Entry &b) {
    return a.first < b.first;
  });
  for (const auto &v : row) if ((v.first >= split) == exterior && v.second != 0.) {
    if (matrix.values.size() == static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Gravity local CSR exceeds 32-bit offsets");
    matrix.colidx.push_back(exterior ? v.first-split : v.first);
    matrix.values.push_back(sign*v.second);
  }
  matrix.rowptr.push_back(static_cast<int>(matrix.values.size()));
}

std::array<Vec, 3> Inverse(std::array<Vec, 3> a, int ndim) {
  std::array<Vec, 3> b{};
  for (int d = 0; d < ndim; ++d) b[d][d] = 1.;
  for (int d = 0; d < ndim; ++d) {
    int pivot = d;
    for (int k = d+1; k < ndim; ++k)
      if (std::abs(a[k][d]) > std::abs(a[pivot][d])) pivot = k;
    if (!(std::abs(a[pivot][d]) > std::numeric_limits<double>::min()))
      throw std::runtime_error("Gravity singular neighbor gradient geometry");
    std::swap(a[d], a[pivot]); std::swap(b[d], b[pivot]);
    const double scale = a[d][d];
    for (int k = 0; k < ndim; ++k) { a[d][k] /= scale; b[d][k] /= scale; }
    for (int r = 0; r < ndim; ++r) if (r != d) {
      const double f = a[r][d];
      for (int k = 0; k < ndim; ++k) {
        a[r][k] -= f*a[d][k]; b[r][k] -= f*b[d][k];
      }
    }
  }
  return b;
}

struct RowWorkspace {
  AccRow a, difference, potential;
  std::array<AccRow,3> g;
  RowWorkspace() {
    a.v.reserve(32); difference.v.reserve(32); potential.v.reserve(48);
    for (auto &row : g) row.v.reserve(32);
  }
};

#if defined(KOKKOS_ENABLE_CUDA)
using PinnedSpace = Kokkos::CudaHostPinnedSpace;
#else
using PinnedSpace = Kokkos::HostSpace;
#endif
// (<gravity>/test_gpu_assembly) the page-locked download buffers of the split builds per level slot
// and component, kept for the next rebuild (page-locked allocations are slow); only the thread
// that owns the device takes them, a rebuild's systems are complete before the next one
struct PinnedPool {
  std::vector<std::array<Kokkos::View<GlobalIndex*, PinnedSpace>,4>> offset, col;
  std::vector<std::array<Kokkos::View<double*, PinnedSpace>,4>> val;
};
PinnedPool &Pinned() {
  static PinnedPool pool;
  // released before Kokkos::finalize
  static const bool hooked = []() {
    Kokkos::push_finalize_hook([]() { Pinned() = PinnedPool{}; });
    return true;
  }();
  (void)hooked;
  return pool;
}
template <class T>
Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> Take(
    Kokkos::View<T*, PinnedSpace> &buffer, std::size_t n, const char *name) {
  if (buffer.extent(0) < n) {
    buffer = Kokkos::View<T*, PinnedSpace>();
    buffer = Kokkos::View<T*, PinnedSpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(name)),n+n/8+1024);
  }
  return Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(buffer.data(),n);
}

int AvailablePhaseCpus() {
#if defined(__linux__)
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (sched_getaffinity(0,sizeof(affinity),&affinity) == 0)
    return std::max(1,CPU_COUNT(&affinity));
#endif
  // Unknown affinity must not cause accidental node-wide oversubscription.
  return 1;
}

}  // namespace

struct MeshGeometry::Impl {
  std::vector<GravityPatch> patches;
  CompositeDomain domain;
  int rank, nranks, nd;
  bool neumann_z = false;
  bool coarse_reconstruction = false;
  Vec length{}, minwidth{};
  double shear;
  std::vector<GlobalIndex> partition, sample_partition, sample_begin;
  std::vector<int> order, row_order, sample_order;
  std::vector<Node> tree;
  GradientCache gradients;
  QuadraticCache quadratics;
  ReconstructionFaces reconstruction_faces;
  struct PhaseContext {
    QuadraticCache quadratics;
    ReconstructionFaces faces;
    double shear = std::numeric_limits<double>::quiet_NaN();
  };
  std::vector<PhaseContext> phase_contexts;
  int phase_threads = 1, setup_threads = 1;
  double phase_rows_s = 0, phase_join_s = 0;   // last BuildPhaseUpdate: row assembly, join
  double build_rows_s = 0, build_join_s = 0;   // last BuildSystem: row assembly, join
  std::array<double,4> build_class{};   // (profile) thread seconds and rows: inner / outer cells
  std::array<double,gpu::kReasons> reason_seconds{};   // (profile) host row thread seconds by reason
  std::vector<int> phase_rows;
  bool phase_rows_ready = false;
  std::array<double,6> device_stats{};   // see LastDeviceStats
  // One worker chunk of a BuildSystem: consecutive owned rows.
  struct Part {
    LocalCsr matrix, boundary_rhs;
    std::array<LocalCsr,3> acceleration, boundary_acceleration;
    std::vector<Real> volume;
    std::vector<CompositeCell> cells;
    std::vector<unsigned char> interface;
    std::vector<int> phase_rows;
    // (profile) host seconds and rows: cells >= 2 cells inside their patch / all others
    double t_inner = 0, t_outer = 0;
    std::array<double,gpu::kReasons> t_reason{};   // (profile, split build) host rows by reason
    long long n_inner = 0, n_outer = 0;
  };
  // <gravity>/test_gpu_assembly split build (PrepareSplit, BuildHostRows, RunDevice, FinishSplit)
  struct Split {
    bool prepared = false, forces = false, host_done = false, device_done = false;
    bool no_device = false;   // no device rows: RunDevice does nothing
    // the classification parameters (PrepareSplit), the host classification done (ClassifySplit),
    // the device's own classification (downloaded, compared in FinishSplitPrepare)
    bool device_level = false, irregular_on = false, classified = false;
    double margin = 12.;
    std::vector<int> device_kind;
    int components = 0;
    // host tables (gpu::HostGeometry points into these) and per owned row: 1 = a device row,
    // else -(the reason it stays on the host)
    // (1: the regular-row kernels of gpu_assembly.hpp, 2: the irregular-row kernels of
    // gpu_irregular.hpp, whose status 1/2 = assembled (2: an interface row), <= 0 = left to the
    // host in FinishSplit)
    gpu::HostGeometry g;
    std::vector<double> width, center, plo, phi;
    std::vector<int> cells, coff, nb, wrap, owned, owned_first, regular, status;
    std::vector<int> cand_off, cand, rorder, irregular_rows;
    std::vector<GlobalIndex> row_begin;
    std::array<long long,gpu::kReasons> why{};   // rows by the first non-regular reason
    long long irregular = 0, fallback = 0;
    std::array<long long,8> failures{};          // irregular rows left to the host, by status
    // the host pass's fits with an interface / x-image fallback (sorted unique ids)
    std::vector<GlobalIndex> interface_ids, x_image_ids;
    // the device rows (host copies): per component (0: A, 1+d: acceleration d) offsets per
    // owned row (+1), columns and values
    std::array<Kokkos::View<GlobalIndex*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>,4> offset, colidx;
    std::array<Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>,4> values;
    // the host pass: its chunks (device rows are empty placeholders) and the output so far
    std::vector<Part> parts;
    LocalCompositeSystem out;
    // FinishSplit in two steps: Prepare (needs the device offsets and statuses: the irregular
    // rows left to the host, the output sizes) and Copy (needs the device entries)
    struct Fallback {
      std::array<LocalCsr,8> c;
      unsigned char interface = 0;
      bool phase = false;
    };
    bool counted = false, join_prepared = false;
    std::vector<int> failed;
    std::vector<Fallback> fallback_rows;
    std::vector<size_t> row_offset;
    std::vector<std::vector<size_t>> entry_offset;
    double join_t0 = 0;
    double prepare_s = 0, upload_s = 0, kernels_s = 0, download_s = 0, join_s = 0;
    std::string memory;   // (diagnostic) free device memory at the device steps
  } split;

  static GlobalIndex Count(const GravityPatch &p) {
    GlobalIndex n = 1;
    for (int c : p.cells) {
      if (c <= 0 || n > std::numeric_limits<GlobalIndex>::max()/c)
        throw std::invalid_argument("Gravity invalid/overflowing patch cell count");
      n *= c;
    }
    return n;
  }
  bool Wall(const GravityPatch &p, int side) const {
    return domain.isolated_z && std::abs((side < 0 ? p.lo[2] : p.hi[2])-
        (side < 0 ? domain.lo[2] : domain.hi[2])) < minwidth[2]*1.e-5;
  }
  GlobalIndex Samples(int p) const {
    return static_cast<GlobalIndex>(patches[p].cells[0])*patches[p].cells[1]*
        (static_cast<int>(Wall(patches[p], -1))+static_cast<int>(Wall(patches[p], 1)));
  }
  Box PatchBox(int p) const {
    Box b;
    for (int d = 0; d < nd; ++d) { b.lo[d] = patches[p].lo[d]; b.hi[d] = patches[p].hi[d]; }
    return b;
  }
  bool Intersects(const Box &a, const Box &b) const {
    for (int d = 0; d < nd; ++d)
      if (a.hi[d] < b.lo[d] || b.hi[d] < a.lo[d]) return false;
    return true;
  }
  int BuildTree(int begin, int end) {
    const int id = tree.size();
    Node node{}; node.begin = begin; node.end = end;
    node.box.lo.fill(std::numeric_limits<double>::max());
    node.box.hi.fill(-std::numeric_limits<double>::max());
    for (int k = begin; k < end; ++k) for (int d = 0; d < nd; ++d) {
      node.box.lo[d] = std::min(node.box.lo[d], static_cast<double>(patches[order[k]].lo[d]));
      node.box.hi[d] = std::max(node.box.hi[d], static_cast<double>(patches[order[k]].hi[d]));
    }
    tree.push_back(node);
    if (end-begin > 8) {
      int axis = 0;
      for (int d = 1; d < nd; ++d)
        if (node.box.hi[d]-node.box.lo[d] > node.box.hi[axis]-node.box.lo[axis]) axis = d;
      const int mid = begin+(end-begin)/2;
      std::nth_element(order.begin()+begin, order.begin()+mid, order.begin()+end,
          [&](int a, int b) { return patches[a].lo[axis]+patches[a].hi[axis] <
                                    patches[b].lo[axis]+patches[b].hi[axis]; });
      const int l = BuildTree(begin, mid), r = BuildTree(mid, end);
      tree[id].left = l; tree[id].right = r;
    }
    return id;
  }
  void Query(int node_id, const Box &box, std::vector<int> &result) const {
    const auto &node = tree[node_id];
    if (!Intersects(node.box, box)) return;
    if (node.left >= 0) { Query(node.left, box, result); Query(node.right, box, result); }
    else for (int k = node.begin; k < node.end; ++k)
      if (Intersects(PatchBox(order[k]), box)) result.push_back(order[k]);
  }
  int Patch(GlobalIndex row) const {
    auto it = std::upper_bound(row_order.begin(), row_order.end(), row,
        [&](GlobalIndex r, int p) { return r < patches[p].row_begin; });
    if (it == row_order.begin() || row < 0 || row >= partition.back())
      throw std::out_of_range("Gravity invalid global cell row");
    return *--it;
  }
  std::array<int, 3> Indices(int p, GlobalIndex row) const {
    auto q = row-patches[p].row_begin;
    const int i = q%patches[p].cells[0]; q /= patches[p].cells[0];
    const int j = q%patches[p].cells[1]; q /= patches[p].cells[1];
    return {{i, j, static_cast<int>(q)}};
  }
  GlobalIndex Id(int p, const std::array<int, 3> &ijk) const {
    return patches[p].row_begin+ijk[0]+static_cast<GlobalIndex>(patches[p].cells[0])*
        (ijk[1]+static_cast<GlobalIndex>(patches[p].cells[1])*ijk[2]);
  }
  CompositeCell Cell(int p, const std::array<int, 3> &ijk) const {
    CompositeCell c{};
    for (int d = 0; d < 3; ++d) {
      c.width[d] = (patches[p].hi[d]-patches[p].lo[d])/patches[p].cells[d];
      c.center[d] = patches[p].lo[d]+(ijk[d]+.5)*c.width[d];
    }
    return c;
  }
  CompositeCell Cell(GlobalIndex row) const { const int p = Patch(row); return Cell(p, Indices(p,row)); }
  GlobalIndex Sample(int p, const std::array<int, 3> &ijk, int side) const {
    return partition.back()+sample_begin[p]+ijk[0]+
        static_cast<GlobalIndex>(patches[p].cells[0])*ijk[1]+
        (side > 0 && Wall(patches[p],-1) ?
         static_cast<GlobalIndex>(patches[p].cells[0])*patches[p].cells[1] : 0);
  }
  std::pair<GlobalIndex,double> Ghost(GlobalIndex column) const {
    const GlobalIndex sample = column-partition.back();
    auto it = std::upper_bound(sample_order.begin(),sample_order.end(),sample,
        [&](GlobalIndex s, int p) { return s < sample_begin[p]; });
    if (it == sample_order.begin() || sample < 0 || sample >= sample_partition.back())
      throw std::out_of_range("Gravity invalid exterior sample column");
    const int p = *--it;
    GlobalIndex offset = sample-sample_begin[p];
    const GlobalIndex plane = static_cast<GlobalIndex>(patches[p].cells[0])*patches[p].cells[1];
    const bool lower = Wall(patches[p],-1) && offset < plane;
    if (!lower && Wall(patches[p],-1)) offset -= plane;
    const std::array<int,3> ijk{{static_cast<int>(offset%patches[p].cells[0]),
        static_cast<int>(offset/patches[p].cells[0]),lower ? 0 : patches[p].cells[2]-1}};
    return {Id(p,ijk),(patches[p].hi[2]-patches[p].lo[2])/patches[p].cells[2]};
  }
  Face MakeFace(GlobalIndex row, const CompositeCell &ci, GlobalIndex other,
                const CompositeCell &cj, const Vec &image, int d, int side) const {
    Face f{}; f.normal = d; f.side = side;
    const auto &left = side > 0 ? ci : cj;
    const auto &right = side > 0 ? cj : ci;
    f.left = side > 0 ? row : other; f.right = side > 0 ? other : row;
    for (int t = 0; t < nd; ++t) {
      const double shift = side*image[t];
      f.delta[t] = right.center[t]+shift-left.center[t];
      if (t == d) continue;
      const double lo = std::max(static_cast<double>(left.center[t]-.5*left.width[t]),
                                right.center[t]-.5*right.width[t]+shift);
      const double hi = std::min(static_cast<double>(left.center[t]+.5*left.width[t]),
                                right.center[t]+.5*right.width[t]+shift);
      if (hi-lo <= minwidth[t]*1.e-8) { f.area = 0.; break; }
      f.area *= hi-lo;
      f.width[t] = hi-lo;
      f.ti[t] = .5*(hi+lo)-left.center[t];
      f.tj[t] = .5*(hi+lo)-right.center[t]-shift;
      // Roundoff in patch coordinates must not turn an aligned seven-point
      // interior into a least-squares stencil (notably for counts such as 24).
      if (std::abs(f.ti[t]) < minwidth[t]*1.e-12) f.ti[t] = 0.;
      if (std::abs(f.tj[t]) < minwidth[t]*1.e-12) f.tj[t] = 0.;
    }
    f.distance = f.delta[d];
    if (f.area > 0. && !(f.distance > 0.))
      throw std::runtime_error("Gravity invalid face normal distance");
    return f;
  }
  std::vector<Face> Faces(GlobalIndex row) const {
    const int p = Patch(row);
    const auto ijk = Indices(p, row);
    const auto ci = Cell(p, ijk);
    std::vector<Face> faces; faces.reserve(2*nd+4);
    Vec coverage{};
    for (int d = 0; d < nd; ++d) for (int side : {-1, 1}) {
      auto next = ijk; next[d] += side;
      if (next[d] >= 0 && next[d] < patches[p].cells[d]) {
        Face f = MakeFace(row,ci,Id(p,next),Cell(p,next),Vec{},d,side);
        coverage[d] += f.area; faces.push_back(f); continue;
      }
      const double edge = ci.center[d]+side*.5*ci.width[d];
      const bool periodic = std::abs(edge-(side < 0 ? domain.lo[d] : domain.hi[d])) <
                            minwidth[d]*1.e-5;
      if (periodic && d == 2 && domain.isolated_z) {
        Face f{}; f.normal = d; f.side = side;
        f.distance = ci.width[d]; f.delta[d] = f.distance;
        f.left = side < 0 ? Sample(p,ijk,side) : row;
        f.right = side < 0 ? row : Sample(p,ijk,side);
        for (int t = 0; t < nd; ++t) if (t != d) {
          f.area *= ci.width[t]; f.width[t] = ci.width[t];
        }
        coverage[d] += f.area; faces.push_back(f); continue;
      }
      const int wraps = periodic && d == 0 ? 1 : 0;
      for (int wrap = -wraps; wrap <= wraps; ++wrap) {
        Vec image{};
        if (periodic) image[d] = side*length[d];
        if (periodic && d == 0) image[1] = side*shear+wrap*length[1];
        Box query{};
        for (int t = 0; t < nd; ++t) {
          query.lo[t] = ci.center[t]-.5*ci.width[t]-image[t];
          query.hi[t] = ci.center[t]+.5*ci.width[t]-image[t];
        }
        const double probe = edge-image[d]+side*minwidth[d]*1.e-6;
        query.lo[d] = probe; query.hi[d] = probe;
        std::vector<int> candidates; Query(0,query,candidates);
        for (int q : candidates) {
          std::array<int,3> first{}, last{};
          bool empty = false;
          for (int t = 0; t < nd; ++t) {
            const double h = (patches[q].hi[t]-patches[q].lo[t])/patches[q].cells[t];
            if (t == d) {
              first[t] = last[t] = static_cast<int>(std::floor((probe-patches[q].lo[t])/h));
            } else {
              first[t] = static_cast<int>(std::floor((query.lo[t]-patches[q].lo[t])/h+1.e-8));
              last[t] = static_cast<int>(std::ceil((query.hi[t]-patches[q].lo[t])/h-1.e-8))-1;
            }
            first[t] = std::max(0,first[t]); last[t] = std::min(patches[q].cells[t]-1,last[t]);
            if (first[t] > last[t]) empty = true;
          }
          if (empty) continue;
          for (int k = first[2]; k <= last[2]; ++k)
            for (int j = first[1]; j <= last[1]; ++j)
              for (int i = first[0]; i <= last[0]; ++i) {
                const std::array<int,3> target{{i,j,k}};
                const auto cj = Cell(q,target);
                if (std::abs(cj.center[d]-side*.5*cj.width[d]+image[d]-edge) >
                    minwidth[d]*1.e-5) continue;
                auto f = MakeFace(row,ci,Id(q,target),cj,image,d,side);
                if (f.area == 0.) continue;
                coverage[d] += f.area; faces.push_back(f);
              }
        }
      }
    }
    double volume = 1.;
    for (int d = 0; d < nd; ++d) volume *= ci.width[d];
    for (int d = 0; d < nd; ++d) {
      const double expected = 2.*volume/ci.width[d];
      if (std::abs(coverage[d]-expected) > expected*1.e-5)
        throw std::runtime_error("Gravity incomplete/overlapping cell-face coverage");
    }
    return faces;
  }
  const std::vector<Face> &CachedReconstructionFaces(GlobalIndex row,
      ReconstructionFaces &cache, std::vector<Face> &uncached) const {
    const auto found = cache.rows.find(row);
    if (found != cache.rows.end()) return found->second;
    auto faces = Faces(row);
    const size_t bytes = ReconstructionFaces::Bytes(faces);
    // This cache serves repeated reconstruction neighborhoods, not a global
    // face expansion. Keep its payload bounded independently of mesh size.
    constexpr size_t limit = 8*1024*1024;
    if (bytes > limit) { uncached = std::move(faces); return uncached; }
    if (cache.bytes+bytes > limit) cache.Clear();
    cache.bytes += bytes;
    return cache.rows.emplace(row,std::move(faces)).first->second;
  }
  const Gradient &Grad(GlobalIndex row, GradientCache &cache) const {
    auto found = cache.find(row);
    if (found != cache.end()) return found->second;
    const auto faces = Faces(row);
    std::array<Vec,3> normal{};
    for (const auto &f : faces) {
      double d2 = 0.;
      for (int d = 0; d < nd; ++d) d2 += f.delta[d]*f.delta[d];
      for (int r = 0; r < nd; ++r) for (int d = 0; d < nd; ++d)
        normal[r][d] += f.area/d2*f.delta[r]*f.delta[d];
    }
    normal = Inverse(normal,nd);
    Gradient g;
    auto add = [&](GlobalIndex column, int d, double value) {
      for (auto &v : g) if (v.first == column) { v.second[d] += value; return; }
      Vec v{}; v[d] = value; g.emplace_back(column,v);
    };
    for (const auto &f : faces) {
      double d2 = 0.;
      for (int d = 0; d < nd; ++d) d2 += f.delta[d]*f.delta[d];
      const double sign = f.side;
      const GlobalIndex other = row == f.left ? f.right : f.left;
      for (int r = 0; r < nd; ++r) {
        double coefficient = 0.;
        for (int d = 0; d < nd; ++d)
          coefficient += normal[r][d]*sign*f.delta[d]*f.area/d2;
        add(other,r,coefficient); add(row,r,-coefficient);
      }
    }
    return cache.emplace(row,std::move(g)).first->second;
  }
  // Point-center quadratic reconstruction. Two face-neighbor rings include
  // mixed moments even on a Cartesian interior (one ring has only axis points).
  // All coordinates are local unwrapped images, including at a shearing seam.
  const Quadratic &Quad(GlobalIndex row, QuadraticCache &cache,
                        ReconstructionFaces &face_cache) const {
    auto found = cache.find(row);
    if (found != cache.end()) return found->second;
    struct Sample { GlobalIndex id; Vec delta; };
    const auto cell = Cell(row);
    std::vector<Sample> samples;
    PositionIndex sample_index;   // (cleared with every samples.clear())
    bool one_sided = !domain.c1_z_interfaces.empty();
    // A cell spanning the entire x period has no same-side x neighbors.
    // On the deepest vacuum levels it must use periodic images from the
    // outset, rather than inferring periodic curvature from one-sided samples.
    // Keep the real source/vacuum z filter independent of this choice.
    const double x_period = domain.hi[0]-domain.lo[0];
    bool one_sided_x = domain.shear_phase_updates &&
        cell.width[0] < x_period*(1.-64.*std::numeric_limits<double>::epsilon());
    Quadratic q;
    auto add = [&](GlobalIndex id, const Vec &delta) {
      const double x = cell.center[0]+delta[0];
      const bool x_image = x < domain.lo[0] || x >= domain.hi[0];
      if (one_sided_x && x_image) return;
      if (one_sided) for (double plane : domain.c1_z_interfaces)
        if ((cell.center[2] < plane) != (cell.center[2]+delta[2] < plane)) return;
      double distance = 0.;
      for (int d = 0; d < nd; ++d)
        distance += (delta[d]/cell.width[d])*(delta[d]/cell.width[d]);
      if (distance < 1.e-20) return;
      auto same = [&](const Sample &s) {
        double error = 0.;
        for (int d = 0; d < nd; ++d) error += std::abs(s.delta[d]-delta[d])/cell.width[d];
        return error < 1.e-8;
      };
      auto key = [&samples](std::size_t i) { return samples[i].id; };
      if (!sample_index.Active()) {
        for (const auto &s : samples) if (s.id == id && same(s)) return;
      } else if (sample_index.Find(id,key,[&](std::size_t i) { return same(samples[i]); }) >= 0) {
        return;
      }
      samples.push_back({id,delta});
      if (sample_index.Active()) sample_index.Appended(id,samples.size()-1,key);
      else if (samples.size() > PositionIndex::kLinear) sample_index.Rebuild(samples.size(),key);
      q.phase_dependent = q.phase_dependent || x_image;
    };
    auto neighbors = [&](GlobalIndex id, const Vec &offset) {
      std::vector<Face> uncached;
      for (const auto &f : CachedReconstructionFaces(id,face_cache,uncached)) {
        const double sign = f.side;
        Vec delta = offset;
        for (int d = 0; d < nd; ++d) delta[d] += sign*f.delta[d];
        add(id == f.left ? f.right : f.left,delta);
      }
    };
    neighbors(row,Vec{});
    auto expand = [&](size_t first, size_t last) {
      for (size_t i = first; i < last; ++i) {
        const auto sample = samples[i];
        if (sample.id < partition.back()) neighbors(sample.id,sample.delta);
      }
    };
    size_t first = 0, last = samples.size();
    expand(first,last); first = last; last = samples.size();
    const int modes = nd+nd*(nd+1)/2;
    auto basis = [&](const Vec &delta) {
      Modes b{};
      for (int d = 0; d < nd; ++d) b[d] = delta[d]/cell.width[d];
      int k = nd;
      for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t)
        b[k++] = b[d]*b[t]*(d == t ? .5 : 1.);
      return b;
    };
    std::array<Modes,9> inverse{};
    auto invert = [&]() {
      std::array<Modes,9> normal{}; inverse = {};
      for (const auto &s : samples) {
        const auto b = basis(s.delta);
        double distance = 0.;
        for (int d = 0; d < nd; ++d) distance += b[d]*b[d];
        const double weight = 1./distance;
        for (int i = 0; i < modes; ++i) for (int j = 0; j < modes; ++j)
          normal[i][j] += weight*b[i]*b[j];
      }
      for (int i = 0; i < modes; ++i) inverse[i][i] = 1.;
      for (int i = 0; i < modes; ++i) {
        int pivot = i;
        for (int j = i+1; j < modes; ++j)
          if (std::abs(normal[j][i]) > std::abs(normal[pivot][i])) pivot = j;
        if (std::abs(normal[pivot][i]) < 1.e-12) return false;
        std::swap(normal[i],normal[pivot]); std::swap(inverse[i],inverse[pivot]);
        const double scale = normal[i][i];
        for (int j = 0; j < modes; ++j) {
          normal[i][j] /= scale; inverse[i][j] /= scale;
        }
        for (int j = 0; j < modes; ++j) if (j != i) {
          const double factor = normal[j][i];
          for (int k = 0; k < modes; ++k) {
            normal[j][k] -= factor*normal[i][k];
            inverse[j][k] -= factor*inverse[i][k];
          }
        }
      }
      return true;
    };
    bool full_rank = invert();
    // Near a join, find enough same-side normal positions rather than falling
    // back merely because the first two rings were anisotropic or incomplete.
    for (int ring = 2; !full_rank && (one_sided || one_sided_x) && ring < 4; ++ring) {
      expand(first,last); first = last; last = samples.size();
      full_rank = invert();
    }
    if (!full_rank && one_sided_x) {
      // Thin x regions may not determine curvature without periodic images.
      // Keep the real source/vacuum z filter strict in this fallback.
      one_sided_x = false; samples.clear(); sample_index.Clear(); q.phase_dependent = false;
      q.x_image_fallback = true;
      neighbors(row,Vec{}); first = 0; last = samples.size();
      expand(first,last); first = last; last = samples.size();
      full_rank = invert();
      for (int ring = 2; !full_rank && one_sided && ring < 4; ++ring) {
        expand(first,last); first = last; last = samples.size();
        full_rank = invert();
      }
    }
    if (!full_rank && one_sided && coarse_reconstruction) {
      // A coarse region may have only one normal cell, so quadratic curvature
      // on that side is unknowable. Keep an elliptic preconditioner stencil;
      // expose this lower-order C1 fallback in hierarchy diagnostics.
      one_sided = false; samples.clear(); sample_index.Clear(); q.interface_fallback = true;
      q.phase_dependent = false;
      neighbors(row,Vec{}); expand(0,samples.size());
      full_rank = invert();
    }
    if (!full_rank)
      throw std::runtime_error("Gravity singular one-sided quadratic reconstruction");
    q.coefficients.reserve(samples.size()+1);
    PositionIndex coefficient_index;
    auto coefficient = [&](GlobalIndex id, const Modes &value) {
      auto &c = q.coefficients;
      auto key = [&c](std::size_t i) { return c[i].first; };
      std::ptrdiff_t found = -1;
      if (!coefficient_index.Active()) {
        for (std::size_t i = 0; i < c.size(); ++i) if (c[i].first == id) { found = i; break; }
      } else {
        found = coefficient_index.Find(id,key,[](std::size_t) { return true; });
      }
      if (found >= 0) {
        for (int i = 0; i < modes; ++i) c[found].second[i] += value[i];
        return;
      }
      c.emplace_back(id,value);
      if (coefficient_index.Active()) coefficient_index.Appended(id,c.size()-1,key);
      else if (c.size() > PositionIndex::kLinear) coefficient_index.Rebuild(c.size(),key);
    };
    Modes center{};
    for (const auto &s : samples) {
      const auto b = basis(s.delta);
      double distance = 0.;
      for (int d = 0; d < nd; ++d) distance += b[d]*b[d];
      Modes value{};
      for (int i = 0; i < modes; ++i) {
        for (int j = 0; j < modes; ++j) value[i] += inverse[i][j]*b[j]/distance;
        center[i] -= value[i];
      }
      coefficient(s.id,value);
    }
    coefficient(row,center);
    return cache.emplace(row,std::move(q)).first->second;
  }
  bool TouchesX(const CompositeCell &cell) const {
    return std::abs(cell.center[0]-.5*cell.width[0]-domain.lo[0]) < minwidth[0]*1.e-5 ||
           std::abs(cell.center[0]+.5*cell.width[0]-domain.hi[0]) < minwidth[0]*1.e-5;
  }
  bool TouchesX(GlobalIndex row) const { return TouchesX(Cell(row)); }
  bool HasOffset(const Vec &offset) const {
    for (int d = 0; d < nd; ++d) if (offset[d] != 0.) return true;
    return false;
  }
  bool AtC1Interface(const Face &f) const {
    if (f.normal != 2 || f.left >= partition.back() || f.right >= partition.back())
      return false;
    const auto left = Cell(f.left);
    const double z = left.center[2]+.5*left.width[2];
    for (double plane : domain.c1_z_interfaces)
      if (std::abs(z-plane) < minwidth[2]*1.e-7) return true;
    return false;
  }
  bool IsPhaseRow(GlobalIndex row) const {
    const auto cell = Cell(row); double volume = 1.;
    for (int d = 0; d < nd; ++d) volume *= cell.width[d];
    QuadraticCache cache; ReconstructionFaces faces;
    RowWorkspace work; bool dependent = false;
    AssembleRow(row,cell,volume,true,work,cache,faces,nullptr,&dependent);
    return dependent;
  }
  void DiscoverPhaseRows() {
    if (phase_rows_ready) return;
    phase_rows.clear();
    const GlobalIndex begin = partition[rank], end = partition[rank+1];
    for (GlobalIndex row = begin; row < end; ++row)
      if (IsPhaseRow(row)) phase_rows.push_back(static_cast<int>(row-begin));
    phase_rows_ready = true;
  }
  void AddPotential(AccRow &target, GlobalIndex column, double value) const {
    if (neumann_z && column >= partition.back()) {
      const auto ghost = Ghost(column);
      Add(target,ghost.first,value); Add(target,column,value*ghost.second);
    } else Add(target,column,value);
  }
  // Add diagonal Hessians using cheap centered axes where all neighbors are
  // aligned and equally sized. Only irregular/one-sided cells need a full fit.
  void AddDiagonal(AccRow &target, GlobalIndex row, const Vec &weights,
                   QuadraticCache &cache, ReconstructionFaces &face_cache,
                   bool *phase_dependent) const {
    const int p = Patch(row); const auto index = Indices(p,row);
    const auto cell = Cell(p,index);
    bool interior = true, near_plane = false;
    for (int d = 0; d < nd; ++d)
      interior = interior && index[d] > 0 && index[d]+1 < patches[p].cells[d];
    for (double plane : domain.c1_z_interfaces)
      near_plane = near_plane || std::abs(std::abs(cell.center[2]-plane)-
                   .5*cell.width[2]) < minwidth[2]*1.e-7;
    std::array<std::array<GlobalIndex,2>,3> neighbor{};
    bool regular = !near_plane && !(domain.shear_phase_updates && TouchesX(cell));
    if (interior && regular) {
      for (int d = 0; d < nd; ++d) for (int side = 0; side < 2; ++side) {
        auto next = index; next[d] += 2*side-1; neighbor[d][side] = Id(p,next);
      }
    } else if (regular) {
      std::vector<Face> uncached;
      const auto &faces = CachedReconstructionFaces(row,face_cache,uncached);
      regular = faces.size() == static_cast<size_t>(2*nd);
      for (const auto &f : faces) {
        const auto other = f.side > 0 ? f.right : f.left;
        regular = regular && !HasOffset(f.ti) && !HasOffset(f.tj) && !AtC1Interface(f);
        if (other < partition.back()) {
          const auto other_cell = Cell(other);
          for (int d = 0; d < nd; ++d)
            regular = regular && std::abs(other_cell.width[d]-cell.width[d]) <
                                 minwidth[d]*1.e-8;
        }
        neighbor[f.normal][f.side > 0 ? 1 : 0] = other;
      }
    }
    if (regular) {
      for (int d = 0; d < nd; ++d) {
        const double coefficient = weights[d]/(cell.width[d]*cell.width[d]);
        AddPotential(target,neighbor[d][0],coefficient);
        AddPotential(target,neighbor[d][1],coefficient);
        AddPotential(target,row,-2.*coefficient);
      }
    } else {
      const auto &q = Quad(row,cache,face_cache);
      if (phase_dependent && q.phase_dependent) *phase_dependent = true;
      for (const auto &v : q.coefficients) {
        double coefficient = 0.; int k = nd;
        for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t) {
          if (d == t) coefficient += weights[d]*v.second[k]/
                                     (cell.width[d]*cell.width[d]);
          ++k;
        }
        AddPotential(target,v.first,coefficient);
      }
    }
  }
  void FacePotential(const Face &face, AccRow &potential, QuadraticCache &cache,
                     ReconstructionFaces &face_cache, bool *phase_dependent) const {
    potential.clear();
    const GlobalIndex n = partition.back();
    if (face.left >= n || face.right >= n) {
      // Mirrored ghost potential is known, or phi_adjacent+h*q in Neumann
      // mode. The adjacent Hessian supplies the point-to-face-average moments.
      const auto row = face.left < n ? face.left : face.right;
      const auto cell = Cell(row);
      AddPotential(potential,face.left,.5); AddPotential(potential,face.right,.5);
      Vec weights{};
      for (int d = 0; d < nd; ++d)
        weights[d] = d == face.normal ? -.125*cell.width[d]*cell.width[d] :
                                       face.width[d]*face.width[d]/24.;
      AddDiagonal(potential,row,weights,cache,face_cache,phase_dependent);
      return;
    }
    const auto left = Cell(face.left), right = Cell(face.right);
    const double weight_left = right.width[face.normal]/
                              (left.width[face.normal]+right.width[face.normal]);
    const bool transverse = HasOffset(face.ti) || HasOffset(face.tj);
    auto side = [&](GlobalIndex row, const CompositeCell &cell,
                    const Vec &offset, double weight) {
      AddPotential(potential,row,weight);
      Vec diagonal{};
      for (int d = 0; d < nd; ++d)
        diagonal[d] = weight*(d == face.normal ? -.125*cell.width[d]*cell.width[d] :
                                                face.width[d]*face.width[d]/24.);
      if (!transverse) {
        AddDiagonal(potential,row,diagonal,cache,face_cache,phase_dependent);
        return;
      }
      const auto &q = Quad(row,cache,face_cache);
      if (phase_dependent && q.phase_dependent) *phase_dependent = true;
      Modes moment{};
      for (int d = 0; d < nd; ++d) moment[d] = weight*offset[d]/cell.width[d];
      int k = nd;
      for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t) {
        moment[k] = weight*offset[d]*offset[t]/(cell.width[d]*cell.width[t])*
                    (d == t ? .5 : 1.);
        if (d == t) moment[k] += diagonal[d]/(cell.width[d]*cell.width[d]);
        ++k;
      }
      for (const auto &v : q.coefficients) {
        double coefficient = 0.;
        for (int i = 0; i < k; ++i) coefficient += moment[i]*v.second[i];
        AddPotential(potential,v.first,coefficient);
      }
    };
    side(face.left,left,face.ti,weight_left);
    side(face.right,right,face.tj,1.-weight_left);
  }
  // Full setup and phase refresh deliberately share this exact physical row
  // routine, including canonical faces and all Dirichlet/Neumann LS couplings.
  void AssembleRow(GlobalIndex row, const CompositeCell &ci, double volume,
                   bool forces, RowWorkspace &work, QuadraticCache &quadratic_cache,
                   ReconstructionFaces &face_cache,
                   bool *is_interface = nullptr, bool *phase_dependent = nullptr) const {
    work.a.clear(); for (auto &component : work.g) component.clear();
    if (is_interface) *is_interface = false;
    if (phase_dependent) *phase_dependent = TouchesX(ci);
    for (const auto &f : Faces(row)) {
      work.difference.clear();
      AddPotential(work.difference,f.right,1.); AddPotential(work.difference,f.left,-1.);
      if (f.left < partition.back() && f.right < partition.back()) {
        const auto left = Cell(f.left), right = Cell(f.right);
        const bool c1 = AtC1Interface(f);
        const double shift = .25*(left.width[f.normal]-right.width[f.normal]);
        if (HasOffset(f.ti) || HasOffset(f.tj) || c1 ||
            left.width[f.normal] != right.width[f.normal]) {
          auto reconstruct = [&](GlobalIndex id, const CompositeCell &cell,
                                 const Vec &offset, double sign) {
            const auto &q = Quad(id,quadratic_cache,face_cache);
            if (phase_dependent && q.phase_dependent) *phase_dependent = true;
            Modes moment{};
            for (int d = 0; d < nd; ++d) moment[d] = sign*offset[d]/cell.width[d];
            int k = nd;
            for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t) {
              moment[k] = sign*offset[d]*offset[t]/(cell.width[d]*cell.width[t])*
                          (d == t ? .5 : 1.);
              if (d == f.normal && t == d) {
                // Real C1 joins need separate curvatures. Smooth equal-width
                // shear faces need a correction that vanishes with alignment.
                if (c1) moment[k] -= sign*.125;
                else moment[k] += .5*f.distance*shift/(cell.width[d]*cell.width[d]);
              }
              ++k;
            }
            for (const auto &v : q.coefficients) {
              double value = 0.;
              for (int i = 0; i < k; ++i) value += moment[i]*v.second[i];
              AddPotential(work.difference,v.first,value);
            }
          };
          reconstruct(f.right,right,f.tj,1.); reconstruct(f.left,left,f.ti,-1.);
        }
      }
      const double sign = f.side;
      for (const auto &v : work.difference) {
        const double flux = f.area/f.distance*v.second;
        // A periodic image can be the same real cell when an axis has one
        // cell. Its two oriented flux contributions cancel in A, while both
        // still contribute to the averaged acceleration below. Row IDs alone
        // cannot distinguish the two sides of such a sheared self-face.
        if (f.left != f.right) Add(work.a,v.first,-sign*flux);
      }
      if (forces) {
        FacePotential(f,work.potential,quadratic_cache,face_cache,phase_dependent);
        for (const auto &v : work.potential)
          Add(work.g[f.normal],v.first,-sign*f.area*v.second/volume);
      }
      const GlobalIndex other = row == f.left ? f.right : f.left;
      if (is_interface && other < partition.back()) {
        const auto cj = Cell(other);
        if (std::abs(ci.width[0]-cj.width[0]) > minwidth[0]*1.e-5) *is_interface = true;
      }
    }
  }
  void AssemblePhaseRange(size_t first, size_t last, LocalCompositeUpdate &out,
                          PhaseContext &context, bool forces) const {
    // A context belongs to exactly one worker. A two-ring fit changes phase
    // when either its center or a first-ring neighbor touches the seam.
    if (context.shear != shear) {
      for (auto it = context.quadratics.begin(); it != context.quadratics.end();) {
        if (it->second.phase_dependent) it = context.quadratics.erase(it);
        else ++it;
      }
      for (auto it = context.faces.rows.begin(); it != context.faces.rows.end();) {
        if (TouchesX(it->first)) {
          context.faces.bytes -= ReconstructionFaces::Bytes(it->second);
          it = context.faces.rows.erase(it);
        } else ++it;
      }
      context.shear = shear;
    }
    const GlobalIndex n = partition.back(), ns = sample_partition.back();
    const GlobalIndex begin = partition[rank];
    auto init = [&](CsrRowUpdate &a, GlobalIndex columns) {
      a.global_rows = n; a.global_cols = columns; a.row_begin = begin;
      a.rows.assign(phase_rows.begin()+first,phase_rows.begin()+last);
      a.rowptr.reserve(last-first+1);
    };
    init(out.matrix,n);
    if (forces) init(out.boundary_rhs,ns);
    for (int d = 0; forces && d < 3; ++d) {
      init(out.acceleration[d],n); init(out.boundary_acceleration[d],ns);
    }
    RowWorkspace work;
    for (size_t i = first; i < last; ++i) {
      const GlobalIndex row = begin+phase_rows[i];
      const auto cell = Cell(row); double volume = 1.;
      for (int d = 0; d < nd; ++d) volume *= cell.width[d];
      AssembleRow(row,cell,volume,forces,work,context.quadratics,context.faces);
      Append(out.matrix,work.a,n,false);
      if (forces) Append(out.boundary_rhs,work.a,n,true,-1.);
      for (int d = 0; forces && d < 3; ++d) {
        Append(out.acceleration[d],work.g[d],n,false);
        Append(out.boundary_acceleration[d],work.g[d],n,true);
      }
    }
  }
  std::vector<GlobalIndex> QueryBox(const Box &box) const {
    std::vector<int> candidates; Query(0,box,candidates);
    std::vector<GlobalIndex> result;
    for (int p : candidates) {
      std::array<int,3> first{}, last{};
      bool empty = false;
      for (int d = 0; d < nd; ++d) {
        const double h = (patches[p].hi[d]-patches[p].lo[d])/patches[p].cells[d];
        first[d] = std::max(0,static_cast<int>(std::floor(
            (box.lo[d]-patches[p].lo[d])/h+1.e-8)));
        last[d] = std::min(patches[p].cells[d]-1,static_cast<int>(std::ceil(
            (box.hi[d]-patches[p].lo[d])/h-1.e-8))-1);
        if (first[d] > last[d]) empty = true;
      }
      if (empty) continue;
      for (int k = first[2]; k <= last[2]; ++k)
        for (int j = first[1]; j <= last[1]; ++j)
          for (int i = first[0]; i <= last[0]; ++i)
            result.push_back(Id(p,{{i,j,k}}));
    }
    return result;
  }
  GlobalIndex TryFindPoint(const Vec &point) const {
    Box query; query.lo = query.hi = point;
    std::vector<int> candidates; Query(0,query,candidates);
    for (int p : candidates) {
      std::array<int,3> ijk{}; bool inside = true;
      for (int d = 0; d < nd; ++d) {
        const double h = (patches[p].hi[d]-patches[p].lo[d])/patches[p].cells[d];
        ijk[d] = static_cast<int>(std::floor((point[d]-patches[p].lo[d])/h));
        if (ijk[d] < 0 || ijk[d] >= patches[p].cells[d]) inside = false;
      }
      if (inside) return Id(p,ijk);
    }
    // A virtual coarse-cell center can lie at a fine-patch vertex. Decimal
    // bounds can place it an ulp outside each half-open index range, despite
    // valid face coverage. Point lookup is only an optimization for an exact
    // leaf average; use volume integration when this probe is ambiguous.
    return -1;
  }
  // Recursion moves to a strictly larger covering leaf before requesting its
  // centered stencil. It therefore terminates at a finite mesh/root scale.
  Row Average(const Box &box, bool &images, bool &shear_images, int depth = 0) const {
    if (depth > 64) throw std::runtime_error("Gravity virtual-cell recursion exceeded 64");
    Vec width{}; double volume = 1.;
    for (int d = 0; d < nd; ++d) {
      width[d] = box.hi[d]-box.lo[d];
      if (!(width[d] > 0.) || width[d] > length[d]*(1.+1.e-8))
        throw std::invalid_argument("Gravity invalid virtual cell extent");
      volume *= width[d];
    }
    if (domain.isolated_z && !neumann_z &&
        (box.hi[2] <= domain.lo[2] || box.lo[2] >= domain.hi[2])) {
      images = true; return {};
    }
    Vec center{};
    for (int d = 0; d < nd; ++d) center[d] = .5*(box.lo[d]+box.hi[d]);
    const double xwrap = std::floor((center[0]-domain.lo[0])/length[0]);
    center[0] -= xwrap*length[0]; center[1] -= xwrap*shear;
    bool crossed = xwrap != 0;
    shear_images = shear_images || xwrap != 0;
    for (int d = 1; d < nd; ++d) {
      if (d == 2 && domain.isolated_z) continue;
      const double wrap = std::floor((center[d]-domain.lo[d])/length[d]);
      center[d] -= wrap*length[d]; crossed = crossed || wrap != 0;
    }
    // Uniform interiors dominate setup. Exact virtual leaf cells need one point
    // lookup and one coefficient, without box fragmentation or reconstruction.
    if (!(domain.isolated_z && (center[2] < domain.lo[2] || center[2] >= domain.hi[2]))) {
      const GlobalIndex row = TryFindPoint(center);
      if (row >= 0) {
        const auto c = Cell(row); bool exact = true;
        for (int d = 0; d < nd; ++d)
          exact = exact && std::abs(width[d]-c.width[d]) < minwidth[d]*1.e-8 &&
              std::abs(center[d]-c.center[d]) < minwidth[d]*1.e-8;
        if (exact) { images = images || crossed; return Row{{row,1.}}; }
      }
    }
    AccRow out;
    auto integrate = [&](const Box &physical) {
      for (GlobalIndex row : QueryBox(physical)) {
        const auto c = Cell(row);
        Vec offset{}; double overlap = 1.; bool reconstruct = false;
        for (int d = 0; d < nd; ++d) {
          const double lo = std::max(physical.lo[d],c.center[d]-.5*c.width[d]);
          const double hi = std::min(physical.hi[d],c.center[d]+.5*c.width[d]);
          overlap *= std::max(0.,hi-lo); offset[d] = .5*(hi+lo)-c.center[d];
          reconstruct = reconstruct || c.width[d] > width[d]*(1.+1.e-8);
        }
        if (overlap == 0.) continue;
        const double weight = overlap/volume; Add(out,row,weight);
        if (!reconstruct) continue;
        for (int d = 0; d < nd; ++d) {
          if (std::abs(offset[d]) < minwidth[d]*1.e-12) continue;
          for (int side : {-1,1}) {
            Box neighbor{};
            for (int t = 0; t < nd; ++t) {
              const double center = c.center[t]+(d == t ? side*c.width[t] : 0.);
              neighbor.lo[t] = center-.5*c.width[t]; neighbor.hi[t] = center+.5*c.width[t];
            }
            const auto fill = Average(neighbor,images,shear_images,depth+1);
            const double factor = side*.5*offset[d]/c.width[d]*weight;
            for (const auto &v : fill) Add(out,v.first,factor*v.second);
          }
        }
      }
    };
    // Split x, then apply its exact shear displacement before splitting y.
    const int xmin = static_cast<int>(std::floor((box.lo[0]-domain.lo[0])/length[0]+1.e-12));
    const int xmax = static_cast<int>(std::ceil((box.hi[0]-domain.lo[0])/length[0]-1.e-12))-1;
    for (int ix = xmin; ix <= xmax; ++ix) {
      Box x = box;
      x.lo[0] = std::max(box.lo[0],domain.lo[0]+ix*length[0])-ix*length[0];
      x.hi[0] = std::min(box.hi[0],domain.hi[0]+ix*length[0])-ix*length[0];
      x.lo[1] -= ix*shear; x.hi[1] -= ix*shear;
      images = images || ix != 0;
      shear_images = shear_images || ix != 0;
      const int ymin = static_cast<int>(std::floor((x.lo[1]-domain.lo[1])/length[1]+1.e-12));
      const int ymax = static_cast<int>(std::ceil((x.hi[1]-domain.lo[1])/length[1]-1.e-12))-1;
      for (int iy = ymin; iy <= ymax; ++iy) {
        Box y = x;
        y.lo[1] = std::max(x.lo[1],domain.lo[1]+iy*length[1])-iy*length[1];
        y.hi[1] = std::min(x.hi[1],domain.hi[1]+iy*length[1])-iy*length[1];
        images = images || iy != 0;
        if (nd == 2) { integrate(y); continue; }
        if (domain.isolated_z) {
          images = images || y.lo[2] < domain.lo[2] || y.hi[2] > domain.hi[2];
          if (neumann_z) {
            if (y.lo[2] < domain.lo[2]) {
              Box reflected = y;
              reflected.lo[2] = 2.*domain.lo[2]-std::min(y.hi[2],static_cast<double>(domain.lo[2]));
              reflected.hi[2] = 2.*domain.lo[2]-y.lo[2];
              integrate(reflected);
            }
            if (y.hi[2] > domain.hi[2]) {
              Box reflected = y;
              reflected.lo[2] = 2.*domain.hi[2]-y.hi[2];
              reflected.hi[2] = 2.*domain.hi[2]-std::max(y.lo[2],static_cast<double>(domain.hi[2]));
              integrate(reflected);
            }
          }
          y.lo[2] = std::max(y.lo[2],static_cast<double>(domain.lo[2]));
          y.hi[2] = std::min(y.hi[2],static_cast<double>(domain.hi[2]));
          if (y.hi[2] > y.lo[2]) integrate(y);
          continue;
        }
        const int zmin = static_cast<int>(std::floor((y.lo[2]-domain.lo[2])/length[2]+1.e-12));
        const int zmax = static_cast<int>(std::ceil((y.hi[2]-domain.lo[2])/length[2]-1.e-12))-1;
        for (int iz = zmin; iz <= zmax; ++iz) {
          Box z = y;
          z.lo[2] = std::max(y.lo[2],domain.lo[2]+iz*length[2])-iz*length[2];
          z.hi[2] = std::min(y.hi[2],domain.hi[2]+iz*length[2])-iz*length[2];
          images = images || iz != 0; integrate(z);
        }
      }
    }
    return std::move(out.v);
  }
};

MeshGeometry::MeshGeometry(std::vector<GravityPatch> patches,
    const CompositeDomain &domain, int rank, int nranks, bool neumann_z,
    bool coarse_reconstruction) : impl_(new Impl) {
  auto &m = *impl_;
  m.patches = std::move(patches); m.domain = domain;
  m.rank = rank; m.nranks = nranks; m.nd = domain.ndim;
  m.neumann_z = neumann_z;
  m.coarse_reconstruction = coarse_reconstruction;
  if (m.patches.empty() || nranks <= 0 || rank < 0 || rank >= nranks ||
      m.nd < 2 || m.nd > 3 || (domain.isolated_z && m.nd != 3) ||
      (neumann_z && !domain.isolated_z))
    throw std::invalid_argument("Gravity invalid patch geometry/domain/rank");
  if (!std::is_sorted(domain.c1_z_interfaces.begin(),domain.c1_z_interfaces.end()))
    throw std::invalid_argument("Gravity C1 planes must be sorted");
  for (size_t i = 0; i < domain.c1_z_interfaces.size(); ++i) {
    const double plane = domain.c1_z_interfaces[i];
    if (m.nd != 3 || !std::isfinite(plane) || plane <= domain.lo[2] ||
        plane >= domain.hi[2] || (i && plane == domain.c1_z_interfaces[i-1]))
      throw std::invalid_argument("Gravity invalid C1 reconstruction plane");
  }
  m.minwidth.fill(std::numeric_limits<double>::max());
  double volume = 0., expected = 1.;
  for (int d = 0; d < m.nd; ++d) {
    m.length[d] = domain.hi[d]-domain.lo[d];
    if (!(m.length[d] > 0.)) throw std::invalid_argument("Gravity invalid domain extent");
    expected *= m.length[d];
  }
  for (const auto &p : m.patches) {
    if (p.owner < 0 || p.owner >= nranks || (m.nd == 2 && p.cells[2] != 1))
      throw std::invalid_argument("Gravity invalid patch owner/inactive count");
    Impl::Count(p);
    double v = 1.;
    for (int d = 0; d < m.nd; ++d) {
      const double extent = p.hi[d]-p.lo[d];
      if (!(extent > 0.) || !std::isfinite(extent) || !std::isfinite(p.lo[d]) ||
          p.lo[d] < domain.lo[d]-m.length[d]*1.e-10 ||
          p.hi[d] > domain.hi[d]+m.length[d]*1.e-10)
        throw std::invalid_argument("Gravity invalid/out-of-domain patch box");
      m.minwidth[d] = std::min(m.minwidth[d],extent/p.cells[d]); v *= extent;
    }
    volume += v;
  }
  if (std::abs(volume-expected) > expected*1.e-5)
    throw std::invalid_argument("Gravity patches do not cover domain volume");
  m.shear = std::fmod(static_cast<double>(domain.shear_offset),m.length[1]);
  if (!std::isfinite(m.shear)) throw std::invalid_argument("Gravity nonfinite shear");
  if (m.shear < 0.) m.shear += m.length[1];
  if (!domain.shear_phase_updates && m.shear != 0.)
    throw std::invalid_argument("Gravity ordinary-periodic geometry requires zero shear");
  m.partition.assign(nranks+1,0); m.sample_partition.assign(nranks+1,0);
  m.sample_begin.resize(m.patches.size());
  for (size_t p = 0; p < m.patches.size(); ++p) {
    const int r = m.patches[p].owner+1;
    const auto count = Impl::Count(m.patches[p]), samples = m.Samples(p);
    if (m.partition[r] > std::numeric_limits<GlobalIndex>::max()-count ||
        m.sample_partition[r] > std::numeric_limits<GlobalIndex>::max()-samples)
      throw std::overflow_error("Gravity global row/sample count overflow");
    m.partition[r] += count; m.sample_partition[r] += samples;
  }
  for (int r = 0; r < nranks; ++r) {
    if (m.partition[r+1] > std::numeric_limits<GlobalIndex>::max()-m.partition[r] ||
        m.sample_partition[r+1] >
        std::numeric_limits<GlobalIndex>::max()-m.sample_partition[r])
      throw std::overflow_error("Gravity global row/sample partition overflow");
    m.partition[r+1] += m.partition[r]; m.sample_partition[r+1] += m.sample_partition[r];
  }
  if (m.partition.back() > std::numeric_limits<GlobalIndex>::max()-m.sample_partition.back())
    throw std::overflow_error("Gravity combined global cell/sample ID overflow");
  auto next = m.partition, sample_next = m.sample_partition;
  for (size_t p = 0; p < m.patches.size(); ++p) {
    const int owner = m.patches[p].owner;
    m.patches[p].row_begin = next[owner]; next[owner] += Impl::Count(m.patches[p]);
    m.sample_begin[p] = sample_next[owner]; sample_next[owner] += m.Samples(p);
  }
  if (m.partition[rank+1]-m.partition[rank] > std::numeric_limits<int>::max())
    throw std::overflow_error("Gravity owned rows exceed 32-bit local count");
  m.order.resize(m.patches.size()); std::iota(m.order.begin(),m.order.end(),0);
  m.row_order = m.order;
  std::sort(m.row_order.begin(),m.row_order.end(),[&](int a, int b) {
    return m.patches[a].row_begin < m.patches[b].row_begin;
  });
  m.BuildTree(0,m.order.size());
  for (int p : m.row_order) if (m.Samples(p) > 0) m.sample_order.push_back(p);
}

MeshGeometry::~MeshGeometry() = default;
MeshGeometry::MeshGeometry(MeshGeometry &&) noexcept = default;
MeshGeometry &MeshGeometry::operator=(MeshGeometry &&) noexcept = default;
const std::vector<GravityPatch> &MeshGeometry::Patches() const { return impl_->patches; }
const std::vector<GlobalIndex> &MeshGeometry::RowPartition() const { return impl_->partition; }
const std::vector<GlobalIndex> &MeshGeometry::SamplePartition() const { return impl_->sample_partition; }
GlobalIndex MeshGeometry::GlobalRows() const { return impl_->partition.back(); }
CompositeCell MeshGeometry::Cell(GlobalIndex row) const { return impl_->Cell(row); }

GlobalIndex MeshGeometry::Locate(const std::array<Real,3> &point) const {
  const auto &m = *impl_;
  Vec mapped{};
  for (int d = 0; d < m.nd; ++d) {
    if (!std::isfinite(point[d])) throw std::invalid_argument("Gravity nonfinite point");
    mapped[d] = point[d];
  }
  const double xwrap = std::floor((mapped[0]-m.domain.lo[0])/m.length[0]);
  mapped[0] -= xwrap*m.length[0]; mapped[1] -= xwrap*m.shear;
  for (int d = 1; d < m.nd; ++d) {
    if (d == 2 && m.domain.isolated_z) continue;
    mapped[d] -= std::floor((mapped[d]-m.domain.lo[d])/m.length[d])*m.length[d];
  }
  Box query{};
  for (int d = 0; d < m.nd; ++d) query.lo[d] = query.hi[d] = mapped[d];
  std::vector<int> candidates; m.Query(0,query,candidates);
  for (int p : candidates) {
    std::array<int,3> ijk{}; bool inside = true;
    for (int d = 0; d < m.nd; ++d) {
      const double h = (m.patches[p].hi[d]-m.patches[p].lo[d])/m.patches[p].cells[d];
      ijk[d] = static_cast<int>(std::floor((mapped[d]-m.patches[p].lo[d])/h));
      if (d == 2 && m.domain.isolated_z && mapped[d] == m.domain.hi[d] &&
          m.patches[p].hi[d] == m.domain.hi[d]) ijk[d] = m.patches[p].cells[d]-1;
      if (ijk[d] < 0 || ijk[d] >= m.patches[p].cells[d]) inside = false;
    }
    if (inside) return m.Id(p,ijk);
  }
  throw std::out_of_range("Gravity point is outside patch cells");
}

std::vector<GlobalIndex> MeshGeometry::QueryBox(const std::array<Real,3> &lo,
    const std::array<Real,3> &hi) const {
  Box box;
  for (int d = 0; d < impl_->nd; ++d) { box.lo[d] = lo[d]; box.hi[d] = hi[d]; }
  return impl_->QueryBox(box);
}

std::vector<std::pair<GlobalIndex,Real>> MeshGeometry::VirtualAverage(
    const std::array<Real,3> &lo, const std::array<Real,3> &hi, bool *images,
    bool *shear_images) const {
  Box box;
  for (int d = 0; d < impl_->nd; ++d) { box.lo[d] = lo[d]; box.hi[d] = hi[d]; }
  bool crossed = false, sheared = false;
  const auto row = impl_->Average(box,crossed,sheared);
  if (images) *images = crossed;
  if (shear_images) *shear_images = sheared;
  std::vector<std::pair<GlobalIndex,Real>> result;
  result.reserve(row.size());
  for (const auto &v : row) if (v.second != 0.) result.emplace_back(v.first,v.second);
  return result;
}

// <gravity>/test_gpu_assembly split build, step 1 (any thread): the host patch tables (cell
// centers and widths from Cell(), aligned same-shape face neighbors from the patch tree, the
// candidate lists of the irregular kernels) and the classification parameters.
void MeshGeometry::PrepareSplit(bool forces, const GpuSplitOptions &options) {
  auto &m = *impl_;
  const auto t0 = std::chrono::steady_clock::now();
  auto &x = m.split;
  x = Impl::Split{};
  x.forces = forces;
  x.components = forces ? 1+m.nd : 1;
  const GlobalIndex begin = m.partition[m.rank], end = m.partition[m.rank+1];
  const int rows = static_cast<int>(end-begin);
  auto &g = x.g;
  g.nd = m.nd; g.shear = m.domain.shear_phase_updates ? 1 : 0; g.n = m.partition.back();
  for (int d = 0; d < 3; ++d) {
    g.dlo[d] = m.domain.lo[d]; g.dhi[d] = m.domain.hi[d];
    g.length[d] = m.length[d]; g.minwidth[d] = m.minwidth[d];
  }
  // more C1 planes than the tables hold: every row stays on the host
  const bool supported = m.domain.c1_z_interfaces.size() <= static_cast<size_t>(gpu::kMaxPlanes);
  g.nplanes = supported ? static_cast<int>(m.domain.c1_z_interfaces.size()) : 0;
  for (int k = 0; k < g.nplanes; ++k) g.planes[k] = m.domain.c1_z_interfaces[k];
  const int np = static_cast<int>(m.patches.size());
  for (int p : m.row_order) if (m.patches[p].owner == m.rank) {
    x.owned.push_back(p); x.owned_first.push_back(static_cast<int>(m.patches[p].row_begin-begin));
  }
  GlobalIndex centers = 0;
  for (const auto &p : m.patches) centers += p.cells[0]+p.cells[1]+p.cells[2];
  if (centers > std::numeric_limits<int>::max()) throw std::overflow_error("Gravity device center table");
  x.width.assign(3*np,0.); x.cells.assign(3*np,0); x.coff.assign(3*np,0);
  x.center.assign(centers,0.); x.row_begin.assign(np,0);
  x.nb.assign(6*np,-1); x.wrap.assign(6*np,0);
  g.width = {x.width.data(),static_cast<size_t>(np)};
  g.cells = {x.cells.data(),static_cast<size_t>(np)};
  g.coff = {x.coff.data(),static_cast<size_t>(np)};
  g.center = {x.center.data(),x.center.size()};
  g.row_begin = {x.row_begin.data(),x.row_begin.size()};
  g.nb = {x.nb.data(),static_cast<size_t>(np)};
  g.wrap = {x.wrap.data(),static_cast<size_t>(np)};
  g.owned = {x.owned.data(),x.owned.size()};
  g.owned_first = {x.owned_first.data(),x.owned_first.size()};
  GlobalIndex offset = 0;
  for (int p = 0; p < np; ++p) {
    for (int d = 0; d < 3; ++d) {
      g.cells(p,d) = m.patches[p].cells[d]; g.coff(p,d) = static_cast<int>(offset);
      for (int i = 0; i < m.patches[p].cells[d]; ++i) {
        const auto c = m.Cell(p,{{i,i,i}});
        g.center(offset+i) = c.center[d];
        if (i == 0) g.width(p,d) = c.width[d];
      }
      offset += m.patches[p].cells[d];
    }
    g.row_begin(p) = m.patches[p].row_begin;
  }
  auto locate = [&](const Vec &point) {
    Box query; query.lo = point; query.hi = point;
    std::vector<int> candidates; m.Query(0,query,candidates);
    for (int q : candidates) {
      bool inside = true;
      for (int t = 0; t < m.nd; ++t)
        inside = inside && point[t] > m.patches[q].lo[t] && point[t] < m.patches[q].hi[t];
      if (inside) return q;
    }
    return -1;
  };
  for (int p = 0; p < np; ++p) for (int k = 0; k < 6; ++k) {
    const int d = k/2, s = k%2 ? 1 : -1;
    if (d >= m.nd) continue;
    const auto &P = m.patches[p];
    const double edge = s > 0 ? P.hi[d] : P.lo[d];
    const double wall = s > 0 ? m.domain.hi[d] : m.domain.lo[d];
    const bool boundary = std::abs(edge-wall) < m.minwidth[d]*1.e-5;
    if (boundary && (d == 0 || (d == 2 && m.domain.isolated_z))) { g.nb(p,k) = -2; continue; }
    Vec point{};
    for (int t = 0; t < m.nd; ++t) point[t] = .5*(P.lo[t]+P.hi[t]);
    point[d] = edge+s*.5*g.width(p,d);
    if (boundary) point[d] -= s*m.length[d];
    const int q = locate(point);
    if (q < 0) continue;
    const auto &Q = m.patches[q];
    bool same = true;
    for (int t = 0; t < m.nd; ++t) {
      const double tol = g.width(p,t)*1.e-9;
      same = same && Q.cells[t] == P.cells[t] &&
             std::abs(g.width(q,t)-g.width(p,t)) <= g.width(p,t)*1.e-12;
      if (t != d) same = same && std::abs(Q.lo[t]-P.lo[t]) <= tol;
      else if (s > 0) same = same && std::abs(Q.lo[t]-(boundary ? m.domain.lo[t] : P.hi[t])) <= tol;
      else same = same && std::abs(Q.hi[t]-(boundary ? m.domain.hi[t] : P.lo[t])) <= tol;
    }
    if (same) { g.nb(p,k) = q; g.wrap(p,k) = boundary ? 1 : 0; }
  }
  // The classification (ClassifySplit on the host, the same predicates on the device in
  // RunDevice).  Irregular rows (options.irregular) go to the irregular kernels unless they touch
  // the x boundary (phase rows) or lie within options.margin cell widths of it or of an
  // isolated-z wall (x images and wall samples stay on the host; the kernels would refuse them,
  // see gpu_irregular.hpp; margin 0 sends those rows to the kernels too, which then leave them to
  // the host fallback).  A level with fewer than options.min_rows owned rows stays on the host
  // (below that the kernels' latency exceeds the host's time for the level).
  const bool irregular = options.irregular;
  const double margin = std::max(0.,options.margin);
  const long long min_rows = options.min_rows;
  x.device_level = rows > 0 && rows >= min_rows && supported && !x.owned.empty();
  x.irregular_on = x.device_level && irregular;
  x.margin = margin;
  if (x.irregular_on) {
    // patch boxes, row order, and per patch face the candidate patches of the Faces() query of
    // any of its cells (an enlarged query of the whole face, in tree order)
    x.plo.assign(3*np,0.); x.phi.assign(3*np,0.);
    for (int p = 0; p < np; ++p) for (int d = 0; d < 3; ++d) {
      x.plo[3*p+d] = m.patches[p].lo[d]; x.phi[3*p+d] = m.patches[p].hi[d];
    }
    x.rorder = m.row_order;
    x.cand_off.assign(6*np+1,0);
    for (int p = 0; p < np; ++p) for (int k = 0; k < 6; ++k) {
      x.cand_off[6*p+k] = static_cast<int>(x.cand.size());
      const int d = k/2, side = k%2 ? 1 : -1;
      if (d >= m.nd) continue;
      const auto &P = m.patches[p];
      const double edge = side > 0 ? P.hi[d] : P.lo[d];
      const bool periodic = std::abs(edge-(side < 0 ? m.domain.lo[d] : m.domain.hi[d])) <
                            m.minwidth[d]*1.e-5;
      if (periodic && (d == 0 || (d == 2 && m.domain.isolated_z))) continue;
      const double image = periodic ? side*m.length[d] : 0.;
      Box box{};
      for (int t = 0; t < m.nd; ++t) {
        box.lo[t] = P.lo[t]-1.e-6*m.minwidth[t]; box.hi[t] = P.hi[t]+1.e-6*m.minwidth[t];
      }
      const double probe = edge-image+side*m.minwidth[d]*1.e-6;
      box.lo[d] = probe-1.e-7*m.minwidth[d]; box.hi[d] = probe+1.e-7*m.minwidth[d];
      std::vector<int> candidates; m.Query(0,box,candidates);
      x.cand.insert(x.cand.end(),candidates.begin(),candidates.end());
    }
    x.cand_off[6*np] = static_cast<int>(x.cand.size());
  }
  x.prepared = true;
  x.prepare_s = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}

// Split build, step 1b (any thread, after PrepareSplit, concurrently with RunDevice): the host
// classification of the owned rows (the device classifies them itself with the same predicates)
void MeshGeometry::ClassifySplit() {
  auto &m = *impl_;
  auto &x = m.split;
  if (!x.prepared) throw std::logic_error("Gravity ClassifySplit before PrepareSplit");
  const auto t0 = std::chrono::steady_clock::now();
  const auto &g = x.g;
  const bool forces = x.forces, irregular = x.irregular_on;
  const double margin = x.margin;
  const int rows = static_cast<int>(m.partition[m.rank+1]-m.partition[m.rank]);
  x.regular.assign(rows,-gpu::kOverflow);
  const int chunks = std::max(1,std::min(64*std::max(1,m.setup_threads),rows/4096+1));
  std::vector<std::array<long long,gpu::kReasons+1>> histogram(chunks);
  if (x.device_level) {
    ParallelChunks(m.setup_threads,chunks,[&](int c) {
      const int first = static_cast<int>(static_cast<long long>(rows)*c/chunks);
      const int last = static_cast<int>(static_cast<long long>(rows)*(c+1)/chunks);
      auto &h = histogram[c];
      h.fill(0);
      for (int row = first; row < last; ++row) {
        const auto ref = gpu::LocalRef(g,row);
        const int why = gpu::Regular(g,ref,forces);
        ++h[why >= 0 && why < gpu::kReasons ? why : gpu::kOverflow];
        if (why == gpu::kRegular) { x.regular[row] = 1; continue; }
        x.regular[row] = -why;
        if (!irregular || why == gpu::kRowTouchesX) continue;
        if (gpu::Far(g,gpu::GetCell(g,ref),margin,m.domain.isolated_z)) {
          x.regular[row] = 2; ++h[gpu::kReasons];
        }
      }
    });
  } else {
    histogram.assign(1,{}); histogram[0][gpu::kOverflow] = rows;
  }
  for (const auto &h : histogram) {
    for (int r = 0; r < gpu::kReasons; ++r) x.why[r] += h[r];
    x.irregular += h[gpu::kReasons];
  }
  if (x.irregular > 0)
    for (int row = 0; row < rows; ++row) if (x.regular[row] == 2) x.irregular_rows.push_back(row);
  x.classified = true;
  x.prepare_s += std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
}

// Split build, device step (the thread that owns the device, after PrepareSplit, concurrently with
// BuildHostRows): upload, the gpu_assembly.hpp kernels (counts, offsets, entries), download.
void MeshGeometry::RunDevice(int slot, const std::function<void()> &counted) {
  auto &m = *impl_;
  auto &x = m.split;
  if (!x.prepared) throw std::logic_error("Gravity RunDevice before PrepareSplit");
  if (!x.device_level) {
    x.no_device = true; x.counted = true;
    if (counted) counted();
    x.device_done = true;
    return;
  }
  using Clock = std::chrono::steady_clock;
  auto seconds = [](Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double>(b-a).count();
  };
  const auto t0 = Clock::now();
  std::ostringstream memory;
  auto mark = [&memory,&t0,&seconds](const char *step) {
    Kokkos::fence();
    memory << ' ' << step << '=' << seconds(t0,Clock::now()) << "s";
#if defined(KOKKOS_ENABLE_CUDA)
    std::size_t free = 0, total = 0;
    cudaMemGetInfo(&free,&total);
    memory << '/' << free/1073741824. << "GiB";
#endif
  };
  // (x.regular: the host classification may still be running)
  const int rows = static_cast<int>(m.partition[m.rank+1]-m.partition[m.rank]);
  const auto &h = x.g;
  gpu::Geometry g;
  g.nd = h.nd; g.shear = h.shear; g.nplanes = h.nplanes; g.n = h.n;
  for (int d = 0; d < 3; ++d) {
    g.dlo[d] = h.dlo[d]; g.dhi[d] = h.dhi[d]; g.length[d] = h.length[d]; g.minwidth[d] = h.minwidth[d];
  }
  for (int k = 0; k < gpu::kMaxPlanes; ++k) g.planes[k] = h.planes[k];
  const size_t np = h.width.n;
  auto upload = [](auto &view, const char *name, const auto *data, size_t n) {
    using V = std::remove_reference_t<decltype(view)>;
    view = V(std::string(name),n);
    typename V::HostMirror::const_type src(data,n);
    Kokkos::deep_copy(view,src);
  };
  upload(g.width,"sg_gpu_width",x.width.data(),np);
  upload(g.cells,"sg_gpu_cells",x.cells.data(),np);
  upload(g.coff,"sg_gpu_center_offset",x.coff.data(),np);
  upload(g.center,"sg_gpu_center",x.center.data(),x.center.size());
  upload(g.row_begin,"sg_gpu_row_begin",x.row_begin.data(),x.row_begin.size());
  upload(g.nb,"sg_gpu_neighbor",x.nb.data(),np);
  upload(g.wrap,"sg_gpu_wrap",x.wrap.data(),np);
  upload(g.owned,"sg_gpu_owned",x.owned.data(),x.owned.size());
  upload(g.owned_first,"sg_gpu_owned_first",x.owned_first.data(),x.owned_first.size());
  gpu::Irregular t;
  t.isolated_z = m.domain.isolated_z ? 1 : 0;
  gpu::DeviceFlags status("sg_gpu_status",x.irregular_on ? rows : 0);
  if (x.irregular_on) {
    upload(t.plo,"sg_gpu_patch_lo",x.plo.data(),np);
    upload(t.phi,"sg_gpu_patch_hi",x.phi.data(),np);
    upload(t.cand_off,"sg_gpu_candidate_offsets",x.cand_off.data(),x.cand_off.size());
    upload(t.cand,"sg_gpu_candidates",x.cand.data(),x.cand.size());
    upload(t.rorder,"sg_gpu_row_order",x.rorder.data(),x.rorder.size());
  }
  // the classification on the device (as ClassifySplit; compared in FinishSplitPrepare) and the
  // list of the irregular rows
  gpu::DeviceFlags regular(Kokkos::view_alloc(Kokkos::WithoutInitializing,"sg_gpu_regular"),rows);
  {
    const bool forces = x.forces, irregular = x.irregular_on, isolated_z = m.domain.isolated_z;
    const double margin = x.margin;
    Kokkos::parallel_for("sg_gpu_classify",Kokkos::RangePolicy<DevExeSpace>(0,rows),
        KOKKOS_LAMBDA(const int row) {
      const auto ref = gpu::LocalRef(g,row);
      const int why = gpu::Regular(g,ref,forces);
      int kind = why == gpu::kRegular ? 1 : -why;
      if (kind < 0 && irregular && why != gpu::kRowTouchesX &&
          gpu::Far(g,gpu::GetCell(g,ref),margin,isolated_z)) kind = 2;
      regular(row) = kind;
    });
    int listed = 0;
    if (irregular) {
      Kokkos::parallel_reduce("sg_gpu_count_listed",Kokkos::RangePolicy<DevExeSpace>(0,rows),
          KOKKOS_LAMBDA(const int row, int &n) { n += regular(row) == 2; },listed);
      t.rows = Kokkos::View<int*, DevMemSpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,"sg_gpu_irregular_rows"),listed);
      auto list = t.rows;
      Kokkos::parallel_scan("sg_gpu_list_irregular",Kokkos::RangePolicy<DevExeSpace>(0,rows),
          KOKKOS_LAMBDA(const int row, int &update, const bool final) {
        if (regular(row) != 2) return;
        if (final) list(update) = row;
        ++update;
      });
    }
    x.device_kind.resize(rows);
    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_kind(x.device_kind.data(),rows);
    Kokkos::deep_copy(h_kind,regular);
  }
  const bool listed_rows = t.rows.extent(0) > 0;
  mark("classify");
  const auto t1 = Clock::now();
  std::array<gpu::DeviceIndex,4> count, col;
  std::array<Kokkos::View<double*, DevMemSpace>,4> val;
  for (int c = 0; c < 4; ++c) count[c] = gpu::DeviceIndex("sg_gpu_offsets",c < x.components ? rows+1 : 1);
  if (rows > 0 && !x.owned.empty()) {
    Kokkos::parallel_for("sg_gpu_count_rows",Kokkos::RangePolicy<DevExeSpace>(0,rows),
        gpu::CountRows{g,regular,x.components,count[0],count[1],count[2],count[3]});
  }
  mark("count");
  gpu::Quads fits;
  std::array<double,4> irregular_s{};
  if (listed_rows) {
    fits = gpu::IrregularCount(g,t,x.forces,x.components,count[0],count[1],count[2],count[3],
                               status,&irregular_s);
    mark("count_irregular");
    memory << "(marks=" << irregular_s[0] << "s fit_counts=" << irregular_s[1] << "s fit_fill="
           << irregular_s[2] << "s row_counts=" << irregular_s[3] << "s fits=" << fits.count
           << " samples=" << fits.samples << ")";
  }
  std::array<GlobalIndex,4> total{};
  for (int c = 0; c < x.components; ++c) {
    Kokkos::parallel_scan("sg_gpu_offsets",Kokkos::RangePolicy<DevExeSpace>(0,rows+1),
                          gpu::ScanCounts{count[c]},total[c]);
  }
  // a row the host left to the device that overflowed the fixed row buffers (impossible for
  // the regular stencils: A has 2 nd+1 entries, an acceleration row at most 17 in 3-D)
  int lost = 0;
  Kokkos::parallel_reduce("sg_gpu_lost_rows",Kokkos::RangePolicy<DevExeSpace>(0,rows),
      KOKKOS_LAMBDA(const int i, int &n) { n += regular(i) == -gpu::kOverflow; },lost);
  if (lost > 0) throw std::logic_error("Gravity device rows overflowed the row buffers");
  // the offsets and statuses now (page-locked buffers of this slot): the joining thread sizes the
  // output while the entries are filled
  auto &pool = Pinned();
  if (static_cast<int>(pool.offset.size()) <= slot) {
    pool.offset.resize(slot+1); pool.col.resize(slot+1); pool.val.resize(slot+1);
  }
  for (int c = 0; c < 4; ++c) {
    x.offset[c] = Take(pool.offset[slot][c],count[c].extent(0),"sg_gpu_pinned_offsets");
    Kokkos::deep_copy(x.offset[c],count[c]);
  }
  if (x.irregular_on) {
    x.status.resize(rows);
    Kokkos::View<int*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> h_status(x.status.data(),rows);
    Kokkos::deep_copy(h_status,status);
  }
  mark("offsets");
  x.counted = true;
  if (counted) counted();
  for (int c = 0; c < 4; ++c) {
    const GlobalIndex n = c < x.components ? total[c] : 0;
    col[c] = gpu::DeviceIndex(Kokkos::view_alloc(Kokkos::WithoutInitializing,"sg_gpu_columns"),n);
    val[c] = Kokkos::View<double*, DevMemSpace>(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,"sg_gpu_values"),n);
  }
  if (rows > 0 && !x.owned.empty()) {
    Kokkos::parallel_for("sg_gpu_fill_rows",Kokkos::RangePolicy<DevExeSpace>(0,rows),
        gpu::FillRows{g,regular,x.components,count[0],count[1],count[2],count[3],
                      col[0],col[1],col[2],col[3],val[0],val[1],val[2],val[3]});
  }
  mark("fill");
  if (listed_rows) {
    gpu::IrregularFill(g,t,fits,x.forces,x.components,count,col,val,status);
    mark("fill_irregular");
  }
  const auto t2 = Clock::now();
  for (int c = 0; c < 4; ++c) {
    x.colidx[c] = Take(pool.col[slot][c],col[c].extent(0),"sg_gpu_pinned_columns");
    x.values[c] = Take(pool.val[slot][c],val[c].extent(0),"sg_gpu_pinned_values");
    Kokkos::deep_copy(x.colidx[c],col[c]);
    Kokkos::deep_copy(x.values[c],val[c]);
  }
  mark("download");
  const auto t3 = Clock::now();
  x.upload_s = seconds(t0,t1); x.kernels_s = seconds(t1,t2); x.download_s = seconds(t2,t3);
  x.memory = memory.str();
  x.device_done = true;
}

LocalCompositeSystem MeshGeometry::BuildSystem(bool forces) {
  return BuildSystemRows(forces,false);
}

// Split build, host step (after PrepareSplit, any thread, concurrently with RunDevice): the rows
// the device does not assemble; the device rows are empty placeholders until FinishSplit.
void MeshGeometry::BuildHostRows() {
  auto &x = impl_->split;
  if (!x.prepared) throw std::logic_error("Gravity BuildHostRows before PrepareSplit");
  BuildSystemRows(x.forces,true);
  x.host_done = true;
}

// Split build, last step (after BuildHostRows and RunDevice): the device rows into their
// placeholders, as a serial join (no floating-point arithmetic).
LocalCompositeSystem MeshGeometry::FinishSplit() {
  FinishSplitPrepare();
  return FinishSplitCopy();
}

void MeshGeometry::FinishSplitPrepare() {
  auto &m = *impl_;
  auto &x = m.split;
  if (!x.host_done || !x.counted) throw std::logic_error("Gravity FinishSplit before both passes");
  if (!x.no_device && x.device_kind != x.regular)
    throw std::logic_error("Gravity device and host row classifications differ");
  const auto t0 = std::chrono::steady_clock::now();
  x.join_t0 = std::chrono::duration<double>(t0.time_since_epoch()).count();
  auto &out = x.out;
  const GlobalIndex begin = m.partition[m.rank];
  const bool forces = x.forces;
  auto component = [](auto &system, int k) -> LocalCsr & {
    if (k == 0) return system.matrix;
    if (k == 1) return system.boundary_rhs;
    if (k < 5) return system.acceleration[k-2];
    return system.boundary_acceleration[k-5];
  };
  // the device component of system component k (-1: none)
  auto device_component = [&](int k) {
    if (k == 0) return 0;
    if (k >= 2 && k < 5 && k-2 < m.nd && forces) return k-1;
    return -1;
  };
  const int components = forces ? 8 : 1;
  const int chunks = static_cast<int>(x.parts.size());
  const int setup_workers = std::max(1,m.setup_threads);
  // irregular rows the device left to the host (status <= 0): assembled here, as in the host pass
  // (fresh caches per chunk; the coefficients do not depend on them)
  const GlobalIndex n = m.partition.back();
  auto &failed = x.failed;
  failed.clear();
  for (int i : x.irregular_rows) if (x.status[i] <= 0) failed.push_back(i);
  x.fallback = static_cast<long long>(failed.size());
  for (int i : failed) ++x.failures[std::min(7,-x.status[i])];
  using Fallback = Impl::Split::Fallback;
  auto &fallback = x.fallback_rows;
  fallback.assign(failed.size(),Fallback{});
  if (!failed.empty()) {
    const int fchunks = std::min(static_cast<int>(failed.size()),8*setup_workers);
    std::vector<std::vector<GlobalIndex>> interface_ids(fchunks), x_image_ids(fchunks);
    ParallelChunks(setup_workers,fchunks,[&](int c) {
      QuadraticCache quadratics; ReconstructionFaces faces; RowWorkspace work;
      const size_t first = failed.size()*c/fchunks, last = failed.size()*(c+1)/fchunks;
      for (size_t i = first; i < last; ++i) {
        const GlobalIndex row = begin+failed[i];
        const auto ci = m.Cell(row); double volume = 1.;
        for (int d = 0; d < m.nd; ++d) volume *= ci.width[d];
        bool is_interface = false, phase_dependent = false;
        m.AssembleRow(row,ci,volume,forces,work,quadratics,faces,
                      forces ? &is_interface : nullptr,&phase_dependent);
        auto &f = fallback[i];
        f.interface = is_interface; f.phase = phase_dependent;
        Append(f.c[0],work.a,n,false);
        if (forces) Append(f.c[1],work.a,n,true,-1.);
        for (int d = 0; forces && d < 3; ++d) {
          Append(f.c[2+d],work.g[d],n,false);
          Append(f.c[5+d],work.g[d],n,true);
        }
      }
      for (const auto &q : quadratics) {
        if (q.second.interface_fallback) interface_ids[c].push_back(q.first);
        if (q.second.x_image_fallback) x_image_ids[c].push_back(q.first);
      }
    });
    auto merge = [](std::vector<GlobalIndex> &all, std::vector<std::vector<GlobalIndex>> &lists) {
      for (auto &list : lists) all.insert(all.end(),list.begin(),list.end());
      std::sort(all.begin(),all.end());
      all.erase(std::unique(all.begin(),all.end()),all.end());
      return all.size();
    };
    out.quadratic_interface_fallbacks = merge(x.interface_ids,interface_ids);
    out.quadratic_x_image_fallbacks = merge(x.x_image_ids,x_image_ids);
    bool phase = false;
    for (size_t i = 0; i < failed.size(); ++i) if (fallback[i].phase) {
      m.phase_rows.push_back(failed[i]); phase = true;
    }
    if (phase) std::sort(m.phase_rows.begin(),m.phase_rows.end());
  }
  auto &row_offset = x.row_offset;
  auto &entry_offset = x.entry_offset;
  row_offset.assign(chunks+1,0);
  entry_offset.assign(components,std::vector<size_t>(chunks+1,0));
  for (int c = 0; c < chunks; ++c) {
    auto &part = x.parts[c];
    const size_t rows = part.matrix.LocalRows();
    row_offset[c+1] = row_offset[c]+rows;
    const auto f0 = std::lower_bound(failed.begin(),failed.end(),static_cast<int>(row_offset[c]));
    const auto f1 = std::lower_bound(failed.begin(),failed.end(),static_cast<int>(row_offset[c+1]));
    for (int k = 0; k < components; ++k) {
      if (component(part,k).LocalRows() != rows)
        throw std::logic_error("Gravity setup part components disagree on rows");
      const int dc = device_component(k);
      size_t device = dc < 0 || x.no_device ? 0 : static_cast<size_t>(
          x.offset[dc](row_offset[c+1])-x.offset[dc](row_offset[c]));
      for (auto f = f0; f != f1; ++f) device += fallback[f-failed.begin()].c[k].values.size();
      entry_offset[k][c+1] = entry_offset[k][c]+component(part,k).values.size()+device;
    }
  }
  for (int k = 0; k < components; ++k)
    if (entry_offset[k][chunks] > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Gravity local CSR exceeds 32-bit offsets");
  const size_t total = row_offset[chunks];
  if (total != x.regular.size()) throw std::logic_error("Gravity split build lost rows");
  const int workers = std::max(1,std::min(setup_workers,chunks));
  ParallelChunks(workers,3*components+(forces ? 3 : 0),[&](int t) {
    if (t < 3*components) {
      auto &a = component(out,t/3);
      if (t%3 == 0) { a.rowptr.resize(total+1); a.rowptr[0] = 0; }
      else if (t%3 == 1) a.colidx.resize(entry_offset[t/3][chunks]);
      else a.values.resize(entry_offset[t/3][chunks]);
    } else if (t == 3*components) {
      out.volume.resize(total);
    } else if (t == 3*components+1) {
      out.cells.resize(total);
    } else {
      out.interface.resize(total);
    }
  });
  x.join_prepared = true;
}

LocalCompositeSystem MeshGeometry::FinishSplitCopy() {
  auto &m = *impl_;
  auto &x = m.split;
  if (!x.join_prepared || !x.device_done) throw std::logic_error("Gravity FinishSplit before both passes");
  auto out = std::move(x.out);
  const bool forces = x.forces;
  auto component = [](auto &system, int k) -> LocalCsr & {
    if (k == 0) return system.matrix;
    if (k == 1) return system.boundary_rhs;
    if (k < 5) return system.acceleration[k-2];
    return system.boundary_acceleration[k-5];
  };
  auto device_component = [&](int k) {
    if (k == 0) return 0;
    if (k >= 2 && k < 5 && k-2 < m.nd && forces) return k-1;
    return -1;
  };
  const int components = forces ? 8 : 1;
  const int chunks = static_cast<int>(x.parts.size());
  const int workers = std::max(1,std::min(std::max(1,m.setup_threads),chunks));
  const auto &failed = x.failed;
  const auto &fallback = x.fallback_rows;
  const auto &row_offset = x.row_offset;
  const auto &entry_offset = x.entry_offset;
  using Fallback = Impl::Split::Fallback;
  auto fallback_of = [&](size_t local) -> const Fallback & {
    const auto it = std::lower_bound(failed.begin(),failed.end(),static_cast<int>(local));
    return fallback[it-failed.begin()];
  };
  ParallelChunks(workers,chunks,[&](int c) {
    auto &part = x.parts[c];
    for (int k = 0; k < components; ++k) {
      auto &source = component(part,k);
      auto &destination = component(out,k);
      const int dc = device_component(k);
      size_t at = entry_offset[k][c];
      for (size_t i = 0; i+1 < source.rowptr.size(); ++i) {
        const size_t local = row_offset[c]+i;
        const int kind = x.regular[local];
        if (kind == 2 && x.status[local] <= 0) {
          const auto &f = fallback_of(local).c[k];
          std::copy(f.colidx.begin(),f.colidx.end(),destination.colidx.begin()+at);
          std::copy(f.values.begin(),f.values.end(),destination.values.begin()+at);
          at += f.values.size();
        } else if (kind > 0) {
          if (dc >= 0) {
            const GlobalIndex k0 = x.offset[dc](local), k1 = x.offset[dc](local+1);
            std::copy(x.colidx[dc].data()+k0,x.colidx[dc].data()+k1,destination.colidx.begin()+at);
            std::copy(x.values[dc].data()+k0,x.values[dc].data()+k1,destination.values.begin()+at);
            at += static_cast<size_t>(k1-k0);
          }
        } else {
          const int s0 = source.rowptr[i], s1 = source.rowptr[i+1];
          std::copy(source.colidx.begin()+s0,source.colidx.begin()+s1,destination.colidx.begin()+at);
          std::copy(source.values.begin()+s0,source.values.begin()+s1,destination.values.begin()+at);
          at += static_cast<size_t>(s1-s0);
        }
        destination.rowptr[local+1] = static_cast<int>(at);
      }
    }
    if (forces) {
      std::copy(part.volume.begin(),part.volume.end(),out.volume.begin()+row_offset[c]);
      std::copy(part.cells.begin(),part.cells.end(),out.cells.begin()+row_offset[c]);
      std::copy(part.interface.begin(),part.interface.end(),out.interface.begin()+row_offset[c]);
      for (size_t local = row_offset[c]; local < row_offset[c+1]; ++local)
        if (x.regular[local] == 2)
          out.interface[local] = x.status[local] > 0 ? x.status[local] == 2
                                                     : fallback_of(local).interface;
    }
    part = Impl::Part{};
  });
  x.parts.clear(); x.failed.clear(); x.fallback_rows.clear();
  for (int c = 0; c < 4; ++c) {
    x.offset[c] = {}; x.colidx[c] = {}; x.values[c] = {};
  }
  x.join_s = std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count()-x.join_t0;
  long long device_rows = x.why[0]+x.irregular-x.fallback;
  m.device_stats = {{static_cast<double>(device_rows),static_cast<double>(x.regular.size()),
                     x.prepare_s,x.upload_s+x.kernels_s,x.download_s,x.join_s}};
  if (SetupTimesOn()) {
    const auto &w = x.why;
    std::cout << "# SG_GPU_ASSEMBLY_ROWS rank=" << m.rank << " forces=" << forces
              << " rows=" << x.regular.size() << " device=" << w[0]
              << " irregular_device=" << x.irregular-x.fallback << " irregular_to_host=" << x.fallback
              << " (x/wall/geometry/samples/rank/duplicate/buffer=" << x.failures[1] << '/'
              << x.failures[2] << '/' << x.failures[3] << '/' << x.failures[4] << '/'
              << x.failures[5] << '/' << x.failures[6] << '/' << x.failures[7] << ")"
              << " first non-regular reason (all non-regular rows):"
              << " touches_x=" << w[1] << " near_c1_plane=" << w[2]
              << " neighbor_touches_x=" << w[3] << " level_jump=" << w[4]
              << " domain_wall=" << w[5] << " face_geometry=" << w[6] << " offset=" << w[7]
              << " c1_face=" << w[8] << " width=" << w[9] << " self_face=" << w[10]
              << " unsupported=" << w[11] << " | prepare_s=" << x.prepare_s
              << " upload_s=" << x.upload_s << " kernels_s=" << x.kernels_s
              << " download_s=" << x.download_s << " join_s=" << x.join_s
              << " host_thread_s by reason:";
    for (int k = 1; k < gpu::kReasons; ++k) std::cout << ' ' << k << '=' << m.reason_seconds[k];
    std::cout << " device:" << x.memory << std::endl;
  }
  x.prepared = x.host_done = x.device_done = x.counted = x.join_prepared = false;
  return out;
}

// (test_gpu_assembly = 2) the host system of the same geometry, compared bitwise with a split
// build's result (every component, volume/cells/interface/phase rows/boundary/fallbacks).
long long MeshGeometry::CheckSplit(const LocalCompositeSystem &split, bool forces, std::string *first_difference) {
  auto &m = *impl_;
  const auto split_phase_rows = m.phase_rows;
  m.phase_rows_ready = false;
  const auto reference = BuildSystemRows(forces,false);
  long long differ = 0;
  std::string first;
  auto compare = [&](const LocalCsr &a, const LocalCsr &b, const char *name) {
    if (a.global_rows != b.global_rows || a.global_cols != b.global_cols ||
        a.row_begin != b.row_begin || a.rowptr.size() != b.rowptr.size()) {
      ++differ; if (first.empty()) first = std::string(name)+" shape"; return;
    }
    for (size_t i = 0; i+1 < a.rowptr.size(); ++i) {
      const int a0 = a.rowptr[i], a1 = a.rowptr[i+1], b0 = b.rowptr[i], b1 = b.rowptr[i+1];
      bool same = a1-a0 == b1-b0;
      if (same && a1 > a0) {
        same = std::memcmp(a.colidx.data()+a0,b.colidx.data()+b0,(a1-a0)*sizeof(GlobalIndex)) == 0
            && std::memcmp(a.values.data()+a0,b.values.data()+b0,(a1-a0)*sizeof(Real)) == 0;
      }
      if (same) continue;
      ++differ;
      if (first.empty()) {
        std::ostringstream o;
        o << name << " local row " << i << " (global " << a.row_begin+GlobalIndex(i) << ") host";
        o.precision(17);
        for (int k = a0; k < a1; ++k) o << ' ' << a.colidx[k] << ':' << a.values[k];
        o << " device";
        for (int k = b0; k < b1; ++k) o << ' ' << b.colidx[k] << ':' << b.values[k];
        first = o.str();
      }
    }
  };
  compare(reference.matrix,split.matrix,"A");
  compare(reference.boundary_rhs,split.boundary_rhs,"boundary_rhs");
  for (int d = 0; d < 3; ++d) {
    compare(reference.acceleration[d],split.acceleration[d],"acceleration");
    compare(reference.boundary_acceleration[d],split.boundary_acceleration[d],"boundary_acceleration");
  }
  auto same_bytes = [](const auto &a, const auto &b) {
    return a.size() == b.size() && (a.empty() ||
        std::memcmp(a.data(),b.data(),a.size()*sizeof(a[0])) == 0);
  };
  const bool others = same_bytes(reference.volume,split.volume) && same_bytes(reference.cells,split.cells)
      && same_bytes(reference.interface,split.interface) && same_bytes(split_phase_rows,m.phase_rows)
      && same_bytes(reference.boundary_cells,split.boundary_cells)
      && same_bytes(reference.boundary_area,split.boundary_area)
      && same_bytes(reference.boundary_width,split.boundary_width)
      && reference.boundary_samples.size() == split.boundary_samples.size()
      && reference.row_partition == split.row_partition
      && reference.sample_partition == split.sample_partition
      && reference.neumann_z == split.neumann_z
      && reference.quadratic_interface_fallbacks == split.quadratic_interface_fallbacks
      && reference.quadratic_x_image_fallbacks == split.quadratic_x_image_fallbacks;
  if (!others) { ++differ; if (first.empty()) first = "volume/cells/interface/phase rows/boundary/fallbacks"; }
  if (first_difference) *first_difference = first;
  return differ;
}

LocalCompositeSystem MeshGeometry::BuildSystemRows(bool forces, bool split) {
  auto &m = *impl_;
  using BuildClock = std::chrono::steady_clock;
  const auto build_t0 = BuildClock::now();
  m.build_join_s = 0;
  // Full setup owns its scratch independently of retained phase-worker caches.
  // In particular, a reused object must never inherit gradients from an earlier
  // full build or an interrupted assembly at a different phase.
  m.gradients.clear();
  m.quadratics.clear();
  m.reconstruction_faces.Clear();
  LocalCompositeSystem out;
  out.neumann_z = m.neumann_z;
  out.row_partition = m.partition; out.sample_partition = m.sample_partition;
  const GlobalIndex begin = m.partition[m.rank], end = m.partition[m.rank+1];
  const GlobalIndex n = m.partition.back(), ns = m.sample_partition.back();
  auto init = [&](LocalCsr &a, GlobalIndex columns) {
    a.global_rows = n; a.global_cols = columns; a.row_begin = begin;
    a.rowptr.reserve(static_cast<size_t>(end-begin)+1);
  };
  init(out.matrix,n);
  if (forces) init(out.boundary_rhs,ns);
  const size_t local_rows = end-begin;
  auto reserve_entries = [&](LocalCsr &a, size_t per_row) {
    if (local_rows <= static_cast<size_t>(std::numeric_limits<int>::max())/per_row) {
      a.colidx.reserve(local_rows*per_row); a.values.reserve(local_rows*per_row);
    }
  };
  reserve_entries(out.matrix,2*m.nd+1);
  for (int d = 0; forces && d < 3; ++d) {
    init(out.acceleration[d],n); init(out.boundary_acceleration[d],ns);
    if (d < m.nd) reserve_entries(out.acceleration[d],2);
  }
  if (forces) {
    out.volume.reserve(end-begin); out.cells.reserve(end-begin); out.interface.reserve(end-begin);
  }
  // Reuse short row storage while visiting owned rows; no complete global cell
  // list, matrix, sample array, or face list is constructed on this rank.
  const bool discover_phase_rows = !m.phase_rows_ready;
  if (discover_phase_rows) m.phase_rows.clear();
  // One worker's consecutive rows. Every row is the same pure function of the geometry;
  // the caches only memoize it, so private caches give the serial coefficients.
  using Part = Impl::Part;
  const bool row_profile = SetupTimesOn();
  // (split build, <gravity>/test_gpu_assembly) rows left to the device: empty placeholders here
  const int *skip = split ? m.split.regular.data() : nullptr;
  auto assemble = [&](GlobalIndex first, GlobalIndex last, Part &part,
                      QuadraticCache &quadratics, ReconstructionFaces &faces) {
    RowWorkspace work;
    for (GlobalIndex row = first; row < last; ++row) {
      const auto row_t0 = row_profile ? BuildClock::now() : BuildClock::time_point{};
      const auto ci = m.Cell(row); double volume = 1.;
      for (int d = 0; d < m.nd; ++d) volume *= ci.width[d];
      if (forces) {
        part.volume.push_back(volume); part.cells.push_back(ci); part.interface.push_back(0);
      }
      if (skip && skip[row-begin] > 0) {
        // a device row: not an interface, not a phase row, no exterior columns
        auto empty = [](LocalCsr &a) { a.rowptr.push_back(static_cast<int>(a.values.size())); };
        empty(part.matrix);
        if (forces) {
          empty(part.boundary_rhs);
          for (int d = 0; d < 3; ++d) { empty(part.acceleration[d]); empty(part.boundary_acceleration[d]); }
        }
        continue;
      }
      bool is_interface = false, phase_dependent = false;
      m.AssembleRow(row,ci,volume,forces,work,quadratics,faces,
                    forces ? &is_interface : nullptr,
                    discover_phase_rows ? &phase_dependent : nullptr);
      if (forces) part.interface.back() = is_interface;
      if (discover_phase_rows && phase_dependent)
        part.phase_rows.push_back(static_cast<int>(row-begin));
      Append(part.matrix,work.a,n,false);
      if (forces) Append(part.boundary_rhs,work.a,n,true,-1.);
      for (int d = 0; forces && d < 3; ++d) {
        Append(part.acceleration[d],work.g[d],n,false);
        Append(part.boundary_acceleration[d],work.g[d],n,true);
      }
      if (row_profile) {
        const double dt = std::chrono::duration<double>(BuildClock::now()-row_t0).count();
        const int p = m.Patch(row); const auto ijk = m.Indices(p,row);
        bool inner = true;
        for (int d = 0; d < m.nd; ++d)
          inner = inner && ijk[d] >= 2 && ijk[d]+2 < m.patches[p].cells[d];
        if (inner) { part.t_inner += dt; ++part.n_inner; } else { part.t_outer += dt; ++part.n_outer; }
        if (skip) {
          const int why = -skip[row-begin];
          if (why > 0 && why < gpu::kReasons) part.t_reason[why] += dt;
        }
      }
    }
  };
  // The full setup runs on SetupThreads() workers; the rows are cut into
  // consecutive chunks (>= 512 rows, up to 16 per worker) taken in any order and joined
  // in row order, which balances the much more expensive interface rows.
  const int workers = static_cast<int>(std::max<GlobalIndex>(1,std::min<GlobalIndex>(
      SetupThreads(),(end-begin)/4096)));
  std::size_t interface_fallbacks = 0, x_image_fallbacks = 0;  // parallel path only
  if (workers == 1 && !split) {
    // Serial: write straight into the output (identical to the original loop).
    Part part;
    part.matrix = std::move(out.matrix); part.boundary_rhs = std::move(out.boundary_rhs);
    for (int d = 0; d < 3; ++d) {
      part.acceleration[d] = std::move(out.acceleration[d]);
      part.boundary_acceleration[d] = std::move(out.boundary_acceleration[d]);
    }
    part.volume = std::move(out.volume); part.cells = std::move(out.cells);
    part.interface = std::move(out.interface);
    assemble(begin,end,part,m.quadratics,m.reconstruction_faces);
    out.matrix = std::move(part.matrix); out.boundary_rhs = std::move(part.boundary_rhs);
    for (int d = 0; d < 3; ++d) {
      out.acceleration[d] = std::move(part.acceleration[d]);
      out.boundary_acceleration[d] = std::move(part.boundary_acceleration[d]);
    }
    out.volume = std::move(part.volume); out.cells = std::move(part.cells);
    out.interface = std::move(part.interface);
    if (discover_phase_rows) m.phase_rows = std::move(part.phase_rows);
    m.build_rows_s = std::chrono::duration<double>(BuildClock::now()-build_t0).count();
    m.build_class = {{part.t_inner,part.t_outer,double(part.n_inner),double(part.n_outer)}};
    m.reason_seconds = part.t_reason;
  } else {
    const GlobalIndex rows = end-begin;
    const int chunks = static_cast<int>(std::max<GlobalIndex>(workers,
        std::min<GlobalIndex>(16*static_cast<GlobalIndex>(workers),rows/512)));
    std::vector<Part> parts(chunks);
    std::vector<QuadraticCache> quadratics(workers);
    std::vector<ReconstructionFaces> faces(workers);
    std::vector<std::exception_ptr> errors(workers);
    std::vector<std::vector<GlobalIndex>> interface_ids(workers), x_image_ids(workers);
    std::atomic<int> next{0};
    // the storage of a chunk reserved as the serial output's (fewer reallocations)
    auto reserve_part = [&](Part &part, GlobalIndex part_rows) {
      const size_t r = static_cast<size_t>(part_rows);
      part.matrix.rowptr.reserve(r+1);
      part.matrix.colidx.reserve(r*(2*m.nd+1)); part.matrix.values.reserve(r*(2*m.nd+1));
      if (!forces) return;
      part.boundary_rhs.rowptr.reserve(r+1);
      for (int d = 0; d < 3; ++d) {
        part.acceleration[d].rowptr.reserve(r+1);
        part.boundary_acceleration[d].rowptr.reserve(r+1);
        if (d < m.nd) {
          part.acceleration[d].colidx.reserve(2*r); part.acceleration[d].values.reserve(2*r);
        }
      }
      part.volume.reserve(r); part.cells.reserve(r); part.interface.reserve(r);
    };
    auto run = [&](int w) {
      try {
        for (int c = next++; c < chunks; c = next++) {
          const GlobalIndex first = begin+rows*c/chunks, last = begin+rows*(c+1)/chunks;
          reserve_part(parts[c],last-first);
          assemble(first,last,parts[c],quadratics[w],faces[w]);
        }
        // Keep only the ids of fallback fits and release this worker's caches here.
        for (const auto &q : quadratics[w]) {
          if (q.second.interface_fallback) interface_ids[w].push_back(q.first);
          if (q.second.x_image_fallback) x_image_ids[w].push_back(q.first);
        }
        quadratics[w] = QuadraticCache{}; faces[w] = ReconstructionFaces{};
      } catch (...) { errors[w] = std::current_exception(); next = chunks; }
    };
    ThreadWorkers().Run(workers,run);
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
    const auto build_t1 = BuildClock::now();
    m.build_rows_s = std::chrono::duration<double>(build_t1-build_t0).count();
    // Concatenation in row order without floating-point arithmetic, as a serial join:
    // exact-size outputs (each vector resized by one task), then every part is copied
    // into its slot and released by a worker. The serial join of all parts and the merge
    // of the fit caches took most of the fine setup.
    auto component = [](auto &system, int k) -> LocalCsr & {
      if (k == 0) return system.matrix;
      if (k == 1) return system.boundary_rhs;
      if (k < 5) return system.acceleration[k-2];
      return system.boundary_acceleration[k-5];
    };
    const int components = forces ? 8 : 1;
    std::vector<size_t> row_offset(chunks+1,0);
    std::vector<std::vector<size_t>> entry_offset(components,std::vector<size_t>(chunks+1,0));
    m.build_class = {{0.,0.,0.,0.}};
    m.reason_seconds = {};
    for (int c = 0; c < chunks; ++c) {
      auto &part = parts[c];
      m.build_class[0] += part.t_inner; m.build_class[1] += part.t_outer;
      for (int k = 0; k < gpu::kReasons; ++k) m.reason_seconds[k] += part.t_reason[k];
      m.build_class[2] += part.n_inner; m.build_class[3] += part.n_outer;
      if (split) {   // the join waits for the device rows (FinishSplit)
        if (discover_phase_rows)
          m.phase_rows.insert(m.phase_rows.end(),part.phase_rows.begin(),part.phase_rows.end());
        continue;
      }
      row_offset[c+1] = row_offset[c]+part.matrix.LocalRows();
      for (int k = 0; k < components; ++k) {
        if (component(part,k).LocalRows() != part.matrix.LocalRows())
          throw std::logic_error("Gravity setup part components disagree on rows");
        entry_offset[k][c+1] = entry_offset[k][c]+component(part,k).values.size();
      }
      if (discover_phase_rows)
        m.phase_rows.insert(m.phase_rows.end(),part.phase_rows.begin(),part.phase_rows.end());
    }
    for (int k = 0; k < components; ++k)
      if (entry_offset[k][chunks] > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::overflow_error("Gravity local CSR exceeds 32-bit offsets");
    if (split) m.split.parts = std::move(parts);
    const size_t total = split ? 0 : row_offset[chunks];
    auto parallel = [&](int tasks, const std::function<void(int)> &task) {
      const int count = std::max(1,std::min(workers,tasks));
      std::vector<std::exception_ptr> failures(count);
      std::atomic<int> claim{0};
      auto work = [&](int w) {
        try { for (int t = claim++; t < tasks; t = claim++) task(t); }
        catch (...) { failures[w] = std::current_exception(); claim = tasks; }
      };
      ThreadWorkers().Run(count,work);
      for (const auto &failure : failures) if (failure) std::rethrow_exception(failure);
    };
    if (!split) parallel(3*components+(forces ? 3 : 0),[&](int t) {
      if (t < 3*components) {
        auto &a = component(out,t/3);
        if (t%3 == 0) a.rowptr.resize(total+1);
        else if (t%3 == 1) a.colidx.resize(entry_offset[t/3][chunks]);
        else a.values.resize(entry_offset[t/3][chunks]);
      } else if (t == 3*components) {
        out.volume.resize(total);
      } else if (t == 3*components+1) {
        out.cells.resize(total);
      } else {
        out.interface.resize(total);
      }
    });
    if (!split) parallel(chunks,[&](int c) {
      auto &part = parts[c];
      for (int k = 0; k < components; ++k) {
        auto &source = component(part,k);
        auto &destination = component(out,k);
        const int offset = static_cast<int>(entry_offset[k][c]);
        for (size_t i = 1; i < source.rowptr.size(); ++i)
          destination.rowptr[row_offset[c]+i] = offset+source.rowptr[i];
        std::copy(source.colidx.begin(),source.colidx.end(),
                  destination.colidx.begin()+entry_offset[k][c]);
        std::copy(source.values.begin(),source.values.end(),
                  destination.values.begin()+entry_offset[k][c]);
      }
      if (forces) {
        std::copy(part.volume.begin(),part.volume.end(),out.volume.begin()+row_offset[c]);
        std::copy(part.cells.begin(),part.cells.end(),out.cells.begin()+row_offset[c]);
        std::copy(part.interface.begin(),part.interface.end(),
                  out.interface.begin()+row_offset[c]);
      }
      part = Part{};
    });
    // The fallback diagnostics count unique fits, as the shared serial cache does.
    auto unique = [](std::vector<std::vector<GlobalIndex>> &ids) {
      std::vector<GlobalIndex> all;
      for (auto &list : ids) all.insert(all.end(),list.begin(),list.end());
      std::sort(all.begin(),all.end());
      return static_cast<std::size_t>(std::unique(all.begin(),all.end())-all.begin());
    };
    interface_fallbacks = unique(interface_ids); x_image_fallbacks = unique(x_image_ids);
    if (split) {   // FinishSplit adds the fits of the irregular rows the device left to the host
      auto ids = [](std::vector<std::vector<GlobalIndex>> &lists) {
        std::vector<GlobalIndex> all;
        for (auto &list : lists) all.insert(all.end(),list.begin(),list.end());
        std::sort(all.begin(),all.end());
        all.erase(std::unique(all.begin(),all.end()),all.end());
        return all;
      };
      m.split.interface_ids = ids(interface_ids); m.split.x_image_ids = ids(x_image_ids);
    }
    m.build_join_s = std::chrono::duration<double>(BuildClock::now()-build_t1).count();
  }
  m.phase_rows_ready = true;
  for (int p : m.row_order) if (forces && m.patches[p].owner == m.rank) {
    for (int side : {-1,1}) if (m.Wall(m.patches[p],side))
      for (int j = 0; j < m.patches[p].cells[1]; ++j)
        for (int i = 0; i < m.patches[p].cells[0]; ++i) {
          const std::array<int,3> ijk{{i,j,side < 0 ? 0 : m.patches[p].cells[2]-1}};
          auto c = m.Cell(p,ijk);
          out.boundary_cells.push_back(m.Id(p,ijk));
          out.boundary_area.push_back(c.width[0]*c.width[1]);
          out.boundary_width.push_back(c.width[2]);
          c.center[2] += side*c.width[2];
          out.boundary_samples.push_back(BoundarySample{c.center,side});
        }
  }
  // Gradient maps are assembly scratch, and must not survive alongside device CSR.
  m.gradients.clear(); m.gradients.rehash(0);
  out.quadratic_interface_fallbacks = interface_fallbacks;
  out.quadratic_x_image_fallbacks = x_image_fallbacks;
  for (const auto &q : m.quadratics) {
    out.quadratic_interface_fallbacks += q.second.interface_fallback;
    out.quadratic_x_image_fallbacks += q.second.x_image_fallback;
  }
  m.quadratics.clear(); m.quadratics.rehash(0);
  m.reconstruction_faces = ReconstructionFaces{};
  if (split) { m.split.out = std::move(out); return LocalCompositeSystem{}; }
  return out;
}

const std::vector<int> &MeshGeometry::PhaseRows() const {
  impl_->DiscoverPhaseRows();
  return impl_->phase_rows;
}

void MeshGeometry::SetPhaseThreads(int threads) {
  if (threads < 0) throw std::invalid_argument("Gravity phase thread count must be nonnegative");
  if (threads == 0) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    // up to 32, two affinity CPUs left to the main thread and the runtime: the prefetched
    // phase assembly then ends within the matter step; the coefficients do not depend on it
    threads = std::max(1,std::min(32,AvailablePhaseCpus()-2));
#else
    threads = 1;
#endif
  }
  impl_->phase_threads = std::min(threads,AvailablePhaseCpus());
}

int MeshGeometry::ResolveSetupThreads(int threads) {
  if (threads < 0) throw std::invalid_argument("Gravity setup thread count must be nonnegative");
  if (threads == 0) {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    threads = AvailablePhaseCpus();
#else
    threads = 1;
#endif
  }
  return std::min(threads,AvailablePhaseCpus());
}

void MeshGeometry::SetSetupThreads(int threads) {
  impl_->setup_threads = ResolveSetupThreads(threads);
}

int MeshGeometry::SetupThreads() const {
  return std::max(1,std::min(impl_->setup_threads,AvailablePhaseCpus()));
}

int MeshGeometry::PhaseThreads() const {
  const auto &m = *impl_;
  const int bound = std::min(m.phase_threads,AvailablePhaseCpus());
  return m.phase_rows_ready ? std::min(bound,static_cast<int>(m.phase_rows.size())) : bound;
}

LocalCompositeUpdate MeshGeometry::BuildPhaseUpdate(Real phase, bool forces) {
  auto &m = *impl_;
  if (!m.phase_rows_ready)
    throw std::logic_error("Gravity phase update requires initial BuildSystem or PhaseRows");
  double shear = std::fmod(static_cast<double>(phase),m.length[1]);
  if (!std::isfinite(shear)) throw std::invalid_argument("Gravity nonfinite shear phase");
  if (shear < 0.) shear += m.length[1];
  if (!m.domain.shear_phase_updates && shear != 0.)
    throw std::invalid_argument("Gravity phase updates are disabled for ordinary periodic x");
  m.shear = shear; m.domain.shear_offset = phase;
  const int workers = std::max(1,PhaseThreads());
  if (m.phase_contexts.size() != static_cast<size_t>(workers)) {
    m.phase_contexts.clear(); m.phase_contexts.resize(workers);
  }
  LocalCompositeUpdate out;
  using Clock = std::chrono::steady_clock;
  const auto t0 = Clock::now();
  if (workers == 1) {
    m.AssemblePhaseRange(0,m.phase_rows.size(),out,m.phase_contexts[0],forces);
    m.phase_rows_s = std::chrono::duration<double>(Clock::now()-t0).count(); m.phase_join_s = 0;
    return out;
  }
  std::vector<LocalCompositeUpdate> parts(workers);
  std::vector<std::exception_ptr> errors(workers);
  auto assemble = [&](int worker) {
    try {
      const size_t first = m.phase_rows.size()*worker/workers;
      const size_t last = m.phase_rows.size()*(worker+1)/workers;
      m.AssemblePhaseRange(first,last,parts[worker],m.phase_contexts[worker],forces);
    } catch (...) { errors[worker] = std::current_exception(); }
  };
  std::vector<std::thread> threads;
  threads.reserve(workers-1);
  try {
    for (int worker = 1; worker < workers; ++worker) threads.emplace_back(assemble,worker);
  } catch (...) {
    for (auto &thread : threads) thread.join();
    throw;
  }
  assemble(0);
  for (auto &thread : threads) thread.join();
  for (const auto &error : errors) if (error) std::rethrow_exception(error);
  const auto t1 = Clock::now();
  // Chunks contain consecutive sorted rows. Concatenation performs no floating
  // arithmetic, preserving serial coefficients and sparse entry order bitwise:
  // every (component, chunk) copies into its own slot of the exact-size outputs, on the
  // phase workers (the serial join took a large part of the fine update).
  auto component = [](LocalCompositeUpdate &u, int k) -> CsrRowUpdate & {
    if (k == 0) return u.matrix;
    if (k == 1) return u.boundary_rhs;
    if (k < 5) return u.acceleration[k-2];
    return u.boundary_acceleration[k-5];
  };
  const int components = forces ? 8 : 1;
  std::vector<std::vector<size_t>> row_at(components,std::vector<size_t>(workers+1,0)),
      entry_at(components,std::vector<size_t>(workers+1,0));
  for (int k = 0; k < components; ++k) {
    for (int w = 0; w < workers; ++w) {
      const auto &part = component(parts[w],k);
      row_at[k][w+1] = row_at[k][w]+part.rows.size();
      entry_at[k][w+1] = entry_at[k][w]+part.values.size();
    }
    if (entry_at[k][workers] > static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Gravity compact phase CSR exceeds 32-bit offsets");
    auto &a = component(out,k); const auto &first = component(parts[0],k);
    a.global_rows = first.global_rows; a.global_cols = first.global_cols;
    a.row_begin = first.row_begin;
  }
  ParallelChunks(workers,3*components,[&](int t) {   // the exact-size outputs
    auto &a = component(out,t/3); const int k = t/3;
    if (t%3 == 0) { a.rows.resize(row_at[k][workers]); a.rowptr.resize(row_at[k][workers]+1); a.rowptr[0] = 0; }
    else if (t%3 == 1) a.colidx.resize(entry_at[k][workers]);
    else a.values.resize(entry_at[k][workers]);
  });
  ParallelChunks(workers,workers*components,[&](int t) {
    const int w = t%workers, k = t/workers;
    auto &part = component(parts[w],k); auto &a = component(out,k);
    const size_t r0 = row_at[k][w], e0 = entry_at[k][w];
    std::copy(part.rows.begin(),part.rows.end(),a.rows.begin()+r0);
    for (size_t i = 1; i < part.rowptr.size(); ++i)
      a.rowptr[r0+i] = static_cast<int>(e0)+part.rowptr[i];
    std::copy(part.colidx.begin(),part.colidx.end(),a.colidx.begin()+e0);
    std::copy(part.values.begin(),part.values.end(),a.values.begin()+e0);
    part = CsrRowUpdate{};   // release each chunk as it is consumed
  });
  m.phase_rows_s = std::chrono::duration<double>(t1-t0).count();
  m.phase_join_s = std::chrono::duration<double>(Clock::now()-t1).count();
  return out;
}

std::array<long long,2> MeshGeometry::SplitRows() const {
  const auto &x = impl_->split;
  return {{x.why[0]+x.irregular,static_cast<long long>(x.regular.size())}};
}

std::array<double,6> MeshGeometry::LastDeviceStats() const { return impl_->device_stats; }

std::array<double,6> MeshGeometry::LastBuildSeconds() const {
  const auto &c = impl_->build_class;
  return {{impl_->build_rows_s,impl_->build_join_s,c[0],c[1],c[2],c[3]}};
}

std::array<double,2> MeshGeometry::LastPhaseUpdateSeconds() const {
  return {{impl_->phase_rows_s,impl_->phase_join_s}};
}
}  // namespace gravity
