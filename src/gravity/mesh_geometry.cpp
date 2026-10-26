//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/mesh_geometry.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>
#if defined(__linux__)
#include <sched.h>
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
};
using QuadraticCache = std::unordered_map<GlobalIndex,Quadratic>;
struct Box { Vec lo{}, hi{}; };
struct Node { Box box; int begin, end, left = -1, right = -1; };
struct Face {
  GlobalIndex left, right;
  int normal, side;
  double area = 1., distance;
  Vec delta{}, ti{}, tj{};
};

void Add(Row &row, GlobalIndex column, double value) {
  // Most rows contain only seven entries; a short contiguous scan avoids a
  // heap node per coefficient and avoids gradient maps on aligned interiors.
  if (value == 0.) return;
  for (auto &v : row) if (v.first == column) { v.second += value; return; }
  row.emplace_back(column, value);
}

template<typename Csr>
void Append(Csr &matrix, Row &row, GlobalIndex split, bool exterior,
            double sign = 1.) {
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
  Row a, difference;
  std::array<Row,3> g;
  RowWorkspace() {
    a.reserve(32); difference.reserve(32);
    for (auto &row : g) row.reserve(32);
  }
};

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

void JoinPhaseRows(CsrRowUpdate &out, const std::vector<CsrRowUpdate *> &parts) {
  size_t rows = 0, nnz = 0;
  for (const auto *part : parts) { rows += part->rows.size(); nnz += part->values.size(); }
  if (nnz > static_cast<size_t>(std::numeric_limits<int>::max()))
    throw std::overflow_error("Gravity compact phase CSR exceeds 32-bit offsets");
  out.global_rows = parts.front()->global_rows; out.global_cols = parts.front()->global_cols;
  out.row_begin = parts.front()->row_begin;
  out.rows.reserve(rows); out.rowptr.reserve(rows+1);
  out.colidx.reserve(nnz); out.values.reserve(nnz);
  for (auto *part : parts) {
    const int offset = static_cast<int>(out.values.size());
    out.rows.insert(out.rows.end(),part->rows.begin(),part->rows.end());
    for (size_t i = 1; i < part->rowptr.size(); ++i) out.rowptr.push_back(offset+part->rowptr[i]);
    out.colidx.insert(out.colidx.end(),part->colidx.begin(),part->colidx.end());
    out.values.insert(out.values.end(),part->values.begin(),part->values.end());
    // Release each compact chunk as it is consumed; no complete fine CSR is copied.
    *part = CsrRowUpdate{};
  }
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
  struct PhaseContext {
    QuadraticCache quadratics;
    double shear = std::numeric_limits<double>::quiet_NaN();
  };
  std::vector<PhaseContext> phase_contexts;
  int phase_threads = 1;
  std::vector<int> phase_rows;
  bool phase_rows_ready = false;

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
        for (int t = 0; t < nd; ++t) if (t != d) f.area *= ci.width[t];
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
  const Quadratic &Quad(GlobalIndex row, QuadraticCache &cache) const {
    auto found = cache.find(row);
    if (found != cache.end()) return found->second;
    struct Sample { GlobalIndex id; Vec delta; };
    const auto cell = Cell(row);
    std::vector<Sample> samples;
    bool one_sided = !domain.c1_z_interfaces.empty();
    auto add = [&](GlobalIndex id, const Vec &delta) {
      if (one_sided) for (double plane : domain.c1_z_interfaces)
        if ((cell.center[2] < plane) != (cell.center[2]+delta[2] < plane)) return;
      double distance = 0.;
      for (int d = 0; d < nd; ++d)
        distance += (delta[d]/cell.width[d])*(delta[d]/cell.width[d]);
      if (distance < 1.e-20) return;
      for (const auto &s : samples) {
        if (s.id != id) continue;
        double error = 0.;
        for (int d = 0; d < nd; ++d) error += std::abs(s.delta[d]-delta[d])/cell.width[d];
        if (error < 1.e-8) return;
      }
      samples.push_back({id,delta});
    };
    Quadratic q;
    auto neighbors = [&](GlobalIndex id, const Vec &offset) {
      q.phase_dependent = q.phase_dependent || TouchesX(id);
      for (const auto &f : Faces(id)) {
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
    for (int ring = 2; !full_rank && one_sided && ring < 4; ++ring) {
      expand(first,last); first = last; last = samples.size();
      full_rank = invert();
    }
    if (!full_rank && one_sided && coarse_reconstruction) {
      // A coarse region may have only one normal cell, so quadratic curvature
      // on that side is unknowable. Keep an elliptic preconditioner stencil;
      // expose this lower-order C1 fallback in hierarchy diagnostics.
      one_sided = false; samples.clear(); q.interface_fallback = true;
      q.phase_dependent = false;
      neighbors(row,Vec{}); expand(0,samples.size());
      full_rank = invert();
    }
    if (!full_rank)
      throw std::runtime_error("Gravity singular one-sided quadratic reconstruction");
    q.coefficients.reserve(samples.size()+1);
    auto coefficient = [&](GlobalIndex id, const Modes &value) {
      for (auto &v : q.coefficients) if (v.first == id) {
        for (int i = 0; i < modes; ++i) v.second[i] += value[i];
        return;
      }
      q.coefficients.emplace_back(id,value);
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
    if (TouchesX(row)) return true;
    // All faces strictly inside one rectangular patch are aligned, so no LS
    // stencil can couple their geometry to a distant shearing face.
    const int p = Patch(row); const auto ijk = Indices(p,row);
    bool interior = true;
    for (int d = 0; d < nd; ++d)
      interior = interior && ijk[d] > 0 && ijk[d]+1 < patches[p].cells[d];
    if (interior && domain.c1_z_interfaces.empty()) return false;
    QuadraticCache cache;
    for (const auto &f : Faces(row)) {
      if (!HasOffset(f.ti) && !HasOffset(f.tj) &&
          !AtC1Interface(f) &&
          (f.left >= partition.back() || f.right >= partition.back() ||
           Cell(f.left).width[f.normal] == Cell(f.right).width[f.normal])) continue;
      for (GlobalIndex id : {f.left,f.right}) if (id < partition.back()) {
        if (Quad(id,cache).phase_dependent) return true;
      }
    }
    return false;
  }
  void DiscoverPhaseRows() {
    if (phase_rows_ready) return;
    phase_rows.clear();
    const GlobalIndex begin = partition[rank], end = partition[rank+1];
    for (GlobalIndex row = begin; row < end; ++row)
      if (IsPhaseRow(row)) phase_rows.push_back(static_cast<int>(row-begin));
    phase_rows_ready = true;
  }
  void AddPotential(Row &target, GlobalIndex column, double value) const {
    if (neumann_z && column >= partition.back()) {
      const auto ghost = Ghost(column);
      Add(target,ghost.first,value); Add(target,column,value*ghost.second);
    } else Add(target,column,value);
  }
  // Full setup and phase refresh deliberately share this exact physical row
  // routine, including canonical faces and all Dirichlet/Neumann LS couplings.
  void AssembleRow(GlobalIndex row, const CompositeCell &ci, double volume,
                   bool forces, RowWorkspace &work, QuadraticCache &quadratic_cache,
                   bool *is_interface = nullptr, bool *phase_dependent = nullptr) const {
    work.a.clear(); for (auto &component : work.g) component.clear();
    if (is_interface) *is_interface = false;
    if (phase_dependent) *phase_dependent = TouchesX(ci);
    for (const auto &f : Faces(row)) {
      work.difference.clear();
      AddPotential(work.difference,f.right,1.); AddPotential(work.difference,f.left,-1.);
      if (f.left < partition.back() && f.right < partition.back()) {
        const auto left = Cell(f.left), right = Cell(f.right);
        if (HasOffset(f.ti) || HasOffset(f.tj) || AtC1Interface(f) ||
            left.width[f.normal] != right.width[f.normal]) {
          auto reconstruct = [&](GlobalIndex id, const CompositeCell &cell,
                                 const Vec &offset, double sign) {
            const auto &q = Quad(id,quadratic_cache);
            if (phase_dependent && q.phase_dependent) *phase_dependent = true;
            Modes moment{};
            for (int d = 0; d < nd; ++d) moment[d] = sign*offset[d]/cell.width[d];
            int k = nd;
            for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t) {
              moment[k] = sign*offset[d]*offset[t]/(cell.width[d]*cell.width[t])*
                          (d == t ? .5 : 1.);
              // Separate one-sided curvatures also handle a C1 source/vacuum
              // join: derivative correction is (hL^2 HL-hR^2 HR)/(8d).
              if (d == f.normal && t == d)
                moment[k] -= sign*.125;
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
        if (forces) Add(work.g[f.normal],v.first,-flux*ci.width[f.normal]/(2.*volume));
      }
      const GlobalIndex other = row == f.left ? f.right : f.left;
      if (is_interface && other < partition.back()) {
        const auto cj = Cell(other);
        if (std::abs(ci.width[0]-cj.width[0]) > minwidth[0]*1.e-5) *is_interface = true;
      }
    }
  }
  void AssemblePhaseRange(size_t first, size_t last, LocalCompositeUpdate &out,
                          PhaseContext &context) const {
    // A context belongs to exactly one worker. A two-ring fit changes phase
    // when either its center or a first-ring neighbor touches the seam.
    if (context.shear != shear) {
      for (auto it = context.quadratics.begin(); it != context.quadratics.end();) {
        if (it->second.phase_dependent) it = context.quadratics.erase(it);
        else ++it;
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
    init(out.matrix,n); init(out.boundary_rhs,ns);
    for (int d = 0; d < 3; ++d) {
      init(out.acceleration[d],n); init(out.boundary_acceleration[d],ns);
    }
    RowWorkspace work;
    for (size_t i = first; i < last; ++i) {
      const GlobalIndex row = begin+phase_rows[i];
      const auto cell = Cell(row); double volume = 1.;
      for (int d = 0; d < nd; ++d) volume *= cell.width[d];
      AssembleRow(row,cell,volume,true,work,context.quadratics);
      Append(out.matrix,work.a,n,false); Append(out.boundary_rhs,work.a,n,true,-1.);
      for (int d = 0; d < 3; ++d) {
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
  Row Average(const Box &box, bool &images, int depth = 0) const {
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
    Row out;
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
            const auto fill = Average(neighbor,images,depth+1);
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
    return out;
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
    const std::array<Real,3> &lo, const std::array<Real,3> &hi, bool *images) const {
  Box box;
  for (int d = 0; d < impl_->nd; ++d) { box.lo[d] = lo[d]; box.hi[d] = hi[d]; }
  bool crossed = false;
  const auto row = impl_->Average(box,crossed);
  if (images) *images = crossed;
  std::vector<std::pair<GlobalIndex,Real>> result;
  result.reserve(row.size());
  for (const auto &v : row) if (v.second != 0.) result.emplace_back(v.first,v.second);
  return result;
}

LocalCompositeSystem MeshGeometry::BuildSystem(bool forces) {
  auto &m = *impl_;
  // Full setup owns its scratch independently of retained phase-worker caches.
  // In particular, a reused object must never inherit gradients from an earlier
  // full build or an interrupted assembly at a different phase.
  m.gradients.clear();
  m.quadratics.clear();
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
  RowWorkspace work;
  const bool discover_phase_rows = !m.phase_rows_ready;
  if (discover_phase_rows) m.phase_rows.clear();
  for (GlobalIndex row = begin; row < end; ++row) {
    const auto ci = m.Cell(row); double volume = 1.;
    for (int d = 0; d < m.nd; ++d) volume *= ci.width[d];
    if (forces) {
      out.volume.push_back(volume); out.cells.push_back(ci); out.interface.push_back(0);
    }
    bool is_interface = false, phase_dependent = false;
    m.AssembleRow(row,ci,volume,forces,work,m.quadratics,forces ? &is_interface : nullptr,
                  discover_phase_rows ? &phase_dependent : nullptr);
    if (forces) out.interface.back() = is_interface;
    if (discover_phase_rows && phase_dependent)
      m.phase_rows.push_back(static_cast<int>(row-begin));
    Append(out.matrix,work.a,n,false);
    if (forces) Append(out.boundary_rhs,work.a,n,true,-1.);
    for (int d = 0; forces && d < 3; ++d) {
      Append(out.acceleration[d],work.g[d],n,false);
      Append(out.boundary_acceleration[d],work.g[d],n,true);
    }
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
  for (const auto &q : m.quadratics)
    out.quadratic_interface_fallbacks += q.second.interface_fallback;
  m.quadratics.clear(); m.quadratics.rehash(0);
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
    threads = 8;
#else
    threads = 1;
#endif
  }
  impl_->phase_threads = std::min(threads,AvailablePhaseCpus());
}

int MeshGeometry::PhaseThreads() const {
  const auto &m = *impl_;
  const int bound = std::min(m.phase_threads,AvailablePhaseCpus());
  return m.phase_rows_ready ? std::min(bound,static_cast<int>(m.phase_rows.size())) : bound;
}

LocalCompositeUpdate MeshGeometry::BuildPhaseUpdate(Real phase) {
  auto &m = *impl_;
  if (!m.phase_rows_ready)
    throw std::logic_error("Gravity phase update requires initial BuildSystem or PhaseRows");
  double shear = std::fmod(static_cast<double>(phase),m.length[1]);
  if (!std::isfinite(shear)) throw std::invalid_argument("Gravity nonfinite shear phase");
  if (shear < 0.) shear += m.length[1];
  m.shear = shear; m.domain.shear_offset = phase;
  const int workers = std::max(1,PhaseThreads());
  if (m.phase_contexts.size() != static_cast<size_t>(workers)) {
    m.phase_contexts.clear(); m.phase_contexts.resize(workers);
  }
  LocalCompositeUpdate out;
  if (workers == 1) {
    m.AssemblePhaseRange(0,m.phase_rows.size(),out,m.phase_contexts[0]);
    return out;
  }
  std::vector<LocalCompositeUpdate> parts(workers);
  std::vector<std::exception_ptr> errors(workers);
  auto assemble = [&](int worker) {
    try {
      const size_t first = m.phase_rows.size()*worker/workers;
      const size_t last = m.phase_rows.size()*(worker+1)/workers;
      m.AssemblePhaseRange(first,last,parts[worker],m.phase_contexts[worker]);
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
  // Chunks contain consecutive sorted rows. Concatenation performs no floating
  // arithmetic, preserving serial coefficients and sparse entry order bitwise.
  std::vector<CsrRowUpdate *> matrices(workers);
  auto join = [&](auto select, CsrRowUpdate &destination) {
    for (int worker = 0; worker < workers; ++worker) matrices[worker] = select(parts[worker]);
    JoinPhaseRows(destination,matrices);
  };
  join([](LocalCompositeUpdate &p) { return &p.matrix; },out.matrix);
  join([](LocalCompositeUpdate &p) { return &p.boundary_rhs; },out.boundary_rhs);
  for (int d = 0; d < 3; ++d) {
    join([&](LocalCompositeUpdate &p) { return &p.acceleration[d]; },out.acceleration[d]);
    join([&](LocalCompositeUpdate &p) { return &p.boundary_acceleration[d]; },out.boundary_acceleration[d]);
  }
  return out;
}
}  // namespace gravity
