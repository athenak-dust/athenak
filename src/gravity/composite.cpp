//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/composite.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace gravity {
namespace {
using Vec = std::array<double, 3>;
using Row = std::map<int, double>;
using Rows = std::vector<Row>;
struct Box { Vec lo, hi; };
struct Node { Box box; int begin, end, left = -1, right = -1; };
struct Face {
  int i, j, normal;
  double area, distance;
  Vec delta, ti, tj;
};

// Balanced geometric tree; each face query costs tree traversal plus intersected cells.
class BoxTree {
 public:
  BoxTree(const std::vector<CompositeCell> &cells, int ndim) : cells_(cells), nd_(ndim) {
    order_.resize(cells.size());
    std::iota(order_.begin(), order_.end(), 0);
    Build(0, order_.size());
  }
  void Query(const Box &box, std::vector<int> &result) const { QueryNode(0, box, result); }
 private:
  const std::vector<CompositeCell> &cells_;
  int nd_;
  std::vector<int> order_;
  std::vector<Node> nodes_;
  bool Intersects(const Box &a, const Box &b) const {
    for (int d = 0; d < nd_; ++d) if (a.hi[d] < b.lo[d] || b.hi[d] < a.lo[d]) return false;
    return true;
  }
  Box CellBox(int i) const {
    Box b{};
    for (int d = 0; d < nd_; ++d) {
      b.lo[d] = cells_[i].center[d] - .5*cells_[i].width[d];
      b.hi[d] = cells_[i].center[d] + .5*cells_[i].width[d];
    }
    return b;
  }
  int Build(int begin, int end) {
    const int id = nodes_.size();
    Node node{}; node.begin = begin; node.end = end;
    node.box.lo.fill(std::numeric_limits<double>::max());
    node.box.hi.fill(-std::numeric_limits<double>::max());
    for (int p = begin; p < end; ++p) {
      auto b = CellBox(order_[p]);
      for (int d = 0; d < nd_; ++d) {
        node.box.lo[d] = std::min(node.box.lo[d], b.lo[d]);
        node.box.hi[d] = std::max(node.box.hi[d], b.hi[d]);
      }
    }
    nodes_.push_back(node);
    if (end-begin > 8) {
      int axis = 0;
      for (int d = 1; d < nd_; ++d)
        if (node.box.hi[d]-node.box.lo[d] > node.box.hi[axis]-node.box.lo[axis]) axis = d;
      const int mid = (begin+end)/2;
      std::nth_element(order_.begin()+begin, order_.begin()+mid, order_.begin()+end,
          [&](int a, int b) { return cells_[a].center[axis] < cells_[b].center[axis]; });
      const int l = Build(begin, mid), r = Build(mid, end);
      nodes_[id].left = l; nodes_[id].right = r;
    }
    return id;
  }
  void QueryNode(int id, const Box &box, std::vector<int> &result) const {
    const auto &node = nodes_[id];
    if (!Intersects(box, node.box)) return;
    if (node.left >= 0) {
      QueryNode(node.left, box, result); QueryNode(node.right, box, result);
    } else {
      for (int p = node.begin; p < node.end; ++p)
        if (Intersects(box, CellBox(order_[p]))) result.push_back(order_[p]);
    }
  }
};

HostCsr Csr(const Rows &rows, int cols) {
  HostCsr out; out.rows = rows.size(); out.cols = cols; out.rowptr.push_back(0);
  for (const auto &row : rows) {
    for (const auto &v : row) if (v.second != 0.) {
      out.colidx.push_back(v.first); out.values.push_back(v.second);
    }
    out.rowptr.push_back(out.values.size());
  }
  return out;
}

Rows Multiply(const HostCsr &a, const HostCsr &b) {
  Rows out(a.rows);
  for (int i = 0; i < a.rows; ++i)
    for (int p = a.rowptr[i]; p < a.rowptr[i+1]; ++p) {
      const int j = a.colidx[p];
      for (int q = b.rowptr[j]; q < b.rowptr[j+1]; ++q)
        out[i][b.colidx[q]] += a.values[p]*b.values[q];
    }
  return out;
}

std::array<Vec, 3> Inverse(std::array<Vec, 3> a, int ndim) {
  std::array<Vec, 3> b{};
  for (int d = 0; d < ndim; ++d) b[d][d] = 1.;
  for (int d = 0; d < ndim; ++d) {
    int pivot = d;
    for (int k = d+1; k < ndim; ++k) if (std::abs(a[k][d]) > std::abs(a[pivot][d])) pivot = k;
    if (!(std::abs(a[pivot][d]) > std::numeric_limits<double>::min()))
      throw std::runtime_error("Composite gravity: singular neighbor gradient geometry");
    std::swap(a[d], a[pivot]); std::swap(b[d], b[pivot]);
    const double scale = a[d][d];
    for (int k = 0; k < ndim; ++k) { a[d][k] /= scale; b[d][k] /= scale; }
    for (int r = 0; r < ndim; ++r) if (r != d) {
      const double f = a[r][d];
      for (int k = 0; k < ndim; ++k) { a[r][k] -= f*a[d][k]; b[r][k] -= f*b[d][k]; }
    }
  }
  return b;
}

std::vector<int> Balanced(int n, int nranks) {
  std::vector<int> p(nranks+1);
  for (int r = 0; r <= nranks; ++r) p[r] = static_cast<long long>(n)*r/nranks;
  return p;
}

void BuildHierarchy(CompositeSystem &system, const std::vector<CompositeCell> &cells,
                    const CompositeDomain &domain, int maxlevel,
                    const std::vector<int> &partition) {
  auto &h = system.hierarchy;
  const int nranks = partition.empty() ? 1 : partition.size()-1;
  // Empty partitions let the solver balance rows using its actual communicator size.
  h.row_partitions.push_back(partition);
  std::vector<Vec> centers(cells.size());
  for (size_t i = 0; i < cells.size(); ++i)
    for (int d = 0; d < domain.ndim; ++d) centers[i][d] = cells[i].center[d];
  Vec spacing{};
  for (int d = 0; d < domain.ndim; ++d)
    spacing[d] = (domain.hi[d]-domain.lo[d])/domain.root_counts[d]*std::pow(2., 1-maxlevel);
  // Do not jump to the uniform root grid when deeper active refinement exists.
  for (int step = 0; centers.size() > 24 && step < 64; ++step) {
    std::map<std::array<long long, 3>, int> keys;
    std::vector<int> labels(centers.size());
    for (size_t i = 0; i < centers.size(); ++i) {
      std::array<long long, 3> key{};
      for (int d = 0; d < domain.ndim; ++d)
        key[d] = std::floor((centers[i][d]-domain.lo[d])/spacing[d]+1.e-9);
      keys.emplace(key, 0);
    }
    int next = 0;
    for (auto &key : keys) key.second = next++;
    for (size_t i = 0; i < centers.size(); ++i) {
      std::array<long long, 3> key{};
      for (int d = 0; d < domain.ndim; ++d)
        key[d] = std::floor((centers[i][d]-domain.lo[d])/spacing[d]+1.e-9);
      labels[i] = keys.at(key);
    }
    if (keys.size() < centers.size()) {
      const auto &a = h.matrices.back();
      Rows prows(a.rows);
      for (int i = 0; i < a.rows; ++i) {
        double diagonal = 0.;
        for (int p = a.rowptr[i]; p < a.rowptr[i+1]; ++p)
          if (a.colidx[p] == i) diagonal = a.values[p];
        if (!(diagonal > 0.)) throw std::runtime_error("Composite gravity: nonpositive diagonal");
        prows[i][labels[i]] += 1.;
        for (int p = a.rowptr[i]; p < a.rowptr[i+1]; ++p)
          prows[i][labels[a.colidx[p]]] -= .65*a.values[p]/diagonal;
      }
      HostCsr prolongation = Csr(prows, keys.size());
      Rows ap = Multiply(a, prolongation), coarse(keys.size());
      for (int i = 0; i < prolongation.rows; ++i)
        for (int p = prolongation.rowptr[i]; p < prolongation.rowptr[i+1]; ++p)
          for (const auto &v : ap[i])
            coarse[prolongation.colidx[p]][v.first] += prolongation.values[p]*v.second;
      h.prolongation.push_back(std::move(prolongation));
      h.matrices.push_back(Csr(coarse, keys.size()));
      h.row_partitions.push_back(partition.empty() ? std::vector<int>{} : Balanced(keys.size(), nranks));
      centers.assign(keys.size(), Vec{});
      for (const auto &key : keys)
        for (int d = 0; d < domain.ndim; ++d)
          centers[key.second][d] = domain.lo[d]+(key.first[d]+.5)*spacing[d];
    }
    for (int d = 0; d < domain.ndim; ++d) spacing[d] *= 2.;
  }
  h.constant_nullspace = !domain.isolated_z;
}
}  // namespace

CompositeSystem BuildCompositeSystem(const std::vector<CompositeCell> &cells,
    const CompositeDomain &domain, const std::vector<int> &fine_row_partition) {
  const int ndim = domain.ndim, n = cells.size();
  if (ndim < 2 || ndim > 3 || n == 0)
    throw std::invalid_argument("Composite gravity requires nonempty 2-D or 3-D leaf cells");
  if (domain.isolated_z && ndim != 3)
    throw std::invalid_argument("Composite gravity isolated z requires three dimensions");
  if (!fine_row_partition.empty() && (fine_row_partition.size() < 2 ||
      fine_row_partition.front() != 0 || fine_row_partition.back() != n ||
      !std::is_sorted(fine_row_partition.begin(), fine_row_partition.end())))
    throw std::invalid_argument("Composite gravity invalid finest row partition");
  Vec length{}, minwidth{};
  minwidth.fill(std::numeric_limits<double>::max());
  for (int d = 0; d < ndim; ++d) {
    length[d] = domain.hi[d]-domain.lo[d];
    if (!(length[d] > 0.) || domain.root_counts[d] <= 0)
      throw std::invalid_argument("Composite gravity invalid domain extent/root count");
  }
  std::vector<int> levels(n);
  int maxlevel = 0;
  CompositeSystem out; out.volume.resize(n, 1.); out.interface.resize(n, 0);
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < ndim; ++d) {
      const double width = cells[i].width[d], root = length[d]/domain.root_counts[d];
      if (!(width > 0.) || !std::isfinite(width) || !std::isfinite(cells[i].center[d]))
        throw std::invalid_argument("Composite gravity invalid cell box");
      const double lev = std::log2(root/width);
      const int level = std::lround(lev);
      if (level < 0 || level > 30 || std::abs(lev-level) > 1.e-5 ||
          (d > 0 && level != levels[i]))
        throw std::invalid_argument("Composite gravity requires dyadically refined leaf widths");
      levels[i] = level;
      const double index = (cells[i].center[d]-.5*width-domain.lo[d])/width;
      if (std::abs(index-std::round(index)) > 1.e-5 || index < -1.e-5 ||
          cells[i].center[d]+.5*width > domain.hi[d]+width*1.e-5)
        throw std::invalid_argument("Composite gravity nonaligned/out-of-domain leaf box");
      minwidth[d] = std::min(minwidth[d], width);
      out.volume[i] *= width;
    }
    maxlevel = std::max(maxlevel, levels[i]);
  }
  double domain_volume = 1.;
  for (int d = 0; d < ndim; ++d) domain_volume *= length[d];
  const double total_volume = std::accumulate(out.volume.begin(), out.volume.end(), 0.);
  if (std::abs(total_volume-domain_volume) > domain_volume*1.e-5)
    throw std::invalid_argument("Composite gravity leaves do not cover domain volume");
  double shear = std::fmod(static_cast<double>(domain.shear_offset), length[1]);
  if (!std::isfinite(shear)) throw std::invalid_argument("Composite gravity nonfinite shear");
  if (shear < 0.) shear += length[1];
  BoxTree tree(cells, ndim);
  std::vector<Face> faces;
  std::vector<Vec> coverage(n, Vec{});
  for (int i = 0; i < n; ++i) for (int d = 0; d < ndim; ++d) {
    const auto &ci = cells[i];
    const double edge = ci.center[d]+.5*ci.width[d];
    const bool periodic = std::abs(edge-domain.hi[d]) < minwidth[d]*1.e-5;
    if (domain.isolated_z && d == 2 && periodic) continue;
    const int firstwrap = periodic && d == 0 ? -1 : 0;
    const int lastwrap = periodic && d == 0 ? 1 : 0;
    for (int wrap = firstwrap; wrap <= lastwrap; ++wrap) {
      Vec image{};
      if (periodic) image[d] = length[d];
      if (periodic && d == 0) image[1] = shear+wrap*length[1];
      Box query{};
      for (int t = 0; t < ndim; ++t) {
        query.lo[t] = ci.center[t]-.5*ci.width[t]-image[t];
        query.hi[t] = ci.center[t]+.5*ci.width[t]-image[t];
      }
      query.lo[d] = edge-image[d]+minwidth[d]*1.e-6;
      query.hi[d] = edge-image[d]+minwidth[d]*2.e-6;
      std::vector<int> candidates;
      tree.Query(query, candidates);
      for (int j : candidates) {
        const auto &cj = cells[j];
        if (std::abs(cj.center[d]-.5*cj.width[d]+image[d]-edge) > minwidth[d]*1.e-5) continue;
        Face f{}; f.i = i; f.j = j; f.normal = d; f.area = 1.;
        for (int t = 0; t < ndim; ++t) {
          f.delta[t] = cj.center[t]+image[t]-ci.center[t];
          if (t == d) continue;
          const double lo = std::max(static_cast<double>(ci.center[t]-.5*ci.width[t]),
                                     cj.center[t]-.5*cj.width[t]+image[t]);
          const double hi = std::min(static_cast<double>(ci.center[t]+.5*ci.width[t]),
                                     cj.center[t]+.5*cj.width[t]+image[t]);
          if (hi-lo <= minwidth[t]*1.e-8) { f.area = 0.; break; }
          f.area *= hi-lo;
          f.ti[t] = .5*(hi+lo)-ci.center[t];
          f.tj[t] = .5*(hi+lo)-cj.center[t]-image[t];
        }
        if (f.area == 0.) continue;
        f.distance = f.delta[d];
        if (!(f.distance > 0.)) throw std::runtime_error("Composite gravity invalid normal distance");
        faces.push_back(f);
        coverage[i][d] += f.area; coverage[j][d] += f.area;
        if (levels[i] != levels[j]) out.interface[i] = out.interface[j] = 1;
      }
    }
  }
  if (domain.isolated_z) {
    for (int i = 0; i < n; ++i) for (int side : {-1, 1}) {
      const double edge = cells[i].center[2]+side*.5*cells[i].width[2];
      const double wall = side < 0 ? domain.lo[2] : domain.hi[2];
      if (std::abs(edge-wall) > minwidth[2]*1.e-5) continue;
      BoundarySample sample{cells[i].center, side};
      sample.center[2] += side*cells[i].width[2];
      const int ghost = n+out.boundary_samples.size();
      out.boundary_samples.push_back(sample);
      Face f{};
      f.i = side < 0 ? ghost : i; f.j = side < 0 ? i : ghost;
      f.normal = 2; f.area = out.volume[i]/cells[i].width[2];
      f.distance = cells[i].width[2]; f.delta[2] = f.distance;
      // Exterior center aligns tangentially: neither ghost gradients nor slopes
      // are needed for this physical face. Ghost values still enter real-cell LS.
      faces.push_back(f);
      coverage[i][2] += f.area;
    }
  }
  for (int i = 0; i < n; ++i) for (int d = 0; d < ndim; ++d) {
    const double expected = 2.*out.volume[i]/cells[i].width[d];
    if (std::abs(coverage[i][d]-expected) > expected*1.e-5)
      throw std::runtime_error("Composite gravity incomplete/overlapping cell-face coverage");
  }
  std::vector<std::array<Vec, 3>> normal(n);
  for (const auto &f : faces) {
    double distance2 = 0.;
    for (int d = 0; d < ndim; ++d) distance2 += f.delta[d]*f.delta[d];
    const double w = f.area/distance2;
    for (int r = 0; r < ndim; ++r) for (int d = 0; d < ndim; ++d) {
      const double v = w*f.delta[r]*f.delta[d];
      if (f.i < n) normal[f.i][r][d] += v;
      if (f.j < n) normal[f.j][r][d] += v;
    }
  }
  for (auto &m : normal) m = Inverse(m, ndim);
  std::vector<std::map<int, Vec>> gradients(n);
  for (const auto &f : faces) {
    double distance2 = 0.;
    for (int d = 0; d < ndim; ++d) distance2 += f.delta[d]*f.delta[d];
    for (int side = 0; side < 2; ++side) {
      const int i = side == 0 ? f.i : f.j, j = side == 0 ? f.j : f.i;
      if (i >= n) continue;
      const double sign = side == 0 ? 1. : -1.;
      for (int r = 0; r < ndim; ++r) {
        double coefficient = 0.;
        for (int d = 0; d < ndim; ++d)
          coefficient += normal[i][r][d]*sign*f.delta[d]*f.area/distance2;
        gradients[i][j][r] += coefficient; gradients[i][i][r] -= coefficient;
      }
    }
  }
  Rows a(n); std::array<Rows, 3> acceleration{{Rows(n), Rows(n), Rows(n)}};
  for (const auto &f : faces) {
    Row difference;
    difference[f.j] += 1.; difference[f.i] -= 1.;
    if (f.j < n) for (const auto &g : gradients[f.j])
      for (int t = 0; t < ndim; ++t) difference[g.first] += f.tj[t]*g.second[t];
    if (f.i < n) for (const auto &g : gradients[f.i])
      for (int t = 0; t < ndim; ++t) difference[g.first] -= f.ti[t]*g.second[t];
    for (const auto &v : difference) {
      const double flux = f.area/f.distance*v.second;
      if (f.i < n) {
        a[f.i][v.first] -= flux;
        acceleration[f.normal][f.i][v.first] -= flux*cells[f.i].width[f.normal]/(2.*out.volume[f.i]);
      }
      if (f.j < n) {
        a[f.j][v.first] += flux;
        acceleration[f.normal][f.j][v.first] -= flux*cells[f.j].width[f.normal]/(2.*out.volume[f.j]);
      }
    }
  }
  const int nsamples = out.boundary_samples.size();
  Rows boundary_rhs(n);
  for (int i = 0; i < n; ++i) {
    auto ghost = a[i].lower_bound(n);
    for (auto it = ghost; it != a[i].end(); ++it)
      boundary_rhs[i][it->first-n] = -it->second;  // move known values to RHS
    a[i].erase(ghost, a[i].end());
  }
  out.boundary_rhs = Csr(boundary_rhs, nsamples);
  out.hierarchy.matrices.push_back(Csr(a, n));
  for (int d = 0; d < 3; ++d) {
    Rows boundary_force(n);
    for (int i = 0; i < n; ++i) {
      auto ghost = acceleration[d][i].lower_bound(n);
      for (auto it = ghost; it != acceleration[d][i].end(); ++it)
        boundary_force[i][it->first-n] = it->second;
      acceleration[d][i].erase(ghost, acceleration[d][i].end());
    }
    out.acceleration[d] = Csr(acceleration[d], n);
    out.boundary_acceleration[d] = Csr(boundary_force, nsamples);
  }
  BuildHierarchy(out, cells, domain, maxlevel, fine_row_partition);
  return out;
}
}  // namespace gravity
