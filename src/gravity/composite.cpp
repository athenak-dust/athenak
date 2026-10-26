//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/composite.hpp"
#include "gravity/mesh_geometry.hpp"

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
  // The reference backend retains its global Galerkin hierarchy and solver,
  // but shares the physical discretization with the distributed builder.
  // One-cell patches preserve caller row order exactly; this intentionally
  // global metadata expansion belongs only to the explicit reference backend.
  std::vector<GravityPatch> patches(n);
  for (int i = 0; i < n; ++i) {
    patches[i].gid = i;
    for (int d = 0; d < 3; ++d) {
      patches[i].lo[d] = cells[i].center[d]-.5*cells[i].width[d];
      patches[i].hi[d] = cells[i].center[d]+.5*cells[i].width[d];
    }
  }
  MeshGeometry geometry(std::move(patches),domain,0,1);
  auto physical = geometry.BuildSystem();
  auto host = [](LocalCsr &&local) {
    HostCsr matrix;
    matrix.rows = local.LocalRows(); matrix.cols = local.global_cols;
    matrix.rowptr = std::move(local.rowptr);
    matrix.values = std::move(local.values);
    matrix.colidx.assign(local.colidx.begin(),local.colidx.end());
    return matrix;
  };
  out.volume = std::move(physical.volume);
  out.interface = std::move(physical.interface);
  out.boundary_samples = std::move(physical.boundary_samples);
  out.hierarchy.matrices.push_back(host(std::move(physical.matrix)));
  out.boundary_rhs = host(std::move(physical.boundary_rhs));
  for (int d = 0; d < 3; ++d) {
    out.acceleration[d] = host(std::move(physical.acceleration[d]));
    out.boundary_acceleration[d] = host(std::move(physical.boundary_acceleration[d]));
  }
  BuildHierarchy(out, cells, domain, maxlevel, fine_row_partition);
  return out;
}
}  // namespace gravity
