//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/geometric_mg.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

namespace gravity {
namespace {
using Row = std::vector<std::pair<GlobalIndex,Real>>;

void Add(Row &row, GlobalIndex column, Real value) {
  if (value == 0.) return;
  for (auto &v : row) if (v.first == column) { v.second += value; return; }
  row.emplace_back(column,value);
}

template<class Matrix>
void Append(Matrix &a, Row &row) {
  std::sort(row.begin(),row.end(),[](const auto &x, const auto &y) { return x.first < y.first; });
  for (const auto &v : row) if (v.second != 0.) {
    if (a.values.size() == static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Gravity transfer CSR exceeds 32-bit local offsets");
    a.colidx.push_back(v.first); a.values.push_back(v.second);
  }
  a.rowptr.push_back(static_cast<int>(a.values.size()));
}

GlobalIndex Count(const std::vector<GravityPatch> &patches) {
  GlobalIndex n = 0;
  for (const auto &p : patches)
    n += static_cast<GlobalIndex>(p.cells[0])*p.cells[1]*p.cells[2];
  return n;
}

std::vector<GravityPatch> Coarsen(const std::vector<GravityPatch> &fine,
                                 const CompositeDomain &domain) {
  auto coarse = fine;
  bool cell_step = false;
  for (auto &p : coarse) for (int d = 0; d < domain.ndim; ++d) if (p.cells[d] > 1) {
    if (p.cells[d]%2 != 0)
      throw std::runtime_error("Gravity geometric coarsening requires even active patch "
                               "cell counts until one; odd large grid cannot coarsen");
    p.cells[d] /= 2; cell_step = true;
  }
  if (cell_step) return coarse;

  // At one cell per patch, merge only complete equal-size aligned siblings.
  // Integer coordinates use the smallest existing patch extent in each axis;
  // dyadic AMR boxes therefore have exact integer family keys even on odd roots.
  std::array<double,3> unit{};
  unit.fill(std::numeric_limits<double>::max());
  for (const auto &p : fine) for (int d = 0; d < domain.ndim; ++d)
    unit[d] = std::min(unit[d],static_cast<double>(p.hi[d]-p.lo[d]));
  using Key = std::array<long long,6>;
  struct Family {
    std::vector<int> child;
    unsigned mask = 0, axes = 0;
    bool valid = true;
  };
  std::map<Key,Family> families;
  std::vector<Key> keys(fine.size());
  for (size_t i = 0; i < fine.size(); ++i) {
    Key key{}; int bits = 0; unsigned axes = 0; bool aligned = true;
    for (int d = 0; d < domain.ndim; ++d) {
      const double extent = fine[i].hi[d]-fine[i].lo[d];
      const double scaled = extent/unit[d];
      const long long width = std::llround(scaled);
      const double lower = (fine[i].lo[d]-domain.lo[d])/unit[d];
      const long long index = std::llround(lower);
      if (width <= 0 || std::abs(scaled-width) > 1.e-6 || std::abs(lower-index) > 1.e-6)
        throw std::runtime_error("Gravity geometric patch siblings are not dyadically aligned");
      // Padding can leave physical patches offset by half their width relative
      // to the extended-domain origin. Such cells remain valid leaves, but are
      // not geometric children of the rounded parent box below.
      aligned = aligned && index%width == 0;
      // An axis already covered by one patch remains unchanged. Requiring two
      // siblings in it would strand valid rectangular power-of-two grids.
      const bool collapsed = std::abs(fine[i].lo[d]-domain.lo[d]) < unit[d]*1.e-6 &&
                             std::abs(fine[i].hi[d]-domain.hi[d]) < unit[d]*1.e-6;
      const long long parent = collapsed ? index : index/(2*width)*(2*width);
      key[d] = parent; key[d+3] = width;
      if (!collapsed) {
        axes |= 1u<<d;
        bits |= (index-parent >= width ? 1 : 0)<<d;
      }
    }
    keys[i] = key;
    auto &family = families[key];
    family.valid = family.valid && aligned;
    if (!family.child.empty() && family.axes != axes) family.valid = false;
    family.axes = axes;
    if (family.mask & (1u<<bits)) family.valid = false;
    family.mask |= 1u<<bits; family.child.push_back(i);
  }
  coarse.clear(); coarse.reserve(fine.size());
  for (size_t i = 0; i < fine.size(); ++i) {
    const auto &key = keys[i]; const auto &family = families.at(key);
    unsigned full = 0; size_t children = 0;
    for (int bits = 0; bits < (1<<domain.ndim); ++bits)
      if ((static_cast<unsigned>(bits) & ~family.axes) == 0) {
        full |= 1u<<bits; ++children;
      }
    bool complete = family.axes != 0 && family.valid && family.mask == full &&
                    family.child.size() == children;
    for (int d = 0; d < domain.ndim; ++d)
      complete = complete && domain.lo[d]+(key[d]+
          (family.axes & (1u<<d) ? 2 : 1)*key[d+3])*unit[d] <=
                            domain.hi[d]+unit[d]*1.e-6;
    if (!complete) { coarse.push_back(fine[i]); continue; }
    if (family.child.front() != static_cast<int>(i)) continue;
    GravityPatch parent = fine[i]; parent.gid = -1;
    for (int child : family.child) parent.owner = std::min(parent.owner,fine[child].owner);
    for (int d = 0; d < domain.ndim; ++d) {
      parent.lo[d] = domain.lo[d]+key[d]*unit[d];
      parent.hi[d] = domain.lo[d]+(key[d]+
          (family.axes & (1u<<d) ? 2 : 1)*key[d+3])*unit[d];
    }
    coarse.push_back(parent);
  }
  if (coarse.size() == fine.size())
    throw std::runtime_error("Gravity geometric hierarchy cannot reduce >128 rows: "
                             "no complete aligned patch sibling family (odd/incomplete root)");
  return coarse;
}

void ProlongationRow(const MeshGeometry &fine, const MeshGeometry &coarse,
    const CompositeDomain &domain, GlobalIndex i, Row &row,
    bool &images, bool &shear_images) {
  const auto cell = fine.Cell(i);
  const GlobalIndex parent = coarse.Locate(cell.center);
  const auto cc = coarse.Cell(parent);
  std::array<int,3> sign{}; int split = 0;
  for (int d = 0; d < domain.ndim; ++d) {
    if (cell.width[d] > cc.width[d]*(1.+1.e-8))
      throw std::runtime_error("Gravity coarse cell is smaller than a fine child");
    const double delta = (cell.center[d]-cc.center[d])/cc.width[d];
    if (std::abs(delta) < 1.e-8) continue;
    if (std::abs(std::abs(delta)-.25) > 1.e-7)
      throw std::runtime_error("Gravity interpolation requires centered dyadic children");
    sign[d] = delta > 0. ? 1 : -1; split |= 1<<d;
  }
  row.clear(); images = false; shear_images = false;
  for (int bits = 0; bits < (1<<domain.ndim); ++bits) {
    if (bits & ~split) continue;
    Real weight = 1.; std::array<Real,3> lo{}, hi{};
    for (int d = 0; d < domain.ndim; ++d) {
      const bool neighbor = bits & (1<<d);
      if (sign[d] != 0) weight *= neighbor ? .25 : .75;
      const double center = cc.center[d]+(neighbor ? sign[d]*cc.width[d] : 0.);
      lo[d] = center-.5*cc.width[d]; hi[d] = center+.5*cc.width[d];
    }
    bool image = false, shear_image = false;
    for (const auto &v : coarse.VirtualAverage(lo,hi,&image,&shear_image))
      Add(row,v.first,weight*v.second);
    images = images || image; shear_images = shear_images || shear_image;
  }
}

void Transfers(const MeshGeometry &fine, const MeshGeometry &coarse,
    const CompositeDomain &domain, int rank, LocalCsr &p, LocalCsr &r,
    TransferDiagnostics &diagnostics, bool neumann_z, std::vector<int> &phase_rows) {
  p.global_rows = fine.GlobalRows(); p.global_cols = coarse.GlobalRows();
  p.row_begin = fine.RowPartition()[rank];
  r.global_rows = coarse.GlobalRows(); r.global_cols = fine.GlobalRows();
  r.row_begin = coarse.RowPartition()[rank];
  const GlobalIndex fine_end = fine.RowPartition()[rank+1];
  const GlobalIndex coarse_end = coarse.RowPartition()[rank+1];
  p.rowptr.reserve(fine_end-p.row_begin+1); r.rowptr.reserve(coarse_end-r.row_begin+1);
  const size_t expected_p = static_cast<size_t>(fine_end-p.row_begin)*(1<<domain.ndim);
  if (expected_p <= static_cast<size_t>(std::numeric_limits<int>::max())) {
    p.colidx.reserve(expected_p); p.values.reserve(expected_p);
  }
  const size_t expected_r = static_cast<size_t>(coarse_end-r.row_begin)*(1<<domain.ndim);
  if (expected_r <= static_cast<size_t>(std::numeric_limits<int>::max())) {
    r.colidx.reserve(expected_r); r.values.reserve(expected_r);
  }
  Row row; row.reserve(64);
  for (GlobalIndex i = p.row_begin; i < fine_end; ++i) {
    const auto cell = fine.Cell(i);
    bool images = false, shear_images = false;
    ProlongationRow(fine,coarse,domain,i,row,images,shear_images);
    // Before the first x-image traversal, geometry routing is phase independent.
    // Thus a row that can depend on phase records this traversal at every phase,
    // even if its final coefficients cancel. No A-row dependency is assumed.
    if (shear_images) phase_rows.push_back(static_cast<int>(i-p.row_begin));
    Real constant = 0.; std::array<Real,3> linear{};
    for (const auto &v : row) {
      constant += v.second;
      if (!images) {
        const auto source = coarse.Cell(v.first);
        for (int d = 0; d < domain.ndim; ++d) linear[d] += v.second*source.center[d];
      }
    }
    if (!domain.isolated_z || neumann_z)
      diagnostics.constant_defect = std::max(diagnostics.constant_defect,std::abs(constant-Real(1)));
    if (!images) {
      ++diagnostics.interior_linear_rows;
      for (int d = 0; d < domain.ndim; ++d)
        diagnostics.interior_linear_defect = std::max(diagnostics.interior_linear_defect,
                                                    std::abs(linear[d]-cell.center[d]));
    }
    Append(p,row);
  }
  for (GlobalIndex i = r.row_begin; i < coarse_end; ++i) {
    const auto cell = coarse.Cell(i); std::array<Real,3> lo{}, hi{};
    double volume = 1.;
    for (int d = 0; d < domain.ndim; ++d) {
      lo[d] = cell.center[d]-.5*cell.width[d]; hi[d] = cell.center[d]+.5*cell.width[d];
      volume *= cell.width[d];
    }
    row.clear(); double covered = 0.;
    for (GlobalIndex child : fine.QueryBox(lo,hi)) {
      const auto c = fine.Cell(child); double v = 1.;
      for (int d = 0; d < domain.ndim; ++d) {
        if (c.center[d]-.5*c.width[d] < lo[d]-cell.width[d]*1.e-8 ||
            c.center[d]+.5*c.width[d] > hi[d]+cell.width[d]*1.e-8)
          throw std::runtime_error("Gravity restriction encountered a partial fine child");
        v *= c.width[d];
      }
      covered += v; Add(row,child,1.);
    }
    diagnostics.restriction_volume_defect = std::max(diagnostics.restriction_volume_defect,
        static_cast<Real>(std::abs(covered-volume)/volume));
    if (std::abs(covered-volume) > volume*1.e-7)
      throw std::runtime_error("Gravity restriction children do not cover parent volume");
    Append(r,row);
  }
}
}  // namespace

GeometricSystem BuildGeometricHierarchy(std::vector<GravityPatch> patches,
    const CompositeDomain &domain, int rank, int nranks, bool neumann_z,
    bool fine_forces) {
  GeometricSystem out;
  out.fine_geometry.reset(new MeshGeometry(std::move(patches),domain,rank,nranks,neumann_z));
  MeshGeometry *fine = out.fine_geometry.get();
  out.fine = fine->BuildSystem(fine_forces);
  out.quadratic_interface_fallbacks.push_back(out.fine.quadratic_interface_fallbacks);
  out.hierarchy.constant_nullspace = !domain.isolated_z || neumann_z;
  out.hierarchy.matrices.push_back(std::move(out.fine.matrix));
  out.hierarchy.row_partitions.push_back(fine->RowPartition());
  out.patches.push_back(fine->Patches());
  for (int level = 0; fine->GlobalRows() > 128; ++level) {
    if (level >= 64) throw std::runtime_error("Gravity geometric hierarchy exceeded 64 levels");
    auto coarse_patches = Coarsen(fine->Patches(),domain);
    if (Count(coarse_patches) >= fine->GlobalRows())
      throw std::runtime_error("Gravity geometric coarsening did not reduce global rows");
    std::unique_ptr<MeshGeometry> coarse(new MeshGeometry(std::move(coarse_patches),domain,
                                                        rank,nranks,neumann_z,true));
    LocalCsr p, r; TransferDiagnostics check; std::vector<int> phase_rows;
    Transfers(*fine,*coarse,domain,rank,p,r,check,neumann_z,phase_rows);
    auto system = coarse->BuildSystem(false);
    out.quadratic_interface_fallbacks.push_back(system.quadratic_interface_fallbacks);
    out.hierarchy.matrices.push_back(std::move(system.matrix));
    out.hierarchy.row_partitions.push_back(coarse->RowPartition());
    out.hierarchy.prolongation.push_back(std::move(p));
    out.hierarchy.restriction.push_back(std::move(r));
    out.prolongation_phase_rows.push_back(std::move(phase_rows));
    out.patches.push_back(coarse->Patches()); out.transfer_checks.push_back(check);
    out.coarse_geometry.push_back(std::move(coarse));
    fine = out.coarse_geometry.back().get();
  }
  return out;
}

CsrRowUpdate BuildProlongationPhaseUpdate(const MeshGeometry &fine,
    const MeshGeometry &coarse, const CompositeDomain &domain, int rank,
    const std::vector<int> &rows) {
  const GlobalIndex begin = fine.RowPartition()[rank];
  const GlobalIndex count = fine.RowPartition()[rank+1]-begin;
  if (!std::is_sorted(rows.begin(),rows.end()) ||
      std::adjacent_find(rows.begin(),rows.end()) != rows.end() ||
      (!rows.empty() && (rows.front() < 0 || rows.back() >= count)))
    throw std::invalid_argument("Gravity invalid prolongation phase rows");
  const int workers = rows.size() < 256 ? 1 :
      std::max(1,std::min(fine.PhaseThreads(),static_cast<int>(rows.size())));
  std::vector<CsrRowUpdate> parts(workers);
  std::vector<std::exception_ptr> errors(workers);
  auto assemble = [&](int worker) {
    try {
      auto &a = parts[worker]; a.global_rows = fine.GlobalRows();
      a.global_cols = coarse.GlobalRows(); a.row_begin = begin;
      const size_t first = rows.size()*worker/workers, last = rows.size()*(worker+1)/workers;
      a.rows.assign(rows.begin()+first,rows.begin()+last);
      Row row; row.reserve(64);
      for (size_t j = first; j < last; ++j) {
        bool images = false, shear_images = false;
        ProlongationRow(fine,coarse,domain,begin+rows[j],row,images,shear_images);
        Append(a,row);
      }
    } catch (...) { errors[worker] = std::current_exception(); }
  };
  std::vector<std::thread> threads;
  try {
    for (int worker = 1; worker < workers; ++worker) threads.emplace_back(assemble,worker);
  } catch (...) { for (auto &thread : threads) thread.join(); throw; }
  assemble(0);
  for (auto &thread : threads) thread.join();
  for (const auto &error : errors) if (error) std::rethrow_exception(error);
  auto out = std::move(parts[0]);
  for (int worker = 1; worker < workers; ++worker) {
    const auto &part = parts[worker]; const int offset = out.values.size();
    if (out.values.size()+part.values.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Gravity prolongation phase CSR exceeds local offsets");
    out.rows.insert(out.rows.end(),part.rows.begin(),part.rows.end());
    out.colidx.insert(out.colidx.end(),part.colidx.begin(),part.colidx.end());
    out.values.insert(out.values.end(),part.values.begin(),part.values.end());
    for (size_t j = 1; j < part.rowptr.size(); ++j) out.rowptr.push_back(offset+part.rowptr[j]);
  }
  return out;
}
}  // namespace gravity
