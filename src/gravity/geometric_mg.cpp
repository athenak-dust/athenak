//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/geometric_mg.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <memory_resource>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

namespace gravity {
namespace {
using Row = std::vector<std::pair<GlobalIndex,Real>>;

// One cache belongs to one fixed coarse geometry/phase and one host worker.
// Keys retain every input coordinate bit; values retain the original weight
// sequence and both dependency flags. Nothing is summed or reordered here.
class AverageCache {
  using Weight = std::pair<GlobalIndex,Real>;
  using Key = std::array<std::uint64_t,6>;
  static constexpr std::size_t kLimit = 16u*1024u*1024u;
  struct Hash {
    std::size_t operator()(const Key &key) const noexcept {
      std::uint64_t value = 0x9e3779b97f4a7c15ULL;
      for (auto bits : key) value ^= bits+0x9e3779b97f4a7c15ULL+(value<<6)+(value>>2);
      return static_cast<std::size_t>(value);
    }
  };
  class Budget final : public std::pmr::memory_resource {
   public:
    std::size_t used = 0, peak = 0;
   private:
    void *do_allocate(std::size_t bytes, std::size_t alignment) override {
      if (bytes > kLimit-used) throw std::bad_alloc();
      void *pointer = std::pmr::new_delete_resource()->allocate(bytes,alignment);
      used += bytes; peak = std::max(peak,used); return pointer;
    }
    void do_deallocate(void *pointer, std::size_t bytes, std::size_t alignment) override {
      std::pmr::new_delete_resource()->deallocate(pointer,bytes,alignment); used -= bytes;
    }
    bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override {
      return this == &other;
    }
  };
  struct Entry {
    std::pmr::vector<Weight> values;
    bool images, shear_images;
    Entry(std::pmr::memory_resource *resource, const Row &weights, bool image, bool shear)
        : values(weights.begin(),weights.end(),resource), images(image), shear_images(shear) {}
  };
 public:
  struct Result {
    Row owned;  // only an uncached query owns this ordinary working vector
    const Weight *cached = nullptr;
    std::size_t count = 0;
    bool stored = false, images = false, shear_images = false;
    std::size_t Size() const { return stored ? count : owned.size(); }
    const Weight &operator[](std::size_t i) const { return stored ? cached[i] : owned[i]; }
  };
  explicit AverageCache(const MeshGeometry &coarse, bool measure = false)
      : coarse_(coarse), arena_(4096,&budget_), cache_(&arena_), measure_(measure) {}

  Result Get(const std::array<Real,3> &lo, const std::array<Real,3> &hi) {
    Key key;
    for (int d = 0; d < 3; ++d) {
      const double lower = lo[d], upper = hi[d];
      std::memcpy(&key[d],&lower,sizeof(lower));
      std::memcpy(&key[d+3],&upper,sizeof(upper));
    }
    ++statistics_.queries;
    const auto found = cache_.find(key);
    if (found != cache_.end()) {
      ++statistics_.hits;
      const auto &entry = found->second;
      return {{},entry.values.data(),entry.values.size(),true,entry.images,entry.shear_images};
    }
    ++statistics_.misses;
    bool images = false, shear_images = false;
    const auto begin = measure_ ? std::chrono::steady_clock::now() :
                                 std::chrono::steady_clock::time_point{};
    auto values = coarse_.VirtualAverage(lo,hi,&images,&shear_images);
    if (measure_) statistics_.virtual_average_seconds +=
        std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
    const bool oversized = values.size() > kLimit/sizeof(Weight);
    if (!exhausted_ && !oversized) {
      try {
        const auto insertion = cache_.try_emplace(key,&arena_,values,images,shear_images);
        const auto &entry = insertion.first->second;
        return {{},entry.values.data(),entry.values.size(),true,entry.images,entry.shear_images};
      } catch (const std::bad_alloc &) {
        // Failed arena growth cannot affect the original query result. Keep
        // existing entries usable, and bypass all subsequent new keys.
        exhausted_ = true;
      }
    }
    ++statistics_.bypasses;
    statistics_.oversized_bypasses += oversized;
    return {std::move(values),nullptr,0,false,images,shear_images};
  }

  ProlongationCacheStats Statistics() const {
    auto result = statistics_;
    result.entries = cache_.size(); result.worker_peak_bytes = budget_.peak;
    return result;
  }
 private:
  const MeshGeometry &coarse_;
  Budget budget_;
  std::pmr::monotonic_buffer_resource arena_;
  std::pmr::unordered_map<Key,Entry,Hash> cache_;
  bool measure_ = false, exhausted_ = false;
  ProlongationCacheStats statistics_;
};

void AddCacheStats(ProlongationCacheStats &total, const ProlongationCacheStats &part) {
  total.queries += part.queries; total.hits += part.hits; total.misses += part.misses;
  total.bypasses += part.bypasses; total.oversized_bypasses += part.oversized_bypasses;
  total.entries += part.entries; total.worker_peak_bytes += part.worker_peak_bytes;
  total.virtual_average_seconds += part.virtual_average_seconds;
}

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
    bool &images, bool &shear_images, AverageCache &cache) {
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
    const auto average = cache.Get(lo,hi);
    for (std::size_t i = 0; i < average.Size(); ++i) {
      const auto &v = average[i];
      Add(row,v.first,weight*v.second);
    }
    images = images || average.images; shear_images = shear_images || average.shear_images;
  }
}

void Transfers(const MeshGeometry &fine, const MeshGeometry &coarse,
    const CompositeDomain &domain, int rank, LocalCsr &p, LocalCsr &r,
    TransferDiagnostics &diagnostics, bool neumann_z, std::vector<int> &phase_rows,
    ProlongationCacheStats &cache_statistics) {
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
  // Both loops are pure per row (the average cache only memoizes queries), so
  // consecutive row chunks of several workers, concatenated in order, give the serial CSR.
  struct PartP {
    LocalCsr p;
    std::vector<int> phase_rows;
    TransferDiagnostics diagnostics;
    ProlongationCacheStats cache;
  };
  struct PartR {
    LocalCsr r;
    Real volume_defect = 0;
  };
  auto prolongation = [&](GlobalIndex first, GlobalIndex last, PartP &part) {
    Row row; row.reserve(64);
    AverageCache cache(coarse);
    auto &diag = part.diagnostics;
    for (GlobalIndex i = first; i < last; ++i) {
      const auto cell = fine.Cell(i);
      bool images = false, shear_images = false;
      ProlongationRow(fine,coarse,domain,i,row,images,shear_images,cache);
      // Before the first x-image traversal, geometry routing is phase independent.
      // Thus a row that can depend on phase records this traversal at every phase,
      // even if its final coefficients cancel. No A-row dependency is assumed.
      if (shear_images) part.phase_rows.push_back(static_cast<int>(i-p.row_begin));
      Real constant = 0.; std::array<Real,3> linear{};
      for (const auto &v : row) {
        constant += v.second;
        if (!images) {
          const auto source = coarse.Cell(v.first);
          for (int d = 0; d < domain.ndim; ++d) linear[d] += v.second*source.center[d];
        }
      }
      if (!domain.isolated_z || neumann_z)
        diag.constant_defect = std::max(diag.constant_defect,std::abs(constant-Real(1)));
      if (!images) {
        ++diag.interior_linear_rows;
        for (int d = 0; d < domain.ndim; ++d)
          diag.interior_linear_defect = std::max(diag.interior_linear_defect,
                                                 std::abs(linear[d]-cell.center[d]));
      }
      Append(part.p,row);
    }
    part.cache = cache.Statistics();
  };  // Each worker frees its P-query cache before R is built.
  auto restriction = [&](GlobalIndex first, GlobalIndex last, PartR &part) {
    Row row; row.reserve(64);
    for (GlobalIndex i = first; i < last; ++i) {
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
      part.volume_defect = std::max(part.volume_defect,
          static_cast<Real>(std::abs(covered-volume)/volume));
      if (std::abs(covered-volume) > volume*1.e-7)
        throw std::runtime_error("Gravity restriction children do not cover parent volume");
      Append(part.r,row);
    }
  };
  auto parallel = [](int workers, auto &&task) {
    std::vector<std::exception_ptr> errors(workers);
    auto run = [&](int w) {
      try { task(w); } catch (...) { errors[w] = std::current_exception(); }
    };
    std::vector<std::thread> threads;
    threads.reserve(workers-1);
    try {
      for (int w = 1; w < workers; ++w) threads.emplace_back(run,w);
    } catch (...) { for (auto &thread : threads) thread.join(); throw; }
    run(0);
    for (auto &thread : threads) thread.join();
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
  };
  auto join = [](LocalCsr &destination, LocalCsr &source) {
    const int offset = static_cast<int>(destination.values.size());
    if (destination.values.size()+source.values.size() >
        static_cast<size_t>(std::numeric_limits<int>::max()))
      throw std::overflow_error("Gravity transfer CSR exceeds 32-bit local offsets");
    for (size_t i = 1; i < source.rowptr.size(); ++i)
      destination.rowptr.push_back(offset+source.rowptr[i]);
    destination.colidx.insert(destination.colidx.end(),source.colidx.begin(),source.colidx.end());
    destination.values.insert(destination.values.end(),source.values.begin(),source.values.end());
    source = LocalCsr{};
  };
  const int threads = fine.SetupThreads();
  // consecutive chunks (>= 512 rows, up to 16 per worker) taken in any order, joined in order
  auto chunking = [threads](GlobalIndex rows, int &workers, int &chunks) {
    workers = static_cast<int>(std::max<GlobalIndex>(1,std::min<GlobalIndex>(threads,rows/4096)));
    chunks = static_cast<int>(std::max<GlobalIndex>(workers,
        std::min<GlobalIndex>(16*static_cast<GlobalIndex>(workers),rows/512)));
  };
  {
    const GlobalIndex rows = fine_end-p.row_begin;
    int workers = 1, chunks = 1;
    chunking(rows,workers,chunks);
    std::vector<PartP> parts(chunks);
    std::atomic<int> next{0};
    parallel(workers,[&](int) {
      for (int c = next++; c < chunks; c = next++)
        prolongation(p.row_begin+rows*c/chunks,p.row_begin+rows*(c+1)/chunks,parts[c]);
    });
    cache_statistics = ProlongationCacheStats{};
    for (auto &part : parts) {
      join(p,part.p);
      phase_rows.insert(phase_rows.end(),part.phase_rows.begin(),part.phase_rows.end());
      diagnostics.constant_defect = std::max(diagnostics.constant_defect,
                                             part.diagnostics.constant_defect);
      diagnostics.interior_linear_defect = std::max(diagnostics.interior_linear_defect,
                                                    part.diagnostics.interior_linear_defect);
      diagnostics.interior_linear_rows += part.diagnostics.interior_linear_rows;
      AddCacheStats(cache_statistics,part.cache);
    }
  }
  {
    const GlobalIndex rows = coarse_end-r.row_begin;
    int workers = 1, chunks = 1;
    chunking(rows,workers,chunks);
    std::vector<PartR> parts(chunks);
    std::atomic<int> next{0};
    parallel(workers,[&](int) {
      for (int c = next++; c < chunks; c = next++)
        restriction(r.row_begin+rows*c/chunks,r.row_begin+rows*(c+1)/chunks,parts[c]);
    });
    for (auto &part : parts) {
      join(r,part.r);
      diagnostics.restriction_volume_defect = std::max(diagnostics.restriction_volume_defect,
                                                       part.volume_defect);
    }
  }
}
}  // namespace

HierarchyBuildTimes &BuildTimes() {
  static HierarchyBuildTimes times;
  return times;
}

GeometricSystem BuildGeometricHierarchy(std::vector<GravityPatch> patches,
    const CompositeDomain &domain, int rank, int nranks, bool neumann_z,
    bool fine_forces, int threads) {
  auto &times = BuildTimes();
  auto now = []() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
  };
  double clock = now();
  auto lap = [&](double &bucket) { const double t = now(); bucket += t-clock; clock = t; };
  GeometricSystem out;
  out.fine_geometry.reset(new MeshGeometry(std::move(patches),domain,rank,nranks,neumann_z));
  MeshGeometry *fine = out.fine_geometry.get();
  fine->SetSetupThreads(threads);
  lap(times.geometry);
  out.fine = fine->BuildSystem(fine_forces);
  lap(times.fine_system);
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
    coarse->SetSetupThreads(threads);
    lap(times.geometry);
    LocalCsr p, r; TransferDiagnostics check; std::vector<int> phase_rows;
    ProlongationCacheStats cache_statistics;
    Transfers(*fine,*coarse,domain,rank,p,r,check,neumann_z,phase_rows,cache_statistics);
    lap(times.transfers);
    out.initial_prolongation_cache.push_back(cache_statistics);
    auto system = coarse->BuildSystem(false);
    lap(times.coarse_system);
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
    const std::vector<int> &rows, ProlongationCacheStats *statistics) {
  const GlobalIndex begin = fine.RowPartition()[rank];
  const GlobalIndex count = fine.RowPartition()[rank+1]-begin;
  if (!std::is_sorted(rows.begin(),rows.end()) ||
      std::adjacent_find(rows.begin(),rows.end()) != rows.end() ||
      (!rows.empty() && (rows.front() < 0 || rows.back() >= count)))
    throw std::invalid_argument("Gravity invalid prolongation phase rows");
  const int workers = rows.size() < 256 ? 1 :
      std::max(1,std::min(fine.PhaseThreads(),static_cast<int>(rows.size())));
  std::vector<CsrRowUpdate> parts(workers);
  std::vector<ProlongationCacheStats> cache_statistics(workers);
  std::vector<std::exception_ptr> errors(workers);
  auto assemble = [&](int worker) {
    try {
      auto &a = parts[worker]; a.global_rows = fine.GlobalRows();
      a.global_cols = coarse.GlobalRows(); a.row_begin = begin;
      const size_t first = rows.size()*worker/workers, last = rows.size()*(worker+1)/workers;
      a.rows.assign(rows.begin()+first,rows.begin()+last);
      Row row; row.reserve(64);
      AverageCache cache(coarse,statistics != nullptr);
      for (size_t j = first; j < last; ++j) {
        bool images = false, shear_images = false;
        ProlongationRow(fine,coarse,domain,begin+rows[j],row,images,shear_images,cache);
        Append(a,row);
      }
      cache_statistics[worker] = cache.Statistics();
    } catch (...) { errors[worker] = std::current_exception(); }
  };
  std::vector<std::thread> threads;
  try {
    for (int worker = 1; worker < workers; ++worker) threads.emplace_back(assemble,worker);
  } catch (...) { for (auto &thread : threads) thread.join(); throw; }
  assemble(0);
  for (auto &thread : threads) thread.join();
  for (const auto &error : errors) if (error) std::rethrow_exception(error);
  if (statistics) {
    *statistics = {};
    for (const auto &part : cache_statistics) AddCacheStats(*statistics,part);
  }
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
