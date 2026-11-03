//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/sparse_mg.hpp"
#include "gravity/host_workers.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <string>
#include <thread>
#include <utility>
#if defined(__linux__)
#include <sched.h>
#endif
#include <unordered_map>

namespace gravity {
int &PhasePatternPadding() {
  static int padding = 2;
  return padding;
}
PhaseUpdateTimes &PhaseTimes() {
  static PhaseUpdateTimes times;
  return times;
}
OperatorSetupTimes &SetupTimes() {
  static OperatorSetupTimes times;
  return times;
}
bool &SetupTimesOn() {
  static bool on = false;
  return on;
}
int &OperatorSetupThreads() {
  static int threads = 1;
  return threads;
}
namespace {
double HostSeconds() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}
// adds the host seconds of its scope to a bucket of SetupTimes()
// (only with <gravity>/profile: SetupTimesOn)
struct SetupLap {
  double &bucket;
  const bool on;
  const double begin;
  explicit SetupLap(double &b) : bucket(b), on(SetupTimesOn()), begin(on ? HostSeconds() : 0.0) {}
  ~SetupLap() { if (on) bucket += HostSeconds()-begin; }
};

// host workers and consecutive chunks for `items` loop iterations of the operator setup
// (one worker per 64k items, at most OperatorSetupThreads(); four chunks per worker)
void SetupChunks(std::size_t items, int &workers, int &chunks) {
  const std::size_t threads = std::max(1,OperatorSetupThreads());
  workers = static_cast<int>(std::max<std::size_t>(1,std::min(threads,items/65536)));
  chunks = workers == 1 ? 1 : 4*workers;
}
using Vector = DvceArray1D<Real>;
using Indices = DvceArray1D<int>;

// the host copies into the pinned stages of a phase update (values of every operator, ~10^6
// to 10^7 entries per update at L3/L4) on up to 8 host threads of this rank's CPU set, in chunks
// of at least 2^16 entries; memcpy of the same bytes
int StageCopyWorkers() {
  static const int n = [] {
    int cpus = 1;
#if defined(__linux__)
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0,sizeof(affinity),&affinity) == 0) cpus = std::max(1,CPU_COUNT(&affinity));
#endif
    return std::min(8,cpus);
  }();
  return n;
}
template <typename T>
void StageCopy(T *dst, const T *src, std::size_t n) {
  constexpr std::size_t piece = std::size_t(1) << 16;
  const int workers = static_cast<int>(std::min<std::size_t>(StageCopyWorkers(),n/piece));
  if (workers <= 1) {
    if (n > 0) std::memcpy(dst,src,n*sizeof(T));
    return;
  }
  const int chunks = 4*workers;
  ParallelChunks(workers,chunks,[&](int c) {
    const std::size_t b = n*c/chunks, e = n*(c+1)/chunks;
    if (e > b) std::memcpy(dst+b,src+b,(e-b)*sizeof(T));
  });
}

// LASSQ representation: ||x|| = scale*sqrt(squares), without squaring an
// unscaled component. A negative scale propagates any NaN/Inf collectively.
// Empty MPI ranks contribute the identity (0,0).
struct ScaledSquares {
  Real scale, squares;
  KOKKOS_INLINE_FUNCTION void Join(const ScaledSquares &b) {
    if (scale < 0 || b.scale < 0) { scale = -1; squares = 0; return; }
    if (b.scale == 0) return;
    if (scale < b.scale) {
      const Real ratio = scale/b.scale;
      squares = b.squares+squares*ratio*ratio;
      scale = b.scale;
    } else {
      const Real ratio = b.scale/scale;
      squares += b.squares*ratio*ratio;
    }
  }
};
static_assert(sizeof(ScaledSquares) == 2*sizeof(Real),"MPI norm pair layout");
struct NormReduction {
  using value_type = ScaledSquares;
  Vector x;
  KOKKOS_INLINE_FUNCTION void init(value_type &s) const { s = {0,0}; }
  KOKKOS_INLINE_FUNCTION void join(value_type &a, const value_type &b) const {
    a.Join(b);
  }
  KOKKOS_INLINE_FUNCTION void operator()(int i, value_type &s) const {
    const Real a = fabs(x(i));
    s.Join({a <= std::numeric_limits<Real>::max() ? a : Real(-1),1});
  }
};
#if MPI_PARALLEL_ENABLED
void MergeScaledSquares(void *input, void *inout, int *length, MPI_Datatype *) {
  auto a = static_cast<ScaledSquares *>(input);
  auto b = static_cast<ScaledSquares *>(inout);
  for (int i = 0; i < *length; ++i) b[i].Join(a[i]);
}
#endif

// the device copy of a host vector: no zero fill of the new view, no host mirror (the
// vector's storage is copied directly)
template <typename T>
DvceArray1D<T> DeviceCopy(const std::vector<T> &data, const char *name) {
  DvceArray1D<T> result(Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(name)),
                        data.size());
  if (!data.empty()) {
    Kokkos::View<const T *, LayoutWrapper, HostMemSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(data.data(),data.size());
    Kokkos::deep_copy(result, host);
  }
  return result;
}

void Validate(const HostCsr &a) {
  if (a.rows < 1 || a.cols < 1 || a.rowptr.size() != static_cast<size_t>(a.rows+1)
      || a.rowptr.front() != 0 || a.colidx.size() != a.values.size()
      || a.rowptr.back() != static_cast<int>(a.values.size())) {
    throw std::runtime_error("self-gravity: invalid CSR shape/row offsets");
  }
  for (int i = 0; i < a.rows; ++i) {
    if (a.rowptr[i] > a.rowptr[i+1]) {
      throw std::runtime_error("self-gravity: decreasing CSR row offsets");
    }
  }
  for (size_t k = 0; k < a.values.size(); ++k) {
    if (a.colidx[k] < 0 || a.colidx[k] >= a.cols || !std::isfinite(a.values[k])) {
      throw std::runtime_error("self-gravity: invalid CSR column/value");
    }
  }
}

HostCsr Transpose(const HostCsr &a) {
  HostCsr t;
  t.rows = a.cols;
  t.cols = a.rows;
  t.rowptr.assign(t.rows+1, 0);
  for (int j : a.colidx) ++t.rowptr[j+1];
  std::partial_sum(t.rowptr.begin(), t.rowptr.end(), t.rowptr.begin());
  auto next = t.rowptr;
  t.colidx.resize(a.values.size());
  t.values.resize(a.values.size());
  for (int i = 0; i < a.rows; ++i) {
    for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) {
      int j = next[a.colidx[k]]++;
      t.colidx[j] = i;
      t.values[j] = a.values[k];
    }
  }
  return t;
}

void Axpy(Real alpha, const Vector &x, const Vector &y) {
  const int n = y.extent_int(0);
  Kokkos::parallel_for("gravity_axpy", Kokkos::RangePolicy<DevExeSpace>(0, n),
                       KOKKOS_LAMBDA(int i) { y(i) += alpha*x(i); });
}

void Scale(Real alpha, const Vector &x) {
  const int n = x.extent_int(0);
  Kokkos::parallel_for("gravity_scale", Kokkos::RangePolicy<DevExeSpace>(0, n),
                       KOKKOS_LAMBDA(int i) { x(i) *= alpha; });
}
// ATHENA_SG_DEVICE_MPI=1: the multigrid halo exchange hands device buffers to MPI (needs a
// GPU-aware MPI; experimental, for timing), otherwise pinned host buffers.
bool DeviceMpi() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_DEVICE_MPI");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// ATHENA_SG_HALO_ALLTOALLV=1: the former halo exchange (one MPI_Alltoallv per matrix-vector
// product, blocking deep copies with global fences), kept for A/B timing.  Default: point-to-
// point messages with the ranks that share halo entries only, copies queued on the device
// stream.  The same values arrive in the same places: results are bitwise identical.
// Operators with at least n nonzeros per row on average (the coarse multigrid levels: 42-69,
// rows up to 300 entries) use the row-team product kernel, which gives bitwise the same
// results; in our tests n = 40 was faster than 16, and n = 1 (the fine level too) slower.
// Default n = 40 on GPU backends, 0 (the
// one-thread-per-row kernel everywhere) on host backends; ATHENA_SG_SPMV_ROWS=<n> overrides
// (0: off).
int SpmvRowTeams() {
  static const int n = [] {
    const char *e = std::getenv("ATHENA_SG_SPMV_ROWS");
    if (e != nullptr) return std::atoi(e);
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    return 40;
#else
    return 0;
#endif
  }();
  return n;
}
// ATHENA_SG_PHASE_ALLOC_OLD=1: the phase updates allocate fresh device and pinned views
// for every changed pattern or halo (cudaMalloc/cudaFree, cudaHostAlloc/cudaFreeHost on the main
// thread, milliseconds per cycle on every rank) and upload the values with blocking copies.
// Default: the views of an operator's phase pattern and halo are subviews of stores that keep
// their capacity (grown by an eighth when too small), new device vectors are zeroed as the fresh
// ones were, and the value uploads are queued on the device stream with one fence per update.
// The same values reach the same places: results are bitwise identical.
bool PhaseAllocOld() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_PHASE_ALLOC_OLD");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// ATHENA_SG_PHASE_FORCE_OLD=1: the force and boundary operators of a phase update keep
// their own two collectives each (validity, ghost change).  Default: their flags join the
// hierarchy's one collective (UpdateHierarchyPhaseRows with extras); the updates run in the
// former order with the same plans.  An MPI-pattern switch: checked collectively.
bool PhaseForceOld() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_PHASE_FORCE_OLD");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// ATHENA_SG_MG_SYNC_COPIES=1: the device-to-device copies and fills of the V-cycle and the
// FGMRES basis are blocking (a device synchronization each, as before).  Default: queued on the
// device stream in the same order; every host read of device data still follows a fence.
bool MgSyncCopies() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_MG_SYNC_COPIES");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// ATHENA_SG_GHOST_MAP_OLD=1: a phase-configured operator keeps a hash map of its ghost
// columns (rebuilt at configuration and whenever its ghost list changes; L4: ~10^6 entries per
// rebuild) to find the base slot of a column.  Default: a binary search in the sorted base ghost
// list (the base slots are its positions; the same slots and ghost lists).
bool GhostMapOld() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_GHOST_MAP_OLD");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// ATHENA_SG_PROJECT_FUSE=1 (opt-in): in the V-cycle a projection followed by a product of
// the same vector (smoothing sweeps, the residual, the restriction, the prolongation of the coarse
// correction) exchanges the halo of the unprojected vector at the same time as the mean's
// MPI_Allreduce (one stream fence for both); the owner then subtracts the mean from its entries
// and every receiver from the received halo values.  This is the same subtraction of the same two
// numbers ONLY IF MPI_Allreduce returns bitwise the same sum on every rank.  The MPI standard asks
// for that, but not every implementation delivers it: UCC 1.3.0 (TL/UCP knomial, radix 4) adds in
// a rank-dependent order, and local sums {1, -1, 2^-54, 0} can come back as 0 on some ranks and
// 2^-54 on others; then a receiver's ghost value differs from the owner's (Codex review;
// OpenMPI 5 tuned/HAN and UCC >= 1.5 add in the same order everywhere).  Hence: off by default
// (every projection completes before the next product packs its halo, as before), and when it is
// requested a probe at every solver construction (ProbeAllreduceIdentical) sums such values and
// compares the result bits of all ranks; a mismatch keeps the former order and prints a note.
// An MPI-pattern switch: checked collectively.
bool ProjectFuse() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_PROJECT_FUSE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// (test hook) ATHENA_SG_PROJECT_FUSE_PROBE_TEST=1: the probe sees rank 0's sums with the last bit
// flipped, as an MPI with rank-dependent sums would deliver them (the probe must then refuse)
bool ProjectFuseProbeTest() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_PROJECT_FUSE_PROBE_TEST");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
// completion of the queued phase uploads (StagedCopy, the value copies) at every exit of a
// public phase update, also by an exception: the pinned stages may be written again as soon as
// the call has returned (Codex review: two consecutive fine updates, or a batch that throws after
// the fine upload, could otherwise rewrite a stage under a copy still in flight).
// Two exits (Codex review, round 3): a normal return calls Complete() as its last statement, a
// plain fence that throws what Kokkos throws on a failed device synchronization (CUDA/HIP errors
// of the queued copies or kernels), so the update does not report success for failed device
// work.  The destructor fences only when Complete() was not reached, i.e. while an exception
// unwinds the call; that fence must not throw (a second exception during unwinding terminates),
// and the exception already in flight is the one the caller sees.  The two fences carry
// different labels (gravity_phase_uploads / gravity_phase_uploads_unwind; the regression check
// gravity_phase_update_check counts them through a Kokkos Tools fence callback).
struct PhaseUploadFence {
  PhaseUploadFence() = default;
  PhaseUploadFence(const PhaseUploadFence &) = delete;
  PhaseUploadFence &operator=(const PhaseUploadFence &) = delete;
  void Complete() {   // the normal exit: may throw
    done_ = true;
    if (PhaseAllocOld()) return;
    Kokkos::fence("gravity_phase_uploads");
  }
  ~PhaseUploadFence() {
    if (done_ || PhaseAllocOld()) return;
    try { Kokkos::fence("gravity_phase_uploads_unwind"); } catch (...) {}
  }
 private:
  bool done_ = false;
};
bool PhaseCollectivesOld() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_PHASE_COLLECTIVES");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
bool HaloAlltoallv() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_SG_HALO_ALLTOALLV");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
}  // namespace

struct SparseMultigrid::Impl {
  // CSR indices are remapped to [local input, sparse ghost input]. Halo plans are
  // built once, exchange only requested entries, and are reused for every SpMV.
  struct Operator {
    int rows = 0, inputs = 0;
    GlobalIndex global_rows = 0, global_cols = 0, row_begin = 0;
    Indices rowptr, colidx, send_indices;
    Vector values, send_device, ghost_device, recv_device;
    Indices recv_slots;
    // pinned staging buffers (page-locked: no driver staging copy; DMA straight from/to them)
    Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace> send_host, ghost_host;
    std::vector<int> send_counts, recv_counts, send_offsets, recv_offsets;
    // the ranks with nonzero send / receive counts (point-to-point halo exchange), set by
    // BuildHalo; requests reused by every exchange
    std::vector<int> send_peers, recv_peers;
#if MPI_PARALLEL_ENABLED
    std::vector<MPI_Request> halo_requests;
#endif
    std::vector<GlobalIndex> input_partition, ghost_ids;
    bool phase_configured = false, phase_active = false;
    Indices phase_map, phase_rows, phase_rowptr, phase_colidx;
    Vector phase_values;
    Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace> phase_values_pinned;   // upload buffer, kept
    std::vector<int> phase_rows_host, phase_rowptr_host;
    std::vector<GlobalIndex> phase_columns_host, phase_base_ghosts;
    std::unordered_map<GlobalIndex,int> phase_ghost_slots;
    // phase_base_ghosts sorted and unique: base slots by binary search, no map
    bool phase_base_sorted = false;
    std::uint64_t phase_masked_base_nnz = 0;
    FineOperatorStats phase_updates;
    // the stores behind the views above that BuildHalo and the phase updates replace
    // (ATHENA_SG_PHASE_ALLOC_OLD=1: fresh views instead)
    struct Stores {
      Indices send_indices, recv_slots, phase_rowptr, phase_colidx;
      Vector send_device, ghost_device, recv_device, phase_values;
      Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace> send_host, ghost_host;
      // pinned staging of the phase pattern uploads (queued copies; fenced once per update)
      Kokkos::View<int*, Kokkos::SharedHostPinnedSpace> stage_rowptr, stage_colidx;
      Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace> stage_values;
    } store;
  };
  // the first n entries of a store (grown to n + n/8 if smaller; the old allocation is
  // released first)
  template <typename V>
  static V FromStore(V &store, std::size_t n, const char *label) {
    if (store.extent(0) < n) {
      store = V();
      store = V(Kokkos::view_alloc(Kokkos::WithoutInitializing,std::string(label)),n+n/8);
    }
    V v = Kokkos::subview(store,std::make_pair(std::size_t(0),n));
    return v;
  }
  // the same, cleared on the device stream as a freshly constructed device vector
  static Vector ZeroFromStore(Vector &store, std::size_t n, const char *label) {
    Vector v = FromStore(store,n,label);
    if (n > 0) Kokkos::deep_copy(DevExeSpace(),v,Real(0));
    return v;
  }
  // the contents of data in the first entries of a device store, through a pinned stage,
  // the copy queued on the device stream (the caller fences before the stage is written again)
  template <typename T>
  static DvceArray1D<T> StagedCopy(DvceArray1D<T> &store,
                                   Kokkos::View<T*, Kokkos::SharedHostPinnedSpace> &stage,
                                   const std::vector<T> &data, const char *label,
                                   const char *stage_label) {
    DvceArray1D<T> v = FromStore(store,data.size(),label);
    if (!data.empty()) {
      auto h = FromStore(stage,data.size(),stage_label);
      StageCopy(h.data(),data.data(),data.size());
      Kokkos::deep_copy(DevExeSpace(),v,h);
    }
    return v;
  }
  // DeviceCopy into a store
  template <typename T>
  static DvceArray1D<T> StoreCopy(DvceArray1D<T> &store, const std::vector<T> &data,
                                  const char *label) {
    if (PhaseAllocOld()) return DeviceCopy(data,label);
    DvceArray1D<T> v = FromStore(store,data.size(),label);
    if (!data.empty()) {
      Kokkos::View<const T *, LayoutWrapper, HostMemSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(data.data(),data.size());
      Kokkos::deep_copy(v,host);
    }
    return v;
  }
  struct Level {
    std::vector<GlobalIndex> partition;
    Operator a, p, r;
    Vector inverse_diagonal, b, x, residual, temp;
  };
  static void AddPhaseStorage(const Operator &op, StorageStats &s) {
    s.logical_csr_bytes += (op.phase_rowptr.size()+op.phase_colidx.size())*sizeof(int)
        +op.phase_values.size()*sizeof(Real);
    s.phase_map_bytes += (op.phase_map.size()+op.phase_rows.size())*sizeof(int);
    s.phase_rows += op.phase_rows_host.size();
    s.phase_nnz += op.phase_values.size();
    s.logical_halo_bytes += op.recv_slots.size()*sizeof(int);
    if (op.recv_device.data() != op.ghost_device.data()) {
      s.logical_halo_bytes += op.recv_device.size()*sizeof(Real);
    }
    s.host_pattern_bytes += (op.input_partition.capacity()+op.ghost_ids.capacity()
        +op.phase_base_ghosts.capacity()+op.phase_columns_host.capacity())*sizeof(GlobalIndex)
        +(op.phase_rows_host.capacity()+op.phase_rowptr_host.capacity())*sizeof(int);
    const auto map_bytes = op.phase_ghost_slots.bucket_count()*sizeof(void*)
        +op.phase_ghost_slots.size()*sizeof(decltype(op.phase_ghost_slots)::value_type);
    s.phase_ghost_map_bytes += map_bytes;
    s.host_pattern_bytes += map_bytes;
  }
  static void AddStorage(const Operator &op, StorageStats &s) {
    s.logical_csr_bytes += (op.rowptr.size()+op.colidx.size())*sizeof(int)
                   +op.values.size()*sizeof(Real);
    s.logical_halo_bytes += op.send_indices.size()*sizeof(int)
        +(op.send_device.size()+op.ghost_device.size()+op.send_host.size()
          +op.ghost_host.size())*sizeof(Real)
        +(op.send_counts.size()+op.recv_counts.size()+op.send_offsets.size()
          +op.recv_offsets.size())*sizeof(int);
    AddPhaseStorage(op,s);
  }
  // the allocations behind an operator's views: [begin, end) byte ranges of the views and
  // of the stores and pinned stages behind them (Storage merges overlapping ranges, so a subview,
  // a view shared by two operators, or a store and its subview count once)
  using Ranges = std::vector<std::pair<std::uintptr_t,std::uintptr_t>>;
  template <typename V>
  static void AddRange(Ranges &r, const V &v) {
    if (v.data() == nullptr || v.span() == 0) return;
    const auto b = reinterpret_cast<std::uintptr_t>(v.data());
    r.emplace_back(b,b+v.span()*sizeof(typename V::value_type));
  }
  static void AddAllocations(const Operator &op, Ranges &csr, Ranges &halo) {
    AddRange(csr,op.rowptr); AddRange(csr,op.colidx); AddRange(csr,op.values);
    AddRange(csr,op.phase_rowptr); AddRange(csr,op.phase_colidx); AddRange(csr,op.phase_values);
    AddRange(csr,op.phase_values_pinned);
    const auto &st = op.store;
    AddRange(csr,st.phase_rowptr); AddRange(csr,st.phase_colidx); AddRange(csr,st.phase_values);
    AddRange(csr,st.stage_rowptr); AddRange(csr,st.stage_colidx); AddRange(csr,st.stage_values);
    AddRange(halo,op.send_indices); AddRange(halo,op.recv_slots);
    AddRange(halo,op.send_device); AddRange(halo,op.ghost_device); AddRange(halo,op.recv_device);
    AddRange(halo,op.send_host); AddRange(halo,op.ghost_host);
    AddRange(halo,st.send_indices); AddRange(halo,st.recv_slots);
    AddRange(halo,st.send_device); AddRange(halo,st.ghost_device); AddRange(halo,st.recv_device);
    AddRange(halo,st.send_host); AddRange(halo,st.ghost_host);
  }
  static std::uint64_t MergedBytes(Ranges r) {
    std::sort(r.begin(),r.end());
    std::uint64_t total = 0;
    std::uintptr_t b = 0, e = 0;
    bool open = false;
    for (const auto &x : r) {
      if (open && x.first <= e) { e = std::max(e,x.second); continue; }
      if (open) total += e-b;
      b = x.first; e = x.second; open = true;
    }
    if (open) total += e-b;
    return total;
  }
  static std::uint64_t CountBytes(const Operator &op) {   // host count/offset vectors
    return (op.send_counts.size()+op.recv_counts.size()+op.send_offsets.size()
            +op.recv_offsets.size())*sizeof(int);
  }
  static std::uint64_t Nonzeros(const Operator &op) {
    return op.phase_active ? op.values.size()-op.phase_masked_base_nnz+op.phase_values.size()
                           : op.values.size();
  }
  std::vector<Level> levels;
  Operator current_a;
  std::vector<int> current_rowptr;
  std::vector<GlobalIndex> current_columns;
  bool current_values_shared = true;
  bool fine_diagonal_dirty = false;
  bool hierarchy_phase_configured = false, terminal_phase_changes = false;
  // The only retained host matrix is the owned portion of the tiny terminal A.
  LocalCsr terminal_matrix;
  FineOperatorStats fine_updates;
  // FGMRES storage belongs to the hierarchy and is reused across solves. Each solve
  // still copies and projects its CURRENT rhs and recomputes the true residual.
  Vector krylov_b, krylov_r, krylov_w, krylov_x;
  std::vector<Vector> krylov_v, krylov_z;
  std::vector<Real> krylov_h, krylov_cosine, krylov_sine, krylov_g, krylov_y;
  int krylov_restart = 0;
  bool nullspace;
  // ATHENA_SG_PROJECT_FUSE=1 and the probe passed (CheckSwitches)
  bool fuse_projection = false;
  int rank = 0, nranks = 1;
  int matrix_applications = 0, vcycles = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
  mutable MPI_Datatype norm_type = MPI_DATATYPE_NULL;
  mutable MPI_Op norm_op = MPI_OP_NULL;
#endif
  // Only rank zero stores and solves the tiny dense coarse problem.
  std::vector<Real> coarse_lu, coarse_rhs;
  std::vector<int> coarse_pivots, coarse_counts, coarse_offsets;

#if MPI_PARALLEL_ENABLED
  explicit Impl(MPI_Comm input_comm) : nullspace(false) {
    MPI_Comm_dup(input_comm, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    CheckSwitches();
  }
  Impl(const HostHierarchy &h, MPI_Comm input_comm) : nullspace(h.constant_nullspace) {
    MPI_Comm_dup(input_comm, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    CheckSwitches();
    try {
      Initialize(h);
    } catch (...) {
      MPI_Comm_free(&comm);
      throw;
    }
  }
  Impl(const DistributedHierarchy &h, MPI_Comm input_comm) : nullspace(h.constant_nullspace) {
    MPI_Comm_dup(input_comm, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
    CheckSwitches();
    try {
      Initialize(h);
    } catch (...) {
      MPI_Comm_free(&comm);
      throw;
    }
  }
#else
  Impl() : nullspace(false) {}
  explicit Impl(const HostHierarchy &h) : nullspace(h.constant_nullspace) { Initialize(h); }
  explicit Impl(const DistributedHierarchy &h) : nullspace(h.constant_nullspace) {
    Initialize(h);
  }
#endif
  ~Impl() {
#if MPI_PARALLEL_ENABLED
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && norm_op != MPI_OP_NULL) MPI_Op_free(&norm_op);
    if (!finalized && norm_type != MPI_DATATYPE_NULL) MPI_Type_free(&norm_type);
    if (!finalized && comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
#endif
  }

#if MPI_PARALLEL_ENABLED
  // The MPI-pattern switches must agree on every rank: a mismatch would pair MPI_Alltoallv
  // with point-to-point halo messages, or Allreduce calls of different sizes.  Collective
  // (called right after MPI_Comm_dup); frees comm before throwing, since no destructor runs.
  void CheckSwitches() {
    const int a = HaloAlltoallv() ? 1 : 0, b = PhaseCollectivesOld() ? 1 : 0;
    const int c = PhaseForceOld() ? 1 : 0, d = ProjectFuse() ? 1 : 0;
    const int e = DeviceMpi() ? 1 : 0;
    int v[10] = {a, b, c, d, e, -a, -b, -c, -d, -e};
    MPI_Allreduce(MPI_IN_PLACE, v, 10, MPI_INT, MPI_MAX, comm);
    if (v[0] != -v[5] || v[1] != -v[6] || v[2] != -v[7] || v[3] != -v[8] || v[4] != -v[9]) {
      MPI_Comm_free(&comm);
      throw std::runtime_error("self-gravity: ATHENA_SG_HALO_ALLTOALLV / "
                               "ATHENA_SG_PHASE_COLLECTIVES / ATHENA_SG_PHASE_FORCE_OLD / "
                               "ATHENA_SG_PROJECT_FUSE / ATHENA_SG_DEVICE_MPI must be set alike "
                               "on every rank");
    }
    // the fused projection only where the probe finds rank-independent sums (collective:
    // every rank has the same ATHENA_SG_PROJECT_FUSE, checked above)
    fuse_projection = ProjectFuse() && ProbeAllreduceIdentical();
    if (ProjectFuse() && !fuse_projection && rank == 0) {
      static bool told = false;
      if (!told) {
        told = true;
        std::cout << "# NOTE (self-gravity): ATHENA_SG_PROJECT_FUSE=1 ignored: MPI_Allreduce "
                  << "returned different sums on different ranks in the probe; every projection "
                  << "completes before the next product (the default order)" << std::endl;
      }
    }
  }
  // Collective.  Sums of probe values with one MPI_Allreduce each, as the projection
  // does (one MPI_ATHENA_REAL, MPI_SUM), then one exact integer reduction of the result bits:
  // true if every rank received bitwise the same sums.  The values are the UCC 1.3 counterexample
  // ({1, -1, 2^-54, 0} by rank modulo 4) in four rotations and a set of near-cancelling numbers.
  bool ProbeAllreduceIdentical() const {
    const Real tiny = std::ldexp(Real(1), -54);
    const Real base[4] = {Real(1), Real(-1), tiny, Real(0)};
    constexpr int kProbes = 5;
    std::int64_t bits[2*kProbes];
    for (int k = 0; k < kProbes; ++k) {
      Real local = k < 4 ? base[(rank+k)%4]
                         : (rank%2 == 0 ? Real(1) : Real(-1))*(Real(1)+rank*std::ldexp(Real(1),-52));
      Real sum = 0;
      MPI_Allreduce(&local, &sum, 1, MPI_ATHENA_REAL, MPI_SUM, comm);
      std::int64_t b = 0;
      std::memcpy(&b, &sum, sizeof(sum) < sizeof(b) ? sizeof(sum) : sizeof(b));
      if (ProjectFuseProbeTest() && rank == 0) b ^= 1;
      bits[k] = b; bits[kProbes+k] = -b;
    }
    // min and max of every result's bits at once: max(b) and max(-b) = -min(b)
    std::int64_t r[2*kProbes];
    MPI_Allreduce(bits, r, 2*kProbes, MPI_INT64_T, MPI_MAX, comm);
    bool same = true;
    for (int k = 0; k < kProbes; ++k) same = same && r[k] == -r[kProbes+k];
    return same;
  }
#endif

  void Require(bool local_good, const char *message) const {
    int good = local_good ? 1 : 0;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &good, 1, MPI_INT, MPI_MIN, comm);
#endif
    if (!good) throw std::runtime_error(std::string("self-gravity: ")+message);
  }

  std::vector<GlobalIndex> Partition(
      const std::vector<std::vector<GlobalIndex>> &supplied, int l, GlobalIndex n) {
    GlobalIndex lo = n, hi = n;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &lo, 1, MPI_INT64_T, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &hi, 1, MPI_INT64_T, MPI_MAX, comm);
#endif
    Require(n > 0 && lo == hi, "inconsistent global matrix dimensions");
    std::vector<GlobalIndex> part;
    if (l < static_cast<int>(supplied.size())) part = supplied[l];
    if (part.empty()) {
      part.resize(nranks+1);
      for (int r = 0; r <= nranks; ++r) {
        part[r] = (n/nranks)*r+((n%nranks)*r)/nranks;
      }
    }
    Require(part.size() == static_cast<size_t>(nranks+1) && part.front() == 0
        && part.back() == n && std::is_sorted(part.begin(), part.end()),
        "invalid row partition");
#if MPI_PARALLEL_ENABLED
    auto minimum = part, maximum = part;
    MPI_Allreduce(MPI_IN_PLACE, minimum.data(), nranks+1, MPI_INT64_T, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, maximum.data(), nranks+1, MPI_INT64_T, MPI_MAX, comm);
    Require(minimum == maximum, "row partitions differ across ranks");
#endif
    Require(part[rank+1]-part[rank] <= std::numeric_limits<int>::max(),
            "local row count exceeds device index capacity");
    return part;
  }

  void CheckLocal(const LocalCsr &a, const std::vector<GlobalIndex> &rows,
                  const std::vector<GlobalIndex> &cols) const {
    bool good = a.global_rows > 0 && a.global_cols > 0
        && !a.rowptr.empty()
        && a.rowptr.size()-1 <= static_cast<size_t>(std::numeric_limits<int>::max())
        && a.values.size() <= static_cast<size_t>(std::numeric_limits<int>::max())
        && a.colidx.size() == a.values.size()
        && a.global_rows == rows.back() && a.global_cols == cols.back()
        && a.row_begin == rows[rank]
        && a.rowptr.size()-1 == static_cast<size_t>(rows[rank+1]-rows[rank]);
    if (good) {
      good = a.rowptr.front() == 0
          && a.rowptr.back() == static_cast<int>(a.values.size());
    }
    if (good) {
      // sorted offsets, columns in range and finite values, checked in chunks
      SetupLap lap(SetupTimes().check);
      const std::size_t steps = a.rowptr.size()-1, nnz = a.values.size();
      int workers = 1, chunks = 1;
      SetupChunks(steps+nnz,workers,chunks);
      std::vector<char> chunk_good(chunks,1);
      ParallelChunks(workers,chunks,[&](int c) {
        bool g = true;
        for (std::size_t i = steps*c/chunks; g && i < steps*(c+1)/chunks; ++i)
          g = !(a.rowptr[i+1] < a.rowptr[i]);
        for (std::size_t k = nnz*c/chunks; g && k < nnz*(c+1)/chunks; ++k)
          g = a.colidx[k] >= 0 && a.colidx[k] < a.global_cols && std::isfinite(a.values[k]);
        chunk_good[c] = g;
      });
      good = std::find(chunk_good.begin(),chunk_good.end(),0) == chunk_good.end();
    }
    Require(good, "invalid owned CSR shape, offsets, columns, values, or ownership");
  }

  // The summed diagonal entries of the owned rows of a (in each row's entry order),
  // computed in row chunks.
  static std::vector<Real> DiagonalSums(const LocalCsr &a) {
    const int n = a.LocalRows();
    std::vector<Real> diagonal(n);
    int workers = 1, chunks = 1;
    SetupChunks(a.values.size(),workers,chunks);
    ParallelChunks(workers,chunks,[&](int c) {
      const int first = static_cast<int>(static_cast<std::int64_t>(n)*c/chunks);
      const int last = static_cast<int>(static_cast<std::int64_t>(n)*(c+1)/chunks);
      for (int i = first; i < last; ++i) {
        Real d = 0;
        for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) {
          if (a.colidx[k] == a.row_begin+i) d += a.values[k];
        }
        diagonal[i] = d;
      }
    });
    return diagonal;
  }

  static LocalCsr Slice(const HostCsr &a, const std::vector<GlobalIndex> &part,
                        int owner) {
    LocalCsr local;
    local.global_rows = a.rows;
    local.global_cols = a.cols;
    local.row_begin = part[owner];
    const int begin = part[owner], end = part[owner+1];
    const int offset = a.rowptr[begin], nnz = a.rowptr[end]-offset;
    local.rowptr.resize(end-begin+1);
    for (int i = begin; i <= end; ++i) local.rowptr[i-begin] = a.rowptr[i]-offset;
    local.colidx.assign(a.colidx.begin()+offset, a.colidx.begin()+offset+nnz);
    local.values.assign(a.values.begin()+offset, a.values.begin()+offset+nnz);
    return local;
  }

  void BuildHalo(Operator &op, const std::vector<GlobalIndex> &ghosts) {
    const auto &inpart = op.input_partition;
    const GlobalIndex input_begin = inpart[rank], input_end = inpart[rank+1];
    Require(ghosts.size() <= static_cast<size_t>(std::numeric_limits<int>::max()-op.inputs),
            "local plus halo vector exceeds device index capacity");
    op.ghost_ids = ghosts;
    op.recv_counts.assign(nranks,0);
    op.send_counts.assign(nranks,0);
    op.recv_offsets.assign(nranks+1,0);
    op.send_offsets.assign(nranks+1,0);
    for (GlobalIndex j : ghosts) {
      const int owner = std::upper_bound(inpart.begin(),inpart.end(),j)-inpart.begin()-1;
      ++op.recv_counts[owner];
    }
#if MPI_PARALLEL_ENABLED
    MPI_Alltoall(op.recv_counts.data(),1,MPI_INT,op.send_counts.data(),1,MPI_INT,comm);
#endif
    GlobalIndex sends = 0, receives = 0;
    for (int r = 0; r < nranks; ++r) {
      sends += op.send_counts[r]; receives += op.recv_counts[r];
    }
    Require(sends <= std::numeric_limits<int>::max()
        && receives <= std::numeric_limits<int>::max(),"halo exceeds MPI count capacity");
    op.send_peers.clear(); op.recv_peers.clear();
    for (int r = 0; r < nranks; ++r) {
      op.recv_offsets[r+1] = op.recv_offsets[r]+op.recv_counts[r];
      op.send_offsets[r+1] = op.send_offsets[r]+op.send_counts[r];
      if (op.send_counts[r] > 0) op.send_peers.push_back(r);
      if (op.recv_counts[r] > 0) op.recv_peers.push_back(r);
    }
#if MPI_PARALLEL_ENABLED
    op.halo_requests.assign(op.send_peers.size()+op.recv_peers.size(),MPI_REQUEST_NULL);
#endif
    // Union ghost slots deliberately retain base indices, so their order need
    // not match MPI owner groups. Receive into grouped order, then scatter.
    std::vector<GlobalIndex> grouped(ghosts.size()), requested(op.send_offsets.back());
    std::vector<int> slots(ghosts.size());
    auto next = op.recv_offsets;
    bool identity = true;
    for (size_t s = 0; s < ghosts.size(); ++s) {
      const int owner = std::upper_bound(inpart.begin(),inpart.end(),ghosts[s])-inpart.begin()-1;
      const int q = next[owner]++;
      grouped[q] = ghosts[s]; slots[q] = s; identity = identity && q == static_cast<int>(s);
    }
#if MPI_PARALLEL_ENABLED
    MPI_Alltoallv(grouped.data(),op.recv_counts.data(),op.recv_offsets.data(),MPI_INT64_T,
                  requested.data(),op.send_counts.data(),op.send_offsets.data(),MPI_INT64_T,comm);
#endif
    std::vector<int> send_indices(requested.size()); bool valid = true;
    for (size_t k = 0; k < requested.size(); ++k) {
      valid = valid && requested[k] >= input_begin && requested[k] < input_end;
      if (valid) send_indices[k] = static_cast<int>(requested[k]-input_begin);
    }
    Require(valid,"halo request addressed to a nonowner");
    if (PhaseAllocOld()) {
      op.send_indices = DeviceCopy(send_indices,"gravity_halo_indices");
      op.send_device = Vector("gravity_halo_send",requested.size());
      op.ghost_device = Vector("gravity_halo_recv",ghosts.size());
      op.recv_device = identity ? op.ghost_device : Vector("gravity_grouped_recv",ghosts.size());
      op.recv_slots = identity ? Indices() : DeviceCopy(slots,"gravity_union_slots");
      op.send_host = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,"gravity_halo_host_send"),
          requested.size());
      op.ghost_host = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,"gravity_halo_host_recv"),ghosts.size());
      return;
    }
    // the same views as subviews of the operator's stores; the device vectors zeroed
    // as the freshly constructed ones were.  The kernels and copies of an earlier halo plan
    // that read these stores were ordered before this point (the copies above fence).
    auto &st = op.store;
    op.send_indices = StoreCopy(st.send_indices,send_indices,"gravity_halo_indices");
    op.send_device = ZeroFromStore(st.send_device,requested.size(),"gravity_halo_send");
    op.ghost_device = ZeroFromStore(st.ghost_device,ghosts.size(),"gravity_halo_recv");
    op.recv_device = identity ? op.ghost_device
        : ZeroFromStore(st.recv_device,ghosts.size(),"gravity_grouped_recv");
    op.recv_slots = identity ? Indices() : StoreCopy(st.recv_slots,slots,"gravity_union_slots");
    op.send_host = FromStore(st.send_host,requested.size(),"gravity_halo_host_send");
    op.ghost_host = FromStore(st.ghost_host,ghosts.size(),"gravity_halo_host_recv");
  }

  // The sorted unique columns of a outside [lo, hi): sorted unique lists of entry chunks,
  // merged pairwise.
  static std::vector<GlobalIndex> GhostColumns(const LocalCsr &a, GlobalIndex lo,
                                               GlobalIndex hi) {
    const std::size_t nnz = a.colidx.size();
    int workers = 1, chunks = 1;
    SetupChunks(nnz,workers,chunks);
    std::vector<std::vector<GlobalIndex>> pieces(chunks);
    ParallelChunks(workers,chunks,[&](int c) {
      auto &g = pieces[c];
      for (std::size_t k = nnz*c/chunks; k < nnz*(c+1)/chunks; ++k) {
        const GlobalIndex j = a.colidx[k];
        if (j < lo || j >= hi) g.push_back(j);
      }
      std::sort(g.begin(),g.end());
      g.erase(std::unique(g.begin(),g.end()),g.end());
    });
    for (int width = 1; width < chunks; width *= 2) {
      const int pairs = (chunks+2*width-1)/(2*width);
      ParallelChunks(workers,pairs,[&](int q) {
        const int left = 2*width*q, right = left+width;
        if (right >= chunks) return;
        auto &x = pieces[left];
        auto &y = pieces[right];
        std::vector<GlobalIndex> merged;
        merged.reserve(x.size()+y.size());
        std::set_union(x.begin(),x.end(),y.begin(),y.end(),std::back_inserter(merged));
        x.swap(merged);
        std::vector<GlobalIndex>().swap(y);
      });
    }
    return std::move(pieces[0]);
  }

  Operator Build(const LocalCsr &a, const std::vector<GlobalIndex> &outpart,
                 const std::vector<GlobalIndex> &inpart) {
    Operator op;
    op.rows = a.LocalRows(); op.inputs = static_cast<int>(inpart[rank+1]-inpart[rank]);
    op.global_rows = a.global_rows; op.global_cols = a.global_cols; op.row_begin = a.row_begin;
    op.input_partition = inpart;
    const GlobalIndex lo = inpart[rank], hi = inpart[rank+1];
    std::vector<GlobalIndex> ghosts;
    {
      SetupLap lap(SetupTimes().ghosts);
      ghosts = GhostColumns(a,lo,hi);
    }
    {
      SetupLap lap(SetupTimes().halo);
      BuildHalo(op,ghosts);
    }
    std::vector<int> columns;
    {
      SetupLap lap(SetupTimes().columns);
      const std::size_t nnz = a.colidx.size();
      columns.resize(nnz);
      int workers = 1, chunks = 1;
      SetupChunks(nnz,workers,chunks);
      const int inputs = op.inputs;
      ParallelChunks(workers,chunks,[&](int c) {
        for (std::size_t k = nnz*c/chunks; k < nnz*(c+1)/chunks; ++k) {
          const GlobalIndex j = a.colidx[k];
          columns[k] = j >= lo && j < hi ? static_cast<int>(j-lo) :
              inputs+static_cast<int>(std::lower_bound(ghosts.begin(),ghosts.end(),j)
                                      -ghosts.begin());
        }
      });
    }
    SetupLap lap(SetupTimes().upload);
    op.rowptr = DeviceCopy(a.rowptr,"gravity_csr_rows");
    op.colidx = DeviceCopy(columns,"gravity_csr_columns");
    op.values = DeviceCopy(a.values,"gravity_csr_values");
    return op;
  }

  void ConfigurePhase(Operator &op, const std::vector<int> &rows) {
    const bool valid = rows.size() <= static_cast<size_t>(op.rows)
        && std::is_sorted(rows.begin(),rows.end())
        && std::adjacent_find(rows.begin(),rows.end()) == rows.end()
        && (rows.empty() || (rows.front() >= 0 && rows.back() < op.rows));
    Require(valid,"invalid sorted local phase row list");
    Require(!op.phase_configured || rows == op.phase_rows_host,
            "phase row set changed; construct a fresh base operator");
    if (op.phase_configured) return;
    op.phase_rows_host = rows;
    op.phase_base_ghosts = op.ghost_ids;
    op.phase_rows = DeviceCopy(rows,"gravity_phase_rows");
    op.phase_map = Indices("gravity_phase_map",op.rows);
    Kokkos::deep_copy(op.phase_map,-1);
    auto indices = op.phase_rows;
    auto mapping = op.phase_map;
    auto rp = op.rowptr;
    const int count = rows.size();
    Kokkos::parallel_for("gravity_configure_phase_rows",Kokkos::RangePolicy<DevExeSpace>(0,count),
                         KOKKOS_LAMBDA(int j) { mapping(indices(j)) = j; });
    std::uint64_t masked = 0;
    Kokkos::parallel_reduce("gravity_phase_base_nnz",Kokkos::RangePolicy<DevExeSpace>(0,count),
    KOKKOS_LAMBDA(int j, std::uint64_t &sum) { sum += rp(indices(j)+1)-rp(indices(j)); },masked);
    op.phase_masked_base_nnz = masked;
    op.phase_base_sorted = !GhostMapOld()
        && std::is_sorted(op.phase_base_ghosts.begin(),op.phase_base_ghosts.end())
        && std::adjacent_find(op.phase_base_ghosts.begin(),op.phase_base_ghosts.end())
           == op.phase_base_ghosts.end();
    if (!rows.empty() && !op.phase_base_sorted) BuildGhostSlots(op);
    op.phase_configured = true;
  }

  void BuildGhostSlots(Operator &op) {
    auto &slots = op.phase_ghost_slots;
    slots.clear();
    slots.reserve(op.ghost_ids.size());
    for (size_t k = 0; k < op.ghost_ids.size(); ++k) slots.emplace(op.ghost_ids[k],k);
    op.phase_updates.host_map_entries = slots.size();
    op.phase_updates.host_map_buckets = slots.bucket_count();
    op.phase_updates.host_map_peak_bytes = std::max(op.phase_updates.host_map_peak_bytes,
        static_cast<std::uint64_t>(slots.bucket_count()*sizeof(void*)
            +slots.size()*sizeof(decltype(op.phase_ghost_slots)::value_type)));
  }

  // Host only: validation, pattern comparison, next ghost list and remapped columns of a
  // compact update, from op's state after its previous update (op is only read).
  PhaseRowPlan Plan(const Operator &op, const CsrRowUpdate &a, bool smoothing_diagonal,
                    int workers = 1) const {
    PhaseRowPlan plan;
    plan.prepared = true;
    plan.version = op.phase_updates.updates;
    bool valid = op.phase_configured && a.global_rows == op.global_rows
        && a.global_cols == op.global_cols && a.row_begin == op.row_begin
        && a.rows == op.phase_rows_host && a.rowptr.size() == a.rows.size()+1
        && a.colidx.size() == a.values.size()
        && a.values.size() <= static_cast<size_t>(std::numeric_limits<int>::max());
    if (valid) {
      valid = a.rowptr.front() == 0 && a.rowptr.back() == static_cast<int>(a.values.size())
          && std::is_sorted(a.rowptr.begin(),a.rowptr.end());
    }
    // Consecutive row chunks on up to `workers` host threads of the calling thread: every
    // chunk writes only its own rows or its own partial result, combined in chunk order, so
    // the plan does not depend on the count.
    const size_t nrows = a.rows.size();
    const int chunks = static_cast<int>(std::max<size_t>(1,std::min<size_t>(
        16*static_cast<size_t>(std::max(1,workers)),nrows/2048+1)));
    auto first = [&](int c) { return nrows*c/chunks; };
    auto parallel = [&](const auto &task) { ParallelChunks(std::max(1,workers),chunks,task); };
    if (valid) {
      std::vector<char> good(chunks,1);
      parallel([&](int c) {
        const size_t j0 = first(c), j1 = first(c+1);
        bool ok = true;
        for (int k = a.rowptr[j0]; ok && k < a.rowptr[j1]; ++k) {
          ok = a.colidx[k] >= 0 && a.colidx[k] < op.global_cols && std::isfinite(a.values[k]);
        }
        if (smoothing_diagonal) for (size_t j = j0; ok && j < j1; ++j) {
          Real diagonal = 0;
          for (int k = a.rowptr[j]; k < a.rowptr[j+1]; ++k) {
            if (a.colidx[k] == a.row_begin+a.rows[j]) diagonal += a.values[k];
          }
          ok = diagonal != 0 && std::isfinite(diagonal) && std::isfinite(Real(1)/diagonal);
        }
        good[c] = ok;
      });
      valid = std::find(good.begin(),good.end(),0) == good.end();
    }
    plan.valid = valid;
    if (!valid) return plan;
    // the stored pattern: the update's own, or (padding) per row the union with the current row
    // when it exceeds the new row by at most `pad` entries (zeros at the columns not in the new row)
    const int pad = PhasePatternPadding();
    const std::vector<int> *rowptr = &a.rowptr;
    const std::vector<GlobalIndex> *colidx = &a.colidx;
    const auto &rp = op.phase_rowptr_host; const auto &ci = op.phase_columns_host;
    if (pad <= 0 || rp.size() != a.rowptr.size()) {
      plan.same = a.rowptr == rp && a.colidx == ci;
      if (plan.same) return plan;
    } else {
      // per row: 0 the same pattern, 1 held by the stored row (a subset with at most `pad`
      // stored zeros), 2 changed
      std::vector<char> kind(nrows);
      parallel([&](int c) {
        for (size_t j = first(c); j < first(c+1); ++j) {
          const int o0 = rp[j], o1 = rp[j+1], n0 = a.rowptr[j], n1 = a.rowptr[j+1];
          if (o1-o0 == n1-n0 && std::equal(ci.begin()+o0,ci.begin()+o1,a.colidx.begin()+n0)) {
            kind[j] = 0;
          } else {
            kind[j] = o1-o0 <= n1-n0+pad && std::includes(ci.begin()+o0,ci.begin()+o1,
                a.colidx.begin()+n0,a.colidx.begin()+n1) ? 1 : 2;
          }
        }
      });
      const bool holds = std::find(kind.begin(),kind.end(),2) == kind.end();
      if (holds && std::find(kind.begin(),kind.end(),1) == kind.end()) {
        plan.same = true;   // the update's own pattern is the stored one
        return plan;
      }
      if (holds) {   // values only, scattered into the stored rows
        plan.same = true; plan.padded = true;
        plan.values.resize(ci.size());
        parallel([&](int c) {
          for (size_t j = first(c); j < first(c+1); ++j) {
            const int o0 = rp[j], o1 = rp[j+1], n0 = a.rowptr[j], n1 = a.rowptr[j+1];
            if (kind[j] == 0) {
              std::copy(a.values.begin()+n0,a.values.begin()+n1,plan.values.begin()+o0);
              continue;
            }
            std::fill(plan.values.begin()+o0,plan.values.begin()+o1,Real(0));
            for (int k = n0, m = o0; k < n1; ++k) {
              while (ci[m] != a.colidx[k]) ++m;
              plan.values[m++] = a.values[k];
            }
          }
        });
        return plan;
      }
      // a changed pattern: per chunk the stored rows (the union where it is within `pad` of the
      // new row, else the new row), then joined in order
      struct Part { std::vector<int> length; std::vector<GlobalIndex> colidx; std::vector<Real> values; };
      std::vector<Part> parts(chunks);
      parallel([&](int c) {
        auto &part = parts[c];
        std::vector<GlobalIndex> merged;
        for (size_t j = first(c); j < first(c+1); ++j) {
          const int o0 = rp[j], o1 = rp[j+1], n0 = a.rowptr[j], n1 = a.rowptr[j+1];
          const size_t before = part.colidx.size();
          // both rows are sorted by column without repeats (Append)
          merged.clear();
          std::set_union(ci.begin()+o0,ci.begin()+o1,a.colidx.begin()+n0,a.colidx.begin()+n1,
                         std::back_inserter(merged));
          if (merged.size() <= static_cast<size_t>(n1-n0+pad)) {
            int k = n0;
            for (GlobalIndex col : merged) {
              part.colidx.push_back(col);
              part.values.push_back(k < n1 && a.colidx[k] == col ? a.values[k++] : Real(0));
            }
          } else {
            part.colidx.insert(part.colidx.end(),a.colidx.begin()+n0,a.colidx.begin()+n1);
            part.values.insert(part.values.end(),a.values.begin()+n0,a.values.begin()+n1);
          }
          part.length.push_back(static_cast<int>(part.colidx.size()-before));
        }
      });
      size_t nnz = 0;
      std::vector<size_t> offset(chunks+1,0);
      for (int c = 0; c < chunks; ++c) offset[c+1] = offset[c]+parts[c].colidx.size();
      nnz = offset[chunks];
      if (nnz > static_cast<size_t>(std::numeric_limits<int>::max())) {
        plan.valid = false; return plan;
      }
      plan.padded = true;
      plan.rowptr.resize(nrows+1); plan.rowptr[0] = 0;
      plan.colidx.resize(nnz); plan.values.resize(nnz);
      parallel([&](int c) {
        auto &part = parts[c];
        int at = static_cast<int>(offset[c]);
        for (size_t i = 0; i < part.length.size(); ++i) {
          at += part.length[i]; plan.rowptr[first(c)+i+1] = at;
        }
        std::copy(part.colidx.begin(),part.colidx.end(),plan.colidx.begin()+offset[c]);
        std::copy(part.values.begin(),part.values.end(),plan.values.begin()+offset[c]);
        part = Part{};
      });
      rowptr = &plan.rowptr; colidx = &plan.colidx;
    }
    const auto &np = *rowptr; const auto &nc = *colidx;
    // (diagnostics) the rows whose stored pattern changes, and up to four of them (first rows)
    if (rp.size() == np.size()) {
      std::vector<std::uint64_t> changed(chunks,0);
      std::vector<std::vector<std::array<GlobalIndex,5>>> sample(chunks);
      parallel([&](int c) {
        for (size_t j = first(c); j < first(c+1); ++j) {
          const int n0 = rp[j+1]-rp[j], n1 = np[j+1]-np[j];
          if (n0 == n1 && std::equal(ci.begin()+rp[j],ci.begin()+rp[j+1],nc.begin()+np[j])) continue;
          ++changed[c];
          if (sample[c].size() < 4) {
            const auto o0 = ci.begin()+rp[j], o1 = ci.begin()+rp[j+1];
            const auto c0 = nc.begin()+np[j], c1 = nc.begin()+np[j+1];
            GlobalIndex added = -1, removed = -1;
            for (auto x = c0; x != c1 && added < 0; ++x) if (std::find(o0,o1,*x) == o1) added = *x;
            for (auto x = o0; x != o1 && removed < 0; ++x) if (std::find(c0,c1,*x) == c1) removed = *x;
            sample[c].push_back({{static_cast<GlobalIndex>(j),n0,n1,added,removed}});
          }
        }
      });
      for (int c = 0; c < chunks; ++c) {
        plan.changed_rows += changed[c];
        for (const auto &x : sample[c]) if (plan.changed_sample.size() < 4) plan.changed_sample.push_back(x);
      }
    }
    const auto begin = op.input_partition[rank], end = op.input_partition[rank+1];
    const int base = static_cast<int>(op.phase_base_ghosts.size());
    // the slot of a base ghost column (its position in the base list, the prefix of the current
    // ghost list), -1 for any other column (read only: safe on several threads)
    auto base_slot = [&op,base](GlobalIndex j) -> int {
      if (op.phase_base_sorted) {
        const auto &g = op.phase_base_ghosts;
        const auto it = std::lower_bound(g.begin(),g.end(),j);
        return it != g.end() && *it == j ? static_cast<int>(it-g.begin()) : -1;
      }
      const auto slot = op.phase_ghost_slots.find(j);
      return slot != op.phase_ghost_slots.end() && slot->second < base ? slot->second : -1;
    };
    plan.ghosts = op.phase_base_ghosts;
    std::vector<std::vector<GlobalIndex>> extras(chunks);
    parallel([&](int c) {
      auto &extra = extras[c];
      for (int k = np[first(c)]; k < np[first(c+1)]; ++k) {
        const GlobalIndex j = nc[k];
        if (j >= begin && j < end) continue;
        if (base_slot(j) < 0) extra.push_back(j);
      }
      std::sort(extra.begin(),extra.end());
      extra.erase(std::unique(extra.begin(),extra.end()),extra.end());
    });
    std::vector<GlobalIndex> extra;
    for (auto &part : extras) extra.insert(extra.end(),part.begin(),part.end());
    std::sort(extra.begin(),extra.end());
    extra.erase(std::unique(extra.begin(),extra.end()),extra.end());
    plan.ghosts.insert(plan.ghosts.end(),extra.begin(),extra.end());
    // The slot of a ghost is its position in the next list: a base ghost keeps its
    // current (prefix) slot, an extra one follows the base in sorted order.
    plan.columns.resize(nc.size());
    parallel([&](int c) {
      for (int k = np[first(c)]; k < np[first(c+1)]; ++k) {
        const GlobalIndex j = nc[k];
        if (j >= begin && j < end) { plan.columns[k] = static_cast<int>(j-begin); continue; }
        const int found = base_slot(j);
        const int position = found >= 0 ? found
            : base+static_cast<int>(std::lower_bound(extra.begin(),extra.end(),j)-extra.begin());
        plan.columns[k] = op.inputs+position;
      }
    });
    return plan;
  }

  // the plan of a phase update: the given one if it was prepared for the operator's current
  // version, else computed into `computed` (host only, no MPI)
  const PhaseRowPlan *PhasePlan(const Operator &op, const CsrRowUpdate &a, bool smoothing_diagonal,
                                const PhaseRowPlan *given, PhaseRowPlan &computed) const {
    if (given == nullptr || !given->prepared || given->version != op.phase_updates.updates) {
      computed = Plan(op,a,smoothing_diagonal);
      return &computed;
    }
    return given;
  }
  // known (xnode): {any rank invalid, any rank's ghost list changed} of this operator, already
  // reduced over all ranks by the caller (one collective for many operators); nullptr: the two
  // collectives here
  bool UpdatePhase(Operator &op, const CsrRowUpdate &a, bool smoothing_diagonal = false,
                   const PhaseRowPlan *given = nullptr, const int *known = nullptr) {
    auto &times = PhaseTimes();
    double clock = HostSeconds();
    auto lap = [&clock](double &bucket) {
      const double now = HostSeconds(); bucket += now-clock; clock = now;
    };
    ++times.calls;
    PhaseRowPlan computed;
    const PhaseRowPlan *plan = given;
    if (plan == nullptr || !plan->prepared || plan->version != op.phase_updates.updates) {
      computed = Plan(op,a,smoothing_diagonal);
      plan = &computed;
    } else {
      ++times.planned;
    }
    lap(times.validate);
    if (known == nullptr) {
      Require(plan->valid,"invalid compact phase CSR, ownership, row set, or smoothing diagonal");
    } else if (known[0]) {
      throw std::runtime_error("self-gravity: invalid compact phase CSR, ownership, row set, "
                               "or smoothing diagonal");
    }
    lap(times.mpi);
    const bool same = plan->same;
    if (!same) ++op.phase_updates.local_pattern_changes;
    const auto &ghosts = same ? op.ghost_ids : plan->ghosts;
    const int local_ghost_change = ghosts != op.ghost_ids;
    int global_ghost_change = local_ghost_change;
    lap(times.ghosts);
#if MPI_PARALLEL_ENABLED
    // A remote rank may change requests addressed to this rank even when our
    // own CSR/ghost list is unchanged. Every rank must rebuild the send plan.
    if (known == nullptr) MPI_Allreduce(MPI_IN_PLACE,&global_ghost_change,1,MPI_INT,MPI_MAX,comm);
    else global_ghost_change = known[1];
#endif
    lap(times.mpi);
    op.phase_updates.local_ghost_changes += local_ghost_change;
    op.phase_updates.global_ghost_changes += global_ghost_change;
    if (global_ghost_change) {
      BuildHalo(op,ghosts);
      ++op.phase_updates.halo_rebuilds;
    } else {
      ++op.phase_updates.halo_reuses;
    }
    lap(times.halo);
    if (local_ghost_change && !op.phase_base_sorted) BuildGhostSlots(op);
    lap(times.ghosts);
    // No MPI occurs below this point: CSR/value uploads are local decisions.
    if (same) {
      // a kept pinned buffer instead of a fresh zero-initialised mirror per update (22
      // operators per phase update: allocation, memset and a pageable copy each)
      const size_t nv = op.phase_values.extent(0);
      Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace> host;
      if (PhaseAllocOld()) {
        if (op.phase_values_pinned.extent(0) != nv) {
          op.phase_values_pinned = Kokkos::View<Real*, Kokkos::SharedHostPinnedSpace>(
              Kokkos::view_alloc(Kokkos::WithoutInitializing,"gravity_phase_values_pinned"),nv);
        }
        host = op.phase_values_pinned;
      } else {   // a store with capacity: no new pinned allocation when nv changes
        host = FromStore(op.phase_values_pinned,nv,"gravity_phase_values_pinned");
      }
      const auto &values = plan->padded ? plan->values : a.values;   // (padded: current pattern)
      const size_t nc = std::min(nv,values.size());
      if (PhaseAllocOld()) {
        for (size_t k = 0; k < nc; ++k) host(k) = values[k];
      } else {
        StageCopy(host.data(),values.data(),nc);   // the same bytes, several threads
      }
      for (size_t k = nc; k < nv; ++k) host(k) = 0;   // (as the zero-initialised mirror did)
      // queued on the device stream; the caller fences once per update before the host
      // writes the pinned buffer again (UpdateHierarchyPhaseRows, DistributedOperator)
      if (PhaseAllocOld()) Kokkos::deep_copy(op.phase_values,host);
      else Kokkos::deep_copy(DevExeSpace(),op.phase_values,host);
      ++op.phase_updates.value_updates;
      ++times.value_only;
      lap(times.upload);
    } else {
      const auto &columns = plan->columns;
      // the stored pattern: the update's own, or its padded union with the current one
      const auto &rowptr = plan->padded ? plan->rowptr : a.rowptr;
      const auto &colidx = plan->padded ? plan->colidx : a.colidx;
      const auto &values = plan->padded ? plan->values : a.values;
      lap(times.remap);
      if (PhaseAllocOld()) {
        if (op.phase_rowptr.size() == rowptr.size()) {
          auto host = Kokkos::create_mirror_view(op.phase_rowptr);
          for (size_t k = 0; k < rowptr.size(); ++k) host(k) = rowptr[k];
          Kokkos::deep_copy(op.phase_rowptr,host);
        } else {
          op.phase_rowptr = DeviceCopy(rowptr,"gravity_phase_rowptr");
        }
        op.phase_colidx = DeviceCopy(columns,"gravity_phase_columns");
        op.phase_values = DeviceCopy(values,"gravity_phase_values");
      } else {
        // the same contents in the operator's stores; the copies are queued on the
        // device stream after the kernels that still read the previous pattern, and the
        // caller fences once per update before the pinned stages are written again
        auto &st = op.store;
        op.phase_rowptr = StagedCopy(st.phase_rowptr,st.stage_rowptr,rowptr,
                                     "gravity_phase_rowptr","gravity_phase_stage_rowptr");
        op.phase_colidx = StagedCopy(st.phase_colidx,st.stage_colidx,columns,
                                     "gravity_phase_columns","gravity_phase_stage_columns");
        op.phase_values = StagedCopy(st.phase_values,st.stage_values,values,
                                     "gravity_phase_values","gravity_phase_stage_values");
      }
      op.phase_rowptr_host = rowptr;
      op.phase_columns_host = colidx;
      ++op.phase_updates.structure_rebuilds;
      lap(times.upload);
    }
    op.phase_active = true;
    ++op.phase_updates.updates;
    return same != 0;
  }

  // the fine phase update with its uploads still queued (internal: the public entries
  // complete them with a PhaseUploadFence)
  void UpdateFineQueued(const CsrRowUpdate &a, const PhaseRowPlan *plan, const int *known) {
    const bool smooth = levels.size() > 1;
    const bool same = UpdatePhase(current_a,a,smooth,plan,known);
    if (smooth) UpdatePhaseDiagonal(current_a,levels[0].inverse_diagonal,fine_diagonal_dirty);
    fine_diagonal_dirty = false;
    ++fine_updates.updates;
    if (same) ++fine_updates.value_updates;
    else ++fine_updates.structure_rebuilds;
  }

  void UpdatePhaseDiagonal(Operator &op, const Vector &inverse, bool all_rows = false) {
    // Diagonals were validated collectively with the compact CSR before upload.
    // Device compact columns use the owned input index for each diagonal entry.
    auto rows = op.phase_rows;
    auto rp = op.phase_rowptr;
    auto ci = op.phase_colidx;
    auto av = op.phase_values;
    if (all_rows) {
      // Full outer-only updates can change nonphase rows too. On re-entering
      // compact mode, rebuild the complete effective fine smoothing diagonal.
      // This volume kernel and its validation are absent from regular updates.
      auto base_rp = op.rowptr;
      auto base_ci = op.colidx;
      auto base_av = op.values;
      auto mapping = op.phase_map;
      const int n = op.rows;
      int invalid = 0;
      Kokkos::parallel_reduce("gravity_full_phase_diagonal",
      Kokkos::RangePolicy<DevExeSpace>(0,n),KOKKOS_LAMBDA(int i, int &bad) {
        const int j = mapping(i);
        Real diagonal = 0;
        if (j >= 0) {
          for (int k = rp(j); k < rp(j+1); ++k) if (ci(k) == i) diagonal += av(k);
        } else {
          for (int k = base_rp(i); k < base_rp(i+1); ++k)
            if (base_ci(k) == i) diagonal += base_av(k);
        }
        const Real inv = diagonal == 0 ? Real(0) : Real(1)/diagonal;
        if (diagonal == 0 || !Kokkos::isfinite(diagonal) || !Kokkos::isfinite(inv)) {
          ++bad;
        } else {
          inverse(i) = inv;
        }
      },invalid);
      Require(invalid == 0,"invalid current fine smoothing diagonal after full update");
      return;
    }
    const int count = op.phase_rows_host.size();
    Kokkos::parallel_for("gravity_phase_diagonal",Kokkos::RangePolicy<DevExeSpace>(0,count),
    KOKKOS_LAMBDA(int j) {
      const int i = rows(j);
      Real diagonal = 0;
      for (int k = rp(j); k < rp(j+1); ++k) if (ci(k) == i) diagonal += av(k);
      inverse(i) = Real(1)/diagonal;
    });
  }

  Operator &CycleMatrix(Level &level) {
    // The production compact phase update shares the current finest matrix and
    // its existing halo with the V-cycle. The complete hierarchy update refreshes
    // coarse A/P in place; the fine-only API leaves them cached. The legacy
    // full-matrix update retains its outer-operator-only contract.
    return &level == &levels[0] && current_a.phase_active ? current_a : level.a;
  }

  void Initialize(const HostHierarchy &h) {
    // Reference adapter only. Production callers provide DistributedHierarchy
    // directly; they never materialize these global host matrices/transposes.
    Require(!h.matrices.empty() && h.prolongation.size()+1 == h.matrices.size(),
            "hierarchy requires L matrices and L-1 P");
    DistributedHierarchy local;
    local.constant_nullspace = h.constant_nullspace;
    local.row_partitions.resize(h.matrices.size());
    for (size_t l = 0; l < h.row_partitions.size(); ++l) {
      if (l < local.row_partitions.size()) {
        local.row_partitions[l].assign(h.row_partitions[l].begin(), h.row_partitions[l].end());
      }
    }
    for (size_t l = 0; l < h.matrices.size(); ++l) {
      bool valid = true;
      try { Validate(h.matrices[l]); } catch (...) { valid = false; }
      Require(valid, "invalid reference host matrix");
      local.row_partitions[l] = Partition(local.row_partitions, l, h.matrices[l].rows);
      local.matrices.push_back(Slice(h.matrices[l],local.row_partitions[l],rank));
    }
    for (size_t l = 0; l < h.prolongation.size(); ++l) {
      bool valid = true;
      try { Validate(h.prolongation[l]); } catch (...) { valid = false; }
      Require(valid && h.prolongation[l].rows == h.matrices[l].rows
          && h.prolongation[l].cols == h.matrices[l+1].rows,
          "invalid reference prolongation");
      local.prolongation.push_back(Slice(h.prolongation[l],local.row_partitions[l],rank));
      local.restriction.push_back(Slice(Transpose(h.prolongation[l]),
                                       local.row_partitions[l+1],rank));
    }
    Initialize(local);
  }

  void Initialize(const DistributedHierarchy &h) {
    const GlobalIndex count = h.matrices.size();
    GlobalIndex minimum = count, maximum = count;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &minimum, 1, MPI_INT64_T, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &maximum, 1, MPI_INT64_T, MPI_MAX, comm);
#endif
    Require(count > 0 && minimum == maximum
        && h.prolongation.size()+1 == h.matrices.size()
        && h.restriction.size()+1 == h.matrices.size(),
        "distributed hierarchy requires consistent L matrices and explicit L-1 P/R");
    int null_min = h.constant_nullspace ? 1 : 0, null_max = null_min;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &null_min, 1, MPI_INT, MPI_MIN, comm);
    MPI_Allreduce(MPI_IN_PLACE, &null_max, 1, MPI_INT, MPI_MAX, comm);
#endif
    Require(null_min == null_max, "inconsistent hierarchy nullspace flag");
    levels.resize(h.matrices.size());
    // the summed diagonal of every level's owned rows (smoothing check, inverse diagonal)
    std::vector<std::vector<Real>> diagonals(levels.size());
    for (size_t l = 0; l < levels.size(); ++l) {
      {
        SetupLap lap(SetupTimes().partition);
        levels[l].partition = Partition(h.row_partitions,l,h.matrices[l].global_rows);
      }
      CheckLocal(h.matrices[l],levels[l].partition,levels[l].partition);
      bool good_diagonal = true;
      {
        SetupLap lap(SetupTimes().diagonal);
        diagonals[l] = DiagonalSums(h.matrices[l]);
        if (l+1 < levels.size()) {
          for (Real diagonal : diagonals[l]) {
            good_diagonal = good_diagonal && diagonal != 0 && std::isfinite(diagonal)
                && std::isfinite(Real(1)/diagonal);
          }
        }
      }
      Require(good_diagonal, "invalid owned smoothing diagonal");
    }
    Require(h.matrices.back().global_rows <= 512,
            "coarsest matrix exceeds 512 rows");
    for (size_t l = 0; l+1 < levels.size(); ++l) {
      CheckLocal(h.prolongation[l],levels[l].partition,levels[l+1].partition);
      CheckLocal(h.restriction[l],levels[l+1].partition,levels[l].partition);
    }
    for (size_t l = 0; l < levels.size(); ++l) {
      auto &level = levels[l];
      const auto &part = level.partition;
      const auto &a = h.matrices[l];
      level.a = Build(a,part,part);
      const int n = a.LocalRows();
      {
        SetupLap lap(SetupTimes().upload);
        auto &diagonal = diagonals[l];
        for (int i = 0; i < n; ++i) {
          const Real d = diagonal[i];
          diagonal[i] = d == 0 ? 0 : 1/d;
        }
        level.inverse_diagonal = DeviceCopy(diagonal, "gravity_inverse_diagonal");
        std::vector<Real>().swap(diagonal);
        level.b = Vector("gravity_level_rhs",n);
        level.x = Vector("gravity_level_solution",n);
        level.residual = Vector("gravity_level_residual",n);
        level.temp = Vector("gravity_level_product",n);
      }
      if (l+1 < levels.size()) {
        level.p = Build(h.prolongation[l],part,levels[l+1].partition);
        level.r = Build(h.restriction[l],levels[l+1].partition,part);
      }
    }
    // ATHENA_SG_LEVEL_STATS=1 (xnode diagnostics): rows, nonzeros, longest row and halo of
    // every local operator of the hierarchy, one line per rank and level
    if (const char *e = std::getenv("ATHENA_SG_LEVEL_STATS"); e != nullptr && e[0] == '1') {
      auto stat = [](const LocalCsr &m, const Operator &op) {
        int longest = 0;
        for (int i = 0; i < m.LocalRows(); ++i)
          longest = std::max(longest, m.rowptr[i+1]-m.rowptr[i]);
        return std::to_string(m.LocalRows())+"/"+std::to_string(m.colidx.size())+"/"
            +std::to_string(longest)+"/"+std::to_string(op.ghost_ids.size())+"/"
            +std::to_string(op.recv_peers.size());
      };
      for (size_t l = 0; l < levels.size(); ++l) {
        std::cout << "# SG_LEVEL_STATS rank=" << rank << " level=" << l
                  << " global_rows=" << h.matrices[l].global_rows
                  << " A(rows/nnz/longest/ghosts/peers)=" << stat(h.matrices[l],levels[l].a);
        if (l+1 < levels.size())
          std::cout << " P=" << stat(h.prolongation[l],levels[l].p)
                    << " R=" << stat(h.restriction[l],levels[l].r);
        std::cout << '\n';
      }
    }
    // Shared CSR views remain immutable in the cached hierarchy. Halo scratch is
    // used sequentially. First refresh detaches outer values; new patterns rebuild CSR.
    SetupLap lap(SetupTimes().coarse);
    current_a = levels[0].a;
    // the current operator shares the cached views, not the stores behind them: its
    // first phase update that rebuilds the halo or the pattern allocates its own (as the fresh
    // views did), so the cached operator's send indices stay immutable
    current_a.store = Operator::Stores{};
    current_rowptr = h.matrices[0].rowptr;
    current_columns = h.matrices[0].colidx;
    current_values_shared = true;
    SetupCoarse(h.matrices.back());
    if (h.matrices.back().global_rows <= 128) terminal_matrix = h.matrices.back();
  }

  void UpdateFine(const LocalCsr &a, const std::vector<GlobalIndex> &partition) {
    const auto &part = levels[0].partition;
    Require(partition.empty() || partition == part,
            "fine ownership changed; construct a fresh hierarchy");
    CheckLocal(a,part,part);
    if (levels.size() == 1 && a.global_rows <= 128) terminal_matrix = a;
    fine_diagonal_dirty = true;
    if (current_a.phase_configured) {
      // A caller switching back to the outer-only full update freezes the most
      // recent phase preconditioner, including its already updated diagonal.
      if (current_a.phase_active) levels[0].a = current_a;
      current_a = Build(a,part,part);
      current_rowptr = a.rowptr;
      current_columns = a.colidx;
      current_values_shared = false;
      ++fine_updates.structure_rebuilds;
      ++fine_updates.updates;
      return;
    }
    int same_pattern = a.rowptr == current_rowptr && a.colidx == current_columns;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&same_pattern,1,MPI_INT,MPI_MIN,comm);
#endif
    if (same_pattern) {
      if (current_values_shared) {
        current_a.values = Vector("gravity_current_values",a.values.size());
        current_values_shared = false;
      }
      auto host = Kokkos::create_mirror_view(current_a.values);
      for (size_t k = 0; k < a.values.size(); ++k) host(k) = a.values[k];
      Kokkos::deep_copy(current_a.values,host);
      ++fine_updates.value_updates;
    } else {
      current_a = Build(a,part,part);
      current_rowptr = a.rowptr;
      current_columns = a.colidx;
      current_values_shared = false;
      ++fine_updates.structure_rebuilds;
    }
    ++fine_updates.updates;
  }

  void SetupCoarse(const LocalCsr &a) {
    const int n = static_cast<int>(a.global_rows), local_n = a.LocalRows();
    coarse_counts.resize(nranks);
    coarse_offsets.resize(nranks+1);
    for (int r = 0; r <= nranks; ++r) {
      coarse_offsets[r] = static_cast<int>(levels.back().partition[r]);
      if (r < nranks) coarse_counts[r] = static_cast<int>(
          levels.back().partition[r+1]-levels.back().partition[r]);
    }
    std::vector<int> local_row_nnz(local_n), row_nnz(rank == 0 ? n : 0);
    for (int i = 0; i < local_n; ++i) local_row_nnz[i] = a.rowptr[i+1]-a.rowptr[i];
    const int local_nnz = static_cast<int>(a.values.size());
    std::vector<int> counts(nranks), offsets(nranks+1,0);
#if MPI_PARALLEL_ENABLED
    MPI_Gather(&local_nnz,1,MPI_INT,counts.data(),1,MPI_INT,0,comm);
    MPI_Gatherv(local_row_nnz.data(),local_n,MPI_INT,row_nnz.data(),
                coarse_counts.data(),coarse_offsets.data(),MPI_INT,0,comm);
#else
    counts[0] = local_nnz;
    row_nnz = local_row_nnz;
#endif
    GlobalIndex total = 0;
    if (rank == 0) for (int c : counts) total += c;
    Require(rank != 0 || total <= std::numeric_limits<int>::max(),
            "coarse CSR gather exceeds MPI count capacity");
    if (rank == 0) for (int r = 0; r < nranks; ++r) offsets[r+1] = offsets[r]+counts[r];
    std::vector<GlobalIndex> columns(rank == 0 ? total : 0);
    std::vector<Real> values(rank == 0 ? total : 0);
#if MPI_PARALLEL_ENABLED
    MPI_Gatherv(a.colidx.data(),local_nnz,MPI_INT64_T,columns.data(),counts.data(),
                offsets.data(),MPI_INT64_T,0,comm);
    MPI_Gatherv(a.values.data(),local_nnz,MPI_ATHENA_REAL,values.data(),counts.data(),
                offsets.data(),MPI_ATHENA_REAL,0,comm);
#else
    columns = a.colidx;
    values = a.values;
#endif
    int good = 1;
    if (rank == 0) {
      coarse_lu.assign(n*n,Real(0));
      coarse_pivots.resize(n);
      coarse_rhs.resize(n);
      int entry = 0;
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < row_nnz[i]; ++k, ++entry) {
          coarse_lu[i*n+static_cast<int>(columns[entry])] += values[entry];
        }
      }
      Real matrix_scale = 0;
      std::vector<Real> column_max(n,Real(0));
      for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
        const Real value = std::abs(coarse_lu[i*n+j]);
        matrix_scale = std::max(matrix_scale,value);
        column_max[j] = std::max(column_max[j],value);
      }
      // A zero column is an extra null direction (except for the legitimate
      // one-cell constant-nullspace system). Check A before adding its gauge.
      if (!(nullspace && n == 1)) for (Real value : column_max)
        if (!(value > 0)) good = 2;
      // Scaling the constant-mode gauge with A keeps the numerical-rank test
      // independent of physical units. For a compatible RHS this changes only
      // the unphysical constant potential. The singleton zero A needs a unit gauge.
      const Real factor_scale = matrix_scale > 0 ? matrix_scale : Real(1);
      if (nullspace) for (Real &value : coarse_lu) value += factor_scale/n;
      const Real pivot_floor = 64*n*std::numeric_limits<Real>::epsilon()*factor_scale;
      for (int k = 0; good == 1 && k < n; ++k) {
        int pivot = k;
        for (int i = k+1; i < n; ++i) {
          if (std::abs(coarse_lu[i*n+k]) > std::abs(coarse_lu[pivot*n+k])) pivot = i;
        }
        coarse_pivots[k] = pivot;
        if (!std::isfinite(coarse_lu[pivot*n+k])
            || std::abs(coarse_lu[pivot*n+k]) <= pivot_floor) {
          good = 3;
          break;
        }
        if (pivot != k) {
          for (int j = 0; j < n; ++j) std::swap(coarse_lu[k*n+j],coarse_lu[pivot*n+j]);
        }
        for (int i = k+1; i < n; ++i) {
          coarse_lu[i*n+k] /= coarse_lu[k*n+k];
          for (int j = k+1; j < n; ++j) coarse_lu[i*n+j] -= coarse_lu[i*n+k]*coarse_lu[k*n+j];
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Bcast(&good,1,MPI_INT,0,comm);
#endif
    if (good == 2) throw std::runtime_error(
        "self-gravity: invalid terminal geometry/operator: zero coarse column");
    if (good != 1) throw std::runtime_error(
        "self-gravity: singular or ill-conditioned terminal operator after constant-mode "
        "gauge; coarse geometry may have an extra nullspace");
  }

  void UpdateTerminal(const CsrRowUpdate &a) {
    // Fine ownership is never gathered. Reassemble only the <=128-row terminal
    // CSR, then reuse the existing current-matrix gather/pivoted factorization.
    LocalCsr next;
    next.global_rows = terminal_matrix.global_rows;
    next.global_cols = terminal_matrix.global_cols;
    next.row_begin = terminal_matrix.row_begin;
    size_t selected = 0;
    for (int i = 0; i < terminal_matrix.LocalRows(); ++i) {
      if (selected < a.rows.size() && a.rows[selected] == i) {
        const int first = a.rowptr[selected], last = a.rowptr[selected+1];
        next.colidx.insert(next.colidx.end(),a.colidx.begin()+first,a.colidx.begin()+last);
        next.values.insert(next.values.end(),a.values.begin()+first,a.values.begin()+last);
        ++selected;
      } else {
        const int first = terminal_matrix.rowptr[i], last = terminal_matrix.rowptr[i+1];
        next.colidx.insert(next.colidx.end(),terminal_matrix.colidx.begin()+first,
                           terminal_matrix.colidx.begin()+last);
        next.values.insert(next.values.end(),terminal_matrix.values.begin()+first,
                           terminal_matrix.values.begin()+last);
      }
      next.rowptr.push_back(static_cast<int>(next.values.size()));
    }
    terminal_matrix = std::move(next);
    SetupCoarse(terminal_matrix);
  }

#if MPI_PARALLEL_ENABLED
  // The halo values of one product: messages only with the ranks in send_peers/recv_peers
  // (no global synchronization; a rank without halo entries returns at once).  The packed
  // send buffer is copied to the pinned host buffer on the device stream, one stream fence
  // before MPI, and the received values are copied back on the stream (the kernels that
  // read them follow on the same stream; the next exchange fences before MPI writes the
  // host buffer again).  ATHENA_SG_DEVICE_MPI=1 hands the device buffers to MPI instead.
  void ExchangeHalo(Operator &op) {
    const int ns = op.send_peers.size(), nr = op.recv_peers.size();
    if (ns == 0 && nr == 0) return;
    DevExeSpace exec;
    const bool device = DeviceMpi();
    const int nsend = op.send_device.extent_int(0);
    if (!device && nsend > 0) {
      Kokkos::deep_copy(exec, Kokkos::subview(op.send_host,std::make_pair(0,nsend)),
                        op.send_device);
    }
    exec.fence("gravity_halo_send_ready");
    PostHalo(op);
    FinishHalo(op);
  }
  // the receives and sends of the packed halo (the send buffer complete)
  void PostHalo(Operator &op) {
    const int ns = op.send_peers.size(), nr = op.recv_peers.size();
    if (ns == 0 && nr == 0) return;
    const bool device = DeviceMpi();
    Real *sbuf = device ? op.send_device.data() : op.send_host.data();
    Real *rbuf = device ? op.recv_device.data() : op.ghost_host.data();
    auto *req = op.halo_requests.data();
    for (int q = 0; q < nr; ++q) {
      const int r = op.recv_peers[q];
      MPI_Irecv(rbuf+op.recv_offsets[r], op.recv_counts[r], MPI_ATHENA_REAL, r, 0, comm,
                &req[q]);
    }
    for (int q = 0; q < ns; ++q) {
      const int r = op.send_peers[q];
      MPI_Isend(sbuf+op.send_offsets[r], op.send_counts[r], MPI_ATHENA_REAL, r, 0, comm,
                &req[nr+q]);
    }
  }
  // completion of PostHalo and the copy of the received values to the device (queued)
  void FinishHalo(Operator &op) {
    const int ns = op.send_peers.size(), nr = op.recv_peers.size();
    if (ns == 0 && nr == 0) return;
    DevExeSpace exec;
    const int nrecv = op.recv_device.extent_int(0);
    MPI_Waitall(nr+ns, op.halo_requests.data(), MPI_STATUSES_IGNORE);
    if (!DeviceMpi() && nrecv > 0) {
      Kokkos::deep_copy(exec, op.recv_device,
                        Kokkos::subview(op.ghost_host,std::make_pair(0,nrecv)));
    }
  }
#endif

  // the halo entries of x into the device send buffer
  void PackHalo(Operator &op, const Vector &x) {
    auto send = op.send_device;
    auto send_indices = op.send_indices;
    const int nsend = send.extent_int(0);
    Kokkos::parallel_for("gravity_halo_pack", Kokkos::RangePolicy<DevExeSpace>(0, nsend),
                         KOKKOS_LAMBDA(int i) { send(i) = x(send_indices(i)); });
  }

  // Project(x) followed by Multiply(op, x, y), see ProjectFuse: the halo of the
  // unprojected x travels while the mean is reduced; owner and receivers subtract the same mean
  void ProjectMultiply(const Vector &x, GlobalIndex global_size, Operator &op, const Vector &y) {
#if MPI_PARALLEL_ENABLED
    if (!nullspace || !fuse_projection || HaloAlltoallv() || DeviceMpi()) {
      Project(x, global_size);
      Multiply(op, x, y);
      return;
    }
    if (x.extent_int(0) != op.inputs || y.extent_int(0) != op.rows) {
      throw std::runtime_error("self-gravity: local matrix/vector shape mismatch");
    }
    DevExeSpace exec;
    PackHalo(op, x);
    const int nsend = op.send_device.extent_int(0);
    const int nrecv = op.recv_device.extent_int(0);
    if (nsend > 0) {
      Kokkos::deep_copy(exec, Kokkos::subview(op.send_host,std::make_pair(0,nsend)),
                        op.send_device);
    }
    // the local sum as in Project (the same reduction); a reduction into a host scalar returns
    // only when the stream has completed it, so the packed halo is in the host buffer too
    Real local = 0;
    const int n = x.extent_int(0);
    Kokkos::parallel_reduce("gravity_mean", Kokkos::RangePolicy<DevExeSpace>(0, n),
                            KOKKOS_LAMBDA(int i, Real &s) { s += x(i); }, local);
    PostHalo(op);
    const Real mean = Sum(local)/global_size;
    // the received values minus the mean: a short halo on the host before its copy to the
    // device (IEEE subtraction, the same result as on the device), a long one on the device
    const int ns = op.send_peers.size(), nr = op.recv_peers.size();
    const bool host_halo = nrecv > 0 && nrecv <= 16384 && (ns > 0 || nr > 0);
    if (ns > 0 || nr > 0) {
      MPI_Waitall(nr+ns, op.halo_requests.data(), MPI_STATUSES_IGNORE);
      if (host_halo) {
        Real *h = op.ghost_host.data();
        for (int q = 0; q < nrecv; ++q) h[q] -= mean;
      }
      if (nrecv > 0) {
        Kokkos::deep_copy(exec, op.recv_device,
                          Kokkos::subview(op.ghost_host,std::make_pair(0,nrecv)));
      }
    }
    Kokkos::parallel_for("gravity_null_projection", Kokkos::RangePolicy<DevExeSpace>(0, n),
                         KOKKOS_LAMBDA(int i) { x(i) -= mean; });
    if (nrecv > 0 && !host_halo) {
      auto received = op.recv_device;
      Kokkos::parallel_for("gravity_null_projection_halo",
                           Kokkos::RangePolicy<DevExeSpace>(0, nrecv),
                           KOKKOS_LAMBDA(int q) { received(q) -= mean; });
    }
    ScatterAndProduct(op, x, y);
#else
    Project(x, global_size);
    Multiply(op, x, y);
#endif
  }

  void Multiply(Operator &op, const Vector &x, const Vector &y) {
    if (x.extent_int(0) != op.inputs || y.extent_int(0) != op.rows) {
      throw std::runtime_error("self-gravity: local matrix/vector shape mismatch");
    }
    auto send = op.send_device;
    PackHalo(op, x);
#if MPI_PARALLEL_ENABLED
    if (!HaloAlltoallv()) {
      ExchangeHalo(op);
    } else if (DeviceMpi()) {
      // GPU-aware MPI: the device buffers themselves (the pack must be complete first)
      Kokkos::fence("gravity_halo_device_mpi");
      MPI_Alltoallv(send.data(), op.send_counts.data(), op.send_offsets.data(),
                    MPI_ATHENA_REAL, op.recv_device.data(), op.recv_counts.data(),
                    op.recv_offsets.data(), MPI_ATHENA_REAL, comm);
    } else {
      Kokkos::deep_copy(op.send_host, send);
      MPI_Alltoallv(op.send_host.data(), op.send_counts.data(), op.send_offsets.data(),
                    MPI_ATHENA_REAL, op.ghost_host.data(), op.recv_counts.data(),
                    op.recv_offsets.data(), MPI_ATHENA_REAL, comm);
      Kokkos::deep_copy(op.recv_device, op.ghost_host);
    }
#else
    Kokkos::deep_copy(op.send_host, send);
    Kokkos::deep_copy(op.recv_device, op.ghost_host);
#endif
    ScatterAndProduct(op, x, y);
  }

  // the received halo values to their ghost slots, then the local product y = A x
  void ScatterAndProduct(Operator &op, const Vector &x, const Vector &y) {
    if (op.recv_slots.extent(0) != 0) {
      auto received = op.recv_device;
      auto targets = op.ghost_device;
      auto slots = op.recv_slots;
      Kokkos::parallel_for("gravity_union_scatter",Kokkos::RangePolicy<DevExeSpace>(0,received.extent_int(0)),
                           KOKKOS_LAMBDA(int q) { targets(slots(q)) = received(q); });
    }
    auto ghosts = op.ghost_device;
    auto rp = op.rowptr;
    auto ci = op.colidx;
    auto av = op.values;
    auto phase_map = op.phase_map;
    auto prp = op.phase_rowptr;
    auto pci = op.phase_colidx;
    auto pav = op.phase_values;
    const bool phase_active = op.phase_active;
    const int ni = op.inputs;
    const int row_teams = SpmvRowTeams();
    if (row_teams > 0 && op.rows > 0
        && static_cast<std::int64_t>(op.colidx.extent(0)) >= static_cast<std::int64_t>(row_teams)*op.rows) {
      // one team per row (xnode; see SpmvRowTeams): the vector lanes load
      // a chunk of the row's values and inputs into team scratch at once, then one lane adds
      // them up in the row order -- the same multiply-adds in the same order as the loop
      // below, so a long row costs a few memory latencies instead of one per entry
      using Policy = Kokkos::TeamPolicy<DevExeSpace>;
      using Member = Policy::member_type;
      using Scratch = Kokkos::View<Real*, DevExeSpace::scratch_memory_space,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
      const int vl = std::min(32, Policy::vector_length_max());
      const size_t bytes = 2*Scratch::shmem_size(vl);
      Kokkos::parallel_for("gravity_sparse_matvec_rows",
          Policy(op.rows, 1, vl).set_scratch_size(0, Kokkos::PerTeam(bytes)),
          KOKKOS_LAMBDA(const Member &m) {
        const int i = m.league_rank();
        Scratch sa(m.team_scratch(0), vl), sx(m.team_scratch(0), vl);
        const int selected = phase_active ? phase_map(i) : -1;
        const bool ph = selected >= 0;
        const int kb = ph ? prp(selected) : rp(i), ke = ph ? prp(selected+1) : rp(i+1);
        Real sum = 0;
        for (int k0 = kb; k0 < ke;) {  // k0 advances by cnt: never past ke (no int overflow)
          const int cnt = (ke-k0 < vl) ? ke-k0 : vl;
          Kokkos::parallel_for(Kokkos::ThreadVectorRange(m, cnt), [&](const int q) {
            const int k = k0+q;
            const int j = ph ? pci(k) : ci(k);
            sa(q) = ph ? pav(k) : av(k);
            sx(q) = j < ni ? x(j) : ghosts(j-ni);
          });
          m.team_barrier();
          Kokkos::single(Kokkos::PerThread(m), [&]() {
            for (int q = 0; q < cnt; ++q) sum += sa(q)*sx(q);
          });
          m.team_barrier();
          k0 += cnt;
        }
        Kokkos::single(Kokkos::PerThread(m), [&]() { y(i) = sum; });
      });
      ++matrix_applications;
      return;
    }
    Kokkos::parallel_for("gravity_sparse_matvec", Kokkos::RangePolicy<DevExeSpace>(0, op.rows),
                         KOKKOS_LAMBDA(int i) {
      Real sum = 0;
      const int selected = phase_active ? phase_map(i) : -1;
      if (selected >= 0) {
        for (int k = prp(selected); k < prp(selected+1); ++k) {
          const int j = pci(k);
          sum += pav(k)*(j < ni ? x(j) : ghosts(j-ni));
        }
      } else {
        for (int k = rp(i); k < rp(i+1); ++k) {
          const int j = ci(k);
          sum += av(k)*(j < ni ? x(j) : ghosts(j-ni));
        }
      }
      y(i) = sum;
    });
    ++matrix_applications;
  }

  Real Sum(Real value) const {
#if MPI_PARALLEL_ENABLED
    Real result;
    MPI_Allreduce(&value, &result, 1, MPI_ATHENA_REAL, MPI_SUM, comm);
    return result;
#else
    return value;
#endif
  }

  Real Dot(const Vector &x, const Vector &y) const {
    Real result = 0;
    const int n = x.extent_int(0);
    Kokkos::parallel_reduce("gravity_dot", Kokkos::RangePolicy<DevExeSpace>(0, n),
                            KOKKOS_LAMBDA(int i, Real &s) { s += x(i)*y(i); }, result);
    return Sum(result);
  }

  ScaledSquares NormData(const Vector &x) const {
    ScaledSquares result;
    Kokkos::parallel_reduce("gravity_safe_norm",
        Kokkos::RangePolicy<DevExeSpace>(0,x.extent_int(0)),NormReduction{x},result);
#if MPI_PARALLEL_ENABLED
    if (norm_type == MPI_DATATYPE_NULL) {
      MPI_Type_contiguous(2,MPI_ATHENA_REAL,&norm_type);
      MPI_Type_commit(&norm_type);
      MPI_Op_create(MergeScaledSquares,1,&norm_op);
    }
    MPI_Allreduce(MPI_IN_PLACE,&result,1,norm_type,norm_op,comm);
#endif
    return result;
  }

  Real Norm(const Vector &x) const {
    const auto s = NormData(x);
    return s.scale < 0 ? std::numeric_limits<Real>::infinity()
                       : s.scale*std::sqrt(s.squares);
  }

  Real Project(const Vector &x, GlobalIndex global_size) const {
    if (!nullspace) return 0;
    Real local = 0;
    const int n = x.extent_int(0);
    Kokkos::parallel_reduce("gravity_mean", Kokkos::RangePolicy<DevExeSpace>(0, n),
                            KOKKOS_LAMBDA(int i, Real &s) { s += x(i); }, local);
    const Real mean = Sum(local)/global_size;
    Kokkos::parallel_for("gravity_null_projection", Kokkos::RangePolicy<DevExeSpace>(0, n),
                         KOKKOS_LAMBDA(int i) { x(i) -= mean; });
    return mean;
  }

  void Residual(Level &l, const Vector &b, const Vector &x, const Vector &r) {
    Multiply(CycleMatrix(l), x, r);
    const int n = r.extent_int(0);
    Kokkos::parallel_for("gravity_residual", Kokkos::RangePolicy<DevExeSpace>(0, n),
                         KOKKOS_LAMBDA(int i) { r(i) = b(i)-r(i); });
    Project(r, l.partition.back());
  }

  void CurrentResidual(const Vector &b, const Vector &x, const Vector &r) {
    Multiply(current_a,x,r);
    const int n = r.extent_int(0);
    Kokkos::parallel_for("gravity_current_residual",Kokkos::RangePolicy<DevExeSpace>(0,n),
                         KOKKOS_LAMBDA(int i) { r(i) = b(i)-r(i); });
    Project(r,levels[0].partition.back());
  }

  void Smooth(Level &l) {
    auto x = l.x;
    auto b = l.b;
    auto temp = l.temp;
    auto diagonal = l.inverse_diagonal;
    const int n = x.extent_int(0);
    for (int sweep = 0; sweep < 3; ++sweep) {
      Multiply(CycleMatrix(l), x, temp);
      Kokkos::parallel_for("gravity_jacobi", Kokkos::RangePolicy<DevExeSpace>(0, n),
                           KOKKOS_LAMBDA(int i) {
        x(i) += Real(0.65)*diagonal(i)*(b(i)-temp(i));
      });
      Project(x, l.partition.back());
    }
  }

  // project = false (fused V-cycle): the final projection of l.x is left to the caller
  void CoarseSolve(Level &l, bool project = true) {
    auto host_b = Kokkos::create_mirror_view_and_copy(HostMemSpace(), l.b);
    auto host_x = Kokkos::create_mirror_view(l.x);
    const int n = static_cast<int>(l.partition.back());
#if MPI_PARALLEL_ENABLED
    MPI_Gatherv(host_b.data(), l.b.extent_int(0), MPI_ATHENA_REAL,
                 coarse_rhs.data(), coarse_counts.data(), coarse_offsets.data(),
                 MPI_ATHENA_REAL, 0, comm);
#else
    for (int i = 0; i < n; ++i) coarse_rhs[i] = host_b(i);
#endif
    if (rank == 0) {
      for (int k = 0; k < n; ++k) std::swap(coarse_rhs[k], coarse_rhs[coarse_pivots[k]]);
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) coarse_rhs[i] -= coarse_lu[i*n+j]*coarse_rhs[j];
      }
      for (int i = n-1; i >= 0; --i) {
        for (int j = i+1; j < n; ++j) coarse_rhs[i] -= coarse_lu[i*n+j]*coarse_rhs[j];
        coarse_rhs[i] /= coarse_lu[i*n+i];
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Scatterv(coarse_rhs.data(), coarse_counts.data(), coarse_offsets.data(),
                  MPI_ATHENA_REAL, host_x.data(), l.x.extent_int(0),
                  MPI_ATHENA_REAL, 0, comm);
#else
    for (int i = 0; i < n; ++i) host_x(i) = coarse_rhs[i];
#endif
    Kokkos::deep_copy(l.x, host_x);
    if (project) Project(l.x, n);
  }

  void Cycle(int index) {
    // xnode profiling: a region per level (nested; no-op without a Kokkos tool)
    Kokkos::Profiling::pushRegion("MG:L" + std::to_string(index));
    auto &l = levels[index];
    Project(l.b, l.partition.back());
    // queued on the device stream (ATHENA_SG_MG_SYNC_COPIES=1: blocking, as before)
    if (MgSyncCopies()) Kokkos::deep_copy(l.x, Real(0));
    else Kokkos::deep_copy(DevExeSpace(), l.x, Real(0));
    if (index+1 == static_cast<int>(levels.size())) {
      CoarseSolve(l);
      Kokkos::Profiling::popRegion();
      return;
    }
    Smooth(l);
    Residual(l, l.b, l.x, l.residual);
    Multiply(l.r, l.residual, levels[index+1].b);
    Cycle(index+1);
    Multiply(l.p, levels[index+1].x, l.temp);
    Axpy(1, l.temp, l.x);
    Smooth(l);
    Kokkos::Profiling::popRegion();
  }

  // Smooth without its last projection, which the caller fuses with the next product
  // (ProjectMultiply); the sweeps after the first fuse the projection of the previous sweep
  void SmoothFused(Level &l) {
    auto x = l.x;
    auto b = l.b;
    auto temp = l.temp;
    auto diagonal = l.inverse_diagonal;
    const int n = x.extent_int(0);
    for (int sweep = 0; sweep < 3; ++sweep) {
      if (sweep == 0) Multiply(CycleMatrix(l), x, temp);
      else ProjectMultiply(x, l.partition.back(), CycleMatrix(l), temp);
      Kokkos::parallel_for("gravity_jacobi", Kokkos::RangePolicy<DevExeSpace>(0, n),
                           KOKKOS_LAMBDA(int i) {
        x(i) += Real(0.65)*diagonal(i)*(b(i)-temp(i));
      });
    }
  }

  // Cycle with every projection that is followed by a product of the same vector fused
  // with that product (ProjectMultiply): the same operations in the same order on the device; it
  // returns with the final projection of levels[index].x left to the caller
  void CycleFused(int index) {
    Kokkos::Profiling::pushRegion("MG:L" + std::to_string(index));
    auto &l = levels[index];
    Project(l.b, l.partition.back());
    if (MgSyncCopies()) Kokkos::deep_copy(l.x, Real(0));
    else Kokkos::deep_copy(DevExeSpace(), l.x, Real(0));
    if (index+1 == static_cast<int>(levels.size())) {
      CoarseSolve(l, false);
      Kokkos::Profiling::popRegion();
      return;
    }
    SmoothFused(l);
    // Residual(l, l.b, l.x, l.residual), its product fused with the last smoothing projection
    ProjectMultiply(l.x, l.partition.back(), CycleMatrix(l), l.residual);
    {
      auto r = l.residual;
      auto b = l.b;
      const int n = r.extent_int(0);
      Kokkos::parallel_for("gravity_residual", Kokkos::RangePolicy<DevExeSpace>(0, n),
                           KOKKOS_LAMBDA(int i) { r(i) = b(i)-r(i); });
    }
    // its projection fused with the restriction
    ProjectMultiply(l.residual, l.partition.back(), l.r, levels[index+1].b);
    CycleFused(index+1);
    // the coarse level's final projection fused with the prolongation
    ProjectMultiply(levels[index+1].x, levels[index+1].partition.back(), l.p, l.temp);
    Axpy(1, l.temp, l.x);
    SmoothFused(l);
    Kokkos::Profiling::popRegion();
  }

  void Precondition(const Vector &rhs, const Vector &result) {
    if (rhs.extent_int(0) != levels[0].b.extent_int(0)
        || result.extent_int(0) != rhs.extent_int(0)) {
      throw std::runtime_error("self-gravity: V-cycle local vector shape mismatch");
    }
    if (MgSyncCopies()) Kokkos::deep_copy(levels[0].b, rhs);
    else Kokkos::deep_copy(DevExeSpace(), levels[0].b, rhs);
    if (!fuse_projection) {
      Cycle(0);
    } else {
      CycleFused(0);
      Project(levels[0].x, levels[0].partition.back());
    }
    if (MgSyncCopies()) Kokkos::deep_copy(result, levels[0].x);
    else Kokkos::deep_copy(DevExeSpace(), result, levels[0].x);
    ++vcycles;
  }

  SolveStats Solve(const Vector &rhs, const Vector &solution, Real tolerance,
                   int max_iterations, int restart) {
    const int n = levels[0].a.rows;
    const GlobalIndex global_n = levels[0].partition.back();
    if (rhs.extent_int(0) != n || solution.extent_int(0) != n || !(tolerance > 0)
        || !std::isfinite(tolerance) || max_iterations < 0 || restart < 1) {
      throw std::runtime_error("self-gravity: invalid FGMRES arguments");
    }
    SolveStats stats;
    const int initial_matvecs = matrix_applications, initial_cycles = vcycles;
    const int m = std::min(restart, std::max(1, max_iterations));
    if (krylov_restart != m) {
      krylov_b = Vector("gravity_krylov_rhs", n);
      krylov_r = Vector("gravity_krylov_residual", n);
      krylov_w = Vector("gravity_krylov_work", n);
      krylov_x = Vector("gravity_krylov_solution", n);
      krylov_v.clear();
      krylov_z.clear();
      for (int j = 0; j <= m; ++j) krylov_v.emplace_back("gravity_arnoldi", n);
      for (int j = 0; j < m; ++j) krylov_z.emplace_back("gravity_flexible_basis", n);
      krylov_h.resize((m+1)*m);
      krylov_cosine.resize(m);
      krylov_sine.resize(m);
      krylov_g.resize(m+1);
      krylov_y.resize(m);
      krylov_restart = m;
    }
    auto b = krylov_b, r = krylov_r, w = krylov_w, x = krylov_x;
    auto &v = krylov_v;
    auto &z = krylov_z;
    auto &h = krylov_h;
    auto &cosine = krylov_cosine;
    auto &sine = krylov_sine;
    auto &g = krylov_g;
    auto &y = krylov_y;
    const auto rhs_data = NormData(rhs), guess_data = NormData(solution);
    if (rhs_data.scale < 0 || guess_data.scale < 0) {
      throw std::runtime_error("self-gravity: nonfinite FGMRES rhs/initial guess");
    }
    const Real largest = std::max(rhs_data.scale,guess_data.scale);
    // Leave ordinary-scale arithmetic unchanged. At extreme scales solve for
    // x/scale with b/scale so projection, coarse solves and Krylov updates stay
    // representable. Divide directly (the reciprocal could overflow).
    const Real safe_low = std::sqrt(std::numeric_limits<Real>::min());
    const Real safe_high = std::sqrt(std::numeric_limits<Real>::max());
    const Real scale = largest > 0 && (largest < safe_low || largest > safe_high)
                       ? largest : Real(1);
    if (rhs_data.scale > 0 && rhs_data.scale/scale == 0) {
      throw std::runtime_error("self-gravity: FGMRES rhs/guess scale range unsupported");
    }
    Kokkos::parallel_for("gravity_normalize_system",Kokkos::RangePolicy<DevExeSpace>(0,n),
        KOKKOS_LAMBDA(int i) { b(i) = rhs(i)/scale; x(i) = solution(i)/scale; });
    stats.removed_rhs_mean = Project(b, global_n)*scale;
    Project(x, global_n);
    const Real norm_b = Norm(b);
    // Zero is an exact gauge-fixed solution of the homogeneous system. Taking
    // it directly avoids spending iterations cancelling an enormous warm guess
    // down to the absolute tolerance used for a zero projected RHS.
    if (norm_b == 0) Kokkos::deep_copy(x,Real(0));
    // Compare dimensionless ratios rather than tolerance*||b||, which could
    // underflow. For zero RHS retain the original absolute-tolerance contract.
    auto relative = [norm_b,scale](Real value) {
      return norm_b > 0 ? value/norm_b : value*scale;
    };
    CurrentResidual(b,x,r);
    Real beta = Norm(r);
    while (std::isfinite(norm_b) && std::isfinite(beta) &&
           std::isfinite(relative(beta)) && relative(beta) > tolerance &&
           stats.iterations < max_iterations) {
      std::fill(h.begin(), h.end(), Real(0));
      std::fill(g.begin(), g.end(), Real(0));
      g[0] = beta;
      if (MgSyncCopies()) Kokkos::deep_copy(v[0], r);
      else Kokkos::deep_copy(DevExeSpace(), v[0], r);
      Scale(1/beta, v[0]);
      int used = 0;
      for (int j = 0; j < m && stats.iterations < max_iterations; ++j) {
        Precondition(v[j], z[j]);
        Kokkos::Profiling::pushRegion("MG:krylov");
        Multiply(current_a,z[j],w);
        Project(w, global_n);
        // Reorthogonalization avoids false convergence near a tight Poisson tolerance.
        for (int pass = 0; pass < 2; ++pass) {
          for (int i = 0; i <= j; ++i) {
            const Real product = Dot(v[i], w);
            h[i*m+j] += product;
            Axpy(-product, v[i], w);
          }
        }
        const Real next = Norm(w);
        Kokkos::Profiling::popRegion();
        h[(j+1)*m+j] = next;
        if (next > 0) {
          if (MgSyncCopies()) Kokkos::deep_copy(v[j+1], w);
          else Kokkos::deep_copy(DevExeSpace(), v[j+1], w);
          Scale(1/next, v[j+1]);
        }
        for (int i = 0; i < j; ++i) {
          const Real first = h[i*m+j], second = h[(i+1)*m+j];
          h[i*m+j] = cosine[i]*first+sine[i]*second;
          h[(i+1)*m+j] = -sine[i]*first+cosine[i]*second;
        }
        const Real first = h[j*m+j], second = h[(j+1)*m+j];
        const Real rotation_norm = std::hypot(first, second);
        cosine[j] = rotation_norm > 0 ? first/rotation_norm : 1;
        sine[j] = rotation_norm > 0 ? second/rotation_norm : 0;
        h[j*m+j] = rotation_norm;
        h[(j+1)*m+j] = 0;
        g[j+1] = -sine[j]*g[j];
        g[j] *= cosine[j];
        ++stats.iterations;
        used = j+1;
        if (relative(std::abs(g[j+1])) <= tolerance || next == 0 || rotation_norm == 0) break;
      }
      bool triangular_ok = true;
      for (int i = used-1; i >= 0; --i) {
        Real value = g[i];
        for (int j = i+1; j < used; ++j) value -= h[i*m+j]*y[j];
        if (h[i*m+i] == 0 || !std::isfinite(h[i*m+i])) {
          triangular_ok = false;
          break;
        }
        y[i] = value/h[i*m+i];
      }
      if (!triangular_ok) break;
      for (int j = 0; j < used; ++j) Axpy(y[j], z[j], x);
      Project(x, global_n);
      // Report and accept only an explicitly recomputed true residual.
      CurrentResidual(b,x,r);
      beta = Norm(r);
      // A huge true/Arnoldi gap signals an unusable flexible basis, e.g. a
      // near-singular coarse solve producing 1e15-size cancelling updates.
      // Include the requested target in the comparison so normal residual
      // replacement near the roundoff floor is not mistaken for breakdown.
      if (relative(beta)/1000 > std::max(relative(std::abs(g[used])),tolerance)) {
        stats.preconditioner_breakdown = true;
        break;
      }
    }
    stats.relative_residual = relative(beta);
    stats.converged = std::isfinite(norm_b) && std::isfinite(beta) &&
        std::isfinite(stats.relative_residual) && stats.relative_residual <= tolerance;
    // Validate before publishing the rescaled answer, including a nonfinite
    // flexible-basis update that might otherwise evade a residual comparison.
    const auto answer = NormData(x);
    if (answer.scale < 0 || answer.scale > std::numeric_limits<Real>::max()/scale) {
      stats.converged = false;
    } else {
      Kokkos::parallel_for("gravity_restore_solution",Kokkos::RangePolicy<DevExeSpace>(0,n),
          KOKKOS_LAMBDA(int i) { solution(i) = x(i)*scale; });
    }
    stats.matrix_applications = matrix_applications-initial_matvecs;
    stats.vcycles = vcycles-initial_cycles;
    return stats;
  }
};

#if MPI_PARALLEL_ENABLED
SparseMultigrid::SparseMultigrid(const HostHierarchy &h, MPI_Comm comm)
    : impl_(new Impl(h, comm)) {}
SparseMultigrid::SparseMultigrid(const DistributedHierarchy &h, MPI_Comm comm)
    : impl_(new Impl(h, comm)) {}
#else
SparseMultigrid::SparseMultigrid(const HostHierarchy &h) : impl_(new Impl(h)) {}
SparseMultigrid::SparseMultigrid(const DistributedHierarchy &h) : impl_(new Impl(h)) {}
#endif
SparseMultigrid::~SparseMultigrid() = default;
int SparseMultigrid::Levels() const { return impl_->levels.size(); }
GlobalIndex SparseMultigrid::LocalBegin(int l) const {
  return impl_->levels.at(l).partition[impl_->rank];
}
GlobalIndex SparseMultigrid::LocalEnd(int l) const {
  return impl_->levels.at(l).partition[impl_->rank+1];
}
int SparseMultigrid::LocalSize(int l) const { return LocalEnd(l)-LocalBegin(l); }
StorageStats SparseMultigrid::Storage() const {
  StorageStats s;
  for (const auto &l : impl_->levels) {
    s.a_nnz += Impl::Nonzeros(&l == &impl_->levels[0] && impl_->current_a.phase_active
                              ? impl_->current_a : l.a);
    s.p_nnz += Impl::Nonzeros(l.p);
    s.r_nnz += Impl::Nonzeros(l.r);
    Impl::AddStorage(l.a,s);
    Impl::AddStorage(l.p,s);
    Impl::AddStorage(l.r,s);
    s.vector_bytes += (l.inverse_diagonal.size()+l.b.size()+l.x.size()
                       +l.residual.size()+l.temp.size())*sizeof(Real);
  }
  const auto &a = impl_->current_a;
  const auto &cached = impl_->levels[0].a;
  s.current_a_nnz = a.values.size();
  if (a.phase_active) {
    s.current_a_nnz = a.values.size()-a.phase_masked_base_nnz+a.phase_values.size();
  }
  if (a.rowptr.data() != cached.rowptr.data()) s.logical_csr_bytes += a.rowptr.size()*sizeof(int);
  if (a.colidx.data() != cached.colidx.data()) s.logical_csr_bytes += a.colidx.size()*sizeof(int);
  if (a.values.data() != cached.values.data()) s.logical_csr_bytes += a.values.size()*sizeof(Real);
  if (a.send_indices.data() != cached.send_indices.data())
    s.logical_halo_bytes += a.send_indices.size()*sizeof(int);
  if (a.send_device.data() != cached.send_device.data())
    s.logical_halo_bytes += a.send_device.size()*sizeof(Real);
  if (a.ghost_device.data() != cached.ghost_device.data())
    s.logical_halo_bytes += a.ghost_device.size()*sizeof(Real);
  if (a.send_host.data() != cached.send_host.data())
    s.logical_halo_bytes += a.send_host.size()*sizeof(Real);
  if (a.ghost_host.data() != cached.ghost_host.data())
    s.logical_halo_bytes += a.ghost_host.size()*sizeof(Real);
  s.logical_halo_bytes += Impl::CountBytes(a);
  Impl::AddPhaseStorage(a,s);
  // the allocations behind all operators, each counted once
  {
    Impl::Ranges csr, halo;
    std::uint64_t counts = Impl::CountBytes(a);
    Impl::AddAllocations(a,csr,halo);
    for (const auto &l : impl_->levels) {
      Impl::AddAllocations(l.a,csr,halo); Impl::AddAllocations(l.p,csr,halo);
      Impl::AddAllocations(l.r,csr,halo);
      counts += Impl::CountBytes(l.a)+Impl::CountBytes(l.p)+Impl::CountBytes(l.r);
    }
    s.csr_bytes = Impl::MergedBytes(std::move(csr));
    s.halo_bytes = Impl::MergedBytes(std::move(halo))+counts;
  }
  s.terminal_host_csr_bytes = impl_->terminal_matrix.rowptr.capacity()*sizeof(int)
      +impl_->terminal_matrix.colidx.capacity()*sizeof(GlobalIndex)
      +impl_->terminal_matrix.values.capacity()*sizeof(Real);
  s.host_pattern_bytes += impl_->current_rowptr.capacity()*sizeof(int)
      +impl_->current_columns.capacity()*sizeof(GlobalIndex);
  s.vector_bytes += (impl_->coarse_lu.size()+impl_->coarse_rhs.size())*sizeof(Real)
      +(impl_->coarse_pivots.size()+impl_->coarse_counts.size()
        +impl_->coarse_offsets.size())*sizeof(int);
  s.vector_bytes += (impl_->krylov_b.size()+impl_->krylov_r.size()
                    +impl_->krylov_w.size()+impl_->krylov_x.size())*sizeof(Real);
  for (const auto &v : impl_->krylov_v) s.vector_bytes += v.size()*sizeof(Real);
  for (const auto &z : impl_->krylov_z) s.vector_bytes += z.size()*sizeof(Real);
  s.vector_bytes += (impl_->krylov_h.capacity()+impl_->krylov_cosine.capacity()
      +impl_->krylov_sine.capacity()+impl_->krylov_g.capacity()
      +impl_->krylov_y.capacity())*sizeof(Real);
  return s;
}
void SparseMultigrid::UpdateFineOperator(const LocalCsr &a,
                                        const std::vector<GlobalIndex> &partition) {
  impl_->UpdateFine(a,partition);
}
FineOperatorStats SparseMultigrid::FineUpdates() const {
  auto stats = impl_->current_a.phase_updates;
  stats.updates = impl_->fine_updates.updates;
  stats.value_updates = impl_->fine_updates.value_updates;
  stats.structure_rebuilds = impl_->fine_updates.structure_rebuilds;
  return stats;
}
FineOperatorStats SparseMultigrid::HierarchyPhaseUpdates() const {
  FineOperatorStats total;
  auto add = [&](const Impl::Operator &op) {
    const auto &s = op.phase_updates;
    total.updates += s.updates; total.value_updates += s.value_updates;
    total.structure_rebuilds += s.structure_rebuilds;
    total.local_pattern_changes += s.local_pattern_changes;
    total.halo_rebuilds += s.halo_rebuilds; total.halo_reuses += s.halo_reuses;
    total.local_ghost_changes += s.local_ghost_changes;
    total.global_ghost_changes += s.global_ghost_changes;
    total.host_map_entries += s.host_map_entries;
    total.host_map_buckets += s.host_map_buckets;
    total.host_map_peak_bytes += s.host_map_peak_bytes;
  };
  add(impl_->current_a);
  for (size_t l = 1; l < impl_->levels.size(); ++l) add(impl_->levels[l].a);
  for (size_t l = 0; l+1 < impl_->levels.size(); ++l) add(impl_->levels[l].p);
  return total;
}
void SparseMultigrid::ConfigureFinePhaseRows(const std::vector<int> &rows) {
  impl_->ConfigurePhase(impl_->current_a,rows);
  std::vector<int>().swap(impl_->current_rowptr);
  std::vector<GlobalIndex>().swap(impl_->current_columns);
}
void SparseMultigrid::UpdateFinePhaseRows(const CsrRowUpdate &a) {
  UpdateFinePhaseRowsPlanned(a,nullptr);
}
void SparseMultigrid::UpdateFinePhaseRowsPlanned(const CsrRowUpdate &a, const PhaseRowPlan *plan) {
  PhaseUploadFence complete;   // the queued uploads, at every exit
  impl_->UpdateFineQueued(a,plan,nullptr);
  complete.Complete();         // normal exit: a device error reaches the caller
}
void SparseMultigrid::UpdateFinePhaseRowsKnown(const CsrRowUpdate &a, const PhaseRowPlan *plan,
                                               const int *known) {
  PhaseUploadFence complete;   // the queued uploads, at every exit
  impl_->UpdateFineQueued(a,plan,known);
  complete.Complete();         // normal exit: a device error reaches the caller
}
void SparseMultigrid::ConfigureHierarchyPhaseRows(
    const std::vector<std::vector<int>> &a_rows,
    const std::vector<std::vector<int>> &p_rows) {
  impl_->Require(a_rows.size() == impl_->levels.size() && p_rows.size()+1 == a_rows.size()
      && impl_->levels.back().partition.back() <= 128,
      "invalid compact hierarchy levels or terminal size");
  ConfigureFinePhaseRows(a_rows[0]);
  for (size_t l = 1; l < a_rows.size(); ++l)
    impl_->ConfigurePhase(impl_->levels[l].a,a_rows[l]);
  for (size_t l = 0; l < p_rows.size(); ++l)
    impl_->ConfigurePhase(impl_->levels[l].p,p_rows[l]);
  int changes = !a_rows.back().empty();
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&changes,1,MPI_INT,MPI_MAX,impl_->comm);
#endif
  impl_->terminal_phase_changes = changes != 0;
  impl_->hierarchy_phase_configured = true;
}
void SparseMultigrid::PrepareHierarchyPhaseRows(const std::vector<CsrRowUpdate> &a,
    const std::vector<CsrRowUpdate> &p, std::vector<PhaseRowPlan> &a_plans,
    std::vector<PhaseRowPlan> &p_plans, const std::vector<char> *due) const {
  a_plans.clear(); p_plans.clear();
  if (!impl_->hierarchy_phase_configured || a.size() != impl_->levels.size()
      || p.size()+1 != a.size() || (due != nullptr && due->size() != a.size()))
    return;   // the update itself reports the inconsistency
  auto is_due = [due](size_t l) { return due == nullptr || (*due)[l] != 0; };
  a_plans.reserve(a.size()); p_plans.reserve(p.size());
  a_plans.push_back(impl_->Plan(impl_->current_a,a[0],impl_->levels.size() > 1));
  for (size_t l = 1; l < a.size(); ++l)
    a_plans.push_back(is_due(l) ? impl_->Plan(impl_->levels[l].a,a[l],l+1 < a.size())
                                : PhaseRowPlan{});
  for (size_t l = 0; l < p.size(); ++l)
    p_plans.push_back(is_due(l+1) ? impl_->Plan(impl_->levels[l].p,p[l],false) : PhaseRowPlan{});
}
// (defined before UpdateHierarchyPhaseRows, which updates the extra operators)
struct DistributedOperator::Impl {
  std::unique_ptr<SparseMultigrid::Impl> backend;
  SparseMultigrid::Impl::Operator op;
#if MPI_PARALLEL_ENABLED
  Impl(const HostCsr &a, const std::vector<int> &rows, const std::vector<int> &cols,
       MPI_Comm comm) : backend(new SparseMultigrid::Impl(comm)) {
#else
  Impl(const HostCsr &a, const std::vector<int> &rows, const std::vector<int> &cols)
      : backend(new SparseMultigrid::Impl()) {
#endif
    bool valid = true;
    try { Validate(a); } catch (...) { valid = false; }
    backend->Require(valid,"invalid reference rectangular CSR");
    std::vector<std::vector<GlobalIndex>> supplied(2);
    supplied[0].assign(rows.begin(),rows.end());
    supplied[1].assign(cols.begin(),cols.end());
    auto row_partition = backend->Partition(supplied,0,a.rows);
    auto col_partition = backend->Partition(supplied,1,a.cols);
    auto local = SparseMultigrid::Impl::Slice(a,row_partition,backend->rank);
    backend->CheckLocal(local,row_partition,col_partition);
    op = backend->Build(local,row_partition,col_partition);
  }
#if MPI_PARALLEL_ENABLED
  Impl(const LocalCsr &a, const std::vector<GlobalIndex> &rows,
       const std::vector<GlobalIndex> &cols, MPI_Comm comm)
      : backend(new SparseMultigrid::Impl(comm)) {
#else
  Impl(const LocalCsr &a, const std::vector<GlobalIndex> &rows,
       const std::vector<GlobalIndex> &cols) : backend(new SparseMultigrid::Impl()) {
#endif
    const std::vector<std::vector<GlobalIndex>> supplied{rows,cols};
    std::vector<GlobalIndex> row_partition, col_partition;
    {
      SetupLap lap(SetupTimes().partition);
      row_partition = backend->Partition(supplied,0,a.global_rows);
      col_partition = backend->Partition(supplied,1,a.global_cols);
    }
    backend->CheckLocal(a,row_partition,col_partition);
    op = backend->Build(a,row_partition,col_partition);
  }
};
// (after DistributedOperator::Impl: it plans the extra operators too)
void SparseMultigrid::PrepareHierarchyPhaseRows(const std::vector<CsrRowUpdate> &a,
    const std::vector<CsrRowUpdate> &p, std::vector<PhaseRowPlan> &a_plans,
    std::vector<PhaseRowPlan> &p_plans, const std::vector<char> *due,
    const std::vector<PhaseExtra> &extra, std::vector<PhaseRowPlan> &extra_plans,
    int workers) const {
  a_plans.clear(); p_plans.clear(); extra_plans.clear();
  // the operators one after the other, each on the workers (the hierarchy first, in the order of
  // the serial preparation)
  if (impl_->hierarchy_phase_configured && a.size() == impl_->levels.size()
      && p.size()+1 == a.size() && (due == nullptr || due->size() == a.size())) {
    auto is_due = [due](size_t l) { return due == nullptr || (*due)[l] != 0; };
    a_plans.reserve(a.size()); p_plans.reserve(p.size());
    a_plans.push_back(impl_->Plan(impl_->current_a,a[0],impl_->levels.size() > 1,workers));
    for (size_t l = 1; l < a.size(); ++l)
      a_plans.push_back(is_due(l) ? impl_->Plan(impl_->levels[l].a,a[l],l+1 < a.size(),workers)
                                  : PhaseRowPlan{});
    for (size_t l = 0; l < p.size(); ++l)
      p_plans.push_back(is_due(l+1) ? impl_->Plan(impl_->levels[l].p,p[l],false,workers)
                                    : PhaseRowPlan{});
  }
  extra_plans.reserve(extra.size());
  for (const auto &x : extra)
    extra_plans.push_back(x.op->impl_->backend->Plan(x.op->impl_->op,*x.a,false,workers));
}
void SparseMultigrid::UpdateHierarchyPhaseRows(const std::vector<CsrRowUpdate> &a,
                                              const std::vector<CsrRowUpdate> &p,
                                              const std::vector<PhaseRowPlan> *a_plans,
                                              const std::vector<PhaseRowPlan> *p_plans,
                                              const std::vector<PhaseExtra> *extra,
                                              const std::vector<char> *due) {
  PhaseUploadFence complete;   // the queued uploads, at every exit (also a throw)
  impl_->Require(impl_->hierarchy_phase_configured && a.size() == impl_->levels.size()
      && p.size()+1 == a.size() && (due == nullptr || (due->size() == a.size() && (*due)[0])),
      "unconfigured or inconsistent compact hierarchy update");
  // a level that is not due keeps A_l, its diagonal and P_{l-1} (operator q: A_l for q < na,
  // P_{q-na} for q >= na)
  auto is_due = [due](size_t l) { return due == nullptr || (*due)[l] != 0; };
  if (a_plans && a_plans->size() != a.size()) a_plans = nullptr;
  if (p_plans && p_plans->size() != p.size()) p_plans = nullptr;
  const bool terminal_full_update = a.size() == 1 && impl_->fine_diagonal_dirty;
  // ATHENA_SG_LEVEL_STATS=1 (xnode diagnostics): the phase rows of the first update
  static bool stats_done = false;
  if (const char *e = std::getenv("ATHENA_SG_LEVEL_STATS"); !stats_done && e != nullptr && e[0] == '1') {
    stats_done = true;
    auto stat = [](const CsrRowUpdate &u) {
      int longest = 0;
      for (size_t i = 0; i+1 < u.rowptr.size(); ++i) longest = std::max(longest, u.rowptr[i+1]-u.rowptr[i]);
      return std::to_string(u.rows.size())+"/"+std::to_string(u.colidx.size())+"/"+std::to_string(longest);
    };
    for (size_t l = 0; l < a.size(); ++l) {
      std::cout << "# SG_LEVEL_PHASE rank=" << impl_->rank << " level=" << l
                << " A(rows/nnz/longest)=" << stat(a[l]);
      if (l < p.size()) std::cout << " P=" << stat(p[l]);
      std::cout << '\n';
    }
  }
  // xnode: the plans of all operators first (host only), then ONE collective for the validity
  // and ghost-change flags of all of them instead of two per operator; the updates below
  // run in the former order with the same plans (results identical).
  // ATHENA_SG_PHASE_COLLECTIVES=1: the former two collectives per operator (A/B timing).
  const bool batched = !PhaseCollectivesOld();
  // the extra operators in this collective unless ATHENA_SG_PHASE_FORCE_OLD=1
  const size_t nx = (extra != nullptr && batched && !PhaseForceOld()) ? extra->size() : 0;
  const size_t na = a.size(), nops = na+p.size();
  std::vector<PhaseRowPlan> computed(nops+nx);
  std::vector<const PhaseRowPlan*> plans(nops+nx,nullptr);
  std::vector<int> flags(2*(nops+nx),0);
  for (size_t x = 0; x < nx; ++x) {   // host only, as the plans below
    const auto &e = (*extra)[x];
    auto &backend = *e.op->impl_->backend;
    auto &op = e.op->impl_->op;
    const size_t q = nops+x;
    plans[q] = backend.PhasePlan(op,*e.a,false,e.plan,computed[q]);
    flags[2*q] = plans[q]->valid ? 0 : 1;
    const auto &ghosts = plans[q]->same ? op.ghost_ids : plans[q]->ghosts;
    flags[2*q+1] = ghosts != op.ghost_ids ? 1 : 0;
  }
  if (batched) {
    for (size_t q = 0; q < nops; ++q) {
      const bool is_a = q < na;
      const size_t l = is_a ? q : q-na;
      if (!is_due(is_a ? l : l+1)) continue;   // kept: no plan, flags 0
      Impl::Operator &op = is_a ? (l == 0 ? impl_->current_a : impl_->levels[l].a)
                                : impl_->levels[l].p;
      const CsrRowUpdate &u = is_a ? a[l] : p[l];
      const bool smooth = is_a && l+1 < na;
      const PhaseRowPlan *given = is_a ? (a_plans ? &(*a_plans)[l] : nullptr)
                                       : (p_plans ? &(*p_plans)[l] : nullptr);
      plans[q] = impl_->PhasePlan(op,u,smooth,given,computed[q]);
      flags[2*q] = plans[q]->valid ? 0 : 1;
      const auto &ghosts = plans[q]->same ? op.ghost_ids : plans[q]->ghosts;
      flags[2*q+1] = ghosts != op.ghost_ids ? 1 : 0;
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,flags.data(),static_cast<int>(flags.size()),MPI_INT,MPI_MAX,
                  impl_->comm);
#endif
    // every reduced validity flag before the first operator is modified: an invalid
    // update of any operator (hierarchy or extra) leaves all of them as they were, on every rank
    for (size_t q = 0; q < nops+nx; ++q) {
      if (flags[2*q]) {
        throw std::runtime_error("self-gravity: invalid compact phase CSR, ownership, row set, "
                                 "or smoothing diagonal");
      }
    }
  }
  auto plan_of = [&](size_t q, const std::vector<PhaseRowPlan> *given, size_t l) {
    return batched ? plans[q] : (given ? &(*given)[l] : nullptr);
  };
  impl_->UpdateFineQueued(a[0],plan_of(0,a_plans,0),batched ? &flags[0] : nullptr);
  for (size_t l = 1; l < a.size(); ++l) {
    if (!is_due(l)) continue;
    auto &level = impl_->levels[l]; const bool smooth = l+1 < a.size();
    impl_->UpdatePhase(level.a,a[l],smooth,plan_of(l,a_plans,l),batched ? &flags[2*l] : nullptr);
    const double t0 = HostSeconds();
    if (smooth) impl_->UpdatePhaseDiagonal(level.a,level.inverse_diagonal);
    PhaseTimes().diagonal += HostSeconds()-t0;
  }
  for (size_t l = 0; l < p.size(); ++l) {
    if (!is_due(l+1)) continue;
    impl_->UpdatePhase(impl_->levels[l].p,p[l],false,plan_of(na+l,p_plans,l),
                       batched ? &flags[2*(na+l)] : nullptr);
  }
  const double t0 = HostSeconds();
  if ((impl_->terminal_phase_changes && is_due(a.size()-1)) || terminal_full_update)
    impl_->UpdateTerminal(a.back());
  PhaseTimes().terminal += HostSeconds()-t0;
  // the extra operators after the hierarchy, as the former separate calls did
  if (extra != nullptr) {
    for (size_t x = 0; x < extra->size(); ++x) {
      const auto &e = (*extra)[x];
      if (x < nx) {
        e.op->impl_->backend->UpdatePhase(e.op->impl_->op,*e.a,false,plans[nops+x],
                                          &flags[2*(nops+x)]);
      } else {
        e.op->UpdatePhaseRows(*e.a,e.plan);
      }
    }
  }
  complete.Complete();   // the only normal exit: a device error reaches the caller
}
bool SparseMultigrid::ProjectionFusion() const { return impl_->fuse_projection; }
SolveStats SparseMultigrid::Solve(const Vector &rhs, const Vector &x, Real tolerance,
                                 int max_iterations, int restart) {
  return impl_->Solve(rhs, x, tolerance, max_iterations, restart);
}
void SparseMultigrid::ApplyMatrix(int l, const Vector &x, const Vector &result) {
  if (l == 0) impl_->Multiply(impl_->current_a,x,result);
  else impl_->Multiply(impl_->levels.at(l).a,x,result);
}
void SparseMultigrid::ApplyProlongation(int l, const Vector &x, const Vector &result) {
  impl_->Multiply(impl_->levels.at(l).p,x,result);
}
void SparseMultigrid::ApplyRestriction(int l, const Vector &x, const Vector &result) {
  impl_->Multiply(impl_->levels.at(l).r,x,result);
}
void SparseMultigrid::ApplyVCycle(const Vector &rhs, const Vector &result) {
  impl_->Precondition(rhs, result);
}

#if MPI_PARALLEL_ENABLED
DistributedOperator::DistributedOperator(const HostCsr &a, const std::vector<int> &rows,
                                         const std::vector<int> &cols, MPI_Comm comm)
    : impl_(new Impl(a, rows, cols, comm)) {}
DistributedOperator::DistributedOperator(const LocalCsr &a,
    const std::vector<GlobalIndex> &rows, const std::vector<GlobalIndex> &cols, MPI_Comm comm)
    : impl_(new Impl(a,rows,cols,comm)) {}
#else
DistributedOperator::DistributedOperator(const HostCsr &a, const std::vector<int> &rows,
                                         const std::vector<int> &cols)
    : impl_(new Impl(a, rows, cols)) {}
DistributedOperator::DistributedOperator(const LocalCsr &a,
    const std::vector<GlobalIndex> &rows, const std::vector<GlobalIndex> &cols)
    : impl_(new Impl(a,rows,cols)) {}
#endif
DistributedOperator::~DistributedOperator() = default;
StorageStats DistributedOperator::Storage() const {
  StorageStats s;
  s.a_nnz = impl_->op.values.size();
  if (impl_->op.phase_active) {
    s.a_nnz = impl_->op.values.size()-impl_->op.phase_masked_base_nnz
        +impl_->op.phase_values.size();
  }
  SparseMultigrid::Impl::AddStorage(impl_->op,s);
  // the allocations, each counted once
  SparseMultigrid::Impl::Ranges csr, halo;
  SparseMultigrid::Impl::AddAllocations(impl_->op,csr,halo);
  s.csr_bytes = SparseMultigrid::Impl::MergedBytes(std::move(csr));
  s.halo_bytes = SparseMultigrid::Impl::MergedBytes(std::move(halo))
      +SparseMultigrid::Impl::CountBytes(impl_->op);
  return s;
}
void DistributedOperator::ConfigurePhaseRows(const std::vector<int> &rows) {
  impl_->backend->ConfigurePhase(impl_->op,rows);
}
void DistributedOperator::UpdatePhaseRows(const CsrRowUpdate &a, const PhaseRowPlan *plan) {
  PhaseUploadFence complete;   // the queued uploads, at every exit
  impl_->backend->UpdatePhase(impl_->op,a,false,plan);
  complete.Complete();         // normal exit: a device error reaches the caller
}
PhaseRowPlan DistributedOperator::PreparePhaseRows(const CsrRowUpdate &a) const {
  return impl_->backend->Plan(impl_->op,a,false);
}
FineOperatorStats DistributedOperator::PhaseUpdates() const { return impl_->op.phase_updates; }
void DistributedOperator::Apply(const Vector &x, const Vector &result) {
  impl_->backend->Multiply(impl_->op, x, result);
}
}  // namespace gravity
