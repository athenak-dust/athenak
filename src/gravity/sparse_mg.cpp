//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/sparse_mg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>

namespace gravity {
namespace {
using Vector = DvceArray1D<Real>;
using Indices = DvceArray1D<int>;

template <typename T>
DvceArray1D<T> DeviceCopy(const std::vector<T> &data, const char *name) {
  DvceArray1D<T> result(std::string(name), data.size());
  auto host = Kokkos::create_mirror_view(result);
  for (std::size_t i = 0; i < data.size(); ++i) host(i) = data[i];
  Kokkos::deep_copy(result, host);
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
    HostArray1D<Real> send_host, ghost_host;
    std::vector<int> send_counts, recv_counts, send_offsets, recv_offsets;
    std::vector<GlobalIndex> input_partition, ghost_ids;
    bool phase_configured = false, phase_active = false;
    Indices phase_map, phase_rows, phase_rowptr, phase_colidx;
    Vector phase_values;
    std::vector<int> phase_rows_host, phase_rowptr_host;
    std::vector<GlobalIndex> phase_columns_host, phase_base_ghosts;
    std::unordered_map<GlobalIndex,int> phase_ghost_slots;
    std::uint64_t phase_masked_base_nnz = 0;
    FineOperatorStats phase_updates;
  };
  struct Level {
    std::vector<GlobalIndex> partition;
    Operator a, p, r;
    Vector inverse_diagonal, b, x, residual, temp;
  };
  static void AddPhaseStorage(const Operator &op, StorageStats &s) {
    s.csr_bytes += (op.phase_rowptr.size()+op.phase_colidx.size())*sizeof(int)
        +op.phase_values.size()*sizeof(Real);
    s.phase_map_bytes += (op.phase_map.size()+op.phase_rows.size())*sizeof(int);
    s.phase_rows += op.phase_rows_host.size();
    s.phase_nnz += op.phase_values.size();
    s.halo_bytes += op.recv_slots.size()*sizeof(int);
    if (op.recv_device.data() != op.ghost_device.data()) {
      s.halo_bytes += op.recv_device.size()*sizeof(Real);
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
    s.csr_bytes += (op.rowptr.size()+op.colidx.size())*sizeof(int)
                   +op.values.size()*sizeof(Real);
    s.halo_bytes += op.send_indices.size()*sizeof(int)
        +(op.send_device.size()+op.ghost_device.size()+op.send_host.size()
          +op.ghost_host.size())*sizeof(Real)
        +(op.send_counts.size()+op.recv_counts.size()+op.send_offsets.size()
          +op.recv_offsets.size())*sizeof(int);
    AddPhaseStorage(op,s);
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
  Vector krylov_b, krylov_r, krylov_w;
  std::vector<Vector> krylov_v, krylov_z;
  std::vector<Real> krylov_h, krylov_cosine, krylov_sine, krylov_g, krylov_y;
  int krylov_restart = 0;
  bool nullspace;
  int rank = 0, nranks = 1;
  int matrix_applications = 0, vcycles = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
#endif
  // Only rank zero stores and solves the tiny dense coarse problem.
  std::vector<Real> coarse_lu, coarse_rhs;
  std::vector<int> coarse_pivots, coarse_counts, coarse_offsets;

#if MPI_PARALLEL_ENABLED
  explicit Impl(MPI_Comm input_comm) : nullspace(false) {
    MPI_Comm_dup(input_comm, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
  }
  Impl(const HostHierarchy &h, MPI_Comm input_comm) : nullspace(h.constant_nullspace) {
    MPI_Comm_dup(input_comm, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nranks);
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
    if (!finalized && comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
#endif
  }

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
          && a.rowptr.back() == static_cast<int>(a.values.size())
          && std::is_sorted(a.rowptr.begin(), a.rowptr.end());
      for (size_t k = 0; good && k < a.values.size(); ++k) {
        good = a.colidx[k] >= 0 && a.colidx[k] < a.global_cols
            && std::isfinite(a.values[k]);
      }
    }
    Require(good, "invalid owned CSR shape, offsets, columns, values, or ownership");
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
    for (int r = 0; r < nranks; ++r) {
      op.recv_offsets[r+1] = op.recv_offsets[r]+op.recv_counts[r];
      op.send_offsets[r+1] = op.send_offsets[r]+op.send_counts[r];
    }
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
    op.send_indices = DeviceCopy(send_indices,"gravity_halo_indices");
    op.send_device = Vector("gravity_halo_send",requested.size());
    op.ghost_device = Vector("gravity_halo_recv",ghosts.size());
    op.recv_device = identity ? op.ghost_device : Vector("gravity_grouped_recv",ghosts.size());
    op.recv_slots = identity ? Indices() : DeviceCopy(slots,"gravity_union_slots");
    op.send_host = HostArray1D<Real>("gravity_halo_host_send",requested.size());
    op.ghost_host = HostArray1D<Real>("gravity_halo_host_recv",ghosts.size());
  }

  Operator Build(const LocalCsr &a, const std::vector<GlobalIndex> &outpart,
                 const std::vector<GlobalIndex> &inpart) {
    Operator op;
    op.rows = a.LocalRows(); op.inputs = static_cast<int>(inpart[rank+1]-inpart[rank]);
    op.global_rows = a.global_rows; op.global_cols = a.global_cols; op.row_begin = a.row_begin;
    op.input_partition = inpart;
    std::vector<GlobalIndex> ghosts;
    for (GlobalIndex j : a.colidx) {
      if (j < inpart[rank] || j >= inpart[rank+1]) ghosts.push_back(j);
    }
    std::sort(ghosts.begin(),ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(),ghosts.end()),ghosts.end());
    BuildHalo(op,ghosts);
    std::vector<int> columns(a.colidx.size());
    for (size_t k = 0; k < a.colidx.size(); ++k) {
      const GlobalIndex j = a.colidx[k];
      columns[k] = j >= inpart[rank] && j < inpart[rank+1] ? static_cast<int>(j-inpart[rank]) :
          op.inputs+static_cast<int>(std::lower_bound(ghosts.begin(),ghosts.end(),j)-ghosts.begin());
    }
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
    if (!rows.empty()) BuildGhostSlots(op);
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

  bool UpdatePhase(Operator &op, const CsrRowUpdate &a, bool smoothing_diagonal = false) {
    bool valid = op.phase_configured && a.global_rows == op.global_rows
        && a.global_cols == op.global_cols && a.row_begin == op.row_begin
        && a.rows == op.phase_rows_host && a.rowptr.size() == a.rows.size()+1
        && a.colidx.size() == a.values.size()
        && a.values.size() <= static_cast<size_t>(std::numeric_limits<int>::max());
    if (valid) {
      valid = a.rowptr.front() == 0 && a.rowptr.back() == static_cast<int>(a.values.size())
          && std::is_sorted(a.rowptr.begin(),a.rowptr.end());
      for (size_t k = 0; valid && k < a.values.size(); ++k) {
        valid = a.colidx[k] >= 0 && a.colidx[k] < op.global_cols && std::isfinite(a.values[k]);
      }
      if (smoothing_diagonal) for (size_t j = 0; valid && j < a.rows.size(); ++j) {
        Real diagonal = 0;
        for (int k = a.rowptr[j]; k < a.rowptr[j+1]; ++k) {
          if (a.colidx[k] == a.row_begin+a.rows[j]) diagonal += a.values[k];
        }
        valid = diagonal != 0 && std::isfinite(diagonal)
            && std::isfinite(Real(1)/diagonal);
      }
    }
    Require(valid,"invalid compact phase CSR, ownership, row set, or smoothing diagonal");
    const bool same = a.rowptr == op.phase_rowptr_host && a.colidx == op.phase_columns_host;
    if (!same) ++op.phase_updates.local_pattern_changes;
    std::vector<GlobalIndex> next_ghosts;
    const auto begin = op.input_partition[rank], end = op.input_partition[rank+1];
    if (!same) {
      next_ghosts = op.phase_base_ghosts;
      std::vector<GlobalIndex> extra;
      for (GlobalIndex j : a.colidx) {
        if (j >= begin && j < end) continue;
        const auto slot = op.phase_ghost_slots.find(j);
        if (slot == op.phase_ghost_slots.end()
            || slot->second >= static_cast<int>(op.phase_base_ghosts.size())) extra.push_back(j);
      }
      std::sort(extra.begin(),extra.end());
      extra.erase(std::unique(extra.begin(),extra.end()),extra.end());
      next_ghosts.insert(next_ghosts.end(),extra.begin(),extra.end());
    }
    const auto &ghosts = same ? op.ghost_ids : next_ghosts;
    const int local_ghost_change = ghosts != op.ghost_ids;
    int global_ghost_change = local_ghost_change;
#if MPI_PARALLEL_ENABLED
    // A remote rank may change requests addressed to this rank even when our
    // own CSR/ghost list is unchanged. Every rank must rebuild the send plan.
    MPI_Allreduce(MPI_IN_PLACE,&global_ghost_change,1,MPI_INT,MPI_MAX,comm);
#endif
    op.phase_updates.local_ghost_changes += local_ghost_change;
    op.phase_updates.global_ghost_changes += global_ghost_change;
    if (global_ghost_change) {
      BuildHalo(op,ghosts);
      ++op.phase_updates.halo_rebuilds;
    } else {
      ++op.phase_updates.halo_reuses;
    }
    if (local_ghost_change) BuildGhostSlots(op);
    // No MPI occurs below this point: CSR/value uploads are local decisions.
    if (same) {
      auto host = Kokkos::create_mirror_view(op.phase_values);
      for (size_t k = 0; k < a.values.size(); ++k) host(k) = a.values[k];
      Kokkos::deep_copy(op.phase_values,host);
      ++op.phase_updates.value_updates;
    } else {
      std::vector<int> columns(a.colidx.size());
      for (size_t k = 0; k < a.colidx.size(); ++k) {
        const GlobalIndex j = a.colidx[k];
        columns[k] = j >= begin && j < end ? static_cast<int>(j-begin)
            : op.inputs+op.phase_ghost_slots.at(j);
      }
      if (op.phase_rowptr.size() == a.rowptr.size()) {
        auto host = Kokkos::create_mirror_view(op.phase_rowptr);
        for (size_t k = 0; k < a.rowptr.size(); ++k) host(k) = a.rowptr[k];
        Kokkos::deep_copy(op.phase_rowptr,host);
      } else {
        op.phase_rowptr = DeviceCopy(a.rowptr,"gravity_phase_rowptr");
      }
      op.phase_colidx = DeviceCopy(columns,"gravity_phase_columns");
      op.phase_values = DeviceCopy(a.values,"gravity_phase_values");
      op.phase_rowptr_host = a.rowptr;
      op.phase_columns_host = a.colidx;
      ++op.phase_updates.structure_rebuilds;
    }
    op.phase_active = true;
    ++op.phase_updates.updates;
    return same != 0;
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
    for (size_t l = 0; l < levels.size(); ++l) {
      levels[l].partition = Partition(h.row_partitions,l,h.matrices[l].global_rows);
      CheckLocal(h.matrices[l],levels[l].partition,levels[l].partition);
      bool good_diagonal = true;
      if (l+1 < levels.size()) {
        const auto &a = h.matrices[l];
        for (int i = 0; i < a.LocalRows(); ++i) {
          Real diagonal = 0;
          for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) {
            if (a.colidx[k] == a.row_begin+i) diagonal += a.values[k];
          }
          good_diagonal = good_diagonal && diagonal != 0 && std::isfinite(diagonal)
              && std::isfinite(Real(1)/diagonal);
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
      std::vector<Real> diagonal(n);
      for (int i = 0; i < n; ++i) {
        Real d = 0;
        for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) {
          if (a.colidx[k] == a.row_begin+i) d += a.values[k];
        }
        diagonal[i] = d == 0 ? 0 : 1/d;
      }
      level.inverse_diagonal = DeviceCopy(diagonal, "gravity_inverse_diagonal");
      level.b = Vector("gravity_level_rhs",n);
      level.x = Vector("gravity_level_solution",n);
      level.residual = Vector("gravity_level_residual",n);
      level.temp = Vector("gravity_level_product",n);
      if (l+1 < levels.size()) {
        level.p = Build(h.prolongation[l],part,levels[l+1].partition);
        level.r = Build(h.restriction[l],levels[l+1].partition,part);
      }
    }
    // Shared CSR views remain immutable in the cached hierarchy. Halo scratch is
    // used sequentially. First refresh detaches outer values; new patterns rebuild CSR.
    current_a = levels[0].a;
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

  void Multiply(Operator &op, const Vector &x, const Vector &y) {
    if (x.extent_int(0) != op.inputs || y.extent_int(0) != op.rows) {
      throw std::runtime_error("self-gravity: local matrix/vector shape mismatch");
    }
    auto send = op.send_device;
    auto send_indices = op.send_indices;
    const int nsend = send.extent_int(0);
    Kokkos::parallel_for("gravity_halo_pack", Kokkos::RangePolicy<DevExeSpace>(0, nsend),
                         KOKKOS_LAMBDA(int i) { send(i) = x(send_indices(i)); });
    Kokkos::deep_copy(op.send_host, send);
#if MPI_PARALLEL_ENABLED
    MPI_Alltoallv(op.send_host.data(), op.send_counts.data(), op.send_offsets.data(),
                  MPI_ATHENA_REAL, op.ghost_host.data(), op.recv_counts.data(),
                  op.recv_offsets.data(), MPI_ATHENA_REAL, comm);
#endif
    Kokkos::deep_copy(op.recv_device, op.ghost_host);
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

  Real Norm(const Vector &x) const { return std::sqrt(Dot(x, x)); }

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

  void CoarseSolve(Level &l) {
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
    Project(l.x, n);
  }

  void Cycle(int index) {
    auto &l = levels[index];
    Project(l.b, l.partition.back());
    Kokkos::deep_copy(l.x, Real(0));
    if (index+1 == static_cast<int>(levels.size())) {
      CoarseSolve(l);
      return;
    }
    Smooth(l);
    Residual(l, l.b, l.x, l.residual);
    Multiply(l.r, l.residual, levels[index+1].b);
    Cycle(index+1);
    Multiply(l.p, levels[index+1].x, l.temp);
    Axpy(1, l.temp, l.x);
    Smooth(l);
  }

  void Precondition(const Vector &rhs, const Vector &result) {
    if (rhs.extent_int(0) != levels[0].b.extent_int(0)
        || result.extent_int(0) != rhs.extent_int(0)) {
      throw std::runtime_error("self-gravity: V-cycle local vector shape mismatch");
    }
    Kokkos::deep_copy(levels[0].b, rhs);
    Cycle(0);
    Kokkos::deep_copy(result, levels[0].x);
    ++vcycles;
  }

  SolveStats Solve(const Vector &rhs, const Vector &x, Real tolerance,
                   int max_iterations, int restart) {
    const int n = levels[0].a.rows;
    const GlobalIndex global_n = levels[0].partition.back();
    if (rhs.extent_int(0) != n || x.extent_int(0) != n || !(tolerance > 0)
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
    auto b = krylov_b, r = krylov_r, w = krylov_w;
    auto &v = krylov_v;
    auto &z = krylov_z;
    auto &h = krylov_h;
    auto &cosine = krylov_cosine;
    auto &sine = krylov_sine;
    auto &g = krylov_g;
    auto &y = krylov_y;
    Kokkos::deep_copy(b, rhs);
    stats.removed_rhs_mean = Project(b, global_n);
    Project(x, global_n);
    const Real norm_b = Norm(b);
    const Real denominator = norm_b > 0 ? norm_b : 1;
    const Real target = tolerance*denominator;
    CurrentResidual(b,x,r);
    Real beta = Norm(r);
    while (beta > target && stats.iterations < max_iterations && std::isfinite(beta)) {
      std::fill(h.begin(), h.end(), Real(0));
      std::fill(g.begin(), g.end(), Real(0));
      g[0] = beta;
      Kokkos::deep_copy(v[0], r);
      Scale(1/beta, v[0]);
      int used = 0;
      for (int j = 0; j < m && stats.iterations < max_iterations; ++j) {
        Precondition(v[j], z[j]);
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
        h[(j+1)*m+j] = next;
        if (next > 0) {
          Kokkos::deep_copy(v[j+1], w);
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
        if (std::abs(g[j+1]) <= target || next == 0 || rotation_norm == 0) break;
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
      if (beta > 1000*std::max(std::abs(g[used]),target)) {
        stats.preconditioner_breakdown = true;
        break;
      }
    }
    stats.relative_residual = beta/denominator;
    stats.converged = std::isfinite(beta) && beta <= target;
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
  if (a.rowptr.data() != cached.rowptr.data()) s.csr_bytes += a.rowptr.size()*sizeof(int);
  if (a.colidx.data() != cached.colidx.data()) s.csr_bytes += a.colidx.size()*sizeof(int);
  if (a.values.data() != cached.values.data()) s.csr_bytes += a.values.size()*sizeof(Real);
  if (a.send_indices.data() != cached.send_indices.data()) s.halo_bytes += a.send_indices.size()*sizeof(int);
  if (a.send_device.data() != cached.send_device.data()) s.halo_bytes += a.send_device.size()*sizeof(Real);
  if (a.ghost_device.data() != cached.ghost_device.data()) s.halo_bytes += a.ghost_device.size()*sizeof(Real);
  if (a.send_host.data() != cached.send_host.data()) s.halo_bytes += a.send_host.size()*sizeof(Real);
  if (a.ghost_host.data() != cached.ghost_host.data()) s.halo_bytes += a.ghost_host.size()*sizeof(Real);
  s.halo_bytes += (a.send_counts.size()+a.recv_counts.size()+a.send_offsets.size()
                  +a.recv_offsets.size())*sizeof(int);
  Impl::AddPhaseStorage(a,s);
  s.terminal_host_csr_bytes = impl_->terminal_matrix.rowptr.capacity()*sizeof(int)
      +impl_->terminal_matrix.colidx.capacity()*sizeof(GlobalIndex)
      +impl_->terminal_matrix.values.capacity()*sizeof(Real);
  s.host_pattern_bytes += impl_->current_rowptr.capacity()*sizeof(int)
      +impl_->current_columns.capacity()*sizeof(GlobalIndex);
  s.vector_bytes += (impl_->coarse_lu.size()+impl_->coarse_rhs.size())*sizeof(Real)
      +(impl_->coarse_pivots.size()+impl_->coarse_counts.size()
        +impl_->coarse_offsets.size())*sizeof(int);
  s.vector_bytes += (impl_->krylov_b.size()+impl_->krylov_r.size()
                    +impl_->krylov_w.size())*sizeof(Real);
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
  const bool smooth = impl_->levels.size() > 1;
  const bool same = impl_->UpdatePhase(impl_->current_a,a,smooth);
  if (smooth) impl_->UpdatePhaseDiagonal(impl_->current_a,impl_->levels[0].inverse_diagonal,
                                        impl_->fine_diagonal_dirty);
  impl_->fine_diagonal_dirty = false;
  ++impl_->fine_updates.updates;
  if (same) ++impl_->fine_updates.value_updates;
  else ++impl_->fine_updates.structure_rebuilds;
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
void SparseMultigrid::UpdateHierarchyPhaseRows(const std::vector<CsrRowUpdate> &a,
                                              const std::vector<CsrRowUpdate> &p) {
  impl_->Require(impl_->hierarchy_phase_configured && a.size() == impl_->levels.size()
      && p.size()+1 == a.size(),"unconfigured or inconsistent compact hierarchy update");
  const bool terminal_full_update = a.size() == 1 && impl_->fine_diagonal_dirty;
  UpdateFinePhaseRows(a[0]);
  for (size_t l = 1; l < a.size(); ++l) {
    auto &level = impl_->levels[l]; const bool smooth = l+1 < a.size();
    impl_->UpdatePhase(level.a,a[l],smooth);
    if (smooth) impl_->UpdatePhaseDiagonal(level.a,level.inverse_diagonal);
  }
  for (size_t l = 0; l < p.size(); ++l) impl_->UpdatePhase(impl_->levels[l].p,p[l]);
  if (impl_->terminal_phase_changes || terminal_full_update) impl_->UpdateTerminal(a.back());
}
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
    auto row_partition = backend->Partition(supplied,0,a.global_rows);
    auto col_partition = backend->Partition(supplied,1,a.global_cols);
    backend->CheckLocal(a,row_partition,col_partition);
    op = backend->Build(a,row_partition,col_partition);
  }
};
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
  return s;
}
void DistributedOperator::ConfigurePhaseRows(const std::vector<int> &rows) {
  impl_->backend->ConfigurePhase(impl_->op,rows);
}
void DistributedOperator::UpdatePhaseRows(const CsrRowUpdate &a) {
  impl_->backend->UpdatePhase(impl_->op,a);
}
FineOperatorStats DistributedOperator::PhaseUpdates() const { return impl_->op.phase_updates; }
void DistributedOperator::Apply(const Vector &x, const Vector &result) {
  impl_->backend->Multiply(impl_->op, x, result);
}
}  // namespace gravity
