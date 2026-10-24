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
    Indices rowptr, colidx, send_indices;
    Vector values, send_device, ghost_device;
    HostArray1D<Real> send_host, ghost_host;
    std::vector<int> send_counts, recv_counts, send_offsets, recv_offsets;
  };
  struct Level {
    std::vector<int> partition;
    Operator a, p, r;
    Vector inverse_diagonal, b, x, residual, temp;
  };
  std::vector<Level> levels;
  bool nullspace;
  int rank = 0, nranks = 1;
  int matrix_applications = 0, vcycles = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
#endif
  // Only rank zero stores and solves the tiny dense coarse problem.
  std::vector<Real> coarse_lu, coarse_rhs;
  std::vector<int> coarse_pivots, coarse_counts;

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
#else
  Impl() : nullspace(false) {}
  explicit Impl(const HostHierarchy &h) : nullspace(h.constant_nullspace) { Initialize(h); }
#endif
  ~Impl() {
#if MPI_PARALLEL_ENABLED
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && comm != MPI_COMM_NULL) MPI_Comm_free(&comm);
#endif
  }

  std::vector<int> Partition(const HostHierarchy &h, int l, int n) {
    std::vector<int> part;
    if (l < static_cast<int>(h.row_partitions.size())) part = h.row_partitions[l];
    if (part.empty()) {
      part.resize(nranks+1);
      for (int r = 0; r <= nranks; ++r) {
        part[r] = static_cast<int>(static_cast<long long>(n)*r/nranks);
      }
    }
    if (part.size() != static_cast<size_t>(nranks+1) || part.front() != 0
        || part.back() != n || !std::is_sorted(part.begin(), part.end())) {
      throw std::runtime_error("self-gravity: invalid row partition");
    }
    return part;
  }

  Operator Build(const HostCsr &a, const std::vector<int> &outpart,
                 const std::vector<int> &inpart) {
    Operator op;
    const int begin = outpart[rank], end = outpart[rank+1];
    const int input_begin = inpart[rank], input_end = inpart[rank+1];
    op.rows = end-begin;
    op.inputs = input_end-input_begin;
    std::vector<int> ghosts;
    for (int k = a.rowptr[begin]; k < a.rowptr[end]; ++k) {
      const int j = a.colidx[k];
      if (j < input_begin || j >= input_end) ghosts.push_back(j);
    }
    std::sort(ghosts.begin(), ghosts.end());
    ghosts.erase(std::unique(ghosts.begin(), ghosts.end()), ghosts.end());
    op.recv_counts.assign(nranks, 0);
    op.send_counts.assign(nranks, 0);
    op.recv_offsets.assign(nranks+1, 0);
    op.send_offsets.assign(nranks+1, 0);
    for (int j : ghosts) {
      const int owner = std::upper_bound(inpart.begin(), inpart.end(), j)-inpart.begin()-1;
      ++op.recv_counts[owner];
    }
#if MPI_PARALLEL_ENABLED
    MPI_Alltoall(op.recv_counts.data(), 1, MPI_INT,
                 op.send_counts.data(), 1, MPI_INT, comm);
#endif
    for (int r = 0; r < nranks; ++r) {
      op.recv_offsets[r+1] = op.recv_offsets[r]+op.recv_counts[r];
      op.send_offsets[r+1] = op.send_offsets[r]+op.send_counts[r];
    }
    std::vector<int> requested(op.send_offsets.back());
#if MPI_PARALLEL_ENABLED
    MPI_Alltoallv(ghosts.data(), op.recv_counts.data(), op.recv_offsets.data(), MPI_INT,
                  requested.data(), op.send_counts.data(), op.send_offsets.data(),
                  MPI_INT, comm);
#endif
    for (int &j : requested) j -= input_begin;
    op.send_indices = DeviceCopy(requested, "gravity_halo_indices");
    op.send_device = Vector("gravity_halo_send", requested.size());
    op.ghost_device = Vector("gravity_halo_recv", ghosts.size());
    op.send_host = HostArray1D<Real>("gravity_halo_host_send", requested.size());
    op.ghost_host = HostArray1D<Real>("gravity_halo_host_recv", ghosts.size());
    std::vector<int> rows(op.rows+1), columns;
    std::vector<Real> vals;
    columns.reserve(a.rowptr[end]-a.rowptr[begin]);
    vals.reserve(columns.capacity());
    for (int i = begin; i < end; ++i) {
      rows[i-begin] = columns.size();
      for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) {
        const int j = a.colidx[k];
        columns.push_back(j >= input_begin && j < input_end ? j-input_begin :
            op.inputs+std::lower_bound(ghosts.begin(), ghosts.end(), j)-ghosts.begin());
        vals.push_back(a.values[k]);
      }
    }
    rows.back() = columns.size();
    op.rowptr = DeviceCopy(rows, "gravity_csr_rows");
    op.colidx = DeviceCopy(columns, "gravity_csr_columns");
    op.values = DeviceCopy(vals, "gravity_csr_values");
    return op;
  }

  void Initialize(const HostHierarchy &h) {
    if (h.matrices.empty() || h.prolongation.size()+1 != h.matrices.size()) {
      throw std::runtime_error("self-gravity: hierarchy requires L matrices and L-1 P");
    }
    levels.resize(h.matrices.size());
    for (size_t l = 0; l < levels.size(); ++l) {
      const auto &a = h.matrices[l];
      Validate(a);
      if (a.rows != a.cols) throw std::runtime_error("self-gravity: A must be square");
      // Host setup matrices are identical on all ranks. Validate smoothing diagonals
      // collectively before entering halo construction, including rows owned elsewhere.
      if (l+1 < levels.size()) {
        for (int i = 0; i < a.rows; ++i) {
          Real diagonal = 0;
          for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) {
            if (a.colidx[k] == i) diagonal += a.values[k];
          }
          if (diagonal == 0 || !std::isfinite(Real(1)/diagonal)) {
            throw std::runtime_error("self-gravity: invalid smoothing diagonal");
          }
        }
      }
      levels[l].partition = Partition(h, l, a.rows);
      if (l+1 < levels.size()) {
        const auto &p = h.prolongation[l];
        Validate(p);
        if (p.rows != a.rows || p.cols != h.matrices[l+1].rows) {
          throw std::runtime_error("self-gravity: prolongation shape mismatch");
        }
      }
    }
    // Bound the exceptional root-only CPU solve; this is never a fine-grid fallback.
    if (h.matrices.back().rows > 512) {
      throw std::runtime_error("self-gravity: coarsest matrix exceeds 512 rows");
    }
    for (size_t l = 0; l < levels.size(); ++l) {
      auto &level = levels[l];
      const auto &part = level.partition;
      const auto &a = h.matrices[l];
      level.a = Build(a, part, part);
      const int n = part[rank+1]-part[rank];
      std::vector<Real> diagonal(n);
      for (int i = part[rank]; i < part[rank+1]; ++i) {
        Real d = 0;
        for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) {
          if (a.colidx[k] == i) d += a.values[k];
        }
        // The one-cell null coarse problem has a zero diagonal and needs no smoother.
        diagonal[i-part[rank]] = d == 0 ? 0 : 1/d;
      }
      level.inverse_diagonal = DeviceCopy(diagonal, "gravity_inverse_diagonal");
      level.b = Vector("gravity_level_rhs", n);
      level.x = Vector("gravity_level_solution", n);
      level.residual = Vector("gravity_level_residual", n);
      level.temp = Vector("gravity_level_product", n);
      if (l+1 < levels.size()) {
        level.p = Build(h.prolongation[l], part, levels[l+1].partition);
        level.r = Build(Transpose(h.prolongation[l]), levels[l+1].partition, part);
      }
    }
    const auto &a = h.matrices.back();
    const int n = a.rows;
    int good = 1;
    if (rank == 0) {
      coarse_lu.assign(n*n, nullspace ? Real(1)/n : Real(0));
      coarse_pivots.resize(n);
      coarse_rhs.resize(n);
      for (int i = 0; i < n; ++i) {
        for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) {
          coarse_lu[i*n+a.colidx[k]] += a.values[k];
        }
      }
      // Partial-pivoted LU permits nonsymmetric Galerkin coarse operators.
      for (int k = 0; k < n; ++k) {
        int pivot = k;
        for (int i = k+1; i < n; ++i) {
          if (std::abs(coarse_lu[i*n+k]) > std::abs(coarse_lu[pivot*n+k])) pivot = i;
        }
        coarse_pivots[k] = pivot;
        if (!std::isfinite(coarse_lu[pivot*n+k])
            || std::abs(coarse_lu[pivot*n+k]) < std::numeric_limits<Real>::min()) {
          good = 0;
          break;
        }
        if (pivot != k) {
          for (int j = 0; j < n; ++j) std::swap(coarse_lu[k*n+j], coarse_lu[pivot*n+j]);
        }
        for (int i = k+1; i < n; ++i) {
          coarse_lu[i*n+k] /= coarse_lu[k*n+k];
          for (int j = k+1; j < n; ++j) {
            coarse_lu[i*n+j] -= coarse_lu[i*n+k]*coarse_lu[k*n+j];
          }
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    MPI_Bcast(&good, 1, MPI_INT, 0, comm);
#endif
    if (!good) throw std::runtime_error("self-gravity: singular/nonfinite coarse LU");
    coarse_counts.resize(nranks);
    for (int r = 0; r < nranks; ++r) {
      coarse_counts[r] = levels.back().partition[r+1]-levels.back().partition[r];
    }
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
    Kokkos::deep_copy(op.ghost_device, op.ghost_host);
    auto ghosts = op.ghost_device;
    auto rp = op.rowptr;
    auto ci = op.colidx;
    auto av = op.values;
    const int ni = op.inputs;
    Kokkos::parallel_for("gravity_sparse_matvec", Kokkos::RangePolicy<DevExeSpace>(0, op.rows),
                         KOKKOS_LAMBDA(int i) {
      Real sum = 0;
      for (int k = rp(i); k < rp(i+1); ++k) {
        const int j = ci(k);
        sum += av(k)*(j < ni ? x(j) : ghosts(j-ni));
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

  Real Project(const Vector &x, int global_size) const {
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
    Multiply(l.a, x, r);
    const int n = r.extent_int(0);
    Kokkos::parallel_for("gravity_residual", Kokkos::RangePolicy<DevExeSpace>(0, n),
                         KOKKOS_LAMBDA(int i) { r(i) = b(i)-r(i); });
    Project(r, l.partition.back());
  }

  void Smooth(Level &l) {
    auto x = l.x;
    auto b = l.b;
    auto temp = l.temp;
    auto diagonal = l.inverse_diagonal;
    const int n = x.extent_int(0);
    for (int sweep = 0; sweep < 3; ++sweep) {
      Multiply(l.a, x, temp);
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
    const int n = l.partition.back();
#if MPI_PARALLEL_ENABLED
    MPI_Gatherv(host_b.data(), l.b.extent_int(0), MPI_ATHENA_REAL,
                 coarse_rhs.data(), coarse_counts.data(), l.partition.data(),
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
    MPI_Scatterv(coarse_rhs.data(), coarse_counts.data(), l.partition.data(),
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
    const int global_n = levels[0].partition.back();
    if (rhs.extent_int(0) != n || x.extent_int(0) != n || !(tolerance > 0)
        || !std::isfinite(tolerance) || max_iterations < 0 || restart < 1) {
      throw std::runtime_error("self-gravity: invalid FGMRES arguments");
    }
    SolveStats stats;
    const int initial_matvecs = matrix_applications, initial_cycles = vcycles;
    Vector b("gravity_krylov_rhs", n), r("gravity_krylov_residual", n);
    Vector w("gravity_krylov_work", n);
    Kokkos::deep_copy(b, rhs);
    stats.removed_rhs_mean = Project(b, global_n);
    Project(x, global_n);
    const Real norm_b = Norm(b);
    const Real denominator = norm_b > 0 ? norm_b : 1;
    const Real target = tolerance*denominator;
    Residual(levels[0], b, x, r);
    Real beta = Norm(r);
    const int m = std::min(restart, std::max(1, max_iterations));
    std::vector<Vector> v, z;
    for (int j = 0; j <= m; ++j) v.emplace_back("gravity_arnoldi", n);
    for (int j = 0; j < m; ++j) z.emplace_back("gravity_flexible_basis", n);
    std::vector<Real> h((m+1)*m), cosine(m), sine(m), g(m+1), y(m);
    while (beta > target && stats.iterations < max_iterations && std::isfinite(beta)) {
      std::fill(h.begin(), h.end(), Real(0));
      std::fill(g.begin(), g.end(), Real(0));
      g[0] = beta;
      Kokkos::deep_copy(v[0], r);
      Scale(1/beta, v[0]);
      int used = 0;
      for (int j = 0; j < m && stats.iterations < max_iterations; ++j) {
        Precondition(v[j], z[j]);
        Multiply(levels[0].a, z[j], w);
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
      Residual(levels[0], b, x, r);
      beta = Norm(r);
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
#else
SparseMultigrid::SparseMultigrid(const HostHierarchy &h) : impl_(new Impl(h)) {}
#endif
SparseMultigrid::~SparseMultigrid() = default;
int SparseMultigrid::Levels() const { return impl_->levels.size(); }
int SparseMultigrid::LocalBegin(int l) const {
  return impl_->levels.at(l).partition[impl_->rank];
}
int SparseMultigrid::LocalEnd(int l) const {
  return impl_->levels.at(l).partition[impl_->rank+1];
}
int SparseMultigrid::LocalSize(int l) const { return LocalEnd(l)-LocalBegin(l); }
SolveStats SparseMultigrid::Solve(const Vector &rhs, const Vector &x, Real tolerance,
                                 int max_iterations, int restart) {
  return impl_->Solve(rhs, x, tolerance, max_iterations, restart);
}
void SparseMultigrid::ApplyMatrix(int l, const Vector &x, const Vector &result) {
  impl_->Multiply(impl_->levels.at(l).a, x, result);
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
    Validate(a);
    HostHierarchy partition_input;
    partition_input.row_partitions = {rows, cols};
    auto row_partition = backend->Partition(partition_input, 0, a.rows);
    auto col_partition = backend->Partition(partition_input, 1, a.cols);
    op = backend->Build(a, row_partition, col_partition);
  }
};
#if MPI_PARALLEL_ENABLED
DistributedOperator::DistributedOperator(const HostCsr &a, const std::vector<int> &rows,
                                         const std::vector<int> &cols, MPI_Comm comm)
    : impl_(new Impl(a, rows, cols, comm)) {}
#else
DistributedOperator::DistributedOperator(const HostCsr &a, const std::vector<int> &rows,
                                         const std::vector<int> &cols)
    : impl_(new Impl(a, rows, cols)) {}
#endif
DistributedOperator::~DistributedOperator() = default;
void DistributedOperator::Apply(const Vector &x, const Vector &result) {
  impl_->backend->Multiply(impl_->op, x, result);
}
}  // namespace gravity
