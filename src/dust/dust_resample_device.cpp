//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_resample_device.cpp
//! \brief The per-patch resampling of dust_resample.cpp
//! (ResamplePatch, ReduceCap, PivotedQR) on the device, same algorithm and the same patches:
//! rows, row scaling, the stream of active sets, Householder QR with column pivoting, null
//! vectors, end-point reduction with the cap, residual check and fallback.
//!
//! One team per patch, patches handed out by a device counter.  Every sum that the host
//! forms serially is formed serially by one team thread in the host order (a QR column, a
//! row of the check, a back substitution, the step search along one null vector), and the
//! maxima / minima the host takes with strict comparisons break ties by the smallest index.
//! The device result therefore follows the host's arithmetic order, except the Householder norm
//! below the QR diagonal, which is a team reduction (its order is Kokkos's and can
//! depend on the backend and the team size).  It can differ from the host by round-off (that sum,
//! fused multiply-add, sqrt / hypot of the device library), which the comparison mode
//! (<dust>/cloud_resample_compare) measures.  Bitwise reproducibility between runs is not
//! guaranteed on every backend: repeated OpenMP runs already differ (no fallback, row residuals
//! <= 1e-13; cause not located, and meeting the constraints does not by itself rule out a race).
//!
//! Exception: with <dust>/cloud_resample_qr_rows = true (the default) the QR
//! update sums each column by row chunks over the team, in another order than the host.  The
//! device and host solves then agree only to round-off, and the end-point reduction can turn a
//! round-off difference into another end point of the null space: the kept clouds and their
//! masses may differ from the host's (both satisfy every constraint to round-off).  Compare
//! with cloud_resample_compare by tolerance, or set cloud_resample_qr_rows = false to recover
//! the host's order.
//!
//! Workspaces live in global memory, one per team slot, in two buckets of patch sizes (at
//! most 64 clouds, more than 64) sized from the largest patch of the bucket; the number of
//! slots is limited by <dust>/cloud_resample_device_mem_mb.  The per-patch sizes (clouds,
//! CSR entries, distinct leaves) are computed on the device; only these few integers per
//! patch go to the host to size the buckets.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "dust_resample.hpp"

namespace dust {
namespace resample {
namespace {

using Policy = Kokkos::TeamPolicy<DevExeSpace>;
using Member = Policy::member_type;
// large patches: up to 512 threads per team, registers limited accordingly
using PolicyWide = Kokkos::TeamPolicy<DevExeSpace, Kokkos::LaunchBounds<512, 1>>;
// with cloud_resample_device_wide_team = 1024: teams of up to 1024 threads
using PolicyWider = Kokkos::TeamPolicy<DevExeSpace, Kokkos::LaunchBounds<1024, 1>>;
// a step search along one null vector is split in this many chunks (32 instead of
// 8, so that 2 x 8 x 32 threads of a 1024-thread team share the scan; the chunks are merged in
// their order with strict comparisons, so the result does not depend on the number)
constexpr int kChunk = 32;
// cloud_resample_qr_rows: row chunks per column in the dot products of the
// row-parallel QR update
constexpr int kQRow = 16;

// sizes of the per-team workspace of a bucket
struct WsSize {
  int nmax = 0, emax = 1, rmax = 8, cmax = 32;   // clouds, CSR entries, rows, active columns
  int lmax = 1, hmax = 4;                        // distinct leaves, hash table slots (power of 2)
  std::size_t nd = 0, ni = 0;                     // doubles and ints per team slot
};

WsSize Sizes(const int nmax, const int emax, const int lmax) {
  WsSize z;
  z.nmax = std::max(nmax, 2);
  z.emax = std::max(emax, 1);
  z.lmax = std::max(lmax, 1);
  z.hmax = 4;
  while (z.hmax < 4*z.lmax) z.hmax *= 2;
  z.rmax = 7*lmax + 8;
  const int rb = std::min(z.rmax, z.nmax);
  z.cmax = std::max(2*rb, rb + 32);
  const std::size_t r = z.rmax, n = z.nmax, c = z.cmax;
  z.nd = 2*r*n + r + n + 2*r*c + 2*c*c + 10*c + 32 + 3*2*c*kChunk + 3*c + c*kQRow + c;
  z.ni = static_cast<std::size_t>(z.hmax) + 2*static_cast<std::size_t>(z.lmax) + r + 2*n + 6*c + 16
       + 2*2*c*kChunk;
  return z;
}


struct Ws {
  double *A, *B, *r0, *w, *bact, *qr, *z, *xs, *vn1, *vn2, *d, *wa, *ua, *tt, *zmx, *sc;
  double *pa0, *pa1, *pzm, *score, *fac;   // chunk partials, scores, update factors
  double *qp, *qs;   // row-chunk partial dot products and the column factors of a QR step
  int *tab, *ltmp, *leaves, *rowidx, *act, *left, *perm, *kk, *fixed, *zr, *cnt, *pk0, *pk1;
};

KOKKOS_INLINE_FUNCTION
Ws Carve(double *db, int *ib, const WsSize &z) {
  Ws ws;
  const std::size_t r = z.rmax, n = z.nmax, c = z.cmax;
  ws.A = db; db += r*n;
  ws.B = db; db += r*n;
  ws.r0 = db; db += r;
  ws.w = db; db += n;
  ws.bact = db; db += r*c;
  ws.qr = db; db += r*c;
  ws.z = db; db += c*c;
  ws.xs = db; db += c*c;
  ws.vn1 = db; db += c;
  ws.vn2 = db; db += c + 1;                // one slot past the norms: Householder beta
  ws.d = db; db += c;
  ws.wa = db; db += c;
  ws.ua = db; db += c;
  ws.tt = db; db += 2*c;
  ws.zmx = db; db += c;
  ws.sc = db; db += 16;                 // 16 scalars
  ws.pa0 = db; db += 2*c*kChunk;
  ws.pa1 = db; db += 2*c*kChunk;
  ws.pzm = db; db += 2*c*kChunk;
  ws.score = db; db += 2*c;
  ws.fac = db; db += c;
  ws.qp = db; db += c*kQRow;
  ws.qs = db;
  ws.tab = ib; ib += z.hmax;
  ws.ltmp = ib; ib += z.lmax;
  ws.leaves = ib; ib += z.lmax;
  ws.rowidx = ib; ib += r;
  ws.act = ib; ib += n;
  ws.left = ib; ib += n;
  ws.perm = ib; ib += c;
  ws.kk = ib; ib += 2*c;
  ws.fixed = ib; ib += c;
  ws.zr = ib; ib += 2*c;
  ws.cnt = ib; ib += 16;                // 16 counters
  ws.pk0 = ib; ib += 2*c*kChunk;
  ws.pk1 = ib;
  return ws;
}

// the distinct values among ne integers, by a team: a hash set with linear probing (tab, H
// slots, H a power of 2 at least twice the number of distinct values), then the members in
// ascending order by their ranks.  Returns how many; sorted values in out when out != nullptr.
// The set does not depend on the insertion order, so the result is deterministic.
template <typename Get>
KOKKOS_INLINE_FUNCTION
int DistinctSorted(const Member &tm, const int ne, const Get &get, int *tab, const int hmask,
                   int *tmp, int *out) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, hmask + 1), [&](const int h) {tab[h] = -1;});
  tm.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ne), [&](const int e) {
    const int key = get(e);
    unsigned h = (static_cast<unsigned>(key)*2654435761u) & static_cast<unsigned>(hmask);
    while (true) {
      const int prev = Kokkos::atomic_compare_exchange(&tab[h], -1, key);
      if (prev == -1 || prev == key) break;
      h = (h + 1) & static_cast<unsigned>(hmask);
    }
  });
  tm.team_barrier();
  int nd = 0;
  if (tmp == nullptr) {
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, hmask + 1), [&](const int h, int &c) {
      if (tab[h] != -1) ++c;
    }, nd);
    return nd;
  }
  Kokkos::parallel_scan(Kokkos::TeamThreadRange(tm, hmask + 1), [&](const int h, int &upd,
                                                                     const bool final) {
    if (tab[h] != -1) {
      if (final) tmp[upd] = tab[h];
      ++upd;
    }
  }, nd);
  tm.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nd), [&](const int i) {
    const int x = tmp[i];
    int r = 0;
    for (int j=0; j<nd; ++j) r += (tmp[j] < x) ? 1 : 0;
    out[r] = x;
  });
  tm.team_barrier();
  return nd;
}

KOKKOS_INLINE_FUNCTION
int LowerBound(const int *v, const int n, const int x) {
  int lo = 0, hi = n;
  while (lo < hi) {
    const int mid = (lo + hi)/2;
    if (v[mid] < x) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// Householder QR with column pivoting of the m x n matrix a, stored by rows (a[i*n + j]) so
// that the threads of a team, one per column, read consecutive addresses: a P = Q R; the
// numerical rank.  Every thread runs the host PivotedQR loops of its column in the host order.
KOKKOS_INLINE_FUNCTION
int PivotedQRTeam(const Member &tm, double *a, const int m, const int n, int *perm, double *vn1,
                  double *vn2, double *qp, double *qs, const bool rows) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int j) {
    perm[j] = j;
    double s = 0.0;
    for (int i=0; i<m; ++i) s += a[static_cast<std::size_t>(i)*n + j]*a[static_cast<std::size_t>(i)*n + j];
    vn1[j] = vn2[j] = Kokkos::sqrt(s);
  });
  tm.team_barrier();
  const int kmax = (m < n) ? m : n;
  const double tol3z = 1.4901161193847656e-08;   // sqrt(DBL_EPSILON)
  double first = 0.0;
  for (int k=0; k<kmax; ++k) {
    // the host keeps the first column of largest norm
    double vmax = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, k, n), [&](const int j, double &v) {
      if (vn1[j] > v) v = vn1[j];
    }, Kokkos::Max<double>(vmax));
    int p = n;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, k, n), [&](const int j, int &v) {
      if (vn1[j] == vmax && j < v) v = j;
    }, Kokkos::Min<int>(p));
    // the pivot norm from the team reduction (bitwise vn1[p]), not from vn1: the leader may
    // already be swapping vn1[p] and vn1[k] below while a thread reads it
    const double vp = (p < n) ? vmax : vn1[k];
    if (p >= n) p = k;
    if (k == 0) first = vp;
    if (!(vp > kRankTol*first) || first == 0.0) {
      tm.team_barrier();
      return k;
    }
    if (p != k) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, m), [&](const int i) {
        double *row = a + static_cast<std::size_t>(i)*n;
        const double t = row[p]; row[p] = row[k]; row[k] = t;
      });
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {
        int ti = perm[p]; perm[p] = perm[k]; perm[k] = ti;
        double t = vn1[p]; vn1[p] = vn1[k]; vn1[k] = t;
        t = vn2[p]; vn2[p] = vn2[k]; vn2[k] = t;
      });
      tm.team_barrier();
    }
    // Householder vector of column k (entries a[i*n + k], i >= k)
    double tau = 0.0, sc = 0.0, beta = 0.0;
    // the norm below the diagonal by a team sum (the host sums serially: round-off order differs)
    double xsq = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, k + 1, m), [&](const int i, double &v) {
      const double vi = a[static_cast<std::size_t>(i)*n + k];
      v += vi*vi;
    }, xsq);
    Kokkos::single(Kokkos::PerTeam(tm), [&](double &tv) {
      const double xnorm = Kokkos::sqrt(xsq);
      tv = 0.0;
      if (xnorm > 0.0) {
        const double alpha = a[static_cast<std::size_t>(k)*n + k];
        const double bt = -Kokkos::copysign(Kokkos::hypot(alpha, xnorm), alpha);
        tv = (bt - alpha)/bt;
        a[static_cast<std::size_t>(k)*n + k] = 1.0/(alpha - bt);   // the scale, for a moment
        vn2[n + 0] = bt;                                           // scratch slot past the norms
      }
    }, tau);
    tm.team_barrier();
    if (tau != 0.0) {
      sc = a[static_cast<std::size_t>(k)*n + k];
      beta = vn2[n + 0];
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, k + 1, m), [&](const int i) {
        a[static_cast<std::size_t>(i)*n + k] *= sc;
      });
      tm.team_barrier();
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {a[static_cast<std::size_t>(k)*n + k] = beta;});
      tm.team_barrier();
    }
    if (!rows) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, k + 1, n), [&](const int j) {
        if (tau != 0.0) {
          double s = a[static_cast<std::size_t>(k)*n + j];
          for (int i=k+1; i<m; ++i) s += a[static_cast<std::size_t>(i)*n + k]*a[static_cast<std::size_t>(i)*n + j];
          s *= tau;
          a[static_cast<std::size_t>(k)*n + j] -= s;
          for (int i=k+1; i<m; ++i) a[static_cast<std::size_t>(i)*n + j] -= s*a[static_cast<std::size_t>(i)*n + k];
        }
        if (vn1[j] != 0.0) {
          double t = Kokkos::fabs(a[static_cast<std::size_t>(k)*n + j])/vn1[j];
          t = Kokkos::fmax(0.0, (1.0 + t)*(1.0 - t));
          const double r = vn1[j]/vn2[j];
          if (t*r*r <= tol3z) {
            double s = 0.0;
            for (int i=k+1; i<m; ++i) {
              const double cij = a[static_cast<std::size_t>(i)*n + j];
              s += cij*cij;
            }
            vn1[j] = vn2[j] = Kokkos::sqrt(s);
          } else {
            vn1[j] *= Kokkos::sqrt(t);
          }
        }
      });
    } else {
      // The update with the rows shared as well.  The dot products of the
      // Householder vector with the columns k+1..n-1 by (column, row chunk) over all threads, the
      // chunk sums added in their order; the rank-one update by matrix element (consecutive
      // threads on consecutive columns of a row).  NOTE: the sums run in another order than the
      // host's, so this QR (and the solve that follows) matches the host only to round-off, not
      // bitwise.  The chunk order is fixed, but the Householder norm above is a team reduction,
      // so this alone does not make the device result bitwise reproducible on every backend.
      const int nc = n - k - 1, nr = m - k - 1;
      if (tau != 0.0 && nc > 0) {
        const int nthr = tm.team_size();
        int nch = (nthr + nc - 1)/nc;
        if (nch > kQRow) nch = kQRow;
        if (nch > nr) nch = nr;
        if (nch < 1) nch = 1;
        const int lr = (nr + nch - 1)/nch;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nc*nch), [&](const int pq) {
          const int j = k + 1 + pq/nch, h = pq % nch;
          const int i0 = k + 1 + h*lr, i1 = (i0 + lr < m) ? i0 + lr : m;
          double s = 0.0;
          for (int i=i0; i<i1; ++i) s += a[static_cast<std::size_t>(i)*n + k]*a[static_cast<std::size_t>(i)*n + j];
          qp[pq] = s;
        });
        tm.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nc), [&](const int jc) {
          const int j = k + 1 + jc;
          double s = a[static_cast<std::size_t>(k)*n + j];
          for (int h=0; h<nch; ++h) s += qp[jc*nch + h];
          s *= tau;
          a[static_cast<std::size_t>(k)*n + j] -= s;
          qs[jc] = s;
        });
        tm.team_barrier();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nr*nc), [&](const int e) {
          const int i = k + 1 + e/nc, jc = e % nc;
          a[static_cast<std::size_t>(i)*n + k + 1 + jc] -= qs[jc]*a[static_cast<std::size_t>(i)*n + k];
        });
        tm.team_barrier();
      }
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, k + 1, n), [&](const int j) {
        if (vn1[j] != 0.0) {
          double t = Kokkos::fabs(a[static_cast<std::size_t>(k)*n + j])/vn1[j];
          t = Kokkos::fmax(0.0, (1.0 + t)*(1.0 - t));
          const double r = vn1[j]/vn2[j];
          if (t*r*r <= tol3z) {
            double s = 0.0;
            for (int i=k+1; i<m; ++i) {
              const double cij = a[static_cast<std::size_t>(i)*n + j];
              s += cij*cij;
            }
            vn1[j] = vn2[j] = Kokkos::sqrt(s);
          } else {
            vn1[j] *= Kokkos::sqrt(t);
          }
        }
      });
    }
    tm.team_barrier();
  }
  return kmax;
}

// The SM clock for the stage profile (0 off the device)
KOKKOS_INLINE_FUNCTION
long long Clk() {
#if defined(__CUDA_ARCH__)
  return clock64();
#else
  return 0;
#endif
}

// end-point reduction of the columns b (m x n, by rows, rows scaled) with masses 0 < w <= u
// until the moving clouds are independent (host ReduceCap); cnt[0..2] steps to zero, to the
// cap, scans.  Null vector c is the column c of z (n x ncol0, by rows, leading dimension ncol0).
KOKKOS_INLINE_FUNCTION
void ReduceCapTeam(const Member &tm, const double *b, const int m, const int n, double *w,
                   const double *u, const Ws &ws, const DvceArray1D<unsigned long long> &st,
                   const bool qrrows) {
  const bool prof = DUST_R_DIAG && st.extent(0) > 0 && tm.team_rank() == 0;   // stage profile
  int *fixed = ws.fixed;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int i) {fixed[i] = 0;});
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, m*n), [&](const int e) {ws.qr[e] = b[e];});
  tm.team_barrier();
  long long c0 = prof ? Clk() : 0;
  const int r = PivotedQRTeam(tm, ws.qr, m, n, ws.perm, ws.vn1, ws.vn2, ws.qp, ws.qs, qrrows);
  if (prof) {const long long c1 = Clk(); Kokkos::atomic_add(&st(1), (unsigned long long)(c1 - c0)); c0 = c1;}
  if (r >= n) return;
  const int ld = n - r;          // leading dimension of z and xs (the initial number of vectors)
  int ncol = ld;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n*ld), [&](const int e) {ws.z[e] = 0.0;});
  tm.team_barrier();
  const double *q = ws.qr;       // R11 = q[i*n + j], i, j < r; R12 e_c = q[i*n + r + c]
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ncol), [&](const int c) {
    double *x = ws.xs;           // x[i] of vector c at xs[i*ld + c]
    for (int i=r-1; i>=0; --i) {
      double s = q[static_cast<std::size_t>(i)*n + r + c];
      for (int j=i+1; j<r; ++j) s -= q[static_cast<std::size_t>(i)*n + j]*x[static_cast<std::size_t>(j)*ld + c];
      x[static_cast<std::size_t>(i)*ld + c] = s/q[static_cast<std::size_t>(i)*n + i];
    }
    for (int i=0; i<r; ++i) ws.z[static_cast<std::size_t>(ws.perm[i])*ld + c] = -x[static_cast<std::size_t>(i)*ld + c];
    ws.z[static_cast<std::size_t>(ws.perm[r + c])*ld + c] = 1.0;
  });
  tm.team_barrier();
  if (prof) {const long long c1 = Clk(); Kokkos::atomic_add(&st(2), (unsigned long long)(c1 - c0)); c0 = c1;}
  if (DUST_R_DIAG && st.extent(0) > 0) {
    // Diagnostic (with amr_profile and DUST_R_DIAG): the basic solution that keeps the r pivot
    // clouds and puts every other cloud of the block at zero, w_B = w_B - Z_B w_N (the null
    // vector c moves cloud perm[r + c] alone among the others): within the bounds?
    auto basic = [&](const int i) {
      const int pi = ws.perm[i];
      double v = w[pi];
      for (int c=0; c<ncol; ++c) v -= ws.z[static_cast<std::size_t>(pi)*ld + c]*w[ws.perm[r + c]];
      return v;
    };
    int nneg = 0, nover = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, r), [&](const int i, int &vn) {
      if (basic(i) < -1.0e-12*u[ws.perm[i]]) ++vn;
    }, nneg);
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, r), [&](const int i, int &vo) {
      if (basic(i) > u[ws.perm[i]]*(1.0 + kCapTol)) ++vo;
    }, nover);
    if (tm.team_rank() == 0) {
      Kokkos::atomic_add(&st(7), 1ULL);
      if (nneg == 0 && nover == 0) Kokkos::atomic_add(&st(8), 1ULL);
      if (nneg == 0 && nover > 0) Kokkos::atomic_add(&st(9), 1ULL);
      if (nneg > 0) Kokkos::atomic_add(&st(10), 1ULL);
    }
    if (prof) c0 = Clk();   // the check's time stays outside stages 1-3 (Codex review)
  }
  while (ncol > 0) {
    int nc = (kNCand < ncol) ? kNCand : ncol;
    // largest steps along -+ the null vectors c0..c1-1 and the largest |entry| of each vector
    // (for the score).  Each (vector, sign) is scanned in kChunk chunks by as many threads; the
    // chunks are merged in their order with the host's strict comparisons, so the first
    // minimum wins exactly as in the host's single loop.
    auto steps = [&](const int c0, const int c1) {
      const int lc = (n + kChunk - 1)/kChunk;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, 2*c0*kChunk, 2*c1*kChunk), [&](const int t) {
        const int e = t/kChunk, h = t%kChunk;
        const int c = e/2, sg = e%2;
        const double sign = sg ? 1.0 : -1.0;
        const int i0 = h*lc, i1 = (i0 + lc < n) ? i0 + lc : n;
        double a0 = kInf, a1 = kInf, zm = 0.0;
        int k0 = 0, k1 = 0;
        for (int i=i0; i<i1; ++i) {
          const double zi = ws.z[static_cast<std::size_t>(i)*ld + c];
          const double sgz = sign*zi;
          if (sgz < 0.0) {
            const double v = w[i]/Kokkos::fabs(zi);
            if (v < a0) {a0 = v; k0 = i;}
          } else if (sgz > 0.0) {
            const double v = (u[i] - w[i])/Kokkos::fabs(zi);
            if (v < a1) {a1 = v; k1 = i;}
          }
          zm = Kokkos::fmax(zm, Kokkos::fabs(zi));
        }
        ws.pa0[t] = a0; ws.pa1[t] = a1; ws.pk0[t] = k0; ws.pk1[t] = k1; ws.pzm[t] = zm;
      });
      tm.team_barrier();
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, 2*c0, 2*c1), [&](const int e) {
        double a0 = kInf, a1 = kInf, zm = 0.0;
        int k0 = 0, k1 = 0;
        for (int h=0; h<kChunk; ++h) {
          const int t = e*kChunk + h;
          if (ws.pa0[t] < a0) {a0 = ws.pa0[t]; k0 = ws.pk0[t];}
          if (ws.pa1[t] < a1) {a1 = ws.pa1[t]; k1 = ws.pk1[t];}
          zm = Kokkos::fmax(zm, ws.pzm[t]);
        }
        const bool zero = a0 <= a1;
        ws.tt[e] = zero ? a0 : a1;
        ws.kk[e] = zero ? k0 : k1;
        ws.zr[e] = zero ? 1 : 0;
        if (e%2 == 0) ws.zmx[e/2] = zm;
      });
      tm.team_barrier();
    };
    steps(0, nc);
    int anyzero = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, 2*nc), [&](const int e, int &v) {
      v += ws.zr[e];
    }, anyzero);
    if (!anyzero && ncol > nc) {
      steps(nc, ncol);
      nc = ncol;
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {++ws.cnt[2];});
    }
    // the best (sign, vector): the host's argmax over (sign, column), minus first; order index
    // o = sg nc + c, the first maximum wins
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, 2*nc), [&](const int o) {
      const int sg = o/nc, c = o%nc, e = 2*c + sg;
      ws.score[o] = !Kokkos::isfinite(ws.tt[e]) ? -1.0
                  : Kokkos::fabs(ws.z[static_cast<std::size_t>(ws.kk[e])*ld + c])/ws.zmx[c] + 2.0*ws.zr[e];
    });
    tm.team_barrier();
    double best = -1.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, 2*nc), [&](const int o, double &v) {
      if (ws.score[o] > v) v = ws.score[o];
    }, Kokkos::Max<double>(best));
    int ob = 2*nc;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, 2*nc), [&](const int o, int &v) {
      if (ws.score[o] == best && ws.score[o] > -1.0 && o < v) v = o;
    }, Kokkos::Min<int>(ob));
    const int code = (ob >= 2*nc || !(best > -1.0)) ? -1 : 2*(ob%nc) + ob/nc;
    if (code < 0) break;
    const int bj = code/2, bs = code%2, e = 2*bj + bs;
    const int k = ws.kk[e];
    const double step = ws.tt[e]*(bs ? 1.0 : -1.0);
    const int zero = ws.zr[e];
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int i) {
      const double di = ws.z[static_cast<std::size_t>(i)*ld + bj];
      ws.d[i] = di;
      w[i] = Kokkos::fmin(Kokkos::fmax(w[i] + step*di, 0.0), u[i]);
    });
    tm.team_barrier();
    Kokkos::single(Kokkos::PerTeam(tm), [&]() {
      if (zero) {
        w[k] = 0.0; ++ws.cnt[0];
      } else {
        w[k] = u[k]; fixed[k] = 1; ++ws.cnt[1];
      }
    });
    // drop column bj and remove cloud k from the remaining null vectors
    if (bj != ncol - 1) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int i) {
        ws.z[static_cast<std::size_t>(i)*ld + bj] = ws.z[static_cast<std::size_t>(i)*ld + ncol - 1];
      });
    }
    tm.team_barrier();
    --ncol;
    const double dk = ws.d[k];
    // the host's per-column update, entry by entry: factors first, then every (i, c) at once
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ncol), [&](const int c) {
      ws.fac[c] = ws.z[static_cast<std::size_t>(k)*ld + c]/dk;
    });
    tm.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n*ncol), [&](const int ec) {
      const int i = ec/ncol, c = ec%ncol;
      const double f = ws.fac[c];
      double &zic = ws.z[static_cast<std::size_t>(i)*ld + c];
      if (i == k) {
        zic = 0.0;
      } else if (f != 0.0) {
        zic -= f*ws.d[i];
      }
    });
    tm.team_barrier();
  }
  if (prof) Kokkos::atomic_add(&st(3), (unsigned long long)(Clk() - c0));
}

// one patch (host ResamplePatch): rows, bounded end-point reduction as a stream, row check.
// The row matrix A (nrow x n), the scaled B (nb x n) and the active block (nb x na) are kept
// by rows.
KOKKOS_INLINE_FUNCTION
void PatchTeam(const Member &tm, const DevicePatchProblem &pb, const int q, const Ws &ws,
               const int hmask, DvceArray1D<double> &mnew, DvceArray2D<int> &outi,
               DvceArray1D<double> &outr) {
  const int a0 = pb.pptr(q), a1 = pb.pptr(q + 1), n = a1 - a0;
  const auto &spd = pb.spd;
  const bool prof = DUST_R_DIAG && pb.stage.extent(0) > 0 && tm.team_rank() == 0;   // stage profile
  const long long cstart = prof ? Clk() : 0;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int j) {
    mnew(a0 + j) = spd(a0 + j, JM);
  });
  Kokkos::single(Kokkos::PerTeam(tm), [&]() {
    for (int c=0; c<16; ++c) ws.cnt[c] = 0;
    for (int c=0; c<NPC; ++c) outi(q, c) = 0;
    outi(q, PN) = n; outi(q, PKEPT) = n; outr(q) = 0.0;
  });
  tm.team_barrier();
  if (n < 2) return;
  // leaves under the clouds of the patch (sorted, distinct)
  const int e0 = pb.sptr(a0), ne = pb.sptr(a1) - e0;
  const int nl = DistinctSorted(tm, ne, [&](const int e) {return pb.stgt(e0 + e);}, ws.tab,
                                hmask, ws.ltmp, ws.leaves);
  const int nrow = 7*nl + 8;
  // mass-weighted mean velocity, host order
  Kokkos::single(Kokkos::PerTeam(tm), [&]() {
    double mt = 0.0, vb[3] = {0.0, 0.0, 0.0};
    for (int a=a0; a<a1; ++a) {
      mt += spd(a, JM);
      for (int k=0; k<3; ++k) vb[k] += spd(a, JM)*spd(a, JVX + k);
    }
    for (int k=0; k<3; ++k) ws.sc[k] = vb[k]/mt;
  });
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nrow*n), [&](const int e) {ws.A[e] = 0.0;});
  tm.team_barrier();
  // rows: leaf mass, leaf momentum about the patch mean, centre of mass, velocity second
  // moments, leaf feedback; one thread per cloud (its column A[i*n + j])
  const double xc = pb.pgeo(q,0), yc = pb.pgeo(q,1), rdx = pb.pgeo(q,2), rdy = pb.pgeo(q,3);
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int j) {
    const int a = a0 + j;
    auto col = [&](const int i) -> double & {return ws.A[static_cast<std::size_t>(i)*n + j];};
    const double c[3] = {spd(a, JVX) - ws.sc[0], spd(a, JVY) - ws.sc[1], spd(a, JVZ) - ws.sc[2]};
    const double f[3] = {(spd(a, JUX) - spd(a, JVX))/spd(a, JTS), (spd(a, JUY) - spd(a, JVY))/spd(a, JTS),
                         (spd(a, JUZ) - spd(a, JVZ))/spd(a, JTS)};
    for (int e=pb.sptr(a); e<pb.sptr(a + 1); ++e) {
      const int l = LowerBound(ws.leaves, nl, pb.stgt(e));
      const double wgt = pb.swgt(e);
      col(l) += wgt;
      for (int k=0; k<3; ++k) {col((1 + k)*nl + l) += wgt*c[k]; col(4*nl + 8 + k*nl + l) += wgt*f[k];}
    }
    double dx = spd(a, JX) - xc, dy = spd(a, JY) - yc;
    dx -= pb.lx*Kokkos::floor(dx/pb.lx + 0.5); dy -= pb.ly*Kokkos::floor(dy/pb.ly + 0.5);
    const int g = 4*nl;
    col(g) = dx/rdx; col(g + 1) = dy/rdy;
    col(g + 2) = c[0]*c[0]; col(g + 3) = c[0]*c[1]; col(g + 4) = c[0]*c[2]; col(g + 5) = c[1]*c[1];
    col(g + 6) = c[1]*c[2]; col(g + 7) = c[2]*c[2];
  });
  tm.team_barrier();
  // scaled rows without the empty ones
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nrow), [&](const int i) {
    double r = 0.0;
    const double *row = ws.A + static_cast<std::size_t>(i)*n;
    for (int j=0; j<n; ++j) r = Kokkos::fmax(r, Kokkos::fabs(row[j]));
    ws.r0[i] = r;
  });
  tm.team_barrier();
  int nb = 0;
  Kokkos::single(Kokkos::PerTeam(tm), [&](int &v) {
    int c = 0;
    for (int i=0; i<nrow; ++i) if (ws.r0[i] > 0.0) ws.rowidx[c++] = i;
    v = c;
  }, nb);
  tm.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int j) {
    for (int s=0; s<nb; ++s) {
      const int i = ws.rowidx[s];
      ws.B[static_cast<std::size_t>(s)*n + j] = ws.A[static_cast<std::size_t>(i)*n + j]/ws.r0[i];
    }
    ws.w[j] = spd(a0 + j, JM);
  });
  tm.team_barrier();
  const int rb = (nb < n) ? nb : n;
  const int cap = (2*rb > rb + 32) ? 2*rb : rb + 32;
  const long long cstream = prof ? Clk() : 0;
  if (prof) {Kokkos::atomic_add(&pb.stage(0), (unsigned long long)(cstream - cstart)); Kokkos::atomic_add(&pb.stage(6), 1ULL);}
  int nact = 0, nxt = 0;
  while (true) {
    const int take = ((n - nxt) < (cap - nact)) ? (n - nxt) : (cap - nact);
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, take), [&](const int i) {
      ws.act[nact + i] = nxt + i;
    });
    tm.team_barrier();
    nxt += take;
    nact += take;
    const int na = nact;
    if (na < 2) break;
    // the active block (nb x na) by rows, one thread per active cloud
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, na), [&](const int i) {
      const int j = ws.act[i];
      for (int s=0; s<nb; ++s) {
        ws.bact[static_cast<std::size_t>(s)*na + i] = ws.B[static_cast<std::size_t>(s)*n + j];
      }
      ws.wa[i] = ws.w[j];
      ws.ua[i] = spd(a0 + j, JCAP);
    });
    tm.team_barrier();
    ReduceCapTeam(tm, ws.bact, nb, na, ws.wa, ws.ua, ws, pb.stage, pb.qr_rows);
    tm.team_barrier();
    int nleft = 0;
    Kokkos::single(Kokkos::PerTeam(tm), [&](int &v) {
      int c = 0;
      for (int i=0; i<na; ++i) {
        ws.w[ws.act[i]] = ws.wa[i];
        if (ws.wa[i] > 0.0 && !ws.fixed[i]) ws.left[c++] = ws.act[i];
      }
      v = c;
    }, nleft);
    tm.team_barrier();
    if (nxt >= n) break;          // the moving clouds left by the last call are independent
    if (nleft == nact) break;     // no room was made: the rest keeps its masses
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nleft), [&](const int i) {
      ws.act[i] = ws.left[i];
    });
    tm.team_barrier();
    nact = nleft;
  }
  const long long ccheck = prof ? Clk() : 0;
  if (prof) Kokkos::atomic_add(&pb.stage(4), (unsigned long long)(ccheck - cstream));   // incl. 1-3
  // row check on the unscaled rows: |A w - A m| / (|A| m), host order per row
  int bad = 0;   // a sum: a Max reducer starts from the lowest int, not from 0
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, n), [&](const int j, int &v) {
    const double u = spd(a0 + j, JCAP);
    if (!(ws.w[j] >= 0.0) || ws.w[j] > u*(1.0 + kCapTol)) ++v;
  }, bad);
  double res = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, nrow), [&](const int i, double &v) {
    double den = 0.0, dif = 0.0;
    const double *row = ws.A + static_cast<std::size_t>(i)*n;
    for (int j=0; j<n; ++j) {
      const double m = spd(a0 + j, JM);
      den += Kokkos::fabs(row[j])*m;
      dif += row[j]*(ws.w[j] - m);
    }
    const double r = Kokkos::fabs(dif)/Kokkos::fmax(den, 1.0e-300);
    if (r > v) v = r;
  }, Kokkos::Max<double>(res));
  const bool fallback = bad || !(res <= kResMax);
  int kept = n, ncap = 0;
  if (!fallback) {
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, n), [&](const int j, int &kv) {
      mnew(a0 + j) = ws.w[j];
      if (ws.w[j] > 0.0) ++kv;
    }, kept);
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, n), [&](const int j, int &cv) {
      if (ws.w[j] > 0.0 && ws.w[j] >= spd(a0 + j, JCAP)*(1.0 - kCapTol)) ++cv;
    }, ncap);
  }
  tm.team_barrier();
  Kokkos::single(Kokkos::PerTeam(tm), [&]() {
    outi(q, PN) = n; outi(q, PKEPT) = kept; outi(q, PNCAP) = ncap; outi(q, PFALLBACK) = fallback ? 2 : 0;
    outi(q, PZERO) = ws.cnt[0]; outi(q, PCAPSTEP) = ws.cnt[1]; outi(q, PSCANS) = ws.cnt[2];
    outi(q, PNROW) = nrow; outi(q, PNB) = nb;
    outr(q) = res;
  });
  tm.team_barrier();
  if (prof) Kokkos::atomic_add(&pb.stage(5), (unsigned long long)(Clk() - ccheck));
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn SolvePatchesDevice
//! \brief See dust_resample.hpp.  timing: [0] sizes, [1] small bucket, [2] large bucket,
//! [3] total, [4] middle bucket, seconds (fenced).

void SolvePatchesDevice(const DevicePatchProblem &pb, DvceArray1D<double> mnew,
                        DvceArray2D<int> outi, DvceArray1D<double> outr, const int mem_mb,
                        const int mid, const int mid_team, const int wide_team,
                        double timing[5]) {
  Kokkos::Timer t_all, t;
  const int npatch = pb.npatch;
  // per patch: clouds, CSR entries (device), distinct leaves (team sort of a copy)
  DvceArray2D<int> sz("rs_dev_sizes", npatch, 3);
  {
    auto pptr = pb.pptr; auto sptr = pb.sptr;
    Kokkos::parallel_for("rs_dev_n", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch),
    KOKKOS_LAMBDA(const int q) {
      sz(q, 0) = pptr(q + 1) - pptr(q);
      sz(q, 1) = sptr(pptr(q + 1)) - sptr(pptr(q));
      sz(q, 2) = 0;
    });
  }
  // Sizes, buckets and patch lists on the device; only a few totals go to the host
  int emax = 1;
  Kokkos::parallel_reduce("rs_dev_emax", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch),
  KOKKOS_LAMBDA(const int q, int &m) {if (sz(q, 0) >= 2 && sz(q, 1) > m) m = sz(q, 1);},
  Kokkos::Max<int>(emax));
  emax = std::max(emax, 1);
  {
    using ScratchInt = Kokkos::View<int*, DevExeSpace::scratch_memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    int hpre = 4;
    while (hpre < 2*emax) hpre *= 2;
    const std::size_t bytes = ScratchInt::shmem_size(hpre);
    auto pptr = pb.pptr; auto sptr = pb.sptr; auto stgt = pb.stgt;
    Kokkos::parallel_for("rs_dev_leaves",
    Policy(npatch, Kokkos::AUTO).set_scratch_size(1, Kokkos::PerTeam(bytes)),
    KOKKOS_LAMBDA(const Member &tm) {
      const int q = tm.league_rank();
      if (sz(q, 0) < 2) return;
      const int e0 = sptr(pptr(q)), ne = sz(q, 1);
      ScratchInt tab(tm.team_scratch(1), hpre);
      const int nl = DistinctSorted(tm, ne, [&](const int e) {return stgt(e0 + e);}, tab.data(),
                                    hpre - 1, nullptr, nullptr);
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {sz(q, 2) = nl;});
    });
  }
  // buckets: 0 at most 64 clouds, 1 (when mid > 64) at most mid clouds, 2 more;
  // patches of one or no cloud only get their counters.  Per bucket the number of patches and
  // the largest clouds / entries / leaves.  mid <= 64: no bucket 1 (the original two buckets)
  constexpr int NB = 3;
  int nqb[NB] = {0, 0, 0}, nmx[NB] = {0, 0, 0}, emx[NB] = {1, 1, 1}, lmx[NB] = {1, 1, 1};
  {
    int r[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    Kokkos::parallel_reduce("rs_dev_buckets", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch),
    KOKKOS_LAMBDA(const int q, int &c0, int &c1, int &c2, int &n0, int &n1, int &n2, int &e0,
                  int &e1, int &e2, int &l0, int &l1, int &l2) {
      const int n = sz(q, 0);
      if (n < 2) return;
      const int b = (n <= 64) ? 0 : ((n <= mid) ? 1 : 2);
      int &c = (b == 0) ? c0 : ((b == 1) ? c1 : c2);
      int &nn = (b == 0) ? n0 : ((b == 1) ? n1 : n2);
      int &ee = (b == 0) ? e0 : ((b == 1) ? e1 : e2);
      int &ll = (b == 0) ? l0 : ((b == 1) ? l1 : l2);
      ++c; nn = Kokkos::max(nn, n); ee = Kokkos::max(ee, sz(q, 1)); ll = Kokkos::max(ll, sz(q, 2));
    }, Kokkos::Sum<int>(r[0]), Kokkos::Sum<int>(r[1]), Kokkos::Sum<int>(r[2]),
       Kokkos::Max<int>(r[3]), Kokkos::Max<int>(r[4]), Kokkos::Max<int>(r[5]),
       Kokkos::Max<int>(r[6]), Kokkos::Max<int>(r[7]), Kokkos::Max<int>(r[8]),
       Kokkos::Max<int>(r[9]), Kokkos::Max<int>(r[10]), Kokkos::Max<int>(r[11]));
    for (int b=0; b<NB; ++b) {
      nqb[b] = r[b];
      nmx[b] = std::max(r[3 + b], 0); emx[b] = std::max(r[6 + b], 1); lmx[b] = std::max(r[9 + b], 1);
    }
  }
  // the patch lists: per bucket the largest patches first and by patch number among equal sizes
  // (balance; the order does not change any result).  One sort of all patches by the key
  // ((b 2^24 + 2^24 - 1 - n) npatch + q; patches of fewer than two clouds last): bucket b is the
  // range off(b) .. off(b) + nqb(b) - 1
  DvceArray1D<std::int64_t> qkey(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_dev_qkey"),
                                 std::max(npatch, 1));
  {
    const std::int64_t np = npatch, top = std::int64_t(1) << 24;
    Kokkos::parallel_for("rs_dev_lists", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch),
    KOKKOS_LAMBDA(const int q) {
      const int n = sz(q, 0);
      const int b = (n <= 64) ? 0 : ((n <= mid) ? 1 : 2);
      qkey(q) = (n < 2) ? std::numeric_limits<std::int64_t>::max()
                        : (b*top + top - 1 - n)*np + q;
    });
    if (npatch > 1) Kokkos::sort(qkey);
  }
  timing[0] = t.seconds();
  {
    auto pptr = pb.pptr; auto spd = pb.spd;
    Kokkos::parallel_for("rs_dev_small", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch),
    KOKKOS_LAMBDA(const int q) {
      const int n = pptr(q + 1) - pptr(q);
      if (n >= 2) return;
      for (int c=0; c<NPC; ++c) outi(q, c) = 0;
      outi(q, PN) = n; outi(q, PKEPT) = n; outr(q) = 0.0;
      for (int a=pptr(q); a<pptr(q + 1); ++a) mnew(a) = spd(a, JM);
    });
  }
  const double budget = static_cast<double>(std::max(mem_mb, 64))*1048576.0;
  for (int b=0; b<NB; ++b) {
    Kokkos::fence();
    t.reset();
    const int nq = nqb[b];
    if (nq > 0) {
      const WsSize ws = Sizes(nmx[b], emx[b], lmx[b]);
      const double per = 8.0*ws.nd + 4.0*ws.ni;
      int nslot = static_cast<int>(std::min<double>(nq, std::max(1.0, budget/per)));
      nslot = std::min(nslot, 4096);
      DvceArray1D<int> qlist(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_dev_list"), nq);
      {
        const int off = (b == 0) ? 0 : ((b == 1) ? nqb[0] : nqb[0] + nqb[1]);
        const std::int64_t np = npatch;
        Kokkos::parallel_for("rs_dev_qlist", Kokkos::RangePolicy<>(DevExeSpace(), 0, nq),
        KOKKOS_LAMBDA(const int i) {qlist(i) = static_cast<int>(qkey(off + i) % np);});
      }
      DvceArray1D<double> dws(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_dev_ws_d"),
                              static_cast<std::size_t>(nslot)*ws.nd);
      DvceArray1D<int> iws(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_dev_ws_i"),
                           static_cast<std::size_t>(nslot)*ws.ni);
      DvceArray1D<int> counter("rs_dev_counter", 1);
      const DevicePatchProblem p = pb;
      auto body = KOKKOS_LAMBDA(const Member &tm) {
        const Ws w = Carve(dws.data() + static_cast<std::size_t>(tm.league_rank())*ws.nd,
                           iws.data() + static_cast<std::size_t>(tm.league_rank())*ws.ni, ws);
        DvceArray1D<double> mn = mnew;
        DvceArray2D<int> oi = outi;
        DvceArray1D<double> orr = outr;
        while (true) {
          int i = 0;
          Kokkos::single(Kokkos::PerTeam(tm), [&](int &v) {
            v = Kokkos::atomic_fetch_add(&counter(0), 1);
          }, i);
          if (i >= nq) break;
          PatchTeam(tm, p, qlist(i), w, ws.hmax - 1, mn, oi, orr);
        }
      };
      // the large patches set the critical path of a resampling: the widest team the kernel
      // allows (at most 512 threads); the small ones keep the automatic size.  However,
      // a 512-thread team of the wide kernel (at most 128 registers per thread) fills the
      // register file of an SM, so the middle bucket runs the same kernel with mid_team threads
      // (several patches per SM)
      if (b == 0) {
        Kokkos::parallel_for("rs_dev_patches_small", Policy(nslot, Kokkos::AUTO), body);
      } else {
        const int tmax = PolicyWide(nslot, 1).team_size_max(body, Kokkos::ParallelForTag());
        int ts = 1;   // the largest power of two the backend allows, at most 512 (or mid_team)
        const int tcap = (b == 1) ? std::max(1, mid_team) : 512;
        while (2*ts <= tmax && 2*ts <= tcap) ts *= 2;
        static bool told[NB] = {false, false, false};
        if (!told[b] && global_variable::my_rank == 0 && !(b == 2 && wide_team > 512)) {
          std::cout << "# device resampling: " << ((b == 1) ? "middle" : "large")
                    << " patches with teams of " << ts << " threads (largest allowed " << tmax << ")"
                    << std::endl;
          told[b] = true;
        }
        if (b == 2 && wide_team > 512) {
          // The large bucket with teams of up to 1024 threads (at most 64
          // registers per thread)
          const int tmax2 = PolicyWider(nslot, 1).team_size_max(body, Kokkos::ParallelForTag());
          int ts2 = 1;
          while (2*ts2 <= tmax2 && 2*ts2 <= wide_team) ts2 *= 2;
          static bool told2 = false;
          if (!told2 && global_variable::my_rank == 0) {
            std::cout << "# device resampling: large patches with teams of " << ts2
                      << " threads (largest allowed " << tmax2 << ", 1024-thread bound)" << std::endl;
            told2 = true;
          }
          Kokkos::parallel_for("rs_dev_patches_large1024", PolicyWider(nslot, ts2), body);
        } else {
          Kokkos::parallel_for((b == 1) ? "rs_dev_patches_mid" : "rs_dev_patches_large",
                               PolicyWide(nslot, ts), body);
        }
      }
    }
    Kokkos::fence();
    timing[(b == 0) ? 1 : ((b == 1) ? 4 : 2)] = t.seconds();
  }
  timing[3] = t_all.seconds();
}

//----------------------------------------------------------------------------------------
//! \fn SummarizePatchesDevice
//! \brief See dust_resample.hpp (the host's loop over the patch infos, on the device).

void SummarizePatchesDevice(DvceArray2D<int> outi, DvceArray1D<double> outr, PatchSummary &s) {
  const int npatch = outi.extent_int(0);
  long long c[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int nmax = 0, nbmax = 0, n64 = 0, n256 = 0;
  double resmax = 0.0;
  Kokkos::parallel_reduce("rs_dev_summary", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch),
  KOKKOS_LAMBDA(const int q, long long &npat, long long &pool, long long &kept, long long &atcap,
                long long &nfall, long long &nred, long long &zs, long long &cs, long long &sc,
                double &rmax, int &nm, int &nbm, int &c64, int &c256) {
    const int n = outi(q, PN);
    if (n == 0) return;
    nm = Kokkos::max(nm, n); nbm = Kokkos::max(nbm, outi(q, PNB));
    if (n >= 64) ++c64;
    if (n >= 256) ++c256;
    ++npat; pool += n; kept += outi(q, PKEPT); atcap += outi(q, PNCAP);
    zs += outi(q, PZERO); cs += outi(q, PCAPSTEP); sc += outi(q, PSCANS);
    if (outi(q, PFALLBACK) != 0) ++nfall; else rmax = Kokkos::fmax(rmax, outr(q));
    if (outi(q, PKEPT) < n) ++nred;
  }, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], Kokkos::Max<double>(resmax),
     Kokkos::Max<int>(nmax), Kokkos::Max<int>(nbmax), n64, n256);
  s.npat = c[0]; s.pool = c[1]; s.kept = c[2]; s.atcap = c[3]; s.nfall = c[4]; s.nred = c[5];
  s.zsteps = c[6]; s.csteps = c[7]; s.scans = c[8];
  s.resmax = std::max(resmax, 0.0); s.nmax = std::max(nmax, 0); s.nbmax = std::max(nbmax, 0);
  s.n64 = n64; s.n256 = n256;
}

}  // namespace resample
}  // namespace dust
