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
//! The device result therefore follows the host's arithmetic order; it can still differ by
//! round-off (fused multiply-add, sqrt / hypot of the device library), which the comparison
//! mode (<dust>/cloud_resample_compare) measures.
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

#include "athena.hpp"
#include "dust_resample.hpp"

namespace dust {
namespace resample {
namespace {

using Policy = Kokkos::TeamPolicy<DevExeSpace>;
using Member = Policy::member_type;

// sizes of the per-team workspace of a bucket
struct WsSize {
  int nmax = 0, emax = 1, rmax = 8, cmax = 32;   // clouds, CSR entries, rows, active columns
  std::size_t nd = 0, ni = 0;                     // doubles and ints per team slot
};

WsSize Sizes(const int nmax, const int emax, const int lmax) {
  WsSize z;
  z.nmax = std::max(nmax, 2);
  z.emax = std::max(emax, 1);
  z.rmax = 7*lmax + 8;
  const int rb = std::min(z.rmax, z.nmax);
  z.cmax = std::max(2*rb, rb + 32);
  const std::size_t r = z.rmax, n = z.nmax, c = z.cmax;
  z.nd = 2*r*n + r + n + 2*r*c + 2*c*c + 10*c + 16;
  z.ni = 2*static_cast<std::size_t>(z.emax) + r + 2*n + 6*c + 16;
  return z;
}

struct Ws {
  double *A, *B, *r0, *w, *bact, *qr, *z, *xs, *vn1, *vn2, *d, *wa, *ua, *tt, *zmx, *sc;
  int *ents, *leaves, *rowidx, *act, *left, *perm, *kk, *fixed, *zr, *cnt;
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
  ws.vn2 = db; db += c;
  ws.d = db; db += c;
  ws.wa = db; db += c;
  ws.ua = db; db += c;
  ws.tt = db; db += 2*c;
  ws.zmx = db; db += c;
  ws.sc = db;                           // 16 scalars
  ws.ents = ib; ib += z.emax;
  ws.leaves = ib; ib += z.emax;
  ws.rowidx = ib; ib += r;
  ws.act = ib; ib += n;
  ws.left = ib; ib += n;
  ws.perm = ib; ib += c;
  ws.kk = ib; ib += 2*c;
  ws.fixed = ib; ib += c;
  ws.zr = ib; ib += 2*c;
  ws.cnt = ib;                          // 16 counters
  return ws;
}

// in-place heapsort of v[0..n-1] by one thread (deterministic, no extra memory)
template <typename V>
KOKKOS_INLINE_FUNCTION
void HeapSort(V &v, const int n) {
  auto sift = [&](int i, const int end) {
    while (true) {
      int c = 2*i + 1;
      if (c >= end) break;
      if (c + 1 < end && v[c] < v[c + 1]) ++c;
      if (!(v[i] < v[c])) break;
      const int t = v[i]; v[i] = v[c]; v[c] = t;
      i = c;
    }
  };
  for (int i=n/2 - 1; i>=0; --i) sift(i, n);
  for (int end=n - 1; end>0; --end) {
    const int t = v[0]; v[0] = v[end]; v[end] = t;
    sift(0, end);
  }
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

// Householder QR with column pivoting of the m x n matrix a (column major, overwritten):
// a P = Q R; the numerical rank (host PivotedQR, one thread per column)
KOKKOS_INLINE_FUNCTION
int PivotedQRTeam(const Member &tm, double *a, const int m, const int n, int *perm, double *vn1,
                  double *vn2) {
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int j) {
    perm[j] = j;
    double s = 0.0;
    const double *c = a + static_cast<std::size_t>(j)*m;
    for (int i=0; i<m; ++i) s += c[i]*c[i];
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
    if (p >= n) p = k;
    if (k == 0) first = vn1[p];
    if (!(vn1[p] > kRankTol*first) || first == 0.0) {
      tm.team_barrier();
      return k;
    }
    if (p != k) {
      double *cp = a + static_cast<std::size_t>(p)*m, *ck = a + static_cast<std::size_t>(k)*m;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, m), [&](const int i) {
        const double t = cp[i]; cp[i] = ck[i]; ck[i] = t;
      });
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {
        int ti = perm[p]; perm[p] = perm[k]; perm[k] = ti;
        double t = vn1[p]; vn1[p] = vn1[k]; vn1[k] = t;
        t = vn2[p]; vn2[p] = vn2[k]; vn2[k] = t;
      });
      tm.team_barrier();
    }
    double *v = a + static_cast<std::size_t>(k)*m;
    double tau = 0.0;
    Kokkos::single(Kokkos::PerTeam(tm), [&](double &tv) {
      double xnorm = 0.0;
      for (int i=k+1; i<m; ++i) xnorm += v[i]*v[i];
      xnorm = Kokkos::sqrt(xnorm);
      tv = 0.0;
      if (xnorm > 0.0) {
        const double alpha = v[k];
        const double beta = -Kokkos::copysign(Kokkos::hypot(alpha, xnorm), alpha);
        tv = (beta - alpha)/beta;
        const double sc = 1.0/(alpha - beta);
        for (int i=k+1; i<m; ++i) v[i] *= sc;
        v[k] = beta;
      }
    }, tau);
    tm.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, k + 1, n), [&](const int j) {
      double *c = a + static_cast<std::size_t>(j)*m;
      if (tau != 0.0) {
        double s = c[k];
        for (int i=k+1; i<m; ++i) s += v[i]*c[i];
        s *= tau;
        c[k] -= s;
        for (int i=k+1; i<m; ++i) c[i] -= s*v[i];
      }
      if (vn1[j] != 0.0) {
        double t = Kokkos::fabs(c[k])/vn1[j];
        t = Kokkos::fmax(0.0, (1.0 + t)*(1.0 - t));
        const double r = vn1[j]/vn2[j];
        if (t*r*r <= tol3z) {
          double s = 0.0;
          for (int i=k+1; i<m; ++i) s += c[i]*c[i];
          vn1[j] = vn2[j] = Kokkos::sqrt(s);
        } else {
          vn1[j] *= Kokkos::sqrt(t);
        }
      }
    });
    tm.team_barrier();
  }
  return kmax;
}

// end-point reduction of the columns b (m x n, rows scaled) with masses 0 < w <= u until the
// moving clouds are independent (host ReduceCap); cnt[0..2] steps to zero, to the cap, scans
KOKKOS_INLINE_FUNCTION
void ReduceCapTeam(const Member &tm, const double *b, const int m, const int n, double *w,
                   const double *u, const Ws &ws) {
  int *fixed = ws.fixed;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int i) {fixed[i] = 0;});
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, m*n), [&](const int e) {ws.qr[e] = b[e];});
  tm.team_barrier();
  const int r = PivotedQRTeam(tm, ws.qr, m, n, ws.perm, ws.vn1, ws.vn2);
  if (r >= n) return;
  int ncol = n - r;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n*ncol), [&](const int e) {ws.z[e] = 0.0;});
  tm.team_barrier();
  const double *q = ws.qr;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ncol), [&](const int c) {
    const double *rhs = q + static_cast<std::size_t>(r + c)*m;
    double *x = ws.xs + static_cast<std::size_t>(c)*r;
    for (int i=r-1; i>=0; --i) {
      double s = rhs[i];
      for (int j=i+1; j<r; ++j) s -= q[static_cast<std::size_t>(j)*m + i]*x[j];
      x[i] = s/q[static_cast<std::size_t>(i)*m + i];
    }
    double *zc = ws.z + static_cast<std::size_t>(c)*n;
    for (int i=0; i<r; ++i) zc[ws.perm[i]] = -x[i];
    zc[ws.perm[r + c]] = 1.0;
  });
  tm.team_barrier();
  while (ncol > 0) {
    int nc = (kNCand < ncol) ? kNCand : ncol;
    // largest steps along -+ the null vectors c0..c1-1, one thread per (vector, sign) in the
    // host order; the largest |entry| of each vector for the score
    auto steps = [&](const int c0, const int c1) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, 2*c0, 2*c1), [&](const int e) {
        const int c = e/2, sg = e%2;
        const double *zc = ws.z + static_cast<std::size_t>(c)*n;
        const double sign = sg ? 1.0 : -1.0;
        double a0 = kInf, a1 = kInf;
        int k0 = 0, k1 = 0;
        for (int i=0; i<n; ++i) {
          const double s = sign*zc[i];
          if (s < 0.0) {
            const double v = w[i]/Kokkos::fabs(zc[i]);
            if (v < a0) {a0 = v; k0 = i;}
          } else if (s > 0.0) {
            const double v = (u[i] - w[i])/Kokkos::fabs(zc[i]);
            if (v < a1) {a1 = v; k1 = i;}
          }
        }
        const bool zero = a0 <= a1;
        ws.tt[e] = zero ? a0 : a1;
        ws.kk[e] = zero ? k0 : k1;
        ws.zr[e] = zero ? 1 : 0;
        if (sg == 0) {
          double zm = 0.0;
          for (int i=0; i<n; ++i) zm = Kokkos::fmax(zm, Kokkos::fabs(zc[i]));
          ws.zmx[c] = zm;
        }
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
    // the best (sign, vector): minus first, as the host's argmax over (sign, column)
    int code = -1;
    Kokkos::single(Kokkos::PerTeam(tm), [&](int &cv) {
      double best = -1.0;
      int bj = -1, bs = 0;
      for (int sg=0; sg<2; ++sg) {
        for (int c=0; c<nc; ++c) {
          const int e = 2*c + sg;
          if (!Kokkos::isfinite(ws.tt[e])) continue;
          const double score = Kokkos::fabs(ws.z[static_cast<std::size_t>(c)*n + ws.kk[e]])/ws.zmx[c]
                             + 2.0*ws.zr[e];
          if (score > best) {best = score; bj = c; bs = sg;}
        }
      }
      cv = (bj < 0) ? -1 : 2*bj + bs;
    }, code);
    tm.team_barrier();
    if (code < 0) break;
    const int bj = code/2, bs = code%2, e = 2*bj + bs;
    const int k = ws.kk[e];
    const double step = ws.tt[e]*(bs ? 1.0 : -1.0);
    const int zero = ws.zr[e];
    const double *zj = ws.z + static_cast<std::size_t>(bj)*n;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int i) {ws.d[i] = zj[i];});
    tm.team_barrier();
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int i) {
      w[i] = Kokkos::fmin(Kokkos::fmax(w[i] + step*ws.d[i], 0.0), u[i]);
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
      double *zl = ws.z + static_cast<std::size_t>(ncol - 1)*n;
      double *zd = ws.z + static_cast<std::size_t>(bj)*n;
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int i) {zd[i] = zl[i];});
    }
    tm.team_barrier();
    --ncol;
    const double dk = ws.d[k];
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ncol), [&](const int c) {
      double *zc = ws.z + static_cast<std::size_t>(c)*n;
      const double f = zc[k]/dk;
      if (f != 0.0) for (int i=0; i<n; ++i) zc[i] -= f*ws.d[i];
      zc[k] = 0.0;
    });
    tm.team_barrier();
  }
}

// one patch (host ResamplePatch): rows, bounded end-point reduction as a stream, row check
KOKKOS_INLINE_FUNCTION
void PatchTeam(const Member &tm, const DevicePatchProblem &pb, const int q, const Ws &ws,
               DvceArray1D<double> &mnew, DvceArray2D<int> &outi, DvceArray1D<double> &outr) {
  const int a0 = pb.pptr(q), a1 = pb.pptr(q + 1), n = a1 - a0;
  const auto &spd = pb.spd;
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
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ne), [&](const int e) {
    ws.ents[e] = pb.stgt(e0 + e);
  });
  tm.team_barrier();
  int nl = 0;
  Kokkos::single(Kokkos::PerTeam(tm), [&](int &v) {
    int *ents = ws.ents;
    HeapSort(ents, ne);
    int c = 0;
    for (int e=0; e<ne; ++e) {
      if (e == 0 || ws.ents[e] != ws.ents[e - 1]) ws.leaves[c++] = ws.ents[e];
    }
    v = c;
  }, nl);
  tm.team_barrier();
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
  // moments, leaf feedback; one thread per cloud (its column)
  const double xc = pb.x1min + (q % pb.np1 + 0.5)*pb.s*pb.rdx;
  const double yc = pb.x2min + (q/pb.np1 + 0.5)*pb.s*pb.rdy;
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, n), [&](const int j) {
    const int a = a0 + j;
    double *col = ws.A + static_cast<std::size_t>(j)*nrow;
    const double c[3] = {spd(a, JVX) - ws.sc[0], spd(a, JVY) - ws.sc[1], spd(a, JVZ) - ws.sc[2]};
    const double f[3] = {(spd(a, JUX) - spd(a, JVX))/spd(a, JTS), (spd(a, JUY) - spd(a, JVY))/spd(a, JTS),
                         (spd(a, JUZ) - spd(a, JVZ))/spd(a, JTS)};
    for (int e=pb.sptr(a); e<pb.sptr(a + 1); ++e) {
      const int l = LowerBound(ws.leaves, nl, pb.stgt(e));
      const double wgt = pb.swgt(e);
      col[l] += wgt;
      for (int k=0; k<3; ++k) {col[(1 + k)*nl + l] += wgt*c[k]; col[4*nl + 8 + k*nl + l] += wgt*f[k];}
    }
    double dx = spd(a, JX) - xc, dy = spd(a, JY) - yc;
    dx -= pb.lx*Kokkos::floor(dx/pb.lx + 0.5); dy -= pb.ly*Kokkos::floor(dy/pb.ly + 0.5);
    double *g = col + 4*nl;
    g[0] = dx/pb.rdx; g[1] = dy/pb.rdy;
    g[2] = c[0]*c[0]; g[3] = c[0]*c[1]; g[4] = c[0]*c[2]; g[5] = c[1]*c[1]; g[6] = c[1]*c[2];
    g[7] = c[2]*c[2];
  });
  tm.team_barrier();
  // scaled rows without the empty ones
  Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, nrow), [&](const int i) {
    double r = 0.0;
    for (int j=0; j<n; ++j) r = Kokkos::fmax(r, Kokkos::fabs(ws.A[static_cast<std::size_t>(j)*nrow + i]));
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
    const double *col = ws.A + static_cast<std::size_t>(j)*nrow;
    double *bc = ws.B + static_cast<std::size_t>(j)*nb;
    for (int s=0; s<nb; ++s) bc[s] = col[ws.rowidx[s]]/ws.r0[ws.rowidx[s]];
    ws.w[j] = spd(a0 + j, JM);
  });
  tm.team_barrier();
  const int rb = (nb < n) ? nb : n;
  const int cap = (2*rb > rb + 32) ? 2*rb : rb + 32;
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
    Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, na), [&](const int i) {
      const int j = ws.act[i];
      const double *src = ws.B + static_cast<std::size_t>(j)*nb;
      double *dst = ws.bact + static_cast<std::size_t>(i)*nb;
      for (int s=0; s<nb; ++s) dst[s] = src[s];
      ws.wa[i] = ws.w[j];
      ws.ua[i] = spd(a0 + j, JCAP);
    });
    tm.team_barrier();
    ReduceCapTeam(tm, ws.bact, nb, na, ws.wa, ws.ua, ws);
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
  // row check on the unscaled rows: |A w - A m| / (|A| m), host order per row
  int bad = 0;   // a sum: a Max reducer starts from the lowest int, not from 0
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, n), [&](const int j, int &v) {
    const double u = spd(a0 + j, JCAP);
    if (!(ws.w[j] >= 0.0) || ws.w[j] > u*(1.0 + kCapTol)) ++v;
  }, bad);
  double res = 0.0;
  Kokkos::parallel_reduce(Kokkos::TeamThreadRange(tm, nrow), [&](const int i, double &v) {
    double den = 0.0, dif = 0.0;
    for (int j=0; j<n; ++j) {
      const double col = ws.A[static_cast<std::size_t>(j)*nrow + i];
      const double m = spd(a0 + j, JM);
      den += Kokkos::fabs(col)*m;
      dif += col*(ws.w[j] - m);
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
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn SolvePatchesDevice
//! \brief See dust_resample.hpp.  timing: [0] sizes, [1] small bucket, [2] large bucket,
//! [3] total, seconds (fenced).

void SolvePatchesDevice(const DevicePatchProblem &pb, DvceArray1D<double> mnew,
                        DvceArray2D<int> outi, DvceArray1D<double> outr, const int mem_mb,
                        double timing[4]) {
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
  auto sz_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), sz);
  int emax = 1;
  for (int q=0; q<npatch; ++q) if (sz_h(q, 0) >= 2) emax = std::max(emax, sz_h(q, 1));
  {
    using ScratchInt = Kokkos::View<int*, DevExeSpace::scratch_memory_space,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    const std::size_t bytes = ScratchInt::shmem_size(emax);
    auto pptr = pb.pptr; auto sptr = pb.sptr; auto stgt = pb.stgt;
    Kokkos::parallel_for("rs_dev_leaves",
    Policy(npatch, Kokkos::AUTO).set_scratch_size(1, Kokkos::PerTeam(bytes)),
    KOKKOS_LAMBDA(const Member &tm) {
      const int q = tm.league_rank();
      if (sz(q, 0) < 2) return;
      const int e0 = sptr(pptr(q)), ne = sz(q, 1);
      ScratchInt v(tm.team_scratch(1), emax);
      Kokkos::parallel_for(Kokkos::TeamThreadRange(tm, ne), [&](const int e) {v(e) = stgt(e0 + e);});
      tm.team_barrier();
      Kokkos::single(Kokkos::PerTeam(tm), [&]() {
        int *a = v.data();
        HeapSort(a, ne);
        int c = 0;
        for (int e=0; e<ne; ++e) if (e == 0 || a[e] != a[e - 1]) ++c;
        sz(q, 2) = c;
      });
    });
  }
  Kokkos::deep_copy(sz_h, sz);
  timing[0] = t.seconds();
  // buckets: at most 64 clouds, more; patches of one or no cloud only get their counters
  std::vector<int> lst[2];
  int nmx[2] = {0, 0}, emx[2] = {1, 1}, lmx[2] = {1, 1};
  for (int q=0; q<npatch; ++q) {
    const int n = sz_h(q, 0);
    if (n < 2) continue;
    const int b = (n <= 64) ? 0 : 1;
    lst[b].push_back(q);
    nmx[b] = std::max(nmx[b], n); emx[b] = std::max(emx[b], sz_h(q, 1));
    lmx[b] = std::max(lmx[b], sz_h(q, 2));
  }
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
  for (int b=0; b<2; ++b) {
    Kokkos::fence();
    t.reset();
    const int nq = static_cast<int>(lst[b].size());
    if (nq > 0) {
      // the largest patches first (balance); the order does not change any result
      std::stable_sort(lst[b].begin(), lst[b].end(),
                       [&](const int x, const int y) {return sz_h(x, 0) > sz_h(y, 0);});
      const WsSize ws = Sizes(nmx[b], emx[b], lmx[b]);
      const double per = 8.0*ws.nd + 4.0*ws.ni;
      int nslot = static_cast<int>(std::min<double>(nq, std::max(1.0, budget/per)));
      nslot = std::min(nslot, 4096);
      DvceArray1D<int> qlist("rs_dev_list", nq);
      {
        auto h = Kokkos::create_mirror_view(qlist);
        for (int i=0; i<nq; ++i) h(i) = lst[b][i];
        Kokkos::deep_copy(qlist, h);
      }
      DvceArray1D<double> dws(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_dev_ws_d"),
                              static_cast<std::size_t>(nslot)*ws.nd);
      DvceArray1D<int> iws(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_dev_ws_i"),
                           static_cast<std::size_t>(nslot)*ws.ni);
      DvceArray1D<int> counter("rs_dev_counter", 1);
      const DevicePatchProblem p = pb;
      Kokkos::parallel_for("rs_dev_patches", Policy(nslot, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Member &tm) {
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
          PatchTeam(tm, p, qlist(i), w, mn, oi, orr);
        }
      });
    }
    Kokkos::fence();
    timing[1 + b] = t.seconds();
  }
  timing[3] = t_all.seconds();
}

}  // namespace resample
}  // namespace dust
