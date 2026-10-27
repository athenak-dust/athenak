//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_resample.cpp
//! \brief Conservative same-level resampling of refinable clouds.
//! Positions, velocities, shapes and stopping times are kept; only the masses of the plain
//! clouds change, a cloud whose mass reaches zero is removed.  Per patch (kc x kc cells of one
//! MeshBlock, so that a patch holds about the same number of leaves on every level)
//! the new masses keep, to round-off, every leaf's deposited mass and momentum, the three
//! components of the frozen-gas drag feedback of every leaf (AuditFeedback), the centre of
//! mass and the six velocity second moments of the patch; a mass never exceeds its cap
//! eta m0 / 4^level.  Clouds above the cap are left alone, and so are the clouds already at
//! the cap.
//!
//! Device: selection of the pool, production gather of the gas velocity, cloud weights as a
//! CSR list.  Host (std::thread over patches): constraint rows, null-space end-point
//! reduction with deterministic end points.  The null vectors come from a Householder QR with
//! column pivoting.  Device: new masses, compaction.
//!
//! By default (<dust>/cloud_resample_device = true) the pool is put in patch order and every
//! patch is solved on the device (dust_resample_device.cpp, same algorithm; false keeps the host
//! solves).
//!
//! Schedule: every cloud_resample_interval cycles (default), or with
//! cloud_resample_schedule = time at the planned times t0 + k dt; see ResampleDue.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"
#include "dust_resample.hpp"
#include <Kokkos_Sort.hpp>

namespace dust {

namespace {

constexpr double kRankTol = 1.0e-13;   // relative size of a pivot column that counts as zero
constexpr double kResMax = 1.0e-10;    // largest accepted relative row residual of a patch
constexpr double kCapTol = 1.0e-12;
constexpr double kPoolTol = 1.0e-9;    // clouds with m <= cap (1 + kPoolTol) are in the pool
constexpr int kNCand = 8;              // null vectors looked at per step
constexpr double kInf = std::numeric_limits<double>::infinity();

struct PatchInfo {
  int n = 0, kept = 0, ncap = 0, fallback = 0, steps_zero = 0, steps_cap = 0, scans = 0;
  double res = 0.0;
  // seconds spent in the stages of this patch
  double t_rows = 0.0, t_scale = 0.0, t_qr = 0.0, t_null = 0.0, t_red = 0.0, t_check = 0.0;
  int nrow = 0, nb = 0;   // rows of the patch and the non-empty ones (sizes for the device port)
};

using StageClock = std::chrono::steady_clock;
inline double Since(const StageClock::time_point &a) {
  return std::chrono::duration<double>(StageClock::now() - a).count();
}

struct Work {
  std::vector<double> a, b, qr, z, d, wa, ua, vn1, vn2, x, w, den, r0;
  std::vector<int> perm, act, left, leaves;
  std::vector<char> fixed;
};

// Householder QR with column pivoting of the m x n matrix a (column major, overwritten):
// a P = Q R.  Returns the numerical rank (pivot column norm > kRankTol x the first one).
int PivotedQR(double *a, const int m, const int n, int *perm, double *vn1, double *vn2) {
  for (int j=0; j<n; ++j) {
    perm[j] = j;
    double s = 0.0;
    const double *c = a + static_cast<std::size_t>(j)*m;
    for (int i=0; i<m; ++i) s += c[i]*c[i];
    vn1[j] = vn2[j] = std::sqrt(s);
  }
  const int kmax = std::min(m, n);
  const double tol3z = std::sqrt(std::numeric_limits<double>::epsilon());
  double first = 0.0;
  for (int k=0; k<kmax; ++k) {
    int p = k;
    for (int j=k+1; j<n; ++j) if (vn1[j] > vn1[p]) p = j;
    if (k == 0) first = vn1[p];
    if (!(vn1[p] > kRankTol*first) || first == 0.0) return k;
    if (p != k) {
      double *cp = a + static_cast<std::size_t>(p)*m, *ck = a + static_cast<std::size_t>(k)*m;
      for (int i=0; i<m; ++i) std::swap(cp[i], ck[i]);
      std::swap(perm[p], perm[k]); std::swap(vn1[p], vn1[k]); std::swap(vn2[p], vn2[k]);
    }
    double *v = a + static_cast<std::size_t>(k)*m;
    double xnorm = 0.0;
    for (int i=k+1; i<m; ++i) xnorm += v[i]*v[i];
    xnorm = std::sqrt(xnorm);
    double tau = 0.0;
    if (xnorm > 0.0) {
      const double alpha = v[k];
      const double beta = -std::copysign(std::hypot(alpha, xnorm), alpha);
      tau = (beta - alpha)/beta;
      const double sc = 1.0/(alpha - beta);
      for (int i=k+1; i<m; ++i) v[i] *= sc;
      v[k] = beta;
    }
    for (int j=k+1; j<n; ++j) {
      double *c = a + static_cast<std::size_t>(j)*m;
      if (tau != 0.0) {
        double s = c[k];
        for (int i=k+1; i<m; ++i) s += v[i]*c[i];
        s *= tau;
        c[k] -= s;
        for (int i=k+1; i<m; ++i) c[i] -= s*v[i];
      }
      if (vn1[j] != 0.0) {
        double t = std::fabs(c[k])/vn1[j];
        t = std::max(0.0, (1.0 + t)*(1.0 - t));
        const double r = vn1[j]/vn2[j];
        if (t*r*r <= tol3z) {
          double s = 0.0;
          for (int i=k+1; i<m; ++i) s += c[i]*c[i];
          vn1[j] = vn2[j] = std::sqrt(s);
        } else {
          vn1[j] *= std::sqrt(t);
        }
      }
    }
  }
  return kmax;
}

// largest step t >= 0 along sign*d with 0 <= w + sign t d <= u, the cloud that stops it and
// whether it stops at zero (rs2_lib._steps)
inline void Step(const double *d, const double *w, const double *u, const int n,
                 const double sign, double &t, int &k, bool &zero) {
  double a0 = kInf, a1 = kInf;
  int k0 = 0, k1 = 0;
  for (int i=0; i<n; ++i) {
    const double s = sign*d[i];
    if (s < 0.0) {
      const double v = w[i]/std::fabs(d[i]);
      if (v < a0) {a0 = v; k0 = i;}
    } else if (s > 0.0) {
      const double v = (u[i] - w[i])/std::fabs(d[i]);
      if (v < a1) {a1 = v; k1 = i;}
    }
  }
  zero = a0 <= a1;
  t = zero ? a0 : a1;
  k = zero ? k0 : k1;
}

// end-point reduction of the columns b (m x n, rows scaled) with masses 0 < w <= u until the
// moving clouds are independent (rs2_lib._reduce_cap, deterministic end points)
void ReduceCap(const double *b, const int m, const int n, double *w, const double *u,
               char *fixed, PatchInfo &info, Work &ws) {
  for (int i=0; i<n; ++i) fixed[i] = 0;
  const auto tq = StageClock::now();
  ws.qr.assign(b, b + static_cast<std::size_t>(m)*n);
  ws.perm.resize(n); ws.vn1.resize(n); ws.vn2.resize(n);
  const int r = PivotedQR(ws.qr.data(), m, n, ws.perm.data(), ws.vn1.data(), ws.vn2.data());
  info.t_qr += Since(tq);
  if (r >= n) return;
  const auto tn = StageClock::now();
  int ncol = n - r;
  ws.z.assign(static_cast<std::size_t>(n)*ncol, 0.0);
  ws.x.resize(r);
  const double *q = ws.qr.data();
  for (int c=0; c<ncol; ++c) {       // null vector c: P [-R11^-1 R12 e_c; e_c]
    const double *rhs = q + static_cast<std::size_t>(r + c)*m;
    for (int i=r-1; i>=0; --i) {
      double s = rhs[i];
      for (int j=i+1; j<r; ++j) s -= q[static_cast<std::size_t>(j)*m + i]*ws.x[j];
      ws.x[i] = s/q[static_cast<std::size_t>(i)*m + i];
    }
    double *zc = ws.z.data() + static_cast<std::size_t>(c)*n;
    for (int i=0; i<r; ++i) zc[ws.perm[i]] = -ws.x[i];
    zc[ws.perm[r + c]] = 1.0;
  }
  info.t_null += Since(tn);
  const auto tr = StageClock::now();
  ws.d.resize(n);
  std::vector<double> t(2*static_cast<std::size_t>(ncol));
  std::vector<int> kk(2*static_cast<std::size_t>(ncol));
  std::vector<char> zr(2*static_cast<std::size_t>(ncol));
  while (ncol > 0) {
    int nc = std::min(kNCand, ncol);
    bool anyzero = false;
    auto steps = [&](const int c0, const int c1) {
      for (int c=c0; c<c1; ++c) {
        const double *zc = ws.z.data() + static_cast<std::size_t>(c)*n;
        for (int sg=0; sg<2; ++sg) {
          bool zero;
          Step(zc, w, u, n, sg ? 1.0 : -1.0, t[2*c + sg], kk[2*c + sg], zero);
          zr[2*c + sg] = zero ? 1 : 0;
          anyzero = anyzero || zero;
        }
      }
    };
    steps(0, nc);
    if (!anyzero && ncol > nc) {steps(nc, ncol); nc = ncol; ++info.scans;}
    double best = -1.0;
    int bj = -1, bs = 0;
    for (int sg=0; sg<2; ++sg) {      // minus first: argmax over (sign, column), ties to the first
      for (int c=0; c<nc; ++c) {
        if (!std::isfinite(t[2*c + sg])) continue;
        const double *zc = ws.z.data() + static_cast<std::size_t>(c)*n;
        double zmax = 0.0;
        for (int i=0; i<n; ++i) zmax = std::max(zmax, std::fabs(zc[i]));
        const double score = std::fabs(zc[kk[2*c + sg]])/zmax + 2.0*zr[2*c + sg];
        if (score > best) {best = score; bj = c; bs = sg;}
      }
    }
    if (bj < 0) break;
    const double *zj = ws.z.data() + static_cast<std::size_t>(bj)*n;
    for (int i=0; i<n; ++i) ws.d[i] = zj[i];
    const int k = kk[2*bj + bs];
    const double step = t[2*bj + bs]*(bs ? 1.0 : -1.0);
    for (int i=0; i<n; ++i) w[i] = std::min(std::max(w[i] + step*ws.d[i], 0.0), u[i]);
    if (zr[2*bj + bs]) {
      w[k] = 0.0; ++info.steps_zero;
    } else {
      w[k] = u[k]; fixed[k] = 1; ++info.steps_cap;
    }
    // drop column bj and remove cloud k from the remaining null vectors
    if (bj != ncol - 1) {
      double *zl = ws.z.data() + static_cast<std::size_t>(ncol - 1)*n;
      double *zd = ws.z.data() + static_cast<std::size_t>(bj)*n;
      for (int i=0; i<n; ++i) zd[i] = zl[i];
    }
    --ncol;
    const double dk = ws.d[k];
    for (int c=0; c<ncol; ++c) {
      double *zc = ws.z.data() + static_cast<std::size_t>(c)*n;
      const double f = zc[k]/dk;
      if (f != 0.0) for (int i=0; i<n; ++i) zc[i] -= f*ws.d[i];
      zc[k] = 0.0;
    }
  }
  info.t_red += Since(tr);
}

// the clouds of one patch (indices a0..a1-1 of the patch-sorted pool): rows, bounded
// end-point reduction as a stream (rs2_lib.prune_cap), row check; new masses in mnew
PatchInfo ResamplePatch(const int a0, const int a1, const double *pd, const int npd,
                        const int *cptr, const int *ctgt, const double *cwgt, const double xc,
                        const double yc, const double lx, const double ly, const double rdx,
                        const double rdy, double *mnew, Work &ws) {
  enum {JX=0, JY=1, JM=2, JVX=3, JVY=4, JVZ=5, JTS=6, JUX=7, JUY=8, JUZ=9, JCAP=10};
  PatchInfo info;
  const int n = a1 - a0;
  info.n = n; info.kept = n;
  auto P = [&](const int c, const int a) {return pd[static_cast<std::size_t>(a)*npd + c];};
  for (int a=a0; a<a1; ++a) mnew[a] = P(JM, a);
  if (n < 2) return info;
  auto ts = StageClock::now();
  // leaves under the clouds of the patch
  ws.leaves.clear();
  for (int e=cptr[a0]; e<cptr[a1]; ++e) ws.leaves.push_back(ctgt[e]);
  std::sort(ws.leaves.begin(), ws.leaves.end());
  ws.leaves.erase(std::unique(ws.leaves.begin(), ws.leaves.end()), ws.leaves.end());
  const int nl = static_cast<int>(ws.leaves.size());
  const int nrow = 7*nl + 8;
  info.nrow = nrow;
  double mt = 0.0, vb[3] = {0.0, 0.0, 0.0};
  for (int a=a0; a<a1; ++a) {
    mt += P(JM, a);
    for (int k=0; k<3; ++k) vb[k] += P(JM, a)*P(JVX + k, a);
  }
  for (int k=0; k<3; ++k) vb[k] /= mt;
  // rows: leaf mass, leaf momentum about the patch mean, centre of
  // mass, velocity second moments, leaf feedback (the leaf volume is a row factor: left out)
  ws.a.assign(static_cast<std::size_t>(nrow)*n, 0.0);
  for (int a=a0; a<a1; ++a) {
    double *col = ws.a.data() + static_cast<std::size_t>(a - a0)*nrow;
    const double c[3] = {P(JVX, a) - vb[0], P(JVY, a) - vb[1], P(JVZ, a) - vb[2]};
    const double f[3] = {(P(JUX, a) - P(JVX, a))/P(JTS, a), (P(JUY, a) - P(JVY, a))/P(JTS, a),
                         (P(JUZ, a) - P(JVZ, a))/P(JTS, a)};
    for (int e=cptr[a]; e<cptr[a+1]; ++e) {
      const int l = static_cast<int>(std::lower_bound(ws.leaves.begin(), ws.leaves.end(), ctgt[e]) -
                                     ws.leaves.begin());
      const double w = cwgt[e];
      col[l] += w;
      for (int k=0; k<3; ++k) {col[(1 + k)*nl + l] += w*c[k]; col[4*nl + 8 + k*nl + l] += w*f[k];}
    }
    double dx = P(JX, a) - xc, dy = P(JY, a) - yc;
    dx -= lx*std::floor(dx/lx + 0.5); dy -= ly*std::floor(dy/ly + 0.5);
    double *g = col + 4*nl;
    g[0] = dx/rdx; g[1] = dy/rdy;
    g[2] = c[0]*c[0]; g[3] = c[0]*c[1]; g[4] = c[0]*c[2]; g[5] = c[1]*c[1]; g[6] = c[1]*c[2];
    g[7] = c[2]*c[2];
  }
  info.t_rows += Since(ts);
  ts = StageClock::now();
  // scaled rows without the empty ones
  ws.r0.assign(nrow, 0.0);
  for (int j=0; j<n; ++j) {
    const double *col = ws.a.data() + static_cast<std::size_t>(j)*nrow;
    for (int i=0; i<nrow; ++i) ws.r0[i] = std::max(ws.r0[i], std::fabs(col[i]));
  }
  int nb = 0;
  for (int i=0; i<nrow; ++i) if (ws.r0[i] > 0.0) ++nb;
  info.nb = nb;
  ws.b.resize(static_cast<std::size_t>(nb)*n);
  for (int j=0; j<n; ++j) {
    const double *col = ws.a.data() + static_cast<std::size_t>(j)*nrow;
    double *bc = ws.b.data() + static_cast<std::size_t>(j)*nb;
    int s = 0;
    for (int i=0; i<nrow; ++i) if (ws.r0[i] > 0.0) bc[s++] = col[i]/ws.r0[i];
  }
  ws.w.resize(n);
  for (int j=0; j<n; ++j) ws.w[j] = P(JM, a0 + j);
  const int rb = std::min(nb, n);
  const int cap = std::max(2*rb, rb + 32);
  ws.act.clear();
  int nxt = 0;
  std::vector<double> bact;
  info.t_scale += Since(ts);
  while (true) {
    const int take = std::min(n - nxt, cap - static_cast<int>(ws.act.size()));
    for (int i=0; i<take; ++i) ws.act.push_back(nxt + i);
    nxt += take;
    const int na = static_cast<int>(ws.act.size());
    if (na < 2) break;
    bact.resize(static_cast<std::size_t>(nb)*na);
    ws.wa.resize(na); ws.ua.resize(na); ws.fixed.resize(na);
    for (int i=0; i<na; ++i) {
      const int j = ws.act[i];
      std::copy(ws.b.data() + static_cast<std::size_t>(j)*nb, ws.b.data() + static_cast<std::size_t>(j + 1)*nb,
                bact.data() + static_cast<std::size_t>(i)*nb);
      ws.wa[i] = ws.w[j]; ws.ua[i] = P(JCAP, a0 + j);
    }
    ReduceCap(bact.data(), nb, na, ws.wa.data(), ws.ua.data(), ws.fixed.data(), info, ws);
    ws.left.clear();
    for (int i=0; i<na; ++i) {
      ws.w[ws.act[i]] = ws.wa[i];
      if (ws.wa[i] > 0.0 && !ws.fixed[i]) ws.left.push_back(ws.act[i]);
    }
    if (nxt >= n) break;      // the moving clouds left by the last call are independent
    if (ws.left.size() == ws.act.size()) break;   // no room was made: the rest keeps its masses
    ws.act.swap(ws.left);
  }
  ts = StageClock::now();
  // row check on the unscaled rows: |A w - A m| / (|A| m)
  bool bad = false;
  double res = 0.0;
  ws.den.assign(nrow, 0.0);
  std::vector<double> dif(nrow, 0.0);
  for (int j=0; j<n; ++j) {
    const double *col = ws.a.data() + static_cast<std::size_t>(j)*nrow;
    const double m = P(JM, a0 + j), u = P(JCAP, a0 + j), dw = ws.w[j] - m;
    if (!(ws.w[j] >= 0.0) || ws.w[j] > u*(1.0 + kCapTol)) bad = true;
    for (int i=0; i<nrow; ++i) {ws.den[i] += std::fabs(col[i])*m; dif[i] += col[i]*dw;}
  }
  for (int i=0; i<nrow; ++i) res = std::max(res, std::fabs(dif[i])/std::max(ws.den[i], 1.0e-300));
  info.res = res;
  info.t_check += Since(ts);
  if (bad || !(res <= kResMax)) {
    info.fallback = 2;
    return info;      // the patch keeps its masses
  }
  info.kept = 0;
  for (int j=0; j<n; ++j) {
    mnew[a0 + j] = ws.w[j];
    if (ws.w[j] > 0.0) {
      ++info.kept;
      if (ws.w[j] >= P(JCAP, a0 + j)*(1.0 - kCapTol)) ++info.ncap;
    }
  }
  return info;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn dust::PoolInTagOrder
//! \brief cloud_resample_order = tag with 64-bit particle tags: the pool
//! (the particles p with slot(p) >= 0, npool of them) in (tag, p) order; on return slot(p) is
//! the rank of p and pofs(k) the particle of rank k.  While tag*npart + p fits into an int64
//! (every tag below 2^63/npart) one int64 key per particle is sorted, which gives the order
//! of the int32 key (tag + 2^31)*npart + p used before; above, a (tag, p) pair is sorted with
//! a comparator.  Both are the same strict order, so the result does not depend on the path.

struct PoolTagKey {std::uint64_t tag; int p;};
struct PoolTagKeyLess {
  KOKKOS_INLINE_FUNCTION
  bool operator()(const PoolTagKey &a, const PoolTagKey &b) const {
    return (a.tag < b.tag) || (a.tag == b.tag && a.p < b.p);
  }
};

void PoolInTagOrder(const DvceArray2D<int> &pi, const DvceArray1D<int> &slot, const int npart,
                    const int npool, const DvceArray1D<int> &pofs) {
  std::uint64_t tmax = 0;
  Kokkos::parallel_reduce("rs_tagmax", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, std::uint64_t &m) {
    if (slot(p) >= 0 && GetTag(pi,p) > m) m = GetTag(pi,p);
  }, Kokkos::Max<std::uint64_t>(tmax));
  const std::int64_t np64 = npart;
  const std::uint64_t fit = static_cast<std::uint64_t>(
      (std::numeric_limits<std::int64_t>::max() - (np64 - 1))/np64);
  if (tmax <= fit) {
    DvceArray1D<std::int64_t> tk(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_tagkey"), npool);
    Kokkos::parallel_for("rs_tagkey", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (slot(p) < 0) return;
      tk(slot(p)) = static_cast<std::int64_t>(GetTag(pi,p))*np64 + p;
    });
    Kokkos::sort(tk);
    Kokkos::parallel_for("rs_tagslot", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int k) {
      const int p = static_cast<int>(tk(k) % np64);
      slot(p) = k; pofs(k) = p;
    });
  } else {
    Kokkos::View<PoolTagKey*, DevMemSpace> tk(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                 "rs_tagpair"), npool);
    Kokkos::parallel_for("rs_tagpair", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (slot(p) < 0) return;
      tk(slot(p)).tag = GetTag(pi,p);
      tk(slot(p)).p = p;
    });
    Kokkos::sort(tk, PoolTagKeyLess());
    Kokkos::parallel_for("rs_tagpslot", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int k) {
      const int p = tk(k).p;
      slot(p) = k; pofs(k) = p;
    });
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ResampleDue
//! \brief Is a resampling due at the start of this cycle?  Cycle mode:
//! every cloud_resample_interval cycles, as before.  Time mode: event k is planned at
//! t0 + k dt and runs at the first cycle that starts at or after that time (the mesh time at
//! the start of the cycle; the hydro time step is never changed for it).  A cycle that has
//! passed several planned times runs once; the others count as skipped, and the next event
//! is the first one planned after the time of this cycle.  The first event of a run is the
//! first planned time not earlier than its start; a restart continues from the schedule
//! record of the file when the origin and interval are unchanged, stops with an error when
//! they changed (unless cloud_resample_schedule_reset), and derives the next event from the
//! restart time when the file has no time-mode record.

bool DustGasDrag::ResampleDue() {
  Mesh *pmesh = pmy_pack->pmesh;
  if (!cloud_resample_by_time) return pmesh->ncycle % cloud_resample_interval == 0;
  const double t = pmesh->time, dt = cloud_resample_dt, t0 = cloud_resample_t0;
  // slack only against round-off of a step that ends exactly on a planned time
  auto eps = [](const double tk) { return 1.0e-12*std::fmax(1.0, std::fabs(tk)); };
  auto planned = [&](const long long k) { return t0 + static_cast<double>(k)*dt; };
  // number of the first planned time not earlier than tt
  auto first_at_or_after = [&](const double tt) {
    long long k = static_cast<long long>(std::ceil((tt - t0)/dt));
    if (k < 0) k = 0;
    while (k > 0 && planned(k - 1) >= tt - eps(planned(k - 1))) --k;
    while (planned(k) < tt - eps(planned(k))) ++k;
    return k;
  };
  if (cloud_resample_next < 0) {
    std::string how;
    if (cloud_resample_clock_restored && cloud_resample_t0_file == t0 &&
        cloud_resample_dt_file == dt) {
      cloud_resample_next = cloud_resample_next_file;
      how = "continued from the schedule record of the restart file";
    } else if (cloud_resample_clock_restored && !cloud_resample_schedule_reset) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "the restart file plans the resampling at t0 = " << cloud_resample_t0_file
                << " + k " << cloud_resample_dt_file << ", the input at t0 = " << t0 << " + k "
                << dt << "; set <dust>/cloud_resample_schedule_reset = true to change it"
                << std::endl;
      std::exit(EXIT_FAILURE);
    } else {
      cloud_resample_next = first_at_or_after(t);
      how = cloud_resample_clock_restored ? "reset at the restart: first planned time not earlier "
                                            "than the restart time"
          : cloud_resample_restarted ? "the restart file has no time-mode schedule record: first "
                                       "planned time not earlier than the restart time"
          : "first planned time not earlier than the start time";
    }
    if (global_variable::my_rank == 0) {
      std::cout << "# resampling schedule: time mode, omega0 = " << cloud_resample_omega
                << ", interval " << dt << " (code time), origin " << t0 << "; next event "
                << cloud_resample_next << " planned at " << planned(cloud_resample_next) << " ("
                << how << ")" << std::endl;
    }
  }
  const double tk = planned(cloud_resample_next);
  if (t < tk - eps(tk)) return false;
  // every planned time up to t (within the slack) is done with this one execution
  long long nxt = static_cast<long long>(std::floor((t - t0)/dt)) + 1;
  if (nxt < cloud_resample_next + 1) nxt = cloud_resample_next + 1;
  while (planned(nxt) <= t + eps(planned(nxt))) ++nxt;
  while (nxt > cloud_resample_next + 1 && planned(nxt - 1) > t + eps(planned(nxt - 1))) --nxt;
  const long long skipped = nxt - cloud_resample_next - 1;
  ++cloud_resample_events;
  cloud_resample_skipped += static_cast<unsigned long long>(skipped);
  cloud_resample_next = nxt;
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MergeDue
//! \brief Is a merge pass due at the start of this cycle?  Cycle mode:
//! every cloud_merge_interval cycles (unchanged).  Time mode: at the planned times
//! t0 + k dt, by the rules of ResampleDue (first cycle that starts at or after the time; one
//! pass however many planned times a cycle passed).  Either way, during a burst (after a
//! regrid that removed blocks) also every cloud_merge_burst_interval cycles.

bool DustGasDrag::MergeDue() {
  Mesh *pmesh = pmy_pack->pmesh;
  bool due = false;
  if (!cloud_merge_by_time) {
    due = (pmesh->ncycle % cloud_merge_interval == 0);
  } else {
    const double t = pmesh->time, dt = cloud_merge_dt, t0 = cloud_merge_t0;
    auto eps = [](const double tk) { return 1.0e-12*std::fmax(1.0, std::fabs(tk)); };
    auto planned = [&](const long long k) { return t0 + static_cast<double>(k)*dt; };
    if (cloud_merge_next < 0) {
      long long k = static_cast<long long>(std::ceil((t - t0)/dt));
      if (k < 0) k = 0;
      while (k > 0 && planned(k - 1) >= t - eps(planned(k - 1))) --k;
      while (planned(k) < t - eps(planned(k))) ++k;
      cloud_merge_next = k;
    }
    if (t >= planned(cloud_merge_next) - eps(planned(cloud_merge_next))) {
      due = true;
      long long nxt = static_cast<long long>(std::floor((t - t0)/dt)) + 1;
      if (nxt < cloud_merge_next + 1) nxt = cloud_merge_next + 1;
      while (planned(nxt) <= t + eps(planned(nxt))) ++nxt;
      cloud_merge_next = nxt;
    }
  }
  if (pmesh->time < cloud_merge_burst_until && pmesh->ncycle % cloud_merge_burst_interval == 0) due = true;
  return due;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ResampleClouds
//! \brief One resampling of the plain clouds (see the file header).  Returns the number of
//! clouds removed.

int DustGasDrag::ResampleClouds(const bool local) {
  ++gravity_mass_epoch;   // R changes the deposited mass (gravity field reuse)
  if (clouds3d) return ResampleClouds3D(local);
  using Clock = std::chrono::steady_clock;
  auto seconds = [](const Clock::time_point &a) {
    return std::chrono::duration<double>(Clock::now() - a).count();
  };
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart = ppar->nprtcl_thispack;
  const auto transfer = smr_transfer;
  const auto lt = leaf_traversal;   // cloud weights on the actual leaves
  // the gas velocity and its differences of every leaf of the current state, the
  // owners' values on the halo blocks, for the gather of the pool (the predictor's packed
  // field).  Before any return: the halo exchange is collective.
  PackGasGather(pmy_pack->phydro->u0);
  // the root cloud mass (reduced over the ranks): also before any return
  if (cloud_resample_m0 <= 0.0) EnsureRootCloudMass();
  if (npart == 0) return 0;
  // the root cloud mass of every species (cap eta m0 / 4^level): cloud_resample_m0 for all of
  // them, else the merge pass's per-species value.  The patches are also split by
  // species, so every species keeps its own leaf masses (one species: no change)
  // A species without such a value (no plain level-0/1 cloud) has a cap of zero: its clouds
  // stay out of the pool.
  const int nsp = std::max(1, nspecies);
  DvceArray1D<Real> m0sp("rs_m0", nsp);
  {
    // a missing value only takes its own species out of the pool (e.g. eps_1 = 0 with a second
    // species); fatal only when no species has one
    auto mh = Kokkos::create_mirror_view(m0sp);
    bool any = false;
    for (int sp=0; sp<nsp; ++sp) {
      mh(sp) = (cloud_resample_m0 > 0.0) ? cloud_resample_m0
             : ((static_cast<int>(cloud_merge_mroot.size()) > sp) ? cloud_merge_mroot[sp] : 0.0);
      if (mh(sp) > 0.0) any = true;
    }
    if (!any) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "cloud_resample needs the root cloud mass: no species has a plain level-0 or "
                << "level-1 cloud; set <dust>/cloud_resample_m0" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    static bool told = false;
    if (!told && global_variable::my_rank == 0) {
      for (int sp=0; sp<nsp; ++sp) {
        if (!(mh(sp) > 0.0)) {
          std::cout << "# NOTE (dust): species " << sp << " has no root cloud mass (no plain level-0 "
                    << "or level-1 cloud of positive mass): its clouds stay out of the resampling"
                    << std::endl;
        }
      }
      told = true;
    }
    Kokkos::deep_copy(m0sp, mh);
  }
  // the patches: inside one MeshBlock, k x k of its cells with k = min(s 2^l, kc) (s x s root
  // cells on the root level, up to the level where they reach kc x kc cells, never more), so a
  // patch holds at most about (kc + 2)^2 leaves on any level.
  const int s = cloud_resample_patch, kc = cloud_resample_patch_cells;
  const int nmbp = pmy_pack->nmb_thispack, bnx1 = pmesh->mb_indcs.nx1, bnx2 = pmesh->mb_indcs.nx2;
  std::vector<int> kblk_h, poff_h(nmbp + 1, 0);   // block patches: k per block, patch offsets
  bool bad = (kc < 1);
  if (!bad) {
    kblk_h.resize(nmbp);
    for (int b=0; b<nmbp; ++b) {
      const int lev = pmy_pack->pmb->mb_lev.h_view(b) - pmesh->root_level;
      int k = s;
      for (int l=0; l<lev && k < kc; ++l) k *= 2;
      k = std::min(k, kc);
      if (bnx1 % k != 0 || bnx2 % k != 0) bad = true;
      // the quadrant split halves the width down to one cell, so k must be a power of two
      if (cloud_resample_patch_split > 0 && (k & (k - 1)) != 0) bad = true;
      kblk_h[b] = k;
      poff_h[b + 1] = poff_h[b] + (bnx1/k)*(bnx2/k);
    }
  }
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "block patches of min(cloud_resample_patch 2^level, "
                 "cloud_resample_patch_cells) cells must divide the MeshBlock (and be "
                 "powers of two with cloud_resample_patch_split > 0)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  DvceArray1D<int> kblk("rs_kblk", nmbp), poff("rs_poff", nmbp + 1);
  {
    auto kb = Kokkos::create_mirror_view(kblk);
    auto po = Kokkos::create_mirror_view(poff);
    for (int b=0; b<nmbp; ++b) kb(b) = kblk_h[b];
    for (int b=0; b<=nmbp; ++b) po(b) = poff_h[b];
    Kokkos::deep_copy(kblk, kb);
    Kokkos::deep_copy(poff, po);
  }
  // local resampling: the blocks of this rank made by the last regrid (no cycle since their
  // refinement) and every block of this rank whose box touches one of them (any rank, corners and
  // periodic images included); clouds anchored elsewhere stay out of the pool
  DvceArray1D<int> rmask("rs_mask", std::max(1, nmbp));
  int nmask = nmbp;
  if (local) {
    auto pmr = pmesh->pmr;
    const int gids0 = pmy_pack->gids, ntot = pmesh->nmb_total;
    const Real lx = pmesh->mesh_size.x1max - pmesh->mesh_size.x1min;
    const Real ly = pmesh->mesh_size.x2max - pmesh->mesh_size.x2min;
    auto box = [&](const int g, Real b[4]) {
      const auto &ll = pmesh->lloc_eachmb[g];
      const int lev = ll.level - pmesh->root_level;
      const Real w1 = lx/(pmesh->nmb_rootx1 << lev), w2 = ly/(pmesh->nmb_rootx2 << lev);
      b[0] = pmesh->mesh_size.x1min + ll.lx1*w1; b[1] = b[0] + w1;
      b[2] = pmesh->mesh_size.x2min + ll.lx2*w2; b[3] = b[2] + w2;
    };
    std::vector<std::array<Real, 4>> fresh;
    for (int g=0; g<ntot; ++g) {
      // (every regrid of the event, cloud_event_dt; 0 for one regrid)
      if (pmr->ncyc_since_ref(g) <= amr_event_max_age) {
        Real b[4]; box(g, b); fresh.push_back({b[0], b[1], b[2], b[3]});
      }
    }
    auto mh = Kokkos::create_mirror_view(rmask);
    nmask = 0;
    for (int m=0; m<nmbp; ++m) {
      Real b[4]; box(gids0 + m, b);
      const Real e1 = 1.0e-9*(b[1] - b[0]), e2 = 1.0e-9*(b[3] - b[2]);
      int on = 0;
      for (const auto &f : fresh) {
        for (int oy=-1; oy<=1 && !on; ++oy) {
          for (int ox=-1; ox<=1 && !on; ++ox) {
            const Real a0 = f[0] + ox*lx, a1 = f[1] + ox*lx, c0 = f[2] + oy*ly, c1 = f[3] + oy*ly;
            if (a0 <= b[1] + e1 && b[0] <= a1 + e1 && c0 <= b[3] + e2 && b[2] <= c1 + e2) on = 1;   // touch
          }
        }
        if (on) break;
      }
      mh(m) = on; nmask += on;
    }
    Kokkos::deep_copy(rmask, mh);
  }
  const bool use_mask = local;
  // with cloud_resample_patch_split = N > 0 every block patch of k x k cells has one
  // slot per cell (kc^2 slots); a patch, or a quadrant of it, whose pool (one species) holds more
  // than N clouds is solved as its four quadrants, recursively down to single cells; each final
  // square is one patch in the slot of its lower-left cell
  const int psplit = cloud_resample_patch_split;
  const int psub = (psplit > 0) ? kc*kc : 1;
  const Real eta = cloud_resample_eta;
  auto t_all = Clock::now();
  Kokkos::fence();
  auto t0 = Clock::now();

  // device: pool flags, slots and CSR offsets
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  DvceArray1D<int> cnt(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_cnt"), npart);
  DvceArray1D<int> slot(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_slot"), npart);
  DvceArray1D<int> coff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_coff"), npart);
  int npool = 0, nnz = 0;
  double tot_b[2] = {0.0, 0.0}, cnt_b[3] = {0.0, 0.0, 0.0};
  // cold gate (cloud_resample_cold_alpha > 0): a block patch (one species) is cold when
  // every velocity component of its candidate clouds spans at most alpha H / tau (H the cell size of
  // its block, tau = min(ts, 1/omega), the velocity tolerance of stage X).  Resampling keeps the
  // leaf masses at the moment of the pass but replaces a lattice (or intact P1 families) by unequal
  // masses, whose deposit is no longer uniform once the clouds move (aliasing): a quiet state is
  // left alone; stage X restores its families exactly where they leave the finer leaves.
  const Real calpha = cloud_resample_cold_alpha, comega = cloud_resample_omega;
  const int nqs = (calpha > 0.0) ? std::max(1, poff_h[nmbp]*nsp) : 1;
  DvceArray2D<Real> crange("rs_cold_range", nqs, 6);
  DvceArray1D<int> ccold("rs_cold", nqs);
  DvceArray1D<int> cq("rs_cold_q", (calpha > 0.0) ? std::max(1, npart) : 1);
  double cnt_cold = 0.0;
  if (calpha > 0.0) {
    Kokkos::parallel_for("rs_cold_init", Kokkos::RangePolicy<>(DevExeSpace(), 0, nqs),
    KOKKOS_LAMBDA(const int q) {
      for (int c=0; c<3; ++c) {crange(q,c) = 1.0e300; crange(q,3+c) = -1.0e300;}
      ccold(q) = 1;
    });
    const int js_ = pmesh->mb_indcs.js, is_ = pmesh->mb_indcs.is;
    const int nc1 = qdep.extent_int(4), nc2 = qdep.extent_int(3);
    auto prc = ppar->prtcl_rdata;
    auto pic = ppar->prtcl_idata;
    // the candidates (the pool tests of rs_flag below) and their block patch
    Kokkos::parallel_for("rs_cold_q", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      cq(p) = -1;
      if (prc(IPM,p) <= 0.0) return;
      const int code = pic(PLEV,p);
      if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) return;
      const int am = lt.Anchor(pic(PGID,p));
      if (use_mask && (am < 0 || am >= nmbp || rmask(am) == 0)) return;
      const int lev = SMRCompositeTransfer::CloudLevel(code);
      const int sp = (pic(PSP,p) >= 0 && pic(PSP,p) < nsp) ? pic(PSP,p) : 0;
      const Real cap = (eta > 0.0) ? eta*m0sp(sp)/static_cast<Real>(1 << (2*lev)) : 1.0e300;
      if (!(prc(IPM,p) <= cap*(1.0 + kPoolTol)) || prc(IPM,p) >= cap*(1.0 - kCapTol)) return;
      const int t = lt.LeafOwner(prc(IPX,p), prc(IPY,p), am);
      if (t < 0) return;
      const int b = t/(nc1*nc2);
      if (b >= nmbp) return;
      const int r = t % (nc1*nc2);
      const int tj = r/nc1 - js_, ti = r % nc1 - is_;
      const int k = kblk(b);
      const int qs = (poff(b) + (tj/k)*(bnx1/k) + ti/k)*nsp + sp;
      cq(p) = qs;
      const Real v[3] = {prc(IPVX,p), prc(IPVY,p), prc(IPVZ,p)};
      for (int c=0; c<3; ++c) {
        Kokkos::atomic_min(&crange(qs,c), v[c]);
        Kokkos::atomic_max(&crange(qs,3+c), v[c]);
      }
    });
    Kokkos::parallel_for("rs_cold_flag", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      const int qs = cq(p);
      if (qs < 0) return;
      const Real ts = prc(IPTS,p);
      const Real tau = (comega > 0.0) ? Kokkos::fmin(ts, 1.0/comega) : ts;
      const int b = lt.LeafOwner(prc(IPX,p), prc(IPY,p), lt.Anchor(pic(PGID,p)))/(nc1*nc2);
      const Real tolv = calpha*lt.box(b,2)/tau;
      for (int c=0; c<3; ++c) {
        if (crange(qs,3+c) - crange(qs,c) > tolv) {Kokkos::atomic_store(&ccold(qs), 0); return;}
      }
    });
  }
  // cnt: number of (leaf, weight) entries of a pool cloud, -1 for the others; totals before;
  // lobed clouds, plain clouds above the cap, zero-mass slots
  const int qdep_n1 = qdep.extent_int(4), qdep_n2 = qdep.extent_int(3);
  const bool multirank = (global_variable::nranks > 1);
  Kokkos::parallel_reduce("rs_flag", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, double &mt, double &px, double &nl, double &nf, double &nz,
                double &ncold) {
    cnt(p) = -1;
    if (pr(IPM,p) <= 0.0) {nz += 1.0; return;}
    mt += pr(IPM,p); px += pr(IPM,p)*pr(IPVX,p);
    const int code = pi(PLEV,p);
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) {nl += 1.0; return;}
    if (use_mask) {   // local resampling: only clouds anchored in the marked blocks
      const int am = lt.Anchor(pi(PGID,p));
      if (am < 0 || am >= nmbp || rmask(am) == 0) return;
    }
    const int sp = (pi(PSP,p) >= 0 && pi(PSP,p) < nsp) ? pi(PSP,p) : 0;
    const Real cap = (eta > 0.0) ? eta*m0sp(sp)/static_cast<Real>(1 << (2*lev)) : 1.0e300;
    if (!(pr(IPM,p) <= cap*(1.0 + kPoolTol))) {nf += 1.0; return;}
    if (pr(IPM,p) >= cap*(1.0 - kCapTol)) {nf += 1.0; return;}   // at the cap: left alone
    if (calpha > 0.0 && cq(p) >= 0 && ccold(cq(p)) == 1) {ncold += 1.0; return;}   // cold patch
    // a cloud whose leaf is on a block of another rank stays out (several
    // ranks; a position within round-off of a periodic face can be routed to this rank's block while
    // its leaf lookup takes the periodic image on the other side, owned elsewhere)
    if (multirank) {
      const int tl = lt.LeafOwner(pr(IPX,p), pr(IPY,p), lt.Anchor(pi(PGID,p)));
      if (tl < 0 || tl/(qdep_n1*qdep_n2) >= nmbp) return;
    }
    int c = 0;
    lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), code, lt.Anchor(pi(PGID,p)),
                          [&](const int, const Real) {++c;});
    cnt(p) = c;
  }, tot_b[0], tot_b[1], cnt_b[0], cnt_b[1], cnt_b[2], cnt_cold);
  cloud_resample_cold_skips += static_cast<unsigned long long>(cnt_cold);
  Kokkos::parallel_scan("rs_slot", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) slot(p) = (cnt(p) >= 0) ? upd : -1;
    upd += (cnt(p) >= 0) ? 1 : 0;
  }, npool);
  if (cloud_resample_tag_order && npool > 1) {
    // cloud_resample_order = tag: the pool in PTAG order, so that the streaming
    // order inside a patch does not depend on where the particles sit in the arrays (storage
    // sorting, several ranks); the CSR offsets follow the slots
    DvceArray1D<int> pofs(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_pofs"), npool);
    // position: rows of y, then x, 2^20 per side of the domain (cloud_resample_order = position)
    const bool bypos = cloud_resample_pos_order;
    if (bypos) {
      DvceArray1D<std::int64_t> tk(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_poskey"), npool);
      const std::int64_t np64 = npart, side = std::int64_t(1) << 20;
      if (static_cast<double>(side)*side*np64 >= 9.0e18) {
        std::cout << "### FATAL ERROR: cloud_resample_order = position: too many particles" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const Real px0 = transfer.x1min, py0 = transfer.x2min;
      const Real plx = transfer.x1max - transfer.x1min, ply = transfer.x2max - transfer.x2min;
      Kokkos::parallel_for("rs_poskey", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        if (slot(p) < 0) return;
        const Real x = SMRCompositeTransfer::WrapPosition(pr(IPX,p), transfer.x1min, transfer.x1max);
        const Real y = SMRCompositeTransfer::WrapPosition(pr(IPY,p), transfer.x2min, transfer.x2max);
        std::int64_t ix = static_cast<std::int64_t>((x - px0)/plx*side);
        std::int64_t iy = static_cast<std::int64_t>((y - py0)/ply*side);
        ix = (ix < 0) ? 0 : ((ix >= side) ? side - 1 : ix);
        iy = (iy < 0) ? 0 : ((iy >= side) ? side - 1 : iy);
        tk(slot(p)) = (iy*side + ix)*np64 + p;
      });
      Kokkos::sort(tk);
      Kokkos::parallel_for("rs_posslot", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
      KOKKOS_LAMBDA(const int k) {
        const int p = static_cast<int>(tk(k) % np64);
        slot(p) = k; pofs(k) = p;
      });
    } else {
      PoolInTagOrder(pi, slot, npart, npool, pofs);   // 64-bit tags
    }
    Kokkos::parallel_scan("rs_coff", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int a, int &upd, const bool final) {
      const int p = pofs(a);
      if (final) coff(p) = upd;
      upd += (cnt(p) > 0) ? cnt(p) : 0;
    }, nnz);
  } else {
    Kokkos::parallel_scan("rs_coff", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
      if (final) coff(p) = upd;
      upd += (cnt(p) > 0) ? cnt(p) : 0;
    }, nnz);
  }
  if (npool < 2) {   // nothing to resample
    ++cloud_resample_pass;
    return 0;
  }

  // device: pool table (x, y, m, v, ts, gathered gas velocity, cap), ids, CSR weights
  const int npd = 11;
  DvceArray2D<Real> pd(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_pd"), npool, npd);
  DvceArray2D<int> pid(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_pid"), npool, 2);
  DvceArray1D<int> cptr(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_cptr"), npool + 1);
  DvceArray1D<int> ctgt(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_ctgt"), nnz);
  DvceArray1D<Real> cwgt(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_cwgt"), nnz);
  auto &indcs = pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  auto gp = gpack;   // PackGasGather above
  const int ncells1 = qdep.extent_int(4), ncells2 = qdep.extent_int(3);
  const int tmode = cloud_transition;
  const int nnz_ = nnz;
  const int nmbp_ = pmy_pack->nmb_thispack;
  Kokkos::parallel_for("rs_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    const int a = slot(p);
    if (p == npart - 1) cptr(npool) = nnz_;
    if (a < 0) return;
    const Real xp = pr(IPX,p), yp = pr(IPY,p);
    const int code = pi(PLEV,p);
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
    lt.ForEachCloudMoment(xp, yp, SMRCompositeTransfer::GatherCode(code, tmode),
                          lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w, const Real m1x, const Real m1y) {
      const int ti = target % ncells1;
      const int remainder = target/ncells1;
      const int tj = remainder % ncells2;
      const int tm = remainder/ncells2;
      // the leaf's block slot may be a halo block: its box from lt, the gas from
      // the packed field (the predictor's gather)
      const Real dx1 = lt.box(tm,2), dx2 = lt.box(tm,3);
      const Real xc = lt.box(tm,0) + (ti - is + 0.5)*dx1;
      const Real yc = lt.box(tm,1) + (tj - js + 0.5)*dx2;
      const Real cx = 0.5*(m1x + transfer.MinImage1(xp - xc)*w)/dx1;
      const Real cy = 0.5*(m1y + transfer.MinImage2(yp - yc)*w)/dx2;
      ugx += w*gp(tm,ks,tj,ti,0) + cx*gp(tm,ks,tj,ti,3) + cy*gp(tm,ks,tj,ti,6);
      ugy += w*gp(tm,ks,tj,ti,1) + cx*gp(tm,ks,tj,ti,4) + cy*gp(tm,ks,tj,ti,7);
      ugz += w*gp(tm,ks,tj,ti,2) + cx*gp(tm,ks,tj,ti,5) + cy*gp(tm,ks,tj,ti,8);
    });
    pd(a,0) = xp; pd(a,1) = yp; pd(a,2) = pr(IPM,p);
    pd(a,3) = pr(IPVX,p); pd(a,4) = pr(IPVY,p); pd(a,5) = pr(IPVZ,p); pd(a,6) = pr(IPTS,p);
    pd(a,7) = ugx; pd(a,8) = ugy; pd(a,9) = ugz;
    const int sp = (pi(PSP,p) >= 0 && pi(PSP,p) < nsp) ? pi(PSP,p) : 0;
    pd(a,10) = (eta > 0.0) ? eta*m0sp(sp)/static_cast<Real>(1 << (2*lev)) : 1.0e300;
    {
      // the leaf that holds the particle: its block and cell
      const int am = lt.Anchor(pi(PGID,p));
      const int t = lt.LeafOwner(xp, yp, am);
      int b = am, ti = 0, tj = 0;
      if (t >= 0) {
        b = t/(ncells1*ncells2);
        const int r = t % (ncells1*ncells2);
        tj = r/ncells1 - js; ti = r % ncells1 - is;
      }
      // the clouds were moved to the rank of their block before (MigrateClouds)
      if (b >= nmbp_) {
        Kokkos::printf("resampling: cloud p=%d tag=%llu x=%.17g y=%.17g PGID=%d anchor=%d leaf=%d "
                       "block=%d of %d (code %d, m=%.6e)\n", p,
                       static_cast<unsigned long long>(GetTag(pi,p)), xp, yp, pi(PGID,p), am,
                       t, b, nmbp_, code, pr(IPM,p));
        Kokkos::abort("resampling: a cloud in a block of another rank");
      }
      const int k = kblk(b);
      int q = poff(b) + (tj/k)*(bnx1/k) + ti/k;
      if (psub > 1) q = q*psub + (tj % k)*k + (ti % k);   // its cell in the patch
      pid(a,0) = q;
    }
    pid(a,0) = pid(a,0)*nsp + sp;   // one species per patch
    pid(a,1) = p;
    cptr(a) = coff(p);
    int e = coff(p);
    lt.ForEachCloudWeight(xp, yp, code, lt.Anchor(pi(PGID,p)), [&](const int target, const Real w) {
      ctgt(e) = target; cwgt(e) = w; ++e;
    });
  });
  // count the pool of every cell slot; per block patch and species one thread
  // splits the squares holding more than N clouds into quadrants (down to single cells) and maps
  // every slot to the lower-left slot of its final square (own) with the square's width there (nsz)
  const int nbase = poff_h[nmbp]*nsp;   // geometric patch x species
  const int nslot = nbase*psub;
  DvceArray1D<int> nsz("rs_split_size", (psub > 1) ? nslot : 1);
  int nsplit = 0;
  if (psub > 1) {
    DvceArray1D<int> pc("rs_split_count", nslot), own("rs_split_owner", nslot);
    Kokkos::deep_copy(own, -1);   // every slot that holds a cloud must be given a patch below
    Kokkos::parallel_for("rs_split_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int a) {Kokkos::atomic_increment(&pc(pid(a,0)));});
    const int ps = psub;
    Kokkos::parallel_reduce("rs_split_tree", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbase),
    KOKKOS_LAMBDA(const int i, int &ns) {
      const int q = i/nsp, sp = i % nsp;
      int lo = 0, hi = nmbp;   // the block of patch q
      while (hi - lo > 1) {
        const int mid = (lo + hi)/2;
        if (poff(mid) <= q) lo = mid; else hi = mid;
      }
      const int k = kblk(lo);
      auto idx = [&](const int x, const int y) {return (q*ps + y*k + x)*nsp + sp;};
      int st[64][3], top = 0;   // squares to look at: lower-left cell and width
      st[top][0] = 0; st[top][1] = 0; st[top][2] = k; ++top;
      while (top > 0) {
        --top;
        const int x0 = st[top][0], y0 = st[top][1], w = st[top][2];
        int tot = 0;
        for (int y=y0; y<y0+w; ++y) for (int x=x0; x<x0+w; ++x) tot += pc(idx(x, y));
        if (tot > psplit && w >= 2 && top + 4 <= 64) {
          const int h = w/2;
          if (w == k) ns += 1;
          for (int c=0; c<4; ++c) {
            st[top][0] = x0 + (c & 1)*h; st[top][1] = y0 + (c >> 1)*h; st[top][2] = h; ++top;
          }
        } else {
          for (int y=y0; y<y0+w; ++y) for (int x=x0; x<x0+w; ++x) own(idx(x, y)) = idx(x0, y0);
          nsz(idx(x0, y0)) = w;
        }
      }
    }, nsplit);
    Kokkos::parallel_for("rs_split_ids", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int a) {
      const int o = own(pid(a,0));
      if (o < 0) Kokkos::abort("resampling: a cell slot without a patch");
      pid(a,0) = o;
    });
  }
  Kokkos::fence();
  const double t_dev = seconds(t0);
  t0 = Clock::now();
  const int npatch = nslot;   // (geometric patch x cell slot) x species
  const double lx = transfer.x1max - transfer.x1min, ly = transfer.x2max - transfer.x2min;
  // per patch: centre and length scale of the centre-of-mass rows (half the patch width)
  DvceArray2D<Real> pgeo(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_pgeo"), npatch, 4);
  Kokkos::parallel_for("rs_pgeo", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch),
  KOKKOS_LAMBDA(const int qs) {
    const int g = qs/nsp, q = g/psub, c = g % psub;   // the geometric patch, cell slot
    int lo = 0, hi = nmbp;   // the block of patch q: poff(b) <= q < poff(b + 1)
    while (hi - lo > 1) {
      const int mid = (lo + hi)/2;
      if (poff(mid) <= q) lo = mid; else hi = mid;
    }
    const int b = lo, k = kblk(b), r = q - poff(b), nbx = bnx1/k, ip = r % nbx, jp = r/nbx;
    const Real d1 = lt.box(b,2), d2 = lt.box(b,3);   // = mb_size of local block b
    pgeo(qs,0) = lt.box(b,0) + (ip + 0.5)*k*d1;
    pgeo(qs,1) = lt.box(b,1) + (jp + 0.5)*k*d2;
    pgeo(qs,2) = 0.5*k*d1; pgeo(qs,3) = 0.5*k*d2;
    if (psub > 1 && nsz(qs) > 0) {   // a final square of w cells
      const int w = nsz(qs), x0 = c % k, y0 = c/k;
      pgeo(qs,0) = lt.box(b,0) + (ip*k + x0 + 0.5*w)*d1;
      pgeo(qs,1) = lt.box(b,1) + (jp*k + y0 + 0.5*w)*d2;
      pgeo(qs,2) = 0.5*w*d1; pgeo(qs,3) = 0.5*w*d2;
    }
  });
  // the patch solves on the host or on the device (cloud_resample_device)
  const bool on_dev = cloud_resample_device;
  const bool on_host = !cloud_resample_device;
  double t_copy = 0.0, t_sort = 0.0, t_host = 0.0, cpu = 0.0;
  int nthread = 0;
  std::vector<PatchInfo> infos(npatch);
  std::vector<int> order;
  std::vector<double> mnew;
  if (on_host) {
    auto pd_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pd);
    auto pgeo_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pgeo);
    auto pid_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pid);
    auto cptr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), cptr);
    auto ctgt_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ctgt);
    auto cwgt_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), cwgt);
    t_copy = seconds(t0);
    t0 = Clock::now();

    // host: clouds in patch order (counting sort), CSR in the same order
    std::vector<int> pptr(npatch + 1, 0);
    order.assign(npool, 0);
    for (int a=0; a<npool; ++a) ++pptr[pid_h(a,0) + 1];
    for (int q=0; q<npatch; ++q) pptr[q+1] += pptr[q];
    {
      std::vector<int> fill(pptr.begin(), pptr.end() - 1);
      for (int a=0; a<npool; ++a) order[fill[pid_h(a,0)]++] = a;
    }
    std::vector<double> spd(static_cast<std::size_t>(npool)*npd), swgt(nnz);
    mnew.assign(npool, 0.0);
    std::vector<int> sptr(npool + 1), stgt(nnz);
    sptr[0] = 0;
    for (int k=0; k<npool; ++k) {
      const int a = order[k];
      for (int c=0; c<npd; ++c) spd[static_cast<std::size_t>(k)*npd + c] = pd_h(a,c);
      int e = sptr[k];
      for (int j=cptr_h(a); j<cptr_h(a+1); ++j) {stgt[e] = ctgt_h(j); swgt[e] = cwgt_h(j); ++e;}
      sptr[k+1] = e;
    }
    t_sort = seconds(t0);
    t0 = Clock::now();
    std::atomic<int> next(0);
    nthread = std::max(1, std::min(cloud_resample_threads, npatch));
    std::vector<double> busy(nthread, 0.0);
    auto worker = [&](const int tid) {
      Work ws;
      auto tw = Clock::now();
      while (true) {
        const int q = next.fetch_add(1);
        if (q >= npatch) break;
        if (pptr[q+1] - pptr[q] < 1) continue;
        infos[q] = ResamplePatch(pptr[q], pptr[q+1], spd.data(), npd, sptr.data(), stgt.data(),
                                 swgt.data(), pgeo_h(q,0), pgeo_h(q,1), lx, ly, pgeo_h(q,2),
                                 pgeo_h(q,3), mnew.data(), ws);
      }
      busy[tid] = seconds(tw);
    };
    if (nthread == 1) {
      worker(0);
    } else {
      std::vector<std::thread> th;
      for (int t=0; t<nthread; ++t) th.emplace_back(worker, t);
      for (auto &t : th) t.join();
    }
    t_host = seconds(t0);
    for (double b : busy) cpu += b;
  }
  resample::PatchSummary dsum;       // device totals
  DvceArray1D<Real> mdev;
  double dtim[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // device: order, sizes, small, large, solve, middle
  if (on_dev) {
    auto td = Clock::now();
    // the pool in patch order on the device: the unique keys patch * npool + a, sorted (the
    // host's counting sort keeps the pool order inside a patch, so does this)
    DvceArray1D<std::int64_t> key(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_key"), npool);
    const std::int64_t npl = npool;
    Kokkos::parallel_for("rs_key", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int a) {key(a) = static_cast<std::int64_t>(pid(a,0))*npl + a;});
    Kokkos::sort(key);
    resample::DevicePatchProblem pb;
    pb.npool = npool; pb.npatch = npatch;
    if (DUST_R_DIAG && ProfOn()) pb.stage = DvceArray1D<unsigned long long>("rs_stage", resample::kNStage);
    pb.qr_rows = cloud_resample_qr_rows;
    pb.lx = lx; pb.ly = ly; pb.pgeo = pgeo;
    pb.pptr = DvceArray1D<int>("rs_pptr", npatch + 1);
    DvceArray1D<int> sorder(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_sorder"), npool);
    DvceArray1D<int> scnt(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_scnt"), npool);
    auto pptr_d = pb.pptr;
    Kokkos::parallel_for("rs_order", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int k) {
      const int a = static_cast<int>(key(k) % npl), q = static_cast<int>(key(k)/npl);
      sorder(k) = a;
      scnt(k) = cptr(a + 1) - cptr(a);
      Kokkos::atomic_inc(&pptr_d(q + 1));
    });
    Kokkos::parallel_scan("rs_pptr", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch + 1),
    KOKKOS_LAMBDA(const int q, int &upd, const bool final) {
      upd += pptr_d(q);
      if (final) pptr_d(q) = upd;
    });
    pb.sptr = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_sptr"), npool + 1);
    auto sptr_d = pb.sptr;
    Kokkos::parallel_scan("rs_sptr", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int k, int &upd, const bool final) {
      if (final) sptr_d(k) = upd;
      upd += scnt(k);
      if (final && k == npl - 1) sptr_d(k + 1) = upd;
    });
    pb.spd = DvceArray2D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_spd"), npool, npd);
    pb.stgt = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_stgt"), nnz);
    pb.swgt = DvceArray1D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_swgt"), nnz);
    auto spd_d = pb.spd; auto stgt_d = pb.stgt; auto swgt_d = pb.swgt;
    Kokkos::parallel_for("rs_sfill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int k) {
      const int a = sorder(k);
      for (int c=0; c<npd; ++c) spd_d(k,c) = pd(a,c);
      int e = sptr_d(k);
      for (int j=cptr(a); j<cptr(a + 1); ++j) {stgt_d(e) = ctgt(j); swgt_d(e) = cwgt(j); ++e;}
    });
    Kokkos::fence();
    dtim[0] = seconds(td);
    DvceArray1D<double> mn("rs_dev_mnew", npool);
    DvceArray2D<int> outi("rs_dev_outi", npatch, resample::NPC);
    DvceArray1D<double> outr("rs_dev_outr", npatch);
    double tim[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    resample::SolvePatchesDevice(pb, mn, outi, outr, cloud_resample_device_mem_mb,
                                 cloud_resample_device_mid, cloud_resample_device_mid_team,
                                 cloud_resample_device_wide_team, tim);
    dtim[1] = tim[0]; dtim[2] = tim[1]; dtim[3] = tim[2]; dtim[4] = tim[3]; dtim[5] = tim[4];
    if (pb.stage.extent(0) > 0) {   // stage clock (amr_profile)
      auto sh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pb.stage);
      for (int c=0; c<resample::kNStage; ++c) resample::StageTotals()[c] += sh(c);
    }
    mdev = DvceArray1D<Real>("rs_mnew_dev", npool);
    auto mdv = mdev;
    Kokkos::parallel_for("rs_unsort", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int k) {mdv(sorder(k)) = mn(k);});
    // the totals reduced on the device
    resample::SummarizePatchesDevice(outi, outr, dsum);
  }
  if (cloud_resample_device) {t_sort = dtim[0]; t_host = dtim[4];}
  t0 = Clock::now();
  long long pool = 0, kept = 0, atcap = 0, nfall = 0, nred = 0, npat = 0, zsteps = 0, csteps = 0, scans = 0;
  double resmax = 0.0;
  double st[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // stage seconds, all threads
  int nmax = 0, nbmax = 0, n64 = 0, n256 = 0;      // largest patch, its rows; patches with >= 64 / 256 clouds
  if (cloud_resample_device) {   // stage seconds are host-only (zero)
    npat = dsum.npat; pool = dsum.pool; kept = dsum.kept; atcap = dsum.atcap; nfall = dsum.nfall;
    nred = dsum.nred; zsteps = dsum.zsteps; csteps = dsum.csteps; scans = dsum.scans;
    resmax = dsum.resmax; nmax = dsum.nmax; nbmax = dsum.nbmax; n64 = dsum.n64; n256 = dsum.n256;
  }
  for (const auto &f : infos) {
    if (cloud_resample_device) break;
    if (f.n == 0) continue;
    st[0] += f.t_rows; st[1] += f.t_scale; st[2] += f.t_qr; st[3] += f.t_null; st[4] += f.t_red;
    st[5] += f.t_check;
    nmax = std::max(nmax, f.n); nbmax = std::max(nbmax, f.nb);
    if (f.n >= 64) ++n64;
    if (f.n >= 256) ++n256;
    ++npat; pool += f.n; kept += f.kept; atcap += f.ncap; zsteps += f.steps_zero; csteps += f.steps_cap;
    scans += f.scans;
    if (f.fallback) ++nfall; else resmax = std::max(resmax, f.res);
    if (f.kept < f.n) ++nred;
  }
  const int nremoved = static_cast<int>(pool - kept);

  // device: new masses, compaction
  if (nremoved > 0 || csteps > 0) {
    if (cloud_resample_device) {
      auto mdv = mdev;
      Kokkos::parallel_for("rs_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
      KOKKOS_LAMBDA(const int a) {pr(IPM, pid(a,1)) = mdv(a);});
    } else {
      Kokkos::View<double*, HostMemSpace> mh("rs_mnew_h", npool);
      for (int k=0; k<npool; ++k) mh(order[k]) = mnew[k];
      DvceArray1D<Real> md("rs_mnew", npool);
      Kokkos::deep_copy(md, mh);
      Kokkos::parallel_for("rs_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
      KOKKOS_LAMBDA(const int a) {pr(IPM, pid(a,1)) = md(a);});
    }
  }
  int nfinal = npart;
  if (nremoved > 0) {
    DvceArray1D<int> koff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs_koff"), npart);
    int nkeep = 0;
    Kokkos::parallel_scan("rs_kscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
      if (final) koff(p) = upd;
      upd += (pr(IPM,p) > 0.0) ? 1 : 0;
    }, nkeep);
    // 25% headroom, as the in-place split keeps it
    DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata,
                          nkeep + nkeep/4);
    DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata,
                         nkeep + nkeep/4);
    Kokkos::parallel_for("rs_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (pr(IPM,p) <= 0.0) return;
      const int k = koff(p);
      for (int c=0; c<nrdata; ++c) prn(c,k) = pr(c,p);
      for (int c=0; c<nidata; ++c) pin(c,k) = pi(c,p);
    });
    ppar->prtcl_rdata = prn;
    ppar->prtcl_idata = pin;
    ppar->nprtcl_thispack = nkeep;
    nfinal = nkeep;
    pmesh->nprtcl_thisrank = nfinal;
    pmesh->nprtcl_eachrank[global_variable::my_rank] = nfinal;
    pmesh->nprtcl_total = nfinal;
  }
  ++cloud_resample_pass;
  cloud_resample_removed += static_cast<unsigned long long>(nremoved);
  if (phst_track) phst_r_removed += static_cast<unsigned long long>(nremoved);
  return nremoved;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ResampleClouds3D
//! \brief ResampleClouds for 3-D refinable clouds, the device solve only.
//! The same pool (plain clouds below their cap eta m0 / 8^level, the cold gate, the local event
//! mask), patches of k x k x k cells inside one MeshBlock with k = min(s 2^level, kc), split
//! into octants down to single cells while a patch holds more than cloud_resample_patch_split
//! clouds, and the rows of every patch: leaf mass, momentum and feedback, the three
//! centre-of-mass rows and the six velocity second moments.  Returns the number of clouds
//! removed.

int DustGasDrag::ResampleClouds3D(const bool local) {
  using Clock = std::chrono::steady_clock;
  auto seconds = [](const Clock::time_point &a) {
    return std::chrono::duration<double>(Clock::now() - a).count();
  };
  using T = SMRCompositeTransfer;
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart = ppar->nprtcl_thispack;
  const auto lt = leaf_traversal3d;
  // (the gas gather field is packed below, once the pool is known)
  // the decision is collective and the pack has a forward halo exchange, so every rank takes
  // part, also one without particles (a rank that returned early made the others fail in the
  // MPI_Allreduce)
  auto gas_if_needed = [&](const int npool_mine) {
    int npool_max = npool_mine;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &npool_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (npool_max >= 2) PackGasGather3D(pmy_pack->phydro->u0);
  };
  if (cloud_resample_m0 <= 0.0) EnsureRootCloudMass();
  if (npart == 0) {gas_if_needed(0); return 0;}
  const int nsp = std::max(1, nspecies);
  DvceArray1D<Real> m0sp("rs3_m0", nsp);
  {
    auto mh = Kokkos::create_mirror_view(m0sp);
    bool any = false;
    for (int sp=0; sp<nsp; ++sp) {
      mh(sp) = (cloud_resample_m0 > 0.0) ? cloud_resample_m0
             : ((static_cast<int>(cloud_merge_mroot.size()) > sp) ? cloud_merge_mroot[sp] : 0.0);
      if (mh(sp) > 0.0) any = true;
    }
    if (!any) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "cloud_resample needs the root cloud mass: set <dust>/cloud_resample_m0"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    Kokkos::deep_copy(m0sp, mh);
  }
  // block patches of k x k x k cells
  const int s = cloud_resample_patch, kc = cloud_resample_patch_cells;
  const int nmbp = pmy_pack->nmb_thispack;
  const int bnx1 = lt.nx1, bnx2 = lt.nx2, bnx3 = lt.nx3;
  std::vector<int> kblk_h(nmbp), poff_h(nmbp + 1, 0);
  bool bad = (kc < 1);
  for (int b=0; b<nmbp && !bad; ++b) {
    const int lev = pmy_pack->pmb->mb_lev.h_view(b) - pmesh->root_level;
    int k = s;
    for (int l=0; l<lev && k < kc; ++l) k *= 2;
    k = std::min(k, kc);
    if (bnx1 % k != 0 || bnx2 % k != 0 || bnx3 % k != 0) bad = true;
    if (cloud_resample_patch_split > 0 && (k & (k - 1)) != 0) bad = true;
    kblk_h[b] = k;
    poff_h[b + 1] = poff_h[b] + (bnx1/k)*(bnx2/k)*(bnx3/k);
  }
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D block patches of min(cloud_resample_patch 2^level, cloud_resample_patch_cells) "
              << "cells must divide the MeshBlock (powers of two with the patch split)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  DvceArray1D<int> kblk("rs3_kblk", nmbp), poff("rs3_poff", nmbp + 1);
  {
    auto kb = Kokkos::create_mirror_view(kblk);
    auto po = Kokkos::create_mirror_view(poff);
    for (int b=0; b<nmbp; ++b) kb(b) = kblk_h[b];
    for (int b=0; b<=nmbp; ++b) po(b) = poff_h[b];
    Kokkos::deep_copy(kblk, kb);
    Kokkos::deep_copy(poff, po);
  }
  // local resampling: the blocks of the last regrid and the blocks touching them
  DvceArray1D<int> rmask("rs3_mask", std::max(1, nmbp));
  if (local) {
    auto pmr = pmesh->pmr;
    const int ntot = pmesh->nmb_total;
    const auto &ms = pmesh->mesh_size;
    const Real lx = ms.x1max - ms.x1min, ly = ms.x2max - ms.x2min, lz = ms.x3max - ms.x3min;
    auto box = [&](const int g, Real b[6]) {
      const auto &ll = pmesh->lloc_eachmb[g];
      const int lev = ll.level - pmesh->root_level;
      const Real w1 = lx/(pmesh->nmb_rootx1 << lev), w2 = ly/(pmesh->nmb_rootx2 << lev);
      const Real w3 = lz/(pmesh->nmb_rootx3 << lev);
      b[0] = ms.x1min + ll.lx1*w1; b[1] = b[0] + w1;
      b[2] = ms.x2min + ll.lx2*w2; b[3] = b[2] + w2;
      b[4] = ms.x3min + ll.lx3*w3; b[5] = b[4] + w3;
    };
    std::vector<std::array<Real, 6>> fresh;
    for (int g=0; g<ntot; ++g) {
      if (pmr->ncyc_since_ref(g) <= amr_event_max_age) {
        Real b[6]; box(g, b); fresh.push_back({b[0], b[1], b[2], b[3], b[4], b[5]});
      }
    }
    auto mh = Kokkos::create_mirror_view(rmask);
    for (int m=0; m<nmbp; ++m) {
      Real b[6]; box(pmy_pack->gids + m, b);
      const Real e1 = 1.0e-9*(b[1] - b[0]), e2 = 1.0e-9*(b[3] - b[2]), e3 = 1.0e-9*(b[5] - b[4]);
      int on = 0;
      for (const auto &f : fresh) {
        for (int oz=-1; oz<=1 && !on; ++oz) {
          for (int oy=-1; oy<=1 && !on; ++oy) {
            for (int ox=-1; ox<=1 && !on; ++ox) {
              const Real a0 = f[0] + ox*lx, a1 = f[1] + ox*lx, c0 = f[2] + oy*ly, c1 = f[3] + oy*ly;
              const Real d0 = f[4] + oz*lz, d1 = f[5] + oz*lz;
              if (a0 <= b[1] + e1 && b[0] <= a1 + e1 && c0 <= b[3] + e2 && b[2] <= c1 + e2 &&
                  d0 <= b[5] + e3 && b[4] <= d1 + e3) on = 1;
            }
          }
        }
        if (on) break;
      }
      mh(m) = on;
    }
    Kokkos::deep_copy(rmask, mh);
  }
  const bool use_mask = local;
  const int psplit = cloud_resample_patch_split;
  const int psub = (psplit > 0) ? kc*kc*kc : 1;
  const Real eta = cloud_resample_eta;
  auto t_all = Clock::now();
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata;
  (void) nrdata;
  const int is = lt.is, js = lt.js, ks = lt.ks;
  const int ncell123 = lt.ncells1*lt.ncells2*lt.ncells3;
  // the block patch (geometric) of a leaf
  auto patch_of = [=] KOKKOS_FUNCTION (const int t, int &b, int &q, int &cell) {
    int kk, jj, ii;
    lt.Decode(t, b, kk, jj, ii);
    const int ti = ii - is, tj = jj - js, tk = kk - ks;
    const int k = kblk(b), nbx = bnx1/k, nby = bnx2/k;
    q = poff(b) + ((tk/k)*nby + tj/k)*nbx + ti/k;
    cell = ((tk % k)*k + (tj % k))*k + (ti % k);
  };
  DvceArray1D<int> cnt(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_cnt"), npart);
  DvceArray1D<int> slot(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_slot"), npart);
  DvceArray1D<int> coff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_coff"), npart);
  int npool = 0, nnz = 0;
  double tot_b[2] = {0.0, 0.0}, cnt_b[3] = {0.0, 0.0, 0.0};
  // the cold gate: a block patch whose candidates span at most alpha H / tau in every
  // velocity component is left alone
  const Real calpha = cloud_resample_cold_alpha, comega = cloud_resample_omega;
  const int nqs = (calpha > 0.0) ? std::max(1, poff_h[nmbp]*nsp) : 1;
  DvceArray2D<Real> crange("rs3_cold_range", nqs, 6);
  DvceArray1D<int> ccold("rs3_cold", nqs);
  DvceArray1D<int> cq("rs3_cold_q", (calpha > 0.0) ? std::max(1, npart) : 1);
  double cnt_cold = 0.0;
  if (calpha > 0.0) {
    Kokkos::parallel_for("rs3_cold_init", Kokkos::RangePolicy<>(DevExeSpace(), 0, nqs),
    KOKKOS_LAMBDA(const int q) {
      for (int c=0; c<3; ++c) {crange(q,c) = 1.0e300; crange(q,3+c) = -1.0e300;}
      ccold(q) = 1;
    });
    Kokkos::parallel_for("rs3_cold_q", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      cq(p) = -1;
      if (pr(IPM,p) <= 0.0) return;
      const int code = pi(PLEV,p);
      if (T::CloudLobe1(code) || T::CloudLobe2(code) || T::CloudLobe3(code)) return;
      const int am = lt.Anchor(pi(PGID,p));
      if (use_mask && rmask(am) == 0) return;
      const int lev = T::CloudLevel(code);
      const int sp = (pi(PSP,p) >= 0 && pi(PSP,p) < nsp) ? pi(PSP,p) : 0;
      const Real cap = (eta > 0.0) ? eta*m0sp(sp)/static_cast<Real>(std::int64_t(1) << (3*lev))
                                   : 1.0e300;
      if (!(pr(IPM,p) <= cap*(1.0 + kPoolTol)) || pr(IPM,p) >= cap*(1.0 - kCapTol)) return;
      const int t = lt.LeafOwner(pr(IPX,p), pr(IPY,p), pr(IPZ,p), am);
      if (t < 0) return;
      int b, q, cell;
      patch_of(t, b, q, cell);
      if (b >= nmbp) return;
      const int qs = q*nsp + sp;
      cq(p) = qs;
      const Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
      for (int c=0; c<3; ++c) {
        Kokkos::atomic_min(&crange(qs,c), v[c]);
        Kokkos::atomic_max(&crange(qs,3+c), v[c]);
      }
    });
    Kokkos::parallel_for("rs3_cold_flag", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      const int qs = cq(p);
      if (qs < 0) return;
      const Real ts = pr(IPTS,p);
      const Real tau = (comega > 0.0) ? Kokkos::fmin(ts, 1.0/comega) : ts;
      const int b = lt.LeafOwner(pr(IPX,p), pr(IPY,p), pr(IPZ,p), lt.Anchor(pi(PGID,p)))/ncell123;
      const Real tolv = calpha*lt.box(b,3)/tau;
      for (int c=0; c<3; ++c) {
        if (crange(qs,3+c) - crange(qs,c) > tolv) {Kokkos::atomic_store(&ccold(qs), 0); return;}
      }
    });
  }
  const bool multirank = (global_variable::nranks > 1);
  Kokkos::parallel_reduce("rs3_flag", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, double &mt, double &px, double &nl, double &nf, double &nz,
                double &ncold) {
    cnt(p) = -1;
    if (pr(IPM,p) <= 0.0) {nz += 1.0; return;}
    mt += pr(IPM,p); px += pr(IPM,p)*pr(IPVX,p);
    const int code = pi(PLEV,p);
    const int lev = T::CloudLevel(code);
    if (T::CloudLobe1(code) || T::CloudLobe2(code) || T::CloudLobe3(code)) {nl += 1.0; return;}
    if (use_mask && rmask(lt.Anchor(pi(PGID,p))) == 0) return;
    const int sp = (pi(PSP,p) >= 0 && pi(PSP,p) < nsp) ? pi(PSP,p) : 0;
    const Real cap = (eta > 0.0) ? eta*m0sp(sp)/static_cast<Real>(std::int64_t(1) << (3*lev))
                                 : 1.0e300;
    if (!(pr(IPM,p) <= cap*(1.0 + kPoolTol))) {nf += 1.0; return;}
    if (pr(IPM,p) >= cap*(1.0 - kCapTol)) {nf += 1.0; return;}
    if (calpha > 0.0 && cq(p) >= 0 && ccold(cq(p)) == 1) {ncold += 1.0; return;}
    // several ranks (as in 2-D): a cloud whose leaf is on a block of another rank stays
    // out (a position within round-off of a periodic face routed to this rank's block while its
    // leaf lookup takes the image on the other side)
    if (multirank) {
      const int tl = lt.LeafOwner(pr(IPX,p), pr(IPY,p), pr(IPZ,p), lt.Anchor(pi(PGID,p)));
      if (tl < 0 || tl/ncell123 >= nmbp) return;
    }
    int c = 0;
    lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pr(IPZ,p), code, lt.Anchor(pi(PGID,p)),
                          [&](const int, const Real) {++c;});
    cnt(p) = c;
  }, tot_b[0], tot_b[1], cnt_b[0], cnt_b[1], cnt_b[2], cnt_cold);
  cloud_resample_cold_skips += static_cast<unsigned long long>(cnt_cold);
  Kokkos::parallel_scan("rs3_slot", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) slot(p) = (cnt(p) >= 0) ? upd : -1;
    upd += (cnt(p) >= 0) ? 1 : 0;
  }, npool);
  if (cloud_resample_tag_order && npool > 1) {   // the pool in PTAG order
    DvceArray1D<int> pofs(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_pofs"), npool);
    PoolInTagOrder(pi, slot, npart, npool, pofs);   // 64-bit tags
    Kokkos::parallel_scan("rs3_coff", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int a, int &upd, const bool final) {
      const int p = pofs(a);
      if (final) coff(p) = upd;
      upd += (cnt(p) > 0) ? cnt(p) : 0;
    }, nnz);
  } else {
    Kokkos::parallel_scan("rs3_coff", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
      if (final) coff(p) = upd;
      upd += (cnt(p) > 0) ? cnt(p) : 0;
    }, nnz);
  }
  // the gas gather field (u0 velocities, with a forward halo
  // exchange on several ranks) is only needed when some rank resamples; the decision is
  // collective because the halo exchange is.  u0 is unchanged since the call above used to be.
  gas_if_needed(npool);
  if (npool < 2) {
    ++cloud_resample_pass;
    return 0;
  }
  // the pool table (x, y, m, v, ts, gathered gas velocity, cap, z), ids, CSR weights
  const int npd = resample::NPD3;
  DvceArray2D<Real> pd(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_pd"), npool, npd);
  DvceArray2D<int> pid(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_pid"), npool, 2);
  DvceArray1D<int> cptr(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_cptr"), npool + 1);
  DvceArray1D<int> ctgt(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_ctgt"), nnz);
  DvceArray1D<Real> cwgt(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_cwgt"), nnz);
  auto gp = gpack3;   // PackGasGather3D above
  const int tmode = cloud_transition;
  const int nnz_ = nnz;
  Kokkos::parallel_for("rs3_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    const int a = slot(p);
    if (p == npart - 1) cptr(npool) = nnz_;
    if (a < 0) return;
    const Real xp = pr(IPX,p), yp = pr(IPY,p), zp = pr(IPZ,p);
    const int code = pi(PLEV,p);
    const int lev = T::CloudLevel(code);
    const int am = lt.Anchor(pi(PGID,p));
    Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
    lt.ForEachCloudMoment(xp, yp, zp, T::GatherCode(code, tmode), am,
    [&](const int target, const Real w, const Real m1x, const Real m1y, const Real m1z) {
      int tb, tk, tj, ti;
      lt.Decode(target, tb, tk, tj, ti);
      const Real dx1 = lt.box(tb,3), dx2 = lt.box(tb,4), dx3 = lt.box(tb,5);
      const Real xc = lt.box(tb,0) + (ti - is + 0.5)*dx1;
      const Real yc = lt.box(tb,1) + (tj - js + 0.5)*dx2;
      const Real zc = lt.box(tb,2) + (tk - ks + 0.5)*dx3;
      const Real cx = 0.5*(m1x + T::MinImage(xp - xc, lt.lx)*w)/dx1;
      const Real cy = 0.5*(m1y + T::MinImage(yp - yc, lt.ly)*w)/dx2;
      const Real cz = 0.5*(m1z + T::MinImage(zp - zc, lt.lz)*w)/dx3;
      ugx += w*gp(tb,tk,tj,ti,0) + cx*gp(tb,tk,tj,ti,3) + cy*gp(tb,tk,tj,ti,6) + cz*gp(tb,tk,tj,ti,9);
      ugy += w*gp(tb,tk,tj,ti,1) + cx*gp(tb,tk,tj,ti,4) + cy*gp(tb,tk,tj,ti,7) + cz*gp(tb,tk,tj,ti,10);
      ugz += w*gp(tb,tk,tj,ti,2) + cx*gp(tb,tk,tj,ti,5) + cy*gp(tb,tk,tj,ti,8) + cz*gp(tb,tk,tj,ti,11);
    });
    pd(a,resample::JX) = xp; pd(a,resample::JY) = yp; pd(a,resample::JZ) = zp;
    pd(a,resample::JM) = pr(IPM,p);
    pd(a,resample::JVX) = pr(IPVX,p); pd(a,resample::JVY) = pr(IPVY,p);
    pd(a,resample::JVZ) = pr(IPVZ,p); pd(a,resample::JTS) = pr(IPTS,p);
    pd(a,resample::JUX) = ugx; pd(a,resample::JUY) = ugy; pd(a,resample::JUZ) = ugz;
    const int sp = (pi(PSP,p) >= 0 && pi(PSP,p) < nsp) ? pi(PSP,p) : 0;
    pd(a,resample::JCAP) = (eta > 0.0) ? eta*m0sp(sp)/static_cast<Real>(std::int64_t(1) << (3*lev))
                                       : 1.0e300;
    const int t = lt.LeafOwner(xp, yp, zp, am);
    if (t < 0) Kokkos::abort("3-D resampling: a pool cloud without a leaf");
    int b, q, cell;
    patch_of(t, b, q, cell);
    if (b >= nmbp) Kokkos::abort("3-D resampling: a cloud in a block of another rank");
    if (psub > 1) q = q*psub + cell;   // its cell in the patch
    pid(a,0) = q*nsp + sp;
    pid(a,1) = p;
    cptr(a) = coff(p);
    int e = coff(p);
    lt.ForEachCloudWeight(xp, yp, zp, code, am, [&](const int target, const Real w) {
      ctgt(e) = target; cwgt(e) = w; ++e;
    });
  });
  // octant split of the crowded patches: every slot to the lowest-corner slot
  // of its final cube (own), with the cube's width there (nsz)
  const int nbase = poff_h[nmbp]*nsp;
  const int nslot = nbase*psub;
  DvceArray1D<int> nsz("rs3_split_size", (psub > 1) ? nslot : 1);
  int nsplit = 0;
  if (psub > 1) {
    DvceArray1D<int> pc("rs3_split_count", nslot), own("rs3_split_owner", nslot);
    Kokkos::deep_copy(own, -1);
    Kokkos::parallel_for("rs3_split_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int a) {Kokkos::atomic_increment(&pc(pid(a,0)));});
    const int ps = psub;
    Kokkos::parallel_reduce("rs3_split_tree", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbase),
    KOKKOS_LAMBDA(const int i, int &ns) {
      const int q = i/nsp, sp = i % nsp;
      int lo = 0, hi = nmbp;
      while (hi - lo > 1) {
        const int mid = (lo + hi)/2;
        if (poff(mid) <= q) lo = mid; else hi = mid;
      }
      const int k = kblk(lo);
      auto idx = [&](const int x, const int y, const int z) {
        return (q*ps + (z*k + y)*k + x)*nsp + sp;
      };
      int st[64][4], top = 0;
      st[top][0] = 0; st[top][1] = 0; st[top][2] = 0; st[top][3] = k; ++top;
      while (top > 0) {
        --top;
        const int x0 = st[top][0], y0 = st[top][1], z0 = st[top][2], w = st[top][3];
        int tot = 0;
        for (int z=z0; z<z0+w; ++z) {
          for (int y=y0; y<y0+w; ++y) for (int x=x0; x<x0+w; ++x) tot += pc(idx(x, y, z));
        }
        if (tot > psplit && w >= 2 && top + 8 <= 64) {
          const int h = w/2;
          if (w == k) ns += 1;
          for (int c=0; c<8; ++c) {
            st[top][0] = x0 + (c & 1)*h; st[top][1] = y0 + ((c >> 1) & 1)*h;
            st[top][2] = z0 + (c >> 2)*h; st[top][3] = h; ++top;
          }
        } else {
          for (int z=z0; z<z0+w; ++z) {
            for (int y=y0; y<y0+w; ++y) {
              for (int x=x0; x<x0+w; ++x) own(idx(x, y, z)) = idx(x0, y0, z0);
            }
          }
          nsz(idx(x0, y0, z0)) = w;
        }
      }
    }, nsplit);
    Kokkos::parallel_for("rs3_split_ids", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int a) {
      const int o = own(pid(a,0));
      if (o < 0) Kokkos::abort("3-D resampling: a cell slot without a patch");
      pid(a,0) = o;
    });
  }
  const int npatch = nslot;
  // per patch: centre and half widths of the centre-of-mass rows (x, y, half x, half y, z, half z)
  DvceArray2D<Real> pgeo(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_pgeo"), npatch, 6);
  Kokkos::parallel_for("rs3_pgeo", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch),
  KOKKOS_LAMBDA(const int qs) {
    const int g = qs/nsp, q = g/psub, c = g % psub;
    int lo = 0, hi = nmbp;
    while (hi - lo > 1) {
      const int mid = (lo + hi)/2;
      if (poff(mid) <= q) lo = mid; else hi = mid;
    }
    const int b = lo, k = kblk(b), r = q - poff(b), nbx = bnx1/k, nby = bnx2/k;
    const int ip = r % nbx, jp = (r/nbx) % nby, kp = r/(nbx*nby);
    const Real d1 = lt.box(b,3), d2 = lt.box(b,4), d3 = lt.box(b,5);
    Real x0 = 0.0, y0 = 0.0, z0 = 0.0, w = k;
    if (psub > 1 && nsz(qs) > 0) {
      w = nsz(qs); x0 = c % k; y0 = (c/k) % k; z0 = c/(k*k);
    }
    pgeo(qs,0) = lt.box(b,0) + (ip*k + x0 + 0.5*w)*d1;
    pgeo(qs,1) = lt.box(b,1) + (jp*k + y0 + 0.5*w)*d2;
    pgeo(qs,2) = 0.5*w*d1; pgeo(qs,3) = 0.5*w*d2;
    pgeo(qs,4) = lt.box(b,2) + (kp*k + z0 + 0.5*w)*d3;
    pgeo(qs,5) = 0.5*w*d3;
  });
  // the device solve, the pool in patch order
  DvceArray1D<std::int64_t> key(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_key"), npool);
  const std::int64_t npl = npool;
  Kokkos::parallel_for("rs3_key", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
  KOKKOS_LAMBDA(const int a) {key(a) = static_cast<std::int64_t>(pid(a,0))*npl + a;});
  Kokkos::sort(key);
  resample::DevicePatchProblem pb;
  pb.npool = npool; pb.npatch = npatch;
  pb.qr_rows = cloud_resample_qr_rows;
  pb.lx = lt.lx; pb.ly = lt.ly; pb.lz = lt.lz; pb.three = true; pb.pgeo = pgeo;
  pb.rank_screen = cloud_resample_rank_screen;
  pb.pptr = DvceArray1D<int>("rs3_pptr", npatch + 1);
  DvceArray1D<int> sorder(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_sorder"), npool);
  DvceArray1D<int> scnt(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_scnt"), npool);
  auto pptr_d = pb.pptr;
  Kokkos::parallel_for("rs3_order", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
  KOKKOS_LAMBDA(const int k) {
    const int a = static_cast<int>(key(k) % npl), q = static_cast<int>(key(k)/npl);
    sorder(k) = a;
    scnt(k) = cptr(a + 1) - cptr(a);
    Kokkos::atomic_inc(&pptr_d(q + 1));
  });
  Kokkos::parallel_scan("rs3_pptr", Kokkos::RangePolicy<>(DevExeSpace(), 0, npatch + 1),
  KOKKOS_LAMBDA(const int q, int &upd, const bool final) {
    upd += pptr_d(q);
    if (final) pptr_d(q) = upd;
  });
  pb.sptr = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_sptr"), npool + 1);
  auto sptr_d = pb.sptr;
  Kokkos::parallel_scan("rs3_sptr", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
  KOKKOS_LAMBDA(const int k, int &upd, const bool final) {
    if (final) sptr_d(k) = upd;
    upd += scnt(k);
    if (final && k == npl - 1) sptr_d(k + 1) = upd;
  });
  pb.spd = DvceArray2D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_spd"), npool, npd);
  pb.stgt = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_stgt"), nnz);
  pb.swgt = DvceArray1D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_swgt"), nnz);
  auto spd_d = pb.spd; auto stgt_d = pb.stgt; auto swgt_d = pb.swgt;
  Kokkos::parallel_for("rs3_sfill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
  KOKKOS_LAMBDA(const int k) {
    const int a = sorder(k);
    for (int c=0; c<npd; ++c) spd_d(k,c) = pd(a,c);
    int e = sptr_d(k);
    for (int j=cptr(a); j<cptr(a + 1); ++j) {stgt_d(e) = ctgt(j); swgt_d(e) = cwgt(j); ++e;}
  });
  DvceArray1D<double> mn("rs3_dev_mnew", npool);
  DvceArray2D<int> outi("rs3_dev_outi", npatch, resample::NPC);
  DvceArray1D<double> outr("rs3_dev_outr", npatch);
  double tim[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  long long scr[2] = {0, 0};
  resample::SolvePatchesDevice(pb, mn, outi, outr, cloud_resample_device_mem_mb,
                               cloud_resample_device_mid, cloud_resample_device_mid_team,
                               cloud_resample_device_wide_team, tim, scr);
  resample::PatchSummary dsum;
  resample::SummarizePatchesDevice(outi, outr, dsum);
  const int nremoved = static_cast<int>(dsum.pool - dsum.kept);
  if (nremoved > 0 || dsum.csteps > 0) {
    Kokkos::parallel_for("rs3_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int k) {pr(IPM, pid(sorder(k),1)) = mn(k);});
  }
  int nfinal = npart;
  if (nremoved > 0) {
    const int nrd = ppar->nrdata, nid = ppar->nidata;
    DvceArray1D<int> koff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rs3_koff"), npart);
    int nkeep = 0;
    Kokkos::parallel_scan("rs3_kscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
      if (final) koff(p) = upd;
      upd += (pr(IPM,p) > 0.0) ? 1 : 0;
    }, nkeep);
    DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrd,
                          nkeep + nkeep/4);
    DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nid,
                         nkeep + nkeep/4);
    Kokkos::parallel_for("rs3_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (pr(IPM,p) <= 0.0) return;
      const int k = koff(p);
      for (int c=0; c<nrd; ++c) prn(c,k) = pr(c,p);
      for (int c=0; c<nid; ++c) pin(c,k) = pi(c,p);
    });
    ppar->prtcl_rdata = prn;
    ppar->prtcl_idata = pin;
    ppar->nprtcl_thispack = nkeep;
    nfinal = nkeep;
    pmesh->nprtcl_thisrank = nfinal;
    pmesh->nprtcl_eachrank[global_variable::my_rank] = nfinal;
    pmesh->nprtcl_total = nfinal;
  }
  ++cloud_resample_pass;
  cloud_resample_removed += static_cast<unsigned long long>(nremoved);
  if (phst_track) phst_r_removed += static_cast<unsigned long long>(nremoved);
  (void) nsplit;
  return nremoved;
}

}  // namespace dust
