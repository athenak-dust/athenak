//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_resample.cpp
//! \brief Conservative same-level resampling of refinable clouds, in the
//! code instead of the offline reference (rs_lib.py / rs2_lib.py: Domain, rows, prune_cap).
//! Positions, velocities, shapes and stopping times are kept; only the masses of the plain
//! clouds change, a cloud whose mass reaches zero is removed.  Per patch of s x s root cells
//! the new masses keep, to round-off, every leaf's deposited mass and momentum, the three
//! components of the frozen-gas drag feedback of every leaf (AuditFeedback), the centre of
//! mass and the six velocity second moments of the patch; a mass never exceeds its cap
//! eta m0 / 4^level.  Clouds above the cap are left alone, and by default
//! (cloud_resample_skip_at_cap = true) so are the clouds already at the cap; with false they
//! stay in the pool and may only lose mass, as in the offline reference.
//!
//! Device: selection of the pool, production gather of the gas velocity, cloud weights as a
//! CSR list.  Host (std::thread over patches): constraint rows, null-space end-point
//! reduction with deterministic end points.  The reference takes its null vectors from an
//! SVD; here they come from a Householder QR with column pivoting, so the surviving set is
//! not the reference's cloud for cloud, while the same rows are kept.  Device: new masses,
//! compaction.  Single rank only, like the merge pass.
//!
//! By default (<dust>/cloud_resample_device = true) the pool is put in patch order
//! and every patch is solved on the device (dust_resample_device.cpp, same algorithm; false keeps
//! the host solves); with
//! cloud_resample_compare = true both run on the same pool and cloud_resample_compare.txt
//! records the differences (the device result is applied only with cloud_resample_device).
//!
//! Schedule: every cloud_resample_interval cycles (default), or with
//! cloud_resample_schedule = time at the planned times t0 + k dt; see ResampleDue.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
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
    for (int sg=0; sg<2; ++sg) {      // minus first, as the reference's argmax over (sign, column)
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
  // rows (rs_lib.rows, level DF): leaf mass, leaf momentum about the patch mean, centre of
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

// <dust>/cloud_resample_dump (diagnostic): the particle table of the pack as
// f64 rows [x1, x2, mass, v1, v2, v3, tag, level code], the layout of the particle rows of the
// test problems' snapshots.  File resample.CYCLE.WHEN.bin, WHEN = pre | post.
void DumpResampleState(particles::Particles *ppar, Mesh *pmesh, const char *when) {
  const int n = ppar->nprtcl_thispack;
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_idata);
  std::vector<double> rows;
  rows.reserve(static_cast<std::size_t>(8)*n);
  for (int p=0; p<n; ++p) {
    rows.insert(rows.end(), {pr(IPX,p), pr(IPY,p), pr(IPM,p), pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p),
                             static_cast<double>(pi(PTAG,p)), static_cast<double>(pi(PLEV,p))});
  }
  char name[64];
  std::snprintf(name, sizeof(name), "resample.%08d.%s.bin", pmesh->ncycle, when);
  std::FILE *file = std::fopen(name, "wb");
  bool ok = (file != nullptr) &&
            std::fprintf(file, "AthenaK resample dump v1 f64\n%.17g %d %d\n", pmesh->time, pmesh->ncycle, n) > 0;
  ok = ok && std::fwrite(rows.data(), sizeof(double), rows.size(), file) == rows.size();
  ok = (file != nullptr) && (std::fclose(file) == 0) && ok;
  if (!ok) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "cannot write " << name << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

// one row of cloud_resample_log.txt, with the header in front of the first; written for every
// pass, also when the pool is too small to resample
void AppendResampleLog(const char *row) {
  std::FILE *fp = std::fopen("cloud_resample_log.txt", "a");
  if (fp == nullptr) return;
  std::fseek(fp, 0, SEEK_END);
  if (std::ftell(fp) == 0) {
    std::fprintf(fp, "# ncycle time npart_before npart_after pool above_cap lobed patches"
                 " patches_reduced kept at_cap fallback max_res_accepted steps_zero steps_cap scans"
                 " M_b M_a PX_b PX_a t_total t_device t_copy t_sort t_host_wall t_host_cpu t_back"
                 " threads nnz s_rows s_scale s_qr s_null s_reduce s_check n_max nb_max n_ge64 n_ge256"
                 " device t_dev_order t_dev_sizes t_dev_small t_dev_large\n");
  }
  std::fputs(row, fp);
  std::fclose(fp);
}

}  // namespace

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
//! restart time when the file has no time-mode record.  Each event is logged in
//! cloud_resample_schedule.txt.

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
  const double tprev = cloud_resample_tprev;
  cloud_resample_tprev = t;
  const double tk = planned(cloud_resample_next);
  if (t < tk - eps(tk)) return false;
  // every planned time up to t (within the slack) is done with this one execution
  long long nxt = static_cast<long long>(std::floor((t - t0)/dt)) + 1;
  if (nxt < cloud_resample_next + 1) nxt = cloud_resample_next + 1;
  while (planned(nxt) <= t + eps(planned(nxt))) ++nxt;
  while (nxt > cloud_resample_next + 1 && planned(nxt - 1) > t + eps(planned(nxt - 1))) --nxt;
  const long long skipped = nxt - cloud_resample_next - 1;
  if (global_variable::my_rank == 0) {
    std::FILE *fp = std::fopen("cloud_resample_schedule.txt", "a");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# event planned_time ncycle time delay skipped time_prev_cycle dt_cycle"
                     " interval omega0\n");
      }
      std::fprintf(fp, "%lld %.17g %d %.17g %.6e %lld %.17g %.6e %.17g %.17g\n",
                   cloud_resample_next, tk, pmesh->ncycle, t, t - tk, skipped, tprev, pmesh->dt,
                   dt, static_cast<double>(cloud_resample_omega));
      std::fclose(fp);
    }
  }
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

int DustGasDrag::ResampleClouds() {
  using Clock = std::chrono::steady_clock;
  auto seconds = [](const Clock::time_point &a) {
    return std::chrono::duration<double>(Clock::now() - a).count();
  };
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart = ppar->nprtcl_thispack;
  const auto transfer = smr_transfer;
  const int nlev = transfer.cloud_nlev;
  if (npart == 0) return 0;
  const bool dump = std::find(cloud_resample_dump.begin(), cloud_resample_dump.end(), pmesh->ncycle) !=
                    cloud_resample_dump.end();
  if (dump) {
    DumpResampleState(ppar, pmesh, "pre");
    DumpResampleState(ppar, pmesh, "post");   // rewritten below if the resampling changes anything
  }
  Real m0 = cloud_resample_m0;
  if (m0 <= 0.0 && !cloud_merge_mroot.empty()) m0 = cloud_merge_mroot[0];
  if (m0 <= 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "cloud_resample needs the root cloud mass: set <dust>/cloud_resample_m0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int s = cloud_resample_patch;
  const int nroot1 = transfer.nfine1 >> nlev, nroot2 = transfer.nfine2 >> nlev;
  if (nroot1 % s != 0 || nroot2 % s != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "cloud_resample_patch must divide the root grid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int np1 = nroot1/s, np2 = nroot2/s;
  const Real eta = cloud_resample_eta;
  const bool skip_cap = cloud_resample_skip_at_cap;
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
  const Real root_dx1 = transfer.root_dx1, root_dx2 = transfer.root_dx2;
  // cnt: number of (leaf, weight) entries of a pool cloud, -1 for the others; totals before;
  // lobed clouds, plain clouds above the cap, zero-mass slots
  Kokkos::parallel_reduce("rs_flag", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, double &mt, double &px, double &nl, double &nf, double &nz) {
    cnt(p) = -1;
    if (pr(IPM,p) <= 0.0) {nz += 1.0; return;}
    mt += pr(IPM,p); px += pr(IPM,p)*pr(IPVX,p);
    const int code = pi(PLEV,p);
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) {nl += 1.0; return;}
    const Real cap = (eta > 0.0) ? eta*m0/static_cast<Real>(1 << (2*lev)) : 1.0e300;
    if (!(pr(IPM,p) <= cap*(1.0 + kPoolTol))) {nf += 1.0; return;}
    if (skip_cap && pr(IPM,p) >= cap*(1.0 - kCapTol)) {nf += 1.0; return;}   // at the cap: left alone
    int c = 0;
    transfer.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), code, [&](const int, const Real) {++c;});
    cnt(p) = c;
  }, tot_b[0], tot_b[1], cnt_b[0], cnt_b[1], cnt_b[2]);
  Kokkos::parallel_scan("rs_slot", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) slot(p) = (cnt(p) >= 0) ? upd : -1;
    upd += (cnt(p) >= 0) ? 1 : 0;
  }, npool);
  Kokkos::parallel_scan("rs_coff", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) coff(p) = upd;
    upd += (cnt(p) > 0) ? cnt(p) : 0;
  }, nnz);
  if (npool < 2) {   // nothing to resample; the pass keeps its log row (all patch columns 0)
    ++cloud_resample_pass;
    char row[1024];
    std::snprintf(row, sizeof(row), "%d %.17g %d %d %d %.0f %.0f 0 0 %d 0 0 0.000e+00 0 0 0"
                  " %.17g %.17g %.17g %.17g %.4e 0 0 0 0 0 0 0 %d 0 0 0 0 0 0 0 0 0 0 %d 0 0 0 0\n",
                  pmesh->ncycle, pmesh->time, npart, npart, npool, cnt_b[1], cnt_b[0], npool, tot_b[0],
                  tot_b[0], tot_b[1], tot_b[1], seconds(t_all), nnz, cloud_resample_device ? 1 : 0);
    AppendResampleLog(row);
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
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &uold = pmy_pack->phydro->u0;
  const int ncells1 = qdep.extent_int(4), ncells2 = qdep.extent_int(3);
  const int tmode = cloud_transition;
  const Real x1min = transfer.x1min, x2min = transfer.x2min;
  const int nnz_ = nnz;
  Kokkos::parallel_for("rs_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    const int a = slot(p);
    if (p == npart - 1) cptr(npool) = nnz_;
    if (a < 0) return;
    const Real xp = pr(IPX,p), yp = pr(IPY,p);
    const int code = pi(PLEV,p);
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
    transfer.ForEachCloudMoment(xp, yp, SMRCompositeTransfer::GatherCode(code, tmode),
    [&](const int target, const Real w, const Real m1x, const Real m1y) {
      const int ti = target % ncells1;
      const int remainder = target/ncells1;
      const int tj = remainder % ncells2;
      const int tm = remainder/ncells2;
      const Real dx1 = mbsize.d_view(tm).dx1, dx2 = mbsize.d_view(tm).dx2;
      const Real xc = mbsize.d_view(tm).x1min + (ti - is + 0.5)*dx1;
      const Real yc = mbsize.d_view(tm).x2min + (tj - js + 0.5)*dx2;
      const Real cx = 0.5*(m1x + transfer.MinImage1(xp - xc)*w)/dx1;
      const Real cy = 0.5*(m1y + transfer.MinImage2(yp - yc)*w)/dx2;
      const Real invrho = 1.0/uold(tm,IDN,ks,tj,ti);
      const Real rip = 1.0/uold(tm,IDN,ks,tj,ti+1), rim = 1.0/uold(tm,IDN,ks,tj,ti-1);
      const Real rjp = 1.0/uold(tm,IDN,ks,tj+1,ti), rjm = 1.0/uold(tm,IDN,ks,tj-1,ti);
      ugx += w*uold(tm,IM1,ks,tj,ti)*invrho
           + cx*(uold(tm,IM1,ks,tj,ti+1)*rip - uold(tm,IM1,ks,tj,ti-1)*rim)
           + cy*(uold(tm,IM1,ks,tj+1,ti)*rjp - uold(tm,IM1,ks,tj-1,ti)*rjm);
      ugy += w*uold(tm,IM2,ks,tj,ti)*invrho
           + cx*(uold(tm,IM2,ks,tj,ti+1)*rip - uold(tm,IM2,ks,tj,ti-1)*rim)
           + cy*(uold(tm,IM2,ks,tj+1,ti)*rjp - uold(tm,IM2,ks,tj-1,ti)*rjm);
      ugz += w*uold(tm,IM3,ks,tj,ti)*invrho
           + cx*(uold(tm,IM3,ks,tj,ti+1)*rip - uold(tm,IM3,ks,tj,ti-1)*rim)
           + cy*(uold(tm,IM3,ks,tj+1,ti)*rjp - uold(tm,IM3,ks,tj-1,ti)*rjm);
    });
    pd(a,0) = xp; pd(a,1) = yp; pd(a,2) = pr(IPM,p);
    pd(a,3) = pr(IPVX,p); pd(a,4) = pr(IPVY,p); pd(a,5) = pr(IPVZ,p); pd(a,6) = pr(IPTS,p);
    pd(a,7) = ugx; pd(a,8) = ugy; pd(a,9) = ugz;
    pd(a,10) = (eta > 0.0) ? eta*m0/static_cast<Real>(1 << (2*lev)) : 1.0e300;
    const Real xw = SMRCompositeTransfer::WrapPosition(xp, transfer.x1min, transfer.x1max);
    const Real yw = SMRCompositeTransfer::WrapPosition(yp, transfer.x2min, transfer.x2max);
    int ix = static_cast<int>(floor((xw - x1min)/root_dx1)), iy = static_cast<int>(floor((yw - x2min)/root_dx2));
    ix = (ix < 0) ? 0 : ((ix >= nroot1) ? nroot1 - 1 : ix);
    iy = (iy < 0) ? 0 : ((iy >= nroot2) ? nroot2 - 1 : iy);
    pid(a,0) = (iy/s)*np1 + ix/s; pid(a,1) = p;
    cptr(a) = coff(p);
    int e = coff(p);
    transfer.ForEachCloudWeight(xp, yp, code, [&](const int target, const Real w) {
      ctgt(e) = target; cwgt(e) = w; ++e;
    });
  });
  Kokkos::fence();
  const double t_dev = seconds(t0);
  t0 = Clock::now();
  const int npatch = np1*np2;
  const double lx = transfer.x1max - transfer.x1min, ly = transfer.x2max - transfer.x2min;
  // The patch solves on the host (default), on the device, or both on the same
  // pool (cloud_resample_compare: the differences are logged, the device result is used)
  const bool on_dev = cloud_resample_device || cloud_resample_compare;
  const bool on_host = !cloud_resample_device || cloud_resample_compare;
  double t_copy = 0.0, t_sort = 0.0, t_host = 0.0, cpu = 0.0;
  int nthread = 0;
  std::vector<PatchInfo> infos(npatch);
  std::vector<int> order;
  std::vector<double> mnew;
  Kokkos::View<Real**, Kokkos::LayoutRight, HostMemSpace> pdc;   // pool table (comparison)
  Kokkos::View<int**, Kokkos::LayoutRight, HostMemSpace> pidc;
  if (on_host) {
    auto pd_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pd);
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
        const double xc = transfer.x1min + (q % np1 + 0.5)*s*root_dx1;
        const double yc = transfer.x2min + (q/np1 + 0.5)*s*root_dx2;
        infos[q] = ResamplePatch(pptr[q], pptr[q+1], spd.data(), npd, sptr.data(), stgt.data(),
                                 swgt.data(), xc, yc, lx, ly, root_dx1, root_dx2, mnew.data(), ws);
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
    if (cloud_resample_compare) {
      pdc = Kokkos::View<Real**, Kokkos::LayoutRight, HostMemSpace>("rs_pdc", npool, npd);
      pidc = Kokkos::View<int**, Kokkos::LayoutRight, HostMemSpace>("rs_pidc", npool, 2);
      for (int a=0; a<npool; ++a) {
        for (int c=0; c<npd; ++c) pdc(a,c) = pd_h(a,c);
        pidc(a,0) = pid_h(a,0); pidc(a,1) = pid_h(a,1);
      }
    }
  }
  std::vector<PatchInfo> dinfos;
  DvceArray1D<Real> mdev;
  double dtim[5] = {0.0, 0.0, 0.0, 0.0, 0.0};   // device: order, sizes, small, large, solve
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
    pb.npool = npool; pb.npatch = npatch; pb.np1 = np1; pb.s = s;
    pb.x1min = transfer.x1min; pb.x2min = transfer.x2min; pb.lx = lx; pb.ly = ly;
    pb.rdx = root_dx1; pb.rdy = root_dx2;
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
    double tim[4] = {0.0, 0.0, 0.0, 0.0};
    resample::SolvePatchesDevice(pb, mn, outi, outr, cloud_resample_device_mem_mb, tim);
    dtim[1] = tim[0]; dtim[2] = tim[1]; dtim[3] = tim[2]; dtim[4] = tim[3];
    mdev = DvceArray1D<Real>("rs_mnew_dev", npool);
    auto mdv = mdev;
    Kokkos::parallel_for("rs_unsort", Kokkos::RangePolicy<>(DevExeSpace(), 0, npool),
    KOKKOS_LAMBDA(const int k) {mdv(sorder(k)) = mn(k);});
    auto oi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), outi);
    auto orr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), outr);
    dinfos.resize(npatch);
    for (int q=0; q<npatch; ++q) {
      PatchInfo f;
      f.n = oi(q, resample::PN); f.kept = oi(q, resample::PKEPT); f.ncap = oi(q, resample::PNCAP);
      f.fallback = oi(q, resample::PFALLBACK); f.steps_zero = oi(q, resample::PZERO);
      f.steps_cap = oi(q, resample::PCAPSTEP); f.scans = oi(q, resample::PSCANS);
      f.nrow = oi(q, resample::PNROW); f.nb = oi(q, resample::PNB); f.res = orr(q);
      dinfos[q] = f;
    }
  }
  if (cloud_resample_compare) {
    // the same pool solved on the host and on the device: per cloud, per patch
    auto md_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), mdev);
    std::vector<double> mh(npool);
    for (int k=0; k<npool; ++k) mh[order[k]] = mnew[k];
    long long rem_h = 0, rem_d = 0, kdiff = 0;
    double dmax = 0.0;
    std::vector<char> pk(npatch, 0), pm(npatch, 0);
    for (int a=0; a<npool; ++a) {
      const double m0a = pdc(a, resample::JM);
      if (!(mh[a] > 0.0)) ++rem_h;
      if (!(md_h(a) > 0.0)) ++rem_d;
      const int q = pidc(a, 0);
      if ((mh[a] > 0.0) != (md_h(a) > 0.0)) {++kdiff; pk[q] = 1;}
      const double rel = std::fabs(mh[a] - md_h(a))/m0a;
      dmax = std::max(dmax, rel);
      if (rel > 1.0e-12) pm[q] = 1;
    }
    long long fh = 0, fd = 0, npk = 0, npm = 0, nzh = 0, nzd = 0, nch = 0, ncd = 0;
    for (int q=0; q<npatch; ++q) {
      npk += pk[q]; npm += pm[q];
      fh += (infos[q].fallback != 0); fd += (dinfos[q].fallback != 0);
      nzh += infos[q].steps_zero; nzd += dinfos[q].steps_zero;
      nch += infos[q].steps_cap; ncd += dinfos[q].steps_cap;
    }
    if (global_variable::my_rank == 0) {
      std::FILE *fc = std::fopen("cloud_resample_compare.txt", "a");
      if (fc != nullptr) {
        std::fseek(fc, 0, SEEK_END);
        if (std::ftell(fc) == 0) {
          std::fprintf(fc, "# ncycle time pool removed_host removed_device clouds_kept_differ "
                       "patches_kept_differ patches_mass_differ(>1e-12) max_rel_mass_diff "
                       "fallback_host fallback_device steps_zero_host steps_zero_device "
                       "steps_cap_host steps_cap_device t_host_wall t_device_solve\n");
        }
        std::fprintf(fc, "%d %.17g %d %lld %lld %lld %lld %lld %.3e %lld %lld %lld %lld %lld %lld %.4e %.4e\n",
                     pmesh->ncycle, pmesh->time, npool, rem_h, rem_d, kdiff, npk, npm, dmax, fh, fd,
                     nzh, nzd, nch, ncd, t_host, dtim[4]);
        std::fclose(fc);
      }
    }
  }
  if (cloud_resample_device) {t_sort = dtim[0]; t_host = dtim[4];}
  t0 = Clock::now();
  long long pool = 0, kept = 0, atcap = 0, nfall = 0, nred = 0, npat = 0, zsteps = 0, csteps = 0, scans = 0;
  double resmax = 0.0;
  double st[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // stage seconds, all threads
  int nmax = 0, nbmax = 0, n64 = 0, n256 = 0;      // largest patch, its rows; patches with >= 64 / 256 clouds
  for (const auto &f : (cloud_resample_device ? dinfos : infos)) {
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
  double tot_a[2] = {0.0, 0.0};
  {
    auto pra = ppar->prtcl_rdata;
    Kokkos::parallel_reduce("rs_tot", Kokkos::RangePolicy<>(DevExeSpace(), 0, nfinal),
    KOKKOS_LAMBDA(const int p, double &mt, double &px) {
      mt += pra(IPM,p); px += pra(IPM,p)*pra(IPVX,p);
    }, tot_a[0], tot_a[1]);
  }
  Kokkos::fence();
  const double t_back = seconds(t0);
  ++cloud_resample_pass;
  cloud_resample_removed += static_cast<unsigned long long>(nremoved);
  {
    char row[1024];
    std::snprintf(row, sizeof(row), "%d %.17g %d %d %lld %.0f %.0f %lld %lld %lld %lld %lld %.3e %lld %lld"
                  " %lld %.17g %.17g %.17g %.17g %.4e %.4e %.4e %.4e %.4e %.4e %.4e %d %d"
                  " %.4e %.4e %.4e %.4e %.4e %.4e %d %d %d %d %d %.4e %.4e %.4e %.4e\n", pmesh->ncycle,
                  pmesh->time, npart, nfinal, pool, cnt_b[1], cnt_b[0], npat, nred, kept, atcap, nfall,
                  resmax, zsteps, csteps, scans, tot_b[0], tot_a[0], tot_b[1], tot_a[1], seconds(t_all),
                  t_dev, t_copy, t_sort, t_host, cpu, t_back, nthread, nnz, st[0], st[1], st[2],
                  st[3], st[4], st[5], nmax, nbmax, n64, n256, cloud_resample_device ? 1 : 0,
                  dtim[0], dtim[1], dtim[2], dtim[3]);
    AppendResampleLog(row);
  }
  if (dump) DumpResampleState(ppar, pmesh, "post");
  return nremoved;
}

}  // namespace dust
