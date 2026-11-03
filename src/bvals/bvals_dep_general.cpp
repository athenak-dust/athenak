//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_dep_general.cpp
//! \brief The additive ghost-deposit exchange of a 3-D mesh with
//! any levels, several MPI ranks and shear-periodic x1 faces (DepRemapPlan, and the general path
//! of MeshBoundaryValuesDep).
//!
//! A particle near a block boundary deposits into ghost cells of its own block; that deposit
//! belongs to the active cells of the blocks the ghost cell overlaps.  Here every ghost cell is
//! given to the active cells it overlaps by volume, in the integer cell units of the finest
//! level R (a cell of level L is 2^(R-L) units wide): the overlaps of the static part are exact
//! integers and the remap conserves the deposit exactly, at any level difference.  Ghost cells
//! beyond a shear-periodic x1 face are moved to the opposite face and shifted in x2 by the shear
//! offset of the particles (bvals_part.cpp: crossing the outer face, x -= Lx and y += ysh), and
//! overlap the target cells there by real fractions (piecewise-constant remap; conservative).
//!
//! The lists are built on the host from the global block list (lloc_eachmb, rank_eachmb; no
//! finest-grid table): for every target cell of this rank the contributing source cells, sorted
//! by (source gid, source cell), so that the sums do not depend on how the blocks are spread over
//! the ranks; and per peer rank the source cells sent there.  Both sides enumerate the pairs of
//! blocks with the same function (the target's neighbours or the source's neighbours; the
//! adjacency is symmetric) and every message size is checked once per build (MPI_Alltoall).
//! Every source ghost cell must be covered exactly once (checked).
//!
//! The shear part changes its overlap structure only when the offset crosses a cell (UpdateShear);
//! in between only its weights change.  The weights are computed on the device from recipes kept
//! there (ATHENA_DEP_WEIGHTS_HOST=1: on the host and uploaded, as before), and the structure of the
//! next one-cell offset is built ahead on a host thread from a snapshot of the blocks
//! (ATHENA_DEP_PREFETCH_OLD=1: a full build at every crossing, as before).  Both give the lists
//! and weights of a full build bit for bit (ATHENA_DEP_VERIFY=1 checks it in the run).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

namespace {

// Capacity of a general deposit-exchange buffer for need values: a quarter above the new
// high-water mark, computed in 64 bits and clamped to the int range that the exchange indexes
// with (never below need)
std::size_t DepBufferCapacity(const std::size_t need) {
  const std::size_t lim = static_cast<std::size_t>(std::numeric_limits<int>::max());
  return std::max(need, std::min(need + need/4, lim));
}

struct BlkR {
  int gid = -1, rank = -1, level = 0;
  std::int64_t lo[3] = {0, 0, 0};   // active box in units of the finest level
  std::int64_t sc = 1;              // cell width in those units
  bool face[2] = {false, false};    // at the inner / outer x1 face
};

struct Geo {
  int R = 0, root = 0;
  int nx[3] = {1, 1, 1}, ng = 0, nc[3] = {1, 1, 1};
  std::int64_t N[3] = {1, 1, 1};    // the domain in units of the finest level
  int nmbroot[3] = {1, 1, 1};
  bool shear = false;
  int x3mode = 0;     // 0 periodic x3, 1 fold at the physical x3 faces, 2 drop
};

BlkR MakeBlk(Mesh *pm, const Geo &g, const int gid) {
  BlkR b;
  const LogicalLocation &ll = pm->lloc_eachmb[gid];
  b.gid = gid;
  b.rank = pm->rank_eachmb[gid];
  b.level = ll.level;
  b.sc = static_cast<std::int64_t>(1) << (g.R - ll.level);
  const std::int64_t lx[3] = {ll.lx1, ll.lx2, ll.lx3};
  for (int d=0; d<3; ++d) b.lo[d] = lx[d]*g.nx[d]*b.sc;
  const std::int64_t nmb1 = static_cast<std::int64_t>(g.nmbroot[0]) << (ll.level - g.root);
  b.face[0] = (ll.lx1 == 0);
  b.face[1] = (ll.lx1 == nmb1 - 1);
  return b;
}

// one overlap of a source interval with a target cell along one direction
// (for the shifted x2 overlaps also what o was computed from: o = min(yb + sh, chi) - max(ya + sh,
// clo); the shear part recomputes its weights from these)
struct Ov {int t; double o; double ya = 0.0, yb = 0.0, clo = 0.0, chi = 0.0;};

// The overlaps along direction d of source index q (array index, ghosts included) of block s with
// the active cells of block t.  mode: 0 ordinary periodic wrap; 1 the shear-mapped x1 index (x
// moved to the opposite face); 2 the x2 index shifted by sh (real, periodic images).  Returns
// whether the source index is valid for the requested part (x1 only).
void Overlaps(const Geo &g, const BlkR &s, const BlkR &t, const int d, const int q,
              const int mode, const double sh, std::vector<Ov> &out) {
  out.clear();
  const std::int64_t a0 = s.lo[d] + (q - g.ng)*s.sc;
  const std::int64_t b0 = a0 + s.sc;
  const std::int64_t tlo = t.lo[d], thi = t.lo[d] + g.nx[d]*t.sc;
  const std::int64_t N = g.N[d];
  if (mode == 2) {   // shifted x2: real interval, images -N, 0, +N of the target
    const double a = static_cast<double>(a0) + sh, b = static_cast<double>(b0) + sh;
    for (int im=-2; im<=2; ++im) {
      const double lo = static_cast<double>(tlo + im*N), hi = static_cast<double>(thi + im*N);
      if (b <= lo || a >= hi) continue;
      const double st = static_cast<double>(t.sc);
      int c0 = static_cast<int>(std::floor((std::max(a, lo) - lo)/st));
      int c1 = static_cast<int>(std::ceil((std::min(b, hi) - lo)/st)) - 1;
      c0 = std::max(c0, 0); c1 = std::min(c1, g.nx[d] - 1);
      for (int c=c0; c<=c1; ++c) {
        const double clo = lo + c*st, chi = clo + st;
        const double o = std::min(b, chi) - std::max(a, clo);
        if (o > 0.0) out.push_back({c, o, static_cast<double>(a0), static_cast<double>(b0), clo, chi});
      }
    }
    return;
  }
  std::int64_t a = a0, b = b0;
  if (mode == 1) {           // x1 beyond a shear face: the opposite face
    if (a < 0) {a += N; b += N;} else {a -= N; b -= N;}
  } else if (d == 2 && g.x3mode != 0) {   // a physical x3 face
    if (a < 0 || a >= N) {
      if (g.x3mode == 2) return;          // dropped
      if (a < 0) {const std::int64_t t = a; a = -b; b = -t;}           // mirror about 0
      else {const std::int64_t t = a; a = 2*N - b; b = 2*N - t;}       // mirror about N
    }
  } else {                   // ordinary periodic wrap (one image: the box is inside [0,N))
    if (a < 0) {a += N; b += N;} else if (a >= N) {a -= N; b -= N;}
  }
  if (b <= tlo || a >= thi) return;
  const std::int64_t lo = std::max(a, tlo), hi = std::min(b, thi);
  const std::int64_t c0 = (lo - tlo)/t.sc, c1 = (hi - 1 - tlo)/t.sc;
  for (std::int64_t c=c0; c<=c1; ++c) {
    const std::int64_t clo = tlo + c*t.sc, chi = clo + t.sc;
    const std::int64_t o = std::min(b, chi) - std::max(a, clo);
    if (o > 0) out.push_back({static_cast<int>(c), static_cast<double>(o)});
  }
}

// (rc: the recipe of w for the shear part, see WeightOf)
struct WRecipe {double ox, oz, ya, yb, clo, chi, tvol; bool neg;};
struct PairEnt {int scell; int tcell; double w; double vol; WRecipe rc;};

// whether the structure of the shear part at offset shR may be cached or reused: the fractional
// part away from 0 and 1 by more than 1e-9 cells, so that every overlap of the structure is at
// least that and none can round to zero (an integer or nearly integer offset: always a full build)
inline bool ShearCacheable(const double shR) {
  const double f = shR - std::floor(shR);
  return f > 1.0e-9 && f < 1.0 - 1.0e-9;
}

// the weight of a shear entry for the offset shR, with the arithmetic of Overlaps and PairOverlap
inline double WeightOf(const WRecipe &r, const double shR) {
  const double sh = r.neg ? -shR : shR;   // (a negation, as the caller of Overlaps: no product to fuse)
  const double a = r.ya + sh, b = r.yb + sh;
  const double oy = std::min(b, r.chi) - std::max(a, r.clo);
  const double vol = r.ox*oy*r.oz;
  return vol/r.tvol;
}

// All overlaps of the ghost cells of s with the active cells of t: part 0 the ordinary ghost
// cells (not beyond a shear face), part 1 the ghost cells beyond a shear-periodic x1 face
// (shifted in x2 by +shR beyond the outer face, -shR beyond the inner face)
void PairOverlap(const Geo &g, const BlkR &s, const BlkR &t, const int part, const double shR,
                 std::vector<PairEnt> &out) {
  out.clear();
  const int nc1 = g.nc[0], nc2 = g.nc[1], nc3 = g.nc[2], ng = g.ng;
  const double tvol = static_cast<double>(t.sc)*static_cast<double>(t.sc)*
                      static_cast<double>(t.sc);
  std::vector<std::vector<Ov>> l1(nc1), l2p(nc2), l2m(nc2), l3(nc3);
  std::vector<int> side(nc1, -1);     // -1 ordinary, 0 beyond the inner face, 1 beyond the outer
  std::vector<Ov> tmp;
  for (int i=0; i<nc1; ++i) {
    const std::int64_t a0 = s.lo[0] + (i - ng)*s.sc;
    if (g.shear && a0 < 0) side[i] = 0;
    if (g.shear && a0 >= g.N[0]) side[i] = 1;
    if ((part == 0) != (side[i] < 0)) continue;
    Overlaps(g, s, t, 0, i, (part == 0) ? 0 : 1, 0.0, l1[i]);
  }
  for (int k=0; k<nc3; ++k) Overlaps(g, s, t, 2, k, 0, 0.0, l3[k]);
  if (part == 0) {
    for (int j=0; j<nc2; ++j) Overlaps(g, s, t, 1, j, 0, 0.0, l2p[j]);
  } else {
    for (int j=0; j<nc2; ++j) {
      Overlaps(g, s, t, 1, j, 2, shR, l2p[j]);    // beyond the outer face: y + ysh
      Overlaps(g, s, t, 1, j, 2, -shR, l2m[j]);   // beyond the inner face: y - ysh
    }
  }
  // only the x1 indices with overlaps, in increasing order (the shear
  // part has 2 nghost of them): the same entries in the same order as a loop over every i
  std::vector<int> ilist;
  for (int i=0; i<nc1; ++i) if (!l1[i].empty()) ilist.push_back(i);
  if (ilist.empty()) return;
  for (int k=0; k<nc3; ++k) {
    if (l3[k].empty()) continue;
    const bool kin = (k >= ng && k < nc3 - ng);
    for (int j=0; j<nc2; ++j) {
      const bool jin = (j >= ng && j < nc2 - ng);
      for (const int i : ilist) {
        const bool iin = (i >= ng && i < nc1 - ng);
        if (iin && jin && kin) continue;           // active cell: not a deposit to move
        const std::vector<Ov> &ly = (part == 1 && side[i] == 0) ? l2m[j] : l2p[j];
        const bool neg = (part == 1 && side[i] == 0);
        if (ly.empty()) continue;
        const int scell = (k*nc2 + j)*nc1 + i;
        for (const Ov &oz : l3[k]) {
          for (const Ov &oy : ly) {
            for (const Ov &ox : l1[i]) {
              const double vol = ox.o*oy.o*oz.o;
              const int tcell = ((oz.t + ng)*nc2 + (oy.t + ng))*nc1 + (ox.t + ng);
              out.push_back({scell, tcell, vol/tvol, vol,
                             {ox.o, oz.o, oy.ya, oy.yb, oy.clo, oy.chi, tvol, neg}});
            }
          }
        }
      }
    }
  }
}

// candidate partners of block b (local index m): itself and its neighbours (part 0), or the
// blocks of the opposite face(s) (part 1)
std::vector<int> Candidates(MeshBlockPack *pp, const BlkR &b, const int m, const int part,
                            const std::vector<int> *face) {
  std::set<int> c;
  if (part == 0) {
    c.insert(b.gid);
    auto &nghbr = pp->pmb->nghbr;
    for (int n=0; n<pp->pmb->nnghbr; ++n) {
      const int gid = nghbr.h_view(m,n).gid;
      if (gid >= 0) c.insert(gid);
    }
  } else {
    if (b.face[1]) c.insert(face[0].begin(), face[0].end());
    if (b.face[0]) c.insert(face[1].begin(), face[1].end());
  }
  return std::vector<int>(c.begin(), c.end());
}

// ATHENA_DEP_PROFILE=1 in the environment prints, on rank 0 every 400
// shear updates, the mean host time per build of its phases (diagnostic only); also the
// prefetch installs and the ATHENA_DEP_VERIFY counts (main thread only)
struct DepProfile {
  bool on = false;
  int nbuild = 0, ncall = 0, nupd = 0;
  double tupd = 0.0;
  double t[7] = {0, 0, 0, 0, 0, 0, 0};   // recv pairs, send pairs, check, layout+Alltoall, sort, device, total
  int ninst = 0, ndisc = 0;              // prefetches installed, discarded
  double twait = 0.0, tinst = 0.0, tinst_a2a = 0.0, tinst_w = 0.0, tthread = 0.0;   // seconds
  long nmis = 0;                         // ATHENA_DEP_VERIFY mismatches
  int nver_w = 0, nver_i = 0;            // verified weight updates, installs
  DepProfile() {const char *e = std::getenv("ATHENA_DEP_PROFILE"); on = (e != nullptr && e[0] == '1');}
};
DepProfile &DepProf() {static DepProfile p; return p;}

// the device arrays of a map keep their capacity between builds (the
// users read the counts nseg / nent / nsend / ndrop, not the extents); grown by 1/4 when too small.
// (std::size_t lengths: the shear recipes are kRecipe = 8 doubles per entry, 8 nent can pass 2^31)
template <typename T>
void Reserve(DvceArray1D<T> &v, const char *label, const std::size_t n) {
  const std::size_t need = std::max<std::size_t>(1, n);
  if (v.extent(0) < need) {
    v = DvceArray1D<T>(std::string(label), std::max<std::size_t>(1, n + n/4));
  }
}
template <typename T>
void Upload(DvceArray1D<T> &v, const char *label, const std::vector<T> &h, const std::size_t n) {
  Reserve(v, label, n);
  if (n == 0) return;
  Kokkos::View<const T *, LayoutWrapper, HostMemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      hv(h.data(), n);
  Kokkos::deep_copy(Kokkos::subview(v, std::make_pair(static_cast<std::size_t>(0), n)), hv);
}

struct RecvPair {int rank, sgid, tgid; std::vector<int> cells; std::vector<PairEnt> ent; int tm;};
struct SendPair {int rank, sgid, tgid; std::vector<int> cells; int sm;};
struct TEnt {int tflat; int sgid; int scell; int sref; double w; WRecipe rc;};

Geo MakeGeo(Mesh *pm, const bool shear, const int x3mode) {
  const auto &indcs = pm->mb_indcs;
  Geo g;
  g.R = pm->max_level; g.root = pm->root_level;
  g.nx[0] = indcs.nx1; g.nx[1] = indcs.nx2; g.nx[2] = indcs.nx3; g.ng = indcs.ng;
  for (int d=0; d<3; ++d) g.nc[d] = g.nx[d] + 2*g.ng;
  g.nmbroot[0] = pm->nmb_rootx1; g.nmbroot[1] = pm->nmb_rootx2; g.nmbroot[2] = pm->nmb_rootx3;
  for (int d=0; d<3; ++d) {
    g.N[d] = static_cast<std::int64_t>(g.nmbroot[d])*g.nx[d]*
             (static_cast<std::int64_t>(1) << (g.R - g.root));
  }
  g.shear = shear;
  g.x3mode = x3mode;
  return g;
}

// What the host part of a build reads: the live mesh (main thread), or a snapshot of the
// blocks a shear part needs (a prefetch thread, which must not read the mesh: it may be
// regridded meanwhile; the snapshot is checked against the mesh sequence before use)
struct BuildIn {
  Geo g;
  int nmb = 0, gids = 0, me = 0;
  MeshBlockPack *pp = nullptr;      // live; nullptr: the snapshot (shear part only)
  std::vector<int> face[2];         // blocks at the inner / outer x1 face (sorted gids)
  std::vector<BlkR> loc, fblk[2];   // snapshot: the local blocks, the face blocks
  bool Blk(const int gid, BlkR &b) const {
    if (pp != nullptr) {b = MakeBlk(pp->pmesh, g, gid); return true;}
    if (gid >= gids && gid < gids + nmb) {b = loc[gid - gids]; return true;}
    for (int f=0; f<2; ++f) {
      auto it = std::lower_bound(face[f].begin(), face[f].end(), gid);
      if (it != face[f].end() && *it == gid) {b = fblk[f][it - face[f].begin()]; return true;}
    }
    return false;   // (not reached: a shear part asks for local and face blocks only)
  }
};

}  // namespace

// The host products of one build of a part (no MPI, no device): what FinishPart checks,
// uploads and keeps.  t: seconds of the phases recv pairs, send pairs, check, layout, sort.
struct DepHostPart {
  std::vector<int> seg_tgt, seg_ptr, src, scell, dcell;
  std::vector<Real> w, dvol;
  std::vector<int> send_rank, send_off, send_cnt, recv_rank, recv_off, recv_cnt;
  int nrecv = 0;
  std::vector<double> recipe;       // the shear part: kRecipe doubles per entry
  int nbad = 0;
  double worst = 0.0;
  bool lookup_failed = false;
  double t[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
};

// The shear part of the next one-cell offset, built on a host thread from a snapshot.
// Its overlap structure (entries, order, message layouts, recipes) is that of any offset with
// the same integer part that is not within 1e-9 of an integer (ShearCacheable: the assumption
// UpdateShearWeights already makes); its weights are recomputed for the actual offset.
struct DepShearPrefetch {
  std::thread worker;
  std::int64_t floor = 0;
  double shR = 0.0;                 // the offset it is built for: floor + 0.5
  int build_seq = -1, mesh_seq = -1;
  BuildIn in;
  DepHostPart hp;
  double seconds = 0.0;             // host time of the build on the thread
  bool ok = false;
};

namespace {

// The lists of one part (0 static, 1 shear) for this rank (see the file header), on the host
void BuildHostPart(const BuildIn &in, const int part, const double shR, DepHostPart &out) {
  const Geo &g = in.g;
  const int nmb = in.nmb, gids = in.gids, me = in.me;
  const int ncell = g.nc[0]*g.nc[1]*g.nc[2];
  std::vector<PairEnt> pe;
  std::vector<TEnt> tent;
  std::vector<RecvPair> rpairs;
  std::vector<SendPair> spairs;
  // the coverage counts of the source ghost cells: every cell of every local block for the static
  // part; for the shear part only the x1 ghost layers (2 nghost wide), since a cell beyond a
  // shear-periodic x1 face is an x1 ghost cell (the full array was 77 MB per rank and
  // build for one 128 x 128 x 512 block)
  const int nsl = 2*g.ng;
  const bool slab = (part == 1);
  std::vector<double> cov(slab ? static_cast<std::size_t>(nmb)*g.nc[2]*g.nc[1]*nsl
                               : static_cast<std::size_t>(nmb)*ncell, 0.0);
  auto covat = [&](const int sm, const int scell) -> double & {
    if (!slab) return cov[static_cast<std::size_t>(sm)*ncell + scell];
    const int i = scell % g.nc[0], jk = scell/g.nc[0];
    const int is = (i < g.ng) ? i : i - (g.nc[0] - nsl);   // [0, ng) and [nc - ng, nc) -> [0, 2 ng)
    return cov[(static_cast<std::size_t>(sm)*g.nc[2]*g.nc[1] + jk)*nsl + is];
  };
  using PClock = std::chrono::steady_clock;
  auto tp = PClock::now();
  auto lap = [&](const int k) {
    const auto now = PClock::now();
    out.t[k] += std::chrono::duration<double>(now - tp).count();
    tp = now;
  };

  // receiving side: the targets of this rank and their sources
  for (int tm=0; tm<nmb; ++tm) {
    BlkR t;
    if (!in.Blk(gids + tm, t)) {out.lookup_failed = true; return;}
    if (part == 1 && !t.face[0] && !t.face[1]) continue;
    for (int sg : Candidates(in.pp, t, tm, part, in.face)) {
      BlkR s;
      if (!in.Blk(sg, s)) {out.lookup_failed = true; return;}
      if (part == 1) {   // a target at the inner face receives from the outer face and vice versa
        if (!((s.face[1] && t.face[0]) || (s.face[0] && t.face[1]))) continue;
      }
      PairOverlap(g, s, t, part, shR, pe);
      if (pe.empty()) continue;
      if (s.rank == me) {
        const int sm = sg - gids;
        for (const PairEnt &e : pe) {
          tent.push_back({tm*ncell + e.tcell, sg, e.scell, sm*ncell + e.scell, e.w, e.rc});
          covat(sm, e.scell) += e.vol;
        }
      } else {
        RecvPair rp;
        rp.rank = s.rank; rp.sgid = sg; rp.tgid = t.gid; rp.tm = tm; rp.ent = pe;
        for (const PairEnt &e : pe) rp.cells.push_back(e.scell);
        std::sort(rp.cells.begin(), rp.cells.end());
        rp.cells.erase(std::unique(rp.cells.begin(), rp.cells.end()), rp.cells.end());
        rpairs.push_back(std::move(rp));
      }
    }
  }
  lap(0);
  // sending side: the sources of this rank with targets on other ranks
  for (int sm=0; sm<nmb; ++sm) {
    BlkR s;
    if (!in.Blk(gids + sm, s)) {out.lookup_failed = true; return;}
    if (part == 1 && !s.face[0] && !s.face[1]) continue;
    for (int tg : Candidates(in.pp, s, sm, part, in.face)) {
      BlkR t;
      if (!in.Blk(tg, t)) {out.lookup_failed = true; return;}
      if (t.rank == me) continue;
      if (part == 1) {
        if (!((s.face[1] && t.face[0]) || (s.face[0] && t.face[1]))) continue;
      }
      PairOverlap(g, s, t, part, shR, pe);
      if (pe.empty()) continue;
      SendPair sp;
      sp.rank = t.rank; sp.sgid = s.gid; sp.tgid = tg; sp.sm = sm;
      for (const PairEnt &e : pe) {
        sp.cells.push_back(e.scell);
        covat(sm, e.scell) += e.vol;
      }
      std::sort(sp.cells.begin(), sp.cells.end());
      sp.cells.erase(std::unique(sp.cells.begin(), sp.cells.end()), sp.cells.end());
      spairs.push_back(std::move(sp));
    }
  }
  lap(1);
  // every ghost cell of the part covered once (dropped cells beyond a physical x3 face: none);
  // the result is reported by FinishPart
  int nbad = 0;
  double worst = 0.0;
  std::vector<double> dvol;
  for (int sm=0; sm<nmb; ++sm) {
    BlkR s;
    if (!in.Blk(gids + sm, s)) {out.lookup_failed = true; return;}
    const double svol = static_cast<double>(s.sc)*static_cast<double>(s.sc)*
                        static_cast<double>(s.sc);
    // (the shear part: only the x1 ghost layers, which hold every cell beyond a shear face)
    std::vector<int> irange;
    for (int i=0; i<g.nc[0]; ++i) if (!slab || i < g.ng || i >= g.nc[0] - g.ng) irange.push_back(i);
    for (int k=0; k<g.nc[2]; ++k) for (int j=0; j<g.nc[1]; ++j) for (const int i : irange) {
      const bool act = (i >= g.ng && i < g.nc[0] - g.ng && j >= g.ng && j < g.nc[1] - g.ng &&
                        k >= g.ng && k < g.nc[2] - g.ng);
      if (act) continue;
      const std::int64_t a0 = s.lo[0] + (i - g.ng)*s.sc;
      const bool beyond = g.shear && (a0 < 0 || a0 >= g.N[0]);
      if ((part == 1) != beyond) continue;
      const std::int64_t c0 = s.lo[2] + (k - g.ng)*s.sc;
      if (g.x3mode == 2 && (c0 < 0 || c0 >= g.N[2])) {   // dropped: recorded, not covered
        if (part == 0) {   // (part 0 is always built from the live mesh)
          out.dcell.push_back(sm*ncell + (k*g.nc[1] + j)*g.nc[0] + i);
          const auto &sz = in.pp->pmb->mb_size.h_view(sm);
          dvol.push_back(sz.dx1*sz.dx2*sz.dx3);
        }
        continue;
      }
      const double c = covat(sm, (k*g.nc[1] + j)*g.nc[0] + i);
      const double err = std::fabs(c - svol)/svol;
      worst = std::max(worst, err);
      if (err > 1.0e-9) ++nbad;
    }
  }
  out.nbad = nbad;
  out.worst = worst;
  out.dvol.resize(dvol.size());
  for (std::size_t q=0; q<dvol.size(); ++q) out.dvol[q] = static_cast<Real>(dvol[q]);
  lap(2);
  // message layouts: pairs sorted by (rank, source gid, target gid)
  auto key = [](int r, int s, int t) {return std::make_tuple(r, s, t);};
  std::sort(rpairs.begin(), rpairs.end(), [&](const RecvPair &x, const RecvPair &y) {
    return key(x.rank, x.sgid, x.tgid) < key(y.rank, y.sgid, y.tgid);});
  std::sort(spairs.begin(), spairs.end(), [&](const SendPair &x, const SendPair &y) {
    return key(x.rank, x.sgid, x.tgid) < key(y.rank, y.sgid, y.tgid);});
  int pos = 0;
  for (const RecvPair &rp : rpairs) {
    if (out.recv_rank.empty() || out.recv_rank.back() != rp.rank) {
      out.recv_rank.push_back(rp.rank); out.recv_off.push_back(pos); out.recv_cnt.push_back(0);
    }
    for (const PairEnt &e : rp.ent) {
      const int q = static_cast<int>(std::lower_bound(rp.cells.begin(), rp.cells.end(), e.scell) -
                                     rp.cells.begin());
      tent.push_back({rp.tm*ncell + e.tcell, rp.sgid, e.scell, -(pos + q + 1), e.w, e.rc});
    }
    pos += static_cast<int>(rp.cells.size());
    out.recv_cnt.back() += static_cast<int>(rp.cells.size());
  }
  out.nrecv = pos;
  for (const SendPair &sp : spairs) {
    if (out.send_rank.empty() || out.send_rank.back() != sp.rank) {
      out.send_rank.push_back(sp.rank);
      out.send_off.push_back(static_cast<int>(out.scell.size()));
      out.send_cnt.push_back(0);
    }
    for (int c : sp.cells) out.scell.push_back(sp.sm*ncell + c);
    out.send_cnt.back() += static_cast<int>(sp.cells.size());
  }
  lap(3);
  // target lists: by target cell, then (source gid, source cell)
  std::sort(tent.begin(), tent.end(), [](const TEnt &x, const TEnt &y) {
    if (x.tflat != y.tflat) return x.tflat < y.tflat;
    if (x.sgid != y.sgid) return x.sgid < y.sgid;
    return x.scell < y.scell;
  });
  for (std::size_t e=0; e<tent.size(); ++e) {
    if (e == 0 || tent[e].tflat != tent[e-1].tflat) {
      out.seg_tgt.push_back(tent[e].tflat);
      out.seg_ptr.push_back(static_cast<int>(e));
    }
  }
  out.seg_ptr.push_back(static_cast<int>(tent.size()));
  const std::size_t nent = tent.size();
  out.src.resize(nent);
  out.w.resize(nent);
  for (std::size_t e=0; e<nent; ++e) {out.src[e] = tent[e].sref; out.w[e] = static_cast<Real>(tent[e].w);}
  // the shear part: its weights in entry order, to recompute them while
  // the overlap structure holds (UpdateShear)
  if (part == 1) {
    constexpr int kR = 8;   // (DepRemapPlan::kRecipe)
    out.recipe.resize(nent*kR);
    for (std::size_t e=0; e<nent; ++e) {
      const WRecipe &r = tent[e].rc;
      double *q = &out.recipe[e*kR];
      q[0] = r.ox; q[1] = r.oz; q[2] = r.ya; q[3] = r.yb; q[4] = r.clo; q[5] = r.chi; q[6] = r.tvol;
      q[7] = r.neg ? 1.0 : 0.0;
    }
  }
  lap(4);
}

bool EnvOn(const char *name) {
  const char *e = std::getenv(name);
  return (e != nullptr && e[0] == '1');
}

}  // namespace

//----------------------------------------------------------------------------------------
// DepRemapPlan constructor

DepRemapPlan::DepRemapPlan(MeshBlockPack *pp, bool shear_x1, Real qshear, Real omega0,
                           int x3mode) :
    pmy_pack(pp), shear_x1_(shear_x1), x3mode_(x3mode), qshear_(qshear), omega0_(omega0) {
  static_assert(kRecipe == 8, "BuildHostPart writes 8 recipe doubles per shear entry");
  dev_weights_ = !EnvOn("ATHENA_DEP_WEIGHTS_HOST");
  prefetch_on_ = !EnvOn("ATHENA_DEP_PREFETCH_OLD");
  verify_ = EnvOn("ATHENA_DEP_VERIFY");
#if MPI_PARALLEL_ENABLED
  // The prefetch path adds one MPI_Allreduce per one-cell shift (UpdateShear): the switches
  // must agree on every rank (collective: the construction is, BuildPart calls MPI_Alltoall)
  if (global_variable::nranks > 1) {
    const int a = prefetch_on_ ? 1 : 0, b = dev_weights_ ? 1 : 0, c = verify_ ? 1 : 0;
    int v[6] = {a, b, c, -a, -b, -c};
    MPI_Allreduce(MPI_IN_PLACE, v, 6, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (v[0] != -v[3] || v[1] != -v[4] || v[2] != -v[5]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "ATHENA_DEP_PREFETCH_OLD / ATHENA_DEP_WEIGHTS_HOST / ATHENA_DEP_VERIFY must "
                << "be set alike on every rank" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif
  static bool said = false;
  if (shear_x1_ && !said && global_variable::my_rank == 0) {
    said = true;
    std::cout << "# DEP_REMAP shear_weights=" << (dev_weights_ ? "device" : "host")
              << " shear_prefetch=" << (prefetch_on_ ? "on" : "off")
              << (verify_ ? " verify=on" : "") << std::endl;
  }
  BuildStatic();
}

DepRemapPlan::~DepRemapPlan() {DiscardPrefetch();}

void DepRemapPlan::BuildStatic() {
  DiscardPrefetch();     // (built from the blocks before this regrid)
  Mesh *pm = pmy_pack->pmesh;
  face_gids_[0].clear(); face_gids_[1].clear();
  if (shear_x1_) {
    const int R = pm->max_level;
    for (int g=0; g<pm->nmb_total; ++g) {
      const LogicalLocation &ll = pm->lloc_eachmb[g];
      const std::int32_t nmb1 = pm->nmb_rootx1 << (ll.level - pm->root_level);
      if (ll.lx1 == 0) face_gids_[0].push_back(g);
      if (ll.lx1 == nmb1 - 1) face_gids_[1].push_back(g);
      (void)R;
    }
  }
  BuildPart(0, 0.0, stat);
  ++build_seq;
  shr_cached_ = false;   // (the blocks may have changed)
  if (shear_x1_ && shear_time >= 0.0) {
    const Real t = shear_time;
    shear_time = -1.0;
    UpdateShear(t);
  }
}

void DepRemapPlan::UpdateShear(Real time) {
  if (!shear_x1_) return;
  if (time == shear_time) return;
  Mesh *pm = pmy_pack->pmesh;
  const Real lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
  const Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  const Real ysh = std::fmod(qshear_*omega0_*lx*time, ly);   // as bvals_part.cpp
  const std::int64_t N2 = static_cast<std::int64_t>(pm->nmb_rootx2)*pm->mb_indcs.nx2*
                          (static_cast<std::int64_t>(1) << (pm->max_level - pm->root_level));
  const double shR = static_cast<double>(ysh)*static_cast<double>(N2)/static_cast<double>(ly);
  // a full build only when the overlap structure changes: the integer
  // part of shR, or shR an integer or not (the shifted cells then meet one target cell, not two)
  const double fl = std::floor(shR);
  using PClock = std::chrono::steady_clock;
  const auto t0 = PClock::now();
  DepProfile &P = DepProf();
  bool done = false;
  if (shr_cached_ && ShearCacheable(shR) && static_cast<std::int64_t>(fl) == shr_floor_ &&
      (shR == fl) == shr_int_) {
    done = ShearWeights(shR);
    if (done) {++shr_updates; P.tupd += std::chrono::duration<double>(PClock::now() - t0).count();}
  }
  if (!done) {
    // the structure changed (normally a one-cell shift): the shear part prefetched for this
    // offset, or a full build.  Every rank comes here at the same calls (the offset is global).
    const std::int64_t f = static_cast<std::int64_t>(fl);
    int took = -1;   // 1 prefetch installed, 0 installed but an overlap vanished, -1 not taken
    if (prefetch_on_) took = TakePrefetch(shR, f);
    if (took < 0) {BuildPart(1, shR, shr); ++shr_builds;}
    bool redo = (took == 0);
#if MPI_PARALLEL_ENABLED
    if (prefetch_on_ && global_variable::nranks > 1) {   // (a rebuild calls MPI_Alltoall)
      int bad = redo ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      redo = (bad != 0);
    }
#endif
    if (redo) {BuildPart(1, shR, shr); ++shr_builds;}
    if (prefetch_on_) {
      // build the next one-cell offset ahead while the plan is driven cell by cell (the drag
      // and gravity plans: every cycle; not a science plan updated once per output)
      const std::int64_t d = (qshear_*omega0_ >= 0.0) ? 1 : -1;   // (time increases)
      auto next = [&](std::int64_t q) {
        q += d;
        if (q >= N2) q -= N2;
        if (q < -N2) q += N2;
        return q;
      };
      const bool driven = have_event_ && (event_floor_ == f || next(event_floor_) == f);
      have_event_ = true;
      event_floor_ = f;
      if (driven) LaunchPrefetch(next(f));
    }
  }
  shear_time = time;
  if (P.on && ++P.ncall % 400 == 0 && global_variable::my_rank == 0) {
    const double fb = (P.nbuild > 0) ? 1.0e3/P.nbuild : 0.0;
    const double fi = (P.ninst > 0) ? 1.0e3/P.ninst : 0.0;
    std::printf("# DEP_PROFILE shear: %d full builds, ms per build: recv pairs %.2f send pairs %.2f check %.2f "
                "layout+Alltoall %.2f sort %.2f device %.2f total %.2f; %d weight updates (%s), %.2f ms each "
                "(entries %d, sends %d); prefetch: %d installs, ms each: wait %.2f install %.2f "
                "(Alltoall %.2f) weights %.2f, thread build %.2f; %d discarded; verify mismatches %ld\n",
                P.nbuild, fb*P.t[0], fb*P.t[1], fb*P.t[2], fb*P.t[3], fb*P.t[4],
                fb*P.t[5], fb*P.t[6], shr_updates, dev_weights_ ? "device" : "host",
                shr_updates > 0 ? 1.0e3*P.tupd/shr_updates : 0.0, shr.nent, shr.nsend, P.ninst,
                fi*P.twait, fi*P.tinst, fi*P.tinst_a2a, fi*P.tinst_w, fi*P.tthread, P.ndisc,
                P.nmis);
    std::fflush(stdout);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void DepRemapPlan::BuildPart
//! \brief The lists of one part (0 static, 1 shear) for this rank (see the file header):
//! built on the host from the live mesh, checked, and uploaded.

void DepRemapPlan::BuildPart(const int part, const double shR, DepRemapMap &map) {
  using PClock = std::chrono::steady_clock;
  const auto tp0 = PClock::now();
  BuildIn in;
  in.g = MakeGeo(pmy_pack->pmesh, shear_x1_, x3mode_);
  in.nmb = pmy_pack->nmb_thispack;
  in.gids = pmy_pack->gids;
  in.me = global_variable::my_rank;
  in.pp = pmy_pack;
  in.face[0] = face_gids_[0];
  in.face[1] = face_gids_[1];
  DepHostPart hp;
  BuildHostPart(in, part, shR, hp);
  const double ta = FinishPart(part, shR, hp, map, true);
  if (part == 1 && DepProf().on) {
    DepProfile &P = DepProf();
    for (int k=0; k<5; ++k) P.t[k] += hp.t[k];
    P.t[3] += ta;
    const double tot = std::chrono::duration<double>(PClock::now() - tp0).count();
    P.t[5] += tot - (hp.t[0] + hp.t[1] + hp.t[2] + hp.t[3] + hp.t[4] + ta);
    P.t[6] += tot;
    ++P.nbuild;
  }
}

//----------------------------------------------------------------------------------------
//! \fn double DepRemapPlan::FinishPart
//! \brief The checks of a host build (coverage; the message sizes with MPI_Alltoall), the
//! upload of its lists into map, and for the shear part its recipes.  upload_w false: the
//! weights are left to the caller (a prefetched part, built for another offset).  Returns the
//! seconds of the message-size check.

double DepRemapPlan::FinishPart(const int part, const double shR, DepHostPart &hp,
                                DepRemapMap &map, const bool upload_w) {
  const int me = global_variable::my_rank;
  if (hp.nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "deposit remap (part " << part << "): " << hp.nbad << " ghost cells on rank " << me
              << " not covered exactly once by active cells (worst relative error " << hp.worst
              << "); blocks narrower than 2 nghost or a level jump > 1?" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  map.recv_rank = std::move(hp.recv_rank);
  map.recv_off = std::move(hp.recv_off);
  map.recv_cnt = std::move(hp.recv_cnt);
  map.send_rank = std::move(hp.send_rank);
  map.send_off = std::move(hp.send_off);
  map.send_cnt = std::move(hp.send_cnt);
  map.nrecv = hp.nrecv;
  map.nsend = static_cast<int>(hp.scell.size());
  using PClock = std::chrono::steady_clock;
  const auto ta0 = PClock::now();
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {   // the message sizes agree on both sides
    const int nr = global_variable::nranks;
    std::vector<int> scnt(nr, 0), rcnt(nr, 0), expect(nr, 0);
    for (std::size_t q=0; q<map.send_rank.size(); ++q) scnt[map.send_rank[q]] = map.send_cnt[q];
    for (std::size_t q=0; q<map.recv_rank.size(); ++q) expect[map.recv_rank[q]] = map.recv_cnt[q];
    MPI_Alltoall(scnt.data(), 1, MPI_INT, rcnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int nmis = 0;
    for (int r=0; r<nr; ++r) if (rcnt[r] != expect[r]) ++nmis;
    if (nmis > 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "deposit remap (part " << part << "): rank " << me << " expects other "
                << "message sizes than " << nmis << " ranks send" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif
  const double ta = std::chrono::duration<double>(PClock::now() - ta0).count();
  map.nseg = static_cast<int>(hp.seg_tgt.size());
  map.nent = static_cast<int>(hp.src.size());
  map.ndrop = static_cast<int>(hp.dcell.size());
  Upload(map.drop_cell, "dep_drop_cell", hp.dcell, map.ndrop);
  Upload(map.drop_vol, "dep_drop_vol", hp.dvol, map.ndrop);
  Upload(map.seg_tgt, "dep_seg_tgt", hp.seg_tgt, map.nseg);
  Upload(map.seg_ptr, "dep_seg_ptr", hp.seg_ptr, map.nseg + 1);
  Upload(map.ent_src, "dep_ent_src", hp.src, map.nent);
  if (upload_w) {
    Upload(map.ent_w, "dep_ent_w", hp.w, map.nent);
  } else {
    Reserve(map.ent_w, "dep_ent_w", map.nent);
  }
  Upload(map.send_cell, "dep_send_cell", hp.scell, map.nsend);
  if (part == 1) {
    shr_recipe_ = std::move(hp.recipe);
    shr_hw_.resize(map.nent);
    const double fl = std::floor(shR);
    shr_floor_ = static_cast<std::int64_t>(fl);
    shr_int_ = (shR == fl);
    shr_cached_ = ShearCacheable(shR);
    if (dev_weights_) Upload(shr_rc_d_, "dep_shr_recipe", shr_recipe_, shr_recipe_.size());
  }
  return ta;
}

//----------------------------------------------------------------------------------------
//! \fn bool DepRemapPlan::UpdateShearWeights
//! \brief The weights of the shear part for offset shR with the entries
//! of the last build (same overlap structure: same integer part of shR, and shR an integer or not
//! as then).  The arithmetic is that of Overlaps and PairOverlap, so the weights are those of a
//! full build bit for bit.  False (nothing changed) if an overlap would vanish.
//! (The host path: ATHENA_DEP_WEIGHTS_HOST=1; HostShearWeights without the upload.)

namespace {
bool HostShearWeights(const std::vector<double> &recipe, const double shR, std::vector<Real> &hw) {
  const std::size_t n = hw.size();
  for (std::size_t e=0; e<n; ++e) {
    const double *q = &recipe[e*8];
    const WRecipe r{q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7] != 0.0};
    const double sh = r.neg ? -shR : shR;
    const double a = r.ya + sh, b = r.yb + sh;
    if (!(std::min(b, r.chi) - std::max(a, r.clo) > 0.0)) return false;
    hw[e] = static_cast<Real>(WeightOf(r, shR));
  }
  return true;
}
}  // namespace

bool DepRemapPlan::UpdateShearWeights(const double shR) {
  if (!HostShearWeights(shr_recipe_, shR, shr_hw_)) return false;
  Upload(shr.ent_w, "dep_ent_w", shr_hw_, shr_hw_.size());
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn bool DepRemapPlan::UpdateShearWeightsDevice
//! \brief UpdateShearWeights on the device, from the device copy of the recipes: the same
//! operations in the same order (sums, differences, two products and a quotient of doubles;
//! no product feeds a sum, so no contraction to a fused multiply-add can change a bit), so the
//! weights are bitwise those of the host path.  False if an overlap would vanish (the weights
//! are then partly overwritten: the caller rebuilds the part, which uploads all of them).

bool DepRemapPlan::UpdateShearWeightsDevice(const double shR) {
  const int n = shr.nent;
  if (n == 0) return true;
  if (shr_rc_d_.extent(0) < static_cast<std::size_t>(n)*kRecipe) {   // (FinishPart uploads them)
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "deposit remap: the device shear recipes hold " << shr_rc_d_.extent(0)
              << " values for " << n << " entries" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto rc = shr_rc_d_;
  auto w = shr.ent_w;
  int bad = 0;
  Kokkos::parallel_reduce("dep_shear_weights", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const int e, int &nb) {
    const std::size_t o = static_cast<std::size_t>(e)*8;
    const double ox = rc(o), oz = rc(o+1), ya = rc(o+2), yb = rc(o+3), clo = rc(o+4),
                 chi = rc(o+5), tvol = rc(o+6);
    const double sh = (rc(o+7) != 0.0) ? -shR : shR;
    const double a = ya + sh, b = yb + sh;
    const double hi = (chi < b) ? chi : b;     // std::min(b, chi)
    const double lo = (a < clo) ? clo : a;     // std::max(a, clo)
    const double oy = hi - lo;
    if (!(oy > 0.0)) {++nb; return;}
    const double vol = ox*oy*oz;
    w(e) = static_cast<Real>(vol/tvol);
  }, Kokkos::Sum<int>(bad));
  return bad == 0;
}

// the weights of the shear part for offset shR, device or host as switched (ATHENA_DEP_VERIFY:
// the device weights compared with the host loop, bitwise)
bool DepRemapPlan::ShearWeights(const double shR) {
  if (!dev_weights_) return UpdateShearWeights(shR);
  const bool ok = UpdateShearWeightsDevice(shR);
  if (ok && verify_) {
    std::vector<Real> hw(shr.nent);
    long nmis = 0;
    if (!HostShearWeights(shr_recipe_, shR, hw)) {
      nmis = 1;
    } else if (shr.nent > 0) {
      auto dw = Kokkos::create_mirror_view_and_copy(
          HostMemSpace(), Kokkos::subview(shr.ent_w, std::make_pair(0, shr.nent)));
      for (int e=0; e<shr.nent; ++e) if (!(dw(e) == hw[e])) ++nmis;
    }
    DepProfile &P = DepProf();
    ++P.nver_w;
    if (nmis > 0) {
      P.nmis += nmis;
      std::printf("# DEP_VERIFY rank %d: device shear weights differ from the host loop in %ld of %d "
                  "entries (offset %.17g)\n", global_variable::my_rank, nmis, shr.nent, shR);
      std::fflush(stdout);
    }
  }
  return ok;
}

//----------------------------------------------------------------------------------------
//! \fn void DepRemapPlan::LaunchPrefetch
//! \brief Starts the host build of the shear part of the offset floor + 0.5 on a thread, from a
//! snapshot of the blocks it reads (local blocks, face blocks; no MPI, no Kokkos on the thread).

void DepRemapPlan::LaunchPrefetch(const std::int64_t floor) {
  DiscardPrefetch();
  Mesh *pm = pmy_pack->pmesh;
  auto p = std::make_unique<DepShearPrefetch>();
  p->floor = floor;
  p->shR = static_cast<double>(floor) + 0.5;
  p->build_seq = build_seq;
  p->mesh_seq = pm->GetAMRLoadBalanceUpdateSeq();
  BuildIn &in = p->in;
  in.g = MakeGeo(pm, shear_x1_, x3mode_);
  in.nmb = pmy_pack->nmb_thispack;
  in.gids = pmy_pack->gids;
  in.me = global_variable::my_rank;
  in.pp = nullptr;
  for (int f=0; f<2; ++f) {
    in.face[f] = face_gids_[f];
    in.fblk[f].reserve(in.face[f].size());
    for (const int gid : in.face[f]) in.fblk[f].push_back(MakeBlk(pm, in.g, gid));
  }
  in.loc.reserve(in.nmb);
  for (int m=0; m<in.nmb; ++m) in.loc.push_back(MakeBlk(pm, in.g, in.gids + m));
  DepShearPrefetch *q = p.get();
  try {
    q->worker = std::thread([q]() {
      try {
        const auto t0 = std::chrono::steady_clock::now();
        BuildHostPart(q->in, 1, q->shR, q->hp);
        q->ok = !q->hp.lookup_failed && q->hp.nbad == 0;
        q->seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      } catch (...) {
        q->ok = false;   // (a full build at the shift reports what goes wrong)
      }
    });
  } catch (...) {
    return;   // no thread (resources): no prefetch, the shift builds as before
  }
  pf_ = std::move(p);
}

void DepRemapPlan::DiscardPrefetch() {
  if (pf_ && pf_->worker.joinable()) pf_->worker.join();
  pf_.reset();
}

//----------------------------------------------------------------------------------------
//! \fn int DepRemapPlan::TakePrefetch
//! \brief The prefetched shear part, if it was built for this integer part of the offset from
//! the current blocks and the offset is cacheable: installed (FinishPart: the message-size
//! check, uploads) and its weights computed for shR.  1 installed, 0 installed but an overlap
//! vanished (the caller rebuilds), -1 not taken (the caller builds).

int DepRemapPlan::TakePrefetch(const double shR, const std::int64_t fl) {
  if (!pf_) return -1;
  using PClock = std::chrono::steady_clock;
  const auto t0 = PClock::now();
  if (pf_->worker.joinable()) pf_->worker.join();
  const auto t1 = PClock::now();
  DepProfile &P = DepProf();
  const bool usable = pf_->ok && pf_->floor == fl && pf_->build_seq == build_seq &&
                      pf_->mesh_seq == pmy_pack->pmesh->GetAMRLoadBalanceUpdateSeq() &&
                      ShearCacheable(shR);
  if (!usable) {
    ++P.ndisc;
    pf_.reset();
    return -1;
  }
  const double tthread = pf_->seconds;
  const double ta = FinishPart(1, shR, pf_->hp, shr, false);
  const auto t2 = PClock::now();
  ++shr_prefetched;
  const bool ok = ShearWeights(shR);
  const auto t3 = PClock::now();
  pf_.reset();
  if (verify_) VerifyInstall(shR);
  ++P.ninst;
  P.twait += std::chrono::duration<double>(t1 - t0).count();
  P.tinst += std::chrono::duration<double>(t2 - t1).count();
  P.tinst_a2a += ta;
  P.tinst_w += std::chrono::duration<double>(t3 - t2).count();
  P.tthread += tthread;
  return ok ? 1 : 0;
}

//----------------------------------------------------------------------------------------
//! \fn void DepRemapPlan::VerifyInstall
//! \brief (ATHENA_DEP_VERIFY=1, diagnostic) The installed shear part against a host build for
//! shR from the live mesh: lists, layouts, recipes and weights, bitwise.  No MPI.

void DepRemapPlan::VerifyInstall(const double shR) {
  BuildIn in;
  in.g = MakeGeo(pmy_pack->pmesh, shear_x1_, x3mode_);
  in.nmb = pmy_pack->nmb_thispack;
  in.gids = pmy_pack->gids;
  in.me = global_variable::my_rank;
  in.pp = pmy_pack;
  in.face[0] = face_gids_[0];
  in.face[1] = face_gids_[1];
  DepHostPart ref;
  BuildHostPart(in, 1, shR, ref);
  long nmis = 0;
  auto cmpd = [&](const auto &dv, const auto &hv, const int n) {
    if (static_cast<int>(hv.size()) < n) {nmis += 1; return;}
    if (n <= 0) return;
    auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                 Kokkos::subview(dv, std::make_pair(0, n)));
    for (int e=0; e<n; ++e) if (!(h(e) == hv[e])) ++nmis;
  };
  if (static_cast<int>(ref.seg_tgt.size()) != shr.nseg ||
      static_cast<int>(ref.src.size()) != shr.nent ||
      static_cast<int>(ref.scell.size()) != shr.nsend || ref.nrecv != shr.nrecv) {
    nmis += 1000000000L;
  } else {
    cmpd(shr.seg_tgt, ref.seg_tgt, shr.nseg);
    cmpd(shr.seg_ptr, ref.seg_ptr, shr.nseg + 1);
    cmpd(shr.ent_src, ref.src, shr.nent);
    cmpd(shr.ent_w, ref.w, shr.nent);
    cmpd(shr.send_cell, ref.scell, shr.nsend);
    if (ref.recv_rank != shr.recv_rank || ref.recv_off != shr.recv_off ||
        ref.recv_cnt != shr.recv_cnt || ref.send_rank != shr.send_rank ||
        ref.send_off != shr.send_off || ref.send_cnt != shr.send_cnt) nmis += 1;
    if (ref.recipe != shr_recipe_) nmis += 1;
    if (dev_weights_ && shr.nent > 0) {
      const std::size_t nr = ref.recipe.size();   // (= kRecipe nent: compared above)
      auto h = Kokkos::create_mirror_view_and_copy(
          HostMemSpace(),
          Kokkos::subview(shr_rc_d_, std::make_pair(static_cast<std::size_t>(0), nr)));
      for (std::size_t e=0; e<nr; ++e) if (!(h(e) == ref.recipe[e])) {++nmis; break;}
    }
  }
  DepProfile &P = DepProf();
  ++P.nver_i;
  if (nmis > 0) P.nmis += nmis;
  if (nmis > 0 || P.nver_i <= 2) {
    std::printf("# DEP_VERIFY rank %d: prefetched shear part (offset %.17g, %d entries) vs a "
                "build for the offset: %ld mismatches\n", global_variable::my_rank, shR, shr.nent,
                nmis);
    std::fflush(stdout);
  }
}

//----------------------------------------------------------------------------------------
// The general path of MeshBoundaryValuesDep

MeshBoundaryValuesDep::~MeshBoundaryValuesDep() {
#if MPI_PARALLEL_ENABLED
  MPI_Comm_free(&comm_dep_);
#endif
}

TaskStatus MeshBoundaryValuesDep::InitRecvDep(const int nvar) {
  if (!General()) return InitRecv(nvar);
  gen_nvar_ = nvar;
#if MPI_PARALLEL_ENABLED
  grreq_.clear();
  gsreq_.clear();
  bool no_errors = true;
  for (int p=0; p<2; ++p) {
    const DepRemapMap &map = (p == 0) ? plan_->stat : plan_->shr;
    // a quarter above the new high-water mark: the shear part of the plan
    // changes its sizes at one-cell shifts, and a new device buffer must be registered with
    // the network layer at its first message (only the valid prefix is used)
    const std::size_t need = std::max<std::size_t>(1, static_cast<std::size_t>(map.nrecv)*nvar);
    if (grbuf_[p].extent(0) < need) Kokkos::realloc(grbuf_[p], DepBufferCapacity(need));
    for (std::size_t q=0; q<map.recv_rank.size(); ++q) {
      MPI_Request req;
      int ierr = MPI_Irecv(grbuf_[p].data() + map.recv_off[q]*nvar, map.recv_cnt[q]*nvar,
                           MPI_ATHENA_REAL, map.recv_rank[q], 31 + p, comm_dep_, &req);
      if (ierr != MPI_SUCCESS) no_errors = false;
      grreq_.push_back(req);
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in posting the deposit receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

TaskStatus MeshBoundaryValuesDep::ClearSendDep() {
  if (!General()) return ClearSend();
#if MPI_PARALLEL_ENABLED
  for (auto &r : gsreq_) MPI_Wait(&r, MPI_STATUS_IGNORE);
  gsreq_.clear();
#endif
  return TaskStatus::complete;
}

TaskStatus MeshBoundaryValuesDep::ClearRecvDep() {
  if (!General()) return ClearRecv();
#if MPI_PARALLEL_ENABLED
  for (auto &r : grreq_) MPI_Wait(&r, MPI_STATUS_IGNORE);
  grreq_.clear();
#endif
  return TaskStatus::complete;
}

void MeshBoundaryValuesDep::GeneralPack(DvceArray5D<Real> &a, const DepRemapMap &map,
                                        DvceArray1D<Real> &buf) {
  const int nvar = a.extent_int(1);
  const std::size_t need = std::max<std::size_t>(1, static_cast<std::size_t>(map.nsend)*nvar);
  if (buf.extent(0) < need) Kokkos::realloc(buf, DepBufferCapacity(need));   // (as grbuf_)
  if (map.nsend == 0) return;
  const int nc1 = a.extent_int(4), nc2 = a.extent_int(3), nc3 = a.extent_int(2);
  const int ncell = nc1*nc2*nc3;
  auto cells = map.send_cell;
  auto b = buf;
  Kokkos::parallel_for("dep_gen_pack", Kokkos::RangePolicy<>(DevExeSpace(), 0, map.nsend),
  KOKKOS_LAMBDA(const int e) {
    const int c = cells(e);
    const int m = c/ncell;
    int r = c - m*ncell;
    const int k = r/(nc2*nc1); r -= k*(nc2*nc1);
    const int j = r/nc1, i = r - j*nc1;
    for (int v=0; v<nvar; ++v) b(e*nvar + v) = a(m,v,k,j,i);
  });
}

void MeshBoundaryValuesDep::GeneralApply(DvceArray5D<Real> &a, const DepRemapMap &map,
                                         const DvceArray1D<Real> &buf) {
  if (map.nseg == 0) return;
  const int nvar = a.extent_int(1);
  const int nc1 = a.extent_int(4), nc2 = a.extent_int(3), nc3 = a.extent_int(2);
  const int ncell = nc1*nc2*nc3;
  auto tgt = map.seg_tgt;
  auto ptr = map.seg_ptr;
  auto src = map.ent_src;
  auto w = map.ent_w;
  auto b = buf;
  Kokkos::parallel_for("dep_gen_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, map.nseg),
  KOKKOS_LAMBDA(const int s) {
    const int c = tgt(s);
    const int m = c/ncell;
    int r = c - m*ncell;
    const int k = r/(nc2*nc1); r -= k*(nc2*nc1);
    const int j = r/nc1, i = r - j*nc1;
    for (int v=0; v<nvar; ++v) {
      Real sum = 0.0;
      for (int e=ptr(s); e<ptr(s+1); ++e) {
        const int q = src(e);
        Real val;
        if (q >= 0) {
          const int sm = q/ncell;
          int sr = q - sm*ncell;
          const int sk = sr/(nc2*nc1); sr -= sk*(nc2*nc1);
          const int sj = sr/nc1, si = sr - sj*nc1;
          val = a(sm,v,sk,sj,si);
        } else {
          val = b((-q - 1)*nvar + v);
        }
        sum += w(e)*val;
      }
      a(m,v,k,j,i) += sum;
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn MeshBoundaryValuesDep::GeneralDropRecord
//! \brief The deposits in the ghost cells beyond a physical x3 face that are dropped
//! (x3 mode drop), integrated over their volumes and added to dropped_sum (per variable).

void MeshBoundaryValuesDep::GeneralDropRecord(DvceArray5D<Real> &a, const DepRemapMap &map) {
  if (map.ndrop == 0) return;   // (fold mode, periodic x3, or no block at a dropped face)
  const int nvar = a.extent_int(1);
  if (static_cast<int>(dropped_sum.size()) != nvar) dropped_sum.assign(nvar, 0.0);
  const int nc1 = a.extent_int(4), nc2 = a.extent_int(3), nc3 = a.extent_int(2);
  const int ncell = nc1*nc2*nc3;
  auto cells = map.drop_cell;
  auto vol = map.drop_vol;
  for (int v=0; v<nvar; ++v) {
    double sum = 0.0;
    Kokkos::parallel_reduce("dep_gen_drop", Kokkos::RangePolicy<>(DevExeSpace(), 0, map.ndrop),
    KOKKOS_LAMBDA(const int e, double &ss) {
      const int c = cells(e);
      const int m = c/ncell;
      int r = c - m*ncell;
      const int k = r/(nc2*nc1); r -= k*(nc2*nc1);
      const int j = r/nc1, i = r - j*nc1;
      ss += static_cast<double>(a(m,v,k,j,i)*vol(e));
    }, Kokkos::Sum<double>(sum));
    dropped_sum[v] += sum;
  }
}
