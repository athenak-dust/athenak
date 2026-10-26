//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals_dep_general.cpp
//! \brief S3a: the additive ghost-deposit exchange of a 3-D mesh with
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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals.hpp"

namespace {

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
struct Ov {int t; double o;};

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
        if (o > 0.0) out.push_back({c, o});
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

struct PairEnt {int scell; int tcell; double w; double vol;};

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
  for (int k=0; k<nc3; ++k) {
    if (l3[k].empty()) continue;
    const bool kin = (k >= ng && k < nc3 - ng);
    for (int j=0; j<nc2; ++j) {
      const bool jin = (j >= ng && j < nc2 - ng);
      for (int i=0; i<nc1; ++i) {
        if (l1[i].empty()) continue;
        const bool iin = (i >= ng && i < nc1 - ng);
        if (iin && jin && kin) continue;           // active cell: not a deposit to move
        const std::vector<Ov> &ly = (part == 1 && side[i] == 0) ? l2m[j] : l2p[j];
        if (ly.empty()) continue;
        const int scell = (k*nc2 + j)*nc1 + i;
        for (const Ov &oz : l3[k]) {
          for (const Ov &oy : ly) {
            for (const Ov &ox : l1[i]) {
              const double vol = ox.o*oy.o*oz.o;
              const int tcell = ((oz.t + ng)*nc2 + (oy.t + ng))*nc1 + (ox.t + ng);
              out.push_back({scell, tcell, vol/tvol, vol});
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

// ATHENA_DEP_PROFILE=1 in the environment prints, on rank 0 every 200
// builds of the shear part, the mean host time per build of its phases (diagnostic only)
struct DepProfile {
  bool on = false;
  int nbuild = 0;
  double t[7] = {0, 0, 0, 0, 0, 0, 0};   // recv pairs, send pairs, check, layout+Alltoall, sort, device, total
  DepProfile() {const char *e = std::getenv("ATHENA_DEP_PROFILE"); on = (e != nullptr && e[0] == '1');}
};
DepProfile &DepProf() {static DepProfile p; return p;}

struct RecvPair {int rank, sgid, tgid; std::vector<int> cells; std::vector<PairEnt> ent; int tm;};
struct SendPair {int rank, sgid, tgid; std::vector<int> cells; int sm;};
struct TEnt {int tflat; int sgid; int scell; int sref; double w;};

}  // namespace

//----------------------------------------------------------------------------------------
// DepRemapPlan constructor

DepRemapPlan::DepRemapPlan(MeshBlockPack *pp, bool shear_x1, Real qshear, Real omega0,
                           int x3mode) :
    pmy_pack(pp), shear_x1_(shear_x1), x3mode_(x3mode), qshear_(qshear), omega0_(omega0) {
  BuildStatic();
}

void DepRemapPlan::BuildStatic() {
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
  BuildPart(1, shR, shr);
  shear_time = time;
}

//----------------------------------------------------------------------------------------
//! \fn void DepRemapPlan::BuildPart
//! \brief The lists of one part (0 static, 1 shear) for this rank (see the file header).

void DepRemapPlan::BuildPart(const int part, const double shR, DepRemapMap &map) {
  Mesh *pm = pmy_pack->pmesh;
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
  g.shear = shear_x1_;
  g.x3mode = x3mode_;
  const int nmb = pmy_pack->nmb_thispack, gids = pmy_pack->gids;
  const int ncell = g.nc[0]*g.nc[1]*g.nc[2];
  const int me = global_variable::my_rank;
  std::vector<PairEnt> pe;
  std::vector<TEnt> tent;
  std::vector<RecvPair> rpairs;
  std::vector<SendPair> spairs;
  std::vector<double> cov(static_cast<std::size_t>(nmb)*ncell, 0.0);
  using PClock = std::chrono::steady_clock;
  const bool prof = (part == 1) && DepProf().on;
  auto tp0 = PClock::now(), tp = tp0;
  auto lap = [&](const int k) {
    if (!prof) return;
    const auto now = PClock::now();
    DepProf().t[k] += std::chrono::duration<double>(now - tp).count();
    tp = now;
  };

  // receiving side: the targets of this rank and their sources
  for (int tm=0; tm<nmb; ++tm) {
    const BlkR t = MakeBlk(pm, g, gids + tm);
    if (part == 1 && !t.face[0] && !t.face[1]) continue;
    for (int sg : Candidates(pmy_pack, t, tm, part, face_gids_)) {
      const BlkR s = MakeBlk(pm, g, sg);
      if (part == 1) {   // a target at the inner face receives from the outer face and vice versa
        if (!((s.face[1] && t.face[0]) || (s.face[0] && t.face[1]))) continue;
      }
      PairOverlap(g, s, t, part, shR, pe);
      if (pe.empty()) continue;
      if (s.rank == me) {
        const int sm = sg - gids;
        for (const PairEnt &e : pe) {
          tent.push_back({tm*ncell + e.tcell, sg, e.scell, sm*ncell + e.scell, e.w});
          cov[static_cast<std::size_t>(sm)*ncell + e.scell] += e.vol;
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
    const BlkR s = MakeBlk(pm, g, gids + sm);
    if (part == 1 && !s.face[0] && !s.face[1]) continue;
    for (int tg : Candidates(pmy_pack, s, sm, part, face_gids_)) {
      const BlkR t = MakeBlk(pm, g, tg);
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
        cov[static_cast<std::size_t>(sm)*ncell + e.scell] += e.vol;
      }
      std::sort(sp.cells.begin(), sp.cells.end());
      sp.cells.erase(std::unique(sp.cells.begin(), sp.cells.end()), sp.cells.end());
      spairs.push_back(std::move(sp));
    }
  }
  lap(1);
  // every ghost cell of the part covered once (dropped cells beyond a physical x3 face: none)
  int nbad = 0;
  double worst = 0.0;
  std::vector<int> dcell;
  std::vector<double> dvol;
  for (int sm=0; sm<nmb; ++sm) {
    const BlkR s = MakeBlk(pm, g, gids + sm);
    const double svol = static_cast<double>(s.sc)*static_cast<double>(s.sc)*
                        static_cast<double>(s.sc);
    for (int k=0; k<g.nc[2]; ++k) for (int j=0; j<g.nc[1]; ++j) for (int i=0; i<g.nc[0]; ++i) {
      const bool act = (i >= g.ng && i < g.nc[0] - g.ng && j >= g.ng && j < g.nc[1] - g.ng &&
                        k >= g.ng && k < g.nc[2] - g.ng);
      if (act) continue;
      const std::int64_t a0 = s.lo[0] + (i - g.ng)*s.sc;
      const bool beyond = g.shear && (a0 < 0 || a0 >= g.N[0]);
      if ((part == 1) != beyond) continue;
      const std::int64_t c0 = s.lo[2] + (k - g.ng)*s.sc;
      if (g.x3mode == 2 && (c0 < 0 || c0 >= g.N[2])) {   // dropped: recorded, not covered
        if (part == 0) {
          dcell.push_back(sm*ncell + (k*g.nc[1] + j)*g.nc[0] + i);
          const auto &sz = pmy_pack->pmb->mb_size.h_view(sm);
          dvol.push_back(sz.dx1*sz.dx2*sz.dx3);
        }
        continue;
      }
      const double c = cov[static_cast<std::size_t>(sm)*ncell + (k*g.nc[1] + j)*g.nc[0] + i];
      const double err = std::fabs(c - svol)/svol;
      worst = std::max(worst, err);
      if (err > 1.0e-9) ++nbad;
    }
  }
  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "deposit remap (part " << part << "): " << nbad << " ghost cells on rank " << me
              << " not covered exactly once by active cells (worst relative error " << worst
              << "); blocks narrower than 2 nghost or a level jump > 1?" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  lap(2);
  // message layouts: pairs sorted by (rank, source gid, target gid)
  auto key = [](int r, int s, int t) {return std::make_tuple(r, s, t);};
  std::sort(rpairs.begin(), rpairs.end(), [&](const RecvPair &x, const RecvPair &y) {
    return key(x.rank, x.sgid, x.tgid) < key(y.rank, y.sgid, y.tgid);});
  std::sort(spairs.begin(), spairs.end(), [&](const SendPair &x, const SendPair &y) {
    return key(x.rank, x.sgid, x.tgid) < key(y.rank, y.sgid, y.tgid);});
  map.recv_rank.clear(); map.recv_off.clear(); map.recv_cnt.clear();
  int pos = 0;
  for (const RecvPair &rp : rpairs) {
    if (map.recv_rank.empty() || map.recv_rank.back() != rp.rank) {
      map.recv_rank.push_back(rp.rank); map.recv_off.push_back(pos); map.recv_cnt.push_back(0);
    }
    for (const PairEnt &e : rp.ent) {
      const int q = static_cast<int>(std::lower_bound(rp.cells.begin(), rp.cells.end(), e.scell) -
                                     rp.cells.begin());
      tent.push_back({rp.tm*ncell + e.tcell, rp.sgid, e.scell, -(pos + q + 1), e.w});
    }
    pos += static_cast<int>(rp.cells.size());
    map.recv_cnt.back() += static_cast<int>(rp.cells.size());
  }
  map.nrecv = pos;
  map.send_rank.clear(); map.send_off.clear(); map.send_cnt.clear();
  std::vector<int> scell;
  for (const SendPair &sp : spairs) {
    if (map.send_rank.empty() || map.send_rank.back() != sp.rank) {
      map.send_rank.push_back(sp.rank);
      map.send_off.push_back(static_cast<int>(scell.size()));
      map.send_cnt.push_back(0);
    }
    for (int c : sp.cells) scell.push_back(sp.sm*ncell + c);
    map.send_cnt.back() += static_cast<int>(sp.cells.size());
  }
  map.nsend = static_cast<int>(scell.size());
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

  lap(3);
  // target lists: by target cell, then (source gid, source cell)
  std::sort(tent.begin(), tent.end(), [](const TEnt &x, const TEnt &y) {
    if (x.tflat != y.tflat) return x.tflat < y.tflat;
    if (x.sgid != y.sgid) return x.sgid < y.sgid;
    return x.scell < y.scell;
  });
  std::vector<int> seg_tgt, seg_ptr;
  for (std::size_t e=0; e<tent.size(); ++e) {
    if (e == 0 || tent[e].tflat != tent[e-1].tflat) {
      seg_tgt.push_back(tent[e].tflat);
      seg_ptr.push_back(static_cast<int>(e));
    }
  }
  seg_ptr.push_back(static_cast<int>(tent.size()));
  map.nseg = static_cast<int>(seg_tgt.size());
  map.nent = static_cast<int>(tent.size());
  lap(4);
  map.seg_tgt = DvceArray1D<int>("dep_seg_tgt", std::max(1, map.nseg));
  map.seg_ptr = DvceArray1D<int>("dep_seg_ptr", map.nseg + 1);
  map.ent_src = DvceArray1D<int>("dep_ent_src", std::max(1, map.nent));
  map.ent_w = DvceArray1D<Real>("dep_ent_w", std::max(1, map.nent));
  map.send_cell = DvceArray1D<int>("dep_send_cell", std::max(1, map.nsend));
  map.ndrop = static_cast<int>(dcell.size());
  map.drop_cell = DvceArray1D<int>("dep_drop_cell", std::max(1, map.ndrop));
  map.drop_vol = DvceArray1D<Real>("dep_drop_vol", std::max(1, map.ndrop));
  {
    auto hc = Kokkos::create_mirror_view(map.drop_cell);
    auto hv = Kokkos::create_mirror_view(map.drop_vol);
    for (int q=0; q<map.ndrop; ++q) {hc(q) = dcell[q]; hv(q) = static_cast<Real>(dvol[q]);}
    Kokkos::deep_copy(map.drop_cell, hc);
    Kokkos::deep_copy(map.drop_vol, hv);
  }
  auto h_tgt = Kokkos::create_mirror_view(map.seg_tgt);
  auto h_ptr = Kokkos::create_mirror_view(map.seg_ptr);
  auto h_src = Kokkos::create_mirror_view(map.ent_src);
  auto h_w = Kokkos::create_mirror_view(map.ent_w);
  auto h_send = Kokkos::create_mirror_view(map.send_cell);
  for (int s=0; s<map.nseg; ++s) h_tgt(s) = seg_tgt[s];
  for (int s=0; s<=map.nseg; ++s) h_ptr(s) = seg_ptr[s];
  for (int e=0; e<map.nent; ++e) {h_src(e) = tent[e].sref; h_w(e) = static_cast<Real>(tent[e].w);}
  for (int e=0; e<map.nsend; ++e) h_send(e) = scell[e];
  Kokkos::deep_copy(map.seg_tgt, h_tgt);
  Kokkos::deep_copy(map.seg_ptr, h_ptr);
  Kokkos::deep_copy(map.ent_src, h_src);
  Kokkos::deep_copy(map.ent_w, h_w);
  Kokkos::deep_copy(map.send_cell, h_send);
  if (prof) {
    lap(5);
    DepProfile &P = DepProf();
    P.t[6] += std::chrono::duration<double>(PClock::now() - tp0).count();
    if (++P.nbuild % 200 == 0 && global_variable::my_rank == 0) {
      const double f = 1.0e3/P.nbuild;
      std::printf("# DEP_PROFILE shear builds %d, ms per build: recv pairs %.2f send pairs %.2f check %.2f "
                  "layout+Alltoall %.2f sort %.2f device %.2f total %.2f (entries %d, sends %d)\n", P.nbuild,
                  f*P.t[0], f*P.t[1], f*P.t[2], f*P.t[3], f*P.t[4], f*P.t[5], f*P.t[6], map.nent, map.nsend);
      std::fflush(stdout);
    }
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
    if (grbuf_[p].extent_int(0) < std::max(1, map.nrecv*nvar)) {
      Kokkos::realloc(grbuf_[p], std::max(1, map.nrecv*nvar));
    }
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
  if (buf.extent_int(0) < std::max(1, map.nsend*nvar)) {
    Kokkos::realloc(buf, std::max(1, map.nsend*nvar));
  }
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
