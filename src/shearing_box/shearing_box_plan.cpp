//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box_plan.cpp
//! \brief The block-local part of the shear-exchange plans.
//! BuildCandidates (per regrid): for every local block on a shear-periodic x1 face, the blocks
//! of the opposite face that overlap it in x3 by at least a margin of nghost cells of the root
//! level's coarse array (2 nghost root cells; a superset of every exchange's partners).  The
//! shear shift moves data only along x2, so these are all the blocks it can ever exchange with
//! until the next regrid.  LayoutPeerMessages: one message per (peer rank, direction).

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::BuildCandidates()

void ShearingBox::BuildCandidates() {
  Mesh *pm = pmy_pack->pmesh;
  const int nmb = pmy_pack->nmb_thispack;
  const int gids = pmy_pack->gids;
  for (int f=0; f<2; ++f) {
    cand_[f].assign(nmb, std::vector<ShearCand>());
    candsort_[f].assign(nmb, std::vector<int>());
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic) return;
  const int root = pm->root_level;
  const int top = std::max(pm->max_level, root + 1);
  const int nx3 = pm->mb_indcs.nx3, ng = pm->mb_indcs.ng;
  const bool zper = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
  // x3 extents in cells of the level `top`
  const std::int64_t nzt = static_cast<std::int64_t>(pm->nmb_rootx3)*nx3 << (top - root);
  const std::int64_t marg = static_cast<std::int64_t>(2*ng) << (top - root);
  auto zext = [&](int level, std::int32_t lx3, std::int64_t &z0, std::int64_t &z1) {
    const std::int64_t w = static_cast<std::int64_t>(nx3) << (top - level);
    z0 = lx3*w; z1 = z0 + w;
  };
  auto nxl = [pm, root](int l) {return pm->nmb_rootx1 << (l - root);};
  // the local face blocks and what they need
  struct Loc {int m, f; std::int64_t z0, z1;};
  std::vector<Loc> loc;
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[gids + m];
    for (int f=0; f<2; ++f) {
      if (ll.lx1 != ((f == 0) ? 0 : nxl(ll.level) - 1)) continue;
      Loc l; l.m = m; l.f = f;
      zext(ll.level, ll.lx3, l.z0, l.z1);
      l.z0 -= marg; l.z1 += marg;
      loc.push_back(l);
    }
  }
  if (loc.empty()) return;
  // one pass over the block list (AthenaK's own); kept: the overlapping opposite blocks only
  for (int g=0; g<pm->nmb_total; ++g) {
    const auto &ll = pm->lloc_eachmb[g];
    const bool on0 = (ll.lx1 == 0), on1 = (ll.lx1 == nxl(ll.level) - 1);
    if (!on0 && !on1) continue;
    std::int64_t s0, s1;
    zext(ll.level, ll.lx3, s0, s1);
    for (const Loc &l : loc) {
      const bool opp = (l.f == 0) ? on1 : on0;
      if (!opp) continue;
      bool hit = false;
      for (int k=(zper ? -1 : 0); k<=(zper ? 1 : 0) && !hit; ++k) {
        hit = (s0 + k*nzt < l.z1) && (l.z0 < s1 + k*nzt);
      }
      if (hit) {
        cand_[l.f][l.m].push_back({g, pm->rank_eachmb[g], ll.level, ll.lx2, ll.lx3});
      }
    }
  }
  // lookup index by (level, lx3, lx2) for the plans
  for (int f=0; f<2; ++f) {
    candsort_[f].assign(nmb, std::vector<int>());
    for (int m=0; m<nmb; ++m) {
      const auto &cv = cand_[f][m];
      auto &cs = candsort_[f][m];
      cs.resize(cv.size());
      for (size_t i=0; i<cv.size(); ++i) cs[i] = static_cast<int>(i);
      std::sort(cs.begin(), cs.end(), [&cv](int a, int b) {
        return std::make_tuple(cv[a].level, cv[a].lx3, cv[a].lx2) <
               std::make_tuple(cv[b].level, cv[b].lx3, cv[b].lx2);});
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::LayoutPeerMessages()
//! \brief key[i] = {dir, peer, k1, k2, k3, k4, k5, size} of the i-th remote piece: sorted by
//! (dir, peer, k1..k5); offsets are consecutive per direction, one message per (dir, peer)

void ShearingBox::LayoutPeerMessages(const std::vector<std::array<long long, 8>> &key,
                                     std::vector<int> &offset, std::vector<int> &mrank,
                                     std::vector<int> &moff, std::vector<int> &msize,
                                     std::vector<int> &msend, int &soff, int &roff) {
  const int n = static_cast<int>(key.size());
  std::vector<int> idx(n);
  for (int i=0; i<n; ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&key](int a, int b) {
    for (int c=0; c<7; ++c) {
      if (key[a][c] != key[b][c]) return key[a][c] < key[b][c];
    }
    return a < b;});
  offset.assign(n, -1);
  mrank.clear(); moff.clear(); msize.clear(); msend.clear();
  soff = 0; roff = 0;
  for (int q=0; q<n; ++q) {
    const int i = idx[q];
    const int dir = static_cast<int>(key[i][0]), peer = static_cast<int>(key[i][1]);
    int &off = (dir == 1) ? soff : roff;
    if (mrank.empty() || msend.back() != dir || mrank.back() != peer) {
      mrank.push_back(peer); moff.push_back(off); msize.push_back(0); msend.push_back(dir);
    }
    offset[i] = off;
    off += static_cast<int>(key[i][7]);
    msize.back() += static_cast<int>(key[i][7]);
  }
}
