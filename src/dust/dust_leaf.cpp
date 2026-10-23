//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_leaf.cpp
//! \brief The actual-leaf traversal: the block geometry of
//! LeafTraversal (dust.hpp), rebuilt with the leaf maps after every regrid, and the
//! diagnostic comparison with the virtual-finest-grid traversal (<dust>/cloud_leaf_check).

#include <algorithm>
#include <unordered_map>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildLeafTraversal
//! \brief Box, cell size, level and finest-cell box of every block slot (the
//! blocks of this rank, then the halo blocks of other ranks next to them, then all other
//! blocks), computed from the logical locations as AthenaK computes mb_size; and the distinct
//! adjacent blocks (with the periodic shift that places each next to the block) of every
//! MeshBlock of the pack, from the AthenaK neighbour table.  Then the halo exchange lists.

void DustGasDrag::BuildLeafTraversal() {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int gids = pmy_pack->gids;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const int nslot = pmy_pack->pmb->nnghbr;
  if (!pm->two_d || nslot != 24) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "leaf traversal: 2-D meshes only (24 neighbour slots)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // block slots.  Local blocks first (slot = gid - gids), then the blocks of other
  // ranks that a cloud of this rank can reach (halo, ascending gid), then every other block.
  // The reach of block m (HaloReach) follows from the coarsest cloud its centre
  // can hold, widened where CheckHaloReach found a cloud reaching further; the halo holds every
  // block of another rank within that reach of a block of this rank (the
  // neighbours alone do not cover a root-level cloud over a deep mesh), found through the root
  // block that holds each block.
  const int ntot = pm->nmb_total;
  std::vector<int> slot_of(ntot, -1), slot_gid;
  slot_gid.reserve(ntot);
  for (int m=0; m<nmb; ++m) {slot_of[gids + m] = m; slot_gid.push_back(gids + m);}
  // the cells of each halo block that the clouds of this rank can reach (the
  // bounding box over this rank's blocks of the block cut by their reach), by gid
  std::unordered_map<int, std::array<int, 4>> hrect;
  {
    std::vector<int> h;
    for (int m=0; m<nmb; ++m) {
      for (int n=0; n<nslot; ++n) {
        const int g = nghbr.h_view(m, n).gid;
        if (g >= 0 && slot_of[g] < 0) h.push_back(g);
      }
    }
    if (global_variable::nranks > 1) {
      const int nl = smr_transfer.cloud_nlev;
      // the widenings found by CheckHaloReach stay until the mesh changes
      if (halo_need_seq_ != pm->GetAMRLoadBalanceUpdateSeq() ||
          static_cast<int>(halo_need_.size()) != nmb) {
        halo_need_.assign(nmb, 0);
        halo_need_seq_ = pm->GetAMRLoadBalanceUpdateSeq();
      }
      halo_reach_.assign(nmb, 0);
      halo_allow_.assign(nmb, 0);
      for (int m=0; m<nmb; ++m) {
        const int lev = pm->lloc_eachmb[gids + m].level - pm->root_level;
        HaloReach(lev, halo_need_[m], halo_reach_[m], halo_allow_[m]);
      }
      const int nbx = pm->nmb_rootx1, nby = pm->nmb_rootx2;
      const std::int64_t bw1 = static_cast<std::int64_t>(indcs.nx1) << nl;    // root block width
      const std::int64_t bw2 = static_cast<std::int64_t>(indcs.nx2) << nl;
      const std::int64_t nf1 = bw1*nbx, nf2 = bw2*nby;
      auto fb = [&](const int g, std::int64_t b[4]) {
        const LogicalLocation &ll = pm->lloc_eachmb[g];
        const int sh = nl - (ll.level - pm->root_level);
        b[0] = static_cast<std::int64_t>(ll.lx1)*indcs.nx1 << sh;
        b[1] = b[0] + (static_cast<std::int64_t>(indcs.nx1) << sh);
        b[2] = static_cast<std::int64_t>(ll.lx2)*indcs.nx2 << sh;
        b[3] = b[2] + (static_cast<std::int64_t>(indcs.nx2) << sh);
      };
      std::vector<std::vector<int>> bucket(static_cast<std::size_t>(nbx)*nby);
      for (int g=0; g<ntot; ++g) {
        const LogicalLocation &ll = pm->lloc_eachmb[g];
        const int lev = ll.level - pm->root_level;
        bucket[static_cast<std::size_t>(ll.lx2 >> lev)*nbx + (ll.lx1 >> lev)].push_back(g);
      }
      auto overlap = [](std::int64_t a0, std::int64_t a1, std::int64_t c0, std::int64_t c1,
                        std::int64_t n) {
        for (int s=-1; s<=1; ++s) {
          if (a0 + s*n < c1 && c0 < a1 + s*n) return true;
        }
        return false;
      };
      for (int m=0; m<nmb; ++m) {
        std::int64_t mb[4];
        fb(gids + m, mb);
        const std::int64_t rf = halo_reach_[m];   // finest cells
        const int kx = static_cast<int>((rf + bw1 - 1)/bw1), ky = static_cast<int>((rf + bw2 - 1)/bw2);
        const int bx = static_cast<int>(mb[0]/bw1), by = static_cast<int>(mb[2]/bw2);
        for (int dy=-ky; dy<=ky; ++dy) {
          for (int dx=-kx; dx<=kx; ++dx) {
            const int cx = ((bx + dx) % nbx + nbx) % nbx, cy = ((by + dy) % nby + nby) % nby;
            for (const int g : bucket[static_cast<std::size_t>(cy)*nbx + cx]) {
              if (slot_of[g] >= 0) continue;
              std::int64_t gb[4];
              fb(g, gb);
              if (overlap(gb[0], gb[1], mb[0] - rf, mb[1] + rf, nf1) &&
                  overlap(gb[2], gb[3], mb[2] - rf, mb[3] + rf, nf2)) {
                h.push_back(g);
                // the part of g within the reach (every periodic image), in g's cells
                const LogicalLocation &gl = pm->lloc_eachmb[g];
                const std::int64_t ratio = static_cast<std::int64_t>(1) << (nl - (gl.level - pm->root_level));
                std::int64_t r[4] = {gb[1], gb[0], gb[3], gb[2]};   // empty
                for (int sft=-1; sft<=1; ++sft) {
                  const std::int64_t a0 = std::max(gb[0], mb[0] - rf - sft*nf1);
                  const std::int64_t a1 = std::min(gb[1], mb[1] + rf - sft*nf1);
                  if (a0 < a1) {r[0] = std::min(r[0], a0); r[1] = std::max(r[1], a1);}
                  const std::int64_t b0 = std::max(gb[2], mb[2] - rf - sft*nf2);
                  const std::int64_t b1 = std::min(gb[3], mb[3] + rf - sft*nf2);
                  if (b0 < b1) {r[2] = std::min(r[2], b0); r[3] = std::max(r[3], b1);}
                }
                const std::array<int, 4> c = {static_cast<int>((r[0] - gb[0])/ratio),
                                              static_cast<int>((r[1] - gb[0] + ratio - 1)/ratio),
                                              static_cast<int>((r[2] - gb[2])/ratio),
                                              static_cast<int>((r[3] - gb[2] + ratio - 1)/ratio)};
                auto it = hrect.find(g);
                if (it == hrect.end()) {
                  hrect[g] = c;
                } else {
                  it->second = {std::min(it->second[0], c[0]), std::max(it->second[1], c[1]),
                                std::min(it->second[2], c[2]), std::max(it->second[3], c[3])};
                }
              }
            }
          }
        }
      }
    }
    std::sort(h.begin(), h.end());
    h.erase(std::unique(h.begin(), h.end()), h.end());
    for (const int g : h) {slot_of[g] = static_cast<int>(slot_gid.size()); slot_gid.push_back(g);}
  }
  const int nfld = static_cast<int>(slot_gid.size());
  // the rectangle of every halo slot: the reachable cells with cloud_halo_strips, else (and for
  // a block without one, which the reach always gives) the whole block
  halo_.rect.assign(nfld - nmb, {0, indcs.nx1, 0, indcs.nx2, 0, 1});
  if (cloud_halo_strips) {
    for (int b=nmb; b<nfld; ++b) {
      auto it = hrect.find(slot_gid[b]);
      if (it != hrect.end()) {
        auto c = it->second;
        c[0] = std::max(0, c[0]); c[1] = std::min(indcs.nx1, c[1]);
        c[2] = std::max(0, c[2]); c[3] = std::min(indcs.nx2, c[3]);
        if (c[0] < c[1] && c[2] < c[3]) halo_.rect[b - nmb] = {c[0], c[1], c[2], c[3], 0, 1};
      }
    }
  }
  for (int g=0; g<ntot; ++g) {
    if (slot_of[g] < 0) {slot_of[g] = static_cast<int>(slot_gid.size()); slot_gid.push_back(g);}
  }
  const int nall = static_cast<int>(slot_gid.size());
  if (global_variable::nranks == 1 && (nfld != nmb || nall != nmb)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "leaf traversal: one rank but " << nfld - nmb << " halo and " << nall - nfld
              << " other blocks" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  leaf_slot_gid = slot_gid;
  {
    leaf_slot_gid_d = DvceArray1D<int>("dust_slot_gid", nall);
    leaf_slot_rank_d = DvceArray1D<int>("dust_slot_rank", nall);
    auto gh = Kokkos::create_mirror_view(leaf_slot_gid_d);
    auto rh = Kokkos::create_mirror_view(leaf_slot_rank_d);
    for (int b=0; b<nall; ++b) {gh(b) = slot_gid[b]; rh(b) = pm->rank_eachmb[slot_gid[b]];}
    Kokkos::deep_copy(leaf_slot_gid_d, gh);
    Kokkos::deep_copy(leaf_slot_rank_d, rh);
  }
  auto &lt = leaf_traversal;
  lt.nmb = nmb;
  lt.nfld = nfld;
  lt.nall = nall;
  lt.gids = gids;
  lt.nx1 = indcs.nx1; lt.nx2 = indcs.nx2; lt.is = indcs.is; lt.js = indcs.js;
  lt.ncells1 = indcs.nx1 + 2*indcs.ng; lt.ncells2 = indcs.nx2 + 2*indcs.ng;
  lt.x1min = pm->mesh_size.x1min; lt.x1max = pm->mesh_size.x1max;
  lt.x2min = pm->mesh_size.x2min; lt.x2max = pm->mesh_size.x2max;
  lt.lx = lt.x1max - lt.x1min; lt.ly = lt.x2max - lt.x2min;
  lt.ilx = 1.0/lt.lx; lt.ily = 1.0/lt.ly;
  lt.root_dx1 = smr_transfer.root_dx1; lt.root_dx2 = smr_transfer.root_dx2;
  lt.iroot_dx1 = 1.0/lt.root_dx1; lt.iroot_dx2 = 1.0/lt.root_dx2;
  lt.box = DvceArray2D<Real>("dust_leaf_box", nall, 6);
  lt.nnb = DvceArray1D<int>("dust_leaf_nnb", nmb);
  lt.nb = DvceArray2D<int>("dust_leaf_nb", nmb, LeafTraversal::kMaxNb);
  lt.nbs = DvceArray3D<int>("dust_leaf_nbs", nmb, LeafTraversal::kMaxNb, 2);
  auto box = Kokkos::create_mirror_view(lt.box);
  auto nnb = Kokkos::create_mirror_view(lt.nnb);
  auto nb = Kokkos::create_mirror_view(lt.nb);
  auto nbs = Kokkos::create_mirror_view(lt.nbs);
  // the box of every slot as the MeshBlock constructor computes mb_size (meshblock.cpp)
  const auto &ms = pm->mesh_size;
  std::vector<Real> bx1max(nall), bx2max(nall);
  for (int b=0; b<nall; ++b) {
    const LogicalLocation &ll = pm->lloc_eachmb[slot_gid[b]];
    const std::int32_t nmbx1 = pm->nmb_rootx1 << (ll.level - pm->root_level);
    const std::int32_t nmbx2 = pm->nmb_rootx2 << (ll.level - pm->root_level);
    const Real x1a = (ll.lx1 == 0) ? ms.x1min : LeftEdgeX(ll.lx1, nmbx1, ms.x1min, ms.x1max);
    const Real x1b = (ll.lx1 == nmbx1 - 1) ? ms.x1max
                     : LeftEdgeX(ll.lx1 + 1, nmbx1, ms.x1min, ms.x1max);
    const Real x2a = (ll.lx2 == 0) ? ms.x2min : LeftEdgeX(ll.lx2, nmbx2, ms.x2min, ms.x2max);
    const Real x2b = (ll.lx2 == nmbx2 - 1) ? ms.x2max
                     : LeftEdgeX(ll.lx2 + 1, nmbx2, ms.x2min, ms.x2max);
    const Real dx1 = (x1b - x1a)/static_cast<Real>(indcs.nx1);
    const Real dx2 = (x2b - x2a)/static_cast<Real>(indcs.nx2);
    box(b, 0) = x1a; box(b, 1) = x2a; box(b, 2) = dx1; box(b, 3) = dx2;
    box(b, 4) = 1.0/dx1; box(b, 5) = 1.0/dx2;
    bx1max[b] = x1b; bx2max[b] = x2b;
    if (b < nmb) {
      const auto &sz = mbsize.h_view(b);
      if (sz.x1min != x1a || sz.x2min != x2a || sz.dx1 != dx1 || sz.dx2 != dx2 ||
          sz.x1max != x1b || sz.x2max != x2b) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "leaf traversal: the box of block " << gids + b << " from its logical "
                  << "location differs from mb_size" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  for (int m=0; m<nmb; ++m) {
    int c = 0;
    for (int n=0; n<nslot; ++n) {
      const int gid = nghbr.h_view(m, n).gid;
      if (gid < 0) continue;
      const int b = slot_of[gid];
      if (b >= nfld) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "leaf traversal: neighbour " << gid << " of block " << gids + m
                  << " has no halo slot" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // the side of m this slot looks to (2-D slots of nghbr_index.hpp: x1 faces 0-3 and 4-7,
      // x2 faces 8-11 and 12-15, x1x2 edges 16-23), and the periodic shift that puts the
      // neighbour there; the centre distance alone cannot tell the two sides apart when a
      // direction has one or two blocks
      int ix = 0, iy = 0;
      if (n < 4) {ix = -1;} else if (n < 8) {ix = 1;} else if (n < 12) {iy = -1;}
      else if (n < 16) {iy = 1;} else {ix = ((n - 16) & 2) - 1; iy = (((n - 16) & 4) >> 1) - 1;}
      const int sx = static_cast<int>(std::lround(((ix > 0) ? box(b, 0) - bx1max[m]
                                                 : ((ix < 0) ? bx1max[b] - box(m, 0)
                                                             : box(b, 0) - box(m, 0)))/lt.lx));
      const int sy = static_cast<int>(std::lround(((iy > 0) ? box(b, 1) - bx2max[m]
                                                 : ((iy < 0) ? bx2max[b] - box(m, 1)
                                                             : box(b, 1) - box(m, 1)))/lt.ly));
      bool seen = (b == m && sx == 0 && sy == 0);
      for (int k=0; k<c && !seen; ++k) seen = (nb(m, k) == b && nbs(m, k, 0) == sx && nbs(m, k, 1) == sy);
      if (seen) continue;
      if (c >= LeafTraversal::kMaxNb) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "leaf traversal: more than " << LeafTraversal::kMaxNb
                  << " distinct neighbours of block " << m << std::endl;
        std::exit(EXIT_FAILURE);
      }
      nb(m, c) = b; nbs(m, c, 0) = sx; nbs(m, c, 1) = sy;
      ++c;
    }
    nnb(m) = c;
  }
  // finest-level integer boxes, levels, the dilation of the split buffer
  {
    const auto &t = smr_transfer;
    lt.nlev = t.cloud_nlev; lt.nfine1 = t.nfine1; lt.nfine2 = t.nfine2;
    lt.fine_dx1 = t.fine_dx1; lt.fine_dx2 = t.fine_dx2;
    lt.cloud_max_level = t.cloud_max_level;
    lt.split_everywhere = t.cloud_split_everywhere; lt.split_overlap = t.cloud_split_overlap;
    if (lt.nlev > LeafTraversal::kMaxLev) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "leaf traversal: more than " << LeafTraversal::kMaxLev << " levels" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int l=0; l<=LeafTraversal::kMaxLev; ++l) {
      lt.rdil[l] = (cloud_split_buffer > 0 && l >= 1 && l <= lt.nlev)
                 ? (cloud_split_buffer << (lt.nlev - l + 1)) : 0;
    }
    lt.blev = DvceArray1D<int>("dust_leaf_blev", nall);
    lt.fbox = DvceArray2D<int>("dust_leaf_fbox", nall, 4);
    auto blev = Kokkos::create_mirror_view(lt.blev);
    auto fbox = Kokkos::create_mirror_view(lt.fbox);
    for (int b=0; b<nall; ++b) {
      const LogicalLocation &ll = pm->lloc_eachmb[slot_gid[b]];
      const int lev = ll.level - pm->root_level;
      const int ratio = 1 << (lt.nlev - lev);
      blev(b) = lev;
      fbox(b, 0) = static_cast<int>(std::lround((box(b, 0) - lt.x1min)/lt.fine_dx1));
      fbox(b, 1) = fbox(b, 0) + indcs.nx1*ratio;
      fbox(b, 2) = static_cast<int>(std::lround((box(b, 1) - lt.x2min)/lt.fine_dx2));
      fbox(b, 3) = fbox(b, 2) + indcs.nx2*ratio;
      if (lev > lt.nlev || fbox(b, 0) != ll.lx1*indcs.nx1*ratio ||
          fbox(b, 2) != ll.lx2*indcs.nx2*ratio) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "leaf traversal: block " << slot_gid[b] << " (level " << lev
                  << ") does not fit the finest grid of " << lt.nlev << " levels" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
    Kokkos::deep_copy(lt.blev, blev);
    Kokkos::deep_copy(lt.fbox, fbox);
    // the finest level among each local block and its neighbours
    lt.nbmax = DvceArray1D<int>("dust_leaf_nbmax", std::max(1, nmb));
    auto nbmax = Kokkos::create_mirror_view(lt.nbmax);
    for (int m=0; m<nmb; ++m) {
      int mx = blev(m);
      for (int k=0; k<nnb(m); ++k) mx = std::max(mx, blev(nb(m, k)));
      nbmax(m) = mx;
    }
    Kokkos::deep_copy(lt.nbmax, nbmax);
    // the ring around each block that its neighbours certainly hold.  With s the
    // smallest neighbour extent, every s-cell of the ring (aligned with the block) touches the
    // block, so the leaf next to the block inside it is a neighbour, at least s wide and aligned:
    // it holds the whole s-cell.  Checked cell by cell here; a block that fails gets 0 (its
    // queries then take the coverage test).
    lt.safe = DvceArray2D<int>("dust_leaf_safe", nmb, 2);
    auto safe = Kokkos::create_mirror_view(lt.safe);
    int nunsafe = 0;
    for (int m=0; m<nmb; ++m) {
      int s1 = 0, s2 = 0;
      for (int k=0; k<nnb(m); ++k) {
        const int b = nb(m, k);
        const int w = fbox(b, 1) - fbox(b, 0), h = fbox(b, 3) - fbox(b, 2);
        s1 = (k == 0 || w < s1) ? w : s1;
        s2 = (k == 0 || h < s2) ? h : s2;
      }
      const int x0 = fbox(m, 0), x1 = fbox(m, 1), y0 = fbox(m, 2), y1 = fbox(m, 3);
      bool ok = (s1 > 0 && s2 > 0 && (x1 - x0) % s1 == 0 && (y1 - y0) % s2 == 0);
      for (int cy=y0 - s2; ok && cy<y1 + s2; cy+=s2) {
        for (int cx=x0 - s1; ok && cx<x1 + s1; cx+=s1) {
          if (cx >= x0 && cx < x1 && cy >= y0 && cy < y1) continue;
          bool held = false;
          for (int k=0; k<nnb(m) && !held; ++k) {
            const int b = nb(m, k);
            const int bx0 = fbox(b, 0) - nbs(m, k, 0)*lt.nfine1, bx1 = fbox(b, 1) - nbs(m, k, 0)*lt.nfine1;
            const int by0 = fbox(b, 2) - nbs(m, k, 1)*lt.nfine2, by1 = fbox(b, 3) - nbs(m, k, 1)*lt.nfine2;
            held = (bx0 <= cx && cx + s1 <= bx1 && by0 <= cy && cy + s2 <= by1);
          }
          ok = held;
        }
      }
      if (!ok) {s1 = 0; s2 = 0; ++nunsafe;}
      safe(m, 0) = s1; safe(m, 1) = s2;
    }
    Kokkos::deep_copy(lt.safe, safe);
    // the index of logical locations (LeafTraversal::Locate), every slot
    {
      std::vector<std::pair<std::int64_t, int>> kv(nall);
      for (int b=0; b<nall; ++b) {
        const int sh = lt.nlev - blev(b);
        kv[b] = {LeafTraversal::LocKey(blev(b), fbox(b, 0)/(indcs.nx1 << sh),
                                       fbox(b, 2)/(indcs.nx2 << sh)), b};
      }
      std::sort(kv.begin(), kv.end());
      lt.lockey = DvceArray1D<std::int64_t>("dust_leaf_lockey", nall);
      lt.locblk = DvceArray1D<int>("dust_leaf_locblk", nall);
      auto kh = Kokkos::create_mirror_view(lt.lockey);
      auto bh = Kokkos::create_mirror_view(lt.locblk);
      for (int k=0; k<nall; ++k) {kh(k) = kv[k].first; bh(k) = kv[k].second;}
      Kokkos::deep_copy(lt.lockey, kh);
      Kokkos::deep_copy(lt.locblk, bh);
      lt.nloc = nall;
    }
    if (nunsafe > 0 && global_variable::my_rank == 0) {
      std::cout << "# NOTE (dust): leaf traversal: " << nunsafe << " of " << nmb << " blocks have "
                << "no certain neighbour ring (their queries take the coverage test)" << std::endl;
    }
  }
  Kokkos::deep_copy(lt.box, box);
  Kokkos::deep_copy(lt.nnb, nnb);
  Kokkos::deep_copy(lt.nb, nb);
  Kokkos::deep_copy(lt.nbs, nbs);
  // the leaf fields written or read on halo blocks hold nfld blocks
  if (qdep.extent_int(0) < nfld) {
    Kokkos::realloc(qdep, nfld, qdep.extent(1), qdep.extent(2), qdep.extent(3), qdep.extent(4));
    Kokkos::deep_copy(qdep, 0.0);
  }
  if (dmom.extent_int(0) < nfld) {
    Kokkos::realloc(dmom, nfld, dmom.extent(1), dmom.extent(2), dmom.extent(3), dmom.extent(4));
    Kokkos::deep_copy(dmom, 0.0);
  }
  BuildHalo(slot_gid);
  // the reach of every block of this rank, for CheckHaloReach (several ranks)
  halo_allow_d_ = DvceArray1D<std::int64_t>("dust_halo_allow", std::max(nmb, 1));
  if (global_variable::nranks > 1) {
    auto ah = Kokkos::create_mirror_view(halo_allow_d_);
    for (int m=0; m<nmb; ++m) ah(m) = halo_allow_[m];
    Kokkos::deep_copy(halo_allow_d_, ah);
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::HaloReach
//! \brief The reach of a block of level lev (finest cells): reach = 3 cells of
//! the coarsest cloud level Lmin that its clouds can have, and allow = reach minus one cell of
//! the block's level (the drift of half a step before the gather), the furthest a cloud's
//! support may extend beyond the block (CheckHaloReach).  The split runs every cycle before
//! any gather or deposit; with the overlap rule a cloud is never coarser than the block that
//! holds its centre, with the older rule at most one level, and right after a regrid both are
//! one level coarser until the next split (the only traversal then is the post-regrid check):
//! Lmin = lev - 1 (overlap) or lev - 2, never below 0 or above cloud_max_level, and 0 when the
//! split is switched off.  <dust>/cloud_halo_coarser sets the levels instead (test).  need: a
//! widening found by CheckHaloReach.

void DustGasDrag::HaloReach(const int lev, const std::int64_t need, std::int64_t &reach,
                            std::int64_t &allow) const {
  const auto &t = smr_transfer;
  const int nl = t.cloud_nlev;
  const int coarser = (cloud_halo_coarser >= 0) ? cloud_halo_coarser : (t.cloud_split_overlap ? 1 : 2);
  int lmin = std::min(std::max(0, lev - coarser), t.cloud_max_level);
  if (!cloud_split_enable) lmin = 0;
  lmin = std::max(0, std::min(lmin, nl));
  const std::int64_t margin = static_cast<std::int64_t>(1) << (nl - std::min(lev, nl));
  // cloud_halo_test_shrink (test only): the rule's reach divided by 2^shrink, too short on
  // purpose, so that CheckHaloReach has to widen it
  const std::int64_t rule = (static_cast<std::int64_t>(3) << (nl - lmin)) >> cloud_halo_test_shrink;
  reach = std::max(rule, need);
  // the trigger's buffer (cells of this level) lies within the reach
  reach = std::max(reach, static_cast<std::int64_t>(halo_trigger_cells) << (nl - std::min(lev, nl)));
  allow = reach - margin;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckLeafTraversal
//! \brief Diagnostic (<dust>/cloud_leaf_check = N: every N cycles): for every particle, the
//! deposit weights (cloud code) and the gather weights and first moments (gather code) of
//! the actual-leaf traversal against the virtual-finest-grid traversal.  It compares two
//! pseudo-random linear functionals of the per-leaf weights (equal sums mean equal per-leaf
//! totals, with overwhelming probability), plus the weight sum (1 unless the support leaves
//! the neighbours) and the owning leaf.
//! One line per check in cloud_leaf_check.txt.

void DustGasDrag::CheckLeafTraversal() {
  RequireCloudTable("cloud_leaf_check");
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const SMRCompositeTransfer tab = smr_transfer;
  const LeafTraversal lt = leaf_traversal;
  const int tmode = cloud_transition;
  const int gids = pmy_pack->gids;
  const Real inv_dx = 1.0/smr_transfer.root_dx1;
  constexpr Real kTol = 1.0e-12;
  Real ew = 0.0, em = 0.0, es = 0.0;
  int nbadw = 0, nbadm = 0, nbads = 0, nbado = 0, nlobed = 0, nglob = 0, nprobe = 0;
  Kokkos::parallel_reduce("dust_leaf_check", Kokkos::RangePolicy<DevExeSpace>(0, npart),
  KOKKOS_LAMBDA(const int p, Real &dw, Real &dm, Real &ds, int &bw, int &bm, int &bs, int &bo,
                int &nlb, int &ng, int &np) {
    auto coef = [](const int target, const Real s) {
      Real v = sin(static_cast<Real>(target)*12.9898 + s)*43758.5453;
      return v - floor(v);
    };
    const Real x1 = pr(IPX,p), x2 = pr(IPY,p);
    const int code = pi(PLEV,p), m = pi(PGID,p) - gids;
    const int gcode = SMRCompositeTransfer::GatherCode(code, tmode);
    // the particle, and four positions half a cloud width away with the particle's block as
    // the anchor (children of a split, half-step positions: they may lie in a neighbour or
    // across a periodic edge)
    const Real h1 = tab.CloudHalfWidth1(code), h2 = tab.CloudHalfWidth2(code);
    for (int q=0; q<5; ++q) {
      const Real px = x1 + ((q == 1) ? -h1 : ((q == 2) ? h1 : 0.0));
      const Real py = x2 + ((q == 3) ? -h2 : ((q == 4) ? h2 : 0.0));
      Real tw[2] = {0.0, 0.0}, tm[2] = {0.0, 0.0};
      tab.ForEachCloudWeight(px, py, code, [&](const int target, const Real w) {
        tw[0] += w*coef(target, 0.3); tw[1] += w*coef(target, 4.1);
      });
      tab.ForEachCloudMoment(px, py, gcode,
      [&](const int target, const Real w, const Real m1x, const Real m1y) {
        tm[0] += w*coef(target, 0.3) + inv_dx*(m1x*coef(target, 1.7) + m1y*coef(target, 2.9));
        tm[1] += w*coef(target, 5.3) + inv_dx*(m1x*coef(target, 6.7) + m1y*coef(target, 7.9));
      });
      Real lw[2] = {0.0, 0.0}, lmo[2] = {0.0, 0.0};
      bool reach = true;
      const Real sum = lt.ForEach<false>(px, py, code, m, [&](const int target, const Real w) {
        lw[0] += w*coef(target, 0.3); lw[1] += w*coef(target, 4.1);
      }, &reach);
      lt.ForEach<true>(px, py, gcode, m,
      [&](const int target, const Real w, const Real m1x, const Real m1y) {
        lmo[0] += w*coef(target, 0.3) + inv_dx*(m1x*coef(target, 1.7) + m1y*coef(target, 2.9));
        lmo[1] += w*coef(target, 5.3) + inv_dx*(m1x*coef(target, 6.7) + m1y*coef(target, 7.9));
      });
      const Real d1 = fmax(fabs(tw[0] - lw[0]), fabs(tw[1] - lw[1]));
      const Real d2 = fmax(fabs(tm[0] - lmo[0]), fabs(tm[1] - lmo[1]));
      const Real d3 = fabs(sum - 1.0);
      dw = fmax(dw, d1); dm = fmax(dm, d2); ds = fmax(ds, d3);
      if (d1 > kTol) ++bw;
      if (d2 > kTol) ++bm;
      if (d3 > kTol) ++bs;
      if (lt.Owner(px, py, m) != tab.LeafOwner(px, py)) ++bo;
      if (!reach) ++ng;
      ++np;
    }
    if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) ++nlb;
  }, Kokkos::Max<Real>(ew), Kokkos::Max<Real>(em), Kokkos::Max<Real>(es), Kokkos::Sum<int>(nbadw),
     Kokkos::Sum<int>(nbadm), Kokkos::Sum<int>(nbads), Kokkos::Sum<int>(nbado),
     Kokkos::Sum<int>(nlobed), Kokkos::Sum<int>(nglob), Kokkos::Sum<int>(nprobe));
  if (global_variable::my_rank == 0) {
    std::FILE *fp = std::fopen("cloud_leaf_check.txt", "a");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# ncycle time particles max_dweight max_dmoment max_dsum bad_weight "
                     "bad_moment bad_sum bad_owner lobed probes global (bad: > 1e-12; 5 probes per "
                     "particle)\n");
      }
      std::fprintf(fp, "%d %.17g %d %.3e %.3e %.3e %d %d %d %d %d %d %d\n", pmy_pack->pmesh->ncycle,
                   pmy_pack->pmesh->time, npart, ew, em, es, nbadw, nbadm, nbads, nbado, nlobed,
                   nprobe, nglob);
      std::fclose(fp);
    }
    std::cout << "# leaf traversal check cycle " << pmy_pack->pmesh->ncycle << ": "
              << npart << " particles, max |d weights| " << ew << ", max |d moments| " << em
              << ", max |weight sum - 1| " << es << "; over 1e-12: " << nbadw << " / " << nbadm
              << " / " << nbads << ", owner differs " << nbado << std::endl;
  }
  CheckLeafEligibility();
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckLeafEligibility
//! \brief Part of cloud_leaf_check: for every particle the split / release
//! tests (the probes of P1Flag: the cloud itself; level - 1 at the particle and half a
//! level-(lev-1) width away in each direction), CloudOverlapsFiner, the merge-cell mask of
//! every level at the particle and one cell away in each direction, on the actual leaves
//! against the table (masks taken directly from the level map as MergeCloudsDevice builds
//! them; the stage C mask was checked too until stage C was removed on 2026-10-20).
//! Counts the differing answers, the true answers of the table (so that the test is not
//! vacuous), and the probes whose box leaves the anchor block and its neighbours.  One line
//! per check in cloud_leaf_elig.txt.

void DustGasDrag::CheckLeafEligibility() {
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const SMRCompositeTransfer tab = smr_transfer;
  const LeafTraversal lt = leaf_traversal;
  const int gids = pmy_pack->gids;
  const Real slack = cloud_split_slack;
  const int B = cloud_merge_buffer;
  const int nlev = tab.cloud_nlev, nf1 = tab.nfine1, nf2 = tab.nfine2;
  int d_ins = 0, d_ovl = 0, d_mrg = 0, t_ins = 0, t_ovl = 0, t_mrg = 0;
  int n_ins = 0, n_mrg = 0, nreach = 0;
  Kokkos::parallel_reduce("dust_leaf_elig", Kokkos::RangePolicy<DevExeSpace>(0, npart),
  KOKKOS_LAMBDA(const int p, int &di, int &dov, int &dmr, int &ti, int &tov, int &tmr,
                int &ni, int &nm, int &nr) {
    using T = SMRCompositeTransfer;
    const Real x = pr(IPX,p), y = pr(IPY,p);
    const int code = pi(PLEV,p), m = pi(PGID,p) - gids;
    const int lev = T::CloudLevel(code);
    // split / release probes
    auto probe = [&](const Real px, const Real py, const int c) {
      bool reach = true;
      const bool a = tab.CloudInsideFiner(px, py, c, slack);
      const bool b = lt.InsideFiner(px, py, c, slack, m, &reach);
      ++ni;
      if (a) ++ti;
      if (a != b) ++di;
      if (!reach) ++nr;
    };
    probe(x, y, code);
    if (SMRCompositeTransfer::CloudLobe1(code) == 0 && SMRCompositeTransfer::CloudLobe2(code) == 0) {
      // P1Event of a plain cloud: its own level, half its width away in each direction
      const Real g1 = tab.CloudHalfWidth1(lev), g2 = tab.CloudHalfWidth2(lev);
      probe(x - g1, y, lev); probe(x + g1, y, lev); probe(x, y - g2, lev); probe(x, y + g2, lev);
    }
    if (lev > 0) {
      const Real h1 = tab.CloudHalfWidth1(lev - 1), h2 = tab.CloudHalfWidth2(lev - 1);
      probe(x, y, lev - 1);
      probe(x - h1, y, lev - 1); probe(x + h1, y, lev - 1);
      probe(x, y - h2, lev - 1); probe(x, y + h2, lev - 1);
    }
    {
      bool reach = true;
      const bool a = tab.CloudOverlapsFiner(x, y, code), b = lt.OverlapsFiner(x, y, code, m, &reach);
      if (a) ++tov;
      if (a != b) ++dov;
      if (!reach) ++nr;
    }
    // merge-cell masks
    const Real xw = T::WrapPosition(x, tab.x1min, tab.x1max), yw = T::WrapPosition(y, tab.x2min, tab.x2max);
    for (int L=0; L<nlev; ++L) {
      const int r = 1 << (nlev - L), n1 = nf1/r, n2 = nf2/r;
      const Real dl1 = tab.root_dx1/static_cast<Real>(1 << L), dl2 = tab.root_dx2/static_cast<Real>(1 << L);
      for (int q=0; q<5; ++q) {
        const Real ox = (q == 1) ? -dl1 : ((q == 2) ? dl1 : 0.0);
        const Real oy = (q == 3) ? -dl2 : ((q == 4) ? dl2 : 0.0);
        const Real qx = T::WrapPosition(xw + ox, tab.x1min, tab.x1max);
        const Real qy = T::WrapPosition(yw + oy, tab.x2min, tab.x2max);
        const int I = T::WrapIndex(static_cast<int>(floor((qx - tab.x1min)/dl1)), n1);
        const int J = T::WrapIndex(static_cast<int>(floor((qy - tab.x2min)/dl2)), n2);
        bool ok = true;
        for (int fj=(J - B)*r; fj<(J + B + 1)*r && ok; ++fj) {
          const int wj = T::WrapIndex(fj, nf2);
          for (int fi=(I - B)*r; fi<(I + B + 1)*r; ++fi) {
            if (tab.cloud_level(wj, T::WrapIndex(fi, nf1)) > L) {ok = false; break;}
          }
        }
        bool reach = true;
        const bool b = lt.MergeOK(xw + ox, yw + oy, L, B, m, &reach);
        if (!reach) ++nr;
        ++nm;
        if (ok) ++tmr;
        if (ok != b) ++dmr;
      }
    }
  }, d_ins, d_ovl, d_mrg, t_ins, t_ovl, t_mrg, n_ins, n_mrg, nreach);
  if (global_variable::my_rank == 0) {
    std::FILE *fp = std::fopen("cloud_leaf_elig.txt", "a");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# ncycle time particles split_probes split_differ split_true overlap_differ "
                     "overlap_true merge_probes merge_differ merge_true global_answers (probes "
                     "beyond the anchor's neighbourhood; the compress columns before "
                     "global_answers went with stage C on 2026-10-20)\n");
      }
      std::fprintf(fp, "%d %.17g %d %d %d %d %d %d %d %d %d %d\n", pmy_pack->pmesh->ncycle,
                   pmy_pack->pmesh->time, npart, n_ins, d_ins, t_ins, d_ovl, t_ovl, n_mrg, d_mrg, t_mrg,
                   nreach);
      std::fclose(fp);
    }
    std::cout << "# leaf eligibility check cycle " << pmy_pack->pmesh->ncycle
              << ": split probes " << n_ins << " differ " << d_ins << " (table true " << t_ins
              << "), overlap differ " << d_ovl << " (true " << t_ovl << "), merge probes " << n_mrg
              << " differ " << d_mrg << " (true " << t_mrg << "), out of reach " << nreach
              << std::endl;
  }
}

}  // namespace dust
