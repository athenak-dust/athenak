//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_leaf.cpp
//! \brief The actual-leaf traversal: the block geometry of
//! LeafTraversal (dust.hpp), rebuilt with the leaf maps after every regrid, and the
//! diagnostic comparison with the virtual-finest-grid traversal (<dust>/cloud_leaf_check).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

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
//! \brief Box, cell size and the distinct adjacent blocks (with the periodic shift that
//! places each next to the block) of every MeshBlock of the pack, from mb_size and the
//! AthenaK neighbour table.  One rank: a neighbour must be a block of this pack.

void DustGasDrag::BuildLeafTraversal() {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const int nslot = pmy_pack->pmb->nnghbr;
  if (!pm->two_d || nslot != 24) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "leaf traversal: 2-D meshes only (24 neighbour slots)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &lt = leaf_traversal;
  lt.nmb = nmb;
  lt.gids = pmy_pack->gids;
  lt.nx1 = indcs.nx1; lt.nx2 = indcs.nx2; lt.is = indcs.is; lt.js = indcs.js;
  lt.ncells1 = indcs.nx1 + 2*indcs.ng; lt.ncells2 = indcs.nx2 + 2*indcs.ng;
  lt.x1min = pm->mesh_size.x1min; lt.x1max = pm->mesh_size.x1max;
  lt.x2min = pm->mesh_size.x2min; lt.x2max = pm->mesh_size.x2max;
  lt.lx = lt.x1max - lt.x1min; lt.ly = lt.x2max - lt.x2min;
  lt.ilx = 1.0/lt.lx; lt.ily = 1.0/lt.ly;
  lt.root_dx1 = smr_transfer.root_dx1; lt.root_dx2 = smr_transfer.root_dx2;
  lt.iroot_dx1 = 1.0/lt.root_dx1; lt.iroot_dx2 = 1.0/lt.root_dx2;
  lt.lobe3_integral = smr_transfer.lobe3_integral;
  lt.box = DvceArray2D<Real>("dust_leaf_box", nmb, 6);
  lt.nnb = DvceArray1D<int>("dust_leaf_nnb", nmb);
  lt.nb = DvceArray2D<int>("dust_leaf_nb", nmb, LeafTraversal::kMaxNb);
  lt.nbs = DvceArray3D<int>("dust_leaf_nbs", nmb, LeafTraversal::kMaxNb, 2);
  auto box = Kokkos::create_mirror_view(lt.box);
  auto nnb = Kokkos::create_mirror_view(lt.nnb);
  auto nb = Kokkos::create_mirror_view(lt.nb);
  auto nbs = Kokkos::create_mirror_view(lt.nbs);
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    box(m, 0) = sz.x1min; box(m, 1) = sz.x2min; box(m, 2) = sz.dx1; box(m, 3) = sz.dx2;
    box(m, 4) = 1.0/sz.dx1; box(m, 5) = 1.0/sz.dx2;
  }
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    int c = 0;
    for (int n=0; n<nslot; ++n) {
      const int gid = nghbr.h_view(m, n).gid;
      if (gid < 0) continue;
      const int b = gid - lt.gids;
      if (b < 0 || b >= nmb) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "leaf traversal: neighbour " << gid << " of block " << m
                  << " is not in this pack (one MPI rank only)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      // the side of m this slot looks to (2-D slots of nghbr_index.hpp: x1 faces 0-3 and 4-7,
      // x2 faces 8-11 and 12-15, x1x2 edges 16-23), and the periodic shift that puts the
      // neighbour there; the centre distance alone cannot tell the two sides apart when a
      // direction has one or two blocks
      int ix = 0, iy = 0;
      if (n < 4) {ix = -1;} else if (n < 8) {ix = 1;} else if (n < 12) {iy = -1;}
      else if (n < 16) {iy = 1;} else {ix = ((n - 16) & 2) - 1; iy = (((n - 16) & 4) >> 1) - 1;}
      const auto &nz = mbsize.h_view(b);
      const int sx = static_cast<int>(std::lround(((ix > 0) ? nz.x1min - sz.x1max
                                                 : ((ix < 0) ? nz.x1max - sz.x1min
                                                             : nz.x1min - sz.x1min))/lt.lx));
      const int sy = static_cast<int>(std::lround(((iy > 0) ? nz.x2min - sz.x2max
                                                 : ((iy < 0) ? nz.x2max - sz.x2min
                                                             : nz.x2min - sz.x2min))/lt.ly));
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
    lt.blev = DvceArray1D<int>("dust_leaf_blev", nmb);
    lt.fbox = DvceArray2D<int>("dust_leaf_fbox", nmb, 4);
    auto blev = Kokkos::create_mirror_view(lt.blev);
    auto fbox = Kokkos::create_mirror_view(lt.fbox);
    auto &mblev = pmy_pack->pmb->mb_lev;
    for (int m=0; m<nmb; ++m) {
      const auto &sz = mbsize.h_view(m);
      const int lev = mblev.h_view(m) - pm->root_level;
      const int ratio = 1 << (lt.nlev - lev);
      blev(m) = lev;
      fbox(m, 0) = static_cast<int>(std::lround((sz.x1min - lt.x1min)/lt.fine_dx1));
      fbox(m, 1) = fbox(m, 0) + indcs.nx1*ratio;
      fbox(m, 2) = static_cast<int>(std::lround((sz.x2min - lt.x2min)/lt.fine_dx2));
      fbox(m, 3) = fbox(m, 2) + indcs.nx2*ratio;
    }
    Kokkos::deep_copy(lt.blev, blev);
    Kokkos::deep_copy(lt.fbox, fbox);
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
    if (nunsafe > 0 && global_variable::my_rank == 0) {
      std::cout << "# NOTE (dust): leaf traversal: " << nunsafe << " of " << nmb << " blocks have "
                << "no certain neighbour ring (their queries take the coverage test)" << std::endl;
    }
  }
  Kokkos::deep_copy(lt.box, box);
  Kokkos::deep_copy(lt.nnb, nnb);
  Kokkos::deep_copy(lt.nb, nb);
  Kokkos::deep_copy(lt.nbs, nbs);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckLeafTraversal
//! \brief Diagnostic (<dust>/cloud_leaf_check = N: every N cycles): for every particle, the
//! deposit weights (cloud code) and the gather weights and first moments (gather code) of
//! the actual-leaf traversal against the virtual-finest-grid traversal.  As the native
//! check (CheckNativeTransfer) it compares two pseudo-random linear functionals of the
//! per-leaf weights (equal sums mean equal per-leaf totals, with overwhelming probability),
//! plus the weight sum (1 unless the support leaves the neighbours) and the owning leaf.
//! One line per check in cloud_leaf_check.txt.

void DustGasDrag::CheckLeafTraversal() {
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
//! every level at the particle and one cell away in each direction, and the stage C mask of
//! every level at the particle, on the actual leaves against the table (masks taken directly
//! from the level map as MergeCloudsDevice builds them).  Counts the differing answers, the
//! true answers of the table (so that the test is not vacuous), and the probes whose box
//! leaves the anchor block and its neighbours.  One line per check in cloud_leaf_elig.txt.

void DustGasDrag::CheckLeafEligibility() {
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const SMRCompositeTransfer tab = smr_transfer;
  const LeafTraversal lt = leaf_traversal;
  const int gids = pmy_pack->gids;
  const Real slack = cloud_split_slack;
  const int B = cloud_merge_buffer, cbuf = cloud_compress_buffer;
  const int nlev = tab.cloud_nlev, nf1 = tab.nfine1, nf2 = tab.nfine2;
  int d_ins = 0, d_ovl = 0, d_mrg = 0, d_cmp = 0, t_ins = 0, t_ovl = 0, t_mrg = 0, t_cmp = 0;
  int n_ins = 0, n_mrg = 0, n_cmp = 0, nreach = 0;
  Kokkos::parallel_reduce("dust_leaf_elig", Kokkos::RangePolicy<DevExeSpace>(0, npart),
  KOKKOS_LAMBDA(const int p, int &di, int &dov, int &dmr, int &dcm, int &ti, int &tov, int &tmr,
                int &tcm, int &ni, int &nm, int &nc, int &nr) {
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
    // stage C masks
    for (int l=1; l<=nlev && l<5; ++l) {
      const int r = 1 << (nlev - l), bf = 2*cbuf*r;
      const int I = static_cast<int>(floor((xw - tab.x1min)/(tab.fine_dx1*r)));
      const int J = static_cast<int>(floor((yw - tab.x2min)/(tab.fine_dx2*r)));
      int mn = 1 << 20;
      for (int fj=J*r - bf; fj<J*r + r + bf; ++fj) {
        const int wj = T::WrapIndex(fj, nf2);
        for (int fi=I*r - bf; fi<I*r + r + bf; ++fi) {
          const int v = tab.cloud_level(wj, T::WrapIndex(fi, nf1));
          mn = (v < mn) ? v : mn;
        }
      }
      bool reach = true;
      const bool a = (mn >= l), b = lt.CompressOK(x, y, l, cbuf, m, &reach);
      if (!reach) ++nr;
      ++nc;
      if (a) ++tcm;
      if (a != b) ++dcm;
    }
  }, d_ins, d_ovl, d_mrg, d_cmp, t_ins, t_ovl, t_mrg, t_cmp, n_ins, n_mrg, n_cmp, nreach);
  if (global_variable::my_rank == 0) {
    std::FILE *fp = std::fopen("cloud_leaf_elig.txt", "a");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# ncycle time particles split_probes split_differ split_true overlap_differ "
                     "overlap_true merge_probes merge_differ merge_true compress_probes "
                     "compress_differ compress_true global_answers (probes beyond the anchor's "
                     "neighbourhood)\n");
      }
      std::fprintf(fp, "%d %.17g %d %d %d %d %d %d %d %d %d %d %d %d %d\n", pmy_pack->pmesh->ncycle,
                   pmy_pack->pmesh->time, npart, n_ins, d_ins, t_ins, d_ovl, t_ovl, n_mrg, d_mrg, t_mrg,
                   n_cmp, d_cmp, t_cmp, nreach);
      std::fclose(fp);
    }
    std::cout << "# leaf eligibility check cycle " << pmy_pack->pmesh->ncycle
              << ": split probes " << n_ins << " differ " << d_ins << " (table true " << t_ins
              << "), overlap differ " << d_ovl << " (true " << t_ovl << "), merge probes " << n_mrg
              << " differ " << d_mrg << " (true " << t_mrg << "), compress probes " << n_cmp
              << " differ " << d_cmp << " (true " << t_cmp << "), out of reach " << nreach
              << std::endl;
  }
}

}  // namespace dust
