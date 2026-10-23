//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud3d.cpp
//! \brief Stage S3b: refinable clouds on a 3-D mesh
//! (<dust>/experimental_3d).  The 2-D paths are not touched; in 3-D these replace them:
//!  - BuildLeafTraversal3D: the block geometry of LeafTraversal3D (dust_leaf3d.hpp);
//!  - CheckLeafTraversal3D: the traversal against a brute-force integration over the finest
//!    cells (weights, first moments, the split eligibility);
//!  - SplitCloudsP1Device3D: the moving P1 split, the 2-D rules along each of three directions;
//!  - CloudPredictor3D / CloudAdvance3D: the two production PC2 cloud kernels (linear gather);
//!  - DepositCloudDensity3D: dust density and momentum on the leaves (diagnostics).
//! Stage 1: one rank, static mesh, split only.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "mesh/nghbr_index.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"

namespace dust {

namespace {

constexpr int kCache3 = 27;   // leaf weights kept from the gather (3 x 3 x 3: a same-level cloud)

struct DPiece3 {Real off; Real frac; int lobe;};

// pieces of a cloud along one direction (as P1Pieces in dust_cloud_device.cpp)
KOKKOS_INLINE_FUNCTION
int P1Pieces3(const int lobe_old, const bool sep_lo, const bool sep_hi, DPiece3 out[3]) {
  int n = 0;
  const Real total = 0.5 + 0.25*((lobe_old & 1) ? 1 : 0) + 0.25*((lobe_old & 2) ? 1 : 0);
  const bool lo = (lobe_old & 1) && sep_lo, hi = (lobe_old & 2) && sep_hi;
  const int lobe_new = lobe_old & ~((lo ? 1 : 0) | (hi ? 2 : 0));
  const Real keep = 0.5 + 0.25*((lobe_new & 1) ? 1 : 0) + 0.25*((lobe_new & 2) ? 1 : 0);
  out[n].off = 0.0; out[n].frac = keep/total; out[n].lobe = lobe_new; ++n;
  if (lo) {out[n].off = -1.0; out[n].frac = 0.25/total; out[n].lobe = 0; ++n;}
  if (hi) {out[n].off = 1.0; out[n].frac = 0.25/total; out[n].lobe = 0; ++n;}
  return n;
}

// the MeshBlock of a new cloud, searched from the block m of its parent
KOKKOS_INLINE_FUNCTION
int NewCloudBlock3(const LeafTraversal3D &lt, const Real x, const Real y, const Real z,
                   const int m, const int ncell123) {
  const int leaf = lt.LeafOwner(x, y, z, m);
  const int b = (leaf < 0) ? m : leaf/ncell123;
  return (b < lt.nmb) ? b : m;
}

// is the particle split (or a lobe released) now?  The 2-D rule (P1Flag) per direction.
KOKKOS_INLINE_FUNCTION
bool P1Flag3(const LeafTraversal3D &t, const int m, const int code, const Real x, const Real y,
             const Real z, const Real slack) {
  using T = SMRCompositeTransfer;
  const int lev = T::CloudLevel(code);
  const int l1 = T::CloudLobe1(code), l2 = T::CloudLobe2(code), l3 = T::CloudLobe3(code);
  if (l1 == 0 && l2 == 0 && l3 == 0) return t.CloudInsideFiner(x, y, z, lev, slack, m);
  if (lev > 0 && t.CloudInsideFiner(x, y, z, lev - 1, slack, m)) {
    const Real h1 = t.CloudHalfWidth1(lev - 1), h2 = t.CloudHalfWidth2(lev - 1);
    const Real h3 = t.CloudHalfWidth3(lev - 1);
    return ((l1 & 1) && t.CloudInsideFiner(x - h1, y, z, lev - 1, slack, m)) ||
           ((l1 & 2) && t.CloudInsideFiner(x + h1, y, z, lev - 1, slack, m)) ||
           ((l2 & 1) && t.CloudInsideFiner(x, y - h2, z, lev - 1, slack, m)) ||
           ((l2 & 2) && t.CloudInsideFiner(x, y + h2, z, lev - 1, slack, m)) ||
           ((l3 & 1) && t.CloudInsideFiner(x, y, z - h3, lev - 1, slack, m)) ||
           ((l3 & 2) && t.CloudInsideFiner(x, y, z + h3, lev - 1, slack, m));
  }
  return false;
}

// pieces of a flagged particle along the three directions (P1Event per direction)
KOKKOS_INLINE_FUNCTION
void P1Event3(const LeafTraversal3D &t, const int m, const int code, const Real x, const Real y,
              const Real z, const Real slack, DPiece3 p1[3], DPiece3 p2[3], DPiece3 p3[3],
              int &n1, int &n2, int &n3, int &clev, bool &parent) {
  using T = SMRCompositeTransfer;
  const int lev = T::CloudLevel(code);
  const int l1 = T::CloudLobe1(code), l2 = T::CloudLobe2(code), l3 = T::CloudLobe3(code);
  if (l1 == 0 && l2 == 0 && l3 == 0) {
    const Real h1 = t.CloudHalfWidth1(lev), h2 = t.CloudHalfWidth2(lev);
    const Real h3 = t.CloudHalfWidth3(lev);
    n1 = P1Pieces3(3, t.CloudInsideFiner(x - h1, y, z, lev, slack, m),
                   t.CloudInsideFiner(x + h1, y, z, lev, slack, m), p1);
    n2 = P1Pieces3(3, t.CloudInsideFiner(x, y - h2, z, lev, slack, m),
                   t.CloudInsideFiner(x, y + h2, z, lev, slack, m), p2);
    n3 = P1Pieces3(3, t.CloudInsideFiner(x, y, z - h3, lev, slack, m),
                   t.CloudInsideFiner(x, y, z + h3, lev, slack, m), p3);
    clev = lev + 1;
    parent = true;
  } else {
    const Real h1 = t.CloudHalfWidth1(lev - 1), h2 = t.CloudHalfWidth2(lev - 1);
    const Real h3 = t.CloudHalfWidth3(lev - 1);
    n1 = P1Pieces3(l1, t.CloudInsideFiner(x - h1, y, z, lev - 1, slack, m),
                   t.CloudInsideFiner(x + h1, y, z, lev - 1, slack, m), p1);
    n2 = P1Pieces3(l2, t.CloudInsideFiner(x, y - h2, z, lev - 1, slack, m),
                   t.CloudInsideFiner(x, y + h2, z, lev - 1, slack, m), p2);
    n3 = P1Pieces3(l3, t.CloudInsideFiner(x, y, z - h3, lev - 1, slack, m),
                   t.CloudInsideFiner(x, y, z + h3, lev - 1, slack, m), p3);
    clev = lev;
    parent = false;
  }
}

std::uint64_t Mix64(std::uint64_t x) {
  x += UINT64_C(0x9e3779b97f4a7c15);
  x = (x ^ (x >> 30))*UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27))*UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}
double Uniform01(const std::uint64_t key) {
  return static_cast<double>(Mix64(key) >> 11)*(1.0/9007199254740992.0);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckClouds3D
//! \brief Refinable clouds in 3-D need <dust>/experimental_3d; stage 1 runs the
//! production split and PC2 kernels on one rank and a static mesh, without G and R.

void DustGasDrag::CheckClouds3D() {
  if (!pmy_pack->pmesh->three_d || smr_transfer_mode != DustSMRTransfer::refinable_cloud) return;
  if (!experimental_3d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "refinable clouds on a 3-D mesh need <dust>/experimental_3d = true"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::vector<std::string> bad;
  if (global_variable::nranks != 1) bad.push_back("several MPI ranks");
  if (pmy_pack->pmesh->adaptive) bad.push_back("an adaptive mesh");
  if (!(coupling == DustCoupling::hybrid && hybrid_force_mode == HybridForceMode::pc2)) {
    bad.push_back("a coupling other than PC2 (coupling = pc2)");
  }
  if (drag_solver != DustDragSolver::local) bad.push_back("drag_solver other than local");
  if (cloud_gather_order != 2 || analytic_gather.on) {
    bad.push_back("gather variants (cloud_gather_order != 2, analytic_gather)");
  }
  if (cloud_merge) bad.push_back("G (cloud_merge = true)");
  if (cloud_resample) bad.push_back("R (cloud_resample = true)");
  if (!cloud_split_device || cloud_split_log || split_oracle) bad.push_back("the host split");
  if (cloud_instant_switch) bad.push_back("cloud_instant_switch");
  if (cloud_sort_interval > 0 || cloud_order_interval > 0) {
    bad.push_back("the particle sort or order index (cloud_sort_interval, cloud_order_interval)");
  }
  if (!bad.empty()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D refinable clouds (stage 1) do not support:";
    for (const auto &b : bad) std::cout << " [" << b << "]";
    std::cout << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "# NOTE (dust): 3-D refinable clouds (stage 1: one rank, static mesh, "
              << "split only, PC2)" << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildLeafTraversal3D
//! \brief The block geometry of LeafTraversal3D: box, cell size, level and finest-cell box of
//! every block (from the logical locations, as AthenaK computes mb_size), the distinct adjacent
//! blocks of every block with the periodic shift that places each next to it (from the 56
//! neighbour slots), the ring its neighbours certainly hold, and the index of logical
//! locations.  Stage 1: one rank (every block is local).

void DustGasDrag::BuildLeafTraversal3D() {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int gids = pmy_pack->gids;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const int nslot = pmy_pack->pmb->nnghbr;
  if (!pm->three_d || nslot != 56) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D leaf traversal: 3-D meshes only (56 neighbour slots)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::nranks != 1 || pm->nmb_total != nmb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D refinable clouds: one MPI rank only (stage 1)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nall = nmb;
  auto &lt = leaf_traversal3d;
  const auto &t = smr_transfer;
  const auto &ms = pm->mesh_size;
  lt.nmb = nmb; lt.nfld = nall; lt.nall = nall; lt.gids = gids;
  lt.nx1 = indcs.nx1; lt.nx2 = indcs.nx2; lt.nx3 = indcs.nx3;
  lt.is = indcs.is; lt.js = indcs.js; lt.ks = indcs.ks;
  lt.ncells1 = indcs.nx1 + 2*indcs.ng; lt.ncells2 = indcs.nx2 + 2*indcs.ng;
  lt.ncells3 = indcs.nx3 + 2*indcs.ng;
  lt.x1min = ms.x1min; lt.x1max = ms.x1max; lt.x2min = ms.x2min; lt.x2max = ms.x2max;
  lt.x3min = ms.x3min; lt.x3max = ms.x3max;
  lt.lx = lt.x1max - lt.x1min; lt.ly = lt.x2max - lt.x2min; lt.lz = lt.x3max - lt.x3min;
  lt.ilx = 1.0/lt.lx; lt.ily = 1.0/lt.ly; lt.ilz = 1.0/lt.lz;
  lt.root_dx1 = t.root_dx1; lt.root_dx2 = t.root_dx2;
  lt.root_dx3 = lt.lz/static_cast<Real>(pm->mesh_indcs.nx3);
  lt.iroot_dx1 = 1.0/lt.root_dx1; lt.iroot_dx2 = 1.0/lt.root_dx2; lt.iroot_dx3 = 1.0/lt.root_dx3;
  lt.nlev = t.cloud_nlev;
  lt.nfine1 = t.nfine1; lt.nfine2 = t.nfine2; lt.nfine3 = pm->mesh_indcs.nx3 << lt.nlev;
  lt.fine_dx1 = t.fine_dx1; lt.fine_dx2 = t.fine_dx2;
  lt.fine_dx3 = lt.root_dx3/static_cast<Real>(1 << lt.nlev);
  lt.cloud_max_level = t.cloud_max_level;
  lt.split_everywhere = t.cloud_split_everywhere; lt.split_overlap = t.cloud_split_overlap;
  if (lt.nlev > LeafTraversal3D::kMaxLev) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D leaf traversal: more than " << LeafTraversal3D::kMaxLev << " levels"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int l=0; l<=LeafTraversal3D::kMaxLev; ++l) {
    lt.rdil[l] = (cloud_split_buffer > 0 && l >= 1 && l <= lt.nlev)
               ? (cloud_split_buffer << (lt.nlev - l + 1)) : 0;
  }
  lt.box = DvceArray2D<Real>("dust_leaf3d_box", nall, 9);
  lt.nnb = DvceArray1D<int>("dust_leaf3d_nnb", std::max(1, nmb));
  lt.nb = DvceArray2D<int>("dust_leaf3d_nb", std::max(1, nmb), LeafTraversal3D::kMaxNb);
  lt.nbs = DvceArray3D<int>("dust_leaf3d_nbs", std::max(1, nmb), LeafTraversal3D::kMaxNb, 3);
  lt.blev = DvceArray1D<int>("dust_leaf3d_blev", nall);
  lt.fbox = DvceArray2D<int>("dust_leaf3d_fbox", nall, 6);
  auto box = Kokkos::create_mirror_view(lt.box);
  auto nnb = Kokkos::create_mirror_view(lt.nnb);
  auto nb = Kokkos::create_mirror_view(lt.nb);
  auto nbs = Kokkos::create_mirror_view(lt.nbs);
  auto blev = Kokkos::create_mirror_view(lt.blev);
  auto fbox = Kokkos::create_mirror_view(lt.fbox);
  std::vector<Real> bmax1(nall), bmax2(nall), bmax3(nall);
  for (int b=0; b<nall; ++b) {
    const LogicalLocation &ll = pm->lloc_eachmb[gids + b];
    const int dl = ll.level - pm->root_level;
    const std::int32_t n1 = pm->nmb_rootx1 << dl, n2 = pm->nmb_rootx2 << dl;
    const std::int32_t n3 = pm->nmb_rootx3 << dl;
    const Real x1a = (ll.lx1 == 0) ? ms.x1min : LeftEdgeX(ll.lx1, n1, ms.x1min, ms.x1max);
    const Real x1b = (ll.lx1 == n1 - 1) ? ms.x1max : LeftEdgeX(ll.lx1 + 1, n1, ms.x1min, ms.x1max);
    const Real x2a = (ll.lx2 == 0) ? ms.x2min : LeftEdgeX(ll.lx2, n2, ms.x2min, ms.x2max);
    const Real x2b = (ll.lx2 == n2 - 1) ? ms.x2max : LeftEdgeX(ll.lx2 + 1, n2, ms.x2min, ms.x2max);
    const Real x3a = (ll.lx3 == 0) ? ms.x3min : LeftEdgeX(ll.lx3, n3, ms.x3min, ms.x3max);
    const Real x3b = (ll.lx3 == n3 - 1) ? ms.x3max : LeftEdgeX(ll.lx3 + 1, n3, ms.x3min, ms.x3max);
    const Real dx1 = (x1b - x1a)/static_cast<Real>(indcs.nx1);
    const Real dx2 = (x2b - x2a)/static_cast<Real>(indcs.nx2);
    const Real dx3 = (x3b - x3a)/static_cast<Real>(indcs.nx3);
    box(b, 0) = x1a; box(b, 1) = x2a; box(b, 2) = x3a;
    box(b, 3) = dx1; box(b, 4) = dx2; box(b, 5) = dx3;
    box(b, 6) = 1.0/dx1; box(b, 7) = 1.0/dx2; box(b, 8) = 1.0/dx3;
    bmax1[b] = x1b; bmax2[b] = x2b; bmax3[b] = x3b;
    const auto &sz = mbsize.h_view(b);
    if (sz.x1min != x1a || sz.x2min != x2a || sz.x3min != x3a || sz.dx1 != dx1 ||
        sz.dx2 != dx2 || sz.dx3 != dx3) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "3-D leaf traversal: the box of block " << gids + b << " from its logical "
                << "location differs from mb_size" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const int ratio = 1 << (lt.nlev - dl);
    blev(b) = dl;
    fbox(b, 0) = ll.lx1*indcs.nx1*ratio; fbox(b, 1) = fbox(b, 0) + indcs.nx1*ratio;
    fbox(b, 2) = ll.lx2*indcs.nx2*ratio; fbox(b, 3) = fbox(b, 2) + indcs.nx2*ratio;
    fbox(b, 4) = ll.lx3*indcs.nx3*ratio; fbox(b, 5) = fbox(b, 4) + indcs.nx3*ratio;
    if (dl > lt.nlev ||
        std::lround((x1a - lt.x1min)/lt.fine_dx1) != fbox(b, 0) ||
        std::lround((x2a - lt.x2min)/lt.fine_dx2) != fbox(b, 2) ||
        std::lround((x3a - lt.x3min)/lt.fine_dx3) != fbox(b, 4)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "3-D leaf traversal: block " << gids + b << " (level " << dl
                << ") does not fit the finest grid of " << lt.nlev << " levels" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // the direction of every neighbour slot
  int dir[56][3];
  for (int n=0; n<56; ++n) dir[n][0] = dir[n][1] = dir[n][2] = 0;
  for (int iz=-1; iz<=1; ++iz) {
    for (int iy=-1; iy<=1; ++iy) {
      for (int ix=-1; ix<=1; ++ix) {
        for (int n2=0; n2<=1; ++n2) {
          for (int n1=0; n1<=1; ++n1) {
            // sub-block indices: two for faces, one for edges (NeighborIndex ignores the second,
            // and a second index of 1 would give another slot), none for corners
            const int nz = (ix != 0) + (iy != 0) + (iz != 0);
            if ((nz >= 2 && n2 != 0) || (nz == 3 && n1 != 0)) continue;
            const int idx = NeighborIndex(ix, iy, iz, n1, n2);
            if (idx >= 0 && idx < 56) {dir[idx][0] = ix; dir[idx][1] = iy; dir[idx][2] = iz;}
          }
        }
      }
    }
  }
  const Real lens[3] = {lt.lx, lt.ly, lt.lz};
  for (int m=0; m<nmb; ++m) {
    int c = 0;
    for (int n=0; n<nslot; ++n) {
      const int g = nghbr.h_view(m, n).gid;
      if (g < 0) continue;
      const int b = g - gids;
      int s[3];
      const Real bmin_b[3] = {box(b, 0), box(b, 1), box(b, 2)};
      const Real bmin_m[3] = {box(m, 0), box(m, 1), box(m, 2)};
      const Real bmax_b[3] = {bmax1[b], bmax2[b], bmax3[b]};
      const Real bmax_m[3] = {bmax1[m], bmax2[m], bmax3[m]};
      for (int d=0; d<3; ++d) {
        // across a face, edge or corner: the gap between the facing sides (0 or a domain length);
        // along it (id = 0) the neighbour overlaps the block: the image whose centre is nearest.
        // (The minimum-side difference rounds the wrong way when a coarser neighbour spans the
        // whole domain in that direction: half a domain length.)
        const int id = dir[n][d];
        const Real gap = (id > 0) ? bmin_b[d] - bmax_m[d]
                       : ((id < 0) ? bmax_b[d] - bmin_m[d]
                                   : 0.5*(bmin_b[d] + bmax_b[d] - bmin_m[d] - bmax_m[d]));
        s[d] = static_cast<int>(std::lround(gap/lens[d]));
      }
      bool seen = (b == m && s[0] == 0 && s[1] == 0 && s[2] == 0);
      for (int k=0; k<c && !seen; ++k) {
        seen = (nb(m, k) == b && nbs(m, k, 0) == s[0] && nbs(m, k, 1) == s[1] &&
                nbs(m, k, 2) == s[2]);
      }
      if (seen) continue;
      if (c >= LeafTraversal3D::kMaxNb) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "3-D leaf traversal: more than " << LeafTraversal3D::kMaxNb
                  << " distinct neighbours of block " << m << std::endl;
        std::exit(EXIT_FAILURE);
      }
      nb(m, c) = b; nbs(m, c, 0) = s[0]; nbs(m, c, 1) = s[1]; nbs(m, c, 2) = s[2];
      ++c;
    }
    nnb(m) = c;
  }
  // the finest level among each block and its neighbours
  lt.nbmax = DvceArray1D<int>("dust_leaf3d_nbmax", std::max(1, nmb));
  auto nbmax = Kokkos::create_mirror_view(lt.nbmax);
  for (int m=0; m<nmb; ++m) {
    int mx = blev(m);
    for (int k=0; k<nnb(m); ++k) mx = std::max(mx, blev(nb(m, k)));
    nbmax(m) = mx;
  }
  // the ring that the neighbours certainly hold, checked cell by cell
  lt.safe = DvceArray2D<int>("dust_leaf3d_safe", std::max(1, nmb), 3);
  auto safe = Kokkos::create_mirror_view(lt.safe);
  int nunsafe = 0;
  for (int m=0; m<nmb; ++m) {
    int s1 = 0, s2 = 0, s3 = 0;
    for (int k=0; k<nnb(m); ++k) {
      const int b = nb(m, k);
      const int w1 = fbox(b, 1) - fbox(b, 0), w2 = fbox(b, 3) - fbox(b, 2);
      const int w3 = fbox(b, 5) - fbox(b, 4);
      s1 = (k == 0 || w1 < s1) ? w1 : s1;
      s2 = (k == 0 || w2 < s2) ? w2 : s2;
      s3 = (k == 0 || w3 < s3) ? w3 : s3;
    }
    const int x0 = fbox(m, 0), x1 = fbox(m, 1), y0 = fbox(m, 2), y1 = fbox(m, 3);
    const int z0 = fbox(m, 4), z1 = fbox(m, 5);
    bool ok = (s1 > 0 && s2 > 0 && s3 > 0 && (x1 - x0) % s1 == 0 && (y1 - y0) % s2 == 0 &&
               (z1 - z0) % s3 == 0);
    for (int cz=z0 - s3; ok && cz<z1 + s3; cz+=s3) {
      for (int cy=y0 - s2; ok && cy<y1 + s2; cy+=s2) {
        for (int cx=x0 - s1; ok && cx<x1 + s1; cx+=s1) {
          if (cx >= x0 && cx < x1 && cy >= y0 && cy < y1 && cz >= z0 && cz < z1) continue;
          bool held = false;
          for (int k=0; k<nnb(m) && !held; ++k) {
            const int b = nb(m, k);
            const int bx0 = fbox(b, 0) - nbs(m, k, 0)*lt.nfine1;
            const int bx1 = fbox(b, 1) - nbs(m, k, 0)*lt.nfine1;
            const int by0 = fbox(b, 2) - nbs(m, k, 1)*lt.nfine2;
            const int by1 = fbox(b, 3) - nbs(m, k, 1)*lt.nfine2;
            const int bz0 = fbox(b, 4) - nbs(m, k, 2)*lt.nfine3;
            const int bz1 = fbox(b, 5) - nbs(m, k, 2)*lt.nfine3;
            held = (bx0 <= cx && cx + s1 <= bx1 && by0 <= cy && cy + s2 <= by1 &&
                    bz0 <= cz && cz + s3 <= bz1);
          }
          ok = held;
        }
      }
    }
    if (!ok) {s1 = 0; s2 = 0; s3 = 0; ++nunsafe;}
    safe(m, 0) = s1; safe(m, 1) = s2; safe(m, 2) = s3;
  }
  // the index of logical locations (Locate)
  {
    std::vector<std::pair<std::int64_t, int>> kv(nall);
    for (int b=0; b<nall; ++b) {
      const int sh = lt.nlev - blev(b);
      kv[b] = {LeafTraversal3D::LocKey(blev(b), fbox(b, 0)/(indcs.nx1 << sh),
                                       fbox(b, 2)/(indcs.nx2 << sh),
                                       fbox(b, 4)/(indcs.nx3 << sh)), b};
    }
    std::sort(kv.begin(), kv.end());
    lt.lockey = DvceArray1D<std::int64_t>("dust_leaf3d_lockey", nall);
    lt.locblk = DvceArray1D<int>("dust_leaf3d_locblk", nall);
    auto kh = Kokkos::create_mirror_view(lt.lockey);
    auto bh = Kokkos::create_mirror_view(lt.locblk);
    for (int k=0; k<nall; ++k) {kh(k) = kv[k].first; bh(k) = kv[k].second;}
    Kokkos::deep_copy(lt.lockey, kh);
    Kokkos::deep_copy(lt.locblk, bh);
    lt.nloc = nall;
  }
  Kokkos::deep_copy(lt.box, box);
  Kokkos::deep_copy(lt.nnb, nnb);
  Kokkos::deep_copy(lt.nb, nb);
  Kokkos::deep_copy(lt.nbs, nbs);
  Kokkos::deep_copy(lt.blev, blev);
  Kokkos::deep_copy(lt.fbox, fbox);
  Kokkos::deep_copy(lt.nbmax, nbmax);
  Kokkos::deep_copy(lt.safe, safe);
  if (global_variable::my_rank == 0) {
    int mxnb = 0;
    for (int m=0; m<nmb; ++m) mxnb = std::max(mxnb, nnb(m));
    std::cout << "# 3-D leaf traversal: " << nmb << " blocks, levels " << lt.nlev
              << ", finest grid " << lt.nfine1 << "x" << lt.nfine2 << "x" << lt.nfine3
              << ", at most " << mxnb << " neighbours"
              << (nunsafe > 0 ? ", " + std::to_string(nunsafe) + " blocks without a certain ring"
                              : "") << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckLeafTraversal3D
//! \brief Checks nsample clouds (random positions, levels, lobes, gather point flags and
//! anchor blocks) against a brute-force integration: every finest cell under the support is
//! integrated exactly (the same 1-D CDF differences on the finest faces) and its weight and first
//! moments are added to the leaf that holds it (Locate).  Also the split eligibility (InsideFiner
//! with the overlap rule) against the brute-force test "a finest cell of the shrunk support lies
//! in a block finer than the cloud".  Stops with an error beyond round-off.

void DustGasDrag::CheckLeafTraversal3D(const int nsample) {
  const auto lt = leaf_traversal3d;
  const int nleaf = lt.nmb*lt.ncells3*lt.ncells2*lt.ncells1;
  DvceArray2D<Real> d("dust_leaf3d_check", nleaf, 4);
  DvceArray1D<Real> res("dust_leaf3d_check_res", 4);
  double maxw = 0.0, maxm = 0.0, maxsum = 0.0;
  int nelig_bad = 0, nglobal = 0;
  const Real slack = cloud_split_slack;
  const std::uint64_t seed = static_cast<std::uint64_t>(pmy_pack->pmesh->ncycle) + 12345;
  for (int s=0; s<nsample; ++s) {
    const std::uint64_t key = seed*UINT64_C(1000003) + static_cast<std::uint64_t>(s)*7919;
    const Real px = lt.x1min + Uniform01(key + 1)*lt.lx;
    const Real py = lt.x2min + Uniform01(key + 2)*lt.ly;
    const Real pz = lt.x3min + Uniform01(key + 3)*lt.lz;
    const int lev = std::min(lt.nlev, static_cast<int>(Uniform01(key + 4)*(lt.nlev + 1)));
    int code = lev;
    if (lev >= 1 && Uniform01(key + 5) < 0.5) {
      code |= (static_cast<int>(Uniform01(key + 6)*4) & 3) << SMRCompositeTransfer::kCloudLobeShift1;
      code |= (static_cast<int>(Uniform01(key + 7)*4) & 3) << SMRCompositeTransfer::kCloudLobeShift2;
      code |= (static_cast<int>(Uniform01(key + 8)*4) & 3) << SMRCompositeTransfer::kCloudLobeShift3;
    }
    if (Uniform01(key + 9) < 0.3) code |= SMRCompositeTransfer::kCloudGatherPoint;
    const int anchor = std::min(lt.nmb - 1, static_cast<int>(Uniform01(key + 10)*lt.nmb));
    Kokkos::deep_copy(d, 0.0);
    Kokkos::parallel_for("dust_leaf3d_check", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
    KOKKOS_LAMBDA(const int) {
      using LT = LeafTraversal;
      using T = SMRCompositeTransfer;
      bool reach = true;
      const Real sum = lt.ForEach<true>(px, py, pz, code, anchor,
      [&](const int t, const Real w, const Real m1x, const Real m1y, const Real m1z) {
        d(t, 0) += w; d(t, 1) += m1x; d(t, 2) += m1y; d(t, 3) += m1z;
      }, &reach);
      res(0) = sum;
      res(3) = reach ? 0.0 : 1.0;
      // brute force over the finest cells of the support (unwrapped indices; Locate wraps)
      const int lv = T::CloudLevel(code);
      const bool point = (code & T::kCloudGatherPoint) != 0;
      const LT::Shape s1 = LT::MakeShape(px, ldexp(lt.root_dx1, -lv), ldexp(lt.iroot_dx1, lv),
                                         T::CloudLobe1(code), point);
      const LT::Shape s2 = LT::MakeShape(py, ldexp(lt.root_dx2, -lv), ldexp(lt.iroot_dx2, lv),
                                         T::CloudLobe2(code), point);
      const LT::Shape s3 = LT::MakeShape(pz, ldexp(lt.root_dx3, -lv), ldexp(lt.iroot_dx3, lv),
                                         T::CloudLobe3(code), point);
      const int ia = static_cast<int>(floor((s1.lo - lt.x1min)/lt.fine_dx1));
      const int ib = static_cast<int>(ceil((s1.hi - lt.x1min)/lt.fine_dx1)) - 1;
      const int ja = static_cast<int>(floor((s2.lo - lt.x2min)/lt.fine_dx2));
      const int jb = static_cast<int>(ceil((s2.hi - lt.x2min)/lt.fine_dx2)) - 1;
      const int ka = static_cast<int>(floor((s3.lo - lt.x3min)/lt.fine_dx3));
      const int kb = static_cast<int>(ceil((s3.hi - lt.x3min)/lt.fine_dx3)) - 1;
      for (int k=ka; k<=kb; ++k) {
        Real w3, g3;
        LT::Cell<true>(s3, LT::FaceAt<true>(s3, lt.x3min + k*lt.fine_dx3),
                       LT::FaceAt<true>(s3, lt.x3min + (k + 1)*lt.fine_dx3), w3, g3);
        for (int j=ja; j<=jb; ++j) {
          Real w2, g2;
          LT::Cell<true>(s2, LT::FaceAt<true>(s2, lt.x2min + j*lt.fine_dx2),
                         LT::FaceAt<true>(s2, lt.x2min + (j + 1)*lt.fine_dx2), w2, g2);
          for (int i=ia; i<=ib; ++i) {
            Real w1, g1;
            LT::Cell<true>(s1, LT::FaceAt<true>(s1, lt.x1min + i*lt.fine_dx1),
                           LT::FaceAt<true>(s1, lt.x1min + (i + 1)*lt.fine_dx1), w1, g1);
            const int t = lt.Locate(i, j, k);
            if (t < 0) {res(1) = 1.0e30; continue;}
            d(t, 0) -= w1*w2*w3; d(t, 1) -= g1*w2*w3; d(t, 2) -= w1*g2*w3; d(t, 3) -= w1*w2*g3;
          }
        }
      }
      // eligibility of a plain cloud of level lv at the position (overlap rule)
      if (lt.split_overlap && lv < lt.nlev && lv < lt.cloud_max_level && !lt.split_everywhere) {
        const bool fast = lt.InsideFiner(px, py, pz, lv, slack, anchor);
        int fa, fb, ga, gb, ha, hb;
        lt.SupportBox(px, py, pz, 1.0 - slack, lv, fa, fb, ga, gb, ha, hb);
        bool brute = false;
        for (int k=ha; k<=hb && !brute; ++k) {
          for (int j=ga; j<=gb && !brute; ++j) {
            for (int i=fa; i<=fb && !brute; ++i) {
              int b = -1;
              lt.Locate(i, j, k, &b);
              if (b >= 0 && lt.blev(b) > lv) brute = true;
            }
          }
        }
        res(2) = (fast == brute) ? 0.0 : 1.0;
      } else {
        res(2) = 0.0;
      }
    });
    double ew = 0.0, em = 0.0;
    Kokkos::parallel_reduce("dust_leaf3d_check_max", Kokkos::RangePolicy<>(DevExeSpace(), 0, nleaf),
    KOKKOS_LAMBDA(const int t, double &a, double &b) {
      a = fmax(a, fabs(d(t, 0)));
      b = fmax(b, fmax(fabs(d(t, 1)), fmax(fabs(d(t, 2)), fabs(d(t, 3)))));
    }, Kokkos::Max<double>(ew), Kokkos::Max<double>(em));
    auto rh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), res);
    const Real hw = lt.root_dx1/static_cast<Real>(1 << lev);   // moments in length units
    maxw = std::max(maxw, ew);
    maxm = std::max(maxm, em/hw);
    maxsum = std::max(maxsum, std::fabs(rh(0) - 1.0) + rh(1));
    if (rh(2) != 0.0) ++nelig_bad;
    if (rh(3) != 0.0) ++nglobal;
  }
  const bool ok = (maxw <= 1.0e-13 && maxm <= 1.0e-13 && maxsum <= 1.0e-13 && nelig_bad == 0);
  if (global_variable::my_rank == 0) {
    std::cout << "# 3-D leaf traversal check, cycle " << pmy_pack->pmesh->ncycle
              << ": " << nsample << " clouds, max |weight - brute| " << maxw
              << ", max |moment - brute|/half-width " << maxm << ", max |sum - 1| " << maxsum
              << ", eligibility mismatches " << nelig_bad << ", global searches " << nglobal
              << (ok ? " OK" : " FAILED") << std::endl;
  }
  if (!ok) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D leaf traversal check failed" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitCloudsP1Device3D
//! \brief The device P1 split in 3-D (SplitCloudsP1Device with a third direction): pieces per
//! particle, exclusive scan for the slots and tags, one fill kernel; the centre piece keeps its
//! parent's slot and tag.  Returns the number of split / released particles.

int DustGasDrag::SplitCloudsP1Device3D() {
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart = ppar->nprtcl_thispack;
  if (npart == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const auto lt = leaf_traversal3d;
  const Real slack = cloud_split_slack;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
  const int sh3 = SMRCompositeTransfer::kCloudLobeShift3;
  const int gids = pmy_pack->gids;
  const int ncell123 = qdep.extent_int(4)*qdep.extent_int(3)*qdep.extent_int(2);
  if (split_np.extent_int(0) < npart) {
    split_np = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                   "dust_split_np"), npart + npart/4);
    split_off = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                    "dust_split_off"), npart + npart/4);
  }
  auto np = split_np;
  auto tagoff = split_off;
  int nflag = 0, nparent = 0;
  Kokkos::parallel_reduce("dust_split3d_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &sf, int &sp) {
    const int code = pi(PLEV,p);
    const Real x = pr(IPX,p), y = pr(IPY,p), z = pr(IPZ,p);
    const int m = lt.Anchor(pi(PGID,p));
    if (!P1Flag3(lt, m, code, x, y, z, slack)) {np(p) = 0; return;}
    DPiece3 p1[3], p2[3], p3[3]; int n1, n2, n3, clev; bool parent;
    P1Event3(lt, m, code, x, y, z, slack, p1, p2, p3, n1, n2, n3, clev, parent);
    np(p) = n1*n2*n3;
    sf += 1;
    if (parent) sp += 1;
  }, Kokkos::Sum<int>(nflag), Kokkos::Sum<int>(nparent));
  if (nflag == 0) return 0;
  EnsureNextTag();
  int ntags = 0;
  Kokkos::parallel_scan("dust_split3d_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) tagoff(p) = upd;
    upd += (np(p) > 0) ? np(p) - 1 : 0;
  }, ntags);
  const int nnew = npart + ntags;
  if (ppar->prtcl_rdata.extent_int(1) < nnew) {
    Kokkos::resize(ppar->prtcl_rdata, nrdata, nnew + nnew/4);
    Kokkos::resize(ppar->prtcl_idata, nidata, nnew + nnew/4);
  }
  auto prv = ppar->prtcl_rdata;
  auto piv = ppar->prtcl_idata;
  const int tag0 = cloud_next_tag, tstride = global_variable::nranks;
  Kokkos::parallel_for("dust_split3d_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (np(p) == 0) return;
    const int code = piv(PLEV,p);
    const Real x = prv(IPX,p), y = prv(IPY,p), z = prv(IPZ,p), mass = prv(IPM,p);
    DPiece3 p1[3], p2[3], p3[3]; int n1, n2, n3, clev; bool parent;
    P1Event3(lt, lt.Anchor(piv(PGID,p)), code, x, y, z, slack, p1, p2, p3, n1, n2, n3, clev,
             parent);
    const Real e1 = lt.CloudHalfWidth1(clev), e2 = lt.CloudHalfWidth2(clev);
    const Real e3 = lt.CloudHalfWidth3(clev);
    const int n12 = n1*n2;
    for (int k=n12*n3-1; k>=0; --k) {
      const int a = k % n1, b = (k/n1) % n2, c = k/n12;
      const int s = (k == 0) ? p : npart + tagoff(p) + k - 1;
      const Real cx = LeafTraversal::Wrap(x + p1[a].off*e1, lt.x1min, lt.x1max);
      const Real cy = LeafTraversal::Wrap(y + p2[b].off*e2, lt.x2min, lt.x2max);
      const Real cz = LeafTraversal::Wrap(z + p3[c].off*e3, lt.x3min, lt.x3max);
      if (k > 0) {
        for (int q=0; q<nrdata; ++q) prv(q,s) = prv(q,p);
        for (int q=0; q<nidata; ++q) piv(q,s) = piv(q,p);
        piv(PTAG,s) = tag0 + (tagoff(p) + k - 1)*tstride;
      }
      prv(IPX,s) = cx; prv(IPX1,s) = cx;
      prv(IPY,s) = cy; prv(IPY1,s) = cy;
      prv(IPZ,s) = cz; prv(IPZ1,s) = cz;
      prv(IPM,s) = mass*p1[a].frac*p2[b].frac*p3[c].frac;
      piv(PLEV,s) = clev | (p1[a].lobe << sh1) | (p2[b].lobe << sh2) | (p3[c].lobe << sh3);
      piv(PGID,s) = gids + NewCloudBlock3(lt, cx, cy, cz, lt.Anchor(piv(PGID,s)), ncell123);
    }
  });
  cloud_next_tag += ntags*global_variable::nranks;
  cloud_split_count += nparent;
  cloud_lobe_split_count += nflag - nparent;
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;
  return nflag;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::PackGatherField3D, PackGasGather3D
//! \brief The 3-D gather field: gas velocity and its central differences in x1, x2, x3 of every
//! cell with both neighbours in each direction, twelve numbers side by side (PackGatherField in
//! 3-D); from a conserved state: the velocity of every cell first (PackGasGather in 3-D).

void DustGasDrag::PackGatherField3D(const DvceArray5D<Real> &vel) {
  const int nmb = pmy_pack->nmb_thispack, nk = vel.extent_int(2);
  const int nj = vel.extent_int(3), ni = vel.extent_int(4);
  if (gpack3.extent_int(0) != std::max(nmb, vel.extent_int(0)) || gpack3.extent_int(1) != nk ||
      gpack3.extent_int(2) != nj || gpack3.extent_int(3) != ni) {
    Kokkos::realloc(gpack3, std::max(nmb, vel.extent_int(0)), nk, nj, ni, 12);
  }
  auto gp = gpack3;
  auto v = vel;
  par_for("dust_gather_pack3d", DevExeSpace(), 0, nmb-1, 1, nk-2, 1, nj-2, 1, ni-2,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    for (int c=0; c<3; ++c) {
      gp(m,k,j,i,c) = v(m,c,k,j,i);
      gp(m,k,j,i,3+c) = v(m,c,k,j,i+1) - v(m,c,k,j,i-1);
      gp(m,k,j,i,6+c) = v(m,c,k,j+1,i) - v(m,c,k,j-1,i);
      gp(m,k,j,i,9+c) = v(m,c,k+1,j,i) - v(m,c,k-1,j,i);
    }
  });
}

void DustGasDrag::PackGasGather3D(const DvceArray5D<Real> &u) {
  auto uold = u;
  if (uvel_pred.extent(0) != uold.extent(0) || uvel_pred.extent(2) != uold.extent(2)) {
    Kokkos::realloc(uvel_pred, uold.extent(0), 3, uold.extent(2), uold.extent(3),
                    uold.extent(4));
  }
  auto uvel = uvel_pred;
  const int nmb_ = pmy_pack->nmb_thispack;
  const int nk_ = uold.extent_int(2), nj_ = uold.extent_int(3), ni_ = uold.extent_int(4);
  par_for("dust_pc2_predictor_gasvel3d", DevExeSpace(), 0, nmb_-1, 0, nk_-1, 0, nj_-1, 0, ni_-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real invrho = 1.0/uold(m,IDN,k,j,i);
    uvel(m,0,k,j,i) = uold(m,IM1,k,j,i)*invrho;
    uvel(m,1,k,j,i) = uold(m,IM2,k,j,i)*invrho;
    uvel(m,2,k,j,i) = uold(m,IM3,k,j,i)*invrho;
  });
  PackGatherField3D(uvel_pred);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudPredictor3D
//! \brief The production refinable-cloud PC2 predictor (DepositDrag, linear gather) in 3-D:
//! linear gather of the step-start gas velocity over the cloud, the drag impulse deposited with
//! the cloud's own weights into qdep (components 1-3, and 4 with rate_dep).

void DustGasDrag::CloudPredictor3D(const bool rate_dep) {
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const Real dt = pmy_pack->pmesh->dt;
  auto &qdep_ = qdep;
  const auto lt = leaf_traversal3d;
  const int tmode = cloud_transition;
  PackGasGather3D(pmy_pack->phydro->u1);
  auto gp = gpack3;
  auto body = KOKKOS_LAMBDA(const int p) {
    using T = SMRCompositeTransfer;
    const Real xp = pr(IPX,p), yp = pr(IPY,p), zp = pr(IPZ,p);
    const int am = lt.Anchor(pi(PGID,p));
    const int lev = pi(PLEV,p);
    Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
    int nt = 0, tt[kCache3];
    Real tw[kCache3];
    const int gcode = T::GatherCode(lev, tmode);
    const bool same = (gcode & ~T::kCloudGatherPoint) == lev;
    lt.ForEachCloudMoment(xp, yp, zp, gcode, am,
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
      if (nt < kCache3) {tt[nt] = target; tw[nt] = w;}
      ++nt;
    });
    const Real mass_rate = pr(IPM,p)/pr(IPTS,p);
    const Real gx = -dt*mass_rate*(ugx - pr(IPVX,p));
    const Real gy = -dt*mass_rate*(ugy - pr(IPVY,p));
    const Real gz = -dt*mass_rate*(ugz - pr(IPVZ,p));
    auto add = [&](const int target, const Real w) {
      int tb, tk, tj, ti;
      lt.Decode(target, tb, tk, tj, ti);
      const Real inv_volume = 1.0/(lt.box(tb,3)*lt.box(tb,4)*lt.box(tb,5));
      DepositAdd(&qdep_(tb,1,tk,tj,ti), inv_volume*w*gx);
      DepositAdd(&qdep_(tb,2,tk,tj,ti), inv_volume*w*gy);
      DepositAdd(&qdep_(tb,3,tk,tj,ti), inv_volume*w*gz);
      if (rate_dep) DepositAdd(&qdep_(tb,4,tk,tj,ti), inv_volume*w*mass_rate);
    };
    if (same && nt <= kCache3) {
      for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
    } else {
      lt.ForEachCloudWeight(xp, yp, zp, lev, am, add);
    }
  };
  CloudParFor("dust_pc2_cloud3d_predictor", npart, body);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudAdvance3D
//! \brief The production refinable-cloud PC2 advance (ExplicitPush, linear gather) in 3-D:
//! linear gather of the midpoint gas velocity u* at the midpoint position, implicit-midpoint
//! kick and drift (the 3-D shearing-box terms as in the 2-D kernel), and the impulse deposited
//! with the cloud's weights into dmom.

void DustGasDrag::CloudAdvance3D() {
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const Real dt = pmy_pack->pmesh->dt;
  const bool br = back_reaction;
  const bool is_sbox = is_shearing_box, is_strat = is_stratified;
  const Real qshear_ = qshear, omega0_ = omega0;
  auto &dmom_ = dmom;
  const auto lt = leaf_traversal3d;
  const int tmode = cloud_transition;
  PackGatherField3D(ustar);
  auto gp = gpack3;
  auto body = KOKKOS_LAMBDA(const int p) {
    using T = SMRCompositeTransfer;
    int nt = 0, tt[kCache3];
    Real tw[kCache3];
    const Real x0 = pr(IPX1,p), y0 = pr(IPY1,p), z0 = pr(IPZ1,p);
    const int am = lt.Anchor(pi(PGID,p));
    const Real vx0 = pr(IPVX1,p), vy0 = pr(IPVY1,p), vz0 = pr(IPVZ1,p);
    const Real xh = x0 + 0.5*dt*vx0;
    const Real yh = y0 + 0.5*dt*(vy0 - (is_sbox ? qshear_*omega0_*x0 : 0.0));
    const Real zh = z0 + 0.5*dt*vz0;
    Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
    const int glev = T::GatherCode(pi(PLEV,p), tmode);
    const bool same = (glev & ~T::kCloudGatherPoint) == pi(PLEV,p);
    lt.ForEachCloudMoment(xh, yh, zh, glev, am,
    [&](const int target, const Real w, const Real m1x, const Real m1y, const Real m1z) {
      int tb, tk, tj, ti;
      lt.Decode(target, tb, tk, tj, ti);
      const Real dx1 = lt.box(tb,3), dx2 = lt.box(tb,4), dx3 = lt.box(tb,5);
      const Real xc = lt.box(tb,0) + (ti - is + 0.5)*dx1;
      const Real yc = lt.box(tb,1) + (tj - js + 0.5)*dx2;
      const Real zc = lt.box(tb,2) + (tk - ks + 0.5)*dx3;
      const Real cx = 0.5*(m1x + T::MinImage(xh - xc, lt.lx)*w)/dx1;
      const Real cy = 0.5*(m1y + T::MinImage(yh - yc, lt.ly)*w)/dx2;
      const Real cz = 0.5*(m1z + T::MinImage(zh - zc, lt.lz)*w)/dx3;
      ugx += w*gp(tb,tk,tj,ti,0) + cx*gp(tb,tk,tj,ti,3) + cy*gp(tb,tk,tj,ti,6) + cz*gp(tb,tk,tj,ti,9);
      ugy += w*gp(tb,tk,tj,ti,1) + cx*gp(tb,tk,tj,ti,4) + cy*gp(tb,tk,tj,ti,7) + cz*gp(tb,tk,tj,ti,10);
      ugz += w*gp(tb,tk,tj,ti,2) + cx*gp(tb,tk,tj,ti,5) + cy*gp(tb,tk,tj,ti,8) + cz*gp(tb,tk,tj,ti,11);
      if (nt < kCache3) {tt[nt] = target; tw[nt] = w;}
      ++nt;
    });
    const Real h = 0.5*dt;
    const Real lambda = h/pr(IPTS,p);
    const Real aa = 1.0 + lambda;
    const Real rx = vx0 + lambda*ugx;
    const Real ry = vy0 + lambda*ugy;
    Real rz = vz0 + lambda*ugz;
    Real vxh, vyh, vzh;
    if (is_sbox) {
      const Real cxy = 2.0*h*omega0_;
      const Real cyx = h*(2.0-qshear_)*omega0_;
      const Real det = aa*aa + cxy*cyx;
      vxh = (aa*rx + cxy*ry)/det;
      vyh = (aa*ry - cyx*rx)/det;
      if (is_strat) rz -= h*SQR(omega0_)*zh;
      vzh = rz/aa;
    } else {
      vxh = rx/aa;
      vyh = ry/aa;
      vzh = rz/aa;
    }
    pr(IPVX,p) = 2.0*vxh - vx0;
    pr(IPVY,p) = 2.0*vyh - vy0;
    pr(IPVZ,p) = 2.0*vzh - vz0;
    pr(IPX,p) = x0 + dt*vxh;
    pr(IPY,p) = y0 + dt*(vyh - (is_sbox ? qshear_*omega0_*xh : 0.0));
    pr(IPZ,p) = z0 + dt*vzh;
    pr(IPX1,p) = xh; pr(IPY1,p) = yh; pr(IPZ1,p) = zh;
    const Real dvx_drag = dt*(ugx-vxh)/pr(IPTS,p);
    const Real dvy_drag = dt*(ugy-vyh)/pr(IPTS,p);
    const Real dvz_drag = dt*(ugz-vzh)/pr(IPTS,p);
    pr(IPRX,p) = dvx_drag;
    pr(IPRY,p) = dvy_drag;
    pr(IPRZ,p) = dvz_drag;
    if (!br) return;
    const Real mass = -pr(IPM,p);
    auto add = [&](const int target, const Real w) {
      int tb, tk, tj, ti;
      lt.Decode(target, tb, tk, tj, ti);
      const Real dw = mass*w/(lt.box(tb,3)*lt.box(tb,4)*lt.box(tb,5));
      DepositAdd(&dmom_(tb,0,tk,tj,ti), dw*dvx_drag);
      DepositAdd(&dmom_(tb,1,tk,tj,ti), dw*dvy_drag);
      DepositAdd(&dmom_(tb,2,tk,tj,ti), dw*dvz_drag);
    };
    if (same && nt <= kCache3) {
      for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
    } else {
      lt.ForEachCloudWeight(xh, yh, zh, pi(PLEV,p), am, add);
    }
  };
  CloudParFor("dust_pc2_cloud3d_advance", npart, body);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositCloudDensity3D
//! \brief Dust density (component 0) and momentum density (1-3) of the clouds on the leaves of
//! this rank (diagnostics; f is (re)allocated to the size of qdep with 4 components).

void DustGasDrag::DepositCloudDensity3D(DvceArray5D<Real> &f) {
  if (f.extent_int(0) != qdep.extent_int(0) || f.extent_int(1) != 4 ||
      f.extent_int(2) != qdep.extent_int(2) || f.extent_int(3) != qdep.extent_int(3) ||
      f.extent_int(4) != qdep.extent_int(4)) {
    Kokkos::realloc(f, qdep.extent(0), 4, qdep.extent(2), qdep.extent(3), qdep.extent(4));
  }
  Kokkos::deep_copy(f, 0.0);
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const auto lt = leaf_traversal3d;
  auto ff = f;
  Kokkos::parallel_for("dust_cloud3d_density", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    const Real m = pr(IPM,p), vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
    lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pr(IPZ,p), pi(PLEV,p), lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w) {
      int tb, tk, tj, ti;
      lt.Decode(target, tb, tk, tj, ti);
      const Real d = m*w/(lt.box(tb,3)*lt.box(tb,4)*lt.box(tb,5));
      DepositAdd(&ff(tb,0,tk,tj,ti), d);
      DepositAdd(&ff(tb,1,tk,tj,ti), d*vx);
      DepositAdd(&ff(tb,2,tk,tj,ti), d*vy);
      DepositAdd(&ff(tb,3,tk,tj,ti), d*vz);
    });
  });
}

}  // namespace dust
