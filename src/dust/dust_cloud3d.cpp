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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
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
  // several ranks (stage 3): <dust>/experimental_mpi and the options of CheckMultiRank
  if (global_variable::nranks != 1 && !experimental_mpi) {
    bad.push_back("several MPI ranks without <dust>/experimental_mpi");
  }
  if (!(coupling == DustCoupling::imex ||
        (coupling == DustCoupling::hybrid && hybrid_force_mode == HybridForceMode::pc2))) {
    bad.push_back("a coupling other than PC2 or IMEX (coupling = pc2 / imex)");
  }
  if (drag_solver != DustDragSolver::local) bad.push_back("drag_solver other than local");
  if (cloud_gather_order != 2 || analytic_gather.on) {
    bad.push_back("gather variants (cloud_gather_order != 2, analytic_gather)");
  }
  // G (stage 2b): the device quantum pass with the pass budget (MergeCloudsDevice3D)
  if (cloud_merge && (cloud_merge_mode != 1 || !cloud_merge_device || cloud_merge_budget_x ||
                      cloud_merge_xprotect || cloud_audit_t1 > 0.0 || cloud_audit_only)) {
    bad.push_back("merging other than the device quantum G (no stage-X budget, X protection, "
                  "audits)");
  }
  // R (stage 2c): the device solve (ResampleClouds3D), without its diagnostics
  if (cloud_resample && (!cloud_resample_device || cloud_resample_compare ||
                         cloud_resample_feedback_check || cloud_resample_stress > 0 ||
                         cloud_resample_stress_copies > 0 || !cloud_resample_dump.empty() ||
                         cloud_resample_patch_stats || cloud_resample_pos_order)) {
    bad.push_back("resampling other than the device solve (no host solve, compare, dump, stress, "
                  "feedback check, patch statistics, position order)");
  }
  if (!cloud_split_device || cloud_split_log || split_oracle) bad.push_back("the host split");
  // (physical x3 faces: the supports fold back at the faces; x3_ghost_deposit = drop
  // is refused by the constructor)
  if (pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic &&
      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::reflect &&
      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::outflow &&
      pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::user) {
    // (user faces, e.g. the Li et al. 2018 gas boundaries; the clouds only need the
    // faces to be non-periodic and <dust>/particle_x3_bc)
    bad.push_back("x3 faces other than periodic, reflect, outflow or user");
  }
  if (cloud_instant_switch) bad.push_back("cloud_instant_switch");
  // 2026-10-25 (Codex perf review item 2): the execution-order index (cloud_order_interval) has
  // 3-D keys; the physical sort (cloud_sort_interval) still has 2-D keys only
  if (cloud_sort_interval > 0) {
    bad.push_back("the particle sort (cloud_sort_interval)");
  }
  if (!bad.empty()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D refinable clouds do not support:";
    for (const auto &b : bad) std::cout << " [" << b << "]";
    std::cout << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (global_variable::my_rank == 0) {
    std::cout << "# NOTE (dust): 3-D refinable clouds (PC2 / IMEX; static or adaptive mesh; "
              << global_variable::nranks << " MPI rank" << (global_variable::nranks > 1 ? "s" : "")
              << ")" << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildLeafTraversal3D
//! \brief The block geometry of LeafTraversal3D: box, cell size, level and finest-cell box of
//! every block (from the logical locations, as AthenaK computes mb_size), the distinct adjacent
//! blocks of every block with the periodic shift that places each next to it (from the 56
//! neighbour slots), the ring its neighbours certainly hold, and the index of logical
//! locations; several ranks (stage 3): the halo slots and the exchange lists (BuildHalo).

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
  const auto &t = smr_transfer;
  // Physical x3 faces (no periodic image in x3)
  const bool zper = pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic;
  // block slots (stage 3): the blocks of this rank (slot = gid - gids), then
  // the blocks of other ranks that a cloud of this rank can reach (halo, ascending gid: the
  // neighbours, and every block within the reach of a block of this rank, HaloReach), then all
  // other blocks; with the box of cells of each halo block that the clouds can reach (strips)
  const int ntot = pm->nmb_total;
  std::vector<int> slot_of(ntot, -1), slot_gid;
  slot_gid.reserve(ntot);
  for (int m=0; m<nmb; ++m) {slot_of[gids + m] = m; slot_gid.push_back(gids + m);}
  std::unordered_map<int, std::array<int, 6>> hrect;
  {
    std::vector<int> h;
    for (int m=0; m<nmb; ++m) {
      for (int n=0; n<nslot; ++n) {
        const int g = nghbr.h_view(m, n).gid;
        if (g >= 0 && slot_of[g] < 0) h.push_back(g);
      }
    }
    if (global_variable::nranks > 1) {
      const int nl = t.cloud_nlev;
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
      const int nbr[3] = {pm->nmb_rootx1, pm->nmb_rootx2, pm->nmb_rootx3};
      const int nxb[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
      std::int64_t bw[3], nf[3];   // root block width, domain (finest cells)
      for (int d=0; d<3; ++d) {
        bw[d] = static_cast<std::int64_t>(nxb[d]) << nl;
        nf[d] = bw[d]*nbr[d];
      }
      auto fb = [&](const int g, std::int64_t b[6]) {
        const LogicalLocation &ll = pm->lloc_eachmb[g];
        const int sh = nl - (ll.level - pm->root_level);
        const std::int64_t lx[3] = {ll.lx1, ll.lx2, ll.lx3};
        for (int d=0; d<3; ++d) {
          b[2*d] = lx[d]*nxb[d] << sh;
          b[2*d + 1] = b[2*d] + (static_cast<std::int64_t>(nxb[d]) << sh);
        }
      };
      std::vector<std::vector<int>> bucket(static_cast<std::size_t>(nbr[0])*nbr[1]*nbr[2]);
      for (int g=0; g<ntot; ++g) {
        const LogicalLocation &ll = pm->lloc_eachmb[g];
        const int lev = ll.level - pm->root_level;
        bucket[(static_cast<std::size_t>(ll.lx3 >> lev)*nbr[1] + (ll.lx2 >> lev))*nbr[0] +
               (ll.lx1 >> lev)].push_back(g);
      }
      auto overlap = [](std::int64_t a0, std::int64_t a1, std::int64_t c0, std::int64_t c1,
                        std::int64_t n, const bool per = true) {
        for (int s=(per ? -1 : 0); s<=(per ? 1 : 0); ++s) {
          if (a0 + s*n < c1 && c0 < a1 + s*n) return true;
        }
        return false;
      };
      for (int m=0; m<nmb; ++m) {
        std::int64_t mb[6];
        fb(gids + m, mb);
        const std::int64_t rf = halo_reach_[m];   // finest cells
        int kk[3], bb[3];
        for (int d=0; d<3; ++d) {
          kk[d] = static_cast<int>((rf + bw[d] - 1)/bw[d]);
          bb[d] = static_cast<int>(mb[2*d]/bw[d]);
        }
        for (int dz=-kk[2]; dz<=kk[2]; ++dz) {
          for (int dy=-kk[1]; dy<=kk[1]; ++dy) {
            for (int dx=-kk[0]; dx<=kk[0]; ++dx) {
              const int cx = ((bb[0] + dx) % nbr[0] + nbr[0]) % nbr[0];
              const int cy = ((bb[1] + dy) % nbr[1] + nbr[1]) % nbr[1];
              if (!zper && (bb[2] + dz < 0 || bb[2] + dz >= nbr[2])) continue;
              const int cz = ((bb[2] + dz) % nbr[2] + nbr[2]) % nbr[2];
              for (const int g : bucket[(static_cast<std::size_t>(cz)*nbr[1] + cy)*nbr[0] + cx]) {
                if (slot_of[g] >= 0) continue;
                std::int64_t gb[6];
                fb(g, gb);
                bool in = true;
                for (int d=0; d<3 && in; ++d) {
                  in = overlap(gb[2*d], gb[2*d + 1], mb[2*d] - rf, mb[2*d + 1] + rf, nf[d],
                               d < 2 || zper);
                }
                if (!in) continue;
                h.push_back(g);
                // the part of g within the reach (every periodic image), in g's cells
                const LogicalLocation &gl = pm->lloc_eachmb[g];
                const std::int64_t ratio =
                    static_cast<std::int64_t>(1) << (nl - (gl.level - pm->root_level));
                std::array<int, 6> c;
                for (int d=0; d<3; ++d) {
                  std::int64_t r0 = gb[2*d + 1], r1 = gb[2*d];   // empty
                  const int sm = (d < 2 || zper) ? 1 : 0;
                  for (int sft=-sm; sft<=sm; ++sft) {
                    const std::int64_t a0 = std::max(gb[2*d], mb[2*d] - rf - sft*nf[d]);
                    const std::int64_t a1 = std::min(gb[2*d + 1], mb[2*d + 1] + rf - sft*nf[d]);
                    if (a0 < a1) {r0 = std::min(r0, a0); r1 = std::max(r1, a1);}
                  }
                  c[2*d] = static_cast<int>((r0 - gb[2*d])/ratio);
                  c[2*d + 1] = static_cast<int>((r1 - gb[2*d] + ratio - 1)/ratio);
                }
                auto it = hrect.find(g);
                if (it == hrect.end()) {
                  hrect[g] = c;
                } else {
                  for (int d=0; d<3; ++d) {
                    it->second[2*d] = std::min(it->second[2*d], c[2*d]);
                    it->second[2*d + 1] = std::max(it->second[2*d + 1], c[2*d + 1]);
                  }
                }
              }
            }
          }
        }
        // Shear-periodic x1: across an x1 face the reach lands at a y that moves
        // with time: every block within the reach of the opposite face whose z range meets the
        // reach, at any y (cells within the reach in x1 and z, every y)
        if (shear3d) {
          for (int side=0; side<2; ++side) {
            const bool cross = (side == 1) ? (mb[1] + rf > nf[0]) : (mb[0] - rf < 0);
            if (!cross) continue;
            for (int g=0; g<ntot; ++g) {
              if (slot_of[g] >= 0) continue;
              std::int64_t gb[6];
              fb(g, gb);
              const bool nearx = (side == 1) ? (gb[0] < rf) : (gb[1] > nf[0] - rf);
              if (!nearx || !overlap(gb[4], gb[5], mb[4] - rf, mb[5] + rf, nf[2], zper)) continue;
              h.push_back(g);
              const LogicalLocation &gl = pm->lloc_eachmb[g];
              const std::int64_t ratio =
                  static_cast<std::int64_t>(1) << (nl - (gl.level - pm->root_level));
              std::int64_t r0 = gb[5], r1 = gb[4];
              for (int sft=(zper ? -1 : 0); sft<=(zper ? 1 : 0); ++sft) {
                const std::int64_t a0 = std::max(gb[4], mb[4] - rf - sft*nf[2]);
                const std::int64_t a1 = std::min(gb[5], mb[5] + rf - sft*nf[2]);
                if (a0 < a1) {r0 = std::min(r0, a0); r1 = std::max(r1, a1);}
              }
              const std::int64_t x0 = (side == 1) ? gb[0] : std::max(gb[0], nf[0] - rf);
              const std::int64_t x1 = (side == 1) ? std::min(gb[1], rf) : gb[1];
              std::array<int, 6> c = {static_cast<int>((x0 - gb[0])/ratio),
                                      static_cast<int>((x1 - gb[0] + ratio - 1)/ratio),
                                      0, indcs.nx2,
                                      static_cast<int>((r0 - gb[4])/ratio),
                                      static_cast<int>((r1 - gb[4] + ratio - 1)/ratio)};
              auto it = hrect.find(g);
              if (it == hrect.end()) {
                hrect[g] = c;
              } else {
                for (int d=0; d<3; ++d) {
                  it->second[2*d] = std::min(it->second[2*d], c[2*d]);
                  it->second[2*d + 1] = std::max(it->second[2*d + 1], c[2*d + 1]);
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
  halo_.rect.assign(nfld - nmb, {0, indcs.nx1, 0, indcs.nx2, 0, indcs.nx3});
  if (cloud_halo_strips) {
    const int nxb[3] = {indcs.nx1, indcs.nx2, indcs.nx3};
    for (int b=nmb; b<nfld; ++b) {
      auto it = hrect.find(slot_gid[b]);
      if (it == hrect.end()) continue;
      auto c = it->second;
      bool ok = true;
      for (int d=0; d<3; ++d) {
        c[2*d] = std::max(0, c[2*d]); c[2*d + 1] = std::min(nxb[d], c[2*d + 1]);
        ok = ok && c[2*d] < c[2*d + 1];
      }
      if (ok) halo_.rect[b - nmb] = c;
    }
  }
  for (int g=0; g<ntot; ++g) {
    if (slot_of[g] < 0) {slot_of[g] = static_cast<int>(slot_gid.size()); slot_gid.push_back(g);}
  }
  const int nall = static_cast<int>(slot_gid.size());
  if (global_variable::nranks == 1 && (nfld != nmb || nall != nmb)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D leaf traversal: one rank but " << nfld - nmb << " halo and " << nall - nfld
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
  auto &lt = leaf_traversal3d;
  const auto &ms = pm->mesh_size;
  lt.nmb = nmb; lt.nfld = nfld; lt.nall = nall; lt.gids = gids;
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
    const LogicalLocation &ll = pm->lloc_eachmb[slot_gid[b]];
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
    if (b < nmb) {
      const auto &sz = mbsize.h_view(b);
      if (sz.x1min != x1a || sz.x2min != x2a || sz.x3min != x3a || sz.dx1 != dx1 ||
          sz.dx2 != dx2 || sz.dx3 != dx3) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "3-D leaf traversal: the box of block " << gids + b << " from its logical "
                  << "location differs from mb_size" << std::endl;
        std::exit(EXIT_FAILURE);
      }
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
                << "3-D leaf traversal: block " << slot_gid[b] << " (level " << dl
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
      const int b = slot_of[g];
      if (b >= nfld) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "3-D leaf traversal: neighbour " << g << " of block " << gids + m
                  << " has no halo slot" << std::endl;
        std::exit(EXIT_FAILURE);
      }
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
          if (!zper && (cz < 0 || cz >= lt.nfine3)) continue;   // (beyond a physical x3 face)
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
  // Shear-periodic x1, the data of the shear neighbours (UpdateShear3D): the finest
  // boxes of all slots, the slots near each x1 face, the reach of each local block (the halo's
  // on several ranks, the level rule on one)
  lt.shear = shear3d;
  lt.x3per = zper;
  lt.x3bc = zper ? 0 : pmy_pack->ppart->pbval_part->x3bc_dust;
  shear3d_lists_ok_ = false;
  shear3d_fbox_.clear(); shear3d_face_[0].clear(); shear3d_face_[1].clear();
  shear3d_reach_.assign(nmb, 0);
  if (shear3d) {
    int rmax = 0;
    for (int m=0; m<nmb; ++m) {
      std::int64_t reach = 0, allow = 0;
      if (global_variable::nranks > 1) {
        reach = halo_reach_[m];
      } else {
        HaloReach(blev(m), 0, reach, allow);
      }
      shear3d_reach_[m] = static_cast<int>(reach);
      rmax = std::max(rmax, shear3d_reach_[m]);
    }
    shear3d_fbox_.resize(nall);
    for (int b=0; b<nall; ++b) {
      for (int q=0; q<6; ++q) shear3d_fbox_[b][q] = fbox(b, q);
      if (fbox(b, 0) < rmax + 1) shear3d_face_[0].push_back(b);
      if (fbox(b, 1) > lt.nfine1 - rmax - 1) shear3d_face_[1].push_back(b);
    }
    lt.sreach = DvceArray1D<int>("dust_leaf3d_sreach", std::max(1, nmb));
    auto rh = Kokkos::create_mirror_view(lt.sreach);
    for (int m=0; m<nmb; ++m) rh(m) = shear3d_reach_[m];
    Kokkos::deep_copy(lt.sreach, rh);
    UpdateShear3D(pm->time);
  }
  halo_allow_d_ = DvceArray1D<std::int64_t>("dust_halo_allow", std::max(nmb, 1));
  if (global_variable::nranks > 1) {
    auto ah = Kokkos::create_mirror_view(halo_allow_d_);
    for (int m=0; m<nmb; ++m) ah(m) = halo_allow_[m];
    Kokkos::deep_copy(halo_allow_d_, ah);
  }
  if (global_variable::my_rank == 0) {
    int mxnb = 0;
    for (int m=0; m<nmb; ++m) mxnb = std::max(mxnb, nnb(m));
    std::cout << "# 3-D leaf traversal: " << nmb << " blocks";
    if (global_variable::nranks > 1) {
      std::cout << " on rank 0 (" << nfld - nmb << " halo, " << ntot << " in all)";
    }
    std::cout << ", levels " << lt.nlev << ", finest grid " << lt.nfine1 << "x" << lt.nfine2
              << "x" << lt.nfine3 << ", at most " << mxnb << " neighbours"
              << (nunsafe > 0 ? ", " + std::to_string(nunsafe) + " blocks without a certain ring"
                              : "") << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::WrapParticles3D, ResetBlocks3D
//! \brief Stage 2a: every position (with its RK register) into the box; then the
//! block of every cloud from its position on the current mesh (LeafOwner on the 3-D leaves) and,
//! with check, the sum of its weights over the leaves (largest |sum - 1| in dwmax) and the
//! particle mass.  Returns how many block ids changed; stops on a particle without a block of
//! this rank or with weights outside the blocks.

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::UpdateShear3D
//! \brief Shear-periodic x1: the shear offset ysh = fmod(q Omega Lx t, Ly) of the
//! time t of the positions to be traversed (bvals_part.cpp) and, when it changed, the shear
//! neighbours of every local block: for each side, the blocks that meet the block translated
//! across that x1 face ((+Lx, -ysh) across the inner face, (-Lx, +ysh) across the outer one)
//! and widened by its reach, with their periodic shifts in y and z (none in x).  Host lists from
//! the finest-cell boxes of the slots near the opposite face; a few per block.

void DustGasDrag::UpdateShear3D(const Real time) {
  if (!shear3d) return;
  auto &lt = leaf_traversal3d;
  const Real ysh = std::fmod(qshear*omega0*lt.lx*time, lt.ly);   // as bvals_part.cpp
  if (shear3d_lists_ok_ && ysh == lt.ysh) return;
  ++shear3d_updates;
  const double yf = static_cast<double>(ysh)/static_cast<double>(lt.fine_dx2);
  // (cloud_shear_list_cache) the lists of the unit interval [k, k + 1) of yf: block boxes and
  // reaches are whole finest cells, so the windows of all yf in it lie in the window of the
  // interval; the traversal still tests every candidate at the current ysh (Candidates, meets)
  const double kf = std::floor(yf);
  lt.ysh = ysh;
  if (shear3d_list_cache && shear3d_lists_ok_ && kf == shear3d_key_) return;
  ++shear3d_rebuilds;
  shear3d_key_ = kf;
  const int nmb = lt.nmb, nf1 = lt.nfine1, nf2 = lt.nfine2, nf3 = lt.nfine3;
  const auto &fb = shear3d_fbox_;
  std::vector<std::vector<int>> lst(2*static_cast<std::size_t>(nmb));
  int kmax = 1;
  for (int m=0; m<nmb; ++m) {
    const int R = shear3d_reach_[m];
    for (int side=0; side<2; ++side) {
      const int sg = 2*side - 1;   // across the outer face (1): x - Lx, y + ysh
      const int x0 = std::max(fb[m][0] - sg*nf1 - R, 0), x1 = std::min(fb[m][1] - sg*nf1 + R, nf1);
      if (x0 >= x1) continue;
      const double ylo = shear3d_list_cache ? std::min(sg*kf, sg*(kf + 1.0)) : sg*yf;
      const double yhi = shear3d_list_cache ? std::max(sg*kf, sg*(kf + 1.0)) : sg*yf;
      const double y0 = fb[m][2] + ylo - R, y1 = fb[m][3] + yhi + R;
      const int z0 = fb[m][4] - R, z1 = fb[m][5] + R;
      auto &l = lst[2*static_cast<std::size_t>(m) + side];
      for (const int b : shear3d_face_[1 - side]) {
        if (!(fb[b][0] < x1 && x0 < fb[b][1])) continue;
        for (int sy=-2; sy<=2; ++sy) {
          const double by0 = fb[b][2] - static_cast<double>(sy)*nf2;
          const double by1 = fb[b][3] - static_cast<double>(sy)*nf2;
          if (!(by0 < y1 && y0 < by1)) continue;
          for (int sz=(lt.x3per ? -1 : 0); sz<=(lt.x3per ? 1 : 0); ++sz) {
            const int bz0 = fb[b][4] - sz*nf3, bz1 = fb[b][5] - sz*nf3;
            if (!(bz0 < z1 && z0 < bz1)) continue;
            if (b >= lt.nfld) {
              std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                        << std::endl << "3-D clouds, shear-periodic x1: shear neighbour slot " << b
                        << " of block " << lt.gids + m << " is beyond the halo" << std::endl;
              std::exit(EXIT_FAILURE);
            }
            l.push_back(LeafTraversal3D::Pack(b, 0, sy, sz));
          }
        }
      }
      kmax = std::max(kmax, static_cast<int>(l.size()));
    }
  }
  if (lt.nsnb.extent_int(0) != std::max(1, nmb) || lt.snb.extent_int(2) < kmax) {
    lt.nsnb = DvceArray2D<int>("dust_leaf3d_nsnb", std::max(1, nmb), 2);
    lt.snb = DvceArray3D<int>("dust_leaf3d_snb", std::max(1, nmb), 2, kmax);
  }
  auto nh = Kokkos::create_mirror_view(lt.nsnb);
  auto sh = Kokkos::create_mirror_view(lt.snb);
  for (int m=0; m<nmb; ++m) {
    for (int side=0; side<2; ++side) {
      const auto &l = lst[2*static_cast<std::size_t>(m) + side];
      nh(m, side) = static_cast<int>(l.size());
      for (std::size_t k=0; k<l.size(); ++k) sh(m, side, k) = l[k];
    }
  }
  Kokkos::deep_copy(lt.nsnb, nh);
  Kokkos::deep_copy(lt.snb, sh);
  shear3d_lists_ok_ = true;
}

void DustGasDrag::WrapParticles3D() {
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  if (npart == 0) return;
  auto pr = ppar->prtcl_rdata;
  const auto &ms = pmy_pack->pmesh->mesh_size;
  const Real x1a = ms.x1min, x1b = ms.x1max, x2a = ms.x2min, x2b = ms.x2max;
  const Real x3a = ms.x3min, x3b = ms.x3max;
  // Across a shear-periodic x1 face y moves by n ysh (with its register)
  const bool sh = leaf_traversal3d.shear;
  const Real ysh = leaf_traversal3d.ysh, ilx = 1.0/(x1b - x1a);
  const bool zper = pmy_pack->pmesh->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic;
  Kokkos::parallel_for("dust_wrap3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (sh) {
      const Real x = pr(IPX,p), xw = SMRCompositeTransfer::WrapPosition(x, x1a, x1b);
      if (xw != x) {
        const Real dy = round((x - xw)*ilx)*ysh;
        pr(IPX,p) = xw; pr(IPX1,p) += xw - x;
        pr(IPY,p) += dy; pr(IPY1,p) += dy;
      }
    }
    SMRCompositeTransfer::WrapParticle(pr, p, x1a, x1b, x2a, x2b);
    if (!zper) return;   // (physical x3 faces: the particle routing reflects or removes)
    const Real z = pr(IPZ,p), zw = SMRCompositeTransfer::WrapPosition(z, x3a, x3b);
    if (zw != z) {pr(IPZ,p) = zw; pr(IPZ1,p) += zw - z;}
  });
}

int DustGasDrag::ResetBlocks3D(const char *when, const bool check, Real &dwmax, double &msum) {
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  dwmax = 0.0; msum = 0.0;
  if (npart == 0) return 0;
  WrapParticles3D();
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const auto lt = leaf_traversal3d;
  const int ncell123 = lt.ncells1*lt.ncells2*lt.ncells3;
  const int nmb = pmy_pack->nmb_thispack, gids = pmy_pack->gids;
  int nchanged = 0, nbad = 0;
  Kokkos::parallel_reduce("dust_reset_blocks3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &changed, int &bad, Real &dw, double &ms) {
    const Real x = pr(IPX,p), y = pr(IPY,p), z = pr(IPZ,p);
    const int leaf = lt.LeafOwner(x, y, z, lt.Anchor(pi(PGID,p)));   // the old id: a guess
    const int owner = (leaf < 0) ? -1 : leaf/ncell123;
    if (owner < 0 || owner >= nmb) {++bad; return;}
    if (pi(PGID,p) != gids + owner) {pi(PGID,p) = gids + owner; ++changed;}
    ms += static_cast<double>(pr(IPM,p));
    if (!check) return;
    int out = 0;
    const Real sw = lt.ForEach<false>(x, y, z, pi(PLEV,p), owner,
                                      [&](const int t, const Real) {
      if (t < 0 || t/ncell123 >= lt.nfld) ++out;
    });
    dw = Kokkos::fmax(dw, Kokkos::fabs(sw - 1.0));
    bad += out;
  }, Kokkos::Sum<int>(nchanged), Kokkos::Sum<int>(nbad), Kokkos::Max<Real>(dwmax),
     Kokkos::Sum<double>(msum));
  if (nbad > 0 || dwmax > 1.0e-10) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "3-D clouds (" << when << "): " << nbad << " particles without a block or with "
              << "weights outside the blocks, largest |sum of weights - 1| = " << dwmax << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return nchanged;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::FrameToBlock3D, CheckHaloReach3D, RouteToOwnerBlocks3D
//! \brief Stage 3: several ranks, the 2-D multi-rank functions with a third
//! direction.  FrameToBlock3D: a particle in the periodic image next to the block it keeps
//! (positions and RK position registers), so that the particle exchange sees a step of one
//! block.  CheckHaloReach3D: how far each cloud's support (two half-widths on a lobed side)
//! extends beyond its block, against the block's allowance; where it exceeds it the block's reach
//! is widened and every rank rebuilds the leaf traversal and the halo.  RouteToOwnerBlocks3D:
//! every particle to the block (and rank) that holds its position.

void DustGasDrag::FrameToBlock3D(const Real tpos) {
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const auto lt = leaf_traversal3d;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  if (npart == 0) return;
  const bool frame = global_variable::nranks > 1;
  // The particle routing (bvals_part.cpp) moves a particle across a shear-periodic x1
  // face by ysh at pm->time; positions at tpos != pm->time (after the push) get the difference
  // first, so that the move is the one of their time
  Real dsh = 0.0, ysp = lt.ysh;   // ysp: the shear offset at the time of the positions
  if (shear3d && shear3d_route_fix && tpos >= 0.0) {
    const Real t0 = pmy_pack->pmesh->time;
    ysp = std::fmod(qshear*omega0*lt.lx*tpos, lt.ly);
    // the difference unfolded: the folded values jump by ly when q Omega lx t passes a
    // multiple of ly, which moved the crossing particles by a whole box in y (routing abort)
    dsh = qshear*omega0*lt.lx*(tpos - t0);
  }
  if (!frame && dsh == 0.0) return;
  const bool sh = lt.shear;
  Kokkos::parallel_for("dust_migrate_frame3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (frame) {
      const int m = lt.Anchor(pi(PGID,p));
      const Real cx = lt.box(m,0) + 0.5*lt.nx1*lt.box(m,3);
      const Real cy = lt.box(m,1) + 0.5*lt.nx2*lt.box(m,4);
      const Real cz = lt.box(m,2) + 0.5*lt.nx3*lt.box(m,5);
      const Real n1 = round((pr(IPX,p) - cx)*lt.ilx);
      const Real sx = lt.lx*n1;
      if (sx != 0.0) {
        pr(IPX,p) -= sx; pr(IPX1,p) -= sx;
        if (sh) {pr(IPY,p) += n1*ysp; pr(IPY1,p) += n1*ysp;}
      }
      const Real sy = lt.ly*round((pr(IPY,p) - cy)*lt.ily);
      const Real sz = lt.x3per ? lt.lz*round((pr(IPZ,p) - cz)*lt.ilz) : 0.0;
      if (sy != 0.0) {pr(IPY,p) -= sy; pr(IPY1,p) -= sy;}
      if (sz != 0.0) {pr(IPZ,p) -= sz; pr(IPZ1,p) -= sz;}
    }
    if (dsh != 0.0) {
      const Real x = pr(IPX,p);
      const Real c = (x >= lt.x1max) ? 1.0 : ((x < lt.x1min) ? -1.0 : 0.0);
      if (c != 0.0) {pr(IPY,p) += c*dsh; pr(IPY1,p) += c*dsh;}
    }
  });
}

bool DustGasDrag::CheckHaloReach3D(const char *when) {
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  const int nmb = pmy_pack->nmb_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const auto lt = leaf_traversal3d;
  auto allow = halo_allow_d_;
  DvceArray1D<std::int64_t> need("dust_halo_need3d", std::max(nmb, 1));
  int nover = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_halo_reach3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &cnt) {
      using T = SMRCompositeTransfer;
      const int m = lt.Anchor(pi(PGID,p));
      Real x = pr(IPX,p), y = pr(IPY,p), z = pr(IPZ,p);
      lt.Frame(m, x, y, z);
      const int code = pi(PLEV,p);
      const int lev = T::CloudLevel(code);
      const Real e1 = (T::CloudLobe1(code) ? 2.0 : 1.0)*lt.CloudHalfWidth1(lev);
      const Real e2 = (T::CloudLobe2(code) ? 2.0 : 1.0)*lt.CloudHalfWidth2(lev);
      const Real e3 = (T::CloudLobe3(code) ? 2.0 : 1.0)*lt.CloudHalfWidth3(lev);
      const Real bx0 = lt.box(m,0), bx1 = bx0 + lt.nx1*lt.box(m,3);
      const Real by0 = lt.box(m,1), by1 = by0 + lt.nx2*lt.box(m,4);
      const Real bz0 = lt.box(m,2), bz1 = bz0 + lt.nx3*lt.box(m,5);
      const Real o1 = Kokkos::fmax(0.0, Kokkos::fmax(bx0 - (x - e1), (x + e1) - bx1));
      const Real o2 = Kokkos::fmax(0.0, Kokkos::fmax(by0 - (y - e2), (y + e2) - by1));
      const Real o3 = Kokkos::fmax(0.0, Kokkos::fmax(bz0 - (z - e3), (z + e3) - bz1));
      const std::int64_t d = static_cast<std::int64_t>(
          Kokkos::fmax(Kokkos::ceil(o1/lt.fine_dx1),
                       Kokkos::fmax(Kokkos::ceil(o2/lt.fine_dx2), Kokkos::ceil(o3/lt.fine_dx3))));
      if (d > allow(m)) {
        const std::int64_t margin = static_cast<std::int64_t>(1) << (lt.nlev - lt.blev(m));
        Kokkos::atomic_max(&need(m), d + margin);
        ++cnt;
      }
    }, Kokkos::Sum<int>(nover));
  }
  long long tot = nover;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &tot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (tot == 0) return false;
  auto nh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), need);
  int nwide = 0;
  for (int m=0; m<nmb; ++m) {
    if (nh(m) > halo_need_[m]) {halo_need_[m] = nh(m); ++nwide;}
  }
  halo_widenings_ += static_cast<unsigned long long>(nwide);
  const int nhalo_before = leaf_traversal3d.nfld - leaf_traversal3d.nmb;
  BuildLeafTraversal3D();
  if (global_variable::my_rank == 0) {
    std::cout << "# dust halo (3-D): " << tot << " clouds reached beyond the halo ("
              << when << ", cycle " << pmy_pack->pmesh->ncycle << "); rank 0 widened " << nwide
              << " blocks, halo blocks " << nhalo_before << " -> "
              << leaf_traversal3d.nfld - leaf_traversal3d.nmb << std::endl;
  }
  return true;
}

int DustGasDrag::RouteToOwnerBlocks3D(const char *when) {
  particles::Particles *ppar = pmy_pack->ppart;
  auto *pb = ppar->pbval_part;
  int nchanged = 0, nbad = 0;
#if MPI_PARALLEL_ENABLED
  WrapParticles3D();
  const int npart = ppar->nprtcl_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const auto lt = leaf_traversal3d;
  const int ncell123 = lt.ncells1*lt.ncells2*lt.ncells3;
  const int nmb = pmy_pack->nmb_thispack, gids = pmy_pack->gids;
  auto sgid = leaf_slot_gid_d;
  auto srank = leaf_slot_rank_d;
  if (pb->sendlist.extent_int(0) < std::max(npart, 1)) Kokkos::realloc(pb->sendlist, std::max(npart, 1));
  Kokkos::deep_copy(pb->send_count.d_view, 0);
  auto psendl = pb->sendlist.d_view;
  int *pcounter = pb->send_count.d_view.data();
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_route_owner3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &changed, int &bad) {
      const int leaf = lt.LeafOwner(pr(IPX,p), pr(IPY,p), pr(IPZ,p), lt.Anchor(pi(PGID,p)));
      const int slot = (leaf < 0) ? -1 : leaf/ncell123;
      if (slot < 0) {++bad; return;}
      const int g = (slot < nmb) ? gids + slot : sgid(slot);
      if (pi(PGID,p) != g) {pi(PGID,p) = g; ++changed;}
      if (slot >= nmb) {
        const int k = Kokkos::atomic_fetch_add(pcounter, 1);
        psendl(k).prtcl_indx = p;
        psendl(k).dest_gid = g;
        psendl(k).dest_rank = srank(slot);
      }
    }, Kokkos::Sum<int>(nchanged), Kokkos::Sum<int>(nbad));
  }
  if (nbad > 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << nbad << " particles have no owning MeshBlock (" << (when ? when : "split")
              << ", 3-D)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Kokkos::deep_copy(pb->send_count.h_view, pb->send_count.d_view);
  pb->nprtcl_send = pb->send_count.h_view(0);
  if (pb->nprtcl_send > 0) {
    auto d = Kokkos::subview(pb->sendlist.d_view, std::make_pair(0, pb->nprtcl_send));
    auto h = Kokkos::subview(pb->sendlist.h_view, std::make_pair(0, pb->nprtcl_send));
    Kokkos::deep_copy(h, d);
  }
  const int nsent = pb->nprtcl_send;
  pb->CountSendsAndRecvs();
  pb->InitPrtclRecv();
  pb->PackAndSendPrtcls();
  while (pb->RecvAndUnpackPrtcls() != TaskStatus::complete) {}
  pb->ClearPrtclSend();
  pb->ClearPrtclRecv();
  Mesh *pm = pmy_pack->pmesh;
  long long tot = 0;
  for (int n=0; n<global_variable::nranks; ++n) tot += pm->nprtcl_eachrank[n];
  pm->nprtcl_total = static_cast<int>(tot);
  long long sums[2] = {nchanged, nsent};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (global_variable::my_rank == 0 && when != nullptr) {   // nullptr: quiet (SplitToLeaves)
    std::cout << "# dust (3-D): MeshBlock ids set from the positions at the " << when
              << ": " << sums[0] << " of " << tot << " changed, " << sums[1]
              << " sent to other ranks" << std::endl;
  }
#else
  (void)pb; (void)nbad; (void)when;
#endif
  return nchanged;
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
  auto lt = leaf_traversal3d;
  // several ranks (stage 3): the leaves with fields (this rank's blocks and the halo); the clouds
  // then sit in their anchor block, at the levels the halo reach covers (HaloReach)
  const bool multi = global_variable::nranks > 1;
  const int nleaf = lt.nfld*lt.ncells3*lt.ncells2*lt.ncells1;
  auto blev_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lt.blev);
  auto box_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), lt.box);
  DvceArray2D<Real> d("dust_leaf3d_check", nleaf, 4);
  DvceArray1D<Real> res("dust_leaf3d_check_res", 4);
  double maxw = 0.0, maxm = 0.0, maxsum = 0.0;
  int nelig_bad = 0, nglobal = 0;
  const Real slack = cloud_split_slack;
  const std::uint64_t seed = static_cast<std::uint64_t>(pmy_pack->pmesh->ncycle) + 12345;
  // shear-periodic x1: every sample at a random shear offset (the shear neighbours
  // rebuilt for it; the offset of the current time restored at the end)
  const Real tshear = (shear3d && qshear*omega0 != 0.0)
                    ? lt.ly/(qshear*omega0*lt.lx) : 0.0;   // the time of one full offset
  for (int s=0; s<nsample; ++s) {
    const std::uint64_t key = seed*UINT64_C(1000003) + static_cast<std::uint64_t>(s)*7919;
    if (tshear > 0.0) {
      UpdateShear3D(tshear*Uniform01(key + 13));
      lt = leaf_traversal3d;
    }
    const int anchor0 = std::min(lt.nmb - 1, static_cast<int>(Uniform01(key + 10)*lt.nmb));
    Real px = lt.x1min + Uniform01(key + 1)*lt.lx;
    Real py = lt.x2min + Uniform01(key + 2)*lt.ly;
    Real pz = lt.x3min + Uniform01(key + 3)*lt.lz;
    int lmin = 0;
    // shear-periodic x1: a third of the clouds within two root cells of an x1 face
    if (lt.shear && !multi && Uniform01(key + 11) < 1.0/3.0) {
      const Real u = 2.0*lt.root_dx1*Uniform01(key + 1);
      px = (Uniform01(key + 12) < 0.5) ? lt.x1min + u : lt.x1max - u;
    }
    // (physical x3 faces: a third of the clouds within two root cells of one)
    if (!lt.x3per && !multi && Uniform01(key + 15) < 1.0/3.0) {
      const Real u = 2.0*lt.root_dx3*Uniform01(key + 3);
      pz = (Uniform01(key + 16) < 0.5) ? lt.x3min + u : lt.x3max - u;
    }
    if (multi) {
      px = box_h(anchor0,0) + Uniform01(key + 1)*lt.nx1*box_h(anchor0,3);
      py = box_h(anchor0,1) + Uniform01(key + 2)*lt.nx2*box_h(anchor0,4);
      pz = box_h(anchor0,2) + Uniform01(key + 3)*lt.nx3*box_h(anchor0,5);
      lmin = std::max(0, blev_h(anchor0) - 1);
    }
    const int lev = std::min(lt.nlev, lmin + static_cast<int>(Uniform01(key + 4)*(lt.nlev + 1 - lmin)));
    int code = lev;
    if (lev >= 1 && Uniform01(key + 5) < 0.5) {
      code |= (static_cast<int>(Uniform01(key + 6)*4) & 3) << SMRCompositeTransfer::kCloudLobeShift1;
      code |= (static_cast<int>(Uniform01(key + 7)*4) & 3) << SMRCompositeTransfer::kCloudLobeShift2;
      code |= (static_cast<int>(Uniform01(key + 8)*4) & 3) << SMRCompositeTransfer::kCloudLobeShift3;
    }
    if (Uniform01(key + 9) < 0.3) code |= SMRCompositeTransfer::kCloudGatherPoint;
    // one rank: two thirds of the clouds anchored in the block that holds them (the neighbour
    // and shear-neighbour lists), the others in a random block (the searches over all blocks)
    const bool own = !multi && Uniform01(key + 14) < 2.0/3.0;
    Kokkos::deep_copy(d, 0.0);
    Kokkos::parallel_for("dust_leaf3d_check", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
    KOKKOS_LAMBDA(const int) {
      using LT = LeafTraversal;
      using T = SMRCompositeTransfer;
      bool reach = true;
      int anchor = anchor0;
      if (own) {
        int b = -1;
        lt.Owner(px, py, pz, anchor0, &b);
        if (b >= 0 && b < lt.nmb) anchor = b;
      }
      // the first moments about the leaf centres, as the gather uses them (m1 + MinImage(p - c) w)
      auto centre = [&](const int t, Real &xc, Real &yc, Real &zc) {
        int tb, tk, tj, ti;
        lt.Decode(t, tb, tk, tj, ti);
        xc = lt.box(tb,0) + (ti - lt.is + 0.5)*lt.box(tb,3);
        yc = lt.box(tb,1) + (tj - lt.js + 0.5)*lt.box(tb,4);
        zc = lt.box(tb,2) + (tk - lt.ks + 0.5)*lt.box(tb,5);
      };
      const Real sum = lt.ForEach<true>(px, py, pz, code, anchor,
      [&](const int t, const Real w, const Real m1x, const Real m1y, const Real m1z) {
        Real xc, yc, zc;
        centre(t, xc, yc, zc);
        d(t, 0) += w;
        d(t, 1) += m1x + T::MinImage(px - xc, lt.lx)*w;
        d(t, 2) += m1y + T::MinImage(py - yc, lt.ly)*w;
        d(t, 3) += m1z + T::MinImage(pz - zc, lt.lz)*w;
      }, &reach);
      res(0) = sum;
      res(3) = reach ? 0.0 : 1.0;
      // brute force over the finest cells of the support (unwrapped indices; Locate wraps); with
      // shear-periodic x1 and the support across an x1 face, the part in the domain and the cloud
      // translated across the face, each over its finest cells in the domain
      const int lv = T::CloudLevel(code);
      const bool point = (code & T::kCloudGatherPoint) != 0;
      const LT::Shape s1a = LT::MakeShape(px, ldexp(lt.root_dx1, -lv), ldexp(lt.iroot_dx1, lv),
                                          T::CloudLobe1(code), point);
      const LT::Shape s3a = LT::MakeShape(pz, ldexp(lt.root_dx3, -lv), ldexp(lt.iroot_dx3, lv),
                                          T::CloudLobe3(code), point);
      const int npx = (lt.shear && (s1a.lo < lt.x1min || s1a.hi > lt.x1max)) ? 2 : 1;
      const Real sgn = (s1a.hi > lt.x1max) ? 1.0 : -1.0;
      // (physical x3 faces: the part beyond a face mirrored back)
      const int npz = (!lt.x3per && (s3a.lo < lt.x3min || s3a.hi > lt.x3max)) ? 2 : 1;
      const Real zf = (s3a.hi > lt.x3max) ? lt.x3max : lt.x3min;
      for (int part=0; part<npx*npz; ++part) {
      const int pxp = part % npx, pzp = part/npx;
      const Real qx = (pxp == 0) ? px : px - sgn*lt.lx;
      const Real qy = (pxp == 0) ? py : py + sgn*lt.ysh;
      const Real qz = (pzp == 0) ? pz : 2.0*zf - pz;
      const int l3 = (pzp == 0) ? T::CloudLobe3(code)
                                : LeafTraversal3D::MirrorLobe(T::CloudLobe3(code));
      const LT::Shape s1 = LT::MakeShape(qx, ldexp(lt.root_dx1, -lv), ldexp(lt.iroot_dx1, lv),
                                         T::CloudLobe1(code), point);
      const LT::Shape s2 = LT::MakeShape(qy, ldexp(lt.root_dx2, -lv), ldexp(lt.iroot_dx2, lv),
                                         T::CloudLobe2(code), point);
      const LT::Shape s3 = LT::MakeShape(qz, ldexp(lt.root_dx3, -lv), ldexp(lt.iroot_dx3, lv),
                                         l3, point);
      int ia = static_cast<int>(floor((s1.lo - lt.x1min)/lt.fine_dx1));
      int ib = static_cast<int>(ceil((s1.hi - lt.x1min)/lt.fine_dx1)) - 1;
      if (npx == 2) {ia = (ia < 0) ? 0 : ia; ib = (ib > lt.nfine1 - 1) ? lt.nfine1 - 1 : ib;}
      const int ja = static_cast<int>(floor((s2.lo - lt.x2min)/lt.fine_dx2));
      const int jb = static_cast<int>(ceil((s2.hi - lt.x2min)/lt.fine_dx2)) - 1;
      int ka = static_cast<int>(floor((s3.lo - lt.x3min)/lt.fine_dx3));
      int kb = static_cast<int>(ceil((s3.hi - lt.x3min)/lt.fine_dx3)) - 1;
      lt.ClipZ(ka, kb);
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
            if (t < 0 || t >= nleaf) {res(1) = 1.0e30; continue;}
            Real xc, yc, zc;
            centre(t, xc, yc, zc);
            const Real w = w1*w2*w3;
            d(t, 0) -= w;
            d(t, 1) -= g1*w2*w3 + T::MinImage(qx - xc, lt.lx)*w;
            d(t, 2) -= w1*g2*w3 + T::MinImage(qy - yc, lt.ly)*w;
            d(t, 3) -= w1*w2*g3 + T::MinImage(qz - zc, lt.lz)*w;
          }
        }
      }
      }
      // eligibility of a plain cloud of level lv at the position (overlap rule)
      if (lt.split_overlap && lv < lt.nlev && lv < lt.cloud_max_level && !lt.split_everywhere) {
        const bool fast = lt.InsideFiner(px, py, pz, lv, slack, anchor);
        bool brute = false;
        int fa, fb, ga, gb, ha, hb;
        lt.SupportBox(px, py, pz, 1.0 - slack, lv, fa, fb, ga, gb, ha, hb);
        // (shear-periodic x1: the part in the domain and the box around the translated position)
        const int side = lt.ShearSide(fa, fb);
        for (int part=0; part<((side >= 0) ? 2 : 1); ++part) {
          if (part == 1) {
            const Real sg = (side == 1) ? 1.0 : -1.0;
            lt.SupportBox(px - sg*lt.lx, py + sg*lt.ysh, pz, 1.0 - slack, lv, fa, fb, ga, gb, ha, hb);
          }
          if (side >= 0) {fa = (fa < 0) ? 0 : fa; fb = (fb > lt.nfine1 - 1) ? lt.nfine1 - 1 : fb;}
          for (int k=ha; k<=hb && !brute; ++k) {
            for (int j=ga; j<=gb && !brute; ++j) {
              for (int i=fa; i<=fb && !brute; ++i) {
                int b = -1;
                lt.Locate(i, j, k, &b);
                if (b >= 0 && lt.blev(b) > lv) brute = true;
              }
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
  if (tshear > 0.0) UpdateShear3D(pmy_pack->pmesh->time);
#if MPI_PARALLEL_ENABLED
  if (multi) {
    double mx[3] = {maxw, maxm, maxsum};
    int ni[2] = {nelig_bad, nglobal};
    MPI_Allreduce(MPI_IN_PLACE, mx, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ni, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    maxw = mx[0]; maxm = mx[1]; maxsum = mx[2]; nelig_bad = ni[0]; nglobal = ni[1];
  }
#endif
  // (the moments about the leaf centres carry the round-off of the positions, about 1e-16 of the
  // domain size: relative to the half-width at the deepest level up to a few 1e-14)
  const bool ok = (maxw <= 1.0e-13 && maxm <= 1.0e-11 && maxsum <= 1.0e-13 && nelig_bad == 0);
  if (global_variable::my_rank == 0) {
    std::cout << "# 3-D leaf traversal check, cycle " << pmy_pack->pmesh->ncycle
              << ": " << nsample << " clouds" << (multi ? " per rank" : "")
              << ", max |weight - brute| " << maxw
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
      Real cx = x + p1[a].off*e1, cy = y + p2[b].off*e2, cz = z + p3[c].off*e3;
      // (shear-periodic x1: y moves with an x wrap; a physical x3 face: a piece
      // beyond it is mirrored back with its x3 lobe swapped)
      const int zmir = lt.WrapPos(cx, cy, cz);
      if (k > 0) {
        for (int q=0; q<nrdata; ++q) prv(q,s) = prv(q,p);
        for (int q=0; q<nidata; ++q) piv(q,s) = piv(q,p);
        piv(PTAG,s) = tag0 + (tagoff(p) + k - 1)*tstride;
      }
      prv(IPX,s) = cx; prv(IPX1,s) = cx;
      prv(IPY,s) = cy; prv(IPY1,s) = cy;
      prv(IPZ,s) = cz; prv(IPZ1,s) = cz;
      prv(IPM,s) = mass*p1[a].frac*p2[b].frac*p3[c].frac;
      piv(PLEV,s) = clev | (p1[a].lobe << sh1) | (p2[b].lobe << sh2) |
                    ((zmir ? LeafTraversal3D::MirrorLobe(p3[c].lobe) : p3[c].lobe) << sh3);
      if (zmir && lt.x3bc == 1) {   // a reflecting face: the velocity mirrored too
        prv(IPVZ,s) = -prv(IPVZ,s); prv(IPVZ1,s) = -prv(IPVZ1,s);
      }
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
  // room for the halo slots (their values come from HaloForwardPack)
  const int nmb = pmy_pack->nmb_thispack, nk = vel.extent_int(2);
  const int nj = vel.extent_int(3), ni = vel.extent_int(4);
  const int nslot = std::max(std::max(nmb, vel.extent_int(0)), leaf_traversal3d.nfld);
  if (gpack3.extent_int(0) != nslot || gpack3.extent_int(1) != nk ||
      gpack3.extent_int(2) != nj || gpack3.extent_int(3) != ni) {
    Kokkos::realloc(gpack3, nslot, nk, nj, ni, 12);
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
  HaloForwardPack(gpack3);   // the owners' values on the halo blocks
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
  UpdateShear3D(pmy_pack->pmesh->time);   // positions at the start of the step
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
  if (cloud_ord_n > 0) {   // (cloud_order_interval) in the cell-sorted index order
    CloudParForOrdered("dust_pc2_cloud3d_predictor", npart, cloud_ord, cloud_ord_n, body);
  } else {
    CloudParFor("dust_pc2_cloud3d_predictor", npart, body);
  }
  // deposits on halo blocks back to their owners; component 4 (stiffness) only when the automatic
  // PC2 / split-BE switch reads it (rate_dep, the same on every rank; 2026-10-25, Codex perf review)
  HaloReverseAdd(qdep, 1, rate_dep ? 4 : 3);
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
  // The gas is sampled in its t^n orbital-advection frame (the remap runs after this
  // gather and the impulse deposit), so the shear-periodic images are those of t^n
  UpdateShear3D(pmy_pack->pmesh->time);
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
    // The gas met at the lab midpoint (xh, yh) sits at yh + q Omega xh dt/2 in the stored
    // field (IPY1 keeps the lab midpoint)
    const Real yg = yh + (is_sbox ? qshear_*omega0_*xh*0.5*dt : 0.0);
    Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
    const int glev = T::GatherCode(pi(PLEV,p), tmode);
    const bool same = (glev & ~T::kCloudGatherPoint) == pi(PLEV,p);
    lt.ForEachCloudMoment(xh, yg, zh, glev, am,
    [&](const int target, const Real w, const Real m1x, const Real m1y, const Real m1z) {
      int tb, tk, tj, ti;
      lt.Decode(target, tb, tk, tj, ti);
      const Real dx1 = lt.box(tb,3), dx2 = lt.box(tb,4), dx3 = lt.box(tb,5);
      const Real xc = lt.box(tb,0) + (ti - is + 0.5)*dx1;
      const Real yc = lt.box(tb,1) + (tj - js + 0.5)*dx2;
      const Real zc = lt.box(tb,2) + (tk - ks + 0.5)*dx3;
      const Real cx = 0.5*(m1x + T::MinImage(xh - xc, lt.lx)*w)/dx1;
      const Real cy = 0.5*(m1y + T::MinImage(yg - yc, lt.ly)*w)/dx2;
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
      lt.ForEachCloudWeight(xh, yg, zh, pi(PLEV,p), am, add);
    }
  };
  if (cloud_ord_n > 0) {   // (cloud_order_interval) in the cell-sorted index order
    CloudParForOrdered("dust_pc2_cloud3d_advance", npart, cloud_ord, cloud_ord_n, body);
  } else {
    CloudParFor("dust_pc2_cloud3d_advance", npart, body);
  }
  if (br) HaloReverseAdd(dmom, 0, 3);   // deposits on halo blocks back to their owners
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositCloudDensity3D
//! \brief Dust density (component 0) and momentum density (1-3) of the clouds on the leaves of
//! this rank (diagnostics; f is (re)allocated to the size of qdep with 4 components).  Several
//! ranks: the deposits on halo blocks are added to their owners (the halo slots keep this rank's
//! share).

void DustGasDrag::DepositCloudDensity3D(DvceArray5D<Real> &f) {
  if (f.extent_int(0) != qdep.extent_int(0) || f.extent_int(1) != 4 ||
      f.extent_int(2) != qdep.extent_int(2) || f.extent_int(3) != qdep.extent_int(3) ||
      f.extent_int(4) != qdep.extent_int(4)) {
    Kokkos::realloc(f, qdep.extent(0), 4, qdep.extent(2), qdep.extent(3), qdep.extent(4));
  }
  Kokkos::deep_copy(f, 0.0);
  UpdateShear3D(pmy_pack->pmesh->time);
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
  HaloReverseAdd(f, 0, 4);   // several ranks: the owners hold every rank's clouds
}

}  // namespace dust
