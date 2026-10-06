//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud.cpp
//! \brief Refinable Lagrangian hat clouds on a single-rank,
//! two-dimensional, static 2:1 mesh: the finest-level virtual grid that maps positions
//! to active leaves, and the exact two-scale split with merging of coincident children.
//!
//! Every particle carries a cloud level PLEV. Its hat cloud has half-width
//! root_dx/2^PLEV per direction, and its weight on an active leaf is the exact integral
//! of the cloud over that leaf (SMRCompositeTransfer::ForEachCloudWeight).  The
//! deposited fields therefore depend only on the continuous cloud density
//! sum_p m_p S_p(x - x_p), which the mesh merely integrates over its leaves.  A cloud
//! changes size only through the identity
//!   S_ell(x) = 1/4 S_{ell/2}(x + ell/2) + 1/2 S_{ell/2}(x) + 1/4 S_{ell/2}(x - ell/2),
//! applied per direction, which leaves that continuous density unchanged.  Children
//! that coincide with existing same-level particles (the shared children of adjacent
//! lattice parents) are merged by centre of mass, so a quiet-start lattice refines to
//! 2^d equal children per parent.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"

namespace dust {
namespace {

int EncodeCloudLeaf(const int m, const int j, const int i,
                    const int ncells2, const int ncells1) {
  return (m*ncells2 + j)*ncells1 + i;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildRefinableCloudTransfer
//! \brief Map every cell of the finest-level virtual grid to the active leaf that
//! contains it, and record that leaf's level.

void DustGasDrag::BuildRefinableCloudTransfer(ParameterInput *pin) {
  Mesh *pmesh = pmy_pack->pmesh;
  auto &indcs = pmesh->mb_indcs;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int nmb = pmy_pack->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;
  // <dust>/cloud_virtual_levels (diagnostic control, default 0): extra levels of the
  // virtual grid below the finest mesh level, so clouds can be finer than every gas leaf
  // (e.g. a uniform gas mesh with the fine clouds and sampling of a refined region; the
  // fine virtual cells map to their coarse leaf exactly as in the coarse part of a mixed
  // mesh)
  const int nlev_mesh = pmesh->max_level - pmesh->root_level;
  const int nvirt = pin->GetOrAddInteger("dust", "cloud_virtual_levels", 0);
  int nlev = nlev_mesh + nvirt;

  if (nlev_mesh < 0 || nvirt < 0 || nlev > 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "refinable_cloud currently requires a uniform mesh or one "
              << "or two levels of 2:1 static refinement (virtual levels included)"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &t = smr_transfer;
  t.enabled = true;
  t.cloud_mode = true;
  t.cloud_nlev = nlev;
  t.ncoarse1 = pmesh->mesh_indcs.nx1;
  t.ncoarse2 = pmesh->mesh_indcs.nx2;
  t.nfine1 = t.ncoarse1 << nlev;
  t.nfine2 = t.ncoarse2 << nlev;
  t.x1min = pmesh->mesh_size.x1min;
  t.x1max = pmesh->mesh_size.x1max;
  t.x2min = pmesh->mesh_size.x2min;
  t.x2max = pmesh->mesh_size.x2max;
  t.root_dx1 = (t.x1max - t.x1min)/static_cast<Real>(t.ncoarse1);
  t.root_dx2 = (t.x2max - t.x2min)/static_cast<Real>(t.ncoarse2);
  t.fine_dx1 = t.root_dx1/static_cast<Real>(1 << nlev);
  t.fine_dx2 = t.root_dx2/static_cast<Real>(1 << nlev);
  // controls: cloud_max_level=0 keeps every cloud at root size (translation-invariant
  // dust on the mixed gas mesh); cloud_split_everywhere refines the whole lattice
  t.cloud_max_level = pin->GetOrAddInteger("dust", "cloud_max_level", nlev);
  t.cloud_split_everywhere = pin->GetOrAddBoolean("dust", "cloud_split_everywhere", false);
  t.cloud_split_overlap = pin->GetOrAddBoolean("dust", "cloud_split_overlap", false);
  t.lobe3_integral = cloud_presplit_x1;

  t.cloud_target = DvceArray2D<int>("dust_cloud_target", t.nfine2, t.nfine1);
  t.cloud_level = DvceArray2D<int>("dust_cloud_level", t.nfine2, t.nfine1);
  auto target = Kokkos::create_mirror_view(t.cloud_target);
  auto level = Kokkos::create_mirror_view(t.cloud_level);
  Kokkos::deep_copy(target, -1);
  Kokkos::deep_copy(level, -1);

  for (int m=0; m<nmb; ++m) {
    const auto &size = mbsize.h_view(m);
    int lev = mblev.h_view(m) - pmesh->root_level;
    int ratio = 1 << (nlev - lev);
    Real off1 = (size.x1min - t.x1min)/t.fine_dx1;
    Real off2 = (size.x2min - t.x2min)/t.fine_dx2;
    int fi0 = static_cast<int>(std::lround(off1));
    int fj0 = static_cast<int>(std::lround(off2));
    if (std::abs(off1 - fi0) > 1.0e-8 || std::abs(off2 - fj0) > 1.0e-8 ||
        std::abs(size.dx1 - ratio*t.fine_dx1) > 1.0e-12*size.dx1 ||
        std::abs(size.dx2 - ratio*t.fine_dx2) > 1.0e-12*size.dx2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "MeshBlock " << m << " is not aligned with the finest "
                << "virtual grid" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int j=0; j<indcs.nx2; ++j) {
      for (int i=0; i<indcs.nx1; ++i) {
        int leaf = EncodeCloudLeaf(m, j + indcs.js, i + indcs.is, ncells2, ncells1);
        for (int sj=0; sj<ratio; ++sj) {
          for (int si=0; si<ratio; ++si) {
            int fj = fj0 + j*ratio + sj;
            int fi = fi0 + i*ratio + si;
            if (fj < 0 || fj >= t.nfine2 || fi < 0 || fi >= t.nfine1 ||
                target(fj,fi) >= 0) {
              std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                        << std::endl << "virtual fine cell outside the mesh or owned "
                        << "twice" << std::endl;
              std::exit(EXIT_FAILURE);
            }
            target(fj,fi) = leaf;
            level(fj,fi) = lev;
          }
        }
      }
    }
  }
  for (int fj=0; fj<t.nfine2; ++fj) {
    for (int fi=0; fi<t.nfine1; ++fi) {
      if (target(fj,fi) < 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "virtual fine cell has no active leaf owner"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  Kokkos::deep_copy(t.cloud_target, target);
  Kokkos::deep_copy(t.cloud_level, level);

  if (global_variable::my_rank == 0) {
    std::cout << "# refinable_cloud map: levels=" << nlev
              << (nvirt > 0 ? " (" + std::to_string(nvirt) + " virtual)" : "") << ", virtual finest grid "
              << t.nfine1 << "x" << t.nfine2 << ", split slack=" << cloud_split_slack
              << ", merge tolerance=" << cloud_merge_tolerance << " finest cells"
              << ", max cloud level=" << t.cloud_max_level
              << ", gather order=" << cloud_gather_order
              << (cloud_force_local ? ", local gas-side drag" : "")
              << (cloud_force_children ? ", drag deposited as children over finer leaves" : "")
              << (cloud_deform ? ", deformable clouds (dv/dx carried)" : "")
              << (cloud_gather_level >= 0 ? ", gather footprint fixed at level " : "")
              << (cloud_gather_level >= 0 ? std::to_string(cloud_gather_level) : "")
              << (cloud_transition > 0 ? ", P1 transition clouds (mode " +
                  std::to_string(cloud_transition) + ")" : "")
              << (cloud_p1_split ? ", moving P1 split without merging" : "")
              << (t.cloud_split_everywhere ? ", split everywhere" : "")
              << (t.cloud_split_overlap ? ", split on first overlap with finer leaves" : "")
              << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RefineClouds
//! \brief Task wrapper: split eligible clouds at the start of every cycle.

TaskStatus DustGasDrag::RefineClouds(Driver *pdrive, int stage) {
  if (smr_transfer_mode != DustSMRTransfer::refinable_cloud) return TaskStatus::complete;
  SplitAndMergeClouds();
  if (cloud_merge && pmy_pack->pmesh->ncycle % cloud_merge_interval == 0) MergeClouds();
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitAndMergeClouds
//! \brief Apply the exact two-scale split to every particle whose (slightly shrunk)
//! cloud support lies on leaves finer than its level, then merge children that
//! coincide with same-level particles.  Runs on the host: splits are rare (a lattice
//! column entering the fine region) and the mesh is single-rank by construction.

int DustGasDrag::SplitAndMergeClouds() {
  if (cloud_p1_split) return SplitCloudsP1();
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  auto transfer = smr_transfer;
  Real slack = cloud_split_slack;

  DvceArray1D<int> flag("dust_cloud_flag", npart);
  int nsplit = 0;
  Kokkos::parallel_reduce("dust_cloud_flag", Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int p, int &sum) {
    int f = transfer.CloudInsideFiner(pr(IPX,p), pr(IPY,p), pi(PLEV,p), slack) ? 1 : 0;
    flag(p) = f;
    sum += f;
  }, Kokkos::Sum<int>(nsplit));
  if (nsplit == 0) return 0;

  auto pr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
  auto pi_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pi);
  auto flag_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), flag);
  auto target_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                      transfer.cloud_target);
  if (cloud_next_tag < 0) cloud_next_tag = pmesh->nprtcl_total;

  const Real x1min = transfer.x1min, x2min = transfer.x2min;
  const Real len1 = transfer.x1max - transfer.x1min;
  const Real len2 = transfer.x2max - transfer.x2min;
  const int nfine1 = transfer.nfine1, nfine2 = transfer.nfine2;
  const Real fdx1 = transfer.fine_dx1, fdx2 = transfer.fine_dx2;
  const Real tol = cloud_merge_tolerance*std::min(fdx1, fdx2);
  const int gids = pmy_pack->gids;
  const int ncells1 = qdep.extent_int(4);
  const int ncells2 = qdep.extent_int(3);
  auto wrap1 = [&](Real x) {
    while (x < x1min) x += len1;
    while (x >= x1min + len1) x -= len1;
    return x;
  };
  auto wrap2 = [&](Real x) {
    while (x < x2min) x += len2;
    while (x >= x2min + len2) x -= len2;
    return x;
  };
  auto image1 = [&](Real d) {  // minimum-image separation
    if (d < -0.5*len1) d += len1;
    if (d >= 0.5*len1) d -= len1;
    return d;
  };
  auto image2 = [&](Real d) {
    if (d < -0.5*len2) d += len2;
    if (d >= 0.5*len2) d -= len2;
    return d;
  };
  auto bin1 = [&](Real x) {
    return SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((x - x1min)/fdx1)), nfine1);
  };
  auto bin2 = [&](Real x) {
    return SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((x - x2min)/fdx2)), nfine2);
  };
  auto owner_gid = [&](Real x, Real y) {
    int leaf = target_h(bin2(y), bin1(x));
    return gids + leaf/(ncells1*ncells2);
  };

  // host particle list: kept particles first, then children (possibly merged)
  const int nchild_max = 9*nsplit;
  const int capacity = npart - nsplit + nchild_max;
  std::vector<Real> R(static_cast<std::size_t>(nrdata)*capacity, 0.0);
  std::vector<int> I(static_cast<std::size_t>(nidata)*capacity, 0);
  auto rr = [&](int c, int s) -> Real& { return R[static_cast<std::size_t>(s)*nrdata + c]; };
  auto ii = [&](int c, int s) -> int& { return I[static_cast<std::size_t>(s)*nidata + c]; };

  // bins of same-level candidates for merging, keyed by finest virtual cell
  std::unordered_map<int, std::vector<int>> bins;
  auto bin_key = [&](Real x, Real y) { return bin2(y)*nfine1 + bin1(x); };

  int nslot = 0;
  for (int p=0; p<npart; ++p) {
    if (flag_h(p)) continue;
    for (int c=0; c<nrdata; ++c) rr(c,nslot) = pr_h(c,p);
    for (int c=0; c<nidata; ++c) ii(c,nslot) = pi_h(c,p);
    if (ii(PLEV,nslot) > 0) bins[bin_key(rr(IPX,nslot), rr(IPY,nslot))].push_back(nslot);
    ++nslot;
  }

  const Real offset[3] = {-0.5, 0.0, 0.5};
  const Real fraction[3] = {0.25, 0.5, 0.25};
  const bool deform = cloud_deform;
  std::vector<Real> child(nrdata, 0.0);
  unsigned long long merged_here = 0;
  for (int p=0; p<npart; ++p) {
    if (!flag_h(p)) continue;
    int lev = pi_h(PLEV,p);
    Real ell1 = transfer.CloudHalfWidth1(lev);
    Real ell2 = transfer.CloudHalfWidth2(lev);
    for (int b=0; b<3; ++b) {
      for (int a=0; a<3; ++a) {
        Real d1 = offset[a]*ell1, d2 = offset[b]*ell2;
        Real x = wrap1(pr_h(IPX,p) + d1);
        Real y = wrap2(pr_h(IPY,p) + d2);
        Real mass = pr_h(IPM,p)*fraction[a]*fraction[b];
        // child attributes: the parent's, except that a deformable cloud's children
        // take the parent's linear velocity field v + G.d at their own centres
        for (int c=0; c<nrdata; ++c) child[c] = pr_h(c,p);
        if (deform) {
          child[IPVX] += pr_h(IPGXX,p)*d1 + pr_h(IPGXY,p)*d2;
          child[IPVY] += pr_h(IPGYX,p)*d1 + pr_h(IPGYY,p)*d2;
          child[IPVZ] += pr_h(IPGZX,p)*d1 + pr_h(IPGZY,p)*d2;
          child[IPVX1] = child[IPVX];
          child[IPVY1] = child[IPVY];
          child[IPVZ1] = child[IPVZ];
        }
        // look for a coincident same-level, same-species particle in the 3x3 bins
        int found = -1;
        int bi = bin1(x), bj = bin2(y);
        for (int dj=-1; dj<=1 && found<0; ++dj) {
          int wj = SMRCompositeTransfer::WrapIndex(bj + dj, nfine2);
          for (int di=-1; di<=1 && found<0; ++di) {
            int wi = SMRCompositeTransfer::WrapIndex(bi + di, nfine1);
            auto it = bins.find(wj*nfine1 + wi);
            if (it == bins.end()) continue;
            for (int s : it->second) {
              if (ii(PLEV,s) != lev + 1 || ii(PSP,s) != pi_h(PSP,p)) continue;
              Real d1 = image1(x - rr(IPX,s));
              Real d2 = image2(y - rr(IPY,s));
              if (d1*d1 + d2*d2 <= tol*tol) {found = s; break;}
            }
          }
        }
        if (found >= 0) {
          // centre-of-mass merge into the existing particle; every real attribute
          // except position and mass is mass-averaged (velocities, registers, rates)
          int s = found;
          Real ms = rr(IPM,s), mt = ms + mass, wc = mass/mt;
          for (int c=0; c<nrdata; ++c) {
            if (c == IPM || c == IPX || c == IPY || c == IPX1 || c == IPY1) continue;
            rr(c,s) = rr(c,s) + wc*(child[c] - rr(c,s));
          }
          Real xs = wrap1(rr(IPX,s) + wc*image1(x - rr(IPX,s)));
          Real ys = wrap2(rr(IPY,s) + wc*image2(y - rr(IPY,s)));
          rr(IPX,s) = xs; rr(IPX1,s) = xs;
          rr(IPY,s) = ys; rr(IPY1,s) = ys;
          rr(IPM,s) = mt;
          ii(PGID,s) = owner_gid(xs, ys);
          ++merged_here;
        } else {
          int s = nslot++;
          for (int c=0; c<nrdata; ++c) rr(c,s) = child[c];
          for (int c=0; c<nidata; ++c) ii(c,s) = pi_h(c,p);
          rr(IPX,s) = x; rr(IPX1,s) = x;
          rr(IPY,s) = y; rr(IPY1,s) = y;
          rr(IPM,s) = mass;
          ii(PLEV,s) = lev + 1;
          ii(PGID,s) = owner_gid(x, y);
          ii(PTAG,s) = cloud_next_tag++;
          bins[bin_key(x, y)].push_back(s);
        }
      }
    }
  }
  cloud_split_count += nsplit;
  cloud_merge_count += merged_here;

  // write the new particle list back
  int nnew = nslot;
  Kokkos::realloc(pr, nrdata, nnew);
  Kokkos::realloc(pi, nidata, nnew);
  auto pr_new = Kokkos::create_mirror_view(pr);
  auto pi_new = Kokkos::create_mirror_view(pi);
  for (int s=0; s<nnew; ++s) {
    for (int c=0; c<nrdata; ++c) pr_new(c,s) = rr(c,s);
    for (int c=0; c<nidata; ++c) pi_new(c,s) = ii(c,s);
  }
  Kokkos::deep_copy(pr, pr_new);
  Kokkos::deep_copy(pi, pi_new);
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;
  return nsplit;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MakeTransitionClouds
//! \brief P1 transition clouds.  After the split, the particle basis
//! reproduces constants exactly but not linear fields: at every split edge the child
//! that has a same-level neighbour on one side only (its partner would have been the
//! child of the unsplit neighbouring parent) carries an uncancelled first moment
//! m H/8.  Each such edge child is merged into that neighbour, which keeps its own
//! position and takes the lobed shape (2/3) S_ell(x - x_p) + (1/3) S_ell(x - x_p -/+ ell)
//! along that direction: the deposited density is unchanged, and the union of coarse
//! hats, transition hats and fine hats is the P1 (FEM) basis of the graded particle
//! lattice.  Applied once to the initial lattice; clouds split later are not processed.

int DustGasDrag::MakeTransitionClouds() {
  if (cloud_transition <= 0 || cloud_p1_split) return 0;
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  auto transfer = smr_transfer;
  auto pr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
  auto pi_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pi);

  const Real x1min = transfer.x1min, x2min = transfer.x2min;
  const Real len1 = transfer.x1max - transfer.x1min;
  const Real len2 = transfer.x2max - transfer.x2min;
  const int nfine1 = transfer.nfine1, nfine2 = transfer.nfine2;
  const Real fdx1 = transfer.fine_dx1, fdx2 = transfer.fine_dx2;
  const Real tol = cloud_merge_tolerance*std::min(fdx1, fdx2);
  auto wrap = [](Real x, Real xmin, Real len) {
    while (x < xmin) x += len;
    while (x >= xmin + len) x -= len;
    return x;
  };
  auto image = [](Real d, Real len) {
    if (d < -0.5*len) d += len;
    if (d >= 0.5*len) d -= len;
    return d;
  };
  auto key = [&](Real x, Real y) {
    int i = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((x - x1min)/fdx1)), nfine1);
    int j = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((y - x2min)/fdx2)), nfine2);
    return std::make_pair(i, j);
  };
  std::unordered_map<int, std::vector<int>> bins;
  for (int p=0; p<npart; ++p) {
    if (pi_h(PLEV,p) == 0) continue;
    auto k = key(pr_h(IPX,p), pr_h(IPY,p));
    bins[k.second*nfine1 + k.first].push_back(p);
  }
  std::vector<char> removed(npart, 0);
  auto find_at = [&](Real x, Real y, int lev, int sp) {
    auto k = key(x, y);
    for (int dj=-1; dj<=1; ++dj) {
      int wj = SMRCompositeTransfer::WrapIndex(k.second + dj, nfine2);
      for (int di=-1; di<=1; ++di) {
        int wi = SMRCompositeTransfer::WrapIndex(k.first + di, nfine1);
        auto it = bins.find(wj*nfine1 + wi);
        if (it == bins.end()) continue;
        for (int s : it->second) {
          if (removed[s] || SMRCompositeTransfer::CloudLevel(pi_h(PLEV,s)) != lev ||
              pi_h(PSP,s) != sp) continue;
          Real d1 = image(x - pr_h(IPX,s), len1), d2 = image(y - pr_h(IPY,s), len2);
          if (d1*d1 + d2*d2 <= tol*tol) return s;
        }
      }
    }
    return -1;
  };

  int ntrans = 0;
  for (int dir=2; dir>=1; --dir) {
    const int shift = (dir == 2) ? SMRCompositeTransfer::kCloudLobeShift2
                                 : SMRCompositeTransfer::kCloudLobeShift1;
    for (int s=0; s<npart; ++s) {
      int code = pi_h(PLEV,s);
      int lev = SMRCompositeTransfer::CloudLevel(code);
      if (removed[s] || lev == 0 || code != lev) continue;   // plain refined clouds only
      Real x = pr_h(IPX,s), y = pr_h(IPY,s);
      int sp = pi_h(PSP,s);
      int lo, hi;
      if (dir == 2) {
        Real ell = transfer.CloudHalfWidth2(lev);
        lo = find_at(x, wrap(y - ell, x2min, len2), lev, sp);
        hi = find_at(x, wrap(y + ell, x2min, len2), lev, sp);
      } else {
        Real ell = transfer.CloudHalfWidth1(lev);
        lo = find_at(wrap(x - ell, x1min, len1), y, lev, sp);
        hi = find_at(wrap(x + ell, x1min, len1), y, lev, sp);
      }
      if ((lo >= 0) == (hi >= 0)) continue;
      int c = (lo >= 0) ? lo : hi;
      if (pi_h(PLEV,c) != lev) continue;          // neighbour is already a transition
      // merge s into c: c keeps its position, other attributes are mass-averaged
      Real mt = pr_h(IPM,c) + pr_h(IPM,s), wc = pr_h(IPM,s)/mt;
      for (int col=0; col<nrdata; ++col) {
        if (col == IPM || col == IPX || col == IPY || col == IPX1 || col == IPY1) continue;
        pr_h(col,c) = pr_h(col,c) + wc*(pr_h(col,s) - pr_h(col,c));
      }
      pr_h(IPM,c) = mt;
      pi_h(PLEV,c) = lev | (((lo >= 0) ? 2 : 1) << shift);
      removed[s] = 1;
      ++ntrans;
    }
  }
  if (ntrans == 0) return 0;

  int nnew = 0;
  for (int p=0; p<npart; ++p) if (!removed[p]) ++nnew;
  Kokkos::realloc(pr, nrdata, nnew);
  Kokkos::realloc(pi, nidata, nnew);
  auto pr_new = Kokkos::create_mirror_view(pr);
  auto pi_new = Kokkos::create_mirror_view(pi);
  int s = 0;
  for (int p=0; p<npart; ++p) {
    if (removed[p]) continue;
    for (int c=0; c<nrdata; ++c) pr_new(c,s) = pr_h(c,p);
    for (int c=0; c<nidata; ++c) pi_new(c,s) = pi_h(c,p);
    ++s;
  }
  Kokkos::deep_copy(pr, pr_new);
  Kokkos::deep_copy(pi, pi_new);
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;
  return ntrans;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitCloudsP1
//! \brief Moving-particle P1 split (<dust>/cloud_p1_split), used in place
//! of SplitAndMergeClouds.  A plain level-l cloud whose support lies on finer leaves is
//! split, but along each direction an edge child is separated only if the lattice
//! neighbour on that side (the virtual level-l cloud one parent spacing H away) would
//! be split too; otherwise the edge child stays attached to the centre child as a
//! transition lobe.  A lobed cloud separates a lobe once that neighbour becomes
//! eligible.  Along a direction, the transition band of eligible parents with an
//! ineligible neighbour is exactly H wide, so every lattice row holds one transition
//! cloud whatever its phase and the particle basis stays P1 while the lattice drifts
//! through the split surface.  Nothing is ever merged: every particle keeps its own
//! attributes, so independent streams and stopping times are preserved.

int DustGasDrag::SplitCloudsP1() {
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  auto transfer = smr_transfer;
  Real slack = cloud_split_slack;
  const bool release = cloud_p1_release_exit;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
  const bool presplit = cloud_presplit_x1;

  // eligible(x, y, l): a plain level-l cloud at (x, y) would be split
  DvceArray1D<int> flag("dust_cloud_flag", npart);
  int nflag = 0;
  Kokkos::parallel_reduce("dust_cloud_p1_flag",
  Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int p, int &sum) {
    const int code = pi(PLEV,p);
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    const int l1 = SMRCompositeTransfer::CloudLobe1(code);
    const int l2 = SMRCompositeTransfer::CloudLobe2(code);
    const Real x = pr(IPX,p), y = pr(IPY,p);
    int f = 0;
    if (l1 == 0 && l2 == 0) {
      f = transfer.CloudInsideFiner(x, y, lev, slack) ? 1 : 0;
      if (presplit && lev == 0 && lev < transfer.cloud_nlev) f = 1;
    } else if (lev > 0 && release && !transfer.CloudInsideFiner(x, y, lev - 1, slack)) {
      f = 1;   // diagnostic: a lobed cloud that has left the split region releases its lobes
    } else if (lev > 0 && transfer.CloudInsideFiner(x, y, lev - 1, slack)) {
      const Real h1 = transfer.CloudHalfWidth1(lev - 1);
      const Real h2 = transfer.CloudHalfWidth2(lev - 1);
      if (((l1 & 1) && transfer.CloudInsideFiner(x - h1, y, lev - 1, slack)) ||
          ((l1 & 2) && transfer.CloudInsideFiner(x + h1, y, lev - 1, slack)) ||
          ((l2 & 1) && transfer.CloudInsideFiner(x, y - h2, lev - 1, slack)) ||
          ((l2 & 2) && transfer.CloudInsideFiner(x, y + h2, lev - 1, slack))) f = 1;
    }
    flag(p) = f;
    sum += f;
  }, Kokkos::Sum<int>(nflag));
  if (nflag == 0) return 0;

  auto pr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
  auto pi_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pi);
  auto flag_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), flag);
  auto target_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                      transfer.cloud_target);
  // host copy of the transfer for the eligibility tests (views point to host mirrors)
  auto level_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_level);
  if (cloud_next_tag < 0) cloud_next_tag = pmesh->nprtcl_total;
  // diagnostic split log (<dust>/cloud_split_log): one S line per event (parent state
  // and the largest leaf change of deposited density and momentum density), C lines for
  // its pieces, W lines (leaf centre, volume, mass before/after) for the first events,
  // and one G line per call with the global mass and momentum before and after
  const bool logsplit = cloud_split_log;
  constexpr unsigned long long kLogWeightEvents = 200;
  std::FILE *fp = nullptr;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &indcs = pmesh->mb_indcs;
  if (logsplit) {
    fp = std::fopen("cloud_split_events.txt", "a");
    if (fp == nullptr) {
      std::cout << "### FATAL ERROR: cannot open cloud_split_events.txt" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (cloud_split_event == 0) {
      std::fprintf(fp, "# S ev ncycle time lobe_release ptag pcode px py pm pvx pvy pvz"
                   " npieces max|drho| max(rho_parent) max|djx| max|djy| max|djz| nleaf"
                   " fam_dm fam_dpx fam_dpy fam_dpz\n# C ev k tag code x y m\n"
                   "# W ev leaf_x1 leaf_x2 leaf_vol m_before m_after\n"
                   "# O ev k shift_x shift_y (split_oracle; C lines then give pre-shift positions)\n"
                   "# P ncycle time nevent nleaf max|drho| rms|drho| max|dFx| max|dFy|"
                   " max|dFz| rms|dF| net_dM (whole call: deposit of all flagged clouds before"
                   " vs all pieces after, leaf densities; dF = (dj - u_gas drho)/t_s, the change"
                   " of the drag source density on the gas)\n"
                   "# G ncycle time nflag npart_before npart_after M_before M_after"
                   " PX_b PX_a PY_b PY_a PZ_b PZ_a\n");
      if (cloud_split_log_leaves) {
        std::fprintf(fp, "# V ncycle time nleaf vol_touched vol_domain"
                     " sqrt(sum V drho^2/vol_domain) sqrt(sum V |dm|^2/vol_domain)"
                     " (cloud_split_log_leaves; dm = t_s dF = dj - u_gas drho, the momentum"
                     " the changed deposit holds relative to the gas)\n"
                     "# D ncycle leaf_x1 leaf_x2 leaf_vol drho dm_x dm_y dm_z (one per"
                     " leaf touched by the call, before the P line)\n");
      }
    }
  }

  const Real x1min = transfer.x1min, x2min = transfer.x2min;
  const Real len1 = transfer.x1max - transfer.x1min;
  const Real len2 = transfer.x2max - transfer.x2min;
  const int nfine1 = transfer.nfine1, nfine2 = transfer.nfine2;
  const Real fdx1 = transfer.fine_dx1, fdx2 = transfer.fine_dx2;
  const int gids = pmy_pack->gids;
  const int ncells1 = qdep.extent_int(4);
  const int ncells2 = qdep.extent_int(3);
  auto wrap = [](Real x, Real xmin, Real len) {
    while (x < xmin) x += len;
    while (x >= xmin + len) x -= len;
    return x;
  };
  auto owner_gid = [&](Real x, Real y) {
    int fi = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((x - x1min)/fdx1)), nfine1);
    int fj = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((y - x2min)/fdx2)), nfine2);
    return gids + target_h(fj, fi)/(ncells1*ncells2);
  };
  // host version of SMRCompositeTransfer::CloudInsideFiner for a plain level-l cloud
  auto eligible = [&](Real x, Real y, int lev) {
    if (lev >= transfer.cloud_nlev || lev >= transfer.cloud_max_level) return false;
    if (transfer.cloud_split_everywhere) return true;
    x = wrap(x, x1min, len1);
    y = wrap(y, x2min, len2);
    Real r1 = (1.0 - slack)*transfer.CloudHalfWidth1(lev);
    Real r2 = (1.0 - slack)*transfer.CloudHalfWidth2(lev);
    int ia = static_cast<int>(std::floor((x - r1 - x1min)/fdx1));
    int ib = static_cast<int>(std::ceil((x + r1 - x1min)/fdx1)) - 1;
    int ja = static_cast<int>(std::floor((y - r2 - x2min)/fdx2));
    int jb = static_cast<int>(std::ceil((y + r2 - x2min)/fdx2)) - 1;
    if (transfer.cloud_split_overlap) {
      for (int fj=ja; fj<=jb; ++fj) {
        int wj = SMRCompositeTransfer::WrapIndex(fj, nfine2);
        for (int fi=ia; fi<=ib; ++fi) {
          if (level_h(wj, SMRCompositeTransfer::WrapIndex(fi, nfine1)) > lev) return true;
        }
      }
      return false;
    }
    for (int fj=ja; fj<=jb; ++fj) {
      int wj = SMRCompositeTransfer::WrapIndex(fj, nfine2);
      for (int fi=ia; fi<=ib; ++fi) {
        if (level_h(wj, SMRCompositeTransfer::WrapIndex(fi, nfine1)) <= lev) return false;
      }
    }
    return true;
  };

  // pieces of a cloud along one direction: offset (in child half-widths), fraction of
  // the cloud's mass, lobe code.  lobe_old is the cloud's lobe set along the direction
  // (3 for a plain parent, whose shape is the centre child with both edge children
  // attached), sep_lo/sep_hi say which attached edge children are separated now.
  struct Piece {Real off; Real frac; int lobe;};
  auto pieces = [](int lobe_old, bool sep_lo, bool sep_hi, Piece out[3]) {
    int n = 0;
    Real total = 0.5 + 0.25*((lobe_old & 1) ? 1 : 0) + 0.25*((lobe_old & 2) ? 1 : 0);
    bool lo = (lobe_old & 1) && sep_lo, hi = (lobe_old & 2) && sep_hi;
    int lobe_new = lobe_old & ~((lo ? 1 : 0) | (hi ? 2 : 0));
    Real keep = 0.5 + 0.25*((lobe_new & 1) ? 1 : 0) + 0.25*((lobe_new & 2) ? 1 : 0);
    out[n++] = {0.0, keep/total, lobe_new};
    if (lo) out[n++] = {-1.0, 0.25/total, 0};
    if (hi) out[n++] = {1.0, 0.25/total, 0};
    return n;
  };

  const int capacity = npart - nflag + 9*nflag;
  std::vector<Real> R(static_cast<std::size_t>(nrdata)*capacity, 0.0);
  std::vector<int> I(static_cast<std::size_t>(nidata)*capacity, 0);
  auto rr = [&](int c, int s) -> Real& { return R[static_cast<std::size_t>(s)*nrdata + c]; };
  auto ii = [&](int c, int s) -> int& { return I[static_cast<std::size_t>(s)*nidata + c]; };
  int nslot = 0;
  for (int p=0; p<npart; ++p) {
    if (flag_h(p)) continue;
    for (int c=0; c<nrdata; ++c) rr(c,nslot) = pr_h(c,p);
    for (int c=0; c<nidata; ++c) ii(c,nslot) = pi_h(c,p);
    ++nslot;
  }

  // <dust>/cloud_split_interp: node lookup on the particle lattice.  All particles
  // derive from the root lattice by two-scale splits and drift with a common background
  // velocity, so their nodes are multiples of the finest half-width from any particle
  // (the anchor); displacements are ~1e-5 of a cell, so rounding is unambiguous.
  const bool interp = cloud_split_interp;
  if (interp && split_oracle) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/cloud_split_interp excludes <problem>/split_oracle" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const Real ef1 = transfer.CloudHalfWidth1(transfer.cloud_nlev);
  const Real ef2 = transfer.CloudHalfWidth2(transfer.cloud_nlev);
  const long long nn1 = std::llround(len1/ef1), nn2 = std::llround(len2/ef2);
  const Real ax = pr_h(IPX,0), ay = pr_h(IPY,0);
  auto node_key = [&](Real px, Real py) {
    long long i = std::llround(wrap(px - ax, 0.0, len1)/ef1) % nn1;
    long long j = std::llround(wrap(py - ay, 0.0, len2)/ef2) % nn2;
    return j*nn1 + i;
  };
  std::unordered_map<long long, int> node_of;
  if (interp) {
    for (int p=0; p<npart; ++p) node_of[node_key(pr_h(IPX,p), pr_h(IPY,p))] = p;
  }
  unsigned long long interp_missing = 0;
  // call-level leaf deposit of all flagged clouds (side 0) and all their pieces (side 1)
  std::unordered_map<int, std::array<Real,8>> pass_acc;
  int nevent_call = 0;

  unsigned long long nparent = 0, nlobe = 0;
  for (int p=0; p<npart; ++p) {
    if (!flag_h(p)) continue;
    ++nevent_call;
    const int code = pi_h(PLEV,p);
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    const Real x = pr_h(IPX,p), y = pr_h(IPY,p);
    Piece p1[3], p2[3];
    int n1, n2, clev;
    if (SMRCompositeTransfer::CloudLobe1(code) == 0 &&
        SMRCompositeTransfer::CloudLobe2(code) == 0) {
      // plain parent: children at level lev+1, edges separated where the neighbour
      // parent is eligible as well
      const Real h1 = transfer.CloudHalfWidth1(lev), h2 = transfer.CloudHalfWidth2(lev);
      const bool px = presplit && lev == 0;
      n1 = pieces(3, px || eligible(x - h1, y, lev), px || eligible(x + h1, y, lev), p1);
      n2 = pieces(3, eligible(x, y - h2, lev), eligible(x, y + h2, lev), p2);
      clev = lev + 1;
      ++nparent;
    } else {
      // lobed cloud of level lev (parent level lev-1): separate the lobes whose
      // neighbour parent has become eligible
      const Real h1 = transfer.CloudHalfWidth1(lev - 1);
      const Real h2 = transfer.CloudHalfWidth2(lev - 1);
      const bool out = release && !eligible(x, y, lev - 1);
      n1 = pieces(SMRCompositeTransfer::CloudLobe1(code), out || eligible(x - h1, y, lev - 1),
                  out || eligible(x + h1, y, lev - 1), p1);
      n2 = pieces(SMRCompositeTransfer::CloudLobe2(code), out || eligible(x, y - h2, lev - 1),
                  out || eligible(x, y + h2, lev - 1), p2);
      clev = lev;
      ++nlobe;
    }
    const Real e1 = transfer.CloudHalfWidth1(clev), e2 = transfer.CloudHalfWidth2(clev);
    const int slot0 = nslot;
    for (int b=0; b<n2; ++b) {
      for (int a=0; a<n1; ++a) {
        int s = nslot++;
        Real cx = wrap(x + p1[a].off*e1, x1min, len1);
        Real cy = wrap(y + p2[b].off*e2, x2min, len2);
        for (int c=0; c<nrdata; ++c) rr(c,s) = pr_h(c,p);
        for (int c=0; c<nidata; ++c) ii(c,s) = pi_h(c,p);
        rr(IPX,s) = cx; rr(IPX1,s) = cx;
        rr(IPY,s) = cy; rr(IPY1,s) = cy;
        rr(IPM,s) = pr_h(IPM,p)*p1[a].frac*p2[b].frac;
        ii(PLEV,s) = clev | (p1[a].lobe << sh1) | (p2[b].lobe << sh2);
        ii(PGID,s) = owner_gid(cx, cy);
        if (a > 0 || b > 0) ii(PTAG,s) = cloud_next_tag++;   // the centre keeps the tag
      }
    }
    // diagnostic position oracle (<problem>/split_oracle): give the pieces of an in-run
    // event the prescribed family-relative displacement, evaluated at their mass
    // centroids (lobe: -/+ ell/3), minus its mass-weighted mean, so that the event's
    // mass centroid is unchanged; velocities are not touched.  pre_x/pre_y keep the
    // pre-shift positions for the split log (lineage needs the lattice offsets).
    std::vector<Real> pre_x, pre_y, shift_x, shift_y;
    if (interp) {
      // node-convention (P1) prolongation: along direction d a piece at offset s e_d
      // (s = -1, +1) takes the quadratic interpolant of displacement and velocity through
      // the nodes at -2 s e_d, 0 and +2 s e_d (this cloud's node and the neighbouring ones:
      // a neighbouring parent, its centre piece, or a virtual node inside a coarse cloud,
      // whose P1 value is the mean of the two (or four) nodes next to it), relative to this
      // cloud: (3 (f_+ - f_0) - (f_- - f_0))/8, or (f_+ - f_0)/2 without the far node.  Corner
      // pieces add both directions.  Displacements are taken on the lattice reference
      // (xi_q = x_q - its node).  Positions stay relative to the centre piece; velocity
      // increments have their mass-weighted mean removed.
      const int np = nslot - slot0;
      pre_x.resize(np); pre_y.resize(np); shift_x.assign(np, 0.0); shift_y.assign(np, 0.0);
      const Real ed[2] = {transfer.CloudHalfWidth1(clev), transfer.CloudHalfWidth2(clev)};
      // state (displacement relative to the node, velocity) at node (tx, ty)
      auto node_state = [&](Real tx, Real ty, Real f[5]) {
        auto exact = [&](Real px, Real py, Real g[5]) {
          auto it = node_of.find(node_key(px, py));
          if (it == node_of.end()) return false;
          const int q = it->second;
          const Real ddx = transfer.MinImage1(pr_h(IPX,q) - px);
          const Real ddy = transfer.MinImage2(pr_h(IPY,q) - py);
          if (std::fabs(ddx) > 0.25*ef1 || std::fabs(ddy) > 0.25*ef2) return false;
          g[0] = ddx; g[1] = ddy;
          g[2] = pr_h(IPVX,q); g[3] = pr_h(IPVY,q); g[4] = pr_h(IPVZ,q);
          return true;
        };
        if (exact(tx, ty, f)) return true;
        Real g[4][5];
        const Real o1[4] = {-ed[0], ed[0], 0.0, 0.0}, o2[4] = {0.0, 0.0, -ed[1], ed[1]};
        for (int pair=0; pair<2; ++pair) {   // midpoint of two nodes along x, then along y
          const int i0 = 2*pair, i1 = 2*pair + 1;
          if (exact(tx + o1[i0], ty + o2[i0], g[0]) && exact(tx + o1[i1], ty + o2[i1], g[1])) {
            for (int c=0; c<5; ++c) f[c] = 0.5*(g[0][c] + g[1][c]);
            return true;
          }
        }
        int n = 0;
        for (int a2=-1; a2<=1; a2+=2) {
          for (int a1=-1; a1<=1; a1+=2) {
            if (!exact(tx + a1*ed[0], ty + a2*ed[1], g[n])) return false;
            ++n;
          }
        }
        for (int c=0; c<5; ++c) f[c] = 0.25*(g[0][c] + g[1][c] + g[2][c] + g[3][c]);
        return true;
      };
      Real f0[5] = {0.0, 0.0, pr_h(IPVX,p), pr_h(IPVY,p), pr_h(IPVZ,p)};
      Real inc[2][3][5];   // [direction][s+1][dx, dy, dvx, dvy, dvz]
      bool ok[2][3] = {{false, false, false}, {false, false, false}};
      for (int d=0; d<2; ++d) {
        Real fs[2][5];
        bool have[2];
        for (int k=0; k<2; ++k) {
          const int s = 2*k - 1;
          have[k] = node_state(x + ((d == 0) ? 2.0*s*ed[0] : 0.0),
                               y + ((d == 1) ? 2.0*s*ed[1] : 0.0), fs[k]);
        }
        for (int k=0; k<2; ++k) {
          if (!have[k]) continue;
          const int s = 2*k - 1;
          ok[d][s+1] = true;
          for (int c=0; c<5; ++c) {
            const Real dp = fs[k][c] - f0[c];
            inc[d][s+1][c] = have[1-k] ? (3.0*dp - (fs[1-k][c] - f0[c]))/8.0 : 0.5*dp;
          }
        }
      }
      std::vector<std::array<Real,3>> dv(np);
      Real mt = 0.0, mv[3] = {0.0, 0.0, 0.0};
      int q = 0;
      for (int b=0; b<n2; ++b) {
        for (int a=0; a<n1; ++a, ++q) {
          const int s1 = static_cast<int>(std::lround(p1[a].off));
          const int s2 = static_cast<int>(std::lround(p2[b].off));
          dv[q] = {0.0, 0.0, 0.0};
          const int sd[2] = {s1, s2};
          for (int d=0; d<2; ++d) {
            if (sd[d] == 0) continue;
            if (!ok[d][sd[d]+1]) {++interp_missing; continue;}
            shift_x[q] += inc[d][sd[d]+1][0];
            shift_y[q] += inc[d][sd[d]+1][1];
            for (int c=0; c<3; ++c) dv[q][c] += inc[d][sd[d]+1][2+c];
          }
          const Real mq = rr(IPM, slot0 + q);
          mt += mq;
          for (int c=0; c<3; ++c) mv[c] += mq*dv[q][c];
        }
      }
      for (int k=0; k<np; ++k) {
        const int s = slot0 + k;
        rr(IPVX,s) += dv[k][0] - mv[0]/mt; rr(IPVX1,s) = rr(IPVX,s);
        rr(IPVY,s) += dv[k][1] - mv[1]/mt; rr(IPVY1,s) = rr(IPVY,s);
        rr(IPVZ,s) += dv[k][2] - mv[2]/mt; rr(IPVZ1,s) = rr(IPVZ,s);
        pre_x[k] = rr(IPX,s); pre_y[k] = rr(IPY,s);
        Real nx = wrap(rr(IPX,s) + shift_x[k], x1min, len1);
        Real ny = wrap(rr(IPY,s) + shift_y[k], x2min, len2);
        rr(IPX,s) = nx; rr(IPX1,s) = nx;
        rr(IPY,s) = ny; rr(IPY1,s) = ny;
        ii(PGID,s) = owner_gid(nx, ny);
      }
    }
    if (split_oracle && pmesh->time > 0.0) {
      const int np = nslot - slot0;
      pre_x.resize(np); pre_y.resize(np); shift_x.resize(np); shift_y.resize(np);
      Real mt = 0.0, mx = 0.0, my = 0.0;
      for (int q=0; q<np; ++q) {
        const int s = slot0 + q, pc = ii(PLEV,s);
        const Real e1c = transfer.CloudHalfWidth1(pc), e2c = transfer.CloudHalfWidth2(pc);
        const int l1 = SMRCompositeTransfer::CloudLobe1(pc);
        const int l2 = SMRCompositeTransfer::CloudLobe2(pc);
        const Real cn = split_oracle_node ? 0.0 : 1.0;   // node convention: no centroid shift
        const Real cx = rr(IPX,s) + cn*((l1 == 1) ? -e1c/3.0 : ((l1 == 2) ? e1c/3.0 : 0.0));
        const Real cy = rr(IPY,s) + cn*((l2 == 1) ? -e2c/3.0 : ((l2 == 2) ? e2c/3.0 : 0.0));
        split_oracle(pmesh->time, cx, cy, shift_x[q], shift_y[q]);
        mt += rr(IPM,s); mx += rr(IPM,s)*shift_x[q]; my += rr(IPM,s)*shift_y[q];
      }
      if (split_oracle_velocity) {
        // family-relative velocity at the same centroids, mass-weighted mean removed
        std::vector<Real> dv1(np), dv2(np), dv3(np);
        Real m1 = 0.0, m2 = 0.0, m3 = 0.0;
        for (int q=0; q<np; ++q) {
          const int s = slot0 + q, pc = ii(PLEV,s);
          const Real e1c = transfer.CloudHalfWidth1(pc), e2c = transfer.CloudHalfWidth2(pc);
          const int l1 = SMRCompositeTransfer::CloudLobe1(pc);
          const int l2 = SMRCompositeTransfer::CloudLobe2(pc);
          const Real cn = split_oracle_node ? 0.0 : 1.0;
          const Real cx = rr(IPX,s) + cn*((l1 == 1) ? -e1c/3.0 : ((l1 == 2) ? e1c/3.0 : 0.0));
          const Real cy = rr(IPY,s) + cn*((l2 == 1) ? -e2c/3.0 : ((l2 == 2) ? e2c/3.0 : 0.0));
          split_oracle_velocity(pmesh->time, cx, cy, dv1[q], dv2[q], dv3[q]);
          m1 += rr(IPM,s)*dv1[q]; m2 += rr(IPM,s)*dv2[q]; m3 += rr(IPM,s)*dv3[q];
        }
        if (split_oracle_node && split_oracle_node_meanfree < 1) {
          // relative to the centre piece (slot0, at the parent's node)
          m1 = mt*dv1[0]; m2 = mt*dv2[0]; m3 = mt*dv3[0];
        }
        for (int q=0; q<np; ++q) {
          const int s = slot0 + q;
          rr(IPVX,s) += dv1[q] - m1/mt; rr(IPVX1,s) = rr(IPVX,s);
          rr(IPVY,s) += dv2[q] - m2/mt; rr(IPVY1,s) = rr(IPVY,s);
          rr(IPVZ,s) += dv3[q] - m3/mt; rr(IPVZ1,s) = rr(IPVZ,s);
        }
      }
      if (split_oracle_node && split_oracle_node_meanfree < 2) {
        mx = mt*shift_x[0]; my = mt*shift_y[0];
      }
      for (int q=0; q<np; ++q) {
        const int s = slot0 + q;
        pre_x[q] = rr(IPX,s); pre_y[q] = rr(IPY,s);
        shift_x[q] -= mx/mt; shift_y[q] -= my/mt;
        Real nx = wrap(rr(IPX,s) + shift_x[q], x1min, len1);
        Real ny = wrap(rr(IPY,s) + shift_y[q], x2min, len2);
        rr(IPX,s) = nx; rr(IPX1,s) = nx;
        rr(IPY,s) = ny; rr(IPY1,s) = ny;
        ii(PGID,s) = owner_gid(nx, ny);
      }
    }
    if (logsplit) {
      // audit of this event: leaf deposit of mass and momentum by the cloud before and
      // by its pieces after the split, at the same instant, with the deposit's weights
      // and active-leaf map (side 0 = before, 1 = after)
      std::unordered_map<int, std::array<Real,8>> acc;
      auto add = [&](Real px, Real py, int pcode, Real m, Real vx, Real vy, Real vz,
                     int side) {
        int i0, j0;
        Real w1[SMRCompositeTransfer::kMaxCloudStencil];
        Real w2[SMRCompositeTransfer::kMaxCloudStencil];
        int m1 = SMRCompositeTransfer::CloudWeights1D(wrap(px, x1min, len1),
                     transfer.CloudHalfWidth1(pcode), x1min, fdx1, i0, w1,
                     SMRCompositeTransfer::CloudLobe1(pcode));
        int m2 = SMRCompositeTransfer::CloudWeights1D(wrap(py, x2min, len2),
                     transfer.CloudHalfWidth2(pcode), x2min, fdx2, j0, w2,
                     SMRCompositeTransfer::CloudLobe2(pcode));
        for (int b=0; b<m2; ++b) {
          int fj = SMRCompositeTransfer::WrapIndex(j0 + b, nfine2);
          for (int a=0; a<m1; ++a) {
            Real w = w2[b]*w1[a];
            if (w == 0.0) continue;
            auto &q = acc[target_h(fj, SMRCompositeTransfer::WrapIndex(i0 + a, nfine1))];
            q[4*side] += m*w;
            q[4*side+1] += m*w*vx;
            q[4*side+2] += m*w*vy;
            q[4*side+3] += m*w*vz;
            auto &qq = pass_acc[target_h(fj, SMRCompositeTransfer::WrapIndex(i0 + a, nfine1))];
            qq[4*side] += m*w;
            qq[4*side+1] += m*w*vx;
            qq[4*side+2] += m*w*vy;
            qq[4*side+3] += m*w*vz;
          }
        }
      };
      const unsigned long long ev = cloud_split_event++;
      add(x, y, code, pr_h(IPM,p), pr_h(IPVX,p), pr_h(IPVY,p), pr_h(IPVZ,p), 0);
      Real fam[4] = {-pr_h(IPM,p), -pr_h(IPM,p)*pr_h(IPVX,p), -pr_h(IPM,p)*pr_h(IPVY,p),
                     -pr_h(IPM,p)*pr_h(IPVZ,p)};
      for (int s=slot0; s<nslot; ++s) {
        add(rr(IPX,s), rr(IPY,s), ii(PLEV,s), rr(IPM,s), rr(IPVX,s), rr(IPVY,s),
            rr(IPVZ,s), 1);
        fam[0] += rr(IPM,s);
        fam[1] += rr(IPM,s)*rr(IPVX,s);
        fam[2] += rr(IPM,s)*rr(IPVY,s);
        fam[3] += rr(IPM,s)*rr(IPVZ,s);
      }
      Real dmax[4] = {0.0, 0.0, 0.0, 0.0}, mmax = 0.0;
      for (auto &kv : acc) {
        int tm = kv.first/(ncells1*ncells2);
        Real vol = mbsize.h_view(tm).dx1*mbsize.h_view(tm).dx2;
        for (int c=0; c<4; ++c) {
          dmax[c] = std::max(dmax[c], std::fabs(kv.second[4+c] - kv.second[c])/vol);
        }
        mmax = std::max(mmax, kv.second[0]/vol);
        if (ev < kLogWeightEvents) {
          int ti = kv.first % ncells1, tj = (kv.first/ncells1) % ncells2;
          std::fprintf(fp, "W %llu %.17g %.17g %.17g %.17g %.17g\n", ev,
                       mbsize.h_view(tm).x1min + (ti - indcs.is + 0.5)*mbsize.h_view(tm).dx1,
                       mbsize.h_view(tm).x2min + (tj - indcs.js + 0.5)*mbsize.h_view(tm).dx2,
                       vol, kv.second[0], kv.second[4]);
        }
      }
      std::fprintf(fp, "S %llu %d %.17g %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g %d"
                   " %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n", ev,
                   pmesh->ncycle, pmesh->time, (clev == lev) ? 1 : 0, pi_h(PTAG,p), code,
                   x, y, pr_h(IPM,p), pr_h(IPVX,p), pr_h(IPVY,p), pr_h(IPVZ,p),
                   nslot - slot0, dmax[0], mmax, dmax[1], dmax[2], dmax[3],
                   static_cast<double>(acc.size()), fam[0], fam[1], fam[2], fam[3]);
      for (int s=slot0; s<nslot; ++s) {
        const int q = s - slot0;
        const bool shifted = !pre_x.empty();
        std::fprintf(fp, "C %llu %d %d %d %.17g %.17g %.17g\n", ev, q, ii(PTAG,s),
                     ii(PLEV,s), shifted ? pre_x[q] : rr(IPX,s),
                     shifted ? pre_y[q] : rr(IPY,s), rr(IPM,s));
        if (shifted) {
          std::fprintf(fp, "O %llu %d %.17g %.17g\n", ev, q, shift_x[q], shift_y[q]);
        }
      }
    }
  }
  if (logsplit) {
    // whole call: net leaf change of the deposited dust density and of the drag source
    // density on the gas, dF = (dj - u drho)/t_s (single species), u the gas velocity of
    // the leaf
    {
      auto u0_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pmy_pack->phydro->u0);
      const Real ts0 = taus.h_view(0);
      const int ks = indcs.ks;
      Real drmax = 0.0, dr2 = 0.0, dfmax[3] = {0.0, 0.0, 0.0}, df2 = 0.0, dmnet = 0.0;
      // volume-weighted sums for the V line (cloud_split_log_leaves)
      Real vtouch = 0.0, vdr2 = 0.0, vdm2 = 0.0;
      for (auto &kv : pass_acc) {
        const int tm = kv.first/(ncells1*ncells2);
        const int ti = kv.first % ncells1, tj = (kv.first/ncells1) % ncells2;
        const Real vol = mbsize.h_view(tm).dx1*mbsize.h_view(tm).dx2;
        const Real drho = (kv.second[4] - kv.second[0])/vol;
        const Real rg = u0_h(tm,IDN,ks,tj,ti);
        const Real ug[3] = {u0_h(tm,IM1,ks,tj,ti)/rg, u0_h(tm,IM2,ks,tj,ti)/rg,
                            u0_h(tm,IM3,ks,tj,ti)/rg};
        drmax = std::max(drmax, std::fabs(drho)); dr2 += drho*drho;
        dmnet += kv.second[4] - kv.second[0];
        Real dm[3];
        for (int c=0; c<3; ++c) {
          dm[c] = (kv.second[5+c] - kv.second[1+c])/vol - ug[c]*drho;
          const Real df = dm[c]/ts0;
          dfmax[c] = std::max(dfmax[c], std::fabs(df)); df2 += df*df;
        }
        if (cloud_split_log_leaves) {
          vtouch += vol; vdr2 += vol*drho*drho;
          vdm2 += vol*(dm[0]*dm[0] + dm[1]*dm[1] + dm[2]*dm[2]);
          std::fprintf(fp, "D %d %.17g %.17g %.17g %.9e %.9e %.9e %.9e\n", pmesh->ncycle,
                       mbsize.h_view(tm).x1min + (ti - indcs.is + 0.5)*mbsize.h_view(tm).dx1,
                       mbsize.h_view(tm).x2min + (tj - indcs.js + 0.5)*mbsize.h_view(tm).dx2,
                       vol, drho, dm[0], dm[1], dm[2]);
        }
      }
      if (cloud_split_log_leaves) {
        const Real vdom = len1*len2;
        std::fprintf(fp, "V %d %.17g %d %.9e %.9e %.9e %.9e\n", pmesh->ncycle, pmesh->time,
                     static_cast<int>(pass_acc.size()), vtouch, vdom, std::sqrt(vdr2/vdom),
                     std::sqrt(vdm2/vdom));
      }
      const Real nl = std::max<Real>(1.0, static_cast<Real>(pass_acc.size()));
      std::fprintf(fp, "P %d %.17g %d %d %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
                   pmesh->ncycle, pmesh->time, nevent_call,
                   static_cast<int>(pass_acc.size()), drmax, std::sqrt(dr2/nl), dfmax[0],
                   dfmax[1], dfmax[2], std::sqrt(df2/nl), dmnet);
      if (interp && interp_missing > 0) {
        std::fprintf(fp, "# interp: %llu piece directions without a neighbouring node\n",
                     interp_missing);
      }
    }
    // global totals of mass and momentum before and after this call
    Real tot[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int p=0; p<npart; ++p) {
      tot[0] += pr_h(IPM,p);
      tot[2] += pr_h(IPM,p)*pr_h(IPVX,p);
      tot[4] += pr_h(IPM,p)*pr_h(IPVY,p);
      tot[6] += pr_h(IPM,p)*pr_h(IPVZ,p);
    }
    for (int s=0; s<nslot; ++s) {
      tot[1] += rr(IPM,s);
      tot[3] += rr(IPM,s)*rr(IPVX,s);
      tot[5] += rr(IPM,s)*rr(IPVY,s);
      tot[7] += rr(IPM,s)*rr(IPVZ,s);
    }
    std::fprintf(fp, "G %d %.17g %d %d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
                 pmesh->ncycle, pmesh->time, nflag, npart, nslot, tot[0], tot[1], tot[2],
                 tot[3], tot[4], tot[5], tot[6], tot[7]);
    std::fclose(fp);
  }
  cloud_split_count += nparent;
  cloud_lobe_split_count += nlobe;

  int nnew = nslot;
  Kokkos::realloc(pr, nrdata, nnew);
  Kokkos::realloc(pi, nidata, nnew);
  auto pr_new = Kokkos::create_mirror_view(pr);
  auto pi_new = Kokkos::create_mirror_view(pi);
  for (int s=0; s<nnew; ++s) {
    for (int c=0; c<nrdata; ++c) pr_new(c,s) = rr(c,s);
    for (int c=0; c<nidata; ++c) pi_new(c,s) = ii(c,s);
  }
  Kokkos::deep_copy(pr, pr_new);
  Kokkos::deep_copy(pi, pi_new);
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;
  return nflag;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MergeClouds
//! \brief Coarsening of refinable clouds (<dust>/cloud_merge, variant A1).  A cell
//! of the level-L grid is a merge cell when no leaf finer than L lies within
//! cloud_merge_buffer level-L cells of it (periodic, Chebyshev distance).  In a merge cell
//!  (1) every lobed level-(L+1) cloud releases its lobes into plain pieces (exact: the
//!      deposit, mass and momentum are unchanged; the transition-2 point gather of the
//!      lobed directions becomes the plain cloud integral);
//!  (2) the plain level-(L+1) clouds of one species are fitted with a mass-weighted linear
//!      velocity field v + G.(x - x_c) (mean only for fewer than five clouds), and clouds
//!      whose residual exceeds alpha*H_L/tau_c, tau_c = min(t_s, 1/Omega), are excluded
//!      one at a time (worst first, refitting each time); the remaining group (at least
//!      two) becomes one plain level-L cloud at its mass centroid with the mass-weighted
//!      mean of every other attribute, so mass and momentum are conserved exactly.  The
//!      excluded clouds are treated as a new group (another stream), and so on.
//! The kinetic energy removed, 1/2 sum m |v - v_mean|^2, splits exactly into the part of
//! the fitted gradient and the residual part; both are accumulated.  Host prototype for a
//! single rank, like SplitCloudsP1; the algorithm is local to a merge cell.

int DustGasDrag::MergeClouds() {
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart = ppar->nprtcl_thispack;
  const auto &transfer = smr_transfer;
  const int nlev = transfer.cloud_nlev;
  if (npart == 0 || nlev == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
  auto pr_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
  auto pi_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pi);
  auto level_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_level);
  auto target_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_target);
  if (cloud_next_tag < 0) cloud_next_tag = pmesh->nprtcl_total;

  const Real x1min = transfer.x1min, x2min = transfer.x2min;
  const Real len1 = transfer.x1max - transfer.x1min;
  const Real len2 = transfer.x2max - transfer.x2min;
  const int nfine1 = transfer.nfine1, nfine2 = transfer.nfine2;
  const Real fdx1 = transfer.fine_dx1, fdx2 = transfer.fine_dx2;
  const int gids = pmy_pack->gids;
  const int ncells1 = qdep.extent_int(4);
  const int ncells2 = qdep.extent_int(3);
  const int B = cloud_merge_buffer;
  auto wrap = [](Real x, Real xmin, Real len) {
    while (x < xmin) x += len;
    while (x >= xmin + len) x -= len;
    return x;
  };
  auto owner_gid = [&](Real x, Real y) {
    int fi = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((x - x1min)/fdx1)), nfine1);
    int fj = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((y - x2min)/fdx2)), nfine2);
    return gids + target_h(fj, fi)/(ncells1*ncells2);
  };

  // merge-cell masks of the level-L grids, L = 0 .. nlev-1
  std::vector<std::vector<char>> merge_cell(nlev);
  std::vector<int> ng1(nlev), ng2(nlev);
  for (int L=0; L<nlev; ++L) {
    const int r = 1 << (nlev - L);
    const int n1 = nfine1/r, n2 = nfine2/r;
    ng1[L] = n1; ng2[L] = n2;
    std::vector<int> maxlev(static_cast<std::size_t>(n1)*n2, -1);
    for (int fj=0; fj<nfine2; ++fj) {
      for (int fi=0; fi<nfine1; ++fi) {
        int &m = maxlev[static_cast<std::size_t>(fj/r)*n1 + fi/r];
        m = std::max(m, static_cast<int>(level_h(fj, fi)));
      }
    }
    merge_cell[L].assign(static_cast<std::size_t>(n1)*n2, 0);
    for (int J=0; J<n2; ++J) {
      for (int I=0; I<n1; ++I) {
        bool ok = true;
        for (int dj=-B; dj<=B && ok; ++dj) {
          const int wj = SMRCompositeTransfer::WrapIndex(J + dj, n2);
          for (int di=-B; di<=B; ++di) {
            if (maxlev[static_cast<std::size_t>(wj)*n1 +
                       SMRCompositeTransfer::WrapIndex(I + di, n1)] > L) {ok = false; break;}
          }
        }
        merge_cell[L][static_cast<std::size_t>(J)*n1 + I] = ok ? 1 : 0;
      }
    }
  }
  // merge cell (index into merge_cell[L]) of a position, or -1
  auto merge_index = [&](Real x, Real y, int L) {
    const Real dx1 = transfer.root_dx1/static_cast<Real>(1 << L);
    const Real dx2 = transfer.root_dx2/static_cast<Real>(1 << L);
    const int I = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((wrap(x, x1min, len1) - x1min)/dx1)), ng1[L]);
    const int J = SMRCompositeTransfer::WrapIndex(
        static_cast<int>(std::floor((wrap(y, x2min, len2) - x2min)/dx2)), ng2[L]);
    const int c = J*ng1[L] + I;
    return merge_cell[L][c] ? c : -1;
  };

  // working list: every particle, plus the pieces of released lobes
  std::vector<std::vector<Real>> R;
  std::vector<std::vector<int>> I;
  R.reserve(npart + npart/2);
  I.reserve(npart + npart/2);
  std::vector<Real> rrow(nrdata);
  std::vector<int> irow(nidata);
  unsigned long long nrelease = 0;
  struct Piece {Real off; Real frac; int lobe;};
  auto release = [](int lobe_old, Piece out[3]) {   // separate every attached lobe
    int n = 0;
    Real total = 0.5 + 0.25*((lobe_old & 1) ? 1 : 0) + 0.25*((lobe_old & 2) ? 1 : 0);
    out[n++] = {0.0, 0.5/total, 0};
    if (lobe_old & 1) out[n++] = {-1.0, 0.25/total, 0};
    if (lobe_old & 2) out[n++] = {1.0, 0.25/total, 0};
    return n;
  };
  for (int p=0; p<npart; ++p) {
    for (int c=0; c<nrdata; ++c) rrow[c] = pr_h(c,p);
    for (int c=0; c<nidata; ++c) irow[c] = pi_h(c,p);
    const int code = irow[PLEV];
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    const int l1 = SMRCompositeTransfer::CloudLobe1(code);
    const int l2 = SMRCompositeTransfer::CloudLobe2(code);
    if ((l1 == 0 && l2 == 0) || lev == 0 || merge_index(rrow[IPX], rrow[IPY], lev - 1) < 0) {
      R.push_back(rrow); I.push_back(irow);
      continue;
    }
    Piece p1[3], p2[3];
    const int n1 = (l1 == 0) ? 1 : release(l1, p1);
    const int n2 = (l2 == 0) ? 1 : release(l2, p2);
    if (l1 == 0) p1[0] = {0.0, 1.0, 0};
    if (l2 == 0) p2[0] = {0.0, 1.0, 0};
    const Real e1 = transfer.CloudHalfWidth1(lev), e2 = transfer.CloudHalfWidth2(lev);
    for (int b=0; b<n2; ++b) {
      for (int a=0; a<n1; ++a) {
        std::vector<Real> r = rrow;
        std::vector<int> i = irow;
        const Real cx = wrap(rrow[IPX] + p1[a].off*e1, x1min, len1);
        const Real cy = wrap(rrow[IPY] + p2[b].off*e2, x2min, len2);
        r[IPX] = cx; r[IPX1] = cx; r[IPY] = cy; r[IPY1] = cy;
        r[IPM] = rrow[IPM]*p1[a].frac*p2[b].frac;
        i[PLEV] = lev;
        i[PGID] = owner_gid(cx, cy);
        if (a > 0 || b > 0) i[PTAG] = cloud_next_tag++;
        R.push_back(r); I.push_back(i);
      }
    }
    ++nrelease;
  }

  // groups: plain clouds of level l >= 1 in a merge cell of level l-1, per species
  std::unordered_map<long long, std::vector<int>> groups;
  const int nw = static_cast<int>(R.size());
  for (int s=0; s<nw; ++s) {
    const int code = I[s][PLEV];
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    if (lev == 0 || SMRCompositeTransfer::CloudLobe1(code) != 0 ||
        SMRCompositeTransfer::CloudLobe2(code) != 0) continue;
    const int c = merge_index(R[s][IPX], R[s][IPY], lev - 1);
    if (c < 0) continue;
    const long long key = (static_cast<long long>(lev)*(1LL << 40) +
                           static_cast<long long>(c))*1024LL + I[s][PSP];
    groups[key].push_back(s);
  }

  std::vector<char> removed(nw, 0);
  std::vector<std::vector<Real>> Rm;   // merged clouds
  std::vector<std::vector<int>> Im;
  unsigned long long nevents = 0, nin = 0, nreject = 0;
  Real dk_grad = 0.0, dk_res = 0.0;
  const Real omega = is_shearing_box ? omega0 : 0.0;
  for (auto &kv : groups) {
    const int lev = static_cast<int>(kv.first/1024LL/(1LL << 40));
    const int L = lev - 1;
    const Real H = std::min(transfer.root_dx1, transfer.root_dx2)/static_cast<Real>(1 << L);
    std::vector<int> pool = kv.second;
    int streams = 0;
    while (pool.size() >= 2 && streams < 8) {
      ++streams;
      std::vector<int> mem = pool, out;
      Real M = 0.0, xc = 0.0, yc = 0.0, vm[3] = {0.0, 0.0, 0.0}, G[3][2];
      Real tol = 0.0;
      bool accepted = false;
      while (mem.size() >= 2) {
        // centroid (minimum image about the first member), mean velocity, stopping time
        const int s0 = mem[0];
        M = 0.0; Real mx = 0.0, my = 0.0, mts = 0.0;
        for (int c=0; c<3; ++c) vm[c] = 0.0;
        for (int s : mem) {
          const Real m = R[s][IPM];
          M += m;
          mx += m*transfer.MinImage1(R[s][IPX] - R[s0][IPX]);
          my += m*transfer.MinImage2(R[s][IPY] - R[s0][IPY]);
          vm[0] += m*R[s][IPVX]; vm[1] += m*R[s][IPVY]; vm[2] += m*R[s][IPVZ];
          mts += m*R[s][IPTS];
        }
        xc = R[s0][IPX] + mx/M; yc = R[s0][IPY] + my/M;
        for (int c=0; c<3; ++c) vm[c] /= M;
        const Real ts = mts/M;
        const Real tau = (omega > 0.0) ? std::min(ts, 1.0/omega) : ts;
        tol = cloud_merge_alpha*H/tau;
        // mass-weighted linear fit of each velocity component (five or more clouds)
        for (int c=0; c<3; ++c) {G[c][0] = 0.0; G[c][1] = 0.0;}
        if (mem.size() >= 5) {
          Real cxx = 0.0, cxy = 0.0, cyy = 0.0, bx[3] = {0.0, 0.0, 0.0}, by[3] = {0.0, 0.0, 0.0};
          for (int s : mem) {
            const Real m = R[s][IPM];
            const Real dx = transfer.MinImage1(R[s][IPX] - xc);
            const Real dy = transfer.MinImage2(R[s][IPY] - yc);
            const Real v[3] = {R[s][IPVX] - vm[0], R[s][IPVY] - vm[1], R[s][IPVZ] - vm[2]};
            cxx += m*dx*dx; cxy += m*dx*dy; cyy += m*dy*dy;
            for (int c=0; c<3; ++c) {bx[c] += m*dx*v[c]; by[c] += m*dy*v[c];}
          }
          const Real det = cxx*cyy - cxy*cxy;
          if (det > 1.0e-6*M*M*H*H*H*H) {
            for (int c=0; c<3; ++c) {
              G[c][0] = (cyy*bx[c] - cxy*by[c])/det;
              G[c][1] = (cxx*by[c] - cxy*bx[c])/det;
            }
          }
        }
        // largest residual
        int worst = -1;
        Real rmax = 0.0;
        for (int k=0; k<static_cast<int>(mem.size()); ++k) {
          const int s = mem[k];
          const Real dx = transfer.MinImage1(R[s][IPX] - xc);
          const Real dy = transfer.MinImage2(R[s][IPY] - yc);
          const Real v[3] = {R[s][IPVX], R[s][IPVY], R[s][IPVZ]};
          Real r2 = 0.0;
          for (int c=0; c<3; ++c) {
            const Real d = v[c] - vm[c] - G[c][0]*dx - G[c][1]*dy;
            r2 += d*d;
          }
          if (r2 > rmax) {rmax = r2; worst = k;}
        }
        if (rmax <= tol*tol) {accepted = true; break;}
        out.push_back(mem[worst]);
        mem.erase(mem.begin() + worst);
      }
      if (!accepted || mem.size() < 2) {
        // nothing mergeable left in this pool: one cloud less to try next time
        if (!mem.empty()) out.insert(out.end(), mem.begin(), mem.end());
        if (out.size() >= pool.size()) {out.erase(out.begin()); ++nreject;}
        pool = out;
        continue;
      }
      // merge the accepted group into one plain level-L cloud
      std::vector<Real> r(nrdata, 0.0);
      std::vector<int> i = I[mem[0]];
      int heavy = mem[0];
      for (int s : mem) {
        const Real m = R[s][IPM];
        if (m > R[heavy][IPM]) heavy = s;
        for (int c=0; c<nrdata; ++c) r[c] += m*R[s][c];
        const Real dx = transfer.MinImage1(R[s][IPX] - xc);
        const Real dy = transfer.MinImage2(R[s][IPY] - yc);
        const Real v[3] = {R[s][IPVX] - vm[0], R[s][IPVY] - vm[1], R[s][IPVZ] - vm[2]};
        for (int c=0; c<3; ++c) {
          const Real g = G[c][0]*dx + G[c][1]*dy;
          dk_grad += 0.5*m*g*g;
          dk_res += 0.5*m*(v[c] - g)*(v[c] - g);
        }
        removed[s] = 1;
      }
      for (int c=0; c<nrdata; ++c) r[c] /= M;
      const Real px = wrap(xc, x1min, len1), py = wrap(yc, x2min, len2);
      r[IPM] = M;
      r[IPX] = px; r[IPX1] = px; r[IPY] = py; r[IPY1] = py;
      r[IPVX] = vm[0]; r[IPVY] = vm[1]; r[IPVZ] = vm[2];
      i[PLEV] = L;
      i[PTAG] = I[heavy][PTAG];
      i[PGID] = owner_gid(px, py);
      Rm.push_back(r); Im.push_back(i);
      ++nevents;
      nin += mem.size();
      pool = out;
    }
    for (int s : pool) if (!removed[s]) ++nreject;
  }

  // totals before (original particles) and after (kept + merged), for the log
  Real tb[4] = {0.0, 0.0, 0.0, 0.0}, ta[4] = {0.0, 0.0, 0.0, 0.0};
  for (int p=0; p<npart; ++p) {
    tb[0] += pr_h(IPM,p); tb[1] += pr_h(IPM,p)*pr_h(IPVX,p);
    tb[2] += pr_h(IPM,p)*pr_h(IPVY,p); tb[3] += pr_h(IPM,p)*pr_h(IPVZ,p);
  }
  int nnew = 0;
  for (int s=0; s<nw; ++s) if (!removed[s]) ++nnew;
  nnew += static_cast<int>(Rm.size());
  Kokkos::realloc(pr, nrdata, nnew);
  Kokkos::realloc(pi, nidata, nnew);
  auto pr_new = Kokkos::create_mirror_view(pr);
  auto pi_new = Kokkos::create_mirror_view(pi);
  Real mlev[3] = {0.0, 0.0, 0.0}, mlobe = 0.0;
  int q = 0;
  auto put = [&](const std::vector<Real> &r, const std::vector<int> &i) {
    for (int c=0; c<nrdata; ++c) pr_new(c,q) = r[c];
    for (int c=0; c<nidata; ++c) pi_new(c,q) = i[c];
    ta[0] += r[IPM]; ta[1] += r[IPM]*r[IPVX]; ta[2] += r[IPM]*r[IPVY]; ta[3] += r[IPM]*r[IPVZ];
    const int code = i[PLEV];
    mlev[std::min(2, SMRCompositeTransfer::CloudLevel(code))] += r[IPM];
    if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) {
      mlobe += r[IPM];
    }
    ++q;
  };
  for (int s=0; s<nw; ++s) if (!removed[s]) put(R[s], I[s]);
  for (std::size_t k=0; k<Rm.size(); ++k) put(Rm[k], Im[k]);
  Kokkos::deep_copy(pr, pr_new);
  Kokkos::deep_copy(pi, pi_new);
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;

  cloud_coarsen_events += nevents;
  cloud_coarsen_in += nin;
  cloud_coarsen_reject += nreject;
  cloud_far_release += nrelease;
  cloud_coarsen_dk_grad += dk_grad;
  cloud_coarsen_dk_res += dk_res;
  // one line per pass: counts, energy removed, conservation and the mass by cloud type
  std::FILE *fp = std::fopen("cloud_merge_log.txt", "a");
  if (fp != nullptr) {
    std::fseek(fp, 0, SEEK_END);
    if (std::ftell(fp) == 0) {
      std::fprintf(fp, "# ncycle time npart_before npart_after far_releases events merged_in"
                   " rejected dK_grad dK_res M_b M_a PX_b PX_a PY_b PY_a PZ_b PZ_a"
                   " m_level0 m_level1 m_level2 m_lobed\n");
    }
    std::fprintf(fp, "%d %.17g %d %d %llu %llu %llu %llu %.9e %.9e %.17g %.17g %.17g %.17g"
                 " %.17g %.17g %.17g %.17g %.9e %.9e %.9e %.9e\n", pmesh->ncycle,
                 pmesh->time, npart, nnew, nrelease, nevents, nin, nreject, dk_grad, dk_res,
                 tb[0], ta[0], tb[1], ta[1], tb[2], ta[2], tb[3], ta[3], mlev[0], mlev[1],
                 mlev[2], mlobe);
    std::fclose(fp);
  }
  return static_cast<int>(nevents);
}

} // namespace dust
