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
#include <cmath>
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
  int nlev = pmesh->max_level - pmesh->root_level;

  if (nlev < 0 || nlev > 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "refinable_cloud currently requires a uniform mesh or one "
              << "or two levels of 2:1 static refinement" << std::endl;
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
    std::cout << "# refinable_cloud map: levels=" << nlev << ", virtual finest grid "
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
              << (t.cloud_split_everywhere ? ", split everywhere" : "") << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::RefineClouds
//! \brief Task wrapper: split eligible clouds at the start of every cycle.

TaskStatus DustGasDrag::RefineClouds(Driver *pdrive, int stage) {
  if (smr_transfer_mode != DustSMRTransfer::refinable_cloud) return TaskStatus::complete;
  SplitAndMergeClouds();
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
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;

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

  unsigned long long nparent = 0, nlobe = 0;
  for (int p=0; p<npart; ++p) {
    if (!flag_h(p)) continue;
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
      n1 = pieces(3, eligible(x - h1, y, lev), eligible(x + h1, y, lev), p1);
      n2 = pieces(3, eligible(x, y - h2, lev), eligible(x, y + h2, lev), p2);
      clev = lev + 1;
      ++nparent;
    } else {
      // lobed cloud of level lev (parent level lev-1): separate the lobes whose
      // neighbour parent has become eligible
      const Real h1 = transfer.CloudHalfWidth1(lev - 1);
      const Real h2 = transfer.CloudHalfWidth2(lev - 1);
      n1 = pieces(SMRCompositeTransfer::CloudLobe1(code), eligible(x - h1, y, lev - 1),
                  eligible(x + h1, y, lev - 1), p1);
      n2 = pieces(SMRCompositeTransfer::CloudLobe2(code), eligible(x, y - h2, lev - 1),
                  eligible(x, y + h2, lev - 1), p2);
      clev = lev;
      ++nlobe;
    }
    const Real e1 = transfer.CloudHalfWidth1(clev), e2 = transfer.CloudHalfWidth2(clev);
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

} // namespace dust
