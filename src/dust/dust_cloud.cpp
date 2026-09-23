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

  if (nlev < 1 || nlev > 2) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "refinable_cloud currently requires one or two levels "
              << "of 2:1 static refinement" << std::endl;
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
  unsigned long long merged_here = 0;
  for (int p=0; p<npart; ++p) {
    if (!flag_h(p)) continue;
    int lev = pi_h(PLEV,p);
    Real ell1 = transfer.CloudHalfWidth1(lev);
    Real ell2 = transfer.CloudHalfWidth2(lev);
    for (int b=0; b<3; ++b) {
      for (int a=0; a<3; ++a) {
        Real x = wrap1(pr_h(IPX,p) + offset[a]*ell1);
        Real y = wrap2(pr_h(IPY,p) + offset[b]*ell2);
        Real mass = pr_h(IPM,p)*fraction[a]*fraction[b];
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
            rr(c,s) = rr(c,s) + wc*(pr_h(c,p) - rr(c,s));
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
          for (int c=0; c<nrdata; ++c) rr(c,s) = pr_h(c,p);
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

} // namespace dust
