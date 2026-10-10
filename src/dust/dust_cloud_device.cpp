//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud_device.cpp
//! \brief Device (Kokkos) implementation of the moving-particle P1 split of refinable
//! clouds.  Same rules as the host SplitCloudsP1, without its
//! diagnostics (split log, cloud_split_interp node lookup, split oracle), which keep the
//! host path.  Flag and count the pieces per particle, exclusive scans for the slots and
//! the new tags, then one fill kernel.  The particle order is the host version's: kept
//! particles in their order, then the pieces of the flagged particles in theirs.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
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

struct DPiece {Real off; Real frac; int lobe;};

// pieces of a cloud along one direction (port of the host 'pieces' lambda): offset in
// child half-widths, fraction of the cloud's mass, lobe code; lobe_old = 3 for a plain
// parent (centre child with both edge children attached)
KOKKOS_INLINE_FUNCTION
int P1Pieces(const int lobe_old, const bool sep_lo, const bool sep_hi, DPiece out[3]) {
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

// is the particle split (or a lobe released) now?  Port of the host flag kernel.
KOKKOS_INLINE_FUNCTION
bool P1Flag(const SMRCompositeTransfer &t, const int code, const Real x, const Real y,
            const Real slack, const bool presplit, const bool release) {
  const int lev = SMRCompositeTransfer::CloudLevel(code);
  const int l1 = SMRCompositeTransfer::CloudLobe1(code);
  const int l2 = SMRCompositeTransfer::CloudLobe2(code);
  if (l1 == 0 && l2 == 0) {
    if (presplit && lev == 0 && lev < t.cloud_nlev) return true;
    return t.CloudInsideFiner(x, y, lev, slack);
  }
  if (lev > 0 && release && !t.CloudInsideFiner(x, y, lev - 1, slack)) return true;
  if (lev > 0 && t.CloudInsideFiner(x, y, lev - 1, slack)) {
    const Real h1 = t.CloudHalfWidth1(lev - 1), h2 = t.CloudHalfWidth2(lev - 1);
    return ((l1 & 1) && t.CloudInsideFiner(x - h1, y, lev - 1, slack)) ||
           ((l1 & 2) && t.CloudInsideFiner(x + h1, y, lev - 1, slack)) ||
           ((l2 & 1) && t.CloudInsideFiner(x, y - h2, lev - 1, slack)) ||
           ((l2 & 2) && t.CloudInsideFiner(x, y + h2, lev - 1, slack));
  }
  return false;
}

// pieces of a flagged particle along both directions (port of the host event code)
KOKKOS_INLINE_FUNCTION
void P1Event(const SMRCompositeTransfer &t, const int code, const Real x, const Real y,
             const Real slack, const bool presplit, const bool release, DPiece p1[3],
             DPiece p2[3], int &n1, int &n2, int &clev, bool &parent) {
  const int lev = SMRCompositeTransfer::CloudLevel(code);
  const int l1 = SMRCompositeTransfer::CloudLobe1(code);
  const int l2 = SMRCompositeTransfer::CloudLobe2(code);
  if (l1 == 0 && l2 == 0) {
    const Real h1 = t.CloudHalfWidth1(lev), h2 = t.CloudHalfWidth2(lev);
    const bool px = presplit && lev == 0;
    n1 = P1Pieces(3, px || t.CloudInsideFiner(x - h1, y, lev, slack),
                  px || t.CloudInsideFiner(x + h1, y, lev, slack), p1);
    n2 = P1Pieces(3, t.CloudInsideFiner(x, y - h2, lev, slack),
                  t.CloudInsideFiner(x, y + h2, lev, slack), p2);
    clev = lev + 1;
    parent = true;
  } else {
    const Real h1 = t.CloudHalfWidth1(lev - 1), h2 = t.CloudHalfWidth2(lev - 1);
    const bool out = release && !t.CloudInsideFiner(x, y, lev - 1, slack);
    n1 = P1Pieces(l1, out || t.CloudInsideFiner(x - h1, y, lev - 1, slack),
                  out || t.CloudInsideFiner(x + h1, y, lev - 1, slack), p1);
    n2 = P1Pieces(l2, out || t.CloudInsideFiner(x, y - h2, lev - 1, slack),
                  out || t.CloudInsideFiner(x, y + h2, lev - 1, slack), p2);
    clev = lev;
    parent = false;
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitCloudsP1Device
//! \brief Device P1 split; returns the number of split/released particles.

int DustGasDrag::SplitCloudsP1Device() {
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart = ppar->nprtcl_thispack;
  if (npart == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const auto transfer = smr_transfer;
  const Real slack = cloud_split_slack;
  const bool release = cloud_p1_release_exit;
  const bool presplit = cloud_presplit_x1;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
  const int gids = pmy_pack->gids;
  const int ncell12 = qdep.extent_int(4)*qdep.extent_int(3);

  // pieces per particle (0 = not flagged), and the numbers of parents / lobed events
  DvceArray1D<int> np(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dust_split_np"), npart);
  int nflag = 0, nparent = 0;
  Kokkos::parallel_reduce("dust_split_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &sf, int &sp) {
    const int code = pi(PLEV,p);
    const Real x = pr(IPX,p), y = pr(IPY,p);
    if (!P1Flag(transfer, code, x, y, slack, presplit, release)) {np(p) = 0; return;}
    DPiece p1[3], p2[3]; int n1, n2, clev; bool parent;
    P1Event(transfer, code, x, y, slack, presplit, release, p1, p2, n1, n2, clev, parent);
    np(p) = n1*n2;
    sf += 1;
    if (parent) sp += 1;
  }, Kokkos::Sum<int>(nflag), Kokkos::Sum<int>(nparent));
  if (nflag == 0) return 0;
  if (cloud_next_tag < 0) cloud_next_tag = pmesh->nprtcl_total;

  // slots of the kept particles, first slots of the pieces, first new tags
  DvceArray1D<int> keepoff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dust_split_keepoff"), npart);
  DvceArray1D<int> pieceoff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dust_split_pieceoff"), npart);
  DvceArray1D<int> tagoff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "dust_split_tagoff"), npart);
  int nkeep = 0, npieces = 0, ntags = 0;
  Kokkos::parallel_scan("dust_split_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) keepoff(p) = upd;
    upd += (np(p) == 0) ? 1 : 0;
  }, nkeep);
  Kokkos::parallel_scan("dust_split_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) pieceoff(p) = upd;
    upd += np(p);
  }, npieces);
  Kokkos::parallel_scan("dust_split_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) tagoff(p) = upd;
    upd += (np(p) > 0) ? np(p) - 1 : 0;
  }, ntags);

  const int nnew = nkeep + npieces;
  DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, nnew);
  DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, nnew);
  const int tag0 = cloud_next_tag;
  Kokkos::parallel_for("dust_split_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (np(p) == 0) {
      const int s = keepoff(p);
      for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
      for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
      return;
    }
    const int code = pi(PLEV,p);
    const Real x = pr(IPX,p), y = pr(IPY,p);
    DPiece p1[3], p2[3]; int n1, n2, clev; bool parent;
    P1Event(transfer, code, x, y, slack, presplit, release, p1, p2, n1, n2, clev, parent);
    const Real e1 = transfer.CloudHalfWidth1(clev), e2 = transfer.CloudHalfWidth2(clev);
    int k = 0;
    for (int b=0; b<n2; ++b) {
      for (int a=0; a<n1; ++a) {
        const int s = nkeep + pieceoff(p) + k;
        const Real cx = SMRCompositeTransfer::WrapPosition(x + p1[a].off*e1, transfer.x1min,
                                                           transfer.x1max);
        const Real cy = SMRCompositeTransfer::WrapPosition(y + p2[b].off*e2, transfer.x2min,
                                                           transfer.x2max);
        for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
        for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
        prn(IPX,s) = cx; prn(IPX1,s) = cx;
        prn(IPY,s) = cy; prn(IPY1,s) = cy;
        prn(IPM,s) = pr(IPM,p)*p1[a].frac*p2[b].frac;
        pin(PLEV,s) = clev | (p1[a].lobe << sh1) | (p2[b].lobe << sh2);
        const int fi = SMRCompositeTransfer::WrapIndex(static_cast<int>(
            Kokkos::floor((cx - transfer.x1min)/transfer.fine_dx1)), transfer.nfine1);
        const int fj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
            Kokkos::floor((cy - transfer.x2min)/transfer.fine_dx2)), transfer.nfine2);
        pin(PGID,s) = gids + transfer.cloud_target(fj, fi)/ncell12;
        if (k > 0) pin(PTAG,s) = tag0 + tagoff(p) + k - 1;   // the centre keeps the tag
        ++k;
      }
    }
  });
  pr = prn;
  pi = pin;
  cloud_next_tag += ntags;
  cloud_split_count += nparent;
  cloud_lobe_split_count += nflag - nparent;
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;
  return nflag;
}


namespace {

constexpr int kNbMax = 96;     // pieces in a centre's tent support
constexpr int kNodeMax = 64;   // nodes of a centre (regular offsets and irregular pieces)

KOKKOS_INLINE_FUNCTION
Real Snap(const Real d, const Real ell) {
  const Real r = Kokkos::round(d/ell);
  return (Kokkos::fabs(d - r*ell) < 0.1*ell) ? r*ell : d;
}

// merge-cell masks of the level-L grids, concatenated, and the level-L cell of a position
struct MergeMask {
  DvceArray1D<char> mask;
  int n1[4], n2[4], off[4];
  Real dx1[4], dx2[4];
  Real x1min, x2min, x1max, x2max;
  KOKKOS_INLINE_FUNCTION
  int Cell(const Real x, const Real y, const int L) const {
    const int I = SMRCompositeTransfer::WrapIndex(static_cast<int>(Kokkos::floor(
        (SMRCompositeTransfer::WrapPosition(x, x1min, x1max) - x1min)/dx1[L])), n1[L]);
    const int J = SMRCompositeTransfer::WrapIndex(static_cast<int>(Kokkos::floor(
        (SMRCompositeTransfer::WrapPosition(y, x2min, x2max) - x2min)/dx2[L])), n2[L]);
    return J*n1[L] + I;
  }
  KOKKOS_INLINE_FUNCTION
  bool In(const Real x, const Real y, const int L) const {
    return mask(off[L] + Cell(x, y, L)) != 0;
  }
};

// fraction of a unit-mass hat (centre c, half-width w) in [a, b] (D3 diagnostic)
KOKKOS_INLINE_FUNCTION
Real HatFrac(const Real c, const Real w, const Real a, const Real b) {
  auto cdf = [](const Real u) -> Real {
    if (u <= -1.0) return 0.0;
    if (u >= 1.0) return 1.0;
    return (u <= 0.0) ? 0.5*(u + 1.0)*(u + 1.0) : 1.0 - 0.5*(1.0 - u)*(1.0 - u);
  };
  return cdf((b - c)/w) - cdf((a - c)/w);
}

// pieces of a released lobed cloud along one direction (port of the host 'release')
KOKKOS_INLINE_FUNCTION
int ReleasePieces(const int lobe_old, DPiece out[3]) {
  if (lobe_old == 0) {out[0].off = 0.0; out[0].frac = 1.0; out[0].lobe = 0; return 1;}
  int n = 0;
  const Real total = 0.5 + 0.25*((lobe_old & 1) ? 1 : 0) + 0.25*((lobe_old & 2) ? 1 : 0);
  out[n].off = 0.0; out[n].frac = 0.5/total; out[n].lobe = 0; ++n;
  if (lobe_old & 1) {out[n].off = -1.0; out[n].frac = 0.25/total; out[n].lobe = 0; ++n;}
  if (lobe_old & 2) {out[n].off = 1.0; out[n].frac = 0.25/total; out[n].lobe = 0; ++n;}
  return n;
}

// counting sort of particles by key (key < 0: not listed): start(key), list
void BuildCSR(const DvceArray1D<int> &key, const int nkey, DvceArray1D<int> &start,
              DvceArray1D<int> &list) {
  const int n = key.extent_int(0);
  DvceArray1D<int> cnt("csr_cnt", nkey);
  Kokkos::parallel_for("csr_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const int p) {
    if (key(p) >= 0) Kokkos::atomic_increment(&cnt(key(p)));
  });
  start = DvceArray1D<int>("csr_start", nkey + 1);
  auto st = start;
  Kokkos::parallel_scan("csr_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, nkey),
  KOKKOS_LAMBDA(const int k, int &upd, const bool final) {
    if (final) st(k) = upd;
    upd += cnt(k);
    if (final && k == nkey - 1) st(nkey) = upd;
  });
  int total = 0;
  Kokkos::deep_copy(total, Kokkos::subview(start, nkey));
  list = DvceArray1D<int>("csr_list", total > 0 ? total : 1);
  auto ls = list;
  Kokkos::deep_copy(cnt, 0);
  Kokkos::parallel_for("csr_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const int p) {
    if (key(p) < 0) return;
    const int k = key(p);
    ls(st(k) + Kokkos::atomic_fetch_add(&cnt(k), 1)) = p;
  });
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MergeCloudsDevice
//! \brief Device version of MergeClouds: far-field lobe release and the P1
//! tent merge with the node rule and the merge-side node convention, as the host prototype.
//! Differences, by construction of the parallel algorithm: centres are the local maxima
//! (mass, then x, then y) among unblocked zone pieces within the blocking radius, in two
//! rounds (the host designates greedily, heaviest first); commits run in 16 colour rounds of
//! the centres' level-L cells (I mod 4, J mod 4), so that the supports, the nodes' implicit
//! masses and the new clouds of concurrent centres never overlap.  On a quiet lattice the
//! centres and the commits are the host's; for irregular particles the order differs.

int DustGasDrag::MergeCloudsDevice() {
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart0 = ppar->nprtcl_thispack;
  const auto transfer = smr_transfer;
  const int nlev = transfer.cloud_nlev;
  if (npart0 == 0 || nlev == 0) return 0;
  if (nlev > 3) {
    std::cout << "### FATAL ERROR: MergeCloudsDevice supports at most 3 cloud levels" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
  const int gids = pmy_pack->gids;
  const int ncell12 = qdep.extent_int(4)*qdep.extent_int(3);
  const int B = cloud_merge_buffer;
  if (cloud_next_tag < 0) cloud_next_tag = pmesh->nprtcl_total;

  // merge-cell masks (host, as MergeClouds), built once: the refinement is static
  MergeMask mm;
  mm.x1min = transfer.x1min; mm.x2min = transfer.x2min;
  mm.x1max = transfer.x1max; mm.x2max = transfer.x2max;
  for (int L=0; L<nlev; ++L) {
    mm.dx1[L] = transfer.root_dx1/static_cast<Real>(1 << L);
    mm.dx2[L] = transfer.root_dx2/static_cast<Real>(1 << L);
  }
  if (merge_mask_built) {
    mm.mask = merge_mask_d;
    for (int L=0; L<nlev; ++L) {
      mm.n1[L] = merge_mask_geom[3*L]; mm.n2[L] = merge_mask_geom[3*L+1];
      mm.off[L] = merge_mask_geom[3*L+2];
    }
  } else {
    auto level_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_level);
    const int nfine1 = transfer.nfine1, nfine2 = transfer.nfine2;
    std::vector<char> all;
    for (int L=0; L<nlev; ++L) {
      const int r = 1 << (nlev - L);
      const int n1 = nfine1/r, n2 = nfine2/r;
      mm.n1[L] = n1; mm.n2[L] = n2; mm.off[L] = static_cast<int>(all.size());
      mm.dx1[L] = transfer.root_dx1/static_cast<Real>(1 << L);
      mm.dx2[L] = transfer.root_dx2/static_cast<Real>(1 << L);
      std::vector<int> maxlev(static_cast<std::size_t>(n1)*n2, -1);
      for (int fj=0; fj<nfine2; ++fj) {
        for (int fi=0; fi<nfine1; ++fi) {
          int &m = maxlev[static_cast<std::size_t>(fj/r)*n1 + fi/r];
          m = std::max(m, static_cast<int>(level_h(fj, fi)));
        }
      }
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
          all.push_back(ok ? 1 : 0);
        }
      }
    }
    mm.mask = DvceArray1D<char>("merge_mask", all.size());
    auto mh = Kokkos::create_mirror_view(mm.mask);
    for (std::size_t k=0; k<all.size(); ++k) mh(k) = all[k];
    Kokkos::deep_copy(mm.mask, mh);
    merge_mask_d = mm.mask;
    for (int L=0; L<nlev; ++L) {
      merge_mask_geom[3*L] = mm.n1[L]; merge_mask_geom[3*L+1] = mm.n2[L];
      merge_mask_geom[3*L+2] = mm.off[L];
    }
    merge_mask_built = true;
  }

  // totals before
  Real tb[4] = {0.0, 0.0, 0.0, 0.0};
  {
    auto pr = ppar->prtcl_rdata;
    Kokkos::parallel_reduce("merge_tot_b", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart0),
    KOKKOS_LAMBDA(const int p, Real &a, Real &b, Real &c, Real &d) {
      const Real m = pr(IPM,p);
      a += m; b += m*pr(IPVX,p); c += m*pr(IPVY,p); d += m*pr(IPVZ,p);
    }, tb[0], tb[1], tb[2], tb[3]);
  }

  // (1) far-field release of lobed level-l clouds in merge cells of level l-1
  int nrelease = 0;
  {
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    DvceArray1D<int> np(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rel_np"), npart);
    Kokkos::parallel_reduce("rel_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &sum) {
      const int code = pi(PLEV,p);
      const int lev = SMRCompositeTransfer::CloudLevel(code);
      const int l1 = SMRCompositeTransfer::CloudLobe1(code);
      const int l2 = SMRCompositeTransfer::CloudLobe2(code);
      if ((l1 == 0 && l2 == 0) || lev == 0 || !mm.In(pr(IPX,p), pr(IPY,p), lev - 1)) {
        np(p) = 0; return;
      }
      DPiece a[3], b[3];
      np(p) = ReleasePieces(l1, a)*ReleasePieces(l2, b);
      sum += 1;
    }, Kokkos::Sum<int>(nrelease));
    if (nrelease > 0) {
      DvceArray1D<int> off(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rel_off"), npart), toff("rel_toff", npart);
      int nnew = 0, ntags = 0;
      Kokkos::parallel_scan("rel_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
        if (final) off(p) = upd;
        upd += (np(p) == 0) ? 1 : np(p);
      }, nnew);
      Kokkos::parallel_scan("rel_tscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
        if (final) toff(p) = upd;
        upd += (np(p) > 0) ? np(p) - 1 : 0;
      }, ntags);
      DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, nnew);
      DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, nnew);
      const int tag0 = cloud_next_tag;
      Kokkos::parallel_for("rel_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        const int s0 = off(p);
        if (np(p) == 0) {
          for (int c=0; c<nrdata; ++c) prn(c,s0) = pr(c,p);
          for (int c=0; c<nidata; ++c) pin(c,s0) = pi(c,p);
          return;
        }
        const int code = pi(PLEV,p);
        const int lev = SMRCompositeTransfer::CloudLevel(code);
        DPiece a[3], b[3];
        const int n1 = ReleasePieces(SMRCompositeTransfer::CloudLobe1(code), a);
        const int n2 = ReleasePieces(SMRCompositeTransfer::CloudLobe2(code), b);
        const Real e1 = transfer.CloudHalfWidth1(lev), e2 = transfer.CloudHalfWidth2(lev);
        int k = 0;
        for (int jb=0; jb<n2; ++jb) {
          for (int ja=0; ja<n1; ++ja) {
            const int s = s0 + k;
            const Real cx = SMRCompositeTransfer::WrapPosition(pr(IPX,p) + a[ja].off*e1,
                                                               transfer.x1min, transfer.x1max);
            const Real cy = SMRCompositeTransfer::WrapPosition(pr(IPY,p) + b[jb].off*e2,
                                                               transfer.x2min, transfer.x2max);
            for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
            for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
            prn(IPX,s) = cx; prn(IPX1,s) = cx; prn(IPY,s) = cy; prn(IPY1,s) = cy;
            prn(IPM,s) = pr(IPM,p)*a[ja].frac*b[jb].frac;
            pin(PLEV,s) = lev;
            const int fi = SMRCompositeTransfer::WrapIndex(static_cast<int>(
                Kokkos::floor((cx - transfer.x1min)/transfer.fine_dx1)), transfer.nfine1);
            const int fj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
                Kokkos::floor((cy - transfer.x2min)/transfer.fine_dx2)), transfer.nfine2);
            pin(PGID,s) = gids + transfer.cloud_target(fj, fi)/ncell12;
            if (k > 0) pin(PTAG,s) = tag0 + toff(p) + k - 1;
            ++k;
          }
        }
      });
      ppar->prtcl_rdata = prn;
      ppar->prtcl_idata = pin;
      ppar->nprtcl_thispack = nnew;
      cloud_next_tag += ntags;
    }
  }

  // root-level cloud mass per species (first pass, as the host)
  if (cloud_merge_mroot.empty()) {
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    int nsp = 1;
    Kokkos::parallel_reduce("merge_nsp", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &mx) {if (pi(PSP,p) + 1 > mx) mx = pi(PSP,p) + 1;},
    Kokkos::Max<int>(nsp));
    cloud_merge_mroot.assign(nsp, 0.0);
    for (int sp=0; sp<nsp; ++sp) {
      Real m0 = 0.0, m1 = 0.0;
      Kokkos::parallel_reduce("merge_mroot", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &a, Real &b) {
        const int code = pi(PLEV,p);
        if (pi(PSP,p) != sp || SMRCompositeTransfer::CloudLobe1(code) ||
            SMRCompositeTransfer::CloudLobe2(code)) return;
        const int l = SMRCompositeTransfer::CloudLevel(code);
        if (l == 0 && pr(IPM,p) > a) a = pr(IPM,p);
        if (l == 1 && pr(IPM,p) > b) b = pr(IPM,p);
      }, Kokkos::Max<Real>(m0), Kokkos::Max<Real>(m1));
      cloud_merge_mroot[sp] = (m0 > 0.0) ? m0 : 4.0*m1;
    }
  }
  const int nsp = static_cast<int>(cloud_merge_mroot.size());
  if (nsp > 4) {
    std::cout << "### FATAL ERROR: MergeCloudsDevice supports at most 4 species" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real mrootsp[4] = {0.0, 0.0, 0.0, 0.0};
  for (int sp=0; sp<nsp; ++sp) mrootsp[sp] = cloud_merge_mroot[sp];

  // counters: 0 commits, 1 contributions, 2 rejected, 3 pending, 4 nc shifts, 5 nc skipped
  // 6..10 (pending log): not ready (mask), lobe wait, over limit, tolerance, min mass
  DvceArray1D<long long> icnt("merge_icnt", 11);
  DvceArray1D<Real> dk("merge_dk", 2);   // dK_grad, dK_res
  const Real omega = is_shearing_box ? omega0 : 0.0;
  const Real alpha = cloud_merge_alpha, fmin = cloud_merge_min_mass;
  const bool nc = cloud_merge_nc;
  const bool lobe_wait = cloud_merge_lobe_wait > 0, lobe_wait_dir = cloud_merge_lobe_wait == 2;
  const bool d3 = cloud_merge_d3;
  const bool absorb = cloud_merge_absorb;
  double absc[5] = {0.0, 0.0, 0.0, 0.0, 0.0};   // absorbed n, absorbed mass, incompatible, no recipient, dK
  const int d3_iter = cloud_merge_d3_iter;
  double d3s[2] = {0.0, 0.0};   // D3 diagnostic: row first-moment mismatch^2 before / after

  // quantum mode: counters over levels, and the stage-C interior masks (built once):
  // a level-lev cell is interior when every fine cell within cloud_compress_buffer cells of the
  // next-coarser level carries a leaf of level >= lev
  double qtot[24] = {0.0};
  unsigned long long q_events = 0, q_in = 0;
  if (cloud_merge_mode == 1 && cloud_compress && !compress_mask_built) {
    auto level_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), transfer.cloud_level);
    const int nfine1 = transfer.nfine1, nfine2 = transfer.nfine2;
    std::vector<char> all;
    for (int lev=0; lev<=nlev && lev<5; ++lev) {
      compress_mask_off[lev] = static_cast<int>(all.size());
      if (lev == 0) continue;
      const int r = 1 << (nlev - lev), m1 = nfine1/r, m2 = nfine2/r;
      const int bf = 2*cloud_compress_buffer*r;
      std::vector<int> rowmin(static_cast<std::size_t>(nfine1)*nfine2);
      for (int fj=0; fj<nfine2; ++fj) {   // min level over the x window
        for (int fi=0; fi<nfine1; ++fi) {
          int mn = 1 << 20;
          for (int d=-bf; d<r+bf; ++d) mn = std::min(mn, static_cast<int>(level_h(fj,
              SMRCompositeTransfer::WrapIndex(fi + d, nfine1))));
          rowmin[static_cast<std::size_t>(fj)*nfine1 + fi] = mn;
        }
      }
      for (int J=0; J<m2; ++J) {
        for (int I=0; I<m1; ++I) {
          int mn = 1 << 20;
          for (int d=-bf; d<r+bf; ++d) {
            mn = std::min(mn, rowmin[static_cast<std::size_t>(SMRCompositeTransfer::WrapIndex(
                J*r + d, nfine2))*nfine1 + I*r]);
          }
          all.push_back(mn >= lev ? 1 : 0);
        }
      }
    }
    compress_mask_d = DvceArray1D<char>("compress_mask", all.size() > 0 ? all.size() : 1);
    auto mh = Kokkos::create_mirror_view(compress_mask_d);
    for (std::size_t k=0; k<all.size(); ++k) mh(k) = all[k];
    Kokkos::deep_copy(compress_mask_d, mh);
    compress_mask_built = true;
  }

  for (int lev=1; lev<=nlev; ++lev) {
    const int L = lev - 1;
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    const Real H1 = mm.dx1[L], H2 = mm.dx2[L], H = Kokkos::min(H1, H2);
    const Real ell1 = transfer.CloudHalfWidth1(lev), ell2 = transfer.CloudHalfWidth2(lev);
    const int n1 = mm.n1[L], n2 = mm.n2[L], ncl = n1*n2, nkey = nsp*ncl;
    // cell keys: plain level-lev pieces (m > 0) and plain level-L clouds, by species
    DvceArray1D<int> kp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_kp"), npart), kc("merge_kc", npart);
    DvceArray1D<int> kl("merge_kl", npart);   // unreleased lobed level-lev clouds (lobe wait)
    Kokkos::parallel_for("merge_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      const int code = pi(PLEV,p);
      kp(p) = -1; kc(p) = -1; kl(p) = -1;
      if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) {
        if ((lobe_wait || absorb) && SMRCompositeTransfer::CloudLevel(code) == lev && pr(IPM,p) > 0.0) {
          kl(p) = pi(PSP,p)*ncl + mm.Cell(pr(IPX,p), pr(IPY,p), L);
        }
        return;
      }
      const int l = SMRCompositeTransfer::CloudLevel(code);
      const int key = pi(PSP,p)*ncl + mm.Cell(pr(IPX,p), pr(IPY,p), L);
      if (l == lev && pr(IPM,p) > 0.0) kp(p) = key;
      if (l == L) kc(p) = key;
    });
    DvceArray1D<int> sp_start, sp_list, sc_start, sc_list, sl_start, sl_list;
    BuildCSR(kp, nkey, sp_start, sp_list);
    BuildCSR(kc, nkey, sc_start, sc_list);
    if (lobe_wait || absorb) BuildCSR(kl, nkey, sl_start, sl_list);
    if (cloud_merge_mode == 1) {
      // ===== quantum mode =====
      // X: exact P1-stencil re-merge (geometric, strict tolerance, velocity bounding box);
      // G: mass-quantum grouping of plain level-lev pieces in level-L merge cells into level-L
      //    clouds (true-velocity and position bounding boxes, K open groups, rotated walks);
      // C: the same walk at level lev in the fine interior (cloud_compress), with a dS budget;
      // then the absorption of residual pieces deep in the merge zone, and the compaction.
      const int nmax = (npart > 0) ? npart : 1;
      DvceArray2D<Real> nr("q_nr", nrdata, nmax);
      DvceArray2D<int> ni("q_ni", nidata, nmax);
      DvceArray1D<int> nkind("q_nkind", nmax);             // 0 X (final), 1 G, 2 C
      DvceArray2D<double> acc("q_acc", 8, nmax);           // m, m dx, m dy, m dx^2, m dy^2, m v^2, ref x, ref y
      DvceArray1D<int> nnew("q_nnew", 1);
      DvceArray1D<double> qc("q_counters", 24);
      const int K = cloud_merge_K, rot = cloud_merge_pass % 16, rev = (cloud_merge_pass/16) % 2;
      const Real ell = Kokkos::min(ell1, ell2);
      // ---- stage X
      if (cloud_merge_stencil) {
        DvceArray1D<int> owner("q_owner", npart);
        Kokkos::deep_copy(owner, -1);
        const Real dtol = cloud_merge_stencil_tol;
        const Real vn1 = H1*H1/6.0, vn2 = H2*H2/6.0, vo1 = ell1*ell1/6.0, vo2 = ell2*ell2/6.0;
        Kokkos::parallel_for("q_stencil", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int c) {
          if (kp(c) < 0 || pr(IPM,c) <= 0.0) return;
          const Real xc0 = pr(IPX,c), yc0 = pr(IPY,c), mc = pr(IPM,c), ts = pr(IPTS,c);
          if (!mm.In(xc0, yc0, L)) return;
          const int sp = pi(PSP,c);
          const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
          const Real tolv = alpha*H/tau;
          int mem[9]; int nm = 0; mem[nm++] = c;
          Real vlo[3] = {pr(IPVX,c), pr(IPVY,c), pr(IPVZ,c)};
          Real vhi[3] = {vlo[0], vlo[1], vlo[2]};
          for (int b=-1; b<=1; ++b) {
            for (int a=-1; a<=1; ++a) {
              if (a == 0 && b == 0) continue;
              const Real w = mc/static_cast<Real>((a != 0 ? 2 : 1)*(b != 0 ? 2 : 1));
              const Real tx = xc0 + a*ell1, ty = yc0 + b*ell2;
              const int key = sp*ncl + mm.Cell(tx, ty, L);
              int best = -1; Real bd = 1.0e30;
              for (int k=sp_start(key); k<sp_start(key+1); ++k) {
                const int q = sp_list(k);
                if (q == c || pr(IPM,q) <= 0.0 || owner(q) >= 0) continue;
                if (Kokkos::fabs(pr(IPM,q) - w) > 1.0e-6*w || pr(IPTS,q) != ts) continue;
                const Real dx = Kokkos::fabs(transfer.MinImage1(pr(IPX,q) - tx));
                const Real dy = Kokkos::fabs(transfer.MinImage2(pr(IPY,q) - ty));
                if (dx > dtol*ell1 || dy > dtol*ell2) continue;
                if (!mm.In(pr(IPX,q), pr(IPY,q), L)) continue;
                const Real d = Kokkos::fmax(dx/ell1, dy/ell2);
                if (d < bd) {bd = d; best = q;}
              }
              if (best < 0) return;
              mem[nm++] = best;
              const Real v[3] = {pr(IPVX,best), pr(IPVY,best), pr(IPVZ,best)};
              for (int d=0; d<3; ++d) {
                vlo[d] = Kokkos::fmin(vlo[d], v[d]); vhi[d] = Kokkos::fmax(vhi[d], v[d]);
                if (vhi[d] - vlo[d] > tolv) return;
              }
            }
          }
          for (int k=0; k<9; ++k) {   // claim; roll back on a conflict (retried next pass)
            if (Kokkos::atomic_compare_exchange(&owner(mem[k]), -1, c) != -1) {
              for (int j=0; j<k; ++j) Kokkos::atomic_exchange(&owner(mem[j]), -1);
              return;
            }
          }
          const int knew = Kokkos::atomic_fetch_add(&nnew(0), 1);
          double M = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, svv = 0.0;
          for (int q=0; q<nrdata; ++q) nr(q,knew) = 0.0;
          for (int k=0; k<9; ++k) {
            const int t = mem[k]; const Real m = pr(IPM,t);
            const Real dx = transfer.MinImage1(pr(IPX,t) - xc0), dy = transfer.MinImage2(pr(IPY,t) - yc0);
            M += m; sx += m*dx; sy += m*dy; sxx += m*dx*dx; syy += m*dy*dy;
            svv += m*(pr(IPVX,t)*pr(IPVX,t) + pr(IPVY,t)*pr(IPVY,t) + pr(IPVZ,t)*pr(IPVZ,t));
            for (int q=0; q<nrdata; ++q) nr(q,knew) += m*pr(q,t);
          }
          for (int q=0; q<nrdata; ++q) nr(q,knew) /= M;
          const Real px = SMRCompositeTransfer::WrapPosition(xc0 + sx/M, transfer.x1min, transfer.x1max);
          const Real py = SMRCompositeTransfer::WrapPosition(yc0 + sy/M, transfer.x2min, transfer.x2max);
          nr(IPM,knew) = M;
          nr(IPX,knew) = px; nr(IPX1,knew) = px; nr(IPY,knew) = py; nr(IPY1,knew) = py;
          for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,c);
          ni(PLEV,knew) = L;
          {
            const int fi = SMRCompositeTransfer::WrapIndex(static_cast<int>(
                Kokkos::floor((px - transfer.x1min)/transfer.fine_dx1)), transfer.nfine1);
            const int fj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
                Kokkos::floor((py - transfer.x2min)/transfer.fine_dx2)), transfer.nfine2);
            ni(PGID,knew) = gids + transfer.cloud_target(fj, fi)/ncell12;
          }
          nkind(knew) = 0;
          const double vb2 = nr(IPVX,knew)*nr(IPVX,knew) + nr(IPVY,knew)*nr(IPVY,knew) +
                             nr(IPVZ,knew)*nr(IPVZ,knew);
          for (int k=0; k<9; ++k) pr(IPM,mem[k]) = 0.0;
          Kokkos::atomic_add(&qc(0), 1.0);
          Kokkos::atomic_add(&qc(1), M);
          Kokkos::atomic_add(&qc(2), 0.5*(svv - M*vb2));
          Kokkos::atomic_add(&qc(3), M*vn1 - (sxx - sx*sx/M + M*vo1));
          Kokkos::atomic_add(&qc(4), M*vn2 - (syy - sy*sy/M + M*vo2));
        });
      }
      // ---- stages G and C: the cell walk.  stage 1 (G): pieces at level lev in level-L merge
      // cells -> level-L clouds; stage 2 (C): pieces lighter than the level-lev quantum in
      // level-lev interior cells -> level-lev clouds
      DvceArray1D<int> gid("q_gid", npart), gknew("q_gknew", nmax), gkey("q_gkey", nmax);
      DvceArray1D<int> ngroup("q_ngroup", 1);
      Kokkos::deep_copy(gid, -1);
      Kokkos::deep_copy(gknew, -1);
      const int nfl = 1 << (nlev - lev);   // fine cells per level-lev cell
      for (int stage=1; stage<=2; ++stage) {
        if (stage == 2 && !cloud_compress) break;
        const int tl = (stage == 1) ? L : lev;                  // target level
        const Real cs1 = (stage == 1) ? H1 : ell1, cs2 = (stage == 1) ? H2 : ell2;
        const int m1 = (stage == 1) ? n1 : transfer.nfine1/nfl, m2 = (stage == 1) ? n2 : transfer.nfine2/nfl;
        const int nc = m1*m2;
        const Real smax1 = ((stage == 1) ? cloud_merge_smax : cloud_compress_smax)*cs1;
        const Real smax2 = ((stage == 1) ? cloud_merge_smax : cloud_compress_smax)*cs2;
        const Real cds = cloud_compress_ds;
        const Real vh1 = cs1*cs1/6.0, vh2 = cs2*cs2/6.0;        // hat variance of the target level
        const auto cmask = compress_mask_d;
        const int coff = compress_mask_off[(lev < 5) ? lev : 4];
        const Real qden = static_cast<Real>(1 << (2*tl));
        const int q0 = (stage == 1) ? 5 : 13;                    // counter block
        DvceArray1D<int> kq("q_key", npart);
        Kokkos::parallel_for("q_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int p) {
          kq(p) = -1;
          if (kp(p) < 0 || pr(IPM,p) <= 0.0) return;          // plain level-lev pieces only
          const Real x = SMRCompositeTransfer::WrapPosition(pr(IPX,p), transfer.x1min, transfer.x1max);
          const Real y = SMRCompositeTransfer::WrapPosition(pr(IPY,p), transfer.x2min, transfer.x2max);
          const Real u = (x - transfer.x1min)/cs1, v = (y - transfer.x2min)/cs2;
          const int I = SMRCompositeTransfer::WrapIndex(static_cast<int>(Kokkos::floor(u)), m1);
          const int J = SMRCompositeTransfer::WrapIndex(static_cast<int>(Kokkos::floor(v)), m2);
          if (stage == 1) {
            if (!mm.In(x, y, L)) return;
          } else {
            if (cmask(coff + J*m1 + I) == 0) return;
            if (pr(IPM,p) >= mrootsp[pi(PSP,p)]/qden*(1.0 - 1.0e-9)) return;
          }
          const int si = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(u - Kokkos::floor(u)))));
          const int sj = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(v - Kokkos::floor(v)))));
          const int sub = ((sj >> 1) << 3) | ((si >> 1) << 2) | ((sj & 1) << 1) | (si & 1);   // Morton
          kq(p) = ((pi(PSP,p)*nc + J*m1 + I) << 4) + sub;
        });
        DvceArray1D<int> qs_start, qs_list;
        BuildCSR(kq, nsp*nc*16, qs_start, qs_list);
        Kokkos::parallel_for("q_walk", Kokkos::RangePolicy<>(DevExeSpace(), 0, nsp*nc),
        KOKKOS_LAMBDA(const int kcell) {
          const int base = kcell << 4;
          if (qs_start(base) == qs_start(base + 16)) return;
          const int sp = kcell/nc, cell = kcell % nc, I = cell % m1, J = cell/m1;
          const Real mq = mrootsp[sp]/qden;
          const Real xcc = transfer.x1min + (I + 0.5)*cs1, ycc = transfer.x2min + (J + 0.5)*cs2;
          Real gM[8], gsx[8], gsy[8], gsxx[8], gsyy[8], vlo[8][3], vhi[8][3];
          Real xlo[8], xhi[8], ylo[8], yhi[8], gts[8];
          int gg[8], gfirst[8], gn[8], nopen = 0;
          auto commit = [&](const int j) {
            if (stage == 2) {
              if (gn[j] < 2) return;                              // a single piece stays as it is
              const Real sx = (gsxx[j] - gsx[j]*gsx[j]/gM[j])/gM[j];
              const Real sy = (gsyy[j] - gsy[j]*gsy[j]/gM[j])/gM[j];
              if (sx > cds*vh1 || sy > cds*vh2) {Kokkos::atomic_add(&qc(16), 1.0); return;}
            }
            const int knew = Kokkos::atomic_fetch_add(&nnew(0), 1);
            gknew(gg[j]) = knew;
            for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,gfirst[j]);
            ni(PLEV,knew) = tl;
            nkind(knew) = stage;
            acc(6,knew) = xcc; acc(7,knew) = ycc;
            Kokkos::atomic_add(&qc(q0), 1.0);
            Kokkos::atomic_add(&qc(q0 + 1), static_cast<double>(gn[j]));
            Kokkos::atomic_add(&qc(q0 + 2), static_cast<double>(gM[j]));
          };
          for (int s=0; s<16; ++s) {
            const int sub = rev ? (rot + 16 - s) % 16 : (rot + s) % 16;
            for (int k=qs_start(base + sub); k<qs_start(base + sub + 1); ++k) {
              const int p = qs_list(k);
              const Real m = pr(IPM,p);
              if (m <= 0.0) continue;
              const Real dx = transfer.MinImage1(pr(IPX,p) - xcc), dy = transfer.MinImage2(pr(IPY,p) - ycc);
              const Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
              const Real ts = pr(IPTS,p);
              const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
              const Real tolv = alpha*Kokkos::fmin(cs1, cs2)/tau;
              int j = -1;
              for (int o=0; o<nopen && j<0; ++o) {
                if (gts[o] != ts) continue;
                bool ok = (Kokkos::fmax(xhi[o], dx) - Kokkos::fmin(xlo[o], dx) <= smax1) &&
                          (Kokkos::fmax(yhi[o], dy) - Kokkos::fmin(ylo[o], dy) <= smax2);
                for (int d=0; d<3 && ok; ++d) ok = Kokkos::fmax(vhi[o][d], v[d]) - Kokkos::fmin(vlo[o][d], v[d]) <= tolv;
                if (ok) j = o;
              }
              if (j < 0) {
                if (nopen == K) {Kokkos::atomic_add(&qc(stage == 1 ? 8 : 21), 1.0); continue;}   // deferred
                j = nopen++;
                gg[j] = Kokkos::atomic_fetch_add(&ngroup(0), 1);
                gkey(gg[j]) = kcell;
                gM[j] = gsx[j] = gsy[j] = gsxx[j] = gsyy[j] = 0.0; gn[j] = 0;
                xlo[j] = xhi[j] = dx; ylo[j] = yhi[j] = dy; gts[j] = ts; gfirst[j] = p;
                for (int d=0; d<3; ++d) {vlo[j][d] = vhi[j][d] = v[d];}
              }
              gid(p) = gg[j];
              gM[j] += m; gsx[j] += m*dx; gsy[j] += m*dy; gsxx[j] += m*dx*dx; gsyy[j] += m*dy*dy; ++gn[j];
              xlo[j] = Kokkos::fmin(xlo[j], dx); xhi[j] = Kokkos::fmax(xhi[j], dx);
              ylo[j] = Kokkos::fmin(ylo[j], dy); yhi[j] = Kokkos::fmax(yhi[j], dy);
              for (int d=0; d<3; ++d) {vlo[j][d] = Kokkos::fmin(vlo[j][d], v[d]); vhi[j][d] = Kokkos::fmax(vhi[j][d], v[d]);}
              if (gM[j] >= mq*(1.0 - 1.0e-12)) {
                commit(j);
                --nopen;   // move the last open group into slot j
                if (j != nopen) {
                  gM[j] = gM[nopen]; gsx[j] = gsx[nopen]; gsy[j] = gsy[nopen]; gsxx[j] = gsxx[nopen];
                  gsyy[j] = gsyy[nopen]; xlo[j] = xlo[nopen]; xhi[j] = xhi[nopen]; ylo[j] = ylo[nopen];
                  yhi[j] = yhi[nopen]; gts[j] = gts[nopen]; gg[j] = gg[nopen]; gfirst[j] = gfirst[nopen];
                  gn[j] = gn[nopen];
                  for (int d=0; d<3; ++d) {vlo[j][d] = vlo[nopen][d]; vhi[j][d] = vhi[nopen][d];}
                }
              }
            }
          }
          for (int o=0; o<nopen; ++o) {
            if (gM[o] >= fmin*mq) commit(o);
            if (gknew(gg[o]) < 0) Kokkos::atomic_add(&qc(stage == 1 ? 9 : 20), static_cast<double>(gn[o]));
          }
        });
        // accumulate the members of committed groups into their new clouds; the pieces are used up
        Kokkos::parallel_for("q_accumulate", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int p) {
          if (kq(p) < 0) return;
          const int g = gid(p);
          if (g < 0) return;
          const int k = gknew(g);
          if (k < 0) return;
          const Real m = pr(IPM,p);
          const Real dx = transfer.MinImage1(pr(IPX,p) - acc(6,k)), dy = transfer.MinImage2(pr(IPY,p) - acc(7,k));
          Kokkos::atomic_add(&acc(0,k), static_cast<double>(m));
          Kokkos::atomic_add(&acc(1,k), static_cast<double>(m*dx));
          Kokkos::atomic_add(&acc(2,k), static_cast<double>(m*dy));
          Kokkos::atomic_add(&acc(3,k), static_cast<double>(m*dx*dx));
          Kokkos::atomic_add(&acc(4,k), static_cast<double>(m*dy*dy));
          Kokkos::atomic_add(&acc(5,k), static_cast<double>(m*(pr(IPVX,p)*pr(IPVX,p) +
                             pr(IPVY,p)*pr(IPVY,p) + pr(IPVZ,p)*pr(IPVZ,p))));
          for (int q=0; q<nrdata; ++q) {
            if (q == IPM || q == IPX || q == IPY || q == IPX1 || q == IPY1) continue;
            Kokkos::atomic_add(&nr(q,k), m*pr(q,p));
          }
          pr(IPM,p) = 0.0;
        });
        // a stage's group ids are not reused by the next stage
        Kokkos::deep_copy(gid, -1);
      }
      int ncommit = 0;
      Kokkos::deep_copy(ncommit, Kokkos::subview(nnew, 0));
      // finalize the G and C clouds: means, centroid, dK and dS
      if (ncommit > 0) {
        const Real vG1 = H1*H1/6.0, vG2 = H2*H2/6.0, vF1 = ell1*ell1/6.0, vF2 = ell2*ell2/6.0;
        Kokkos::parallel_for("q_finalize", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
        KOKKOS_LAMBDA(const int k) {
          if (nkind(k) == 0) return;
          const double M = acc(0,k);
          for (int q=0; q<nrdata; ++q) {
            if (q == IPM || q == IPX || q == IPY || q == IPX1 || q == IPY1) continue;
            nr(q,k) /= M;
          }
          const Real px = SMRCompositeTransfer::WrapPosition(acc(6,k) + acc(1,k)/M, transfer.x1min, transfer.x1max);
          const Real py = SMRCompositeTransfer::WrapPosition(acc(7,k) + acc(2,k)/M, transfer.x2min, transfer.x2max);
          nr(IPM,k) = M;
          nr(IPX,k) = px; nr(IPX1,k) = px; nr(IPY,k) = py; nr(IPY1,k) = py;
          const int fi = SMRCompositeTransfer::WrapIndex(static_cast<int>(
              Kokkos::floor((px - transfer.x1min)/transfer.fine_dx1)), transfer.nfine1);
          const int fj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
              Kokkos::floor((py - transfer.x2min)/transfer.fine_dx2)), transfer.nfine2);
          ni(PGID,k) = gids + transfer.cloud_target(fj, fi)/ncell12;
          const double vb2 = nr(IPVX,k)*nr(IPVX,k) + nr(IPVY,k)*nr(IPVY,k) + nr(IPVZ,k)*nr(IPVZ,k);
          const bool g = (nkind(k) == 1);
          const double vn1 = g ? vG1 : vF1, vn2 = g ? vG2 : vF2;
          const int b = g ? 10 : 17;
          Kokkos::atomic_add(&qc(b), 0.5*(acc(5,k) - M*vb2));
          Kokkos::atomic_add(&qc(b + 1), M*vn1 - (acc(3,k) - acc(1,k)*acc(1,k)/M + M*vF1));
          Kokkos::atomic_add(&qc(b + 2), M*vn2 - (acc(4,k) - acc(2,k)*acc(2,k)/M + M*vF2));
        });
      }
      // ---- absorption of G residuals deep in the merge zone: a piece left in an
      // uncommitted group, whose 3x3 level-L cells are all merge cells and with no unreleased lobed
      // cloud within reach, goes to the level-L clouds (existing and new X/G) whose tents cover it
      // and whose velocity is within tol per component; tent-weighted, conserving mass, momentum
      // and the first moment; dK logged
      if (absorb) {
        DvceArray1D<int> nhead("q_nhead", nkey), nnext("q_nnext", ncommit > 0 ? ncommit : 1);
        Kokkos::deep_copy(nhead, -1);
        if (ncommit > 0) {
          Kokkos::parallel_for("q_newlist", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
          KOKKOS_LAMBDA(const int k) {
            if (nkind(k) == 2) return;
            const int key = ni(PSP,k)*ncl + mm.Cell(nr(IPX,k), nr(IPY,k), L);
            nnext(k) = Kokkos::atomic_exchange(&nhead(key), k);
          });
        }
        DvceArray2D<Real> accP("qabs_accP", 7, npart), accN("qabs_accN", 7, ncommit > 0 ? ncommit : 1);
        DvceArray1D<double> ac("qabs_count", 4);
        auto tentc = [=] KOKKOS_FUNCTION (const Real dx, const Real dy) {
          return Kokkos::fmax(0.0, 1.0 - Kokkos::fabs(dx)/H1)*Kokkos::fmax(0.0, 1.0 - Kokkos::fabs(dy)/H2);
        };
        Kokkos::parallel_for("qabs_orphans", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int o) {
          if (kp(o) < 0 || pr(IPM,o) <= 0.0) return;
          const Real xo = pr(IPX,o), yo = pr(IPY,o), mo = pr(IPM,o);
          if (!mm.In(xo, yo, L)) return;
          const int sp = pi(PSP,o), cell = kp(o) - sp*ncl, I = cell % n1, J = cell/n1;
          for (int dj=-1; dj<=1; ++dj) {   // deep: the 3x3 cells are merge cells
            for (int di=-1; di<=1; ++di) {
              const Real xn = transfer.x1min + (SMRCompositeTransfer::WrapIndex(I + di, n1) + 0.5)*H1;
              const Real yn = transfer.x2min + (SMRCompositeTransfer::WrapIndex(J + dj, n2) + 0.5)*H2;
              if (!mm.In(xn, yn, L)) return;
            }
          }
          for (int dj=-2; dj<=2; ++dj) {   // no unreleased lobed cloud within reach
            for (int di=-2; di<=2; ++di) {
              const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                              SMRCompositeTransfer::WrapIndex(I + di, n1);
              for (int k=sl_start(key); k<sl_start(key+1); ++k) {
                const int t = sl_list(k);
                if (Kokkos::fabs(transfer.MinImage1(pr(IPX,t) - xo)) < H1 + ell1 &&
                    Kokkos::fabs(transfer.MinImage2(pr(IPY,t) - yo)) < H2 + ell2) return;
              }
            }
          }
          const Real ts = pr(IPTS,o);
          const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
          const Real tol = alpha*H/tau;
          const Real vo[3] = {pr(IPVX,o), pr(IPVY,o), pr(IPVZ,o)};
          auto compatible = [&](const Real a, const Real b, const Real c, const Real t) {
            return t == ts && Kokkos::fabs(vo[0] - a) <= tol && Kokkos::fabs(vo[1] - b) <= tol &&
                   Kokkos::fabs(vo[2] - c) <= tol;
          };
          Real wsum = 0.0; int nrec = 0;
          for (int ps=0; ps<2; ++ps) {   // 0: weight sum, 1: accumulate
            for (int dj=-1; dj<=1; ++dj) {
              for (int di=-1; di<=1; ++di) {
                const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                                SMRCompositeTransfer::WrapIndex(I + di, n1);
                for (int k=sc_start(key); k<sc_start(key+1); ++k) {
                  const int q = sc_list(k);
                  const Real dx = transfer.MinImage1(xo - pr(IPX,q)), dy = transfer.MinImage2(yo - pr(IPY,q));
                  const Real w = tentc(dx, dy);
                  if (w <= 0.0 || pr(IPM,q) <= 0.0) continue;
                  if (ps == 0) ++nrec;
                  if (!compatible(pr(IPVX,q), pr(IPVY,q), pr(IPVZ,q), pr(IPTS,q))) continue;
                  if (ps == 0) {wsum += w; continue;}
                  const Real dm = w/wsum*mo;
                  Kokkos::atomic_add(&accP(0,q), dm);
                  Kokkos::atomic_add(&accP(1,q), dm*dx); Kokkos::atomic_add(&accP(2,q), dm*dy);
                  Kokkos::atomic_add(&accP(3,q), dm*vo[0]); Kokkos::atomic_add(&accP(4,q), dm*vo[1]);
                  Kokkos::atomic_add(&accP(5,q), dm*vo[2]);
                  Kokkos::atomic_add(&accP(6,q), dm*(vo[0]*vo[0] + vo[1]*vo[1] + vo[2]*vo[2]));
                }
                for (int q=nhead(key); q>=0; q=nnext(q)) {
                  const Real dx = transfer.MinImage1(xo - nr(IPX,q)), dy = transfer.MinImage2(yo - nr(IPY,q));
                  const Real w = tentc(dx, dy);
                  if (w <= 0.0) continue;
                  if (ps == 0) ++nrec;
                  if (!compatible(nr(IPVX,q), nr(IPVY,q), nr(IPVZ,q), nr(IPTS,q))) continue;
                  if (ps == 0) {wsum += w; continue;}
                  const Real dm = w/wsum*mo;
                  Kokkos::atomic_add(&accN(0,q), dm);
                  Kokkos::atomic_add(&accN(1,q), dm*dx); Kokkos::atomic_add(&accN(2,q), dm*dy);
                  Kokkos::atomic_add(&accN(3,q), dm*vo[0]); Kokkos::atomic_add(&accN(4,q), dm*vo[1]);
                  Kokkos::atomic_add(&accN(5,q), dm*vo[2]);
                  Kokkos::atomic_add(&accN(6,q), dm*(vo[0]*vo[0] + vo[1]*vo[1] + vo[2]*vo[2]));
                }
              }
            }
            if (ps == 0 && wsum <= 0.0) {Kokkos::atomic_add(&ac(nrec > 0 ? 2 : 3), 1.0); return;}
          }
          Kokkos::atomic_add(&ac(0), 1.0);
          Kokkos::atomic_add(&ac(1), static_cast<double>(mo));
          pr(IPM,o) = 0.0;
        });
        auto fin = [=] KOKKOS_FUNCTION (const Real M, const Real *a, Real &x, Real &y, Real *v) -> double {
          const Real Mn = M + a[0];
          const double kb = 0.5*M*(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) + 0.5*a[6];
          x = x + a[1]/Mn; y = y + a[2]/Mn;
          for (int q=0; q<3; ++q) v[q] = (M*v[q] + a[3 + q])/Mn;
          return kb - 0.5*Mn*(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        };
        double dkp = 0.0, dkn = 0.0;
        Kokkos::parallel_reduce("qabs_finP", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int q, double &sum) {
          if (accP(0,q) <= 0.0) return;
          Real a[7]; for (int c=0; c<7; ++c) a[c] = accP(c,q);
          Real x = pr(IPX,q), y = pr(IPY,q), v[3] = {pr(IPVX,q), pr(IPVY,q), pr(IPVZ,q)};
          sum += fin(pr(IPM,q), a, x, y, v);
          x = SMRCompositeTransfer::WrapPosition(x, transfer.x1min, transfer.x1max);
          y = SMRCompositeTransfer::WrapPosition(y, transfer.x2min, transfer.x2max);
          pr(IPM,q) += a[0];
          pr(IPX,q) = x; pr(IPX1,q) = x; pr(IPY,q) = y; pr(IPY1,q) = y;
          pr(IPVX,q) = v[0]; pr(IPVY,q) = v[1]; pr(IPVZ,q) = v[2];
          pr(IPVX1,q) = v[0]; pr(IPVY1,q) = v[1]; pr(IPVZ1,q) = v[2];
        }, dkp);
        if (ncommit > 0) {
          Kokkos::parallel_reduce("qabs_finN", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
          KOKKOS_LAMBDA(const int q, double &sum) {
            if (accN(0,q) <= 0.0) return;
            Real a[7]; for (int c=0; c<7; ++c) a[c] = accN(c,q);
            Real x = nr(IPX,q), y = nr(IPY,q), v[3] = {nr(IPVX,q), nr(IPVY,q), nr(IPVZ,q)};
            sum += fin(nr(IPM,q), a, x, y, v);
            x = SMRCompositeTransfer::WrapPosition(x, transfer.x1min, transfer.x1max);
            y = SMRCompositeTransfer::WrapPosition(y, transfer.x2min, transfer.x2max);
            nr(IPM,q) += a[0];
            nr(IPX,q) = x; nr(IPX1,q) = x; nr(IPY,q) = y; nr(IPY1,q) = y;
            nr(IPVX,q) = v[0]; nr(IPVY,q) = v[1]; nr(IPVZ,q) = v[2];
            nr(IPVX1,q) = v[0]; nr(IPVY1,q) = v[1]; nr(IPVZ1,q) = v[2];
          }, dkn);
        }
        auto ach = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ac);
        for (int c=0; c<4; ++c) absc[c] += ach(c);
        absc[4] += dkp + dkn;
      }
      auto qch = Kokkos::create_mirror_view_and_copy(HostMemSpace(), qc);
      for (int c=0; c<24; ++c) qtot[c] += qch(c);
      q_events += static_cast<unsigned long long>(qch(0) + qch(5));
      q_in += static_cast<unsigned long long>(9.0*qch(0) + qch(6));
      // compaction: drop the used-up pieces, append the new clouds
      DvceArray1D<int> koff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_koff"), npart);
      int nkeep = 0;
      Kokkos::parallel_scan("q_kscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
        if (final) koff(p) = upd;
        upd += (pr(IPM,p) > 0.0) ? 1 : 0;
      }, nkeep);
      if (nkeep == npart && ncommit == 0) continue;
      const int ntot = nkeep + ncommit;
      DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, ntot);
      DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, ntot);
      Kokkos::parallel_for("q_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        if (pr(IPM,p) <= 0.0) return;
        const int s = koff(p);
        for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
        for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
      });
      Kokkos::parallel_for("q_append", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
      KOKKOS_LAMBDA(const int k) {
        for (int c=0; c<nrdata; ++c) prn(c,nkeep + k) = nr(c,k);
        for (int c=0; c<nidata; ++c) pin(c,nkeep + k) = ni(c,k);
      });
      ppar->prtcl_rdata = prn;
      ppar->prtcl_idata = pin;
      ppar->nprtcl_thispack = ntot;
      continue;
    }
    auto tent = [=] KOKKOS_FUNCTION (const Real dx, const Real dy) {
      return Kokkos::fmax(0.0, 1.0 - Kokkos::fabs(Snap(dx, ell1))/H1)*
             Kokkos::fmax(0.0, 1.0 - Kokkos::fabs(Snap(dy, ell2))/H2);
    };
    // centres: two rounds of local maxima among unblocked zone pieces
    DvceArray1D<int> state(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_state"), npart);   // 0 none, 1 candidate, 2 centre, 3 blocked
    const Real b1 = H1 - 0.5*ell1, b2 = H2 - 0.5*ell2;
    Kokkos::parallel_for("merge_cand", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      state(p) = 0;
      if (kp(p) < 0 || !mm.In(pr(IPX,p), pr(IPY,p), L)) return;
      const int sp = pi(PSP,p), cell = kp(p) - sp*ncl, I = cell % n1, J = cell/n1;
      for (int dj=-1; dj<=1; ++dj) {
        for (int di=-1; di<=1; ++di) {
          const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                          SMRCompositeTransfer::WrapIndex(I + di, n1);
          for (int k=sc_start(key); k<sc_start(key+1); ++k) {
            const int q = sc_list(k);
            if (Kokkos::fabs(transfer.MinImage1(pr(IPX,p) - pr(IPX,q))) < b1 &&
                Kokkos::fabs(transfer.MinImage2(pr(IPY,p) - pr(IPY,q))) < b2) {state(p) = 3; return;}
          }
        }
      }
      state(p) = 1;
    });
    for (int round=0; round<2; ++round) {
      DvceArray1D<int> win(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_win"), npart);
      Kokkos::parallel_for("merge_mis", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        win(p) = 0;
        if (state(p) != 1) return;
        const int sp = pi(PSP,p), cell = kp(p) - sp*ncl, I = cell % n1, J = cell/n1;
        const Real mp = pr(IPM,p), xp = pr(IPX,p), yp = pr(IPY,p);
        for (int dj=-1; dj<=1; ++dj) {
          for (int di=-1; di<=1; ++di) {
            const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                            SMRCompositeTransfer::WrapIndex(I + di, n1);
            for (int k=sp_start(key); k<sp_start(key+1); ++k) {
              const int q = sp_list(k);
              if (q == p || (state(q) != 1 && state(q) != 2)) continue;
              if (Kokkos::fabs(transfer.MinImage1(xp - pr(IPX,q))) >= b1 ||
                  Kokkos::fabs(transfer.MinImage2(yp - pr(IPY,q))) >= b2) continue;
              if (state(q) == 2) return;   // within the radius of an earlier centre
              const Real mq = pr(IPM,q), xq = pr(IPX,q), yq = pr(IPY,q);
              const bool before = (mq > mp) || (mq == mp && (xq < xp || (xq == xp && yq < yp)));
              if (before) return;
            }
          }
        }
        win(p) = 1;
      });
      Kokkos::parallel_for("merge_mis_set", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {if (win(p)) state(p) = 2;});
    }
    int ncentre = 0;
    Kokkos::parallel_reduce("merge_ncentre", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &sum) {if (state(p) == 2) sum += 1;}, Kokkos::Sum<int>(ncentre));
    if (ncentre == 0 && !absorb) continue;   // with absorption, orphans are looked for in every pass
    // rank of a centre among the centres of its own cell (two centres may share a cell when
    // they are more than the blocking radius apart in one direction): sub-rounds by rank
    DvceArray1D<int> rank(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_rank"), npart);
    int maxrank = 0;
    Kokkos::parallel_reduce("merge_rank", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &mx) {
      rank(p) = -1;
      if (state(p) != 2) return;
      const Real mp = pr(IPM,p), xp = pr(IPX,p), yp = pr(IPY,p);
      int r = 0;
      for (int k=sp_start(kp(p)); k<sp_start(kp(p)+1); ++k) {
        const int q = sp_list(k);
        if (q == p || state(q) != 2) continue;
        const Real mq = pr(IPM,q), xq = pr(IPX,q), yq = pr(IPY,q);
        if ((mq > mp) || (mq == mp && (xq < xp || (xq == xp && yq < yp)))) ++r;
      }
      rank(p) = r;
      if (r > mx) mx = r;
    }, Kokkos::Max<int>(maxrank));
    // new level-L clouds of this pass, and their per-cell linked lists
    DvceArray2D<Real> nr(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_nr"), nrdata, ncentre);
    DvceArray2D<int> ni(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_ni"), nidata, ncentre);
    DvceArray1D<int> nhead("merge_nhead", nkey), nnext("merge_nnext", ncentre);
    DvceArray1D<int> nnew("merge_nnew", 1);
    Kokkos::deep_copy(nhead, -1);
    // D3 diagnostic: per new cloud, the x first-moment mismatch of its contributions in the
    // five coarse cell rows around it (relative to its own position), and its row index
    const Real hcz = transfer.CloudHalfWidth2(L);
    DvceArray2D<Real> dprof("d3_prof", 5, d3 ? ncentre : 1);
    DvceArray1D<int> drow("d3_row", d3 ? ncentre : 1);
    for (int round=0; round<16*(maxrank + 1); ++round) {
      const int col = round % 16, sub = round/16;
      Kokkos::parallel_for("merge_commit", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int c) {
        if (state(c) != 2 || rank(c) != sub) return;
        const int sp = pi(PSP,c), cell = kp(c) - sp*ncl, I = cell % n1, J = cell/n1;
        if ((I % 4) + 4*(J % 4) != col) return;
        if (pr(IPM,c) <= 0.0) return;
        const Real mref = mrootsp[sp]/static_cast<Real>(1 << (2*L)), mnode = 0.25*mref;
        const Real xc0 = pr(IPX,c), yc0 = pr(IPY,c);
        // neighbours with a positive tent weight
        int nb[kNbMax]; int nnb = 0; bool ready = true, over = false; int why = 6;
        for (int dj=-1; dj<=1 && ready; ++dj) {
          for (int di=-1; di<=1 && ready; ++di) {
            const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                            SMRCompositeTransfer::WrapIndex(I + di, n1);
            for (int k=sp_start(key); k<sp_start(key+1); ++k) {
              const int t = sp_list(k);
              if (t == c || pr(IPM,t) <= 0.0) continue;
              const Real dx = transfer.MinImage1(pr(IPX,t) - xc0);
              const Real dy = transfer.MinImage2(pr(IPY,t) - yc0);
              if (Kokkos::fabs(dx) >= H1*(1.0 - 1.0e-12) || Kokkos::fabs(dy) >= H2*(1.0 - 1.0e-12)) continue;
              if (tent(dx, dy) <= 0.0) continue;
              if (!mm.In(pr(IPX,t), pr(IPY,t), L)) {ready = false; break;}
              if (nnb == kNbMax) {over = true; ready = false; why = 8; break;}
              nb[nnb++] = t;
            }
          }
        }
        // lobe wait: an unreleased lobed cloud whose released pieces could fall in this
        // centre's tent makes the centre wait: reach H + ell in both directions
        // (cloud_merge_lobe_wait = 1), or H plus ell only along the cloud's lobes (= 2)
        if (ready && lobe_wait) {
          // the reach (up to H + ell) can span two level-L cells: search +-2 cells
          for (int dj=-2; dj<=2 && ready; ++dj) {
            for (int di=-2; di<=2 && ready; ++di) {
              const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                              SMRCompositeTransfer::WrapIndex(I + di, n1);
              for (int k=sl_start(key); k<sl_start(key+1); ++k) {
                const int t = sl_list(k);
                const int ct = pi(PLEV,t);
                const Real r1 = H1 + (lobe_wait_dir ? (SMRCompositeTransfer::CloudLobe1(ct) ? ell1 : 0.0) : ell1);
                const Real r2 = H2 + (lobe_wait_dir ? (SMRCompositeTransfer::CloudLobe2(ct) ? ell2 : 0.0) : ell2);
                if (Kokkos::fabs(transfer.MinImage1(pr(IPX,t) - xc0)) < r1*(1.0 - 1.0e-12) &&
                    Kokkos::fabs(transfer.MinImage2(pr(IPY,t) - yc0)) < r2*(1.0 - 1.0e-12)) {ready = false; why = 7; break;}
              }
            }
          }
        }
        if (!ready) {Kokkos::atomic_increment(&icnt(3)); Kokkos::atomic_increment(&icnt(why)); return;}
        (void) over;
        // nodes: key (r1, r2) for regular offsets (within 0.1 ell), an own node otherwise
        int nkey1[kNodeMax], nkey2[kNodeMax], nq0[kNodeMax], nnode = 0;
        int pnode[kNbMax + 1], pidx[kNbMax + 1]; int npc = 0;
        auto add_piece = [&](const int t) -> bool {
          const Real dx = transfer.MinImage1(pr(IPX,t) - xc0);
          const Real dy = transfer.MinImage2(pr(IPY,t) - yc0);
          const int r1 = static_cast<int>(Kokkos::round(dx/ell1));
          const int r2 = static_cast<int>(Kokkos::round(dy/ell2));
          const bool reg = Kokkos::fabs(dx - r1*ell1) < 0.1*ell1 && Kokkos::fabs(dy - r2*ell2) < 0.1*ell2;
          int k = -1;
          if (reg) {
            for (int n=0; n<nnode; ++n) if (nkey1[n] == r1 && nkey2[n] == r2 && nq0[n] >= 0) {k = n; break;}
          }
          if (k < 0) {
            if (nnode == kNodeMax) return false;
            k = nnode++;
            nkey1[k] = reg ? r1 : 1000000 + t; nkey2[k] = reg ? r2 : 0; nq0[k] = t;
          }
          pnode[npc] = k; pidx[npc] = t; ++npc;
          return true;
        };
        if (!add_piece(c)) {Kokkos::atomic_increment(&icnt(3)); Kokkos::atomic_increment(&icnt(8)); return;}
        for (int k=0; k<nnb; ++k) if (!add_piece(nb[k])) {Kokkos::atomic_increment(&icnt(3)); Kokkos::atomic_increment(&icnt(8)); return;}
        // node sums, implicit content, shares and effective positions
        Real E[kNodeMax], Ex[kNodeMax], Ey[kNodeMax], Im[kNodeMax], Ix[kNodeMax], Iy[kNodeMax];
        Real share[kNodeMax], ex[kNodeMax], ey[kNodeMax], ox0[kNodeMax], oy0[kNodeMax];
        for (int n=0; n<nnode; ++n) {E[n] = Ex[n] = Ey[n] = 0.0;}
        for (int k=0; k<npc; ++k) {
          const int t = pidx[k], n = pnode[k], q0 = nq0[n];
          E[n] += pr(IPM,t);
          Ex[n] += pr(IPM,t)*transfer.MinImage1(pr(IPX,t) - pr(IPX,q0));
          Ey[n] += pr(IPM,t)*transfer.MinImage2(pr(IPY,t) - pr(IPY,q0));
        }
        for (int n=0; n<nnode; ++n) {
          const int q0 = nq0[n];
          ox0[n] = transfer.MinImage1(pr(IPX,q0) - xc0);
          oy0[n] = transfer.MinImage2(pr(IPY,q0) - yc0);
          Im[n] = Ix[n] = Iy[n] = 0.0;
          const Real xn = pr(IPX,q0), yn = pr(IPY,q0);
          const int ncell = mm.Cell(xn, yn, L), In_ = ncell % n1, Jn = ncell/n1;
          for (int dj=-1; dj<=1; ++dj) {
            for (int di=-1; di<=1; ++di) {
              const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(Jn + dj, n2)*n1 +
                              SMRCompositeTransfer::WrapIndex(In_ + di, n1);
              for (int k=sc_start(key); k<sc_start(key+1); ++k) {
                const int q = sc_list(k);
                const Real dx = transfer.MinImage1(xn - pr(IPX,q)), dy = transfer.MinImage2(yn - pr(IPY,q));
                const Real w = 0.25*pr(IPM,q)*tent(dx, dy);
                if (w <= 0.0) continue;
                Im[n] += w; Ix[n] += w*(Snap(dx, ell1) - dx); Iy[n] += w*(Snap(dy, ell2) - dy);
              }
              for (int q=nhead(key); q>=0; q=nnext(q)) {
                const Real dx = transfer.MinImage1(xn - nr(IPX,q)), dy = transfer.MinImage2(yn - nr(IPY,q));
                const Real w = 0.25*nr(IPM,q)*tent(dx, dy);
                if (w <= 0.0) continue;
                Im[n] += w; Ix[n] += w*(Snap(dx, ell1) - dx); Iy[n] += w*(Snap(dy, ell2) - dy);
              }
            }
          }
          const Real w = tent(ox0[n], oy0[n]);
          share[n] = (E[n] > 0.0) ? Kokkos::fmin(E[n], w*(E[n] + Im[n])) : 0.0;
          if (E[n] > 0.0 && E[n] - share[n] < 1.0e-6*mnode) share[n] = E[n];
          if (E[n] > 0.0) {
            if (nc) {ex[n] = (Ix[n] + Ex[n])/(Im[n] + E[n]); ey[n] = (Iy[n] + Ey[n])/(Im[n] + E[n]);}
            else {ex[n] = Ex[n]/E[n]; ey[n] = Ey[n]/E[n];}
          }
        }
        // contributions and the velocity tolerance (drop the worst until all fit)
        Real cm[kNbMax + 1], cox[kNbMax + 1], coy[kNbMax + 1]; bool act[kNbMax + 1];
        int ncon = 0;
        for (int k=0; k<npc; ++k) {
          const int t = pidx[k], n = pnode[k];
          const Real s = (E[n] > 0.0) ? share[n]*pr(IPM,t)/E[n] : 0.0;
          cm[k] = s; cox[k] = ox0[n] + ex[n]; coy[k] = oy0[n] + ey[n]; act[k] = s > 0.0;
          if (act[k]) ++ncon;
        }
        Real M = 0.0, xc = 0.0, yc = 0.0, vm[3] = {0.0, 0.0, 0.0}, G[3][2];
        bool ok = false;
        while (ncon > 0) {
          M = 0.0; Real mx = 0.0, my = 0.0, mts = 0.0;
          vm[0] = vm[1] = vm[2] = 0.0;
          for (int k=0; k<npc; ++k) {
            if (!act[k]) continue;
            const int s = pidx[k]; const Real m = cm[k];
            M += m; mx += m*cox[k]; my += m*coy[k];
            vm[0] += m*pr(IPVX,s); vm[1] += m*pr(IPVY,s); vm[2] += m*pr(IPVZ,s);
            mts += m*pr(IPTS,s);
          }
          xc = mx/M; yc = my/M;
          for (int q=0; q<3; ++q) vm[q] /= M;
          const Real ts = mts/M;
          const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
          const Real tol = alpha*H/tau;
          for (int q=0; q<3; ++q) {G[q][0] = 0.0; G[q][1] = 0.0;}
          if (ncon >= 5) {
            Real cxx = 0.0, cxy = 0.0, cyy = 0.0, bx[3] = {0.0, 0.0, 0.0}, by[3] = {0.0, 0.0, 0.0};
            for (int k=0; k<npc; ++k) {
              if (!act[k]) continue;
              const int s = pidx[k]; const Real m = cm[k];
              const Real dx = cox[k] - xc, dy = coy[k] - yc;
              const Real v[3] = {pr(IPVX,s) - vm[0], pr(IPVY,s) - vm[1], pr(IPVZ,s) - vm[2]};
              cxx += m*dx*dx; cxy += m*dx*dy; cyy += m*dy*dy;
              for (int q=0; q<3; ++q) {bx[q] += m*dx*v[q]; by[q] += m*dy*v[q];}
            }
            const Real det = cxx*cyy - cxy*cxy;
            if (det > 1.0e-6*M*M*H*H*H*H) {
              for (int q=0; q<3; ++q) {
                G[q][0] = (cyy*bx[q] - cxy*by[q])/det;
                G[q][1] = (cxx*by[q] - cxy*bx[q])/det;
              }
            }
          }
          int worst = -1; Real rmax = 0.0;
          for (int k=0; k<npc; ++k) {
            if (!act[k]) continue;
            const int s = pidx[k];
            const Real dx = cox[k] - xc, dy = coy[k] - yc;
            const Real v[3] = {pr(IPVX,s), pr(IPVY,s), pr(IPVZ,s)};
            Real r2 = 0.0;
            for (int q=0; q<3; ++q) {const Real d = v[q] - vm[q] - G[q][0]*dx - G[q][1]*dy; r2 += d*d;}
            if (r2 > rmax) {rmax = r2; worst = k;}
          }
          if (rmax <= tol*tol) {ok = true; break;}
          if (pidx[worst] == c) break;
          act[worst] = false; --ncon;
        }
        if (!ok || M < fmin*mref) {
          Kokkos::atomic_increment(&icnt(2)); Kokkos::atomic_increment(&icnt(ok ? 10 : 9)); return;
        }
        // commit
        const int knew = Kokkos::atomic_fetch_add(&nnew(0), 1);
        for (int q=0; q<nrdata; ++q) nr(q,knew) = 0.0;
        Real dkg = 0.0, dkr = 0.0; int nin = 0;
        for (int k=0; k<npc; ++k) {
          if (!act[k]) continue;
          const int s = pidx[k]; const Real m = cm[k];
          for (int q=0; q<nrdata; ++q) nr(q,knew) += m*pr(q,s);
          const Real dx = cox[k] - xc, dy = coy[k] - yc;
          const Real v[3] = {pr(IPVX,s) - vm[0], pr(IPVY,s) - vm[1], pr(IPVZ,s) - vm[2]};
          for (int q=0; q<3; ++q) {
            const Real g = G[q][0]*dx + G[q][1]*dy;
            dkg += 0.5*m*g*g; dkr += 0.5*m*(v[q] - g)*(v[q] - g);
          }
          ++nin;
        }
        for (int q=0; q<nrdata; ++q) nr(q,knew) /= M;
        const Real px = SMRCompositeTransfer::WrapPosition(xc0 + xc, transfer.x1min, transfer.x1max);
        const Real py = SMRCompositeTransfer::WrapPosition(yc0 + yc, transfer.x2min, transfer.x2max);
        nr(IPM,knew) = M;
        nr(IPX,knew) = px; nr(IPX1,knew) = px; nr(IPY,knew) = py; nr(IPY1,knew) = py;
        nr(IPVX,knew) = vm[0]; nr(IPVY,knew) = vm[1]; nr(IPVZ,knew) = vm[2];
        if (d3) {
          const int mz = static_cast<int>(Kokkos::floor((py - transfer.x2min)/H2));
          drow(knew) = mz;
          for (int j=0; j<5; ++j) dprof(j,knew) = 0.0;
          for (int k=0; k<npc; ++k) {
            if (!act[k]) continue;
            const int t = pidx[k];
            const Real dxp = transfer.MinImage1(pr(IPX,t) - px);
            const Real zt = py + transfer.MinImage2(pr(IPY,t) - py);
            for (int j=0; j<5; ++j) {
              const Real a = transfer.x2min + (mz - 2 + j)*H2;
              dprof(j,knew) -= cm[k]*dxp*HatFrac(zt, ell2, a, a + H2);
            }
          }
        }
        for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,c);
        ni(PLEV,knew) = L;
        {
          const int fi = SMRCompositeTransfer::WrapIndex(static_cast<int>(
              Kokkos::floor((px - transfer.x1min)/transfer.fine_dx1)), transfer.nfine1);
          const int fj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
              Kokkos::floor((py - transfer.x2min)/transfer.fine_dx2)), transfer.nfine2);
          ni(PGID,knew) = gids + transfer.cloud_target(fj, fi)/ncell12;
        }
        const int key = sp*ncl + mm.Cell(px, py, L);
        nnext(knew) = Kokkos::atomic_exchange(&nhead(key), knew);
        // take the shares; the node convention shifts what is left of each node
        Real taken[kNodeMax];
        for (int n=0; n<nnode; ++n) taken[n] = 0.0;
        for (int k=0; k<npc; ++k) {
          if (!act[k]) continue;
          const int s = pidx[k];
          taken[pnode[k]] += cm[k];
          pr(IPM,s) -= cm[k];
          if (s == c || pr(IPM,s) <= 0.0) pr(IPM,s) = 0.0;
        }
        long long nsh = 0, nsk = 0;
        if (nc) {
          for (int n=0; n<nnode; ++n) {
            const Real rem = E[n] - taken[n];
            if (rem <= 1.0e-6*mnode || taken[n] <= 0.0) continue;
            const int q0 = nq0[n];
            const Real dx = transfer.MinImage1(pr(IPX,q0) - px), dy = transfer.MinImage2(pr(IPY,q0) - py);
            const Real In = 0.25*M*tent(dx, dy);
            const Real xr = (ex[n]*(Im[n] + In + rem) - Ix[n] - In*(Snap(dx, ell1) - dx))/rem;
            const Real yr = (ey[n]*(Im[n] + In + rem) - Iy[n] - In*(Snap(dy, ell2) - dy))/rem;
            const Real sx = xr - Ex[n]/E[n], sy = yr - Ey[n]/E[n];
            if (Kokkos::fabs(sx) >= 0.1*ell1 || Kokkos::fabs(sy) >= 0.1*ell2) {++nsk; continue;}
            for (int k=0; k<npc; ++k) {
              if (pnode[k] != n) continue;
              const int t = pidx[k];
              if (pr(IPM,t) <= 0.0) continue;
              const Real nx = SMRCompositeTransfer::WrapPosition(pr(IPX,t) + sx, transfer.x1min, transfer.x1max);
              const Real ny = SMRCompositeTransfer::WrapPosition(pr(IPY,t) + sy, transfer.x2min, transfer.x2max);
              pr(IPX,t) = nx; pr(IPX1,t) = nx; pr(IPY,t) = ny; pr(IPY1,t) = ny;
              const int fi = SMRCompositeTransfer::WrapIndex(static_cast<int>(
                  Kokkos::floor((nx - transfer.x1min)/transfer.fine_dx1)), transfer.nfine1);
              const int fj = SMRCompositeTransfer::WrapIndex(static_cast<int>(
                  Kokkos::floor((ny - transfer.x2min)/transfer.fine_dx2)), transfer.nfine2);
              pi(PGID,t) = gids + transfer.cloud_target(fj, fi)/ncell12;
            }
            ++nsh;
          }
        }
        Kokkos::atomic_increment(&icnt(0));
        Kokkos::atomic_add(&icnt(1), static_cast<long long>(nin));
        Kokkos::atomic_add(&icnt(4), nsh);
        Kokkos::atomic_add(&icnt(5), nsk);
        Kokkos::atomic_add(&dk(0), dkg);
        Kokkos::atomic_add(&dk(1), dkr);
      });
    }
    // compaction: drop the used-up pieces, append the new level-L clouds
    int ncommit = 0;
    Kokkos::deep_copy(ncommit, Kokkos::subview(nnew, 0));
    if (d3 && ncommit > 0) {
      // Jacobi least squares for x shifts dX of this pass's new clouds: in each coarse cell row
      // of a column (new clouds within H1/2 in x), the deposit x first moment of the new clouds
      // matches that of the shares they took.  Previously committed clouds stay fixed.
      DvceArray1D<Real> dx0("d3_dx0", ncommit), dxa("d3_dxa", ncommit), dxb("d3_dxb", ncommit);
      DvceArray1D<double> ssum("d3_ssum", 2);
      auto rowres = [=] KOKKOS_FUNCTION (const int k, const int m, const DvceArray1D<Real> &dX) -> Real {
        const Real xk = nr(IPX,k), zk = nr(IPY,k);
        const int sp = ni(PSP,k), cell = mm.Cell(xk, zk, L), I = cell % n1, J = cell/n1;
        Real r = 0.0;
        for (int dj=-3; dj<=3; ++dj) {
          for (int di=-1; di<=1; ++di) {
            const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                            SMRCompositeTransfer::WrapIndex(I + di, n1);
            for (int q=nhead(key); q>=0; q=nnext(q)) {
              if (Kokkos::fabs(transfer.MinImage1(nr(IPX,q) - xk)) >= 0.5*H1) continue;
              const Real zq = zk + transfer.MinImage2(nr(IPY,q) - zk);
              const int mzq = static_cast<int>(Kokkos::floor((zq - transfer.x2min)/H2));
              const int j = m - mzq + 2;
              if (j < 0 || j > 4) continue;
              const Real a = transfer.x2min + m*H2;
              r += dprof(j,q) + nr(IPM,q)*HatFrac(zq, hcz, a, a + H2)*dX(q);
            }
          }
        }
        return r;
      };
      auto rows_of = [=] KOKKOS_FUNCTION (const int k) -> int {
        return static_cast<int>(Kokkos::floor((nr(IPY,k) - transfer.x2min)/H2));
      };
      Kokkos::parallel_for("d3_s0", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
      KOKKOS_LAMBDA(const int k) {
        dxa(k) = 0.0;
        const int mz = rows_of(k);
        double s0 = 0.0;
        for (int m=mz-2; m<=mz+2; ++m) {const Real r = rowres(k, m, dx0); s0 += r*r;}
        Kokkos::atomic_add(&ssum(0), s0);
      });
      for (int it=0; it<d3_iter; ++it) {
        auto &src = (it % 2 == 0) ? dxa : dxb;
        auto &dst = (it % 2 == 0) ? dxb : dxa;
        Kokkos::parallel_for("d3_jacobi", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
        KOKKOS_LAMBDA(const int k) {
          const Real zk = nr(IPY,k), Mk = nr(IPM,k);
          const int mz = rows_of(k);
          Real num = 0.0, den = 0.0;
          for (int m=mz-1; m<=mz+1; ++m) {
            const Real a = transfer.x2min + m*H2;
            const Real w = Mk*HatFrac(zk, hcz, a, a + H2);
            if (w <= 0.0) continue;
            num += w*rowres(k, m, src); den += w*w;
          }
          dst(k) = (den > 0.0) ? src(k) - 0.7*num/den : src(k);
        });
      }
      auto &fin = (d3_iter % 2 == 0) ? dxa : dxb;
      Kokkos::parallel_for("d3_s1", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
      KOKKOS_LAMBDA(const int k) {
        const int mz = rows_of(k);
        double s1 = 0.0;
        for (int m=mz-2; m<=mz+2; ++m) {const Real r = rowres(k, m, fin); s1 += r*r;}
        Kokkos::atomic_add(&ssum(1), s1);
      });
      Kokkos::parallel_for("d3_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
      KOKKOS_LAMBDA(const int k) {
        const Real nx = SMRCompositeTransfer::WrapPosition(nr(IPX,k) + fin(k), transfer.x1min, transfer.x1max);
        nr(IPX,k) = nx; nr(IPX1,k) = nx;
      });
      auto sh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ssum);
      d3s[0] += sh(0); d3s[1] += sh(1);
    }
    if (absorb) {
      // orphan absorption: a plain level-lev piece in a merge cell that is blocked
      // (can never be a centre) and lies in no tent of a live candidate or centre is given to
      // the velocity-compatible level-L clouds (existing and new) whose tents cover it, with
      // P1 tent weights; mass, momentum and the first moment are conserved, dK is logged
      DvceArray2D<Real> accP("abs_accP", 7, npart), accN("abs_accN", 7, ncommit > 0 ? ncommit : 1);
      DvceArray1D<double> ac("abs_count", 4);
      auto tentc = [=] KOKKOS_FUNCTION (const Real dx, const Real dy) {
        return Kokkos::fmax(0.0, 1.0 - Kokkos::fabs(dx)/H1)*Kokkos::fmax(0.0, 1.0 - Kokkos::fabs(dy)/H2);
      };
      // blocked after this pass's commits: a level-L cloud (existing or new) within H - ell/2
      DvceArray1D<int> blk("abs_blocked", npart);
      Kokkos::parallel_for("abs_blocked", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        blk(p) = 0;
        if (kp(p) < 0 || pr(IPM,p) <= 0.0) return;
        const Real xp = pr(IPX,p), yp = pr(IPY,p);
        const int sp = pi(PSP,p), cell = kp(p) - sp*ncl, I = cell % n1, J = cell/n1;
        for (int dj=-1; dj<=1; ++dj) {
          for (int di=-1; di<=1; ++di) {
            const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                            SMRCompositeTransfer::WrapIndex(I + di, n1);
            for (int k=sc_start(key); k<sc_start(key+1); ++k) {
              const int q = sc_list(k);
              if (Kokkos::fabs(transfer.MinImage1(xp - pr(IPX,q))) < b1 &&
                  Kokkos::fabs(transfer.MinImage2(yp - pr(IPY,q))) < b2) {blk(p) = 1; return;}
            }
            for (int q=nhead(key); q>=0; q=nnext(q)) {
              if (Kokkos::fabs(transfer.MinImage1(xp - nr(IPX,q))) < b1 &&
                  Kokkos::fabs(transfer.MinImage2(yp - nr(IPY,q))) < b2) {blk(p) = 1; return;}
            }
          }
        }
      });
      Kokkos::parallel_for("abs_orphans", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int o) {
        // an orphan: blocked, in a merge cell, no unblocked piece (a possible future centre) in its
        // tent and no unreleased lobed cloud whose pieces could reach it
        if (kp(o) < 0 || blk(o) == 0 || pr(IPM,o) <= 0.0) return;
        const Real xo = pr(IPX,o), yo = pr(IPY,o), mo = pr(IPM,o);
        if (!mm.In(xo, yo, L)) return;
        const int sp = pi(PSP,o), cell = kp(o) - sp*ncl, I = cell % n1, J = cell/n1;
        for (int dj=-2; dj<=2; ++dj) {
          for (int di=-2; di<=2; ++di) {
            const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                            SMRCompositeTransfer::WrapIndex(I + di, n1);
            if (dj >= -1 && dj <= 1 && di >= -1 && di <= 1) {
              for (int k=sp_start(key); k<sp_start(key+1); ++k) {
                const int q = sp_list(k);
                if (q == o || pr(IPM,q) <= 0.0 || blk(q) != 0) continue;
                if (Kokkos::fabs(transfer.MinImage1(pr(IPX,q) - xo)) < H1 &&
                    Kokkos::fabs(transfer.MinImage2(pr(IPY,q) - yo)) < H2) return;   // still reachable
              }
            }
            for (int k=sl_start(key); k<sl_start(key+1); ++k) {
              const int t = sl_list(k);
              if (Kokkos::fabs(transfer.MinImage1(pr(IPX,t) - xo)) < H1 + ell1 &&
                  Kokkos::fabs(transfer.MinImage2(pr(IPY,t) - yo)) < H2 + ell2) return;   // lobe pending
            }
          }
        }
        const Real ts = pr(IPTS,o);
        const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
        const Real tol = alpha*H/tau;
        const Real vo[3] = {pr(IPVX,o), pr(IPVY,o), pr(IPVZ,o)};
        auto compatible = [&](const Real a, const Real b, const Real c) {
          const Real d0 = vo[0] - a, d1 = vo[1] - b, d2 = vo[2] - c;
          return d0*d0 + d1*d1 + d2*d2 <= tol*tol;
        };
        Real wsum = 0.0; int nrec = 0;
        for (int pass=0; pass<2; ++pass) {   // 0: weight sum, 1: accumulate
          for (int dj=-1; dj<=1; ++dj) {
            for (int di=-1; di<=1; ++di) {
              const int key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                              SMRCompositeTransfer::WrapIndex(I + di, n1);
              for (int k=sc_start(key); k<sc_start(key+1); ++k) {
                const int q = sc_list(k);
                const Real dx = transfer.MinImage1(xo - pr(IPX,q)), dy = transfer.MinImage2(yo - pr(IPY,q));
                const Real w = tentc(dx, dy);
                if (w <= 0.0 || pr(IPM,q) <= 0.0) continue;
                if (pass == 0) ++nrec;
                if (!compatible(pr(IPVX,q), pr(IPVY,q), pr(IPVZ,q))) continue;
                if (pass == 0) {wsum += w; continue;}
                const Real dm = w/wsum*mo;
                Kokkos::atomic_add(&accP(0,q), dm);
                Kokkos::atomic_add(&accP(1,q), dm*dx); Kokkos::atomic_add(&accP(2,q), dm*dy);
                Kokkos::atomic_add(&accP(3,q), dm*vo[0]); Kokkos::atomic_add(&accP(4,q), dm*vo[1]);
                Kokkos::atomic_add(&accP(5,q), dm*vo[2]);
                Kokkos::atomic_add(&accP(6,q), dm*(vo[0]*vo[0] + vo[1]*vo[1] + vo[2]*vo[2]));
              }
              for (int q=nhead(key); q>=0; q=nnext(q)) {
                if (ni(PSP,q) != sp) continue;
                const Real dx = transfer.MinImage1(xo - nr(IPX,q)), dy = transfer.MinImage2(yo - nr(IPY,q));
                const Real w = tentc(dx, dy);
                if (w <= 0.0) continue;
                if (pass == 0) ++nrec;
                if (!compatible(nr(IPVX,q), nr(IPVY,q), nr(IPVZ,q))) continue;
                if (pass == 0) {wsum += w; continue;}
                const Real dm = w/wsum*mo;
                Kokkos::atomic_add(&accN(0,q), dm);
                Kokkos::atomic_add(&accN(1,q), dm*dx); Kokkos::atomic_add(&accN(2,q), dm*dy);
                Kokkos::atomic_add(&accN(3,q), dm*vo[0]); Kokkos::atomic_add(&accN(4,q), dm*vo[1]);
                Kokkos::atomic_add(&accN(5,q), dm*vo[2]);
                Kokkos::atomic_add(&accN(6,q), dm*(vo[0]*vo[0] + vo[1]*vo[1] + vo[2]*vo[2]));
              }
            }
          }
          if (pass == 0 && wsum <= 0.0) {
            Kokkos::atomic_add(&ac(nrec > 0 ? 2 : 3), 1.0);
            return;
          }
        }
        Kokkos::atomic_add(&ac(0), 1.0);
        Kokkos::atomic_add(&ac(1), static_cast<double>(mo));
        pr(IPM,o) = 0.0;
      });
      auto fin = [=] KOKKOS_FUNCTION (const Real M, const Real *a, Real &x, Real &y, Real *v) -> double {
        const Real Mn = M + a[0];
        const double kb = 0.5*M*(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) + 0.5*a[6];
        x = x + a[1]/Mn; y = y + a[2]/Mn;
        for (int q=0; q<3; ++q) v[q] = (M*v[q] + a[3 + q])/Mn;
        return kb - 0.5*Mn*(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
      };
      double dkp = 0.0, dkn = 0.0;
      Kokkos::parallel_reduce("abs_finP", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int q, double &sum) {
        if (accP(0,q) <= 0.0) return;
        Real a[7]; for (int c=0; c<7; ++c) a[c] = accP(c,q);
        Real x = pr(IPX,q), y = pr(IPY,q), v[3] = {pr(IPVX,q), pr(IPVY,q), pr(IPVZ,q)};
        sum += fin(pr(IPM,q), a, x, y, v);
        x = SMRCompositeTransfer::WrapPosition(x, transfer.x1min, transfer.x1max);
        y = SMRCompositeTransfer::WrapPosition(y, transfer.x2min, transfer.x2max);
        pr(IPM,q) += a[0];
        pr(IPX,q) = x; pr(IPX1,q) = x; pr(IPY,q) = y; pr(IPY1,q) = y;
        pr(IPVX,q) = v[0]; pr(IPVY,q) = v[1]; pr(IPVZ,q) = v[2];
        pr(IPVX1,q) = v[0]; pr(IPVY1,q) = v[1]; pr(IPVZ1,q) = v[2];
      }, dkp);
      if (ncommit > 0) {
        Kokkos::parallel_reduce("abs_finN", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
        KOKKOS_LAMBDA(const int q, double &sum) {
          if (accN(0,q) <= 0.0) return;
          Real a[7]; for (int c=0; c<7; ++c) a[c] = accN(c,q);
          Real x = nr(IPX,q), y = nr(IPY,q), v[3] = {nr(IPVX,q), nr(IPVY,q), nr(IPVZ,q)};
          sum += fin(nr(IPM,q), a, x, y, v);
          x = SMRCompositeTransfer::WrapPosition(x, transfer.x1min, transfer.x1max);
          y = SMRCompositeTransfer::WrapPosition(y, transfer.x2min, transfer.x2max);
          nr(IPM,q) += a[0];
          nr(IPX,q) = x; nr(IPX1,q) = x; nr(IPY,q) = y; nr(IPY1,q) = y;
          nr(IPVX,q) = v[0]; nr(IPVY,q) = v[1]; nr(IPVZ,q) = v[2];
          nr(IPVX1,q) = v[0]; nr(IPVY1,q) = v[1]; nr(IPVZ1,q) = v[2];
        }, dkn);
      }
      auto ach = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ac);
      for (int c=0; c<4; ++c) absc[c] += ach(c);
      absc[4] += dkp + dkn;
      if (ncommit == 0 && ach(0) == 0.0) continue;   // nothing changed at this level: no compaction
    }
    DvceArray1D<int> koff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_koff"), npart);
    int nkeep = 0;
    Kokkos::parallel_scan("merge_kscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
      if (final) koff(p) = upd;
      upd += (pr(IPM,p) > 0.0) ? 1 : 0;
    }, nkeep);
    const int ntot = nkeep + ncommit;
    DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, ntot);
    DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, ntot);
    Kokkos::parallel_for("merge_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (pr(IPM,p) <= 0.0) return;
      const int s = koff(p);
      for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
      for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
    });
    Kokkos::parallel_for("merge_append", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
    KOKKOS_LAMBDA(const int k) {
      for (int c=0; c<nrdata; ++c) prn(c,nkeep + k) = nr(c,k);
      for (int c=0; c<nidata; ++c) pin(c,nkeep + k) = ni(c,k);
    });
    ppar->prtcl_rdata = prn;
    ppar->prtcl_idata = pin;
    ppar->nprtcl_thispack = ntot;
  }

  // totals after, mass by cloud type, counters; the log line of the host version
  const int nfinal = ppar->nprtcl_thispack;
  Real ta[4] = {0.0, 0.0, 0.0, 0.0}, ml[4] = {0.0, 0.0, 0.0, 0.0};
  {
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    Kokkos::parallel_reduce("merge_tot_a", Kokkos::RangePolicy<>(DevExeSpace(), 0, nfinal),
    KOKKOS_LAMBDA(const int p, Real &a, Real &b, Real &c, Real &d, Real &l0, Real &l1, Real &l2,
                  Real &lob) {
      const Real m = pr(IPM,p);
      a += m; b += m*pr(IPVX,p); c += m*pr(IPVY,p); d += m*pr(IPVZ,p);
      const int code = pi(PLEV,p);
      const int l = SMRCompositeTransfer::CloudLevel(code);
      if (l == 0) l0 += m; else if (l == 1) l1 += m; else l2 += m;
      if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) lob += m;
    }, ta[0], ta[1], ta[2], ta[3], ml[0], ml[1], ml[2], ml[3]);
  }
  pmesh->nprtcl_thisrank = nfinal;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nfinal;
  pmesh->nprtcl_total = nfinal;
  auto ic = Kokkos::create_mirror_view_and_copy(HostMemSpace(), icnt);
  auto dkh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), dk);
  const unsigned long long nevents = ic(0) + q_events, nin = ic(1) + q_in, nreject = ic(2), npending = ic(3);
  ++cloud_merge_pass;
  const unsigned long long nc_shift = ic(4), nc_skip = ic(5);
  cloud_coarsen_events += nevents;
  cloud_coarsen_in += nin;
  cloud_coarsen_reject += nreject;
  cloud_far_release += nrelease;
  cloud_coarsen_dk_grad += dkh(0);
  cloud_coarsen_dk_res += dkh(1);
  std::FILE *fp = std::fopen("cloud_merge_log.txt", "a");
  if (fp != nullptr) {
    std::fseek(fp, 0, SEEK_END);
    if (std::ftell(fp) == 0) {
      std::fprintf(fp, "# ncycle time npart_before npart_after far_releases commits"
                   " contributions rejected(tolerance/min mass) pending dK_grad dK_res M_b M_a"
                   " PX_b PX_a PY_b PY_a PZ_b PZ_a m_level0 m_level1 m_level2 m_lobed"
                   " nc_shifts nc_skipped (device)\n");
    }
    std::fprintf(fp, "%d %.17g %d %d %d %llu %llu %llu %llu %.9e %.9e %.17g %.17g %.17g"
                 " %.17g %.17g %.17g %.17g %.17g %.9e %.9e %.9e %.9e %llu %llu", pmesh->ncycle,
                 pmesh->time, npart0, nfinal, nrelease, nevents, nin, nreject, npending, dkh(0),
                 dkh(1), tb[0], ta[0], tb[1], ta[1], tb[2], ta[2], tb[3], ta[3], ml[0], ml[1],
                 ml[2], ml[3], nc_shift, nc_skip);
    // D3 diagnostic columns appended to the line: row first-moment mismatch^2 before / after
    if (d3) std::fprintf(fp, " %.9e %.9e", d3s[0], d3s[1]);
    // orphan absorption columns: absorbed n, absorbed mass, incompatible, no recipient, dK
    if (absorb) std::fprintf(fp, " %.0f %.9e %.0f %.0f %.9e", absc[0], absc[1], absc[2], absc[3], absc[4]);
    // pending-reason columns: not ready (mask), lobe wait, over limit, tolerance, min mass
    // quantum-mode columns: X families, mass, dK, dSx, dSy; G groups, pieces, mass,
    // deferred, residual pieces, dK, dSx, dSy; C groups, pieces, mass, rejected (dS), dK, dSx, dSy,
    // residual pieces, deferred
    if (cloud_merge_mode == 1) {
      for (int c=0; c<22; ++c) std::fprintf(fp, " %.9e", qtot[c]);
    }
    if (cloud_merge_pending_log) std::fprintf(fp, " %llu %llu %llu %llu %llu", static_cast<unsigned long long>(ic(6)),
        static_cast<unsigned long long>(ic(7)), static_cast<unsigned long long>(ic(8)),
        static_cast<unsigned long long>(ic(9)), static_cast<unsigned long long>(ic(10)));
    std::fprintf(fp, "\n");
    std::fclose(fp);
  }
  (void) sh1; (void) sh2;
  return static_cast<int>(nevents);
}

}  // namespace dust
