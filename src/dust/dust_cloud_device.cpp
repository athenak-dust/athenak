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

#include <iostream>

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
  DvceArray1D<int> np("dust_split_np", npart);
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
  DvceArray1D<int> keepoff("dust_split_keepoff", npart);
  DvceArray1D<int> pieceoff("dust_split_pieceoff", npart);
  DvceArray1D<int> tagoff("dust_split_tagoff", npart);
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
  DvceArray2D<Real> prn("prtcl_rdata", nrdata, nnew);
  DvceArray2D<int> pin("prtcl_idata", nidata, nnew);
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

}  // namespace dust
