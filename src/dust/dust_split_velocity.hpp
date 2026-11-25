#ifndef DUST_DUST_SPLIT_VELOCITY_HPP_
#define DUST_DUST_SPLIT_VELOCITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_split_velocity.hpp
//! \brief The velocities of the pieces of a P1 split or lobe release with
//! <dust>/cloud_split_velocity = mesh_gradient (dust_split_velocity.cpp): the helpers the
//! fill kernels of the device splits call.

#include "athena.hpp"

namespace dust {

// the centroid of a cloud along one direction relative to its position: a transition lobe
// (1 lower, 2 upper) is the shape (2/3) S_ell(x - x_p) + (1/3) S_ell(x - x_p -/+ ell), whose
// centroid is -/+ ell/3; no lobe and both lobes (the hat of half-width 2 ell) are symmetric
KOKKOS_INLINE_FUNCTION
Real SplitLobeShift(const int lobe, const Real ell) {
  return (lobe == 2) ? ell/3.0 : ((lobe == 1) ? -ell/3.0 : 0.0);
}

// piece velocity: v += G d for the event of row `row` of g (G_ij at g(row, 3 i + j), the
// parent's mean-flow gradient, zero where the split falls back to the parent's velocity);
// d the offset of the piece's centroid from the parent's; zsign = -1 for a piece mirrored at a
// reflecting x3 face (its vz was flipped).  Both the velocity and its RK register.
template <class PR, class GV>
KOKKOS_INLINE_FUNCTION
void SplitAddGradient(const PR &pr, const int s, const GV &g, const int row, const Real d0,
                      const Real d1, const Real d2, const Real zsign) {
  const Real dvx = g(row,0)*d0 + g(row,1)*d1 + g(row,2)*d2;
  const Real dvy = g(row,3)*d0 + g(row,4)*d1 + g(row,5)*d2;
  const Real dvz = zsign*(g(row,6)*d0 + g(row,7)*d1 + g(row,8)*d2);
  pr(IPVX,s) += dvx; pr(IPVX1,s) += dvx;
  pr(IPVY,s) += dvy; pr(IPVY1,s) += dvy;
  pr(IPVZ,s) += dvz; pr(IPVZ1,s) += dvz;
}

//----------------------------------------------------------------------------------------
// the margin of the range bound and the cloud count of the diagnostic's count verdict (constants:
// test_split_gradient_range_margin, _range_kind, _nmin, _count and the count rule
// retired)
constexpr Real kSplitGBMargin = 0.5;
constexpr int kSplitGBNmin = 8;
// (test) <dust>/test_split_gradient_bound = none | range (dust_split_velocity.cpp):
// the extra columns of a split_vgrad row (from kGBCol), written by the fit when the key is given
constexpr int kGBCol = 25;
enum : int {
  kGBUmin = 0,     // [3] min over the populated support cells of (cell-mean v - v_parent)
  kGBUmax = 3,     // [3] max
  kGBNmin = 6,     // the fewest clouds depositing into a populated support cell
  kGBNmean = 7,    // the fit-weighted mean of the clouds per cell
  kGBNeff = 8,     // the effective number of cells, (sum a)^2 / sum a^2
  kGBLmin = 9,     // the smallest eigenvalue of E^-1 A E^-1 (A the centroid covariance)
  kGBWmax = 10,    // the largest weight fraction a_i / sum a
  kGBSt0 = 11,     // the fit's status before the count rule (1 applied)
  kGBVG = 12,      // V_G of the fit (before any bound)
  kGBCntOff = 13,  // 1: the count rule switches it off (applied in count mode)
  kGBS = 14,       // the range factor s (written by the fill)
  kGBNcell = 15,   // the populated support cells
  kGBG = 16,       // [9] G of the fit before any bound
  kGBNum = 25
};
// device counters per split call
enum : int {
  kGCEvents = 0, kGCFit = 1, kGCCountV = 2, kGCCountA = 3, kGCRangeV = 4, kGCRangeA = 5,
  kGCRange0 = 6, kGCRangeHalf = 7, kGCBoth = 8, kGCSmp = 9,
  kGCRHist = 10,   // [12] R = V_G / dv: < 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 30, 100, >= 100
  kGCNHist = 22,   // [12] nmin: < 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64, >= 64
  kGCSHist = 34,   // [6] s: 0, (0, 0.25), [0.25, 0.5), [0.5, 0.75), [0.75, 1), 1
  kGCRange0A = 40,   // the range bound applied with s = 0 (kGCRange0: the verdict, every mode)
  kGCMHist = 41,   // [12] nmean, the bins of nmin
  kGCNum = 53
};
// device maxima per split call: R, V_G, |G|, max |G d| of the fit, max |dv| of the pieces applied
enum : int {kGMR = 0, kGMVG = 1, kGMG = 2, kGMDvRaw = 3, kGMDvApp = 4, kGMNum = 5};
constexpr int kGSmpRows = 32;   // samples per call (events with R above the sample threshold)
constexpr int kGSmpCols = 40;

// (test) the range bound of one event in the fill kernel, before its pieces are written:
// dk[j][i] the centroid offset of piece index i along direction j (nk[j] pieces).  With the
// fit's G (raw, before any bound) the largest and smallest increment (G d)_c over the pieces
// (separable: a sum over the directions of the extremes along each); the range factor s is the
// largest value in [0, 1] that keeps v_parent + s (G d)_c within [lo_c, hi_c] for every piece and
// component, lo/hi the range of the support cells' mean velocities widened by margin times its
// span (margin kSplitGBMargin) and extended to v_parent itself (s = 0, the parent's velocity, is
// always inside).  mode 1
// (range): G and V_G of the row scaled by s (uniformly: the pieces keep the parent's momentum).
// full (test_split_gradient_diag: the statistics kernel wrote every extra
// column from kGBCol): the fit's own status and G from those columns, the verdict in every mode,
// counters, maxima and samples for the diagnostic.  Otherwise (the default range bound): an
// applied fit is one with V_G > 0 in the row, its G is the row's, the range of the cells' mean
// velocities is at column ucol; only the bounded events and those with s = 0 are counted.
// The pieces' offsets dk come from the count pass's separation flags (as the fill's).
template <class GV, class CV, class MV, class SV>
KOKKOS_INLINE_FUNCTION
void SplitGradientBound(const GV &g, const int row, const int mode,
                        const bool full, const int ucol, const Real rsmp,
                        const Real dk[3][3], const int nk[3], const Real x, const Real y,
                        const Real z, const CV &cnt, const MV &mx, const SV &smp) {
  if (full ? (g(row, kGBCol + kGBSt0) != 1.0) : !(g(row,13) > 0.0)) return;
  constexpr Real margin = kSplitGBMargin;
  Real gr[9];
  for (int n=0; n<9; ++n) gr[n] = full ? g(row, kGBCol + kGBG + n) : g(row,n);
  Real s = 1.0, dv2 = 0.0;
  for (int c=0; c<3; ++c) {
    Real ihi = 0.0, ilo = 0.0;
    for (int j=0; j<3; ++j) {
      Real a = gr[3*c + j]*dk[j][0], b = a;
      for (int i=1; i<nk[j]; ++i) {
        const Real v = gr[3*c + j]*dk[j][i];
        a = fmax(a, v); b = fmin(b, v);
      }
      ihi += a; ilo += b;
    }
    const Real umin = g(row, ucol + kGBUmin + c), umax = g(row, ucol + kGBUmax + c);
    const Real span = umax - umin;
    dv2 += span*span;
    const Real hi = fmax(0.0, umax + margin*span), lo = fmin(0.0, umin - margin*span);
    if (ihi > 0.0) s = fmin(s, hi/ihi);
    if (ilo < 0.0) s = fmin(s, lo/ilo);
  }
  s = fmax(0.0, fmin(1.0, s));
  // a factor at round-off (a range edge at v_parent, u = O(1e-18)) is 0: the parent's velocity
  // (otherwise the check's |sum m dv| / (M s V_G) divides round-off by round-off)
  if (s < 1.0e-12) s = 0.0;
  if (!full) {   // the default range bound: scale, count the bounded events
    if (s < 1.0) {
      for (int n=0; n<9; ++n) g(row,n) *= s;
      g(row,13) *= s;
      Kokkos::atomic_add(&cnt(kGCRangeA), 1ULL);
      if (s <= 0.0) Kokkos::atomic_add(&cnt(kGCRange0A), 1ULL);
    }
    return;
  }
  // the largest |G d| over the pieces (the fit's G)
  Real dmax2 = 0.0;
  for (int a=0; a<nk[0]; ++a) {
    for (int b=0; b<nk[1]; ++b) {
      for (int c=0; c<nk[2]; ++c) {
        const Real d0 = dk[0][a], d1 = dk[1][b], d2 = dk[2][c];
        const Real u = gr[0]*d0 + gr[1]*d1 + gr[2]*d2, v = gr[3]*d0 + gr[4]*d1 + gr[5]*d2;
        const Real w = gr[6]*d0 + gr[7]*d1 + gr[8]*d2;
        dmax2 = fmax(dmax2, u*u + v*v + w*w);
      }
    }
  }
  const Real dmax = sqrt(dmax2);
  const bool coff = g(row, kGBCol + kGBCntOff) > 0.5;
  const bool bound = s < 1.0;
  g(row, kGBCol + kGBS) = s;
  if (mode == 1 && bound) {
    for (int n=0; n<9; ++n) g(row,n) *= s;
    g(row,13) *= s;
  }
  // the pieces' largest velocity change as applied (none: the fit's; count: 0 where switched off)
  const Real dapp = (mode == 1) ? s*dmax : ((mode == 2 && coff) ? 0.0 : dmax);
  if (bound) Kokkos::atomic_add(&cnt(kGCRangeV), 1ULL);
  if (bound && mode == 1) Kokkos::atomic_add(&cnt(kGCRangeA), 1ULL);
  if (s <= 0.0) Kokkos::atomic_add(&cnt(kGCRange0), 1ULL);
  if (s <= 0.0 && mode == 1) Kokkos::atomic_add(&cnt(kGCRange0A), 1ULL);
  if (s < 0.5) Kokkos::atomic_add(&cnt(kGCRangeHalf), 1ULL);
  if (bound && coff) Kokkos::atomic_add(&cnt(kGCBoth), 1ULL);
  const int sb = (s <= 0.0) ? 0 : ((s < 0.25) ? 1 : ((s < 0.5) ? 2 : ((s < 0.75) ? 3 :
                                                                      ((s < 1.0) ? 4 : 5))));
  Kokkos::atomic_add(&cnt(kGCSHist + sb), 1ULL);
  Kokkos::atomic_max(&mx(kGMDvRaw), static_cast<double>(dmax));
  Kokkos::atomic_max(&mx(kGMDvApp), static_cast<double>(dapp));
  // a sample of the events whose V_G exceeds rsmp times the range of the cell means
  const Real dv = sqrt(dv2), vg = g(row, kGBCol + kGBVG);
  const Real r = (dv > 0.0) ? vg/dv : 1.0e30;
  if (rsmp > 0.0 && r > rsmp) {
    const unsigned long long k = Kokkos::atomic_fetch_add(&cnt(kGCSmp), 1ULL);
    if (k < static_cast<unsigned long long>(kGSmpRows)) {
      const int q = static_cast<int>(k);
      smp(q,0) = x; smp(q,1) = y; smp(q,2) = z;
      smp(q,3) = r; smp(q,4) = vg; smp(q,5) = dv; smp(q,6) = dmax; smp(q,7) = s;
      smp(q,8) = g(row, kGBCol + kGBNcell); smp(q,9) = g(row, kGBCol + kGBNmin);
      smp(q,10) = g(row, kGBCol + kGBNmean); smp(q,11) = g(row, kGBCol + kGBNeff);
      smp(q,12) = g(row, kGBCol + kGBLmin); smp(q,13) = g(row, kGBCol + kGBWmax);
      smp(q,14) = coff ? 1.0 : 0.0; smp(q,15) = g(row,12);
      for (int c=0; c<3; ++c) {
        smp(q,16 + c) = g(row, kGBCol + kGBUmin + c);
        smp(q,19 + c) = g(row, kGBCol + kGBUmax + c);
        smp(q,22 + c) = g(row, 9 + c);
      }
      for (int n=0; n<9; ++n) smp(q,25 + n) = gr[n];
      smp(q,34) = nk[0]*nk[1]*nk[2];
    }
  }
}

}  // namespace dust

#endif  // DUST_DUST_SPLIT_VELOCITY_HPP_
