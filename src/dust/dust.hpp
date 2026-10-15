#ifndef DUST_DUST_HPP_
#define DUST_DUST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust.hpp
//! \brief definitions for DustGasDrag class: Lagrangian dust particles coupled to a
//! Hydro gas by stiff mutual drag.  The module supports the imex2+ IMEX(4,3,2) method
//! (Krapp et al. 2024), plus an RK2 hybrid of resolved PC2 midpoint coupling and a stiff
//! full-step split-BE fallback.  Both use the particle-mesh scatter/gather method of
//! Yang & Johansen (2016) with exact momentum-conserving back-reaction (PMBR), and the
//! O(N) closed-form per-cell drag solve of Benitez-Llambay, Krapp & Pessah (2019)
//! generalized to per-particle stopping times.
//!
//! This module couples the Particles and Hydro modules the same way IonNeutral couples
//! Hydro and MHD: it owns the combined task graph, the deposited fields, and the drag
//! kernels, while particle storage/migration stays in Particles and fluxes/BCs in Hydro.

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#include "coordinates/cell_locations.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class Driver;
class MeshBlockPack;
class ShearingBoxCC;

// constants that enumerate particle-mesh deposit schemes
enum class DustDeposit {ngp=0, cic=1, tsc=2};

// contract for per-particle stopping times.  The default species_fixed mode keeps the
// configured species table authoritative; particle_static permits arbitrary positive
// per-particle values that remain constant; dynamic refreshes their global maximum once
// per cycle.
enum class DustStoppingTimeMode {species_fixed=0, particle_static=1, dynamic=2};

// `local` is the regression baseline. `applya` executes one non-mutating coupled
// operator application and then retains the local answer, for marginal-cost timing.
// `dc1` and `dc2` unconditionally accept one and two fixed defect corrections.
enum class DustDragSolver {local=0, applya=1, dc1=2, dc2=3, pcg=4, adaptive=5};

// Time coupling between the hydro RK update and particle drag. Input coupling=pc2 maps
// to the hybrid task path with PC2 forced for every cycle. Input coupling=hybrid can use
// PC2 while drag and feedback are resolved and split backward-Euler otherwise.
enum class DustCoupling {imex=0, hybrid=1};
enum class HybridMode {pc2=0, split_be=1};
enum class HybridForceMode {automatic=0, pc2=1, split_be=2};

// Experimental transfer choices on the narrowly supported static-SMR path. The
// owner_overlap choice is retained only as a negative control. refinable_cloud gives
// every particle its own hat cloud: weights are exact integrals of
// the cloud over active leaf cells, and clouds change size only by the exact B-spline
// two-scale split (see DustGasDrag::SplitAndMergeClouds).
enum class DustSMRTransfer {none=0, owner_overlap=1, smooth_p2=2, refinable_cloud=3};

// Device-copyable description of the smooth composite transfer. Each virtual coarse
// or fine cell stores a short linear map to active leaf-cell degrees of freedom. The
// same ForEachWeight() traversal is consumed by every gather and deposit operation.
// In cloud_mode the struct instead holds the finest-level virtual grid of the
// refinable-cloud transfer, and ForEachWeight(x1,x2,lev,f) dispatches to it.
struct SMRCompositeTransfer {
  // Largest 1-D cloud stencil on the finest virtual grid: 2*2^(nlev-lev)+1 cells, so
  // nine cells allow a root cloud to sit on leaves up to two levels finer.
  static constexpr int kMaxCloudStencil = 9;
  static constexpr int kCloudWeightCache = 25;   // leaf weights a kernel keeps from its gather

  // Cloud code stored in PLEV: bits 0-3 the cloud level, bits 4-5 (x2) and 6-7 (x1) a
  // transition lobe (0 none, 1 lower, 2 upper, 3 both); bit 8 is set only on gather calls and
  // evaluates the leaf reconstructions at the particle along lobed directions.  A
  // transition cloud (<dust>/cloud_transition) is the rigid union of a
  // level-lev child and the unpaired edge child next to it: along a lobed direction
  // its shape is (2/3) S_ell(x - x_p) + (1/3) S_ell(x - x_p -/+ ell), the P1 (FEM)
  // transition hat with half-widths ell and 2 ell, peaked at the particle.
  static constexpr int kCloudLevelMask = 15;
  static constexpr int kCloudLobeShift2 = 4;
  static constexpr int kCloudLobeShift1 = 6;
  static constexpr int kCloudGatherPoint = 256;
  KOKKOS_INLINE_FUNCTION
  static int CloudLevel(const int code) {return code & kCloudLevelMask;}
  KOKKOS_INLINE_FUNCTION
  static int CloudLobe1(const int code) {return (code >> kCloudLobeShift1) & 3;}
  KOKKOS_INLINE_FUNCTION
  static int CloudLobe2(const int code) {return (code >> kCloudLobeShift2) & 3;}
  // Code handed to gathers: transition=1 gathers with the plain child hat, transition=2
  // with the lobed weights and the point value of each leaf's linear reconstruction.
  KOKKOS_INLINE_FUNCTION
  static int GatherCode(const int code, const int transition) {
    if (transition == 1) return code & kCloudLevelMask;
    if (transition == 2) return code | kCloudGatherPoint;
    return code;
  }

  bool enabled = false;
  bool cloud_mode = false;
  int cloud_nlev = 0;                 // max_level - root_level
  int cloud_max_level = 0;            // clouds are never split beyond this level
  bool cloud_split_everywhere = false; // control: split regardless of the local mesh
  bool cloud_split_overlap = false;    // split once the (shrunk) support touches finer leaves
                                       // instead of once it lies entirely on them
  bool lobe3_integral = false;         // diagnostic (<dust>/cloud_presplit_x1): a lobe-3
                                       // direction (parent width) of a transition-2 gather
                                       // takes the cloud integral like a plain cloud
  Real root_dx1 = 0.0, root_dx2 = 0.0;
  Real fine_dx1 = 0.0, fine_dx2 = 0.0;
  DvceArray2D<int> cloud_target;      // finest virtual cell -> encoded active leaf
  DvceArray2D<int> cloud_level;       // finest virtual cell -> owner level - root_level
  // Level-native traversal (<dust>/cloud_native_transfer): for
  // L = 0..cloud_nlev, the encoded leaf of every level-L cell that is itself an active
  // leaf (-1 otherwise), row-major from native_offset[L] with rows of native_n1[L].
  static constexpr int kMaxNativeLevels = 3;
  bool native = false;
  int native_n1[kMaxNativeLevels] = {0, 0, 0};
  int native_n2[kMaxNativeLevels] = {0, 0, 0};
  int native_offset[kMaxNativeLevels] = {0, 0, 0};
  DvceArray1D<int> native_target;
  bool measure_correction = true;
  int ncoarse1 = 0, ncoarse2 = 0;
  int nfine1 = 0, nfine2 = 0;
  int nfine_blocks = 0;
  int correction_tiles = 0;
  Real x1min = 0.0, x1max = 0.0;
  Real x2min = 0.0, x2max = 0.0;
  Real transition_width = 0.0;
  Real interface_x1[2] = {0.0, 0.0};
  Real interface_coarse_sign[2] = {0.0, 0.0};
  DvceArray2D<int> coarse_count;
  DvceArray3D<int> coarse_target;
  DvceArray3D<Real> coarse_weight;
  DvceArray2D<int> fine_count;
  DvceArray3D<int> fine_target;
  DvceArray3D<Real> fine_weight;
  DvceArray2D<Real> fine_block_bounds;
  DvceArray1D<int> correction_count;
  DvceArray2D<int> correction_target;
  DvceArray2D<Real> correction_weight;

  KOKKOS_INLINE_FUNCTION
  static int WrapIndex(const int index, const int extent) {
    int wrapped = index % extent;
    return (wrapped < 0) ? wrapped + extent : wrapped;
  }

  KOKKOS_INLINE_FUNCTION
  static Real WrapPosition(const Real x, const Real xmin, const Real xmax) {
    Real length = xmax - xmin;
    Real wrapped = x;
    while (wrapped < xmin) wrapped += length;
    while (wrapped >= xmax) wrapped -= length;
    return wrapped;
  }

  KOKKOS_INLINE_FUNCTION
  static void VirtualTSC(const Real x, const Real xmin, const Real xmax, const int ncells,
                         int index[3], Real weight[3]) {
    Real dx = (xmax - xmin)/static_cast<Real>(ncells);
    Real coordinate = (x - xmin)/dx - 0.5;
    int nearest = static_cast<int>(floor(coordinate + 0.5));
    Real offset = coordinate - static_cast<Real>(nearest);
    index[0] = WrapIndex(nearest - 1, ncells);
    index[1] = WrapIndex(nearest,     ncells);
    index[2] = WrapIndex(nearest + 1, ncells);
    weight[0] = 0.5*SQR(0.5 - offset);
    weight[1] = 0.75 - SQR(offset);
    weight[2] = 0.5*SQR(0.5 + offset);
  }

  // Cumulative distribution of the unit hat cloud S(s) = max(0, 1-|s|).
  KOKKOS_INLINE_FUNCTION
  static Real HatCDF(const Real s) {
    if (s <= -1.0) return 0.0;
    if (s >= 1.0) return 1.0;
    return (s <= 0.0) ? 0.5*SQR(1.0 + s) : 1.0 - 0.5*SQR(1.0 - s);
  }

  // Exact 1-D integrals of a hat cloud of half-width ell centred at x over the
  // consecutive virtual cells [i0, i0+count) of spacing dx.  Consecutive CDF
  // differences telescope, so the weights sum to one for any cell partition.
  // A nonzero lobe (1 lower, 2 upper) gives the transition shape
  // (2/3) S_ell(x - x_p) + (1/3) S_ell(x - x_p -/+ ell) instead.
  // Lobe 3 (both edge children attached, a parent left unsplit along this direction by
  // the moving P1 split) is the hat of half-width 2 ell by the two-scale identity.
  KOKKOS_INLINE_FUNCTION
  static int CloudWeights1D(const Real x, const Real ell_in, const Real xmin, const Real dx,
                            int &i0, Real w[kMaxCloudStencil], const int lobe_in = 0) {
    const Real ell = (lobe_in == 3) ? 2.0*ell_in : ell_in;
    const int lobe = (lobe_in == 3) ? 0 : lobe_in;
    Real inv_ell = 1.0/ell;
    Real sl = (lobe == 1) ? -ell : ell;
    Real lo = (lobe == 1) ? x - 2.0*ell : x - ell;
    Real hi = (lobe == 2) ? x + 2.0*ell : x + ell;
    i0 = static_cast<int>(floor((lo - xmin)/dx));
    int i1 = static_cast<int>(ceil((hi - xmin)/dx));
    int count = i1 - i0;
    if (count > kMaxCloudStencil) count = kMaxCloudStencil;
    Real prev = HatCDF((xmin + i0*dx - x)*inv_ell);
    Real prevl = lobe ? HatCDF((xmin + i0*dx - x - sl)*inv_ell) : 0.0;
    for (int a=0; a<count; ++a) {
      Real next = HatCDF((xmin + (i0 + a + 1)*dx - x)*inv_ell);
      w[a] = next - prev;
      prev = next;
      if (lobe) {
        Real nextl = HatCDF((xmin + (i0 + a + 1)*dx - x - sl)*inv_ell);
        w[a] = (2.0/3.0)*w[a] + (1.0/3.0)*(nextl - prevl);
        prevl = nextl;
      }
    }
    return count;
  }

  // Antiderivative of s*(1-|s|), the first moment of the unit hat, clamped outside
  // [-1,1] so that differences give the moment of the hat over any interval.
  KOKKOS_INLINE_FUNCTION
  static Real HatMoment(const Real s) {
    if (s <= -1.0 || s >= 1.0) return 1.0/6.0;
    return (s <= 0.0) ? 0.5*SQR(s) + SQR(s)*s/3.0 : 0.5*SQR(s) - SQR(s)*s/3.0;
  }

  // CloudWeights1D plus the first moment g[a] = int_cell S(x) (x - x_particle) dx of the
  // unit-mass hat over each virtual cell (about the particle position, length units).
  // With a lobe the moments are those of the transition shape about x_p; point=true
  // (lobed directions of a cloud_transition=2 gather) returns zero moments, so that
  // the gather takes the value of each leaf's reconstruction at the particle.
  KOKKOS_INLINE_FUNCTION
  static int CloudWeightsMoments1D(const Real x, const Real ell_in, const Real xmin,
                                   const Real dx, int &i0, Real w[kMaxCloudStencil],
                                   Real g[kMaxCloudStencil], const int lobe_in = 0,
                                   const bool point = false) {
    const bool both = (lobe_in == 3);
    const Real ell = both ? 2.0*ell_in : ell_in;
    const int lobe = both ? 0 : lobe_in;
    Real inv_ell = 1.0/ell;
    Real sl = (lobe == 1) ? -ell : ell;
    Real lo = (lobe == 1) ? x - 2.0*ell : x - ell;
    Real hi = (lobe == 2) ? x + 2.0*ell : x + ell;
    i0 = static_cast<int>(floor((lo - xmin)/dx));
    int i1 = static_cast<int>(ceil((hi - xmin)/dx));
    int count = i1 - i0;
    if (count > kMaxCloudStencil) count = kMaxCloudStencil;
    Real s = (xmin + i0*dx - x)*inv_ell;
    Real prev = HatCDF(s), gprev = HatMoment(s);
    Real t = lobe ? (xmin + i0*dx - x - sl)*inv_ell : 0.0;
    Real prevl = HatCDF(t), gprevl = HatMoment(t);
    for (int a=0; a<count; ++a) {
      s = (xmin + (i0 + a + 1)*dx - x)*inv_ell;
      Real next = HatCDF(s), gnext = HatMoment(s);
      w[a] = next - prev;
      g[a] = (both && point) ? 0.0 : ell*(gnext - gprev);
      prev = next;
      gprev = gnext;
      if (lobe) {
        t = (xmin + (i0 + a + 1)*dx - x - sl)*inv_ell;
        Real nextl = HatCDF(t), gnextl = HatMoment(t);
        Real wl = nextl - prevl;
        w[a] = (2.0/3.0)*w[a] + (1.0/3.0)*wl;
        g[a] = point ? 0.0 : (2.0/3.0)*g[a] + (1.0/3.0)*(ell*(gnextl - gprevl) + sl*wl);
        prevl = nextl;
        gprevl = gnextl;
      }
    }
    return count;
  }

  // Minimum-image displacement in a periodic direction.
  KOKKOS_INLINE_FUNCTION
  static Real MinImage(const Real d, const Real length) {
    return d - length*round(d/length);
  }
  KOKKOS_INLINE_FUNCTION
  Real MinImage1(const Real d) const {return MinImage(d, x1max - x1min);}
  KOKKOS_INLINE_FUNCTION
  Real MinImage2(const Real d) const {return MinImage(d, x2max - x2min);}

  KOKKOS_INLINE_FUNCTION
  Real CloudHalfWidth1(const int lev) const {
    return root_dx1/static_cast<Real>(1 << (lev & kCloudLevelMask));
  }
  KOKKOS_INLINE_FUNCTION
  Real CloudHalfWidth2(const int lev) const {
    return root_dx2/static_cast<Real>(1 << (lev & kCloudLevelMask));
  }

  // Encoded leaf of the level-l cell (j, i) (indices wrapped), or -1 when that cell is
  // not an active leaf of level l.
  KOKKOS_INLINE_FUNCTION
  int NativeTarget(const int l, const int j, const int i) const {
    return native_target(native_offset[l] + WrapIndex(j, native_n2[l])*native_n1[l] +
                         WrapIndex(i, native_n1[l]));
  }

  // CloudWeightsMoments1D (kMoments) or CloudWeights1D on at most three cells, with the
  // loop unrolled so the arrays stay in registers.  Returns the cell count; a count above
  // three leaves w and g unset (the caller falls back to the general traversal).
  template <bool kMoments>
  KOKKOS_INLINE_FUNCTION
  static int CloudWeights3(const Real x, const Real ell_in, const Real xmin, const Real dx,
                           int &i0, Real w[3], Real g[3], const int lobe_in,
                           const bool point) {
    const bool both = (lobe_in == 3);
    const Real ell = both ? 2.0*ell_in : ell_in;
    const int lobe = both ? 0 : lobe_in;
    Real inv_ell = 1.0/ell;
    Real sl = (lobe == 1) ? -ell : ell;
    Real lo = (lobe == 1) ? x - 2.0*ell : x - ell;
    Real hi = (lobe == 2) ? x + 2.0*ell : x + ell;
    i0 = static_cast<int>(floor((lo - xmin)/dx));
    int count = static_cast<int>(ceil((hi - xmin)/dx)) - i0;
    if (count > 3) return count;
    Real s = (xmin + i0*dx - x)*inv_ell;
    Real prev = HatCDF(s), gprev = kMoments ? HatMoment(s) : 0.0;
    Real t = lobe ? (xmin + i0*dx - x - sl)*inv_ell : 0.0;
    Real prevl = HatCDF(t), gprevl = kMoments ? HatMoment(t) : 0.0;
#pragma unroll
    for (int a=0; a<3; ++a) {
      w[a] = 0.0;
      g[a] = 0.0;
      if (a >= count) continue;
      s = (xmin + (i0 + a + 1)*dx - x)*inv_ell;
      Real next = HatCDF(s);
      w[a] = next - prev;
      prev = next;
      if constexpr (kMoments) {
        Real gnext = HatMoment(s);
        g[a] = (both && point) ? 0.0 : ell*(gnext - gprev);
        gprev = gnext;
      }
      if (lobe) {
        t = (xmin + (i0 + a + 1)*dx - x - sl)*inv_ell;
        Real nextl = HatCDF(t);
        Real wl = nextl - prevl;
        w[a] = (2.0/3.0)*w[a] + (1.0/3.0)*wl;
        prevl = nextl;
        if constexpr (kMoments) {
          Real gnextl = HatMoment(t);
          g[a] = point ? 0.0 : (2.0/3.0)*g[a] + (1.0/3.0)*(ell*(gnextl - gprevl) + sl*wl);
          gprevl = gnextl;
        }
      }
    }
    return count;
  }

  // Level-native traversal.  When the support of the cloud `code` at the wrapped position
  // (x1, x2) covers at most 3x3 cells of the level L of the leaf holding the particle and
  // every one of them is an active leaf of level L, integrate the cloud over those leaves
  // directly, one call per leaf, and return true.  This is the canonical TSC footprint of a
  // plain level-L cloud; a finer cloud on coarser leaves covers at most 2x2.  A cloud's
  // integral over a leaf is the sum of its integrals over the leaf's virtual finest
  // sub-cells, so the weights and first moments equal those of the finest-grid traversal
  // up to rounding.  Returns false, without calling `function`, otherwise (support across
  // levels, or a lobed cloud wider than three cells).
  template <bool kMoments, typename Function>
  KOKKOS_INLINE_FUNCTION
  bool NativeTraversal(const Real x1, const Real x2, const int code,
                       const Function &function) const {
    int l = 0;
    for (; l<=cloud_nlev; ++l) {
      Real d1 = root_dx1/static_cast<Real>(1 << l), d2 = root_dx2/static_cast<Real>(1 << l);
      if (NativeTarget(l, static_cast<int>(floor((x2 - x2min)/d2)),
                       static_cast<int>(floor((x1 - x1min)/d1))) >= 0) break;
    }
    if (l > cloud_nlev) return false;
    Real d1 = root_dx1/static_cast<Real>(1 << l), d2 = root_dx2/static_cast<Real>(1 << l);
    bool point = kMoments && (code & kCloudGatherPoint) != 0;
    int i0, j0;
    Real w1[3], w2[3], g1[3], g2[3];
    int n1 = CloudWeights3<kMoments>(x1, CloudHalfWidth1(code), x1min, d1, i0, w1, g1,
                                     CloudLobe1(code),
                                     point && !(lobe3_integral && CloudLobe1(code) == 3));
    if (n1 > 3) return false;
    int n2 = CloudWeights3<kMoments>(x2, CloudHalfWidth2(code), x2min, d2, j0, w2, g2,
                                     CloudLobe2(code),
                                     point && !(lobe3_integral && CloudLobe2(code) == 3));
    if (n2 > 3) return false;
    int target[3][3];
#pragma unroll
    for (int b=0; b<3; ++b) {
#pragma unroll
      for (int a=0; a<3; ++a) {
        target[b][a] = -1;
        if (a >= n1 || b >= n2) continue;
        target[b][a] = NativeTarget(l, j0 + b, i0 + a);
        if (target[b][a] < 0) return false;
      }
    }
#pragma unroll
    for (int b=0; b<3; ++b) {
#pragma unroll
      for (int a=0; a<3; ++a) {
        if (a >= n1 || b >= n2) continue;
        Real w = w2[b]*w1[a];
        if constexpr (kMoments) {
          Real m1x = w2[b]*g1[a], m1y = w1[a]*g2[b];
          if (w == 0.0 && m1x == 0.0 && m1y == 0.0) continue;
          function(target[b][a], w, m1x, m1y);
        } else {
          if (w == 0.0) continue;
          function(target[b][a], w);
        }
      }
    }
    return true;
  }

  // Cloud-integrated weights on active leaves: the virtual finest cells overlapped by
  // the support are integrated exactly and forwarded to their owning leaf, so a
  // coarse leaf receives the sum of its sub-cell integrals (its own exact integral).
  // With `native`, a support on leaves of one level takes the level-native traversal.
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachCloudWeight(const Real particle_x1, const Real particle_x2, const int lev,
                          const Function &function) const {
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
    if (native && NativeTraversal<false>(x1, x2, lev, function)) return;
    int i0, j0;
    Real w1[kMaxCloudStencil], w2[kMaxCloudStencil];
    int n1 = CloudWeights1D(x1, CloudHalfWidth1(lev), x1min, fine_dx1, i0, w1,
                            CloudLobe1(lev));
    int n2 = CloudWeights1D(x2, CloudHalfWidth2(lev), x2min, fine_dx2, j0, w2,
                            CloudLobe2(lev));
    for (int b=0; b<n2; ++b) {
      if (w2[b] == 0.0) continue;
      int fj = WrapIndex(j0 + b, nfine2);
      for (int a=0; a<n1; ++a) {
        Real w = w2[b]*w1[a];
        if (w == 0.0) continue;
        function(cloud_target(fj, WrapIndex(i0 + a, nfine1)), w);
      }
    }
  }

  // True when the support of a level-lev cloud, shrunk by slack*ell on each side, lies
  // entirely on leaves finer than lev, i.e. the particle may be split.  With
  // cloud_split_overlap it suffices that the shrunk support touches a finer leaf, so the
  // finer leaves only ever see split clouds and the size transition sits on coarser cells.
  KOKKOS_INLINE_FUNCTION
  bool CloudInsideFiner(const Real particle_x1, const Real particle_x2, const int lev,
                        const Real slack) const {
    if (CloudLobe1(lev) || CloudLobe2(lev)) return false;   // transition clouds stay
    if (lev >= cloud_nlev || lev >= cloud_max_level) return false;
    if (cloud_split_everywhere) return true;
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
    Real r1 = (1.0 - slack)*CloudHalfWidth1(lev);
    Real r2 = (1.0 - slack)*CloudHalfWidth2(lev);
    int ia = static_cast<int>(floor((x1 - r1 - x1min)/fine_dx1));
    int ib = static_cast<int>(ceil((x1 + r1 - x1min)/fine_dx1)) - 1;
    int ja = static_cast<int>(floor((x2 - r2 - x2min)/fine_dx2));
    int jb = static_cast<int>(ceil((x2 + r2 - x2min)/fine_dx2)) - 1;
    if (cloud_split_overlap) {
      for (int fj=ja; fj<=jb; ++fj) {
        int wj = WrapIndex(fj, nfine2);
        for (int fi=ia; fi<=ib; ++fi) {
          if (cloud_level(wj, WrapIndex(fi, nfine1)) > lev) return true;
        }
      }
      return false;
    }
    for (int fj=ja; fj<=jb; ++fj) {
      int wj = WrapIndex(fj, nfine2);
      for (int fi=ia; fi<=ib; ++fi) {
        if (cloud_level(wj, WrapIndex(fi, nfine1)) <= lev) return false;
      }
    }
    return true;
  }

  // Cloud traversal that also hands the callback the first moments of the cloud over
  // the virtual cell about the particle position: function(target, w, m1x, m1y).  A
  // gather of a linearly reconstructed leaf field u_c + s.(x - x_c) is then
  // sum(w*u_c + (m1x + (x_p - x_c)*w)*s_x + (m1y + (y_p - y_c)*w)*s_y), exact for a
  // linear field and second-order accurate for any cloud/leaf size combination.
  // With `native`, a support on leaves of one level takes the level-native traversal.
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachCloudMoment(const Real particle_x1, const Real particle_x2, const int lev,
                          const Function &function) const {
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
    if (native && NativeTraversal<true>(x1, x2, lev, function)) return;
    int i0, j0;
    Real w1[kMaxCloudStencil], w2[kMaxCloudStencil];
    Real g1[kMaxCloudStencil], g2[kMaxCloudStencil];
    bool point = (lev & kCloudGatherPoint) != 0;
    bool point1 = point && !(lobe3_integral && CloudLobe1(lev) == 3);
    bool point2 = point && !(lobe3_integral && CloudLobe2(lev) == 3);
    int n1 = CloudWeightsMoments1D(x1, CloudHalfWidth1(lev), x1min, fine_dx1, i0, w1, g1,
                                   CloudLobe1(lev), point1);
    int n2 = CloudWeightsMoments1D(x2, CloudHalfWidth2(lev), x2min, fine_dx2, j0, w2, g2,
                                   CloudLobe2(lev), point2);
    for (int b=0; b<n2; ++b) {
      if (w2[b] == 0.0 && g2[b] == 0.0) continue;
      int fj = WrapIndex(j0 + b, nfine2);
      for (int a=0; a<n1; ++a) {
        Real w = w2[b]*w1[a];
        Real m1x = w2[b]*g1[a];
        Real m1y = w1[a]*g2[b];
        if (w == 0.0 && m1x == 0.0 && m1y == 0.0) continue;
        function(cloud_target(fj, WrapIndex(i0 + a, nfine1)), w, m1x, m1y);
      }
    }
  }

  // True if any finest virtual cell under the support of a level-lev cloud belongs to
  // a leaf finer than lev (the cloud overlaps a finer region, split or not).
  KOKKOS_INLINE_FUNCTION
  bool CloudOverlapsFiner(const Real particle_x1, const Real particle_x2,
                          const int code) const {
    const int lev = CloudLevel(code);
    if (lev >= cloud_nlev) return false;
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
    Real r1 = CloudHalfWidth1(lev), r2 = CloudHalfWidth2(lev);
    int ia = static_cast<int>(floor((x1 - r1 - x1min)/fine_dx1));
    int ib = static_cast<int>(ceil((x1 + r1 - x1min)/fine_dx1)) - 1;
    int ja = static_cast<int>(floor((x2 - r2 - x2min)/fine_dx2));
    int jb = static_cast<int>(ceil((x2 + r2 - x2min)/fine_dx2)) - 1;
    for (int fj=ja; fj<=jb; ++fj) {
      int wj = WrapIndex(fj, nfine2);
      for (int fi=ia; fi<=ib; ++fi) {
        if (cloud_level(wj, WrapIndex(fi, nfine1)) > lev) return true;
      }
    }
    return false;
  }

  // The 3^2 children of a level-lev cloud in the exact two-scale identity: level lev+1
  // clouds at offsets (-1/2, 0, +1/2) of the parent half-width per direction with mass
  // fractions (1/4, 1/2, 1/4); function(child_x1, child_x2, fraction).
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachChild(const Real particle_x1, const Real particle_x2, const int lev,
                    const Function &function) const {
    const Real offset[3] = {-0.5, 0.0, 0.5};
    const Real fraction[3] = {0.25, 0.5, 0.25};
    Real h1 = CloudHalfWidth1(lev), h2 = CloudHalfWidth2(lev);
    for (int b=0; b<3; ++b) {
      for (int a=0; a<3; ++a) {
        function(particle_x1 + offset[a]*h1, particle_x2 + offset[b]*h2,
                 fraction[a]*fraction[b]);
      }
    }
  }

  // Encoded active leaf that owns a (wrapped) position; used to assign PGID to
  // particles created by a split or a merge.
  KOKKOS_INLINE_FUNCTION
  int LeafOwner(const Real particle_x1, const Real particle_x2) const {
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
    int fi = WrapIndex(static_cast<int>(floor((x1 - x1min)/fine_dx1)), nfine1);
    int fj = WrapIndex(static_cast<int>(floor((x2 - x2min)/fine_dx2)), nfine2);
    return cloud_target(fj, fi);
  }

  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachWeight(const Real particle_x1, const Real particle_x2, const int lev,
                     const Function &function) const {
    if (cloud_mode) {
      ForEachCloudWeight(particle_x1, particle_x2, lev, function);
    } else {
      ForEachWeight(particle_x1, particle_x2, function);
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real FineFraction(const Real particle_x1, const Real particle_x2) const {
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
    Real length1 = x1max - x1min;
    Real length2 = x2max - x2min;
    Real minimum_distance = 1.0e30;
    for (int block=0; block<nfine_blocks; ++block) {
      for (int shift2=-1; shift2<=1; ++shift2) {
        Real lower2 = fine_block_bounds(block,2) + shift2*length2;
        Real upper2 = fine_block_bounds(block,3) + shift2*length2;
        Real distance2 = (x2 < lower2) ? lower2 - x2
                       : ((x2 > upper2) ? x2 - upper2 : 0.0);
        for (int shift1=-1; shift1<=1; ++shift1) {
          Real lower1 = fine_block_bounds(block,0) + shift1*length1;
          Real upper1 = fine_block_bounds(block,1) + shift1*length1;
          Real distance1 = (x1 < lower1) ? lower1 - x1
                         : ((x1 > upper1) ? x1 - upper1 : 0.0);
          Real distance = sqrt(SQR(distance1) + SQR(distance2));
          minimum_distance = fmin(minimum_distance, distance);
        }
      }
    }
    if (minimum_distance <= 0.0) return 1.0;
    Real s = fmax(0.0, 1.0 - minimum_distance/transition_width);
    // Quintic smoothstep: alpha, alpha', and alpha'' match at both band edges.
    return s*s*s*(10.0 + s*(-15.0 + 6.0*s));
  }

  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachWeight(const Real particle_x1, const Real particle_x2,
                     const Function &function) const {
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
    Real alpha = FineFraction(x1, x2);
    int index1[3], index2[3];
    Real weight1[3], weight2[3];

    VirtualTSC(x1, x1min, x1max, ncoarse1, index1, weight1);
    VirtualTSC(x2, x2min, x2max, ncoarse2, index2, weight2);
    Real coarse_fraction = 1.0 - alpha;
    for (int b=0; b<3; ++b) {
      for (int a=0; a<3; ++a) {
        Real virtual_weight = coarse_fraction*weight2[b]*weight1[a];
        int count = coarse_count(index2[b], index1[a]);
        for (int entry=0; entry<count; ++entry) {
          function(coarse_target(index2[b],index1[a],entry),
                   virtual_weight*coarse_weight(index2[b],index1[a],entry));
        }
      }
    }

    VirtualTSC(x1, x1min, x1max, nfine1, index1, weight1);
    VirtualTSC(x2, x2min, x2max, nfine2, index2, weight2);
    for (int b=0; b<3; ++b) {
      for (int a=0; a<3; ++a) {
        Real virtual_weight = alpha*weight2[b]*weight1[a];
        int count = fine_count(index2[b], index1[a]);
        for (int entry=0; entry<count; ++entry) {
          function(fine_target(index2[b],index1[a],entry),
                   virtual_weight*fine_weight(index2[b],index1[a],entry));
        }
      }
    }

    if (!measure_correction) return;

    // Measure-preserving correction. Each interface/y-tile defect is the integral of
    // the raw blended row relative to G_H. The normalized compact profiles integrate
    // to one, so subtracting them restores every active-leaf column integral while
    // retaining the tile defect's zero zeroth and first moments.
    Real domain_length1 = x1max - x1min;
    Real coarse_dx2 = (x2max - x2min)/static_cast<Real>(ncoarse2);
    int yindex[3];
    Real yweight[3];
    VirtualTSC(x2, x2min, x2max, ncoarse2, yindex, yweight);
    for (int interface=0; interface<2; ++interface) {
      Real offset = x1 - interface_x1[interface];
      if (offset < -0.5*domain_length1) offset += domain_length1;
      if (offset >= 0.5*domain_length1) offset -= domain_length1;
      Real distance = interface_coarse_sign[interface]*offset;
      if (!(distance > 0.0 && distance < transition_width)) continue;
      Real s = distance/transition_width;
      Real profile1 = 140.0*s*s*s*SQR(1.0-s)*(1.0-s)/transition_width;
      for (int b=0; b<3; ++b) {
        int tile = interface*ncoarse2 + yindex[b];
        Real profile = profile1*yweight[b]/coarse_dx2;
        int count = correction_count(tile);
        for (int entry=0; entry<count; ++entry) {
          function(correction_target(tile,entry),
                   -profile*correction_weight(tile,entry));
        }
      }
    }
  }
};

//----------------------------------------------------------------------------------------
//! \struct DustGasDragTaskIDs
//  \brief container to hold TaskIDs of all dust+hydro tasks

struct DustGasDragTaskIDs {
  // "before_timeintegrator" tasks
  TaskID gswitch, cloud;
  // "before_stagen" tasks
  TaskID h_irecv, irecvd;
  // "stagen" tasks
  TaskID first2;
  TaskID h_flux, h_sendf, h_recvf, h_rkupdt, h_srctrms;
  TaskID gatwid, push;
  TaskID p_newgid, p_cnt, p_irecv, p_sendp, p_recvp;
  TaskID scat, sendd, recvd, solve, sendus, recvus, sendus_shr, recvus_shr;
  TaskID gkp, sendbr, recvbr, apply;
  TaskID p2_newgid, p2_cnt, p2_irecv, p2_sendp, p2_recvp;
  TaskID h_sendu_oa, h_recvu_oa, h_restu, h_sendu, h_recvu, h_sendu_shr, h_recvu_shr;
  TaskID h_bcs, h_prol, h_c2p, h_newdt, newdt, newdt2;
  // "after_stagen" tasks
  TaskID h_csend, h_crecv, p_csend, p_crecv, cleard;
};

namespace dust {

// Particle deposits need atomic updates on parallel execution spaces because stencil
// footprints overlap.  Kokkos::Serial has only one worker, however, so its generic
// compare-and-swap atomic adds synchronization overhead without providing protection.
// Keep the operation order unchanged while compiling that overhead out of serial CPU
// builds; all genuinely parallel backends retain Kokkos atomics.
KOKKOS_INLINE_FUNCTION
void DepositAdd(Real *destination, const Real value) {
#if defined(KOKKOS_ENABLE_SERIAL)
  constexpr bool serial_exec = std::is_same_v<DevExeSpace, Kokkos::Serial>;
#else
  constexpr bool serial_exec = false;
#endif
  if constexpr (serial_exec) {
    *destination += value;
  } else {
    Kokkos::atomic_add(destination, value);
  }
}

// Shared NGP/CIC/TSC stencil for deposits, matrix-free gathers, and the final kick.
// One helper is used everywhere so the coupled operator and conservative commit cannot
// silently acquire different particle-mesh weights.
KOKKOS_INLINE_FUNCTION
void PMWeights(const Real x, const Real xmin, const Real xmax, const int nx,
               const int is, const int scheme, int &ip, Real w[3]) {
  // Express the particle position directly in cell coordinates.  Besides being the
  // same uniform-grid geometry as CellCenterX, this uses one division rather than
  // forming dx and then dividing by it twice.
  Real cell = (x - xmin)*(static_cast<Real>(nx)/(xmax - xmin));
  int ig = static_cast<int>(cell + 1.0) - 1;
  Real del = cell - (static_cast<Real>(ig) + 0.5);
  ip = ig + is;
  if (scheme == 0) {
    w[0] = 0.0;
    w[1] = 1.0;
    w[2] = 0.0;
  } else if (scheme == 1) {
    w[0] = fmax(0.0, -del);
    w[1] = 1.0 - fabs(del);
    w[2] = fmax(0.0, del);
  } else {
    w[0] = 0.5*SQR(0.5 - del);
    w[1] = 0.75 - SQR(del);
    w[2] = 0.5*SQR(0.5 + del);
  }
}

//----------------------------------------------------------------------------------------
//! \class DustGasDrag

class DustGasDrag {
 public:
  DustGasDrag(MeshBlockPack *ppack, ParameterInput *pin);
  ~DustGasDrag();

  // data
  int nspecies;              // number of dust species (per-species stopping times)
  bool back_reaction;        // dust exerts drag on gas (PMBR); false = test particles
  bool gamma_switch;         // switch gamma to 1/2 when dt > max stopping time (Krapp24)
  bool stopping_times_initialized;  // post-pgen stopping-time contract has been checked
  bool is_shearing_box;      // <shearing_box> block present
  bool is_stratified;        // vertical gravity (3D shearing box only)
  Real qshear, omega0;       // shearing box parameters (0 if no shearing box)
  Real const_accel_phi = 0.0; // constant azimuthal particle acceleration (2-D r-z, PC2)
  Real dt_cfl;               // particle CFL number for transport timestep
  Real taus_max;             // largest stopping time over all species
  Real taus_min;             // smallest stopping time over all species
  Real dust_to_gas;          // total dust/gas mass ratio for default mass normalization
  DualArray1D<Real> taus;    // per-species stopping times
  DustDeposit deposit;       // particle-mesh deposit scheme (tsc default)
  DustStoppingTimeMode stopping_time_mode;
  DustDragSolver drag_solver;
  DustCoupling coupling;
  HybridMode hybrid_mode;
  HybridForceMode hybrid_force_mode;
  Real hybrid_enter_zeta, hybrid_enter_chi;
  Real hybrid_exit_zeta, hybrid_exit_chi;
  Real hybrid_last_zeta = 0.0;
  Real hybrid_last_chi = 0.0;
  Real hybrid_feedback_rate_max = 0.0;
  bool hybrid_have_metric = false;
  unsigned long long hybrid_pc2_cycles = 0;
  unsigned long long hybrid_split_be_cycles = 0;
  DustSMRTransfer smr_transfer_mode = DustSMRTransfer::none;
  SMRCompositeTransfer smr_transfer;
  // refinable-cloud controls and counters (SplitAndMergeClouds)
  Real cloud_split_slack = 0.02;      // support test shrinks the cloud by this fraction
  Real cloud_merge_tolerance = 0.05;  // coincidence radius in finest-cell units
  int cloud_gather_order = 1;         // 2: gather a linear reconstruction of the leaves
  bool cloud_force_local = false;     // gas-side drag uses the leaf's own velocity
  bool cloud_force_children = false;  // clouds overlapping finer leaves deposit the drag
                                      // as their 3^d two-scale children (kick unchanged)
  int cloud_gather_level = -1;        // diagnostic: >= 0 gathers every cloud with the
                                      // footprint of that level (deposit keeps the cloud)
  int cloud_transition = 0;           // P1 transition clouds at split edges:
                                      // 1 gathers with the child hat, 2 with the lobed
                                      // weights and point-valued reconstructions
  bool cloud_deform = false;          // deformable clouds: each particle carries dv_i/dx_j,
                                      // split children take v + G.d, and the drag is
                                      // deposited with the linear dust velocity field
  int cloud_next_tag = -1;            // next PTAG for created particles (-1 = unset)
  bool cloud_p1_split = false;        // moving-particle P1 split: edge children
                                      // separate only where the neighbour parent splits,
                                      // no merging (requires cloud_transition > 0)
  bool cloud_p1_release_exit = false; // diagnostic: lobed clouds that leave the split
                                      // region release their lobes (old-rule exit edge)
  bool cloud_merge = false;           // coarsening: far from finer
                                      // leaves, release lobes and merge fine clouds that are
                                      // indistinguishable at the coarser level
  int cloud_merge_buffer = 2;         // merge cells lie at least this many cells (of the
                                      // target level) from any finer leaf
  Real cloud_merge_alpha = 0.2;       // residual velocity tolerance alpha*H/tau_c,
                                      // tau_c = min(t_s, 1/Omega)
  int cloud_merge_interval = 10;      // cycles between coarsening passes
  Real cloud_merge_min_mass = 0.5;    // a merge commits only with >= this fraction of m_ref
  bool cloud_merge_device = false;  // run MergeClouds on the device
  Real cloud_merge_tstop = -1.0;      // diagnostic: no merge passes after this time
  // merge passes on a physical-time schedule (cloud_merge_schedule = time,
  // cloud_merge_dt_omega) instead of every cloud_merge_interval cycles, and a burst of passes
  // every cloud_merge_burst_interval cycles for cloud_merge_burst_dt_omega / omega0 after each
  // regrid that removed blocks; resampling once more right after each regrid that added blocks
  bool cloud_merge_by_time = false;
  Real cloud_merge_dt = 0.0, cloud_merge_t0 = 0.0;
  long long cloud_merge_next = -1;
  Real cloud_merge_burst_dt = 0.0, cloud_merge_burst_until = -1.0;
  int cloud_merge_burst_interval = 50;
  bool cloud_merge_clock_restored = false;
  bool cloud_resample_after_refine = false, amr_refined_pending = false;
  unsigned long long cloud_resample_extra = 0;
  int amr_nmb_before = -1;
  bool MergeDue();
  DvceArray1D<char> merge_mask_d;     // device merge-cell masks of the level-L grids (built once,
  int merge_mask_geom[12] = {0};       // static SMR): per level n1, n2, offset
  bool merge_mask_built = false;
  bool cloud_merge_nc = true;         // merge-side node convention
  bool cloud_merge_absorb = false;    // orphan absorption into existing coarse clouds
  bool cloud_merge_pending_log = false;  // log why designated centres did not commit
  bool cloud_merge_d3 = false;        // diagnostic: per-pass coarse-row first-moment
  int cloud_merge_d3_iter = 30;       // matching of the new clouds' x positions (D3), Jacobi iterations
  int cloud_merge_lobe_wait = 0;      // a centre waits while an unreleased lobed cloud is near
                                      // (1: reach H + ell, 2: reach per lobe direction)
  std::vector<Real> cloud_merge_mroot; // per species: root-level cloud mass (set on first pass)
  // quantum mode: stage X exact P1-stencil re-merge, stage G mass-quantum grouping
  // (cross-level), stage C same-level compression in the fine interior
  int cloud_merge_mode = 0;           // 0 tent, 1 quantum
  bool cloud_merge_stencil = true;    // quantum: stage X on
  Real cloud_merge_stencil_tol = 0.02;  // stage X offset tolerance (units of the piece half-width)
  Real cloud_merge_smax = 1.0;        // stage G group position extent (units of the target cell)
  int cloud_merge_K = 4;              // open groups per cell walk (1..8)
  Real cloud_audit_t0 = -1.0;          // merge audit (diagnostic): dump every merge pass in
  Real cloud_audit_t1 = -1.0;          // [t0, t1]; stages in full when the pass changed anything
  bool cloud_audit_compact = true;     // quiet passes: compact particle + gas snapshot
  bool cloud_merge_xprotect = false;   // stage X's waiting candidates are not
                                       // walked by G/C or absorbed in the same pass
  Real cloud_absorb_budget = -1.0;     // > 0, largest leaf change of mass (and
                                       // momentum) an absorption may make, relative to the leaf
  Real cloud_absorb_floor = 0.1;       // ... with the leaf mass floored at this fraction of a
                                       // root cloud mass per root-cell volume
  Real cloud_merge_budget = -1.0;      // > 0, one budget of leaf mass / momentum
                                       // change for the whole pass (stages G, C, absorption)
  int cloud_merge_budget_evals = 4;    // ... evaluations per stage before the rest is rejected
  Real cloud_merge_budget_prior = 0.0; // ... testing only: fraction of the budget already used
  bool cloud_audit_rejected = false;   // full audit file when G/C groups were budget-rejected
  bool cloud_merge_budget_x = false;   // stage X is also decided by the pass budget
  bool cloud_audit_only = false;       // a merge pass writes its entry audit and returns
  bool cloud_resample = false;         // conservative same-level resampling in the code
  int cloud_resample_interval = 625;   // ... cycles between resamplings
  Real cloud_resample_eta = 2.0;       // ... mass cap eta m0 / 4^level (<= 0: no cap)
  int cloud_resample_patch = 2;        // ... patch size in root cells
  int cloud_resample_threads = 1;      // ... host threads of the patch loop
  Real cloud_resample_m0 = -1.0;       // ... root cloud mass (<= 0: the merge pass's value)
  std::vector<int> cloud_resample_dump;     // cycles whose resampling writes the particle table before and after
  bool cloud_resample_skip_at_cap = true;   // ... clouds at their cap stay out of the pool
  unsigned long long cloud_resample_pass = 0, cloud_resample_removed = 0;
  // resampling at planned physical times t0 + k dt instead of every N cycles
  bool cloud_resample_by_time = false;      // <dust>/cloud_resample_schedule = time
  Real cloud_resample_dt = 0.0;             // ... interval in code time (cloud_resample_dt, or
                                            //     cloud_resample_dt_omega / <shearing_box>/omega0)
  Real cloud_resample_t0 = 0.0;             // ... time origin of the planned times
  Real cloud_resample_omega = 0.0;          // ... <shearing_box>/omega0 (0 without a shearing box)
  bool cloud_resample_schedule_reset = false;  // ... a restart may change the origin / interval
  long long cloud_resample_next = -1;       // ... number of the next event (-1: set at the first check)
  unsigned long long cloud_resample_events = 0, cloud_resample_skipped = 0;
  bool cloud_resample_restarted = false;    // ... this run continues from a restart file
  bool cloud_resample_clock_restored = false;  // ... which had a time-mode schedule record:
  Real cloud_resample_t0_file = 0.0, cloud_resample_dt_file = 0.0;  // its origin, interval
  long long cloud_resample_next_file = -1;  // ... and next event
  Real cloud_resample_tprev = -1.0;         // ... start time of the previous cycle (log only)
  int cloud_merge_g_lag = 2;          // stage G only in level-L cells whose (2 lag + 1)^2
                                      // neighbourhood is in the merge zone: the outer band is
                                      // left to stage X (found in the LinA tests)
  bool cloud_compress = false;        // stage C on (quantum mode only)
  Real cloud_compress_smax = 0.5;     // stage C group extent (units of the piece's own cell)
  Real cloud_compress_ds = 0.1;       // stage C |dS| budget per direction, fraction of M Var(hat)
  int cloud_compress_buffer = 4;      // stage C cells: no coarser leaf within this many cells of
                                      // the next-coarser level
  int cloud_merge_pass = 0;           // pass counter (walk rotation)
  DvceArray1D<char> compress_mask_d;  // stage C interior masks per level (built once)
  int compress_mask_off[5] = {0};
  bool compress_mask_built = false;
  unsigned long long cloud_coarsen_events = 0, cloud_coarsen_in = 0;
  unsigned long long cloud_coarsen_reject = 0, cloud_far_release = 0;
  Real cloud_coarsen_dk_grad = 0.0, cloud_coarsen_dk_res = 0.0;
  int cloud_split_buffer = 0;         // refined dust region reaches this many next-coarser
                                      // cells beyond the refined gas
  int cloud_native_check = 0;         // diagnostic: every this many cycles, compare the
                                      // level-native and finest-grid traversals
  DvceArray1D<int> split_np, split_off;  // device split work arrays, kept between cycles
  bool cloud_presplit_x1 = false;     // diagnostic: every root cloud is
                                      // split along x1 only (x2 keeps the parent width as
                                      // lobe 3, gathered as a cloud integral), so a z-normal
                                      // refinement edge has no along-edge width jump
  bool cloud_split_enable = true;     // test only: false = no splitting at all
  bool cloud_split_interp = false;    // P1 split/release: pieces take the node-convention
                                      // (linear, P1) prolongation of displacement and velocity
                                      // from the neighbouring nodes of the particle lattice,
                                      // positions relative to the centre piece, velocity
                                      // increments mass-mean-free (single-stream prototype)
  bool cloud_split_device = false;    // P1 split on the device; the split log,
                                      // cloud_split_interp and split_oracle keep the host path
  bool cloud_split_log = false;       // diagnostic: log every P1 split event (parent,
                                      // children, leaf deposit audit) to a text file
  bool cloud_split_log_leaves = false; // diagnostic (with cloud_split_log): per split call,
                                      // a V line (volume-weighted norms of the net leaf
                                      // change) and a D line per touched leaf
  // Diagnostic prescribed gas (set by a problem generator, e.g. <problem>/prescribed_gas
  // of streaming_linear): fills the gas conserved variables of every cell, ghosts
  // included, with a prescribed state at time t.  With PC2 and back_reaction=false the
  // stage-2 midpoint gas velocity seen by the dust is formed from the prescribed states
  // at t and t+dt, and the gas is reset to the prescribed state at the end of each step.
  std::function<void(Real, DvceArray5D<Real>&)> prescribed_gas;
  // Diagnostic position oracle (set by a problem generator, <problem>/split_oracle of
  // streaming_linear): the exact dust displacement (dx, dz) at position (x, z) and time t,
  // used to place the pieces of in-run P1 splits and lobe releases (SplitCloudsP1).
  std::function<void(Real, Real, Real, Real&, Real&)> split_oracle;
  // Optional velocity part of the oracle (<problem>/split_oracle_velocity): the exact dust
  // velocity perturbation (dvx, dvz, dvphi) at (x, z, t); the pieces get its family-relative
  // part (mass-weighted mean removed, so the event momentum is unchanged).
  std::function<void(Real, Real, Real, Real&, Real&, Real&)> split_oracle_velocity;
  // Node convention of the oracle (<problem>/split_oracle_node): the pieces take the
  // prescribed displacement (and velocity) at their nodes relative to the centre piece,
  // which keeps the parent's state; the event's mass centroid then moves by the P1
  // difference at a lobe release instead of being preserved.
  bool split_oracle_node = false;
  // Conservation variants of the node convention (<problem>/split_oracle_node_meanfree):
  // 1 removes the mass-weighted mean of the nodal velocity increments (event momentum
  // conserved), 2 also that of the nodal displacement increments (event mass first
  // moment conserved); 0 keeps both relative to the centre piece.
  int split_oracle_node_meanfree = 0;
  // Diagnostic analytic gather (<problem>/analytic_gather of streaming_linear): the PC2
  // midpoint gas velocity seen by each particle is replaced by the analytic eigenmode at
  // t+dt/2 and the predicted midpoint position, with the response of the
  // cloud_transition = 2 gather: the hat transform sinc^2(k ell/2) along plain
  // directions (cloud integral), the point value along lobed ones; point_all
  // (<problem>/analytic_gather_point) takes the point value for every cloud instead, so
  // that all cloud shapes see the same field; centroid_lobes
  // (<problem>/analytic_gather_centroid) samples a lobed direction at the mass centroid
  // of the lobed shape (node -/+ ell/3) instead of the node.  Plain data so that the
  // advance kernel can capture it.
  struct AnalyticGather {
    bool on = false;
    bool point_all = false;
    bool centroid_lobes = false;
    Real kx = 0.0, kz = 0.0, srate = 0.0, wrate = 0.0;
    Real ux0 = 0.0, up0 = 0.0;          // background velocity (x1 radial, x3 azimuthal)
    Real uxr = 0.0, uxi = 0.0, uzr = 0.0, uzi = 0.0, upr = 0.0, upi = 0.0;
  };
  AnalyticGather analytic_gather;
  unsigned long long cloud_split_event = 0;
  unsigned long long cloud_split_count = 0, cloud_merge_count = 0;
  unsigned long long cloud_lobe_split_count = 0;
  bool UsesCompositeTransfer() const {
    return (smr_transfer_mode == DustSMRTransfer::smooth_p2 ||
            smr_transfer_mode == DustSMRTransfer::refinable_cloud);
  }
  Real drag_rtol, drag_atol;
  int drag_iter_max, drag_diagnostic_interval;
  Real adaptive_order_c, adaptive_rtol_max, adaptive_tref, adaptive_state_floor;

  // deposited fields, dimensioned (nmb, nvar, ncells3, ncells2, ncells1)
  DvceArray5D<Real> qdep;    // [0]=Q, [1-3]=P/predictor impulse, [4]=feedback rate
  DvceArray5D<Real> ustar;   // nvar=3: provisional drag-corrected gas velocity u*
  DvceArray5D<Real> uvel_pred;  // nvar=3: old-state gas velocity of the linear cloud predictor
  DvceArray5D<Real> gpack;      // (m, k, j, i, 9): gas velocity and its central differences in
                                // x1 and x2, side by side, for the linear cloud gather
  DvceArray5D<Real> dmom;    // nvar=3: PMBR momentum deposit; becomes R_g after apply
  DvceArray5D<Real> cdummy;  // 1-element dummy coarse array for ustar copy exchange
  DvceArray5D<Real> solver_r;   // coupled-solver residual
  DvceArray5D<Real> solver_p;   // PCG search direction
  DvceArray5D<Real> solver_ap;  // matrix-free A*x / A*p work field

  // Boundary communication objects
  MeshBoundaryValuesDep *pbval_qp;  // additive exchange of (Q,P) ghost deposits
  MeshBoundaryValuesDep *pbval_dm;  // additive exchange of PMBR ghost deposits
  MeshBoundaryValuesCC  *pbval_us;  // copy exchange to fill u* ghost zones
  ShearingBoxCC *psbox_us = nullptr;  // shear-periodic remap of u* x1 ghost zones (3D)
  MeshBoundaryValuesCC *pbval_solver_copy = nullptr;
  MeshBoundaryValuesDep *pbval_solver_add = nullptr;

  // Cumulative rank-local timing and globally consistent iteration diagnostics.
  unsigned long long solver_stage_count = 0;
  unsigned long long solver_applya_count = 0;
  unsigned long long solver_halo_count = 0;
  unsigned long long solver_reduction_count = 0;
  unsigned long long solver_fast_accept_count = 0;
  unsigned long long solver_pcg_stage_count = 0;
  double solver_wall_seconds = 0.0;
  // Exact bounded histogram over [0, drag_iter_max], allocated only on rank 0.
  std::vector<unsigned long long> solver_pcg_iteration_hist;
  Real solver_last_residual = -1.0;  // negative means a true residual was not sampled
  Real solver_last_epsmax = -1.0;  // negative means this mode did not sample epsilon_c
  Real solver_last_error_bound = -1.0;
  Real solver_last_acceptance_target = -1.0;
  Real solver_last_state_scale = -1.0;
  int solver_last_iterations = 0;
  bool solver_last_fast_accept = false;

  // container to hold names of TaskIDs
  DustGasDragTaskIDs id;

  // functions...
  void AssembleDustGasDragTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // dust tasks are dormant on the trivial assembly stage of imex2+ (beta=0, gam0=1)
  bool ActiveStage(Driver *pdrive, int stage) const;
  bool FirstMigrationActive(Driver *pdrive, int stage) const;
  bool SecondMigrationActive(Driver *pdrive, int stage) const;
  Real DragStep(Driver *pdrive) const;
  // default particle mass/species/stopping-time initialization (pgen may override)
  void SetDefaultMasses(ParameterInput *pin);
  void BuildSMRCompositeTransfer(ParameterInput *pin);
  void BuildRefinableCloudTransfer(ParameterInput *pin);
  // experimental dynamic mesh (dust_amr.cpp): rebuild the leaf maps, mark
  // the cached masks stale, set PGID from the positions and the particle time step again
  void BeforeRegrid();                // particle count and checksum before the mesh changes
  void RebuildAfterRegrid(Driver *pdrive, ParameterInput *pin);
  unsigned long long amr_checksum = 0ULL;
  int amr_npart_before = -1;
  // Dynamic-mesh state across a restart.  The driver sets
  // amr_check_after_output while it writes the outputs of a cycle (the mesh check of that
  // cycle follows them); a restart file keeps it and the per-block cooldown ages
  // (MeshRefinement::ncyc_since_ref), and ResumeAmrAfterRestart restores the ages and does
  // the pending check before the first cycle.
  bool amr_check_after_output = false;
  bool amr_restart_state_found = false, amr_restart_pending = false;
  std::vector<int> amr_restart_ages;
  void ResumeAmrAfterRestart(Driver *pdrive, ParameterInput *pin);
  bool experimental_amr = false;      // <dust>/experimental_amr
  int amr_regrids = 0;                // regrids seen by RebuildAfterRegrid
  bool cloud_map_announced = false;   // the map banner is printed once
  // native restart (dust_restart.cpp): the records of what changes during
  // a run and cannot be rebuilt from the particles
  void PackRestartState(std::vector<char> &b) const;
  void UnpackRestartState(const char *p, std::size_t nbytes);
  int SplitAndMergeClouds();   // returns the number of clouds split
  int MakeTransitionClouds();  // P1 transition clouds at split edges
  int SplitCloudsP1();         // moving-particle P1 split
  int SplitCloudsP1Device();   // device P1 split without the diagnostics
  TaskStatus RefineClouds(Driver *pdrive, int stage);
  int MergeClouds();
  int MergeCloudsDevice();    // device version of MergeClouds
  void AuditFeedback(DvceArray2D<Real> &fb);   // frozen-gas predictor feedback
  int ResampleClouds();       // returns the number of clouds removed
  bool ResampleDue();         // is a resampling due at the start of this cycle?
  void PackGatherField(const DvceArray5D<Real> &vel);   // fills gpack from a velocity array
  void CheckNativeTransfer();  // level-native against finest-grid traversal

  // ...in "before_timeintegrator" list
  TaskStatus GammaSwitch(Driver *pdrive, int stage);
  // Must be public because NVCC extended host/device lambdas cannot be enclosed by a
  // private or protected member function.
  void RefreshStoppingTimeMaximum();  // validate IPTS; refresh taus_max when requested
  // ...in "before_stagen" list
  TaskStatus InitRecvDep(Driver *pdrive, int stage);
  // ...in "stagen" list
  TaskStatus FirstTwoImpRK(Driver *pdrive, int stage);     // stage 1 register copies
  TaskStatus ExplicitPush(Driver *pdrive, int stage);      // rotation kick + drift
  TaskStatus UpdateParticleGIDs(Driver *pdrive, int stage);
  TaskStatus CountParticleSends(Driver *pdrive, int stage);
  TaskStatus InitParticleRecv(Driver *pdrive, int stage);
  TaskStatus SendParticles(Driver *pdrive, int stage);
  TaskStatus RecvParticles(Driver *pdrive, int stage);
  TaskStatus UpdateParticleGIDs2(Driver *pdrive, int stage);
  TaskStatus CountParticleSends2(Driver *pdrive, int stage);
  TaskStatus InitParticleRecv2(Driver *pdrive, int stage);
  TaskStatus SendParticles2(Driver *pdrive, int stage);
  TaskStatus RecvParticles2(Driver *pdrive, int stage);
  TaskStatus AddDragHistoryGas(Driver *pdrive, int stage); // u0 += a_twid*dt*R_g
  TaskStatus DepositDrag(Driver *pdrive, int stage);       // scatter Q,P
  TaskStatus SendDepQP(Driver *pdrive, int stage);
  TaskStatus RecvDepQP(Driver *pdrive, int stage);
  TaskStatus GasImplicitSolve(Driver *pdrive, int stage);  // u* = (rho*u+P)/(rho+Q)
  void UpdateHybridMetric(Driver *pdrive, int stage);
  TaskStatus SolveCoupledStage(Driver *pdrive, int stage);
  void ApplyCoupledOperator(DvceArray5D<Real> &field, DvceArray5D<Real> &result,
                            Real a_dt);
  void ApplyDefectCorrection(DvceArray5D<Real> &field, DvceArray5D<Real> &work,
                             Real a_dt);
  // Internal shared implementation.  This must remain public because it encloses NVCC
  // extended host/device lambdas; callers should use one of the explicit methods above.
  void ApplyCoupledOperatorImpl(DvceArray5D<Real> &field, DvceArray5D<Real> &result,
                                Real a_dt, bool correct_field);
  void CompleteCopyExchange(DvceArray5D<Real> &field);
  void CompleteAddExchange(DvceArray5D<Real> &field);
  void GlobalDot(DvceArray5D<Real> &left, DvceArray5D<Real> &right, Real value[3]);
  void GlobalPreconditionedNorm(DvceArray5D<Real> &residual, Real value[3]);
  int StrictPCG(Real a_dt, bool residual_is_current);
  Real AdaptiveErrorBound(Real a_dt, Real residual_norm, Real &state_scale,
                          Real &acceptance_target);
  TaskStatus SendUstar(Driver *pdrive, int stage);
  TaskStatus RecvUstar(Driver *pdrive, int stage);
  TaskStatus SendUstarShr(Driver *pdrive, int stage);
  TaskStatus RecvUstarShr(Driver *pdrive, int stage);
  TaskStatus GatherKickPMBR(Driver *pdrive, int stage);    // gather+kick+record+scatter
  TaskStatus SendPMBR(Driver *pdrive, int stage);
  TaskStatus RecvPMBR(Driver *pdrive, int stage);
  TaskStatus ApplyPMBR(Driver *pdrive, int stage);         // u0 += dmom; dmom -> R_g
  TaskStatus NewTimeStep(Driver *pdrive, int stage);
  TaskStatus NewTimeStep2(Driver *pdrive, int stage);
  TaskStatus ComputeNewTimeStep(Driver *pdrive, int stage);
  // ...in "after_stagen" list
  TaskStatus ClearParticleSend(Driver *pdrive, int stage);
  TaskStatus ClearParticleRecv(Driver *pdrive, int stage);
  TaskStatus ClearDep(Driver *pdrive, int stage);

 private:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this DustGasDrag
};

// scheduled refinement request for tests (dust_amr.cpp); the schedule is read from <dust>
int SetupScheduledRefinement(ParameterInput *pin);
void ScheduledRefinement(MeshBlockPack *pmbp);
// dust-density refinement request (dust_amr.cpp)
void SetupDensityTrigger(ParameterInput *pin);
void DensityRefinement(MeshBlockPack *pmbp);
// particles and dust state in the native restart file (dust_restart.cpp)
std::size_t ParticleRestartHeadBytes();
std::uint64_t ParticleRestartSectionBytes(const char *head);
void PackParticleRestart(MeshBlockPack *pmbp, std::vector<char> &buf);
void UnpackParticleRestart(MeshBlockPack *pmbp, const std::vector<char> &buf);

} // namespace dust
#endif // DUST_DUST_HPP_
