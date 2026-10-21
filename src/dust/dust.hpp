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

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <chrono>
#include <string>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "coordinates/cell_locations.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// a device function kept out of line (rare paths of the hot cloud kernels)
#if defined(__CUDACC__)
#define DUST_NOINLINE __noinline__
#else
#define DUST_NOINLINE __attribute__((noinline))
#endif

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
// two-scale split (see DustGasDrag::SplitCloudsP1).
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
  bool cloud_table = false;            // cloud_target / cloud_level built
  int cloud_nlev = 0;                 // max_level - root_level
  int cloud_max_level = 0;            // clouds are never split beyond this level
  bool cloud_split_everywhere = false; // control: split regardless of the local mesh
  bool cloud_split_overlap = true;     // split once the (shrunk) support touches finer leaves
                                       // instead of once it lies entirely on them
  Real root_dx1 = 0.0, root_dx2 = 0.0;
  Real fine_dx1 = 0.0, fine_dx2 = 0.0;
  DvceArray2D<int> cloud_target;      // finest virtual cell -> encoded active leaf
  DvceArray2D<int> cloud_level;       // finest virtual cell -> owner level - root_level
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

  // Cloud-integrated weights on active leaves: the virtual finest cells overlapped by
  // the support are integrated exactly and forwarded to their owning leaf, so a
  // coarse leaf receives the sum of its sub-cell integrals (its own exact integral).
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachCloudWeight(const Real particle_x1, const Real particle_x2, const int lev,
                          const Function &function) const {
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
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
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachCloudMoment(const Real particle_x1, const Real particle_x2, const int lev,
                          const Function &function) const {
    Real x1 = WrapPosition(particle_x1, x1min, x1max);
    Real x2 = WrapPosition(particle_x2, x2min, x2max);
    int i0, j0;
    Real w1[kMaxCloudStencil], w2[kMaxCloudStencil];
    Real g1[kMaxCloudStencil], g2[kMaxCloudStencil];
    bool point = (lev & kCloudGatherPoint) != 0;
    bool point1 = point, point2 = point;
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
//! \struct LeafTraversal
//! Block-local traversal of the actual leaves under a
//! refinable cloud, without the global virtual finest grid of SMRCompositeTransfer.
//!
//! Each MeshBlock knows its box, its level and its adjacent MeshBlocks (from the AthenaK
//! neighbour table, every distinct neighbour once, with the periodic shift that puts it next
//! to the block).  A cloud owned by block m is integrated exactly over every leaf cell of m
//! and of its neighbours that its support overlaps: per direction the integral of the cloud
//! shape over the leaf interval is a difference of the hat CDF (and of its first moment),
//! the formulas of SMRCompositeTransfer::CloudWeightsMoments1D evaluated at the leaf faces
//! instead of the faces of the virtual finest cells.  The integral over a leaf is the sum of
//! the integrals over its finest sub-cells, so the weights and first moments equal those of
//! the table up to round-off; lobes, gather codes and the point rule are handled as there.
//! The sum of the weights is returned so that a support reaching beyond the neighbours
//! (possible only for clouds much coarser than the leaves) is detected, not truncated.
//!
//! Block slots: [0, nmb) the blocks of this rank (slot = gid - gids), [nmb, nfld)
//! the blocks of other ranks next to them (halo blocks: their leaf fields are exchanged, see
//! DustHalo), [nfld, nall) every other block of the mesh (geometry only, for the eligibility
//! queries that leave the neighbourhood and for Locate).  One rank: nfld = nall = nmb.

struct LeafTraversal {
  static constexpr int kMaxNb = 16;    // distinct neighbours of a 2-D block (12 at most)
  int nmb = 0, gids = 0;
  int nfld = 0, nall = 0;              // slots with leaf fields (local + halo), all slots
  int nx1 = 0, nx2 = 0, is = 0, js = 0, ncells1 = 0, ncells2 = 0;
  Real lx = 0.0, ly = 0.0, x1min = 0.0, x2min = 0.0, x1max = 0.0, x2max = 0.0;
  Real ilx = 0.0, ily = 0.0;           // 1/lx, 1/ly
  Real root_dx1 = 0.0, root_dx2 = 0.0;
  Real iroot_dx1 = 0.0, iroot_dx2 = 0.0;   // 1/root_dx1, 1/root_dx2
  DvceArray2D<Real> box;               // (b, 0..5): x1min, x2min, dx1, dx2, 1/dx1, 1/dx2 of slot b
  DvceArray1D<int> nnb;                // number of neighbours of local block m
  DvceArray2D<int> nb;                 // (m, k): slot of neighbour k (local or halo)
  DvceArray3D<int> nbs;                // (m, k, 0..1): periodic shift (in domain lengths)
  // Split / merge eligibility on integer coordinates of the finest level (no
  // stored grid).  Block m covers the finest cells [fbox(m,0), fbox(m,1)) x [fbox(m,2),
  // fbox(m,3)); blev(m) is its level above the root.  The region where the (dilated) level
  // map of the table exceeds lev is the union of the blocks of level >= lev + 1, each widened
  // by rdil[lev + 1] finest cells (cloud_split_buffer): a higher level has a
  // smaller region and a smaller widening, so it lies inside.
  static constexpr int kMaxLev = 16;
  int nlev = 0, nfine1 = 0, nfine2 = 0;
  Real fine_dx1 = 0.0, fine_dx2 = 0.0;
  int rdil[kMaxLev + 1] = {0};
  int cloud_max_level = 0;
  bool split_everywhere = false, split_overlap = false;
  DvceArray1D<int> blev;
  DvceArray2D<int> fbox;
  // (m) the finest level among local block m and its neighbours, i.e. among the
  // candidates of a query that needs no global search
  DvceArray1D<int> nbmax;
  // (m, 0..1): width of the ring around block m (finest cells, per direction) that its
  // neighbours certainly hold: the smallest neighbour extent, checked when built (0 if the
  // check fails); a query inside it needs no coverage test
  DvceArray2D<int> safe;
  // sorted logical-location keys of the blocks and their blocks (Locate)
  DvceArray1D<std::int64_t> lockey;
  DvceArray1D<int> locblk;
  int nloc = 0;

  KOKKOS_INLINE_FUNCTION
  static Real Wrap(const Real x, const Real xmin, const Real xmax) {
    const Real l = xmax - xmin;
    Real w = x;
    while (w < xmin) w += l;
    while (w >= xmax) w -= l;
    return w;
  }

  // the cloud shape along one direction (SMRCompositeTransfer::CloudWeightsMoments1D)
  struct Shape {
    Real x = 0.0, ell = 1.0, inv = 1.0, sl = 0.0, lo = 0.0, hi = 0.0;
    int lobe = 0;
    bool both = false, point = false;
  };
  KOKKOS_INLINE_FUNCTION
  static Shape MakeShape(const Real x, const Real ell_in, const Real inv_in, const int lobe_in,
                         const bool point) {
    Shape s;
    s.both = (lobe_in == 3);
    s.ell = s.both ? 2.0*ell_in : ell_in;
    s.lobe = s.both ? 0 : lobe_in;
    s.sl = (s.lobe == 1) ? -s.ell : s.ell;
    s.lo = (s.lobe == 1) ? x - 2.0*s.ell : x - s.ell;
    s.hi = (s.lobe == 2) ? x + 2.0*s.ell : x + s.ell;
    s.x = x;
    s.point = point;
    s.inv = s.both ? 0.5*inv_in : inv_in;   // = 1/ell (powers of two: exact)
    return s;
  }
  // the shape's cumulative values at a face z of one direction: the hat CDF of the plain part
  // and of the lobe, and (kMoments) their first-moment antiderivatives.  A cell takes the
  // differences of its two faces, so a run of cells evaluates each face once.
  struct Face {Real c = 0.0, cl = 0.0, h = 0.0, hl = 0.0;};
  template <bool kMoments>
  KOKKOS_INLINE_FUNCTION
  static Face FaceAt(const Shape &s, const Real z) {
    using T = SMRCompositeTransfer;
    Face f;
    const Real sz = (z - s.x)*s.inv;
    f.c = T::HatCDF(sz);
    if constexpr (kMoments) f.h = T::HatMoment(sz);
    if (s.lobe) {
      const Real tz = (z - s.x - s.sl)*s.inv;
      f.cl = T::HatCDF(tz);
      if constexpr (kMoments) f.hl = T::HatMoment(tz);
    }
    return f;
  }
  // weight and (kMoments) first moment (about the particle, length units) over the cell
  // between the faces a and b
  template <bool kMoments>
  KOKKOS_INLINE_FUNCTION
  static void Cell(const Shape &s, const Face &a, const Face &b, Real &w, Real &g) {
    w = b.c - a.c;
    g = 0.0;
    if constexpr (kMoments) g = (s.both && s.point) ? 0.0 : s.ell*(b.h - a.h);
    if (s.lobe) {
      const Real wl = b.cl - a.cl;
      w = (2.0/3.0)*w + (1.0/3.0)*wl;
      if constexpr (kMoments) {
        g = s.point ? 0.0 : (2.0/3.0)*g + (1.0/3.0)*(s.ell*(b.hl - a.hl) + s.sl*wl);
      }
    }
  }

  // ---- candidate blocks (Codex reviews) -------------------------------------------------
  // Every query is answered in the frame of the anchor block m (the periodic image of the
  // position next to m).  Local: m and its distinct neighbours, each with the periodic shift
  // (domain lengths, from the direction of its neighbour slot) that puts it on that side of
  // m, so the two images of a block that is the neighbour on both sides (one or two blocks
  // across the domain) both appear.  Global: every block of the pack in every periodic image
  // that meets the query box [ia, ib] x [ja, jb] (finest cells, inclusive) widened by r, so a
  // query of any width is answered; used when the anchor's neighbourhood does not hold the
  // whole query.  f(b, sx, sy): block b shifted by -(sx, sy) domain lengths.
  KOKKOS_INLINE_FUNCTION
  static int FloorDiv(const int a, const int n) {
    const int q = a/n;
    return (a % n != 0 && a < 0) ? q - 1 : q;
  }
  // the images k (shift +k n) of [c0 - r, c1 + r) that meet [a, b]: k0..k1 (none if k0 > k1)
  KOKKOS_INLINE_FUNCTION
  static void ImageRange(const int c0, const int c1, const int r, const int n, const int a,
                         const int b, int &k0, int &k1) {
    k0 = FloorDiv(a - (c1 + r), n) + 1;
    k1 = FloorDiv(b - (c0 - r), n);
  }
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForCandidates(const int m, const bool global, const int ia, const int ib, const int ja,
                     const int jb, const int r, const Function &f) const {
    if (!global) {
      f(m, 0, 0);
      for (int k=0; k<nnb(m); ++k) f(nb(m, k), nbs(m, k, 0), nbs(m, k, 1));
      return;
    }
    for (int b=0; b<nall; ++b) {
      int kx0, kx1, ky0, ky1;
      ImageRange(fbox(b, 0), fbox(b, 1), r, nfine1, ia, ib, kx0, kx1);
      ImageRange(fbox(b, 2), fbox(b, 3), r, nfine2, ja, jb, ky0, ky1);
      for (int ky=ky0; ky<=ky1; ++ky) {
        for (int kx=kx0; kx<=kx1; ++kx) f(b, -kx, -ky);
      }
    }
  }
  // the periodic image of (x1, x2) next to block m
  KOKKOS_INLINE_FUNCTION
  void Frame(const int m, Real &x1, Real &x2) const {
    const Real cx = box(m, 0) + 0.5*nx1*box(m, 2), cy = box(m, 1) + 0.5*nx2*box(m, 3);
    x1 -= lx*round((x1 - cx)*ilx);
    x2 -= ly*round((x2 - cy)*ily);
  }
  // finest-cell box [x0, x1) x [y0, y1) of block b in the image (sx, sy), widened by r
  KOKKOS_INLINE_FUNCTION
  void ImageRect(const int b, const int sx, const int sy, const int r, int &x0, int &x1, int &y0,
                 int &y1) const {
    x0 = fbox(b, 0) - sx*nfine1 - r; x1 = fbox(b, 1) - sx*nfine1 + r;
    y0 = fbox(b, 2) - sy*nfine2 - r; y1 = fbox(b, 3) - sy*nfine2 + r;
  }
  // does the finest-cell box [ia, ib] x [ja, jb] (inclusive) meet a candidate block of level
  // >= minlev widened by r?
  KOKKOS_INLINE_FUNCTION
  bool Meets(const int m, const int ia, const int ib, const int ja, const int jb,
             const int minlev, const int r, const bool global) const {
    bool hit = false;
    ForCandidates(m, global, ia, ib, ja, jb, r, [&](const int b, const int sx, const int sy) {
      if (hit || blev(b) < minlev) return;
      int x0, x1, y0, y1;
      ImageRect(b, sx, sy, r, x0, x1, y0, y1);
      if (ib < x0 || ia >= x1 || jb < y0 || ja >= y1) return;
      hit = true;
    });
    return hit;
  }
  // is that box entirely covered by them?  A scan by row bands, without a list of boxes (any
  // number of blocks and images): from the first cell (x, y) of a band not yet known to be
  // covered, take the candidate box holding it that reaches furthest in x, continue at its x
  // end; the rows up to the lowest top among the boxes taken are then covered over the whole
  // width, and the next band starts there.  A cell held by no box: not covered.
  KOKKOS_INLINE_FUNCTION
  bool Covered(const int m, const int ia, const int ib, const int ja, const int jb,
               const int minlev, const int r, const bool global) const {
    int y = ja;
    while (y <= jb) {
      int x = ia, top = jb + 1;
      while (x <= ib) {
        int xe = x, ye = y;
        ForCandidates(m, global, x, x, y, y, r, [&](const int b, const int sx, const int sy) {
          if (blev(b) < minlev) return;
          int x0, x1, y0, y1;
          ImageRect(b, sx, sy, r, x0, x1, y0, y1);
          if (x < x0 || x >= x1 || y < y0 || y >= y1) return;
          if (x1 > xe || (x1 == xe && y1 > ye)) {xe = x1; ye = y1;}
        });
        if (xe == x) return false;
        top = (ye < top) ? ye : top;
        x = xe;
      }
      y = top;
    }
    return true;
  }
  // true when the box widened by r is not held by the anchor's neighbourhood: blocks further
  // away may matter, so the global candidates are used (a missing block is never read as
  // "no finer leaf there")
  KOKKOS_INLINE_FUNCTION
  bool NeedGlobal(const int m, const int ia, const int ib, const int ja, const int jb,
                  const int r) const {
    int x0, x1, y0, y1;
    ImageRect(m, 0, 0, 0, x0, x1, y0, y1);
    // fast path: inside the ring around m that its neighbours certainly hold (safe)
    const int s1 = safe(m, 0), s2 = safe(m, 1);
    if (x0 - s1 <= ia - r && x1 + s1 > ib + r && y0 - s2 <= ja - r && y1 + s2 > jb + r) {
      return false;
    }
    return !Covered(m, ia - r, ib + r, ja - r, jb + r, 0, 0, false);
  }

  // the candidate blocks (block, image) of a support [lo1, hi1] x [lo2, hi2] (anchor frame) that
  // does not lie inside the anchor block m: those of the anchor's neighbourhood (or of all
  // blocks, when the neighbourhood does not hold the support) whose image box meets it.  In
  // batches of at most kBatch from cursor (0 at the start); returns the count and sets cursor to
  // the next batch, or -1.  Not inlined: the coverage test and the global
  // enumeration stay out of the hot kernels, which only loop over the batch.
  static constexpr int kBatch = 24;
  // a candidate as one int: block << 8 | (sx + 8) << 4 | (sy + 8), shifts in -8..7
  KOKKOS_INLINE_FUNCTION
  static int Pack(const int b, const int sx, const int sy) {
    return (b << 8) | ((sx + 8) << 4) | (sy + 8);
  }
  KOKKOS_INLINE_FUNCTION
  static int UnpackSx(const int c) {return ((c >> 4) & 15) - 8;}
  KOKKOS_INLINE_FUNCTION
  static int UnpackSy(const int c) {return (c & 15) - 8;}
  KOKKOS_FUNCTION DUST_NOINLINE
  int Candidates(const Real lo1, const Real hi1, const Real lo2, const Real hi2, const int m,
                 int &cursor, bool &global, int *cand) const {
    const int ia = static_cast<int>(floor((lo1 - x1min)/fine_dx1));
    const int ib = static_cast<int>(ceil((hi1 - x1min)/fine_dx1)) - 1;
    const int ja = static_cast<int>(floor((lo2 - x2min)/fine_dx2));
    const int jb = static_cast<int>(ceil((hi2 - x2min)/fine_dx2)) - 1;
    auto meets = [&](const int b, const int sx, const int sy) {
      const Real bx0 = box(b, 0) - sx*lx, by0 = box(b, 1) - sy*ly;
      return !(hi1 <= bx0 || lo1 >= bx0 + nx1*box(b, 2) ||
               hi2 <= by0 || lo2 >= by0 + nx2*box(b, 3));
    };
    int n = 0;
    if (cursor == 0) {
      global = NeedGlobal(m, ia, ib, ja, jb, 0);
      if (!global) {
        ForCandidates(m, false, ia, ib, ja, jb, 0, [&](const int b, const int sx, const int sy) {
          if (!meets(b, sx, sy)) return;
          cand[n++] = Pack(b, sx, sy);   // at most 1 + kMaxNb <= kBatch
        });
        cursor = -1;
        return n;
      }
    }
    // all blocks, every image that meets the support's finest cells (one cell wider against
    // round-off), from block cursor - 1 on; a block's images never straddle two batches
    int b = (cursor > 0) ? cursor - 1 : 0;
    for (; b<nall; ++b) {
      int kx0, kx1, ky0, ky1;
      ImageRange(fbox(b, 0), fbox(b, 1), 0, nfine1, ia - 1, ib + 1, kx0, kx1);
      ImageRange(fbox(b, 2), fbox(b, 3), 0, nfine2, ja - 1, jb + 1, ky0, ky1);
      const int nimg = (kx1 >= kx0 && ky1 >= ky0) ? (kx1 - kx0 + 1)*(ky1 - ky0 + 1) : 0;
      if (nimg > kBatch || kx0 < -7 || kx1 > 8 || ky0 < -7 || ky1 > 8) {
        Kokkos::abort("LeafTraversal::Candidates: a support meets too many periodic images");
      }
      if (n + nimg > kBatch) break;
      for (int ky=ky0; ky<=ky1; ++ky) {
        for (int kx=kx0; kx<=kx1; ++kx) {
          if (!meets(b, -kx, -ky)) continue;
          // a block without leaf fields here (beyond the halo of this rank): the weights would be
          // lost
          if (b >= nfld) Kokkos::abort("LeafTraversal::Candidates: a cloud reaches beyond the halo");
          cand[n++] = Pack(b, -kx, -ky);
        }
      }
    }
    cursor = (b < nall) ? b + 1 : -1;
    return n;
  }

  // every leaf cell under the cloud `code` at (x1, x2), anchor block m: function(target, w,
  // m1x, m1y) with kMoments (moments about the particle), function(target, w) without.
  // Returns the sum of the weights.  The callback appears once, in loops that are not
  // unrolled (code size of the hot kernels).
  template <bool kMoments, typename Function>
  KOKKOS_INLINE_FUNCTION
  Real ForEach(const Real px1, const Real px2, const int code, const int m,
               const Function &function, bool *reach = nullptr) const {
    using T = SMRCompositeTransfer;
    Real x1 = px1, x2 = px2;
    Frame(m, x1, x2);
    const int lev = T::CloudLevel(code);
    const bool point = kMoments && (code & T::kCloudGatherPoint) != 0;
    const int lb1 = T::CloudLobe1(code), lb2 = T::CloudLobe2(code);
    const Shape s1 = MakeShape(x1, ldexp(root_dx1, -lev), ldexp(iroot_dx1, lev), lb1, point);
    const Shape s2 = MakeShape(x2, ldexp(root_dx2, -lev), ldexp(iroot_dx2, lev), lb2, point);
    // a support inside the anchor block meets only its cells (the common case)
    const Real ax0 = box(m, 0), ay0 = box(m, 1);
    const bool inside = (s1.lo >= ax0 && s1.hi <= ax0 + nx1*box(m, 2) &&
                         s2.lo >= ay0 && s2.hi <= ay0 + nx2*box(m, 3));
    int cand[kBatch];
    int ncand = 1, cursor = -1;
    bool global = false;
    cand[0] = Pack(m, 0, 0);
    if (!inside) {
      cursor = 0;
      ncand = Candidates(s1.lo, s1.hi, s2.lo, s2.hi, m, cursor, global, cand);
    }
    if (reach != nullptr) *reach = !global;
    Real wsum = 0.0;
    while (true) {
#pragma unroll 1
      for (int k=0; k<ncand; ++k) {
        const int b = cand[k] >> 8;
        const Real dx = box(b, 2), dy = box(b, 3);
        const Real bx0 = box(b, 0) - UnpackSx(cand[k])*lx, by0 = box(b, 1) - UnpackSy(cand[k])*ly;
        const Real idx = box(b, 4), idy = box(b, 5);   // 1/dx, 1/dy
        int i0 = static_cast<int>(floor((s1.lo - bx0)*idx));
        int i1 = static_cast<int>(ceil((s1.hi - bx0)*idx)) - 1;
        int j0 = static_cast<int>(floor((s2.lo - by0)*idy));
        int j1 = static_cast<int>(ceil((s2.hi - by0)*idy)) - 1;
        i0 = (i0 < 0) ? 0 : i0; i1 = (i1 > nx1 - 1) ? nx1 - 1 : i1;
        j0 = (j0 < 0) ? 0 : j0; j1 = (j1 > nx2 - 1) ? nx2 - 1 : j1;
        // the x1 integrals in runs of kRun cells, in registers across the rows
        constexpr int kRun = 4;
#pragma unroll 1
        for (int ic=i0; ic<=i1; ic+=kRun) {
          const int nr = (i1 - ic + 1 < kRun) ? i1 - ic + 1 : kRun;
          Real w1[kRun], g1[kRun];
          Face f1 = FaceAt<kMoments>(s1, bx0 + ic*dx);
#pragma unroll
          for (int a=0; a<kRun; ++a) {
            w1[a] = 0.0; g1[a] = 0.0;
            if (a < nr) {
              const Face fn = FaceAt<kMoments>(s1, bx0 + (ic + a + 1)*dx);
              Cell<kMoments>(s1, f1, fn, w1[a], g1[a]);
              f1 = fn;
            }
          }
          Face f2 = FaceAt<kMoments>(s2, by0 + j0*dy);
#pragma unroll 1
          for (int j=j0; j<=j1; ++j) {
            const Face fn = FaceAt<kMoments>(s2, by0 + (j + 1)*dy);
            Real w2, g2;
            Cell<kMoments>(s2, f2, fn, w2, g2);
            f2 = fn;
            if (w2 == 0.0 && (!kMoments || g2 == 0.0)) continue;
            const int row = (b*ncells2 + j + js)*ncells1 + ic + is;
#pragma unroll 1
            for (int a=0; a<nr; ++a) {
              // the run's values by selection, so that they stay in registers
              const Real wa = (a == 0) ? w1[0] : ((a == 1) ? w1[1] : ((a == 2) ? w1[2] : w1[3]));
              const Real w = w2*wa;
              wsum += w;
              if constexpr (kMoments) {
                const Real ga = (a == 0) ? g1[0] : ((a == 1) ? g1[1] : ((a == 2) ? g1[2] : g1[3]));
                const Real m1x = w2*ga, m1y = wa*g2;
                if (w == 0.0 && m1x == 0.0 && m1y == 0.0) continue;
                function(row + a, w, m1x, m1y);
              } else {
                if (w == 0.0) continue;
                function(row + a, w);
              }
            }
          }
        }
      }
      if (cursor < 0) break;
      ncand = Candidates(s1.lo, s1.hi, s2.lo, s2.hi, m, cursor, global, cand);
    }
    return wsum;
  }

  // encoded leaf cell that holds (x1, x2), searched from the anchor block m (its own box, its
  // neighbours, then every block); -1 if no block holds it.  *bout: the block.  A position on a
  // block face can round outside both blocks (y = -1.7e-17 against a face at 0: (y - y0)/dy is
  // exactly nx2 below and negative above), so a last pass accepts 1e-9 cells around the blocks
  // and clamps the cell (the table's finest-cell index always gave a leaf).
  // The index of logical locations.  A block of level l at logical (lx1, lx2)
  // (its finest-cell box over the block width in finest cells) has the key LocKey(l, lx1, lx2);
  // lockey holds the keys of all blocks sorted, locblk the block of each.  The leaf cell with
  // the finest-cell index (ic, jc) (periodic) is found by trying l = 0, 1, ... (the leaves tile
  // the domain, one level matches), each a binary search: no grid, memory O(blocks).
  KOKKOS_INLINE_FUNCTION
  static std::int64_t LocKey(const int l, const int lx1, const int lx2) {
    return (static_cast<std::int64_t>(l) << 56) | (static_cast<std::int64_t>(lx2) << 28) |
           static_cast<std::int64_t>(lx1);
  }
  KOKKOS_INLINE_FUNCTION
  int Locate(const int ic_in, const int jc_in, int *bout = nullptr) const {
    const int ic = ((ic_in % nfine1) + nfine1) % nfine1, jc = ((jc_in % nfine2) + nfine2) % nfine2;
    for (int l=0; l<=nlev; ++l) {
      const int sh = nlev - l, w1 = nx1 << sh, w2 = nx2 << sh;
      const std::int64_t key = LocKey(l, ic/w1, jc/w2);
      int lo = 0, hi = nloc;
      while (lo < hi) {
        const int mid = (lo + hi)/2;
        if (lockey(mid) < key) lo = mid + 1; else hi = mid;
      }
      if (lo < nloc && lockey(lo) == key) {
        const int b = locblk(lo);
        if (bout != nullptr) *bout = b;
        const int i = (ic - fbox(b, 0)) >> sh, j = (jc - fbox(b, 2)) >> sh;
        return (b*ncells2 + j + js)*ncells1 + i + is;
      }
    }
    return -1;
  }
  // The search beyond the neighbours is the index of logical locations
  // (Locate, O(levels log blocks)); the scan of all blocks remains for a structure without it.
  KOKKOS_INLINE_FUNCTION
  int Owner(const Real px1, const Real px2, const int m, int *bout = nullptr) const {
    Real x1 = px1, x2 = px2;
    Frame(m, x1, x2);
    int res = -1, bb = -1;
    const int ic = static_cast<int>(floor((x1 - x1min)/fine_dx1));
    const int jc = static_cast<int>(floor((x2 - x2min)/fine_dx2));
    for (int pass=0; pass<3 && res < 0; ++pass) {
      if (pass == 1 && nloc > 0) {
        res = Locate(ic, jc, &bb);
        break;
      }
      const Real tol = (pass == 2) ? 1.0e-9 : 0.0;
      ForCandidates(m, pass >= 1, ic - 1, ic + 1, jc - 1, jc + 1, 0,
                    [&](const int b, const int sx, const int sy) {
        if (res >= 0) return;
        const Real fi = (x1 - (box(b, 0) - sx*lx))/box(b, 2);
        const Real fj = (x2 - (box(b, 1) - sy*ly))/box(b, 3);
        if (!(fi >= -tol && fi < nx1 + tol && fj >= -tol && fj < nx2 + tol)) return;
        int i = static_cast<int>(floor(fi)), j = static_cast<int>(floor(fj));
        i = (i < 0) ? 0 : ((i > nx1 - 1) ? nx1 - 1 : i);
        j = (j < 0) ? 0 : ((j > nx2 - 1) ? nx2 - 1 : j);
        res = (b*ncells2 + j + js)*ncells1 + i + is;
        bb = b;
      });
    }
    if (bout != nullptr) *bout = bb;
    return res;
  }

  // ---- split / merge eligibility ---------------------------------------------------------
  // the region of the table's level map above lev (see the members), on the anchor's
  // neighbourhood or, when that does not hold the box widened by the dilation, on all blocks
  KOKKOS_INLINE_FUNCTION
  bool AnyAbove(const int m, const int ia, const int ib, const int ja, const int jb,
                const int lev, bool *reach = nullptr) const {
    if (lev + 1 > nlev) return false;
    const int r = rdil[lev + 1];
    const bool g = NeedGlobal(m, ia, ib, ja, jb, r);
    if (reach != nullptr) *reach = !g;
    // no candidate as fine as lev + 1: Meets finds nothing
    if (!g && m < nmb && lev + 1 > nbmax(m) && ia <= ib && ja <= jb) return false;
    return Meets(m, ia, ib, ja, jb, lev + 1, r, g);
  }
  KOKKOS_INLINE_FUNCTION
  bool AllAbove(const int m, const int ia, const int ib, const int ja, const int jb,
                const int lev, bool *reach = nullptr) const {
    if (lev + 1 > nlev) return false;
    const int r = rdil[lev + 1];
    const bool g = NeedGlobal(m, ia, ib, ja, jb, r);
    if (reach != nullptr) *reach = !g;
    // no candidate as fine as lev + 1: a non-empty box is not covered
    if (!g && m < nmb && lev + 1 > nbmax(m) && ia <= ib && ja <= jb) return false;
    return Covered(m, ia, ib, ja, jb, lev + 1, r, g);
  }
  // SMRCompositeTransfer::CloudInsideFiner on the actual leaves (anchor block m)
  KOKKOS_INLINE_FUNCTION
  bool InsideFiner(const Real px1, const Real px2, const int code, const Real slack, const int m,
                   bool *reach = nullptr) const {
    using T = SMRCompositeTransfer;
    if (reach != nullptr) *reach = true;
    if (T::CloudLobe1(code) || T::CloudLobe2(code)) return false;
    const int lev = code & T::kCloudLevelMask;
    if (lev >= nlev || lev >= cloud_max_level) return false;
    if (split_everywhere) return true;
    Real x1 = px1, x2 = px2;
    Frame(m, x1, x2);
    const Real r1 = (1.0 - slack)*root_dx1/static_cast<Real>(1 << lev);
    const Real r2 = (1.0 - slack)*root_dx2/static_cast<Real>(1 << lev);
    const int ia = static_cast<int>(floor((x1 - r1 - x1min)/fine_dx1));
    const int ib = static_cast<int>(ceil((x1 + r1 - x1min)/fine_dx1)) - 1;
    const int ja = static_cast<int>(floor((x2 - r2 - x2min)/fine_dx2));
    const int jb = static_cast<int>(ceil((x2 + r2 - x2min)/fine_dx2)) - 1;
    return split_overlap ? AnyAbove(m, ia, ib, ja, jb, lev, reach)
                         : AllAbove(m, ia, ib, ja, jb, lev, reach);
  }
  // SMRCompositeTransfer::CloudOverlapsFiner on the actual leaves
  KOKKOS_INLINE_FUNCTION
  bool OverlapsFiner(const Real px1, const Real px2, const int code, const int m,
                     bool *reach = nullptr) const {
    if (reach != nullptr) *reach = true;
    const int lev = SMRCompositeTransfer::CloudLevel(code);
    if (lev >= nlev) return false;
    Real x1 = px1, x2 = px2;
    Frame(m, x1, x2);
    const Real r1 = root_dx1/static_cast<Real>(1 << lev), r2 = root_dx2/static_cast<Real>(1 << lev);
    const int ia = static_cast<int>(floor((x1 - r1 - x1min)/fine_dx1));
    const int ib = static_cast<int>(ceil((x1 + r1 - x1min)/fine_dx1)) - 1;
    const int ja = static_cast<int>(floor((x2 - r2 - x2min)/fine_dx2));
    const int jb = static_cast<int>(ceil((x2 + r2 - x2min)/fine_dx2)) - 1;
    return AnyAbove(m, ia, ib, ja, jb, lev, reach);
  }
  // the merge-cell mask of MergeCloudsDevice: the level-L cell holding (x1, x2) and the B
  // cells around it carry no dilated level above L
  KOKKOS_INLINE_FUNCTION
  bool MergeOK(const Real px1, const Real px2, const int L, const int B, const int m,
               bool *reach = nullptr) const {
    Real x1 = px1, x2 = px2;
    Frame(m, x1, x2);
    const int r = 1 << (nlev - L);
    const int I = static_cast<int>(floor((x1 - x1min)/(root_dx1/static_cast<Real>(1 << L))));
    const int J = static_cast<int>(floor((x2 - x2min)/(root_dx2/static_cast<Real>(1 << L))));
    if (reach != nullptr) *reach = true;
    return !AnyAbove(m, (I - B)*r, (I + B + 1)*r - 1, (J - B)*r, (J + B + 1)*r - 1, L, reach);
  }

  // ---- the cloud interface of SMRCompositeTransfer on the actual
  // leaves, with the anchor block m of the cloud (its particle's MeshBlock; a new cloud of a
  // split or merge takes its parent's).  The results do not depend on the anchor; a good
  // anchor only keeps the queries local.
  // the anchor of a particle's MeshBlock id; an id outside the pack (e.g. before the ids are
  // set again after a regrid) gives block 0, and the queries then fall back to all blocks
  KOKKOS_INLINE_FUNCTION
  Real CloudHalfWidth1(const int lev) const {
    return root_dx1/static_cast<Real>(1 << (lev & SMRCompositeTransfer::kCloudLevelMask));
  }
  KOKKOS_INLINE_FUNCTION
  Real CloudHalfWidth2(const int lev) const {
    return root_dx2/static_cast<Real>(1 << (lev & SMRCompositeTransfer::kCloudLevelMask));
  }
  KOKKOS_INLINE_FUNCTION
  int Anchor(const int pgid) const {
    const int m = pgid - gids;
    return (m >= 0 && m < nmb) ? m : 0;
  }
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachCloudWeight(const Real x1, const Real x2, const int code, const int m,
                          const Function &function) const {
    ForEach<false>(x1, x2, code, m, function);
  }
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachCloudMoment(const Real x1, const Real x2, const int code, const int m,
                          const Function &function) const {
    ForEach<true>(x1, x2, code, m, function);
  }
  KOKKOS_INLINE_FUNCTION
  bool CloudInsideFiner(const Real x1, const Real x2, const int code, const Real slack,
                        const int m) const {
    return InsideFiner(x1, x2, code, slack, m);
  }
  KOKKOS_INLINE_FUNCTION
  bool CloudOverlapsFiner(const Real x1, const Real x2, const int code, const int m) const {
    return OverlapsFiner(x1, x2, code, m);
  }
  KOKKOS_INLINE_FUNCTION
  int LeafOwner(const Real x1, const Real x2, const int m) const {return Owner(x1, x2, m);}
};

// the name of a per-rank log file: unchanged on one rank, name.rNNNN.ext on several
inline std::string RankFile(const std::string &name) {
  if (global_variable::nranks == 1) return name;
  char sfx[16];
  std::snprintf(sfx, sizeof(sfx), ".r%04d", global_variable::my_rank);
  const auto dot = name.rfind('.');
  return (dot == std::string::npos) ? name + sfx : name.substr(0, dot) + sfx + name.substr(dot);
}

// An append-mode log file opened once per process and kept open, instead of an
// open and close per line (a GB200 test measured about 1 ms per open/close for the sort log; for the
// trigger and event logs a GB200 test found no difference); callers write and fflush
inline std::FILE *AppendLog(const std::string &name) {
  static std::map<std::string, std::FILE*> open;
  auto it = open.find(name);
  if (it != open.end()) return it->second;
  std::FILE *fp = std::fopen(name.c_str(), "a");
  if (fp != nullptr) open[name] = fp;
  return fp;
}

//----------------------------------------------------------------------------------------
//! \struct DustHalo
//! The exchange lists of the dust halo (dust_halo.cpp), cell by cell.  Per
//! peer rank p (ascending), the cells of this rank's blocks sent to it, scell[soff[p] ..
//! soff[p+1]), and the cells of the halo slots received from it, rcell[roff[p] .. roff[p+1]),
//! as flat leaf indices (block slot, j, i); the blocks in ascending gid, each block's cells
//! row by row over its rectangle (the whole block, or with <dust>/cloud_halo_strips the
//! part that the clouds of the receiving rank can reach).  One rank: no peers.
struct DustHalo {
  int nx1 = 0, nx2 = 0, is = 0, js = 0, ncells1 = 0, ncells2 = 0;
  int hslot0 = 0, hslot1 = 0;          // the halo slots [hslot0, hslot1)
  int npeer = 0;
  unsigned long long builds = 0;
  // seconds of the exchanges, forward [0..3] and reverse [4..7]: waiting for the kernels before
  // the exchange, packing, MPI (posting, the receives, the sends), unpacking; and their counts
  double tsec[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  unsigned long long nexch[2] = {0, 0};
  std::vector<int> peer, soff, roff;
  DvceArray1D<int> scell, rcell;
  DvceArray1D<Real> sbuf, rbuf;
  long long nblk_send = 0, nblk_recv = 0;   // blocks in the lists (for the log)
  // the rectangle [i0, i1) x [j0, j1) (interior cell indices) of every halo slot that this rank
  // receives, by halo slot (slot - nmb); and the guard of the strips (test)
  std::vector<std::array<int, 4>> rect;
  DvceArray2D<int> rect_d;
  bool guard = false;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm = MPI_COMM_NULL;
  bool comm_set = false;
#endif
};

// the weights of a particle on the composite transfer, as
// SMRCompositeTransfer::ForEachWeight(x1, x2, code, f): refinable clouds on the actual leaves
// (anchor block m), the smooth transfer (smooth_p2, owner_overlap) on its own maps
template <typename Function>
KOKKOS_INLINE_FUNCTION
void ForEachCompositeWeight(const SMRCompositeTransfer &t, const LeafTraversal &lt,
                            const Real x1, const Real x2, const int code, const int m,
                            const Function &function) {
  if (t.cloud_mode) {
    lt.ForEachCloudWeight(x1, x2, code, m, function);
  } else {
    t.ForEachWeight(x1, x2, function);
  }
}

//----------------------------------------------------------------------------------------
//! \struct DustGasDragTaskIDs
//  \brief container to hold TaskIDs of all dust+hydro tasks

// the phase profile is on only with <dust>/amr_profile = true; off, the timers neither
// fence nor time (every fence makes the host wait for the device, about eight per cycle)
inline bool &ProfOn() {
  static bool on = false;
  return on;
}
// fenced timer of one phase (the kernels before are finished first; their time is not counted)
struct ProfTimer {
  double &acc;
  long long &cnt;
  std::chrono::steady_clock::time_point t0;
  ProfTimer(double &a, long long &c) : acc(a), cnt(c) {
    if (!ProfOn()) return;
    Kokkos::fence();
    t0 = std::chrono::steady_clock::now();
  }
  ~ProfTimer() {
    if (!ProfOn()) return;
    Kokkos::fence();
    acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    ++cnt;
  }
};

// the known velocity field of <dust>/test_ustar_linear: each component linear in x1 and x2
struct UstarLinearField {
  Real x0 = 0.0, y0 = 0.0, lx = 1.0, ly = 1.0;
  KOKKOS_INLINE_FUNCTION Real A(const int c) const {return (c == 0) ? 0.1 : ((c == 1) ? -0.2 : 0.05);}
  KOKKOS_INLINE_FUNCTION Real Bx(const int c) const {return ((c == 0) ? 0.3 : ((c == 1) ? -0.7 : 0.2))/lx;}
  KOKKOS_INLINE_FUNCTION Real By(const int c) const {return ((c == 0) ? -0.4 : ((c == 1) ? 0.25 : 0.6))/ly;}
  KOKKOS_INLINE_FUNCTION Real Value(const int c, const Real x, const Real y) const {
    return A(c) + Bx(c)*(x - x0) + By(c)*(y - y0);
  }
};

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

// 2026-10-20 cleanup: a key whose option was removed.  An input or
// restart file that still writes it is accepted when the value asks for the behaviour that
// remains (keeps: usually the old default); otherwise the run stops instead of silently running
// another method.  The removed paths still run in the version tagged pre-cleanup-20261020.
// The guard sits where the key used to be read; block: the input block (<dust> or <problem>).
inline void RemovedKey(ParameterInput *pin, const char *key, const bool keeps, const char *what,
                       const char *block = "dust") {
  if (!pin->DoesParameterExist(block, key) || keeps) return;
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
            << "<" << block << ">/" << key << " = " << pin->GetString(block, key) << ": " << what
            << " was removed on 2026-10-20 (it runs in the version tagged pre-cleanup-20261020)"
            << std::endl;
  std::exit(EXIT_FAILURE);
}
// the key is absent or false
inline bool KeyFalse(ParameterInput *pin, const char *key, const char *block = "dust") {
  return !pin->DoesParameterExist(block, key) || !pin->GetBoolean(block, key);
}

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

// Tried and removed: splitting each of the two production cloud
// kernels into a gather kernel and a deposit kernel that forms the leaf weights again (fewer
// registers per kernel) was 10% slower in wall time on GB200; keeping the 25-entry
// leaf-weight cache in team scratch (shared memory, teams of 128) instead of per-thread local
// arrays changed nothing measurable (277 s against 280 s).  Both gave results identical to this
// kernel on the CPU.  ncu: 128 registers, 25% occupancy, waiting on L1 data.
// The cloud kernels over particles 0..n-1 with a launch bound: 0 none, 1
// <256, 2> (at most 128 registers, default: fastest on GB200), 2 <128, 3> (at most 168), 3
// <256, 3> (at most 85); occupancy against spills.  Rixin: a
// compile-time choice, -DDUST_CLOUD_LAUNCH_BOUNDS=n (the run-time key <dust>/cloud_launch_bounds
// compiled all four variants of every cloud kernel); the key is accepted only with the compiled
// value (dust.cpp)
#ifndef DUST_CLOUD_LAUNCH_BOUNDS
#define DUST_CLOUD_LAUNCH_BOUNDS 1
#endif
#if DUST_CLOUD_LAUNCH_BOUNDS == 1
using CloudPolicy = Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<256, 2>>;
#elif DUST_CLOUD_LAUNCH_BOUNDS == 2
using CloudPolicy = Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<128, 3>>;
#elif DUST_CLOUD_LAUNCH_BOUNDS == 3
using CloudPolicy = Kokkos::RangePolicy<DevExeSpace, Kokkos::LaunchBounds<256, 3>>;
#elif DUST_CLOUD_LAUNCH_BOUNDS == 0
using CloudPolicy = Kokkos::RangePolicy<DevExeSpace>;
#else
#error "DUST_CLOUD_LAUNCH_BOUNDS must be 0, 1, 2 or 3"
#endif
template <typename Function>
void CloudParFor(const std::string &name, const int n, const Function &f) {
  Kokkos::parallel_for(name, CloudPolicy(0, n), f);
}

// cloud_order_interval: the same kernels over the particles in the order of an
// index (ord(k) for k < nidx, k itself for the particles appended since the index was built)
template <typename Function>
void CloudParForOrdered(const std::string &name, const int n,
                        const DvceArray1D<int> &ord, const int nidx, const Function &f) {
  auto g = KOKKOS_LAMBDA(const int k) {f((k < nidx) ? ord(k) : k);};
  CloudParFor(name, n, g);
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
  int cloud_gather_order = 1;         // 2: gather a linear reconstruction of the leaves
  int cloud_transition = 0;           // P1 transition clouds at split edges:
                                      // 1 gathers with the child hat, 2 with the lobed
                                      // weights and point-valued reconstructions
  int cloud_next_tag = -1;            // next PTAG for created particles (-1 = unset)
  void EnsureNextTag();               // ... largest tag + 1, once
  bool cloud_p1_split = true;         // moving-particle P1 split: edge children
                                      // separate only where the neighbour parent splits,
                                      // no merging (requires cloud_transition > 0); the
                                      // only split since 2026-10-20 (the old 3x3 one is gone)
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
  // merge passes on a physical-time schedule (cloud_merge_schedule = time,
  // cloud_merge_dt_omega) instead of every cloud_merge_interval cycles, and a burst of passes
  // every cloud_merge_burst_interval cycles for cloud_merge_burst_dt_omega / omega0 after each
  // regrid that deleted blocks
  bool cloud_merge_by_time = false;
  Real cloud_merge_dt = 0.0, cloud_merge_t0 = 0.0;
  long long cloud_merge_next = -1;
  Real cloud_merge_burst_dt = 0.0, cloud_merge_burst_until = -1.0;
  int cloud_merge_burst_interval = 50;
  bool cloud_merge_clock_restored = false;
  bool MergeDue();
  // One resampling after every regrid, once the particle work of that regrid
  // is done (the split of the next cycle and, with merging on, the first merge pass of a
  // derefinement); on by default in dynamic-mesh runs with resampling.  An event is a regrid
  // that created (nnew) or deleted (ndel) blocks, whatever the net change of the block count.
  bool cloud_resample_after_regrid = true;
  bool amr_event_pending = false;
  // For the scaling on several ranks: a regrid event may wait cloud_event_dt (code
  // time; <dust>/cloud_event_dt_omega / omega0) after the first regrid it covers; the regrids in
  // between make one event (counts summed, the blocks of all of them fresh for the local R).  0:
  // the cycle after each regrid, as before.  first_ncycle < 0: none recorded (handle at once).
  Real cloud_event_dt = 0.0;
  Real amr_event_first_time = -1.0e300;
  long long amr_event_first_ncycle = -1;
  int amr_event_max_age = 0;   // the local R's fresh blocks: cycles since their change <= this
  int amr_event_nnew = 0, amr_event_ndel = 0;  // summed if two regrids precede the handling
  int amr_event_regrid = 0;                    // regrid number (amr_regrids) of the last one
  long long amr_event_handled = 0, amr_event_merged = 0;
  int amr_regrid_nnew = 0, amr_regrid_ndel = 0;  // of the regrid in progress (BeforeRegrid)
  void HandleRegridEvent(const bool merge_due);
  bool cloud_merge_nc = true;         // merge-side node convention
  bool cloud_merge_pending_log = false;  // log why designated centres did not commit
  int cloud_merge_lobe_wait = 0;      // a centre waits while an unreleased lobed cloud is near
                                      // (1: reach H + ell, 2: reach per lobe direction)
  std::vector<Real> cloud_merge_mroot; // per species: root-level cloud mass (set on first pass)
  // quantum mode: stage X exact P1-stencil re-merge, stage G mass-quantum grouping
  // (cross-level); stage C (same-level compression) was removed on 2026-10-20
  int cloud_merge_mode = 0;           // 0 tent, 1 quantum
  bool cloud_merge_stencil = true;    // quantum: stage X on
  Real cloud_merge_stencil_tol = 0.02;  // stage X offset tolerance (units of the piece half-width)
  Real cloud_merge_smax = 1.0;        // stage G group position extent (units of the target cell)
  int cloud_merge_K = 4;              // open groups per cell walk (1..8)
  bool cloud_merge_exact_ts = true;   // merged clouds keep the common ts of their members
  Real cloud_audit_t0 = -1.0;          // merge audit (diagnostic): dump every merge pass in
  Real cloud_audit_t1 = -1.0;          // [t0, t1]; stages in full when the pass changed anything
  bool cloud_audit_compact = true;     // quiet passes: compact particle + gas snapshot
  bool cloud_merge_xprotect = false;   // stage X's waiting candidates are not
                                       // walked by G in the same pass
  Real cloud_merge_budget = -1.0;      // > 0, one budget of leaf mass / momentum
                                       // change for the whole pass (stage G; X with budget_x)
  Real cloud_absorb_floor = 0.1;       // ... with the leaf mass floored at this fraction of a
                                       // root cloud mass per root-cell volume (the name is
                                       // from the absorption budget)
  int cloud_merge_budget_evals = 4;    // ... evaluations per stage before the rest is rejected
  bool cloud_audit_rejected = false;   // full audit file when G groups were budget-rejected
  bool cloud_merge_budget_x = false;   // stage X is also decided by the pass budget
  bool cloud_audit_only = false;       // a merge pass writes its entry audit and returns
  bool cloud_resample = false;         // conservative same-level resampling in the code
  int cloud_resample_interval = 625;   // ... cycles between resamplings
  Real cloud_resample_eta = 2.0;       // ... mass cap eta m0 / 4^level (<= 0: no cap)
  int cloud_resample_patch = 2;        // ... patches inside one MeshBlock, of
                                       //     this many root cells per side on the root level
  int cloud_resample_patch_cells = 4;  // ... at most this many cells of the block per side
  int cloud_resample_patch_split = 1024;   // ... a patch (or quadrant) with more pool
                                      //     clouds is split into quadrants, down to cells (0: never)
  int cloud_resample_threads = 1;      // ... host threads of the patch loop
  bool cloud_resample_device = true;   // ... the patch solves on the device (default)
  bool cloud_resample_qr_rows = true;  // ... row-parallel QR update on the device
                                       //     (matches the host only to round-off)
  int cloud_resample_device_mem_mb = 2048;  // ... workspace limit of the device solves
  int cloud_resample_device_mid = 0;   // ... patches of 65..mid clouds in a middle
  int cloud_resample_device_mid_team = 128;   //     bucket with teams of this many threads (0: off)
  int cloud_resample_device_wide_team = 512;  // ... large-bucket team (1024: a 1024-thread kernel)
  bool cloud_resample_compare = false; // ... diagnostic: host and device on the same pool, compared
  bool cloud_resample_patch_stats = false;   // diagnostic: patches by level and size
  bool cloud_resample_feedback_check = false;   // ... diagnostic: AuditFeedback before / after
  int cloud_resample_stress = 0;       // ... diagnostic: coincident copies, every N-th
  int cloud_resample_stress_copies = 0;   //     candidate; K copies of the first one
  bool cloud_resample_stress_done = false;
  bool cloud_resample_tag_order = false;   // ... pool in PTAG order (else array order)
  bool cloud_resample_pos_order = false;   // ... pool in position order instead
  void StressResamplePool();
  Real cloud_resample_m0 = -1.0;       // ... root cloud mass (<= 0: the merge pass's value)
  std::vector<int> cloud_resample_dump;     // cycles whose resampling writes the particle table before and after
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
  int cloud_merge_pass = 0;           // pass counter (walk rotation)
  unsigned long long cloud_coarsen_events = 0, cloud_coarsen_in = 0;
  unsigned long long cloud_coarsen_reject = 0, cloud_far_release = 0;
  Real cloud_coarsen_dk_grad = 0.0, cloud_coarsen_dk_res = 0.0;
  int cloud_split_buffer = 0;         // refined dust region reaches this many next-coarser
                                      // cells beyond the refined gas
  DvceArray1D<int> split_np, split_off;  // device split work arrays, kept between cycles
  bool cloud_split_enable = true;     // test only: false = no splitting at all
  // Instant switch, the OO control on the cloud framework: at the
  // start of every cycle (and in the initial split) each cloud takes the level of the leaf that
  // holds its centre; no split, merge or resampling, fixed particle number and masses
  bool cloud_instant_switch = false;
  unsigned long long cloud_switch_count = 0;   // level changes made by the instant switch
  bool cloud_split_device = false;    // P1 split on the device; the split log,
                                      // and split_oracle keep the host path
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
  // cloud_merge_count: coincident children merged by the old 3x3 split (removed 2026-10-20);
  // stays 0, kept in the restart record and the summary line
  unsigned long long cloud_split_count = 0, cloud_merge_count = 0;
  unsigned long long cloud_lobe_split_count = 0;
  // the switching metric (the cell feedback stiffness, its reduction and the
  // stiffness deposit in qdep[4]) is read only by the automatic PC2 / split-BE switch
  bool NeedsHybridMetric() const {
    return coupling == DustCoupling::hybrid && hybrid_force_mode == HybridForceMode::automatic;
  }
  // Codex review: with a fixed mode the cached metric (from a restart written by
  // an automatic run, say) is marked invalid, so that a later automatic run never starts from it
  void InvalidateHybridMetric() {
    hybrid_have_metric = false;
    hybrid_feedback_rate_max = 0.0;
    hybrid_last_zeta = 0.0;
    hybrid_last_chi = 0.0;
  }
  bool UsesCompositeTransfer() const {
    return (smr_transfer_mode == DustSMRTransfer::smooth_p2 ||
            smr_transfer_mode == DustSMRTransfer::refinable_cloud);
  }
  // u* is exchanged into the ghost cells for the TSC gather and for the linear cloud gather
  // (central differences); the piecewise-constant composite gathers read leaf cells only
  bool UstarGhosts() const {
    return !UsesCompositeTransfer() ||
           (smr_transfer_mode == DustSMRTransfer::refinable_cloud && cloud_gather_order == 2);
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
  void BeforeRegrid(int nnew, int ndel);  // particle count and checksum before the mesh changes
  // For the scaling on several ranks: <dust>/load_balance_cloud_weight w > 0 gives the
  // new blocks of a regrid the cost 1 + w * clouds / (mean clouds per block) in AthenaK's load
  // balance (otherwise every block costs 1).  The clouds of every old block are counted in
  // BeforeRegrid (all ranks); RegridCosts maps them to the new blocks (a refined block passes 1/4
  // of its clouds to each child, a derefined parent takes the sum of its children).
  Real lb_cloud_weight = 4.0;   // Rixin: default 4
  std::vector<float> lb_old_counts;
  void RegridCosts(const int *newtoold, const LogicalLocation *new_lloc, const int new_nmb, float *cost);
  int ResetPGIDFromPositions(const char *when);  // refinable clouds
  void RebuildAfterRegrid(Driver *pdrive, ParameterInput *pin);
  // The old owner_overlap transfer on a dynamic mesh (dust_amr.cpp).  The
  // block boxes (leaf_traversal: boxes only, read by the density trigger), the finest-block table
  // of the block of a position and the ghost-to-leaf remaps of the deposit exchanges, for the
  // current mesh; the block of every particle from its position; the TSC deposit on the owner
  // blocks with its ghost parts added to the leaves they overlap (density, and with moments the
  // momentum and the raw second moments: 7 components), checked against the particle mass
  void BuildOwnerOverlapMesh(const bool remap);
  int ResetPGIDOwnerOverlap(const char *when);
  void OwnerOverlapDeposit(DvceArray5D<Real> &a, const bool moments);
  DvceArray2D<int> oo_block_map;
  DvceArray5D<Real> oo_rho;
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
  int SwitchCloudsToLeafLevel();   // instant switch; returns the clouds changed
  int SplitCloudsP1();         // moving-particle P1 split
  int SplitCloudsP1Device();   // device P1 split without the diagnostics
  TaskStatus RefineClouds(Driver *pdrive, int stage);
  int MergeClouds();
  int MergeCloudsDevice();    // device version of MergeClouds
  void AuditFeedback(DvceArray2D<Real> &fb);   // frozen-gas predictor feedback
  int ResampleClouds(const bool local = false);   // returns the number of clouds removed;
                              // local: only clouds anchored in blocks new since the
                              // regrid and their neighbours
  bool cloud_resample_event_local = true;    // on by default: the resampling
                                             // after a regrid is local (the scheduled one global)
  bool cloud_resample_after_derefine = false;   // R after a pure derefinement event
  bool ResampleDue();         // is a resampling due at the start of this cycle?
  void PackGatherField(const DvceArray5D<Real> &vel);   // fills gpack from a velocity array
  void PackGasGather(const DvceArray5D<Real> &u);       // ... from a conserved state
  // block-local traversal of the actual leaves (dust_leaf.cpp)
  LeafTraversal leaf_traversal;
  int cloud_leaf_check = 0;           // diagnostic: every this many cycles, leaf vs table
  // timing diagnostic: storage order of the particles
  int cloud_sort_interval = 0;        // reorder every this many cycles (0: never)
  bool cloud_sort_random = false;     // diagnostic: a random permutation instead
  bool cloud_merge_position_order = false;   // diagnostic: G walk takes pieces by position
  DvceArray2D<Real> sort_spare_r;     // the arrays the reordered particles are written to
  DvceArray2D<int> sort_spare_i;
  void SortParticles();
  // an index of the particles in (block, cell) order for the two production cloud
  // kernels, without moving them (cloud_order_interval = N: rebuilt every N cycles and whenever
  // the check at the end of RefineClouds fails; 0: off; default 8 on the production PC2 path)
  int cloud_order_interval = 0;
  DvceArray1D<int> cloud_ord, cloud_ord_tag;
  int cloud_ord_n = 0;
  unsigned long long cloud_ord_builds = 0;
  void UpdateCloudOrder();
  void BuildLeafTraversal();
  // multi-rank refinable clouds (<dust>/experimental_mpi): the gid of every block
  // slot of the leaf traversal, the halo lists and the exchanges (dust_halo.cpp)
  bool experimental_mpi = false;
  // test (Codex review): u* set to a known linear field before its ghost
  // exchange; the packed gather field on the leaf cells and every cloud's linear gather are compared
  // with the exact values (CheckUstarLinear).  skip_exchange (negative control, one rank): no u*
  // ghost exchange
  bool test_ustar_linear = false, test_ustar_skip_exchange = false;
  // the buffer of the density trigger (amr_rho_map = block), cells of a block's level: part of the
  // reach of the halo, so that the strips hold the buffer cells (Codex review)
  int halo_trigger_cells = 0;
  // profile: fenced wall-clock seconds and calls of the AMR and cloud-management
  // phases, summed over the run (rank 0's), printed as one DUST_AMR_PROFILE line at the end
  enum ProfPhase {kProfTrigger, kProfAmr, kProfRegridMesh, kProfRebuild, kProfRefineClouds, kProfSplit,
                  kProfEvent, kProfSchedR, kProfSchedG, kProfSort, kProfOrder,
                  kProfTrigDeposit, kProfTrigCopy, kProfTrigBlocks, kProfTrigLog,
                  kProfN};   // kProfTrig*: trigger parts
  double prof_time[kProfN] = {0.0};
  long long prof_count[kProfN] = {0};
  void CheckUstarLinear(const int stage);
  std::vector<int> leaf_slot_gid;
  DvceArray1D<int> leaf_slot_gid_d, leaf_slot_rank_d;   // gid and rank of every block slot
  DustHalo halo_;
  // work arrays of the density trigger kept between checks (allocating and freeing
  // device arrays in every check synchronizes the device); reallocated when a size changes
  DvceArray2D<Real> trig_f;
  DvceArray1D<Real> trig_q, trig_area;
  DvceArray1D<int> trig_bad, trig_fine;
  DvceArray2D<int> trig_rect;
  void BuildHalo(const std::vector<int> &slot_gid);
  void CheckMultiRank();
  void MigrateClouds();
  void FrameToBlock();
  // the reach of the halo per block of this rank (finest cells): the level rule
  // (HaloReach), the widenings found by CheckHaloReach (kept until the mesh changes), and the
  // furthest a support may extend beyond each block
  int cloud_halo_coarser = -1;         // test: levels a cloud may be coarser than its block (-1: auto)
  int cloud_halo_test_shrink = 0;      // test: the rule's reach divided by 2^this (forces widenings)
  bool cloud_halo_strips = true;       // exchange only the reachable cells of each block
  bool cloud_halo_strip_guard = false; // test: NaN outside the strips before forward, zero check before reverse
  std::vector<std::int64_t> halo_need_, halo_reach_, halo_allow_;
  int halo_need_seq_ = -1;
  DvceArray1D<std::int64_t> halo_allow_d_;
  unsigned long long halo_widenings_ = 0;
  void HaloReach(const int lev, const std::int64_t need, std::int64_t &reach, std::int64_t &allow) const;
  bool CheckHaloReach(const char *when);
  int SplitToLeaves();   // the initial split of a problem generator
  int RouteToOwnerBlocks(const char *when);   // after a regrid / restart
  // several ranks: seconds and calls of 0 MigrateClouds, 1 the migration after the
  // push, 2 the kernels these waited for (a fence before each); the halo exchanges keep their
  // own split (DustHalo::tsec)
  double mpi_time[4] = {0.0, 0.0, 0.0, 0.0};
  unsigned long long mpi_count[4] = {0, 0, 0, 0};
  int cloud_particle_dump = -1;       // diagnostic: cycle at whose start the particles are written
  void DumpParticles();
  void HaloForwardPack(DvceArray5D<Real> &gp);
  void HaloReverseAdd(DvceArray5D<Real> &f, const int c0, const int nc);
  void HaloForwardFlat(DvceArray2D<Real> &f, const int nc);
  void HaloReverseAddFlat(DvceArray2D<Real> &f, const int nc);
  void HaloForwardFlags(DvceArray1D<int> &f);
  void HaloReverseAddFlags(DvceArray1D<int> &f);
  void RequireCloudTable(const char *what) const;   // fatal without the table
  void EnsureRootCloudMass();          // cloud_merge_mroot once (merge pass, resampling caps)
  void CheckLeafTraversal();
  void CheckLeafEligibility();        // called by CheckLeafTraversal

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
int ParticleRestartSections(const char *head);
void UnpackParticleRestart(MeshBlockPack *pmbp, const int nsec,
                           const std::function<void(int, std::vector<char>&)> &read);

} // namespace dust
#endif // DUST_DUST_HPP_
