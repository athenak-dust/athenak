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

  bool enabled = false;
  bool cloud_mode = false;
  int cloud_nlev = 0;                 // max_level - root_level
  int cloud_max_level = 0;            // clouds are never split beyond this level
  bool cloud_split_everywhere = false; // control: split regardless of the local mesh
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
  KOKKOS_INLINE_FUNCTION
  static int CloudWeights1D(const Real x, const Real ell, const Real xmin, const Real dx,
                            int &i0, Real w[kMaxCloudStencil]) {
    Real inv_ell = 1.0/ell;
    i0 = static_cast<int>(floor((x - ell - xmin)/dx));
    int i1 = static_cast<int>(ceil((x + ell - xmin)/dx));
    int count = i1 - i0;
    if (count > kMaxCloudStencil) count = kMaxCloudStencil;
    Real prev = HatCDF((xmin + i0*dx - x)*inv_ell);
    for (int a=0; a<count; ++a) {
      Real next = HatCDF((xmin + (i0 + a + 1)*dx - x)*inv_ell);
      w[a] = next - prev;
      prev = next;
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
  KOKKOS_INLINE_FUNCTION
  static int CloudWeightsMoments1D(const Real x, const Real ell, const Real xmin,
                                   const Real dx, int &i0, Real w[kMaxCloudStencil],
                                   Real g[kMaxCloudStencil]) {
    Real inv_ell = 1.0/ell;
    i0 = static_cast<int>(floor((x - ell - xmin)/dx));
    int i1 = static_cast<int>(ceil((x + ell - xmin)/dx));
    int count = i1 - i0;
    if (count > kMaxCloudStencil) count = kMaxCloudStencil;
    Real s = (xmin + i0*dx - x)*inv_ell;
    Real prev = HatCDF(s), gprev = HatMoment(s);
    for (int a=0; a<count; ++a) {
      s = (xmin + (i0 + a + 1)*dx - x)*inv_ell;
      Real next = HatCDF(s), gnext = HatMoment(s);
      w[a] = next - prev;
      g[a] = ell*(gnext - gprev);
      prev = next;
      gprev = gnext;
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
    return root_dx1/static_cast<Real>(1 << lev);
  }
  KOKKOS_INLINE_FUNCTION
  Real CloudHalfWidth2(const int lev) const {
    return root_dx2/static_cast<Real>(1 << lev);
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
    int n1 = CloudWeights1D(x1, CloudHalfWidth1(lev), x1min, fine_dx1, i0, w1);
    int n2 = CloudWeights1D(x2, CloudHalfWidth2(lev), x2min, fine_dx2, j0, w2);
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
  // entirely on leaves finer than lev, i.e. the particle may be split.
  KOKKOS_INLINE_FUNCTION
  bool CloudInsideFiner(const Real particle_x1, const Real particle_x2, const int lev,
                        const Real slack) const {
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
    int n1 = CloudWeightsMoments1D(x1, CloudHalfWidth1(lev), x1min, fine_dx1, i0, w1, g1);
    int n2 = CloudWeightsMoments1D(x2, CloudHalfWidth2(lev), x2min, fine_dx2, j0, w2, g2);
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
                          const int lev) const {
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
  int cloud_next_tag = -1;            // next PTAG for created particles (-1 = unset)
  unsigned long long cloud_split_count = 0, cloud_merge_count = 0;
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
  int SplitAndMergeClouds();   // returns the number of clouds split
  TaskStatus RefineClouds(Driver *pdrive, int stage);

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

} // namespace dust
#endif // DUST_DUST_HPP_
