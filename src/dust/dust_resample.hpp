#ifndef DUST_DUST_RESAMPLE_HPP_
#define DUST_DUST_RESAMPLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_resample.hpp
//! \brief Shared pieces of the same-level resampling of refinable clouds: the
//! tolerances of the per-patch algorithm, the pool table columns, and the device solver of
//! dust_resample_device.cpp, which runs the algorithm of the host
//! ResamplePatch on the device.

#include <limits>
#include "athena.hpp"

namespace dust {
namespace resample {

constexpr double kRankTol = 1.0e-13;   // relative size of a pivot column that counts as zero
constexpr double kResMax = 1.0e-10;    // largest accepted relative row residual of a patch
constexpr double kCapTol = 1.0e-12;
constexpr double kPoolTol = 1.0e-9;    // clouds with m <= cap (1 + kPoolTol) are in the pool
constexpr int kNCand = 8;              // null vectors looked at per step
constexpr double kInf = std::numeric_limits<double>::infinity();

// columns of the pool table (one row per pool cloud)
enum PoolColumn {JX=0, JY=1, JM=2, JVX=3, JVY=4, JVZ=5, JTS=6, JUX=7, JUY=8, JUZ=9, JCAP=10, NPD=11};

// per-patch counters returned by the device solver (columns of outi)
enum PatchCount {PN=0, PKEPT=1, PNCAP=2, PFALLBACK=3, PZERO=4, PCAPSTEP=5, PSCANS=6, PNROW=7, PNB=8,
                 NPC=9};

// The pool sorted by patch (k = 0..npool-1, patch q holds k in [pptr(q), pptr(q+1))): rows
// spd (npool x NPD), CSR weights sptr / stgt / swgt in the same order.  New masses in mnew
// (sorted order), counters in outi (npatch x NPC) and the accepted residual in outr (npatch).
// Patch q is centred at (x1min + (q % np1 + 0.5) s rdx, x2min + (q / np1 + 0.5) s rdy).
// Workspaces are allocated per team slot, with at most mem_mb megabytes in all.
struct DevicePatchProblem {
  DvceArray2D<Real> spd;
  DvceArray1D<int> sptr, stgt, pptr;
  DvceArray1D<Real> swgt;
  int npool = 0, npatch = 0, np1 = 0, s = 1;
  double x1min = 0.0, x2min = 0.0, lx = 0.0, ly = 0.0, rdx = 0.0, rdy = 0.0;
};
void SolvePatchesDevice(const DevicePatchProblem &pb, DvceArray1D<double> mnew,
                        DvceArray2D<int> outi, DvceArray1D<double> outr, const int mem_mb,
                        double timing[4]);

}  // namespace resample
}  // namespace dust
#endif  // DUST_DUST_RESAMPLE_HPP_
