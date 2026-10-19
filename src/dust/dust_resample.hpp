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
// Patch q is centred at (pgeo(q,0), pgeo(q,1)); its centre-of-mass rows are measured in units of
// (pgeo(q,2), pgeo(q,3)).
// Workspaces are allocated per team slot, with at most mem_mb megabytes in all.
struct DevicePatchProblem {
  DvceArray2D<Real> spd;
  DvceArray1D<int> sptr, stgt, pptr;
  DvceArray1D<Real> swgt;
  DvceArray2D<Real> pgeo;
  int npool = 0, npatch = 0;
  double lx = 0.0, ly = 0.0;
  // amr_profile only: device clock cycles of the patch solves by stage, summed over
  // the teams (kNStage entries; empty = off)
  DvceArray1D<unsigned long long> stage;
  bool qr_rows = false;   // the QR update shares the rows as well (cloud_resample_qr_rows)
};
// stages: 0 rows and scaling, 1 QR, 2 null vectors, 3 end-point steps, 4 the whole stream
// (1-3 included), 5 row check, 6 patches timed; StageTotals() sums them over the run (host).
// Counts, not cycles: 7 QR blocks with a null space, 8 of them whose basic solution
// on the pivot columns (the other clouds at zero) lies within 0 <= w <= cap, 9 within the bounds
// except above a cap, 10 with a negative mass
constexpr int kNStage = 11;
inline unsigned long long *StageTotals() {
  static unsigned long long t[kNStage] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  return t;
}
// mid > 64: patches of 65 .. mid clouds form a middle bucket solved with teams of mid_team threads
// (three buckets); mid <= 64: the two buckets of <= 64 and > 64 clouds.  wide_team > 512: the large bucket
// with teams of up to 1024 threads (a kernel bound to 1024 threads).
void SolvePatchesDevice(const DevicePatchProblem &pb, DvceArray1D<double> mnew,
                        DvceArray2D<int> outi, DvceArray1D<double> outr, const int mem_mb,
                        const int mid, const int mid_team, const int wide_team,
                        double timing[5]);

// Totals of one resampling over its patches (the log row), reduced on the device from outi /
// outr, so that no per-patch data goes to the host: patches with clouds, pool,
// kept, at the cap, fallbacks, reduced patches, steps to zero / to the cap, scans; the largest
// accepted residual; the largest patch, its rows; patches with >= 64 / 256 clouds.
struct PatchSummary {
  long long npat = 0, pool = 0, kept = 0, atcap = 0, nfall = 0, nred = 0, zsteps = 0, csteps = 0,
            scans = 0;
  double resmax = 0.0;
  int nmax = 0, nbmax = 0, n64 = 0, n256 = 0;
};
void SummarizePatchesDevice(DvceArray2D<int> outi, DvceArray1D<double> outr, PatchSummary &s);

}  // namespace resample
}  // namespace dust
#endif  // DUST_DUST_RESAMPLE_HPP_
