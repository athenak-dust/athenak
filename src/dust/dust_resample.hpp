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

// The stage clock (DUST_R_STAGES) and the basic-solution check (DUST_R_BASIS) of the
// device patch solves are compiled only with -DDUST_R_DIAG=1.  Compiled in but switched off they
// still changed the solve kernels (local CUDA build, sm_89: the small-patch kernel 168 -> 178
// registers, the 512- and 1024-thread kernels 24 / 32 more stack bytes per thread, 64 more bytes
// of shared memory each).  With DUST_R_DIAG = 0, amr_profile still times the phases on the host.
#ifndef DUST_R_DIAG
#define DUST_R_DIAG 0
#endif
// Rixin: the column-by-column QR update of the device patch solves
// (<dust>/cloud_resample_qr_rows = false, one thread per column in the host's order, bitwise
// comparable with the host solver) is compiled only with -DDUST_R_QR_COLUMNS=1 (the reference
// build); the production build has the row-parallel update only, and an input or
// restart asking for false stops there (dust.cpp).  The host R solver is not affected.
#ifndef DUST_R_QR_COLUMNS
#define DUST_R_QR_COLUMNS 0
#endif

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
// except above a cap, 10 with a negative mass.
// Scope (Codex review): the totals are this rank's only (DUST_R_STAGES prints
// rank 0's, not a sum over ranks); the clock cycles exist only on CUDA (Clk() is 0 elsewhere),
// the counts 6-10 on every backend.  The time of the basic-solution check (7-10) is outside
// stages 1-3 and so lands in the stream's rest (4 minus 1-3; before 2026-10-20 it fell into 2).  With amr_profile off these branches do
// not run, but compiled in they still cost registers and stack: hence DUST_R_DIAG above (the
// cost is read from the compiler's resource report, not the runtime condition).
constexpr int kNStage = 11;
constexpr int kStagePatches = 6;   // the number of timed patches: > 0 once a profiled R has run
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
