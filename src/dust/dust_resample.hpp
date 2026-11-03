#ifndef DUST_DUST_RESAMPLE_HPP_
#define DUST_DUST_RESAMPLE_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_resample.hpp
//! \brief Shared pieces of the same-level resampling of refinable clouds: the tolerances of
//! the per-patch algorithm, the pool table columns, and the device solver of
//! dust_resample_device.cpp, which runs the algorithm of the host ResamplePatch on the device.

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
// The column-by-column QR update of the device patch solves (<dust>/cloud_resample_qr_rows =
// false, one thread per column in the host's order, bitwise comparable with the host solver) is
// compiled only with -DDUST_R_QR_COLUMNS=1 (the reference build); the production build has the
// row-parallel update only, and an input or restart asking for false stops there (dust.cpp).
// The host R solver is not affected.
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
enum PoolColumn {JX=0, JY=1, JM=2, JVX=3, JVY=4, JVZ=5, JTS=6, JUX=7, JUY=8, JUZ=9, JCAP=10, NPD=11,
                 JZ=11, NPD3=12};   // x3 position (3-D pools only)

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
  // 3-D refinable clouds: pools with JZ, pgeo (x, y, z centre, three half widths) and a
  // third centre-of-mass row; the 2-D rows and their order are unchanged
  bool three = false;
  double lz = 0.0;
  // amr_profile only: device clock cycles of the patch solves by stage, summed over
  // the teams (kNStage entries; empty = off)
  DvceArray1D<unsigned long long> stage;
  bool qr_rows = false;   // the QR update shares the rows as well (cloud_resample_qr_rows)
  // 3-D, <dust>/cloud_resample_rank_screen: patches whose clouds do not outnumber their nominal
  // rows (7 leaves + 6 + centre-of-mass rows, + the cross-moment rows in a flagged patch) are
  // left as they are, before any bucket
  bool rank_screen = false;
};
// stages: 0 rows and scaling, 1 QR, 2 null vectors, 3 end-point steps, 4 the whole stream
// (1-3 included), 5 row check, 6 patches timed; StageTotals() sums them over the run (host).
// Counts, not cycles: 7 QR blocks with a null space, 8 of them whose basic solution
// on the pivot columns (the other clouds at zero) lies within 0 <= w <= cap, 9 within the bounds
// except above a cap, 10 with a negative mass.
// Scope: the totals are this rank's only (DUST_R_STAGES prints rank 0's, not a sum over ranks);
// the clock cycles exist only on CUDA (Clk() is 0 elsewhere), the counts 6-10 on every backend.
// The time of the basic-solution check (7-10) is outside stages 1-3 and so lands in the
// stream's rest (4 minus 1-3).  With amr_profile off these branches do not run, but compiled in
// they still cost registers and stack: hence DUST_R_DIAG above (the cost is read from the
// compiler's resource report, not the runtime condition).
constexpr int kNStage = 11;
constexpr int kStagePatches = 6;   // the number of timed patches: > 0 once a profiled R has run
inline unsigned long long *StageTotals() {
  static unsigned long long t[kNStage] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  return t;
}
// <dust>/cloud_resample_xv_rows (3-D; default angmom): xv = 0 none, 1 angmom, 2 full.  The rows
// follow the feedback rows (the rows of xv = 0 and their order are unchanged):
//   angmom: sum w (x - c) x (v - u) / h, the three components, h the largest half width of
//           the patch (one length, so the rows are the physical angular momentum),
//   full:   sum w ((x - c)_a / h_a) (v - u)_b, a, b = 1..3 (h_a the half widths, as the
//           centre-of-mass rows),
// c the patch centre (pgeo), x the position unwrapped about it as for the centre-of-mass rows,
// u the patch's mass-weighted mean velocity before R (the reference of the momentum rows).  The
// solver scales every row by its largest entry, so the absolute scale above is cosmetic.  Each
// mode is its own instantiation of the solve kernels, so xv = 0 compiles the kernels as before.
// The rows go only to the flagged patches (xflag below).  A patch is unflagged only when all its
// pool clouds have one level L, every footprint leaf is at level >= L, no cloud support crosses a
// physical x3 face and the footprint spans less than a period in every periodic direction; there
// the existing rows already fix all nine cross moments and the six spatial second moments (the
// hat of a cloud is box * box of the level-L cell width: leaf values a_l = the centre of the
// level-L cell holding leaf l give sum_l a_l W_il = x_i; the cell-integrated hat is a quadratic
// B-spline on the level-L grid, which reproduces x^2 up to one constant per width), so those
// patches are solved by the xv = 0 kernels, with the result of none.  Rows per flagged patch:
// angmom 3 (+ 3 in a rotating frame, xv_rot), full 9 (+ 3).
KOKKOS_INLINE_FUNCTION
constexpr int XVRows(const int xv) {return (xv == 1) ? 3 : ((xv == 2) ? 9 : 0);}
// mid > 64: patches of 65 .. mid clouds form a middle bucket solved with teams of mid_team threads;
// mid <= 64: only the two buckets of at most 64 and of more than 64 clouds.  wide_team > 512: the
// large bucket with teams of up to 1024 threads (a kernel bound to 1024 threads).
void SolvePatchesDevice(const DevicePatchProblem &pb, DvceArray1D<double> mnew,
                        DvceArray2D<int> outi, DvceArray1D<double> outr, const int mem_mb,
                        const int mid, const int mid_team, const int wide_team,
                        double timing[5], long long *screened = nullptr, const int xv = 0,
                        const DvceArray1D<int> &xflag = DvceArray1D<int>(),
                        const bool xv_rot = false, const double xv_q = 0.0);
// screened (pb.rank_screen): [0] patches left out by the screen, [1] their clouds
// xv: the cross-moment rows above (3-D only; 2-D callers leave 0).  xflag (npatch): the patches
// that get them (ResampleClouds3D, the rule above: several cloud levels, a cloud finer than a
// footprint leaf, an x3 fold or an ambiguous unwrapping); every other patch is solved by the
// xv = 0 kernels, bitwise as without the rows.  xv != 0 with an empty xflag: every patch.
// xv_rot (shearing box, omega0 != 0) with shear parameter xv_q: each flagged patch also gets the
// three rows (q - 1) S_xz, S_yz, (1 - q) S_xx + S_yy of S = sum w (x - c)(x - c)^T, so that the
// inertial J (stored v2 is shear-relative: J_in = J + omega (those three)) is kept as well.

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
