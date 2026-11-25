#ifndef DUST_DUST_IMEX3D_GATHER_HPP_
#define DUST_DUST_IMEX3D_GATHER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_imex3d_gather.hpp
//! \brief The clouds' linear-moment u* gather of the IMEX stages (3-D), in one place for the
//! kick (CloudGatherImplicit3D) and the operator of the defect correction (dust_imex3d_dc.cpp),
//! so that both evaluate the same expressions in the same order.  Included by dust.hpp users
//! after dust.hpp.
//!
//! kBlend: the gather blend of the center rule, ubar = sum w u* + (1 - sum w |f|) (slope terms),
//! f from fq (its sign marks the cells with an overlap share); otherwise ubar = sum (w u* + slope
//! terms).  kMark (only without kBlend): mark = some weight on a cell whose fq is negative (the
//! floor's overlap-share cells), for the overlap-skip of the consistency step.  The leaf cells
//! and weights of the moment traversal are kept (at most `cache` of them) for the deposit.

namespace dust {

template <bool kBlend, bool kMark, typename GP, typename FQ>
KOKKOS_INLINE_FUNCTION
void CloudLinearGather3D(const LeafTraversal3D &lt, const GP &gp, const FQ &fq,
                         const Real xp, const Real yp, const Real zp, const int glev,
                         const int am, const int is, const int js, const int ks, Real ug[3],
                         int &nt, int *tt, Real *tw, const int cache, int &ndp) {
  using T = CloudGeometry;
  Real us[3] = {0.0, 0.0, 0.0};   // the slope terms (blend)
  Real fs = 0.0;                  // sum w f: the dust share under the cloud (blend)
  ug[0] = 0.0; ug[1] = 0.0; ug[2] = 0.0;
  nt = 0; ndp = 0;
  lt.ForEachCloudMoment(xp, yp, zp, glev, am,
  [&](const int target, const Real w, const Real m1x, const Real m1y, const Real m1z) {
    int b, k, j, i;
    lt.Decode(target, b, k, j, i);
    const Real dx = lt.box(b,3), dy = lt.box(b,4), dz = lt.box(b,5);
    const Real xc = lt.box(b,0) + (i - is + 0.5)*dx;
    const Real yc = lt.box(b,1) + (j - js + 0.5)*dy;
    const Real zc = lt.box(b,2) + (k - ks + 0.5)*dz;
    const Real cx = 0.5*(m1x + T::MinImage(xp - xc, lt.lx)*w)/dx;
    const Real cy = 0.5*(m1y + T::MinImage(yp - yc, lt.ly)*w)/dy;
    const Real cz = 0.5*(m1z + T::MinImage(zp - zc, lt.lz)*w)/dz;
    if constexpr (kBlend) {
      const Real fv = fq(b,0,k,j,i);   // negative: a cell with an overlap share (floor mode)
      fs += w*fabs(fv);
      if (fv < 0.0 && w != 0.0) ndp = 1;
      for (int c=0; c<3; ++c) {
        ug[c] += w*gp(b,k,j,i,c);
        us[c] += cx*gp(b,k,j,i,3+c) + cy*gp(b,k,j,i,6+c) + cz*gp(b,k,j,i,9+c);
      }
    } else {
      if constexpr (kMark) {
        if (fq(b,0,k,j,i) < 0.0 && w != 0.0) ndp = 1;
      }
      for (int c=0; c<3; ++c) {
        ug[c] += w*gp(b,k,j,i,c) + cx*gp(b,k,j,i,3+c) +
                 cy*gp(b,k,j,i,6+c) + cz*gp(b,k,j,i,9+c);
      }
    }
    if (nt < cache) {tt[nt] = target; tw[nt] = w;}
    ++nt;
  });
  if constexpr (kBlend) {
    // the slope part fades with the dust share: 1 - sum w f (0 where the dust dominates the solve)
    const Real wl = fmax(0.0, 1.0 - fs);
    for (int c=0; c<3; ++c) ug[c] += wl*us[c];
  }
}


// The kick of the IMEX stage from the gathered ubar (dv = c_p (ubar - v), the rate dv/a_dt for the
// next working stage) and the deposit of the actual impulse with the cloud's own weights (the
// cached ones when the gather used the deposit's shape and all of them fit).
template <typename PR, typename DM>
KOKKOS_INLINE_FUNCTION
void CloudKickDeposit3D(const LeafTraversal3D &lt, const PR &pr, const int p, const Real ug[3],
                        const Real a_dt, const bool br, const DM &dm, const std::size_t sd,
                        const bool same, const int nt, const int *tt, const Real *tw,
                        const int cache, const Real xp, const Real yp, const Real zp,
                        const int lev, const int am) {
  const Real cj = a_dt/(pr(IPTS,p) + a_dt);
  const Real dvx = cj*(ug[0] - pr(IPVX,p));
  const Real dvy = cj*(ug[1] - pr(IPVY,p));
  const Real dvz = cj*(ug[2] - pr(IPVZ,p));
  pr(IPVX,p) += dvx; pr(IPVY,p) += dvy; pr(IPVZ,p) += dvz;
  // The next working stage consumes rates, not impulses, in a_twid[2][2].
  pr(IPRX,p) = dvx/a_dt; pr(IPRY,p) = dvy/a_dt; pr(IPRZ,p) = dvz/a_dt;
  if (!br) return;
  const Real mass = -pr(IPM,p);
  auto add = [&](const int target, const Real w) {
    int b, k, j, i;
    lt.Decode(target, b, k, j, i);
    const Real dw = mass*w/(lt.box(b,3)*lt.box(b,4)*lt.box(b,5));
    // Deposit the actual applied particle impulse. This remains conservative with
    // a linear gather and across refinement even though that gather differs from W.
    Real v[3] = {dw*dvx, dw*dvy, dw*dvz};
    DepositAddWarp(&dm(b,0,k,j,i), sd, target, v);
  };
  if (same && nt <= cache) {
    for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
  } else {
    lt.ForEachCloudWeight(xp, yp, zp, lev, am, add);
  }
}

}  // namespace dust

#endif  // DUST_DUST_IMEX3D_GATHER_HPP_
