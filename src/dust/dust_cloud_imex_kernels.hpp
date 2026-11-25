#ifndef DUST_DUST_CLOUD_IMEX_KERNELS_HPP_
#define DUST_DUST_CLOUD_IMEX_KERNELS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud_imex_kernels.hpp
//! \brief The refinable clouds' IMEX kernels written once for both traversals on LeafApi:
//! the particle kernel of the dc1 / dc2 operator (CloudOperatorKernelT).
//! Included by dust_imex3d_dc.cpp and dust_cloud2d_dc.cpp after dust.hpp.

namespace dust {

// the operator's particle kernel without the gather blend on either traversal (the 2-D
// kernel of the former CloudCoupledOperator2D and the plain 3-D operator kernel written once): the
// kick's gather (CloudGatherKickT: LeafApi::GatherLinear) of every cloud times m_p c_p, deposited with the weights of the kick's impulse deposit
// (own: with the cloud's own weights, the operator check); kKeep: also the gathered velocity
template <class LTT, bool kKeep, typename PR, typename PI, typename GP, typename RS, typename GV>
void CloudOperatorKernelT(const LTT &lt, const PR &pr, const PI &pi, const GP &gp, const RS &res,
                          const GV &gv, const Real a_dt, const Real sh,
                          const bool own, const int is, const int js,
                          const int ks, const int npart, const DvceArray1D<int> &ord,
                          const int nord) {
  using A = LeafApi<LTT>;
  constexpr int kC = A::kCache;
  const std::size_t sr = res.stride(1);
  auto body = KOKKOS_LAMBDA(const int p) {
    using T = CloudGeometry;
    int nt = 0, tt[kC];
    Real tw[kC];
    Real x[3];
    A::LoadImplicit(pr, p, sh, x);
    const int am = lt.Anchor(pi(PGID,p)), lev = pi(PLEV,p);
    const int glev = T::GatherCode(lev);
    const bool same = (glev & ~T::kCloudGatherPoint) == lev;
    Real ug[3] = {0.0, 0.0, 0.0};
    A::ForEachMoment(lt, x, glev, am,
    [&](const int target, const Real w, const Real m1x, const Real m1y, const Real m1z) {
      A::GatherLinear(lt, gp, target, w, m1x, m1y, m1z, x, is, js, ks, ug);
      if (nt < kC) {tt[nt] = target; tw[nt] = w;}
      ++nt;
    });
    // (a plain if on the template constant: an extended lambda may not first-capture in an
    // if constexpr)
    if (kKeep) {gv(p,0) = ug[0]; gv(p,1) = ug[1]; gv(p,2) = ug[2];}
    const Real mc = pr(IPM,p)*(a_dt/(pr(IPTS,p) + a_dt));
    auto add = [&](const int target, const Real w) {
      int b, k, j, i;
      A::Decode(lt, target, b, k, j, i);
      const Real dw = mc*w/A::Measure(lt, b);
      Real v[3] = {dw*ug[0], dw*ug[1], dw*ug[2]};
      A::template DepositN<3>(res, sr, target, b, k, j, i, v);
    };
    // the deposit weights of the kick's impulse (CloudGatherKickT)
    if (!own && same && nt <= kC) {
      for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
    } else {
      A::ForEachWeight(lt, x, lev, am, add);
    }
  };
  const bool k3 = (A::kDim == 3);
  if (nord) {
    CloudParForOrdered(k3 ? "dust_cloud_dc_operator" : "dust_cloud2d_dc_operator", npart, ord, nord,
                       body);
  } else {
    CloudParFor(k3 ? "dust_cloud_dc_operator_storage" : "dust_cloud2d_dc_operator_storage", npart,
                body);
  }
}

}  // namespace dust

#endif  // DUST_DUST_CLOUD_IMEX_KERNELS_HPP_
