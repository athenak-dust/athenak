//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_imex3d.cpp
//! \brief Refinable-cloud IMEX2+ drag on actual 3-D leaves. The common task graph owns
//! the explicit push, migration, gas local solve and drag history. These two kernels use
//! each particle's own stopping time, the cloud mass assignment for Q/P and PMBR, and
//! the linear moment gather for u*. They never add position drift or external forces.

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

// The stage position geometry is distinct from ShearRouteTime, whose crossing offset
// also compensates for background drift in the stages still to come.
// Form the abscissa from the same low-storage recurrence as ExplicitPush, including a
// gamma switch. The final beta=0 assembly stage never reaches either kernel.
Real DustGasDrag::CloudImplicitTime3D(Driver *pdrive, int stage) const {
  Real c = 0.0;
  for (int s=0; s<stage; ++s) c = pdrive->gam0[s]*c + pdrive->beta[s];
  return pmy_pack->pmesh->time + c*pmy_pack->pmesh->dt;
}

void DustGasDrag::CloudDepositImplicit3D(Driver *pdrive, int stage) {
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const Real a_dt = DragStep(pdrive);
  UpdateShear3D(CloudImplicitTime3D(pdrive, stage));
  const auto lt = leaf_traversal3d;
  auto qp = qdep;  // cleared, including halo slots, by DepositDrag
  auto body = KOKKOS_LAMBDA(const int p) {
    const Real mass_c = pr(IPM,p)*a_dt/(pr(IPTS,p) + a_dt);
    const Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
    lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pr(IPZ,p), pi(PLEV,p),
                          lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w) {
      int b, k, j, i;
      lt.Decode(target, b, k, j, i);
      const Real dw = mass_c*w/(lt.box(b,3)*lt.box(b,4)*lt.box(b,5));
      DepositAdd(&qp(b,0,k,j,i), dw);
      DepositAdd(&qp(b,1,k,j,i), dw*vx);
      DepositAdd(&qp(b,2,k,j,i), dw*vy);
      DepositAdd(&qp(b,3,k,j,i), dw*vz);
    });
  };
  // Migration can change the number of particles between the two working stages.
  if (cloud_ord_n > 0 && cloud_ord_n <= npart) {
    CloudParForOrdered("dust_imex_cloud3d_deposit", npart, cloud_ord, cloud_ord_n, body);
  } else {
    CloudParFor("dust_imex_cloud3d_deposit", npart, body);
  }
  HaloReverseAdd(qdep, 0, 4);
}

void DustGasDrag::CloudGatherImplicit3D(Driver *pdrive, int stage) {
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const Real a_dt = DragStep(pdrive);
  const bool br = back_reaction;
  const int tmode = cloud_transition;
  UpdateShear3D(CloudImplicitTime3D(pdrive, stage));
  const auto lt = leaf_traversal3d;
  // The common u* exchange has already filled same-level and coarse/fine ghosts,
  // including physical x3 boundaries. Pack owners' slopes before forwarding to halo.
  PackGatherField3D(ustar);
  auto gp = gpack3;
  auto dm = dmom;  // cleared, including halo slots, by GatherKickPMBR
  auto body = KOKKOS_LAMBDA(const int p) {
    using T = SMRCompositeTransfer;
    constexpr int cache = 27;
    int nt = 0, tt[cache];
    Real tw[cache];
    const Real xp = pr(IPX,p), yp = pr(IPY,p), zp = pr(IPZ,p);
    const int am = lt.Anchor(pi(PGID,p)), lev = pi(PLEV,p);
    const int glev = T::GatherCode(lev, tmode);
    const bool same = (glev & ~T::kCloudGatherPoint) == lev;
    Real ug[3] = {0.0, 0.0, 0.0};
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
      for (int c=0; c<3; ++c) {
        ug[c] += w*gp(b,k,j,i,c) + cx*gp(b,k,j,i,3+c) +
                 cy*gp(b,k,j,i,6+c) + cz*gp(b,k,j,i,9+c);
      }
      if (nt < cache) {tt[nt] = target; tw[nt] = w;}
      ++nt;
    });
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
      DepositAdd(&dm(b,0,k,j,i), dw*dvx);
      DepositAdd(&dm(b,1,k,j,i), dw*dvy);
      DepositAdd(&dm(b,2,k,j,i), dw*dvz);
    };
    if (same && nt <= cache) {
      for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
    } else {
      lt.ForEachCloudWeight(xp, yp, zp, lev, am, add);
    }
  };
  if (cloud_ord_n > 0 && cloud_ord_n <= npart) {
    CloudParForOrdered("dust_imex_cloud3d_gather", npart, cloud_ord, cloud_ord_n, body);
  } else {
    CloudParFor("dust_imex_cloud3d_gather", npart, body);
  }
  if (br) HaloReverseAdd(dmom, 0, 3);
}

}  // namespace dust
