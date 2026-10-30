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
#include "hydro/hydro.hpp"
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
  // the gas is in its t^n orbital-advection frame: shear images of t^n, positions shifted
  const Real sh = ImplicitShearShift(pdrive, stage);
  UpdateShear3D(pmy_pack->pmesh->time);
  const auto lt = leaf_traversal3d;
  auto qp = qdep;  // cleared, including halo slots, by DepositDrag
  const std::size_t sq = qp.stride(1);
  auto body = KOKKOS_LAMBDA(const int p) {
    const Real mass_c = pr(IPM,p)*a_dt/(pr(IPTS,p) + a_dt);
    const Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
    lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p) + sh*pr(IPX,p), pr(IPZ,p), pi(PLEV,p),
                          lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w) {
      int b, k, j, i;
      lt.Decode(target, b, k, j, i);
      const Real dw = mass_c*w/(lt.box(b,3)*lt.box(b,4)*lt.box(b,5));
      Real v[4] = {dw, dw*vx, dw*vy, dw*vz};
      DepositAddWarp(&qp(b,0,k,j,i), sq, target, v);
    });
  };
  // Migration can change the number of particles between the two working stages.
  if (const int nord = CloudOrderLength(npart)) {
    CloudParForOrdered("dust_imex_cloud3d_deposit", npart, cloud_ord, nord, body);
  } else {
    CloudParFor("dust_imex_cloud3d_deposit_storage", npart, body);   // (label: the storage order)
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
  const Real sh = ImplicitShearShift(pdrive, stage);   // as in CloudDepositImplicit3D
  UpdateShear3D(pmy_pack->pmesh->time);
  const auto lt = leaf_traversal3d;
  // The common u* exchange has already filled same-level and coarse/fine ghosts,
  // including physical x3 boundaries. Pack owners' slopes before forwarding to halo.
  PackGatherField3D(ustar);
  auto gp = gpack3;
  auto dm = dmom;  // cleared, including halo slots, by GatherKickPMBR
  const std::size_t sd = dm.stride(1);
  auto body = KOKKOS_LAMBDA(const int p) {
    using T = SMRCompositeTransfer;
    constexpr int cache = 27;
    int nt = 0, tt[cache];
    Real tw[cache];
    const Real xp = pr(IPX,p), yp = pr(IPY,p) + sh*pr(IPX,p), zp = pr(IPZ,p);
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
      Real v[3] = {dw*dvx, dw*dvy, dw*dvz};
      DepositAddWarp(&dm(b,0,k,j,i), sd, target, v);
    };
    if (same && nt <= cache) {
      for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
    } else {
      lt.ForEachCloudWeight(xp, yp, zp, lev, am, add);
    }
  };
  if (const int nord = CloudOrderLength(npart)) {
    CloudParForOrdered("dust_imex_cloud3d_gather", npart, cloud_ord, nord, body);
  } else {
    CloudParFor("dust_imex_cloud3d_gather_storage", npart, body);   // (label: the storage order)
  }
  if (br) HaloReverseAdd(dmom, 0, 3);
}


// Bound-clump NaN fix: the local implicit solve is lumped, u*_c = (rho u + P)/(rho + Q)
// per cell, while each cloud couples to several cells (deposit W, gather from neighbouring u*).  Giving
// the gas the particles' actual impulse D_c (as PMBR alone does) leaves the gas with D_c - G_c, G_c =
// rho (u*_c - u_c) the lumped change, which the light gas of a dust-dominated cell feels Q/rho times
// amplified: the gas there ran away to tens of c_s in a collapsing clump (epsilon ~ 2e4).  Here the gas
// receives G_c plus its mass share rho/(rho + Q) of the residual, and the clouds of the cell take the
// rest: r_c = (D_c - G_c)/(rho + Q_c), gas += G_c + rho r_c, v_p += c_p sum_c W_cp r_c.  Total momentum
// is exact; a gas-dominated cell (Q << rho) is unchanged.  dmom becomes the applied gas rate.
void DustGasDrag::CloudImplicitConsistency3D(Driver *pdrive, int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0 = pmy_pack->phydro->u0;
  const Real a_dt = DragStep(pdrive);
  const Real inv_adt = 1.0/((pdrive->a_impl)*(pmy_pack->pmesh->dt));
  if (imex_resid.extent(0) != ustar.extent(0) || imex_resid.extent(2) != ustar.extent(2) ||
      imex_resid.extent(3) != ustar.extent(3) || imex_resid.extent(4) != ustar.extent(4)) {
    Kokkos::realloc(imex_resid, ustar.extent(0), 3, ustar.extent(2), ustar.extent(3), ustar.extent(4));
  }
  Kokkos::deep_copy(DevExeSpace(), imex_resid, 0.0);
  auto res = imex_resid;
  auto us = ustar;
  auto qp = qdep;
  auto dm = dmom;
  par_for("dust_imex_consistent_gas", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real rho = u0(m,IDN,k,j,i);
    const Real inv = 1.0/(rho + qp(m,0,k,j,i));
    for (int c=0; c<3; ++c) {
      const Real g = rho*us(m,c,k,j,i) - u0(m,IM1+c,k,j,i);
      const Real r = (dm(m,c,k,j,i) - g)*inv;
      const Real dgas = g + rho*r;
      u0(m,IM1+c,k,j,i) += dgas;
      dm(m,c,k,j,i) = dgas*inv_adt;
      res(m,c,k,j,i) = r;
    }
  });
  // the residual velocity of each cell back to its clouds, with the deposit's weights and frame
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const Real sh = ImplicitShearShift(pdrive, stage);
  UpdateShear3D(pmy_pack->pmesh->time);
  PackGatherField3D(imex_resid);   // owners' values on the halo blocks
  const auto lt = leaf_traversal3d;
  auto gp = gpack3;
  auto body = KOKKOS_LAMBDA(const int p) {
    if (pr(IPM,p) <= 0.0) return;
    Real rv[3] = {0.0, 0.0, 0.0};
    lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p) + sh*pr(IPX,p), pr(IPZ,p), pi(PLEV,p),
                          lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w) {
      int b, k, j, i;
      lt.Decode(target, b, k, j, i);
      for (int c=0; c<3; ++c) rv[c] += w*gp(b,k,j,i,c);
    });
    const Real cj = a_dt/(pr(IPTS,p) + a_dt);
    pr(IPVX,p) += cj*rv[0]; pr(IPVY,p) += cj*rv[1]; pr(IPVZ,p) += cj*rv[2];
    // the drag rate of this stage (consumed by the next working stage) includes the correction
    pr(IPRX,p) += cj*rv[0]/a_dt; pr(IPRY,p) += cj*rv[1]/a_dt; pr(IPRZ,p) += cj*rv[2]/a_dt;
  };
  // in cell order as the two kernels above (each cloud writes only its own velocity)
  if (const int nord = CloudOrderLength(npart)) {
    CloudParForOrdered("dust_imex_consistent_dust", npart, cloud_ord, nord, body);
  } else {
    CloudParFor("dust_imex_consistent_dust_storage", npart, body);   // (label: the storage order)
  }
}

}  // namespace dust
