//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_pm.cpp
//! \brief Particle-mesh kernels for IMEX drag stages and for the PC2/split-BE hybrid.
//! The implicit chain is executed with a=a_impl*dt in each working IMEX stage, or once
//! with a=dt in the split-BE branch:
//!
//!  (1) DepositDrag (scatter): with c_j = a/(t_s,j + a) and mu_j = m_j*W_jk/V, atomically
//!      accumulate Q_k = sum_j mu_j*c_j and P_k = sum_j mu_j*c_j*v_j.
//!  (2) [additive halo exchange of (Q,P), see MeshBoundaryValuesDep]
//!  (3) GasImplicitSolve (per cell): u*_k = (rho*u + P)_k / (rho + Q)_k, which is the
//!      closed-form solution of the momentum-conserving backward-Euler drag pair
//!      (Benitez-Llambay, Krapp & Pessah 2019 / Krapp & Benitez-Llambay 2020).
//!  (4) [copy exchange fills u* ghost zones]
//!  (5) GatherKickPMBR (fused gather + kick + record + scatter): interpolate u~*_j with
//!      the SAME weights, kick dv_j = c_j*(u~*_j - v_j), record the drag rate
//!      R_j = dv_j/a, and atomically deposit the momentum back-reaction (PMBR)
//!      -m_j*W_jk*dv_j/V into dmom. Using identical W and dv for the particle kick and
//!      the gas deposit makes total momentum conservation exact by construction (YJ16).
//!  (6) [additive halo exchange of dmom]
//!  (7) ApplyPMBR: u0 += dmom over active cells, then dmom /= a in place, turning dmom
//!      into the gas drag-rate field R_g = -sum_j m_j*W_jk*R_j/V consumed by the
//!      a_twid history term of the next stage (AddDragHistoryGas). This works because
//!      dmom = a*R_g identically.

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositDrag
//! \brief Zero the (Q,P) field and scatter the drag-weighted particle sums into it.

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::PackGatherField
//! \brief Gas velocity and its central differences in x1 and x2 of every cell with both
//! neighbours, nine numbers side by side.  The linear cloud gather then reads one
//! short run of memory per leaf instead of fifteen values spread over nine rows of three
//! arrays; the values and the order of the operations are those of the unpacked gather.

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::PackGasGather
//! \brief The gas velocity of every cell (ghosts included) of a conserved state u into
//! uvel_pred, then PackGatherField: the gather field of the linear cloud gather from u (the
//! predictor's old state; also the resampling's current state, which reads it
//! through the halo like the predictor).

void DustGasDrag::PackGasGather(const DvceArray5D<Real> &u) {
  auto uold = u;
  if (uvel_pred.extent(0) != uold.extent(0)) {
    Kokkos::realloc(uvel_pred, uold.extent(0), 3, uold.extent(2), uold.extent(3),
                    uold.extent(4));
  }
  auto uvel = uvel_pred;
  const int ks = pmy_pack->pmesh->mb_indcs.ks;
  const int nmb_ = pmy_pack->nmb_thispack;   // not the capacity of the array
  const int nj_ = static_cast<int>(uold.extent(3)), ni_ = static_cast<int>(uold.extent(4));
  par_for("dust_pc2_predictor_gasvel",DevExeSpace(),0,nmb_-1,0,nj_-1,0,ni_-1,
  KOKKOS_LAMBDA(const int m, const int j, const int i) {
    const Real invrho = 1.0/uold(m,IDN,ks,j,i);
    uvel(m,0,ks,j,i) = uold(m,IM1,ks,j,i)*invrho;
    uvel(m,1,ks,j,i) = uold(m,IM2,ks,j,i)*invrho;
    uvel(m,2,ks,j,i) = uold(m,IM3,ks,j,i)*invrho;
  });
  PackGatherField(uvel_pred);
}

void DustGasDrag::PackGatherField(const DvceArray5D<Real> &vel) {
  // Room for the halo slots (their values come from HaloForwardPack)
  const int nmb = std::max(vel.extent_int(0), leaf_traversal.nfld), nk = vel.extent_int(2);
  const int nj = vel.extent_int(3), ni = vel.extent_int(4);
  if (gpack.extent_int(0) != nmb || gpack.extent_int(1) != nk || gpack.extent_int(2) != nj ||
      gpack.extent_int(3) != ni) {
    Kokkos::realloc(gpack, nmb, nk, nj, ni, 9);
  }
  auto gp = gpack;
  auto v = vel;
  const int ks = pmy_pack->pmesh->mb_indcs.ks;
  // only the blocks that exist: with adaptive refinement the arrays hold more slots
  par_for("dust_gather_pack",DevExeSpace(),0,pmy_pack->nmb_thispack-1,1,nj-2,1,ni-2,
  KOKKOS_LAMBDA(const int m, const int j, const int i) {
    for (int c=0; c<3; ++c) {
      gp(m,ks,j,i,c) = v(m,c,ks,j,i);
      gp(m,ks,j,i,3+c) = v(m,c,ks,j,i+1) - v(m,c,ks,j,i-1);
      gp(m,ks,j,i,6+c) = v(m,c,ks,j+1,i) - v(m,c,ks,j-1,i);
    }
  });
  HaloForwardPack(gpack);   // the owners' values on the halo blocks
}

TaskStatus DustGasDrag::DepositDrag(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}
  // The stiffness deposit qdep[4] feeds the automatic switch's metric and
  // the stiffness monitor / dt limit / predictor damping
  const bool rate_dep = NeedsHybridMetric() || StiffnessActive();

  if (coupling == DustCoupling::hybrid) {
    // Split-BE deposits Q/P in its fused preparation traversal.  PC2 needs only the
    // fresh old-state feedback predictor during stage 1.
    if (hybrid_mode != HybridMode::pc2 || stage != 1) return TaskStatus::complete;
    Kokkos::deep_copy(DevExeSpace(), qdep, 0.0);

    particles::Particles *ppar = pmy_pack->ppart;
    auto &pr = ppar->prtcl_rdata;
    auto &pi = ppar->prtcl_idata;
    int npart = ppar->nprtcl_thispack;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    bool three_d = pmy_pack->pmesh->three_d;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto gids = pmy_pack->gids;
    int scheme = static_cast<int>(deposit);
    Real dt = pmy_pack->pmesh->dt;
    auto &uold = pmy_pack->phydro->u1;
    auto &qdep_ = qdep;

    if (UsesCompositeTransfer()) {
      auto transfer = smr_transfer;
      auto lt = leaf_traversal;   // cloud weights on the actual leaves
      int ncells1 = qdep.extent_int(4);
      int ncells2 = qdep.extent_int(3);
      bool linear = (cloud_gather_order == 2) && transfer.cloud_mode;
      int tmode = cloud_transition;
      if (linear) {
        // Production refinable-cloud predictor: the linear gather and the
        // cloud-weight deposit.  (The general kernel dust_pc2_cloud_predictor with the
        // force / deformation variants was removed.)
        // The gas velocity of every cell (ghosts included) is formed once; the gather then
        // reads velocities, as the advance does with u*, instead of dividing five momenta
        // by five densities in each of the cloud's cells.
        if (clouds3d) {CloudPredictor3D(rate_dep); return TaskStatus::complete;}
        PackGasGather(uold);
        auto gp = gpack;
        // (a split into gather + deposit kernels and a shared-memory weight
        // cache were tried and removed, see CloudParFor in dust.hpp)
        auto pred_body = KOKKOS_LAMBDA(const int p) {
          Real xp = pr(IPX,p), yp = pr(IPY,p);
          const int am = lt.Anchor(pi(PGID,p));   // anchor block of the cloud
          int lev = pi(PLEV,p);
          Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
          // the gather's leaf weights are the deposit's: keep them
          int nt = 0, tt[SMRCompositeTransfer::kCloudWeightCache];
          Real tw[SMRCompositeTransfer::kCloudWeightCache];
          const int gcode = SMRCompositeTransfer::GatherCode(lev, tmode);
          // same shape for the gather and the deposit (the point flag changes moments only)
          const bool same = (gcode & ~SMRCompositeTransfer::kCloudGatherPoint) == lev;
          lt.ForEachCloudMoment(xp, yp, gcode, am,
          [&](const int target, const Real w, const Real m1x, const Real m1y) {
            int ti = target % ncells1;
            int remainder = target/ncells1;
            int tj = remainder % ncells2;
            int tm = remainder/ncells2;
            // the leaf's block slot may be a halo block: its box from lt
            Real dx1 = lt.box(tm,2), dx2 = lt.box(tm,3);
            Real xc = lt.box(tm,0) + (ti - is + 0.5)*dx1;
            Real yc = lt.box(tm,1) + (tj - js + 0.5)*dx2;
            Real cx = 0.5*(m1x + transfer.MinImage1(xp - xc)*w)/dx1;
            Real cy = 0.5*(m1y + transfer.MinImage2(yp - yc)*w)/dx2;
            ugx += w*gp(tm,ks,tj,ti,0) + cx*gp(tm,ks,tj,ti,3) + cy*gp(tm,ks,tj,ti,6);
            ugy += w*gp(tm,ks,tj,ti,1) + cx*gp(tm,ks,tj,ti,4) + cy*gp(tm,ks,tj,ti,7);
            ugz += w*gp(tm,ks,tj,ti,2) + cx*gp(tm,ks,tj,ti,5) + cy*gp(tm,ks,tj,ti,8);
            if (nt < SMRCompositeTransfer::kCloudWeightCache) {tt[nt] = target; tw[nt] = w;}
            ++nt;
          });
          Real mass_rate = pr(IPM,p)/pr(IPTS,p);
          Real gx = -dt*mass_rate*(ugx - pr(IPVX,p));
          Real gy = -dt*mass_rate*(ugy - pr(IPVY,p));
          Real gz = -dt*mass_rate*(ugz - pr(IPVZ,p));
          auto add = [&](const int target, const Real w) {
            int ti = target % ncells1;
            int remainder = target/ncells1;
            int tj = remainder % ncells2;
            int tm = remainder/ncells2;
            Real inv_volume = 1.0/(lt.box(tm,2)*lt.box(tm,3));
            DepositAdd(&qdep_(tm,1,ks,tj,ti), inv_volume*w*gx);
            DepositAdd(&qdep_(tm,2,ks,tj,ti), inv_volume*w*gy);
            DepositAdd(&qdep_(tm,3,ks,tj,ti), inv_volume*w*gz);
            if (rate_dep) DepositAdd(&qdep_(tm,4,ks,tj,ti), inv_volume*w*mass_rate);
          };
          if (same && nt <= SMRCompositeTransfer::kCloudWeightCache) {
            for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
          } else {
            lt.ForEachCloudWeight(xp, yp, lev, am, add);
          }
        };
        if (cloud_ord_n > 0) {   // in the order of the cell-sorted index
          CloudParForOrdered("dust_pc2_cloud_predictor_linear", npart, cloud_ord,
                             cloud_ord_n, pred_body);
        } else {
          CloudParFor("dust_pc2_cloud_predictor_linear", npart, pred_body);
        }
        // Deposits on halo blocks back to their owners; component 4 only for the
        // automatic switch (rate_dep, 2026-10-25, Codex perf review)
        HaloReverseAdd(qdep, 1, rate_dep ? 4 : 3);
        return TaskStatus::complete;
      }
      par_for("dust_pc2_composite_predictor",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
        Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
        ForEachCompositeWeight(transfer, lt, pr(IPX,p), pr(IPY,p),
                               SMRCompositeTransfer::GatherCode(pi(PLEV,p), tmode),
                               lt.Anchor(pi(PGID,p)),
        [&](const int target, const Real particle_weight) {
          int ti = target % ncells1;
          int remainder = target/ncells1;
          int tj = remainder % ncells2;
          int tm = remainder/ncells2;
          Real invrho = 1.0/uold(tm,IDN,ks,tj,ti);
          ugx += particle_weight*uold(tm,IM1,ks,tj,ti)*invrho;
          ugy += particle_weight*uold(tm,IM2,ks,tj,ti)*invrho;
          ugz += particle_weight*uold(tm,IM3,ks,tj,ti)*invrho;
        });

        Real mass_rate = pr(IPM,p)/pr(IPTS,p);
        Real gx = -dt*mass_rate*(ugx-pr(IPVX,p));
        Real gy = -dt*mass_rate*(ugy-pr(IPVY,p));
        Real gz = -dt*mass_rate*(ugz-pr(IPVZ,p));
        ForEachCompositeWeight(transfer, lt, pr(IPX,p), pr(IPY,p), pi(PLEV,p),
                               lt.Anchor(pi(PGID,p)),
        [&](const int target, const Real particle_weight) {
          int ti = target % ncells1;
          int remainder = target/ncells1;
          int tj = remainder % ncells2;
          int tm = remainder/ncells2;
          Real volume = mbsize.d_view(tm).dx1*mbsize.d_view(tm).dx2;
          Real density_weight = particle_weight/volume;
          DepositAdd(&qdep_(tm,1,ks,tj,ti), density_weight*gx);
          DepositAdd(&qdep_(tm,2,ks,tj,ti), density_weight*gy);
          DepositAdd(&qdep_(tm,3,ks,tj,ti), density_weight*gz);
          if (rate_dep) DepositAdd(&qdep_(tm,4,ks,tj,ti), density_weight*mass_rate);
        });
      });
      return TaskStatus::complete;
    }

    par_for("dust_pc2_predictor",DevExeSpace(),0,(npart-1),
    KOKKOS_LAMBDA(const int p) {
      int m = pi(PGID,p) - gids;
      int ip, jp, kp;
      Real wx[3], wy[3], wz[3];
      PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
                scheme, ip, wx);
      PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
                scheme, jp, wy);
      if (three_d) {
        PMWeights(pr(IPZ,p), mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks,
                  scheme, kp, wz);
      } else {
        kp = ks;
        wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
      }
      Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
      int clo = three_d ? 0 : 1, chi = three_d ? 2 : 1;
      for (int c=clo; c<=chi; ++c) {
        for (int b=0; b<3; ++b) {
          Real wcb = wz[c]*wy[b];
          if (wcb == 0.0) continue;
          for (int a=0; a<3; ++a) {
            Real w = wcb*wx[a];
            int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
            Real invrho = 1.0/uold(m,IDN,kk,jj,ii);
            ugx += w*uold(m,IM1,kk,jj,ii)*invrho;
            ugy += w*uold(m,IM2,kk,jj,ii)*invrho;
            ugz += w*uold(m,IM3,kk,jj,ii)*invrho;
          }
        }
      }
      Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
      if (three_d) vol *= mbsize.d_view(m).dx3;
      Real rate = pr(IPM,p)/(pr(IPTS,p)*vol);
      Real gx = -dt*rate*(ugx-pr(IPVX,p));
      Real gy = -dt*rate*(ugy-pr(IPVY,p));
      Real gz = -dt*rate*(ugz-pr(IPVZ,p));
      for (int c=clo; c<=chi; ++c) {
        for (int b=0; b<3; ++b) {
          Real wcb = wz[c]*wy[b];
          if (wcb == 0.0) continue;
          for (int a=0; a<3; ++a) {
            Real w = wcb*wx[a];
            int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
            DepositAdd(&qdep_(m,1,kk,jj,ii), w*gx);
            DepositAdd(&qdep_(m,2,kk,jj,ii), w*gy);
            DepositAdd(&qdep_(m,3,kk,jj,ii), w*gz);
            if (rate_dep) DepositAdd(&qdep_(m,4,kk,jj,ii), w*rate);
          }
        }
      }
    });
    return TaskStatus::complete;
  }

  Kokkos::deep_copy(DevExeSpace(), qdep, 0.0);

  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto gids = pmy_pack->gids;
  int scheme = static_cast<int>(deposit);
  Real a_dt = DragStep(pdrive);
  auto &qdep_ = qdep;

  if (UsesCompositeTransfer()) {
    auto transfer = smr_transfer;
    auto lt = leaf_traversal;   // cloud weights on the actual leaves
    int ncells1 = qdep.extent_int(4);
    int ncells2 = qdep.extent_int(3);
    const bool cloud = transfer.cloud_mode;
    auto scat_body = KOKKOS_LAMBDA(const int p) {
      Real cj = a_dt/(pr(IPTS,p) + a_dt);
      Real mass_c = pr(IPM,p)*cj;
      Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
      ForEachCompositeWeight(transfer, lt, pr(IPX,p), pr(IPY,p), pi(PLEV,p),
                             lt.Anchor(pi(PGID,p)),
      [&](const int target, const Real particle_weight) {
        int ti = target % ncells1;
        int remainder = target/ncells1;
        int tj = remainder % ncells2;
        int tm = remainder/ncells2;
        // the leaf's block slot may be a halo block: its box from lt
        Real receiving_volume = cloud ? lt.box(tm,2)*lt.box(tm,3)
                                      : mbsize.d_view(tm).dx1*mbsize.d_view(tm).dx2;
        Real density_weight = mass_c*particle_weight/receiving_volume;
        DepositAdd(&qdep_(tm,0,ks,tj,ti), density_weight);
        DepositAdd(&qdep_(tm,1,ks,tj,ti), density_weight*vx);
        DepositAdd(&qdep_(tm,2,ks,tj,ti), density_weight*vy);
        DepositAdd(&qdep_(tm,3,ks,tj,ti), density_weight*vz);
      });
    };
    if (cloud && cloud_ord_n > 0 && cloud_ord_n <= npart) {
      // In the order of the cell-sorted index (a permutation of its slots,
      // which the migration of this stage may have refilled)
      CloudParForOrdered("dust_composite_scatter", npart, cloud_ord,
                         cloud_ord_n, scat_body);
    } else {
      par_for("dust_composite_scatter",DevExeSpace(),0,(npart-1), scat_body);
    }
    if (cloud) HaloReverseAdd(qdep, 0, 4);   // halo deposits back to their owners
    return TaskStatus::complete;
  }

  par_for("dust_scatter",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp;
    Real wx[3], wy[3], wz[3];
    PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
              scheme, ip, wx);
    PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
              scheme, jp, wy);
    if (three_d) {
      PMWeights(pr(IPZ,p), mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks,
                scheme, kp, wz);
    } else {
      kp = ks;
      wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
    }
    // guard against particles that escaped migration (indicates an upstream bug)
    if (ip < (is-1) || ip > (is+nx1) || jp < (js-1) || jp > (js+nx2) ||
        (three_d && (kp < (ks-1) || kp > (ks+nx3)))) {
      Kokkos::printf("DustGasDrag halo violation: p=%d gid=%d x=(%.6e %.6e %.6e) "
                     "block x1=[%.4e,%.4e] x2=[%.4e,%.4e] x3=[%.4e,%.4e] "
                     "ip,jp,kp=(%d %d %d)\n", p, pi(PGID,p),
                     pr(IPX,p), pr(IPY,p), pr(IPZ,p),
                     mbsize.d_view(m).x1min, mbsize.d_view(m).x1max,
                     mbsize.d_view(m).x2min, mbsize.d_view(m).x2max,
                     mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, ip, jp, kp);
      Kokkos::abort("DustGasDrag: particle outside deposit halo");
    }

    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    Real cj  = a_dt/(pr(IPTS,p) + a_dt);
    Real muc = pr(IPM,p)*cj/vol;
    Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);

    int clo = three_d ? 0 : 1, chi = three_d ? 2 : 1;
    for (int c=clo; c<=chi; ++c) {
      for (int b=0; b<3; ++b) {
        Real wcb = wz[c]*wy[b]*muc;
        if (wcb == 0.0) continue;
        for (int a=0; a<3; ++a) {
          Real w = wcb*wx[a];
          int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
          DepositAdd(&qdep_(m,0,kk,jj,ii), w);
          DepositAdd(&qdep_(m,1,kk,jj,ii), w*vx);
          DepositAdd(&qdep_(m,2,kk,jj,ii), w*vy);
          DepositAdd(&qdep_(m,3,kk,jj,ii), w*vz);
        }
      }
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::UpdateStiffness
//! \brief The PC2 feedback stiffness of this cycle, chi = dt max_cells
//! sum_p w m_p/(t_s,p V rho_g) from the stage-1 predictor deposit qdep[4] (u0 still holds the
//! start-of-step gas at this point), reduced over all ranks; one log line per cycle on rank 0.

void DustGasDrag::UpdateStiffness(const Real dt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int ncells = pmy_pack->nmb_thispack*nx3*nx2*nx1;
  auto &u0 = pmy_pack->phydro->u0;
  auto &qdep_ = qdep;
  Real rate = 0.0;
  Kokkos::parallel_reduce("dust_stiffness", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
  KOKKOS_LAMBDA(const int idx, Real &maximum) {
    int q = idx;
    const int i = is + q % nx1; q /= nx1;
    const int j = js + q % nx2; q /= nx2;
    const int k = ks + q % nx3; q /= nx3;
    maximum = fmax(maximum, qdep_(q,4,k,j,i)/u0(q,IDN,k,j,i));
  }, Kokkos::Max<Real>(rate));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &rate, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  stiff_rate_max = rate;
  stiff_chi_last = dt*rate;
  stiff_chi_run_max = std::max(stiff_chi_run_max, stiff_chi_last);
  if (stiff_log && global_variable::my_rank == 0) {
    std::FILE *fp = AppendLog("dust_stiffness_log.txt");
    if (fp != nullptr) {
      if (LogEmpty(fp)) {
        std::fprintf(fp, "# PC2 feedback stiffness per cycle. ncycle time dt "
                     "chi=dt*max(sum w m/(t_s V rho_g)) dt_limit=stiff_thresh/max_rate\n");
      }
      std::fprintf(fp, "%d %.10g %.6e %.6e %.6e\n", pmy_pack->pmesh->ncycle, pmy_pack->pmesh->time,
                   dt, stiff_chi_last, (stiff_thresh > 0.0 && rate > 0.0) ? stiff_thresh/rate : -1.0);
      std::fflush(fp);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::UpdateHybridMetric
//! \brief Reduce the cell feedback stiffness deposited in qdep[4].  A PC2 cycle may
//! still fall back before committing its predictor; a split-BE cycle records the metric
//! for the next cycle's hysteretic choice.

void DustGasDrag::UpdateHybridMetric(Driver *pdrive, int stage) {
  // Only the automatic switch reads the metric; with a fixed mode (coupling = pc2,
  // hybrid_force_mode = pc2 / split_be) the full-mesh reduction and its host wait are skipped
  // and the metric stays "not computed" (hybrid_have_metric = false)
  if (!NeedsHybridMetric()) {InvalidateHybridMetric(); return;}
  hybrid_last_zeta = pmy_pack->pmesh->dt/taus_min;
  hybrid_last_chi = 0.0;
  if (back_reaction) {
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nx1 = ie-is+1, nx2 = je-js+1, nx3 = ke-ks+1;
    int nmb = pmy_pack->nmb_thispack;
    int ncells = nmb*nx3*nx2*nx1;
    auto &rho_state = (hybrid_mode == HybridMode::pc2 && stage == 1) ?
                      pmy_pack->phydro->u1 : pmy_pack->phydro->u0;
    auto &qdep_ = qdep;
    Real dt = pmy_pack->pmesh->dt;
    Real local_chi = 0.0;
    Kokkos::parallel_reduce("dust_hybrid_chi",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
    KOKKOS_LAMBDA(const int idx, Real &maximum) {
      int q = idx;
      int i = is + q % nx1; q /= nx1;
      int j = js + q % nx2; q /= nx2;
      int k = ks + q % nx3; q /= nx3;
      int m = q;
      Real value = dt*qdep_(m,4,k,j,i)/rho_state(m,IDN,k,j,i);
      maximum = fmax(maximum, value);
    }, Kokkos::Max<Real>(local_chi));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &local_chi, 1, MPI_ATHENA_REAL, MPI_MAX,
                  MPI_COMM_WORLD);
#endif
    hybrid_last_chi = local_chi;
    hybrid_feedback_rate_max = local_chi/dt;
  } else {
    hybrid_feedback_rate_max = 0.0;
  }
  hybrid_have_metric = true;
  if (hybrid_force_mode == HybridForceMode::automatic &&
      hybrid_mode == HybridMode::pc2 && stage == 1 &&
      (hybrid_last_zeta > hybrid_enter_zeta || hybrid_last_chi > hybrid_enter_chi)) {
    hybrid_mode = HybridMode::split_be;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::GasImplicitSolve
//! \brief Closed-form per-cell drag solve for the provisional gas velocity u*. Uses
//! conserved variables (primitives are stale mid-stage). Active cells only; ghost values
//! of u* are then filled by the copy exchange.
//! With back_reaction=false (test particles) the deposited dust weights must NOT enter
//! the solve: particles are kicked implicitly toward the unmodified gas velocity
//! u* = m_g/rho_g, and the gas is never updated (PMBR is skipped in ApplyPMBR). Folding
//! qdep into u* here would make test-particle trajectories depend on their own assigned
//! masses and on the other test particles.

TaskStatus DustGasDrag::GasImplicitSolve(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  if (coupling == DustCoupling::hybrid) {
    if (hybrid_mode == HybridMode::pc2 && stage == 1) {
      UpdateHybridMetric(pdrive, stage);
      if (hybrid_mode == HybridMode::pc2 && back_reaction) {
        if (StiffnessActive()) UpdateStiffness(pmy_pack->pmesh->dt);
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int is = indcs.is, ie = indcs.ie;
        int js = indcs.js, je = indcs.je;
        int ks = indcs.ks, ke = indcs.ke;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto &u0 = pmy_pack->phydro->u0;
        auto &qdep_ = qdep;
        const bool damp = pc2_feedback_damping;
        const Real dt = pmy_pack->pmesh->dt;
        par_for("dust_pc2_apply_predictor",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
          if (damp) {
            // Athena-C's feedback predictor (Bai & Stone 2010 §2.4 footnote): the
            // predictor feedback of a stiff cell is reduced by 1/max(1, chi_cell); qdep keeps the
            // scaled value, which ApplyPMBR subtracts again at stage 2 (momentum stays exact)
            const Real chi = dt*qdep_(m,4,k,j,i)/u0(m,IDN,k,j,i);
            if (chi > 1.0) {
              const Real s = 1.0/chi;
              qdep_(m,1,k,j,i) *= s; qdep_(m,2,k,j,i) *= s; qdep_(m,3,k,j,i) *= s;
            }
          }
          u0(m,IM1,k,j,i) += qdep_(m,1,k,j,i);
          u0(m,IM2,k,j,i) += qdep_(m,2,k,j,i);
          u0(m,IM3,k,j,i) += qdep_(m,3,k,j,i);
        });
      }
      return TaskStatus::complete;
    }
    if (hybrid_mode == HybridMode::split_be && stage == 2) {
      UpdateHybridMetric(pdrive, stage);
    } else {
      return TaskStatus::complete;
    }
  }
  if (back_reaction && drag_solver != DustDragSolver::local) {
    return SolveCoupledStage(pdrive, stage);
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0 = pmy_pack->phydro->u0;
  auto &qdep_ = qdep;
  auto &ustar_ = ustar;
  bool br = back_reaction;

  par_for("dust_ustar",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    Real qd0 = br ? qdep_(m,0,k,j,i) : 0.0;
    Real qd1 = br ? qdep_(m,1,k,j,i) : 0.0;
    Real qd2 = br ? qdep_(m,2,k,j,i) : 0.0;
    Real qd3 = br ? qdep_(m,3,k,j,i) : 0.0;
    Real denom = 1.0/(u0(m,IDN,k,j,i) + qd0);
    ustar_(m,0,k,j,i) = (u0(m,IM1,k,j,i) + qd1)*denom;
    ustar_(m,1,k,j,i) = (u0(m,IM2,k,j,i) + qd2)*denom;
    ustar_(m,2,k,j,i) = (u0(m,IM3,k,j,i) + qd3)*denom;
  });
  if (test_ustar_linear) {   // test: the known linear field on the active cells
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto &ms = pmy_pack->pmesh->mesh_size;
    UstarLinearField fld;
    fld.x0 = 0.5*(ms.x1min + ms.x1max); fld.y0 = 0.5*(ms.x2min + ms.x2max);
    fld.lx = ms.x1max - ms.x1min; fld.ly = ms.x2max - ms.x2min;
    par_for("dust_ustar_test",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real x = mbsize.d_view(m).x1min + (i - is + 0.5)*mbsize.d_view(m).dx1;
      const Real y = mbsize.d_view(m).x2min + (j - js + 0.5)*mbsize.d_view(m).dx2;
      for (int c=0; c<3; ++c) ustar_(m,c,k,j,i) = fld.Value(c, x, y);
    });
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckUstarLinear
//! \brief Test (<dust>/test_ustar_linear): with u* the known linear field on the
//! active cells, after its ghost exchange and the packing (PackGatherField, with the halo):
//! (1) on every leaf cell of this rank's blocks, the packed value and the two central
//! differences against the exact ones (the edge cells read the ghosts: same level, restricted
//! or prolongated); (2) for every cloud, the linear gather from the packed field against the
//! same gather from the exact values (same weights and moments; the halo slots included).
//! Cells within 5.5 cells of their level and clouds within 6 root cells of the domain edge are left
//! out (the field is not periodic; a prolongated ghost reads the slope of its coarse cell).  One line per call on rank 0 with the largest differences over all ranks.

void DustGasDrag::CheckUstarLinear(const int stage) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks, nx1 = indcs.nx1, nx2 = indcs.nx2;
  const int nmb = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &ms = pmy_pack->pmesh->mesh_size;
  UstarLinearField fld;
  fld.x0 = 0.5*(ms.x1min + ms.x1max); fld.y0 = 0.5*(ms.x2min + ms.x2max);
  fld.lx = ms.x1max - ms.x1min; fld.ly = ms.x2max - ms.x2min;
  const Real x1min = ms.x1min, x1max = ms.x1max, x2min = ms.x2min, x2max = ms.x2max;
  const Real rdx = fld.lx/pmy_pack->pmesh->mesh_indcs.nx1, rdy = fld.ly/pmy_pack->pmesh->mesh_indcs.nx2;
  auto gp = gpack;
  // (1) the leaf cells of this rank
  Real ev = 0.0, ex = 0.0, ey = 0.0;
  int ncell = 0, nbad_cell = 0, nbad_cloud = 0;
  const int ncb = nx1*nx2;
  Kokkos::parallel_reduce("dust_ustar_check_cells", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmb*ncb),
  KOKKOS_LAMBDA(const int idx, Real &v, Real &sx, Real &sy, int &n, int &bad) {
    const int m = idx/ncb, j = (idx % ncb)/nx1 + js, i = idx % nx1 + is;
    const Real dx = mbsize.d_view(m).dx1, dy = mbsize.d_view(m).dx2;
    const Real x = mbsize.d_view(m).x1min + (i - is + 0.5)*dx, y = mbsize.d_view(m).x2min + (j - js + 0.5)*dy;
    // the edge rows: a central difference reads one cell further, and a prolongated ghost uses the slope
    // of its coarse cell, which reads one coarse cell further (across the periodic edge the test field
    // jumps): 5.5 cells of the cell's level
    if (x - 5.5*dx < x1min || x + 5.5*dx > x1max || y - 5.5*dy < x2min || y + 5.5*dy > x2max) return;
    ++n;
    for (int q=0; q<9; ++q) if (!Kokkos::isfinite(gp(m,ks,j,i,q))) {++bad; return;}   // fmax drops NaN
    for (int c=0; c<3; ++c) {
      v = fmax(v, fabs(gp(m,ks,j,i,c) - fld.Value(c, x, y)));
      sx = fmax(sx, fabs(gp(m,ks,j,i,3+c) - 2.0*dx*fld.Bx(c)));
      sy = fmax(sy, fabs(gp(m,ks,j,i,6+c) - 2.0*dy*fld.By(c)));
    }
  }, Kokkos::Max<Real>(ev), Kokkos::Max<Real>(ex), Kokkos::Max<Real>(ey), Kokkos::Sum<int>(ncell),
     Kokkos::Sum<int>(nbad_cell));
  // (2) the clouds
  particles::Particles *ppar = pmy_pack->ppart;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  auto transfer = smr_transfer;
  auto lt = leaf_traversal;
  const int ncells1 = ustar.extent_int(4), ncells2 = ustar.extent_int(3);
  const int tmode = cloud_transition;
  const Real mx = 6.0*rdx, my = 6.0*rdy;
  Real eg = 0.0, ua = 0.0;
  int nchk = 0;
  Kokkos::parallel_reduce("dust_ustar_check_clouds", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, Real &e, Real &u, int &n, int &bad) {
    const Real x0 = pr(IPX,p), y0 = pr(IPY,p);
    if (x0 - mx < x1min || x0 + mx > x1max || y0 - my < x2min || y0 + my > x2max) return;
    ++n;
    const int glev = SMRCompositeTransfer::GatherCode(pi(PLEV,p), tmode);
    Real ug[3] = {0.0, 0.0, 0.0}, ux[3] = {0.0, 0.0, 0.0};
    lt.ForEachCloudMoment(x0, y0, glev, lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w, const Real m1x, const Real m1y) {
      const int ti = target % ncells1, tj = (target/ncells1) % ncells2, tm = target/(ncells1*ncells2);
      const Real dx1 = lt.box(tm,2), dx2 = lt.box(tm,3);
      const Real xc = lt.box(tm,0) + (ti - is + 0.5)*dx1, yc = lt.box(tm,1) + (tj - js + 0.5)*dx2;
      const Real cx = 0.5*(m1x + transfer.MinImage1(x0 - xc)*w)/dx1;
      const Real cy = 0.5*(m1y + transfer.MinImage2(y0 - yc)*w)/dx2;
      for (int c=0; c<3; ++c) {
        ug[c] += w*gp(tm,ks,tj,ti,c) + cx*gp(tm,ks,tj,ti,3+c) + cy*gp(tm,ks,tj,ti,6+c);
        ux[c] += w*fld.Value(c, xc, yc) + cx*2.0*dx1*fld.Bx(c) + cy*2.0*dx2*fld.By(c);
      }
    });
    for (int c=0; c<3; ++c) {
      if (!Kokkos::isfinite(ug[c])) {++bad; return;}
      e = fmax(e, fabs(ug[c] - ux[c])); u = fmax(u, fabs(ux[c]));
    }
  }, Kokkos::Max<Real>(eg), Kokkos::Max<Real>(ua), Kokkos::Sum<int>(nchk), Kokkos::Sum<int>(nbad_cloud));
  // the worst cells (test output): where the differences exceed 1e-12
  if (ex > 1.0e-12 || ey > 1.0e-12 || ev > 1.0e-12) {
    auto gph = Kokkos::create_mirror_view_and_copy(HostMemSpace(), gp);
    int shown = 0;
    for (int m=0; m<nmb && shown < 6; ++m) {
      const auto &sz = mbsize.h_view(m);
      const auto &ll = pmy_pack->pmesh->lloc_eachmb[pmy_pack->gids + m];
      for (int j=js; j<js+nx2 && shown < 6; ++j) {
        for (int i=is; i<is+nx1 && shown < 6; ++i) {
          const Real x = sz.x1min + (i - is + 0.5)*sz.dx1, y = sz.x2min + (j - js + 0.5)*sz.dx2;
          if (x - 5.5*sz.dx1 < x1min || x + 5.5*sz.dx1 > x1max || y - 5.5*sz.dx2 < x2min ||
              y + 5.5*sz.dx2 > x2max) continue;
          for (int c=0; c<3; ++c) {
            const Real d1 = gph(m,ks,j,i,3+c) - 2.0*sz.dx1*fld.Bx(c);
            const Real d2 = gph(m,ks,j,i,6+c) - 2.0*sz.dx2*fld.By(c);
            const Real d0 = gph(m,ks,j,i,c) - fld.Value(c, x, y);
            if (std::fabs(d0) > 1.0e-12 || std::fabs(d1) > 1.0e-12 || std::fabs(d2) > 1.0e-12) {
              std::printf("#   rank %d block gid %d level %d lx (%d,%d) cell (%d,%d) x %.6f y %.6f comp %d: "
                          "value %.3e x1diff %.3e x2diff %.3e\n", global_variable::my_rank,
                          pmy_pack->gids + m, ll.level, ll.lx1, ll.lx2, i - is, j - js, x, y, c, d0, d1, d2);
              ++shown;
              break;
            }
          }
        }
      }
    }
  }
  Real mxv[5] = {ev, ex, ey, eg, ua};
  long long cnt[4] = {ncell, nchk, nbad_cell, nbad_cloud};
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, mxv, 5, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, cnt, 4, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank == 0) {
    std::printf("# DUST_USTAR_CHECK cycle %d stage %d: %lld leaf cells, max |value - exact| %.3e, "
                "|x1 difference - exact| %.3e, |x2 difference - exact| %.3e; %lld clouds, "
                "max |gather - exact| %.3e (largest |u| %.3e); non-finite cells %lld clouds %lld\n",
                pmy_pack->pmesh->ncycle, stage, cnt[0], mxv[0], mxv[1], mxv[2], cnt[1], mxv[3], mxv[4],
                cnt[2], cnt[3]);
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::GatherKickPMBR
//! \brief Fused gather + implicit particle kick + drag-rate record + PMBR scatter.
//! Weights are computed once and used for both the gather and the back-reaction deposit,
//! which makes the momentum exchange antisymmetric to round-off.

TaskStatus DustGasDrag::GatherKickPMBR(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  if (coupling == DustCoupling::hybrid &&
      !(hybrid_mode == HybridMode::split_be && stage == 2)) {
    return TaskStatus::complete;
  }

  bool br = back_reaction;
  if (br) {
    Kokkos::deep_copy(DevExeSpace(), dmom, 0.0);
  }

  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto gids = pmy_pack->gids;
  int scheme = static_cast<int>(deposit);
  Real a_dt = DragStep(pdrive);
  Real dt = pmy_pack->pmesh->dt;
  bool split_be = (coupling == DustCoupling::hybrid);
  bool shear3d = is_shearing_box && three_d;
  Real qo = qshear*omega0;
  auto &ustar_ = ustar;
  auto &dmom_ = dmom;

  if (UsesCompositeTransfer()) {
    auto transfer = smr_transfer;
    auto lt = leaf_traversal;   // cloud weights on the actual leaves
    int ncells1 = ustar.extent_int(4);
    int ncells2 = ustar.extent_int(3);
    int tmode = cloud_transition;
    if (transfer.cloud_mode) {
      // Refinable clouds (2026-10-17): the gather reads the packed u* field (velocity and
      // its central differences, the owners' values on halo blocks).
      // cloud_gather_order = 2 gathers the linear reconstruction of u* over the cloud, with
      // the lobed weights and point values of transition mode 2, as the PC2 midpoint gather
      // does with its midpoint velocity; order 1 the cell values (piecewise constant).  The
      // impulse is deposited with the cloud's own weights.
      const bool linear = (cloud_gather_order == 2);
      PackGatherField(ustar);
      if (test_ustar_linear) CheckUstarLinear(stage);   // test
      auto gp = gpack;
      auto gk_body = KOKKOS_LAMBDA(const int p) {
        // the gather's leaf weights are the deposit's when the shapes agree: keep them
        int nt = 0, tt[SMRCompositeTransfer::kCloudWeightCache];
        Real tw[SMRCompositeTransfer::kCloudWeightCache];
        Real x0 = pr(IPX,p), y0 = pr(IPY,p);
        const int am = lt.Anchor(pi(PGID,p));   // anchor block of the cloud
        const int lev = pi(PLEV,p);
        const int glev = SMRCompositeTransfer::GatherCode(lev, tmode);
        // same shape for the gather and the deposit (the point flag changes moments only)
        const bool same = (glev & ~SMRCompositeTransfer::kCloudGatherPoint) == lev;
        Real ux = 0.0, uy = 0.0, uz = 0.0;
        if (linear) {
          lt.ForEachCloudMoment(x0, y0, glev, am,
          [&](const int target, const Real w, const Real m1x, const Real m1y) {
            int ti = target % ncells1;
            int remainder = target/ncells1;
            int tj = remainder % ncells2;
            int tm = remainder/ncells2;
            // the leaf's block slot may be a halo block: its box from lt
            Real dx1 = lt.box(tm,2), dx2 = lt.box(tm,3);
            Real xc = lt.box(tm,0) + (ti - is + 0.5)*dx1;
            Real yc = lt.box(tm,1) + (tj - js + 0.5)*dx2;
            Real cx = 0.5*(m1x + transfer.MinImage1(x0 - xc)*w)/dx1;
            Real cy = 0.5*(m1y + transfer.MinImage2(y0 - yc)*w)/dx2;
            ux += w*gp(tm,ks,tj,ti,0) + cx*gp(tm,ks,tj,ti,3) + cy*gp(tm,ks,tj,ti,6);
            uy += w*gp(tm,ks,tj,ti,1) + cx*gp(tm,ks,tj,ti,4) + cy*gp(tm,ks,tj,ti,7);
            uz += w*gp(tm,ks,tj,ti,2) + cx*gp(tm,ks,tj,ti,5) + cy*gp(tm,ks,tj,ti,8);
            if (nt < SMRCompositeTransfer::kCloudWeightCache) {tt[nt] = target; tw[nt] = w;}
            ++nt;
          });
        } else {
          lt.ForEachCloudWeight(x0, y0, glev, am,
          [&](const int target, const Real w) {
            int ti = target % ncells1;
            int remainder = target/ncells1;
            int tj = remainder % ncells2;
            int tm = remainder/ncells2;
            ux += w*gp(tm,ks,tj,ti,0);
            uy += w*gp(tm,ks,tj,ti,1);
            uz += w*gp(tm,ks,tj,ti,2);
            if (nt < SMRCompositeTransfer::kCloudWeightCache) {tt[nt] = target; tw[nt] = w;}
            ++nt;
          });
        }

        Real cj = a_dt/(pr(IPTS,p) + a_dt);
        Real dvx = cj*(ux - pr(IPVX,p));
        Real dvy = cj*(uy - pr(IPVY,p));
        Real dvz = cj*(uz - pr(IPVZ,p));
        pr(IPVX,p) += dvx;
        pr(IPVY,p) += dvy;
        pr(IPVZ,p) += dvz;
        pr(IPRX,p) = dvx/a_dt;
        pr(IPRY,p) = dvy/a_dt;
        pr(IPRZ,p) = dvz/a_dt;

        if (br) {
          Real mass = -pr(IPM,p);
          auto add = [&](const int target, const Real particle_weight) {
            int ti = target % ncells1;
            int remainder = target/ncells1;
            int tj = remainder % ncells2;
            int tm = remainder/ncells2;
            Real receiving_volume = lt.box(tm,2)*lt.box(tm,3);
            Real density_weight = mass*particle_weight/receiving_volume;
            DepositAdd(&dmom_(tm,0,ks,tj,ti), density_weight*dvx);
            DepositAdd(&dmom_(tm,1,ks,tj,ti), density_weight*dvy);
            DepositAdd(&dmom_(tm,2,ks,tj,ti), density_weight*dvz);
          };
          if (same && nt <= SMRCompositeTransfer::kCloudWeightCache) {
            for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
          } else {
            lt.ForEachCloudWeight(x0, y0, lev, am, add);
          }
        }
        if (split_be) {
          pr(IPX,p) += dt*pr(IPVX,p);
          pr(IPY,p) += dt*(pr(IPVY,p) - (shear3d ? qo*x0 : 0.0));
          if (three_d) pr(IPZ,p) += dt*pr(IPVZ,p);
        }
      };
      if (cloud_ord_n > 0 && cloud_ord_n <= npart) {
        // In the order of the cell-sorted index (a permutation of its slots)
        CloudParForOrdered("dust_cloud_gather_kick", npart, cloud_ord,
                           cloud_ord_n, gk_body);
      } else {
        CloudParFor("dust_cloud_gather_kick", npart, gk_body);
      }
      if (br) HaloReverseAdd(dmom, 0, 3);   // halo deposits back to their owners
      return TaskStatus::complete;
    }
    par_for("dust_composite_gather",DevExeSpace(),0,(npart-1),
    KOKKOS_LAMBDA(const int p) {
      Real x0 = pr(IPX,p), y0 = pr(IPY,p);
      Real ux = 0.0, uy = 0.0, uz = 0.0;
      ForEachCompositeWeight(transfer, lt, x0, y0,
                             SMRCompositeTransfer::GatherCode(pi(PLEV,p), tmode),
                             lt.Anchor(pi(PGID,p)),
      [&](const int target, const Real particle_weight) {
        int ti = target % ncells1;
        int remainder = target/ncells1;
        int tj = remainder % ncells2;
        int tm = remainder/ncells2;
        ux += particle_weight*ustar_(tm,0,ks,tj,ti);
        uy += particle_weight*ustar_(tm,1,ks,tj,ti);
        uz += particle_weight*ustar_(tm,2,ks,tj,ti);
      });

      Real cj = a_dt/(pr(IPTS,p) + a_dt);
      Real dvx = cj*(ux - pr(IPVX,p));
      Real dvy = cj*(uy - pr(IPVY,p));
      Real dvz = cj*(uz - pr(IPVZ,p));
      pr(IPVX,p) += dvx;
      pr(IPVY,p) += dvy;
      pr(IPVZ,p) += dvz;
      pr(IPRX,p) = dvx/a_dt;
      pr(IPRY,p) = dvy/a_dt;
      pr(IPRZ,p) = dvz/a_dt;

      if (br) {
        Real mass = -pr(IPM,p);
        ForEachCompositeWeight(transfer, lt, x0, y0, pi(PLEV,p), lt.Anchor(pi(PGID,p)),
        [&](const int target, const Real particle_weight) {
          int ti = target % ncells1;
          int remainder = target/ncells1;
          int tj = remainder % ncells2;
          int tm = remainder/ncells2;
          Real receiving_volume = mbsize.d_view(tm).dx1*mbsize.d_view(tm).dx2;
          Real density_weight = mass*particle_weight/receiving_volume;
          DepositAdd(&dmom_(tm,0,ks,tj,ti), density_weight*dvx);
          DepositAdd(&dmom_(tm,1,ks,tj,ti), density_weight*dvy);
          DepositAdd(&dmom_(tm,2,ks,tj,ti), density_weight*dvz);
        });
      }
      if (split_be) {
        pr(IPX,p) += dt*pr(IPVX,p);
        pr(IPY,p) += dt*(pr(IPVY,p) - (shear3d ? qo*x0 : 0.0));
        if (three_d) pr(IPZ,p) += dt*pr(IPVZ,p);
      }
    });
    return TaskStatus::complete;
  }

  par_for("dust_gather",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp;
    Real wx[3], wy[3], wz[3];
    PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
              scheme, ip, wx);
    PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
              scheme, jp, wy);
    if (three_d) {
      PMWeights(pr(IPZ,p), mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks,
                scheme, kp, wz);
    } else {
      kp = ks;
      wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
    }

    // gather provisional gas velocity at particle position
    Real ux = 0.0, uy = 0.0, uz = 0.0;
    int clo = three_d ? 0 : 1, chi = three_d ? 2 : 1;
    for (int c=clo; c<=chi; ++c) {
      for (int b=0; b<3; ++b) {
        Real wcb = wz[c]*wy[b];
        if (wcb == 0.0) continue;
        for (int a=0; a<3; ++a) {
          Real w = wcb*wx[a];
          int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
          ux += w*ustar_(m,0,kk,jj,ii);
          uy += w*ustar_(m,1,kk,jj,ii);
          uz += w*ustar_(m,2,kk,jj,ii);
        }
      }
    }

    // implicit drag kick and drag-rate record
    Real cj  = a_dt/(pr(IPTS,p) + a_dt);
    Real dvx = cj*(ux - pr(IPVX,p));
    Real dvy = cj*(uy - pr(IPVY,p));
    Real dvz = cj*(uz - pr(IPVZ,p));
    pr(IPVX,p) += dvx;
    pr(IPVY,p) += dvy;
    pr(IPVZ,p) += dvz;
    pr(IPRX,p) = dvx/a_dt;
    pr(IPRY,p) = dvy/a_dt;
    pr(IPRZ,p) = dvz/a_dt;

    // The split fallback holds positions fixed during the coupled solve, then uses the
    // relaxed velocity for its deliberately first-order drift.
    if (split_be) {
      Real x0 = pr(IPX,p);
      pr(IPX,p) += dt*pr(IPVX,p);
      pr(IPY,p) += dt*(pr(IPVY,p) - (shear3d ? qo*x0 : 0.0));
      if (three_d) pr(IPZ,p) += dt*pr(IPVZ,p);
    }

    // momentum back-reaction deposit with the SAME weights and dv
    if (br) {
      Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
      if (three_d) {vol *= mbsize.d_view(m).dx3;}
      Real fac = -pr(IPM,p)/vol;
      for (int c=clo; c<=chi; ++c) {
        for (int b=0; b<3; ++b) {
          Real wcb = wz[c]*wy[b]*fac;
          if (wcb == 0.0) continue;
          for (int a=0; a<3; ++a) {
            Real w = wcb*wx[a];
            int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
            DepositAdd(&dmom_(m,0,kk,jj,ii), w*dvx);
            DepositAdd(&dmom_(m,1,kk,jj,ii), w*dvy);
            DepositAdd(&dmom_(m,2,kk,jj,ii), w*dvz);
          }
        }
      }
    }
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ApplyPMBR
//! \brief Adds the (halo-exchanged) back-reaction momentum deposit to the gas conserved
//! variables over active cells, then rescales dmom in place into the gas drag-rate field
//! R_g = dmom/(a_impl*dt) consumed by AddDragHistoryGas at the next stage.

TaskStatus DustGasDrag::ApplyPMBR(Driver *pdrive, int stage) {
  if (prescribed_gas && coupling == DustCoupling::hybrid && stage == 2 &&
      ActiveStage(pdrive, stage)) {
    // diagnostic prescribed gas (back_reaction=false): reset the gas to the prescribed
    // end-of-step state before the boundary tail
    prescribed_gas(pmy_pack->pmesh->time + pmy_pack->pmesh->dt, pmy_pack->phydro->u0);
  }
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}

  if (coupling == DustCoupling::hybrid) {
    if (stage != 2) return TaskStatus::complete;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb1 = pmy_pack->nmb_thispack - 1;
    auto &u0 = pmy_pack->phydro->u0;
    auto &dmom_ = dmom;
    auto &qdep_ = qdep;
    bool pc2 = (hybrid_mode == HybridMode::pc2);
    Real wpred = pdrive->gam1[pdrive->nexp_stages-1];
    par_for("dust_hybrid_commit",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      u0(m,IM1,k,j,i) += dmom_(m,0,k,j,i) - (pc2 ? wpred*qdep_(m,1,k,j,i) : 0.0);
      u0(m,IM2,k,j,i) += dmom_(m,1,k,j,i) - (pc2 ? wpred*qdep_(m,2,k,j,i) : 0.0);
      u0(m,IM3,k,j,i) += dmom_(m,2,k,j,i) - (pc2 ? wpred*qdep_(m,3,k,j,i) : 0.0);
    });
    return TaskStatus::complete;
  }

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0 = pmy_pack->phydro->u0;
  auto &dmom_ = dmom;
  Real inv_adt = 1.0/((pdrive->a_impl)*(pmy_pack->pmesh->dt));

  par_for("dust_pmbr",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u0(m,IM1,k,j,i) += dmom_(m,0,k,j,i);
    u0(m,IM2,k,j,i) += dmom_(m,1,k,j,i);
    u0(m,IM3,k,j,i) += dmom_(m,2,k,j,i);
    dmom_(m,0,k,j,i) *= inv_adt;
    dmom_(m,1,k,j,i) *= inv_adt;
    dmom_(m,2,k,j,i) *= inv_adt;
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::AddDragHistoryGas
//! \brief Adds the a_twid history combination of the recorded gas drag rates to the
//! partially-updated conserved variables, mirroring block (i) of
//! IonNeutral::ImpRKUpdate. For imex2+ the only nonzero coefficient is a_twid[2][2],
//! which weights the drag rate recorded at the previous (stage-1) implicit solve and is
//! consumed at explicit stage 2.

TaskStatus DustGasDrag::AddDragHistoryGas(Driver *pdrive, int stage) {
  if (coupling != DustCoupling::imex) {return TaskStatus::complete;}
  if (stage != 2 || !back_reaction) {return TaskStatus::complete;}

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0 = pmy_pack->phydro->u0;
  auto &dmom_ = dmom;   // holds R_g recorded at the previous stage
  Real atw_dt = (pdrive->a_twid[2][2])*(pmy_pack->pmesh->dt);

  par_for("dust_gatwid",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    u0(m,IM1,k,j,i) += atw_dt*dmom_(m,0,k,j,i);
    u0(m,IM2,k,j,i) += atw_dt*dmom_(m,1,k,j,i);
    u0(m,IM3,k,j,i) += atw_dt*dmom_(m,2,k,j,i);
  });

  return TaskStatus::complete;
}

} // namespace dust
