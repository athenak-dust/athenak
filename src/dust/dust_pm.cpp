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

TaskStatus DustGasDrag::DepositDrag(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}

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
            DepositAdd(&qdep_(m,4,kk,jj,ii), w*rate);
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
//! \fn DustGasDrag::UpdateHybridMetric
//! \brief Reduce the cell feedback stiffness deposited in qdep[4].  A PC2 cycle may
//! still fall back before committing its predictor; a split-BE cycle records the metric
//! for the next cycle's hysteretic choice.

void DustGasDrag::UpdateHybridMetric(Driver *pdrive, int stage) {
  if (coupling != DustCoupling::hybrid) return;
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
        auto &indcs = pmy_pack->pmesh->mb_indcs;
        int is = indcs.is, ie = indcs.ie;
        int js = indcs.js, je = indcs.je;
        int ks = indcs.ks, ke = indcs.ke;
        int nmb1 = pmy_pack->nmb_thispack - 1;
        auto &u0 = pmy_pack->phydro->u0;
        auto &qdep_ = qdep;
        par_for("dust_pc2_apply_predictor",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
        KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
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

  return TaskStatus::complete;
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
