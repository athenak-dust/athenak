//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_pm.cpp
//! \brief Particle-mesh kernels for the implicit gas-dust drag solve, executed once per
//! working IMEX stage with implicit coefficient a = a_impl*dt:
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

namespace dust {

namespace {

// Launch a scheme- and dimension-specialized scatter.  Compact supports remove all
// zero-weight NGP/CIC stencil entries before the GPU kernel is instantiated.
template <DustDeposit Scheme, bool ThreeD, typename MBSizeView>
void DepositDragKernel(DvceArray2D<Real> pr, DvceArray2D<int> pi,
                       MBSizeView mbsize, DvceArray5D<Real> qdep,
                       const int npart, const int gids,
                       const int is, const int js, const int ks,
                       const int nx1, const int nx2, const int nx3,
                       const Real a_dt) {
  constexpr int nstencil = PMStencilWidth<Scheme>::value;
  constexpr int nk = ThreeD ? nstencil : 1;
  const char *label = (Scheme == DustDeposit::ngp) ? "dust_scatter_ngp" :
                      (Scheme == DustDeposit::cic) ? "dust_scatter_cic" :
                                                    "dust_scatter_tsc";

  par_for(label,DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp, i0, j0, k0;
    Real wx[nstencil], wy[nstencil], wz[nk];
    // Capture z-dimension bounds outside if constexpr for CUDA extended lambdas.
    const int nx3_ = nx3;
    kp = ks;
    k0 = ks;
    wz[0] = 1.0;
    PMCompactWeights<Scheme>(pr(IPX,p), mbsize(m).x1min, mbsize(m).x1max,
                             nx1, is, ip, i0, wx);
    PMCompactWeights<Scheme>(pr(IPY,p), mbsize(m).x2min, mbsize(m).x2max,
                             nx2, js, jp, j0, wy);
    if constexpr (ThreeD) {
      PMCompactWeights<Scheme>(pr(IPZ,p), mbsize(m).x3min, mbsize(m).x3max,
                               nx3_, ks, kp, k0, wz);
    }

    // guard against particles that escaped migration (indicates an upstream bug)
    if (ip < (is-1) || ip > (is+nx1) || jp < (js-1) || jp > (js+nx2) ||
        (ThreeD && (kp < (ks-1) || kp > (ks+nx3)))) {
      Kokkos::printf("DustGasDrag halo violation: p=%d gid=%d x=(%.6e %.6e %.6e) "
                     "block x1=[%.4e,%.4e] x2=[%.4e,%.4e] x3=[%.4e,%.4e] "
                     "ip,jp,kp=(%d %d %d)\n", p, pi(PGID,p),
                     pr(IPX,p), pr(IPY,p), pr(IPZ,p),
                     mbsize(m).x1min, mbsize(m).x1max,
                     mbsize(m).x2min, mbsize(m).x2max,
                     mbsize(m).x3min, mbsize(m).x3max, ip, jp, kp);
      Kokkos::abort("DustGasDrag: particle outside deposit halo");
    }

    Real vol = mbsize(m).dx1*mbsize(m).dx2;
    if constexpr (ThreeD) {vol *= mbsize(m).dx3;}
    Real inv_vol = 1.0/vol;
    Real cj  = a_dt/(pr(IPTS,p) + a_dt);
    Real muc = pr(IPM,p)*cj*inv_vol;
    Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);

    for (int c=0; c<nk; ++c) {
      for (int b=0; b<nstencil; ++b) {
        Real wcb = wz[c]*wy[b]*muc;
        if (wcb == 0.0) {continue;}
        for (int a=0; a<nstencil; ++a) {
          Real w = wcb*wx[a];
          if (w == 0.0) {continue;}
          int kk = k0+c, jj = j0+b, ii = i0+a;
          Kokkos::atomic_add(&qdep(m,0,kk,jj,ii), w);
          Kokkos::atomic_add(&qdep(m,1,kk,jj,ii), w*vx);
          Kokkos::atomic_add(&qdep(m,2,kk,jj,ii), w*vy);
          Kokkos::atomic_add(&qdep(m,3,kk,jj,ii), w*vz);
        }
      }
    }
  });
}

// Launch a matching compact gather/kick/back-reaction kernel.  Scheme specialization is
// important here because both the interpolation and PMBR scatter traverse the stencil.
template <DustDeposit Scheme, bool ThreeD, typename MBSizeView>
void GatherKickPMBRKernel(DvceArray2D<Real> pr, DvceArray2D<int> pi,
                          MBSizeView mbsize, DvceArray5D<Real> ustar,
                          DvceArray5D<Real> dmom, const int npart,
                          const int gids, const int is, const int js, const int ks,
                          const int nx1, const int nx2, const int nx3,
                          const Real a_dt, const Real inv_a_dt,
                          const bool back_reaction) {
  constexpr int nstencil = PMStencilWidth<Scheme>::value;
  constexpr int nk = ThreeD ? nstencil : 1;
  const char *label = (Scheme == DustDeposit::ngp) ? "dust_gather_ngp" :
                      (Scheme == DustDeposit::cic) ? "dust_gather_cic" :
                                                    "dust_gather_tsc";

  par_for(label,DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp, i0, j0, k0;
    Real wx[nstencil], wy[nstencil], wz[nk];
    // Capture z-dimension bounds outside if constexpr for CUDA extended lambdas.
    const int nx3_ = nx3;
    kp = ks;
    k0 = ks;
    wz[0] = 1.0;
    PMCompactWeights<Scheme>(pr(IPX,p), mbsize(m).x1min, mbsize(m).x1max,
                             nx1, is, ip, i0, wx);
    PMCompactWeights<Scheme>(pr(IPY,p), mbsize(m).x2min, mbsize(m).x2max,
                             nx2, js, jp, j0, wy);
    if constexpr (ThreeD) {
      PMCompactWeights<Scheme>(pr(IPZ,p), mbsize(m).x3min, mbsize(m).x3max,
                               nx3_, ks, kp, k0, wz);
    }

    // gather provisional gas velocity at particle position
    Real ux = 0.0, uy = 0.0, uz = 0.0;
    for (int c=0; c<nk; ++c) {
      for (int b=0; b<nstencil; ++b) {
        Real wcb = wz[c]*wy[b];
        if (wcb == 0.0) {continue;}
        for (int a=0; a<nstencil; ++a) {
          Real w = wcb*wx[a];
          if (w == 0.0) {continue;}
          int kk = k0+c, jj = j0+b, ii = i0+a;
          ux += w*ustar(m,0,kk,jj,ii);
          uy += w*ustar(m,1,kk,jj,ii);
          uz += w*ustar(m,2,kk,jj,ii);
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
    pr(IPRX,p) = dvx*inv_a_dt;
    pr(IPRY,p) = dvy*inv_a_dt;
    pr(IPRZ,p) = dvz*inv_a_dt;

    // momentum back-reaction deposit with the SAME weights and dv
    if (back_reaction) {
      Real vol = mbsize(m).dx1*mbsize(m).dx2;
      if constexpr (ThreeD) {vol *= mbsize(m).dx3;}
      Real inv_vol = 1.0/vol;
      Real fac = -pr(IPM,p)*inv_vol;
      for (int c=0; c<nk; ++c) {
        for (int b=0; b<nstencil; ++b) {
          Real wcb = wz[c]*wy[b]*fac;
          if (wcb == 0.0) {continue;}
          for (int a=0; a<nstencil; ++a) {
            Real w = wcb*wx[a];
            if (w == 0.0) {continue;}
            int kk = k0+c, jj = j0+b, ii = i0+a;
            Kokkos::atomic_add(&dmom(m,0,kk,jj,ii), w*dvx);
            Kokkos::atomic_add(&dmom(m,1,kk,jj,ii), w*dvy);
            Kokkos::atomic_add(&dmom(m,2,kk,jj,ii), w*dvz);
          }
        }
      }
    }
  });
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositDrag
//! \brief Zero the (Q,P) field and scatter the drag-weighted particle sums into it.

TaskStatus DustGasDrag::DepositDrag(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}

  Kokkos::deep_copy(DevExeSpace(), qdep, 0.0);

  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  bool three_d = pmy_pack->pmesh->three_d;
  auto mbsize = pmy_pack->pmb->mb_size.d_view;
  auto gids = pmy_pack->gids;
  Real a_dt = (pdrive->a_impl)*(pmy_pack->pmesh->dt);

  if (three_d) {
    if (deposit == DustDeposit::ngp) {
      DepositDragKernel<DustDeposit::ngp,true>(
          pr, pi, mbsize, qdep, npart, gids, is, js, ks, nx1, nx2, nx3, a_dt);
    } else if (deposit == DustDeposit::cic) {
      DepositDragKernel<DustDeposit::cic,true>(
          pr, pi, mbsize, qdep, npart, gids, is, js, ks, nx1, nx2, nx3, a_dt);
    } else {
      DepositDragKernel<DustDeposit::tsc,true>(
          pr, pi, mbsize, qdep, npart, gids, is, js, ks, nx1, nx2, nx3, a_dt);
    }
  } else {
    if (deposit == DustDeposit::ngp) {
      DepositDragKernel<DustDeposit::ngp,false>(
          pr, pi, mbsize, qdep, npart, gids, is, js, ks, nx1, nx2, nx3, a_dt);
    } else if (deposit == DustDeposit::cic) {
      DepositDragKernel<DustDeposit::cic,false>(
          pr, pi, mbsize, qdep, npart, gids, is, js, ks, nx1, nx2, nx3, a_dt);
    } else {
      DepositDragKernel<DustDeposit::tsc,false>(
          pr, pi, mbsize, qdep, npart, gids, is, js, ks, nx1, nx2, nx3, a_dt);
    }
  }

  return TaskStatus::complete;
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
  auto mbsize = pmy_pack->pmb->mb_size.d_view;
  auto gids = pmy_pack->gids;
  Real a_dt = (pdrive->a_impl)*(pmy_pack->pmesh->dt);
  Real inv_a_dt = 1.0/a_dt;

  if (three_d) {
    if (deposit == DustDeposit::ngp) {
      GatherKickPMBRKernel<DustDeposit::ngp,true>(
          pr, pi, mbsize, ustar, dmom, npart, gids, is, js, ks, nx1, nx2, nx3,
          a_dt, inv_a_dt, br);
    } else if (deposit == DustDeposit::cic) {
      GatherKickPMBRKernel<DustDeposit::cic,true>(
          pr, pi, mbsize, ustar, dmom, npart, gids, is, js, ks, nx1, nx2, nx3,
          a_dt, inv_a_dt, br);
    } else {
      GatherKickPMBRKernel<DustDeposit::tsc,true>(
          pr, pi, mbsize, ustar, dmom, npart, gids, is, js, ks, nx1, nx2, nx3,
          a_dt, inv_a_dt, br);
    }
  } else {
    if (deposit == DustDeposit::ngp) {
      GatherKickPMBRKernel<DustDeposit::ngp,false>(
          pr, pi, mbsize, ustar, dmom, npart, gids, is, js, ks, nx1, nx2, nx3,
          a_dt, inv_a_dt, br);
    } else if (deposit == DustDeposit::cic) {
      GatherKickPMBRKernel<DustDeposit::cic,false>(
          pr, pi, mbsize, ustar, dmom, npart, gids, is, js, ks, nx1, nx2, nx3,
          a_dt, inv_a_dt, br);
    } else {
      GatherKickPMBRKernel<DustDeposit::tsc,false>(
          pr, pi, mbsize, ustar, dmom, npart, gids, is, js, ks, nx1, nx2, nx3,
          a_dt, inv_a_dt, br);
    }
  }

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ApplyPMBR
//! \brief Adds the (halo-exchanged) back-reaction momentum deposit to the gas conserved
//! variables over active cells, then rescales dmom in place into the gas drag-rate field
//! R_g = dmom/(a_impl*dt) consumed by AddDragHistoryGas at the next stage.

TaskStatus DustGasDrag::ApplyPMBR(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}

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
