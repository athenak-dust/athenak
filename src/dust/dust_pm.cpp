//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_pm.cpp
//! \brief Particle-mesh kernels for IMEX drag stages and for PC2.
//! The implicit chain is executed with a=a_impl*dt in each working IMEX stage:
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
//! neighbors, nine numbers side by side.  The linear cloud gather then reads one
//! short run of memory per leaf instead of fifteen values spread over nine rows of three
//! arrays; the values and the order of the operations are those of the unpacked gather.

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::PackGasGather
//! \brief The gas velocity of every cell (ghosts included) of a conserved state u into
//! uvel_pred, then PackGatherField: the gather field of the linear cloud gather from u (the
//! predictor's old state, or the resampling's current state, which reads it
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
  // room for the halo slots (their values come from HaloForwardPack)
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
  // the stiffness deposit qdep[4] feeds the stiffness monitor / dt limit / predictor damping
  const bool rate_dep = StiffnessActive();

  if (coupling == DustCoupling::pc2) {
    // PC2 needs only the fresh old-state feedback predictor during stage 1.
    if (stage != 1) return TaskStatus::complete;
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

    if (UsesRefinableClouds()) {
      // Production refinable-cloud predictor: the linear gather and the cloud-weight
      // deposit.
      // The gas velocity of every cell (ghosts included) is formed once; the gather then
      // reads velocities, as the advance does with u*, instead of dividing five momenta
      // by five densities in each of the cloud's cells.
      // (one kernel for both traversals, CloudPredictorT in dust_cloud3d.cpp)
      if (cloud_dim == 3) {CloudPredictorT(leaf_traversal3d, rate_dep); return TaskStatus::complete;}   // 3-D clouds
      CloudPredictorT(leaf_traversal, rate_dep);
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

  if (cloud_dim == 3) {
    CloudDepositImplicitT(leaf_traversal3d, pdrive, stage);
    return TaskStatus::complete;
  }
  if (UsesRefinableClouds()) {
    // (the kernel of both traversals, CloudDepositImplicitT in dust_imex3d.cpp)
    CloudDepositImplicitT(leaf_traversal, pdrive, stage);
    return TaskStatus::complete;
  }

  const Real sh = ImplicitShearShift(pdrive, stage);   // t^n gas frame
  par_for("dust_scatter",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp;
    Real wx[3], wy[3], wz[3];
    PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
              scheme, ip, wx);
    PMWeights(pr(IPY,p) + sh*pr(IPX,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
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
//! start-of-step gas at this point), reduced over all ranks (the particle history, phst, reads
//! its interval maximum).  With stiff_thresh <= 0 nothing reads the global value during the
//! cycle: the rank's records stay on the device (no host wait, no MPI collective) until
//! SyncStiffness.

void DustGasDrag::UpdateStiffness(const Real dt) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int ncells = pmy_pack->nmb_thispack*nx3*nx2*nx1;
  auto &u0 = pmy_pack->phydro->u0;
  auto &qdep_ = qdep;
  auto stiffness = KOKKOS_LAMBDA(const int idx, Real &maximum) {
    int q = idx;
    const int i = is + q % nx1; q /= nx1;
    const int j = js + q % nx2; q /= nx2;
    const int k = ks + q % nx3; q /= nx3;
    maximum = fmax(maximum, qdep_(q,4,k,j,i)/u0(q,IDN,k,j,i));
  };
  if (stiff_thresh <= 0.0) {
    // diagnostics only: the records stay on the device (dust.hpp, SyncStiffness)
    if (!stiff_diag_dev.is_allocated()) {
      stiff_rate_dev = Kokkos::View<Real, DevMemSpace>("dust_stiff_rate");
      stiff_diag_dev = DvceArray1D<Real>("dust_stiff_diag", 3);
      auto h = Kokkos::create_mirror_view(stiff_diag_dev);
      h(0) = stiff_chi_last; h(1) = stiff_chi_run_max; h(2) = phst_chi_max;
      Kokkos::deep_copy(stiff_diag_dev, h);
    }
    Kokkos::parallel_reduce("dust_stiffness", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
                            stiffness, Kokkos::Max<Real, DevMemSpace>(stiff_rate_dev));
    auto rate_d = stiff_rate_dev;
    auto diag = stiff_diag_dev;
    const bool phst = phst_track;
    Kokkos::parallel_for("dust_stiffness_record", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
    KOKKOS_LAMBDA(const int) {
      const Real chi = dt*rate_d();
      diag(0) = chi;
      diag(1) = (diag(1) < chi) ? chi : diag(1);              // std::max(run max, chi)
      if (phst) diag(2) = (diag(2) < chi) ? chi : diag(2);    // std::max(interval max, chi)
    });
    return;
  }
  Real rate = 0.0;
  Kokkos::parallel_reduce("dust_stiffness", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
                          stiffness, Kokkos::Max<Real>(rate));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &rate, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  stiff_rate_max = rate;
  stiff_chi_last = dt*rate;
  stiff_chi_run_max = std::max(stiff_chi_run_max, stiff_chi_last);
  if (phst_track) {   // the interval records of the particle history
    phst_chi_max = std::max(phst_chi_max, stiff_chi_last);
    if (stiff_thresh > 0.0 && rate > 0.0) {
      const Real d = stiff_thresh/rate;
      phst_dt_stiff_min = (phst_dt_stiff_min < 0.0) ? d : std::min(phst_dt_stiff_min, d);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SyncStiffness
//! \brief Collective: the device records of UpdateStiffness (stiff_thresh <= 0) reduced over the
//! ranks into stiff_chi_last, stiff_chi_run_max and phst_chi_max (no-op before the first record,
//! which every rank makes in the same cycle; with stiff_thresh > 0 the host values are current)

void DustGasDrag::SyncStiffness() {
  if (!stiff_diag_dev.is_allocated()) return;
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), stiff_diag_dev);
  Real v[3] = {h(0), h(1), h(2)};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, v, 3, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  stiff_chi_last = v[0];
  stiff_chi_run_max = v[1];
  phst_chi_max = v[2];
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
  if (coupling == DustCoupling::pc2) {
    if (stage == 1) {
      if (back_reaction) {
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
    }
    return TaskStatus::complete;
  }
  // (hidden test_qrho_diag: the coupled dust-to-gas ratio of this stage, read-only)
  if (qrho_diag_ > 0 && back_reaction && (pmy_pack->pmesh->ncycle % qrho_diag_) == 0) {
    QrhoDiag(pdrive, stage);
  }
  // (refinable clouds, 2-D and 3-D: the lumped solve below and their own dc1 / dc2 correction)
  if (back_reaction && drag_solver != DustDragSolver::local && !UsesRefinableClouds()) {
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
  // refinable clouds (2-D and 3-D), drag_solver = dc1 / dc2: the defect correction of u* (every
  // rank, every working stage)
  if (cloud_dc1) CloudDefectCorrection(pdrive, stage);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::GatherKickPMBR
//! \brief Fused gather + implicit particle kick + drag-rate record + PMBR scatter.
//! Weights are computed once and used for both the gather and the back-reaction deposit,
//! which makes the momentum exchange antisymmetric to round-off.

TaskStatus DustGasDrag::GatherKickPMBR(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  if (coupling == DustCoupling::pc2) return TaskStatus::complete;   // (the kick is in ExplicitPush)

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
  auto &ustar_ = ustar;
  auto &dmom_ = dmom;

  if (cloud_dim == 3) {
    if (cloud_dc1_opcheck) CloudDcOpCheckBefore(pdrive, stage);
    CloudGatherImplicit3D(pdrive, stage);
    if (cloud_dc1_opcheck) CloudDcOpCheckAfterKick(pdrive, stage);
    return TaskStatus::complete;
  }
  if (UsesRefinableClouds()) {
    // Refinable clouds: the gather reads the packed u* field (velocity and its central
    // differences, with the owners' values on halo blocks).
    // The gather is the linear reconstruction of u* over the cloud, with the lobed weights and
    // point values of transition mode 2, as the PC2 midpoint gather does with its midpoint
    // velocity.  The impulse is deposited with the cloud's own weights.
    // (test_cloud_dc1_opcheck, 2-D clouds: the operator on u* before the kick; it packs its own
    // fields, so it comes before the kick's pack)
    if (cloud_dc1_opcheck) CloudDcOpCheckBefore(pdrive, stage);
    PackGatherField(ustar);
    // (the kernel of both traversals, CloudGatherKickT in dust_imex3d.cpp; it sums
    // dmom to the owners)
    CloudGatherKickT(leaf_traversal, gpack, a_dt, 0.0, "dust_cloud_gather_kick");
    if (cloud_dc1_opcheck) CloudDcOpCheckAfterKick(pdrive, stage);
    return TaskStatus::complete;
  }

  const Real sh = ImplicitShearShift(pdrive, stage);   // gas in its t^n frame
  if (point_dc_opcheck) CloudDcOpCheckBefore(pdrive, stage);   // (test_point_dc_opcheck)
  par_for("dust_gather",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp;
    Real wx[3], wy[3], wz[3];
    PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
              scheme, ip, wx);
    PMWeights(pr(IPY,p) + sh*pr(IPX,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
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
  if (point_dc_opcheck) CloudDcOpCheckAfterKick(pdrive, stage);

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ApplyPMBR
//! \brief Adds the (halo-exchanged) back-reaction momentum deposit to the gas conserved
//! variables over active cells, then rescales dmom in place into the gas drag-rate field
//! R_g = dmom/(a_impl*dt) consumed by AddDragHistoryGas at the next stage.

TaskStatus DustGasDrag::ApplyPMBR(Driver *pdrive, int stage) {
  if (prescribed_gas && coupling == DustCoupling::pc2 && stage == 2 &&
      ActiveStage(pdrive, stage)) {
    // diagnostic prescribed gas (back_reaction=false): reset the gas to the prescribed
    // end-of-step state before the boundary tail
    prescribed_gas(pmy_pack->pmesh->time + pmy_pack->pmesh->dt, pmy_pack->phydro->u0);
  }
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}

  if (coupling == DustCoupling::pc2) {
    if (stage != 2) return TaskStatus::complete;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, ie = indcs.ie;
    int js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int nmb1 = pmy_pack->nmb_thispack - 1;
    auto &u0 = pmy_pack->phydro->u0;
    auto &dmom_ = dmom;
    auto &qdep_ = qdep;
    Real wpred = pdrive->gam1[pdrive->nexp_stages-1];
    par_for("dust_pc2_commit",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      u0(m,IM1,k,j,i) += dmom_(m,0,k,j,i) - wpred*qdep_(m,1,k,j,i);
      u0(m,IM2,k,j,i) += dmom_(m,1,k,j,i) - wpred*qdep_(m,2,k,j,i);
      u0(m,IM3,k,j,i) += dmom_(m,2,k,j,i) - wpred*qdep_(m,3,k,j,i);
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
  if (cloud_dc1_opcheck || point_dc_opcheck) CloudDcOpCheckDmom(pdrive, stage);   // (dc1 checks)
  if (cloud_dim == 3 && imex_consistent_gas) {   // consistent gas / dust split of the residual
    CloudImplicitConsistency3D(pdrive, stage);
    return TaskStatus::complete;
  }
  // 2-D clouds (IMEX working stages; PC2 returned above): the same split with the
  // overlap weights (dust_imex3d.cpp)
  if (cloud_dim != 3 && UsesRefinableClouds() && imex_consistent_gas) {
    CloudImplicitConsistencyOverlapT(leaf_traversal, pdrive, stage);   // (overlap weights)
    return TaskStatus::complete;
  }
  if (cloud_dim != 3 && !UsesRefinableClouds() && imex_consistent_gas) {
    // Point particles (canonical TSC): as for the clouds, the gas takes the lumped
    // solve's change G_c = rho (u*_c - u_c) plus its mass share rho/(rho + Q_c) of the residual
    // D_c - G_c; the dust share r_c Q_c V of a cell goes to the particles whose TSC center cell it
    // is, in proportion to m_p c_p (the center cell is an active cell of the particle's own block:
    // no ghost values are needed).  A cell that is no particle's center gives the whole residual
    // to the gas (as before).  Total momentum is exact.
    particles::Particles *ppar = pmy_pack->ppart;
    auto &pr = ppar->prtcl_rdata;
    auto &pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    auto &mbsize = pmy_pack->pmb->mb_size;
    const auto gids = pmy_pack->gids;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const bool three_d = pmy_pack->pmesh->three_d;
    const int scheme = static_cast<int>(deposit);
    const Real a_dt = DragStep(pdrive);
    if (imex_resid.extent(0) != ustar.extent(0) || imex_resid.extent(1) != 4 ||
        imex_resid.extent(2) != ustar.extent(2) || imex_resid.extent(3) != ustar.extent(3) ||
        imex_resid.extent(4) != ustar.extent(4)) {
      Kokkos::realloc(imex_resid, ustar.extent(0), 4, ustar.extent(2), ustar.extent(3),
                      ustar.extent(4));
    }
    Kokkos::deep_copy(DevExeSpace(), imex_resid, 0.0);
    auto res = imex_resid;   // 0-2: the dust correction per unit coupled mass, 3: center-cell m c/V
    auto center = KOKKOS_LAMBDA(const int p, int &m, int &ip, int &jp, int &kp) {
      m = pi(PGID,p) - gids;
      Real w[3];
      PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is, scheme, ip, w);
      PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js, scheme, jp, w);
      if (three_d) {
        PMWeights(pr(IPZ,p), mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks, scheme, kp, w);
      } else {
        kp = ks;
      }
    };
    par_for("dust_imex_point_center_mass", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
      if (pr(IPM,p) <= 0.0) return;
      int m, ip, jp, kp;
      center(p, m, ip, jp, kp);
      Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
      if (three_d) vol *= mbsize.d_view(m).dx3;
      Kokkos::atomic_add(&res(m,3,kp,jp,ip), pr(IPM,p)*(a_dt/(pr(IPTS,p) + a_dt))/vol);
    });
    auto us = ustar;
    auto qp = qdep;
    par_for("dust_imex_point_consistent_gas",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real rho = u0(m,IDN,k,j,i), qt = res(m,3,k,j,i);
      const Real q = qp(m,0,k,j,i);
      for (int c=0; c<3; ++c) {
        Real dgas = dmom_(m,c,k,j,i);
        if (qt > 0.0) {
          const Real g = rho*us(m,c,k,j,i) - u0(m,IM1+c,k,j,i);
          const Real r = (dmom_(m,c,k,j,i) - g)/(rho + q);
          dgas = g + rho*r;
          res(m,c,k,j,i) = r*q/qt;
        }
        u0(m,IM1+c,k,j,i) += dgas;
        dmom_(m,c,k,j,i) = dgas*inv_adt;
      }
    });
    par_for("dust_imex_point_consistent_dust", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
      if (pr(IPM,p) <= 0.0) return;
      int m, ip, jp, kp;
      center(p, m, ip, jp, kp);
      const Real cj = a_dt/(pr(IPTS,p) + a_dt);
      for (int c=0; c<3; ++c) {
        const Real dv = cj*res(m,c,kp,jp,ip);
        pr(IPVX+2*c,p) += dv;
        pr(IPRX+c,p) += dv/a_dt;
      }
    });
    return TaskStatus::complete;
  }

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
