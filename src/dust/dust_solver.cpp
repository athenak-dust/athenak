//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_solver.cpp
//! \brief Matrix-free coupled particle-mesh drag solvers on a periodic uniform mesh.
//!
//! For each velocity component the backward-Euler stage Schur complement is
//!
//!   A = diag(rho_g) + G^T diag(m_p*c_p/V) G,
//!   b = momentum_g + G^T diag(m_p*c_p/V) v_p,
//!
//! where c_p=a/(t_stop,p+a).  Matching PMCompactWeights in G and G^T makes A SPD.  All
//! trial operations below are non-mutating: particle velocities and gas momentum are
//! changed only later by GatherKickPMBR and ApplyPMBR after one field has been accepted.
//!
//! The communication-bearing solves below deliberately block inside one task.  This is
//! safe under AthenaK's current execution contract: one MeshBlockPack contains all local
//! MeshBlocks on a rank, IMEX stages invoke this task sequentially, and every numerical
//! continue/accept decision is computed from globally reduced scalars.  A future design
//! with multiple packs, overlapping stages, or concurrent collective streams must move
//! solver progress to a stage-wide controller rather than call these loops concurrently.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "driver/driver.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust {
namespace {

[[noreturn]] void SolverFatal(const std::string &message) {
  std::cerr << "### FATAL ERROR in dust coupled solver on rank "
            << global_variable::my_rank << std::endl << message << std::endl;
#if MPI_PARALLEL_ENABLED
  // This routine is called while the solver may have outstanding point-to-point
  // operations or may be between collectives.  Terminate the whole MPI job rather than
  // leave peer ranks spinning in a boundary exchange or blocked in the next collective.
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

void RequireComplete(TaskStatus status, const char *operation) {
  if (status == TaskStatus::fail) {
    SolverFatal(std::string("Boundary operation failed: ") + operation);
  }
}

// Scheme- and dimension-specialized particle contribution to the matrix-free operator.
// Compact supports are essential for NGP/CIC: the former runtime-width loop issued
// reads and atomics for zero-weight cells.
template<DustDeposit Scheme, bool ThreeD>
void ApplyCoupledParticleTerm(DvceArray2D<Real> pr, DvceArray2D<int> pi,
                              DvceArray1D<RegionSize> mbsize,
                              DvceArray5D<Real> field, DvceArray5D<Real> result,
                              const int npart, const int gids,
                              const int is, const int js, const int ks,
                              const int nx1, const int nx2, const int nx3,
                              const Real a_dt) {
  constexpr int nstencil = PMStencilWidth<Scheme>::value;
  constexpr int nk = ThreeD ? nstencil : 1;
  const char *label = (Scheme == DustDeposit::ngp) ? "dust_applya_gtsg_ngp" :
                      (Scheme == DustDeposit::cic) ? "dust_applya_gtsg_cic" :
                                                    "dust_applya_gtsg_tsc";

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

    Real gathered[3] = {0.0, 0.0, 0.0};
    for (int c=0; c<nk; ++c) {
      for (int b=0; b<nstencil; ++b) {
        Real wcb = wz[c]*wy[b];
        if (wcb == 0.0) {continue;}
        for (int a=0; a<nstencil; ++a) {
          Real w = wcb*wx[a];
          if (w == 0.0) {continue;}
          int kk = k0+c, jj = j0+b, ii = i0+a;
          gathered[0] += w*field(m,0,kk,jj,ii);
          gathered[1] += w*field(m,1,kk,jj,ii);
          gathered[2] += w*field(m,2,kk,jj,ii);
        }
      }
    }

    Real vol = mbsize(m).dx1*mbsize(m).dx2;
    if constexpr (ThreeD) {vol *= mbsize(m).dx3;}
    Real cj = a_dt/(pr(IPTS,p) + a_dt);
    Real fac = pr(IPM,p)*cj/vol;
    for (int c=0; c<nk; ++c) {
      for (int b=0; b<nstencil; ++b) {
        Real wcb = wz[c]*wy[b]*fac;
        if (wcb == 0.0) {continue;}
        for (int a=0; a<nstencil; ++a) {
          Real w = wcb*wx[a];
          if (w == 0.0) {continue;}
          int kk = k0+c, jj = j0+b, ii = i0+a;
          Kokkos::atomic_add(&result(m,0,kk,jj,ii), w*gathered[0]);
          Kokkos::atomic_add(&result(m,1,kk,jj,ii), w*gathered[1]);
          Kokkos::atomic_add(&result(m,2,kk,jj,ii), w*gathered[2]);
        }
      }
    }
  });
}

template<DustDeposit Scheme, bool ThreeD>
Real AdaptiveParticleState(DvceArray2D<Real> pr, DvceArray2D<int> pi,
                           DvceArray1D<RegionSize> mbsize,
                           DvceArray5D<Real> field, const int npart,
                           const int gids, const int is, const int js, const int ks,
                           const int nx1, const int nx2, const int nx3,
                           const Real a_dt) {
  constexpr int nstencil = PMStencilWidth<Scheme>::value;
  constexpr int nk = ThreeD ? nstencil : 1;
  const char *label = (Scheme == DustDeposit::ngp) ? "dust_adapt_particle_state_ngp" :
                      (Scheme == DustDeposit::cic) ? "dust_adapt_particle_state_cic" :
                                                    "dust_adapt_particle_state_tsc";

  Real state = 0.0;
  Kokkos::parallel_reduce(label,Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int p, Real &sum) {
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

    Real gu[3] = {0.0, 0.0, 0.0};
    for (int c=0; c<nk; ++c) {
      for (int b=0; b<nstencil; ++b) {
        Real wcb = wz[c]*wy[b];
        if (wcb == 0.0) {continue;}
        for (int a=0; a<nstencil; ++a) {
          Real w = wcb*wx[a];
          if (w == 0.0) {continue;}
          int kk = k0+c, jj = j0+b, ii = i0+a;
          gu[0] += w*field(m,0,kk,jj,ii);
          gu[1] += w*field(m,1,kk,jj,ii);
          gu[2] += w*field(m,2,kk,jj,ii);
        }
      }
    }

    Real cj = a_dt/(pr(IPTS,p)+a_dt);
    Real vx = pr(IPVX,p)+cj*(gu[0]-pr(IPVX,p));
    Real vy = pr(IPVY,p)+cj*(gu[1]-pr(IPVY,p));
    Real vz = pr(IPVZ,p)+cj*(gu[2]-pr(IPVZ,p));
    sum += pr(IPM,p)*(vx*vx+vy*vy+vz*vz);
  },state);
  return state;
}

} // namespace

//----------------------------------------------------------------------------------------
//! Complete one reusable copy exchange.  This boundary object is distinct from the
//! one-shot ustar exchange posted by InitRecvDep at the beginning of the IMEX stage.

void DustGasDrag::CompleteCopyExchange(DvceArray5D<Real> &field) {
  RequireComplete(pbval_solver_copy->InitRecv(3), "copy InitRecv");
  RequireComplete(pbval_solver_copy->PackAndSendCC(field, cdummy), "copy send");
  TaskStatus status;
  do {
    status = pbval_solver_copy->RecvAndUnpackCC(field, cdummy);
    RequireComplete(status, "copy receive");
  } while (status == TaskStatus::incomplete);
  RequireComplete(pbval_solver_copy->ClearSend(), "copy ClearSend");
  RequireComplete(pbval_solver_copy->ClearRecv(), "copy ClearRecv");
  ++solver_halo_count;
}

//----------------------------------------------------------------------------------------
//! Complete one reusable additive exchange of a particle-deposited field.

void DustGasDrag::CompleteAddExchange(DvceArray5D<Real> &field) {
  RequireComplete(pbval_solver_add->InitRecv(3), "add InitRecv");
  RequireComplete(pbval_solver_add->PackAndSendDeposit(field), "add send");
  TaskStatus status;
  do {
    status = pbval_solver_add->RecvAndSumDeposit(field);
    RequireComplete(status, "add receive");
  } while (status == TaskStatus::incomplete);
  RequireComplete(pbval_solver_add->ClearSend(), "add ClearSend");
  RequireComplete(pbval_solver_add->ClearRecv(), "add ClearRecv");
  ++solver_halo_count;
}

//----------------------------------------------------------------------------------------
//! Apply A to all three velocity components using one fused particle gather/scatter.

void DustGasDrag::ApplyCoupledOperator(DvceArray5D<Real> &field,
                                       DvceArray5D<Real> &result, Real a_dt) {
  CompleteCopyExchange(field);

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  int ncells1 = result.extent_int(4);
  int ncells2 = result.extent_int(3);
  int ncells3 = result.extent_int(2);
  auto &u0 = pmy_pack->phydro->u0;
  auto &field_ = field;
  auto &result_ = result;
  // Initialize the particle-scatter target and gas diagonal in one pass.  Ghost cells
  // must start at zero because the additive exchange transfers only particle deposits.
  par_for("dust_applya_init",DevExeSpace(),0,nmb1,0,(ncells3-1),0,(ncells2-1),
          0,(ncells1-1),
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    bool active = (i >= is && i <= ie && j >= js && j <= je &&
                   k >= ks && k <= ke);
    if (active) {
      Real rho = u0(m,IDN,k,j,i);
      result_(m,0,k,j,i) = rho*field_(m,0,k,j,i);
      result_(m,1,k,j,i) = rho*field_(m,1,k,j,i);
      result_(m,2,k,j,i) = rho*field_(m,2,k,j,i);
    } else {
      result_(m,0,k,j,i) = 0.0;
      result_(m,1,k,j,i) = 0.0;
      result_(m,2,k,j,i) = 0.0;
    }
  });

  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  bool three_d = pmy_pack->pmesh->three_d;
  auto mbsize = pmy_pack->pmb->mb_size.d_view;
  auto gids = pmy_pack->gids;

  if (three_d) {
    if (deposit == DustDeposit::ngp) {
      ApplyCoupledParticleTerm<DustDeposit::ngp,true>(
          pr, pi, mbsize, field, result, npart, gids,
          is, js, ks, nx1, nx2, nx3, a_dt);
    } else if (deposit == DustDeposit::cic) {
      ApplyCoupledParticleTerm<DustDeposit::cic,true>(
          pr, pi, mbsize, field, result, npart, gids,
          is, js, ks, nx1, nx2, nx3, a_dt);
    } else {
      ApplyCoupledParticleTerm<DustDeposit::tsc,true>(
          pr, pi, mbsize, field, result, npart, gids,
          is, js, ks, nx1, nx2, nx3, a_dt);
    }
  } else {
    if (deposit == DustDeposit::ngp) {
      ApplyCoupledParticleTerm<DustDeposit::ngp,false>(
          pr, pi, mbsize, field, result, npart, gids,
          is, js, ks, nx1, nx2, nx3, a_dt);
    } else if (deposit == DustDeposit::cic) {
      ApplyCoupledParticleTerm<DustDeposit::cic,false>(
          pr, pi, mbsize, field, result, npart, gids,
          is, js, ks, nx1, nx2, nx3, a_dt);
    } else {
      ApplyCoupledParticleTerm<DustDeposit::tsc,false>(
          pr, pi, mbsize, field, result, npart, gids,
          is, js, ks, nx1, nx2, nx3, a_dt);
    }
  }

  CompleteAddExchange(result);
  ++solver_applya_count;
}

//----------------------------------------------------------------------------------------
//! Volume-weighted global dot products, one MPI collective for three components.

void DustGasDrag::GlobalDot(DvceArray5D<Real> &left, DvceArray5D<Real> &right,
                            Real value[3]) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int ncells = nmb*nx3*nx2*nx1;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto left_ = left;
  auto right_ = right;
  Real local[3] = {0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("dust_solver_dot",Kokkos::RangePolicy<>(DevExeSpace(),0,ncells),
  KOKKOS_LAMBDA(const int idx, Real &sum0, Real &sum1, Real &sum2) {
    int q = idx;
    int i = q % nx1; q /= nx1;
    int j = q % nx2; q /= nx2;
    int k = q % nx3; q /= nx3;
    int m = q;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) vol *= mbsize.d_view(m).dx3;
    sum0 += vol*left_(m,0,k+ks,j+js,i+is)*right_(m,0,k+ks,j+js,i+is);
    sum1 += vol*left_(m,1,k+ks,j+js,i+is)*right_(m,1,k+ks,j+js,i+is);
    sum2 += vol*left_(m,2,k+ks,j+js,i+is)*right_(m,2,k+ks,j+js,i+is);
  }, Kokkos::Sum<Real>(local[0]), Kokkos::Sum<Real>(local[1]),
     Kokkos::Sum<Real>(local[2]));
#if MPI_PARALLEL_ENABLED
  int ierr = MPI_Allreduce(local, value, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) SolverFatal("MPI_Allreduce failed in GlobalDot");
#else
  for (int d=0; d<3; ++d) value[d] = local[d];
#endif
  ++solver_reduction_count;
}

//----------------------------------------------------------------------------------------
//! Global squared P^{-1} norm with P=diag(rho+Q), including cell volume.

void DustGasDrag::GlobalPreconditionedNorm(DvceArray5D<Real> &residual, Real value[3]) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int ncells = nmb*nx3*nx2*nx1;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &u0 = pmy_pack->phydro->u0;
  auto &qdep_ = qdep;
  auto residual_ = residual;
  Real local[3] = {0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("dust_solver_pnorm",Kokkos::RangePolicy<>(DevExeSpace(),0,ncells),
  KOKKOS_LAMBDA(const int idx, Real &sum0, Real &sum1, Real &sum2) {
    int q = idx;
    int i = q % nx1; q /= nx1;
    int j = q % nx2; q /= nx2;
    int k = q % nx3; q /= nx3;
    int m = q;
    int ii = i+is, jj = j+js, kk = k+ks;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) vol *= mbsize.d_view(m).dx3;
    Real invp = vol/(u0(m,IDN,kk,jj,ii) + qdep_(m,0,kk,jj,ii));
    Real rv0 = residual_(m,0,kk,jj,ii);
    Real rv1 = residual_(m,1,kk,jj,ii);
    Real rv2 = residual_(m,2,kk,jj,ii);
    sum0 += invp*rv0*rv0;
    sum1 += invp*rv1*rv1;
    sum2 += invp*rv2*rv2;
  }, Kokkos::Sum<Real>(local[0]), Kokkos::Sum<Real>(local[1]),
     Kokkos::Sum<Real>(local[2]));
#if MPI_PARALLEL_ENABLED
  int ierr = MPI_Allreduce(local, value, 3, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) SolverFatal("MPI_Allreduce failed in residual norm");
#else
  for (int d=0; d<3; ++d) value[d] = local[d];
#endif
  ++solver_reduction_count;
}

//----------------------------------------------------------------------------------------
//! Strict row-sum-preconditioned CG.  The three scalar recurrences share matvec passes,
//! messages, and collectives but have independent alpha/beta/convergence decisions.

int DustGasDrag::StrictPCG(Real a_dt, bool residual_is_current) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0 = pmy_pack->phydro->u0;
  auto &qdep_ = qdep;
  auto &x = ustar;
  auto &r = solver_r;
  auto &p = solver_p;
  auto &ap = solver_ap;

  if (!residual_is_current) {
    ApplyCoupledOperator(x, ap, a_dt);
    par_for("dust_pcg_initial",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real pinv = 1.0/(u0(m,IDN,k,j,i) + qdep_(m,0,k,j,i));
      for (int d=0; d<3; ++d) {
        r(m,d,k,j,i) = u0(m,IM1+d,k,j,i) + qdep_(m,1+d,k,j,i) - ap(m,d,k,j,i);
        p(m,d,k,j,i) = pinv*r(m,d,k,j,i);
      }
    });
  } else {
    par_for("dust_pcg_restart",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real pinv = 1.0/(u0(m,IDN,k,j,i) + qdep_(m,0,k,j,i));
      p(m,0,k,j,i) = pinv*r(m,0,k,j,i);
      p(m,1,k,j,i) = pinv*r(m,1,k,j,i);
      p(m,2,k,j,i) = pinv*r(m,2,k,j,i);
    });
  }

  Real rz[3];
  GlobalDot(r, p, rz);

  // Compute ||b||_{P^-1} without allocating another mesh field.
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmy_pack->nmb_thispack;
  int ncells = nmb*nx3*nx2*nx1;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  Real bnorm_local[3] = {0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("dust_pcg_bnorm",Kokkos::RangePolicy<>(DevExeSpace(),0,ncells),
  KOKKOS_LAMBDA(const int idx, Real &sum0, Real &sum1, Real &sum2) {
    int q = idx;
    int i = q % nx1; q /= nx1;
    int j = q % nx2; q /= nx2;
    int k = q % nx3; q /= nx3;
    int m = q;
    int ii=i+is, jj=j+js, kk=k+ks;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) vol *= mbsize.d_view(m).dx3;
    Real invp = vol/(u0(m,IDN,kk,jj,ii) + qdep_(m,0,kk,jj,ii));
    Real bv0 = u0(m,IM1,kk,jj,ii) + qdep_(m,1,kk,jj,ii);
    Real bv1 = u0(m,IM2,kk,jj,ii) + qdep_(m,2,kk,jj,ii);
    Real bv2 = u0(m,IM3,kk,jj,ii) + qdep_(m,3,kk,jj,ii);
    sum0 += invp*bv0*bv0;
    sum1 += invp*bv1*bv1;
    sum2 += invp*bv2*bv2;
  }, Kokkos::Sum<Real>(bnorm_local[0]), Kokkos::Sum<Real>(bnorm_local[1]),
     Kokkos::Sum<Real>(bnorm_local[2]));
  Real bnorm2[3];
#if MPI_PARALLEL_ENABLED
  int ierr = MPI_Allreduce(bnorm_local, bnorm2, 3, MPI_ATHENA_REAL, MPI_SUM,
                           MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) SolverFatal("MPI_Allreduce failed for PCG RHS norm");
#else
  for (int d=0; d<3; ++d) bnorm2[d] = bnorm_local[d];
#endif
  ++solver_reduction_count;

  Real target[3];
  bool converged[3];
  for (int d=0; d<3; ++d) {
    target[d] = drag_atol + drag_rtol*std::sqrt(std::max(bnorm2[d], 0.0));
    converged[d] = std::sqrt(std::max(rz[d], 0.0)) <= target[d];
  }

  // `iteration` is the number of completed PCG updates.  Including drag_iter_max here
  // gives an iterate that converged on the final allowed update one verification-only
  // pass; it does not permit an additional PCG update beyond the configured budget.
  for (int iteration=0; iteration<=drag_iter_max; ++iteration) {
    if (converged[0] && converged[1] && converged[2]) {
      ApplyCoupledOperator(x, ap, a_dt);
      par_for("dust_pcg_true_r",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        for (int d=0; d<3; ++d) {
          r(m,d,k,j,i) = u0(m,IM1+d,k,j,i) + qdep_(m,1+d,k,j,i) - ap(m,d,k,j,i);
        }
      });
      Real true_norm2[3];
      GlobalPreconditionedNorm(r, true_norm2);
      bool true_converged = true;
      solver_last_residual = 0.0;
      for (int d=0; d<3; ++d) {
        Real norm = std::sqrt(std::max(true_norm2[d], 0.0));
        solver_last_residual = std::max(solver_last_residual, norm);
        converged[d] = norm <= target[d];
        true_converged = true_converged && converged[d];
        rz[d] = true_norm2[d];
      }
      if (true_converged) return iteration;

      // Recursive convergence was optimistic: restart from the recomputed true residual.
      par_for("dust_pcg_true_restart",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real pinv = 1.0/(u0(m,IDN,k,j,i) + qdep_(m,0,k,j,i));
        for (int d=0; d<3; ++d) p(m,d,k,j,i) = pinv*r(m,d,k,j,i);
      });
    }

    if (iteration == drag_iter_max) break;
    int update_iteration = iteration + 1;
    ApplyCoupledOperator(p, ap, a_dt);
    Real pap[3];
    GlobalDot(p, ap, pap);
    for (int d=0; d<3; ++d) {
      if (!converged[d] && (!(pap[d] > 0.0) || !std::isfinite(pap[d]))) {
        SolverFatal("PCG lost positive definiteness at iteration " +
                    std::to_string(update_iteration));
      }
    }
    Real alpha0 = converged[0] ? 0.0 : rz[0]/pap[0];
    Real alpha1 = converged[1] ? 0.0 : rz[1]/pap[1];
    Real alpha2 = converged[2] ? 0.0 : rz[2]/pap[2];
    par_for("dust_pcg_xr",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      x(m,0,k,j,i) += alpha0*p(m,0,k,j,i);
      x(m,1,k,j,i) += alpha1*p(m,1,k,j,i);
      x(m,2,k,j,i) += alpha2*p(m,2,k,j,i);
      r(m,0,k,j,i) -= alpha0*ap(m,0,k,j,i);
      r(m,1,k,j,i) -= alpha1*ap(m,1,k,j,i);
      r(m,2,k,j,i) -= alpha2*ap(m,2,k,j,i);
    });

    Real rz_new[3];
    GlobalPreconditionedNorm(r, rz_new);
    bool now_converged[3];
    for (int d=0; d<3; ++d) {
      now_converged[d] = std::sqrt(std::max(rz_new[d], 0.0)) <= target[d];
    }
    Real beta0 = (converged[0] || now_converged[0]) ? 0.0 : rz_new[0]/rz[0];
    Real beta1 = (converged[1] || now_converged[1]) ? 0.0 : rz_new[1]/rz[1];
    Real beta2 = (converged[2] || now_converged[2]) ? 0.0 : rz_new[2]/rz[2];
    par_for("dust_pcg_p",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      Real pinv = 1.0/(u0(m,IDN,k,j,i) + qdep_(m,0,k,j,i));
      p(m,0,k,j,i) = now_converged[0] ? 0.0 : pinv*r(m,0,k,j,i) + beta0*p(m,0,k,j,i);
      p(m,1,k,j,i) = now_converged[1] ? 0.0 : pinv*r(m,1,k,j,i) + beta1*p(m,1,k,j,i);
      p(m,2,k,j,i) = now_converged[2] ? 0.0 : pinv*r(m,2,k,j,i) + beta2*p(m,2,k,j,i);
    });
    for (int d=0; d<3; ++d) {
      converged[d] = now_converged[d];
      rz[d] = rz_new[d];
    }
  }

  SolverFatal("PCG failed to reach the true-residual tolerance in " +
              std::to_string(drag_iter_max) + " iterations");
}

//----------------------------------------------------------------------------------------
//! Conservative adaptive candidate bound and its mass-weighted state scale.

Real DustGasDrag::AdaptiveErrorBound(Real a_dt, Real residual_norm, Real &state_scale,
                                     Real &acceptance_target) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is=indcs.is, js=indcs.js, ks=indcs.ks;
  int nx1=indcs.nx1, nx2=indcs.nx2, nx3=indcs.nx3;
  int nmb=pmy_pack->nmb_thispack;
  int ncells=nmb*nx3*nx2*nx1;
  bool three_d=pmy_pack->pmesh->three_d;
  auto &mbsize=pmy_pack->pmb->mb_size;
  auto &u0=pmy_pack->phydro->u0;
  auto &qdep_=qdep;
  auto &x=ustar;

  Real eps_local=0.0;
  Kokkos::parallel_reduce("dust_adapt_eps",Kokkos::RangePolicy<>(DevExeSpace(),0,ncells),
  KOKKOS_LAMBDA(const int idx, Real &maximum) {
    int q=idx;
    int i=q%nx1; q/=nx1;
    int j=q%nx2; q/=nx2;
    int k=q%nx3; q/=nx3;
    int m=q;
    int ii=i+is, jj=j+js, kk=k+ks;
    maximum=fmax(maximum,qdep_(m,0,kk,jj,ii)/u0(m,IDN,kk,jj,ii));
  },Kokkos::Max<Real>(eps_local));

  particles::Particles *ppar=pmy_pack->ppart;
  auto &pr=ppar->prtcl_rdata;
  auto &pi=ppar->prtcl_idata;
  int npart=ppar->nprtcl_thispack;
  Real cmax_local=0.0;
  Kokkos::parallel_reduce("dust_adapt_cmax",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int p, Real &maximum) {
    maximum=fmax(maximum,a_dt/(pr(IPTS,p)+a_dt));
  },Kokkos::Max<Real>(cmax_local));
  Real maxima_local[2]={eps_local,cmax_local}, maxima[2];
#if MPI_PARALLEL_ENABLED
  int ierr=MPI_Allreduce(maxima_local,maxima,2,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) SolverFatal("MPI_Allreduce failed for adaptive maxima");
#else
  maxima[0]=maxima_local[0]; maxima[1]=maxima_local[1];
#endif
  ++solver_reduction_count;
  solver_last_epsmax=maxima[0];

  Real gas_state_local=0.0;
  Kokkos::parallel_reduce("dust_adapt_gas_state",
  Kokkos::RangePolicy<>(DevExeSpace(),0,ncells),KOKKOS_LAMBDA(const int idx,Real &sum) {
    int q=idx;
    int i=q%nx1; q/=nx1;
    int j=q%nx2; q/=nx2;
    int k=q%nx3; q/=nx3;
    int m=q;
    int ii=i+is, jj=j+js, kk=k+ks;
    Real vol=mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) vol*=mbsize.d_view(m).dx3;
    Real usq=SQR(x(m,0,kk,jj,ii))+SQR(x(m,1,kk,jj,ii))+SQR(x(m,2,kk,jj,ii));
    sum+=vol*u0(m,IDN,kk,jj,ii)*usq;
  },gas_state_local);

  auto gids=pmy_pack->gids;
  Real particle_state_local=0.0;
  auto mbsize_d=mbsize.d_view;
  if (three_d) {
    if (deposit == DustDeposit::ngp) {
      particle_state_local=AdaptiveParticleState<DustDeposit::ngp,true>(
          pr,pi,mbsize_d,x,npart,gids,is,js,ks,nx1,nx2,nx3,a_dt);
    } else if (deposit == DustDeposit::cic) {
      particle_state_local=AdaptiveParticleState<DustDeposit::cic,true>(
          pr,pi,mbsize_d,x,npart,gids,is,js,ks,nx1,nx2,nx3,a_dt);
    } else {
      particle_state_local=AdaptiveParticleState<DustDeposit::tsc,true>(
          pr,pi,mbsize_d,x,npart,gids,is,js,ks,nx1,nx2,nx3,a_dt);
    }
  } else {
    if (deposit == DustDeposit::ngp) {
      particle_state_local=AdaptiveParticleState<DustDeposit::ngp,false>(
          pr,pi,mbsize_d,x,npart,gids,is,js,ks,nx1,nx2,nx3,a_dt);
    } else if (deposit == DustDeposit::cic) {
      particle_state_local=AdaptiveParticleState<DustDeposit::cic,false>(
          pr,pi,mbsize_d,x,npart,gids,is,js,ks,nx1,nx2,nx3,a_dt);
    } else {
      particle_state_local=AdaptiveParticleState<DustDeposit::tsc,false>(
          pr,pi,mbsize_d,x,npart,gids,is,js,ks,nx1,nx2,nx3,a_dt);
    }
  }

  Real state_local=gas_state_local+particle_state_local,state_global;
#if MPI_PARALLEL_ENABLED
  ierr=MPI_Allreduce(&state_local,&state_global,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) SolverFatal("MPI_Allreduce failed for adaptive state scale");
#else
  state_global=state_local;
#endif
  ++solver_reduction_count;
  state_scale=std::sqrt(std::max(state_global,0.0));

  Real epsilon=maxima[0];
  Real theta=epsilon/(1.0+epsilon);
  Real response=(1.0+epsilon)*std::sqrt((1.0+epsilon)*theta*theta+maxima[1]*theta);
  Real outer_dt=pmy_pack->pmesh->dt;
  Real rtol_step=std::min(adaptive_rtol_max,
                          adaptive_order_c*std::pow(outer_dt/adaptive_tref,3));
  acceptance_target=drag_atol+rtol_step*std::max(state_scale,adaptive_state_floor);
  return response*residual_norm;
}

//----------------------------------------------------------------------------------------
//! Build the local guess and execute the selected correction/strict/adaptive path.

TaskStatus DustGasDrag::SolveCoupledStage(Driver *pdrive, int stage) {
  auto started=std::chrono::steady_clock::now();
  Real a_dt=(pdrive->a_impl)*(pmy_pack->pmesh->dt);
  auto &indcs=pmy_pack->pmesh->mb_indcs;
  int is=indcs.is,ie=indcs.ie,js=indcs.js,je=indcs.je,ks=indcs.ks,ke=indcs.ke;
  int nmb1=pmy_pack->nmb_thispack-1;
  auto &u0=pmy_pack->phydro->u0;
  auto &qdep_=qdep;
  auto &x=ustar;
  auto &r=solver_r;
  auto &ap=solver_ap;

  par_for("dust_solver_local_guess",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(const int m,const int k,const int j,const int i) {
    Real pinv=1.0/(u0(m,IDN,k,j,i)+qdep_(m,0,k,j,i));
    x(m,0,k,j,i)=(u0(m,IM1,k,j,i)+qdep_(m,1,k,j,i))*pinv;
    x(m,1,k,j,i)=(u0(m,IM2,k,j,i)+qdep_(m,2,k,j,i))*pinv;
    x(m,2,k,j,i)=(u0(m,IM3,k,j,i)+qdep_(m,3,k,j,i))*pinv;
  });

  solver_last_iterations=0;
  solver_last_fast_accept=false;
  solver_last_residual=-1.0;
  solver_last_epsmax=-1.0;
  solver_last_error_bound=-1.0;
  solver_last_acceptance_target=-1.0;
  solver_last_state_scale=-1.0;
  if (drag_solver == DustDragSolver::applya) {
    ApplyCoupledOperator(x,ap,a_dt);
  } else if (drag_solver == DustDragSolver::dc1 ||
             drag_solver == DustDragSolver::dc2 ||
             drag_solver == DustDragSolver::adaptive) {
    // Fixed defect correction: DC1/adaptive take one sweep and DC2 takes two. DC2 is
    // deliberately accepted after its second sweep without a residual reduction or a
    // PCG fallback, giving a deterministic two-ApplyA production cost.
    int nsweeps=(drag_solver == DustDragSolver::dc2)?2:1;
    for (int sweep=0;sweep<nsweeps;++sweep) {
      ApplyCoupledOperator(x,ap,a_dt);
      par_for("dust_solver_dc",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(const int m,const int k,const int j,const int i) {
        Real pinv=1.0/(u0(m,IDN,k,j,i)+qdep_(m,0,k,j,i));
        for (int d=0;d<3;++d) {
          Real residual =
              u0(m,IM1+d,k,j,i)+qdep_(m,1+d,k,j,i)-ap(m,d,k,j,i);
          x(m,d,k,j,i)+=pinv*residual;
        }
      });
    }
    if (drag_solver == DustDragSolver::adaptive) {
      ApplyCoupledOperator(x,ap,a_dt);
      par_for("dust_adaptive_true_r",DevExeSpace(),0,nmb1,ks,ke,js,je,is,ie,
      KOKKOS_LAMBDA(const int m,const int k,const int j,const int i) {
        for (int d=0;d<3;++d) {
          r(m,d,k,j,i)=u0(m,IM1+d,k,j,i)+qdep_(m,1+d,k,j,i)-ap(m,d,k,j,i);
        }
      });
      Real norm2[3];
      GlobalPreconditionedNorm(r,norm2);
      Real residual_norm=std::sqrt(std::max(norm2[0]+norm2[1]+norm2[2],0.0));
      solver_last_residual=residual_norm;
      Real state_scale,target;
      Real bound=AdaptiveErrorBound(a_dt,residual_norm,state_scale,target);
      solver_last_error_bound=bound;
      solver_last_acceptance_target=target;
      solver_last_state_scale=state_scale;
      if (bound <= target) {
        solver_last_fast_accept=true;
        ++solver_fast_accept_count;
      } else {
        solver_last_iterations=StrictPCG(a_dt,true);
        ++solver_pcg_stage_count;
        if (global_variable::my_rank == 0) {
          ++solver_pcg_iteration_hist[static_cast<std::size_t>(solver_last_iterations)];
        }
      }
    }
  } else if (drag_solver == DustDragSolver::pcg) {
    solver_last_iterations=StrictPCG(a_dt,false);
    ++solver_pcg_stage_count;
    if (global_variable::my_rank == 0) {
      ++solver_pcg_iteration_hist[static_cast<std::size_t>(solver_last_iterations)];
    }
  }

  Kokkos::fence();
  solver_wall_seconds+=std::chrono::duration<double>(
      std::chrono::steady_clock::now()-started).count();
  ++solver_stage_count;
  if (drag_diagnostic_interval>0 && stage==2 &&
      (pmy_pack->pmesh->ncycle%drag_diagnostic_interval)==0 &&
      global_variable::my_rank==0) {
    std::cout << "# DUST_SOLVER_DIAG cycle=" << pmy_pack->pmesh->ncycle
              << " eps_c_max=" << solver_last_epsmax
              << " true_residual=" << solver_last_residual
              << " alg_error_bound=" << solver_last_error_bound
              << " acceptance_target=" << solver_last_acceptance_target
              << " state_scale=" << solver_last_state_scale
              << " pcg_iterations=" << solver_last_iterations
              << " fast_accept=" << (solver_last_fast_accept?1:0) << std::endl;
  }
  return TaskStatus::complete;
}

} // namespace dust
