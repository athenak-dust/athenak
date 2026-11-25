//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_imex3d_dc.cpp
//! \brief <dust>/drag_solver = dc1 (dc2) for 3-D refinable clouds, the default with coupling =
//! imex: one (two) defect-correction sweep(s) of the lumped IMEX drag solve.  Per working stage,
//! after the lumped solve x0 = (M + P)/(rho + Q) of every leaf cell (GasImplicitSolve):
//!
//!   x1 = x0 + (rho + Q)^-1 (M + P - rho x0 - D M G x0),
//!
//! with G the clouds' linear-moment u* gather of the kick (CloudGatherImplicit3D: cell values
//! and slopes, the cloud moments, GatherCode), M = diag(m_p c_p) and D
//! the deposit of the kick's impulse (the cloud's own weights).  The ghost cells of x0 come from
//! a blocking exchange with its own boundary objects in the order of the task chain (restrict,
//! exchange, fine x3 values, shear collection, coarse x3 values / fill / prolongation, fine x3
//! values), at the t^n shear offset; the clouds' positions take ImplicitShearShift as in the
//! kick.  x1 replaces u* before the common u* exchange; the kick and the gas split
//! (imex_consistent_gas) then run as before.  The dt-dependent gather blend
//! (test_imex_residual_to_center_upto_nominal) is off on this path, in A and in the kick (it
//! alone keeps the scheme first order; test_cloud_dc1_blend, which kept it, retired).
//! The center rule and its floor stay in the gas split.
//!
//! The blocking calls run inside one task.  This is safe under the current execution
//! contract (one MeshBlockPack per rank, the IMEX stages call this task in turn on every rank,
//! with or without particles); see dust_solver.cpp.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "shearing_box/shearing_box.hpp"
#include "dust.hpp"
#include "dust_imex3d_gather.hpp"
#include "dust_cloud_imex_kernels.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust {
namespace {

void DcRequire(const TaskStatus status, const char *what) {
  if (status == TaskStatus::fail) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "dc1: boundary operation failed: " << what << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    std::exit(EXIT_FAILURE);
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::EnsureCloudDcField
//! \brief A 3-component leaf field with the cell extents of u* and at least nslot_min block
//! slots (the halo slots of the deposits: leaf_traversal3d.nfld, at least the local blocks); every
//! extent is checked at every use (a rebuild of the leaf traversal or a wider halo changes nfld).

void DustGasDrag::EnsureCloudDcField(DvceArray5D<Real> &f, const int nslot_min) {
  const int nslot = std::max(nslot_min, pmy_pack->nmb_thispack);
  if (f.extent_int(0) < nslot || f.extent_int(1) != 3 || f.extent(2) != ustar.extent(2) ||
      f.extent(3) != ustar.extent(3) || f.extent(4) != ustar.extent(4)) {
    // (a little room, so that a slightly wider halo does not reallocate again)
    Kokkos::realloc(f, nslot + nslot/8 + 1, 3, ustar.extent(2), ustar.extent(3), ustar.extent(4));
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ExchangeUstarBlocking
//! \brief The ghost cells of a 3-component velocity field (and its coarse buffer) as the task
//! chain fills those of u* (SendUstar .. RecvUstarShr, FinishUstarGhosts), completed at once
//! with the private objects pbval_dc / psbox_dc (own communicators, requests and buffers; the
//! u* receives the stage has posted are not touched).  The shear offset is that of t^n, as in
//! InitRecvDep.  Used in 2-D and 3-D, by the clouds' dc1 / dc2 and by the point particles' coupled
//! solvers (formerly ExchangeUstarBlocking3D).

void DustGasDrag::ExchangeUstarBlocking(DvceArray5D<Real> &field, DvceArray5D<Real> &cfield) {
  Mesh *pm = pmy_pack->pmesh;
  ProfTimer tx(dc_prof_t_[0], dc_prof_n_[0]);   // (<dust>/amr_profile: the whole exchange)
  // (1) restrict
  if (pm->multilevel) pm->pmr->RestrictCC(field, cfield);
  // (2) the ordinary exchange; the shear receives (or the pull plan) of t^n posted first
  DcRequire(pbval_dc->InitRecv(3), "u* InitRecv");
  if (psbox_dc != nullptr) DcRequire(psbox_dc->InitRecv(pm->time), "u* shear InitRecv");
  DcRequire(pbval_dc->PackAndSendCC(field, cfield), "u* send");
  TaskStatus st;
  do {
    st = pbval_dc->RecvAndUnpackCC(field, cfield);
    DcRequire(st, "u* receive");
  } while (st == TaskStatus::incomplete);
  if (psbox_dc != nullptr) {
    // (3) collected ghost zones: the fine x3 boundary values before the shear (SendUstarShr)
    if (psbox_dc->face_levels) UstarX3BCs(field, false);
    // (4) the shear collection, fine and coarse
    DcRequire(psbox_dc->PackAndSendCC(field, ReconstructionMethod::plm), "u* shear send");
    if (psbox_dc->coarse_on) {
      DcRequire(psbox_dc->PackAndSendCoarseCC(cfield, ReconstructionMethod::plm),
                "u* shear coarse send");
    }
    do {
      st = psbox_dc->RecvAndUnpackCC(field);
      DcRequire(st, "u* shear receive");
    } while (st == TaskStatus::incomplete);
    if (psbox_dc->coarse_on) {
      do {
        st = psbox_dc->RecvAndUnpackCoarseCC(cfield);
        DcRequire(st, "u* shear coarse receive");
      } while (st == TaskStatus::incomplete);
    }
  }
  // (5) coarse x3 values, fill, prolongation; (6) fine x3 values
  FinishUstarGhosts(field, cfield, pbval_dc);
  // drain the private requests
  DcRequire(pbval_dc->ClearSend(), "u* ClearSend");
  DcRequire(pbval_dc->ClearRecv(), "u* ClearRecv");
  if (psbox_dc != nullptr) {
    DcRequire(psbox_dc->ClearSend(), "u* shear ClearSend");
    DcRequire(psbox_dc->ClearRecv(), "u* shear ClearRecv");
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudCoupledOperator, CloudCoupledOperatorT
//! \brief result = D M G field: the gather of the kick (CloudGatherImplicit3D, the same
//! arithmetic, without the gather blend) of every cloud times m_p c_p, deposited with
//! the weights of the kick's impulse deposit, the halo slots summed to their owners.  The ghost
//! cells of field must be filled.  The particles are not changed.  gath (optional): every
//! cloud's gathered velocity (npart, 3).

namespace {
// the traversal of a dimension (a copy is taken by the caller)
template <class LTT> struct TraversalOfT;
template <> struct TraversalOfT<LeafTraversal> {
  static const LeafTraversal &Get(const DustGasDrag &d) {return d.leaf_traversal;}
};
template <> struct TraversalOfT<LeafTraversal3D> {
  static const LeafTraversal3D &Get(const DustGasDrag &d) {return d.leaf_traversal3d;}
};
}  // namespace

// (the 2-D and 3-D host steps written once; the dimension dispatch in one place)
void DustGasDrag::CloudCoupledOperator(Driver *pdrive, int stage, DvceArray5D<Real> &field,
                                       DvceArray5D<Real> &result,
                                       DvceArray2D<Real> *gath, const bool own_weights) {
  if (cloud_dim == 3) {
    CloudCoupledOperatorT<LeafTraversal3D>(pdrive, stage, field, result, gath, own_weights);
  } else {
    CloudCoupledOperatorT<LeafTraversal>(pdrive, stage, field, result, gath, own_weights);
  }
}

template <class LTT>
void DustGasDrag::CloudCoupledOperatorT(Driver *pdrive, int stage, DvceArray5D<Real> &field,
                                        DvceArray5D<Real> &result,
                                        DvceArray2D<Real> *gath, const bool own_weights) {
  constexpr bool k3 = (LeafApi<LTT>::kDim == 3);
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const Real a_dt = DragStep(pdrive);
  // 3-D: the shear shift of the kick's gather (CloudGatherImplicit3D) and the shear geometry at
  // the time of the step.  2-D: the own-weights deposit of the operator check
  Real sh = 0.0;
  if constexpr (k3) {
    sh = ImplicitShearShift(pdrive, stage);
    UpdateShear3D(pmy_pack->pmesh->time);
  } else {
    (void)stage;
  }
  const LTT lt = TraversalOfT<LTT>::Get(*this);
  {
    ProfTimer tp(dc_prof_t_[1], dc_prof_n_[1]);   // the pack and its forward exchange
    if constexpr (k3) {
      PackGatherField3D(field);
    } else {
      PackGatherField(field);
    }
  }
  auto gp = k3 ? gpack3 : gpack;
  EnsureCloudDcField(result, lt.nfld);
  Kokkos::deep_copy(DevExeSpace(), result, 0.0);
  const bool keep = (gath != nullptr);
  if (keep && (gath->extent_int(0) < npart || gath->extent_int(1) != 3)) {
    Kokkos::realloc(*gath, std::max(1, npart), 3);
  }
  DvceArray2D<Real> gv;
  if (keep) gv = *gath;
  const int nord = CloudOrderLength(npart);
  {
    ProfTimer tk(dc_prof_t_[2], dc_prof_n_[2]);   // the particle kernel
    if (keep) {
      CloudOperatorKernelT<LTT, true>(lt, pr, pi, gp, result, gv, a_dt, sh,
                                      own_weights, is, js, ks, npart, cloud_ord, nord);
    } else {
      CloudOperatorKernelT<LTT, false>(lt, pr, pi, gp, result, gv, a_dt, sh,
                                       own_weights, is, js, ks, npart, cloud_ord, nord);
    }
  }
  ProfTimer tr(dc_prof_t_[3], dc_prof_n_[3]);   // the reverse sum of the halo slots
  HaloReverseAdd(result, 0, 3);
}
template void DustGasDrag::CloudCoupledOperatorT<LeafTraversal>(Driver *, int,
    DvceArray5D<Real> &, DvceArray5D<Real> &, DvceArray2D<Real> *, const bool);
template void DustGasDrag::CloudCoupledOperatorT<LeafTraversal3D>(Driver *, int,
    DvceArray5D<Real> &, DvceArray5D<Real> &, DvceArray2D<Real> *, const bool);

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudDefectCorrection
//! \brief cloud_dc_sweeps sweeps (dc1: 1, dc2: 2) of x <- x + (rho + Q)^-1 (M + P - rho x -
//! D M G x) on u* (the leaf cells of this rank); called by every rank at every working stage.
//! The operator: CloudCoupledOperator (both dimensions).
//! The ghost cells of x by the blocking exchange (the linear gather reads them)

void DustGasDrag::CloudDefectCorrection(Driver *pdrive, int stage) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  // dmom is free here: the drag history of the stage has read it (AddDragHistoryGas precedes the
  // solve) and the kick clears it before its deposit; its slots cover the halo (nfld).  No
  // gather blend (as the kick's on the dc1 path, imex_gather_blend off)
  for (int s=0; s<cloud_dc_sweeps; ++s) {
    ExchangeUstarBlocking(ustar, dc_cfield);
    CloudCoupledOperator(pdrive, stage, ustar, dmom, nullptr, false);
    ProfTimer tc(dc_prof_t_[4], dc_prof_n_[4]);   // the correction of the leaf cells
    auto &u0 = pmy_pack->phydro->u0;
    auto qp = qdep;
    auto x = ustar;
    auto wk = dmom;
    par_for("dust_cloud_dc_correct", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const Real rho = u0(m,IDN,k,j,i);
      const Real pinv = 1.0/(rho + qp(m,0,k,j,i));
      for (int d=0; d<3; ++d) {
        const Real av = wk(m,d,k,j,i) + rho*x(m,d,k,j,i);
        const Real rv = u0(m,IM1+d,k,j,i) + qp(m,1+d,k,j,i) - av;
        x(m,d,k,j,i) += pinv*rv;
      }
    });
  }
  if (cloud_dc1_opcheck) CloudDcOpCheckPoison();
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudDcOpCheck*
//! \brief test_cloud_dc1_opcheck: the operator of the correction against the kick, at every
//! working stage.  (Poison) after the correction the ghost cells of u* and its coarse buffer
//! get the value 3e3, which the task chain must overwrite where the gather reads.  (Before, in
//! GatherKickPMBR) the operator on u* as the task chain left it (path A), and on a copy of the
//! active cells whose ghost cells (-7e3) and coarse buffer are filled by the blocking exchange
//! (path B): every cloud's gather and the owners' D M G x of both, and the particles unchanged
//! by them.  (AfterKick) the kick's gather, v + dv/c_p from the stored velocity and the rates,
//! against path A.  (Dmom, ApplyPMBR before the gas split) the owners' impulse deposit dmom
//! against P - D M G u* of path A.  Rank 0 prints the maxima over the ranks per stage.

namespace {
constexpr Real kPoisonA = 3.0e3;
constexpr Real kPoisonB = -7.0e3;
}  // namespace

void DustGasDrag::CloudDcOpCheckPoison() {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto x = ustar;
  const int nk = x.extent_int(2), nj = x.extent_int(3), ni = x.extent_int(4);
  par_for("dust_cloud_dc_poison", DevExeSpace(), 0, nmb1, 0, nk-1, 0, nj-1, 0, ni-1,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    if (k >= ks && k <= ke && j >= js && j <= je && i >= is && i <= ie) return;
    for (int c=0; c<3; ++c) x(m,c,k,j,i) = kPoisonA;
  });
  if (pmy_pack->pmesh->multilevel) Kokkos::deep_copy(DevExeSpace(), cdummy, kPoisonA);
}

void DustGasDrag::CloudDcOpCheckBefore(Driver *pdrive, int stage) {
  for (int q=0; q<8; ++q) dc_check_[q] = 0.0;
  auto pr = pmy_pack->ppart->prtcl_rdata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  if (dc_psave.extent_int(0) != 6 || dc_psave.extent_int(1) < npart) {
    Kokkos::realloc(dc_psave, 6, std::max(1, npart));
  }
  auto ps = dc_psave;
  par_for("dust_cloud_dc_psave", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
    ps(0,p) = pr(IPVX,p); ps(1,p) = pr(IPVY,p); ps(2,p) = pr(IPVZ,p);
    ps(3,p) = pr(IPRX,p); ps(4,p) = pr(IPRY,p); ps(5,p) = pr(IPRZ,p);
  });
  // (the kick's gather: no blend on the dc1 path)
  // the operator of the correction (point particles: PointOperatorParticles and the additive
  // exchange; clouds: CloudCoupledOperator), and the copy exchange of path B
  auto dcop = [&](DvceArray5D<Real> &f, DvceArray5D<Real> &res, DvceArray2D<Real> *g) {
    if (cloud_dim == 3 || UsesRefinableClouds()) {   // refinable clouds, 2-D or 3-D
      CloudCoupledOperator(pdrive, stage, f, res, g, false);
    } else {
      // (shaped as u*: the additive exchange works on its block slots)
      if (res.extent(0) != ustar.extent(0) || res.extent(1) != 3 ||
          res.extent(2) != ustar.extent(2) ||
          res.extent(3) != ustar.extent(3) || res.extent(4) != ustar.extent(4)) {
        Kokkos::realloc(res, ustar.extent(0), 3, ustar.extent(2), ustar.extent(3), ustar.extent(4));
      }
      Kokkos::deep_copy(DevExeSpace(), res, 0.0);
      PointOperatorParticles(f, res, DragStep(pdrive), g);
      CompleteAddExchange(res);
    }
  };
  // path A: u* with the ghost cells of the task chain
  dcop(ustar, dc_res_a, &dc_gath_a);
  // path B: the active cells of u*, own ghost cells and coarse buffer, the blocking exchange
  // (the extents of u* itself: the exchange objects and the restriction work on all its block
  // slots)
  if (dc_tfield.extent(0) != ustar.extent(0) || dc_tfield.extent(1) != ustar.extent(1) ||
      dc_tfield.extent(2) != ustar.extent(2) || dc_tfield.extent(3) != ustar.extent(3) ||
      dc_tfield.extent(4) != ustar.extent(4)) {
    Kokkos::realloc(dc_tfield, ustar.extent(0), ustar.extent(1), ustar.extent(2), ustar.extent(3),
                    ustar.extent(4));
  }
  if (dc_tcfield.extent(0) != dc_cfield.extent(0) || dc_tcfield.extent(1) != dc_cfield.extent(1) ||
      dc_tcfield.extent(2) != dc_cfield.extent(2) || dc_tcfield.extent(3) != dc_cfield.extent(3) ||
      dc_tcfield.extent(4) != dc_cfield.extent(4)) {
    Kokkos::realloc(dc_tcfield, dc_cfield.extent(0), dc_cfield.extent(1), dc_cfield.extent(2),
                    dc_cfield.extent(3), dc_cfield.extent(4));
  }
  Kokkos::deep_copy(DevExeSpace(), dc_tcfield, kPoisonB);
  {
    const auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
    const int ks = indcs.ks, ke = indcs.ke;
    const int nmb1 = pmy_pack->nmb_thispack - 1;
    auto t = dc_tfield;
    auto x = ustar;
    const int nk = x.extent_int(2), nj = x.extent_int(3), ni = x.extent_int(4);
    par_for("dust_cloud_dc_tcopy", DevExeSpace(), 0, nmb1, 0, nk-1, 0, nj-1, 0, ni-1,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      const bool act = (k >= ks && k <= ke && j >= js && j <= je && i >= is && i <= ie);
      for (int c=0; c<3; ++c) t(m,c,k,j,i) = act ? x(m,c,k,j,i) : kPoisonB;
    });
  }
  if (cloud_dim == 3) {
    ExchangeUstarBlocking(dc_tfield, dc_tcfield);
  } else if (UsesRefinableClouds()) {
    // 2-D clouds: the linear gather reads ghost cells
    ExchangeUstarBlocking(dc_tfield, dc_tcfield);
  } else {
    PointCopyExchange(dc_tfield);
  }
  dcop(dc_tfield, dc_res_b, &dc_gath_b);
  // the gathers of both paths, the particles unchanged
  Real m0 = 0.0, m2 = 0.0;
  if (npart > 0) {
    auto ga = dc_gath_a, gb = dc_gath_b;
    Kokkos::parallel_reduce("dust_cloud_dc_check_gather",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &a, Real &b) {
      for (int c=0; c<3; ++c) a = fmax(a, fabs(ga(p,c) - gb(p,c)));
      b = fmax(b, fabs(pr(IPVX,p) - ps(0,p)));
      b = fmax(b, fabs(pr(IPVY,p) - ps(1,p)));
      b = fmax(b, fabs(pr(IPVZ,p) - ps(2,p)));
      b = fmax(b, fabs(pr(IPRX,p) - ps(3,p)));
      b = fmax(b, fabs(pr(IPRY,p) - ps(4,p)));
      b = fmax(b, fabs(pr(IPRZ,p) - ps(5,p)));
      // (a NaN anywhere: report it)
      for (int c=0; c<3; ++c) if (ga(p,c) != ga(p,c) || gb(p,c) != gb(p,c)) a = 1.0e300;
    }, Kokkos::Max<Real>(m0), Kokkos::Max<Real>(m2));
  }
  dc_check_[0] = m0;
  dc_check_[2] = m2;
  // the owners' results of both paths
  {
    const auto &indcs = pmy_pack->pmesh->mb_indcs;
    const int is = indcs.is, js = indcs.js, ks = indcs.ks;
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int ncells = pmy_pack->nmb_thispack*nx3*nx2*nx1;
    auto ra = dc_res_a, rb = dc_res_b;
    Real m1 = 0.0;
    Kokkos::parallel_reduce("dust_cloud_dc_check_result",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
    KOKKOS_LAMBDA(const int idx, Real &a) {
      int q = idx;
      const int i = q % nx1 + is; q /= nx1;
      const int j = q % nx2 + js; q /= nx2;
      const int k = q % nx3 + ks; q /= nx3;
      const int m = q;
      for (int c=0; c<3; ++c) {
        const Real d = fabs(ra(m,c,k,j,i) - rb(m,c,k,j,i));
        a = fmax(a, (d == d) ? d : 1.0e300);
      }
    }, Kokkos::Max<Real>(m1));
    dc_check_[1] = m1;
  }
  // 2-D clouds: the cached deposit weights against the clouds' own (ForEachCloudWeight of their
  // level), and the counts of the deposit branches
  if (cloud_dim != 3 && UsesRefinableClouds()) CloudDcOpCheck2D(pdrive, stage);
}

void DustGasDrag::CloudDcOpCheckAfterKick(Driver *pdrive, int stage) {
  (void)stage;
  auto pr = pmy_pack->ppart->prtcl_rdata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const Real a_dt = DragStep(pdrive);
  Real m3 = 0.0, m4 = 0.0;
  if (npart > 0) {
    auto ga = dc_gath_a;
    auto ps = dc_psave;
    Kokkos::parallel_reduce("dust_cloud_dc_check_kick",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &a, Real &b) {
      const Real cj = a_dt/(pr(IPTS,p) + a_dt);
      // the kick: dv = c_p (ubar - v), the rate dv/a_dt
      const Real gk[3] = {ps(0,p) + pr(IPRX,p)*a_dt/cj, ps(1,p) + pr(IPRY,p)*a_dt/cj,
                          ps(2,p) + pr(IPRZ,p)*a_dt/cj};
      for (int c=0; c<3; ++c) {
        const Real d = fabs(gk[c] - ga(p,c));
        a = fmax(a, (d == d) ? d : 1.0e300);
        b = fmax(b, fabs(ga(p,c)));
      }
    }, Kokkos::Max<Real>(m3), Kokkos::Max<Real>(m4));
  }
  dc_check_[3] = m3;
  dc_check_[4] = m4;
}

void DustGasDrag::CloudDcOpCheckDmom(Driver *pdrive, int stage) {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int ncells = pmy_pack->nmb_thispack*nx3*nx2*nx1;
  auto ra = dc_res_a;
  auto qp = qdep;
  auto dm = dmom;
  Real m5 = 0.0, m6 = 0.0;
  Kokkos::parallel_reduce("dust_cloud_dc_check_dmom",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
  KOKKOS_LAMBDA(const int idx, Real &a, Real &b) {
    int q = idx;
    const int i = q % nx1 + is; q /= nx1;
    const int j = q % nx2 + js; q /= nx2;
    const int k = q % nx3 + ks; q /= nx3;
    const int m = q;
    for (int c=0; c<3; ++c) {
      // dmom = D(-m (c (G u* - v))) = P - D M G u*
      const Real d = fabs(dm(m,c,k,j,i) - (qp(m,1+c,k,j,i) - ra(m,c,k,j,i)));
      a = fmax(a, (d == d) ? d : 1.0e300);
      b = fmax(b, fabs(dm(m,c,k,j,i)));
    }
  }, Kokkos::Max<Real>(m5), Kokkos::Max<Real>(m6));
  dc_check_[5] = m5;
  dc_check_[6] = m6;
  long long np = pmy_pack->ppart->nprtcl_thispack;
  long long nzero = (np == 0) ? 1 : 0;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, dc_check_, 8, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &np, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &nzero, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(6) << "# DC1_OPCHECK cycle " << pmy_pack->pmesh->ncycle
              << " stage " << stage << " particles " << np << " ranks_without " << nzero
              << " | gather A-B " << dc_check_[0] << " result A-B " << dc_check_[1]
              << " | particles changed by the operator " << dc_check_[2]
              << " | kick-op gather " << dc_check_[3] << " (|ubar| " << dc_check_[4] << ")"
              << " | dmom-(P-DMGu) " << dc_check_[5] << " (|dmom| " << dc_check_[6] << ")";
    if (cloud_dim != 3 && UsesRefinableClouds()) {
      std::cout << " | 2-D: cached-vs-own deposit weights " << dc_check_[7] << ", clouds with "
                << "nt > " << CloudGeometry::kCloudWeightCache << " (same shape) "
                << dc_cnt2d_[0] << ", lobed (gather shape != deposit shape) " << dc_cnt2d_[1];
    }
    std::cout << std::endl;
  }
  (void)pdrive;
}


//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudDcSkipCheck
//! \brief test_cloud_dc1_opcheck, floor mode: the consistency step skips the overlap gather of
//! the clouds the kick did not mark (imex_need = 0).  Here every skipped cloud gathers the
//! overlap part (gpack3 components 3-5, as the consistency step left them) in full: the largest
//! |value| over them must be exactly 0.  Also the share of the clouds that need the gather.

void DustGasDrag::CloudDcSkipCheck(Driver *pdrive, int stage) {
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const Real sh = ImplicitShearShift(pdrive, stage);
  const auto lt = leaf_traversal3d;
  auto gf = gpack3;
  auto nd = imex_need;
  // (a Max reduction starts at its identity, -DBL_MAX for double: with no cloud checked on any
  // rank it would stay there, so the count of the checked clouds goes with it)
  Real vmax = 0.0;
  int nneed = 0, nchk = 0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_cloud_dc_skipcheck",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &a, int &n, int &nc) {
      if (pr(IPM,p) <= 0.0) return;
      if (nd(p) != 0) {++n; return;}
      ++nc;
      Real rv[3] = {0.0, 0.0, 0.0};
      lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p) + sh*pr(IPX,p), pr(IPZ,p), pi(PLEV,p),
                            lt.Anchor(pi(PGID,p)),
      [&](const int target, const Real w) {
        int b, k, j, i;
        lt.Decode(target, b, k, j, i);
        for (int c=0; c<3; ++c) rv[c] += w*gf(b,k,j,i,3+c);
      });
      for (int c=0; c<3; ++c) a = fmax(a, (rv[c] == rv[c]) ? fabs(rv[c]) : 1.0e300);
    }, Kokkos::Max<Real>(vmax), Kokkos::Sum<int>(nneed), Kokkos::Sum<int>(nchk));
  }
  if (nchk == 0) vmax = 0.0;   // (nothing checked on this rank: no -DBL_MAX into the maximum)
  long long cnt[3] = {nneed, npart, nchk};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, cnt, 3, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(6) << "# DC1_SKIPCHECK cycle " << pmy_pack->pmesh->ncycle
              << " stage " << stage << " | skipped clouds' overlap part by a full gather: nchecked="
              << cnt[2] << " max ";
    if (cnt[2] > 0) {
      std::cout << vmax;
    } else {
      std::cout << "n/a";
    }
    std::cout << " | clouds that need the overlap gather " << cnt[0] << " of " << cnt[1]
              << std::endl;
  }
}

}  // namespace dust
