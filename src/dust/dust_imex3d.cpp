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

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"
#include "dust_imex3d_gather.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust {

// test_imex_residual_to_center_upto_nominal, floor: the centered clouds' share of a
// cell's dust residual from their
// coupled mass density qt and the cell's B (CloudCenterMass3D): 1 for qt >= B, qt/B below (never
// 1/B first: it overflows for tiny B), 0 without centered clouds or with an invalid B (underflow).
KOKKOS_INLINE_FUNCTION
static Real FloorShare(const Real qt, const Real b) {
  if (!(qt > 0.0) || !(b > 0.0)) return 0.0;
  return (qt >= b) ? 1.0 : qt/b;
}

// The stage position geometry is distinct from ShearRouteTime, whose crossing offset
// also compensates for background drift in the stages still to come.
// Form the abscissa from the same low-storage recurrence as ExplicitPush, including a
// gamma switch. The final beta=0 assembly stage never reaches either kernel.
Real DustGasDrag::CloudImplicitTime3D(Driver *pdrive, int stage) const {
  Real c = 0.0;
  for (int s=0; s<stage; ++s) c = pdrive->gam0[s]*c + pdrive->beta[s];
  return pmy_pack->pmesh->time + c*pmy_pack->pmesh->dt;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudDepositImplicitT
//! \brief The IMEX working stage's deposit of the clouds on either traversal (the 2-D
//! kernel of DepositDrag and the 3-D one written once): Q = m c_p W / V and P = Q v into qdep 0-3
//! (cleared by DepositDrag, halo slots included), the halo slots then summed to their owners.  The
//! per-dimension expressions (the coupled mass, the 3-D orbital-advection position, the atomics)
//! in LeafApi.

template <class LTT>
void DustGasDrag::CloudDepositImplicitT(const LTT &lt_in, Driver *pdrive, int stage) {
  using A = LeafApi<LTT>;
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const Real a_dt = DragStep(pdrive);
  Real sh = 0.0;
  if constexpr (A::kDim == 3) {
    // the gas is in its t^n orbital-advection frame: shear images of t^n, positions shifted
    sh = ImplicitShearShift(pdrive, stage);
    UpdateShear3D(pmy_pack->pmesh->time);
  }
  const LTT lt = lt_in;
  auto qp = qdep;  // cleared, including halo slots, by DepositDrag
  const std::size_t sq = qp.stride(1);
  auto body = KOKKOS_LAMBDA(const int p) {
    Real x[3];
    A::LoadImplicit(pr, p, sh, x);
    const Real mass_c = A::CoupledMass(pr(IPM,p), pr(IPTS,p), a_dt);
    const Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
    A::ForEachWeight(lt, x, pi(PLEV,p), lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w) {
      int b, k, j, i;
      A::Decode(lt, target, b, k, j, i);
      const Real dw = mass_c*w/A::Measure(lt, b);
      Real v[4] = {dw, dw*vx, dw*vy, dw*vz};
      A::template DepositN<4>(qp, sq, target, b, k, j, i, v);
    });
  };
  // Migration can change the number of particles between the two working stages.
  if (const int nord = CloudOrderLength(npart)) {
    CloudParForOrdered(A::kDim == 3 ? "dust_imex_cloud3d_deposit" : "dust_composite_scatter", npart,
                       cloud_ord, nord, body);
  } else {   // (label: the storage order)
    CloudParFor(A::kDim == 3 ? "dust_imex_cloud3d_deposit_storage" : "dust_composite_scatter", npart,
                body);
  }
  HaloReverseAdd(qdep, 0, 4);   // halo deposits back to their owners
}
template void DustGasDrag::CloudDepositImplicitT<LeafTraversal>(const LeafTraversal &, Driver *, int);
template void DustGasDrag::CloudDepositImplicitT<LeafTraversal3D>(const LeafTraversal3D &, Driver *,
                                                                  int);

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudGatherKickT
//! \brief The IMEX stage's gather of u* and kick of the clouds without the center rule's blend or
//! marks, on either traversal (the 2-D kernel of GatherKickPMBR and the plain 3-D
//! kernel of CloudGatherImplicit3D written once): the linear-moment gather of the packed u* (gp),
//! the kick dv = c_p (ubar - v), the rate dv / h for
//! the next stage, the actual impulse deposited into dmom with the cloud's weights (the gather's
//! when the shapes agree).  dmom is cleared by the caller and summed to the owners here.

template <class LTT>
void DustGasDrag::CloudGatherKickT(const LTT &lt, const DvceArray5D<Real> &gpk, const Real a_dt,
                                   const Real sh, const char *label) {
  using A = LeafApi<LTT>;
  constexpr int kC = A::kCache;
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const bool br = back_reaction;
  auto gp = gpk;
  auto dm = dmom;  // cleared, including halo slots, by GatherKickPMBR
  const std::size_t sd = dm.stride(1);
  auto body = KOKKOS_LAMBDA(const int p) {
    using T = CloudGeometry;
    int nt = 0, tt[kC];
    Real tw[kC];
    Real x[3];
    A::LoadImplicit(pr, p, sh, x);
    const int am = lt.Anchor(pi(PGID,p)), lev = pi(PLEV,p);
    const int glev = T::GatherCode(lev);
    // same shape for the gather and the deposit (the point flag changes moments only)
    const bool same = (glev & ~T::kCloudGatherPoint) == lev;
    Real ug[3] = {0.0, 0.0, 0.0};
    A::ForEachMoment(lt, x, glev, am,
    [&](const int target, const Real w, const Real m1x, const Real m1y, const Real m1z) {
      A::GatherLinear(lt, gp, target, w, m1x, m1y, m1z, x, is, js, ks, ug);
      if (nt < kC) {tt[nt] = target; tw[nt] = w;}
      ++nt;
    });
    // the kick (dv = c_p (ubar - v)); the next working stage consumes rates, not impulses
    const Real cj = a_dt/(pr(IPTS,p) + a_dt);
    const Real dvx = cj*(ug[0] - pr(IPVX,p));
    const Real dvy = cj*(ug[1] - pr(IPVY,p));
    const Real dvz = cj*(ug[2] - pr(IPVZ,p));
    pr(IPVX,p) += dvx; pr(IPVY,p) += dvy; pr(IPVZ,p) += dvz;
    pr(IPRX,p) = dvx/a_dt; pr(IPRY,p) = dvy/a_dt; pr(IPRZ,p) = dvz/a_dt;
    if (br) {
      const Real mass = -pr(IPM,p);
      // the actual applied particle impulse (conservative with a linear gather and across
      // refinement, although that gather differs from W)
      auto add = [&](const int target, const Real w) {
        int b, k, j, i;
        A::Decode(lt, target, b, k, j, i);
        const Real dw = mass*w/A::Measure(lt, b);
        Real v[3] = {dw*dvx, dw*dvy, dw*dvz};
        A::template DepositN<3>(dm, sd, target, b, k, j, i, v);
      };
      if (same && nt <= kC) {
        for (int q=0; q<nt; ++q) if (tw[q] != 0.0) add(tt[q], tw[q]);
      } else {
        A::ForEachWeight(lt, x, lev, am, add);
      }
    }
  };
  if (const int nord = CloudOrderLength(npart)) {
    CloudParForOrdered(label, npart, cloud_ord, nord, body);
  } else {
    CloudParFor(A::kDim == 3 ? "dust_imex_cloud3d_gather_storage" : label, npart, body);
  }
  if (br) HaloReverseAdd(dmom, 0, 3);   // halo deposits back to their owners
}
template void DustGasDrag::CloudGatherKickT<LeafTraversal>(const LeafTraversal &,
    const DvceArray5D<Real> &, const Real, const Real, const char *);
template void DustGasDrag::CloudGatherKickT<LeafTraversal3D>(const LeafTraversal3D &,
    const DvceArray5D<Real> &, const Real, const Real, const char *);

void DustGasDrag::CloudGatherImplicit3D(Driver *pdrive, int stage) {
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const Real a_dt = DragStep(pdrive);
  const bool br = back_reaction;
  const Real sh = ImplicitShearShift(pdrive, stage);   // as in CloudDepositImplicitT
  // floor mode: the centered mass of every leaf cell before the gather (it depends only on the
  // positions, masses and the stage's drag step), so that the blend kernel can mark the clouds
  // that cover a cell with an overlap share (CloudImplicitConsistency3D)
  // (only with back-reaction: without it the consistency step never runs)
  const bool mark = br && imex_resid_center && imex_resid_center_floor && imex_consistent_gas;
  if (mark) CloudCenterMass3D(pdrive, stage);
  UpdateShear3D(pmy_pack->pmesh->time);
  const auto lt = leaf_traversal3d;
  // The common u* exchange has already filled same-level and coarse/fine ghosts,
  // including physical x3 boundaries. Pack owners' slopes before forwarding to halo.
  PackGatherField3D(ustar);
  auto gp = gpack3;
  auto dm = dmom;  // cleared, including halo slots, by GatherKickPMBR
  const std::size_t sd = dm.stride(1);
  // floor mode with the dc1 correction (drag_solver = dc1 / dc2, no gather blend): the cells
  // with an overlap share get the same mark (a negative fq) for the overlap-skip of the
  // consistency step
  const bool nbmark = mark && cloud_dc1 && !imex_gather_blend;
  if (imex_gather_blend || nbmark) {
    // gather blend (test_imex_residual_to_center_upto_nominal): f = Q/(rho + Q) of every
    // leaf cell, the dust share of the lumped
    // solve (qdep[0] after its halo sum; rho as in the solve), owners' values on the halo slots
    auto &indcs_ = pmy_pack->pmesh->mb_indcs;
    const int ie = indcs_.ie, je = indcs_.je, ke = indcs_.ke;
    const int nmb1 = pmy_pack->nmb_thispack - 1;
    // floor mode: f carries a minus sign where the cell has an overlap share (no centered cloud, or
    // fewer than one nominal cloud: the only cells that may have an overlap residual later); the
    // kernel reads |f|, so the mark costs no second load (f = 0 there: the smallest normal number)
    if (imex_fq.extent(0) != qdep.extent(0) || imex_fq.extent(2) != qdep.extent(2) ||
        imex_fq.extent(3) != qdep.extent(3) || imex_fq.extent(4) != qdep.extent(4)) {
      Kokkos::realloc(imex_fq, qdep.extent(0), 1, qdep.extent(2), qdep.extent(3), qdep.extent(4));
    }
    const Real ftiny = std::numeric_limits<Real>::min();
    auto fq = imex_fq;
    auto qp = qdep;
    auto qcn = imex_qcen;
    const Real bs = imex_bsingle;
    const bool fmulti = mark && imex_floor_multi;
    auto &u0 = pmy_pack->phydro->u0;
    if (nbmark) {
      // (no blend: only the mark, -1 on a cell with an overlap share, 0 elsewhere)
      par_for("dust_imex_fmark", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real qt = qcn(m,0,k,j,i);
        fq(m,0,k,j,i) = (FloorShare(qt, fmulti ? qcn(m,1,k,j,i) : bs) < 1.0) ? -1.0 : 0.0;
      });
    } else {
      par_for("dust_imex_fdust", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        const Real q = br ? qp(m,0,k,j,i) : 0.0;
        const Real fv = q/(u0(m,IDN,k,j,i) + q);
        fq(m,0,k,j,i) = fv;
        if (mark) {
          const Real qt = qcn(m,0,k,j,i);
          if (FloorShare(qt, fmulti ? qcn(m,1,k,j,i) : bs) < 1.0) fq(m,0,k,j,i) = -fmax(fv, ftiny);
        }
      });
    }
    HaloForward5D(imex_fq, 0, 1);
    if (mark && imex_need.extent_int(0) < npart) {   // (capacity in 64 bits, as bvals_part.cpp)
      const long long cap = static_cast<long long>(npart) + npart/8 + 1;
      Kokkos::realloc(imex_need, static_cast<int>(std::min<long long>(
                                     cap, std::numeric_limits<int>::max())));
    }
  }
  auto fq = imex_fq;
  auto nd = imex_need;
  constexpr int cache = 27;
  // the blend kernel (the slope terms kept apart, dust_imex3d_gather.hpp) / the plain kernel / the
  // plain kernel with the overlap marks (dc1 in floor mode); one gather helper for all of them
  // and for the dc1 operator
  auto body_b = KOKKOS_LAMBDA(const int p) {
    using T = CloudGeometry;
    int nt, tt[cache], ndp;
    Real tw[cache], ug[3];
    const Real xp = pr(IPX,p), yp = pr(IPY,p) + sh*pr(IPX,p), zp = pr(IPZ,p);
    const int am = lt.Anchor(pi(PGID,p)), lev = pi(PLEV,p);
    const int glev = T::GatherCode(lev);
    const bool same = (glev & ~T::kCloudGatherPoint) == lev;
    CloudLinearGather3D<true, false>(lt, gp, fq, xp, yp, zp, glev, am, is, js, ks, ug, nt, tt, tw,
                                     cache, ndp);
    // (a gather that is not the deposit's own weights covers other cells: no skipping then)
    if (mark) nd(p) = (ndp != 0 || !same) ? 1 : 0;
    CloudKickDeposit3D(lt, pr, p, ug, a_dt, br, dm, sd, same, nt, tt, tw, cache, xp, yp, zp, lev,
                       am);
  };
  auto body_m = KOKKOS_LAMBDA(const int p) {
    using T = CloudGeometry;
    int nt, tt[cache], ndp;
    Real tw[cache], ug[3];
    const Real xp = pr(IPX,p), yp = pr(IPY,p) + sh*pr(IPX,p), zp = pr(IPZ,p);
    const int am = lt.Anchor(pi(PGID,p)), lev = pi(PLEV,p);
    const int glev = T::GatherCode(lev);
    const bool same = (glev & ~T::kCloudGatherPoint) == lev;
    CloudLinearGather3D<false, true>(lt, gp, fq, xp, yp, zp, glev, am, is, js, ks, ug, nt, tt, tw,
                                     cache, ndp);
    nd(p) = (ndp != 0 || !same) ? 1 : 0;
    CloudKickDeposit3D(lt, pr, p, ug, a_dt, br, dm, sd, same, nt, tt, tw, cache, xp, yp, zp, lev,
                       am);
  };
  if (imex_gather_blend) {
    if (const int nord = CloudOrderLength(npart)) {
      CloudParForOrdered("dust_imex_cloud3d_gather_blend", npart, cloud_ord, nord, body_b);
    } else {
      CloudParFor("dust_imex_cloud3d_gather_blend_storage", npart, body_b);
    }
  } else if (nbmark) {
    if (const int nord = CloudOrderLength(npart)) {
      CloudParForOrdered("dust_imex_cloud3d_gather_mark", npart, cloud_ord, nord, body_m);
    } else {
      CloudParFor("dust_imex_cloud3d_gather_mark_storage", npart, body_m);
    }
  } else {
    // the plain kernel: the one of both traversals (it sums dmom to the owners)
    CloudGatherKickT(lt, gpack3, a_dt, sh, "dust_imex_cloud3d_gather");
    return;
  }
  if (br) HaloReverseAdd(dmom, 0, 3);
}


// The root cloud mass of every species estimated from the clouds present, for the floor of the
// center rule while the merge / resampling passes have no value yet: the largest plain level-0
// cloud, else eight times the largest plain level-1 cloud (the rule of EnsureRootCloudMass),
// reduced over the ranks.  It does not touch cloud_merge_mroot: the G / R passes set their own
// reference when they first run, as without the center rule.
std::vector<Real> DustGasDrag::ImexRootMassEstimate(const int nsp) {
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  std::vector<Real> out(nsp, 0.0);
  for (int sp=0; sp<nsp; ++sp) {
    Real m0 = 0.0, m1 = 0.0;
    Kokkos::parallel_reduce("dust_imex_root_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &a, Real &b) {
      using T = CloudGeometry;
      const int code = pi(PLEV,p);
      if (pi(PSP,p) != sp || T::CloudLobe1(code) || T::CloudLobe2(code) || T::CloudLobe3(code)) {
        return;
      }
      const int l = T::CloudLevel(code);
      if (l == 0 && pr(IPM,p) > a) a = pr(IPM,p);
      if (l == 1 && pr(IPM,p) > b) b = pr(IPM,p);
    }, Kokkos::Max<Real>(m0), Kokkos::Max<Real>(m1));
#if MPI_PARALLEL_ENABLED
    Real mm[2] = {m0, m1};
    MPI_Allreduce(MPI_IN_PLACE, mm, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
    m0 = mm[0]; m1 = mm[1];
#endif
    // (Kokkos::Max starts from the lowest number: without a plain level-0 / level-1 cloud of the
    // species both stay negative)
    out[sp] = (m0 > 0.0) ? m0 : ((m1 > 0.0) ? 8.0*m1 : 0.0);
  }
  return out;
}

// Center rule (test_imex_residual_to_center_upto_nominal): imex_qcen = the coupled mass
// density of the clouds centered in each leaf
// cell (the leaf that holds the center in the deposit's frame; halo slots back to their owners).
// Floor mode also keeps every cloud's center leaf for the kick (imex_leaf).  A nominal cloud of
// level l has the mass m_root/8^l, i.e. the mass density rho_nom = m_root/V_root in a leaf cell of
// its level, the same on every level.  The centered clouds take the share s = min(1, qt/B) of the
// cell's dust residual, B = the largest c_p rho_nom(species) among them: the size of each center
// kick, |c_p s r Q/qt|, is then at most |r Q|/rho_nom of its own species, for any mix of species
// and stopping times (a bound on the center part, not a stability theorem).  One species with one
// fixed stopping time: B = c_p rho_nom is the same for every cloud (imex_bsingle, no second
// deposit); otherwise imex_qcen component 1 holds B (atomic maximum).  FloorShare below turns
// (qt, B) into s in both places that need it (the gather's mark and the consistency step).
void DustGasDrag::CloudCenterMass3D(Driver *pdrive, int stage) {
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const Real a_dt = DragStep(pdrive);
  const Real sh = ImplicitShearShift(pdrive, stage);
  const auto lt = leaf_traversal3d;
  const bool cfloor = imex_resid_center_floor;
  if (cfloor) {
    // The reference (root) cloud mass of every species: the value of the merge and resampling
    // passes once they have one (cloud_resample_m0, or cloud_merge_mroot; never written from
    // here), until then an own estimate made once (ImexRootMassEstimate).  A species without any
    // gets a negative entry: a cloud of it sets B of its center cell to the largest number, so
    // that cell sends (practically) its whole dust share by the overlap weights.
    const int nsp = std::max(1, nspecies);
    std::vector<Real> want(nsp, 0.0);
    bool missing = false;
    for (int sp=0; sp<nsp; ++sp) {
      want[sp] = (cloud_resample_m0 > 0.0) ? cloud_resample_m0
               : ((static_cast<int>(cloud_merge_mroot.size()) > sp) ? cloud_merge_mroot[sp] : 0.0);
      if (!(want[sp] > 0.0)) missing = true;
    }
    // (cloud_merge_mroot is the same on every rank, so this collective call is too).  The own
    // estimate is made once and written to the restart file (record imex_mown), so that a
    // restart does not estimate again from clouds that have split or left since.
    if (!imex_mown.empty() && static_cast<int>(imex_mown.size()) != nsp) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "the restart record imex_mown holds " << imex_mown.size() << " species, the run "
                << nsp << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (missing && imex_mown.empty()) imex_mown = ImexRootMassEstimate(nsp);
    std::vector<int> own(nsp, 0);
    for (int sp=0; sp<nsp; ++sp) {
      if (!(want[sp] > 0.0) && !imex_mown.empty()) {want[sp] = imex_mown[sp]; own[sp] = 1;}
    }
    if (want != imex_mref) {
      imex_mref = want;
      if (imex_bnom.extent_int(0) != nsp) Kokkos::realloc(imex_bnom, nsp);
      auto bh = Kokkos::create_mirror_view(imex_bnom);
      const auto &ms = pmy_pack->pmesh->mesh_size;
      const Real vroot = ms.dx1*ms.dx2*ms.dx3;
      for (int sp=0; sp<nsp; ++sp) {
        bh(sp) = (want[sp] > 0.0) ? want[sp]/vroot : -1.0;   // (negative: no reference)
        if (global_variable::my_rank == 0) {
          std::cout << "# NOTE (dust): test_imex_residual_to_center_upto_nominal: species " << sp
                    << " root cloud mass " << want[sp]
                    << ((want[sp] > 0.0) ? (own[sp] ? (imex_mown_restored ? " (own estimate, from "
                        "the restart file)" : " (own estimate)") : " (merge / resampling value)")
                        : " (none: its center cells use the overlap weights)")
                    << ", nominal center mass density " << bh(sp) << std::endl;
        }
      }
      Kokkos::deep_copy(imex_bnom, bh);
      imex_rho_nom = bh(0);
      // one species with one fixed stopping time: every cloud has the same c_p and nominal
      // density, so B is the same for every cell; otherwise the center pass also keeps the
      // largest c_p rho_nom of the centered clouds (imex_qcen component 1)
      imex_floor_multi = nsp > 1 || stopping_time_mode != DustStoppingTimeMode::species_fixed;
    }
  }
  const bool multi = cfloor && imex_floor_multi;
  // B of every cell on the one-species path: c_p rho_nom (the largest number without a root mass)
  if (cfloor) {
    imex_bsingle = (imex_rho_nom > 0.0) ? (a_dt/(taus_min + a_dt))*imex_rho_nom
                                        : std::numeric_limits<Real>::max();
  }
  const int nqc = multi ? 2 : 1;
  if (imex_qcen.extent(0) != qdep.extent(0) || imex_qcen.extent(2) != qdep.extent(2) ||
      imex_qcen.extent(3) != qdep.extent(3) || imex_qcen.extent(4) != qdep.extent(4) ||
      imex_qcen.extent_int(1) != nqc) {
    Kokkos::realloc(imex_qcen, qdep.extent(0), nqc, qdep.extent(2), qdep.extent(3), qdep.extent(4));
  }
  Kokkos::deep_copy(DevExeSpace(), imex_qcen, 0.0);
  UpdateShear3D(pmy_pack->pmesh->time);
  auto qc = imex_qcen;
  const std::size_t sc = qc.stride(1);
  auto cbody = KOKKOS_LAMBDA(const int p) {
    if (pr(IPM,p) <= 0.0) return;
    const int leaf = lt.LeafOwner(pr(IPX,p), pr(IPY,p) + sh*pr(IPX,p), pr(IPZ,p),
                                  lt.Anchor(pi(PGID,p)));
    if (leaf < 0) return;
    int b, k, j, i;
    lt.Decode(leaf, b, k, j, i);
    Real v[1] = {pr(IPM,p)*(a_dt/(pr(IPTS,p) + a_dt))/(lt.box(b,3)*lt.box(b,4)*lt.box(b,5))};
    DepositAddWarp(&qc(b,0,k,j,i), sc, leaf, v);
  };
  if (cfloor) {
    if (imex_leaf.extent_int(0) < npart) {   // (capacity in 64 bits, as bvals_part.cpp)
      const long long cap = static_cast<long long>(npart) + npart/8 + 1;
      Kokkos::realloc(imex_leaf, static_cast<int>(std::min<long long>(
                                     cap, std::numeric_limits<int>::max())));
    }
    auto lf = imex_leaf;
    auto bn = imex_bnom;
    const int nspm1 = imex_bnom.extent_int(0) - 1;
    const Real bmax = std::numeric_limits<Real>::max();
    auto cbody_f = KOKKOS_LAMBDA(const int p) {   // cbody, and the leaf kept for the kick
      if (pr(IPM,p) <= 0.0) {lf(p) = -1; return;}
      const int leaf = lt.LeafOwner(pr(IPX,p), pr(IPY,p) + sh*pr(IPX,p), pr(IPZ,p),
                                    lt.Anchor(pi(PGID,p)));
      lf(p) = leaf;
      if (leaf < 0) return;
      int b, k, j, i;
      lt.Decode(leaf, b, k, j, i);
      const Real cp = a_dt/(pr(IPTS,p) + a_dt);
      Real v[1] = {pr(IPM,p)*cp/(lt.box(b,3)*lt.box(b,4)*lt.box(b,5))};
      DepositAddWarp(&qc(b,0,k,j,i), sc, leaf, v);
      if (multi) {
        // the coupled nominal density of this cloud: with it alone at the center and at its
        // nominal mass, qt would be this value (the floor compares qt with the largest one)
        const int sp = pi(PSP,p) < 0 ? 0 : (pi(PSP,p) > nspm1 ? nspm1 : pi(PSP,p));
        Kokkos::atomic_max(&qc(b,1,k,j,i), (bn(sp) > 0.0) ? cp*bn(sp) : bmax);
      }
    };
    if (const int nord = CloudOrderLength(npart)) {
      CloudParForOrdered("dust_imex_center_mass_floor", npart, cloud_ord, nord, cbody_f);
    } else {
      CloudParFor("dust_imex_center_mass_floor_storage", npart, cbody_f);
    }
  } else if (const int nord = CloudOrderLength(npart)) {
    CloudParForOrdered("dust_imex_center_mass", npart, cloud_ord, nord, cbody);
  } else {
    CloudParFor("dust_imex_center_mass_storage", npart, cbody);
  }
  HaloReverseAdd(imex_qcen, 0, 1);
  if (multi) HaloReverseMax(imex_qcen, 1, 1);
}


// Bound-clump NaN fix: the local implicit solve is lumped, u*_c = (rho u + P)/(rho + Q)
// per cell, while each cloud couples to several cells (deposit W, gather from neighboring u*).  Giving
// the gas the particles' actual impulse D_c (as PMBR alone does) leaves the gas with D_c - G_c, G_c =
// rho (u*_c - u_c) the lumped change, which the light gas of a dust-dominated cell feels Q/rho times
// amplified: the gas there ran away to tens of c_s in a collapsing clump (epsilon ~ 2e4).  Here the gas
// receives G_c plus its mass share rho/(rho + Q) of the residual, and the clouds of the cell take the
// rest: r_c = (D_c - G_c)/(rho + Q_c), gas += G_c + rho r_c, v_p += c_p sum_c W_cp r_c.  Total momentum
// is exact; a gas-dominated cell (Q << rho) is unchanged.  dmom becomes the applied gas rate.
void DustGasDrag::CloudImplicitConsistency3D(Driver *pdrive, int stage) {
  // the overlap split (center rule off): the function of both traversals
  if (!imex_resid_center) {CloudImplicitConsistencyOverlapT(leaf_traversal3d, pdrive, stage); return;}
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je, ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0 = pmy_pack->phydro->u0;
  const Real a_dt = DragStep(pdrive);
  const Real inv_adt = 1.0/((pdrive->a_impl)*(pmy_pack->pmesh->dt));
  const int nres = (imex_resid_center && imex_resid_center_floor) ? 6 : 3;
  if (imex_resid.extent(0) != ustar.extent(0) || imex_resid.extent(2) != ustar.extent(2) ||
      imex_resid.extent(3) != ustar.extent(3) || imex_resid.extent(4) != ustar.extent(4) ||
      imex_resid.extent_int(1) != nres) {
    Kokkos::realloc(imex_resid, ustar.extent(0), nres, ustar.extent(2), ustar.extent(3), ustar.extent(4));
  }
  Kokkos::deep_copy(DevExeSpace(), imex_resid, 0.0);
  auto res = imex_resid;
  auto us = ustar;
  auto qp = qdep;
  auto dm = dmom;
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const Real sh = ImplicitShearShift(pdrive, stage);
  const auto lt = leaf_traversal3d;
  // center rule: the coupled mass density of the clouds centered in each leaf cell
  // (CloudCenterMass3D)
  const bool center = imex_resid_center;
  // floor: the centered clouds take the share min(1, n) of the cell's dust
  // residual, n = their mass in nominal clouds of the leaf's level; the rest goes by the default
  // overlap weights, and so does the whole dust share of a cell that is no cloud's center (n = 0).
  // In floor mode the gather has already run the center pass of this stage.
  const bool cfloor = center && imex_resid_center_floor;
  if (center && !cfloor) CloudCenterMass3D(pdrive, stage);
  const Real bs = imex_bsingle;
  const bool fmulti = cfloor && imex_floor_multi;
  auto qc = imex_qcen;
  par_for("dust_imex_consistent_gas", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real rho = u0(m,IDN,k,j,i);
    const Real q = qp(m,0,k,j,i);
    const Real inv = 1.0/(rho + q);
    const Real qt = center ? qc(m,0,k,j,i) : 0.0;
    // the centered clouds' share of the dust residual (1 without the floor)
    // (the gather's mark uses the same call: unmarked = share 1 = no overlap part)
    const Real scen = cfloor ? FloorShare(qt, fmulti ? qc(m,1,k,j,i) : bs) : 1.0;
    for (int c=0; c<3; ++c) {
      const Real g = rho*us(m,c,k,j,i) - u0(m,IM1+c,k,j,i);
      const Real r = (dm(m,c,k,j,i) - g)*inv;
      Real dgas = g + rho*r;
      Real rd = r;
      if (center) {
        // the dust share r q V of the cell to the clouds centered in it, per unit of their coupled
        // mass; a cell that is no cloud's center gives the whole residual to the gas (point path)
        if (qt > 0.0) {
          rd = cfloor ? scen*r*q/qt : r*q/qt;
        } else if (cfloor) {
          rd = 0.0;   // floor mode: the whole dust share by the overlap weights (below), the limit
                      // of a cell whose centered mass goes to zero; the gas keeps its mass share
        } else {
          dgas = dm(m,c,k,j,i);
          rd = 0.0;
        }
      }
      u0(m,IM1+c,k,j,i) += dgas;
      dm(m,c,k,j,i) = dgas*inv_adt;
      res(m,c,k,j,i) = rd;
      // the rest of the dust share, per unit coupled mass of every cloud over the cell
      if (cfloor) res(m,3+c,k,j,i) = (qt > 0.0) ? (1.0 - scen)*r : r;
    }
  });
  // the residual velocity of each cell back to its clouds, with the deposit's weights and frame
  UpdateShear3D(pmy_pack->pmesh->time);
  if (cfloor) {
    // components 0-2: the centered part, 3-5: the overlap part (PackGatherField3D's slots)
    const int nmb = pmy_pack->nmb_thispack, nk = res.extent_int(2);
    const int nj = res.extent_int(3), ni = res.extent_int(4);
    const int nslot = std::max(std::max(nmb, res.extent_int(0)), leaf_traversal3d.nfld);
    if (gpack3.extent_int(0) != nslot || gpack3.extent_int(1) != nk ||
        gpack3.extent_int(2) != nj || gpack3.extent_int(3) != ni) {
      Kokkos::realloc(gpack3, nslot, nk, nj, ni, 12);
    }
    auto gf = gpack3;
    par_for("dust_imex_resid_pack_floor", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
      for (int c=0; c<6; ++c) gf(m,k,j,i,c) = res(m,c,k,j,i);
    });
    HaloForwardPack(gpack3, 6);   // the owners' values on the halo blocks
    // with the gather blend, its kernel has marked the clouds that cover a cell with an overlap
    // share (imex_need); the others only read their center cell (imex_leaf, the center pass)
    // (also the plain kernel with the marks: dc1 in floor mode)
    const bool skip = imex_gather_blend || (cloud_dc1 && imex_consistent_gas);
    auto nd = imex_need;
    auto lf = imex_leaf;
    auto body_f = KOKKOS_LAMBDA(const int p) {
      if (pr(IPM,p) <= 0.0) return;
      Real rv[3] = {0.0, 0.0, 0.0};
      if (!skip || nd(p) != 0) {
        lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p) + sh*pr(IPX,p), pr(IPZ,p), pi(PLEV,p),
                              lt.Anchor(pi(PGID,p)),
        [&](const int target, const Real w) {
          int b, k, j, i;
          lt.Decode(target, b, k, j, i);
          for (int c=0; c<3; ++c) rv[c] += w*gf(b,k,j,i,3+c);
        });
      }
      const int leaf = lf(p);
      if (leaf >= 0) {
        int b, k, j, i;
        lt.Decode(leaf, b, k, j, i);
        for (int c=0; c<3; ++c) rv[c] += gf(b,k,j,i,c);
      }
      const Real cj = a_dt/(pr(IPTS,p) + a_dt);
      pr(IPVX,p) += cj*rv[0]; pr(IPVY,p) += cj*rv[1]; pr(IPVZ,p) += cj*rv[2];
      pr(IPRX,p) += cj*rv[0]/a_dt; pr(IPRY,p) += cj*rv[1]/a_dt; pr(IPRZ,p) += cj*rv[2]/a_dt;
    };
    if (const int nord = CloudOrderLength(npart)) {
      CloudParForOrdered("dust_imex_consistent_dust_floor", npart, cloud_ord, nord, body_f);
    } else {
      CloudParFor("dust_imex_consistent_dust_floor_storage", npart, body_f);
    }
    // test_cloud_dc1_opcheck: the overlap part of every skipped cloud by the full gather (it
    // must be exactly 0), and the share of the clouds that need the overlap gather
    if (cloud_dc1_opcheck && skip) CloudDcSkipCheck(pdrive, stage);
    return;
  }
  PackGatherField3D(imex_resid);   // owners' values on the halo blocks
  auto gp = gpack3;
  auto body = KOKKOS_LAMBDA(const int p) {
    if (pr(IPM,p) <= 0.0) return;
    Real rv[3] = {0.0, 0.0, 0.0};
    if (center) {
      // the same leaf as in the center-mass pass (positions unchanged in between)
      const int leaf = lt.LeafOwner(pr(IPX,p), pr(IPY,p) + sh*pr(IPX,p), pr(IPZ,p),
                                    lt.Anchor(pi(PGID,p)));
      if (leaf < 0) return;
      int b, k, j, i;
      lt.Decode(leaf, b, k, j, i);
      for (int c=0; c<3; ++c) rv[c] = gp(b,k,j,i,c);
    } else {
      lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p) + sh*pr(IPX,p), pr(IPZ,p), pi(PLEV,p),
                            lt.Anchor(pi(PGID,p)),
      [&](const int target, const Real w) {
        int b, k, j, i;
        lt.Decode(target, b, k, j, i);
        for (int c=0; c<3; ++c) rv[c] += w*gp(b,k,j,i,c);
      });
    }
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


//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudImplicitConsistencyOverlapT
//! \brief The consistent gas / dust split of the IMEX residual with the overlap weights on either
//! traversal (CloudImplicitConsistency2D and the overlap branch of
//! CloudImplicitConsistency3D, center rule off, written once): with G_c = rho (u*_c - u_c) the lumped
//! change and D_c the kick's impulse deposit, r_c = (D_c - G_c)/(rho + Q_c); the gas takes
//! G_c + rho r_c, every cloud v_p += c_p sum_c W_cp r_c with the weights of the Q deposit; the rates
//! IPR* take dv / h as well, dmom the applied gas rate.  test_cloud_dc1_opcheck: the two closure
//! checks (DC1_SHARECHECK).

template <class LTT>
void DustGasDrag::CloudImplicitConsistencyOverlapT(const LTT &lt_in, Driver *pdrive, int stage) {
  using A = LeafApi<LTT>;
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
  const int ks = indcs.ks, ke = indcs.ke;
  const int nmb1 = pmy_pack->nmb_thispack - 1;
  auto &u0 = pmy_pack->phydro->u0;
  const Real a_dt = DragStep(pdrive);
  const Real inv_adt = 1.0/((pdrive->a_impl)*(pmy_pack->pmesh->dt));
  Real sh = 0.0;
  if constexpr (A::kDim == 3) {
    // the deposit's frame: the gas in its t^n orbital-advection frame
    sh = ImplicitShearShift(pdrive, stage);
    UpdateShear3D(pmy_pack->pmesh->time);
  }
  const LTT lt = lt_in;
  EnsureCloudDcField(cloud_res_ovl, lt.nfld);
  Kokkos::deep_copy(DevExeSpace(), cloud_res_ovl, 0.0);
  auto res = cloud_res_ovl;
  auto us = ustar;
  auto qp = qdep;
  auto dm = dmom;
  const bool check = cloud_dc1_opcheck;
  // (check) the cell closure max |dM_g + Q r - D| over max |D|, and sum V Q r of the owners' cells,
  // from the old values, before they are changed below
  Real cl_cell = 0.0, dmax = 0.0, sq[3] = {0.0, 0.0, 0.0};
  if (check) {
    const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    const int ncells = pmy_pack->nmb_thispack*nx3*nx2*nx1;
    Kokkos::parallel_reduce("dust_cloud_split_check", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
    KOKKOS_LAMBDA(const int idx, Real &a, Real &b, Real &s0, Real &s1, Real &s2) {
      const int i = idx % nx1 + is;
      const int j = (idx/nx1) % nx2 + js;
      const int k = (idx/(nx1*nx2)) % nx3 + ks;
      const int m = idx/(nx1*nx2*nx3);
      const Real rho = u0(m,IDN,k,j,i), q = qp(m,0,k,j,i);
      const Real inv = 1.0/(rho + q);
      const Real vol = A::Measure(lt, m);
      Real s[3];
      for (int c=0; c<3; ++c) {
        const Real g = rho*us(m,c,k,j,i) - u0(m,IM1+c,k,j,i);
        const Real r = (dm(m,c,k,j,i) - g)*inv;
        const Real dgas = g + rho*r;
        a = fmax(a, fabs(dgas + q*r - dm(m,c,k,j,i)));
        b = fmax(b, fabs(dm(m,c,k,j,i)));
        s[c] = vol*q*r;
      }
      s0 += s[0]; s1 += s[1]; s2 += s[2];
    }, Kokkos::Max<Real>(cl_cell), Kokkos::Max<Real>(dmax), Kokkos::Sum<Real>(sq[0]),
       Kokkos::Sum<Real>(sq[1]), Kokkos::Sum<Real>(sq[2]));
  }
  // the gas: G_c + rho r_c; r_c to the residual field (from the old values first)
  par_for(A::kDim == 3 ? "dust_imex_consistent_gas" : "dust_cloud2d_consistent_gas", DevExeSpace(), 0, nmb1, ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
    const Real rho = u0(m,IDN,k,j,i);
    const Real q = qp(m,0,k,j,i);
    const Real inv = 1.0/(rho + q);
    for (int c=0; c<3; ++c) {
      const Real g = rho*us(m,c,k,j,i) - u0(m,IM1+c,k,j,i);
      const Real r = (dm(m,c,k,j,i) - g)*inv;
      const Real dgas = g + rho*r;
      u0(m,IM1+c,k,j,i) += dgas;
      dm(m,c,k,j,i) = dgas*inv_adt;
      res(m,c,k,j,i) = r;
    }
  });
  HaloForward5D(cloud_res_ovl, 0, 3);   // the owners' values on the halo slots
  // the clouds: v_p += c_p sum_c W_cp r_c with the weights of the Q deposit
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  auto body = KOKKOS_LAMBDA(const int p, Real &s0, Real &s1, Real &s2) {
    if (pr(IPM,p) <= 0.0) return;
    Real rv[3] = {0.0, 0.0, 0.0};
    Real x[3];
    A::LoadImplicit(pr, p, sh, x);
    A::ForEachWeight(lt, x, pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int target, const Real w) {
      int b, k, j, i;
      A::Decode(lt, target, b, k, j, i);
      for (int c=0; c<3; ++c) rv[c] += w*res(b,c,k,j,i);
    });
    const Real cj = a_dt/(pr(IPTS,p) + a_dt);
    pr(IPVX,p) += cj*rv[0]; pr(IPVY,p) += cj*rv[1]; pr(IPVZ,p) += cj*rv[2];
    // the drag rate of this stage (consumed by the next working stage) includes the correction
    pr(IPRX,p) += cj*rv[0]/a_dt; pr(IPRY,p) += cj*rv[1]/a_dt; pr(IPRZ,p) += cj*rv[2]/a_dt;
    const Real mp = pr(IPM,p);
    s0 += mp*cj*rv[0]; s1 += mp*cj*rv[1]; s2 += mp*cj*rv[2];
  };
  if (!check) {
    auto body1 = KOKKOS_LAMBDA(const int p) {Real a = 0.0, b = 0.0, c = 0.0; body(p, a, b, c);};
    if (const int nord = CloudOrderLength(npart)) {
      CloudParForOrdered(A::kDim == 3 ? "dust_imex_consistent_dust" : "dust_cloud2d_consistent_dust", npart, cloud_ord, nord, body1);
    } else {
      CloudParFor(A::kDim == 3 ? "dust_imex_consistent_dust_storage" : "dust_cloud2d_consistent_dust_storage", npart, body1);
    }
    return;
  }
  Real sp[3] = {0.0, 0.0, 0.0};
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_cloud_consistent_dust_check",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, npart), body,
                            Kokkos::Sum<Real>(sp[0]), Kokkos::Sum<Real>(sp[1]),
                            Kokkos::Sum<Real>(sp[2]));
  }
  Real mx[2] = {cl_cell, dmax};
  Real sm[6] = {sp[0], sp[1], sp[2], sq[0], sq[1], sq[2]};
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, mx, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, sm, 6, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank == 0) {
    Real worst = 0.0, scale = 0.0;
    for (int c=0; c<3; ++c) {
      worst = std::max(worst, std::abs(sm[c] - sm[3+c]));
      scale = std::max(scale, std::abs(sm[3+c]));
    }
    std::cout << std::setprecision(6) << "# DC1_SHARECHECK cycle " << pmy_pack->pmesh->ncycle
              << " stage " << stage << " | cell: max |dM_g + Q r - D| " << mx[0] << " (|D| "
              << mx[1] << ") | clouds' momentum - sum V Q r: " << worst << " (|sum V Q r| "
              << scale << "; x " << sm[0] << " vs " << sm[3] << ")" << std::endl;
  }
}
template void DustGasDrag::CloudImplicitConsistencyOverlapT<LeafTraversal>(const LeafTraversal &,
                                                                           Driver *, int);
template void DustGasDrag::CloudImplicitConsistencyOverlapT<LeafTraversal3D>(
    const LeafTraversal3D &, Driver *, int);

}  // namespace dust
