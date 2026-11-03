//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_split_velocity.cpp
//! \brief <dust>/cloud_split_velocity = mesh_gradient: the pieces of a device P1 split or lobe
//! release get the parent's velocity plus the mean-flow velocity gradient of the deposited dust
//! times their centroid offset, where the deposit is single-stream; elsewhere (and with the
//! default cloud_split_velocity = parent) they keep the parent's velocity.
//!
//! On a cycle where some rank splits:
//!  - DepositSplitMoments(3D): per species, on the actual leaves with the clouds' own weights,
//!    rho, rho dX, rho dX dX^T, rho v, rho |v|^2 (dX = X - x_c, X the cloud's centroid, x_c the
//!    cell centre, the cloud framed next to the cell's block; v relative to the species'
//!    mass-weighted mean velocity, so that the second moment keeps small dispersions about a
//!    drift common to the species; it does not remove the cancellation where the local velocity
//!    differs strongly from that mean: the round-off of the excess below is then about
//!    eps |v - v_ref|^2, against c^2 V_G^2); several ranks: halo reverse-add and forward.  For
//!    any linear field carried by the clouds the cell means U = (rho v)/rho and
//!    X = x_c + (rho dX)/rho obey U = u0 + G X exactly, whatever the
//!    density, the cloud levels and lobes, and the velocity variance within a cell is exactly
//!    G V_c G^T (V_c the covariance of the cloud centroids that deposit into the cell).
//!  - SplitVelocityGradient(3D), one thread per event with two or more pieces: weighted least
//!    squares of U against X over the parent's support (weights: the parent's own leaf weights
//!    times rho), G = B A^-1.  Single-stream test: the velocity variance that the fitted linear
//!    field does not account for (total - between-cell explained - within-cell linear part
//!    G Vbar G^T), sigma_x^2, must not exceed cloud_split_dispersion_max^2 V_G^2, V_G^2 =
//!    sum_ij (G_ij e_j)^2 the velocity change of G over the parent's half-widths e; sigma_x is
//!    zero (round-off) for a single-stream linear field on any mesh, about the stream spread for
//!    several streams.  |sigma_x^2| <= c^2 V_G^2 is a conservative screen, not a strict
//!    single-stream test: it accepts several streams whose spread is small against V_G (or whose
//!    between- and within-cell parts cancel in the sum), and it rejects a single stream whose
//!    field is curved on the parent's scale.  Where the populated cells of the support do not
//!    spread over every direction (a lobed cloud over one layer of coarser cells), the fit is
//!    repeated on the plain support of the next coarser level (within the halo: the supports are
//!    checked by CheckSplitVelocityReach before the split); if that is ill-conditioned too, the
//!    parent's velocity.  G (zero on a fallback) goes to split_vgrad, row tagoff(p).
//!  - the fill kernels of the splits add G d to each piece (SplitAddGradient): d is the
//!    piece's centroid offset from the parent's centroid, and sum_k m_k d_k = 0 for the P1
//!    pieces, so sum_k m_k G d_k = 0 and the pieces carry the parent's momentum in an unbounded
//!    (periodic) domain.  At a reflecting x3 face this holds for the tangential components only:
//!    a piece mirrored there has its v_z, the added (G d)_z included, reversed by the fill (the
//!    wall's impulse, -2 m_k v_z,k, as in the An split), so the normal momentum of the pieces
//!    differs from the parent's by that impulse.
//!  - the lobe releases inside the merge pass (G) keep the parent's velocity (An); so does the
//!    host split, which mesh_gradient does not support (fatal).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"
#include "dust_split_velocity.hpp"

namespace dust {

namespace {

// 2-D: [0] rho, [1..2] rho dX, [3..5] rho dX dX (xx yy xy), [6..8] rho v, [9] rho |v|^2
constexpr int kNC2 = 10;
// 3-D: [0] rho, [1..3] rho dX, [4..9] rho dX dX (xx yy zz xy xz yz), [10..12] rho v, [13] rho |v|^2
constexpr int kNC3 = 14;
// split_vgrad row: G (9), v_parent (3), mass, V_G (applied, else 0); diagnostics (check key):
// [14] ratio sigma_x/V_G, [15] status, [16] cells, [17] sigma_x^2, [18] G Vbar G^T, [19] V_G^2,
// [20..22] the parent's position, [23] its code, [24] its anchor block slot
constexpr int kRow = 25;

bool AnyRankSV(const bool v) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    int x = v ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return x != 0;
  }
#endif
  return v;
}

// the weighted moments of one event's support (velocities relative to the parent's)
struct SVMoments {
  Real s0 = 0.0, sx[3] = {0.0, 0.0, 0.0}, su[3] = {0.0, 0.0, 0.0};
  Real sxx[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // 00 11 22 01 02 12
  Real sux[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Real svc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};   // sum a V_c (00 11 22 01 02 12)
  Real svv = 0.0;   // sum w rho |v|^2
  KOKKOS_INLINE_FUNCTION
  void Add(const Real a, const Real x0, const Real x1, const Real x2, const Real u0,
           const Real u1, const Real u2, const Real wvv, const Real v[6]) {
    s0 += a;
    sx[0] += a*x0; sx[1] += a*x1; sx[2] += a*x2;
    su[0] += a*u0; su[1] += a*u1; su[2] += a*u2;
    sxx[0] += a*x0*x0; sxx[1] += a*x1*x1; sxx[2] += a*x2*x2;
    sxx[3] += a*x0*x1; sxx[4] += a*x0*x2; sxx[5] += a*x1*x2;
    sux[0] += a*u0*x0; sux[1] += a*u0*x1; sux[2] += a*u0*x2;
    sux[3] += a*u1*x0; sux[4] += a*u1*x1; sux[5] += a*u1*x2;
    sux[6] += a*u2*x0; sux[7] += a*u2*x1; sux[8] += a*u2*x2;
    for (int n=0; n<6; ++n) svc[n] += a*v[n];
    svv += wvv;
  }
};

// the fit: G (3 x D, zero columns beyond D) into g; returns 1 applied, 2 dispersion fallback,
// 3 ill-conditioned, 4 no gradient (uniform flow); vg = the velocity change of the fitted G over
// the half-widths e.  Velocities relative to the species' reference velocity (vp: the parent's).
KOKKOS_INLINE_FUNCTION
int SVFit(const SVMoments &q, const int ndim, const Real vpx, const Real vpy, const Real vpz,
          const Real e0, const Real e1, const Real e2, const Real c2, Real g[9], Real &vg,
          Real diag[3]) {
  for (int n=0; n<9; ++n) g[n] = 0.0;
  vg = 0.0;
  diag[0] = diag[1] = diag[2] = 0.0;
  if (!(q.s0 > 0.0)) return 3;
  const Real is0 = 1.0/q.s0;
  const Real mx[3] = {q.sx[0]*is0, q.sx[1]*is0, q.sx[2]*is0};
  const Real mu[3] = {q.su[0]*is0, q.su[1]*is0, q.su[2]*is0};
  // A = position covariance (symmetric), B = velocity-position covariance
  Real a00 = q.sxx[0]*is0 - mx[0]*mx[0], a11 = q.sxx[1]*is0 - mx[1]*mx[1];
  Real a22 = q.sxx[2]*is0 - mx[2]*mx[2], a01 = q.sxx[3]*is0 - mx[0]*mx[1];
  Real a02 = q.sxx[4]*is0 - mx[0]*mx[2], a12 = q.sxx[5]*is0 - mx[1]*mx[2];
  Real b[9];
  for (int i=0; i<3; ++i) {
    for (int j=0; j<3; ++j) b[3*i + j] = q.sux[3*i + j]*is0 - mu[i]*mx[j];
  }
  // the inverse of A (D x D)
  Real ai[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // the populated cells must spread over every direction (cell-mean centroids at least about
  // 0.03 e apart; a support on a single layer of coarse cells has no spread along it, and its G
  // would be round-off divided by round-off)
  const Real amin = 1.0e-3;
  if (ndim == 3) {
    if (!(a00 > amin*e0*e0 && a11 > amin*e1*e1 && a22 > amin*e2*e2)) return 3;
    const Real c00 = a11*a22 - a12*a12, c01 = a02*a12 - a01*a22, c02 = a01*a12 - a02*a11;
    const Real det = a00*c00 + a01*c01 + a02*c02;
    if (!(det > 1.0e-6*a00*a11*a22)) return 3;
    const Real id = 1.0/det;
    ai[0] = c00*id; ai[1] = c01*id; ai[2] = c02*id;
    ai[3] = c01*id; ai[4] = (a00*a22 - a02*a02)*id; ai[5] = (a01*a02 - a00*a12)*id;
    ai[6] = c02*id; ai[7] = ai[5]; ai[8] = (a00*a11 - a01*a01)*id;
  } else {
    if (!(a00 > amin*e0*e0 && a11 > amin*e1*e1)) return 3;
    const Real det = a00*a11 - a01*a01;
    if (!(det > 1.0e-6*a00*a11)) return 3;
    const Real id = 1.0/det;
    ai[0] = a11*id; ai[1] = -a01*id; ai[3] = -a01*id; ai[4] = a00*id;
  }
  // G = B A^-1; the explained variance tr(G B^T); V_G^2 = sum (G_ij e_j)^2
  const Real e[3] = {e0, e1, e2};
  Real expl = 0.0, vg2 = 0.0;
  for (int i=0; i<3; ++i) {
    for (int j=0; j<ndim; ++j) {
      Real s = 0.0;
      for (int k=0; k<ndim; ++k) s += b[3*i + k]*ai[3*k + j];
      g[3*i + j] = s;
      expl += s*b[3*i + j];
      vg2 += (s*e[j])*(s*e[j]);
    }
  }
  // total variance: sum w rho |v|^2 / sum a - |mean v|^2 (raw mean = mu + v_parent); the
  // within-cell variance of the fitted linear field, G Vbar G^T
  const Real ux = mu[0] + vpx, uy = mu[1] + vpy, uz = mu[2] + vpz;
  const Real tvar = q.svv*is0 - (ux*ux + uy*uy + uz*uz);
  const Real vb[9] = {q.svc[0]*is0, q.svc[3]*is0, q.svc[4]*is0,
                      q.svc[3]*is0, q.svc[1]*is0, q.svc[5]*is0,
                      q.svc[4]*is0, q.svc[5]*is0, q.svc[2]*is0};
  Real lin = 0.0;
  for (int i=0; i<3; ++i) {
    for (int j=0; j<ndim; ++j) {
      for (int k=0; k<ndim; ++k) lin += g[3*i + j]*vb[3*j + k]*g[3*i + k];
    }
  }
  const Real excess = tvar - expl - lin;
  vg = sqrt(vg2);
  diag[0] = excess; diag[1] = lin; diag[2] = vg2;
  // no gradient above round-off (a uniform flow): nothing to add
  if (vg2 <= 1.0e-24*q.svv*is0) {
    for (int n=0; n<9; ++n) g[n] = 0.0;
    vg = 0.0;
    return 4;
  }
  // a single-stream linear field has excess = 0 up to round-off; several streams make it
  // positive, a G that over-predicts the within-cell spread negative
  if (fabs(excess) <= c2*vg2) return 1;
  for (int n=0; n<9; ++n) g[n] = 0.0;
  return 2;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ReadSplitVelocityKeys
//! \brief The keys, read only when given (a run without them keeps the base parameter list and
//! restart files).

void DustGasDrag::ReadSplitVelocityKeys(ParameterInput *pin) {
  if (pin->DoesParameterExist("dust", "cloud_split_velocity")) {
    const std::string v = pin->GetString("dust", "cloud_split_velocity");
    if (v == "parent") {
      cloud_split_velocity = 0;
    } else if (v == "mesh_gradient") {
      cloud_split_velocity = 1;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dust>/cloud_split_velocity must be parent or mesh_gradient" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (pin->DoesParameterExist("dust", "cloud_split_dispersion_max")) {
    cloud_split_dispersion_max = pin->GetReal("dust", "cloud_split_dispersion_max");
  }
  if (pin->DoesParameterExist("dust", "cloud_split_velocity_check")) {
    cloud_split_velocity_check = pin->GetBoolean("dust", "cloud_split_velocity_check");
  }
  if (cloud_split_velocity == 1 && (!cloud_split_device || !(cloud_split_dispersion_max >= 0.0))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<dust>/cloud_split_velocity = mesh_gradient needs the device split "
              << "(cloud_split_device = true) and cloud_split_dispersion_max >= 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (cloud_split_velocity == 1 && global_variable::my_rank == 0) {
    std::cout << "# dust: cloud_split_velocity = mesh_gradient, cloud_split_dispersion_max = "
              << cloud_split_dispersion_max << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitVelocityPrepare
//! \brief Called by the device splits once the events are known (nrows: the rows of
//! split_vgrad, the pieces beyond the first of all events; 0 on the early returns).  Off:
//! false at once.  On: when any rank has rows, the moments are deposited (collective on several
//! ranks) and the gradient of every event with two or more pieces fitted.  Returns whether this
//! rank's fill adds the offsets.

bool DustGasDrag::SplitVelocityPrepare(const int nrows) {
  if (cloud_split_velocity == 0) return false;
  if (!AnyRankSV(nrows > 0)) return false;
  if (split_vel_cnt.extent_int(0) != 7) {
    split_vel_cnt = Kokkos::View<unsigned long long*, DevMemSpace>("dust_split_vel_cnt", 7);
  }
  // the reference velocity of the deposit (any value is exact; near the species' mean it keeps
  // the second moment's round-off small): refreshed every 64 deposits, at the same deposits on
  // every rank (all ranks deposit together)
  if (split_vel_deposits % 64 == 0 || split_vref.extent_int(0) != 3*nspecies) {
    SplitReferenceVelocity();
  }
  if (clouds3d) {
    DepositSplitMoments3D();
  } else {
    DepositSplitMoments();
  }
  ++split_vel_deposits;
  if (nrows == 0) return false;
  if (split_vgrad.extent_int(0) < nrows || split_vgrad.extent_int(1) != kRow) {
    split_vgrad = DvceArray2D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                       "dust_split_vgrad"), nrows + nrows/4, kRow);
  }
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  if (clouds3d) {
    SplitVelocityGradient3D(npart);
  } else {
    SplitVelocityGradient(npart);
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckSplitVelocityReach
//! \brief (mesh_gradient, several ranks) CheckHaloReach for every support the fit may read:
//! the cloud's actual support and, for a cloud of level >= 1, the plain support of the next
//! coarser level around its position, which the fit takes when the parent's own support is
//! ill-conditioned (SplitVelocityGradient(3D), attempt 1: twice the half-width on every side).
//! The event parents are not known before the split, so every cloud is checked.  A cloud at
//! most one level coarser than its block (the overlap rule) reaches at most 4 cells of the
//! block's level beyond it with the retry support, inside the default allowance (HaloReach:
//! 3 cells of level lev - 1 less one cell of lev); a coarser cloud (held back by
//! cloud_split_budget, the older halo rule, or one level behind right after a regrid) can reach
//! further, and then the halo is widened and the leaf traversal rebuilt as in CheckHaloReach.
//! Called from SplitCloudsP1 in place of CheckHaloReach("split velocity"), before the device
//! split copies the traversal, so the moments' deposit, its halo exchange and the fit all see
//! the widened halo (Codex review: the retry support could run past the halo, or read halo cells
//! whose moments had not been exchanged).  Returns whether the halo was rebuilt.

bool DustGasDrag::CheckSplitVelocityReach() {
  if (global_variable::nranks == 1) return false;
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  const int nmb = pmy_pack->nmb_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  auto allow = halo_allow_d_;
  DvceArray1D<std::int64_t> need("dust_halo_need_sv", std::max(nmb, 1));
  int nover = 0;
  if (npart > 0 && clouds3d) {
    const auto lt = leaf_traversal3d;
    Kokkos::parallel_reduce("dust_halo_reach_sv3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &cnt) {
      using T = SMRCompositeTransfer;
      const int m = lt.Anchor(pi(PGID,p));
      Real x = pr(IPX,p), y = pr(IPY,p), z = pr(IPZ,p);
      lt.Frame(m, x, y, z);
      const int code = pi(PLEV,p);
      const int lev = T::CloudLevel(code);
      // the retry support (plain, level lev - 1) contains the actual one (two half-widths of
      // lev at most on a lobed side)
      const Real e1 = (lev > 0) ? lt.CloudHalfWidth1(lev - 1)
                                : (T::CloudLobe1(code) ? 2.0 : 1.0)*lt.CloudHalfWidth1(lev);
      const Real e2 = (lev > 0) ? lt.CloudHalfWidth2(lev - 1)
                                : (T::CloudLobe2(code) ? 2.0 : 1.0)*lt.CloudHalfWidth2(lev);
      const Real e3 = (lev > 0) ? lt.CloudHalfWidth3(lev - 1)
                                : (T::CloudLobe3(code) ? 2.0 : 1.0)*lt.CloudHalfWidth3(lev);
      const Real bx0 = lt.box(m,0), bx1 = bx0 + lt.nx1*lt.box(m,3);
      const Real by0 = lt.box(m,1), by1 = by0 + lt.nx2*lt.box(m,4);
      const Real bz0 = lt.box(m,2), bz1 = bz0 + lt.nx3*lt.box(m,5);
      const Real o1 = Kokkos::fmax(0.0, Kokkos::fmax(bx0 - (x - e1), (x + e1) - bx1));
      const Real o2 = Kokkos::fmax(0.0, Kokkos::fmax(by0 - (y - e2), (y + e2) - by1));
      const Real o3 = Kokkos::fmax(0.0, Kokkos::fmax(bz0 - (z - e3), (z + e3) - bz1));
      const std::int64_t d = static_cast<std::int64_t>(
          Kokkos::fmax(Kokkos::ceil(o1/lt.fine_dx1),
                       Kokkos::fmax(Kokkos::ceil(o2/lt.fine_dx2), Kokkos::ceil(o3/lt.fine_dx3))));
      if (d > allow(m)) {
        const std::int64_t margin = static_cast<std::int64_t>(1) << (lt.nlev - lt.blev(m));
        Kokkos::atomic_max(&need(m), d + margin);
        ++cnt;
      }
    }, Kokkos::Sum<int>(nover));
  } else if (npart > 0) {
    const auto lt = leaf_traversal;
    Kokkos::parallel_reduce("dust_halo_reach_sv2d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &cnt) {
      using T = SMRCompositeTransfer;
      const int m = lt.Anchor(pi(PGID,p));
      Real x = pr(IPX,p), y = pr(IPY,p);
      lt.Frame(m, x, y);
      const int code = pi(PLEV,p);
      const int lev = T::CloudLevel(code);
      const Real e1 = (lev > 0) ? lt.CloudHalfWidth1(lev - 1)
                                : (T::CloudLobe1(code) ? 2.0 : 1.0)*lt.CloudHalfWidth1(lev);
      const Real e2 = (lev > 0) ? lt.CloudHalfWidth2(lev - 1)
                                : (T::CloudLobe2(code) ? 2.0 : 1.0)*lt.CloudHalfWidth2(lev);
      const Real bx0 = lt.box(m,0), bx1 = bx0 + lt.nx1*lt.box(m,2);
      const Real by0 = lt.box(m,1), by1 = by0 + lt.nx2*lt.box(m,3);
      const Real o1 = Kokkos::fmax(0.0, Kokkos::fmax(bx0 - (x - e1), (x + e1) - bx1));
      const Real o2 = Kokkos::fmax(0.0, Kokkos::fmax(by0 - (y - e2), (y + e2) - by1));
      const std::int64_t d = static_cast<std::int64_t>(Kokkos::fmax(Kokkos::ceil(o1/lt.fine_dx1),
                                                                     Kokkos::ceil(o2/lt.fine_dx2)));
      if (d > allow(m)) {
        const std::int64_t margin = static_cast<std::int64_t>(1) << (lt.nlev - lt.blev(m));
        Kokkos::atomic_max(&need(m), d + margin);
        ++cnt;
      }
    }, Kokkos::Sum<int>(nover));
  }
  long long tot = nover;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &tot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (tot == 0) return false;
  // the widening of CheckHaloReach (kept until the mesh changes), then every rank rebuilds
  auto nh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), need);
  int nwide = 0;
  for (int m=0; m<nmb; ++m) {
    if (nh(m) > halo_need_[m]) {halo_need_[m] = nh(m); ++nwide;}
  }
  halo_widenings_ += static_cast<unsigned long long>(nwide);
  split_vel_halo_rebuilds += 1;
  const int nhalo_before = clouds3d ? leaf_traversal3d.nfld - leaf_traversal3d.nmb
                                    : leaf_traversal.nfld - leaf_traversal.nmb;
  if (clouds3d) {
    BuildLeafTraversal3D();
  } else {
    BuildLeafTraversal();
  }
  if (global_variable::my_rank == 0) {
    std::cout << "# dust halo: " << tot << " clouds' split-velocity supports (incl. the coarser "
              << "retry support) reached beyond the halo (cycle " << pmy_pack->pmesh->ncycle
              << "); rank 0 widened " << nwide << " blocks, halo blocks " << nhalo_before
              << " -> " << (clouds3d ? leaf_traversal3d.nfld - leaf_traversal3d.nmb
                                     : leaf_traversal.nfld - leaf_traversal.nmb) << std::endl;
  }
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitReferenceVelocity
//! \brief The mass-weighted mean velocity of every species over all ranks (split_vref, 3 per
//! species): the deposited velocities are taken relative to it, so that the second moment does
//! not lose the small dispersions (LinA: 1e-9 against a drift of 1e-2) to round-off.

DvceArray1D<Real> DustGasDrag::SplitReferenceVelocity() {
  const int nsp = nspecies;
  if (split_vref.extent_int(0) != 3*nsp) split_vref = DvceArray1D<Real>("dust_split_vref", 3*nsp);
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  std::vector<double> sums(4*nsp, 0.0);
  for (int s=0; s<nsp; ++s) {
    double a = 0.0, b = 0.0, c = 0.0, d = 0.0;
    Kokkos::parallel_reduce("dust_split_vref", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, double &sa, double &sb, double &sc, double &sd) {
      if (pi(PSP,p) != s) return;
      const double m = pr(IPM,p);
      sa += m; sb += m*pr(IPVX,p); sc += m*pr(IPVY,p); sd += m*pr(IPVZ,p);
    }, a, b, c, d);
    sums[4*s] = a; sums[4*s+1] = b; sums[4*s+2] = c; sums[4*s+3] = d;
  }
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, sums.data(), 4*nsp, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  auto h = Kokkos::create_mirror_view(split_vref);
  for (int s=0; s<nsp; ++s) {
    const double m = (sums[4*s] > 0.0) ? sums[4*s] : 1.0;
    for (int c=0; c<3; ++c) h(3*s + c) = static_cast<Real>(sums[4*s + 1 + c]/m);
  }
  Kokkos::deep_copy(split_vref, h);
  return split_vref;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositSplitMoments3D / DepositSplitMoments
//! \brief The moments of every cloud on the actual leaves (see the file comment).

void DustGasDrag::DepositSplitMoments3D() {
  const int nc = kNC3*nspecies;
  // the slots the leaf traversal can reach (local + halo), not qdep's preallocated
  // max_nmb_per_rank: the array is zeroed at every deposit
  const int nslot = std::max(std::max(1, pmy_pack->nmb_thispack), leaf_traversal3d.nfld);
  if (split_mom.extent_int(0) != nslot || split_mom.extent_int(1) != nc ||
      split_mom.extent_int(2) != qdep.extent_int(2) || split_mom.extent_int(3) != qdep.extent_int(3) ||
      split_mom.extent_int(4) != qdep.extent_int(4)) {
    split_mom = DvceArray5D<Real>("dust_split_mom", nslot, nc, qdep.extent(2), qdep.extent(3),
                                  qdep.extent(4));
  } else {
    Kokkos::deep_copy(split_mom, 0.0);
  }
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const auto lt = leaf_traversal3d;
  auto f = split_mom;
  const int is = lt.is, js = lt.js, ks = lt.ks;
  auto vr = split_vref;
  const int nsp = nspecies;
  const std::size_t cstride = f.stride(1);   // between two components of one cell
  // in the cell order of the cloud kernels where there is an index (lanes of a warp that add to
  // the same cell sum in registers first, DepositAddWarp)
  const int nidx = (cloud_ord_n > 0 && (cloud_ord_n <= npart || cloud_order_keep)) ? cloud_ord_n : 0;
  auto body = KOKKOS_LAMBDA(const int p) {
    using T = SMRCompositeTransfer;
    const int sp = pi(PSP,p);
    const int code = pi(PLEV,p), lev = T::CloudLevel(code), c0 = kNC3*sp;
    const Real m = pr(IPM,p), vx = pr(IPVX,p) - vr(3*sp), vy = pr(IPVY,p) - vr(3*sp+1);
    const Real vz = pr(IPVZ,p) - vr(3*sp+2);
    const Real v2 = vx*vx + vy*vy + vz*vz;
    const Real x = pr(IPX,p), y = pr(IPY,p), z = pr(IPZ,p);
    const Real xc = x + SplitLobeShift(T::CloudLobe1(code), lt.CloudHalfWidth1(lev));
    const Real yc = y + SplitLobeShift(T::CloudLobe2(code), lt.CloudHalfWidth2(lev));
    const Real zc = z + SplitLobeShift(T::CloudLobe3(code), lt.CloudHalfWidth3(lev));
    lt.ForEachCloudWeight(x, y, z, code, lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w) {
      if (w == 0.0) return;
      int tb, tk, tj, ti;
      lt.Decode(target, tb, tk, tj, ti);
      const Real d = m*w/(lt.box(tb,3)*lt.box(tb,4)*lt.box(tb,5));
      Real xq = xc, yq = yc, zq = zc;
      lt.Frame(tb, xq, yq, zq);
      const Real dx = xq - (lt.box(tb,0) + (ti - is + 0.5)*lt.box(tb,3));
      const Real dy = yq - (lt.box(tb,1) + (tj - js + 0.5)*lt.box(tb,4));
      const Real dz = zq - (lt.box(tb,2) + (tk - ks + 0.5)*lt.box(tb,5));
      Real q[kNC3] = {d, d*dx, d*dy, d*dz, d*dx*dx, d*dy*dy, d*dz*dz, d*dx*dy, d*dx*dz,
                      d*dy*dz, d*vx, d*vy, d*vz, d*v2};
      Real *dst = &f(tb,c0,tk,tj,ti);
      DepositAddWarp<kNC3>(dst, cstride, target*nsp + sp, q);
    });
  };
  if (nidx > 0) {
    CloudParForOrdered("dust_split_moments3d", npart, cloud_ord, nidx, body);
  } else {
    CloudParFor("dust_split_moments3d", npart, body);
  }
  // several ranks: the owners' sums, then the owners' values on the halo slots
  HaloReverseAdd(split_mom, 0, nc);
  HaloForward5D(split_mom, 0, nc);
}

void DustGasDrag::DepositSplitMoments() {
  const int nc = kNC2*nspecies;
  const int nslot = std::max(std::max(1, pmy_pack->nmb_thispack), leaf_traversal.nfld);
  if (split_mom.extent_int(0) != nslot || split_mom.extent_int(1) != nc ||
      split_mom.extent_int(2) != qdep.extent_int(2) || split_mom.extent_int(3) != qdep.extent_int(3) ||
      split_mom.extent_int(4) != qdep.extent_int(4)) {
    split_mom = DvceArray5D<Real>("dust_split_mom", nslot, nc, qdep.extent(2), qdep.extent(3),
                                  qdep.extent(4));
  } else {
    Kokkos::deep_copy(split_mom, 0.0);
  }
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const auto lt = leaf_traversal;
  auto f = split_mom;
  const int is = lt.is, js = lt.js, ks = pmy_pack->pmesh->mb_indcs.ks;
  const int nc1 = lt.ncells1, nc2 = lt.ncells2;
  auto vr = split_vref;
  const int nsp = nspecies;
  const std::size_t cstride = f.stride(1);
  const int nidx = (cloud_ord_n > 0 && (cloud_ord_n <= npart || cloud_order_keep)) ? cloud_ord_n : 0;
  auto body = KOKKOS_LAMBDA(const int p) {
    using T = SMRCompositeTransfer;
    const int sp = pi(PSP,p);
    const int code = pi(PLEV,p), lev = T::CloudLevel(code), c0 = kNC2*sp;
    const Real m = pr(IPM,p), vx = pr(IPVX,p) - vr(3*sp), vy = pr(IPVY,p) - vr(3*sp+1);
    const Real vz = pr(IPVZ,p) - vr(3*sp+2);
    const Real v2 = vx*vx + vy*vy + vz*vz;
    const Real x = pr(IPX,p), y = pr(IPY,p);
    const Real xc = x + SplitLobeShift(T::CloudLobe1(code), lt.CloudHalfWidth1(lev));
    const Real yc = y + SplitLobeShift(T::CloudLobe2(code), lt.CloudHalfWidth2(lev));
    lt.ForEachCloudWeight(x, y, code, lt.Anchor(pi(PGID,p)), [&](const int target, const Real w) {
      if (w == 0.0) return;
      const int ti = target % nc1, r = target/nc1, tj = r % nc2, tm = r/nc2;
      const Real d = m*w/(lt.box(tm,2)*lt.box(tm,3));
      Real xq = xc, yq = yc;
      lt.Frame(tm, xq, yq);
      const Real dx = xq - (lt.box(tm,0) + (ti - is + 0.5)*lt.box(tm,2));
      const Real dy = yq - (lt.box(tm,1) + (tj - js + 0.5)*lt.box(tm,3));
      Real q[kNC2] = {d, d*dx, d*dy, d*dx*dx, d*dy*dy, d*dx*dy, d*vx, d*vy, d*vz, d*v2};
      Real *dst = &f(tm,c0,ks,tj,ti);
      DepositAddWarp<kNC2>(dst, cstride, target*nsp + sp, q);
    });
  };
  if (nidx > 0) {
    CloudParForOrdered("dust_split_moments2d", npart, cloud_ord, nidx, body);
  } else {
    CloudParFor("dust_split_moments2d", npart, body);
  }
  HaloReverseAdd(split_mom, 0, nc);
  HaloForward5D(split_mom, 0, nc);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitVelocityGradient3D / SplitVelocityGradient
//! \brief The fit of every event with two or more pieces (row tagoff(p) of split_vgrad).

void DustGasDrag::SplitVelocityGradient3D(const int npart) {
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const auto lt = leaf_traversal3d;
  auto f = split_mom;
  auto g = split_vgrad;
  auto np = split_np;
  auto off = split_off;
  const int is = lt.is, js = lt.js, ks = lt.ks;
  const Real c2 = cloud_split_dispersion_max*cloud_split_dispersion_max;
  const bool chk = cloud_split_velocity_check;
  auto vr = split_vref;
  auto cnt = split_vel_cnt;   // device counters (no host synchronization per split cycle)
  // (check) the retry supports against the halo allowance of the anchor block (several ranks)
  const bool multi = global_variable::nranks > 1;
  auto allow = halo_allow_d_;
  Kokkos::parallel_for("dust_split_vgrad3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (np(p) < 2) return;
    using T = SMRCompositeTransfer;
    const int row = off(p);
    const int code = pi(PLEV,p), lev = T::CloudLevel(code), c0 = kNC3*pi(PSP,p);
    const Real e1 = lt.CloudHalfWidth1(lev), e2 = lt.CloudHalfWidth2(lev);
    const Real e3 = lt.CloudHalfWidth3(lev);
    const Real x = pr(IPX,p), y = pr(IPY,p), z = pr(IPZ,p);
    const Real xc = x + SplitLobeShift(T::CloudLobe1(code), e1);
    const Real yc = y + SplitLobeShift(T::CloudLobe2(code), e2);
    const Real zc = z + SplitLobeShift(T::CloudLobe3(code), e3);
    const int sp = pi(PSP,p);
    const Real vx = pr(IPVX,p) - vr(3*sp), vy = pr(IPVY,p) - vr(3*sp+1);
    const Real vz = pr(IPVZ,p) - vr(3*sp+2);
    // the parent's own support; where its populated cells do not spread over every direction
    // (status 3), once more on the plain support of the next coarser level (twice as wide)
    Real gg[9], vg, dg[3];
    int st = 3, ncell = 0;
#pragma unroll 1
    for (int attempt=0; attempt<2 && st == 3; ++attempt) {
    if (attempt == 1 && lev == 0) break;
    if (attempt == 1 && chk) {
      // (check) a retry, and whether its support (plain, level lev - 1) reaches beyond the
      // anchor block's halo allowance: CheckSplitVelocityReach widens the halo for exactly this
      // support before the split, so the second counter must stay 0
      Kokkos::atomic_add(&cnt(5), 1ULL);
      if (multi) {
        const int m = lt.Anchor(pi(PGID,p));
        Real xf = x, yf = y, zf = z;
        lt.Frame(m, xf, yf, zf);
        const Real r1 = lt.CloudHalfWidth1(lev - 1), r2 = lt.CloudHalfWidth2(lev - 1);
        const Real r3 = lt.CloudHalfWidth3(lev - 1);
        const Real bx0 = lt.box(m,0), bx1 = bx0 + lt.nx1*lt.box(m,3);
        const Real by0 = lt.box(m,1), by1 = by0 + lt.nx2*lt.box(m,4);
        const Real bz0 = lt.box(m,2), bz1 = bz0 + lt.nx3*lt.box(m,5);
        const Real o1 = Kokkos::fmax(0.0, Kokkos::fmax(bx0 - (xf - r1), (xf + r1) - bx1));
        const Real o2 = Kokkos::fmax(0.0, Kokkos::fmax(by0 - (yf - r2), (yf + r2) - by1));
        const Real o3 = Kokkos::fmax(0.0, Kokkos::fmax(bz0 - (zf - r3), (zf + r3) - bz1));
        const Real d = Kokkos::fmax(Kokkos::ceil(o1/lt.fine_dx1),
                                    Kokkos::fmax(Kokkos::ceil(o2/lt.fine_dx2),
                                                 Kokkos::ceil(o3/lt.fine_dx3)));
        if (static_cast<std::int64_t>(d) > allow(m)) Kokkos::atomic_add(&cnt(6), 1ULL);
      }
    }
    const int tcode = (attempt == 0) ? code : lev - 1;
    SVMoments q;
    ncell = 0;
    lt.ForEachCloudWeight(x, y, z, tcode, lt.Anchor(pi(PGID,p)),
    [&](const int target, const Real w) {
      if (w == 0.0) return;
      int tb, tk, tj, ti;
      lt.Decode(target, tb, tk, tj, ti);
      const Real rho = f(tb,c0,tk,tj,ti);
      if (!(rho > 0.0)) return;
      const Real ir = 1.0/rho;
      Real xq = xc, yq = yc, zq = zc;
      lt.Frame(tb, xq, yq, zq);
      // the cell's mean centroid and mean velocity, relative to the parent's; the covariance
      // of the centroids within the cell
      const Real m0 = f(tb,c0+1,tk,tj,ti)*ir, m1 = f(tb,c0+2,tk,tj,ti)*ir;
      const Real m2 = f(tb,c0+3,tk,tj,ti)*ir;
      const Real X0 = lt.box(tb,0) + (ti - is + 0.5)*lt.box(tb,3) - xq + m0;
      const Real X1 = lt.box(tb,1) + (tj - js + 0.5)*lt.box(tb,4) - yq + m1;
      const Real X2 = lt.box(tb,2) + (tk - ks + 0.5)*lt.box(tb,5) - zq + m2;
      const Real vc[6] = {f(tb,c0+4,tk,tj,ti)*ir - m0*m0, f(tb,c0+5,tk,tj,ti)*ir - m1*m1,
                          f(tb,c0+6,tk,tj,ti)*ir - m2*m2, f(tb,c0+7,tk,tj,ti)*ir - m0*m1,
                          f(tb,c0+8,tk,tj,ti)*ir - m0*m2, f(tb,c0+9,tk,tj,ti)*ir - m1*m2};
      ++ncell;
      q.Add(w*rho, X0, X1, X2, f(tb,c0+10,tk,tj,ti)*ir - vx, f(tb,c0+11,tk,tj,ti)*ir - vy,
            f(tb,c0+12,tk,tj,ti)*ir - vz, w*f(tb,c0+13,tk,tj,ti), vc);
    });
    st = SVFit(q, 3, vx, vy, vz, e1, e2, e3, c2, gg, vg, dg);
    }
    for (int n=0; n<9; ++n) g(row,n) = gg[n];
    g(row,9) = pr(IPVX,p); g(row,10) = pr(IPVY,p); g(row,11) = pr(IPVZ,p); g(row,12) = pr(IPM,p);
    g(row,13) = (st == 1) ? vg : 0.0;
    if (chk) {
      g(row,14) = (dg[2] > 0.0) ? sqrt(fabs(dg[0])/dg[2]) : -1.0;
      g(row,15) = st; g(row,16) = ncell; g(row,17) = dg[0]; g(row,18) = dg[1]; g(row,19) = dg[2];
      g(row,20) = pr(IPX,p); g(row,21) = pr(IPY,p); g(row,22) = pr(IPZ,p); g(row,23) = code;
      g(row,24) = lt.Anchor(pi(PGID,p));
    }
    // counters: [0] events, [st] by status (1 applied, 2 dispersion, 3 conditioning, 4 uniform)
    Kokkos::atomic_add(&cnt(0), 1ULL);
    Kokkos::atomic_add(&cnt(st), 1ULL);
  });
}

void DustGasDrag::SplitVelocityGradient(const int npart) {
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const auto lt = leaf_traversal;
  auto f = split_mom;
  auto g = split_vgrad;
  auto np = split_np;
  auto off = split_off;
  const int is = lt.is, js = lt.js, ks = pmy_pack->pmesh->mb_indcs.ks;
  const int nc1 = lt.ncells1, nc2 = lt.ncells2;
  const Real c2 = cloud_split_dispersion_max*cloud_split_dispersion_max;
  const bool chk = cloud_split_velocity_check;
  auto vr = split_vref;
  auto cnt = split_vel_cnt;   // device counters (no host synchronization per split cycle)
  // (check) the retry supports against the halo allowance of the anchor block (several ranks)
  const bool multi = global_variable::nranks > 1;
  auto allow = halo_allow_d_;
  Kokkos::parallel_for("dust_split_vgrad2d", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (np(p) < 2) return;
    using T = SMRCompositeTransfer;
    const int row = off(p);
    const int code = pi(PLEV,p), lev = T::CloudLevel(code), c0 = kNC2*pi(PSP,p);
    const Real e1 = lt.CloudHalfWidth1(lev), e2 = lt.CloudHalfWidth2(lev);
    const Real x = pr(IPX,p), y = pr(IPY,p);
    const Real xc = x + SplitLobeShift(T::CloudLobe1(code), e1);
    const Real yc = y + SplitLobeShift(T::CloudLobe2(code), e2);
    const int sp = pi(PSP,p);
    const Real vx = pr(IPVX,p) - vr(3*sp), vy = pr(IPVY,p) - vr(3*sp+1);
    const Real vz = pr(IPVZ,p) - vr(3*sp+2);
    Real gg[9], vg, dg[3];
    int st = 3, ncell = 0;
#pragma unroll 1
    for (int attempt=0; attempt<2 && st == 3; ++attempt) {
    if (attempt == 1 && lev == 0) break;
    if (attempt == 1 && chk) {   // (check) as in 3-D
      Kokkos::atomic_add(&cnt(5), 1ULL);
      if (multi) {
        const int m = lt.Anchor(pi(PGID,p));
        Real xf = x, yf = y;
        lt.Frame(m, xf, yf);
        const Real r1 = lt.CloudHalfWidth1(lev - 1), r2 = lt.CloudHalfWidth2(lev - 1);
        const Real bx0 = lt.box(m,0), bx1 = bx0 + lt.nx1*lt.box(m,2);
        const Real by0 = lt.box(m,1), by1 = by0 + lt.nx2*lt.box(m,3);
        const Real o1 = Kokkos::fmax(0.0, Kokkos::fmax(bx0 - (xf - r1), (xf + r1) - bx1));
        const Real o2 = Kokkos::fmax(0.0, Kokkos::fmax(by0 - (yf - r2), (yf + r2) - by1));
        const Real d = Kokkos::fmax(Kokkos::ceil(o1/lt.fine_dx1), Kokkos::ceil(o2/lt.fine_dx2));
        if (static_cast<std::int64_t>(d) > allow(m)) Kokkos::atomic_add(&cnt(6), 1ULL);
      }
    }
    const int tcode = (attempt == 0) ? code : lev - 1;
    SVMoments q;
    ncell = 0;
    lt.ForEachCloudWeight(x, y, tcode, lt.Anchor(pi(PGID,p)), [&](const int target, const Real w) {
      if (w == 0.0) return;
      const int ti = target % nc1, r = target/nc1, tj = r % nc2, tm = r/nc2;
      const Real rho = f(tm,c0,ks,tj,ti);
      if (!(rho > 0.0)) return;
      const Real ir = 1.0/rho;
      Real xq = xc, yq = yc;
      lt.Frame(tm, xq, yq);
      const Real m0 = f(tm,c0+1,ks,tj,ti)*ir, m1 = f(tm,c0+2,ks,tj,ti)*ir;
      const Real X0 = lt.box(tm,0) + (ti - is + 0.5)*lt.box(tm,2) - xq + m0;
      const Real X1 = lt.box(tm,1) + (tj - js + 0.5)*lt.box(tm,3) - yq + m1;
      const Real vc[6] = {f(tm,c0+3,ks,tj,ti)*ir - m0*m0, f(tm,c0+4,ks,tj,ti)*ir - m1*m1, 0.0,
                          f(tm,c0+5,ks,tj,ti)*ir - m0*m1, 0.0, 0.0};
      ++ncell;
      q.Add(w*rho, X0, X1, 0.0, f(tm,c0+6,ks,tj,ti)*ir - vx, f(tm,c0+7,ks,tj,ti)*ir - vy,
            f(tm,c0+8,ks,tj,ti)*ir - vz, w*f(tm,c0+9,ks,tj,ti), vc);
    });
    st = SVFit(q, 2, vx, vy, vz, e1, e2, 0.0, c2, gg, vg, dg);
    }
    for (int n=0; n<9; ++n) g(row,n) = gg[n];
    g(row,9) = pr(IPVX,p); g(row,10) = pr(IPVY,p); g(row,11) = pr(IPVZ,p); g(row,12) = pr(IPM,p);
    g(row,13) = (st == 1) ? vg : 0.0;
    if (chk) {
      g(row,14) = (dg[2] > 0.0) ? sqrt(fabs(dg[0])/dg[2]) : -1.0;
      g(row,15) = st; g(row,16) = ncell; g(row,17) = dg[0]; g(row,18) = dg[1]; g(row,19) = dg[2];
      g(row,20) = pr(IPX,p); g(row,21) = pr(IPY,p); g(row,22) = 0.0; g(row,23) = code;
      g(row,24) = lt.Anchor(pi(PGID,p));
    }
    // counters: [0] events, [st] by status (1 applied, 2 dispersion, 3 conditioning, 4 uniform)
    Kokkos::atomic_add(&cnt(0), 1ULL);
    Kokkos::atomic_add(&cnt(st), 1ULL);
  });
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitVelocityCheck
//! \brief (cloud_split_velocity_check) after the fill: for every event with the gradient
//! applied, |sum_k m_k (v_k - v_parent)| / (M V_G), the largest kept in split_vel_dpmax (pieces
//! mirrored at a reflecting x3 face, whose vz the split flips, are not separated out).

void DustGasDrag::SplitVelocityCheck(const int npart, const int nnew) {
  if (!cloud_split_velocity_check || nnew <= npart) return;
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto g = split_vgrad;
  auto np = split_np;
  auto off = split_off;
  Real worst = 0.0;
  Kokkos::parallel_reduce("dust_split_vcheck", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, Real &w) {
    if (np(p) < 2) return;
    const int row = off(p);
    if (!(g(row,13) > 0.0)) return;
    Real dp[3] = {0.0, 0.0, 0.0};
    for (int k=0; k<np(p); ++k) {
      const int s = (k == 0) ? p : npart + off(p) + k - 1;
      const Real m = pr(IPM,s);
      dp[0] += m*(pr(IPVX,s) - g(row,9));
      dp[1] += m*(pr(IPVY,s) - g(row,10));
      dp[2] += m*(pr(IPVZ,s) - g(row,11));
    }
    const Real r = sqrt(dp[0]*dp[0] + dp[1]*dp[1] + dp[2]*dp[2])/(g(row,12)*g(row,13));
    if (r > w) w = r;
  }, Kokkos::Max<Real>(worst));
  split_vel_dpmax = std::max(split_vel_dpmax, static_cast<double>(worst));
  // the ratio sigma_res / V_G of the events (status 1 and 2), and a few fallbacks in the log
  const int nrows = nnew - npart;
  auto gh = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                Kokkos::subview(g, std::make_pair(0, nrows), Kokkos::ALL));
  auto nph = Kokkos::create_mirror_view_and_copy(HostMemSpace(), np);
  auto offh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), off);
  static const double edge[11] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0, 2.0, 5.0, 10.0};
  // (the scope of a restricted deposit) the blocks of this rank that hold an event parent, and
  // those blocks with their MeshBlock neighbours
  {
    const int nmb = pmy_pack->nmb_thispack;
    std::vector<char> ev(nmb, 0), dil(nmb, 0);
    for (int p=0; p<npart; ++p) {
      if (nph(p) < 2) continue;
      const int m = static_cast<int>(gh(offh(p),24));
      if (m >= 0 && m < nmb) ev[m] = 1;
    }
    const auto &nb = pmy_pack->pmb->nghbr;
    for (int m=0; m<nmb; ++m) {
      if (!ev[m]) continue;
      dil[m] = 1;
      for (int n=0; n<pmy_pack->pmb->nnghbr; ++n) {
        const int g2 = nb.h_view(m,n).gid - pmy_pack->gids;
        if (g2 >= 0 && g2 < nmb) dil[g2] = 1;
      }
    }
    long long ne = 0, nd = 0;
    for (int m=0; m<nmb; ++m) {ne += ev[m]; nd += dil[m];}
    split_vel_blk[0] += static_cast<double>(ne)/std::max(1, nmb);
    split_vel_blk[1] += static_cast<double>(nd)/std::max(1, nmb);
    split_vel_blk[2] += 1.0;
  }
  for (int p=0; p<npart; ++p) {
    if (nph(p) < 2) continue;
    const int row = offh(p);
    const int st = static_cast<int>(gh(row,15));
    if (st != 1 && st != 2) continue;
    const double r = gh(row,14);
    int b = 0;
    while (b < 11 && r >= edge[b]) ++b;
    ++split_vel_hist[b];
    if ((st == 2 && split_vel_samples < 20) || (st == 1 && split_vel_samples_app < 20)) {
      if (st == 2) ++split_vel_samples; else ++split_vel_samples_app;
      std::printf("# SPLIT_VELOCITY_%s rank=%d cycle=%d x=%.6f %.6f %.6f code=%d ratio=%.4g "
                  "cells=%d sigx2=%.6g lin=%.6g VG2=%.6g G=%.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g %.4g\n",
                  (st == 2) ? "FALLBACK" : "APPLIED", global_variable::my_rank,
                  pmy_pack->pmesh->ncycle, gh(row,20), gh(row,21), gh(row,22),
                  static_cast<int>(gh(row,23)), r, static_cast<int>(gh(row,16)), gh(row,17),
                  gh(row,18), gh(row,19), gh(row,0), gh(row,1), gh(row,2), gh(row,3), gh(row,4),
                  gh(row,5), gh(row,6), gh(row,7), gh(row,8));
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitVelocitySummary
//! \brief One line per rank at the end (mesh_gradient only).

void DustGasDrag::SplitVelocitySummary() {
  if (cloud_split_velocity == 0) return;
  if (split_vel_cnt.extent_int(0) == 7 && Kokkos::is_initialized()) {
    auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), split_vel_cnt);
    split_vel_events = h(0); split_vel_applied = h(1); split_vel_disp = h(2);
    split_vel_cond = h(3); split_vel_uniform = h(4);
    split_vel_retry = h(5); split_vel_retry_out = h(6);
  }
  std::ostringstream os;
  os << "# DUST_SPLIT_VELOCITY rank=" << global_variable::my_rank << " deposits="
     << split_vel_deposits << " events=" << split_vel_events << " gradient=" << split_vel_applied
     << " fallback_dispersion=" << split_vel_disp << " fallback_conditioning=" << split_vel_cond
     << " uniform=" << split_vel_uniform << " halo_widenings=" << split_vel_halo_rebuilds;
  if (cloud_split_velocity_check) {
    os << " retries=" << split_vel_retry << " retries_beyond_halo=" << split_vel_retry_out;
    os << " max_dp_over_MVG=" << split_vel_dpmax << " ratio_hist(<0.1,..,0.7,1,2,5,10,>10)=";
    for (int b=0; b<12; ++b) os << (b ? "," : "") << split_vel_hist[b];
    if (split_vel_blk[2] > 0.0) {
      os << " mean_block_fraction_with_events=" << split_vel_blk[0]/split_vel_blk[2]
         << " with_neighbours=" << split_vel_blk[1]/split_vel_blk[2];
    }
  }
  os << "\n";
  std::cout << os.str() << std::flush;
}

}  // namespace dust
