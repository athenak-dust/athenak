//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_science.cpp
//! \brief Science output of the dust.
//!
//!  - ScienceRepresentation: the rule by which a particle's mass is spread over space, as
//!    "key = value" lines for the header of the particle file (pbin, outputs/pbin.cpp), so
//!    that a reader needs no knowledge of this code version's defaults.
//!  - DepositScienceFields: the dust mass and momentum densities on the active cells with that
//!    rule, recomputed from the particles of the output time (never the stage-leftover qdep):
//!      refinable clouds, 3-D: the exact integrals of the cloud shapes over the active leaves
//!        (DepositCloudDensity3D, the leaf traversal of the coupling; shear-periodic images,
//!        physical x3 faces mirrored back);
//!      refinable clouds, 2-D (and smooth_p2): the composite weights of the coupling on the
//!        leaves (ForEachCompositeWeight), the halo parts added to their owners;
//!      point particles, 3-D: the coupling's stencil (<dust>/deposit) on the owner block, the
//!        ghost parts remapped conservatively onto the leaves they overlap (DepRemapPlan:
//!        uniform, owner_overlap SMR / AMR, shear-periodic x1, physical x3 faces folded back);
//!      point particles, 2-D: owner_overlap on a refined mesh (OwnerOverlapDeposit), on a
//!        uniform mesh the owner stencil with the coupling's additive ghost exchange.
//!    The masses and momenta of the particles and of the fields are compared (science_audit,
//!    one log line per call); outputs never change the particle, gas or coupling state.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

namespace {
[[noreturn]] void ScienceFail(const std::string &message) {
  std::cerr << "### FATAL ERROR in dust science output: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
const char *StencilName(const DustDeposit d) {
  return (d == DustDeposit::tsc) ? "tsc" : (d == DustDeposit::cic) ? "cic" : "ngp";
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ScienceRepresentationId
//! \brief refinable_hat (refinable clouds), smooth_p2 (the 2-D composite control), or
//! <stencil>_owner (point particles: the coupling's stencil on the owner block's cells).

std::string DustGasDrag::ScienceRepresentationId() const {
  if (smr_transfer_mode == DustSMRTransfer::refinable_cloud) return "refinable_hat";
  if (smr_transfer_mode == DustSMRTransfer::smooth_p2) return "smooth_p2";
  return std::string(StencilName(deposit)) + "_owner";
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ScienceRepresentation
//! \brief The representation rule and its constants, one "key = value" per line.

void DustGasDrag::ScienceRepresentation(std::ostream &os) const {
  Mesh *pm = pmy_pack->pmesh;
  const int ndim = pm->three_d ? 3 : (pm->two_d ? 2 : 1);
  const Real rdx[3] = {(pm->mesh_size.x1max - pm->mesh_size.x1min)/pm->mesh_indcs.nx1,
                       (pm->mesh_size.x2max - pm->mesh_size.x2min)/pm->mesh_indcs.nx2,
                       (pm->mesh_size.x3max - pm->mesh_size.x3min)/pm->mesh_indcs.nx3};
  const std::string id = ScienceRepresentationId();
  os << std::setprecision(17);
  os << "representation = " << id << "\n";
  os << "representation_version = 1\n";
  os << "root_dx1 = " << rdx[0] << "\nroot_dx2 = " << rdx[1] << "\nroot_dx3 = " << rdx[2]
     << "\n";
  os << "mesh_levels = " << (pm->max_level - pm->root_level) << "\n";
  if (id == "refinable_hat") {
    os << "cloud_kernel = hat  # S(s) = max(0, 1 - |s|), s = (x - x_p)/ell, per direction\n";
    os << "cloud_half_width = root_dx_d * 2^-level  # ell_d of a cloud of level `level`; "
       << "the same physical width on any leaf\n";
    os << "cloud_lobe_code = 0 none, 1 lower, 2 upper, 3 both  # per direction\n";
    os << "cloud_lobe_shape = lobe 1/2: (2/3) S_ell(x - x_p) + (1/3) S_ell(x - x_p -/+ ell); "
       << "lobe 3: hat of half-width 2 ell\n";
    os << "cloud_support = lobe 0 [x_p - ell, x_p + ell], 1 [x_p - 2 ell, x_p + ell], "
       << "2 [x_p - ell, x_p + 2 ell], 3 [x_p - 2 ell, x_p + 2 ell]\n";
    os << "cloud_max_level = " << smr_transfer.cloud_max_level << "\n";
    os << "cloud_dims = " << ndim << "  # directions with a cloud shape (a 2-D run has none "
       << "along x3)\n";
    os << "mesh_deposit = exact integral of the cloud shape over the active leaf cells "
       << "(uniform level-0 mesh: canonical TSC)\n";
  } else if (id == "smooth_p2") {
    os << "mesh_deposit = smooth_p2 composite transfer (2-D control; see the source)\n";
  } else {
    os << "stencil = " << StencilName(deposit) << "  # on the cells of the owner MeshBlock "
       << "(the block whose half-open box holds x_p)\n";
    os << "transfer = "
       << ((smr_transfer_mode == DustSMRTransfer::owner_overlap) ? "owner_overlap" : "none")
       << "  # owner_overlap: deposits in ghost cells are added to the active leaf cells "
       << "they overlap, by volume\n";
    os << "mesh_deposit = owner-block stencil, ghost-cell deposits remapped conservatively\n";
  }
  if (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic) {
    os << "x1_faces = shear_periodic  # a point beyond the outer x1 face is (x - Lx, "
       << "y + q Omega Lx t mod Ly)\n";
  }
  if (pm->three_d && pm->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic) {
    os << "x3_walls = fold  # the part of a deposit beyond a physical x3 face is mirrored "
       << "back (mass conserving)\n";
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositScienceFields
//! \brief See the file header.  out is (re)allocated to at least (nmb, 4, nk, nj, ni).

void DustGasDrag::DepositScienceFields(DvceArray5D<Real> &out, const bool audit,
                                       const bool mass_only) {
  Mesh *pm = pmy_pack->pmesh;
  auto &ind = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack, gids = pmy_pack->gids;
  const int nc1 = ind.nx1 + 2*ind.ng;
  const int nc2 = (ind.nx2 > 1) ? ind.nx2 + 2*ind.ng : 1;
  const int nc3 = (ind.nx3 > 1) ? ind.nx3 + 2*ind.ng : 1;
  const int is = ind.is, ie = ind.ie, js = ind.js, je = ind.je, ks = ind.ks, ke = ind.ke;
  if (out.extent_int(0) < nmb || out.extent_int(1) < 4 || out.extent_int(2) != nc3 ||
      out.extent_int(3) != nc2 || out.extent_int(4) != nc1) {
    Kokkos::realloc(out, std::max(1, nmb), 4, nc3, nc2, nc1);
  }
  Kokkos::deep_copy(out, 0.0);
  auto o = out;
  particles::Particles *ppar = pmy_pack->ppart;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int np = ppar->nprtcl_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool three_d = pm->three_d;
  if (pm->one_d) ScienceFail("dust science fields need a 2-D or 3-D mesh");

  if (smr_transfer_mode == DustSMRTransfer::refinable_cloud && three_d) {
    // ---- 3-D refinable clouds: the exact leaf integrals of the coupling's traversal
    if (!clouds3d) ScienceFail("3-D refinable clouds need <dust>/experimental_3d = true");
    if (!mass_only) {
      DepositCloudDensity3D(science_work_);   // collective (halo), shear geometry at pm->time
    } else {
      // the mass component of DepositCloudDensity3D alone (the same operations)
      if (science_mass_.extent_int(0) != qdep.extent_int(0) ||
          science_mass_.extent_int(1) != 1 || science_mass_.extent_int(2) != qdep.extent_int(2) ||
          science_mass_.extent_int(3) != qdep.extent_int(3) ||
          science_mass_.extent_int(4) != qdep.extent_int(4)) {
        Kokkos::realloc(science_mass_, qdep.extent(0), 1, qdep.extent(2), qdep.extent(3),
                        qdep.extent(4));
      }
      Kokkos::deep_copy(science_mass_, 0.0);
      UpdateShear3D(pm->time);
      const auto lt = leaf_traversal3d;
      auto ff = science_mass_;
      Kokkos::parallel_for("dust_science_cloud3d_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, np),
      KOKKOS_LAMBDA(const int p) {
        const Real m = pr(IPM,p);
        lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pr(IPZ,p), pi(PLEV,p), lt.Anchor(pi(PGID,p)),
        [&](const int target, const Real w) {
          int tb, tk, tj, ti;
          lt.Decode(target, tb, tk, tj, ti);
          DepositAdd(&ff(tb,0,tk,tj,ti), m*w/(lt.box(tb,3)*lt.box(tb,4)*lt.box(tb,5)));
        });
      });
      HaloReverseAdd(science_mass_, 0, 1);
    }
    auto f = mass_only ? science_mass_ : science_work_;
    const int nc = mass_only ? 1 : 4;
    par_for("dust_science_cloud3d", DevExeSpace(), 0, nmb-1, ks, ke, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      for (int c=0; c<nc; ++c) o(m,c,k,j,i) = f(m,c,k,j,i);
    });
  } else if (UsesCompositeTransfer()) {
    // ---- 2-D refinable clouds (and the smooth_p2 control): composite weights on the leaves
    const auto transfer = smr_transfer;
    const auto lt = leaf_traversal;
    const bool cloud = transfer.cloud_mode;
    const int ncb = nc1*nc2;
    const int nslot = (global_variable::nranks > 1) ? std::max(nmb, lt.nfld) : nmb;
    const int nc = mass_only ? 1 : 4;
    DvceArray2D<Real> f0("dust_science_leaf2d", nc, static_cast<std::size_t>(nslot)*ncb);
    par_for("dust_science_composite2d", DevExeSpace(), 0, np-1, KOKKOS_LAMBDA(const int p) {
      const Real mass = pr(IPM,p);
      if (mass == 0.0) return;
      const Real v1 = pr(IPVX,p), v2 = pr(IPVY,p), v3 = pr(IPVZ,p);
      ForEachCompositeWeight(transfer, lt, pr(IPX,p), pr(IPY,p), pi(PLEV,p),
                             lt.Anchor(pi(PGID,p)),
      [&](const int target, const Real w) {
        const int tm = target/ncb;
        const Real area = cloud ? lt.box(tm,2)*lt.box(tm,3)
                                : mbsize.d_view(tm).dx1*mbsize.d_view(tm).dx2;
        const Real d = mass*w/area;
        DepositAdd(&f0(0,target), d);
        if (mass_only) return;
        DepositAdd(&f0(1,target), d*v1);
        DepositAdd(&f0(2,target), d*v2);
        DepositAdd(&f0(3,target), d*v3);
      });
    });
    HaloReverseAddFlat(f0, nc);   // several ranks: the halo parts to their owners
    par_for("dust_science_composite2d_copy", DevExeSpace(), 0, nmb-1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      const std::size_t t = (static_cast<std::size_t>(m)*nc2 + j)*nc1 + i;
      for (int c=0; c<nc; ++c) o(m,c,ks,j,i) = f0(c,t);
    });
  } else if (three_d) {
    // ---- 3-D point particles: owner stencil + conservative ghost remap
    DepositPointScienceFields(pm->time, out, mass_only);
  } else if (smr_transfer_mode == DustSMRTransfer::owner_overlap) {
    // ---- 2-D point particles on a refined mesh: the owner_overlap leaf deposit
    DvceArray5D<Real> &work = mass_only ? science_mass_ : science_work_;
    OwnerOverlapDeposit(work, !mass_only);   // stops on a mass mismatch
    auto g = work;
    const int nc = mass_only ? 1 : 4;
    par_for("dust_science_oo2d_copy", DevExeSpace(), 0, nmb-1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      for (int c=0; c<nc; ++c) o(m,c,ks,j,i) = g(m,c,ks,j,i);
    });
  } else {
    // ---- 2-D point particles on a uniform mesh: owner stencil + the coupling's additive
    // ghost exchange (pbval_qp, 5 components as in the coupling; idle between cycles)
    if (pm->multilevel) ScienceFail("2-D point particles on a refined mesh need owner_overlap");
    if (pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic) {
      ScienceFail("2-D point particles with shear-periodic x1 faces are not supported");
    }
    if (science_work_.extent_int(0) != nmb || science_work_.extent_int(1) != 5 ||
        science_work_.extent_int(2) != nc3 || science_work_.extent_int(3) != nc2 ||
        science_work_.extent_int(4) != nc1) {
      science_work_ = DvceArray5D<Real>("dust_science_point2d", std::max(1, nmb), 5, nc3, nc2,
                                        nc1);
    }
    Kokkos::deep_copy(science_work_, 0.0);
    auto a = science_work_;
    const int scheme = static_cast<int>(deposit);
    const int nx1 = ind.nx1, nx2 = ind.nx2;
    int bad = 0;
    Kokkos::parallel_reduce("dust_science_point2d", Kokkos::RangePolicy<DevExeSpace>(0, np),
    KOKKOS_LAMBDA(const int p, int &nbad) {
      const Real mass = pr(IPM,p);
      if (mass == 0.0) return;
      const int m = pi(PGID,p) - gids;
      if (m < 0 || m >= nmb) {++nbad; return;}
      int ip, jp;
      Real wx[3], wy[3];
      PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is, scheme,
                ip, wx);
      PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js, scheme,
                jp, wy);
      if (ip < 1 || ip > nc1 - 2 || jp < 1 || jp > nc2 - 2) {++nbad; return;}
      const Real d = mass/(mbsize.d_view(m).dx1*mbsize.d_view(m).dx2);
      const Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
      for (int b=0; b<3; ++b) {
        for (int c=0; c<3; ++c) {
          const Real w = wy[b]*wx[c]*d;
          if (w == 0.0) continue;
          const int jj = jp + b - 1, ii = ip + c - 1;
          DepositAdd(&a(m,0,ks,jj,ii), w);
          if (mass_only) continue;
          for (int q=0; q<3; ++q) DepositAdd(&a(m,1+q,ks,jj,ii), w*v[q]);
        }
      }
    }, Kokkos::Sum<int>(bad));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (bad > 0) ScienceFail(std::to_string(bad) + " particles outside their block's stencil");
    pbval_qp->InitRecvDep(5);
    pbval_qp->PackAndSendDeposit(science_work_);
    while (pbval_qp->RecvAndSumDeposit(science_work_) != TaskStatus::complete) {}
    while (pbval_qp->ClearSendDep() != TaskStatus::complete) {}
    while (pbval_qp->ClearRecvDep() != TaskStatus::complete) {}
    const int nc = mass_only ? 1 : 4;
    par_for("dust_science_point2d_copy", DevExeSpace(), 0, nmb-1, js, je, is, ie,
    KOKKOS_LAMBDA(int m, int j, int i) {
      for (int c=0; c<nc; ++c) o(m,c,ks,j,i) = a(m,c,ks,j,i);
    });
  }

  if (!audit) return;   // (the particle history makes its own sums)
  // ---- audit: the particles' mass and momentum against the fields' (all ranks)
  double s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  double pabs = 0.0;   // sum of m |v| (1-norm): the scale of the momentum round-off
  {
    // two-level sums (chunks of particles, then the chunks; per block, then the blocks), so that
    // the round-off of the audit itself stays far below its 1e-10 threshold on any backend
    constexpr int kChunk = 4096;
    const int nchunk = (np + kChunk - 1)/kChunk;
    double ps[5] = {0, 0, 0, 0, 0};
    Kokkos::parallel_reduce("dust_science_audit_p", Kokkos::RangePolicy<DevExeSpace>(0, nchunk),
    KOKKOS_LAMBDA(const int c, double &m0, double &m1, double &m2, double &m3, double &ma) {
      double a[5] = {0, 0, 0, 0, 0};
      const int p1 = (c + 1)*kChunk < np ? (c + 1)*kChunk : np;
      for (int p=c*kChunk; p<p1; ++p) {
        const double m = static_cast<double>(pr(IPM,p));
        const double v[3] = {static_cast<double>(pr(IPVX,p)), static_cast<double>(pr(IPVY,p)),
                             static_cast<double>(pr(IPVZ,p))};
        a[0] += m;
        a[1] += m*v[0];
        a[2] += m*v[1];
        a[3] += m*v[2];
        a[4] += Kokkos::fabs(m)*(Kokkos::fabs(v[0]) + Kokkos::fabs(v[1]) + Kokkos::fabs(v[2]));
      }
      m0 += a[0]; m1 += a[1]; m2 += a[2]; m3 += a[3]; ma += a[4];
    }, Kokkos::Sum<double>(ps[0]), Kokkos::Sum<double>(ps[1]), Kokkos::Sum<double>(ps[2]),
       Kokkos::Sum<double>(ps[3]), Kokkos::Sum<double>(ps[4]));
    pabs = ps[4];
    double gs[4] = {0, 0, 0, 0};
    Kokkos::parallel_reduce("dust_science_audit_g", Kokkos::RangePolicy<DevExeSpace>(0, nmb),
    KOKKOS_LAMBDA(const int m, double &g0, double &g1, double &g2, double &g3) {
      double vol = static_cast<double>(mbsize.d_view(m).dx1)*mbsize.d_view(m).dx2;
      if (three_d) vol *= mbsize.d_view(m).dx3;
      double a[4] = {0, 0, 0, 0};
      for (int k=ks; k<=ke; ++k) {
        for (int j=js; j<=je; ++j) {
          double b[4] = {0, 0, 0, 0};
          for (int i=is; i<=ie; ++i) {
            for (int c=0; c<4; ++c) b[c] += o(m,c,k,j,i);
          }
          for (int c=0; c<4; ++c) a[c] += b[c];
        }
      }
      g0 += vol*a[0]; g1 += vol*a[1]; g2 += vol*a[2]; g3 += vol*a[3];
    }, Kokkos::Sum<double>(gs[0]), Kokkos::Sum<double>(gs[1]), Kokkos::Sum<double>(gs[2]),
       Kokkos::Sum<double>(gs[3]));
    s[0] = ps[0]; s[1] = gs[0];
    for (int c=0; c<3; ++c) {s[2+c] = ps[1+c]; s[5+c] = gs[1+c];}
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, s, 8, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &pabs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int c=0; c<8; ++c) science_audit[c] = s[c];
  const double rel_mass = (s[0] != 0.0) ? std::fabs(s[1] - s[0])/std::fabs(s[0])
                                        : std::fabs(s[1]);
  double perr = 0.0;
  for (int c=0; c<3; ++c) perr = std::max(perr, std::fabs(s[5+c] - s[2+c]));
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(10) << "# DUST_SCIENCE_FIELDS t=" << pm->time
              << " cycle=" << pm->ncycle << " representation=" << ScienceRepresentationId()
              << " mass_particles=" << s[0] << " mass_mesh=" << s[1]
              << " rel_mass_error=" << std::setprecision(3) << rel_mass
              << " momentum_error/sum(m|v|)=" << ((pabs > 0.0) ? perr/pabs : perr)
              << std::endl;
    if (!(rel_mass <= 1.0e-10)) {
      std::cout << "### WARNING (dust science fields): the deposited dust mass differs from the "
                << "particle mass by " << rel_mass << " (relative)" << std::endl;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ParticleHistory
//! \brief The quantities of one particle-history row (file_type = phst)
//! after time and dt, global over the ranks (every rank gets them):
//!   row[0] n_particles (active rows, mass != 0); row[1] mass; row[2..4] momentum (stored
//!   velocities); row[5] ekin = sum m |v|^2/2; row[6] z_rms = sqrt(sum m z^2/sum m), z = x3 in
//!   3-D, x2 (the vertical of the 2-D r-z box) in 2-D; row[7] rho_max, the largest dust density
//!   on the active cells (DepositScienceFields, mass only: the representation of the run,
//!   deposited from the particles of this time, bit for bit dust_total_rho); row[8..10] the centre of that cell, row[11] its level above
//!   the root (if rho_max = 0: centre 0, level -1); row[12] chi_max, row[13] dt_stiff_min (the
//!   interval records of UpdateStiffness; -1 without back-reaction, without a PC2 cycle in the
//!   interval, or, for dt_stiff_min, without stiff_thresh); row[14] coarsening events (G groups,
//!   and stage X families where X is on) and row[15] particles removed by R since the previous
//!   row.
//! Resets the interval records.  Collective.

void DustGasDrag::ParticleHistory(double row[16]) {
  Mesh *pm = pmy_pack->pmesh;
  particles::Particles *ppar = pmy_pack->ppart;
  auto pr = ppar->prtcl_rdata;
  const int np = ppar->nprtcl_thispack;
  const int zc = pm->three_d ? IPZ : IPY;
  // ---- particle sums: per chunk on the device, the chunks in order on the host
  constexpr int kChunk = 4096, kSums = 7;
  const int nchunk = (np + kChunk - 1)/kChunk;
  DvceArray2D<double> part("phst_chunks", std::max(1, nchunk), kSums);
  Kokkos::parallel_for("phst_particle_sums", Kokkos::RangePolicy<DevExeSpace>(0, nchunk),
  KOKKOS_LAMBDA(const int c) {
    double a[kSums] = {0, 0, 0, 0, 0, 0, 0};
    const int p1 = (c + 1)*kChunk < np ? (c + 1)*kChunk : np;
    for (int p=c*kChunk; p<p1; ++p) {
      const double m = static_cast<double>(pr(IPM,p));
      if (m == 0.0) continue;   // outflow tombstones
      const double v1 = pr(IPVX,p), v2 = pr(IPVY,p), v3 = pr(IPVZ,p), z = pr(zc,p);
      a[0] += 1.0;
      a[1] += m;
      a[2] += m*v1; a[3] += m*v2; a[4] += m*v3;
      a[5] += 0.5*m*(v1*v1 + v2*v2 + v3*v3);
      a[6] += m*z*z;
    }
    for (int q=0; q<kSums; ++q) part(c,q) = a[q];
  });
  auto ph = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part);
  double sums[kSums + 2] = {0};
  for (int c=0; c<nchunk; ++c) for (int q=0; q<kSums; ++q) sums[q] += ph(c,q);
  sums[kSums] = static_cast<double>(phst_g_events);
  sums[kSums + 1] = static_cast<double>(phst_r_removed);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums, kSums + 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  for (int q=0; q<6; ++q) row[q] = sums[q];
  row[6] = (sums[1] > 0.0) ? std::sqrt(sums[6]/sums[1]) : 0.0;

  // ---- the densest cell of the deposit
  DepositScienceFields(science_phst_, false, true);   // the mass component only
  auto &ind = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  const int nx1 = ind.nx1, nx2 = ind.nx2, nx3 = ind.nx3;
  const int is = ind.is, js = ind.js, ks = ind.ks;
  const int ncell = nx1*nx2*nx3;
  auto f = science_phst_;
  using ML = Kokkos::MaxLoc<Real, int>;
  typename ML::value_type best;
  best.val = 0.0; best.loc = -1;
  if (nmb > 0) {
    Kokkos::parallel_reduce("phst_rho_max", Kokkos::RangePolicy<DevExeSpace>(0, nmb*ncell),
    KOKKOS_LAMBDA(const int q, typename ML::value_type &b) {
      const int m = q/ncell, r = q % ncell;
      const Real v = f(m, 0, ks + r/(nx1*nx2), js + (r/nx1) % nx2, is + r % nx1);
      if (v > b.val || (v == b.val && q < b.loc)) {b.val = v; b.loc = q;}
    }, ML(best));
  }
  struct {double val; int rank;} mine = {static_cast<double>(best.val), global_variable::my_rank};
  if (best.loc < 0) mine.val = -1.0;
  auto glob = mine;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&mine, &glob, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
#endif
  double where[4] = {0.0, 0.0, 0.0, -1.0};
  if (glob.rank == global_variable::my_rank && best.loc >= 0 && best.val > 0.0) {
    const int m = best.loc/ncell, r = best.loc % ncell;
    const int k = r/(nx1*nx2), j = (r/nx1) % nx2, i = r % nx1;
    const auto &sz = pmy_pack->pmb->mb_size.h_view(m);
    where[0] = sz.x1min + (i + 0.5)*sz.dx1;
    where[1] = sz.x2min + (j + 0.5)*sz.dx2;
    where[2] = pm->three_d ? sz.x3min + (k + 0.5)*sz.dx3 : 0.0;
    where[3] = pm->lloc_eachmb[pmy_pack->gids + m].level - pm->root_level;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(where, 4, MPI_DOUBLE, glob.rank, MPI_COMM_WORLD);
#endif
  row[7] = std::max(0.0, glob.val);
  for (int c=0; c<4; ++c) row[8 + c] = where[c];

  // ---- the interval records (UpdateStiffness, the G / R counters), then reset
  row[12] = back_reaction ? static_cast<double>(phst_chi_max) : -1.0;
  row[13] = (back_reaction && stiff_thresh > 0.0) ? static_cast<double>(phst_dt_stiff_min) : -1.0;
  row[14] = sums[kSums];
  row[15] = sums[kSums + 1];
  phst_chi_max = -1.0;
  phst_dt_stiff_min = -1.0;
  phst_g_events = 0;
  phst_r_removed = 0;
}

}  // namespace dust
