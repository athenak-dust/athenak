//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_damping_amr.cpp
//! \brief Problem generator for the drag damping test on refined meshes: uniform dust
//! moving at <problem>/dust_test_v0 (default 0.1) along x1 through uniform gas at rest, without
//! rotation (no <shearing_box>).  The centre-of-mass velocity is conserved and the relative
//! velocity decays as exp(-(1 + eps) t/t_s), so the gas velocity is
//!     u(t) = eps v0/(1 + eps) (1 - exp(-(1 + eps) t/t_s))
//! (one species).  As dust_damping, but with the particle set-up of dust_setup: point
//! particles on a lattice (or random), or with <problem>/cloud_root_random = true root parents
//! (<problem>/root_placement = lattice | random | generic) that split exactly to the leaves
//! of a refined mesh as refinable clouds.  In 3-D with one species of root parents the
//! history carries the errors against the exact solution (e_rhog, e_ux, e_rhod, e_uperp, the
//! non-1-D parts y_rhog, y_rhod, and the leaf mass error dmass).  <problem>/dust_test_vz0 adds a
//! uniform vz to the 3-D root parents (with dust_test_vz_mirror its sign follows z).
//! <problem>/ba_* adds the 2-D diagnostic outputs of dust_snapshots.cpp.
//!
//! With dust_test_v0 = 0 this is also the quiet set-up of uniform dust at rest that
//! dust_freefall and dust_jeans start from.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "pgen/dust_amr/dust_setup.hpp"
#include "pgen/dust_amr/dust_snapshots.hpp"
#include "dust/dust_split_velocity.hpp"
#include "globals.hpp"

namespace {
dust_setup::DustTest3D dtest3d_;

void DampingHistory(HistoryData *pdata, Mesh *pm) {
  if (!dust_setup::SpeciesHistory(pdata, pm)) return;
  dust_snapshots::HistoryColumns(pdata, pm);
  dust_setup::DustTest3DHistory(pdata, pm, dtest3d_);
  dust_snapshots::Write(pm);
}

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust_damping_amr" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

// Test fields of <dust>/cloud_split_velocity (no drag and no forces: the clouds move
// ballistically).  <problem>/test_vgrad = "G00 G01 ... G22" (G_ij = dv_i/dx_j; in 2-D the
// columns j = 0, 1): every cloud gets v_base + G (X - c) at its centroid X (c = test_vgrad_c,
// default the domain centre).  The field stays linear under ballistic motion,
// u(x, t) = v_base + G (I + G t)^-1 (x - c - v_base t).  <problem>/test_two_stream_v = V: every
// cloud becomes two, v_base +- V along x1, masses m f and m (1 - f), f = 1/2 + a sin(2 pi x/L1)
// (a = test_two_stream_amp).  At the end one line # SPLIT_VELOCITY_TEST: linear, the largest
// |v - u(X, t)| of the clouds whose centroid lies in the box test_check (fractions of the
// domain, default 0.3 .. 0.7 per direction), children (level > 0) separately; two streams,
// the largest distance of any cloud's velocity from the nearer stream velocity.
struct SVTest {
  bool lin = false, two = false;
  int ndim = 2;
  Real g[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, c[3] = {0, 0, 0}, vb[3] = {0, 0, 0};
  Real V = 0.0, amp = 0.25;
  Real chk[6] = {0, 0, 0, 0, 0, 0};
  Real rdx[3] = {1, 1, 1}, xmin[3] = {0, 0, 0}, len[3] = {1, 1, 1};
} svt_;

void SVCentroid(const Real *x, const int code, Real X[3]) {
  using T = SMRCompositeTransfer;
  const int lev = T::CloudLevel(code);
  const int lobe[3] = {T::CloudLobe1(code), T::CloudLobe2(code), T::CloudLobe3(code)};
  for (int d=0; d<3; ++d) {
    X[d] = (d < svt_.ndim) ? x[d] + dust::SplitLobeShift(lobe[d], svt_.rdx[d]/(1 << lev)) : 0.0;
  }
}

void SVTestFinal(ParameterInput *, Mesh *pm) {
  auto *part = pm->pmb_pack->ppart;
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_idata);
  const int np = part->nprtcl_thispack, D = svt_.ndim;
  const Real t = pm->time;
  // (I + G t)^-1 of the spatial block
  Real mi[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  if (svt_.lin) {
    Real m[9];
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
      m[3*i + j] = ((i == j) ? 1.0 : 0.0) + ((i < D && j < D) ? svt_.g[3*i + j]*t : 0.0);
    }
    const Real c00 = m[4]*m[8] - m[5]*m[7], c01 = m[5]*m[6] - m[3]*m[8];
    const Real c02 = m[3]*m[7] - m[4]*m[6];
    const Real det = m[0]*c00 + m[1]*c01 + m[2]*c02;
    mi[0] = c00/det; mi[1] = (m[2]*m[7] - m[1]*m[8])/det; mi[2] = (m[1]*m[5] - m[2]*m[4])/det;
    mi[3] = c01/det; mi[4] = (m[0]*m[8] - m[2]*m[6])/det; mi[5] = (m[2]*m[3] - m[0]*m[5])/det;
    mi[6] = c02/det; mi[7] = (m[1]*m[6] - m[0]*m[7])/det; mi[8] = (m[0]*m[4] - m[1]*m[3])/det;
  }
  // [0] max error all checked, [1] children, [2] checked, [3] children checked, [4] lev max
  Real r[5] = {0, 0, 0, 0, 0};
  for (int p=0; p<np; ++p) {
    const Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
    const Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
    const int code = pi(PLEV,p), lev = SMRCompositeTransfer::CloudLevel(code);
    Real X[3];
    SVCentroid(x, code, X);
    Real err = 0.0;
    if (svt_.lin) {
      bool in = true;
      for (int d=0; d<D; ++d) in = in && X[d] >= svt_.chk[2*d] && X[d] <= svt_.chk[2*d+1];
      if (!in) continue;
      Real y[3] = {0, 0, 0}, x0[3] = {0, 0, 0};
      for (int d=0; d<D; ++d) y[d] = X[d] - svt_.c[d] - svt_.vb[d]*t;
      for (int i=0; i<D; ++i) for (int j=0; j<D; ++j) x0[i] += mi[3*i + j]*y[j];
      for (int i=0; i<3; ++i) {
        Real u = svt_.vb[i];
        for (int j=0; j<D; ++j) u += svt_.g[3*i + j]*x0[j];
        err = std::max(err, std::fabs(v[i] - u));
      }
    } else {
      Real ea = 0.0, eb = 0.0;
      for (int i=0; i<3; ++i) {
        const Real s = (i == 0) ? svt_.V : 0.0;
        ea = std::max(ea, std::fabs(v[i] - (svt_.vb[i] + s)));
        eb = std::max(eb, std::fabs(v[i] - (svt_.vb[i] - s)));
      }
      err = std::min(ea, eb);
    }
    r[0] = std::max(r[0], err); r[2] += 1.0;
    if (lev > 0) {r[1] = std::max(r[1], err); r[3] += 1.0;}
    r[4] = std::max(r[4], static_cast<Real>(lev));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &r[0], 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &r[2], 2, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &r[4], 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  Real gmax = 0.0;
  for (int n=0; n<9; ++n) gmax = std::max(gmax, std::fabs(svt_.g[n]));
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(6) << "# SPLIT_VELOCITY_TEST mode="
              << (svt_.lin ? "linear" : "two_stream") << " t=" << t << " checked=" << r[2]
              << " children=" << r[3] << " max_level=" << r[4] << " max_err=" << r[0]
              << " max_err_children=" << r[1] << " scale="
              << (svt_.lin ? gmax*svt_.rdx[0] : svt_.V) << std::endl;
  }
}

void SVTestInit(ParameterInput *pin, Mesh *pm, const Real v0) {
  // every test key is read only when present (before the return below: nothing is added to the
  // parameters, so the parameter dump and the restart header of a run without them are those of
  // the base; Codex review: GetOrAddReal added test_two_stream_v = 0 to every run)
  svt_.lin = pin->DoesParameterExist("problem", "test_vgrad");
  svt_.V = pin->DoesParameterExist("problem", "test_two_stream_v")
           ? pin->GetReal("problem", "test_two_stream_v") : 0.0;
  svt_.two = svt_.V != 0.0;
  if (!svt_.lin && !svt_.two) return;
  if (svt_.lin && svt_.two) Fatal("test_vgrad and test_two_stream_v exclude each other");
  if (pin->GetOrAddReal("problem", "dust_test_vz0", 0.0) != 0.0) Fatal("test fields need dust_test_vz0 = 0");
  auto *pack = pm->pmb_pack;
  auto *part = pack->ppart;
  svt_.ndim = pm->three_d ? 3 : 2;
  const auto &box = pm->mesh_size;
  const Real lo[3] = {box.x1min, box.x2min, box.x3min}, hi[3] = {box.x1max, box.x2max, box.x3max};
  const int nx[3] = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  for (int d=0; d<3; ++d) {
    svt_.xmin[d] = lo[d]; svt_.len[d] = hi[d] - lo[d]; svt_.rdx[d] = svt_.len[d]/nx[d];
    svt_.c[d] = 0.5*(lo[d] + hi[d]);
    svt_.chk[2*d] = lo[d] + 0.3*svt_.len[d]; svt_.chk[2*d+1] = lo[d] + 0.7*svt_.len[d];
  }
  svt_.vb[0] = v0;
  if (svt_.lin) {
    std::istringstream is(pin->GetString("problem", "test_vgrad"));
    for (int n=0; n<9; ++n) if (!(is >> svt_.g[n])) Fatal("test_vgrad needs 9 numbers");
    if (pin->DoesParameterExist("problem", "test_vgrad_c")) {
      std::istringstream ic(pin->GetString("problem", "test_vgrad_c"));
      for (int d=0; d<svt_.ndim; ++d) if (!(ic >> svt_.c[d])) Fatal("test_vgrad_c: one number per direction");
    }
  }
  if (pin->DoesParameterExist("problem", "test_check")) {
    std::istringstream ic(pin->GetString("problem", "test_check"));
    for (int d=0; d<svt_.ndim; ++d) {
      Real a, b;
      if (!(ic >> a >> b)) Fatal("test_check: two fractions per direction");
      svt_.chk[2*d] = lo[d] + a*svt_.len[d]; svt_.chk[2*d+1] = lo[d] + b*svt_.len[d];
    }
  }
  svt_.amp = pin->GetOrAddReal("problem", "test_two_stream_amp", 0.25);
  const int np = part->nprtcl_thispack;
  const int nout = svt_.two ? 2*np : np;
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_idata);
  Kokkos::View<Real**, Kokkos::LayoutRight, HostMemSpace> rr("svt_r", part->nrdata, std::max(nout, 1));
  Kokkos::View<int**, Kokkos::LayoutRight, HostMemSpace> ii("svt_i", part->nidata, std::max(nout, 1));
  std::uint64_t tag0 = 0;
  const std::uint64_t tstride = static_cast<std::uint64_t>(global_variable::nranks);
  if (svt_.two) {
    pack->pdust->EnsureNextTag();
    tag0 = pack->pdust->IssueTags(static_cast<std::uint64_t>(np));
  }
  for (int p=0; p<np; ++p) {
    const Real x[3] = {pr(IPX,p), pr(IPY,p), pr(IPZ,p)};
    Real X[3];
    SVCentroid(x, pi(PLEV,p), X);
    for (int c=0; c<part->nrdata; ++c) rr(c,p) = pr(c,p);
    for (int c=0; c<part->nidata; ++c) ii(c,p) = pi(c,p);
    if (svt_.lin) {
      const int iv[3] = {IPVX, IPVY, IPVZ};
      for (int i=0; i<3; ++i) {
        Real dv = 0.0;
        for (int j=0; j<svt_.ndim; ++j) dv += svt_.g[3*i + j]*(X[j] - svt_.c[j]);
        rr(iv[i],p) += dv;
      }
    } else {
      const Real f = 0.5 + svt_.amp*std::sin(2.0*M_PI*(X[0] - svt_.xmin[0])/svt_.len[0]);
      const int q = np + p;
      for (int c=0; c<part->nrdata; ++c) rr(c,q) = rr(c,p);
      for (int c=0; c<part->nidata; ++c) ii(c,q) = ii(c,p);
      rr(IPM,p) *= f; rr(IPM,q) *= (1.0 - f);
      rr(IPVX,p) += svt_.V; rr(IPVX,q) -= svt_.V;
      const std::uint64_t t = tag0 + static_cast<std::uint64_t>(p)*tstride;
      ii(PTAG,q) = TagLo(t); ii(PTAGH,q) = TagHi(t);
    }
    rr(IPVX1,p) = rr(IPVX,p); rr(IPVY1,p) = rr(IPVY,p); rr(IPVZ1,p) = rr(IPVZ,p);
    if (svt_.two) {
      const int q = np + p;
      rr(IPVX1,q) = rr(IPVX,q); rr(IPVY1,q) = rr(IPVY,q); rr(IPVZ1,q) = rr(IPVZ,q);
    }
  }
  if (nout > part->prtcl_rdata.extent_int(1)) {
    Kokkos::realloc(part->prtcl_rdata, part->nrdata, nout);
    Kokkos::realloc(part->prtcl_idata, part->nidata, nout);
  }
  auto prh = Kokkos::create_mirror_view(part->prtcl_rdata);
  auto pih = Kokkos::create_mirror_view(part->prtcl_idata);
  Kokkos::deep_copy(prh, part->prtcl_rdata);
  Kokkos::deep_copy(pih, part->prtcl_idata);
  for (int p=0; p<nout; ++p) {
    for (int c=0; c<part->nrdata; ++c) prh(c,p) = rr(c,p);
    for (int c=0; c<part->nidata; ++c) pih(c,p) = ii(c,p);
  }
  Kokkos::deep_copy(part->prtcl_rdata, prh);
  Kokkos::deep_copy(part->prtcl_idata, pih);
  if (svt_.two) {
    part->nprtcl_thispack = nout;
    pm->nprtcl_thisrank = nout;
#if MPI_PARALLEL_ENABLED
    MPI_Allgather(&nout, 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
#else
    pm->nprtcl_eachrank[0] = nout;
#endif
    pm->nprtcl_total = 0;
    for (int r=0; r<global_variable::nranks; ++r) pm->nprtcl_total += pm->nprtcl_eachrank[r];
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustDampingAMR()

void ProblemGenerator::DustDampingAMR(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (!dust_setup::DustActive(pmy_mesh_)) Fatal("needs <particles> and <dust>");
  if (pin->GetOrAddString("problem", "dust_test", "damping") != "damping") {
    Fatal("<problem>/dust_test must be damping (dusty_wave_amr is the wave)");
  }
  if (pin->DoesBlockExist("shearing_box")) Fatal("a drag test without rotation: no <shearing_box>");
  dust_setup::RejectKeys(pin, "dust_damping_amr", {
      {"stratified_dust", "false", "dust_si_strat"},
      {"stratified_dust_sampling", "weight", "dust_si_strat"},
      {"dust_zcut", "0", "dust_si_strat"},
      {"nsh_vertical", "false", "dust_si_strat"},
      {"vertical_bc", "none", "dust_si_strat"},
      {"two_stream_v", "0", "dust_nsh"},
      {"two_stream_v_upper", "0", "dust_nsh"},
      {"blob_amp", "0", "dust_nsh"},
      {"gas_vx_amp", "0", "dust_nsh"},
      {"dust_ymode_amp", "0", "dust_nsh"}});
  dust_snapshots::Setup(pin, pmy_mesh_, true);
  dust_setup::SetupHistory(pin);
  user_hist_func = DampingHistory;
  dust_setup::EnrollRefinement(pin, pmy_mesh_, this);
  if (restart) {
    dust_snapshots::Restart(pin, pmy_mesh_);
    dust_setup::LogRestartParticles(pmy_mesh_);
    return;
  }
  if (pmbp->phydro == nullptr) Fatal("needs a <hydro> block");

  std::vector<Real> eps, taus;
  dust_setup::ReadSpecies(pin, pmbp, false, eps, taus);
  const int nspec = static_cast<int>(eps.size());
  const Real v0 = pin->GetOrAddReal("problem", "dust_test_v0", 0.1);
  std::vector<Real> vx(nspec, v0), vp(nspec, 0.0);
  const Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  dust_setup::InitUniformGas(pmy_mesh_, rho0, 0.0, 0.0);

  const bool cloud_root_random = pin->GetOrAddBoolean("problem", "cloud_root_random", false);
  if (pmy_mesh_->three_d && nspec == 1 && cloud_root_random) {   // the 3-D history errors
    dtest3d_.on = true;
    dtest3d_.rho0 = rho0; dtest3d_.eps = eps[0]; dtest3d_.ts = taus[0];
    dtest3d_.v0 = v0;
  }
  dust_setup::DustInit d;
  d.rho0 = rho0;
  d.eps = eps;
  d.vx = vx;
  d.vp = vp;
  d.placement = "lattice";
  d.vz_add = pin->GetOrAddReal("problem", "dust_test_vz0", 0.0);
  d.vz_mirror = pin->GetOrAddBoolean("problem", "dust_test_vz_mirror", false);
  dust_setup::PlaceParticles(pin, pmy_mesh_, d);
  SVTestInit(pin, pmy_mesh_, v0);   // (tests of <dust>/cloud_split_velocity)
  if (svt_.lin || svt_.two) pgen_final_func = SVTestFinal;
  dust_setup::SetParticleDt(pmy_mesh_, vx, vp, 0.0);
}
