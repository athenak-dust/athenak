//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_freefall.cpp
//! \brief Free fall of a cold uniform sphere of dust clouds under its
//! own gravity only, the analytic check before the bound-clump validation.
//!
//! Physics: <gravity>/source = dust, kick = dust (the gas neither feels nor makes gravity); no
//! drag (back_reaction = false and taus_1 >> t_ff); no shear.  The gas stays uniform at rest.
//! (With drag on, the same set-up is a dust-gas collapse without an analytic solution.)
//! The gravity solver is periodic: it removes the mean source rhobar = M/V_box, so a uniform
//! sphere stays uniform and every shell collapses homologously, r(t) = f(t) r(0), with
//!   f'' = -(4 pi G rho_s/3)/f^2 + (4 pi G rhobar/3) f,   f(0) = 1, f'(0) = 0
//! (the periodic images add only a cubic l = 4 term ~ (R/L)^5).  Without the mean, f = 0 at
//! t_ff = pi sqrt(3/(8 rho_s 4piG)).
//!
//! Initial state: one root-level cloud at each point of an ff_n^3 sub-lattice of every root cell
//! whose point lies inside ff_radius of the centre, mass rho_s dx^3/ff_n^3, at rest; then the
//! usual initial split to the leaves.  Diagnostics every ff_dt (and at the end), rank 0, one line
//! "# FREEFALL {json}" and freefall_history.jsonl: the Lagrangian radii r_q (radius about the
//! centre of mass that holds the mass fraction q) over f(t) r_q(0), of the particle centres and of
//! the clouds expanded by repeated P1 splits to ff_repr_level (no jump at a refinement); kinetic and
//! potential energy
//! W = 1/2 sum rho_d (phi - <phi>) V over the leaf cells, the energy drift (K + W - E0)/|W0|; the
//! centre of mass; the peak deposited density; the leaf blocks per level.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "dust/dust.hpp"
#include "dust/dust_species.hpp"
#include "globals.hpp"
#include "gravity/self_gravity.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {
struct FreeFall {
  Real radius = 0.25, rho = 1.0, x = 0.5, y = 0.5, z = 0.5, dt = 0.02, fourpig = 1.0;
  Real rhobar = 0.0, mass = 0.0, tff = 0.0;
  int n = 1;
  int next = 0;
  bool have0 = false;
  Real e0 = 0.0, w0 = 0.0;
  std::vector<Real> q = {0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9};
  std::vector<Real> rq0, rqm0;
  int rep = 0;       // ff_repr_level: the level the clouds are expanded to for rqm
} ff;

void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust_freefall: " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

// f(t) and f'(t) of the homologous collapse with the removed mean (RK4, fine steps)
std::pair<Real, Real> Homologous(Real t) {
  const Real a = ff.fourpig*ff.rho/3.0, b = ff.fourpig*ff.rhobar/3.0;
  auto acc = [a, b](Real f) {return -a/(f*f) + b*f;};
  const int nstep = 20000;
  const Real h = t/nstep;
  Real f = 1.0, v = 0.0;
  for (int s=0; s<nstep && t > 0.0; ++s) {
    const Real k1f = v, k1v = acc(f);
    const Real k2f = v + 0.5*h*k1v, k2v = acc(f + 0.5*h*k1f);
    const Real k3f = v + 0.5*h*k2v, k3v = acc(f + 0.5*h*k2f);
    const Real k4f = v + h*k3v, k4v = acc(f + h*k3f);
    f += h/6.0*(k1f + 2*k2f + 2*k3f + k4f);
    v += h/6.0*(k1v + 2*k2v + 2*k3v + k4v);
    if (!(f > 1.0e-4)) return {0.0, -std::numeric_limits<Real>::infinity()};
  }
  return {f, v};
}

int Owner(MeshBlockPack *pack, const std::array<Real, 3> &p) {
  for (int b=0; b<pack->nmb_thispack; ++b) {
    const auto s = pack->pmb->mb_size.h_view(b);
    if (p[0] >= s.x1min && p[0] < s.x1max && p[1] >= s.x2min && p[1] < s.x2max &&
        p[2] >= s.x3min && p[2] < s.x3max) return b;
  }
  return -1;
}

// radii holding the mass fractions ff.q of the (r, m) pairs (all on rank 0 after Gatherv)
std::vector<Real> Quantiles(const std::vector<double> &rm_local, Real mass) {
  std::vector<double> all;
#if MPI_PARALLEL_ENABLED
  int cnt = static_cast<int>(rm_local.size());
  std::vector<int> counts(global_variable::nranks), displ(global_variable::nranks, 0);
  MPI_Gather(&cnt, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (global_variable::my_rank == 0) {
    for (int r=1; r<global_variable::nranks; ++r) displ[r] = displ[r-1] + counts[r-1];
    all.resize(displ.back() + counts.back());
  }
  MPI_Gatherv(rm_local.data(), cnt, MPI_DOUBLE, all.data(), counts.data(), displ.data(),
              MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
  all = rm_local;
#endif
  std::vector<Real> rq(ff.q.size(), 0.0);
  if (global_variable::my_rank != 0) return rq;
  const std::size_t nall = all.size()/2;
  std::vector<std::pair<double, double>> v(nall);
  for (std::size_t p=0; p<nall; ++p) v[p] = {all[2*p], all[2*p+1]};
  std::sort(v.begin(), v.end());
  double cum = 0.0;
  std::size_t iq = 0;
  for (std::size_t p=0; p<nall && iq<ff.q.size(); ++p) {
    cum += v[p].second;
    while (iq < ff.q.size() && cum >= ff.q[iq]*mass) {rq[iq] = v[p].first; ++iq;}
  }
  return rq;
}

void Observe(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto *dust = pack->pdust;
  auto *grav = pack->pgrav;
  auto *part = pack->ppart;
  if (grav->last_time != pm->time) grav->Solve(pm->time);
  DvceArray5D<Real> rho;
  dust->DepositGravityDensity(pm->time, rho);
  auto rh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rho);
  auto ph = Kokkos::create_mirror_view_and_copy(HostMemSpace(), grav->phi);
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_idata);
  const auto ind = pm->mb_indcs;
  // leaf sums: sum rho V, sum phi V, sum rho phi V, V; peak; blocks per level
  Real s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  Real peak = 0.0;
  for (int b=0; b<pack->nmb_thispack; ++b) {
    const auto sz = pack->pmb->mb_size.h_view(b);
    const Real vol = sz.dx1*sz.dx2*sz.dx3;
    const int lev = pm->lloc_eachmb[pack->gids + b].level - pm->root_level;
    s[4 + std::min(lev, 3)] += 1.0;
    for (int k=ind.ks; k<=ind.ke; ++k) for (int j=ind.js; j<=ind.je; ++j)
      for (int i=ind.is; i<=ind.ie; ++i) {
        const Real d = rh(b,0,k,j,i), p = ph(b,0,k,j,i);
        s[0] += d*vol; s[1] += p*vol; s[2] += d*p*vol; s[3] += vol;
        peak = std::max(peak, d);
      }
  }
  // particles: mass, mass-weighted position, kinetic energy, max speed
  Real a[7] = {0, 0, 0, 0, 0, 0, 0};
  Real vmax = 0.0;
  const int np = part->nprtcl_thispack;
  for (int p=0; p<np; ++p) {
    const Real m = pr(IPM,p);
    a[0] += m; a[1] += m*pr(IPX,p); a[2] += m*pr(IPY,p); a[3] += m*pr(IPZ,p);
    const Real v2 = pr(IPVX,p)*pr(IPVX,p) + pr(IPVY,p)*pr(IPVY,p) + pr(IPVZ,p)*pr(IPVZ,p);
    a[4] += 0.5*m*v2; a[5] += m*(pr(IPVX,p) + pr(IPVY,p) + pr(IPVZ,p));
    vmax = std::max(vmax, std::sqrt(v2));
  }
  a[6] = np;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, s, 8, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, a, 7, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &peak, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  const Real mass = a[0], cm[3] = {a[1]/mass, a[2]/mass, a[3]/mass};
  const Real kin = a[4];
  const Real pot = 0.5*(s[2] - s[0]*s[1]/s[3]);   // 1/2 sum rho (phi - <phi>) V
  // Lagrangian radii about the centre of mass: of the particle centres (rq), and of every cloud
  // expanded into the points of repeated P1 splits down to the level ff_repr_level (per direction
  // offsets -h/2, 0, +h/2 with weights 1/4, 1/2, 1/4, h the half-width; the children of an actual
  // split sit exactly on these points, so these radii (rqm) do not jump at a refinement; lobes of
  // clouds at block edges are ignored)
  std::vector<double> rm(2*static_cast<std::size_t>(np));
  for (int p=0; p<np; ++p) {
    const Real dx = pr(IPX,p) - cm[0], dy = pr(IPY,p) - cm[1], dz = pr(IPZ,p) - cm[2];
    rm[2*p] = std::sqrt(dx*dx + dy*dy + dz*dz); rm[2*p+1] = pr(IPM,p);
  }
  const std::vector<Real> rq = Quantiles(rm, mass);
  const Real rdx[3] = {(pm->mesh_size.x1max - pm->mesh_size.x1min)/pm->mesh_indcs.nx1,
                       (pm->mesh_size.x2max - pm->mesh_size.x2min)/pm->mesh_indcs.nx2,
                       (pm->mesh_size.x3max - pm->mesh_size.x3min)/pm->mesh_indcs.nx3};
  std::vector<std::vector<std::pair<Real, Real>>> table(ff.rep + 1);   // [depth]: (offset/h, weight)
  table[0] = {{0.0, 1.0}};
  for (int d=1; d<=ff.rep; ++d) {
    const Real sc = 1.0/static_cast<Real>(1 << d);
    for (const auto &e : table[d-1]) {
      table[d].push_back({e.first - sc, 0.25*e.second});
      table[d].push_back({e.first, 0.5*e.second});
      table[d].push_back({e.first + sc, 0.25*e.second});
    }
  }
  std::vector<double> cm2;
  for (int p=0; p<np; ++p) {
    const int lev = pi(PLEV,p) & 15;
    const int depth = std::max(0, ff.rep - lev);
    const Real h[3] = {rdx[0]/(1 << lev), rdx[1]/(1 << lev), rdx[2]/(1 << lev)};
    const auto &t = table[depth];
    for (const auto &ez : t) for (const auto &ey : t) for (const auto &ex : t) {
      const Real x = pr(IPX,p) + ex.first*h[0] - cm[0], y = pr(IPY,p) + ey.first*h[1] - cm[1];
      const Real z = pr(IPZ,p) + ez.first*h[2] - cm[2];
      cm2.push_back(std::sqrt(x*x + y*y + z*z));
      cm2.push_back(pr(IPM,p)*ex.second*ey.second*ez.second);
    }
  }
  const std::vector<Real> rqm = Quantiles(cm2, mass);
  if (global_variable::my_rank != 0) return;
  if (!ff.have0) {
    ff.rq0 = rq; ff.rqm0 = rqm; ff.e0 = kin + pot; ff.w0 = pot; ff.have0 = true;
  }
  const auto fv = Homologous(pm->time);
  std::ostringstream o;
  o << std::setprecision(10) << "{\"time\":" << pm->time << ",\"t_over_tff\":" << pm->time/ff.tff
    << ",\"cycle\":" << pm->ncycle << ",\"f\":" << fv.first << ",\"fdot\":";
  if (std::isfinite(fv.second)) o << fv.second; else o << "null";   // after the analytic collapse
  o << ",\"rq_over_f_rq0\":[";
  for (std::size_t n=0; n<rq.size(); ++n) {
    o << (n ? "," : "") << ((fv.first > 0 && ff.rq0[n] > 0) ? rq[n]/(fv.first*ff.rq0[n]) : -1.0);
  }
  o << "],\"rqm_over_f_rqm0\":[";
  for (std::size_t n=0; n<rqm.size(); ++n) {
    o << (n ? "," : "") << ((fv.first > 0 && ff.rqm0[n] > 0) ? rqm[n]/(fv.first*ff.rqm0[n]) : -1.0);
  }
  o << "],\"rqm\":[";
  for (std::size_t n=0; n<rqm.size(); ++n) o << (n ? "," : "") << rqm[n];
  o << "],\"rq\":[";
  for (std::size_t n=0; n<rq.size(); ++n) o << (n ? "," : "") << rq[n];
  o << "],\"K\":" << kin << ",\"W\":" << pot << ",\"dE_over_W0\":" << (kin + pot - ff.e0)/std::fabs(ff.w0)
    << ",\"mass\":" << mass << ",\"leaf_mass\":" << s[0] << ",\"cm\":[" << cm[0] - ff.x << ","
    << cm[1] - ff.y << "," << cm[2] - ff.z << "],\"momentum_sum\":" << a[5] << ",\"vmax\":" << vmax
    << ",\"rho_peak\":" << peak << ",\"rho_uniform_analytic\":"
    << ((fv.first > 0) ? ff.rho/(fv.first*fv.first*fv.first) : -1.0)
    << ",\"particles\":" << static_cast<long long>(a[6]) << ",\"blocks_per_level\":[" << s[4] << "," << s[5] << "," << s[6]
    << "," << s[7] << "],\"residual\":" << grav->last_solve.relative_residual << "}";
  std::cout << "# FREEFALL " << o.str() << std::endl;
  std::ofstream out("freefall_history.jsonl", std::ios::app);
  out << o.str() << "\n";
}

void Work(Mesh *pm) {
  if (pm->time + 1.0e-12 >= ff.next*ff.dt) {
    Observe(pm);
    ff.next = static_cast<int>(std::floor(pm->time/ff.dt + 1.0e-9)) + 1;
  }
}

void Final(ParameterInput *, Mesh *pm) {Observe(pm);}
}  // namespace

void ProblemGenerator::DustFreeFall(ParameterInput *pin, const bool restart) {
  dust::RequireLegacySpeciesInit(pin, "dust_freefall");
  auto *pack = pmy_mesh_->pmb_pack;
  if (pack->pdust == nullptr || pack->ppart == nullptr || pack->pgrav == nullptr ||
      !pmy_mesh_->three_d || !pack->pdust->Clouds3D() || pack->pdust->nspecies != 1) {
    Fatal("needs 3D refinable clouds (one species) and <gravity>");
  }
  if (pin->GetOrAddReal("problem", "etavk", 0.0) != 0.0 ||
      pin->GetString("gravity", "source") != "dust" || pin->GetString("gravity", "kick") != "dust") {
    Fatal("needs etavk = 0, <gravity>/source = kick = dust");
  }
  // With drag (back_reaction = true, a finite taus_1) the sphere collapses through the
  // gas; f(t) then is only the drag-free reference, and runs are compared with each other
  // (AMR against uniform fine)
  ff.fourpig = pin->GetReal("gravity", "four_pi_G");
  ff.radius = pin->GetOrAddReal("problem", "ff_radius", 0.25);
  // default: t_ff = 1 without the mean
  ff.rho = pin->GetOrAddReal("problem", "ff_rho", 3.0*M_PI*M_PI/(8.0*ff.fourpig));
  const auto &box = pmy_mesh_->mesh_size;
  ff.x = pin->GetOrAddReal("problem", "ff_x", 0.5*(box.x1min + box.x1max));
  ff.y = pin->GetOrAddReal("problem", "ff_y", 0.5*(box.x2min + box.x2max));
  ff.z = pin->GetOrAddReal("problem", "ff_z", 0.5*(box.x3min + box.x3max));
  ff.dt = pin->GetOrAddReal("problem", "ff_dt", 0.02);
  ff.n = pin->GetOrAddInteger("problem", "ff_n", 1);
  ff.rep = pin->GetOrAddInteger("problem", "ff_repr_level", pmy_mesh_->max_level - pmy_mesh_->root_level);
  if (ff.rep < 0 || ff.rep > 4) Fatal("ff_repr_level must be 0..4");
  ff.tff = M_PI*std::sqrt(3.0/(8.0*ff.rho*ff.fourpig));
  if (!(ff.radius > 0 && ff.rho > 0 && ff.dt > 0 && ff.n >= 1)) Fatal("invalid ff_* parameters");
  {   // the gas is irrelevant (no drag, no gravity): the damping fixture at rest
    const bool split = pack->pdust->cloud_split_enable;
    pack->pdust->cloud_split_enable = false;
    DustNSH(pin, restart);
    pack->pdust->cloud_split_enable = split;
  }
  user_work_in_loop_func = Work;
  pgen_final_func = Final;
  const Real lx = box.x1max - box.x1min, ly = box.x2max - box.x2min, lz = box.x3max - box.x3min;
  const int nx = pmy_mesh_->mesh_indcs.nx1, ny = pmy_mesh_->mesh_indcs.nx2,
            nz = pmy_mesh_->mesh_indcs.nx3;
  const Real dx[3] = {lx/nx, ly/ny, lz/nz};
  const Real mcl = ff.rho*dx[0]*dx[1]*dx[2]/(ff.n*ff.n*ff.n);
  // the mass the solver removes: the sphere's actual (lattice) mass over the box
  long long ninside = 0;
  auto inside = [&](const std::array<Real, 3> &p) {
    const Real a = p[0] - ff.x, b = p[1] - ff.y, c = p[2] - ff.z;
    return a*a + b*b + c*c < ff.radius*ff.radius;
  };
  auto point = [&](int i, int j, int k, int si, int sj, int sk) {
    return std::array<Real, 3>{box.x1min + (i + (si + 0.5)/ff.n)*dx[0],
                               box.x2min + (j + (sj + 0.5)/ff.n)*dx[1],
                               box.x3min + (k + (sk + 0.5)/ff.n)*dx[2]};
  };
  struct Sample {std::array<Real, 3> x; long long tag; int block;};
  std::vector<Sample> local;
  for (int k=0; k<nz; ++k) for (int j=0; j<ny; ++j) for (int i=0; i<nx; ++i)
    for (int sk=0; sk<ff.n; ++sk) for (int sj=0; sj<ff.n; ++sj) for (int si=0; si<ff.n; ++si) {
      const auto p = point(i, j, k, si, sj, sk);
      if (!inside(p)) continue;
      ++ninside;
      if (restart) continue;
      const int b = Owner(pack, p);
      const long long tag = ((static_cast<long long>(k)*ny + j)*nx + i)*ff.n*ff.n*ff.n +
                            (sk*ff.n + sj)*ff.n + si;
      if (b >= 0) local.push_back({p, tag, b});
    }
  ff.mass = mcl*ninside;
  ff.rhobar = ff.mass/(lx*ly*lz);
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(12) << "# FREEFALL_SETUP radius=" << ff.radius << " rho_s=" << ff.rho
              << " clouds=" << ninside << " mass=" << ff.mass << " R_eff="
              << std::cbrt(3.0*ff.mass/(4.0*M_PI*ff.rho)) << " rhobar=" << ff.rhobar
              << " t_ff=" << ff.tff << " n=" << ff.n << " dx=" << dx[0] << std::endl;
  }
  if (restart) {
    ff.next = static_cast<int>(std::floor(pmy_mesh_->time/ff.dt + 1.0e-9)) + 1;
    return;
  }
  auto *part = pack->ppart;
  const int count = static_cast<int>(local.size());
  Kokkos::realloc(part->prtcl_rdata, part->nrdata, std::max(count, 1));
  Kokkos::realloc(part->prtcl_idata, part->nidata, std::max(count, 1));
  auto pr = Kokkos::create_mirror_view(part->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view(part->prtcl_idata);
  for (int p=0; p<count; ++p) {
    for (int v=0; v<part->nrdata; ++v) pr(v,p) = 0.0;
    for (int v=0; v<part->nidata; ++v) pi(v,p) = 0;
    const auto &s = local[p];
    pi(PGID,p) = pack->gids + s.block;
    pi(PTAG,p) = static_cast<int>(s.tag);
    pi(PLEV,p) = 0; pi(PSP,p) = 0;
    pr(IPX,p) = pr(IPX1,p) = s.x[0]; pr(IPY,p) = pr(IPY1,p) = s.x[1];
    pr(IPZ,p) = pr(IPZ1,p) = s.x[2];
    pr(IPM,p) = mcl; pr(IPTS,p) = pack->pdust->taus.h_view(0);
  }
  Kokkos::deep_copy(part->prtcl_rdata, pr);
  Kokkos::deep_copy(part->prtcl_idata, pi);
  part->nprtcl_thispack = count; pmy_mesh_->nprtcl_thisrank = count;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&count, 1, MPI_INT, pmy_mesh_->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
#else
  pmy_mesh_->nprtcl_eachrank[0] = count;
#endif
  pmy_mesh_->nprtcl_total = 0;
  for (int r=0; r<global_variable::nranks; ++r) pmy_mesh_->nprtcl_total += pmy_mesh_->nprtcl_eachrank[r];
  if (pmy_mesh_->nprtcl_total != ninside) Fatal("lattice clouds were not owned exactly once");
  pack->pdust->cloud_merge_mroot.assign(1, mcl);
  pack->pdust->cloud_resample_m0 = mcl;
  pack->pdust->EnsureNextTag();
  pack->pdust->SplitToLeaves();
  part->dtnew = std::numeric_limits<float>::max();
}
