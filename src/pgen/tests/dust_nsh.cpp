//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_nsh.cpp
//! \brief Problem generator for the multi-species Nakagawa-Sei-Hayashi (NSH) drift
//! equilibrium test in an unstratified shearing box (Nakagawa et al. 1986;
//! Benitez-Llambay et al. 2019 Sec. 3.4; Krapp et al. 2024 Sec. 3.6). The gas feels a
//! constant radial acceleration a_x = 2*Omega*etavk (the pressure-gradient mimic,
//! applied through the standard <hydro_srcterms> const_accel machinery so that dust
//! particles do not feel it), and gas + N dust species settle into uniform steady drift
//! velocities that solve the linear system
//!     0 =  2*Omega*u_phi + a_x + sum_s eps_s*(v_x,s - u_x)/tau_s
//!     0 = -(2-q)*Omega*u_x     + sum_s eps_s*(v_phi,s - u_phi)/tau_s
//!     0 =  2*Omega*v_phi,s - (v_x,s - u_x)/tau_s
//!     0 = -(2-q)*Omega*v_x,s - (v_phi,s - u_phi)/tau_s
//! (velocities relative to the background shear). By default the problem uses a quiet
//! particle lattice, initializes this equilibrium exactly, and verifies it is HELD TO
//! ROUND-OFF: because the
//! rotation/shear/forcing kicks and the implicit drag solve are composed unsplit inside
//! the IMEX stages, the discrete update has the continuum equilibrium as an exact fixed
//! point (any Strang-like splitting error would appear as secular drift).
//! <problem>/particle_placement=random instead gives a reproducible warm start at the
//! same NSH velocities for nonlinear streaming-instability calculations.
//! A physical dust-free equilibrium is obtained by omitting both <particles> and <dust>.
//! In that mode there are no dust species or stopping times, so the solution reduces to
//! u_x=0 and u_phi=-etavk.
//!
//! Works in the 2D r-z shearing box (azimuthal components in IM3/IPVZ, plain periodic
//! boundaries) and in 3D (azimuthal in IM2/IPVY, shear-periodic x1 boundaries).
//! Errors written to "<basename>-errs.dat"; per-species drift history available with
//! <problem>/user_hist = true.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>     // fopen(), fprintf()
#include <unistd.h>   // truncate()
#include <iostream>
#include <limits>
#include <string>
#include <vector>

// Athena++ headers
#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs/outputs.hpp"

// functions for error output and user history
void DustNSHErrors(ParameterInput *pin, Mesh *pm);
void DustNSHHistory(HistoryData *pdata, Mesh *pm);
void RunAuditReplay(ParameterInput *pin, Mesh *pm);   // streaming_linear.cpp

namespace {

bool ba_snapshots_ = false;
int ba_snapshot_number_ = 0;
// native restart: the numbers of the next snapshot and field frame are kept
// in the input block <problem> (rst_ba_snapshot_number, rst_ba_field_number), which every
// restart file carries; the distribution file is continued after a restart
ParameterInput *ba_pin_ = nullptr;
bool ba_ppdf_resume_ = false;
Real ba_snapshot_dt_ = 40.0;
std::string ba_snapshot_basename_;

// Diagnostic only: the v2 LinA layout retains the exact particle masses and P1
// lobe codes needed to reconstruct the cloud integral on every active leaf.
void WriteBASnapshot(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto &ind = pm->mb_indcs;
  auto gas = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pack->phydro->u0);
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                               pack->ppart->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                               pack->ppart->prtcl_idata);
  int ncell = pack->nmb_thispack*ind.nx1*ind.nx2;
  int npart = pack->ppart->nprtcl_thispack;
  std::vector<double> cells;
  std::vector<double> particles;
  cells.reserve(8*ncell);
  particles.reserve(8*npart);
  for (int m=0; m<pack->nmb_thispack; ++m) {
    const auto &size = pack->pmb->mb_size.h_view(m);
    for (int j=ind.js; j<=ind.je; ++j) {
      for (int i=ind.is; i<=ind.ie; ++i) {
        cells.insert(cells.end(), {size.x1min + (i-ind.is+0.5)*size.dx1,
            size.x2min + (j-ind.js+0.5)*size.dx2, size.dx1, size.dx2,
            gas(m,IDN,ind.ks,j,i), gas(m,IM1,ind.ks,j,i),
            gas(m,IM2,ind.ks,j,i), gas(m,IM3,ind.ks,j,i)});
      }
    }
  }
  for (int p=0; p<npart; ++p) {
    particles.insert(particles.end(), {pr(IPX,p), pr(IPY,p), pr(IPM,p),
        pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p), static_cast<double>(pi(PTAG,p)),
        static_cast<double>(pi(PLEV,p))});
  }
  char number[32];
  std::snprintf(number, sizeof(number), ".%05d.ba", ba_snapshot_number_++);
  if (ba_pin_ != nullptr) ba_pin_->SetInteger("problem", "rst_ba_snapshot_number", ba_snapshot_number_);
  std::string filename = ba_snapshot_basename_ + number;
  FILE *file = std::fopen(filename.c_str(), "wb");
  if (file == nullptr) {
    std::cerr << "Cannot open BA snapshot " << filename << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool ok = std::fprintf(file, "AthenaK BA snapshot v2 f64-native\n"
      "%.17g %d %d %d\n", pm->time, pm->ncycle, ncell, npart) > 0;
  ok = (std::fwrite(cells.data(), sizeof(double), cells.size(), file)
        == cells.size()) && ok;
  ok = (std::fwrite(particles.data(), sizeof(double), particles.size(), file)
        == particles.size()) && ok;
  ok = (std::fclose(file) == 0) && ok;
  if (!ok) {
    std::cerr << "Failed writing BA snapshot " << filename << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

bool ba_fields_ = false;
bool ba_history_ = false;
int ba_field_number_ = 0;
bool ba_geom_written_ = false;   // the geometry file is written with the first field file
int ba_geom_seq_ = -1;           // mesh update sequence of the geometry last written
Real ba_field_dt_ = 1.0;
constexpr int kBAFields = 11;

// Diagnostic only: the distribution of the dust density measured at
// the particle positions, weighted by the mass each particle represents, appended to
// <basename>.ppdf with every field snapshot.  The leaf densities are the run's own deposit
// (WriteBAFields).  Two estimates per particle, both a normalised point-TSC interpolation
// of a uniform periodic grid at the particle centre (self contribution included):
//   native   the grid of the level of the leaf that holds the particle; finer leaves are
//            restricted to it conservatively, coarser leaves prolonged piecewise constant
//   common   the reference grid of ba_ppdf_nref cells per x1 extent (leaves restricted)
// Regions by particle centre: all; x1 below / above the middle of the box; the same two
// without the bands of two reference cells around the middle and the box edge.
// Record: kPPScal doubles (time, cycle, particles, mass, mean density, largest native and
// common estimate; for native then common the mass fraction above 1, 3, 10, 30, 100, 300;
// the mass fractions of the four regions), then 2 x 5 x (kPPBins + 2) floats: mass
// fraction per bin of log10(density / mean), first / last bin = below / above the range.
bool ba_ppdf_ = false;
bool ba_fields_write_ = true;
bool ba_fields_global_test_ = false;   // test: the several-rank writer on one rank
bool ba_ppdf_header_ = false;
int ba_ppdf_nref_ = 128;
constexpr int kPPBins = 1000, kPPScal = 23;
constexpr double kPPLo = -6.0, kPPHi = 4.0;

// The particle loop and the record of WriteBAParticlePDF, on grids of the whole mesh (grid[L]: the
// density on level L, lmap: the level of the leaf of every finest cell, ref: the reference grid).
template <typename PrtclHost>
void FinishBAParticlePDF(Mesh *pm, const std::vector<std::vector<double>> &grid,
                         const std::vector<signed char> &lmap, const int nlev,
                         const std::vector<double> &ref, const int nr1, const int nr2,
                         const PrtclHost &prh, const int npart, const double mtot) {
  const double x1min = pm->mesh_size.x1min, x2min = pm->mesh_size.x2min;
  const double lx = pm->mesh_size.x1max - x1min, ly = pm->mesh_size.x2max - x2min;
  const int n1 = pm->mesh_indcs.nx1, n2 = pm->mesh_indcs.nx2;
  const int nf1 = n1 << nlev, nf2 = n2 << nlev;
  const double hr1 = lx/nr1;
  auto tsc = [&](const std::vector<double> &g, const int m1, const int m2, const double x,
                 const double y) {
    const double q1 = (x - x1min)/(lx/m1) - 0.5, q2 = (y - x2min)/(ly/m2) - 0.5;
    const int c1 = static_cast<int>(std::floor(q1 + 0.5)), c2 = static_cast<int>(std::floor(q2 + 0.5));
    const double d1 = q1 - c1, d2 = q2 - c2;
    const double w1[3] = {0.5*SQR(0.5-d1), 0.75-d1*d1, 0.5*SQR(0.5+d1)};
    const double w2[3] = {0.5*SQR(0.5-d2), 0.75-d2*d2, 0.5*SQR(0.5+d2)};
    double v = 0.0;
    for (int b=0; b<3; ++b) {
      const int jj = ((c2 + b - 1) % m2 + m2) % m2;
      for (int a=0; a<3; ++a) v += w2[b]*w1[a]*g[static_cast<std::size_t>(jj)*m1 + ((c1 + a - 1) % m1 + m1) % m1];
    }
    return v;
  };
  const double rbar = mtot/(lx*ly), band = 2.0*hr1, xmid = x1min + 0.5*lx;
  const int nb2 = kPPBins + 2;
  std::vector<double> hist(static_cast<std::size_t>(10)*nb2, 0.0);
  const double thr[6] = {1.0, 3.0, 10.0, 30.0, 100.0, 300.0};
  double above[2][6] = {{0.0}}, rmax[2] = {0.0, 0.0}, mreg[4] = {0.0, 0.0, 0.0, 0.0};
  for (int p=0; p<npart; ++p) {
    const double mass = prh(IPM,p);
    if (!(mass > 0.0)) continue;
    double x = prh(IPX,p), y = prh(IPY,p);
    x -= lx*std::floor((x - x1min)/lx); y -= ly*std::floor((y - x2min)/ly);
    const int fi = std::min(nf1 - 1, static_cast<int>((x - x1min)/(lx/nf1)));
    const int fj = std::min(nf2 - 1, static_cast<int>((y - x2min)/(ly/nf2)));
    const int L = lmap[static_cast<std::size_t>(fj)*nf1 + fi];
    const double r[2] = {tsc(grid[L], n1 << L, n2 << L, x, y)/rbar, tsc(ref, nr1, nr2, x, y)/rbar};
    const double dedge = std::min(x - x1min, x1min + lx - x), dmid = std::fabs(x - xmid);
    const bool left = x < xmid, clear = dedge > band && dmid > band;
    const int reg[3] = {0, left ? 1 : 2, clear ? (left ? 3 : 4) : -1};
    mreg[left ? 0 : 1] += mass;
    if (clear) mreg[left ? 2 : 3] += mass;
    for (int e=0; e<2; ++e) {
      int bin = 0;
      if (r[e] > 0.0) {
        const double sb = (std::log10(r[e]) - kPPLo)/(kPPHi - kPPLo)*kPPBins;
        bin = (sb < 0.0) ? 0 : ((sb >= kPPBins) ? kPPBins + 1 : 1 + static_cast<int>(sb));
      }
      for (int k=0; k<3; ++k) if (reg[k] >= 0) hist[(static_cast<std::size_t>(e)*5 + reg[k])*nb2 + bin] += mass;
      for (int k=0; k<6; ++k) if (r[e] > thr[k]) above[e][k] += mass;
      rmax[e] = std::max(rmax[e], r[e]);
    }
  }
  // several ranks (2026-10-17, U2048): every rank evaluates its particles on the global grids;
  // the sums go to rank 0, which writes the record
  long long ntot = npart;
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, hist.data(), static_cast<int>(hist.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &above[0][0], 12, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, mreg, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, rmax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &ntot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank != 0) return;
  double scal[kPPScal] = {pm->time, static_cast<double>(pm->ncycle), static_cast<double>(ntot),
                          mtot, rbar, rmax[0], rmax[1]};
  for (int e=0; e<2; ++e) for (int k=0; k<6; ++k) scal[7 + 6*e + k] = above[e][k]/mtot;
  for (int k=0; k<4; ++k) scal[19 + k] = mreg[k]/mtot;
  std::vector<float> out(hist.size());
  for (std::size_t k=0; k<hist.size(); ++k) out[k] = static_cast<float>(hist[k]/mtot);
  const std::string name = ba_snapshot_basename_ + ".ppdf";
  if (ba_ppdf_resume_) {
    // first record after a restart: keep the leading records that are earlier than this one
    // (a file written past the restart point is cut back), then append.  The records are
    // told by their time, not by the frame number: a run continued from a saved state does
    // not start its file at frame zero.
    ba_ppdf_resume_ = false;
    FILE *old = std::fopen(name.c_str(), "rb");
    if (old != nullptr) {
      char line[256];
      const bool two = std::fgets(line, sizeof(line), old) != nullptr &&
                       std::fgets(line, sizeof(line), old) != nullptr;
      const long head = two ? std::ftell(old) : -1;
      std::fseek(old, 0, SEEK_END);
      const long size = std::ftell(old);
      const long rec = static_cast<long>(kPPScal*sizeof(double) + hist.size()*sizeof(float));
      long keep = 0;
      if (head > 0) {
        const long nrec = (size - head)/rec;
        const double tnow = scal[0];
        for (long k=0; k<nrec; ++k) {
          double tk = 0.0;
          std::fseek(old, head + k*rec, SEEK_SET);
          if (std::fread(&tk, sizeof(double), 1, old) != 1 ||
              !(tk < tnow - 1.0e-12*std::fabs(tnow))) break;
          keep = k + 1;
        }
      }
      std::fclose(old);
      if (head > 0) {
        if (size != head + rec*keep && truncate(name.c_str(), head + rec*keep) != 0) {
          std::cerr << "Cannot cut " << name << " back to the restart point" << std::endl;
          std::exit(EXIT_FAILURE);
        }
        ba_ppdf_header_ = true;
      }
    }
  }
  FILE *file = std::fopen(name.c_str(), ba_ppdf_header_ ? "ab" : "wb");
  bool ok = file != nullptr;
  if (ok && !ba_ppdf_header_) {
    ok = std::fprintf(file, "AthenaK BA particle density distribution v1\n%d 2 5 %d %.17g %.17g %d\n",
                      kPPScal, kPPBins, kPPLo, kPPHi, nr1) > 0;
    ba_ppdf_header_ = true;
  }
  ok = ok && std::fwrite(scal, sizeof(double), kPPScal, file) == kPPScal;
  ok = ok && std::fwrite(out.data(), sizeof(float), out.size(), file) == out.size();
  ok = (file != nullptr) && (std::fclose(file) == 0) && ok;
  if (!ok) {
    std::cerr << "Failed writing BA particle distribution " << name << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <typename FieldHost, typename PrtclHost>
void WriteBAParticlePDF(Mesh *pm, const FieldHost &fh, const PrtclHost &prh, const int npart,
                        const double mtot) {
  auto *pack = pm->pmb_pack;
  auto &ind = pm->mb_indcs;
  auto &mbsize = pack->pmb->mb_size;
  const int nmb = pack->nmb_thispack, is = ind.is, js = ind.js, nx1 = ind.nx1, nx2 = ind.nx2;
  const double x1min = pm->mesh_size.x1min, x2min = pm->mesh_size.x2min;
  const double lx = pm->mesh_size.x1max - x1min, ly = pm->mesh_size.x2max - x2min;
  const int n1 = pm->mesh_indcs.nx1, n2 = pm->mesh_indcs.nx2;
  const double rdx = lx/n1;
  std::vector<int> blev(nmb);
  int nlev = 0;
  for (int m=0; m<nmb; ++m) {
    blev[m] = static_cast<int>(std::lround(std::log2(rdx/mbsize.h_view(m).dx1)));
    nlev = std::max(nlev, blev[m]);
  }
  const int nr1 = ba_ppdf_nref_;
  const int nr2 = static_cast<int>(std::lround(nr1*ly/lx));
  const double hr1 = lx/nr1, hr2 = ly/nr2;
  if (nlev > 4 || blev.empty() || rdx > hr1*(1.0 + 1.0e-9) || n1 % nr1 != 0) {
    std::cerr << "BA particle distribution: needs at most 4 levels and a reference grid"
              << " (ba_ppdf_nref) that divides the root grid" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nf1 = n1 << nlev, nf2 = n2 << nlev;
  std::vector<std::vector<double>> grid(nlev + 1);
  for (int L=0; L<=nlev; ++L) grid[L].assign(static_cast<std::size_t>(n1 << L)*(n2 << L), 0.0);
  std::vector<signed char> lmap(static_cast<std::size_t>(nf1)*nf2, 0);
  std::vector<double> ref(static_cast<std::size_t>(nr1)*nr2, 0.0);
  for (int m=0; m<nmb; ++m) {
    const auto &size = mbsize.h_view(m);
    const int l = blev[m];
    const int i0 = static_cast<int>(std::lround((size.x1min - x1min)/size.dx1));
    const int j0 = static_cast<int>(std::lround((size.x2min - x2min)/size.dx2));
    for (int j=0; j<nx2; ++j) {
      for (int i=0; i<nx1; ++i) {
        const double rho = fh(m,0,js+j,is+i);
        const int ci = i0 + i, cj = j0 + j;
        for (int L=0; L<=nlev; ++L) {
          const int m1 = n1 << L;
          if (L <= l) {
            const int sh = l - L;
            grid[L][static_cast<std::size_t>(cj >> sh)*m1 + (ci >> sh)] += rho/static_cast<double>(1 << (2*sh));
          } else {
            const int sh = L - l, w = 1 << sh;
            for (int b=0; b<w; ++b)
              for (int a=0; a<w; ++a)
                grid[L][static_cast<std::size_t>((cj << sh) + b)*m1 + (ci << sh) + a] = rho;
          }
        }
        const int sh = nlev - l, w = 1 << sh;
        for (int b=0; b<w; ++b)
          for (int a=0; a<w; ++a)
            lmap[static_cast<std::size_t>((cj << sh) + b)*nf1 + (ci << sh) + a] = static_cast<signed char>(l);
        const int ir = std::min(nr1 - 1, static_cast<int>((size.x1min + (i + 0.5)*size.dx1 - x1min)/hr1));
        const int jr = std::min(nr2 - 1, static_cast<int>((size.x2min + (j + 0.5)*size.dx2 - x2min)/hr2));
        ref[static_cast<std::size_t>(jr)*nr1 + ir] += rho*size.dx1*size.dx2/(hr1*hr2);
      }
    }
  }
  FinishBAParticlePDF(pm, grid, lmap, nlev, ref, nr1, nr2, prh, npart, mtot);
}

// Several ranks (2026-10-17, U2048): the field snapshot of a uniform mesh with the canonical
// TSC deposit.  Every rank deposits its particles on the global periodic grid; the sums are
// reduced (mass and momentum to all ranks for the second pass, the central moments and the gas
// to rank 0), and rank 0 writes the cells in row order (x fastest) with that geometry, in the
// formats of WriteBAFields.  The particle distribution is evaluated by every rank on the global
// density and reduced (FinishBAParticlePDF).
void WriteBAFieldsGlobal(Mesh *pm) {
#if MPI_PARALLEL_ENABLED
  auto *pack = pm->pmb_pack;
  auto *pdust = pack->pdust;
  auto &ind = pm->mb_indcs;
  const int nmb = pack->nmb_thispack, is = ind.is, js = ind.js, ks = ind.ks;
  const int nx1 = ind.nx1, nx2 = ind.nx2;
  auto &pr = pack->ppart->prtcl_rdata;
  const int npart = pack->ppart->nprtcl_thispack;
  auto &mbsize = pack->pmb->mb_size;
  if (pm->multilevel || pdust->UsesCompositeTransfer() || pdust->deposit != DustDeposit::tsc) {
    std::cerr << "BA fields on several ranks: only the canonical TSC deposit on a uniform mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int n1 = pm->mesh_indcs.nx1, n2 = pm->mesh_indcs.nx2;
  const Real gx1 = pm->mesh_size.x1min, gx2 = pm->mesh_size.x2min;
  const Real h1 = (pm->mesh_size.x1max - gx1)/n1, h2 = (pm->mesh_size.x2max - gx2)/n2;
  const std::size_t ncell = static_cast<std::size_t>(n1)*n2;
  DvceArray3D<Real> g0("ba_g0", 4, n2, n1), g1("ba_g1", 3, n2, n1);
  Kokkos::deep_copy(DevExeSpace(), g0, 0.0);
  Kokkos::deep_copy(DevExeSpace(), g1, 0.0);
  const Real inv_area = 1.0/(h1*h2);
  for (int pass=0; pass<2; ++pass) {
    par_for("ba_fields_global", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
      const Real mass = pr(IPM,p);
      const Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
      const Real q1 = (pr(IPX,p) - gx1)/h1 - 0.5, q2 = (pr(IPY,p) - gx2)/h2 - 0.5;
      const int c1 = static_cast<int>(floor(q1 + 0.5)), c2 = static_cast<int>(floor(q2 + 0.5));
      const Real d1 = q1 - c1, d2 = q2 - c2;
      const Real w1[3] = {0.5*SQR(0.5-d1), 0.75-d1*d1, 0.5*SQR(0.5+d1)};
      const Real w2[3] = {0.5*SQR(0.5-d2), 0.75-d2*d2, 0.5*SQR(0.5+d2)};
      for (int b=0; b<3; ++b) {
        const int jj = ((c2 + b - 1) % n2 + n2) % n2;
        for (int a=0; a<3; ++a) {
          const int ii = ((c1 + a - 1) % n1 + n1) % n1;
          const Real d = mass*w1[a]*w2[b]*inv_area;
          if (pass == 0) {
            dust::DepositAdd(&g0(0,jj,ii), d);
            for (int c=0; c<3; ++c) dust::DepositAdd(&g0(1+c,jj,ii), d*v[c]);
          } else {
            const Real rho = g0(0,jj,ii);
            for (int c=0; c<3; ++c) {
              const Real dv = v[c] - g0(1+c,jj,ii)/rho;
              dust::DepositAdd(&g1(c,jj,ii), d*dv*dv);
            }
          }
        }
      }
    });
    if (pass == 0) {   // the mass and momentum of all ranks, on every rank
      auto h0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g0);
      MPI_Allreduce(MPI_IN_PLACE, h0.data(), static_cast<int>(4*ncell), MPI_ATHENA_REAL, MPI_SUM,
                    MPI_COMM_WORLD);
      Kokkos::deep_copy(g0, h0);
    }
  }
  auto h0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g0);
  auto h1v = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g1);
  const bool root = global_variable::my_rank == 0;
  MPI_Reduce(root ? MPI_IN_PLACE : h1v.data(), h1v.data(), static_cast<int>(3*ncell), MPI_ATHENA_REAL,
             MPI_SUM, 0, MPI_COMM_WORLD);
  // the gas of every block in the global grid (zero elsewhere), summed on rank 0
  std::vector<double> gas(4*ncell, 0.0);
  auto u0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pack->phydro->u0);
  for (int m=0; m<nmb; ++m) {
    const auto &size = mbsize.h_view(m);
    const int i0 = static_cast<int>(std::lround((size.x1min - gx1)/h1));
    const int j0 = static_cast<int>(std::lround((size.x2min - gx2)/h2));
    for (int j=0; j<nx2; ++j)
      for (int i=0; i<nx1; ++i) {
        const std::size_t k = static_cast<std::size_t>(j0 + j)*n1 + i0 + i;
        gas[k] = u0(m,IDN,ks,js+j,is+i); gas[ncell + k] = u0(m,IM1,ks,js+j,is+i);
        gas[2*ncell + k] = u0(m,IM2,ks,js+j,is+i); gas[3*ncell + k] = u0(m,IM3,ks,js+j,is+i);
      }
  }
  MPI_Reduce(root ? MPI_IN_PLACE : gas.data(), gas.data(), static_cast<int>(4*ncell), MPI_DOUBLE, MPI_SUM,
             0, MPI_COMM_WORLD);
  auto prh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
  double mtot = 0.0;
  long long ntot = npart;
  for (int p=0; p<npart; ++p) mtot += prh(IPM,p);
  MPI_Allreduce(MPI_IN_PLACE, &mtot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ntot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  double mdep = 0.0;
  for (std::size_t k=0; k<ncell; ++k) mdep += h0(0, k/n1, k%n1)*h1*h2;
  if (std::fabs(mdep - mtot) > 1.0e-8*mtot) {
    if (root) std::fprintf(stderr, "BA fields: deposited mass %.17g != particle mass %.17g\n", mdep, mtot);
    std::exit(EXIT_FAILURE);
  }
  if (ba_ppdf_) {
    const int nr1 = ba_ppdf_nref_;
    const double lx = pm->mesh_size.x1max - gx1, ly = pm->mesh_size.x2max - gx2;
    const int nr2 = static_cast<int>(std::lround(nr1*ly/lx));
    if (n1 % nr1 != 0) {
      std::cerr << "BA particle distribution: ba_ppdf_nref must divide the root grid" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const double hr1 = lx/nr1, hr2 = ly/nr2;
    std::vector<std::vector<double>> grid(1, std::vector<double>(ncell));
    std::vector<signed char> lmap(ncell, 0);
    std::vector<double> ref(static_cast<std::size_t>(nr1)*nr2, 0.0);
    for (int j=0; j<n2; ++j)
      for (int i=0; i<n1; ++i) {
        const double rho = h0(0,j,i);
        grid[0][static_cast<std::size_t>(j)*n1 + i] = rho;
        const int ir = std::min(nr1 - 1, static_cast<int>((i + 0.5)*h1/hr1));
        const int jr = std::min(nr2 - 1, static_cast<int>((j + 0.5)*h2/hr2));
        ref[static_cast<std::size_t>(jr)*nr1 + ir] += rho*h1*h2/(hr1*hr2);
      }
    FinishBAParticlePDF(pm, grid, lmap, 0, ref, nr1, nr2, prh, npart, mtot);
  }
  const int number = ba_field_number_++;
  if (ba_pin_ != nullptr) ba_pin_->SetInteger("problem", "rst_ba_field_number", ba_field_number_);
  if (!ba_fields_write_ || !root) return;
  auto write = [](const std::string &name, const std::string &header, const void *data,
                  std::size_t size, std::size_t count) {
    FILE *file = std::fopen(name.c_str(), "wb");
    bool ok = (file != nullptr) && std::fputs(header.c_str(), file) >= 0;
    ok = ok && std::fwrite(data, size, count, file) == count;
    ok = (file != nullptr) && (std::fclose(file) == 0) && ok;
    if (!ok) {
      std::cerr << "Failed writing BA field file " << name << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };
  char line[160];
  if (!ba_geom_written_) {   // a uniform mesh: once
    std::vector<double> geom;
    geom.reserve(4*ncell);
    for (int j=0; j<n2; ++j)
      for (int i=0; i<n1; ++i) geom.insert(geom.end(), {gx1 + (i + 0.5)*h1, gx2 + (j + 0.5)*h2, h1, h2});
    std::snprintf(line, sizeof(line), "AthenaK BA field geometry v1 f64\n%zu\n", ncell);
    write(ba_snapshot_basename_ + ".bafield.geom", line, geom.data(), sizeof(double), geom.size());
    ba_geom_written_ = true;
  }
  std::vector<float> out;
  out.reserve(static_cast<std::size_t>(kBAFields)*ncell);
  for (int j=0; j<n2; ++j)
    for (int i=0; i<n1; ++i) {
      const std::size_t k = static_cast<std::size_t>(j)*n1 + i;
      out.insert(out.end(), {static_cast<float>(gas[k]), static_cast<float>(gas[ncell + k]),
                             static_cast<float>(gas[2*ncell + k]), static_cast<float>(gas[3*ncell + k])});
      for (int c=0; c<4; ++c) out.push_back(static_cast<float>(h0(c,j,i)));
      for (int c=0; c<3; ++c) out.push_back(static_cast<float>(h1v(c,j,i)));
    }
  std::snprintf(line, sizeof(line), "AthenaK BA fields v1 f32\n%.17g %d %zu %d %.17g %lld\n",
                pm->time, pm->ncycle, ncell, kBAFields, mtot, ntot);
  char name[32];
  std::snprintf(name, sizeof(name), ".%05d.baf", number);
  write(ba_snapshot_basename_ + name, line, out.data(), sizeof(float), out.size());
#else
  (void)pm;
#endif
}

// Diagnostic only: compact per-leaf field snapshot.  Gas density and
// momenta, and the dust cloud integral of mass, momentum and the CENTRAL second
// velocity moments about each leaf's own mean (two passes, so a cold stream gives zero
// instead of a cancellation residue).  The deposit is the exact one the run uses: the
// composite-transfer cloud weights, or canonical TSC on a uniform periodic mesh.
// Geometry (x, z, dx, dz per active leaf, f64) is written once; fields are f32.
void WriteBAFields(Mesh *pm) {
  if (global_variable::nranks > 1 || ba_fields_global_test_) {WriteBAFieldsGlobal(pm); return;}
  auto *pack = pm->pmb_pack;
  auto *pdust = pack->pdust;
  auto &ind = pm->mb_indcs;
  int nmb = pack->nmb_thispack;
  int is = ind.is, js = ind.js, ks = ind.ks;
  int nx1 = ind.nx1, nx2 = ind.nx2;
  int ncells1 = nx1 + 2*ind.ng, ncells2 = nx2 + 2*ind.ng;
  auto &pr = pack->ppart->prtcl_rdata;
  auto &pi = pack->ppart->prtcl_idata;
  int npart = pack->ppart->nprtcl_thispack;
  auto &mbsize = pack->pmb->mb_size;
  DvceArray4D<Real> f("ba_fields", nmb, 7, ncells2, ncells1);
  Kokkos::deep_copy(DevExeSpace(), f, 0.0);
  bool composite = pdust->UsesCompositeTransfer();
  auto transfer = pdust->smr_transfer;
  auto lt = pdust->leaf_traversal;   // the actual leaves
  // canonical TSC: global periodic stencil on the uniform root grid -> leaf index
  int n1 = pm->mesh_indcs.nx1, n2 = pm->mesh_indcs.nx2;
  Real gx1 = pm->mesh_size.x1min, gx2 = pm->mesh_size.x2min;
  Real h1 = (pm->mesh_size.x1max - gx1)/n1, h2 = (pm->mesh_size.x2max - gx2)/n2;
  DualArray2D<int> tgt("ba_tsc_target", composite ? 1 : n2, composite ? 1 : n1);
  if (!composite) {
    if (pm->multilevel || pdust->deposit != DustDeposit::tsc) {
      std::cerr << "BA fields: canonical deposit requires TSC on a uniform mesh (static SMR with "
                << "owner_overlap: no leaf deposit to write; with coupling = pc2 the run does not "
                << "deposit a density either; use f64 snapshots and an offline estimator)"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int m=0; m<nmb; ++m) {
      const auto &size = mbsize.h_view(m);
      int i0 = static_cast<int>(std::lround((size.x1min - gx1)/h1));
      int j0 = static_cast<int>(std::lround((size.x2min - gx2)/h2));
      for (int j=0; j<nx2; ++j)
        for (int i=0; i<nx1; ++i)
          tgt.h_view(j0+j, i0+i) = (m*ncells2 + js + j)*ncells1 + is + i;
    }
    tgt.template modify<HostMemSpace>();
    tgt.template sync<DevExeSpace>();
  }
  auto tgt_ = tgt.d_view;
  // pass 0: mass and momentum; pass 1: central second moments about the leaf mean
  for (int pass=0; pass<2; ++pass) {
    par_for("ba_fields_deposit", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
      Real mass = pr(IPM,p);
      Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
      auto add = [&](const int target, const Real w) {
        if (w == 0.0) return;
        int ti = target % ncells1;
        int tj = (target/ncells1) % ncells2;
        int tm = target/(ncells1*ncells2);
        Real d = mass*w/(mbsize.d_view(tm).dx1*mbsize.d_view(tm).dx2);
        if (pass == 0) {
          dust::DepositAdd(&f(tm,0,tj,ti), d);
          for (int c=0; c<3; ++c) dust::DepositAdd(&f(tm,1+c,tj,ti), d*v[c]);
        } else {
          Real rho = f(tm,0,tj,ti);
          for (int c=0; c<3; ++c) {
            Real dv = v[c] - f(tm,1+c,tj,ti)/rho;
            dust::DepositAdd(&f(tm,4+c,tj,ti), d*dv*dv);
          }
        }
      };
      if (composite) {
        ForEachCompositeWeight(transfer, lt, pr(IPX,p), pr(IPY,p), pi(PLEV,p), lt.Anchor(pi(PGID,p)), add);
      } else {
        Real q1 = (pr(IPX,p) - gx1)/h1 - 0.5, q2 = (pr(IPY,p) - gx2)/h2 - 0.5;
        int c1 = static_cast<int>(floor(q1 + 0.5)), c2 = static_cast<int>(floor(q2 + 0.5));
        Real d1 = q1 - c1, d2 = q2 - c2;
        Real w1[3] = {0.5*SQR(0.5-d1), 0.75-d1*d1, 0.5*SQR(0.5+d1)};
        Real w2[3] = {0.5*SQR(0.5-d2), 0.75-d2*d2, 0.5*SQR(0.5+d2)};
        for (int b=0; b<3; ++b) {
          int jj = ((c2 + b - 1) % n2 + n2) % n2;
          for (int a=0; a<3; ++a) {
            int ii = ((c1 + a - 1) % n1 + n1) % n1;
            add(tgt_(jj,ii), w1[a]*w2[b]);
          }
        }
      }
    });
  }
  auto fh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), f);
  auto gas = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pack->phydro->u0);
  auto prh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
  double mtot = 0.0, mdep = 0.0;
  for (int p=0; p<npart; ++p) mtot += prh(IPM,p);
  int ncell = nmb*nx1*nx2;
  std::vector<float> out;
  out.reserve(static_cast<std::size_t>(kBAFields)*ncell);
  std::vector<double> geom;
  // the geometry is written again after every change of the mesh; the
  // later files carry the number of the first field frame they belong to
  const int mesh_seq = pm->GetAMRLoadBalanceUpdateSeq();
  bool first = !ba_geom_written_ || mesh_seq != ba_geom_seq_;
  if (first) geom.reserve(4*ncell);
  for (int m=0; m<nmb; ++m) {
    const auto &size = mbsize.h_view(m);
    for (int j=js; j<js+nx2; ++j) {
      for (int i=is; i<is+nx1; ++i) {
        if (first) geom.insert(geom.end(), {size.x1min + (i-is+0.5)*size.dx1,
            size.x2min + (j-js+0.5)*size.dx2, size.dx1, size.dx2});
        mdep += fh(m,0,j,i)*size.dx1*size.dx2;
        out.insert(out.end(), {static_cast<float>(gas(m,IDN,ks,j,i)),
            static_cast<float>(gas(m,IM1,ks,j,i)), static_cast<float>(gas(m,IM2,ks,j,i)),
            static_cast<float>(gas(m,IM3,ks,j,i))});
        for (int c=0; c<7; ++c) out.push_back(static_cast<float>(fh(m,c,j,i)));
      }
    }
  }
  if (std::fabs(mdep - mtot) > 1.0e-8*mtot) {   // rounding of ~1e7 particles on ~4e6 cells (U2048)
    std::fprintf(stderr, "BA fields: deposited mass %.17g != particle mass %.17g\n", mdep, mtot);
    std::exit(EXIT_FAILURE);
  }
  if (ba_ppdf_) WriteBAParticlePDF(pm, fh, prh, npart, mtot);
  if (!ba_fields_write_) {
    ++ba_field_number_;
    if (ba_pin_ != nullptr) ba_pin_->SetInteger("problem", "rst_ba_field_number", ba_field_number_);
    return;
  }
  auto write = [](const std::string &name, const std::string &header, const void *data,
                  std::size_t size, std::size_t count) {
    FILE *file = std::fopen(name.c_str(), "wb");
    bool ok = (file != nullptr) && std::fputs(header.c_str(), file) >= 0;
    ok = ok && std::fwrite(data, size, count, file) == count;
    ok = (file != nullptr) && (std::fclose(file) == 0) && ok;
    if (!ok) {
      std::cerr << "Failed writing BA field file " << name << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };
  char line[160];
  if (first) {
    std::snprintf(line, sizeof(line), "AthenaK BA field geometry v1 f64\n%d\n", ncell);
    char gnum[32] = "";
    if (ba_geom_written_) std::snprintf(gnum, sizeof(gnum), ".%05d", ba_field_number_);
    write(ba_snapshot_basename_ + gnum + ".bafield.geom", line, geom.data(), sizeof(double),
          geom.size());
    ba_geom_written_ = true;
    ba_geom_seq_ = mesh_seq;
  }
  std::snprintf(line, sizeof(line), "AthenaK BA fields v1 f32\n%.17g %d %d %d %.17g %d\n",
                pm->time, pm->ncycle, ncell, kBAFields, mtot, npart);
  char number[32];
  std::snprintf(number, sizeof(number), ".%05d.baf", ba_field_number_++);
  if (ba_pin_ != nullptr) ba_pin_->SetInteger("problem", "rst_ba_field_number", ba_field_number_);
  write(ba_snapshot_basename_ + number, line, out.data(), sizeof(float), out.size());
}

//----------------------------------------------------------------------------------------
//! \fn HashUniform01
//! \brief Reproducible stateless pseudo-random number in [0,1), suitable for device code.

KOKKOS_INLINE_FUNCTION
Real HashUniform01(std::uint64_t key) {
  // SplitMix64 finalizer. Using particle tags as keys makes the initial condition
  // independent of Kokkos execution order and therefore reproducible on CPU and GPU.
  key += UINT64_C(0x9e3779b97f4a7c15);
  key = (key ^ (key >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  key = (key ^ (key >> 27)) * UINT64_C(0x94d049bb133111eb);
  key ^= key >> 31;
#if SINGLE_PRECISION_ENABLED
  // All integers through 2^24 are exactly representable as float.  Using 24 bits keeps
  // the largest result at 1 - 2^-24; casting a 53-bit value to float could round to 1.
  return static_cast<Real>(key >> 40) * static_cast<Real>(1.0/16777216.0);
#else
  return static_cast<Real>(key >> 11) * static_cast<Real>(1.0/9007199254740992.0);
#endif
}

//----------------------------------------------------------------------------------------
//! \fn SolveNSH
//! \brief Solves the (2+2N)x(2+2N) linear system for the multi-species NSH drift
//! equilibrium by Gaussian elimination with partial pivoting. Unknown ordering:
//! [u_x, u_phi, v_x,1, v_phi,1, ..., v_x,N, v_phi,N].

void SolveNSH(const Real omega0, const Real qshear, const Real ax,
              const std::vector<Real> &eps, const std::vector<Real> &taus,
              Real &ugx, Real &ugp, std::vector<Real> &vx, std::vector<Real> &vp) {
  int ns = static_cast<int>(eps.size());
  if (ns == 0) {
    ugx = 0.0;
    ugp = -ax/(2.0*omega0);
    vx.clear();
    vp.clear();
    return;
  }
  int n = 2 + 2*ns;
  std::vector<std::vector<Real>> a(n, std::vector<Real>(n+1, 0.0));

  // gas radial:  2*Om*u_phi + sum eps_s/tau_s*(v_x,s - u_x) = -a_x
  a[0][1] = 2.0*omega0;
  for (int s=0; s<ns; ++s) {
    a[0][0]     -= eps[s]/taus[s];
    a[0][2+2*s] += eps[s]/taus[s];
  }
  a[0][n] = -ax;
  // gas azimuthal: -(2-q)*Om*u_x + sum eps_s/tau_s*(v_phi,s - u_phi) = 0
  a[1][0] = -(2.0-qshear)*omega0;
  for (int s=0; s<ns; ++s) {
    a[1][1]     -= eps[s]/taus[s];
    a[1][3+2*s] += eps[s]/taus[s];
  }
  // dust radial / azimuthal for each species
  for (int s=0; s<ns; ++s) {
    int rx = 2+2*s, rp = 3+2*s;
    a[rx][rp] = 2.0*omega0;
    a[rx][rx] -= 1.0/taus[s];
    a[rx][0]  += 1.0/taus[s];
    a[rp][rx] = -(2.0-qshear)*omega0;
    a[rp][rp] -= 1.0/taus[s];
    a[rp][1]  += 1.0/taus[s];
  }

  // Gaussian elimination with partial pivoting
  for (int c=0; c<n; ++c) {
    int piv = c;
    for (int r=c+1; r<n; ++r) {
      if (fabs(a[r][c]) > fabs(a[piv][c])) {piv = r;}
    }
    std::swap(a[c], a[piv]);
    for (int r=c+1; r<n; ++r) {
      Real f = a[r][c]/a[c][c];
      for (int cc=c; cc<=n; ++cc) {a[r][cc] -= f*a[c][cc];}
    }
  }
  std::vector<Real> x(n);
  for (int r=n-1; r>=0; --r) {
    Real sum = a[r][n];
    for (int cc=r+1; cc<n; ++cc) {sum -= a[r][cc]*x[cc];}
    x[r] = sum/a[r][r];
  }

  ugx = x[0];
  ugp = x[1];
  vx.resize(ns);
  vp.resize(ns);
  for (int s=0; s<ns; ++s) {
    vx[s] = x[2+2*s];
    vp[s] = x[3+2*s];
  }
}

//----------------------------------------------------------------------------------------
//! \fn ReadNSHParams
//! \brief Reads the shearing-box and active dust-mixture parameters.

void ReadNSHParams(ParameterInput *pin, MeshBlockPack *pmbp,
                   Real &omega0, Real &qshear, Real &etavk,
                   std::vector<Real> &eps, std::vector<Real> &taus) {
  dust::DustGasDrag *pdust = pmbp->pdust;
  int ns = (pdust != nullptr) ? pdust->nspecies : 0;
  omega0 = pin->GetReal("shearing_box","omega0");
  qshear = pin->GetReal("shearing_box","qshear");
  etavk  = pin->GetReal("problem","etavk");
  eps.resize(ns);
  taus.resize(ns);
  bool has_eps = pin->DoesParameterExist("problem","eps_1");
  for (int s=0; s<ns; ++s) {
    if (has_eps) {
      eps[s] = pin->GetReal("problem","eps_" + std::to_string(s+1));
    } else {
      eps[s] = pdust->dust_to_gas/static_cast<Real>(ns);
    }
    taus[s] = pdust->taus.h_view(s);
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustNSH()
//! \brief Problem Generator for the multi-species NSH drift equilibrium

void ProblemGenerator::DustNSH(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  bool has_particles = (pmbp->ppart != nullptr);
  bool has_dust = (pmbp->pdust != nullptr);
  if (has_particles != has_dust) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "NSH requires both <particles> and <dust>, or neither "
              << "for the dust-free gas equilibrium" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool dust_active = has_particles && has_dust;
  bool random_placement = false;
  bool cloud_root_random = pin->GetOrAddBoolean("problem", "cloud_root_random", false);
  ba_snapshots_ = pin->GetOrAddBoolean("problem", "ba_f64_snapshots", false);
  if (ba_snapshots_) {
    if (global_variable::nranks != 1 || !dust_active ||
        !pin->GetOrAddBoolean("problem", "user_hist", false)) {
      std::cerr << "BA snapshots require dust, one MPI rank and user_hist=true"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ba_snapshot_basename_ = pin->GetString("job", "basename");
    ba_snapshot_number_ = 0;
    ba_snapshot_dt_ = pin->GetOrAddReal("problem", "ba_snapshot_dt", 40.0);
    if (ba_snapshot_dt_ <= 0.0) {
      std::cerr << "ba_snapshot_dt must be positive" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  ba_fields_ = pin->GetOrAddBoolean("problem", "ba_fields", false);
  ba_ppdf_ = pin->GetOrAddBoolean("problem", "ba_ppdf", false);
  ba_fields_write_ = pin->GetOrAddBoolean("problem", "ba_fields_write", true);
  ba_fields_global_test_ = pin->GetOrAddBoolean("problem", "ba_fields_global_test", false);
  ba_ppdf_nref_ = pin->GetOrAddInteger("problem", "ba_ppdf_nref", 128);
  if (ba_ppdf_ && !ba_fields_) {
    std::cerr << "ba_ppdf is written with the BA field snapshots: set ba_fields = true"
              << " (ba_fields_write = false skips the field files)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ba_history_ = pin->GetOrAddBoolean("problem", "ba_history", false);
  // several ranks: fields of the canonical TSC deposit on a uniform mesh (WriteBAFieldsGlobal)
  const bool fields_mpi = dust_active && !pmy_mesh_->multilevel &&
      pmbp->pdust->smr_transfer_mode == DustSMRTransfer::none && !ba_history_;
  if (ba_fields_ || ba_history_) {
    if ((global_variable::nranks != 1 && !fields_mpi) || !dust_active || pmy_mesh_->three_d ||
        !pin->GetOrAddBoolean("problem", "user_hist", false)) {
      std::cerr << "BA fields/history require 2-D dust, one MPI rank and user_hist=true"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ba_snapshot_basename_ = pin->GetString("job", "basename");
    ba_field_number_ = 0;
    ba_field_dt_ = pin->GetOrAddReal("problem", "ba_field_dt", 1.0);
    if (ba_field_dt_ <= 0.0) {
      std::cerr << "ba_field_dt must be positive" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // canonical_root: the same root parents with the standard deposit on a uniform mesh,
  // so a uniform control shares the parent positions of a refinable-cloud run
  bool canonical_root = cloud_root_random && dust_active &&
      pmbp->pdust->smr_transfer_mode == DustSMRTransfer::none && !pmy_mesh_->multilevel;
  // several ranks: refinable clouds with <dust>/experimental_mpi; every rank
  // draws the same parents and keeps those in its blocks
  const bool ranks_ok = (global_variable::nranks == 1) ||
      (dust_active && pmbp->pdust->experimental_mpi && !canonical_root);
  if (cloud_root_random && (!ranks_ok || !dust_active ||
      !(pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud ||
        canonical_root) || pmy_mesh_->three_d)) {
    std::cerr << "cloud_root_random requires single-rank (or <dust>/experimental_mpi) 2-D "
              << "dust with refinable_cloud transfer, or the canonical deposit on a uniform "
              << "mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (dust_active) {
    std::string placement =
        pin->GetOrAddString("problem","particle_placement","lattice");
    random_placement = (placement.compare("random") == 0);
    if (!random_placement && placement.compare("lattice") != 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "<problem>/particle_placement must be lattice or random"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  bool check_equilibrium =
      pin->GetOrAddBoolean("problem","check_equilibrium",!random_placement);
  pgen_final_func = check_equilibrium ? DustNSHErrors : nullptr;
  // Keep this enrolled in dust-free runs as well so an otherwise unchanged input
  // with user_hist=true remains valid; the callback emits no dust columns in that mode.
  user_hist_func = DustNSHHistory;
  // experimental dynamic mesh: the scheduled refinement request of
  // <dust>/amr_sched_* answers <amr_criterionN>/method = user
  if (pmy_mesh_->adaptive && pin->DoesBlockExist("dust")) {
    dust::SetupScheduledRefinement(pin);
    user_ref_func = dust::ScheduledRefinement;
    // the dust-density request instead of the schedule
    if (pin->GetOrAddBoolean("dust", "amr_density_trigger", false)) {
      dust::SetupDensityTrigger(pin);
      user_ref_func = dust::DensityRefinement;
    }
  }
  ba_pin_ = pin;
  if (restart) {
    // native restart: gas, particles and the dust counters come from the
    // file.  The output numbers continue from the values the file carries in <problem>; a file
    // without them falls back on the time.  No file is touched here (the run directory is set
    // later); the distribution file is cut back and continued at its next record.
    if (ba_snapshots_) {
      ba_snapshot_number_ = pin->GetOrAddInteger("problem", "rst_ba_snapshot_number",
          static_cast<int>(std::ceil((pmy_mesh_->time - 1.0e-10)/ba_snapshot_dt_)));
    }
    if (ba_fields_) {
      ba_field_number_ = pin->GetOrAddInteger("problem", "rst_ba_field_number",
          static_cast<int>(std::ceil((pmy_mesh_->time - 1.0e-10)/ba_field_dt_)));
      ba_ppdf_resume_ = ba_ppdf_;
      // on a mesh that may have changed, the first frame after the restart writes its own
      // (numbered) geometry file instead of replacing the one of frame zero
      ba_geom_written_ = ba_field_number_ > 0;
    }
    if (pmy_mesh_->pmb_pack->pdust != nullptr && pmy_mesh_->pmb_pack->ppart != nullptr &&
        pmy_mesh_->pmb_pack->ppart->nprtcl_thispack == 0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "restart of a dust run without particles in the restart file" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    return;
  }

  if (pmbp->phydro == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH test requires a <hydro> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pin->DoesBlockExist("shearing_box"))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH test requires a <shearing_box> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real omega0, qshear, etavk;
  std::vector<Real> eps, taus;
  ReadNSHParams(pin, pmbp, omega0, qshear, etavk, eps, taus);
  int nspec = static_cast<int>(eps.size());
  Real ax = 2.0*omega0*etavk;

  // the radial forcing must be applied to the gas by the standard const_accel source
  // term (so that dust particles do not feel it); verify the input file sets it up
  {
    bool ok = pin->GetOrAddBoolean("hydro_srcterms","const_accel",false);
    Real val = ok ? pin->GetReal("hydro_srcterms","const_accel_val") : 0.0;
    int dir = ok ? pin->GetInteger("hydro_srcterms","const_accel_dir") : 0;
    if (!ok || dir != 1 || fabs(val - ax) > 1.0e-12*fabs(ax)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "NSH test requires <hydro_srcterms> const_accel = true, "
                << "const_accel_dir = 1, const_accel_val = 2*omega0*etavk = " << ax
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // equilibrium drift velocities
  Real ugx, ugp;
  std::vector<Real> vx, vp;
  SolveNSH(omega0, qshear, ax, eps, taus, ugx, ugp, vx, vp);
  if (global_variable::my_rank == 0) {
    if (dust_active) {
      std::cout << "# NSH equilibrium: u_gx=" << ugx << " u_gphi=" << ugp << std::endl;
      for (int s=0; s<nspec; ++s) {
        std::cout << "#   species " << s+1 << ": v_x=" << vx[s]
                  << " v_phi=" << vp[s] << std::endl;
      }
    } else {
      std::cout << "# Dust-free NSH equilibrium: u_gx=" << ugx
                << " u_gphi=" << ugp << std::endl;
    }
  }

  // initialize uniform gas at the equilibrium; azimuthal component is IM2 in 3D and
  // IM3 in the 2D r-z geometry (matching ShearingBoxCC::SourceTermsCC)
  Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  bool three_d = pmy_mesh_->three_d;
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->phydro->u0;
  par_for("nsh_gas", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0*ugx;
    u0(m,IM2,k,j,i) = three_d ? rho0*ugp : 0.0;
    u0(m,IM3,k,j,i) = three_d ? 0.0 : rho0*ugp;
  });

  // There are no particle or dust modules to initialize in the dust-free equilibrium.
  if (!dust_active) return;

  // Initialize particles at the per-species equilibrium velocities. The lattice is the
  // quiet start used by the NSH regression test. Random placement is the warm start for
  // nonlinear streaming-instability calculations (e.g. Johansen et al. 2007 Run BA).
  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &mbsize = pmbp->pmb->mb_size;
  auto gids = pmbp->gids;
  int npart_permb = npart/nmb;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  int ppc_int = npart_permb/ncells;
  if ((!random_placement &&
       ((npart_permb != ppc_int*ncells) || (ppc_int % nspec != 0))) ||
      (random_placement && (npart % nspec != 0))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH lattice placement requires <particles>/ppc to be an integer "
              << "multiple of <dust>/nspecies; random placement requires the total "
              << "number of particles per rank to be divisible by <dust>/nspecies"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int lnx1 = indcs.nx1, lnx2 = indcs.nx2, lnx3 = indcs.nx3;
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);
  std::uint64_t random_seed = static_cast<std::uint64_t>(
      pin->GetOrAddInteger("problem","random_seed",1));
  auto &taus_ = pmbp->pdust->taus;
  // per-species device table: [s][0]=v_x, [s][1]=v_phi, [s][2]=mass factor
  DualArray2D<Real> spdat("nsh_spdat",nspec,3);
  for (int s=0; s<nspec; ++s) {
    spdat.h_view(s,0) = vx[s];
    spdat.h_view(s,1) = vp[s];
    spdat.h_view(s,2) = eps[s]*rho0*static_cast<Real>(nspec)/ppc;
  }
  spdat.template modify<HostMemSpace>();
  spdat.template sync<DevExeSpace>();

  if (cloud_root_random) {
    // Four particles per ROOT cell, placed independently over the whole box.
    // Refinement changes the cloud shape through the exact P1 split, never the
    // prescribed parent mass. This is a warm Poisson seed, not a quiet lattice.
    auto transfer = pmbp->pdust->smr_transfer;
    auto lt = pmbp->pdust->leaf_traversal;
    if (canonical_root) {   // only the root geometry is used below
      const auto &ms = pmy_mesh_->mesh_size;
      const auto &mi = pmy_mesh_->mesh_indcs;
      transfer.x1min = ms.x1min; transfer.x1max = ms.x1max;
      transfer.x2min = ms.x2min; transfer.x2max = ms.x2max;
      transfer.ncoarse1 = mi.nx1; transfer.ncoarse2 = mi.nx2;
      transfer.root_dx1 = (ms.x1max - ms.x1min)/mi.nx1;
      transfer.root_dx2 = (ms.x2max - ms.x2min)/mi.nx2;
    }
    int nroot = transfer.ncoarse1*transfer.ncoarse2;
    int nparent = static_cast<int>(ppc)*nroot;
    if (ppc != static_cast<Real>(static_cast<int>(ppc)) || nparent <= 0) {
      std::cerr << "cloud_root_random requires positive integer root ppc" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int ncells1 = pmbp->pdust->qdep.extent_int(4);
    int ncells2 = pmbp->pdust->qdep.extent_int(3);
    // the parents of this rank: those whose leaf is in one of its blocks, in
    // the order of their global index p, which stays the tag (one rank: all of them)
    const int nlocal_mb = nmb;
    DvceArray1D<int> pslot("nsh_root_slot", nparent);
    par_for("nsh_cloud_root_owner", DevExeSpace(), 0, nparent-1, KOKKOS_LAMBDA(const int p) {
      std::uint64_t key = static_cast<std::uint64_t>(p)
                        + random_seed*UINT64_C(0x9e3779b97f4a7c15);
      Real x = transfer.x1min + HashUniform01(key + UINT64_C(0x632be59bd9b4e019))
                             *(transfer.x1max-transfer.x1min);
      Real y = transfer.x2min + HashUniform01(key + UINT64_C(0x8cb92baa3f3d8dd7))
                             *(transfer.x2max-transfer.x2min);
      int m = 0;
      if (canonical_root) {
        for (int b=0; b<nmb; ++b) {
          if (x >= mbsize.d_view(b).x1min && x < mbsize.d_view(b).x1max &&
              y >= mbsize.d_view(b).x2min && y < mbsize.d_view(b).x2max) m = b;
        }
      } else {
        const int leaf = lt.LeafOwner(x, y, 0);   // the actual leaves
        m = (leaf < 0) ? 0 : leaf/(ncells1*ncells2);
      }
      pslot(p) = m;
    });
    DvceArray1D<int> pnew("nsh_root_new", nparent);
    int nkeep = 0;
    Kokkos::parallel_scan("nsh_root_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, nparent),
    KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
      const bool keep = (pslot(p) < nlocal_mb);
      if (final) pnew(p) = keep ? upd : -1;
      if (keep) ++upd;
    }, nkeep);
    Kokkos::realloc(pr, ppar->nrdata, nkeep);
    Kokkos::realloc(pi, ppar->nidata, nkeep);
    ppar->nprtcl_thispack = nkeep;
    pmy_mesh_->nprtcl_thisrank = nkeep;
    pmy_mesh_->nprtcl_eachrank[global_variable::my_rank] = nkeep;
    pmy_mesh_->nprtcl_total = nparent;
    auto pr0 = ppar->prtcl_rdata;
    auto pi0 = ppar->prtcl_idata;
    auto gids0 = pmbp->gids;
    int nrdata = ppar->nrdata;
    Real mass_unit = transfer.root_dx1*transfer.root_dx2;
    par_for("nsh_cloud_root_random", DevExeSpace(), 0, nparent-1,
    KOKKOS_LAMBDA(const int q) {
      const int p = pnew(q);
      if (p < 0) return;
      std::uint64_t key = static_cast<std::uint64_t>(q)
                        + random_seed*UINT64_C(0x9e3779b97f4a7c15);
      Real x = transfer.x1min + HashUniform01(key + UINT64_C(0x632be59bd9b4e019))
                             *(transfer.x1max-transfer.x1min);
      Real y = transfer.x2min + HashUniform01(key + UINT64_C(0x8cb92baa3f3d8dd7))
                             *(transfer.x2max-transfer.x2min);
      const int m = pslot(q);
      pi0(PGID,p) = gids0+m;
      pi0(PTAG,p) = q;
      pi0(PSP,p) = q % nspec;
      pi0(PLEV,p) = 0;
      for (int c=0; c<nrdata; ++c) pr0(c,p) = 0.0;
      pr0(IPX,p) = x; pr0(IPX1,p) = x;
      pr0(IPY,p) = y; pr0(IPY1,p) = y;
      pr0(IPTS,p) = taus_.d_view(q % nspec);
      pr0(IPM,p) = spdat.d_view(q % nspec,2)*mass_unit;
      pr0(IPVX,p) = spdat.d_view(q % nspec,0);
      pr0(IPVZ,p) = spdat.d_view(q % nspec,1);
      pr0(IPVX1,p) = pr0(IPVX,p);
      pr0(IPVZ1,p) = pr0(IPVZ,p);
    });
    // the tag counter is set collectively before the splits
    if (!canonical_root) pmbp->pdust->EnsureNextTag();
    if (!canonical_root) pmbp->pdust->SplitToLeaves();
    std::cout << "# BA random root " << (canonical_root ? "particles (canonical): "
              : "clouds: ") << nparent << " parents -> " << ppar->nprtcl_thispack
              << (canonical_root ? " particles" : " particles after P1 presplit")
              << std::endl;
  } else {
  par_for("nsh_part", DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m, s;
    if (random_placement) {
      // Offset the tag stream before hashing so different seeds generate independent
      // position sets.  XOR with the raw tag is not sufficient: for consecutive tags
      // spanning complete power-of-two ranges it only permutes the same set of keys.
      std::uint64_t key = static_cast<std::uint64_t>(pi(PTAG,p))
                        + random_seed*UINT64_C(0x9e3779b97f4a7c15);
      m = static_cast<int>(HashUniform01(key)*static_cast<Real>(nmb));
      if (m > (nmb-1)) {m = nmb-1;}
      Real ux = HashUniform01(key + UINT64_C(0x632be59bd9b4e019));
      Real uy = HashUniform01(key + UINT64_C(0x8cb92baa3f3d8dd7));
      Real uz = HashUniform01(key + UINT64_C(0x58f38ded6f7c55b5));
      pr(IPX,p) = mbsize.d_view(m).x1min
                + ux*(mbsize.d_view(m).x1max - mbsize.d_view(m).x1min);
      pr(IPY,p) = mbsize.d_view(m).x2min
                + uy*(mbsize.d_view(m).x2max - mbsize.d_view(m).x2min);
      pr(IPZ,p) = three_d ? mbsize.d_view(m).x3min
                + uz*(mbsize.d_view(m).x3max - mbsize.d_view(m).x3min) : 0.0;
      // Stripe species by the local particle index. Unlike PTAG % nspec, this remains
      // balanced when <particles>/assign_tag=rank_order and nranks shares a factor with
      // nspec. The divisibility check above guarantees equal counts in this pack.
      s = p % nspec;
    } else {
      m = p/npart_permb;
      if (m > (nmb-1)) {m = nmb-1;}
      int q = p - m*npart_permb;
      int c = q/ppc_int;
      s = (q % ppc_int) % nspec;
      int i = c % lnx1;
      int j = (c/lnx1) % lnx2;
      int k = c/(lnx1*lnx2);
      pr(IPX,p) = CellCenterX(i, lnx1, mbsize.d_view(m).x1min,
                             mbsize.d_view(m).x1max);
      pr(IPY,p) = CellCenterX(j, lnx2, mbsize.d_view(m).x2min,
                             mbsize.d_view(m).x2max);
      pr(IPZ,p) = three_d ?
          CellCenterX(k, lnx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max) : 0.0;
    }
    pi(PGID,p) = gids + m;
    pi(PSP,p) = s;
    pr(IPTS,p) = taus_.d_view(s);
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    pr(IPM,p) = spdat.d_view(s,2)*vol;
    pr(IPVX,p) = spdat.d_view(s,0);
    pr(IPVY,p) = three_d ? spdat.d_view(s,1) : 0.0;
    pr(IPVZ,p) = three_d ? 0.0 : spdat.d_view(s,1);
    pr(IPRX,p) = 0.0;
    pr(IPRY,p) = 0.0;
    pr(IPRZ,p) = 0.0;
  });
  }

  // two cold streams along x2 for the multi-stream regrid test; the
  // particles below the midplane get +two_stream_v added to their x2 velocity, the others
  // two_stream_v_upper (default -two_stream_v) (2-D lattice or random placement)
  const Real two_stream_v = pin->GetOrAddReal("problem", "two_stream_v", 0.0);
  const Real two_stream_vu = pin->GetOrAddReal("problem", "two_stream_v_upper", -two_stream_v);
  if (two_stream_v != 0.0 || two_stream_vu != 0.0) {
    if (three_d || cloud_root_random) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/two_stream_v needs a 2-D run without cloud_root_random" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    auto &prs = ppar->prtcl_rdata;
    par_for("nsh_two_stream", DevExeSpace(), 0, ppar->nprtcl_thispack-1, KOKKOS_LAMBDA(const int p) {
      prs(IPVY,p) += (prs(IPY,p) < 0.0) ? two_stream_v : two_stream_vu;
    });
  }

  // a dust blob; the particles within blob_r of (blob_x1, blob_x2)
  // (periodic distance) carry (1 + blob_amp) times their mass (2-D lattice or random placement)
  const Real blob_amp = pin->GetOrAddReal("problem", "blob_amp", 0.0);
  if (blob_amp != 0.0) {
    if (three_d || cloud_root_random) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/blob_amp needs a 2-D run without cloud_root_random" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real bx = pin->GetReal("problem", "blob_x1"), by = pin->GetReal("problem", "blob_x2");
    const Real br = pin->GetReal("problem", "blob_r");
    const Real lx = pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min;
    const Real ly = pmy_mesh_->mesh_size.x2max - pmy_mesh_->mesh_size.x2min;
    auto &prb = ppar->prtcl_rdata;
    par_for("nsh_blob", DevExeSpace(), 0, ppar->nprtcl_thispack-1, KOKKOS_LAMBDA(const int p) {
      Real dx = prb(IPX,p) - bx, dy = prb(IPY,p) - by;
      dx -= lx*std::round(dx/lx); dy -= ly*std::round(dy/ly);
      if (dx*dx + dy*dy < br*br) prb(IPM,p) *= (1.0 + blob_amp);
    });
  }

  // particle timestep from the largest drift speed (transport is radial drift here)
  Real vmax = 1.0e-30;
  vmax = std::max(vmax, std::max(std::fabs(two_stream_v), std::fabs(two_stream_vu)));
  for (int s=0; s<nspec; ++s) {
    vmax = std::max(vmax, fabs(vx[s]));
    vmax = std::max(vmax, fabs(vp[s]));
  }
  Real dtnew = std::numeric_limits<float>::max();
  auto &msize = pmbp->pmb->mb_size;
  dtnew = std::min(dtnew, msize.h_view(0).dx1/vmax);
  dtnew = std::min(dtnew, msize.h_view(0).dx2/vmax);
  if (three_d) {dtnew = std::min(dtnew, msize.h_view(0).dx3/vmax);}
  ppar->dtnew = dtnew;

  // <problem>/merge_audit_replay = audit-format entry state (diagnostic): one
  // merge pass from a saved state, then exit
  if (!pin->GetOrAddString("problem", "merge_audit_replay", "").empty()) {
    RunAuditReplay(pin, pmy_mesh_);
    // the replayed particles get their MeshBlock ids from their positions
    // (the audit file does not carry valid ones for this mesh)
    if (pmbp->pdust != nullptr &&
        pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud) {
      pmbp->pdust->ResetPGIDFromPositions("audit replay");
    }
    // only reached with merge_audit_restart: the run continues from the restored time, so the BA
    // outputs are numbered by time and resume at the next multiple of their interval
    // (numbering from zero would write one file per history call until it caught up)
    if (ba_fields_) {
      ba_field_number_ = static_cast<int>(std::ceil((pmy_mesh_->time - 1.0e-10)/ba_field_dt_));
    }
    if (ba_snapshots_) {
      ba_snapshot_number_ =
          static_cast<int>(std::ceil((pmy_mesh_->time - 1.0e-10)/ba_snapshot_dt_));
    }
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustNSHErrors()
//! \brief Measures the deviation of the mean gas and per-species dust drift velocities
//! from the NSH equilibrium at the end of the run and appends the errors to
//! "<basename>-errs.dat". Both the coupled and dust-free equilibria are fixed points,
//! so all errors should be at round-off level.

void DustNSHErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  int nspec = (pmbp->pdust != nullptr) ? pmbp->pdust->nspecies : 0;
  bool three_d = pm->three_d;

  Real omega0, qshear, etavk;
  std::vector<Real> eps, taus;
  ReadNSHParams(pin, pmbp, omega0, qshear, etavk, eps, taus);
  Real ugx, ugp;
  std::vector<Real> vx, vp;
  SolveNSH(omega0, qshear, 2.0*omega0*etavk, eps, taus, ugx, ugp, vx, vp);

  // mean gas velocities over active cells
  auto &indcs = pm->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  int iazim = three_d ? IM2 : IM3;
  Real gmx = 0.0, gmp = 0.0, gm = 0.0;
  Kokkos::parallel_reduce("nsh_gsum",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mx, Real &mp, Real &mass) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    mx += u0(m,IM1,k,j,i);
    mp += u0(m,iazim,k,j,i);
    mass += u0(m,IDN,k,j,i);
  }, Kokkos::Sum<Real>(gmx), Kokkos::Sum<Real>(gmp), Kokkos::Sum<Real>(gm));

  // per-species mean dust velocities (absent in the dust-free equilibrium)
  std::vector<Real> pvx(nspec), pvp(nspec), pn(nspec);
  if (nspec > 0) {
    auto &pr = ppar->prtcl_rdata;
    auto &pi = ppar->prtcl_idata;
    int npart = ppar->nprtcl_thispack;
    int ivazim = three_d ? IPVY : IPVZ;
    for (int s=0; s<nspec; ++s) {
      Real sx = 0.0, sp = 0.0, sn = 0.0;
      Kokkos::parallel_reduce("nsh_psum",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
      KOKKOS_LAMBDA(const int &p, Real &x_, Real &p_, Real &n_) {
        if (pi(PSP,p) == s) {
          x_ += pr(IPVX,p);
          p_ += pr(ivazim,p);
          n_ += 1.0;
        }
      }, Kokkos::Sum<Real>(sx), Kokkos::Sum<Real>(sp), Kokkos::Sum<Real>(sn));
      pvx[s] = sx;
      pvp[s] = sp;
      pn[s] = sn;
    }
  }

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &gmx, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gmp, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gm,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  if (nspec > 0) {
    MPI_Allreduce(MPI_IN_PLACE, pvx.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pvp.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, pn.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif

  Real err_ugx = fabs(gmx/gm - ugx);
  Real err_ugp = fabs(gmp/gm - ugp);

  if (global_variable::my_rank == 0) {
    std::string fname;
    fname.assign(pin->GetString("job","basename"));
    fname.append("-errs.dat");
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      std::fclose(pfile);
      pfile = std::fopen(fname.c_str(), "a");
    } else {
      pfile = std::fopen(fname.c_str(), "w");
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  err_ugx  err_ugphi");
      if (nspec > 0) {std::fprintf(pfile, "  err_vx_s  err_vphi_s ...");}
      std::fprintf(pfile, "\n");
    }
    std::fprintf(pfile, "%04d  %04d  %04d  %05d  %e  %e", pm->mesh_indcs.nx1,
                 pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle, err_ugx, err_ugp);
    for (int s=0; s<nspec; ++s) {
      std::fprintf(pfile, "  %e  %e", fabs(pvx[s]/pn[s] - vx[s]),
                   fabs(pvp[s]/pn[s] - vp[s]));
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustNSHHistory()
//! \brief User history output: per-species sums of the radial and azimuthal dust drift
//! velocities and particle counts (summed over ranks; divide sums by counts for means).

void DustNSHHistory(HistoryData *pdata, Mesh *pm) {
  if (pm->pmb_pack->ppart == nullptr || pm->pmb_pack->pdust == nullptr) {
    pdata->nhist = 0;
    return;
  }
  particles::Particles *ppar = pm->pmb_pack->ppart;
  int nspec = pm->pmb_pack->pdust->nspecies;
  bool three_d = pm->three_d;
  int nsout = std::min(nspec, NHISTORY_VARIABLES/3);
  pdata->nhist = 3*nsout;

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  int ivazim = three_d ? IPVY : IPVZ;
  for (int s=0; s<nsout; ++s) {
    pdata->label[3*s  ] = "vxsum_" + std::to_string(s+1);
    pdata->label[3*s+1] = "vpsum_" + std::to_string(s+1);
    pdata->label[3*s+2] = "np_" + std::to_string(s+1);
    Real sx = 0.0, sp = 0.0, sn = 0.0;
    Kokkos::parallel_reduce("nsh_hist",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &x_, Real &p_, Real &n_) {
      if (pi(PSP,p) == s) {
        x_ += pr(IPVX,p);
        p_ += pr(ivazim,p);
        n_ += 1.0;
      }
    }, Kokkos::Sum<Real>(sx), Kokkos::Sum<Real>(sp), Kokkos::Sum<Real>(sn));
    pdata->hdata[3*s  ] = sx;
    pdata->hdata[3*s+1] = sp;
    pdata->hdata[3*s+2] = sn;
  }
  if (ba_history_) {
    // mass-weighted dust moments (all species) and the cumulative cloud
    // machinery counters; the merge passes themselves are in cloud_merge_log.txt
    int h = 3*nsout;
    if (h + 14 > NHISTORY_VARIABLES) {
      std::cerr << "BA history: too many species for the history columns" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const char *labels[14] = {"msum", "mvx", "mvz", "mvp", "mvx2", "mvz2", "mvp2",
        "nsplit", "nlobesplit", "ncoarse", "ncoarsein", "ncoarserej", "nfar_rel",
        "pcapacity"};
    for (int c=0; c<14; ++c) pdata->label[h+c] = labels[c];
    Real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0;
    Kokkos::parallel_reduce("ba_hist1",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &a0, Real &a1, Real &a2, Real &a3) {
      Real m = pr(IPM,p);
      a0 += m; a1 += m*pr(IPVX,p); a2 += m*pr(IPVY,p); a3 += m*pr(IPVZ,p);
    }, Kokkos::Sum<Real>(s0), Kokkos::Sum<Real>(s1), Kokkos::Sum<Real>(s2),
       Kokkos::Sum<Real>(s3));
    Kokkos::parallel_reduce("ba_hist2",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &a4, Real &a5, Real &a6) {
      Real m = pr(IPM,p);
      a4 += m*SQR(pr(IPVX,p)); a5 += m*SQR(pr(IPVY,p)); a6 += m*SQR(pr(IPVZ,p));
    }, Kokkos::Sum<Real>(s4), Kokkos::Sum<Real>(s5), Kokkos::Sum<Real>(s6));
    Real sums[7] = {s0, s1, s2, s3, s4, s5, s6};
    for (int c=0; c<7; ++c) pdata->hdata[h+c] = sums[c];
    auto *pd = pm->pmb_pack->pdust;
    pdata->hdata[h+7] = static_cast<Real>(pd->cloud_split_count);
    pdata->hdata[h+8] = static_cast<Real>(pd->cloud_lobe_split_count);
    pdata->hdata[h+9] = static_cast<Real>(pd->cloud_coarsen_events);
    pdata->hdata[h+10] = static_cast<Real>(pd->cloud_coarsen_in);
    pdata->hdata[h+11] = static_cast<Real>(pd->cloud_coarsen_reject);
    pdata->hdata[h+12] = static_cast<Real>(pd->cloud_far_release);
    pdata->hdata[h+13] = static_cast<Real>(npart);   // the arrays may hold headroom (in-place split)
    pdata->nhist = h + 14;
  }
  if (ba_snapshots_ && pm->time + 1.0e-10 >= ba_snapshot_dt_*ba_snapshot_number_)
    WriteBASnapshot(pm);
  if (ba_fields_ && pm->time + 1.0e-10 >= ba_field_dt_*ba_field_number_)
    WriteBAFields(pm);
  return;
}
