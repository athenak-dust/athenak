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
#include <complex>
#include <iomanip>
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
// 2026-10-21: the schedule index of the next snapshot / field frame (due at dt*index), equal to
// the file number unless a restart changed the interval (ScheduleIndex)
int ba_snapshot_k_ = 0, ba_field_k_ = 0;
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
bool dust_moments_hist_ = false;   // msum, mvx, mvz, mz2 columns in the user history
// 3-D particle snapshots: every p3d_snapshot_dt, per rank p3d.NNNNN.rRRRRR.bin with
// "P3D1", the time (double), the particle count (int), then x, y, z, m (float32 arrays)
Real p3d_dt_ = -1.0;
int p3d_number_ = 0;
void WriteP3DSnapshot(Mesh *pm) {
  auto *ppar = pm->pmb_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_rdata);
  char fn[64];
  std::snprintf(fn, sizeof(fn), "p3d.%05d.r%06d.bin", p3d_number_, global_variable::my_rank);
  std::FILE *fp = std::fopen(fn, "wb");
  if (fp == nullptr) {std::cerr << "cannot open " << fn << std::endl; std::exit(EXIT_FAILURE);}
  const double t = pm->time;
  std::fwrite("P3D1", 1, 4, fp);
  std::fwrite(&t, sizeof(double), 1, fp);
  std::fwrite(&npart, sizeof(int), 1, fp);
  std::vector<float> buf(npart);
  for (int c : {static_cast<int>(IPX), static_cast<int>(IPY), static_cast<int>(IPZ),
                static_cast<int>(IPM)}) {
    for (int p=0; p<npart; ++p) buf[p] = static_cast<float>(pr(c,p));
    std::fwrite(buf.data(), sizeof(float), npart, fp);
  }
  std::fclose(fp);
  ++p3d_number_;
}
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
#if MPI_PARALLEL_ENABLED
  // several ranks (refinable clouds, 2026-10-21): the grids of the whole mesh, from all ranks
  if (global_variable::nranks > 1)
    MPI_Allreduce(MPI_IN_PLACE, &nlev, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
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
#if MPI_PARALLEL_ENABLED
  // every finest cell lies in one leaf of one rank: the others hold zeros there (level 0 in lmap)
  if (global_variable::nranks > 1) {
    for (auto &g : grid)
      MPI_Allreduce(MPI_IN_PLACE, g.data(), static_cast<int>(g.size()), MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, lmap.data(), static_cast<int>(lmap.size()), MPI_SIGNED_CHAR, MPI_MAX,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, ref.data(), static_cast<int>(ref.size()), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
  }
#endif
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
          const Real wab = w1[a]*w2[b];
          if (wab == 0.0) continue;   // as the block writer: an empty cell gets nothing (0/0 below)
          const Real d = mass*wab*inv_area;
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
  // the geometry (row order) with the first frame of this process: after a restart it is written
  // again, numbered with that frame, since the earlier frames may come from the block writer (one
  // rank, block order) or from another directory (Codex review)
  const int mesh_seq = pm->GetAMRLoadBalanceUpdateSeq();
  if (!ba_geom_written_ || mesh_seq != ba_geom_seq_) {
    std::vector<double> geom;
    geom.reserve(4*ncell);
    for (int j=0; j<n2; ++j)
      for (int i=0; i<n1; ++i) geom.insert(geom.end(), {gx1 + (i + 0.5)*h1, gx2 + (j + 0.5)*h2, h1, h2});
    std::snprintf(line, sizeof(line), "AthenaK BA field geometry v1 f64\n%zu\n", ncell);
    char gnum[32] = "";
    if (ba_geom_written_) std::snprintf(gnum, sizeof(gnum), ".%05d", number);
    write(ba_snapshot_basename_ + gnum + ".bafield.geom", line, geom.data(), sizeof(double), geom.size());
    ba_geom_written_ = true;
    ba_geom_seq_ = mesh_seq;
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

#if MPI_PARALLEL_ENABLED
// Several ranks: rank 0 receives the vectors of all ranks, concatenated in rank order (the
// other ranks keep theirs).  Rank order is gid order, so the result is the block order of one
// rank with the same mesh.
template <typename T>
void GatherToRoot(std::vector<T> &v, MPI_Datatype type) {
  const int nr = global_variable::nranks, me = global_variable::my_rank;
  int n = static_cast<int>(v.size());
  std::vector<int> cnt(nr, 0), off(nr, 0);
  MPI_Gather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<T> all;
  if (me == 0) {
    long long tot = 0;
    for (int r=0; r<nr; ++r) {off[r] = static_cast<int>(tot); tot += cnt[r];}
    if (tot > std::numeric_limits<int>::max()) {
      std::cerr << "BA fields: too many leaf cells for one MPI_Gatherv" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    all.resize(static_cast<std::size_t>(tot));
  }
  MPI_Gatherv(v.data(), n, type, all.data(), cnt.data(), off.data(), type, 0, MPI_COMM_WORLD);
  if (me == 0) v.swap(all);
}
#endif

// Diagnostic only: compact per-leaf field snapshot.  Gas density and
// momenta, and the dust cloud integral of mass, momentum and the CENTRAL second
// velocity moments about each leaf's own mean (two passes, so a cold stream gives zero
// instead of a cancellation residue).  The deposit is the exact one the run uses: the
// composite-transfer cloud weights, or canonical TSC on a uniform periodic mesh.
// Geometry (x, z, dx, dz per active leaf, f64) is written once; fields are f32.
// Several ranks (2026-10-21): refinable clouds with <dust>/experimental_mpi deposit on the leaves
// of their own and of the halo blocks; the halo parts are added to their owners
// (and, for the second pass, the owners' mass and momentum sent back), and rank 0 writes the
// leaves of all ranks in gid order, i.e. the files of one rank with the same mesh.  The
// canonical TSC deposit of a uniform mesh on several ranks is WriteBAFieldsGlobal.
void WriteBAFields(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto *pdust = pack->pdust;
  const bool composite = pdust->UsesCompositeTransfer();
  const int nranks = global_variable::nranks;
  if (nranks > 1 && !composite) {WriteBAFieldsGlobal(pm); return;}
  if (nranks > 1 && !(pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud &&
                      pdust->experimental_mpi)) {
    std::cerr << "BA fields on several ranks: refinable clouds need <dust>/experimental_mpi"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &ind = pm->mb_indcs;
  int nmb = pack->nmb_thispack;
  int is = ind.is, js = ind.js, ks = ind.ks;
  int nx1 = ind.nx1, nx2 = ind.nx2;
  int ncells1 = nx1 + 2*ind.ng, ncells2 = nx2 + 2*ind.ng;
  const int ncb = ncells1*ncells2;
  auto &pr = pack->ppart->prtcl_rdata;
  auto &pi = pack->ppart->prtcl_idata;
  int npart = pack->ppart->nprtcl_thispack;
  auto &mbsize = pack->pmb->mb_size;
  auto transfer = pdust->smr_transfer;
  auto lt = pdust->leaf_traversal;   // the actual leaves
  // the leaf fields of every block slot with fields, flat ((slot*ncells2 + j)*ncells1 + i): the
  // local blocks, then (several ranks) the halo blocks; f0 mass and momentum, f1 the moments
  const int nslot = (nranks > 1) ? lt.nfld : nmb;
  DvceArray2D<Real> f0("ba_fields0", 4, static_cast<std::size_t>(nslot)*ncb);   // zero
  DvceArray2D<Real> f1("ba_fields1", 3, static_cast<std::size_t>(nslot)*ncb);
  // canonical TSC: global periodic stencil on the uniform root grid -> leaf index
  int n1 = pm->mesh_indcs.nx1, n2 = pm->mesh_indcs.nx2;
  Real gx1 = pm->mesh_size.x1min, gx2 = pm->mesh_size.x2min;
  Real h1 = (pm->mesh_size.x1max - gx1)/n1, h2 = (pm->mesh_size.x2max - gx2)/n2;
  // the old owner_overlap transfer on a multilevel mesh: its own leaf deposit (TSC on the
  // owner blocks, the ghost parts added to the leaves they overlap); one pass, the central
  // moments from the raw ones (a ghost cell feeds leaves with different means)
  const bool oo = pdust->smr_transfer_mode == DustSMRTransfer::owner_overlap && pm->multilevel;
  DualArray2D<int> tgt("ba_tsc_target", (composite || oo) ? 1 : n2, (composite || oo) ? 1 : n1);
  if (!composite && !oo) {
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
  if (oo) {
    DvceArray5D<Real> g;
    pdust->OwnerOverlapDeposit(g, true);
    par_for("ba_fields_oo", DevExeSpace(), 0, nmb*ncb - 1, KOKKOS_LAMBDA(const int k) {
      const int m = k/ncb, r = k % ncb, j = r/ncells1, i = r % ncells1;
      const Real rho = g(m,0,ks,j,i);
      f0(0,k) = rho;
      for (int c=0; c<3; ++c) {
        const Real mom = g(m,1+c,ks,j,i);
        f0(1+c,k) = mom;
        f1(c,k) = (rho > 0.0) ? g(m,4+c,ks,j,i) - mom*mom/rho : 0.0;
      }
    });
  }
  // pass 0: mass and momentum; pass 1: central second moments about the leaf mean
  for (int pass=0; pass<2 && !oo; ++pass) {
    par_for("ba_fields_deposit", DevExeSpace(), 0, npart-1, KOKKOS_LAMBDA(const int p) {
      Real mass = pr(IPM,p);
      Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
      auto add = [&](const int target, const Real w) {
        if (w == 0.0) return;
        int tm = target/ncb;
        // a halo slot: its box from the leaf traversal (computed as mb_size, dust_leaf.cpp;
        // smooth_p2 builds no traversal but has no halo)
        Real area = (tm < nmb) ? mbsize.d_view(tm).dx1*mbsize.d_view(tm).dx2
                               : lt.box(tm,2)*lt.box(tm,3);
        Real d = mass*w/area;
        if (pass == 0) {
          dust::DepositAdd(&f0(0,target), d);
          for (int c=0; c<3; ++c) dust::DepositAdd(&f0(1+c,target), d*v[c]);
        } else {
          Real rho = f0(0,target);
          for (int c=0; c<3; ++c) {
            Real dv = v[c] - f0(1+c,target)/rho;
            dust::DepositAdd(&f1(c,target), d*dv*dv);
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
    // several ranks: the halo parts to their owners; for the second pass the owners' mass and
    // momentum back to the halo copies (no-ops on one rank)
    if (pass == 0) {
      pdust->HaloReverseAddFlat(f0, 4);
      pdust->HaloForwardFlat(f0, 4);
    } else {
      pdust->HaloReverseAddFlat(f1, 3);
    }
  }
  auto h0 = Kokkos::create_mirror_view_and_copy(HostMemSpace(), f0);
  auto h1v = Kokkos::create_mirror_view_and_copy(HostMemSpace(), f1);
  auto fh = [&h0, &h1v, ncb, ncells1](const int m, const int c, const int j, const int i) {
    const std::size_t k = static_cast<std::size_t>(m)*ncb + static_cast<std::size_t>(j)*ncells1 + i;
    return static_cast<double>(c < 4 ? h0(c, k) : h1v(c - 4, k));
  };
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
  long long ntot = npart, ncell_tot = ncell;
#if MPI_PARALLEL_ENABLED
  if (nranks > 1) {
    double sums[2] = {mtot, mdep};
    long long counts[2] = {ntot, ncell_tot};
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    mtot = sums[0]; mdep = sums[1]; ntot = counts[0]; ncell_tot = counts[1];
  }
#endif
  if (std::fabs(mdep - mtot) > 1.0e-8*mtot) {   // rounding of ~1e7 particles on ~4e6 cells (U2048)
    if (global_variable::my_rank == 0)
      std::fprintf(stderr, "BA fields: deposited mass %.17g != particle mass %.17g\n", mdep, mtot);
    std::exit(EXIT_FAILURE);
  }
  if (ba_ppdf_) WriteBAParticlePDF(pm, fh, prh, npart, mtot);
  if (!ba_fields_write_) {
    ++ba_field_number_;
    if (ba_pin_ != nullptr) ba_pin_->SetInteger("problem", "rst_ba_field_number", ba_field_number_);
    return;
  }
#if MPI_PARALLEL_ENABLED
  if (nranks > 1) {
    GatherToRoot(out, MPI_FLOAT);
    if (first) GatherToRoot(geom, MPI_DOUBLE);
  }
#endif
  const bool root = global_variable::my_rank == 0;
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
    std::snprintf(line, sizeof(line), "AthenaK BA field geometry v1 f64\n%lld\n", ncell_tot);
    char gnum[32] = "";
    if (ba_geom_written_) std::snprintf(gnum, sizeof(gnum), ".%05d", ba_field_number_);
    if (root) write(ba_snapshot_basename_ + gnum + ".bafield.geom", line, geom.data(),
                    sizeof(double), geom.size());
    ba_geom_written_ = true;
    ba_geom_seq_ = mesh_seq;
  }
  std::snprintf(line, sizeof(line), "AthenaK BA fields v1 f32\n%.17g %d %lld %d %.17g %lld\n",
                pm->time, pm->ncycle, ncell_tot, kBAFields, mtot, ntot);
  char number[32];
  std::snprintf(number, sizeof(number), ".%05d.baf", ba_field_number_++);
  if (ba_pin_ != nullptr) ba_pin_->SetInteger("problem", "rst_ba_field_number", ba_field_number_);
  if (root) write(ba_snapshot_basename_ + number, line, out.data(), sizeof(float), out.size());
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
//! \fn ScheduleIndex
//! \brief 2026-10-21: the schedule index of the next BA output after a restart at time t.  The
//! stored file number is kept when it fits the interval (dt*(number-1) <= t < dt*number); a
//! restart with another interval (e.g. ba_field_dt on the command line) continues the numbering
//! and starts the schedule at the next multiple of the new interval after t (before, the
//! schedule was dt*number: a smaller interval wrote one frame per history call until it caught
//! up, a larger one wrote none for a long time).

int ScheduleIndex(const int number, const Real dt, const Real t) {
  const bool fits = (dt*number + 1.0e-10 > t) && (dt*(number - 1) <= t + 1.0e-10);
  return fits ? number : static_cast<int>(std::floor((t + 1.0e-10)/dt)) + 1;
}

//----------------------------------------------------------------------------------------
//! \struct RootPlacement
//! \brief The position of root parent q (cloud_root_random): mode 0 the
//! independent random draw (the original hash), 1 a square lattice (ppc = n^2, cell-centred
//! offsets), 2 a generic arrangement for problems along x1: x1 offsets (k + 1/2)/ppc in the root
//! cell, x2 offsets frac((k + 1/2) phi) (phi the golden ratio), the same in every root cell, so
//! that for any integer ppc each x1 sub-lattice is a full shift family of the mesh (a uniform
//! TSC / hat deposit on the root level).  With kwv != 0 the x1 position is displaced by the
//! eigenmode displacement Re[xi_s exp(i k x0)] (dust_test = wave).

struct RootPlacement {
  int mode = 0, ppc = 1, nside = 1, nc1 = 1, nspec = 1;
  std::uint64_t seed = 0;
  Real x1min = 0.0, x2min = 0.0, lx = 1.0, ly = 1.0, hx = 1.0, hy = 1.0;
  Real kwv = 0.0;
  Real xir[4] = {0.0, 0.0, 0.0, 0.0}, xii[4] = {0.0, 0.0, 0.0, 0.0};
  Real dvr[4] = {0.0, 0.0, 0.0, 0.0}, dvi[4] = {0.0, 0.0, 0.0, 0.0};
  KOKKOS_INLINE_FUNCTION void Base(const int q, Real &x, Real &y) const {
    if (mode == 0) {
      std::uint64_t key = static_cast<std::uint64_t>(q) + seed*UINT64_C(0x9e3779b97f4a7c15);
      x = x1min + HashUniform01(key + UINT64_C(0x632be59bd9b4e019))*lx;
      y = x2min + HashUniform01(key + UINT64_C(0x8cb92baa3f3d8dd7))*ly;
      return;
    }
    const int c = q/ppc, k = q - c*ppc;
    const int ci = c % nc1, cj = c/nc1;
    Real fx, fy;
    if (mode == 1) {
      fx = (static_cast<Real>(k % nside) + 0.5)/nside;
      fy = (static_cast<Real>(k/nside) + 0.5)/nside;
    } else {
      fx = (static_cast<Real>(k) + 0.5)/ppc;
      const Real g = (static_cast<Real>(k) + 0.5)*0.6180339887498949;
      fy = g - Kokkos::floor(g);
    }
    x = x1min + (ci + fx)*hx;
    y = x2min + (cj + fy)*hy;
  }
  // the position (displaced, wrapped into the box) and the x1 velocity of the eigenmode
  KOKKOS_INLINE_FUNCTION void operator()(const int q, Real &x, Real &y, Real &vx) const {
    Real x0;
    Base(q, x0, y);
    x = x0; vx = 0.0;
    if (kwv != 0.0) {
      const int s = q % nspec;
      const Real c = Kokkos::cos(kwv*x0), sn = Kokkos::sin(kwv*x0);
      x = x0 + xir[s]*c - xii[s]*sn;
      vx = dvr[s]*c - dvi[s]*sn;
      x -= lx*Kokkos::floor((x - x1min)/lx);
    }
  }
};

//----------------------------------------------------------------------------------------
//! \fn DustTestOmega
//! \brief dust_test = wave: the right-travelling damped root of the dusty-wave dispersion
//! relation omega^2 (1 + sum_s eps_s/(1 - i omega t_s)) = k^2 cs^2 (as dusty_wave.cpp).

std::complex<Real> DustTestOmega(const Real kwv, const Real cs, const std::vector<Real> &eps,
                                 const std::vector<Real> &ts) {
  const std::complex<Real> iunit(0.0, 1.0);
  Real epstot = 0.0;
  for (const Real e : eps) epstot += e;
  std::complex<Real> omega(kwv*cs/std::sqrt(1.0 + epstot), 0.0);
  for (int it=0; it<100; ++it) {
    std::complex<Real> ssum(0.0, 0.0), dsum(0.0, 0.0);
    for (std::size_t s=0; s<eps.size(); ++s) {
      const std::complex<Real> den = 1.0 - iunit*omega*ts[s];
      ssum += eps[s]/den;
      dsum += eps[s]*iunit*ts[s]/(den*den);
    }
    const std::complex<Real> f = omega*omega*(1.0 + ssum) - SQR(kwv*cs);
    const std::complex<Real> fp = 2.0*omega*(1.0 + ssum) + omega*omega*dsum;
    const std::complex<Real> d = f/fp;
    omega -= d;
    if (std::abs(d) < 1.0e-15*std::abs(omega)) break;
  }
  return omega;
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
  // The several-rank field writer on one rank (test) is gone
  dust::RemovedKey(pin, "ba_fields_global_test",
                   !ba_fields_ || dust::KeyFalse(pin, "ba_fields_global_test", "problem"),
                   "the several-rank field writer on one rank (test)", "problem");
  ba_ppdf_nref_ = pin->GetOrAddInteger("problem", "ba_ppdf_nref", 128);
  if (ba_ppdf_ && !ba_fields_) {
    std::cerr << "ba_ppdf is written with the BA field snapshots: set ba_fields = true"
              << " (ba_fields_write = false skips the field files)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ba_history_ = pin->GetOrAddBoolean("problem", "ba_history", false);
  dust_moments_hist_ = pin->GetOrAddBoolean("problem", "dust_moments_hist", false);
  p3d_dt_ = pin->GetOrAddReal("problem", "p3d_snapshot_dt", -1.0);
  // several ranks: fields of the canonical TSC deposit on a uniform mesh (WriteBAFieldsGlobal), or
  // of refinable clouds with <dust>/experimental_mpi (2026-10-21, WriteBAFields); not ba_history
  const bool fields_mpi = dust_active && !ba_history_ &&
      ((!pmy_mesh_->multilevel && pmbp->pdust->smr_transfer_mode == DustSMRTransfer::none) ||
       (pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud &&
        pmbp->pdust->experimental_mpi));
  if (ba_fields_ || ba_history_) {
    if ((global_variable::nranks != 1 && !fields_mpi) || !dust_active || pmy_mesh_->three_d ||
        !pin->GetOrAddBoolean("problem", "user_hist", false)) {
      std::cerr << "BA fields/history require 2-D dust, user_hist=true and one MPI rank (several "
                << "ranks: fields only, canonical TSC on a uniform mesh or refinable clouds with "
                << "<dust>/experimental_mpi)" << std::endl;
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
  // also the old owner_overlap transfer on any mesh (point particles, each in the block
  // that holds it), so that it starts from the parents of the refinable-cloud runs
  bool canonical_root = cloud_root_random && dust_active &&
      ((pmbp->pdust->smr_transfer_mode == DustSMRTransfer::none && !pmy_mesh_->multilevel) ||
       pmbp->pdust->smr_transfer_mode == DustSMRTransfer::owner_overlap);
  // several ranks: refinable clouds with <dust>/experimental_mpi; every rank
  // draws the same parents and keeps those in its blocks
  const bool ranks_ok = (global_variable::nranks == 1) ||
      (dust_active && pmbp->pdust->experimental_mpi && !canonical_root);
  if (cloud_root_random && (!ranks_ok || !dust_active ||
      !(pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud ||
        canonical_root) || pmy_mesh_->three_d)) {
    std::cerr << "cloud_root_random requires single-rank (or <dust>/experimental_mpi) 2-D "
              << "dust with refinable_cloud transfer, or (one rank) the canonical deposit on a "
              << "uniform mesh or the owner_overlap transfer" << std::endl;
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
      ba_snapshot_k_ = ScheduleIndex(ba_snapshot_number_, ba_snapshot_dt_, pmy_mesh_->time);
    }
    if (ba_fields_) {
      ba_field_number_ = pin->GetOrAddInteger("problem", "rst_ba_field_number",
          static_cast<int>(std::ceil((pmy_mesh_->time - 1.0e-10)/ba_field_dt_)));
      ba_field_k_ = ScheduleIndex(ba_field_number_, ba_field_dt_, pmy_mesh_->time);
      if (ba_field_k_ != ba_field_number_ && global_variable::my_rank == 0) {
        std::cout << "# BA fields: the interval changed at the restart: frames numbered from "
                  << ba_field_number_ << ", next at t = " << ba_field_dt_*ba_field_k_ << std::endl;
      }
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
  // <problem>/dust_test = damping | wave, the one-dimensional drag tests without
  // rotation on the root parents (cloud_root_random): the deceleration of uniform dust moving at
  // <problem>/dust_test_v0 through gas at rest, or the damped dusty sound wave (amp, wave_n
  // wavelengths along x1); <problem>/root_placement = random | lattice | generic
  const std::string dust_test = pin->GetOrAddString("problem", "dust_test", "none");
  const bool dtest = (dust_test != "none");
  if (dtest && dust_test != "damping" && dust_test != "wave") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/dust_test must be none, damping or wave" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // in 3-D with the lattice (or random) placement instead of the root parents
  if (dtest && (!dust_active || pin->DoesBlockExist("shearing_box") ||
                (pmy_mesh_->three_d && cloud_root_random))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/dust_test needs dust (2-D: root clouds or the lattice; 3-D: the "
              << "lattice) and no <shearing_box>" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!dtest && !(pin->DoesBlockExist("shearing_box"))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH test requires a <shearing_box> block" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real omega0 = 0.0, qshear = 0.0, etavk = 0.0;
  std::vector<Real> eps, taus;
  if (dtest) {
    const int ns = pmbp->pdust->nspecies;
    eps.resize(ns); taus.resize(ns);
    const bool has_eps = pin->DoesParameterExist("problem", "eps_1");
    for (int s=0; s<ns; ++s) {
      eps[s] = has_eps ? pin->GetReal("problem", "eps_" + std::to_string(s+1))
                       : pmbp->pdust->dust_to_gas/static_cast<Real>(ns);
      taus[s] = pmbp->pdust->taus.h_view(s);
    }
    pgen_final_func = nullptr;   // no NSH equilibrium to check
  } else {
    ReadNSHParams(pin, pmbp, omega0, qshear, etavk, eps, taus);
  }
  int nspec = static_cast<int>(eps.size());
  Real ax = 2.0*omega0*etavk;

  // the radial forcing must be applied to the gas by the standard const_accel source
  // term (so that dust particles do not feel it); verify the input file sets it up
  if (!dtest) {
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
  Real ugx = 0.0, ugp = 0.0;
  std::vector<Real> vx(nspec, 0.0), vp(nspec, 0.0);
  if (dtest) {
    if (dust_test == "damping") {
      const Real v0 = pin->GetOrAddReal("problem", "dust_test_v0", 0.1);
      for (int s=0; s<nspec; ++s) vx[s] = v0;
    }
  } else {
    SolveNSH(omega0, qshear, ax, eps, taus, ugx, ugp, vx, vp);
  }
  if (global_variable::my_rank == 0 && !dtest) {
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
  // dust_test = wave: the gas part of the eigenmode (velocity normalisation, as dusty_wave.cpp)
  RootPlacement place;
  if (dust_test == "wave") {
    if (nspec > 4) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "dust_test = wave: at most 4 species" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real amp = pin->GetOrAddReal("problem", "amp", 1.0e-6);
    const int nwave = pin->GetOrAddInteger("problem", "wave_n", 1);
    const Real cs = pmbp->phydro->peos->eos_data.iso_cs;
    const Real kwv = 2.0*M_PI*nwave/(pmy_mesh_->mesh_size.x1max - pmy_mesh_->mesh_size.x1min);
    const std::complex<Real> iunit(0.0, 1.0);
    const std::complex<Real> omega = DustTestOmega(kwv, cs, eps, taus);
    const std::complex<Real> du = amp*cs, drho = (kwv/omega)*rho0*du;
    place.kwv = kwv;
    for (int s=0; s<nspec; ++s) {
      const std::complex<Real> dv = du/(1.0 - iunit*omega*taus[s]);
      const std::complex<Real> xi = iunit*dv/omega;   // displacement: d(xi)/dt = dv
      place.xir[s] = xi.real(); place.xii[s] = xi.imag();
      place.dvr[s] = dv.real(); place.dvi[s] = dv.imag();
    }
    if (global_variable::my_rank == 0) {
      std::cout << std::setprecision(15) << "# dust_test wave: k = " << kwv << ", omega = "
                << omega.real() << " " << omega.imag() << " i, |drho| = " << std::abs(drho)
                << std::endl;
    }
    const Real drr = drho.real(), dri = drho.imag(), dur = du.real(), dui = du.imag();
    const int is0 = indcs.is, nx1 = indcs.nx1;
    auto &mbs = pmbp->pmb->mb_size;
    par_for("dust_test_gas", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real x = CellCenterX(i - is0, nx1, mbs.d_view(m).x1min, mbs.d_view(m).x1max);
      const Real c = cos(kwv*x), sn = sin(kwv*x);
      const Real rho = rho0 + drr*c - dri*sn, v = dur*c - dui*sn;
      u0(m,IDN,k,j,i) = rho;
      u0(m,IM1,k,j,i) = rho*v;
      u0(m,IM2,k,j,i) = 0.0;
      u0(m,IM3,k,j,i) = 0.0;
    });
  }

  // <problem>/stratified_dust: the hydrostatic gas (also without dust, as a control);
  // the dust part follows the particle set-up below
  if (pin->GetOrAddBoolean("problem", "stratified_dust", false) && three_d &&
      pin->GetOrAddBoolean("shearing_box", "stratified", false)) {
    const Real hgas = pmbp->phydro->peos->eos_data.iso_cs/omega0;
    const int ks0 = indcs.ks, nz3 = indcs.nx3;
    auto &mbs = pmbp->pmb->mb_size;
    par_for("strat_gas", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real z = CellCenterX(k - ks0, nz3, mbs.d_view(m).x3min, mbs.d_view(m).x3max);
      const Real g = exp(-0.5*z*z/(hgas*hgas));
      for (int v=IDN; v<=IM3; ++v) u0(m,v,k,j,i) *= g;
    });
  }

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
  // <problem>/lattice_sub: in 3-D the ppc particles of a cell on an n^3 sub-lattice
  // (offsets (a + 1/2)/n) instead of all at the cell centre; ppc = n^3 times nspecies
  int nsub = 1;
  if (pin->GetOrAddBoolean("problem", "lattice_sub", false)) {
    nsub = static_cast<int>(std::lround(std::cbrt(static_cast<double>(ppc_int/nspec))));
    if (!three_d || nsub*nsub*nsub*nspec != ppc_int) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "lattice_sub: 3-D and ppc = n^3 * nspecies" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
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
    {
      const std::string rp = pin->GetOrAddString("problem", "root_placement", "random");
      place.mode = (rp == "lattice") ? 1 : ((rp == "generic") ? 2 : 0);
      place.ppc = static_cast<int>(ppc);
      place.nside = static_cast<int>(std::lround(std::sqrt(ppc)));
      if ((rp != "random" && rp != "lattice" && rp != "generic") ||
          (place.mode == 1 && place.nside*place.nside != place.ppc)) {
        std::cerr << "<problem>/root_placement must be random, lattice (ppc a square) or "
                  << "generic" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      place.nc1 = transfer.ncoarse1;
      place.nspec = nspec;
      place.seed = random_seed;
      place.x1min = transfer.x1min; place.x2min = transfer.x2min;
      place.lx = transfer.x1max - transfer.x1min;
      place.ly = transfer.x2max - transfer.x2min;
      place.hx = transfer.root_dx1; place.hy = transfer.root_dx2;
    }
    // the parents of this rank: those whose leaf is in one of its blocks, in
    // the order of their global index p, which stays the tag (one rank: all of them)
    const int nlocal_mb = nmb;
    DvceArray1D<int> pslot("nsh_root_slot", nparent);
    par_for("nsh_cloud_root_owner", DevExeSpace(), 0, nparent-1, KOKKOS_LAMBDA(const int p) {
      Real x, y, vw;
      place(p, x, y, vw);
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
      Real x, y, vw;
      place(q, x, y, vw);
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
      pr0(IPVX,p) = spdat.d_view(q % nspec,0) + vw;
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
      if (nsub > 1) {   // the sub-lattice point of this particle (species cycle fastest)
        const int r = (q % ppc_int)/nspec;
        const int a = r % nsub, b = (r/nsub) % nsub, cc = r/(nsub*nsub);
        pr(IPX,p) += ((a + 0.5)/nsub - 0.5)*mbsize.d_view(m).dx1;
        pr(IPY,p) += ((b + 0.5)/nsub - 0.5)*mbsize.d_view(m).dx2;
        pr(IPZ,p) += ((cc + 0.5)/nsub - 0.5)*mbsize.d_view(m).dx3;
      }
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
    // dust_test = wave in 3-D: the eigenmode displacement and velocity of the lattice
    if (place.kwv != 0.0) {
      const Real x0 = pr(IPX,p);
      const Real c = Kokkos::cos(place.kwv*x0), sn = Kokkos::sin(place.kwv*x0);
      pr(IPX,p) = x0 + place.xir[s]*c - place.xii[s]*sn;
      pr(IPVX,p) += place.dvr[s]*c - place.dvi[s]*sn;
    }
  });
  }

  // <problem>/deptest: a test of the general deposit remap (3-D, one rank): every
  // particle is placed at random within deptest_w of the two x1 faces, with a random mass, then
  // moved by deptest_shift along x1 (crossing a shear-periodic face: y shifted by the offset of
  // time deptest_t, as a crossing particle); the remapped mass density is written
  // (DustGasDrag::DepositRemapTest) and the run stops
  if (pin->GetOrAddBoolean("problem", "deptest", false)) {
    const Real w = pin->GetReal("problem", "deptest_w");
    const Real shx = pin->GetOrAddReal("problem", "deptest_shift", 0.0);
    const Real tt = pin->GetOrAddReal("problem", "deptest_t", 0.0);
    const std::string tag = pin->GetOrAddString("problem", "deptest_tag", "a");
    const auto &ms = pmy_mesh_->mesh_size;
    const Real lx = ms.x1max - ms.x1min, ly = ms.x2max - ms.x2min, lz = ms.x3max - ms.x3min;
    const bool shear = pin->DoesBlockExist("shearing_box") &&
        pmy_mesh_->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic;
    const Real ysh = shear ? std::fmod(pin->GetReal("shearing_box", "qshear")*
                                       pin->GetReal("shearing_box", "omega0")*lx*tt, ly) : 0.0;
    // the same global set of deptest_n particles on every rank; each keeps those in its blocks
    // (in the order of the global index, so a block's deposit does not depend on the ranks)
    const int nglob = pin->GetOrAddInteger("problem", "deptest_n", ppar->nprtcl_thispack);
    const Real x1a = ms.x1min, x2a = ms.x2min, x3a = ms.x3min;
    const Real x1b = ms.x1max, x2b = ms.x2max;
    const int nmbl = nmb;
    auto pos = KOKKOS_LAMBDA(const int q, Real &x, Real &y, Real &z, Real &mq) {
      std::uint64_t key = static_cast<std::uint64_t>(q) + random_seed*UINT64_C(0x9e3779b97f4a7c15);
      const Real u0 = HashUniform01(key + UINT64_C(0x632be59bd9b4e019));
      const Real u1 = HashUniform01(key + UINT64_C(0x8cb92baa3f3d8dd7));
      const Real u2 = HashUniform01(key + UINT64_C(0x58f38ded6f7c55b5));
      const Real u3 = HashUniform01(key + UINT64_C(0x1f83d9abfb41bd6b));
      x = (u0 < 0.5) ? (x1b - 2.0*u0*w) : (x1a + 2.0*(u0 - 0.5)*w);
      y = x2a + u1*ly; z = x3a + u2*lz;
      x += shx;
      if (x >= x1b) {x -= lx; y += ysh;}
      if (x < x1a) {x += lx; y -= ysh;}
      if (y >= x2b) y -= ly;
      if (y < x2a) y += ly;
      mq = 1.0 + u3;
    };
    DvceArray1D<int> qslot("nsh_deptest_slot", std::max(1, nglob));
    par_for("nsh_deptest_owner", DevExeSpace(), 0, nglob-1, KOKKOS_LAMBDA(const int q) {
      Real x, y, z, mq;
      pos(q, x, y, z, mq);
      int mm = -1;
      for (int b=0; b<nmbl; ++b) {
        if (x >= mbsize.d_view(b).x1min && x < mbsize.d_view(b).x1max &&
            y >= mbsize.d_view(b).x2min && y < mbsize.d_view(b).x2max &&
            z >= mbsize.d_view(b).x3min && z < mbsize.d_view(b).x3max) mm = b;
      }
      qslot(q) = mm;
    });
    DvceArray1D<int> qnew("nsh_deptest_new", std::max(1, nglob));
    int nkeep = 0;
    Kokkos::parallel_scan("nsh_deptest_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, nglob),
    KOKKOS_LAMBDA(const int q, int &upd, const bool final) {
      if (final) qnew(q) = (qslot(q) >= 0) ? upd : -1;
      if (qslot(q) >= 0) ++upd;
    }, nkeep);
    Kokkos::realloc(ppar->prtcl_rdata, ppar->nrdata, std::max(1, nkeep));
    Kokkos::realloc(ppar->prtcl_idata, ppar->nidata, std::max(1, nkeep));
    ppar->nprtcl_thispack = nkeep;
    auto prt = ppar->prtcl_rdata;
    auto pit = ppar->prtcl_idata;
    const int nrd = ppar->nrdata;
    par_for("nsh_deptest_set", DevExeSpace(), 0, nglob-1, KOKKOS_LAMBDA(const int q) {
      const int p = qnew(q);
      if (p < 0) return;
      Real x, y, z, mq;
      pos(q, x, y, z, mq);
      for (int c=0; c<nrd; ++c) prt(c,p) = 0.0;
      prt(IPX,p) = x; prt(IPY,p) = y; prt(IPZ,p) = z; prt(IPM,p) = mq;
      pit(PGID,p) = gids + qslot(q);
      pit(PTAG,p) = q;
    });
    if (!three_d) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "deptest: 3-D only" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    pmbp->pdust->DepositRemapTest(tt, tag);
#if MPI_PARALLEL_ENABLED
    MPI_Finalize();
#endif
    std::exit(EXIT_SUCCESS);
  }

  // S3c: <problem>/stratified_dust (3-D shearing box with <shearing_box>/stratified):
  // isothermal hydrostatic gas rho0 exp(-z^2/2H^2), H = cs/omega0, the velocities of the set-up
  // above (NSH, zero for etavk = 0), dust masses times exp(-z^2/2 Hp^2) (<problem>/dust_hp,
  // default H: the dust-to-gas ratio of the set-up everywhere), <problem>/dust_vz0 added to vz
  if (pin->GetOrAddBoolean("problem", "stratified_dust", false)) {
    if (!three_d || !pin->GetOrAddBoolean("shearing_box", "stratified", false) || cloud_root_random) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "stratified_dust: 3-D, <shearing_box>/stratified = true, lattice or random "
                << "placement" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real hgas = pmbp->phydro->peos->eos_data.iso_cs/omega0;
    const Real hp = pin->GetOrAddReal("problem", "dust_hp", hgas);
    const Real vz0 = pin->GetOrAddReal("problem", "dust_vz0", 0.0);
    auto &prs = ppar->prtcl_rdata;
    par_for("strat_dust", DevExeSpace(), 0, ppar->nprtcl_thispack-1, KOKKOS_LAMBDA(const int p) {
      const Real z = prs(IPZ,p);
      prs(IPM,p) *= exp(-0.5*z*z/(hp*hp));
      prs(IPVZ,p) += vz0;
    });
    if (global_variable::my_rank == 0) {
      std::cout << "# stratified dust: H = " << hgas << ", Hp = " << hp << ", vz0 = " << vz0
                << std::endl;
    }
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
      ba_field_k_ = ba_field_number_;
    }
    if (ba_snapshots_) {
      ba_snapshot_number_ =
          static_cast<int>(std::ceil((pmy_mesh_->time - 1.0e-10)/ba_snapshot_dt_));
      ba_snapshot_k_ = ba_snapshot_number_;
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
  // the particle snapshots ride on the history calls
  if (p3d_dt_ > 0.0 && pm->time >= p3d_number_*p3d_dt_ - 1.0e-10*p3d_dt_) WriteP3DSnapshot(pm);
  if (dust_moments_hist_) {   // all species: sum m, m vx, m vz, m z^2
    int h = pdata->nhist;
    if (h + 4 > NHISTORY_VARIABLES) {
      std::cerr << "dust_moments_hist: too many history columns" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const char *labels[4] = {"msum", "mvx", "mvz", "mz2"};
    for (int c=0; c<4; ++c) pdata->label[h+c] = labels[c];
    Real s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    Kokkos::parallel_reduce("dust_mom_hist",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &a0, Real &a1, Real &a2, Real &a3) {
      const Real m = pr(IPM,p);
      a0 += m; a1 += m*pr(IPVX,p); a2 += m*pr(IPVZ,p); a3 += m*pr(IPZ,p)*pr(IPZ,p);
    }, Kokkos::Sum<Real>(s0), Kokkos::Sum<Real>(s1), Kokkos::Sum<Real>(s2),
       Kokkos::Sum<Real>(s3));
    pdata->hdata[h] = s0; pdata->hdata[h+1] = s1; pdata->hdata[h+2] = s2; pdata->hdata[h+3] = s3;
    pdata->nhist = h + 4;
    // with <dust>/x3_ghost_deposit = drop: the momentum of the dropped back-reaction deposits
    const auto &ds = pm->pmb_pack->pdust->pbval_dm->dropped_sum;
    if (ds.size() == 3 && pdata->nhist + 3 <= NHISTORY_VARIABLES) {
      const char *dl[3] = {"dropmx", "dropmy", "dropmz"};
      for (int c=0; c<3; ++c) {
        pdata->label[pdata->nhist + c] = dl[c];
        pdata->hdata[pdata->nhist + c] = static_cast<Real>(ds[c]);
      }
      pdata->nhist += 3;
    }
  }
  if (ba_history_) {
    // mass-weighted dust moments (all species) and the cumulative cloud
    // machinery counters; the merge passes themselves are in cloud_merge_log.txt
    int h = pdata->nhist;   // after any columns above (3*nsout without them)
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
  if (ba_snapshots_ && pm->time + 1.0e-10 >= ba_snapshot_dt_*ba_snapshot_k_) {
    WriteBASnapshot(pm);
    ++ba_snapshot_k_;
  }
  if (ba_fields_ && pm->time + 1.0e-10 >= ba_field_dt_*ba_field_k_) {
    WriteBAFields(pm);
    ++ba_field_k_;
  }
  return;
}
