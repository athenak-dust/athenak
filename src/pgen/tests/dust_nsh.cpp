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
#include <sstream>
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
#include "dust/dust_species.hpp"
#include "outputs/outputs.hpp"

// functions for error output and user history
void DustNSHErrors(ParameterInput *pin, Mesh *pm);
void DustNSHHistory(HistoryData *pdata, Mesh *pm);
void RunAuditReplay(ParameterInput *pin, Mesh *pm);   // streaming_linear.cpp
// the outflow vertical boundaries of Li et al. (2018) and the mass renormalisation
void LiVerticalBCs(Mesh *pm);
void LiVerticalBCsCoarse(Mesh *pm);
void LiRenormalize(Mesh *pm);

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
        pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p), static_cast<double>(GetTag(pi,p)),
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
// Diagnostic: the x momentum of the dust (m vx) and of the gas (M1 dV) as
// compensated (Neumaier) sums on the host, columns mvx_c and gpx_c, to tell a summation
// error of the ordinary history columns from a conservation error.
bool dust_moments_exact_ = false;
// <problem>/vertical_bc = li2018: the outflow vertical boundaries of Li, Youdin &
// Simon (2018, §2.2; Simon et al. 2011, appendix) as in Athena's par_strat3d.c (strat_ix3,
// strat_ox3, mass_cons / Userwork_in_loop): in the ghost cells the density of the last physical
// cell times exp(-(z^2 - z_b^2)/(2 H^2)), its x and y velocities, its z velocity unless that
// flows into the box (then 0); after every cycle the gas density of the whole box is scaled back
// to the initial mean (the momenta are left as they are) with the floor vbc_density_floor
struct LiVBC {
  bool on = false;
  Real h2 = 1.0;          // (cs/omega0)^2
  Real floor = 1.0e-4;    // Athena's D_FLOOR
  Real mean0 = 0.0;       // the initial mean gas density of the box
  Real ratio = 1.0;       // the ratio of the last renormalisation
  Real dm = 0.0;          // the gas mass put back so far
};
LiVBC livbc_;
Real LiGasMass(Mesh *pm);      // below
Real LiBoxVolume(Mesh *pm);
// 3-D particle snapshots: every p3d_snapshot_dt, per rank p3d.NNNNN.rRRRRR.bin with
// "P3D1", the time (double), the particle count (int), then x, y, z, m (float32 arrays)
Real p3d_dt_ = -1.0;
Real p3d_t0_ = 0.0;   // frame k at p3d_t0 + k p3d_dt (p3d_snapshot_t0)
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
  // native restart: the number of the next frame travels in the restart file
  if (ba_pin_ != nullptr) ba_pin_->SetInteger("problem", "rst_p3d_number", p3d_number_);
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
  // unequal ppc_N: the slots of a root cell in species blocks, every species on its
  // own sub-lattice (sslot, skk: species and index within it of slot k; sppc, snside: per
  // species); legacy and equal ppc_N: the interleaved layout, species q % nspec
  bool blocks = false;
  DvceArray1D<int> sslot, skk, sppc, snside;
  // per-species H_p of the Gaussian sampling (empty: the single hp)
  DvceArray1D<Real> shp;
  KOKKOS_INLINE_FUNCTION int Species(const int q) const {
    return blocks ? sslot(q % ppc) : q % nspec;
  }
  // the slot of parent q within its root cell, on the sub-lattice of its species
  KOKKOS_INLINE_FUNCTION void Slot(const int k, int &kl, int &pp, int &ns) const {
    if (blocks) {
      const int sp = sslot(k);
      kl = skk(k); pp = sppc(sp); ns = snside(sp);
    } else {
      kl = k; pp = ppc; ns = nside;
    }
  }
  KOKKOS_INLINE_FUNCTION void Base(const int q, Real &x, Real &y) const {
    if (mode == 0) {
      std::uint64_t key = static_cast<std::uint64_t>(q) + seed*UINT64_C(0x9e3779b97f4a7c15);
      x = x1min + HashUniform01(key + UINT64_C(0x632be59bd9b4e019))*lx;
      y = x2min + HashUniform01(key + UINT64_C(0x8cb92baa3f3d8dd7))*ly;
      return;
    }
    const int c = q/ppc, k0 = q - c*ppc;
    int k, pp, ns;
    Slot(k0, k, pp, ns);
    const int ci = c % nc1, cj = c/nc1;
    Real fx, fy;
    if (mode == 1) {
      fx = (static_cast<Real>(k % ns) + 0.5)/ns;
      fy = (static_cast<Real>(k/ns) + 0.5)/ns;
    } else {
      fx = (static_cast<Real>(k) + 0.5)/pp;
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
      const int s = Species(q);
      const Real c = Kokkos::cos(kwv*x0), sn = Kokkos::sin(kwv*x0);
      x = x0 + xir[s]*c - xii[s]*sn;
      vx = dvr[s]*c - dvi[s]*sn;
      x -= lx*Kokkos::floor((x - x1min)/lx);
    }
  }
  // 3-D refinable clouds: root cells nc1 x nc2 x nc3; mode 1 a cubic lattice (ppc = n^3),
  // mode 2 x1 offsets (k + 1/2)/ppc and the R2 sequence frac((k + 1/2) g1), frac((k + 1/2) g2) in
  // x2, x3 (g1, g2 the inverse plastic number and its square), mode 0 an independent random draw
  int nc2 = 1;
  Real x3min = 0.0, lz = 1.0, hz = 1.0;
  // gauss: mode 0 with z ~ N(0, hp) truncated to the box (Box-Muller, drawn again
  // with the next key when outside; Athena's par_strat3d.c draws Normal() until inside)
  bool gauss = false;
  Real hp = 1.0;
  // <problem>/dust_zcut > 0: the Gaussian truncated to |z| < zcut as well
  Real zcut = 0.0;
  KOKKOS_INLINE_FUNCTION void Base3(const int q, Real &x, Real &y, Real &z) const {
    if (mode == 0) {
      std::uint64_t key = static_cast<std::uint64_t>(q) + seed*UINT64_C(0x9e3779b97f4a7c15);
      x = x1min + HashUniform01(key + UINT64_C(0x632be59bd9b4e019))*lx;
      y = x2min + HashUniform01(key + UINT64_C(0x8cb92baa3f3d8dd7))*ly;
      if (gauss) {
        z = 0.0;
        const Real hps = (shp.extent_int(0) > 0) ? shp(Species(q)) : hp;
        for (int t=0; t<64; ++t) {
          const std::uint64_t kz = key + UINT64_C(0x58f38ded6f7c55b5) +
                                   static_cast<std::uint64_t>(t)*UINT64_C(0x2545f4914f6cdd1d);
          const Real u1 = Kokkos::fmax(HashUniform01(kz), 1.0e-300);
          const Real u2 = HashUniform01(kz + UINT64_C(0x1d8e4e27c47d124f));
          const Real zz = hps*Kokkos::sqrt(-2.0*Kokkos::log(u1))*Kokkos::cos(2.0*M_PI*u2);
          if (zz >= x3min && zz < x3min + lz && (zcut <= 0.0 || Kokkos::fabs(zz) < zcut)) {
            z = zz;
            break;
          }
        }
        return;
      }
      z = x3min + HashUniform01(key + UINT64_C(0x58f38ded6f7c55b5))*lz;
      return;
    }
    const int c = q/ppc, k0 = q - c*ppc;
    int k, pp, ns;
    Slot(k0, k, pp, ns);
    const int ci = c % nc1, cj = (c/nc1) % nc2, ck = c/(nc1*nc2);
    Real fx, fy, fz;
    if (mode == 1) {
      fx = (static_cast<Real>(k % ns) + 0.5)/ns;
      fy = (static_cast<Real>((k/ns) % ns) + 0.5)/ns;
      fz = (static_cast<Real>(k/(ns*ns)) + 0.5)/ns;
    } else {
      fx = (static_cast<Real>(k) + 0.5)/pp;
      const Real g1 = (static_cast<Real>(k) + 0.5)*0.7548776662466927;
      const Real g2 = (static_cast<Real>(k) + 0.5)*0.5698402909980532;
      fy = g1 - Kokkos::floor(g1);
      fz = g2 - Kokkos::floor(g2);
    }
    x = x1min + (ci + fx)*hx;
    y = x2min + (cj + fy)*hy;
    z = x3min + (ck + fz)*hz;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const int q, Real &x, Real &y, Real &z,
                                         Real &vx) const {
    Real x0;
    Base3(q, x0, y, z);
    x = x0; vx = 0.0;
    if (kwv != 0.0) {
      const int s = Species(q);
      const Real c = Kokkos::cos(kwv*x0), sn = Kokkos::sin(kwv*x0);
      x = x0 + xir[s]*c - xii[s]*sn;
      vx = dvr[s]*c - dvi[s]*sn;
      x -= lx*Kokkos::floor((x - x1min)/lx);
    }
  }
};

// <problem>/nsh_vertical: the multi-species NSH drift with the local dust-to-gas
// ratio eps_s(z) = eps_s exp(-z^2 (1/Hp^2 - 1/H^2)/2), solved on the host on a fine table in z
// (Athena's par_strat3d.c with ipert = 0 solves MultiNSH at the cell faces); columns u_gx, u_gphi,
// then v_x, v_phi of each species; linear interpolation on the device
struct NSHTableZ {
  DvceArray2D<Real> t;
  Real z0 = 0.0, dz = 1.0;
  int nt = 0;
  KOKKOS_INLINE_FUNCTION Real At(const Real z, const int c) const {
    Real f = (z - z0)/dz;
    int i = static_cast<int>(Kokkos::floor(f));
    i = (i < 0) ? 0 : ((i > nt - 2) ? nt - 2 : i);
    f -= static_cast<Real>(i);
    return (1.0 - f)*t(i,c) + f*t(i + 1,c);
  }
};

// the drag tests in 3-D (dust_test with 3-D refinable clouds): the parameters of the
// exact solution for the errors that DustNSHHistory computes in the code
struct DustTest3D {
  bool on = false, wave = false;
  bool nsh = false;   // the NSH equilibrium (shearing box): gas at (ux0, uy0), uniform dust
  Real ux0 = 0.0, uy0 = 0.0;
  // the NSH dust with the density rho0 eps (1 + amp cos(ky y)) at t = 0, advected
  // with the drift (vx, vy) and the shear: the phase ky (y - vy t + q Omega (x t - vx t^2/2));
  // e_rhod is then the L1 error over amp, y_rhod the same within two root cells of an x1 face
  bool ymode = false;
  Real yamp = 0.0, ky = 0.0, yvx = 0.0, yvy = 0.0, qom = 0.0, hface = 0.0;
  Real x1min = 0.0, x1max = 0.0;
  Real rho0 = 1.0, eps = 1.0, ts = 1.0, v0 = 0.0;
  Real kwv = 0.0, wr = 0.0, wi = 0.0;
  Real dgr = 0.0, dgi = 0.0, dur = 0.0, dui = 0.0, ddr = 0.0, ddi = 0.0;   // drho_g, du, drho_d
};
DustTest3D dtest3d_;
KOKKOS_INLINE_FUNCTION
void DTest3DCell(const int q, const int nx1, const int nx2, const int nx3, const int is,
                 const int js, const int ks, int &m, int &k, int &j, int &i) {
  i = q % nx1;
  int r = q/nx1;
  j = r % nx2;
  r /= nx2;
  k = r % nx3;
  m = r/nx3;
  i += is; j += js; k += ks;
}

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

// the midplane ratios of the new species keys as the problem generator resolved
// them (Z_N converted); empty before (a restarted run: epsilon_N directly, Z_N recomputed there)
std::vector<Real> nsh_eps_resolved;

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
  // <problem>/eps_N: to be deprecated after the shared epsilon_N/Z_N species
  // API is implemented; preserve this pgen's reference-density meaning.
  bool has_eps = pin->DoesParameterExist("problem","eps_1");
  for (int s=0; s<ns; ++s) {
    if (has_eps) {
      eps[s] = pin->GetReal("problem","eps_" + std::to_string(s+1));
    } else {
      eps[s] = pdust->dust_to_gas/static_cast<Real>(ns);
    }
    taus[s] = pdust->taus.h_view(s);
  }
  if (pdust != nullptr && dust::Species(pin).new_init) {
    const dust::SpeciesTable &spt = dust::Species(pin);
    for (int s=0; s<ns; ++s) eps[s] = static_cast<Real>(spt.value[s]);
    if (static_cast<int>(nsh_eps_resolved.size()) == ns) eps = nsh_eps_resolved;
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
  dust_moments_exact_ = pin->GetOrAddBoolean("problem", "dust_moments_exact", false);
  p3d_dt_ = pin->GetOrAddReal("problem", "p3d_snapshot_dt", -1.0);
  p3d_t0_ = pin->GetOrAddReal("problem", "p3d_snapshot_t0", 0.0);
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
      (dust_active && pmbp->pdust->experimental_mpi && !canonical_root) ||
      (dust_active && canonical_root && pmy_mesh_->three_d);   // 3-D point particles
  // 3-D with refinable clouds (<dust>/experimental_3d) or the canonical deposit
  // (several ranks: the clouds only, ranks_ok)
  const bool root3d_ok = dust_active && (pmbp->pdust->Clouds3D() || canonical_root);
  if (cloud_root_random && (!ranks_ok || !dust_active ||
      !(pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud ||
        canonical_root) || (pmy_mesh_->three_d && !root3d_ok))) {
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
  // Drag-only tests do not have an NSH equilibrium or a shearing-box block.
  // Install this consistently before the restart branch returns as well.
  const bool drag_test = pin->GetOrAddString("problem", "dust_test", "none") != "none";
  pgen_final_func = check_equilibrium && !drag_test ? DustNSHErrors : nullptr;
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
  // <problem>/vertical_bc = li2018: the outflow vertical boundaries of Li et al.
  // (2018) with the mass renormalisation after every cycle; x3 faces of type user
  {
    const std::string vbc = pin->GetOrAddString("problem", "vertical_bc", "none");
    if (vbc == "li2018") {
      const auto &mbc = pmy_mesh_->mesh_bcs;
      if (!pmy_mesh_->three_d || pmbp->phydro == nullptr ||
          mbc[BoundaryFace::inner_x3] != BoundaryFlag::user ||
          mbc[BoundaryFace::outer_x3] != BoundaryFlag::user ||
          !pin->GetOrAddBoolean("shearing_box", "stratified", false)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "vertical_bc = li2018: 3-D hydro, <shearing_box>/stratified = true, "
                  << "ix3_bc = ox3_bc = user" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const Real cs = pmbp->phydro->peos->eos_data.iso_cs;
      const Real om = pin->GetReal("shearing_box", "omega0");
      livbc_.on = true;
      livbc_.h2 = (cs/om)*(cs/om);
      livbc_.floor = pin->GetOrAddReal("problem", "vbc_density_floor", 1.0e-4);
      user_bcs_func = LiVerticalBCs;
      user_bcs_coarse_func = LiVerticalBCsCoarse;
      user_work_in_loop_func = LiRenormalize;
      user_work_keeps_particles = true;   // it rescales the gas density only
      if (restart) {   // the initial mean density and the running total from the restart file
        livbc_.mean0 = pin->GetReal("problem", "vbc_mean0");
        livbc_.dm = pin->GetOrAddReal("problem", "vbc_dm", 0.0);
      }
    } else if (vbc != "none") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/vertical_bc must be none or li2018" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  if (restart) {
    // native restart: gas, particles and the dust counters come from the
    // file.  The output numbers continue from the values the file carries in <problem>; a file
    // without them falls back on the time.  No file is touched here (the run directory is set
    // later); the distribution file is cut back and continued at its next record.
    if (p3d_dt_ > 0.0) {   // the 3-D particle snapshots continue their numbering
      p3d_number_ = pin->GetOrAddInteger("problem", "rst_p3d_number",
          std::max(0, static_cast<int>(std::ceil((pmy_mesh_->time - p3d_t0_ - 1.0e-10)/p3d_dt_))));
    }
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
    if (pmy_mesh_->pmb_pack->pdust != nullptr && pmy_mesh_->pmb_pack->ppart != nullptr) {
      // 2026-10-26 fix: the total over the ranks; a rank may hold no particles (blocks far from
      // the midplane: u640 seed 4 on 128 ranks stopped here on every such rank)
      long long np = pmy_mesh_->pmb_pack->ppart->nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
      MPI_Allreduce(MPI_IN_PLACE, &np, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
      // The typed restart reader already distinguishes a missing/corrupt
      // particle extension from valid zero-count sections. All dust may have
      // left an outflow domain; that is a legitimate gas-only continuation.
      if (np == 0 && global_variable::my_rank == 0)
        std::cout << "# dust restart: valid empty particle population restored" << std::endl;
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
  // in 3-D with the lattice (or random) placement instead of the root parents;
  // or with the 3-D root parents of refinable clouds (<dust>/experimental_3d) or of the
  // canonical deposit
  if (dtest && (!dust_active || pin->DoesBlockExist("shearing_box") ||
                (pmy_mesh_->three_d && cloud_root_random && !root3d_ok))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/dust_test needs dust (2-D: root clouds or the lattice; 3-D: the "
              << "lattice, or the root parents of 3-D refinable clouds) and no <shearing_box>"
              << std::endl;
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
    // <problem>/eps_N: to be deprecated with the shared species API.
    // Keep damping/wave compatibility identical to the NSH reader above.
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
  // the new species keys: abundances (epsilon_N, or Z_N converted to the midplane
  // ratio below), per-species ppc and H_p from the resolved table; legacy inputs untouched
  const dust::SpeciesTable *spt = dust_active ? &dust::Species(pin) : nullptr;
  const bool new_init = (spt != nullptr) && spt->new_init;
  const bool strat_setup = new_init && pmy_mesh_->three_d &&
      pin->GetOrAddBoolean("problem", "stratified_dust", false) &&
      pin->DoesBlockExist("shearing_box") &&
      pin->GetOrAddBoolean("shearing_box", "stratified", false);
  std::vector<Real> hps;            // per-species H_p (new keys, stratified)
  std::vector<Real> finside;        // fraction of each species' Gaussian inside its sampling range
  Real mg_box = 0.0;                // the initial gas mass of the box (Z with domain)
  if (new_init) {
    if (spt->abundance == "Z" && !strat_setup) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Z_N (column ratios) need the stratified 3-D set-up (<problem>/stratified_dust "
                << "and <shearing_box>/stratified); use epsilon_N otherwise" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int sp=0; sp<nspec; ++sp) eps[sp] = static_cast<Real>(spt->value[sp]);
    if (strat_setup) {
      const Real hg = pmbp->phydro->peos->eos_data.iso_cs/omega0;
      const auto hv = dust::SpeciesHp(pin, *spt, hg, false);
      hps.assign(hv.begin(), hv.end());
      const auto &ms = pmy_mesh_->mesh_size;
      const bool gs = pin->GetOrAddString("problem", "stratified_dust_sampling", "weight") ==
                      "gaussian";
      const Real zc = gs ? pin->GetOrAddReal("problem", "dust_zcut", 0.0) : 0.0;
      const Real zlo = (zc > 0.0) ? std::max(ms.x3min, -zc) : ms.x3min;
      const Real zhi = (zc > 0.0) ? std::min(ms.x3max, zc) : ms.x3max;
      finside.resize(nspec);
      for (int sp=0; sp<nspec; ++sp) {
        const Real r2 = std::sqrt(2.0)*hps[sp];
        finside[sp] = 0.5*(std::erf(zhi/r2) - std::erf(zlo/r2));
      }
      if (spt->abundance == "Z" && spt->column_normalization == "domain") {
        // the gas mass of the box as the initialisation below sets it (cell centres)
        const Real rho0g = pin->GetOrAddReal("problem", "rho0", 1.0);
        auto &ind = pmy_mesh_->mb_indcs;
        auto &mbs = pmbp->pmb->mb_size;
        const int ks0 = ind.ks, nz3 = ind.nx3, nx = ind.nx1, ny = ind.nx2;
        const int nmbl = pmbp->nmb_thispack;
        double mg = 0.0;
        Kokkos::parallel_reduce("nsh_gas_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmbl*nz3),
        KOKKOS_LAMBDA(const int q, double &a) {
          const int m = q/nz3, k = q % nz3;
          const Real z = CellCenterX(k, nz3, mbs.d_view(m).x3min, mbs.d_view(m).x3max);
          const Real rho = rho0g*exp(-0.5*z*z/(hg*hg));
          a += static_cast<double>(rho)*mbs.d_view(m).dx1*mbs.d_view(m).dx2*mbs.d_view(m).dx3*nx*ny;
        }, mg);
        (void)ks0;
#if MPI_PARALLEL_ENABLED
        MPI_Allreduce(MPI_IN_PLACE, &mg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        mg_box = mg;
      }
      if (spt->abundance == "Z") {
        // the midplane ratio of each Gaussian: full_profile Z H_g/H_p (the whole column), domain
        // the column that puts Z M_gas(box) into the sampling range
        const Real rho0g = pin->GetOrAddReal("problem", "rho0", 1.0);
        const Real area = (ms.x1max - ms.x1min)*(ms.x2max - ms.x2min);
        for (int sp=0; sp<nspec; ++sp) {
          if (spt->column_normalization == "full_profile") {
            eps[sp] = static_cast<Real>(spt->value[sp])*hg/hps[sp];
          } else {
            eps[sp] = static_cast<Real>(spt->value[sp])*mg_box/
                      (rho0g*std::sqrt(2.0*M_PI)*hps[sp]*finside[sp]*area);
          }
        }
      }
    }
  }

  if (new_init) nsh_eps_resolved = eps;   // the reference of DustNSHErrors

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
  // <problem>/nsh_vertical: the NSH drift with the local dust-to-gas ratio
  // eps_s(z) = eps_s exp(-z^2 (1/Hp^2 - 1/H^2)/2) for the gas (the mean of the two cell faces) and
  // the particles (their height), on a table of at least 64 points per root cell in z
  const bool nsh_vertical = !dtest && dust_active && pmy_mesh_->three_d &&
                            pin->GetOrAddBoolean("problem", "nsh_vertical", false);
  NSHTableZ nshtab;
  if (nsh_vertical) {
    if (!pin->GetOrAddBoolean("problem", "stratified_dust", false) ||
        !pin->GetOrAddBoolean("shearing_box", "stratified", false)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "nsh_vertical needs stratified_dust in a stratified shearing box" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real hg = pmbp->phydro->peos->eos_data.iso_cs/omega0;
    // the new keys: per-species H_p, and a truncated profile is not renormalised
    const Real hpz = new_init ? hps[0] : pin->GetOrAddReal("problem", "dust_hp", hg);
    const auto &ms = pmy_mesh_->mesh_size;
    const int nt = std::max(4097, 64*pmy_mesh_->mesh_indcs.nx3 + 1);
    nshtab.nt = nt;
    nshtab.z0 = ms.x3min;
    nshtab.dz = (ms.x3max - ms.x3min)/static_cast<Real>(nt - 1);
    nshtab.t = DvceArray2D<Real>("nsh_ztab", nt, 2 + 2*nspec);
    auto th = Kokkos::create_mirror_view(nshtab.t);
    const Real a = 1.0/(hpz*hpz) - 1.0/(hg*hg);
    // <problem>/dust_zcut: the truncated profile of the Gaussian sampling, the
    // same total dust mass (higher by 1/erf inside, zero outside)
    const Real zcut = pin->GetOrAddReal("problem", "dust_zcut", 0.0);
    const Real fcut = (zcut > 0.0) ? 1.0/std::erf(zcut/(std::sqrt(2.0)*hpz)) : 1.0;
    std::vector<Real> as(nspec, a), fcs(nspec, fcut);
    if (new_init) {
      for (int s=0; s<nspec; ++s) {as[s] = 1.0/(hps[s]*hps[s]) - 1.0/(hg*hg); fcs[s] = 1.0;}
    }
    std::vector<Real> ez(nspec), vxz(nspec, 0.0), vpz(nspec, 0.0);
    for (int n=0; n<nt; ++n) {
      const Real z = nshtab.z0 + n*nshtab.dz;
      const bool in = (zcut <= 0.0 || std::fabs(z) < zcut);
      for (int s=0; s<nspec; ++s) ez[s] = in ? eps[s]*fcs[s]*std::exp(-0.5*z*z*as[s]) : 0.0;
      Real gx = 0.0, gp = 0.0;
      SolveNSH(omega0, qshear, ax, ez, taus, gx, gp, vxz, vpz);
      th(n,0) = gx; th(n,1) = gp;
      for (int s=0; s<nspec; ++s) {th(n,2 + 2*s) = vxz[s]; th(n,3 + 2*s) = vpz[s];}
    }
    Kokkos::deep_copy(nshtab.t, th);
    if (global_variable::my_rank == 0) {
      const int n0 = (nt - 1)/2;
      std::cout << "# NSH(z): " << nt << " heights, Hp = " << hpz << "; at z = "
                << nshtab.z0 + n0*nshtab.dz << ": u_gx = " << th(n0,0) << " u_gphi = " << th(n0,1)
                << " v_x = " << th(n0,2) << " v_phi = " << th(n0,3) << std::endl;
    }
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
  // <problem>/gas_vx_amp, gas_vx_ny: a non-axisymmetric gas velocity u_x += A sin(k_y y)
  // (gas_vx_ny wavelengths along x2; divergence free) for the check of the frame in which the drag
  // samples the gas inside a step of the 3-D shearing box (the cell averages of the sine)
  {
    const Real gamp = pin->GetOrAddReal("problem", "gas_vx_amp", 0.0);
    if (gamp != 0.0 && three_d) {
      const auto &ms = pmy_mesh_->mesh_size;
      const Real ky = 2.0*M_PI*pin->GetOrAddReal("problem", "gas_vx_ny", 1.0)/(ms.x2max - ms.x2min);
      auto &mbs = pmbp->pmb->mb_size;
      const int js0 = indcs.js, nx2 = indcs.nx2;
      par_for("nsh_gas_vx_ky", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
      KOKKOS_LAMBDA(int m, int k, int j, int i) {
        const Real dy = mbs.d_view(m).dx2;
        const Real y = CellCenterX(j - js0, nx2, mbs.d_view(m).x2min, mbs.d_view(m).x2max);
        const Real avg = (cos(ky*(y - 0.5*dy)) - cos(ky*(y + 0.5*dy)))/(ky*dy);   // <sin> over the cell
        u0(m,IM1,k,j,i) += u0(m,IDN,k,j,i)*gamp*avg;
      });
    }
  }
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
    if (three_d && nspec == 1) {   // the exact solution for the 3-D history errors
      const std::complex<Real> dv = du/(1.0 - iunit*omega*taus[0]);
      const std::complex<Real> drd = (kwv/omega)*eps[0]*rho0*dv;
      dtest3d_.wave = true;
      dtest3d_.kwv = kwv; dtest3d_.wr = omega.real(); dtest3d_.wi = omega.imag();
      dtest3d_.dgr = drho.real(); dtest3d_.dgi = drho.imag();
      dtest3d_.dur = du.real(); dtest3d_.dui = du.imag();
      dtest3d_.ddr = drd.real(); dtest3d_.ddi = drd.imag();
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
    const bool nshz = nsh_vertical;
    const NSHTableZ ntab = nshtab;
    par_for("strat_gas", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
    KOKKOS_LAMBDA(int m, int k, int j, int i) {
      const Real z = CellCenterX(k - ks0, nz3, mbs.d_view(m).x3min, mbs.d_view(m).x3max);
      const Real g = exp(-0.5*z*z/(hgas*hgas));
      for (int v=IDN; v<=IM3; ++v) u0(m,v,k,j,i) *= g;
      if (nshz) {   // the NSH drift at this height: the mean of the two faces
        const Real dz = mbs.d_view(m).dx3, rho = u0(m,IDN,k,j,i);
        u0(m,IM1,k,j,i) = rho*0.5*(ntab.At(z - 0.5*dz, 0) + ntab.At(z + 0.5*dz, 0));
        u0(m,IM2,k,j,i) = rho*0.5*(ntab.At(z - 0.5*dz, 1) + ntab.At(z + 0.5*dz, 1));
        u0(m,IM3,k,j,i) = 0.0;
      }
    });
  }
  // the initial mean gas density of the box for the renormalisation (17 digits in
  // <problem>/vbc_mean0, which the restart files carry)
  if (livbc_.on) {
    livbc_.mean0 = LiGasMass(pmy_mesh_)/LiBoxVolume(pmy_mesh_);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", livbc_.mean0);
    pin->SetString("problem", "vbc_mean0", buf);
    if (global_variable::my_rank == 0) {
      std::cout << "# vertical_bc = li2018: initial mean gas density " << buf << std::endl;
    }
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
  // the new keys: ppc_total = sum of ppc_N particles per cell; equal ppc_N keep the
  // legacy interleaved layout, unequal ones put each species in a block of slots
  const bool sblocks = new_init && !spt->equal_ppc;
  const int ptot = new_init ? spt->ppc_total : nspec;
  const bool bad_layout = new_init ?
      ((!random_placement && ((npart_permb != ppc_int*ncells) || (ppc_int != ptot))) ||
       (random_placement && (npart % ptot != 0))) :
      ((!random_placement &&
        ((npart_permb != ppc_int*ncells) || (ppc_int % nspec != 0))) ||
       (random_placement && (npart % nspec != 0)));
  if (bad_layout) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH lattice placement requires <particles>/ppc to be an integer "
              << "multiple of <dust>/nspecies; random placement requires the total "
              << "number of particles per rank to be divisible by <dust>/nspecies"
              << " (new keys: the sum of ppc_N per cell)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // the slot layout of a cell (q % ptot -> species and index within it) and, per species, its
  // ppc and the side of its square / cubic sub-lattice (0 if it is not a square / cube)
  DualArray1D<int> lay_s("nsh_lay_s", std::max(1, ptot)), lay_k("nsh_lay_k", std::max(1, ptot));
  DualArray1D<int> lay_ppc("nsh_lay_ppc", nspec), lay_n2("nsh_lay_n2", nspec);
  DualArray1D<int> lay_n3("nsh_lay_n3", nspec);
  for (int k=0; k<ptot; ++k) {
    int sp = k % nspec, kk = k/nspec;
    if (new_init) spt->SlotSpecies(k, sp, kk);
    lay_s.h_view(k) = sp; lay_k.h_view(k) = kk;
  }
  for (int sp=0; sp<nspec; ++sp) {
    const int pp = new_init ? spt->ppc[sp] : 1;
    const int n2 = static_cast<int>(std::lround(std::sqrt(static_cast<double>(pp))));
    const int n3 = static_cast<int>(std::lround(std::cbrt(static_cast<double>(pp))));
    lay_ppc.h_view(sp) = pp;
    lay_n2.h_view(sp) = (n2*n2 == pp) ? n2 : 0;
    lay_n3.h_view(sp) = (n3*n3*n3 == pp) ? n3 : 0;
  }
  for (auto *a : {&lay_s, &lay_k, &lay_ppc, &lay_n2, &lay_n3}) {
    a->template modify<HostMemSpace>();
    a->template sync<DevExeSpace>();
  }
  int lnx1 = indcs.nx1, lnx2 = indcs.nx2, lnx3 = indcs.nx3;
  // <problem>/lattice_sub: in 3-D the ppc particles of a cell on an n^3 sub-lattice
  // (offsets (a + 1/2)/n) instead of all at the cell centre; ppc = n^3 times nspecies
  int nsub = 1;
  if (pin->GetOrAddBoolean("problem", "lattice_sub", false)) {
    if (sblocks) {   // every species on its own cubic sub-lattice
      bool ok = three_d;
      for (int sp=0; sp<nspec; ++sp) ok = ok && (lay_n3.h_view(sp) > 0);
      if (!ok) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "lattice_sub with unequal ppc_N: 3-D and every ppc_N a cube" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      nsub = 2;   // (> 1: the sub-lattice is on; the side is per species)
    } else {
      nsub = static_cast<int>(std::lround(std::cbrt(static_cast<double>(ppc_int/nspec))));
      if (!three_d || nsub*nsub*nsub*nspec != ppc_int) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "lattice_sub: 3-D and ppc = n^3 * nspecies" << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  Real ppc = static_cast<Real>(dust::RootPpc(pin));   // new species keys: sum of ppc_N
  std::uint64_t random_seed = static_cast<std::uint64_t>(
      pin->GetOrAddInteger("problem","random_seed",1));
  auto &taus_ = pmbp->pdust->taus;
  // per-species device table: [s][0]=v_x, [s][1]=v_phi, [s][2]=mass factor
  DualArray2D<Real> spdat("nsh_spdat",nspec,3);
  for (int s=0; s<nspec; ++s) {
    spdat.h_view(s,0) = vx[s];
    spdat.h_view(s,1) = vp[s];
    // unequal ppc_N: eps_s rho0/ppc_s; legacy and equal ppc_N: the same value
    // as eps_s rho0 nspec/ppc, written as before (bit for bit the legacy masses)
    spdat.h_view(s,2) = sblocks ? eps[s]*rho0/static_cast<Real>(spt->ppc[s])
                                : eps[s]*rho0*static_cast<Real>(nspec)/ppc;
  }
  spdat.template modify<HostMemSpace>();
  spdat.template sync<DevExeSpace>();
  // the new species keys: per species the target and the actual dust mass (all
  // ranks) in the log; Z with column_normalization = domain is scaled to Z M_gas(box) exactly
  auto species_audit = [&](const bool allow_rescale) {
    if (!new_init) return;
    auto prs = ppar->prtcl_rdata;
    auto pis = ppar->prtcl_idata;
    const int np = ppar->nprtcl_thispack;
    std::vector<double> act(nspec, 0.0);
    for (int sp=0; sp<nspec; ++sp) {
      double a = 0.0;
      Kokkos::parallel_reduce("nsh_species_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, np),
      KOKKOS_LAMBDA(const int p, double &v) {if (pis(PSP,p) == sp) v += prs(IPM,p);}, a);
      act[sp] = a;
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, act.data(), nspec, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    const auto &ms = pmy_mesh_->mesh_size;
    const Real area = (ms.x1max - ms.x1min)*(ms.x2max - ms.x2min);
    const Real vbox = three_d ? area*(ms.x3max - ms.x3min) : area;
    const bool domain = (spt->abundance == "Z" && spt->column_normalization == "domain");
    std::vector<double> tgt(nspec);
    for (int sp=0; sp<nspec; ++sp) {
      if (!strat_setup) {
        tgt[sp] = eps[sp]*rho0*vbox;
      } else if (domain) {
        tgt[sp] = spt->value[sp]*mg_box;
      } else {
        tgt[sp] = eps[sp]*rho0*std::sqrt(2.0*M_PI)*hps[sp]*finside[sp]*area;
      }
    }
    // the scaling corrects a real mismatch (the weight-sampled lattice); a gaussian sample has the
    // target already, and its sum here is only accurate to round-off (serial: N eps)
    bool rescale = false;
    for (int sp=0; sp<nspec; ++sp) {
      if (act[sp] > 0.0 && std::fabs(act[sp]/tgt[sp] - 1.0) > 1.0e-11) rescale = true;
    }
    if (domain && allow_rescale && rescale) {
      DualArray1D<Real> fac("nsh_species_rescale", nspec);
      for (int sp=0; sp<nspec; ++sp) fac.h_view(sp) = (act[sp] > 0.0) ? tgt[sp]/act[sp] : 1.0;
      fac.template modify<HostMemSpace>();
      fac.template sync<DevExeSpace>();
      auto f = fac.d_view;
      par_for("nsh_species_rescale", DevExeSpace(), 0, np-1, KOKKOS_LAMBDA(const int p) {
        prs(IPM,p) *= f(pis(PSP,p));
      });
    }
    if (global_variable::my_rank == 0) {
      std::ostringstream o;
      o << std::setprecision(17) << "# dust species masses"
        << (strat_setup ? ", stratified" : "") << ":\n";
      for (int sp=0; sp<nspec; ++sp) {
        o << "#   species " << sp + 1 << ": " << spt->abundance << " = " << spt->value[sp];
        if (strat_setup) {
          o << ", H_p = " << hps[sp] << ", midplane epsilon0 = " << eps[sp]
            << ", fraction of the profile in the sampling range " << finside[sp];
        }
        o << ", target mass " << tgt[sp] << ", particle mass " << act[sp] << ", rel_error "
          << std::setprecision(3) << ((tgt[sp] > 0.0) ? act[sp]/tgt[sp] - 1.0 : 0.0)
          << std::setprecision(17)
          << ((domain && allow_rescale && rescale) ? " (before the scaling to the target)" : "")
          << "\n";
      }
      std::cout << o.str() << std::flush;
    }
  };

  // the 3-D history errors (DustNSHHistory), with the root parents (cloud_root_random)
  if (dtest && three_d && nspec == 1 && cloud_root_random) {
    dtest3d_.on = true;
    dtest3d_.rho0 = rho0; dtest3d_.eps = eps[0]; dtest3d_.ts = taus[0];
    dtest3d_.v0 = (dust_test == "damping") ? pin->GetOrAddReal("problem", "dust_test_v0", 0.1)
                                          : 0.0;
  }
  // the NSH equilibrium of 3-D refinable clouds in the shearing box, with the same
  // error columns (gas against its equilibrium, dust density against eps rho0)
  if (!dtest && three_d && nspec == 1 && cloud_root_random && pmbp->pdust->Clouds3D()) {
    dtest3d_.on = true;
    dtest3d_.nsh = true;
    dtest3d_.rho0 = rho0; dtest3d_.eps = eps[0]; dtest3d_.ts = taus[0];
    dtest3d_.ux0 = ugx; dtest3d_.uy0 = ugp;
    const Real yamp = pin->GetOrAddReal("problem", "dust_ymode_amp", 0.0);
    if (yamp != 0.0) {
      const auto &ms = pmy_mesh_->mesh_size;
      dtest3d_.ymode = true;
      dtest3d_.yamp = yamp;
      dtest3d_.ky = 2.0*M_PI*pin->GetOrAddReal("problem", "dust_ymode_n", 1.0)/(ms.x2max - ms.x2min);
      dtest3d_.yvx = vx[0]; dtest3d_.yvy = vp[0]; dtest3d_.qom = qshear*omega0;
      dtest3d_.hface = 2.0*(ms.x1max - ms.x1min)/pmy_mesh_->mesh_indcs.nx1;
      dtest3d_.x1min = ms.x1min; dtest3d_.x1max = ms.x1max;
    }
  }
  if (cloud_root_random && three_d) {
    // the root parents in 3-D (refinable clouds with <dust>/experimental_3d, or the
    // canonical deposit on a uniform mesh), one rank: random, lattice (ppc = n^3) or generic
    const auto &ms = pmy_mesh_->mesh_size;
    const auto &mi = pmy_mesh_->mesh_indcs;
    const int nc1 = mi.nx1, nc2 = mi.nx2, nc3 = mi.nx3;
    const int nparent = static_cast<int>(ppc)*nc1*nc2*nc3;
    if (ppc != static_cast<Real>(static_cast<int>(ppc)) || nparent <= 0) {
      std::cerr << "cloud_root_random requires positive integer root ppc" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const std::string rp = pin->GetOrAddString("problem", "root_placement", "random");
    place.mode = (rp == "lattice") ? 1 : ((rp == "generic") ? 2 : 0);
    place.ppc = static_cast<int>(ppc);
    place.nside = static_cast<int>(std::lround(std::cbrt(ppc)));
    bool lattice_ok = (place.nside*place.nside*place.nside == place.ppc);
    if (sblocks) {   // every species on its own lattice: every ppc_N a cube
      lattice_ok = true;
      for (int sp=0; sp<nspec; ++sp) lattice_ok = lattice_ok && (lay_n3.h_view(sp) > 0);
      place.blocks = true;
      place.sslot = lay_s.d_view; place.skk = lay_k.d_view;
      place.sppc = lay_ppc.d_view; place.snside = lay_n3.d_view;
    }
    if ((rp != "random" && rp != "lattice" && rp != "generic") ||
        (place.mode == 1 && !lattice_ok)) {
      std::cerr << "<problem>/root_placement must be random, lattice (ppc a cube; with unequal "
                << "ppc_N every ppc_N a cube) or generic" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    place.nc1 = nc1; place.nc2 = nc2; place.nspec = nspec; place.seed = random_seed;
    place.x1min = ms.x1min; place.x2min = ms.x2min; place.x3min = ms.x3min;
    place.lx = ms.x1max - ms.x1min; place.ly = ms.x2max - ms.x2min; place.lz = ms.x3max - ms.x3min;
    place.hx = place.lx/nc1; place.hy = place.ly/nc2; place.hz = place.lz/nc3;
    // <problem>/stratified_dust_sampling = gaussian: equal masses, z ~ N(0, Hp)
    // truncated to the box (Li et al. 2018); weight (default): uniform z, masses times
    // exp(-z^2/2 Hp^2)
    const std::string ssamp = pin->GetOrAddString("problem", "stratified_dust_sampling", "weight");
    if (ssamp == "gaussian") {
      if (place.mode != 0 || !pin->GetOrAddBoolean("problem", "stratified_dust", false) ||
          !pin->GetOrAddBoolean("shearing_box", "stratified", false)) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "stratified_dust_sampling = gaussian: stratified_dust, a stratified shearing "
                  << "box and root_placement = random" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      place.gauss = true;
      if (new_init) {   // per-species H_p
        DualArray1D<Real> hd("nsh_species_hp", nspec);
        for (int sp=0; sp<nspec; ++sp) hd.h_view(sp) = hps[sp];
        hd.template modify<HostMemSpace>();
        hd.template sync<DevExeSpace>();
        place.shp = hd.d_view;
        place.hp = hps[0];
      } else {
        place.hp = pin->GetOrAddReal("problem", "dust_hp",
                                     pmbp->phydro->peos->eos_data.iso_cs/omega0);
      }
      // truncation: the same total dust mass (the equal masses are unchanged, all
      // samples inside), so the dust density inside is higher by 1/erf(zcut/(sqrt(2) Hp)); the
      // NSH(z) table uses the same profile (nsh_vertical)
      place.zcut = pin->GetOrAddReal("problem", "dust_zcut", 0.0);
      if (place.zcut > 0.0 && global_variable::my_rank == 0) {
        std::cout << "# dust_zcut: Gaussian truncated to |z| < " << place.zcut;
        if (new_init) {   // the profile is cut, not renormalised
          std::cout << ", the mass of the profile inside only (new species keys)" << std::endl;
        } else {
          std::cout << ", density inside x " << 1.0/std::erf(place.zcut/(std::sqrt(2.0)*place.hp))
                    << std::endl;
        }
      }
    } else if (pin->GetOrAddReal("problem", "dust_zcut", 0.0) > 0.0) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/dust_zcut needs stratified_dust_sampling = gaussian" << std::endl;
      std::exit(EXIT_FAILURE);
    } else if (ssamp != "weight") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/stratified_dust_sampling must be weight or gaussian" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    // <problem>/cloud_seed = leaf: every parent becomes a cloud at the level of the
    // leaf that holds it (no lobes; the P1 split then only acts on the clouds whose supports
    // reach finer leaves), so that the clouds start from the same samples as a uniform-mesh run
    // of particles on the finest grid; root (default): root-level parents split to the leaves
    const std::string cseed = pin->GetOrAddString("problem", "cloud_seed", "root");
    if ((cseed != "root" && cseed != "leaf") || (cseed == "leaf" && canonical_root)) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<problem>/cloud_seed must be root or leaf (leaf: refinable clouds)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const bool seed_leaf = (cseed == "leaf");
    // the parents of this rank (several ranks): every rank draws all of them and
    // keeps those whose leaf is in one of its blocks, in the order of their global index q,
    // which stays the tag (one rank: all of them)
    const auto lt3 = pmbp->pdust->leaf_traversal3d;
    const bool canon = canonical_root;
    const int ncell123 = (indcs.nx1 + 2*indcs.ng)*(indcs.nx2 + 2*indcs.ng)*(indcs.nx3 + 2*indcs.ng);
    DvceArray1D<int> pslot("nsh_root3d_slot", nparent);
    par_for("nsh_cloud_root3d_owner", DevExeSpace(), 0, nparent-1, KOKKOS_LAMBDA(const int q) {
      Real x, y, z, vw;
      place(q, x, y, z, vw);
      int m = 0;
      if (canon) {
        m = nmb;   // not in a block of this rank: dropped (one rank: never)
        for (int b=0; b<nmb; ++b) {
          if (x >= mbsize.d_view(b).x1min && x < mbsize.d_view(b).x1max &&
              y >= mbsize.d_view(b).x2min && y < mbsize.d_view(b).x2max &&
              z >= mbsize.d_view(b).x3min && z < mbsize.d_view(b).x3max) m = b;
        }
      } else {
        const int leaf = lt3.LeafOwner(x, y, z, 0);
        m = (leaf < 0) ? 0 : leaf/ncell123;
      }
      pslot(q) = m;
    });
    DvceArray1D<int> pnew("nsh_root3d_new", nparent);
    int nkeep = 0;
    Kokkos::parallel_scan("nsh_root3d_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, nparent),
    KOKKOS_LAMBDA(const int q, int &upd, const bool final) {
      const bool keep = (pslot(q) < nmb);
      if (final) pnew(q) = keep ? upd : -1;
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
    const int nrdata = ppar->nrdata;
    const Real mass_unit = place.hx*place.hy*place.hz;
    // the y-modulated NSH dust (dust_ymode_amp), the mass of each parent
    const Real yamp = dtest3d_.ymode ? dtest3d_.yamp : 0.0, ky = dtest3d_.ky;
    // <problem>/stratified_dust on the root parents (masses times
    // exp(-z^2/2 Hp^2), dust_vz0 added to vz), before the split
    const bool strat = pin->GetOrAddBoolean("problem", "stratified_dust", false) &&
                       pin->GetOrAddBoolean("shearing_box", "stratified", false);
    const Real shp = (strat && !new_init) ? pin->GetOrAddReal("problem", "dust_hp",
        pmbp->phydro->peos->eos_data.iso_cs/omega0) : 1.0;
    const Real svz0 = strat ? pin->GetOrAddReal("problem", "dust_vz0", 0.0) : 0.0;
    // a uniform dust vz on the root parents of any 3-D set-up; with
    // dust_test_vz_mirror its sign follows z (a state mirrored about z = 0, the image of a box
    // with reflecting x3 faces)
    const Real tvz0 = pin->GetOrAddReal("problem", "dust_test_vz0", 0.0);
    const bool tvzm = pin->GetOrAddBoolean("problem", "dust_test_vz_mirror", false);
    const bool gsamp = place.gauss;
    const Real gfac = std::sqrt(2.0*M_PI)*place.hp/place.lz;
    // per species: the Gaussian factor of the equal masses and the H_p of the weights (legacy:
    // the single values; per species, the Gaussian masses times the fraction of
    // the profile inside the sampling range, so a truncated profile carries less mass)
    DualArray1D<Real> sgf("nsh_species_gfac", nspec), shpv("nsh_species_hpw", nspec);
    for (int sp=0; sp<nspec; ++sp) {
      sgf.h_view(sp) = new_init && strat ? std::sqrt(2.0*M_PI)*hps[sp]/place.lz*finside[sp] : gfac;
      shpv.h_view(sp) = new_init && strat ? hps[sp] : shp;
    }
    sgf.template modify<HostMemSpace>(); sgf.template sync<DevExeSpace>();
    shpv.template modify<HostMemSpace>(); shpv.template sync<DevExeSpace>();
    auto sgfd = sgf.d_view;
    auto shpd = shpv.d_view;
    auto mblev0 = pmbp->pmb->mb_lev;
    const int rootlev0 = pmy_mesh_->root_level;
    const bool nshz = nsh_vertical;
    const NSHTableZ ntab = nshtab;
    par_for("nsh_cloud_root3d", DevExeSpace(), 0, nparent-1, KOKKOS_LAMBDA(const int q) {
      const int p = pnew(q);
      if (p < 0) return;
      Real x, y, z, vw;
      place(q, x, y, z, vw);
      const int sp = place.Species(q);
      pi0(PGID,p) = gids0 + pslot(q);
      SetTag(pi0, p, static_cast<std::uint64_t>(q));
      pi0(PSP,p) = sp;
      pi0(PLEV,p) = seed_leaf ? (mblev0.d_view(pslot(q)) - rootlev0) : 0;
      for (int c=0; c<nrdata; ++c) pr0(c,p) = 0.0;
      pr0(IPX,p) = x; pr0(IPX1,p) = x;
      pr0(IPY,p) = y; pr0(IPY1,p) = y;
      pr0(IPZ,p) = z; pr0(IPZ1,p) = z;
      pr0(IPTS,p) = taus_.d_view(sp);
      pr0(IPM,p) = spdat.d_view(sp,2)*mass_unit*((yamp != 0.0) ? 1.0 + yamp*cos(ky*y) : 1.0);
      pr0(IPVX,p) = spdat.d_view(sp,0) + vw;
      pr0(IPVY,p) = spdat.d_view(sp,1);
      pr0(IPVX1,p) = pr0(IPVX,p);
      pr0(IPVY1,p) = pr0(IPVY,p);
      if (strat) {
        // gaussian: the equal mass eps rho0 sqrt(2 pi) Hp Lx Ly / (parents of the
        // species) = (the uniform root mass) sqrt(2 pi) Hp / Lz
        pr0(IPM,p) *= gsamp ? sgfd(sp) : exp(-0.5*z*z/(shpd(sp)*shpd(sp)));
        pr0(IPVZ,p) = svz0; pr0(IPVZ1,p) = svz0;
      }
      if (nshz) {   // the NSH drift at the height of the parent
        pr0(IPVX,p) = ntab.At(z, 2 + 2*sp) + vw;
        pr0(IPVY,p) = ntab.At(z, 3 + 2*sp);
        pr0(IPVX1,p) = pr0(IPVX,p);
        pr0(IPVY1,p) = pr0(IPVY,p);
      }
      if (tvz0 != 0.0) {
        pr0(IPVZ,p) += (tvzm && z < 0.0) ? -tvz0 : tvz0;
        pr0(IPVZ1,p) = pr0(IPVZ,p);
      }
    });
    if (strat && global_variable::my_rank == 0) {
      std::cout << "# stratified dust (3-D root " << (canonical_root ? "particles" : "clouds")
                << "): Hp = ";
      if (new_init) {for (int sp=0; sp<nspec; ++sp) std::cout << (sp ? "/" : "") << hps[sp];}
      else std::cout << shp;
      std::cout << ", vz0 = " << svz0 << ", sampling "
                << (gsamp ? "gaussian (equal masses)" : "weight") << (nshz ? ", NSH(z)" : "")
                << std::endl;
    }
    species_audit(true);   // before the split
    // the tag counter is set collectively before the splits
    if (!canonical_root) {
      pmbp->pdust->EnsureNextTag();
      // the root cloud mass of G and R (the largest root parent) is taken now,
      // before any split: taken later (at the first G or R) it is the largest root cloud still
      // unsplit, which for stratified masses lies far from the refined midplane (4e-6 of the
      // midplane mass for Hp = 0.02 H and a band |z| < 0.1 H).  Equal masses: the same value.
      if (seed_leaf) {
        // the root cloud mass of G and R: 8^lmax times the largest cloud, so that a
        // seeded cloud on the finest level is one quantum there (the clouds are not root parents).
        // Per species (Codex review): the largest cloud of all species
        // gave a light species the quantum of the heaviest one
        const int nsp = std::max(1, nspec);
        std::vector<Real> mmax(nsp, 0.0);
        int lmax = 0;
        auto prq = ppar->prtcl_rdata;
        auto piq = ppar->prtcl_idata;
        const int npq = ppar->nprtcl_thispack;
        Kokkos::parallel_reduce("nsh_seed_leaf_lev", Kokkos::RangePolicy<>(DevExeSpace(), 0, npq),
        KOKKOS_LAMBDA(const int p, int &l) {
          const int lv = piq(PLEV,p) & 15;   // the level bits of the cloud code
          if (lv > l) l = lv;
        }, Kokkos::Max<int>(lmax));
        for (int sp=0; sp<nsp; ++sp) {
          Kokkos::parallel_reduce("nsh_seed_leaf_m", Kokkos::RangePolicy<>(DevExeSpace(), 0, npq),
          KOKKOS_LAMBDA(const int p, Real &a) {
            if (piq(PSP,p) == sp && prq(IPM,p) > a) a = prq(IPM,p);
          }, Kokkos::Max<Real>(mmax[sp]));
        }
#if MPI_PARALLEL_ENABLED
        MPI_Allreduce(MPI_IN_PLACE, mmax.data(), nsp, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &lmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        const Real f8 = static_cast<Real>(std::int64_t(1) << (3*lmax));
        pmbp->pdust->cloud_merge_mroot.assign(nsp, 0.0);
        for (int sp=0; sp<nsp; ++sp) pmbp->pdust->cloud_merge_mroot[sp] = mmax[sp]*f8;
        if (global_variable::my_rank == 0) {
          std::cout << "# cloud_seed = leaf: clouds on their leaf levels (finest " << lmax
                    << "), root cloud mass of G and R per species 8^" << lmax << " x the largest cloud:";
          for (int sp=0; sp<nsp; ++sp) std::cout << " " << mmax[sp]*f8;
          std::cout << std::endl;
        }
      }
      pmbp->pdust->EnsureRootCloudMass();
      pmbp->pdust->SplitToLeaves();
    }
    long long nnow = ppar->nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &nnow, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank == 0) {
      std::cout << "# 3-D root " << (canonical_root ? "particles (canonical): " : "clouds: ")
                << nparent << " parents -> " << nnow << " particles" << std::endl;
    }
  } else if (cloud_root_random) {
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
      bool lattice_ok = (place.nside*place.nside == place.ppc);
      if (sblocks) {   // every species on its own lattice: every ppc_N a square
        lattice_ok = true;
        for (int sp=0; sp<nspec; ++sp) lattice_ok = lattice_ok && (lay_n2.h_view(sp) > 0);
        place.blocks = true;
        place.sslot = lay_s.d_view; place.skk = lay_k.d_view;
        place.sppc = lay_ppc.d_view; place.snside = lay_n2.d_view;
      }
      if ((rp != "random" && rp != "lattice" && rp != "generic") ||
          (place.mode == 1 && !lattice_ok)) {
        std::cerr << "<problem>/root_placement must be random, lattice (ppc a square; with "
                  << "unequal ppc_N every ppc_N a square) or generic" << std::endl;
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
      SetTag(pi0, p, static_cast<std::uint64_t>(q));
      const int sq = place.Species(q);
      pi0(PSP,p) = sq;
      pi0(PLEV,p) = 0;
      for (int c=0; c<nrdata; ++c) pr0(c,p) = 0.0;
      pr0(IPX,p) = x; pr0(IPX1,p) = x;
      pr0(IPY,p) = y; pr0(IPY1,p) = y;
      pr0(IPTS,p) = taus_.d_view(sq);
      pr0(IPM,p) = spdat.d_view(sq,2)*mass_unit;
      pr0(IPVX,p) = spdat.d_view(sq,0) + vw;
      pr0(IPVZ,p) = spdat.d_view(sq,1);
      pr0(IPVX1,p) = pr0(IPVX,p);
      pr0(IPVZ1,p) = pr0(IPVZ,p);
    });
    species_audit(true);   // before the split
    // the tag counter is set collectively before the splits
    if (!canonical_root) pmbp->pdust->EnsureNextTag();
    if (!canonical_root) pmbp->pdust->SplitToLeaves();
    std::cout << "# BA random root " << (canonical_root ? "particles (canonical): "
              : "clouds: ") << nparent << " parents -> " << ppar->nprtcl_thispack
              << (canonical_root ? " particles" : " particles after P1 presplit")
              << std::endl;
  } else {
  auto lsd = lay_s.d_view, lkd = lay_k.d_view, ln3d = lay_n3.d_view;
  par_for("nsh_part", DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m, s;
    if (random_placement) {
      // Offset the tag stream before hashing so different seeds generate independent
      // position sets.  XOR with the raw tag is not sufficient: for consecutive tags
      // spanning complete power-of-two ranges it only permutes the same set of keys.
      std::uint64_t key = GetTag(pi,p)
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
      s = sblocks ? lsd(p % ptot) : p % nspec;   // the slot layout
    } else {
      m = p/npart_permb;
      if (m > (nmb-1)) {m = nmb-1;}
      int q = p - m*npart_permb;
      int c = q/ppc_int;
      s = sblocks ? lsd(q % ppc_int) : (q % ppc_int) % nspec;
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
        // (unequal ppc_N: the cubic sub-lattice of the particle's species)
        const int r = sblocks ? lkd(q % ppc_int) : (q % ppc_int)/nspec;
        const int ns = sblocks ? ln3d(s) : nsub;
        const int a = r % ns, b = (r/ns) % ns, cc = r/(ns*ns);
        pr(IPX,p) += ((a + 0.5)/ns - 0.5)*mbsize.d_view(m).dx1;
        pr(IPY,p) += ((b + 0.5)/ns - 0.5)*mbsize.d_view(m).dx2;
        pr(IPZ,p) += ((cc + 0.5)/ns - 0.5)*mbsize.d_view(m).dx3;
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
      SetTag(pit, p, static_cast<std::uint64_t>(q));
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
  // (the 3-D root clouds got it in their set-up)
  if (pin->GetOrAddBoolean("problem", "stratified_dust", false) &&
      !(cloud_root_random && three_d && (pmbp->pdust->Clouds3D() || canonical_root))) {
    // (the 3-D root parents, clouds or canonical point particles, got it above)
    if (!three_d || !pin->GetOrAddBoolean("shearing_box", "stratified", false) || cloud_root_random) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "stratified_dust: 3-D, <shearing_box>/stratified = true, lattice or random "
                << "placement" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const Real hgas = pmbp->phydro->peos->eos_data.iso_cs/omega0;
    // the new keys: per-species H_p; legacy: the single dust_hp
    const Real hp = new_init ? hps[0] : pin->GetOrAddReal("problem", "dust_hp", hgas);
    DualArray1D<Real> hpv("nsh_strat_hp", nspec);
    for (int sp=0; sp<nspec; ++sp) hpv.h_view(sp) = new_init ? hps[sp] : hp;
    hpv.template modify<HostMemSpace>(); hpv.template sync<DevExeSpace>();
    auto hpd = hpv.d_view;
    const Real vz0 = pin->GetOrAddReal("problem", "dust_vz0", 0.0);
    auto &prs = ppar->prtcl_rdata;
    auto &pis = ppar->prtcl_idata;
    const bool nshz = nsh_vertical;
    const NSHTableZ ntab = nshtab;
    par_for("strat_dust", DevExeSpace(), 0, ppar->nprtcl_thispack-1, KOKKOS_LAMBDA(const int p) {
      const Real z = prs(IPZ,p);
      const Real h = hpd(pis(PSP,p));
      prs(IPM,p) *= exp(-0.5*z*z/(h*h));
      prs(IPVZ,p) += vz0;
      if (nshz) {   // the NSH drift at the height of the particle
        const int s = pis(PSP,p);
        prs(IPVX,p) = ntab.At(z, 2 + 2*s);
        prs(IPVY,p) = ntab.At(z, 3 + 2*s);
      }
    });
    if (global_variable::my_rank == 0) {
      std::cout << "# stratified dust: H = " << hgas << ", Hp = " << hp << ", vz0 = " << vz0
                << std::endl;
    }
  }
  if (!cloud_root_random) species_audit(true);   // lattice / random placement

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
//! \fn DustTest3DErrors
//! \brief The errors of the 3-D drag tests, with the definitions of dtest_metrics.py
//! computed in the code (leaf cells, volume weighted).  damping: rms(rho_g - rho0),
//! rms(u_x - u(t)), rms(rho_d/(eps rho0) - 1), rms|u_perp|; wave: L1 errors over the amplitudes
//! (gas and dust density as cell averages, u_x at the cell centre) and rms|u_perp|/|du|.  Both:
//! the rms deviation of rho_g and rho_d from their x-column means (the part that is not one-
//! dimensional; over the amplitudes for the wave, rho_d over eps rho0 for damping) and the dust
//! mass on the leaves over the particle mass, minus one.  Dust entries are -1 without clouds.

void DustTest3DErrors(Mesh *pm, Real out[7]) {
  auto *pmbp = pm->pmb_pack;
  auto *pd = pmbp->pdust;
  const DustTest3D c = dtest3d_;
  const bool clouds = pd->Clouds3D();
  DvceArray5D<Real> fd;
  if (clouds) pd->DepositCloudDensity3D(fd);
  auto u0 = pmbp->phydro->u0;
  const auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  auto mbsize = pmbp->pmb->mb_size;
  auto mblev = pmbp->pmb->mb_lev;
  const int root = pm->root_level;
  const int nlev = pm->max_level - pm->root_level;
  const int nf1 = pm->mesh_indcs.nx1 << nlev;
  const Real xmin = pm->mesh_size.x1min;
  DvceArray2D<Real> col("dtest3d_col", (nlev + 1)*nf1, 4);
  const int nmb = pmbp->nmb_thispack;
  const int ncell = nmb*nx3*nx2*nx1;
  const Real t = pm->time;
  const Real eps_rho = c.eps*c.rho0;
  const Real vcm = c.eps*c.v0/(1.0 + c.eps);
  const Real uexact = vcm*(1.0 - std::exp(-t*(1.0 + c.eps)/c.ts));
  const Real grow = std::exp(c.wi*t);
  Real vtot = 0.0, eg = 0.0, eu = 0.0, ed = 0.0, ep = 0.0, mleaf = 0.0;
  Kokkos::parallel_reduce("dtest3d_err", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int q, Real &v_, Real &g_, Real &u_, Real &d_) {
    int m, k, j, i;
    DTest3DCell(q, nx1, nx2, nx3, is, js, ks, m, k, j, i);
    const Real dx = mbsize.d_view(m).dx1;
    const Real vol = dx*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
    const Real rg = u0(m,IDN,k,j,i), ux = u0(m,IM1,k,j,i)/rg;
    const Real rd = clouds ? fd(m,0,k,j,i) : 0.0;
    const Real x = CellCenterX(i - is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    v_ += vol;
    if (c.wave) {
      const Real ph = c.kwv*x - c.wr*t, cp = grow*cos(ph), sp = grow*sin(ph);
      const Real sinc = sin(0.5*c.kwv*dx)/(0.5*c.kwv*dx);
      g_ += vol*fabs(rg - (c.rho0 + (c.dgr*cp - c.dgi*sp)*sinc));
      u_ += vol*fabs(ux - (c.dur*cp - c.dui*sp));
      d_ += vol*fabs(rd - (eps_rho + (c.ddr*cp - c.ddi*sp)*sinc));
    } else if (c.nsh) {
      g_ += vol*SQR(rg - c.rho0);
      u_ += vol*SQR(ux - c.ux0);
      if (c.ymode) {
        const Real y = CellCenterX(j - js, nx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
        const Real dy = mbsize.d_view(m).dx2;
        const Real kx = c.ky*c.qom*t;
        const Real ph = c.ky*(y - c.yvy*t + c.qom*(x*t - 0.5*c.yvx*t*t));
        const Real ay = 0.5*c.ky*dy, ax = 0.5*kx*dx;
        const Real sinc = ((ay == 0.0) ? 1.0 : sin(ay)/ay)*((ax == 0.0) ? 1.0 : sin(ax)/ax);
        d_ += vol*fabs(rd - eps_rho*(1.0 + c.yamp*sinc*cos(ph)));
      } else {
        d_ += vol*SQR(rd/eps_rho - 1.0);
      }
    } else {
      g_ += vol*SQR(rg - c.rho0);
      u_ += vol*SQR(ux - uexact);
      d_ += vol*SQR(rd/eps_rho - 1.0);
    }
    const int l = mblev.d_view(m) - root;
    const int ix = static_cast<int>(floor((x - xmin)/dx));
    const int key = l*nf1 + ix;
    Kokkos::atomic_add(&col(key,0), vol*rg);
    Kokkos::atomic_add(&col(key,1), vol*rd);
    Kokkos::atomic_add(&col(key,2), vol);
  }, Kokkos::Sum<Real>(vtot), Kokkos::Sum<Real>(eg), Kokkos::Sum<Real>(eu), Kokkos::Sum<Real>(ed));
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {   // the column sums of all ranks
    auto ch = Kokkos::create_mirror_view_and_copy(HostMemSpace(), col);
    MPI_Allreduce(MPI_IN_PLACE, ch.data(), static_cast<int>(ch.size()), MPI_ATHENA_REAL, MPI_SUM,
                  MPI_COMM_WORLD);
    Kokkos::deep_copy(col, ch);
  }
#endif
  Real yg = 0.0, yd = 0.0, yf = 0.0, vf = 0.0;
  Kokkos::parallel_reduce("dtest3d_err2", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
  KOKKOS_LAMBDA(const int q, Real &p_, Real &yg_, Real &yd_, Real &ml_) {
    int m, k, j, i;
    DTest3DCell(q, nx1, nx2, nx3, is, js, ks, m, k, j, i);
    const Real dx = mbsize.d_view(m).dx1;
    const Real vol = dx*mbsize.d_view(m).dx2*mbsize.d_view(m).dx3;
    const Real rg = u0(m,IDN,k,j,i);
    const Real rd = clouds ? fd(m,0,k,j,i) : 0.0;
    if (c.nsh) {
      p_ += vol*(SQR(u0(m,IM2,k,j,i)/rg - c.uy0) + SQR(u0(m,IM3,k,j,i)/rg));
    } else {
      p_ += vol*(SQR(u0(m,IM2,k,j,i)) + SQR(u0(m,IM3,k,j,i)))/SQR(rg);
    }
    const Real x = CellCenterX(i - is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    const int key = (mblev.d_view(m) - root)*nf1 + static_cast<int>(floor((x - xmin)/dx));
    yg_ += vol*SQR(rg - col(key,0)/col(key,2));
    yd_ += vol*SQR(rd - col(key,1)/col(key,2));
    ml_ += vol*rd;
  }, Kokkos::Sum<Real>(ep), Kokkos::Sum<Real>(yg), Kokkos::Sum<Real>(yd), Kokkos::Sum<Real>(mleaf));
  if (c.nsh && c.ymode) {   // the y-mode error within two root cells of an x1 face
    Kokkos::parallel_reduce("dtest3d_face", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(const int q, Real &f_, Real &v_) {
      int m, k, j, i;
      DTest3DCell(q, nx1, nx2, nx3, is, js, ks, m, k, j, i);
      const Real dx = mbsize.d_view(m).dx1, dy = mbsize.d_view(m).dx2;
      const Real vol = dx*dy*mbsize.d_view(m).dx3;
      const Real x = CellCenterX(i - is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
      if (x - c.x1min > c.hface && c.x1max - x > c.hface) return;
      const Real y = CellCenterX(j - js, nx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
      const Real rd = clouds ? fd(m,0,k,j,i) : 0.0;
      const Real kx = c.ky*c.qom*t;
      const Real ph = c.ky*(y - c.yvy*t + c.qom*(x*t - 0.5*c.yvx*t*t));
      const Real ay = 0.5*c.ky*dy, ax = 0.5*kx*dx;
      const Real sinc = ((ay == 0.0) ? 1.0 : sin(ay)/ay)*((ax == 0.0) ? 1.0 : sin(ax)/ax);
      f_ += vol*fabs(rd - eps_rho*(1.0 + c.yamp*sinc*cos(ph)));
      v_ += vol;
    }, Kokkos::Sum<Real>(yf), Kokkos::Sum<Real>(vf));
  }
  auto &pr = pmbp->ppart->prtcl_rdata;
  Real mpart = 0.0;
  Kokkos::parallel_reduce("dtest3d_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0,
                          pmbp->ppart->nprtcl_thispack),
  KOKKOS_LAMBDA(const int p, Real &a) {a += pr(IPM,p);}, Kokkos::Sum<Real>(mpart));
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {   // the sums of all ranks (every rank gets the same out)
    Real sums[11] = {vtot, eg, eu, ed, ep, yg, yd, mleaf, mpart, yf, vf};
    MPI_Allreduce(MPI_IN_PLACE, sums, 11, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
    vtot = sums[0]; eg = sums[1]; eu = sums[2]; ed = sums[3]; ep = sums[4]; yg = sums[5];
    yd = sums[6]; mleaf = sums[7]; mpart = sums[8]; yf = sums[9]; vf = sums[10];
  }
#endif
  if (c.wave) {
    const Real ag = std::hypot(c.dgr, c.dgi), au = std::hypot(c.dur, c.dui);
    const Real ad = std::hypot(c.ddr, c.ddi);
    out[0] = eg/vtot/ag; out[1] = eu/vtot/au; out[2] = ed/vtot/ad;
    out[3] = std::sqrt(ep/vtot)/au; out[4] = std::sqrt(yg/vtot)/ag; out[5] = std::sqrt(yd/vtot)/ad;
  } else {
    out[0] = std::sqrt(eg/vtot); out[1] = std::sqrt(eu/vtot); out[2] = std::sqrt(ed/vtot);
    out[3] = std::sqrt(ep/vtot); out[4] = std::sqrt(yg/vtot); out[5] = std::sqrt(yd/vtot)/eps_rho;
    if (c.nsh && c.ymode) {
      out[2] = ed/vtot/(eps_rho*c.yamp);
      out[5] = (vf > 0.0) ? yf/vf/(eps_rho*c.yamp) : 0.0;
    }
  }
  out[6] = (mpart > 0.0) ? mleaf/mpart - 1.0 : 0.0;
  if (!clouds) {out[2] = -1.0; out[5] = -1.0; out[6] = -1.0;}
}

//----------------------------------------------------------------------------------------
//! \fn void DustNSHHistory()
//! \brief User history output: per-species sums of the radial and azimuthal dust drift
//! velocities and particle counts (summed over ranks; divide sums by counts for means).

void DustNSHHistory(HistoryData *pdata, Mesh *pm) {
  if (pm->pmb_pack->ppart == nullptr || pm->pmb_pack->pdust == nullptr) {
    pdata->nhist = 0;
    if (livbc_.on) {   // the gas-only control keeps the renormalisation columns
      const bool r0 = (global_variable::my_rank == 0);
      pdata->label[0] = "vbc_mratio";
      pdata->label[1] = "vbc_dm";
      pdata->hdata[0] = r0 ? livbc_.ratio : 0.0;
      pdata->hdata[1] = r0 ? livbc_.dm : 0.0;
      pdata->nhist = 2;
    }
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
  if (p3d_dt_ > 0.0 && pm->time >= p3d_t0_ + p3d_number_*p3d_dt_ - 1.0e-10*p3d_dt_) WriteP3DSnapshot(pm);
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
  if (dust_moments_exact_ && pdata->nhist + 2 <= NHISTORY_VARIABLES) {
    // compensated sums, rank-local; few ranks add only O(eps |sum|)
    auto prh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pr);
    auto *pack = pm->pmb_pack;
    auto &ind = pm->mb_indcs;
    auto u0h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pack->phydro->u0);
    double sum[2] = {0.0, 0.0}, cmp[2] = {0.0, 0.0};
    auto add = [&](int k, double x) {
      const double t = sum[k] + x;
      cmp[k] += (std::abs(sum[k]) >= std::abs(x)) ? (sum[k] - t) + x : (x - t) + sum[k];
      sum[k] = t;
    };
    for (int p=0; p<npart; ++p) add(0, prh(IPM,p)*prh(IPVX,p));
    for (int m=0; m<pack->nmb_thispack; ++m) {
      const auto &size = pack->pmb->mb_size.h_view(m);
      const double vol = size.dx1*size.dx2*size.dx3;
      for (int k=ind.ks; k<=ind.ke; ++k) {
        for (int j=ind.js; j<=ind.je; ++j) {
          for (int i=ind.is; i<=ind.ie; ++i) add(1, u0h(m,IM1,k,j,i)*vol);
        }
      }
    }
    pdata->label[pdata->nhist] = "mvx_c";
    pdata->label[pdata->nhist + 1] = "gpx_c";
    pdata->hdata[pdata->nhist] = sum[0] + cmp[0];
    pdata->hdata[pdata->nhist + 1] = sum[1] + cmp[1];
    pdata->nhist += 2;
  }
  if (livbc_.on && pdata->nhist + 2 <= NHISTORY_VARIABLES) {
    // the last renormalisation ratio and the gas mass put back so far (rank 0
    // only: the history sums the ranks)
    const bool r0 = (global_variable::my_rank == 0);
    pdata->label[pdata->nhist] = "vbc_mratio";
    pdata->label[pdata->nhist + 1] = "vbc_dm";
    pdata->hdata[pdata->nhist] = r0 ? livbc_.ratio : 0.0;
    pdata->hdata[pdata->nhist + 1] = r0 ? livbc_.dm : 0.0;
    pdata->nhist += 2;
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
  if (dtest3d_.on) {   // the 3-D drag-test errors
    const int h = pdata->nhist;
    const char *labels[7] = {"e_rhog", "e_ux", "e_rhod", "e_uperp", "y_rhog", "y_rhod", "dmass"};
    Real v[7];
    DustTest3DErrors(pm, v);
    // the history adds the ranks: the values once, from rank 0
    const Real f = (global_variable::my_rank == 0) ? 1.0 : 0.0;
    for (int c=0; c<7; ++c) {pdata->label[h+c] = labels[c]; pdata->hdata[h+c] = f*v[c];}
    pdata->nhist = h + 7;
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

//----------------------------------------------------------------------------------------
//! \fn LiVerticalBCs, LiVerticalBCsCoarse
//! \brief The outflow vertical boundaries of Li et al. (2018) on u0 or, for the
//! ghost prolongation next to the x3 faces of a refined mesh, on coarse_u0 with the
//! coarse indices; the same rules as Athena's strat_ix3 / strat_ox3 (isothermal)

namespace {
void LiVerticalBCsApply(Mesh *pm, const bool coarse) {
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  const int nx1 = coarse ? indcs.cnx1 : indcs.nx1;
  const int nx2 = coarse ? indcs.cnx2 : indcs.nx2;
  const int nx3 = coarse ? indcs.cnx3 : indcs.nx3;
  const int n1 = nx1 + 2*ng, n2 = nx2 + 2*ng;
  const int ks = coarse ? indcs.cks : indcs.ks, ke = coarse ? indcs.cke : indcs.ke;
  const int nmb = pmbp->nmb_thispack;
  auto &mb_bcs = pmbp->pmb->mb_bcs;
  auto &size = pmbp->pmb->mb_size;
  auto u = coarse ? pmbp->phydro->coarse_u0 : pmbp->phydro->u0;
  const Real h2 = livbc_.h2;
  par_for("li_vbc", DevExeSpace(), 0, (nmb-1), 0, (n2-1), 0, (n1-1),
  KOKKOS_LAMBDA(int m, int j, int i) {
    const Real zmin = size.d_view(m).x3min, zmax = size.d_view(m).x3max;
    if (mb_bcs.d_view(m, BoundaryFace::inner_x3) == BoundaryFlag::user) {
      const Real zb = CellCenterX(0, nx3, zmin, zmax);
      const Real d = u(m,IDN,ks,j,i);
      const Real v1 = u(m,IM1,ks,j,i)/d, v2 = u(m,IM2,ks,j,i)/d, v3 = u(m,IM3,ks,j,i)/d;
      const bool in = (u(m,IM3,ks,j,i) >= 0.0);   // copied, it would flow into the box
      for (int k=0; k<ng; ++k) {
        const Real z = CellCenterX(-1 - k, nx3, zmin, zmax);
        const Real rho = d*exp(-(z*z - zb*zb)/(2.0*h2));
        u(m,IDN,ks-k-1,j,i) = rho;
        u(m,IM1,ks-k-1,j,i) = rho*v1;
        u(m,IM2,ks-k-1,j,i) = rho*v2;
        u(m,IM3,ks-k-1,j,i) = in ? 0.0 : rho*v3;
      }
    }
    if (mb_bcs.d_view(m, BoundaryFace::outer_x3) == BoundaryFlag::user) {
      const Real zt = CellCenterX(nx3 - 1, nx3, zmin, zmax);
      const Real d = u(m,IDN,ke,j,i);
      const Real v1 = u(m,IM1,ke,j,i)/d, v2 = u(m,IM2,ke,j,i)/d, v3 = u(m,IM3,ke,j,i)/d;
      const bool in = (u(m,IM3,ke,j,i) <= 0.0);
      for (int k=0; k<ng; ++k) {
        const Real z = CellCenterX(nx3 + k, nx3, zmin, zmax);
        const Real rho = d*exp(-(z*z - zt*zt)/(2.0*h2));
        u(m,IDN,ke+k+1,j,i) = rho;
        u(m,IM1,ke+k+1,j,i) = rho*v1;
        u(m,IM2,ke+k+1,j,i) = rho*v2;
        u(m,IM3,ke+k+1,j,i) = in ? 0.0 : rho*v3;
      }
    }
  });
}

// the gas mass of the box (all ranks, the leaf blocks of every level)
Real LiGasMass(Mesh *pm) {
  auto *pmbp = pm->pmb_pack;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nji = nx2*nx1, nkji = nx3*nji, nmkji = pmbp->nmb_thispack*nkji;
  auto &u0 = pmbp->phydro->u0;
  auto &size = pmbp->pmb->mb_size;
  Real sum = 0.0;
  Kokkos::parallel_reduce("li_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, Real &s) {
    const int m = idx/nkji, k = (idx - m*nkji)/nji, j = (idx - m*nkji - k*nji)/nx1;
    const int i = idx - m*nkji - k*nji - j*nx1;
    s += u0(m,IDN,ks+k,js+j,is+i)*size.d_view(m).dx1*size.d_view(m).dx2*size.d_view(m).dx3;
  }, Kokkos::Sum<Real>(sum));
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return sum;
}

Real LiBoxVolume(Mesh *pm) {
  const auto &ms = pm->mesh_size;
  return (ms.x1max - ms.x1min)*(ms.x2max - ms.x2min)*(ms.x3max - ms.x3min);
}
}  // namespace

void LiVerticalBCs(Mesh *pm) {LiVerticalBCsApply(pm, false);}
void LiVerticalBCsCoarse(Mesh *pm) {LiVerticalBCsApply(pm, true);}

//----------------------------------------------------------------------------------------
//! \fn LiRenormalize
//! \brief At the start of every cycle (ProblemGenerator::user_work_in_loop_func):
//! the gas density of the whole box scaled back to the initial mean, the momenta left as they
//! are, with the floor vbc_density_floor (Athena's mass_cons / Userwork_in_loop, which runs after
//! the step).  Every cell, the ghost cells too, then the primitives from the conserved variables
//! everywhere, so that the state stays consistent and a restart continues bitwise.

void LiRenormalize(Mesh *pm) {
  if (!livbc_.on) return;
  const Real vol = LiBoxVolume(pm);
  const Real mean = LiGasMass(pm)/vol;
  const Real ratio = livbc_.mean0/mean;
  livbc_.ratio = ratio;
  livbc_.dm += (livbc_.mean0 - mean)*vol;
  auto *pmbp = pm->pmb_pack;
  auto &u0 = pmbp->phydro->u0;
  auto &w0 = pmbp->phydro->w0;
  const int n1m1 = u0.extent_int(4) - 1, n2m1 = u0.extent_int(3) - 1, n3m1 = u0.extent_int(2) - 1;
  const Real dfloor = livbc_.floor;
  par_for("li_renorm", DevExeSpace(), 0, (pmbp->nmb_thispack-1), 0, n3m1, 0, n2m1, 0, n1m1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = fmax(u0(m,IDN,k,j,i)*ratio, dfloor);
  });
  pmbp->phydro->peos->ConsToPrim(u0, w0, false, 0, n1m1, 0, n2m1, 0, n3m1);
  // the running total travels in the restart files
  if (ba_pin_ != nullptr) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", livbc_.dm);
    ba_pin_->SetString("problem", "vbc_dm", buf);
  }
}
