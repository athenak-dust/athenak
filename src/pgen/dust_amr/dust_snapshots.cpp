//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_snapshots.cpp
//! \brief The 2-D diagnostic outputs of the dust problem generators (dust_nsh,
//! dust_si_unstrat, dust_damping_amr, dusty_wave_amr), keys <problem>/ba_* (named after the
//! 2-D streaming-instability run BA of Johansen & Youdin 2007, for which they were written):
//!   ba_f64_snapshots   gas cells and particles in double precision (one rank), every
//!                      ba_snapshot_dt
//!   ba_fields          per-leaf gas and dust fields (mass, momentum, central second velocity
//!                      moments of the dust), every ba_field_dt; ba_fields_write = false keeps
//!                      only the particle density distribution
//!   ba_ppdf            the distribution of the dust density at the particle positions,
//!                      appended to <basename>.ppdf with every field snapshot
//!   ba_history         mass-weighted dust moments and cloud counters in the user history

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>     // fopen(), fprintf()
#include <unistd.h>   // truncate()
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs/outputs.hpp"
#include "pgen/dust_amr/dust_snapshots.hpp"

namespace dust_snapshots {

namespace {

bool ba_snapshots_ = false;
int ba_snapshot_number_ = 0;
// native restart: the numbers of the next snapshot and field frame are kept
// in the input block <problem> (rst_ba_snapshot_number, rst_ba_field_number), which every
// restart file carries; the distribution file is continued after a restart
ParameterInput *ba_pin_ = nullptr;
bool ba_ppdf_resume_ = false;
Real ba_snapshot_dt_ = 40.0;
// the schedule index of the next snapshot / field frame (due at dt*index), equal to
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
  // several ranks: every rank evaluates its particles on the global grids;
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
  // several ranks (refinable clouds): the grids of the whole mesh, from all ranks
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

// Several ranks: the field snapshot of a uniform mesh with the canonical
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
  // rank, block order) or from another directory
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
// Several ranks: refinable clouds deposit on the leaves
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
  if (nranks > 1 && pdust->smr_transfer_mode != DustSMRTransfer::refinable_cloud) {
    std::cerr << "BA fields on several ranks: refinable clouds, or point particles on a uniform "
              << "mesh" << std::endl;
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
                << "deposit a density either; use the pbin output and an offline estimator)"
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
        // a halo slot: its box from the leaf traversal (computed as mb_size, dust_leaf.cpp)
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
  if (std::fabs(mdep - mtot) > 1.0e-8*mtot) {   // rounding of ~1e7 particles on ~4e6 cells
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
//! \fn ScheduleIndex
//! \brief the schedule index of the next BA output after a restart at time t.  The
//! stored file number is kept when it fits the interval (dt*(number-1) <= t < dt*number); a
//! restart with another interval (e.g. ba_field_dt on the command line) continues the numbering
//! and starts the schedule at the next multiple of the new interval after t (a schedule at
//! dt*number would write one frame per history call until it caught up for a smaller
//! interval, and none for a long time for a larger one).

int ScheduleIndex(const int number, const Real dt, const Real t) {
  const bool fits = (dt*number + 1.0e-10 > t) && (dt*(number - 1) <= t + 1.0e-10);
  return fits ? number : static_cast<int>(std::floor((t + 1.0e-10)/dt)) + 1;
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn Setup
//! \brief reads and checks the keys <problem>/ba_*

void Setup(ParameterInput *pin, Mesh *pm, const bool dust_active) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  ba_pin_ = pin;
  ba_snapshots_ = pin->GetOrAddBoolean("problem", "ba_f64_snapshots", false);
  if (ba_snapshots_) {
    if (global_variable::nranks != 1 || !dust_active || pm->three_d ||
        !pin->GetOrAddBoolean("problem", "user_hist", false)) {
      std::cerr << "BA snapshots require 2-D dust, one MPI rank and user_hist=true"
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
  ba_ppdf_nref_ = pin->GetOrAddInteger("problem", "ba_ppdf_nref", 128);
  if (ba_ppdf_ && !ba_fields_) {
    std::cerr << "ba_ppdf is written with the BA field snapshots: set ba_fields = true"
              << " (ba_fields_write = false skips the field files)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  ba_history_ = pin->GetOrAddBoolean("problem", "ba_history", false);
  // several ranks: fields of the canonical TSC deposit on a uniform mesh (WriteBAFieldsGlobal), or
  // of refinable clouds (WriteBAFields); not ba_history
  const bool fields_mpi = dust_active && !ba_history_ &&
      ((!pm->multilevel && pmbp->pdust->smr_transfer_mode == DustSMRTransfer::none) ||
       pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud);
  if (ba_fields_ || ba_history_) {
    if ((global_variable::nranks != 1 && !fields_mpi) || !dust_active || pm->three_d ||
        !pin->GetOrAddBoolean("problem", "user_hist", false)) {
      std::cerr << "BA fields/history require 2-D dust, user_hist=true and one MPI rank (several "
                << "ranks: fields only, canonical TSC on a uniform mesh or refinable clouds)"
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
}

//----------------------------------------------------------------------------------------
//! \fn Restart
//! \brief native restart: the output numbers continue from the values the file carries in
//! <problem>; a file without them falls back on the time.  No file is touched here (the run
//! directory is set later); the distribution file is cut back and continued at its next
//! record.

void Restart(ParameterInput *pin, Mesh *pm) {
  if (ba_snapshots_) {
    ba_snapshot_number_ = pin->GetOrAddInteger("problem", "rst_ba_snapshot_number",
        static_cast<int>(std::ceil((pm->time - 1.0e-10)/ba_snapshot_dt_)));
    ba_snapshot_k_ = ScheduleIndex(ba_snapshot_number_, ba_snapshot_dt_, pm->time);
  }
  if (ba_fields_) {
    ba_field_number_ = pin->GetOrAddInteger("problem", "rst_ba_field_number",
        static_cast<int>(std::ceil((pm->time - 1.0e-10)/ba_field_dt_)));
    ba_field_k_ = ScheduleIndex(ba_field_number_, ba_field_dt_, pm->time);
    if (ba_field_k_ != ba_field_number_ && global_variable::my_rank == 0) {
      std::cout << "# BA fields: the interval changed at the restart: frames numbered from "
                << ba_field_number_ << ", next at t = " << ba_field_dt_*ba_field_k_ << std::endl;
    }
    ba_ppdf_resume_ = ba_ppdf_;
    // on a mesh that may have changed, the first frame after the restart writes its own
    // (numbered) geometry file instead of replacing the one of frame zero
    ba_geom_written_ = ba_field_number_ > 0;
  }
}

//----------------------------------------------------------------------------------------
//! \fn HistoryColumns
//! \brief ba_history: mass-weighted dust moments (all species) and the cumulative cloud
//! machinery counters; the merge passes themselves are in cloud_merge_log.txt

void HistoryColumns(HistoryData *pdata, Mesh *pm) {
  if (!ba_history_) return;
  auto &pr = pm->pmb_pack->ppart->prtcl_rdata;
  const int npart = pm->pmb_pack->ppart->nprtcl_thispack;
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

//----------------------------------------------------------------------------------------
//! \fn Write
//! \brief the snapshots and field frames that are due at this history call

void Write(Mesh *pm) {
  if (ba_snapshots_ && pm->time + 1.0e-10 >= ba_snapshot_dt_*ba_snapshot_k_) {
    WriteBASnapshot(pm);
    ++ba_snapshot_k_;
  }
  if (ba_fields_ && pm->time + 1.0e-10 >= ba_field_dt_*ba_field_k_) {
    WriteBAFields(pm);
    ++ba_field_k_;
  }
}

}  // namespace dust_snapshots
