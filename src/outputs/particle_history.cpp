//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_history.cpp
//! \brief Particle history output (file_type = phst), the AthenaK counterpart of the
//! Athena-C dust code's dump_particle_history.c.  One line per output time in
//! <basename>.phst with
//!   global scalars: npart, mass, x/y/z momentum, x/y/z kinetic energy (velocities as
//!     the particles carry them: shear-relative in a shearing box), escaped mass (dust
//!     module), and the maximum of the particle-mesh dust density (dust module: the
//!     module's own deposit with its ghost exchange; the Roche-density diagnostic);
//!   per species s (dust module; one "species" otherwise): n_s, <x>, <y>, <z>, <vx>,
//!     <vy>, <vz>, and the number-weighted standard deviations sig_x, sig_y, sig_z
//!     (= the dust scale height H_d), sig_vx, sig_vy, sig_vz -- Athena-C's "var"
//!     columns, sqrt(sum (q - <q>)^2/(n - 1)).
//! Means and deviations are two exact passes with MPI sums, not a one-pass variance.

#include <cstdio>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// constructor

ParticleHistoryOutput::ParticleHistoryOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  if (pm->pmb_pack->ppart == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "phst output requested in block '" << op.block_name
              << "' but no particles are constructed" << std::endl;
    exit(EXIT_FAILURE);
  }
  nspecies_ = (pm->pmb_pack->pdust != nullptr) ? pm->pmb_pack->pdust->nspecies : 1;
  header_written_ = false;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleHistoryOutput::LoadOutputData()
//! \brief global sums and the two-pass per-species statistics (rank-local sums here,
//! reduced in WriteOutputFile); the maximum density is reduced here as a max

void ParticleHistoryOutput::LoadOutputData(Mesh *pm) {
  particles::Particles *ppar = pm->pmb_pack->ppart;
  dust::DustGasDrag *pdust = pm->pmb_pack->pdust;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  const bool has_mass = (ppar->particle_type == ParticleType::dust);
  const int nsp = nspecies_;

  // ---- global scalars: n, mass, momenta, kinetic energies ----------------------
  Real gs[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  Kokkos::parallel_reduce("phst_global",
      Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, Real &n_, Real &m_, Real &px, Real &py, Real &pz,
                Real &kx, Real &ky, Real &kz) {
    Real m = has_mass ? pr(IPM,p) : 1.0;
    Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
    n_ += 1.0; m_ += m;
    px += m*vx; py += m*vy; pz += m*vz;
    kx += 0.5*m*vx*vx; ky += 0.5*m*vy*vy; kz += 0.5*m*vz*vz;
  }, Kokkos::Sum<Real>(gs[0]), Kokkos::Sum<Real>(gs[1]), Kokkos::Sum<Real>(gs[2]),
     Kokkos::Sum<Real>(gs[3]), Kokkos::Sum<Real>(gs[4]), Kokkos::Sum<Real>(gs[5]),
     Kokkos::Sum<Real>(gs[6]), Kokkos::Sum<Real>(gs[7]));
  Real escaped = (pdust != nullptr) ? pdust->escaped_mass : 0.0;
  Real dmax = 0.0;
  if (pdust != nullptr) {
    pdust->AssembleDustDensityNow();
    auto &rhod = pdust->rho_dust;
    auto &indcs = pm->mb_indcs;
    int is = indcs.is, ie = indcs.ie, js = indcs.js, je = indcs.je;
    int ks = indcs.ks, ke = indcs.ke;
    int ni = ie-is+1, nj = je-js+1, nk = ke-ks+1;
    int nmkji = (pm->pmb_pack->nmb_thispack)*nk*nj*ni;
    Kokkos::parallel_reduce("phst_dmax", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
    KOKKOS_LAMBDA(int idx, Real &lmax) {
      int i = is + (idx % ni);
      int j = js + ((idx/ni) % nj);
      int k = ks + ((idx/(ni*nj)) % nk);
      int m = idx/(ni*nj*nk);
      lmax = fmax(lmax, rhod(m,0,k,j,i));
    }, Kokkos::Max<Real>(dmax));
  }
  global_.assign(gs, gs + 8);
  global_.push_back(escaped);
  global_.push_back(dmax);

  // ---- per species, pass 1: counts and sums of positions and velocities ---------------
  sums1_.assign(7*nsp, 0.0);
  for (int s=0; s<nsp; ++s) {
    Real a[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Kokkos::parallel_reduce("phst_mean",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &n_, Real &x, Real &y, Real &z, Real &vx, Real &vy,
                  Real &vz) {
      int sp = has_mass ? pi(PSP,p) : 0;
      if (sp != s) return;
      n_ += 1.0;
      x += pr(IPX,p); y += pr(IPY,p); z += pr(IPZ,p);
      vx += pr(IPVX,p); vy += pr(IPVY,p); vz += pr(IPVZ,p);
    }, Kokkos::Sum<Real>(a[0]), Kokkos::Sum<Real>(a[1]), Kokkos::Sum<Real>(a[2]),
       Kokkos::Sum<Real>(a[3]), Kokkos::Sum<Real>(a[4]), Kokkos::Sum<Real>(a[5]),
       Kokkos::Sum<Real>(a[6]));
    for (int q=0; q<7; ++q) sums1_[7*s+q] = a[q];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, sums1_.data(), 7*nsp, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  // means (all ranks hold them)
  means_.assign(6*nsp, 0.0);
  for (int s=0; s<nsp; ++s) {
    Real n = std::max(sums1_[7*s], 1.0);
    for (int q=0; q<6; ++q) means_[6*s+q] = sums1_[7*s+1+q]/n;
  }
  // ---- pass 2: squared deviations ---------------------------------------------
  sums2_.assign(6*nsp, 0.0);
  DvceArray1D<Real> d_means("phst_means", 6*nsp);
  {
    auto h = Kokkos::create_mirror_view(d_means);
    for (int q=0; q<6*nsp; ++q) h(q) = means_[q];
    Kokkos::deep_copy(d_means, h);
  }
  for (int s=0; s<nsp; ++s) {
    Real b[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Kokkos::parallel_reduce("phst_disp",
        Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &x, Real &y, Real &z, Real &vx, Real &vy, Real &vz) {
      int sp = has_mass ? pi(PSP,p) : 0;
      if (sp != s) return;
      x += SQR(pr(IPX,p) - d_means(6*s));
      y += SQR(pr(IPY,p) - d_means(6*s+1));
      z += SQR(pr(IPZ,p) - d_means(6*s+2));
      vx += SQR(pr(IPVX,p) - d_means(6*s+3));
      vy += SQR(pr(IPVY,p) - d_means(6*s+4));
      vz += SQR(pr(IPVZ,p) - d_means(6*s+5));
    }, Kokkos::Sum<Real>(b[0]), Kokkos::Sum<Real>(b[1]), Kokkos::Sum<Real>(b[2]),
       Kokkos::Sum<Real>(b[3]), Kokkos::Sum<Real>(b[4]), Kokkos::Sum<Real>(b[5]));
    for (int q=0; q<6; ++q) sums2_[6*s+q] = b[q];
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParticleHistoryOutput::WriteOutputFile()

void ParticleHistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  const int nsp = nspecies_;
  // reduce the global sums (max for the density) and the deviation sums to the root
#if MPI_PARALLEL_ENABLED
  Real gmax = global_[9];
  if (global_variable::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, global_.data(), 9, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &gmax, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, sums2_.data(), 6*nsp, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(global_.data(), global_.data(), 9, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&gmax, &gmax, 1, MPI_ATHENA_REAL, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(sums2_.data(), sums2_.data(), 6*nsp, MPI_ATHENA_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
  }
  global_[9] = gmax;
#endif
  if (global_variable::my_rank != 0) {
    AdvanceTime(pm, pin);
    return;
  }
  std::string fname = out_params.file_basename + ".phst";
  FILE *pfile = std::fopen(fname.c_str(), "a");
  if (pfile == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Output file '" << fname << "' could not be opened"
              << std::endl;
    exit(EXIT_FAILURE);
  }
  if (!header_written_) {
    int c = 1;
    std::fprintf(pfile, "# AthenaK particle history data\n");
    std::fprintf(pfile, "#  [%d]=time  [%d]=dt", c, c+1); c += 2;
    const char *gl[10] = {"npart", "mass", "px", "py", "pz", "kex", "key", "kez",
                          "m_escaped", "dpm_max"};
    for (int q=0; q<10; ++q) {std::fprintf(pfile, "  [%d]=%s", c++, gl[q]);}
    const char *sl[13] = {"n", "x_avg", "y_avg", "z_avg", "vx_avg", "vy_avg", "vz_avg",
                          "sig_x", "sig_y", "sig_z", "sig_vx", "sig_vy", "sig_vz"};
    for (int s=0; s<nsp; ++s) {
      for (int q=0; q<13; ++q) {std::fprintf(pfile, "  [%d]=%s_%d", c++, sl[q], s+1);}
    }
    std::fprintf(pfile, "\n");
    header_written_ = true;
  }
  std::fprintf(pfile, out_params.data_format.c_str(), pm->time);
  std::fprintf(pfile, out_params.data_format.c_str(), pm->dt);
  const char *fmt = out_params.data_format.c_str();
  for (int q=0; q<10; ++q) {std::fprintf(pfile, fmt, global_[q]);}
  for (int s=0; s<nsp; ++s) {
    Real n = sums1_[7*s];
    std::fprintf(pfile, out_params.data_format.c_str(), n);
    for (int q=0; q<6; ++q) {std::fprintf(pfile, fmt, means_[6*s+q]);}
    Real denom = std::max(n - 1.0, 1.0);
    for (int q=0; q<6; ++q) {
      std::fprintf(pfile, fmt, std::sqrt(sums2_[6*s+q]/denom));
    }
  }
  std::fprintf(pfile, "\n");
  std::fclose(pfile);
  AdvanceTime(pm, pin);
}

void ParticleHistoryOutput::AdvanceTime(Mesh *pm, ParameterInput *pin) {
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
