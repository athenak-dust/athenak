//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_selfforce.cpp
//! \brief The spurious self-force of dust
//! clouds through the production chain deposit -> multigrid -> force -> gather.  The gather is the
//! exact transpose of the deposit, so on a uniform mesh the self-force of a cloud vanishes (the
//! force operator is antisymmetric and commutes with the Laplacian); at a coarse-fine interface it
//! does not.  The dust_nsh damping fixture sets up the mesh, gas and clouds; at the final time the
//! particles are replaced by probes:
//!   <problem>/sf_mode = single: one cloud of level sf_level at x = sf_x0 ... sf_x1 (sf_n points),
//!       y = sf_y, z = sf_z: its acceleration in units of G m/h^2 (h = the cloud's half-width);
//!   <problem>/sf_mode = pair: two equal clouds at x and x + sf_pair_dx (x = sf_x0 ... sf_x1):
//!       both accelerations and the net force (a0 + a1)/(G m/h^2), which an exact pair force leaves
//!       zero (Newton's third law).
//! One line per probe ("# SELFFORCE" header, then numbers) and a summary line "# SELFFORCE_MAX".
//! Gas is uniform; the gas source is excluded (<gravity>/source = dust) so only the clouds act.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "dust/dust.hpp"
#include "dust/dust_species.hpp"
#include "globals.hpp"
#include "gravity/self_gravity.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {
struct SelfForceProbe {
  std::string mode = "single";
  int level = 1, n = 41;
  Real y = 0.53, z = 0.47, x0 = 0.4, x1 = 0.6, mass = 1.0e-3, fourpig = 1.0, pair_dx = 0.0;
  Real xref = 0.5;   // the interface the offsets are printed from
} sf;

void SelfForceFinal(ParameterInput *pin, Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto *dust = pack->pdust;
  auto *part = pack->ppart;
  auto *grav = pack->pgrav;
  dust->cloud_split_enable = false;
  const bool pair = (sf.mode == "pair");
  const Real G = sf.fourpig/(4.0*M_PI);
  const Real h = (pm->mesh_size.x1max - pm->mesh_size.x1min)/pm->mesh_indcs.nx1/
                 static_cast<Real>(1 << sf.level);
  const Real scale = G*sf.mass/(h*h);
  if (global_variable::my_rank == 0) {
    std::printf("# SELFFORCE mode=%s level=%d h=%.6e G=%.6e m=%.3e scale=G*m/h^2=%.6e pair_dx=%.6e\n",
                sf.mode.c_str(), sf.level, h, G, sf.mass, scale, sf.pair_dx);
    std::printf(pair ? "# x  (x-xref)/h  a0x a1x (a0+a1)x (a0+a1)y (a0+a1)z [G m/h^2]  dep_mass  residual\n"
                     : "# x  (x-xref)/h  ax ay az [G m/h^2]  dep_mass  residual\n");
  }
  Real amax = 0.0, netmax = 0.0;
  for (int q=0; q<sf.n; ++q) {
    const Real x = (sf.n > 1) ? sf.x0 + (sf.x1 - sf.x0)*q/static_cast<Real>(sf.n - 1) : sf.x0;
    auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_rdata);
    auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), part->prtcl_idata);
    const int keep = (global_variable::my_rank == 0) ? (pair ? 2 : 1) : 0;   // all on rank 0
    if (pr.extent_int(1) < keep) {
      std::cout << "### FATAL ERROR in dust_selfforce: particle arrays too small" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    for (int c=0; c<keep; ++c) {
      for (int r=0; r<part->nrdata; ++r) pr(r,c) = 0.0;
      pr(IPX,c) = pr(IPX1,c) = x + c*sf.pair_dx;
      pr(IPY,c) = pr(IPY1,c) = sf.y;
      pr(IPZ,c) = pr(IPZ1,c) = sf.z;
      pr(IPM,c) = sf.mass;
      pr(IPTS,c) = dust->taus.h_view(0);
      pi(PLEV,c) = sf.level; pi(PSP,c) = 0; SetTag(pi, c, static_cast<std::uint64_t>(c));
      // points: the leaf block (on this rank) that contains the probe
      for (int b=0; b<pack->nmb_thispack; ++b) {
        const auto sz = pack->pmb->mb_size.h_view(b);
        if (pr(IPX,c) >= sz.x1min && pr(IPX,c) < sz.x1max && pr(IPY,c) >= sz.x2min &&
            pr(IPY,c) < sz.x2max && pr(IPZ,c) >= sz.x3min && pr(IPZ,c) < sz.x3max) {
          pi(PGID,c) = pack->gids + b;
        }
      }
    }
    Kokkos::deep_copy(part->prtcl_rdata, pr);
    Kokkos::deep_copy(part->prtcl_idata, pi);
    part->nprtcl_thispack = keep;
    pm->nprtcl_thisrank = keep;
    // the block ids of the probes: 3-D clouds by the leaf traversal (single rank for points)
    if (dust->Clouds3D()) dust->ResetPGIDFromPositions("selfforce probe");   // (points: set above)
    grav->Invalidate();
    grav->Solve(pm->time);
    DvceArray5D<Real> rho;
    dust->DepositGravityDensity(pm->time, rho);
    auto rh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rho);
    const auto ind = pm->mb_indcs;
    Real dm = 0.0;
    for (int b=0; b<pack->nmb_thispack; ++b) {
      const auto s = pack->pmb->mb_size.h_view(b);
      for (int k=ind.ks; k<=ind.ke; ++k) for (int j=ind.js; j<=ind.je; ++j)
        for (int i=ind.is; i<=ind.ie; ++i) dm += s.dx1*s.dx2*s.dx3*rh(b,0,k,j,i);
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &dm, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
    DvceArray2D<Real> pa;
    dust->GatherGravityAcceleration(pm->time, grav->acceleration, pa);
    auto ph = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pa);
    if (global_variable::my_rank == 0) {
      if (pair) {
        const Real n0 = (ph(0,0) + ph(0,1))/scale, n1 = (ph(1,0) + ph(1,1))/scale,
                   n2 = (ph(2,0) + ph(2,1))/scale;
        netmax = std::max(netmax, std::sqrt(n0*n0 + n1*n1 + n2*n2));
        amax = std::max(amax, std::max(std::abs(ph(0,0)), std::abs(ph(0,1)))/scale);
        std::printf("%.8f %+.4f %+.6e %+.6e %+.6e %+.6e %+.6e %.12e %.3e\n", x, (x - sf.xref)/h,
                    ph(0,0)/scale, ph(0,1)/scale, n0, n1, n2, dm, grav->last_solve.relative_residual);
      } else {
        amax = std::max(amax, std::sqrt(ph(0,0)*ph(0,0) + ph(1,0)*ph(1,0) + ph(2,0)*ph(2,0))/scale);
        std::printf("%.8f %+.4f %+.6e %+.6e %+.6e %.12e %.3e\n", x, (x - sf.xref)/h,
                    ph(0,0)/scale, ph(1,0)/scale, ph(2,0)/scale, dm, grav->last_solve.relative_residual);
      }
      std::fflush(stdout);
    }
  }
  if (global_variable::my_rank == 0) {
    std::printf("# SELFFORCE_MAX mode=%s max|a|/scale=%.6e max|net|/scale=%.6e\n", sf.mode.c_str(),
                amax, netmax);
  }
}
}  // namespace

void ProblemGenerator::DustSelfForce(ParameterInput *pin, const bool restart) {
  dust::RequireLegacySpeciesInit(pin, "dust_selfforce");
  sf.mode = pin->GetOrAddString("problem", "sf_mode", "single");
  if (sf.mode != "single" && sf.mode != "pair") {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/sf_mode must be single or pair" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  sf.level = pin->GetOrAddInteger("problem", "sf_level", 1);
  sf.y = pin->GetOrAddReal("problem", "sf_y", 0.53);
  sf.z = pin->GetOrAddReal("problem", "sf_z", 0.47);
  sf.x0 = pin->GetOrAddReal("problem", "sf_x0", 0.4);
  sf.x1 = pin->GetOrAddReal("problem", "sf_x1", 0.6);
  sf.n = pin->GetOrAddInteger("problem", "sf_n", 41);
  sf.mass = pin->GetOrAddReal("problem", "sf_mass", 1.0e-3);
  sf.pair_dx = pin->GetOrAddReal("problem", "sf_pair_dx", 0.0);
  sf.xref = pin->GetOrAddReal("problem", "sf_xref", 0.5);
  sf.fourpig = pin->GetReal("gravity", "four_pi_G");
  if (sf.mode == "pair" && sf.pair_dx == 0.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<problem>/sf_mode = pair needs sf_pair_dx != 0" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto *dust = pmy_mesh_->pmb_pack->pdust;
  const bool split = dust->cloud_split_enable;
  dust->cloud_split_enable = false;
  DustNSH(pin, restart);
  dust->cloud_split_enable = split;
  user_work_in_loop_func = nullptr;
  pgen_final_func = SelfForceFinal;
}
