#ifndef PGEN_DUST_AMR_DUST_SETUP_HPP_
#define PGEN_DUST_AMR_DUST_SETUP_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_setup.hpp
//! \brief The pieces that the dust problem generators share (dust_nsh, dust_si_unstrat,
//! dust_si_strat, dust_damping_amr, dusty_wave_amr, and through dust_damping_amr also
//! dust_freefall and dust_jeans):
//!   - the dust species (abundance and stopping time of every species)
//!   - the multi-species NSH drift equilibrium (Nakagawa et al. 1986)
//!   - the particle set-up: point particles on a lattice or at random positions, or the
//!     root parents of refinable clouds (<problem>/cloud_root_random), split to the leaves
//!   - the user history columns common to these problems, and the 3-D drag-test errors
//!   - the refinement requests of a dynamic mesh (scheduled, or by the dust density)

#include <cstdint>
#include <string>
#include <vector>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"

namespace dust_setup {

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
//! \struct RootPlacement
//! \brief the position of root parent q (cloud_root_random): mode 0 the
//! independent random draw (the original hash), 1 a square lattice (ppc = n^2, cell-centred
//! offsets), 2 a generic arrangement for problems along x1: x1 offsets (k + 1/2)/ppc in the root
//! cell, x2 offsets frac((k + 1/2) phi) (phi the golden ratio), the same in every root cell, so
//! that for any integer ppc each x1 sub-lattice is a full shift family of the mesh (a uniform
//! TSC / hat deposit on the root level).  With kwv != 0 the x1 position is displaced by the
//! eigenmode displacement Re[xi_s exp(i k x0)] (dusty_wave_amr).

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
  // root cells nc1 x nc2 x nc3; mode 1 a cubic lattice (ppc = n^3),
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

//----------------------------------------------------------------------------------------
//! \struct NSHTableZ
//! \brief the multi-species NSH drift with the local dust-to-gas ratio
//! eps_s(z) = eps_s exp(-z^2 (1/Hp^2 - 1/H^2)/2) (dust_si_strat, <problem>/nsh_vertical), solved
//! on the host on a fine table in z (Athena's par_strat3d.c with ipert = 0 solves MultiNSH at the
//! cell faces); columns u_gx, u_gphi, then v_x, v_phi of each species; linear interpolation on
//! the device

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

//----------------------------------------------------------------------------------------
//! \struct DustTest3D
//! \brief the parameters of the exact solutions behind the 3-D error columns of the
//! history (DustTest3DErrors): the drag tests of dust_damping_amr and dusty_wave_amr, and
//! the NSH equilibrium of 3-D refinable clouds in dust_nsh

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

//----------------------------------------------------------------------------------------
//! \struct StratDust
//! \brief the vertical profile of the dust that dust_si_strat hands to PlaceParticles

struct StratDust {
  bool on = false;
  bool new_keys = false;        // per-species H_p and abundances from the species table
  std::vector<Real> hp;         // H_p per species (legacy keys: <problem>/dust_hp for all)
  std::vector<Real> finside;    // new keys: the fraction of each Gaussian in the sampling range
  std::vector<Real> wnorm;      // new keys: the factor of the weights (redraw: 1/finside)
  bool redraw = false;          // Z with column_normalization = redraw
  bool domain = false;          // Z with column_normalization = domain
  Real mg_box = 0.0;            // domain: the initial gas mass of the box
  bool gauss = false;           // <problem>/stratified_dust_sampling = gaussian
  Real zcut = 0.0;              // <problem>/dust_zcut
  Real vz0 = 0.0;               // <problem>/dust_vz0
  bool ztab_on = false;         // <problem>/nsh_vertical
  NSHTableZ ztab;
};

//----------------------------------------------------------------------------------------
//! \struct DustInit
//! \brief what a problem generator fixes before PlaceParticles: the dust of every species
//! at t = 0 (uniform, or the vertical profile of StratDust) and its velocities

struct DustInit {
  Real rho0 = 1.0;
  std::vector<Real> eps;        // dust-to-gas ratio per species (midplane ratio if stratified)
  std::vector<Real> vx, vp;     // x1 and azimuthal velocity per species (azimuthal: x2 in 3-D,
                                // x3 in the 2-D r-z shearing box)
  // the default of <problem>/particle_placement (point particles): lattice or random
  std::string placement = "lattice";
  // the dusty-wave eigenmode: displacement xi and velocity dv of every species at wavenumber kwv
  Real kwv = 0.0;
  Real xir[4] = {0.0, 0.0, 0.0, 0.0}, xii[4] = {0.0, 0.0, 0.0, 0.0};
  Real dvr[4] = {0.0, 0.0, 0.0, 0.0}, dvi[4] = {0.0, 0.0, 0.0, 0.0};
  // dust_nsh: the masses of the 3-D root parents times 1 + yamp cos(ky y)
  Real yamp = 0.0, ky = 0.0;
  // dust_damping_amr: a uniform vz on the 3-D root parents; with vz_mirror its sign follows z
  Real vz_add = 0.0;
  bool vz_mirror = false;
  StratDust strat;
};

// the dust species: eps_s from <problem>/eps_N (or <dust>/dust_to_gas over the species), or
// the new species keys (epsilon_N; Z_N only with allow_z, the caller converts them), and taus_s
bool DustActive(Mesh *pm);
void ReadSpecies(ParameterInput *pin, MeshBlockPack *pmbp, const bool allow_z,
                 std::vector<Real> &eps, std::vector<Real> &taus);
// the multi-species NSH drift equilibrium, unknowns [u_x, u_phi, v_x,s, v_phi,s]
void SolveNSH(const Real omega0, const Real qshear, const Real ax,
              const std::vector<Real> &eps, const std::vector<Real> &taus,
              Real &ugx, Real &ugp, std::vector<Real> &vx, std::vector<Real> &vp);
// the radial forcing 2 Omega etavk on the gas through <hydro_srcterms>/const_accel
void CheckRadialForcing(ParameterInput *pin, const Real ax);
// uniform gas rho0 with the velocity (ugx, ugp); azimuthal is IM2 in 3-D, IM3 in 2-D
void InitUniformGas(Mesh *pm, const Real rho0, const Real ugx, const Real ugp);
// the particles of every species (lattice, random, or root parents of refinable clouds)
void PlaceParticles(ParameterInput *pin, Mesh *pm, const DustInit &d);
// the particle time step from the largest drift speed
void SetParticleDt(Mesh *pm, const std::vector<Real> &vx, const std::vector<Real> &vp,
                   const Real vextra);
// the refinement request of a dynamic mesh: <dust>/amr_sched_* or the dust density
void EnrollRefinement(ParameterInput *pin, Mesh *pm, ProblemGenerator *pgen);
// restart: a note when no particle is left (a valid state with outflow boundaries)
void LogRestartParticles(Mesh *pm);
// the problem generator stops when <problem> sets one of these keys to a value other than
// its off value (the key belongs to another problem generator)
void RejectKeys(ParameterInput *pin, const std::string &pgen,
                const std::vector<std::vector<std::string>> &keys);
// the history columns of every dust problem: per species the sums of v_x, v_phi and the
// count; with <problem>/dust_moments_hist the mass moments, with dust_moments_exact the
// compensated x momenta.  Returns false (no columns) without dust.
void SetupHistory(ParameterInput *pin);
bool SpeciesHistory(HistoryData *pdata, Mesh *pm);
// the 3-D error columns e_rhog, e_ux, e_rhod, e_uperp, y_rhog, y_rhod, dmass
void DustTest3DHistory(HistoryData *pdata, Mesh *pm, const DustTest3D &c);

}  // namespace dust_setup

#endif  // PGEN_DUST_AMR_DUST_SETUP_HPP_
