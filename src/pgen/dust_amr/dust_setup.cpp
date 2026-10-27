//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_setup.cpp
//! \brief The set-up pieces shared by the dust problem generators (see dust_setup.hpp).

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

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
#include "pgen/pgen.hpp"
#include "pgen/dust_amr/dust_setup.hpp"

namespace dust_setup {

namespace {
bool dust_moments_hist_ = false;   // msum, mvx, mvz, mz2 columns in the user history
// Diagnostic: the x momentum of the dust (m vx) and of the gas (M1 dV) as
// compensated (Neumaier) sums on the host, columns mvx_c and gpx_c, to tell a summation
// error of the ordinary history columns from a conservation error.
bool dust_moments_exact_ = false;

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust problem generator" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

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
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustActive
//! \brief true with both <particles> and <dust>, false with neither (the dust-free gas
//! equilibrium); one without the other stops the run

bool DustActive(Mesh *pm) {
  const bool has_particles = (pm->pmb_pack->ppart != nullptr);
  const bool has_dust = (pm->pmb_pack->pdust != nullptr);
  if (has_particles != has_dust) {
    Fatal("the dust problems need both <particles> and <dust>, or neither (the dust-free gas)");
  }
  return has_particles && has_dust;
}

//----------------------------------------------------------------------------------------
//! \fn ReadSpecies
//! \brief eps_s and taus_s of every species.

void ReadSpecies(ParameterInput *pin, MeshBlockPack *pmbp, const bool allow_z,
                 std::vector<Real> &eps, std::vector<Real> &taus) {
  dust::DustGasDrag *pdust = pmbp->pdust;
  const int ns = (pdust != nullptr) ? pdust->nspecies : 0;
  eps.resize(ns);
  taus.resize(ns);
  // <problem>/eps_N: the legacy abundance keys (the reference-density ratio of every species)
  const bool has_eps = pin->DoesParameterExist("problem", "eps_1");
  for (int s=0; s<ns; ++s) {
    eps[s] = has_eps ? pin->GetReal("problem", "eps_" + std::to_string(s+1))
                     : pdust->dust_to_gas/static_cast<Real>(ns);
    taus[s] = pdust->taus.h_view(s);
  }
  if (pdust != nullptr && dust::Species(pin).new_init) {
    const dust::SpeciesTable &spt = dust::Species(pin);
    if (spt.abundance == "Z" && !allow_z) {
      Fatal("Z_N (column ratios) need the stratified 3-D set-up of dust_si_strat; use "
            "epsilon_N otherwise");
    }
    for (int s=0; s<ns; ++s) eps[s] = static_cast<Real>(spt.value[s]);
  }
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
//! \fn CheckRadialForcing
//! \brief the radial forcing must be applied to the gas by the standard const_accel source
//! term (so that dust particles do not feel it); verify the input file sets it up

void CheckRadialForcing(ParameterInput *pin, const Real ax) {
  bool ok = pin->GetOrAddBoolean("hydro_srcterms","const_accel",false);
  Real val = ok ? pin->GetReal("hydro_srcterms","const_accel_val") : 0.0;
  int dir = ok ? pin->GetInteger("hydro_srcterms","const_accel_dir") : 0;
  if (!ok || dir != 1 || fabs(val - ax) > 1.0e-12*fabs(ax)) {
    std::ostringstream o;
    o << "the NSH set-up requires <hydro_srcterms> const_accel = true, const_accel_dir = 1, "
      << "const_accel_val = 2*omega0*etavk = " << ax;
    Fatal(o.str());
  }
}

//----------------------------------------------------------------------------------------
//! \fn InitUniformGas
//! \brief uniform gas at (ugx, ugp); the azimuthal component is IM2 in 3D and IM3 in the 2D
//! r-z geometry (matching ShearingBoxCC::SourceTermsCC)

void InitUniformGas(Mesh *pm, const Real rho0, const Real ugx, const Real ugp) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  if (pmbp->phydro == nullptr) Fatal("the dust problems require a <hydro> block");
  const bool three_d = pm->three_d;
  auto &indcs = pm->mb_indcs;
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
}

//----------------------------------------------------------------------------------------
//! \fn PlaceParticles
//! \brief Initialize particles at the per-species velocities of d.  Point particles:
//! <problem>/particle_placement = lattice (the quiet start used by the NSH regression test;
//! with lattice_sub in 3-D on an n^3 sub-lattice of every cell) or random (the warm start for
//! nonlinear streaming-instability calculations, e.g. Johansen et al. 2007 Run BA).
//! <problem>/cloud_root_random = true: root parents placed over the whole box
//! (<problem>/root_placement = random | lattice | generic) and split to the leaves as
//! refinable clouds, or (canonical_root) kept as point particles with the standard deposit on
//! a uniform mesh or with the owner_overlap transfer.

void PlaceParticles(ParameterInput *pin, Mesh *pm, const DustInit &d) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const bool three_d = pm->three_d;
  const std::vector<Real> &eps = d.eps, &vx = d.vx, &vp = d.vp;
  const StratDust &sd = d.strat;
  const Real rho0 = d.rho0;
  const int nspec = static_cast<int>(eps.size());
  bool cloud_root_random = pin->GetOrAddBoolean("problem", "cloud_root_random", false);
  // canonical_root: the same root parents with the standard deposit on a uniform mesh,
  // so a uniform control shares the parent positions of a refinable-cloud run
  // also the old owner_overlap transfer on any mesh (point particles, each in the block
  // that holds it), so that it starts from the parents of the refinable-cloud runs
  bool canonical_root = cloud_root_random &&
      ((pmbp->pdust->smr_transfer_mode == DustSMRTransfer::none && !pm->multilevel) ||
       pmbp->pdust->smr_transfer_mode == DustSMRTransfer::owner_overlap);
  // several ranks: refinable clouds; every rank draws the same parents and keeps
  // those in its blocks
  const bool ranks_ok = (global_variable::nranks == 1) || !canonical_root ||
      (canonical_root && three_d);   // 3-D point particles
  // 3-D with refinable clouds or the canonical deposit
  // (several ranks: the clouds only, ranks_ok)
  const bool root3d_ok = pmbp->pdust->Clouds3D() || canonical_root;
  if (cloud_root_random && (!ranks_ok ||
      !(pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud ||
        canonical_root) || (three_d && !root3d_ok))) {
    Fatal("cloud_root_random requires (several ranks: refinable clouds) 2-D dust with "
          "refinable_cloud transfer, or (one rank) the canonical deposit on a uniform mesh or "
          "the owner_overlap transfer; in 3-D the root parents of 3-D refinable clouds");
  }
  std::string placement = pin->GetOrAddString("problem","particle_placement",d.placement);
  const bool random_placement = (placement.compare("random") == 0);
  if (!random_placement && placement.compare("lattice") != 0) {
    Fatal("<problem>/particle_placement must be lattice or random");
  }
  if (sd.on && (!three_d || (cloud_root_random && !root3d_ok))) {
    Fatal("stratified dust: 3-D, <shearing_box>/stratified = true, lattice or random "
          "placement, or the 3-D root parents");
  }
  const dust::SpeciesTable *spt = &dust::Species(pin);
  const bool new_init = spt->new_init;
  // the stratified set-up of the new species keys (per-species H_p, Z_N)
  const bool strat_setup = new_init && sd.on;
  const std::vector<Real> &hps = sd.hp, &finside = sd.finside, &wnorm = sd.wnorm;
  const bool redraw = sd.redraw;
  auto &indcs = pm->mb_indcs;
  int nmb = pmbp->nmb_thispack;
  RootPlacement place;
  place.kwv = d.kwv;
  for (int s=0; s<4; ++s) {
    place.xir[s] = d.xir[s]; place.xii[s] = d.xii[s];
    place.dvr[s] = d.dvr[s]; place.dvi[s] = d.dvi[s];
  }

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
    Fatal("lattice placement requires <particles>/ppc to be an integer multiple of "
          "<dust>/nspecies; random placement requires the total number of particles per rank "
          "to be divisible by <dust>/nspecies (new keys: the sum of ppc_N per cell)");
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
      if (!ok) Fatal("lattice_sub with unequal ppc_N: 3-D and every ppc_N a cube");
      nsub = 2;   // (> 1: the sub-lattice is on; the side is per species)
    } else {
      nsub = static_cast<int>(std::lround(std::cbrt(static_cast<double>(ppc_int/nspec))));
      if (!three_d || nsub*nsub*nsub*nspec != ppc_int) {
        Fatal("lattice_sub: 3-D and ppc = n^3 * nspecies");
      }
    }
  }
  Real ppc = static_cast<Real>(dust::RootPpc(pin));   // new keys: sum of ppc_N
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
    const auto &ms = pm->mesh_size;
    const Real area = (ms.x1max - ms.x1min)*(ms.x2max - ms.x2min);
    const Real vbox = three_d ? area*(ms.x3max - ms.x3min) : area;
    const bool domain = sd.domain;
    std::vector<double> tgt(nspec);
    for (int sp=0; sp<nspec; ++sp) {
      if (!strat_setup) {
        tgt[sp] = eps[sp]*rho0*vbox;
      } else if (domain) {
        tgt[sp] = spt->value[sp]*sd.mg_box;
      } else {
        tgt[sp] = eps[sp]*rho0*std::sqrt(2.0*M_PI)*hps[sp]*(redraw ? 1.0 : finside[sp])*area;
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
          if (redraw) {
            o << " (redraw: all of Z Sigma_g A inside, midplane ratio " << eps[sp]/finside[sp] << ")";
          }
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

  if (cloud_root_random && three_d) {
    // the root parents in 3-D (refinable clouds, or the
    // canonical deposit on a uniform mesh), one rank: random, lattice (ppc = n^3) or generic
    const auto &ms = pm->mesh_size;
    const auto &mi = pm->mesh_indcs;
    const int nc1 = mi.nx1, nc2 = mi.nx2, nc3 = mi.nx3;
    const int nparent = static_cast<int>(ppc)*nc1*nc2*nc3;
    if (ppc != static_cast<Real>(static_cast<int>(ppc)) || nparent <= 0) {
      Fatal("cloud_root_random requires positive integer root ppc");
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
      Fatal("<problem>/root_placement must be random, lattice (ppc a cube; with unequal "
            "ppc_N every ppc_N a cube) or generic");
    }
    place.nc1 = nc1; place.nc2 = nc2; place.nspec = nspec; place.seed = random_seed;
    place.x1min = ms.x1min; place.x2min = ms.x2min; place.x3min = ms.x3min;
    place.lx = ms.x1max - ms.x1min; place.ly = ms.x2max - ms.x2min; place.lz = ms.x3max - ms.x3min;
    place.hx = place.lx/nc1; place.hy = place.ly/nc2; place.hz = place.lz/nc3;
    // stratified dust, sampling gaussian: equal masses, z ~ N(0, Hp) truncated to the box
    // (Li et al. 2018); weight: uniform z, masses times exp(-z^2/2 Hp^2)
    if (sd.on && sd.gauss) {
      if (place.mode != 0) {
        Fatal("stratified_dust_sampling = gaussian: root_placement = random");
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
        place.hp = hps[0];   // the single <problem>/dust_hp
      }
      // truncation: the same total dust mass (the equal masses are unchanged, all
      // samples inside), so the dust density inside is higher by 1/erf(zcut/(sqrt(2) Hp)); the
      // NSH(z) table uses the same profile (nsh_vertical)
      place.zcut = sd.zcut;
      if (place.zcut > 0.0 && global_variable::my_rank == 0) {
        std::cout << "# dust_zcut: Gaussian truncated to |z| < " << place.zcut;
        if (new_init && !redraw) {   // epsilon_N, full_profile, domain: cut
          std::cout << ", the mass of the profile inside only (new species keys)" << std::endl;
        } else if (new_init) {   // redraw: all of the column inside, per species
          std::cout << ", redraw: all of Z Sigma_g A inside, density inside x 1/f_in (per species, "
                    << "the mass audit below)" << std::endl;
        } else {
          std::cout << ", density inside x " << 1.0/std::erf(place.zcut/(std::sqrt(2.0)*place.hp))
                    << std::endl;
        }
      }
    }
    // <problem>/cloud_seed = leaf: every parent becomes a cloud at the level of the
    // leaf that holds it (no lobes; the P1 split then only acts on the clouds whose supports
    // reach finer leaves), so that the clouds start from the same samples as a uniform-mesh run
    // of particles on the finest grid; root (default): root-level parents split to the leaves
    const std::string cseed = pin->GetOrAddString("problem", "cloud_seed", "root");
    if ((cseed != "root" && cseed != "leaf") || (cseed == "leaf" && canonical_root)) {
      Fatal("<problem>/cloud_seed must be root or leaf (leaf: refinable clouds)");
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
    pm->nprtcl_thisrank = nkeep;
    pm->nprtcl_eachrank[global_variable::my_rank] = nkeep;
    pm->nprtcl_total = nparent;
    auto pr0 = ppar->prtcl_rdata;
    auto pi0 = ppar->prtcl_idata;
    auto gids0 = pmbp->gids;
    const int nrdata = ppar->nrdata;
    const Real mass_unit = place.hx*place.hy*place.hz;
    // the y-modulated NSH dust (dust_ymode_amp), the mass of each parent
    const Real yamp = d.yamp, ky = d.ky;
    // stratified dust on the root parents (masses times exp(-z^2/2 Hp^2), dust_vz0 added to
    // vz), before the split
    const bool strat = sd.on;
    const Real shp = (strat && !new_init) ? hps[0] : 1.0;
    const Real svz0 = strat ? sd.vz0 : 0.0;
    // a uniform dust vz on the root parents (dust_damping_amr); with vz_mirror its sign
    // follows z (a state mirrored about z = 0, the image of a box with reflecting x3 faces)
    const Real tvz0 = d.vz_add;
    const bool tvzm = d.vz_mirror;
    const bool gsamp = place.gauss;
    const Real gfac = std::sqrt(2.0*M_PI)*place.hp/place.lz;
    // per species: the Gaussian factor of the equal masses and the H_p of the weights (legacy:
    // the single values; per species, the Gaussian masses times the fraction of
    // the profile inside the sampling range, so a truncated profile carries less mass)
    // redraw: the gaussian masses without f_in (all of the column inside), the
    // weights times 1/f_in
    DualArray1D<Real> sgf("nsh_species_gfac", nspec), shpv("nsh_species_hpw", nspec);
    DualArray1D<Real> swn("nsh_species_wnorm", nspec);
    for (int sp=0; sp<nspec; ++sp) {
      sgf.h_view(sp) = new_init && strat ? std::sqrt(2.0*M_PI)*hps[sp]/place.lz*
                                           (redraw ? 1.0 : finside[sp]) : gfac;
      shpv.h_view(sp) = new_init && strat ? hps[sp] : shp;
      swn.h_view(sp) = (new_init && strat) ? wnorm[sp] : 1.0;
    }
    sgf.template modify<HostMemSpace>(); sgf.template sync<DevExeSpace>();
    shpv.template modify<HostMemSpace>(); shpv.template sync<DevExeSpace>();
    swn.template modify<HostMemSpace>(); swn.template sync<DevExeSpace>();
    auto sgfd = sgf.d_view;
    auto shpd = shpv.d_view;
    auto swnd = swn.d_view;
    const bool wscale = redraw && !gsamp;
    auto mblev0 = pmbp->pmb->mb_lev;
    const int rootlev0 = pm->root_level;
    const bool nshz = sd.ztab_on;
    const NSHTableZ ntab = sd.ztab;
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
        if (wscale) pr0(IPM,p) *= swnd(sp);
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
        // per species: the largest cloud of all species would give a light species the
        // quantum of the heaviest one
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
    // ppc particles per ROOT cell, placed over the whole box.  Refinement changes the
    // cloud shape through the exact P1 split, never the prescribed parent mass.
    auto transfer = pmbp->pdust->smr_transfer;
    auto lt = pmbp->pdust->leaf_traversal;
    if (canonical_root) {   // only the root geometry is used below
      const auto &ms = pm->mesh_size;
      const auto &mi = pm->mesh_indcs;
      transfer.x1min = ms.x1min; transfer.x1max = ms.x1max;
      transfer.x2min = ms.x2min; transfer.x2max = ms.x2max;
      transfer.ncoarse1 = mi.nx1; transfer.ncoarse2 = mi.nx2;
      transfer.root_dx1 = (ms.x1max - ms.x1min)/mi.nx1;
      transfer.root_dx2 = (ms.x2max - ms.x2min)/mi.nx2;
    }
    int nroot = transfer.ncoarse1*transfer.ncoarse2;
    int nparent = static_cast<int>(ppc)*nroot;
    if (ppc != static_cast<Real>(static_cast<int>(ppc)) || nparent <= 0) {
      Fatal("cloud_root_random requires positive integer root ppc");
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
        Fatal("<problem>/root_placement must be random, lattice (ppc a square; with "
              "unequal ppc_N every ppc_N a square) or generic");
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
    pm->nprtcl_thisrank = nkeep;
    pm->nprtcl_eachrank[global_variable::my_rank] = nkeep;
    pm->nprtcl_total = nparent;
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
    std::cout << "# 2-D random root " << (canonical_root ? "particles (canonical): "
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
      // the dusty-wave eigenmode displacement and velocity of the lattice
      if (place.kwv != 0.0) {
        const Real x0 = pr(IPX,p);
        const Real c = Kokkos::cos(place.kwv*x0), sn = Kokkos::sin(place.kwv*x0);
        pr(IPX,p) = x0 + place.xir[s]*c - place.xii[s]*sn;
        pr(IPVX,p) += place.dvr[s]*c - place.dvi[s]*sn;
      }
    });
  }

  // stratified point particles (lattice or random placement): dust masses times
  // exp(-z^2/2 Hp^2), dust_vz0 added to vz (the 3-D root parents got it in their set-up)
  if (sd.on && !cloud_root_random) {
    // the new keys: per-species H_p; legacy: the single dust_hp
    const Real hp = hps[0];
    DualArray1D<Real> hpv("nsh_strat_hp", nspec), wnv("nsh_strat_wnorm", nspec);
    for (int sp=0; sp<nspec; ++sp) {
      hpv.h_view(sp) = new_init ? hps[sp] : hp;
      wnv.h_view(sp) = new_init ? wnorm[sp] : 1.0;   // redraw: 1/f_in
    }
    hpv.template modify<HostMemSpace>(); hpv.template sync<DevExeSpace>();
    wnv.template modify<HostMemSpace>(); wnv.template sync<DevExeSpace>();
    auto hpd = hpv.d_view;
    auto wnd = wnv.d_view;
    const bool wscale = redraw;
    const Real vz0 = sd.vz0;
    auto &prs = ppar->prtcl_rdata;
    auto &pis = ppar->prtcl_idata;
    const bool nshz = sd.ztab_on;
    const NSHTableZ ntab = sd.ztab;
    par_for("strat_dust", DevExeSpace(), 0, ppar->nprtcl_thispack-1, KOKKOS_LAMBDA(const int p) {
      const Real z = prs(IPZ,p);
      const Real h = hpd(pis(PSP,p));
      prs(IPM,p) *= exp(-0.5*z*z/(h*h));
      if (wscale) prs(IPM,p) *= wnd(pis(PSP,p));
      prs(IPVZ,p) += vz0;
      if (nshz) {   // the NSH drift at the height of the particle
        const int s = pis(PSP,p);
        prs(IPVX,p) = ntab.At(z, 2 + 2*s);
        prs(IPVY,p) = ntab.At(z, 3 + 2*s);
      }
    });
    if (global_variable::my_rank == 0) {
      const Real hgas = pmbp->phydro->peos->eos_data.iso_cs/
                        pin->GetReal("shearing_box", "omega0");
      std::cout << "# stratified dust: H = " << hgas << ", Hp = " << hp << ", vz0 = " << vz0
                << std::endl;
    }
  }
  if (!cloud_root_random) species_audit(true);   // lattice / random placement
}

//----------------------------------------------------------------------------------------
//! \fn SetParticleDt
//! \brief particle timestep from the largest drift speed (transport is radial drift here)

void SetParticleDt(Mesh *pm, const std::vector<Real> &vx, const std::vector<Real> &vp,
                   const Real vextra) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  Real vmax = 1.0e-30;
  vmax = std::max(vmax, vextra);
  for (std::size_t s=0; s<vx.size(); ++s) {
    vmax = std::max(vmax, fabs(vx[s]));
    vmax = std::max(vmax, fabs(vp[s]));
  }
  Real dtnew = std::numeric_limits<float>::max();
  auto &msize = pmbp->pmb->mb_size;
  dtnew = std::min(dtnew, msize.h_view(0).dx1/vmax);
  dtnew = std::min(dtnew, msize.h_view(0).dx2/vmax);
  if (pm->three_d) {dtnew = std::min(dtnew, msize.h_view(0).dx3/vmax);}
  pmbp->ppart->dtnew = dtnew;
}

//----------------------------------------------------------------------------------------
//! \fn EnrollRefinement
//! \brief dynamic mesh: the scheduled refinement request of <dust>/amr_sched_* answers
//! <amr_criterionN>/method = user; with <dust>/amr_density_trigger the dust-density request

void EnrollRefinement(ParameterInput *pin, Mesh *pm, ProblemGenerator *pgen) {
  if (pm->adaptive && pin->DoesBlockExist("dust")) {
    dust::SetupScheduledRefinement(pin);
    pgen->user_ref_func = dust::ScheduledRefinement;
    if (pin->GetOrAddBoolean("dust", "amr_density_trigger", false)) {
      dust::SetupDensityTrigger(pin);
      pgen->user_ref_func = dust::DensityRefinement;
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn LogRestartParticles
//! \brief native restart: gas, particles and the dust counters come from the file.  The
//! total over the ranks may be zero (all dust left an outflow domain), which is a valid
//! gas-only continuation; the typed restart reader already tells a missing or corrupt
//! particle extension from valid zero-count sections.

void LogRestartParticles(Mesh *pm) {
  if (pm->pmb_pack->pdust == nullptr || pm->pmb_pack->ppart == nullptr) return;
  long long np = pm->pmb_pack->ppart->nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &np, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  if (np == 0 && global_variable::my_rank == 0)
    std::cout << "# dust restart: valid empty particle population restored" << std::endl;
}

//----------------------------------------------------------------------------------------
//! \fn RejectKeys
//! \brief every entry {key, off value, problem generator that takes it}: a <problem> key set
//! to anything but its off value stops the run with the name of the right problem generator

void RejectKeys(ParameterInput *pin, const std::string &pgen,
                const std::vector<std::vector<std::string>> &keys) {
  for (const auto &e : keys) {
    if (!pin->DoesParameterExist("problem", e[0])) continue;
    const std::string v = pin->GetString("problem", e[0]);
    char *end = nullptr;
    const double off = std::strtod(e[1].c_str(), &end);
    // a boolean (restart files store them as 0 / 1)
    auto lower = [](std::string x) {
      for (auto &ch : x) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
      return x;
    };
    auto as_bool = [&](const std::string &x, int &b) {
      const std::string l = lower(x);
      b = (l == "true" || l == "1") ? 1 : ((l == "false" || l == "0") ? 0 : -1);
      return b >= 0;
    };
    int boff = -1, bval = -1;
    bool is_off;
    if (as_bool(e[1], boff) && (e[1] == "true" || e[1] == "false")) {
      is_off = as_bool(v, bval) && (bval == boff);
    } else if (end != e[1].c_str() && *end == '\0') {   // a number: compare the values
      char *vend = nullptr;
      const double val = std::strtod(v.c_str(), &vend);
      is_off = (vend != v.c_str()) && (val == off);
    } else {
      is_off = (v == e[1]);
    }
    if (!is_off) {
      Fatal("<problem>/" + e[0] + " = " + v + " is not a key of pgen_name = " + pgen +
            "; it belongs to pgen_name = " + e[2]);
    }
  }
}

//----------------------------------------------------------------------------------------
//! \fn SetupHistory, SpeciesHistory
//! \brief User history output: per-species sums of the radial and azimuthal dust drift
//! velocities and particle counts (summed over ranks; divide sums by counts for means), and
//! the optional mass moments (<problem>/dust_moments_hist) and compensated x momenta
//! (<problem>/dust_moments_exact).

void SetupHistory(ParameterInput *pin) {
  dust_moments_hist_ = pin->GetOrAddBoolean("problem", "dust_moments_hist", false);
  dust_moments_exact_ = pin->GetOrAddBoolean("problem", "dust_moments_exact", false);
}

bool SpeciesHistory(HistoryData *pdata, Mesh *pm) {
  pdata->nhist = 0;
  if (pm->pmb_pack->ppart == nullptr || pm->pmb_pack->pdust == nullptr) return false;
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
  if (dust_moments_hist_) {   // all species: sum m, m vx, m vz, m z^2
    int h = pdata->nhist;
    if (h + 4 > NHISTORY_VARIABLES) Fatal("dust_moments_hist: too many history columns");
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
  return true;
}

//----------------------------------------------------------------------------------------
//! \fn DustTest3DErrors
//! \brief The errors of the 3-D drag tests, computed in the code (leaf cells, volume
//! weighted).  damping: rms(rho_g - rho0),
//! rms(u_x - u(t)), rms(rho_d/(eps rho0) - 1), rms|u_perp|; wave: L1 errors over the amplitudes
//! (gas and dust density as cell averages, u_x at the cell centre) and rms|u_perp|/|du|.  Both:
//! the rms deviation of rho_g and rho_d from their x-column means (the part that is not one-
//! dimensional; over the amplitudes for the wave, rho_d over eps rho0 for damping) and the dust
//! mass on the leaves over the particle mass, minus one.  Dust entries are -1 without clouds.

namespace {
void DustTest3DErrors(Mesh *pm, const DustTest3D &c, Real out[7]) {
  auto *pmbp = pm->pmb_pack;
  auto *pd = pmbp->pdust;
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
}  // namespace

void DustTest3DHistory(HistoryData *pdata, Mesh *pm, const DustTest3D &c) {
  if (!c.on) return;
  const int h = pdata->nhist;
  const char *labels[7] = {"e_rhog", "e_ux", "e_rhod", "e_uperp", "y_rhog", "y_rhod", "dmass"};
  Real v[7];
  DustTest3DErrors(pm, c, v);
  // the history adds the ranks: the values once, from rank 0
  const Real f = (global_variable::my_rank == 0) ? 1.0 : 0.0;
  for (int k=0; k<7; ++k) {pdata->label[h+k] = labels[k]; pdata->hdata[h+k] = f*v[k];}
  pdata->nhist = h + 7;
}

}  // namespace dust_setup
