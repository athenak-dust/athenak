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

namespace {

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
//! \brief Reads shearing box and dust parameters shared by pgen/errors/history.

void ReadNSHParams(ParameterInput *pin, MeshBlockPack *pmbp,
                   Real &omega0, Real &qshear, Real &etavk,
                   std::vector<Real> &eps, std::vector<Real> &taus) {
  dust::DustGasDrag *pdust = pmbp->pdust;
  int ns = pdust->nspecies;
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
  std::string placement = pin->GetOrAddString("problem","particle_placement","lattice");
  bool random_placement = (placement.compare("random") == 0);
  if (!random_placement && placement.compare("lattice") != 0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<problem>/particle_placement must be lattice or random"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  bool check_equilibrium =
      pin->GetOrAddBoolean("problem","check_equilibrium",!random_placement);
  pgen_final_func = check_equilibrium ? DustNSHErrors : nullptr;
  user_hist_func = DustNSHHistory;   // used only when <problem>/user_hist = true
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "NSH test requires <hydro>, <particles>, and <dust> blocks" << std::endl;
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
  int nspec = pmbp->pdust->nspecies;
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
    std::cout << "# NSH equilibrium: u_gx=" << ugx << " u_gphi=" << ugp << std::endl;
    for (int s=0; s<nspec; ++s) {
      std::cout << "#   species " << s+1 << ": v_x=" << vx[s] << " v_phi=" << vp[s]
                << std::endl;
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

  // particle timestep from the largest drift speed (transport is radial drift here)
  Real vmax = 1.0e-30;
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

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustNSHErrors()
//! \brief Measures the deviation of the mean gas and per-species dust drift velocities
//! from the NSH equilibrium at the end of the run and appends the errors to
//! "<basename>-errs.dat". Since the equilibrium is an exact fixed point of the unsplit
//! IMEX update, all errors should be at round-off level.

void DustNSHErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  int nspec = pmbp->pdust->nspecies;
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

  // per-species mean dust velocities
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  int ivazim = three_d ? IPVY : IPVZ;
  std::vector<Real> pvx(nspec), pvp(nspec), pn(nspec);
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

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &gmx, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gmp, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gm,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pvx.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pvp.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, pn.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
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
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  err_ugx  err_ugphi  "
                          "err_vx_s  err_vphi_s ...\n");
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
  return;
}
