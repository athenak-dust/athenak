//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_damping.cpp
//! \brief Problem generator for the N-species gas-dust drag damping test (Krapp et al.
//! 2024, Sec. 3.1; Benitez-Llambay et al. 2019, Sec. 3.1). A uniform gas at rest is
//! coupled to uniformly-distributed dust particles with per-species stopping times and
//! initial velocities. The spatially-uniform system obeys the linear ODE system
//!    du/dt   = sum_s eps_s*(v_s - u)/t_s,s
//!    dv_s/dt = -(v_s - u)/t_s,s
//! whose reference solution is integrated on the host with a fine-step RK4 at the end
//! of the run. Errors in the mean gas and per-species dust velocities, and the change
//! in total (gas+dust) momentum -- which must be conserved to round-off -- are written
//! to "dust_damping-errs.dat".

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
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

#include <Kokkos_Random.hpp>

// function to compute errors in solution at end of run
void DustDampingErrors(ParameterInput *pin, Mesh *pm);
// optional user history: per-species dust velocity sums and counts
void DustDampingHistory(HistoryData *pdata, Mesh *pm);

namespace {
// initial total x-momentum (per rank), stored for the conservation check
Real p0x_thisrank = 0.0;

// per-species initial dust velocities: <problem>/vd_1..N if present, else the default
// staircase v0*(s+1)/nspecies
std::vector<Real> InitialDustVelocities(ParameterInput *pin, int nspec) {
  Real v0 = pin->GetOrAddReal("problem","v0",1.0);
  std::vector<Real> vd(nspec);
  for (int s=0; s<nspec; ++s) {
    std::string key = "vd_" + std::to_string(s+1);
    if (pin->DoesParameterExist("problem", key)) {
      vd[s] = pin->GetReal("problem", key);
    } else {
      vd[s] = v0*static_cast<Real>(s+1)/static_cast<Real>(nspec);
    }
  }
  return vd;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustDamping()
//! \brief Problem Generator for gas-dust drag damping test

void ProblemGenerator::DustDamping(ParameterInput *pin, const bool restart) {
  pgen_final_func = DustDampingErrors;
  user_hist_func = DustDampingHistory;   // used only when <problem>/user_hist = true
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dust damping test requires <hydro>, <particles>, and <dust> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // initialize uniform gas with x-velocity <problem>/vgas0 (default at rest)
  Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  Real vgas0 = pin->GetOrAddReal("problem","vgas0",0.0);
  auto &indcs = pmy_mesh_->mb_indcs;
  int &ng = indcs.ng;
  int n1 = indcs.nx1 + 2*ng;
  int n2 = (indcs.nx2 > 1)? (indcs.nx2 + 2*ng) : 1;
  int n3 = (indcs.nx3 > 1)? (indcs.nx3 + 2*ng) : 1;
  int nmb = pmbp->nmb_thispack;
  auto &u0 = pmbp->phydro->u0;
  auto &eos = pmbp->phydro->peos->eos_data;
  bool is_ideal = eos.is_ideal;
  Real pgas = is_ideal ? pin->GetOrAddReal("problem","pgas",1.0) : 0.0;
  Real gm1 = eos.gamma - 1.0;
  par_for("dustdamp_gas", DevExeSpace(),0,(nmb-1),0,(n3-1),0,(n2-1),0,(n1-1),
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0*vgas0;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
    if (is_ideal) {u0(m,IEN,k,j,i) = pgas/gm1 + 0.5*rho0*SQR(vgas0);}
  });

  // initialize particles: uniformly distributed at random positions
  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &mbsize = pmbp->pmb->mb_size;
  auto gids = pmbp->gids;
  bool three_d = pmy_mesh_->three_d;
  int npart_permb = npart/nmb;

  // Particle placement: "lattice" puts <particles>/ppc particles at every cell center,
  // with ppc an integer multiple of nspecies so every cell holds an identical mixture
  // of all species. The deposited fields are then exactly uniform and the ODE reference
  // solution is exact for the discrete system. "random" tests momentum conservation
  // and the halo exchange with spatial fluctuations.
  bool lattice = pin->GetOrAddBoolean("problem","lattice",true);
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  int ppc_int = npart_permb/ncells;
  int nspec_ = pmbp->pdust->nspecies;
  // Optional third placement mode: a uniform particle lattice of lat_nx1 x lat_nx2
  // (x lat_nx3) nodes per MeshBlock, allowing non-integer ppc with uniform spacing
  // (e.g. ppc=1.5 on an 8x8 block via lat_nx1=12, lat_nx2=8). Requires lattice=false
  // and a single species.
  int lat_nx1 = pin->GetOrAddInteger("problem","lat_nx1",0);
  int lat_nx2 = pin->GetOrAddInteger("problem","lat_nx2",1);
  int lat_nx3 = pin->GetOrAddInteger("problem","lat_nx3",1);
  bool ulat = (lat_nx1 > 0);
  if (ulat && (lattice || (nspec_ != 1) ||
               (npart_permb != lat_nx1*lat_nx2*lat_nx3))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "uniform-lattice placement requires lattice=false, "
              << "nspecies=1, and ppc*ncells = lat_nx1*lat_nx2*lat_nx3" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (lattice && ((npart_permb != ppc_int*ncells) || (ppc_int % nspec_ != 0))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "lattice placement requires <particles>/ppc to be an "
              << "integer multiple of <dust>/nspecies" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int lnx1 = indcs.nx1, lnx2 = indcs.nx2, lnx3 = indcs.nx3;
  Kokkos::Random_XorShift64_Pool<> rand_pool64(pmbp->gids);
  par_for("dustdamp_part_pos", DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = p/npart_permb;
    if (m > (nmb-1)) {m = nmb-1;}
    pi(PGID,p) = gids + m;

    if (ulat) {
      int q = p - m*npart_permb;
      int i = q % lat_nx1;
      int j = (q/lat_nx1) % lat_nx2;
      int k = q/(lat_nx1*lat_nx2);
      pr(IPX,p) = CellCenterX(i, lat_nx1, mbsize.d_view(m).x1min,
                              mbsize.d_view(m).x1max);
      pr(IPY,p) = CellCenterX(j, lat_nx2, mbsize.d_view(m).x2min,
                              mbsize.d_view(m).x2max);
      pr(IPZ,p) = three_d ?
          CellCenterX(k, lat_nx3, mbsize.d_view(m).x3min,
                      mbsize.d_view(m).x3max) : 0.0;
    } else if (lattice) {
      int q = p - m*npart_permb;
      int c = q/ppc_int;    // cell index within block; all ppc particles at its center
      int i = c % lnx1;
      int j = (c/lnx1) % lnx2;
      int k = c/(lnx1*lnx2);
      pr(IPX,p) = CellCenterX(i, lnx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
      pr(IPY,p) = CellCenterX(j, lnx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
      pr(IPZ,p) = three_d ?
          CellCenterX(k, lnx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max) : 0.0;
    } else {
      auto rand_gen = rand_pool64.get_state();
      pr(IPX,p) = mbsize.d_view(m).x1min +
                  rand_gen.frand()*(mbsize.d_view(m).x1max - mbsize.d_view(m).x1min);
      pr(IPY,p) = mbsize.d_view(m).x2min +
                  rand_gen.frand()*(mbsize.d_view(m).x2max - mbsize.d_view(m).x2min);
      if (three_d) {
        pr(IPZ,p) = mbsize.d_view(m).x3min +
                    rand_gen.frand()*(mbsize.d_view(m).x3max - mbsize.d_view(m).x3min);
      } else {
        pr(IPZ,p) = 0.0;
      }
      rand_pool64.free_state(rand_gen);  // free state for use by other threads
    }
  });

  // assign species, stopping times, and masses (default striping/normalization)
  pmbp->pdust->SetDefaultMasses(pin);

  // In lattice mode reassign species within each cell so every cell holds an identical
  // mixture of all species; then set per-species initial velocities (<problem>/vd_s or
  // the default staircase), and optionally per-species dust-to-gas ratios via
  // <problem>/eps_1..N which override the equal-mass default of SetDefaultMasses
  int nspec = pmbp->pdust->nspecies;
  std::vector<Real> vdh = InitialDustVelocities(pin, nspec);
  bool has_eps = pin->DoesParameterExist("problem","eps_1");
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);
  DualArray2D<Real> spdat("dustdamp_spdat",nspec,2);  // [s][0]=vd, [s][1]=mass factor
  for (int s=0; s<nspec; ++s) {
    spdat.h_view(s,0) = vdh[s];
    Real eps_s = has_eps ? pin->GetReal("problem","eps_" + std::to_string(s+1)) : 0.0;
    spdat.h_view(s,1) = eps_s*rho0*static_cast<Real>(nspec)/ppc;
  }
  spdat.template modify<HostMemSpace>();
  spdat.template sync<DevExeSpace>();
  auto &taus_ = pmbp->pdust->taus;
  par_for("dustdamp_part_vel", DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = p/npart_permb;
    if (m > (nmb-1)) {m = nmb-1;}
    if (lattice) {
      int q = p - m*npart_permb;
      int s = (q % ppc_int) % nspec;
      pi(PSP,p) = s;
      pr(IPTS,p) = taus_.d_view(s);
    }
    int s = pi(PSP,p);
    if (has_eps) {
      Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
      if (three_d) {vol *= mbsize.d_view(m).dx3;}
      pr(IPM,p) = spdat.d_view(s,1)*vol;
    }
    pr(IPVX,p) = spdat.d_view(s,0);
    pr(IPVY,p) = 0.0;
    pr(IPVZ,p) = 0.0;
    pr(IPRX,p) = 0.0;
    pr(IPRY,p) = 0.0;
    pr(IPRZ,p) = 0.0;
  });

  // record initial total x-momentum on this rank (dust + gas)
  Real psum = 0.0;
  Kokkos::parallel_reduce("dustdamp_p0",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int &p, Real &sum) {
    sum += pr(IPM,p)*pr(IPVX,p);
  }, Kokkos::Sum<Real>(psum));
  p0x_thisrank = psum;
  for (int m=0; m<nmb; ++m) {
    Real volm = (pmbp->pmb->mb_size.h_view(m).x1max - pmbp->pmb->mb_size.h_view(m).x1min)
               *(pmbp->pmb->mb_size.h_view(m).x2max - pmbp->pmb->mb_size.h_view(m).x2min);
    if (three_d) {
      volm *= (pmbp->pmb->mb_size.h_view(m).x3max - pmbp->pmb->mb_size.h_view(m).x3min);
    }
    p0x_thisrank += rho0*vgas0*volm;
  }

  // set particle timestep from initial velocities
  Real dtnew = std::numeric_limits<float>::max();
  auto &msize = pmbp->pmb->mb_size;
  Real vmax = 1.0e-30;
  for (int s=0; s<nspec; ++s) {vmax = std::max(vmax, fabs(vdh[s]));}
  dtnew = std::min(dtnew, msize.h_view(0).dx1/vmax);
  dtnew = std::min(dtnew, msize.h_view(0).dx2/vmax);
  if (three_d) {dtnew = std::min(dtnew, msize.h_view(0).dx3/vmax);}
  ppar->dtnew = dtnew;

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustDampingHistory()
//! \brief User history output: per-species sums of dust x-velocity and particle counts
//! (summed over MPI ranks by the history machinery; divide sum by count to get the mean
//! species velocity at each output time). Enable with <problem>/user_hist = true.

void DustDampingHistory(HistoryData *pdata, Mesh *pm) {
  particles::Particles *ppar = pm->pmb_pack->ppart;
  int nspec = pm->pmb_pack->pdust->nspecies;
  int nsout = std::min(nspec, NHISTORY_VARIABLES/2);
  pdata->nhist = 2*nsout;

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  for (int s=0; s<nsout; ++s) {
    pdata->label[2*s  ] = "vdsum_" + std::to_string(s+1);
    pdata->label[2*s+1] = "np_" + std::to_string(s+1);
    Real vs = 0.0, ns = 0.0;
    Kokkos::parallel_reduce("dustdamp_hist",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &v_, Real &n_) {
      if (pi(PSP,p) == s) {
        v_ += pr(IPVX,p);
        n_ += 1.0;
      }
    }, Kokkos::Sum<Real>(vs), Kokkos::Sum<Real>(ns));
    pdata->hdata[2*s  ] = vs;
    pdata->hdata[2*s+1] = ns;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustDampingErrors()
//! \brief Computes errors in the mean gas and per-species dust velocities against a
//! fine-step RK4 reference solution of the damping ODE system, plus the total momentum
//! conservation error, and appends them to "dust_damping-errs.dat".

void DustDampingErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  dust::DustGasDrag *pdust = pmbp->pdust;
  particles::Particles *ppar = pmbp->ppart;
  int nspec = pdust->nspecies;
  Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);

  // (1) volume-integrated gas x-momentum and mass over active cells
  auto &indcs = pm->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;
  bool three_d = pm->three_d;

  Real gas_mom = 0.0, gas_mass = 0.0;
  Kokkos::parallel_reduce("dampgas_sum",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &mom, Real &mass) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    mom  += vol*u0(m,IM1,k,j,i);
    mass += vol*u0(m,IDN,k,j,i);
  }, Kokkos::Sum<Real>(gas_mom), Kokkos::Sum<Real>(gas_mass));

  // (2) per-species dust velocity sums and mass, and dust x-momentum
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  std::vector<Real> vsum(nspec), msum(nspec), nsum(nspec);
  Real dust_mom = 0.0;
  for (int s=0; s<nspec; ++s) {
    Real vs = 0.0, ms = 0.0, ns = 0.0;
    Kokkos::parallel_reduce("damppar_sum",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &v_, Real &m_, Real &n_) {
      if (pi(PSP,p) == s) {
        v_ += pr(IPVX,p);
        m_ += pr(IPM,p);
        n_ += 1.0;
      }
    }, Kokkos::Sum<Real>(vs), Kokkos::Sum<Real>(ms), Kokkos::Sum<Real>(ns));
    vsum[s] = vs;
    msum[s] = ms;
    nsum[s] = ns;
  }
  Real dmom = 0.0;
  Kokkos::parallel_reduce("damppar_mom",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int &p, Real &sum) {
    sum += pr(IPM,p)*pr(IPVX,p);
  }, Kokkos::Sum<Real>(dmom));
  dust_mom = dmom;
  Real p0x = p0x_thisrank;

#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &gas_mom,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &gas_mass, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &dust_mom, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &p0x,      1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, vsum.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, msum.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, nsum.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif

  // (3) reference solution: RK4 integration of the damping ODEs with fine steps
  std::vector<Real> vref = InitialDustVelocities(pin, nspec);
  std::vector<Real> eps(nspec), ts(nspec);
  Real uref = pin->GetOrAddReal("problem","vgas0",0.0);
  for (int s=0; s<nspec; ++s) {
    eps[s] = msum[s]/gas_mass;
    ts[s] = pdust->taus.h_view(s);
  }
  // step count adapts to the stiffest rate (1+eps_tot)/ts_min so the explicit RK4
  // reference stays deep inside its stability region (|h*lambda| <= 1/30) even for
  // strongly coupled cases (e.g. ts = 1e-3 with eps = 1e3 -> lambda ~ 1e6)
  Real eps_tot = 0.0, its_max = 0.0;
  for (int s=0; s<nspec; ++s) {
    eps_tot += eps[s];
    its_max = std::max(its_max, 1.0/ts[s]);
  }
  const int nref = std::max(100000,
      static_cast<int>(30.0*pm->time*(1.0 + eps_tot)*its_max));
  Real h = pm->time/static_cast<Real>(nref);
  const int ny = 1 + 2*nspec;
  std::vector<Real> k1(ny), k2(ny), k3(ny), k4(ny), y(ny), yt(ny);
  // y[0] = u, y[1..nspec] = v_s, y[nspec+1..2*nspec] = S_s (distance traveled)
  y[0] = uref;
  for (int s=0; s<nspec; ++s) {y[s+1] = vref[s]; y[nspec+1+s] = 0.0;}
  auto rhs = [&](const std::vector<Real> &w, std::vector<Real> &dw) {
    dw[0] = 0.0;
    for (int s=0; s<nspec; ++s) {
      dw[0]   += eps[s]*(w[s+1] - w[0])/ts[s];
      dw[s+1] = -(w[s+1] - w[0])/ts[s];
      dw[nspec+1+s] = w[s+1];
    }
  };
  for (int n=0; n<nref; ++n) {
    rhs(y, k1);
    for (int q=0; q<ny; ++q) {yt[q] = y[q] + 0.5*h*k1[q];}
    rhs(yt, k2);
    for (int q=0; q<ny; ++q) {yt[q] = y[q] + 0.5*h*k2[q];}
    rhs(yt, k3);
    for (int q=0; q<ny; ++q) {yt[q] = y[q] + h*k3[q];}
    rhs(yt, k4);
    for (int q=0; q<ny; ++q) {
      y[q] += (h/6.0)*(k1[q] + 2.0*k2[q] + 2.0*k3[q] + k4[q]);
    }
  }

  // (4) errors
  Real err_u = fabs(gas_mom/gas_mass - y[0]);
  std::vector<Real> err_v(nspec);
  for (int s=0; s<nspec; ++s) {
    err_v[s] = fabs(vsum[s]/nsum[s] - y[s+1]);
  }
  Real err_mom = fabs((gas_mom + dust_mom) - p0x);

  // (4b) per-species mean x-displacement error vs the RK4 reference S_s (the BS10
  // particle-gas deceleration test). Initial positions are reconstructed from PTAG,
  // which requires the serial sequential tag scheme and lattice placement; otherwise
  // NaN columns are written.
  std::vector<Real> err_S(nspec, std::numeric_limits<Real>::quiet_NaN());
  bool lattice = pin->GetOrAddBoolean("problem","lattice",true);
  int lat_nx1 = pin->GetOrAddInteger("problem","lat_nx1",0);
  bool ulat = (lat_nx1 > 0);
  if ((lattice || ulat) && global_variable::nranks == 1) {
    int nmb = pmbp->nmb_thispack;
    int npart_permb = npart/nmb;
    int ncells = nx1*nx2*nx3;
    int ppc_int = ulat ? 1 : npart_permb/ncells;
    Real Lx = pm->mesh_size.x1max - pm->mesh_size.x1min;
    DualArray1D<Real> sref("decel_sref",nspec);
    for (int s=0; s<nspec; ++s) {sref.h_view(s) = y[nspec+1+s];}
    sref.template modify<HostMemSpace>();
    sref.template sync<DevExeSpace>();
    int lnx1 = ulat ? lat_nx1 : nx1;
    for (int s=0; s<nspec; ++s) {
      Real ssum = 0.0, smin = (std::numeric_limits<Real>::max)(), smax = -smin;
      Kokkos::parallel_reduce("decel_ssum",
                              Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
      KOKKOS_LAMBDA(const int &p, Real &sum_, Real &mn_, Real &mx_) {
        if (pi(PSP,p) == s) {
          int t = pi(PTAG,p);
          int m0 = t/npart_permb;
          int q = t - m0*npart_permb;
          int c = q/ppc_int;
          int i = c % lnx1;
          Real x0 = CellCenterX(i, lnx1, mbsize.d_view(m0).x1min,
                                mbsize.d_view(m0).x1max);
          Real dxp = pr(IPX,p) - x0;
          Real sr = sref.d_view(s);
          while (dxp <  sr - 0.5*Lx) {dxp += Lx;}
          while (dxp >= sr + 0.5*Lx) {dxp -= Lx;}
          sum_ += dxp;
          mn_ = fmin(mn_, dxp);
          mx_ = fmax(mx_, dxp);
        }
      }, Kokkos::Sum<Real>(ssum), Kokkos::Min<Real>(smin), Kokkos::Max<Real>(smax));
      err_S[s] = fabs(ssum/nsum[s] - y[nspec+1+s]);
      if (global_variable::my_rank == 0 && s == 0) {
        std::printf("decel diag: species 0 displacement mean/min/max = "
                    "%.6e / %.6e / %.6e (S_ref = %.6e)\n",
                    ssum/nsum[s], smin, smax, y[nspec+1+s]);
      }
    }
  }

  // (5) root writes results to file
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
      std::fprintf(pfile,
                   "# Nx1  Nx2  Nx3  Ncycle  err_u  err_v_s...  err_mom  err_S_s...\n");
    }
    std::fprintf(pfile, "%04d  %04d  %04d  %05d  %e", pm->mesh_indcs.nx1,
                 pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle, err_u);
    for (int s=0; s<nspec; ++s) {
      std::fprintf(pfile, "  %e", err_v[s]);
    }
    std::fprintf(pfile, "  %e", err_mom);
    for (int s=0; s<nspec; ++s) {
      std::fprintf(pfile, "  %e", err_S[s]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}
