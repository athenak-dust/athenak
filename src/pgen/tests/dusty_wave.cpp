//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dusty_wave.cpp
//! \brief Problem generator for the damped dusty sound wave test (Laibe & Price 2011;
//! Benitez-Llambay et al. 2019 Sec. 3.2; Krapp et al. 2024 Sec. 3.3). An isothermal gas
//! is initialized with the exact (complex) eigenmode of the linearized gas-dust system
//! for a right-traveling sound wave along x1, with N dust species of dust-to-gas ratios
//! eps_s (<problem>/eps_1..N, default dust_to_gas/nspecies each). Perturbations evolve
//! as exp(i(k x - omega t)) with omega the root of
//!     omega^2 * (1 + sum_s eps_s/(1 - i*omega*t_s)) = k^2 cs^2 ,
//! solved by complex Newton iteration. The eigenvector may be normalized so that either
//! the gas velocity amplitude is amp*cs (<problem>/normalization = velocity, default) or
//! the gas density amplitude is amp*rho0 and real (normalization = density, matching
//! Table 2 of Benitez-Llambay et al. 2019 and Fig. 4 of Krapp et al. 2024).
//!
//! At the end of the run, L1 errors of the gas density/velocity fields and of the
//! per-species particle velocities against the analytic damped wave are appended to
//! "<basename>-errs.dat"; second-order convergence in resolution is expected. With
//! <problem>/user_hist = true, the history output additionally records the evolution of
//! the density of every fluid in the first cell column (x ~ 0): the summed gas density
//! over the column, and the TSC-deposited dust column mass per species.

// C headers

// C++ headers
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>     // fopen(), fprintf()
#include <iostream>
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
void DustyWaveErrors(ParameterInput *pin, Mesh *pm);
void DustyWaveHistory(HistoryData *pdata, Mesh *pm);

namespace {

//----------------------------------------------------------------------------------------
//! \fn DustyWaveOmega
//! \brief Solves the dusty-wave dispersion relation for the right-traveling damped root
//! by complex Newton iteration, starting from the tightly-coupled limit.

std::complex<Real> DustyWaveOmega(const Real kwv, const Real cs,
                                  const std::vector<Real> &eps,
                                  const std::vector<Real> &ts) {
  const std::complex<Real> iunit(0.0, 1.0);
  int nspec = static_cast<int>(eps.size());
  Real epstot = 0.0;
  for (int s=0; s<nspec; ++s) {epstot += eps[s];}
  std::complex<Real> omega(kwv*cs/sqrt(1.0 + epstot), 0.0);
  for (int it=0; it<100; ++it) {
    std::complex<Real> ssum(0.0,0.0), dsum(0.0,0.0);
    for (int s=0; s<nspec; ++s) {
      std::complex<Real> den = 1.0 - iunit*omega*ts[s];
      ssum += eps[s]/den;
      dsum += eps[s]*iunit*ts[s]/(den*den);
    }
    std::complex<Real> f  = omega*omega*(1.0 + ssum) - SQR(kwv*cs);
    std::complex<Real> fp = 2.0*omega*(1.0 + ssum) + omega*omega*dsum;
    std::complex<Real> domega = f/fp;
    omega -= domega;
    if (std::abs(domega) < 1.0e-15*std::abs(omega)) {break;}
  }
  return omega;
}

//----------------------------------------------------------------------------------------
//! \fn WaveSetup
//! \brief Reads wave parameters and computes the complex eigenmode amplitudes with the
//! chosen normalization. Shared by the problem generator and the error function.

void WaveSetup(ParameterInput *pin, MeshBlockPack *pmbp, Mesh *pmesh,
               Real &kwv, Real &cs, Real &rho0, Real &amp,
               std::vector<Real> &eps, std::vector<Real> &ts,
               std::complex<Real> &omega, std::complex<Real> &du,
               std::complex<Real> &drho, std::vector<std::complex<Real>> &dv) {
  dust::DustGasDrag *pdust = pmbp->pdust;
  int nspec = pdust->nspecies;
  rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  amp  = pin->GetOrAddReal("problem","amp",1.0e-6);
  cs = pmbp->phydro->peos->eos_data.iso_cs;
  Real x1size = pmesh->mesh_size.x1max - pmesh->mesh_size.x1min;
  kwv = 2.0*M_PI/x1size;

  eps.resize(nspec);
  ts.resize(nspec);
  bool has_eps = pin->DoesParameterExist("problem","eps_1");
  for (int s=0; s<nspec; ++s) {
    if (has_eps) {
      eps[s] = pin->GetReal("problem","eps_" + std::to_string(s+1));
    } else {
      eps[s] = pdust->dust_to_gas/static_cast<Real>(nspec);
    }
    ts[s] = pdust->taus.h_view(s);
  }
  omega = DustyWaveOmega(kwv, cs, eps, ts);

  // eigenmode amplitudes: delta_rho_g = (k/omega)*rho0*delta_u;
  // delta_v_s = delta_u/(1 - i*omega*ts)
  const std::complex<Real> iunit(0.0, 1.0);
  du = amp*cs;
  drho = (kwv/omega)*rho0*du;
  dv.resize(nspec);
  for (int s=0; s<nspec; ++s) {
    dv[s] = du/(1.0 - iunit*omega*ts[s]);
  }
  // optionally renormalize so the gas density amplitude is amp*rho0 and real (BKP19)
  std::string norm = pin->GetOrAddString("problem","normalization","velocity");
  if (norm.compare("density") == 0) {
    std::complex<Real> cfac = amp*rho0/drho;
    du *= cfac;
    drho = amp*rho0;
    for (int s=0; s<nspec; ++s) {dv[s] *= cfac;}
  }
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustyWave()
//! \brief Problem Generator for the damped dusty sound wave

void ProblemGenerator::DustyWave(ParameterInput *pin, const bool restart) {
  pgen_final_func = DustyWaveErrors;
  user_hist_func = DustyWaveHistory;   // used only when <problem>/user_hist = true
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dusty wave test requires <hydro>, <particles>, and <dust> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  dust::DustGasDrag *pdust = pmbp->pdust;
  particles::Particles *ppar = pmbp->ppart;
  int nspec = pdust->nspecies;

  Real kwv, cs, rho0, amp;
  std::vector<Real> eps, ts;
  std::complex<Real> omega, du, drho;
  std::vector<std::complex<Real>> dv;
  WaveSetup(pin, pmbp, pmy_mesh_, kwv, cs, rho0, amp, eps, ts, omega, du, drho, dv);

  // displacement field amplitude imprinting delta_rhod_s/rhod_s = Re[(k/w) dv e^{ikx}]
  const std::complex<Real> iunit(0.0, 1.0);
  std::vector<std::complex<Real>> xi(nspec);
  for (int s=0; s<nspec; ++s) {
    xi[s] = iunit*(kwv/omega)*dv[s]/kwv;
  }

  // initialize gas
  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nmb = pmbp->nmb_thispack;
  int nx1 = indcs.nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;
  Real drho_r = drho.real(), drho_i = drho.imag();
  Real du_r = du.real(), du_i = du.imag();
  par_for("dustywave_gas", DevExeSpace(),0,(nmb-1),ks,ke,js,je,is,ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    Real x1v = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    Real rho = rho0 + drho_r*cos(kwv*x1v) - drho_i*sin(kwv*x1v);
    Real vx  = du_r*cos(kwv*x1v) - du_i*sin(kwv*x1v);
    u0(m,IDN,k,j,i) = rho;
    u0(m,IM1,k,j,i) = rho*vx;
    u0(m,IM2,k,j,i) = 0.0;
    u0(m,IM3,k,j,i) = 0.0;
  });

  // initialize particles: one particle of every species at each cell center, displaced
  // by the (linear) eigenmode displacement field to imprint the dust density wave
  int npart = ppar->nprtcl_thispack;
  int npart_permb = npart/nmb;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  int ppc_int = npart_permb/ncells;
  if ((npart_permb != ppc_int*ncells) || (ppc_int % nspec != 0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Dusty wave requires <particles>/ppc to be an integer multiple of "
              << "<dust>/nspecies" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto gids = pmbp->gids;
  bool three_d = pmy_mesh_->three_d;
  int lnx1 = indcs.nx1, lnx2 = indcs.nx2, lnx3 = indcs.nx3;
  auto &taus_ = pdust->taus;
  Real ppc = pin->GetOrAddReal("particles","ppc",1.0);

  // copy per-species data into device-usable array:
  // [s][0,1]=Re,Im dv; [s][2,3]=Re,Im xi; [s][4]=mass factor
  DualArray2D<Real> damp("dwave_amp",nspec,5);
  for (int s=0; s<nspec; ++s) {
    damp.h_view(s,0) = dv[s].real();
    damp.h_view(s,1) = dv[s].imag();
    damp.h_view(s,2) = xi[s].real();
    damp.h_view(s,3) = xi[s].imag();
    damp.h_view(s,4) = eps[s]*rho0*static_cast<Real>(nspec)/ppc;
  }
  damp.template modify<HostMemSpace>();
  damp.template sync<DevExeSpace>();

  par_for("dustywave_part", DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = p/npart_permb;
    if (m > (nmb-1)) {m = nmb-1;}
    pi(PGID,p) = gids + m;
    int q = p - m*npart_permb;
    int c = q/ppc_int;
    int s = (q % ppc_int) % nspec;
    int i = c % lnx1;
    int j = (c/lnx1) % lnx2;
    int k = c/(lnx1*lnx2);
    Real x0 = CellCenterX(i, lnx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    // displace position to imprint the dust density perturbation
    Real x = x0 + damp.d_view(s,2)*cos(kwv*x0) - damp.d_view(s,3)*sin(kwv*x0);
    pr(IPX,p) = x;
    pr(IPY,p) = CellCenterX(j, lnx2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max);
    pr(IPZ,p) = three_d ?
        CellCenterX(k, lnx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max) : 0.0;
    pi(PSP,p) = s;
    pr(IPTS,p) = taus_.d_view(s);
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    pr(IPM,p) = damp.d_view(s,4)*vol;
    pr(IPVX,p) = damp.d_view(s,0)*cos(kwv*x0) - damp.d_view(s,1)*sin(kwv*x0);
    pr(IPVY,p) = 0.0;
    pr(IPVZ,p) = 0.0;
    pr(IPRX,p) = 0.0;
    pr(IPRY,p) = 0.0;
    pr(IPRZ,p) = 0.0;
  });

  // particle timestep
  ppar->dtnew = mbsize.h_view(0).dx1/std::max(amp*cs, 1.0e-30);

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustyWaveErrors()
//! \brief Computes L1 errors of the gas density/velocity fields and of the per-species
//! particle velocities against the analytic damped traveling wave, and appends them to
//! "<basename>-errs.dat". Columns: Nx1 Nx2 Nx3 Ncycle L1_drho L1_dvx Re(omega)
//! Im(omega) e_tot L1_dvp_1 ... L1_dvp_N, where e_tot = L1_drho + L1_dvx + sum_s
//! L1_dvp_s is the analogue of Eq. (46) of Krapp et al. (2024) with the dust
//! contributions evaluated at the particle positions.

void DustyWaveErrors(ParameterInput *pin, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  int nspec = pmbp->pdust->nspecies;

  Real kwv, cs, rho0, amp;
  std::vector<Real> eps, ts;
  std::complex<Real> omega, du, drho;
  std::vector<std::complex<Real>> dv;
  WaveSetup(pin, pmbp, pm, kwv, cs, rho0, amp, eps, ts, omega, du, drho, dv);

  // analytic solution at t: fields evolve as exp(i(kx - omega*t))
  const std::complex<Real> iunit(0.0, 1.0);
  std::complex<Real> phase = std::exp(-iunit*omega*pm->time);
  std::complex<Real> du_t = du*phase;
  std::complex<Real> drho_t = drho*phase;
  Real du_r = du_t.real(), du_i = du_t.imag();
  Real drho_r = drho_t.real(), drho_i = drho_t.imag();

  auto &indcs = pm->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;

  Real l1_rho = 0.0, l1_vx = 0.0;
  Kokkos::parallel_reduce("dwave_err",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &e_rho, Real &e_vx) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real x1v = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    Real rho_a = rho0 + drho_r*cos(kwv*x1v) - drho_i*sin(kwv*x1v);
    Real vx_a  = du_r*cos(kwv*x1v) - du_i*sin(kwv*x1v);
    e_rho += fabs(u0(m,IDN,k,j,i) - rho_a);
    e_vx  += fabs(u0(m,IM1,k,j,i)/u0(m,IDN,k,j,i) - vx_a);
  }, Kokkos::Sum<Real>(l1_rho), Kokkos::Sum<Real>(l1_vx));

  // per-species particle velocity errors against the analytic eigenmode
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  std::vector<Real> l1_vp(nspec), np_s(nspec);
  for (int s=0; s<nspec; ++s) {
    std::complex<Real> dv_t = dv[s]*phase;
    Real dv_r = dv_t.real(), dv_i = dv_t.imag();
    Real es = 0.0, ns = 0.0;
    Kokkos::parallel_reduce("dwave_perr",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &e_, Real &n_) {
      if (pi(PSP,p) == s) {
        Real vx_a = dv_r*cos(kwv*pr(IPX,p)) - dv_i*sin(kwv*pr(IPX,p));
        e_ += fabs(pr(IPVX,p) - vx_a);
        n_ += 1.0;
      }
    }, Kokkos::Sum<Real>(es), Kokkos::Sum<Real>(ns));
    l1_vp[s] = es;
    np_s[s] = ns;
  }

  int ncells_total = pm->mesh_indcs.nx1*pm->mesh_indcs.nx2*pm->mesh_indcs.nx3;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &l1_rho, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &l1_vx,  1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, l1_vp.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, np_s.data(), nspec, MPI_ATHENA_REAL, MPI_SUM,
                MPI_COMM_WORLD);
#endif
  l1_rho /= static_cast<Real>(ncells_total);
  l1_vx  /= static_cast<Real>(ncells_total);
  Real e_tot = l1_rho + l1_vx;
  for (int s=0; s<nspec; ++s) {
    l1_vp[s] /= np_s[s];
    e_tot += l1_vp[s];
  }

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
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  L1_drho  L1_dvx  Re(omega)  "
                          "Im(omega)  e_tot  L1_dvp_s...\n");
    }
    std::fprintf(pfile, "%04d  %04d  %04d  %05d  %e  %e  %e  %e  %e", pm->mesh_indcs.nx1,
                 pm->mesh_indcs.nx2, pm->mesh_indcs.nx3, pm->ncycle, l1_rho, l1_vx,
                 omega.real(), omega.imag(), e_tot);
    for (int s=0; s<nspec; ++s) {
      std::fprintf(pfile, "  %e", l1_vp[s]);
    }
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }

  return;
}

//----------------------------------------------------------------------------------------
//! \fn void DustyWaveHistory()
//! \brief User history output: evolution of the density of every fluid at the first cell
//! column (x ~ x1min + dx/2). Records the summed gas density over the column cells, and
//! the TSC-deposited dust column mass of every species (divide by the column volume and
//! cell count in post-processing). Values are rank-local sums combined by the history
//! machinery; use <output>/data_format = %.14e to resolve small perturbations.

void DustyWaveHistory(HistoryData *pdata, Mesh *pm) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  particles::Particles *ppar = pmbp->ppart;
  int nspec = pmbp->pdust->nspecies;
  int nsout = std::min(nspec, NHISTORY_VARIABLES-1);
  pdata->nhist = 1 + nsout;
  pdata->label[0] = "grho_col";
  for (int s=0; s<nsout; ++s) {
    pdata->label[1+s] = "dmcol_" + std::to_string(s+1);
  }

  // measurement column: center of the first cell of the global grid
  Real x1min_m = pm->mesh_size.x1min;
  Real x1size = pm->mesh_size.x1max - x1min_m;
  Real dx = x1size/static_cast<Real>(pm->mesh_indcs.nx1);
  Real xcol = x1min_m + 0.5*dx;

  // gas: sum of density over all active cells in the column
  auto &indcs = pm->mb_indcs;
  int is = indcs.is;
  int js = indcs.js;
  int ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmkji = (pmbp->nmb_thispack)*nx3*nx2*nx1;
  const int nkji = nx3*nx2*nx1;
  const int nji = nx2*nx1;
  auto &u0 = pmbp->phydro->u0;
  auto &mbsize = pmbp->pmb->mb_size;
  Real grho = 0.0;
  Kokkos::parallel_reduce("dwave_hcol",Kokkos::RangePolicy<>(DevExeSpace(),0,nmkji),
  KOKKOS_LAMBDA(const int &idx, Real &sum) {
    int m = (idx)/nkji;
    int k = (idx - m*nkji)/nji;
    int j = (idx - m*nkji - k*nji)/nx1;
    int i = (idx - m*nkji - k*nji - j*nx1) + is;
    k += ks;
    j += js;
    Real x1v = CellCenterX(i-is, nx1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max);
    if (fabs(x1v - xcol) < 0.1*dx) {sum += u0(m,IDN,k,j,i);}
  }, Kokkos::Sum<Real>(grho));
  pdata->hdata[0] = grho;

  // dust: TSC-weighted column mass per species; periodic wrap in x handled by distance
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  for (int s=0; s<nsout; ++s) {
    Real msum = 0.0;
    Kokkos::parallel_reduce("dwave_hdep",Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
    KOKKOS_LAMBDA(const int &p, Real &sum) {
      if (pi(PSP,p) == s) {
        Real d = pr(IPX,p) - xcol;
        d -= x1size*round(d/x1size);   // periodic wrap
        Real ad = fabs(d)/dx;
        Real w = 0.0;
        if (ad <= 0.5) {
          w = 0.75 - ad*ad;
        } else if (ad <= 1.5) {
          w = 0.5*SQR(1.5 - ad);
        }
        sum += w*pr(IPM,p);
      }
    }, Kokkos::Sum<Real>(msum));
    pdata->hdata[1+s] = msum;
  }
  return;
}
