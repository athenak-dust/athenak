//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_si_strat.cpp
//! \brief Problem generator for the streaming instability in a vertically stratified 3-D
//! shearing box (<shearing_box>/stratified = true; e.g. Li, Youdin & Simon 2018; Li & Youdin
//! 2021).  The gas is isothermal and hydrostatic, rho0 exp(-z^2/2H^2) with H = cs/omega0, and
//! feels the radial forcing 2 Omega etavk through <hydro_srcterms>/const_accel; the dust of
//! every species has a Gaussian layer of thickness H_p.  The velocities are the NSH drift of the
//! midplane ratios, or with <problem>/nsh_vertical = true the NSH drift of the local dust-to-gas
//! ratio at every height (Athena's par_strat3d.c with ipert = 0).
//!
//! The dust: the new species keys (<dust>/epsilon_N or Z_N, H_p_N, ppc_N; see the README) or
//! the legacy ones (<problem>/eps_N, dust_hp).  <problem>/stratified_dust_sampling = gaussian
//! draws equal-mass root parents with z ~ N(0, H_p) (root_placement = random; dust_zcut > 0
//! truncates the Gaussian at |z| < dust_zcut); weight (default) places the particles uniformly
//! and multiplies the masses by exp(-z^2/2H_p^2).  <problem>/dust_vz0 adds a uniform vz.
//! With <problem>/cloud_root_random = true the parents are 3-D refinable clouds that split to
//! the leaves of a refined mesh (on a uniform mesh, the same parents as point particles).
//!
//! <problem>/vertical_bc = li2018 (with ix3_bc = ox3_bc = user): the outflow vertical
//! boundaries of Li, Youdin & Simon (2018) and the gas mass renormalisation after every cycle.
//! Without <particles> and <dust> the problem is the gas-only control (hydrostatic gas at the
//! dust-free drift u_phi = -etavk).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
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

namespace {

// the outflow vertical boundaries of Li et al. (2018) and the mass renormalisation
void LiVerticalBCs(Mesh *pm);
void LiVerticalBCsCoarse(Mesh *pm);
void LiRenormalize(Mesh *pm);
void SIStratHistory(HistoryData *pdata, Mesh *pm);
ParameterInput *strat_pin_ = nullptr;

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

[[noreturn]] void Fatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust_si_strat" << std::endl << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::DustSIStrat()

void ProblemGenerator::DustSIStrat(ParameterInput *pin, const bool restart) {
  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  const bool dust_active = dust_setup::DustActive(pmy_mesh_);
  dust_setup::RejectKeys(pin, "dust_si_strat", {
      {"dust_test", "none", "dust_damping_amr or dusty_wave_amr"},
      {"dust_test_vz0", "0", "dust_damping_amr"},
      {"stratified_dust", "true", "dust_nsh or dust_si_unstrat"},
      {"two_stream_v", "0", "dust_nsh"},
      {"two_stream_v_upper", "0", "dust_nsh"},
      {"blob_amp", "0", "dust_nsh"},
      {"gas_vx_amp", "0", "dust_nsh"},
      {"dust_ymode_amp", "0", "dust_nsh"},
      {"ba_f64_snapshots", "false", "dust_si_unstrat (2-D)"},
      {"ba_fields", "false", "dust_si_unstrat (2-D)"},
      {"ba_history", "false", "dust_si_unstrat (2-D)"}});
  if (!pmy_mesh_->three_d || !pin->DoesBlockExist("shearing_box") ||
      !pin->GetOrAddBoolean("shearing_box", "stratified", false)) {
    Fatal("needs a 3-D mesh and <shearing_box>/stratified = true");
  }
  dust_setup::SetupHistory(pin);
  user_hist_func = SIStratHistory;
  dust_setup::EnrollRefinement(pin, pmy_mesh_, this);
  strat_pin_ = pin;
  // <problem>/vertical_bc = li2018: the outflow vertical boundaries of Li et al.
  // (2018) with the mass renormalisation after every cycle; x3 faces of type user
  {
    const std::string vbc = pin->GetOrAddString("problem", "vertical_bc", "none");
    if (vbc == "li2018") {
      const auto &mbc = pmy_mesh_->mesh_bcs;
      if (pmbp->phydro == nullptr ||
          mbc[BoundaryFace::inner_x3] != BoundaryFlag::user ||
          mbc[BoundaryFace::outer_x3] != BoundaryFlag::user) {
        Fatal("vertical_bc = li2018: hydro and ix3_bc = ox3_bc = user");
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
      Fatal("<problem>/vertical_bc must be none or li2018");
    }
  }
  if (restart) {
    dust_setup::LogRestartParticles(pmy_mesh_);
    return;
  }
  if (pmbp->phydro == nullptr) Fatal("needs a <hydro> block");

  const Real omega0 = pin->GetReal("shearing_box","omega0");
  const Real qshear = pin->GetReal("shearing_box","qshear");
  const Real etavk = pin->GetReal("problem","etavk");
  std::vector<Real> eps, taus;
  dust_setup::ReadSpecies(pin, pmbp, true, eps, taus);
  int nspec = static_cast<int>(eps.size());
  Real ax = 2.0*omega0*etavk;
  const Real hg = pmbp->phydro->peos->eos_data.iso_cs/omega0;
  // <problem>/stratified_dust_sampling = gaussian (equal masses, z ~ N(0, H_p), the root
  // parents) or weight (uniform z, masses times exp(-z^2/2 H_p^2)); dust_zcut with gaussian only
  const std::string ssamp = pin->GetOrAddString("problem", "stratified_dust_sampling", "weight");
  if (ssamp != "weight" && ssamp != "gaussian") {
    Fatal("<problem>/stratified_dust_sampling must be weight or gaussian");
  }
  const bool gs = (ssamp == "gaussian");
  if (pin->GetOrAddReal("problem", "dust_zcut", 0.0) > 0.0 && !gs) {
    Fatal("<problem>/dust_zcut needs stratified_dust_sampling = gaussian");
  }
  if (gs && dust_active && !pin->GetOrAddBoolean("problem", "cloud_root_random", false)) {
    Fatal("stratified_dust_sampling = gaussian draws root parents: cloud_root_random = true, "
          "root_placement = random");
  }
  // the new species keys: abundances (epsilon_N, or Z_N converted to the midplane
  // ratio below), per-species ppc and H_p from the resolved table; legacy inputs untouched
  const dust::SpeciesTable *spt = dust_active ? &dust::Species(pin) : nullptr;
  const bool new_init = (spt != nullptr) && spt->new_init;
  dust_setup::StratDust sd;
  sd.on = true;
  sd.new_keys = new_init;
  sd.gauss = gs;
  sd.vz0 = pin->GetOrAddReal("problem", "dust_vz0", 0.0);
  std::vector<Real> &hps = sd.hp;   // per-species H_p
  std::vector<Real> &finside = sd.finside;   // fraction of each Gaussian inside its range
  // Z with column_normalization = redraw (the default, Athena-C):
  // the whole column Z Sigma_g A inside the cut (box, or dust_zcut inside it): gaussian samples
  // drawn again outside the cut with unchanged masses, weights divided by f_in (wnorm)
  bool &redraw = sd.redraw;
  std::vector<Real> &wnorm = sd.wnorm;   // per species: the factor of the weights (redraw: 1/f_in)
  Real &mg_box = sd.mg_box;              // the initial gas mass of the box (Z with domain)
  if (new_init) {
    for (int sp=0; sp<nspec; ++sp) eps[sp] = static_cast<Real>(spt->value[sp]);
    const auto hv = dust::SpeciesHp(pin, *spt, hg, false);
    hps.assign(hv.begin(), hv.end());
    const auto &ms = pmy_mesh_->mesh_size;
    const Real zc = gs ? pin->GetOrAddReal("problem", "dust_zcut", 0.0) : 0.0;
    const Real zlo = (zc > 0.0) ? std::max(ms.x3min, -zc) : ms.x3min;
    const Real zhi = (zc > 0.0) ? std::min(ms.x3max, zc) : ms.x3max;
    finside.resize(nspec);
    for (int sp=0; sp<nspec; ++sp) {
      const Real r2 = std::sqrt(2.0)*hps[sp];
      finside[sp] = 0.5*(std::erf(zhi/r2) - std::erf(zlo/r2));
    }
    redraw = (spt->abundance == "Z" && spt->column_normalization == "redraw");
    sd.domain = (spt->abundance == "Z" && spt->column_normalization == "domain");
    wnorm.assign(nspec, 1.0);
    if (redraw) for (int sp=0; sp<nspec; ++sp) wnorm[sp] = 1.0/finside[sp];
    if (sd.domain) {
      // the gas mass of the box as the initialisation below sets it (cell centres)
      const Real rho0g = pin->GetOrAddReal("problem", "rho0", 1.0);
      auto &ind = pmy_mesh_->mb_indcs;
      auto &mbs = pmbp->pmb->mb_size;
      const int nz3 = ind.nx3, nx = ind.nx1, ny = ind.nx2;
      const int nmbl = pmbp->nmb_thispack;
      double mg = 0.0;
      Kokkos::parallel_reduce("nsh_gas_mass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmbl*nz3),
      KOKKOS_LAMBDA(const int q, double &a) {
        const int m = q/nz3, k = q % nz3;
        const Real z = CellCenterX(k, nz3, mbs.d_view(m).x3min, mbs.d_view(m).x3max);
        const Real rho = rho0g*exp(-0.5*z*z/(hg*hg));
        a += static_cast<double>(rho)*mbs.d_view(m).dx1*mbs.d_view(m).dx2*mbs.d_view(m).dx3*nx*ny;
      }, mg);
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
        if (spt->column_normalization != "domain") {   // full_profile and redraw
          eps[sp] = static_cast<Real>(spt->value[sp])*hg/hps[sp];
        } else {
          eps[sp] = static_cast<Real>(spt->value[sp])*mg_box/
                    (rho0g*std::sqrt(2.0*M_PI)*hps[sp]*finside[sp]*area);
        }
      }
    }
  } else {   // legacy keys: the single <problem>/dust_hp (default H) for every species
    hps.assign(nspec, pin->GetOrAddReal("problem", "dust_hp", hg));
    finside.assign(nspec, 1.0);
    wnorm.assign(nspec, 1.0);
  }
  dust_setup::CheckRadialForcing(pin, ax);

  // equilibrium drift velocities of the midplane ratios
  Real ugx = 0.0, ugp = 0.0;
  std::vector<Real> vx(nspec, 0.0), vp(nspec, 0.0);
  dust_setup::SolveNSH(omega0, qshear, ax, eps, taus, ugx, ugp, vx, vp);
  // <problem>/nsh_vertical: the NSH drift with the local dust-to-gas ratio
  // eps_s(z) = eps_s exp(-z^2 (1/Hp^2 - 1/H^2)/2) for the gas (the mean of the two cell faces) and
  // the particles (their height), on a table of at least 64 points per root cell in z
  const bool nsh_vertical = dust_active &&
                            pin->GetOrAddBoolean("problem", "nsh_vertical", false);
  dust_setup::NSHTableZ &nshtab = sd.ztab;
  sd.ztab_on = nsh_vertical;
  if (nsh_vertical) {
    // the new keys: per-species H_p, and a truncated profile is not renormalised
    const Real hpz = hps[0];
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
      for (int s=0; s<nspec; ++s) {
        as[s] = 1.0/(hps[s]*hps[s]) - 1.0/(hg*hg);
        fcs[s] = redraw ? 1.0/finside[s] : 1.0;   // redraw: the column inside the cut
      }
    }
    std::vector<Real> ez(nspec), vxz(nspec, 0.0), vpz(nspec, 0.0);
    for (int n=0; n<nt; ++n) {
      const Real z = nshtab.z0 + n*nshtab.dz;
      const bool in = (zcut <= 0.0 || std::fabs(z) < zcut);
      for (int s=0; s<nspec; ++s) ez[s] = in ? eps[s]*fcs[s]*std::exp(-0.5*z*z*as[s]) : 0.0;
      Real gx = 0.0, gp = 0.0;
      dust_setup::SolveNSH(omega0, qshear, ax, ez, taus, gx, gp, vxz, vpz);
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
  if (global_variable::my_rank == 0) {
    if (dust_active) {
      std::cout << "# NSH equilibrium (midplane): u_gx=" << ugx << " u_gphi=" << ugp << std::endl;
      for (int s=0; s<nspec; ++s) {
        std::cout << "#   species " << s+1 << ": v_x=" << vx[s]
                  << " v_phi=" << vp[s] << std::endl;
      }
    } else {
      std::cout << "# Dust-free NSH equilibrium: u_gx=" << ugx
                << " u_gphi=" << ugp << std::endl;
    }
  }

  // the hydrostatic gas: the uniform gas at the equilibrium times exp(-z^2/2H^2), with the
  // NSH(z) drift at every height (nsh_vertical)
  const Real rho0 = pin->GetOrAddReal("problem","rho0",1.0);
  dust_setup::InitUniformGas(pmy_mesh_, rho0, ugx, ugp);
  {
    auto &indcs = pmy_mesh_->mb_indcs;
    const int ng = indcs.ng, nmb = pmbp->nmb_thispack;
    const int n1 = indcs.nx1 + 2*ng, n2 = indcs.nx2 + 2*ng, n3 = indcs.nx3 + 2*ng;
    auto &u0 = pmbp->phydro->u0;
    const Real hgas = hg;
    const int ks0 = indcs.ks, nz3 = indcs.nx3;
    auto &mbs = pmbp->pmb->mb_size;
    const bool nshz = nsh_vertical;
    const dust_setup::NSHTableZ ntab = nshtab;
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

  // the gas-only control has no particles
  if (!dust_active) return;

  sd.zcut = gs ? pin->GetOrAddReal("problem", "dust_zcut", 0.0) : 0.0;
  dust_setup::DustInit d;
  d.rho0 = rho0;
  d.eps = eps;
  d.vx = vx;
  d.vp = vp;
  d.placement = "lattice";
  d.strat = sd;
  dust_setup::PlaceParticles(pin, pmy_mesh_, d);
  dust_setup::SetParticleDt(pmy_mesh_, vx, vp, 0.0);
}

namespace {

//----------------------------------------------------------------------------------------
//! \fn SIStratHistory
//! \brief the per-species drift sums of dust_setup, and with vertical_bc = li2018 the last
//! renormalisation ratio and the gas mass put back so far (the gas-only control too)

void SIStratHistory(HistoryData *pdata, Mesh *pm) {
  dust_setup::SpeciesHistory(pdata, pm);
  if (livbc_.on && pdata->nhist + 2 <= NHISTORY_VARIABLES) {
    // rank 0 only: the history sums the ranks
    const bool r0 = (global_variable::my_rank == 0);
    pdata->label[pdata->nhist] = "vbc_mratio";
    pdata->label[pdata->nhist + 1] = "vbc_dm";
    pdata->hdata[pdata->nhist] = r0 ? livbc_.ratio : 0.0;
    pdata->hdata[pdata->nhist + 1] = r0 ? livbc_.dm : 0.0;
    pdata->nhist += 2;
  }
}

//----------------------------------------------------------------------------------------
//! \fn LiVerticalBCs, LiVerticalBCsCoarse
//! \brief the outflow vertical boundaries of Li et al. (2018) on u0 or, for the
//! ghost prolongation next to the x3 faces of a refined mesh, on coarse_u0 with the
//! coarse indices; the same rules as Athena's strat_ix3 / strat_ox3 (isothermal)

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

void LiVerticalBCs(Mesh *pm) {LiVerticalBCsApply(pm, false);}
void LiVerticalBCsCoarse(Mesh *pm) {LiVerticalBCsApply(pm, true);}

//----------------------------------------------------------------------------------------
//! \fn LiRenormalize
//! \brief at the start of every cycle (ProblemGenerator::user_work_in_loop_func):
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
  // the running total travels in the restart files
  if (strat_pin_ != nullptr) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", livbc_.dm);
    strat_pin_->SetString("problem", "vbc_dm", buf);
  }
}

}  // namespace
