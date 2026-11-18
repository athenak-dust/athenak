//========================================================================================
// AthenaK self-gravity validation problems. BSD-3-Clause; see LICENSE.
//========================================================================================
// poisson: cell-averaged periodic/sheared sinusoidal density, stationary field check.
// slab: nonzero mean isolated slab, with a horizontal sinusoidal density perturbation.
// slab_evolve: the same positive slab initialization, then freely evolving hydro;
//              final diagnostics are measured state, not a frozen analytic solution.
// jeans: physical isothermal/adiabatic linear mode initialization, without external forcing.
// shear: physical isothermal shearing wave, compared externally with its linear ODE.
// topology: explicitly diagnostic source reset before refinement checks, not hydro dynamics.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "mesh/nghbr_index.hpp"
#include "eos/eos.hpp"
#include "hydro/hydro.hpp"
#include "gravity/self_gravity.hpp"
#include "outputs/outputs.hpp"
#include "pgen/pgen.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"

namespace {
struct WaveData {
  Real rho0, amp, phase, kx, ky, kz, qomega, x0, y0, z0;
  Real fourpig, frequency, growth, pressure;
  Real refine_start, refine_end, band_lo, band_hi;
  Real band2_lo, band2_hi, band3_lo, band3_hi;
  bool jeans, shear, topology, slab, slab_evolve, scheduled_amr;
  bool spitzer;
  Real zs, zc, cs2;   // spitzer: z0 = cs/sqrt(2 pi G rho0), the slab center, cs^2
  bool reflect;       // spitzer: <problem>/test_spitzer_reflect (ghost vz mirrored, no mass flux)
};
WaveData wave;

[[noreturn]] void WaveFail(const std::string &message) {
  std::cerr << "self_gravity_wave: " << message << '\n';
  std::exit(EXIT_FAILURE);
}

KOKKOS_INLINE_FUNCTION
Real Sinc(Real value) { return fabs(value) < Real(1.e-10) ? Real(1) : sin(value)/value; }

void InitializeKnownWave(Mesh *pm, Real time) {
  auto *pack = pm->pmb_pack;
  const auto ind = pm->mb_indcs;
  const auto sv = wave;
  const Real kx = sv.kx+sv.qomega*time*sv.ky;
  const Real k2 = kx*kx+sv.ky*sv.ky+sv.kz*sv.kz;
  const Real amplitude = sv.amp*std::exp(sv.growth*time);
  auto u = pack->phydro->u0;
  auto size = pack->pmb->mb_size.d_view;
  const auto eos = pack->phydro->peos->eos_data;
  par_for("sg_wave_initial", DevExeSpace(), 0, pack->nmb_thispack-1,
          ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto s = size(m);
    const Real x = s.x1min+(i-ind.is+Real(.5))*s.dx1-sv.x0;
    const Real y = s.x2min+(j-ind.js+Real(.5))*s.dx2-sv.y0;
    const Real z = s.x3min+(k-ind.ks+Real(.5))*s.dx3-sv.z0;
    const Real theta = kx*x+sv.ky*y+sv.kz*z+sv.phase-sv.frequency*time;
    const Real average = Sinc(Real(.5)*kx*s.dx1)*Sinc(Real(.5)*sv.ky*s.dx2)*
                         Sinc(Real(.5)*sv.kz*s.dx3);
    const Real density = sv.rho0*(1+amplitude*average*cos(theta));
    u(m,IDN,k,j,i) = density;
    const Real momentum_amplitude = sv.rho0*amplitude*average/k2*
        (sv.frequency*cos(theta)-sv.growth*sin(theta));
    u(m,IM1,k,j,i) = momentum_amplitude*kx;
    u(m,IM2,k,j,i) = momentum_amplitude*sv.ky;
    u(m,IM3,k,j,i) = momentum_amplitude*sv.kz;
    if (eos.is_ideal) {
      const Real p2 = SQR(u(m,IM1,k,j,i))+SQR(u(m,IM2,k,j,i))+SQR(u(m,IM3,k,j,i));
      const Real pressure = sv.jeans ?
          sv.pressure*(1+eos.gamma*amplitude*average*cos(theta)) : sv.pressure;
      u(m,IEN,k,j,i) = pressure/(eos.gamma-1)+Real(.5)*p2/density;
    }
  });
}

array_sum::GlobalSum WaveMoments(Mesh *pm) {
  const auto sv = wave;
  const auto ind = pm->mb_indcs;
  const Real kx = sv.kx+sv.qomega*pm->time*sv.ky;
  const Real knorm = std::sqrt(kx*kx+sv.ky*sv.ky+sv.kz*sv.kz);
  auto u = pm->pmb_pack->phydro->u0;
  auto size = pm->pmb_pack->pmb->mb_size.d_view;
  const int nc = ind.nx1*ind.nx2*ind.nx3;
  const int n = pm->pmb_pack->nmb_thispack*nc;
  const bool three_d = pm->three_d;
  const bool ideal_jeans = sv.jeans && pm->pmb_pack->phydro->peos->eos_data.is_ideal;
  const auto *gravity = pm->pmb_pack->pgrav;
  const Real source_mean = gravity == nullptr ? 0 : gravity->last_source_mean;
  auto phi = gravity == nullptr ? DvceArray5D<Real>() : gravity->phi;
  array_sum::GlobalSum result;
  Kokkos::parallel_reduce("sg_wave_moments",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(int q, array_sum::GlobalSum &sum) {
    const int m=q/nc;
    int t=q%nc;
    const int i=t%ind.nx1+ind.is; t/=ind.nx1;
    const int j=t%ind.nx2+ind.js; t/=ind.nx2;
    const int k=t+ind.ks;
    const auto s=size(m);
    const Real x=s.x1min+(i-ind.is+Real(.5))*s.dx1-sv.x0;
    const Real y=s.x2min+(j-ind.js+Real(.5))*s.dx2-sv.y0;
    const Real z=s.x3min+(k-ind.ks+Real(.5))*s.dx3-sv.z0;
    const Real theta=kx*x+sv.ky*y+sv.kz*z+sv.phase;
    const Real average=Sinc(Real(.5)*kx*s.dx1)*Sinc(Real(.5)*sv.ky*s.dx2)*
                       Sinc(Real(.5)*sv.kz*s.dx3);
    const Real c=average*cos(theta), sn=average*sin(theta);
    const Real vol=s.dx1*s.dx2*(three_d ? s.dx3 : Real(1));
    const Real density=u(m,IDN,k,j,i)/sv.rho0-1;
    const Real momentum=(kx*u(m,IM1,k,j,i)+sv.ky*u(m,IM2,k,j,i)+
                         sv.kz*u(m,IM3,k,j,i))/(sv.rho0*knorm);
    sum.the_array[0]+=vol*density*c;
    sum.the_array[1]+=vol*density*sn;
    sum.the_array[2]+=vol*momentum*c;
    sum.the_array[3]+=vol*momentum*sn;
    sum.the_array[4]+=vol*c*c;
    sum.the_array[5]+=vol*sn*sn;
    sum.the_array[6]+=vol*c*sn;
    for (int d=0;d<3;++d) {
      const Real momentum_component=u(m,IM1+d,k,j,i)/sv.rho0;
      sum.the_array[7+2*d]+=vol*momentum_component*c;
      sum.the_array[8+2*d]+=vol*momentum_component*sn;
      sum.the_array[14+d]+=vol*u(m,IM1+d,k,j,i);
    }
    sum.the_array[13]+=vol*u(m,IDN,k,j,i);
    sum.the_array[17]+=vol;
    if (ideal_jeans) {
      // Match the native gas-energy history volume, including the homogeneous
      // third-coordinate extent in 2D. The periodic mean makes this gauge free.
      sum.the_array[19]+=Real(.5)*s.dx1*s.dx2*s.dx3*
          (u(m,IDN,k,j,i)-source_mean)*phi(m,0,k,j,i);
    }
  }, Kokkos::Sum<array_sum::GlobalSum>(result));
  result.the_array[18]=pm->pmb_pack->nmb_thispack;
  auto *orbital=pm->pmb_pack->phydro->porb_u;
  if (pm->three_d && orbital!=nullptr) {
    for (int m=0;m<pm->pmb_pack->nmb_thispack;++m) {
      if (orbital->xlev.h_view(m,0)!=0 || orbital->xlev.h_view(m,1)!=0) {
        result.the_array[19]+=1;
      }
    }
  }
  return result;
}

void WaveHistory(HistoryData *data, Mesh *pm) {
  const auto moments=WaveMoments(pm);
  const char *labels[] = {"rho_cos_integral","rho_sin_integral","mom_cos_integral",
                         "mom_sin_integral","basis_cc","basis_ss","basis_cs",
                         "mom1_cos_integral","mom1_sin_integral","mom2_cos_integral",
                         "mom2_sin_integral","mom3_cos_integral","mom3_sin_integral",
                         "mass","momentum1","momentum2","momentum3","volume","blocks",
                         "oa_mixed_blocks"};
  data->nhist=20;
  for (int q=0;q<20;++q) {data->label[q]=labels[q]; data->hdata[q]=moments.the_array[q];}
  if (wave.jeans && pm->pmb_pack->phydro->peos->eos_data.is_ideal) {
    data->label[19]="grav_potential_energy";
  }
}

// The shearing-wave amplitude is not the frozen initial sinusoid. Save measured
// finite-volume projections for comparison to an independently integrated ODE.
void ShearDiagnostics(Mesh *pm) {
  auto moments=WaveMoments(pm);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,moments.the_array,20,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  const auto &b=moments.the_array;
  const Real determinant=b[4]*b[5]-b[6]*b[6];
  if (global_variable::my_rank!=0) return;
  std::cout << std::setprecision(16) << "# SELF_GRAVITY_WAVE {\"mode\":\"shear\",\"time\":"
            << pm->time << ",\"cycle\":" << pm->ncycle << ",\"blocks\":" << pm->nmb_total;
  const char *names[]={"density","momentum1_over_rho0","momentum2_over_rho0",
                       "momentum3_over_rho0"};
  const int offsets[]={0,7,9,11};
  for (int d=0;d<4;++d) {
    const int q=offsets[d];
    std::cout << ",\"" << names[d] << "_mode_cos\":" << (b[q]*b[5]-b[q+1]*b[6])/determinant
              << ",\"" << names[d] << "_mode_sin\":" << (b[q+1]*b[4]-b[q]*b[6])/determinant;
  }
  std::cout << ",\"mass\":" << b[13] << ",\"momentum1\":" << b[14]
            << ",\"momentum2\":" << b[15] << ",\"momentum3\":" << b[16]
            << ",\"volume\":" << b[17]
            << ",\"oa_mixed_blocks\":" << b[19]
            << ",\"gravity_enabled\":" << (pm->pmb_pack->pgrav != nullptr ? "true" : "false");
  if (const auto *gravity = pm->pmb_pack->pgrav) {
    std::cout << ",\"iterations\":" << gravity->last_solve.iterations
              << ",\"residual\":" << gravity->last_solve.relative_residual
              << ",\"solves\":" << gravity->solve_count
              << ",\"rebuilds\":" << gravity->rebuild_count;
  }
  std::cout << "}\n";
}

// mode = spitzer: the isothermal self-gravitating slab rho0 sech^2((z - zc)/z0),
// z0 = cs/sqrt(2 pi G rho0), at rest, isolated in z (Spitzer 1942); its cell averages
KOKKOS_INLINE_FUNCTION
Real SpitzerAverage(const WaveData &sv, const Real zlo, const Real zhi) {
  return sv.rho0*sv.zs*(tanh((zhi - sv.zc)/sv.zs) - tanh((zlo - sv.zc)/sv.zs))/(zhi - zlo);
}

void SpitzerInitialize(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  const auto ind = pm->mb_indcs;
  const auto sv = wave;
  auto u = pack->phydro->u0;
  auto size = pack->pmb->mb_size.d_view;
  par_for("sg_spitzer_initial", DevExeSpace(), 0, pack->nmb_thispack-1,
          ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    const auto s = size(m);
    const Real zlo = s.x3min+(k-ind.ks)*s.dx3;
    u(m,IDN,k,j,i) = SpitzerAverage(sv,zlo,zlo+s.dx3);
    u(m,IM1,k,j,i) = 0; u(m,IM2,k,j,i) = 0; u(m,IM3,k,j,i) = 0;
  });
}

// ix3_bc / ox3_bc = user: the ghost density hydrostatic in the self-gravity field of the
// boundary cell, cs^2 dln(rho)/dz = g_b - 4 pi G rho_b (z - z_b) (dust_si_strat's li2018 rule
// velocity copied unless it flows into the box (then 0; test_spitzer_reflect: mirrored); before a
// field of the current mesh exists,
// the analytic profile.  Coarse arrays: g_b the mean of the 2x2x2 fine boundary cells.
void SpitzerBCsApply(Mesh *pm, const bool coarse) {
  auto *pack = pm->pmb_pack;
  const auto ind = pm->mb_indcs;
  const auto sv = wave;
  const int ng = ind.ng;
  const int nx1 = coarse ? ind.cnx1 : ind.nx1, nx2 = coarse ? ind.cnx2 : ind.nx2;
  const int nx3 = coarse ? ind.cnx3 : ind.nx3;
  const int n1 = nx1+2*ng, n2 = nx2+2*ng;
  const int ks = coarse ? ind.cks : ind.ks, ke = coarse ? ind.cke : ind.ke;
  const int ais = coarse ? ind.cis : ind.is, aie = coarse ? ind.cie : ind.ie;
  const int ajs = coarse ? ind.cjs : ind.js, aje = coarse ? ind.cje : ind.je;
  const int fis = ind.is, fjs = ind.js, fks = ind.ks, fke = ind.ke;
  auto &mb_bcs = pack->pmb->mb_bcs;
  auto &size = pack->pmb->mb_size;
  auto u = coarse ? pack->phydro->coarse_u0 : pack->phydro->u0;
  auto *pg = pack->pgrav;
  const bool sg = pg != nullptr && pg->FieldCurrent();
  DvceArray5D<Real> acc;
  if (sg) acc = pg->acceleration;
  const Real tpg = sg ? 0.5*pg->four_pi_G : 0.0;
  par_for("sg_spitzer_bc", DevExeSpace(), 0, pack->nmb_thispack-1, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int j, int i) {
    const Real zmin = size.d_view(m).x3min, dz = (size.d_view(m).x3max-zmin)/nx3;
    Real gl = 0, gu = 0;
    if (sg) {
      const int ic = (i < ais) ? ais : ((i > aie) ? aie : i);
      const int jc = (j < ajs) ? ajs : ((j > aje) ? aje : j);
      if (coarse) {
        const int fi = fis+2*(ic-ais), fj = fjs+2*(jc-ajs);
        for (int dk=0; dk<2; ++dk) for (int dj=0; dj<2; ++dj) for (int di=0; di<2; ++di) {
          gl += acc(m,2,fks+dk,fj+dj,fi+di);
          gu += acc(m,2,fke-dk,fj+dj,fi+di);
        }
        gl *= Real(.125); gu *= Real(.125);
      } else {
        gl = acc(m,2,fks,jc,ic); gu = acc(m,2,fke,jc,ic);
      }
    }
    for (int side=0; side<2; ++side) {
      const auto face = side == 0 ? BoundaryFace::inner_x3 : BoundaryFace::outer_x3;
      if (mb_bcs.d_view(m,face) != BoundaryFlag::user) continue;
      const int kb = side == 0 ? ks : ke;
      const Real zb = zmin+(kb-ks+Real(.5))*dz;
      const Real d = u(m,IDN,kb,j,i);
      const Real v1 = u(m,IM1,kb,j,i)/d, v2 = u(m,IM2,kb,j,i)/d, v3 = u(m,IM3,kb,j,i)/d;
      const bool in = side == 0 ? v3 >= 0 : v3 <= 0;
      const Real gb = side == 0 ? gl : gu;
      for (int g=0; g<ng; ++g) {
        const int kg = side == 0 ? ks-1-g : ke+1+g;
        const Real z = zmin+(kg-ks+Real(.5))*dz, dzb = z-zb;
        Real rho;
        if (sg) {
          rho = d*exp((gb*dzb-tpg*d*dzb*dzb)/sv.cs2);
        } else {
          const Real cb = cosh((zb-sv.zc)/sv.zs), cg = cosh((z-sv.zc)/sv.zs);
          rho = d*SQR(cb/cg);
        }
        u(m,IDN,kg,j,i) = rho;
        u(m,IM1,kg,j,i) = rho*v1;
        u(m,IM2,kg,j,i) = rho*v2;
        if (sv.reflect) {   // the mirror cell, vz reversed
          const int km = side == 0 ? ks+g : ke-g;
          u(m,IM3,kg,j,i) = -rho*u(m,IM3,km,j,i)/u(m,IDN,km,j,i);
        } else {
          u(m,IM3,kg,j,i) = in ? Real(0) : rho*v3;
        }
      }
    }
  });
}
void SpitzerBCs(Mesh *pm) {SpitzerBCsApply(pm,false);}
void SpitzerBCsCoarse(Mesh *pm) {SpitzerBCsApply(pm,true);}

KOKKOS_INLINE_FUNCTION bool ExteriorX3(const BoundaryFlag f) {
  return f == BoundaryFlag::user || f == BoundaryFlag::outflow || f == BoundaryFlag::diode;
}

// the history of mode = spitzer (sums of this rank; the history sums the ranks): mass,
// sum V (rho - rho_ex)^2, sum V rho_ex^2, sum V |rho - rho_ex|, sum 1/2 rho vz^2 V, sum 1/2 rho v^2 V,
// sum mz V, the outward mass flux rho vz A at the boundary cells of the x3 faces (top, bottom), volume,
// and sum V rho of g_h^2, g_z^2, (g_z - g_ex)^2 of the self-gravity field
void SpitzerHistory(HistoryData *data, Mesh *pm) {
  auto *pack = pm->pmb_pack;
  const auto ind = pm->mb_indcs;
  const auto sv = wave;
  auto u = pack->phydro->u0;
  auto size = pack->pmb->mb_size.d_view;
  auto &mb_bcs = pack->pmb->mb_bcs;
  const int nc = ind.nx1*ind.nx2*ind.nx3, n = pack->nmb_thispack*nc;
  const int is = ind.is, js = ind.js, ks = ind.ks, ke = ind.ke;
  // the self-gravity field of the last solve (when it belongs to this mesh): mass-weighted
  // horizontal g^2, g_z^2 and (g_z - g_ex)^2, g_ex = -(2 cs^2/z0) tanh((z - zc)/z0)
  const bool field = pack->pgrav != nullptr && pack->pgrav->FieldCurrent();
  DvceArray5D<Real> acc;
  if (field) acc = pack->pgrav->acceleration;
  array_sum::GlobalSum r;
  Kokkos::parallel_reduce("sg_spitzer_hist",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(int q, array_sum::GlobalSum &sum) {
    const int m = q/nc;
    int t = q%nc;
    const int i = t%ind.nx1+is; t /= ind.nx1;
    const int j = t%ind.nx2+js; t /= ind.nx2;
    const int k = t+ks;
    const auto s = size(m);
    const Real vol = s.dx1*s.dx2*s.dx3;
    const Real zlo = s.x3min+(k-ks)*s.dx3;
    const Real ex = SpitzerAverage(sv,zlo,zlo+s.dx3);
    const Real d = u(m,IDN,k,j,i);
    const Real m1 = u(m,IM1,k,j,i), m2 = u(m,IM2,k,j,i), m3 = u(m,IM3,k,j,i);
    sum.the_array[0] += vol*d;
    sum.the_array[1] += vol*SQR(d-ex);
    sum.the_array[2] += vol*ex*ex;
    sum.the_array[3] += vol*fabs(d-ex);
    sum.the_array[4] += vol*Real(.5)*m3*m3/d;
    sum.the_array[5] += vol*Real(.5)*(m1*m1+m2*m2+m3*m3)/d;
    sum.the_array[6] += vol*m3;
    if (k == ke && ExteriorX3(mb_bcs.d_view(m,BoundaryFace::outer_x3)))
      sum.the_array[7] += s.dx1*s.dx2*m3;
    if (k == ks && ExteriorX3(mb_bcs.d_view(m,BoundaryFace::inner_x3)))
      sum.the_array[8] -= s.dx1*s.dx2*m3;
    sum.the_array[9] += vol;
    if (field) {
      const Real z = zlo+Real(.5)*s.dx3;
      const Real gex = -2*sv.cs2/sv.zs*tanh((z-sv.zc)/sv.zs);
      sum.the_array[10] += vol*d*(SQR(acc(m,0,k,j,i))+SQR(acc(m,1,k,j,i)));
      sum.the_array[11] += vol*d*SQR(acc(m,2,k,j,i));
      sum.the_array[12] += vol*d*SQR(acc(m,2,k,j,i)-gex);
    }
  }, Kokkos::Sum<array_sum::GlobalSum>(r));
  const char *labels[] = {"mass","drho2","rhoex2","drho1","ekz","ek","mom3",
                          "fz_top","fz_bot","volume","rho_gh2","rho_gz2","rho_dgz2"};
  data->nhist = 13;
  for (int q=0; q<13; ++q) {data->label[q] = labels[q]; data->hdata[q] = r.the_array[q];}
}

// final diagnostics of mode = spitzer: max |rho/rho_ex - 1|, max |vz|/cs, rms
void SpitzerDiagnostics(Mesh *pm) {
  auto *pack = pm->pmb_pack;
  auto u = Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->phydro->u0);
  const auto ind = pm->mb_indcs;
  Real dmax = 0, vmax = 0, sums[3] = {};
  for (int m=0; m<pack->nmb_thispack; ++m) {
    const auto s = pack->pmb->mb_size.h_view(m);
    const Real vol = s.dx1*s.dx2*s.dx3;
    for (int k=ind.ks; k<=ind.ke; ++k) for (int j=ind.js; j<=ind.je; ++j)
      for (int i=ind.is; i<=ind.ie; ++i) {
        const Real zlo = s.x3min+(k-ind.ks)*s.dx3;
        const Real ex = SpitzerAverage(wave,zlo,zlo+s.dx3), d = u(m,IDN,k,j,i);
        dmax = std::max(dmax,std::abs(d/ex-1));
        vmax = std::max(vmax,std::abs(u(m,IM3,k,j,i)/d));
        sums[0] += vol*SQR(d-ex); sums[1] += vol*ex*ex; sums[2] += vol*d;
      }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&dmax,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&vmax,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,sums,3,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank != 0) return;
  std::cout << std::setprecision(16) << "# SELF_GRAVITY_WAVE {\"mode\":\"spitzer\",\"time\":"
            << pm->time << ",\"cycle\":" << pm->ncycle << ",\"blocks\":" << pm->nmb_total
            << ",\"z0\":" << wave.zs << ",\"rel_density_linf\":" << dmax
            << ",\"rel_density_l2\":" << std::sqrt(sums[0]/sums[1])
            << ",\"vz_linf_over_cs\":" << vmax/std::sqrt(wave.cs2) << ",\"mass\":" << sums[2];
  if (const auto *gravity = pack->pgrav) {
    std::cout << ",\"iterations\":" << gravity->last_solve.iterations
              << ",\"residual\":" << gravity->last_solve.relative_residual
              << ",\"solves\":" << gravity->solve_count << ",\"rebuilds\":" << gravity->rebuild_count
              << ",\"kdk_reuse_hits\":" << gravity->cache_hit_count;
  }
  std::cout << "}\n";
}

// A physical, non-resetting slab control. Mean/momentum/positivity are meaningful
// after hydro/AMR evolution; the initial sinusoid is not an evolved exact solution.
void SlabEvolutionDiagnostics(Mesh *pm) {
  auto *pack=pm->pmb_pack;
  if (pack->pgrav != nullptr) pack->pgrav->Solve(pm->time);
  auto moments=WaveMoments(pm);
  auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->phydro->u0);
  HostArray5D<Real> phi,acc;
  if (pack->pgrav != nullptr) {
    phi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->pgrav->phi);
    acc=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->pgrav->acceleration);
  }
  const auto ind=pm->mb_indcs;
  const auto eos=pack->phydro->peos->eos_data;
  Real minimum[2]={std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()};
  Real maximum_density=0;
  int nonfinite=0;
  for (int m=0;m<pack->nmb_thispack;++m) for (int k=ind.ks;k<=ind.ke;++k)
    for (int j=ind.js;j<=ind.je;++j) for (int i=ind.is;i<=ind.ie;++i) {
      const Real rho=u(m,IDN,k,j,i);
      Real momentum2=0;
      for (int d=0;d<3;++d) momentum2+=SQR(u(m,IM1+d,k,j,i));
      const Real pressure=eos.is_ideal ? (eos.gamma-1)*(u(m,IEN,k,j,i)-Real(.5)*momentum2/rho)
                                         : SQR(eos.iso_cs)*rho;
      minimum[0]=std::min(minimum[0],rho); minimum[1]=std::min(minimum[1],pressure);
      maximum_density=std::max(maximum_density,rho);
      for (int q=0;q<u.extent_int(1);++q) nonfinite+=!std::isfinite(u(m,q,k,j,i));
      if (pack->pgrav != nullptr) {
        nonfinite+=!std::isfinite(phi(m,0,k,j,i));
        for (int d=0;d<3;++d) nonfinite+=!std::isfinite(acc(m,d,k,j,i));
      }
    }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,moments.the_array,20,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,minimum,2,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&maximum_density,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&nonfinite,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  if (global_variable::my_rank != 0) return;
  const auto &b=moments.the_array;
  std::cout << std::setprecision(16) << "# SELF_GRAVITY_WAVE {\"mode\":\"slab_evolve\",\"time\":"
            << pm->time << ",\"cycle\":" << pm->ncycle << ",\"blocks\":" << pm->nmb_total
            << ",\"gravity_enabled\":" << (pack->pgrav != nullptr ? "true" : "false")
            << ",\"mass\":" << b[13] << ",\"momentum1\":" << b[14]
            << ",\"momentum2\":" << b[15] << ",\"momentum3\":" << b[16]
            << ",\"volume\":" << b[17] << ",\"density_min\":" << minimum[0]
            << ",\"density_max\":" << maximum_density << ",\"pressure_min\":" << minimum[1]
            << ",\"nonfinite_values\":" << nonfinite;
  if (const auto *gravity=pack->pgrav) {
    std::cout << ",\"iterations\":" << gravity->last_solve.iterations
              << ",\"residual\":" << gravity->last_solve.relative_residual
              << ",\"solves\":" << gravity->solve_count << ",\"rebuilds\":" << gravity->rebuild_count;
  }
  std::cout << "}\n";
}

void WaveErrors(ParameterInput *, Mesh *pm) {
  auto *pack=pm->pmb_pack;
  if (wave.shear) {ShearDiagnostics(pm); return;}
  if (wave.spitzer) {SpitzerDiagnostics(pm); return;}
  if (wave.slab_evolve) {SlabEvolutionDiagnostics(pm); return;}
  if (wave.topology) InitializeKnownWave(pm,pm->time);
  pack->pgrav->Solve(pm->time);
  auto u=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->phydro->u0);
  auto phi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->pgrav->phi);
  auto acc=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->pgrav->acceleration);
  const auto ind=pm->mb_indcs;
  const Real kx=wave.kx+wave.qomega*pm->time*wave.ky;
  const Real kv[3]={kx,wave.ky,wave.kz};
  const Real k2=kx*kx+wave.ky*wave.ky+wave.kz*wave.kz;
  const Real amplitude=wave.amp*std::exp(wave.growth*pm->time);
  const Real coefficient=wave.fourpig*wave.rho0*amplitude/k2;
  // Gauge the sampled continuous reference with the same volume-weighted convention.
  Real sums[12]={}, maxima[3]={};
  for (int pass=0;pass<2;++pass) {
    for (int m=0;m<pack->nmb_thispack;++m) {
      const auto s=pack->pmb->mb_size.h_view(m);
      const Real vol=s.dx1*s.dx2*(pm->three_d ? s.dx3 : Real(1));
      bool interface_face[6]={};
      for (int d=0;d<(pm->three_d ? 3 : 2);++d) {
        for (int side=0;side<2;++side) {
          int offset[3]={}; offset[d]=2*side-1;
          for (int f1=0;f1<2;++f1) {
            for (int f2=0;f2<(pm->three_d ? 2 : 1);++f2) {
              const int slot=NeighborIndex(offset[0],offset[1],offset[2],f1,f2);
              const auto nb=pack->pmb->nghbr.h_view(m,slot);
              interface_face[2*d+side] |= nb.gid>=0 && nb.lev!=pack->pmb->mb_lev.h_view(m);
            }
          }
        }
      }
      for (int k=ind.ks;k<=ind.ke;++k) for (int j=ind.js;j<=ind.je;++j) {
        for (int i=ind.is;i<=ind.ie;++i) {
          const Real x=s.x1min+(i-ind.is+Real(.5))*s.dx1-wave.x0;
          const Real y=s.x2min+(j-ind.js+Real(.5))*s.dx2-wave.y0;
          const Real z=s.x3min+(k-ind.ks+Real(.5))*s.dx3-wave.z0;
          const Real theta=kx*x+wave.ky*y+wave.kz*z+wave.phase-wave.frequency*pm->time;
          Real exact=-coefficient*std::cos(theta);
          Real slab_shape=1, slab_sinh=0, zeta=0;
          if (wave.slab) {
            const Real half=0.5*(pm->mesh_size.x3max-pm->mesh_size.x3min);
            zeta=z-half;
            const Real knorm=std::sqrt(k2);
            const Real ep=std::exp(-knorm*(half-zeta));
            const Real em=std::exp(-knorm*(half+zeta));
            slab_shape=1-0.5*(ep+em);
            slab_sinh=0.5*(ep-em);
            exact=0.5*wave.fourpig*wave.rho0*(zeta*zeta+half*half)
                  -coefficient*slab_shape*std::cos(theta);
          }
          if (pass==0) {sums[0]+=vol; sums[1]+=vol*exact; continue;}
          const Real gauge=wave.slab ? 0 : sums[1]/sums[0];
          const Real error=phi(m,0,k,j,i)-(exact-gauge);
          Real force_error=0, force_exact=0;
          for (int d=0;d<(pm->three_d ? 3 : 2);++d) {
            Real g=-coefficient*kv[d]*slab_shape*std::sin(theta);
            if (wave.slab && d==2) {
              g=-wave.fourpig*wave.rho0*zeta
                -coefficient*std::sqrt(k2)*slab_sinh*std::cos(theta);
            }
            force_error+=SQR(acc(m,d,k,j,i)-g); force_exact+=g*g;
          }
          const bool interface=(i==ind.is&&interface_face[0])||(i==ind.ie&&interface_face[1])||
              (j==ind.js&&interface_face[2])||(j==ind.je&&interface_face[3])||
              (pm->three_d&&((k==ind.ks&&interface_face[4])||(k==ind.ke&&interface_face[5])));
          sums[2]+=vol*error*error; sums[3]+=vol*force_error;
          sums[4]+=vol*SQR(exact-gauge); sums[5]+=vol*force_exact;
          if (interface) {sums[6]+=vol; sums[7]+=vol*error*error; sums[8]+=vol*force_error;}
          const Real average=Sinc(Real(.5)*kx*s.dx1)*Sinc(Real(.5)*wave.ky*s.dx2)*
                             Sinc(Real(.5)*wave.kz*s.dx3);
          sums[9]+=vol*SQR(u(m,IDN,k,j,i)-wave.rho0*(1+amplitude*average*std::cos(theta)));
          maxima[0]=std::max(maxima[0],std::abs(error));
          maxima[1]=std::max(maxima[1],std::sqrt(force_error));
          if (interface) maxima[2]=std::max(maxima[2],std::sqrt(force_error));
        }
      }
    }
#if MPI_PARALLEL_ENABLED
    if (pass==0) MPI_Allreduce(MPI_IN_PLACE,sums,2,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,sums+2,10,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,maxima,3,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  auto moments=WaveMoments(pm);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,moments.the_array,7,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  const auto &b=moments.the_array;
  const Real determinant=b[4]*b[5]-b[6]*b[6];
  const Real cosine=(b[0]*b[5]-b[1]*b[6])/determinant;
  const Real sine=(b[1]*b[4]-b[0]*b[6])/determinant;
  if (global_variable::my_rank==0) {
    std::cout << std::setprecision(16) << "# SELF_GRAVITY_WAVE {\"mode\":\""
      << (wave.jeans ? "jeans" : wave.topology ? "topology_diagnostic" : wave.slab ? "slab" : "poisson")
      << "\",\"time\":" << pm->time << ",\"cycle\":" << pm->ncycle
      << ",\"blocks\":" << pm->nmb_total << ",\"cells\":" << pm->nmb_total*ind.nx1*ind.nx2*ind.nx3
      << ",\"phi_l2\":" << std::sqrt(sums[2]/sums[0])
      << ",\"phi_relative_l2\":" << std::sqrt(sums[2]/sums[4])
      << ",\"phi_linf\":" << maxima[0] << ",\"acceleration_l2\":" << std::sqrt(sums[3]/sums[0])
      << ",\"acceleration_relative_l2\":" << std::sqrt(sums[3]/sums[5])
      << ",\"acceleration_linf\":" << maxima[1]
      << ",\"interface_volume\":" << sums[6]
      << ",\"interface_phi_l2\":" << (sums[6]>0 ? std::sqrt(sums[7]/sums[6]) : 0)
      << ",\"interface_acceleration_l2\":" << (sums[6]>0 ? std::sqrt(sums[8]/sums[6]) : 0)
      << ",\"interface_acceleration_linf\":" << maxima[2]
      << ",\"density_l2\":" << std::sqrt(sums[9]/sums[0])
      << ",\"density_mode_cos\":" << cosine << ",\"density_mode_sin\":" << sine
      << ",\"expected_mode_cos\":" << amplitude*std::cos(wave.frequency*pm->time)
      << ",\"expected_mode_sin\":" << amplitude*std::sin(wave.frequency*pm->time)
      << ",\"growth\":" << wave.growth << ",\"frequency\":" << wave.frequency
      << ",\"iterations\":" << pack->pgrav->last_solve.iterations
      << ",\"residual\":" << pack->pgrav->last_solve.relative_residual
      << ",\"solves\":" << pack->pgrav->solve_count
      << ",\"rebuilds\":" << pack->pgrav->rebuild_count << "}\n";
  }
}

void WaveRefinement(MeshBlockPack *pack) {
  auto *pm=pack->pmesh;
  // This explicit topology-only mode restores a known analytic source, so source
  // interpolation errors and physical hydro evolution cannot conceal stale geometry.
  if (wave.topology) WaveErrors(nullptr,pm);
  auto &flags=pm->pmr->refine_flag;
  const int first=pm->gids_eachrank[global_variable::my_rank];
  int requests[2]={};
  for (int m=0;m<pack->nmb_thispack;++m) {
    const auto s=pack->pmb->mb_size.h_view(m);
    const Real center=(s.x1min+s.x1max)*Real(.5);
    const Real center2=(s.x2min+s.x2max)*Real(.5);
    const Real center3=(s.x3min+s.x3max)*Real(.5);
    const int desired=(pm->time>=wave.refine_start && pm->time<wave.refine_end &&
                       center>wave.band_lo && center<wave.band_hi &&
                       center2>wave.band2_lo && center2<wave.band2_hi &&
                       center3>wave.band3_lo && center3<wave.band3_hi) ? 1 : 0;
    const int level=pm->lloc_eachmb[first+m].level-pm->root_level;
    flags.h_view(first+m)=desired>level ? 1 : desired<level ? -1 : 0;
    if (desired>level) ++requests[0];
    if (desired<level) ++requests[1];
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,requests,2,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
#endif
  if (wave.scheduled_amr && global_variable::my_rank==0 && (requests[0] || requests[1])) {
    std::cout << std::setprecision(16) << "# SELF_GRAVITY_AMR {\"time\":" << pm->time
              << ",\"cycle\":" << pm->ncycle << ",\"blocks_before\":" << pm->nmb_total
              << ",\"requested_refine\":" << requests[0]
              << ",\"requested_derefine\":" << requests[1] << "}\n";
  }
  flags.template modify<HostMemSpace>();
  flags.template sync<DevExeSpace>();
}
}  // namespace

void ProblemGenerator::SelfGravityWave(ParameterInput *pin, const bool restart) {
  auto *pm=pmy_mesh_;
  auto *pack=pm->pmb_pack;
  if (pack->phydro==nullptr) WaveFail("requires hydro");
  const auto mode=pin->GetOrAddString("problem","mode","poisson");
  wave.jeans=mode=="jeans"; wave.topology=mode=="topology";
  wave.slab_evolve=mode=="slab_evolve";
  wave.slab=mode=="slab" || wave.slab_evolve; wave.shear=mode=="shear";
  wave.spitzer=mode=="spitzer";
  if (wave.spitzer) {
    // the isothermal self-gravitating slab in equilibrium (no rotation), isolated z gravity;
    // hydro x3 faces user (hydrostatic ghosts, SpitzerBCs) or outflow; optional scheduled AMR band
    if (pack->pgrav==nullptr || !pm->three_d || pack->phydro->peos->eos_data.is_ideal ||
        pack->phydro->psbox_u!=nullptr ||
        pin->GetOrAddString("gravity","vertical_bc","periodic")!="isolated" ||
        pin->GetString("time","evolution")!="dynamic") {
      WaveFail("spitzer mode requires 3D isothermal hydro without rotation, dynamic evolution, "
               "and gravity with vertical_bc=isolated");
    }
    wave.rho0=pin->GetOrAddReal("problem","rho0",1);
    wave.fourpig=pack->pgrav->four_pi_G;
    wave.cs2=SQR(pack->phydro->peos->eos_data.iso_cs);
    // hidden test key (only looked up): reflecting ghost vz instead of the li2018 outflow rule
    wave.reflect=pin->DoesParameterExist("problem","test_spitzer_reflect") &&
                 pin->GetBoolean("problem","test_spitzer_reflect");
    wave.zs=std::sqrt(2*wave.cs2/(wave.fourpig*wave.rho0));   // cs/sqrt(2 pi G rho0)
    wave.zc=Real(.5)*(pm->mesh_size.x3min+pm->mesh_size.x3max);
    wave.x0=pm->mesh_size.x1min; wave.y0=pm->mesh_size.x2min; wave.z0=pm->mesh_size.x3min;
    wave.kx=wave.ky=wave.kz=wave.qomega=wave.frequency=wave.growth=wave.amp=0;
    wave.scheduled_amr=pin->GetOrAddBoolean("problem","scheduled_amr",false);
    wave.refine_start=pin->GetOrAddReal("problem","refine_start",Real(.02));
    wave.refine_end=pin->GetOrAddReal("problem","refine_end",Real(.06));
    wave.band_lo=pin->GetOrAddReal("problem","refine_x1min",pm->mesh_size.x1min);
    wave.band_hi=pin->GetOrAddReal("problem","refine_x1max",pm->mesh_size.x1max);
    wave.band2_lo=pin->GetOrAddReal("problem","refine_x2min",pm->mesh_size.x2min);
    wave.band2_hi=pin->GetOrAddReal("problem","refine_x2max",pm->mesh_size.x2max);
    wave.band3_lo=pin->GetOrAddReal("problem","refine_x3min",wave.zc-wave.zs);
    wave.band3_hi=pin->GetOrAddReal("problem","refine_x3max",wave.zc+wave.zs);
    pgen_final_func=WaveErrors;
    user_hist_func=SpitzerHistory;
    if (user_bcs) {user_bcs_func=SpitzerBCs; user_bcs_coarse_func=SpitzerBCsCoarse;}
    if (wave.scheduled_amr) {
      if (!pm->adaptive) WaveFail("refinement schedule requires adaptive mesh");
      user_ref_func=WaveRefinement;
    }
    if (global_variable::my_rank==0) {
      std::cout << std::setprecision(10) << "# spitzer slab: rho0 = " << wave.rho0 << ", z0 = "
                << wave.zs << ", center " << wave.zc << ", box half height "
                << Real(.5)*(pm->mesh_size.x3max-pm->mesh_size.x3min) << " = "
                << Real(.5)*(pm->mesh_size.x3max-pm->mesh_size.x3min)/wave.zs << " z0\n";
    }
    if (!restart) SpitzerInitialize(pm);
    return;
  }
  if (pack->pgrav==nullptr && !wave.shear && !wave.slab_evolve) {
    WaveFail("analytic/static gravity controls require gravity/enabled=true");
  }
  wave.scheduled_amr=pin->GetOrAddBoolean("problem","scheduled_amr",false);
  if (!wave.jeans && !wave.shear && !wave.topology && !wave.slab && mode!="poisson") WaveFail("unknown mode");
  const bool isolated=pin->GetOrAddString("gravity","vertical_bc","periodic")=="isolated";
  if (wave.slab != isolated || (wave.slab && !pm->three_d)) {
    WaveFail("use 3D mode=slab or slab_evolve for isolated-z gravity; other modes require periodic gravity");
  }
  if (wave.jeans && pack->phydro->psbox_u!=nullptr) {
    WaveFail("Jeans mode requires hydro without rotation/shear");
  }
  if (wave.shear && (!pm->three_d || pack->phydro->peos->eos_data.is_ideal ||
      pack->phydro->psbox_u==nullptr || pack->phydro->psbox_u->is_stratified)) {
    WaveFail("shear mode requires 3D unstratified isothermal shearing-box hydro");
  }
  if (!wave.jeans && !wave.shear && !wave.topology && !wave.slab_evolve
      && pin->GetString("time","evolution")!="static") {
    WaveFail("Poisson field check requires time/evolution=static");
  }
  wave.rho0=pin->GetOrAddReal("problem","rho0",1);
  wave.amp=pin->GetOrAddReal("problem","amp",wave.jeans || wave.shear ? Real(1.e-6) : Real(.1));
  wave.phase=pin->GetOrAddReal("problem","phase",Real(.37));
  wave.pressure=pin->GetOrAddReal("problem","pressure",1);
  const int nx=pin->GetOrAddInteger("problem","nwx",1);
  const int ny=pin->GetOrAddInteger("problem","nwy",wave.jeans ? 0 : 1);
  const int nz=pin->GetOrAddInteger("problem","nwz",pm->three_d && !wave.jeans && !wave.slab ? 1 : 0);
  if (wave.slab && nz!=0) WaveFail("analytic slab mode requires nwz=0");
  if ((!pm->three_d && nz!=0) || (nx==0&&ny==0&&nz==0) || wave.rho0<=0 ||
      wave.amp<0 || (wave.amp==0 && !wave.slab) || wave.amp>=1) {
    WaveFail("invalid density, amplitude, or wavenumbers");
  }
  wave.x0=pm->mesh_size.x1min; wave.y0=pm->mesh_size.x2min; wave.z0=pm->mesh_size.x3min;
  wave.kx=2*M_PI*nx/(pm->mesh_size.x1max-wave.x0);
  wave.ky=2*M_PI*ny/(pm->mesh_size.x2max-wave.y0);
  wave.kz=2*M_PI*nz/(pm->mesh_size.x3max-wave.z0);
  wave.qomega=pack->phydro->psbox_u==nullptr ? 0 :
      pack->phydro->psbox_u->qshear*pack->phydro->psbox_u->omega0;
  wave.fourpig=pack->pgrav==nullptr ? 0 : pack->pgrav->four_pi_G;
  wave.frequency=wave.growth=0;
  if (wave.jeans) {
    const auto eos=pack->phydro->peos->eos_data;
    if (eos.is_ideal && (wave.pressure<=0 || eos.gamma*wave.amp>=1)) {
      WaveFail("ideal Jeans mode requires positive pressure and gamma*amp<1");
    }
    const Real cs2=eos.is_ideal ? eos.gamma*wave.pressure/wave.rho0 : SQR(eos.iso_cs);
    const Real dispersion=cs2*(SQR(wave.kx)+SQR(wave.ky)+SQR(wave.kz))-wave.fourpig*wave.rho0;
    if (dispersion>0) wave.frequency=std::sqrt(dispersion);
    else wave.growth=std::sqrt(-dispersion);
  }
  wave.refine_start=pin->GetOrAddReal("problem","refine_start",Real(.02));
  wave.refine_end=pin->GetOrAddReal("problem","refine_end",Real(.06));
  wave.band_lo=pin->GetOrAddReal("problem","refine_x1min",wave.x0+Real(.25)*(pm->mesh_size.x1max-wave.x0));
  wave.band_hi=pin->GetOrAddReal("problem","refine_x1max",wave.x0+Real(.75)*(pm->mesh_size.x1max-wave.x0));
  wave.band2_lo=pin->GetOrAddReal("problem","refine_x2min",wave.y0);
  wave.band2_hi=pin->GetOrAddReal("problem","refine_x2max",pm->mesh_size.x2max);
  wave.band3_lo=pin->GetOrAddReal("problem","refine_x3min",wave.z0);
  wave.band3_hi=pin->GetOrAddReal("problem","refine_x3max",pm->mesh_size.x3max);
  pgen_final_func=WaveErrors;
  user_hist_func=WaveHistory;
  if (wave.scheduled_amr && !wave.jeans && !wave.shear && !wave.slab_evolve) {
    WaveFail("scheduled_amr is supported only for physical Jeans/shear/slab_evolve modes");
  }
  if (wave.topology || wave.scheduled_amr) {
    if (!pm->adaptive) WaveFail("refinement schedule requires adaptive mesh");
    user_ref_func=WaveRefinement;
  }
  if (restart) return;  // diagnostics reenrolled, restored fields/time are retained
  if (pin->DoesParameterExist("problem","initial_time")) {
    if (wave.jeans || wave.shear) WaveFail("initial_time is a Poisson/topology diagnostic option");
    pm->time=pin->GetReal("problem","initial_time");
  }
  InitializeKnownWave(pm,pm->time);
}
