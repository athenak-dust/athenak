// Physical growing eigenmode of the full two-fluid gas+dust self-gravity system.
// Initialization and diagnostics only; no trajectory or later source is imposed.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "dust/dust.hpp"
#include "dust/dust_species.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "gravity/self_gravity.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {
struct JeansMode {
  Real rho, eps, cs, tau, kappa, k, phase, sigma, dg, ug, dd, vd;
};
JeansMode mode;

void Require(bool good,const char *message) {
  int ok=good;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&ok,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
#endif
  if (ok) return;
  std::cerr<<"dust_jeans: "<<message<<'\n';
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
void GlobalSum(Real *values,int count) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,values,count,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
}
Real Sinc(Real x) {return std::abs(x)<1.e-12 ? Real(1) : std::sin(x)/x;}
std::array<Real,2> Projection(Real cc,Real cs,Real ss,Real bc,Real bs) {
  const Real determinant=cc*ss-cs*cs;
  return {(ss*bc-cs*bs)/determinant,(cc*bs-cs*bc)/determinant};
}

void JeansHistory(HistoryData *history,Mesh *pm) {
  auto *pack=pm->pmb_pack;
  auto *particles=pack->ppart;
  const auto ind=pm->mb_indcs;
  DvceArray5D<Real> mapped;
  pack->pdust->DepositGravityDensity(pm->time,mapped);
  auto rd=Kokkos::create_mirror_view_and_copy(HostMemSpace(),mapped);
  auto gas=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->phydro->u0);
  auto pr=Kokkos::create_mirror_view_and_copy(HostMemSpace(),particles->prtcl_rdata);
  auto acceleration=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->pgrav->acceleration);
  DvceArray2D<Real> particle_acceleration;
  pack->pdust->GatherGravityAcceleration(pm->time,pack->pgrav->acceleration,particle_acceleration);
  auto pa=Kokkos::create_mirror_view_and_copy(HostMemSpace(),particle_acceleration);
  // Whole-leaf normal equations, norm, force and harmonic moments. No x-column
  // averaging: the residual includes all transverse cell-to-cell structure.
  Real a[33]={};
  Real rho_min=std::numeric_limits<Real>::max(),dust_min=rho_min;
  bool finite=true;
  for (int b=0;b<pack->nmb_thispack;++b) {
    const auto s=pack->pmb->mb_size.h_view(b);
    const Real volume=s.dx1*s.dx2*s.dx3;
    for (int k=ind.ks;k<=ind.ke;++k) for (int j=ind.js;j<=ind.je;++j)
      for (int i=ind.is;i<=ind.ie;++i) {
        const Real theta=mode.k*(s.x1min+(i-ind.is+.5)*s.dx1)+mode.phase;
        const Real c=Sinc(.5*mode.k*s.dx1)*std::cos(theta);
        const Real sn=Sinc(.5*mode.k*s.dx1)*std::sin(theta);
        const Real c2=Sinc(mode.k*s.dx1)*std::cos(2*theta);
        const Real s2=Sinc(mode.k*s.dx1)*std::sin(2*theta);
        const Real rho=gas(b,IDN,k,j,i),dust=rd(b,0,k,j,i);
        const Real drg=rho-mode.rho,drd=dust-mode.eps*mode.rho;
        const Real momentum=gas(b,IM1,k,j,i),g=acceleration(b,0,k,j,i);
        rho_min=std::min(rho_min,rho);dust_min=std::min(dust_min,dust);
        finite=finite && std::isfinite(rho) && std::isfinite(dust) &&
            std::isfinite(momentum) && std::isfinite(g) && rho>0 && dust>=0;
        a[0]+=volume; a[1]+=volume*c*c; a[2]+=volume*c*sn; a[3]+=volume*sn*sn;
        a[4]+=volume*drg*c; a[5]+=volume*drg*sn;
        a[6]+=volume*momentum*c/mode.rho; a[7]+=volume*momentum*sn/mode.rho;
        a[8]+=volume*drd*c; a[9]+=volume*drd*sn;
        a[10]+=volume*drg*drg; a[11]+=volume*drd*drd;
        a[12]+=volume*rho; a[19]+=volume*rho*g;
        a[21]+=volume*drg*c2; a[22]+=volume*drg*s2;
        a[23]+=volume*drd*c2; a[24]+=volume*drd*s2;
        a[25]+=volume*c2*c2; a[26]+=volume*c2*s2; a[27]+=volume*s2*s2;
        a[28]+=volume*g*c; a[29]+=volume*g*sn;
        a[30]+=volume*momentum;
      }
  }
  for (int p=0;p<particles->nprtcl_thispack;++p) {
    const Real mass=pr(IPM,p),x=pr(IPX,p),v=pr(IPVX,p);
    const Real theta=mode.k*x+mode.phase;
    finite=finite && std::isfinite(mass) && std::isfinite(x) && std::isfinite(v) &&
        std::isfinite(pa(0,p)) && mass>=0;
    a[13]+=mass; a[15]+=mass*std::cos(theta); a[16]+=mass*std::sin(theta);
    a[17]+=mass*v*std::cos(theta); a[18]+=mass*v*std::sin(theta);
    a[20]+=mass*pa(0,p); a[31]+=mass*v; a[32]+=1;
  }
  Require(finite,"nonfinite/negative evolved state");
  GlobalSum(a,33);
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&rho_min,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&dust_min,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
#endif
  const auto gd=Projection(a[1],a[2],a[3],a[4],a[5]);
  const auto gu=Projection(a[1],a[2],a[3],a[6],a[7]);
  const auto dd=Projection(a[1],a[2],a[3],a[8],a[9]);
  const auto gg=Projection(a[1],a[2],a[3],a[28],a[29]);
  const auto g2=Projection(a[25],a[26],a[27],a[21],a[22]);
  const auto d2=Projection(a[25],a[26],a[27],a[23],a[24]);
  const Real gas_res=std::sqrt(std::max(Real(0),(a[10]-gd[0]*a[4]-gd[1]*a[5])/a[0]));
  const Real dust_res=std::sqrt(std::max(Real(0),(a[11]-dd[0]*a[8]-dd[1]*a[9])/a[0]));
  const Real pc=2*a[15]/a[0],ps=2*a[16]/a[0];
  const Real vc=2*a[17]/(a[0]*mode.eps*mode.rho),vs=2*a[18]/(a[0]*mode.eps*mode.rho);
  // The history machinery reduces these again. Only rank0 contributes the
  // already-global diagnostic values, avoiding any rank-dependent normalization.
  const char *labels[20]={"gcos","gsin","ucos","usin","dcos","dsin","pcos","psin",
      "vcos","vsin","gmass","dmass","total_px","gforce","dforce","gres","dres",
      "gravity_cos","gravity_sin","particles"};
  const Real values[20]={gd[0],gd[1],gu[0],gu[1],dd[0],dd[1],pc,ps,vc,vs,a[12],a[13],
      a[30]+a[31],a[19],a[20],gas_res,dust_res,gg[0],gg[1],a[32]};
  history->nhist=20;
  for (int n=0;n<20;++n) {
    history->label[n]=labels[n];
    history->hdata[n]=global_variable::my_rank==0 ? values[n] : 0;
  }
  if (global_variable::my_rank==0) {
    auto pair=[](const std::array<Real,2> &p) {std::cout<<'['<<p[0]<<','<<p[1]<<']';};
    std::cout<<std::setprecision(17)<<"# DUST_JEANS_MODES {\"time\":"<<pm->time
      <<",\"cycle\":"<<pm->ncycle<<",\"gas_density\":";pair(gd);
    std::cout<<",\"gas_velocity\":";pair(gu);
    std::cout<<",\"dust_mapped\":";pair(dd);
    std::cout<<",\"particle_density\":["<<pc<<','<<ps<<"],\"particle_velocity\":["<<vc<<','<<vs<<']';
    std::cout<<",\"gravity_g1\":";pair(gg);
    std::cout<<",\"gas_k2\":";pair(g2);std::cout<<",\"dust_k2\":";pair(d2);
    std::cout<<",\"gas_residual_l2\":"<<gas_res<<",\"dust_residual_l2\":"<<dust_res
      <<",\"gas_mass\":"<<a[12]<<",\"dust_mass\":"<<a[13]<<",\"total_px\":"<<a[30]+a[31]
      <<",\"gas_force\":"<<a[19]<<",\"dust_force\":"<<a[20]
      <<",\"rho_min\":"<<rho_min<<",\"dust_min\":"<<dust_min
      <<",\"volume\":"<<a[0]<<",\"particles\":"<<a[32]
      <<",\"source_mean\":"<<pack->pgrav->last_source_mean
      <<",\"gravity_residual\":"<<pack->pgrav->last_solve.relative_residual
      <<",\"gram\":["<<a[1]<<','<<a[2]<<','<<a[3]<<"]}\n";
  }
}
void JeansFinal(ParameterInput *,Mesh *pm) {
  HistoryData data(PhysicsModule::UserDefined);JeansHistory(&data,pm);
}
}  // namespace

void ProblemGenerator::DustJeans(ParameterInput *pin,const bool restart) {
  dust::RequireLegacySpeciesInit(pin, "dust_jeans");
  auto *pack=pmy_mesh_->pmb_pack;
  Require(pack->pdust && pack->ppart && pack->phydro && pack->pgrav &&
      pack->pdust->nspecies==1 && !pin->DoesBlockExist("shearing_box") &&
      !pack->phydro->peos->eos_data.is_ideal,"requires periodic isothermal gas+one dust species+SG");
  mode.rho=pin->GetReal("problem","rho0");mode.eps=pin->GetReal("problem","eps_1");
  mode.tau=pin->GetReal("dust","taus_1");mode.cs=pack->phydro->peos->eos_data.iso_cs;
  mode.kappa=pin->GetReal("gravity","four_pi_G");
  mode.k=2*M_PI/(pmy_mesh_->mesh_size.x1max-pmy_mesh_->mesh_size.x1min);
  mode.phase=pin->GetReal("problem","jeans_phase");mode.sigma=pin->GetReal("problem","jeans_sigma");
  mode.dg=pin->GetReal("problem","jeans_dg");mode.ug=pin->GetReal("problem","jeans_ug");
  mode.dd=pin->GetReal("problem","jeans_dd");mode.vd=pin->GetReal("problem","jeans_vd");
  const Real lhs[4]={mode.k*mode.rho*mode.ug,
      (mode.kappa/mode.k-mode.k*mode.cs*mode.cs/mode.rho)*mode.dg-
        mode.eps/mode.tau*mode.ug+mode.kappa/mode.k*mode.dd+mode.eps/mode.tau*mode.vd,
      mode.k*mode.eps*mode.rho*mode.vd,
      mode.kappa/mode.k*(mode.dg+mode.dd)+(mode.ug-mode.vd)/mode.tau};
  const Real state[4]={mode.dg,mode.ug,mode.dd,mode.vd};
  Real error=0,scale=0;
  for (int q=0;q<4;++q) {error=std::max(error,std::abs(lhs[q]-mode.sigma*state[q]));scale=std::max(scale,std::abs(mode.sigma*state[q]));}
  Require(mode.sigma>0 && error<1.e-11*std::max(scale,Real(1.e-30)),"input vector is not the full-SG eigenpair");
  // DustNSH supplies the common quiet root lattice, physical masses and IPTS.
  // Imprint the mode BEFORE any representation split, so mixed RC begins from
  // precisely the same physical parent samples as its own coarse uniform run.
  const bool splitting=pack->pdust->cloud_split_enable;
  pack->pdust->cloud_split_enable=false;
  DustNSH(pin,restart);
  pack->pdust->cloud_split_enable=splitting;
  user_hist_func=JeansHistory;pgen_final_func=JeansFinal;user_work_in_loop_func=nullptr;
  if (restart) return;
  const auto ind=pmy_mesh_->mb_indcs;
  auto gas=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->phydro->u0);
  for (int b=0;b<pack->nmb_thispack;++b) {
    const auto s=pack->pmb->mb_size.h_view(b);
    for (int k=ind.ks;k<=ind.ke;++k) for (int j=ind.js;j<=ind.je;++j)
      for (int i=ind.is;i<=ind.ie;++i) {
        const Real theta=mode.k*(s.x1min+(i-ind.is+.5)*s.dx1)+mode.phase;
        const Real c=Sinc(.5*mode.k*s.dx1)*std::cos(theta);
        const Real sn=Sinc(.5*mode.k*s.dx1)*std::sin(theta);
        const Real sn2=Sinc(mode.k*s.dx1)*std::sin(2*theta);
        gas(b,IDN,k,j,i)=mode.rho+mode.dg*c;
        gas(b,IM1,k,j,i)=-mode.rho*mode.ug*sn-.5*mode.dg*mode.ug*sn2;
        gas(b,IM2,k,j,i)=gas(b,IM3,k,j,i)=0;
      }
  }
  Kokkos::deep_copy(pack->phydro->u0,gas);
  auto pr=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->ppart->prtcl_rdata);
  auto pi=Kokkos::create_mirror_view_and_copy(HostMemSpace(),pack->ppart->prtcl_idata);
  std::ofstream audit("jeans_initial.r"+std::to_string(global_variable::my_rank)+".csv");
  audit<<std::setprecision(17)<<"tag,x0,y0,z0,mass,x,vx\n";
  for (int p=0;p<pack->ppart->nprtcl_thispack;++p) {
    const Real x0=pr(IPX,p),theta=mode.k*x0+mode.phase;
    const Real x=x0-mode.dd/(mode.k*mode.eps*mode.rho)*std::sin(theta);
    const Real vx=-mode.vd*std::sin(theta);
    audit<<pi(PTAG,p)<<','<<x0<<','<<pr(IPY,p)<<','<<pr(IPZ,p)<<','<<pr(IPM,p)<<','<<x<<','<<vx<<'\n';
    pr(IPX,p)=pr(IPX1,p)=x;pr(IPVX,p)=pr(IPVX1,p)=vx;
    pr(IPVY,p)=pr(IPVY1,p)=pr(IPVZ,p)=pr(IPVZ1,p)=0;
  }
  audit.close();Kokkos::deep_copy(pack->ppart->prtcl_rdata,pr);
  if (pack->pdust->Clouds3D()) {
    pack->pdust->ResetPGIDFromPositions("Jeans eigenmode imprint");
    pack->pdust->SplitToLeaves();
  }
  if (global_variable::my_rank==0)
    std::cout<<std::setprecision(17)<<"# DUST_JEANS_SETUP {\"sigma\":"<<mode.sigma
      <<",\"kappa\":"<<mode.kappa<<",\"k\":"<<mode.k<<",\"rho0\":"<<mode.rho
      <<",\"eps\":"<<mode.eps<<",\"tau\":"<<mode.tau<<",\"phase\":"<<mode.phase
      <<",\"dg\":"<<mode.dg<<",\"ug\":"<<mode.ug<<",\"dd\":"<<mode.dd<<",\"vd\":"<<mode.vd
      <<",\"input_eigenpair_max_residual\":"<<error<<"}\n";
}
