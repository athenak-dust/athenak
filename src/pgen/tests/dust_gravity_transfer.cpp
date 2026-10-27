// Native application checks of the gravity adapter, not a second implementation
// of its kernels. Initialize with dust_nsh/damping fixtures; test at final time.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "athena.hpp"
#include "dust/dust.hpp"
#include "globals.hpp"
#include "gravity/self_gravity.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "pgen/pgen.hpp"

namespace {
void Require(bool ok,const char *message) {
  int good=ok;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&good,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
#endif
  if (good) return;
  std::cerr<<"### DUST_GRAVITY_TRANSFER failed: "<<message<<std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
void Sum(Real *x,int n) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,x,n,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
}
Real Maximum(Real x) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&x,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  return x;
}

void CheckTransfer(ParameterInput *pin,Mesh *pm) {
  auto *pack=pm->pmb_pack;
  auto *dust=pack->pdust;
  auto *particles=pack->ppart;
  Require(dust!=nullptr && particles!=nullptr && dust->GravityTransferSupported(),
          "unsupported fixture");
  const auto ind=pm->mb_indcs;
  const int nmb=pack->nmb_thispack,npart=particles->nprtcl_thispack;
  const Real time=pin->GetOrAddReal("problem","gravity_test_time",pm->time);
  const bool shape_sweep=pin->GetOrAddBoolean("problem","gravity_test_shape_sweep",false);
  const bool clouds=dust->Clouds3D();
  Require(!shape_sweep || clouds,"shape sweep requires refinable clouds, not point TSC");
  auto original_codes=Kokkos::create_mirror(particles->prtcl_idata);
  Kokkos::deep_copy(original_codes,particles->prtcl_idata);
  if (shape_sweep) {
    // An explicit frozen-state kernel test, not a physical split operation:
    // cover all 64 combinations of valid directional lobe codes, at widths
    // no larger than the original root-cloud support. Restore exact codes below.
    auto test_codes=Kokkos::create_mirror(particles->prtcl_idata);
    Kokkos::deep_copy(test_codes,particles->prtcl_idata);
    for (int p=0;p<npart;++p) {
      using T=::SMRCompositeTransfer;
      const unsigned int pattern=static_cast<unsigned int>(GetTag(test_codes,p)%64);
      const int level=std::max(1,T::CloudLevel(test_codes(PLEV,p)));
      test_codes(PLEV,p)=level | ((pattern&3)<<T::kCloudLobeShift1) |
          (((pattern>>2)&3)<<T::kCloudLobeShift2) |
          (((pattern>>4)&3)<<T::kCloudLobeShift3);
    }
    Kokkos::deep_copy(particles->prtcl_idata,test_codes);
  }
  DvceArray5D<Real> density;
  dust->DepositGravityDensity(time,density);
  auto rho=Kokkos::create_mirror_view_and_copy(HostMemSpace(),density);
  // A true copy is required: mirror_view_and_copy may alias the original on
  // Serial/OpenMP, which would invalidate the before/after kick comparison.
  auto pr=Kokkos::create_mirror(particles->prtcl_rdata);
  Kokkos::deep_copy(pr,particles->prtcl_rdata);
  const auto original_particles=pr;
  auto codes=Kokkos::create_mirror_view_and_copy(HostMemSpace(),particles->prtcl_idata);
  Real assignment_wall_particles=0;
  if (dust->PointGravityTransferSupported() &&
      pm->mesh_bcs[BoundaryFace::inner_x3]!=BoundaryFlag::periodic) {
    for (int p=0;p<npart;++p) if (pr(IPM,p)>0) {
      const auto s=pack->pmb->mb_size.h_view(codes(PGID,p)-pack->gids);
      if (pr(IPZ,p)<pm->mesh_size.x3min+s.dx3 ||
          pr(IPZ,p)>pm->mesh_size.x3max-s.dx3) ++assignment_wall_particles;
    }
    Sum(&assignment_wall_particles,1);
  }
  Real shapes[4]={0,0,0,0};
  for (int p=0;p<npart;++p) if (clouds && pr(IPM,p)>0) {
    const int code=codes(PLEV,p);
    const bool l1=::SMRCompositeTransfer::CloudLobe1(code)!=0;
    const bool l2=::SMRCompositeTransfer::CloudLobe2(code)!=0;
    const bool l3=::SMRCompositeTransfer::CloudLobe3(code)!=0;
    shapes[0]+=!(l1||l2||l3); shapes[1]+=l1; shapes[2]+=l2; shapes[3]+=l3;
  }
  Sum(shapes,4);
  Real mass[2]={0,0};
  for (int p=0;p<npart;++p) mass[0]+=pr(IPM,p);
  DvceArray5D<Real> field("gravity_test_field",nmb,3,density.extent(2),
                         density.extent(3),density.extent(4));
  auto fh=Kokkos::create_mirror_view(field);
  for (int b=0;b<nmb;++b) {
    const auto s=pack->pmb->mb_size.h_view(b);
    const Real volume=s.dx1*s.dx2*s.dx3;
    for (int k=ind.ks;k<=ind.ke;++k) for (int j=ind.js;j<=ind.je;++j)
      for (int i=ind.is;i<=ind.ie;++i) {
        mass[1]+=volume*rho(b,0,k,j,i);
        const Real x=s.x1min+(i-ind.is+.5)*s.dx1;
        const Real y=s.x2min+(j-ind.js+.5)*s.dx2;
        const Real z=s.x3min+(k-ind.ks+.5)*s.dx3;
        fh(b,0,k,j,i)=.3+std::sin(11*x+5*y-3*z);
        fh(b,1,k,j,i)=-.7+std::cos(7*y+13*z);
        fh(b,2,k,j,i)=.2+std::sin(17*z+19*x);
      }
  }
  Sum(mass,2);
  const Real mass_error=std::abs(mass[1]-mass[0])/(1+std::abs(mass[0]));
  Require(mass_error<1.e-10,"physical particle mass differs from owned-leaf mass");
  Kokkos::deep_copy(field,fh);
  DvceArray2D<Real> gathered;
  dust->GatherGravityAcceleration(time,field,gathered);
  auto gp=Kokkos::create_mirror_view_and_copy(HostMemSpace(),gathered);
  Real inner[6]={};
  for (int p=0;p<npart;++p) for (int d=0;d<3;++d) inner[d]+=pr(IPM,p)*gp(d,p);
  for (int b=0;b<nmb;++b) {
    const auto s=pack->pmb->mb_size.h_view(b);
    for (int k=ind.ks;k<=ind.ke;++k) for (int j=ind.js;j<=ind.je;++j)
      for (int i=ind.is;i<=ind.ie;++i) for (int d=0;d<3;++d)
        inner[3+d]+=s.dx1*s.dx2*s.dx3*rho(b,0,k,j,i)*fh(b,d,k,j,i);
  }
  Sum(inner,6);
  Real adjoint_error=0;
  for (int d=0;d<3;++d)
    adjoint_error=std::max(adjoint_error,std::abs(inner[d]-inner[3+d])/(1+std::abs(inner[d])));
  Require(adjoint_error<1.e-10,"mass-weighted deposition/gather identity");

  // Uniform periodic control against the established native ghost-deposit exchange.
  // It shares only the canonical TSC stencil, not the new remap or communication.
  const bool native_tsc=pin->GetOrAddBoolean("problem","gravity_test_native_tsc",false);
  Real native_error=0;
  if (native_tsc) {
    Require(!clouds && !pm->multilevel && pm->strictly_periodic && !dust->pbval_qp->General(),
            "native TSC control requires uniform periodic point dust with native deposit exchange");
    DvceArray5D<Real> reference("gravity_native_tsc_control",nmb,1,
                              density.extent(2),density.extent(3),density.extent(4));
    Kokkos::deep_copy(reference,0.0);
    auto particle=particles->prtcl_rdata;
    auto ids=particles->prtcl_idata;
    auto sizes=pack->pmb->mb_size.d_view;
    const int gids=pack->gids;
    Kokkos::parallel_for("gravity_native_tsc_deposit",Kokkos::RangePolicy<DevExeSpace>(0,npart),
      KOKKOS_LAMBDA(int p) {
        if (particle(IPM,p)==0) return;
        const int b=ids(PGID,p)-gids;
        const auto s=sizes(b);
        int ip,jp,kp; Real wx[3],wy[3],wz[3];
        dust::PMWeights(particle(IPX,p),s.x1min,s.x1max,ind.nx1,ind.is,2,ip,wx);
        dust::PMWeights(particle(IPY,p),s.x2min,s.x2max,ind.nx2,ind.js,2,jp,wy);
        dust::PMWeights(particle(IPZ,p),s.x3min,s.x3max,ind.nx3,ind.ks,2,kp,wz);
        const Real rd=particle(IPM,p)/(s.dx1*s.dx2*s.dx3);
        for (int c=0;c<3;++c) for (int j=0;j<3;++j) for (int i=0;i<3;++i)
          dust::DepositAdd(&reference(b,0,kp+c-1,jp+j-1,ip+i-1),rd*wz[c]*wy[j]*wx[i]);
      });
    dust->pbval_qp->InitRecvDep(1);
    dust->pbval_qp->PackAndSendDeposit(reference);
    while (dust->pbval_qp->RecvAndSumDeposit(reference)!=TaskStatus::complete) {}
    dust->pbval_qp->ClearSendDep();
    dust->pbval_qp->ClearRecvDep();
    auto native=Kokkos::create_mirror_view_and_copy(HostMemSpace(),reference);
    for (int b=0;b<nmb;++b) for (int k=ind.ks;k<=ind.ke;++k)
      for (int j=ind.js;j<=ind.je;++j) for (int i=ind.is;i<=ind.ie;++i)
        native_error=std::max(native_error,std::abs(native(b,0,k,j,i)-rho(b,0,k,j,i))/
                                         (1+std::abs(rho(b,0,k,j,i))));
    native_error=Maximum(native_error);
    Require(native_error<1.e-10,"uniform TSC differs from native periodic deposit");
  }

  // A distinct existing diagnostic deposit provides a useful owned/halo and
  // channel-layout control at the actual particle time, where it is defined.
  Real old_density_error=0;
  if (clouds && time==pm->time) {
    DvceArray5D<Real> reference;
    dust->DepositCloudDensity3D(reference);
    auto old=Kokkos::create_mirror_view_and_copy(HostMemSpace(),reference);
    for (int b=0;b<nmb;++b) for (int k=ind.ks;k<=ind.ke;++k)
      for (int j=ind.js;j<=ind.je;++j) for (int i=ind.is;i<=ind.ie;++i)
        old_density_error=std::max(old_density_error,
            std::abs(old(b,0,k,j,i)-rho(b,0,k,j,i))/(1+std::abs(rho(b,0,k,j,i))));
    old_density_error=Maximum(old_density_error);
    Require(old_density_error<1.e-10,"separate diagnostic density control");
  }
  const Real constant[3]={.375,-.25,1.125};
  for (int b=0;b<nmb;++b) for (int k=ind.ks;k<=ind.ke;++k)
    for (int j=ind.js;j<=ind.je;++j) for (int i=ind.is;i<=ind.ie;++i)
      for (int d=0;d<3;++d) fh(b,d,k,j,i)=constant[d];
  Kokkos::deep_copy(field,fh);
  dust->GatherGravityAcceleration(time,field,gathered);
  gp=Kokkos::create_mirror_view_and_copy(HostMemSpace(),gathered);
  Real constant_error=0;
  for (int p=0;p<npart;++p) if (pr(IPM,p)>0)
    for (int d=0;d<3;++d) constant_error=std::max(constant_error,std::abs(gp(d,p)-constant[d]));
  constant_error=Maximum(constant_error);
  Require(constant_error<1.e-10,"constant gravity changed by cloud/shear/halo map");
  const Real kick_time=.123;
  dust->KickGravity(kick_time,gathered);
  auto kicked=Kokkos::create_mirror_view_and_copy(HostMemSpace(),particles->prtcl_rdata);
  Real kick_error=0;
  for (int p=0;p<npart;++p) for (int q=0;q<kicked.extent_int(0);++q) {
    Real expected=pr(q,p);
    for (int d=0;d<3;++d) if (q==IPVX+2*d && pr(IPM,p)>0)
      expected+=kick_time*constant[d];
    kick_error=std::max(kick_error,std::abs(kicked(q,p)-expected));
  }
  kick_error=Maximum(kick_error);
  Kokkos::deep_copy(particles->prtcl_rdata,original_particles);
  Kokkos::deep_copy(particles->prtcl_idata,original_codes);
  Require(kick_error<1.e-10,"kick changed positions, mass, history or wrong velocity component");
  if (global_variable::my_rank==0)
    std::cout<<std::setprecision(16)<<"# DUST_GRAVITY_TRANSFER {\"time\":"<<time
      <<",\"particle_mass\":"<<mass[0]<<",\"leaf_mass\":"<<mass[1]
      <<",\"mass_error\":"<<mass_error<<",\"adjoint_error\":"<<adjoint_error
      <<",\"constant_error\":"<<constant_error<<",\"kick_error\":"<<kick_error
      <<",\"diagnostic_density_error\":"<<old_density_error
      <<",\"diagnostic_density_checked\":"<<((clouds && time==pm->time) ? "true" : "false")
      <<",\"native_tsc_checked\":"<<(native_tsc ? "true" : "false")
      <<",\"native_tsc_density_error\":"<<native_error
      <<",\"point_assignment_wall_particles\":"<<assignment_wall_particles
      <<",\"representation\":\""<<(clouds ? "refinable_cloud" : "owner_grid_tsc")<<"\""
      <<",\"diagnostic_shape_sweep\":"<<(shape_sweep ? "true" : "false")
      <<",\"shape_counts_plain_x1_x2_x3\":["<<shapes[0]<<','<<shapes[1]<<','
      <<shapes[2]<<','<<shapes[3]<<"],\"passed\":true}\n";
}
}  // namespace

void ProblemGenerator::DustGravityTransfer(ParameterInput *pin,const bool restart) {
  DustNSH(pin,restart);
  auto *pack=pmy_mesh_->pmb_pack;
  Require(pack->pdust!=nullptr && pack->pdust->GravityTransferSupported(),
          "requires a supported 3D cloud or TSC point initializer");
  if (!restart && pin->GetOrAddBoolean("problem","gravity_test_mass_modulation",false)) {
    Require(pack->pdust->PointGravityTransferSupported(),"mass-modulation fixture requires points");
    auto pr=pack->ppart->prtcl_rdata;
    auto pi=pack->ppart->prtcl_idata;
    Kokkos::parallel_for("gravity_test_unequal_particle_masses",
      Kokkos::RangePolicy<DevExeSpace>(0,pack->ppart->nprtcl_thispack),KOKKOS_LAMBDA(int p) {
        const std::uint64_t tag=GetTag(pi,p);
        pr(IPM,p)*=.5+1.5*static_cast<Real>(tag%17)/17;
      });
  }
  const int empty=pin->GetOrAddInteger("problem","gravity_test_empty_rank",-1);
  if (!restart && empty>=0) {
    Require(empty<global_variable::nranks,"empty rank outside communicator");
    if (global_variable::my_rank==empty) pack->ppart->nprtcl_thispack=0;
    int mine=pack->ppart->nprtcl_thispack;
#if MPI_PARALLEL_ENABLED
    MPI_Allgather(&mine,1,MPI_INT,pmy_mesh_->nprtcl_eachrank,1,MPI_INT,MPI_COMM_WORLD);
#else
    pmy_mesh_->nprtcl_eachrank[0]=mine;
#endif
    pmy_mesh_->nprtcl_thisrank=mine;
    pmy_mesh_->nprtcl_total=0;
    for (int r=0;r<global_variable::nranks;++r)
      pmy_mesh_->nprtcl_total+=pmy_mesh_->nprtcl_eachrank[r];
  }
  pgen_final_func=CheckTransfer;
}
