//========================================================================================
// Point-particle gravity: owner-grid TSC followed by the actual leaf-density remap.
// The gather is its mass transpose, not the native gas ghost interpolation.
// Licensed under the 3-clause BSD License; see LICENSE.
//========================================================================================
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "athena.hpp"
#include "dust/dust.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"

namespace dust {
namespace {
[[noreturn]] void PointFail(const char *message) {
  std::cerr << "### FATAL ERROR in dust point gravity: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
void CheckPointParticles(int bad) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (bad) PointFail("invalid particle mass, owner, position or TSC support");
}
void Resize(DvceArray5D<Real> &a,const char *name,int nb,int nv,int nk,int nj,int ni) {
  if (a.extent_int(0)!=nb || a.extent_int(1)!=nv || a.extent_int(2)!=nk ||
      a.extent_int(3)!=nj || a.extent_int(4)!=ni)
    a=DvceArray5D<Real>(std::string(name),nb,nv,nk,nj,ni);
}
}  // namespace

// Own communicator/buffers keep gravity independent of in-flight drag exchanges.
// The plan contains block/ghost overlap metadata, never a global particle/cell table.
struct GravityPointTransfer {
  std::unique_ptr<DepRemapPlan> plan;
  int mesh_seq=-1;
  DvceArray1D<Real> outgoing,incoming;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm=MPI_COMM_NULL;
  GravityPointTransfer() {MPI_Comm_dup(MPI_COMM_WORLD,&comm);}
  ~GravityPointTransfer() {
    int finalized=0;
    MPI_Finalized(&finalized);
    if (!finalized && comm!=MPI_COMM_NULL) MPI_Comm_free(&comm);
  }
#endif
  // For the transpose, sends/receives exchange roles. Contributions from different
  // target ranks are summed on the source owner after the exchange.
  void Exchange(const DepRemapMap &map,int nv,bool transpose) {
    DevExeSpace().fence();
#if MPI_PARALLEL_ENABLED
    const auto &sr=transpose ? map.recv_rank : map.send_rank;
    const auto &so=transpose ? map.recv_off : map.send_off;
    const auto &sc=transpose ? map.recv_cnt : map.send_cnt;
    const auto &rr=transpose ? map.send_rank : map.recv_rank;
    const auto &ro=transpose ? map.send_off : map.recv_off;
    const auto &rc=transpose ? map.send_cnt : map.recv_cnt;
    std::vector<MPI_Request> requests;
    for (std::size_t q=0;q<rr.size();++q) {
      MPI_Request r;
      MPI_Irecv(incoming.data()+nv*ro[q],nv*rc[q],MPI_ATHENA_REAL,rr[q],0,comm,&r);
      requests.push_back(r);
    }
    for (std::size_t q=0;q<sr.size();++q) {
      MPI_Request r;
      MPI_Isend(outgoing.data()+nv*so[q],nv*sc[q],MPI_ATHENA_REAL,sr[q],0,comm,&r);
      requests.push_back(r);
    }
    if (!requests.empty())
      MPI_Waitall(static_cast<int>(requests.size()),requests.data(),MPI_STATUSES_IGNORE);
#endif
  }
  void Buffers(const DepRemapMap &map,int nv) {
    const int n=std::max(1,std::max(map.nsend,map.nrecv)*nv);
    if (outgoing.extent_int(0)<n) {
      outgoing=DvceArray1D<Real>("gravity_point_send",n);
      incoming=DvceArray1D<Real>("gravity_point_recv",n);
    }
    Kokkos::deep_copy(outgoing,0.0);
    Kokkos::deep_copy(incoming,0.0);
  }
  void DepositMap(const DepRemapMap &map,DvceArray5D<Real> &field) {
    const int ni=field.extent_int(4),nj=field.extent_int(3),nk=field.extent_int(2);
    const int nc=ni*nj*nk;
    Buffers(map,1);
    auto out=outgoing,in=incoming;
    auto cells=map.send_cell;
    Kokkos::parallel_for("gravity_point_pack",Kokkos::RangePolicy<DevExeSpace>(0,map.nsend),
      KOKKOS_LAMBDA(int e) {
        const int c=cells(e),b=c/nc,r=c%nc;
        out(e)=field(b,0,r/(nj*ni),(r/ni)%nj,r%ni);
      });
    Exchange(map,1,false);
    auto tgt=map.seg_tgt,ptr=map.seg_ptr,src=map.ent_src;
    auto w=map.ent_w;
    Kokkos::parallel_for("gravity_point_remap",Kokkos::RangePolicy<DevExeSpace>(0,map.nseg),
      KOKKOS_LAMBDA(int s) {
        const int c=tgt(s),b=c/nc,r=c%nc;
        Real sum=0;
        for (int e=ptr(s);e<ptr(s+1);++e) {
          const int q=src(e),sb=q/nc,sr=q%nc;
          const Real value=q<0 ? in(-q-1) : field(sb,0,sr/(nj*ni),(sr/ni)%nj,sr%ni);
          sum+=w(e)*value;
        }
        field(b,0,r/(nj*ni),(r/ni)%nj,r%ni)+=sum;
      });
  }
  void GatherMap(const DepRemapMap &map,const DvceArray5D<Real> &owned,
                 DvceArray5D<Real> &field,MeshBlockPack *pack) {
    const int ni=field.extent_int(4),nj=field.extent_int(3),nk=field.extent_int(2);
    const int nc=ni*nj*nk;
    Buffers(map,3);
    auto out=outgoing,in=incoming;
    auto tgt=map.seg_tgt,ptr=map.seg_ptr,src=map.ent_src;
    auto w=map.ent_w;
    auto size=pack->pmb->mb_size.d_view;
    // R_density(e)=overlap/V_target. Accumulate V_target*g first;
    // local and received contributions then divide by V_source exactly once.
    Kokkos::parallel_for("gravity_point_transpose",Kokkos::RangePolicy<DevExeSpace>(0,map.nseg),
      KOKKOS_LAMBDA(int s) {
        const int c=tgt(s),b=c/nc,r=c%nc;
        const Real vt=size(b).dx1*size(b).dx2*size(b).dx3;
        for (int e=ptr(s);e<ptr(s+1);++e) {
          const int q=src(e);
          if (q<0) {
            for (int d=0;d<3;++d)
              DepositAdd(&out(3*(-q-1)+d),w(e)*vt*owned(b,d,r/(nj*ni),(r/ni)%nj,r%ni));
          } else {
            const int sb=q/nc,sr=q%nc;
            const Real vs=size(sb).dx1*size(sb).dx2*size(sb).dx3;
            for (int d=0;d<3;++d)
              DepositAdd(&field(sb,d,sr/(nj*ni),(sr/ni)%nj,sr%ni),
                         w(e)*vt/vs*owned(b,d,r/(nj*ni),(r/ni)%nj,r%ni));
          }
        }
      });
    Exchange(map,3,true);
    auto cells=map.send_cell;
    Kokkos::parallel_for("gravity_point_unpack_transpose",
      Kokkos::RangePolicy<DevExeSpace>(0,map.nsend),KOKKOS_LAMBDA(int e) {
        const int c=cells(e),b=c/nc,r=c%nc;
        const Real vs=size(b).dx1*size(b).dx2*size(b).dx3;
        for (int d=0;d<3;++d)
          DepositAdd(&field(b,d,r/(nj*ni),(r/ni)%nj,r%ni),in(3*e+d)/vs);
      });
  }
};

bool DustGasDrag::PointGravityTransferSupported() const {
  const auto *pm=pmy_pack->pmesh;
  return pm->three_d && !clouds3d && deposit==DustDeposit::tsc &&
      (smr_transfer_mode==DustSMRTransfer::owner_overlap ||
       (smr_transfer_mode==DustSMRTransfer::none && !pm->multilevel));
}

void DustGasDrag::PreparePointGravity(Real time) {
  if (!PointGravityTransferSupported() || !std::isfinite(time))
    PointFail("requires 3D uniform canonical TSC or owner_overlap TSC and finite field time");
  auto *pm=pmy_pack->pmesh;
  if (!gravity_points_) gravity_points_=std::make_shared<GravityPointTransfer>();
  auto &state=*gravity_points_;
  const int seq=pm->GetAMRLoadBalanceUpdateSeq();
  if (!state.plan) {
    const bool shear=pm->mesh_bcs[BoundaryFace::inner_x1]==BoundaryFlag::shear_periodic;
    const bool physical=pm->mesh_bcs[BoundaryFace::inner_x3]!=BoundaryFlag::periodic;
    state.plan.reset(new DepRemapPlan(pmy_pack,shear,qshear,omega0,physical ? 1 : 0));
    state.mesh_seq=seq;
    if (global_variable::my_rank==0)
      std::cout << "# DUST_GRAVITY_POINTS assignment=owner_tsc+leaf_density_remap"
                << " x3_assignment=" << (physical ? "fold" : "periodic")
                << " drag_deposit_policy=independent" << std::endl;
  } else if (state.mesh_seq!=seq) {
    state.plan->BuildStatic();
    state.mesh_seq=seq;
  }
  state.plan->UpdateShear(time);
}

void DustGasDrag::DepositPointGravityDensity(Real time,DvceArray5D<Real> &owned) {
  PreparePointGravity(time);
  const auto ind=pmy_pack->pmesh->mb_indcs;
  const int nb=pmy_pack->nmb_thispack,gids=pmy_pack->gids;
  const int ni=ind.nx1+2*ind.ng,nj=ind.nx2+2*ind.ng,nk=ind.nx3+2*ind.ng;
  Resize(gravity_density_halo_,"gravity_point_density",nb,1,nk,nj,ni);
  Resize(owned,"gravity_dust_density",nb,1,nk,nj,ni);
  Kokkos::deep_copy(gravity_density_halo_,0.0);
  Kokkos::deep_copy(owned,0.0);
  auto field=gravity_density_halo_;
  auto size=pmy_pack->pmb->mb_size.d_view;
  auto pr=pmy_pack->ppart->prtcl_rdata;
  auto pi=pmy_pack->ppart->prtcl_idata;
  const int np=pmy_pack->ppart->nprtcl_thispack;
  int bad=0;
  Kokkos::parallel_reduce("gravity_point_deposit",Kokkos::RangePolicy<DevExeSpace>(0,np),
    KOKKOS_LAMBDA(int p,int &invalid) {
      const Real mass=pr(IPM,p);
      if (!(mass>=0) || !Kokkos::isfinite(mass)) {++invalid;return;}
      if (mass==0) return;
      const int b=pi(PGID,p)-gids;
      if (b<0 || b>=nb || !Kokkos::isfinite(pr(IPX,p)) ||
          !Kokkos::isfinite(pr(IPY,p)) || !Kokkos::isfinite(pr(IPZ,p))) {++invalid;return;}
      const auto s=size(b);
      int ip,jp,kp;Real wx[3],wy[3],wz[3];
      PMWeights(pr(IPX,p),s.x1min,s.x1max,ind.nx1,ind.is,2,ip,wx);
      PMWeights(pr(IPY,p),s.x2min,s.x2max,ind.nx2,ind.js,2,jp,wy);
      PMWeights(pr(IPZ,p),s.x3min,s.x3max,ind.nx3,ind.ks,2,kp,wz);
      if (ip<1 || ip>=ni-1 || jp<1 || jp>=nj-1 || kp<1 || kp>=nk-1) {++invalid;return;}
      const Real density=mass/(s.dx1*s.dx2*s.dx3);
      for (int c=0;c<3;++c) for (int j=0;j<3;++j) for (int i=0;i<3;++i)
        DepositAdd(&field(b,0,kp+c-1,jp+j-1,ip+i-1),density*wz[c]*wy[j]*wx[i]);
    },bad);
  CheckPointParticles(bad);
  auto &state=*gravity_points_;
  state.DepositMap(state.plan->stat,gravity_density_halo_);
  if (state.plan->Shear()) state.DepositMap(state.plan->shr,gravity_density_halo_);
  auto destination=owned;
  par_for("gravity_point_owned",DevExeSpace(),0,nb-1,ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
    KOKKOS_LAMBDA(int b,int k,int j,int i) {destination(b,0,k,j,i)=field(b,0,k,j,i);});
}

void DustGasDrag::GatherPointGravityAcceleration(Real time,const DvceArray5D<Real> &owned,
                                                DvceArray2D<Real> &result) {
  PreparePointGravity(time);
  const auto ind=pmy_pack->pmesh->mb_indcs;
  const int nb=pmy_pack->nmb_thispack,gids=pmy_pack->gids;
  const int ni=ind.nx1+2*ind.ng,nj=ind.nx2+2*ind.ng,nk=ind.nx3+2*ind.ng;
  if (owned.extent_int(0)<nb || owned.extent_int(1)<3 || owned.extent_int(2)<nk ||
      owned.extent_int(3)<nj || owned.extent_int(4)<ni) PointFail("stale owned force field");
  Resize(gravity_acceleration_halo_,"gravity_point_force",nb,3,nk,nj,ni);
  Kokkos::deep_copy(gravity_acceleration_halo_,0.0);
  auto field=gravity_acceleration_halo_;
  // Owner active cells are the identity part of the effective deposit.
  par_for("gravity_point_force_identity",DevExeSpace(),0,nb-1,
    ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,KOKKOS_LAMBDA(int b,int k,int j,int i) {
      for (int d=0;d<3;++d) field(b,d,k,j,i)=owned(b,d,k,j,i);
    });
  auto &state=*gravity_points_;
  state.GatherMap(state.plan->stat,owned,gravity_acceleration_halo_,pmy_pack);
  if (state.plan->Shear()) state.GatherMap(state.plan->shr,owned,gravity_acceleration_halo_,pmy_pack);
  auto pr=pmy_pack->ppart->prtcl_rdata;
  auto pi=pmy_pack->ppart->prtcl_idata;
  auto size=pmy_pack->pmb->mb_size.d_view;
  const int np=pmy_pack->ppart->nprtcl_thispack;
  if (result.extent_int(0)!=3 || result.extent_int(1)!=np)
    result=DvceArray2D<Real>("gravity_particle_acceleration",3,np);
  auto acceleration=result;
  int bad=0;
  Kokkos::parallel_reduce("gravity_point_gather",Kokkos::RangePolicy<DevExeSpace>(0,np),
    KOKKOS_LAMBDA(int p,int &invalid) {
      for (int d=0;d<3;++d) acceleration(d,p)=0;
      const Real mass=pr(IPM,p);
      if (!(mass>=0) || !Kokkos::isfinite(mass)) {++invalid;return;}
      if (mass==0) return;
      const int b=pi(PGID,p)-gids;
      if (b<0 || b>=nb || !Kokkos::isfinite(pr(IPX,p)) ||
          !Kokkos::isfinite(pr(IPY,p)) || !Kokkos::isfinite(pr(IPZ,p))) {++invalid;return;}
      const auto s=size(b);
      int ip,jp,kp;Real wx[3],wy[3],wz[3];
      PMWeights(pr(IPX,p),s.x1min,s.x1max,ind.nx1,ind.is,2,ip,wx);
      PMWeights(pr(IPY,p),s.x2min,s.x2max,ind.nx2,ind.js,2,jp,wy);
      PMWeights(pr(IPZ,p),s.x3min,s.x3max,ind.nx3,ind.ks,2,kp,wz);
      if (ip<1 || ip>=ni-1 || jp<1 || jp>=nj-1 || kp<1 || kp>=nk-1) {++invalid;return;}
      for (int c=0;c<3;++c) for (int j=0;j<3;++j) for (int i=0;i<3;++i)
        for (int d=0;d<3;++d)
          acceleration(d,p)+=wz[c]*wy[j]*wx[i]*field(b,d,kp+c-1,jp+j-1,ip+i-1);
      for (int d=0;d<3;++d) if (!Kokkos::isfinite(acceleration(d,p))) ++invalid;
    },bad);
  CheckPointParticles(bad);
}
}  // namespace dust
