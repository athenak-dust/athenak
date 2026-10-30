//========================================================================================
// AthenaK: mass-assignment gravity adapter for independently moving dust clouds.
// Licensed under the 3-clause BSD License; see LICENSE.
//========================================================================================
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"

namespace dust {
namespace {
[[noreturn]] void GravityTransferFail(const char *message) {
  std::cerr << "### FATAL ERROR in dust gravity transfer: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
void CheckParticles(int bad) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if (bad) GravityTransferFail("invalid mass, particle owner or cloud weights");
}
void ResizeField(DvceArray5D<Real> &field, const char *name, int blocks,
                 int components, int nk, int nj, int ni) {
  if (field.extent_int(0)!=blocks || field.extent_int(1)!=components ||
      field.extent_int(2)!=nk || field.extent_int(3)!=nj || field.extent_int(4)!=ni)
    field=DvceArray5D<Real>(std::string(name),blocks,components,nk,nj,ni);
}
}  // namespace

bool DustGasDrag::GravityTransferSupported() const {
  return PointGravityTransferSupported() || (clouds3d && pmy_pack->pmesh->three_d &&
      smr_transfer_mode==DustSMRTransfer::refinable_cloud &&
      (pmy_pack->pmesh->mesh_bcs[0]!=BoundaryFlag::shear_periodic || shear3d));
}

void DustGasDrag::DepositGravityDensity(Real time, DvceArray5D<Real> &owned) {
  if (PointGravityTransferSupported()) {DepositPointGravityDensity(time,owned);return;}
  if (!GravityTransferSupported() || !std::isfinite(time))
    GravityTransferFail("requires 3D refinable clouds and a finite geometry time");
  UpdateShear3D(time);
  const auto lt=leaf_traversal3d;
  const auto ind=pmy_pack->pmesh->mb_indcs;
  const int nmb=pmy_pack->nmb_thispack;
  const int ni=ind.nx1+2*ind.ng, nj=ind.nx2+2*ind.ng, nk=ind.nx3+2*ind.ng;
  ResizeField(gravity_density_halo_,"gravity_dust_mass_halo",lt.nfld,1,nk,nj,ni);
  ResizeField(owned,"gravity_dust_density",nmb,1,nk,nj,ni);
  Kokkos::deep_copy(gravity_density_halo_,0.0);
  Kokkos::deep_copy(owned,0.0);
  auto field=gravity_density_halo_;
  auto *particles=pmy_pack->ppart;
  auto pr=particles->prtcl_rdata;
  auto pi=particles->prtcl_idata;
  const int npart=particles->nprtcl_thispack, gids=pmy_pack->gids;
  int bad=0;
  // the launch bound and the cell order of the drag kernels (CloudPolicy, cloud_ord): the same
  // clouds, each once; only the order of the atomic sums changes
  const int nord=CloudOrderLength(npart), nq=(nord>npart) ? nord : npart;
  auto ord=cloud_ord;
  Kokkos::parallel_reduce(nord>0 ? "gravity_dust_deposit" : "gravity_dust_deposit_storage",
      CloudPolicy(0,nq),KOKKOS_LAMBDA(int q,int &invalid) {
    const int p=(q<nord) ? ord(q) : q;
    if (p>=npart) return;   // (cloud_order_keep) an index slot past the end
    const Real mass=pr(IPM,p);
    if (!(mass>=0) || !Kokkos::isfinite(mass)) {++invalid;return;}
    if (mass==0) return;  // existing outflow tombstones carry no gravity mass
    if (!Kokkos::isfinite(pr(IPX,p)) || !Kokkos::isfinite(pr(IPY,p)) ||
        !Kokkos::isfinite(pr(IPZ,p))) {++invalid;return;}
    const int anchor=pi(PGID,p)-gids;
    if (anchor<0 || anchor>=nmb) {++invalid;return;}
    Real sum=0;
    lt.ForEachCloudWeight(pr(IPX,p),pr(IPY,p),pr(IPZ,p),pi(PLEV,p),anchor,
      [&](int target,Real weight) {
        int b,k,j,i;lt.Decode(target,b,k,j,i);
        const Real volume=lt.box(b,3)*lt.box(b,4)*lt.box(b,5);
        // plain atomic add: the warp aggregation (DepositAddWarp) made this one-component
        // deposit about 23% slower on two of four GH200 ranks
        DepositAdd(&field(b,0,k,j,i),mass*weight/volume);
        sum+=weight;
      });
    if (!Kokkos::isfinite(sum) || fabs(sum-1)>Real(1.e-9)) ++invalid;
  },bad);
  CheckParticles(bad);
  // Collective even on ranks with no particles: those ranks can own targets.
  HaloReverseAdd(gravity_density_halo_,0,1);
  auto destination=owned;
  par_for("gravity_dust_owned_density",DevExeSpace(),0,nmb-1,
      ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
      KOKKOS_LAMBDA(int b,int k,int j,int i) {destination(b,0,k,j,i)=field(b,0,k,j,i);});
}

void DustGasDrag::GatherGravityAcceleration(Real time,
    const DvceArray5D<Real> &owned,DvceArray2D<Real> &particle_acceleration,bool record) {
  ++gravity_halo_fills_;   // any earlier recording no longer matches the halo field
  gravity_gather_nsnap_=-1;
  if (PointGravityTransferSupported()) {
    GatherPointGravityAcceleration(time,owned,particle_acceleration);return;
  }
  if (!GravityTransferSupported() || !std::isfinite(time))
    GravityTransferFail("requires 3D refinable clouds and a finite geometry time");
  UpdateShear3D(time);
  const auto lt=leaf_traversal3d;
  const auto ind=pmy_pack->pmesh->mb_indcs;
  const int nmb=pmy_pack->nmb_thispack;
  const int ni=ind.nx1+2*ind.ng, nj=ind.nx2+2*ind.ng, nk=ind.nx3+2*ind.ng;
  if (owned.extent_int(0)<nmb || owned.extent_int(1)<3 ||
      owned.extent_int(2)<nk || owned.extent_int(3)<nj || owned.extent_int(4)<ni)
    GravityTransferFail("owned acceleration does not cover active hydro cells");
  ResizeField(gravity_acceleration_halo_,"gravity_dust_force_halo",lt.nfld,3,nk,nj,ni);
  Kokkos::deep_copy(gravity_acceleration_halo_,0.0);
  auto field=gravity_acceleration_halo_;
  par_for("gravity_pack_owned_force",DevExeSpace(),0,nmb-1,
      ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
      KOKKOS_LAMBDA(int b,int k,int j,int i) {
        for (int d=0;d<3;++d) field(b,d,k,j,i)=owned(b,d,k,j,i);
      });
  // This is a vector in fixed Cartesian axes: no velocity/shear boost is added.
  HaloForward5D(gravity_acceleration_halo_,0,3);
  auto *particles=pmy_pack->ppart;
  const int npart=particles->nprtcl_thispack, gids=pmy_pack->gids;
  if (particle_acceleration.extent_int(0)!=3 ||
      particle_acceleration.extent_int(1)!=npart)
    particle_acceleration=DvceArray2D<Real>("gravity_particle_acceleration",3,npart);
  auto result=particle_acceleration;
  auto pr=particles->prtcl_rdata;
  auto pi=particles->prtcl_idata;
  int bad=0;
  // as the deposit: launch bound and cell order (each cloud writes only its own result)
  const int nord=CloudOrderLength(npart), nq=(nord>npart) ? nord : npart;
  auto ord=cloud_ord;
  Kokkos::parallel_reduce(nord>0 ? "gravity_dust_gather" : "gravity_dust_gather_storage",
      CloudPolicy(0,nq),KOKKOS_LAMBDA(int q,int &invalid) {
    const int p=(q<nord) ? ord(q) : q;
    if (p>=npart) return;   // (cloud_order_keep) an index slot past the end
    for (int d=0;d<3;++d) result(d,p)=0;
    const Real mass=pr(IPM,p);
    if (!(mass>=0) || !Kokkos::isfinite(mass)) {++invalid;return;}
    if (mass==0) return;
    if (!Kokkos::isfinite(pr(IPX,p)) || !Kokkos::isfinite(pr(IPY,p)) ||
        !Kokkos::isfinite(pr(IPZ,p))) {++invalid;return;}
    const int anchor=pi(PGID,p)-gids;
    if (anchor<0 || anchor>=nmb) {++invalid;return;}
    Real sum=0, acc[3]={0,0,0};   // (in registers: the same sums in the same order)
    lt.ForEachCloudWeight(pr(IPX,p),pr(IPY,p),pr(IPZ,p),pi(PLEV,p),anchor,
      [&](int target,Real weight) {
        int b,k,j,i;lt.Decode(target,b,k,j,i);
        for (int d=0;d<3;++d) acc[d]+=weight*field(b,d,k,j,i);
        sum+=weight;
      });
    for (int d=0;d<3;++d) result(d,p)=acc[d];
    if (!Kokkos::isfinite(sum) || fabs(sum-1)>Real(1.e-9)) ++invalid;
    for (int d=0;d<3;++d) if (!Kokkos::isfinite(acc[d])) ++invalid;
  },bad);
  CheckParticles(bad);
  if (record) {
    RecordGravityGather(time);
    gravity_gather_target_=particle_acceleration.data();
  }
}

void DustGasDrag::RecordGravityGather(Real time) {
  auto *particles=pmy_pack->ppart;
  const int npart=particles->nprtcl_thispack;
  if (gravity_gather_rsnap_.extent_int(1)<npart) {
    gravity_gather_rsnap_=DvceArray2D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
        "gravity_gather_rsnap"),4,npart);
    gravity_gather_isnap_=DvceArray2D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
        "gravity_gather_isnap"),2,npart);
  }
  auto rs=gravity_gather_rsnap_;
  auto is=gravity_gather_isnap_;
  auto pr=particles->prtcl_rdata;
  auto pi=particles->prtcl_idata;
  Kokkos::parallel_for("gravity_gather_record",Kokkos::RangePolicy<DevExeSpace>(0,npart),
    KOKKOS_LAMBDA(int p) {
      rs(0,p)=pr(IPX,p); rs(1,p)=pr(IPY,p); rs(2,p)=pr(IPZ,p); rs(3,p)=pr(IPM,p);
      is(0,p)=pi(PLEV,p); is(1,p)=pi(PGID,p);
    });
  gravity_gather_nsnap_=npart;
  gravity_gather_time_=time;
  gravity_gather_fill_=gravity_halo_fills_;
}

long long DustGasDrag::RegatherGravityAccelerationChanged(Real time,
    DvceArray2D<Real> &particle_acceleration) {
  if (PointGravityTransferSupported() || !GravityTransferSupported()
      || gravity_gather_nsnap_<0 || time!=gravity_gather_time_
      || gravity_gather_fill_!=gravity_halo_fills_
      || particle_acceleration.data()!=gravity_gather_target_
      || particle_acceleration.extent_int(0)!=3
      || particle_acceleration.extent_int(1)!=gravity_gather_nsnap_) return -1;
  UpdateShear3D(time);
  const auto lt=leaf_traversal3d;
  const auto ind=pmy_pack->pmesh->mb_indcs;
  const int nmb=pmy_pack->nmb_thispack;
  const int ni=ind.nx1+2*ind.ng, nj=ind.nx2+2*ind.ng, nk=ind.nx3+2*ind.ng;
  if (gravity_acceleration_halo_.extent_int(0)!=lt.nfld ||
      gravity_acceleration_halo_.extent_int(1)!=3 || gravity_acceleration_halo_.extent_int(2)!=nk ||
      gravity_acceleration_halo_.extent_int(3)!=nj || gravity_acceleration_halo_.extent_int(4)!=ni)
    GravityTransferFail("selective gravity regather: the exchanged field does not fit the mesh");
  auto field=gravity_acceleration_halo_;
  auto *particles=pmy_pack->ppart;
  const int npart=particles->nprtcl_thispack, gids=pmy_pack->gids, nold=gravity_gather_nsnap_;
  auto old=particle_acceleration;
  auto result=old;
  if (npart!=nold) result=DvceArray2D<Real>("gravity_particle_acceleration",3,npart);
  const bool copy=npart!=nold;
  auto rs=gravity_gather_rsnap_;
  auto is=gravity_gather_isnap_;
  auto pr=particles->prtcl_rdata;
  auto pi=particles->prtcl_idata;
  int bad=0, changed=0;
  const int nord=CloudOrderLength(npart), nq=(nord>npart) ? nord : npart;
  auto ord=cloud_ord;
  Kokkos::parallel_reduce(nord>0 ? "gravity_dust_regather" : "gravity_dust_regather_storage",
      CloudPolicy(0,nq),KOKKOS_LAMBDA(int q,int &invalid,int &count) {
    const int p=(q<nord) ? ord(q) : q;
    if (p>=npart) return;   // (cloud_order_keep) an index slot past the end
    if (p<nold && rs(0,p)==pr(IPX,p) && rs(1,p)==pr(IPY,p) && rs(2,p)==pr(IPZ,p) &&
        rs(3,p)==pr(IPM,p) && is(0,p)==pi(PLEV,p) && is(1,p)==pi(PGID,p)) {
      if (copy) for (int d=0;d<3;++d) result(d,p)=old(d,p);
      return;
    }
    ++count;
    // the body of GatherGravityAcceleration for this cloud
    for (int d=0;d<3;++d) result(d,p)=0;
    const Real mass=pr(IPM,p);
    if (!(mass>=0) || !Kokkos::isfinite(mass)) {++invalid;return;}
    if (mass==0) return;
    if (!Kokkos::isfinite(pr(IPX,p)) || !Kokkos::isfinite(pr(IPY,p)) ||
        !Kokkos::isfinite(pr(IPZ,p))) {++invalid;return;}
    const int anchor=pi(PGID,p)-gids;
    if (anchor<0 || anchor>=nmb) {++invalid;return;}
    Real sum=0, acc[3]={0,0,0};   // (in registers: the same sums in the same order)
    lt.ForEachCloudWeight(pr(IPX,p),pr(IPY,p),pr(IPZ,p),pi(PLEV,p),anchor,
      [&](int target,Real weight) {
        int b,k,j,i;lt.Decode(target,b,k,j,i);
        for (int d=0;d<3;++d) acc[d]+=weight*field(b,d,k,j,i);
        sum+=weight;
      });
    for (int d=0;d<3;++d) result(d,p)=acc[d];
    if (!Kokkos::isfinite(sum) || fabs(sum-1)>Real(1.e-9)) ++invalid;
    for (int d=0;d<3;++d) if (!Kokkos::isfinite(acc[d])) ++invalid;
  },bad,changed);
  CheckParticles(bad);
  particle_acceleration=result;
  RecordGravityGather(time);
  gravity_gather_target_=particle_acceleration.data();
  return changed;
}

void DustGasDrag::KickGravity(Real dt,const DvceArray2D<Real> &acceleration) {
  auto *particles=pmy_pack->ppart;
  const int npart=particles->nprtcl_thispack;
  if (!std::isfinite(dt) || acceleration.extent_int(0)!=3 ||
      acceleration.extent_int(1)!=npart)
    GravityTransferFail("invalid kick time or stale particle acceleration array");
  auto pr=particles->prtcl_rdata;
  Kokkos::parallel_for("gravity_dust_kick",Kokkos::RangePolicy<DevExeSpace>(0,npart),
    KOKKOS_LAMBDA(int p) {
      if (pr(IPM,p)==0) return;
      for (int d=0;d<3;++d) pr(IPVX+2*d,p)+=dt*acceleration(d,p);
    });
}

Real DustGasDrag::GravityAccelerationTimeStep(const DvceArray2D<Real> &acceleration) {
  auto *particles=pmy_pack->ppart;
  const int npart=particles->nprtcl_thispack, gids=pmy_pack->gids;
  if (acceleration.extent_int(0)!=3 || acceleration.extent_int(1)!=npart)
    GravityTransferFail("stale particle field in acceleration timestep");
  auto pr=particles->prtcl_rdata;
  auto pi=particles->prtcl_idata;
  const auto lt=leaf_traversal3d;
  const bool points=PointGravityTransferSupported();
  auto sizes=pmy_pack->pmb->mb_size.d_view;
  Real limit=std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("gravity_dust_acceleration_timestep",
      Kokkos::RangePolicy<DevExeSpace>(0,npart),KOKKOS_LAMBDA(int p,Real &minimum) {
    if (pr(IPM,p)==0) return;
    const int b=pi(PGID,p)-gids;
    if (points) {
      const Real width[3]={sizes(b).dx1,sizes(b).dx2,sizes(b).dx3};
      for (int d=0;d<3;++d) {
        const Real a=fabs(acceleration(d,p));
        if (a>0) minimum=fmin(minimum,sqrt(width[d]/a));
      }
      return;
    }
    const int level=pi(PLEV,p)&SMRCompositeTransfer::kCloudLevelMask;
    const Real width[3]={lt.root_dx1/static_cast<Real>(1<<level),
        lt.root_dx2/static_cast<Real>(1<<level),lt.root_dx3/static_cast<Real>(1<<level)};
    for (int d=0;d<3;++d) {
      const Real a=fabs(acceleration(d,p));
      if (a>0) minimum=fmin(minimum,sqrt(fmin(width[d],lt.box(b,3+d))/a));
    }
  },Kokkos::Min<Real>(limit));
  return limit;  // the gravity module applies its cfl, Mesh reduces all ranks
}
}  // namespace dust
