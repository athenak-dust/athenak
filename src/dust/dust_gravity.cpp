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
  return clouds3d && pmy_pack->pmesh->three_d &&
      smr_transfer_mode==DustSMRTransfer::refinable_cloud &&
      (pmy_pack->pmesh->mesh_bcs[0]!=BoundaryFlag::shear_periodic || shear3d);
}

void DustGasDrag::DepositGravityDensity(Real time, DvceArray5D<Real> &owned) {
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
  Kokkos::parallel_reduce("gravity_dust_deposit",
      Kokkos::RangePolicy<DevExeSpace>(0,npart),KOKKOS_LAMBDA(int p,int &invalid) {
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
    const DvceArray5D<Real> &owned,DvceArray2D<Real> &particle_acceleration) {
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
  Kokkos::parallel_reduce("gravity_dust_gather",
      Kokkos::RangePolicy<DevExeSpace>(0,npart),KOKKOS_LAMBDA(int p,int &invalid) {
    for (int d=0;d<3;++d) result(d,p)=0;
    const Real mass=pr(IPM,p);
    if (!(mass>=0) || !Kokkos::isfinite(mass)) {++invalid;return;}
    if (mass==0) return;
    if (!Kokkos::isfinite(pr(IPX,p)) || !Kokkos::isfinite(pr(IPY,p)) ||
        !Kokkos::isfinite(pr(IPZ,p))) {++invalid;return;}
    const int anchor=pi(PGID,p)-gids;
    if (anchor<0 || anchor>=nmb) {++invalid;return;}
    Real sum=0;
    lt.ForEachCloudWeight(pr(IPX,p),pr(IPY,p),pr(IPZ,p),pi(PLEV,p),anchor,
      [&](int target,Real weight) {
        int b,k,j,i;lt.Decode(target,b,k,j,i);
        for (int d=0;d<3;++d) result(d,p)+=weight*field(b,d,k,j,i);
        sum+=weight;
      });
    if (!Kokkos::isfinite(sum) || fabs(sum-1)>Real(1.e-9)) ++invalid;
    for (int d=0;d<3;++d) if (!Kokkos::isfinite(result(d,p))) ++invalid;
  },bad);
  CheckParticles(bad);
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
  Real limit=std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("gravity_dust_acceleration_timestep",
      Kokkos::RangePolicy<DevExeSpace>(0,npart),KOKKOS_LAMBDA(int p,Real &minimum) {
    if (pr(IPM,p)==0) return;
    const int b=pi(PGID,p)-gids;
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
