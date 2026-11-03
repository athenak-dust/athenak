//========================================================================================
// AthenaK: mass-assignment gravity adapter for independently moving dust clouds.
// Licensed under the 3-clause BSD License; see LICENSE.
//========================================================================================
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
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

// The atomic adds of the gravity mass deposit on a GPU (environment ATHENA_SG_DEPOSIT_ATOMIC,
// read once).  In the cell order of the cloud kernels (cloud_ord) the lanes of a warp often
// deposit into the same leaf cells in the same step, and atomic adds to one address serialize:
// in clumps the plain deposit took several times as long on one
// rank as on another, and every rank waits for the slowest in the check that follows.
//   warp (default): DepositAddWarp, as the drag deposits: the lanes of a warp that add to the same
//            cell in the same step sum in registers, one of them adds (even times on
//            every rank, where the plain adds varied by a factor of two; a variant that aggregated
//            only clouds sharing their first cell with another lane was slower everywhere)
//   plain  : one atomic add per cloud and cell (the former path)
//   cycle  : (diagnostic) plain and warp in turn, one per call (kernel labels
//            gravity_dust_deposit_plain and gravity_dust_deposit; the plain path has the label
//            gravity_dust_deposit_plain, the warp path the former label gravity_dust_deposit)
//   compare: (diagnostic) the warp deposit, then two plain deposits into scratch fields; rank 0
//            prints the largest |warp - plain| and |plain - plain| over the largest density
// Both add the same terms; only the order of the floating-point sums changes, as between any two
// runs with atomics.  Host backends (and CUDA before compute capability 7.0) always add in the
// plain order (DepositAddWarp falls back to DepositAdd), so CPU results do not depend on the mode.
enum DepositAtomicMode {kDepositPlain = 0, kDepositWarp = 1, kDepositCycle = 2,
                        kDepositCompare = 3};
int DepositAtomicMode() {
  static const int mode = [] {
    const char *e = std::getenv("ATHENA_SG_DEPOSIT_ATOMIC");
    const std::string s = (e == nullptr) ? "" : e;
    if (s.empty() || s == "warp") return static_cast<int>(kDepositWarp);
    if (s == "plain") return static_cast<int>(kDepositPlain);
    if (s == "cycle") return static_cast<int>(kDepositCycle);
    if (s == "compare") return static_cast<int>(kDepositCompare);
    std::cerr << "### FATAL ERROR: ATHENA_SG_DEPOSIT_ATOMIC=" << s
              << " (warp, plain, cycle or compare)" << std::endl;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
    std::exit(EXIT_FAILURE);
  }();
  return mode;
}
}  // namespace

// One gravity mass deposit of all clouds into field (cleared by the caller) with the atomic
// adds of MODE (kDepositPlain or kDepositWarp); returns the invalid clouds.
template <int MODE>
int GravityDepositKernel(const char *label, const LeafTraversal3D &lt,
                         const DvceArray5D<Real> &field, const DvceArray2D<Real> &pr,
                         const DvceArray2D<int> &pi, const DvceArray1D<int> &ord,
                         const int npart, const int nord, const int gids, const int nmb) {
  int bad = 0;
  const int nq=(nord>npart) ? nord : npart;
  Kokkos::parallel_reduce(label,CloudPolicy(0,nq),KOKKOS_LAMBDA(int q,int &invalid) {
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
        // (the cell address outside the constexpr branches: nvcc does not let an extended
        // lambda first capture a variable inside one)
        Real *dst=&field(b,0,k,j,i);
        Real v[1]={mass*weight/volume};
        if constexpr (MODE==kDepositPlain) {
          DepositAdd(dst,v[0]);
        } else {
          DepositAddWarp<1>(dst,0,target,v);
        }
        sum+=weight;
      });
    if (!Kokkos::isfinite(sum) || fabs(sum-1)>Real(1.e-9)) ++invalid;
  },bad);
  return bad;
}

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
  // the launch bound and the cell order of the drag kernels (CloudPolicy, cloud_ord): the same
  // clouds, each once; only the order of the atomic sums changes (also between the atomic modes,
  // ATHENA_SG_DEPOSIT_ATOMIC above)
  const int nord=CloudOrderLength(npart);
  auto ord=cloud_ord;
  const int config=DepositAtomicMode();
  int mode=(config==kDepositCompare) ? static_cast<int>(kDepositWarp) : config;
  static int calls=0;   // (cycle) the calls so far
  // compare adds an MPI_Allreduce: the choice must agree on every rank, checked at the first
  // call (collective: every rank calls this deposit, see CheckParticles below)
  static bool compare_checked=false;
  if (!compare_checked) {
#if MPI_PARALLEL_ENABLED
    const int a=(config==kDepositCompare) ? 1 : 0;
    int v[2]={a,-a};
    MPI_Allreduce(MPI_IN_PLACE,v,2,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if (v[0]!=-v[1]) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "ATHENA_SG_DEPOSIT_ATOMIC=compare must be set on every rank or on none"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
#endif
    compare_checked=true;
  }
  if (config==kDepositCycle) mode=(calls++)%2;
  auto deposit=[&](const int m, const DvceArray5D<Real> &f) {
    if (m==kDepositPlain) {
      return GravityDepositKernel<kDepositPlain>(nord>0 ? "gravity_dust_deposit_plain"
          : "gravity_dust_deposit_plain_storage",lt,f,pr,pi,ord,npart,nord,gids,nmb);
    }
    return GravityDepositKernel<kDepositWarp>(nord>0 ? "gravity_dust_deposit"
        : "gravity_dust_deposit_storage",lt,f,pr,pi,ord,npart,nord,gids,nmb);
  };
  const int bad=deposit(mode,field);
  if (config==kDepositCompare) {
    // the same deposit with plain atomic adds, twice: the difference of the mode against the
    // plain path next to the difference of two plain runs (both round-off of the atomic order)
    DvceArray5D<Real> plain1("gravity_deposit_compare_1",lt.nfld,1,nk,nj,ni);
    DvceArray5D<Real> plain2("gravity_deposit_compare_2",lt.nfld,1,nk,nj,ni);
    deposit(kDepositPlain,plain1);
    deposit(kDepositPlain,plain2);
    const std::size_t nall=field.size();
    const Real *pf=field.data(), *p1=plain1.data(), *p2=plain2.data();
    Real vmax=0, d_mode=0, d_plain=0;
    const auto range=Kokkos::RangePolicy<DevExeSpace>(0,nall);
    Kokkos::parallel_reduce("gravity_deposit_compare",range,KOKKOS_LAMBDA(const std::size_t q,
        Real &a) {a=Kokkos::fmax(a,Kokkos::fabs(p1[q]));},Kokkos::Max<Real>(vmax));
    Kokkos::parallel_reduce("gravity_deposit_compare",range,KOKKOS_LAMBDA(const std::size_t q,
        Real &a) {a=Kokkos::fmax(a,Kokkos::fabs(pf[q]-p1[q]));},Kokkos::Max<Real>(d_mode));
    Kokkos::parallel_reduce("gravity_deposit_compare",range,KOKKOS_LAMBDA(const std::size_t q,
        Real &a) {a=Kokkos::fmax(a,Kokkos::fabs(p2[q]-p1[q]));},Kokkos::Max<Real>(d_plain));
    Real red[3]={vmax,d_mode,d_plain};
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,red,3,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
    if (global_variable::my_rank==0) {
      std::cout << std::setprecision(6) << "# SG_DEPOSIT_COMPARE time=" << time
                << " max_density=" << red[0];
      if (red[0]>0) {
        std::cout << " warp_vs_plain=" << red[1]/red[0] << " plain_vs_plain=" << red[2]/red[0];
      } else {   // no dust mass: absolute differences, the relative ones are undefined
        std::cout << " warp_vs_plain_abs=" << red[1] << " plain_vs_plain_abs=" << red[2]
                  << " relative=n/a";
      }
      std::cout << '\n';
    }
  }
  CheckParticles(bad);
  // Collective even on ranks with no particles: those ranks can own targets.
  HaloReverseAdd(gravity_density_halo_,0,1);
  auto destination=owned;
  par_for("gravity_dust_owned_density",DevExeSpace(),0,nmb-1,
      ind.ks,ind.ke,ind.js,ind.je,ind.is,ind.ie,
      KOKKOS_LAMBDA(int b,int k,int j,int i) {destination(b,0,k,j,i)=field(b,0,k,j,i);});
}

namespace {
bool GatherAllocOld() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_DUST_GATHER_ALLOC");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
}  // namespace

DvceArray2D<Real> DustGasDrag::GravityAccelerationStorage(int npart) {
  gravity_acc_turn_ ^= 1;
  auto &store = gravity_acc_store_[gravity_acc_turn_];
  const std::size_t need = 3*static_cast<std::size_t>(npart > 0 ? npart : 1);
  if (store.extent(0) < need) {
    store = DvceArray1D<Real>();   // release before the larger allocation
    store = DvceArray1D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
        "gravity_particle_acceleration_store"),need + need/8);
  }
  DvceArray2D<Real> v(store.data(),3,npart);
  Kokkos::deep_copy(DevExeSpace(),v,0.0);   // as the zero-initialised allocation it replaces
  return v;
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
  // components innermost for the gathers (the same values, so the same sums in the same order)
  if (gravity_force_pack_.extent_int(0)!=lt.nfld || gravity_force_pack_.extent_int(1)!=nk ||
      gravity_force_pack_.extent_int(2)!=nj || gravity_force_pack_.extent_int(3)!=ni)
    gravity_force_pack_=DvceArray5D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
        "gravity_dust_force_pack"),lt.nfld,nk,nj,ni,4);
  auto fpack=gravity_force_pack_;
  par_for("gravity_force_interleave",DevExeSpace(),0,lt.nfld-1,0,nk-1,0,nj-1,0,ni-1,
      KOKKOS_LAMBDA(int b,int k,int j,int i) {
        for (int d=0;d<3;++d) fpack(b,k,j,i,d)=field(b,d,k,j,i);
        fpack(b,k,j,i,3)=0;
      });
  auto *particles=pmy_pack->ppart;
  const int npart=particles->nprtcl_thispack, gids=pmy_pack->gids;
  if (particle_acceleration.extent_int(0)!=3 ||
      particle_acceleration.extent_int(1)!=npart)
    particle_acceleration=(record && !GatherAllocOld()) ? GravityAccelerationStorage(npart) :
        DvceArray2D<Real>("gravity_particle_acceleration",3,npart);
  auto result=particle_acceleration;
  auto pr=particles->prtcl_rdata;
  auto pi=particles->prtcl_idata;
  auto fp=gravity_force_pack_;
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
        for (int d=0;d<3;++d) acc[d]+=weight*fp(b,k,j,i,d);
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
    // an eighth of headroom above the new high-water mark, as the particle send list: a rank
    // whose particle count keeps growing (a collapsing clump) no longer frees and allocates
    // the snapshot (several GB, a large cost per cycle) at every gather; only the first
    // gravity_gather_nsnap_ columns are used.  ATHENA_DUST_GATHER_SNAP_OLD=1: the exact size.
    static const bool exact = [] {
      const char *e = std::getenv("ATHENA_DUST_GATHER_SNAP_OLD");
      return e != nullptr && e[0] == '1';
    }();
    const int cap = exact ? npart : static_cast<int>(std::min<long long>(
        static_cast<long long>(npart) + npart/8, std::numeric_limits<int>::max()));
    gravity_gather_rsnap_=DvceArray2D<Real>();   // free first: no peak with both copies
    gravity_gather_isnap_=DvceArray2D<int>();
    gravity_gather_rsnap_=DvceArray2D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
        "gravity_gather_rsnap"),4,cap);
    gravity_gather_isnap_=DvceArray2D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
        "gravity_gather_isnap"),2,cap);
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
  if (gravity_force_pack_.extent_int(0)!=lt.nfld || gravity_force_pack_.extent_int(1)!=nk ||
      gravity_force_pack_.extent_int(2)!=nj || gravity_force_pack_.extent_int(3)!=ni)
    GravityTransferFail("selective gravity regather: the packed force does not fit the mesh");
  auto fp=gravity_force_pack_;
  auto *particles=pmy_pack->ppart;
  const int npart=particles->nprtcl_thispack, gids=pmy_pack->gids, nold=gravity_gather_nsnap_;
  auto old=particle_acceleration;
  auto result=old;
  if (npart!=nold) result=GatherAllocOld() ?
      DvceArray2D<Real>("gravity_particle_acceleration",3,npart) : GravityAccelerationStorage(npart);
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
        for (int d=0;d<3;++d) acc[d]+=weight*fp(b,k,j,i,d);
        sum+=weight;
      });
    for (int d=0;d<3;++d) result(d,p)=acc[d];
    if (!Kokkos::isfinite(sum) || fabs(sum-1)>Real(1.e-9)) ++invalid;
    for (int d=0;d<3;++d) if (!Kokkos::isfinite(acc[d])) ++invalid;
  },bad,changed);
  // the old snapshot is read no more: drop these references, so that RecordGravityGather
  // really frees it before it allocates a larger one (else both are held at the peak)
  rs=DvceArray2D<Real>();
  is=DvceArray2D<int>();
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
