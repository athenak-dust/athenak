//========================================================================================
// AthenaK: composite AMR self-gravity. BSD-3-Clause; see LICENSE.
//========================================================================================
#include "gravity/self_gravity.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "eos/eos.hpp"
#include "globals.hpp"
#include "gravity/composite.hpp"
#include "gravity/slab_boundary.hpp"
#include "hydro/hydro.hpp"
#include "parameter_input.hpp"
#include "shearing_box/shearing_box.hpp"
#include "shearing_box/orbital_advection.hpp"

namespace gravity {
namespace {
[[noreturn]] void Fail(const std::string &message) {
  std::cerr << "### FATAL ERROR in self-gravity: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
Real Sum(Real value) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_ATHENA_REAL, MPI_SUM, MPI_COMM_WORLD);
#endif
  return value;
}
}  // namespace

SelfGravity::SelfGravity(MeshBlockPack *pack, ParameterInput *pin) : pack_(pack) {
  auto *pm = pack->pmesh;
  if (pack->phydro == nullptr || pack->pmhd != nullptr || pack->pdust != nullptr ||
      pack->ppart != nullptr ||
      pack->prad != nullptr || pack->pcoord->is_special_relativistic ||
      pack->pcoord->is_general_relativistic || pack->pcoord->is_dynamical_relativistic ||
      pm->nmb_packs_thisrank != 1 || pm->one_d) {
    Fail("currently requires Newtonian 2D/3D hydro, one pack per rank; dust coupling "
         "is a future consumer of the source/acceleration API");
  }
  if (pin->GetOrAddString("gravity", "solver", "multigrid") != "multigrid") {
    Fail("solver must be multigrid");
  }
  const auto evolution = pin->GetString("time", "evolution");
  if (evolution != "dynamic" && evolution != "static") {
    Fail("requires dynamic or static evolution");
  }
  if (evolution == "dynamic") {
    const auto integrator = pin->GetOrAddString("time", "integrator", "rk2");
    if (integrator != "rk1" && integrator != "rk2" && integrator != "rk3") {
      Fail("gravity half-kicks currently support rk1/rk2/rk3 only");
    }
  }
  auto *orbital = pack->phydro->porb_u;
  if (orbital != nullptr) {
    if (pack->phydro->psbox_u == nullptr ||
        (!pm->three_d && orbital->shearing_box_r_phi)) {
      Fail("gravity orbital splitting requires a supported 3D shearing box");
    }
    // The existing 2D r-z box has Coriolis terms but no orbital remap.
    orbital_split_ = pm->three_d && evolution == "dynamic";
  }
  four_pi_G = pin->GetReal("gravity", "four_pi_G");
  tolerance_ = pin->GetOrAddReal("gravity", "rtol", 1.0e-10);
  gravity_cfl_ = pin->GetOrAddReal("gravity", "cfl", 0.25);
  max_iterations_ = pin->GetOrAddInteger("gravity", "max_iterations", 150);
  restart_ = pin->GetOrAddInteger("gravity", "gmres_restart", 24);
  report_each_solve_ = pin->GetOrAddBoolean("gravity", "report_each_solve", false);
  vertical_bc_ = pin->GetOrAddString("gravity", "vertical_bc", "periodic");
  if (!std::isfinite(four_pi_G) || four_pi_G <= 0 || !std::isfinite(tolerance_) ||
      tolerance_ <= 0 || tolerance_ >= 1 || !std::isfinite(gravity_cfl_) ||
      gravity_cfl_ <= 0 || gravity_cfl_ > 0.5 || max_iterations_ < 1 || restart_ < 2) {
    Fail("invalid gravitational constant, tolerance, iteration count, or timestep factor");
  }
  if (vertical_bc_ != "periodic" && vertical_bc_ != "isolated") {
    Fail("gravity/vertical_bc must be periodic or isolated");
  }
  if (vertical_bc_ == "isolated") {
    if (!pm->three_d) Fail("vertical isolated gravity requires 3D (x3 is vertical)");
    slab_modes_x_ = pin->GetOrAddInteger("gravity", "slab_modes_x", 0);
    slab_modes_y_ = pin->GetOrAddInteger("gravity", "slab_modes_y", 0);
    if (slab_modes_x_ < 0 || slab_modes_y_ < 0) Fail("negative slab mode cutoff");
  }
  for (int d = 0; d < (pm->three_d ? 3 : 2); ++d) {
    if (d == 2 && vertical_bc_ == "isolated") continue;
    const auto left = pm->mesh_bcs[2*d], right = pm->mesh_bcs[2*d+1];
    const bool periodic = left == BoundaryFlag::periodic && right == BoundaryFlag::periodic;
    const bool shear = d == 0 && pack->phydro->psbox_u != nullptr &&
        left == BoundaryFlag::shear_periodic && right == BoundaryFlag::shear_periodic;
    if (!periodic && !shear) Fail("unsupported gravity boundary in direction " +
                                std::to_string(d+1));
  }
  dtnew = std::numeric_limits<Real>::max();
}

SelfGravity::~SelfGravity() {
  if (global_variable::my_rank == 0) {
    std::cout << std::setprecision(16) << "# SELF_GRAVITY_SUMMARY solves=" << solve_count
              << " rebuilds=" << rebuild_count << " iterations=" << last_solve.iterations
              << " relative_residual=" << last_solve.relative_residual
              << " source_mean=" << last_source_mean << " time=" << last_time << '\n';
  }
}

void SelfGravity::Invalidate() {
  topology_ = -1;
  phase_valid_ = false;
}

void SelfGravity::SetHydroBoundaryTime(Real time) {
  if (!std::isfinite(time)) Fail("nonfinite hydro boundary time");
  hydro_boundary_time_ = time;
  hydro_boundary_time_valid_ = true;
}

void SelfGravity::OrbitalStep(Real dt) {
  if (!UsesOrbitalSplit() || !std::isfinite(dt) || dt <= 0) {
    Fail("invalid gravity orbital halfstep");
  }
  auto *hydro = pack_->phydro;
  auto *orbital = hydro->porb_u;
  auto *pm = pack_->pmesh;
  const Real full_dt = pm->dt;
  // The public remap reads pm->dt, but never pm->time. Restrict this temporary
  // change to the completed remap and restore it before residual RK stages.
  pm->dt = dt;
  auto finish = [](auto operation) {
    TaskStatus status;
    do { status = operation(); } while (status == TaskStatus::incomplete);
    if (status != TaskStatus::complete) Fail("orbital halfstep task failed");
  };
  finish([&]() { return orbital->InitRecv(); });
  finish([&]() { return orbital->PackAndSendCC(hydro->u0); });
  finish([&]() { return orbital->RecvAndUnpackCC(hydro->u0, hydro->recon_method); });
  // Both halfsteps retain the native conservative correction at mixed x2 faces.
  finish([&]() { return orbital->SendFluxCC(); });
  finish([&]() { return orbital->RecvFluxCC(hydro->u0); });
  finish([&]() { return orbital->ClearSend(); });
  finish([&]() { return orbital->ClearRecv(); });
  Kokkos::fence();
  pm->dt = full_dt;
}

void SelfGravity::Rebuild(Real shear_offset) {
  auto *pm = pack_->pmesh;
  const auto ind = pm->mb_indcs;
  const int nc = ind.nx1*ind.nx2*ind.nx3;
  if (pm->nmb_total > std::numeric_limits<int>::max()/nc) {
    Fail("composite cell indexing exceeds the current 32-bit CSR capacity");
  }
  const int n = pm->nmb_total*nc;
  CompositeDomain domain;
  domain.ndim = pm->three_d ? 3 : 2;
  domain.lo = {pm->mesh_size.x1min, pm->mesh_size.x2min, pm->mesh_size.x3min};
  domain.hi = {pm->mesh_size.x1max, pm->mesh_size.x2max, pm->mesh_size.x3max};
  domain.root_counts = {pm->mesh_indcs.nx1, pm->mesh_indcs.nx2, pm->mesh_indcs.nx3};
  domain.shear_offset = shear_offset;
  domain.isolated_z = vertical_bc_ == "isolated";
  const int mbnc[3] = {ind.nx1, ind.nx2, ind.nx3};
  std::vector<CompositeCell> cells(n);
  for (int m = 0; m < pm->nmb_total; ++m) {
    const auto loc = pm->lloc_eachmb[m];
    const int ell = loc.level-pm->root_level;
    const int block[3] = {loc.lx1, loc.lx2, loc.lx3};
    for (int k = 0; k < ind.nx3; ++k) {
      for (int j = 0; j < ind.nx2; ++j) {
        for (int i = 0; i < ind.nx1; ++i) {
          const int q = m*nc + (k*ind.nx2+j)*ind.nx1+i;
          const int ijk[3] = {i,j,k};
          for (int d = 0; d < 3; ++d) {
            if (d >= domain.ndim) {
              cells[q].width[d] = domain.hi[d]-domain.lo[d];
              cells[q].center[d] = 0.5*(domain.lo[d]+domain.hi[d]);
            } else {
              const Real h = std::ldexp((domain.hi[d]-domain.lo[d]) /
                                       domain.root_counts[d], -ell);
              cells[q].width[d] = h;
              cells[q].center[d] = domain.lo[d]+(block[d]*mbnc[d]+ijk[d]+0.5)*h;
            }
          }
        }
      }
    }
  }
  partition_.resize(global_variable::nranks+1);
  for (int r = 0; r < global_variable::nranks; ++r) {
    partition_[r] = pm->gids_eachrank[r]*nc;
  }
  partition_.back() = n;
  const int begin = partition_[global_variable::my_rank];
  const int local = pack_->nmb_thispack*nc;
  for (int m = 0; m < pack_->nmb_thispack; ++m) {
    if (pack_->pmb->mb_gid.h_view(m)*nc != begin+m*nc) {
      Fail("local block layout does not match the global contiguous cell partition");
    }
  }
  try {
    system_ = std::make_unique<CompositeSystem>(
        BuildCompositeSystem(cells, domain, partition_));
    mg_ = std::make_unique<SparseMultigrid>(system_->hierarchy);
    for (int d = 0; d < domain.ndim; ++d) {
      force_[d] = std::make_unique<DistributedOperator>(system_->acceleration[d],
                                                       partition_, partition_);
    }
    if (domain.isolated_z) {
      const int nb = system_->boundary_samples.size();
      std::vector<int> bp(global_variable::nranks+1);
      for (int r = 0; r <= global_variable::nranks; ++r) {
        bp[r] = static_cast<long long>(nb)*r/global_variable::nranks;
      }
      slab_ = std::make_unique<SlabBoundary>(cells,domain,system_->boundary_samples,
                                            partition_,bp,slab_modes_x_,slab_modes_y_);
      boundary_values_ = DvceArray1D<Real>("sg_slab_boundary",
          bp[global_variable::my_rank+1]-bp[global_variable::my_rank]);
      boundary_rhs_op_ = std::make_unique<DistributedOperator>(system_->boundary_rhs,
                                                               partition_,bp);
      for (int d = 0; d < 3; ++d) {
        boundary_force_[d] = std::make_unique<DistributedOperator>(
            system_->boundary_acceleration[d],partition_,bp);
      }
    }
  } catch (const std::exception &e) {
    Fail(e.what());
  }
  rhs_ = DvceArray1D<Real>("sg_rhs",local);
  solution_ = DvceArray1D<Real>("sg_phi",local);
  work_ = DvceArray1D<Real>("sg_work",local);
  volume_ = DvceArray1D<Real>("sg_volume",local);
  auto vol = Kokkos::create_mirror_view(volume_);
  for (int q = 0; q < local; ++q) vol(q) = system_->volume[begin+q];
  Kokkos::deep_copy(volume_,vol);
  const int ni = ind.nx1+2*ind.ng;
  const int nj = ind.nx2+2*ind.ng;
  const int nk = pm->three_d ? ind.nx3+2*ind.ng : 1;
  phi = DvceArray5D<Real>("sg_phi_pack",pack_->nmb_thispack,1,nk,nj,ni);
  acceleration = DvceArray5D<Real>("sg_acc_pack",pack_->nmb_thispack,3,nk,nj,ni);
  extra_density_ = DvceArray5D<Real>("sg_extra_density",pack_->nmb_thispack,1,nk,nj,ni);
  topology_ = pm->GetAMRLoadBalanceUpdateSeq();
  phase_ = shear_offset;
  phase_valid_ = true;
  ++rebuild_count;
}

void SelfGravity::Solve(Real time) {
  auto *pm = pack_->pmesh;
  Real shift = 0;
  if (pm->mesh_bcs[0] == BoundaryFlag::shear_periodic) {
    const auto *ps = pack_->phydro->psbox_u;
    const Real ly = pm->mesh_size.x2max-pm->mesh_size.x2min;
    shift = std::fmod(-ps->qshear*ps->omega0*
                     (pm->mesh_size.x1max-pm->mesh_size.x1min)*time, ly);
  }
  if (!phase_valid_ || topology_ != pm->GetAMRLoadBalanceUpdateSeq() || shift != phase_) {
    Rebuild(shift);
  }
  const auto ind = pm->mb_indcs;
  const int n = pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  auto u = pack_->phydro->u0;
  auto extra = extra_density_;
  Kokkos::deep_copy(extra,0.0);
  if (extra_source_) extra_source_(time,extra);
  auto volume = volume_;
  auto b = rhs_;
  Real local_mass = 0;
  Kokkos::parallel_reduce("sg_source_mass",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q, Real &sum) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    sum += (u(t,IDN,k,j,i)+extra(t,0,k,j,i))*volume(q);
  },local_mass);
  Real total_volume = (pm->mesh_size.x1max-pm->mesh_size.x1min)*
                      (pm->mesh_size.x2max-pm->mesh_size.x2min);
  if (pm->three_d) total_volume *= pm->mesh_size.x3max-pm->mesh_size.x3min;
  last_source_mean = Sum(local_mass)/total_volume;
  const Real mean = slab_ ? 0 : last_source_mean, fourpig = four_pi_G;
  auto work = work_;
  Kokkos::parallel_for("sg_source",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    work(q) = u(t,IDN,k,j,i)+extra(t,0,k,j,i);
    b(q) = -fourpig*(work(q)-mean)*volume(q);
  });
  if (slab_) {
    slab_->Compute(work_,four_pi_G,boundary_values_);
    boundary_rhs_op_->Apply(boundary_values_,work_);
    Kokkos::parallel_for("sg_add_slab_rhs",Kokkos::RangePolicy<DevExeSpace>(0,n),
    KOKKOS_LAMBDA(const int q) { b(q) += work(q); });
  }
  Kokkos::deep_copy(solution_,0.0);
  try {
    last_solve = mg_->Solve(rhs_,solution_,tolerance_,max_iterations_,restart_);
  } catch (const std::exception &e) {
    Fail(e.what());
  }
  if (!last_solve.converged || !std::isfinite(last_solve.relative_residual)) {
    Fail("multigrid did not reach the requested residual; refusing a stale gravity field");
  }
  auto sol = solution_;
  Real local_phi = 0;
  Kokkos::parallel_reduce("sg_volume_gauge",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q, Real &sum) { sum += sol(q)*volume(q); },local_phi);
  const Real gauge = slab_ ? 0 : Sum(local_phi)/total_volume;
  auto potential = phi;
  Kokkos::parallel_for("sg_store_phi",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    sol(q) -= gauge;
    potential(t,0,k,j,i) = sol(q);
  });
  auto acc = acceleration;
  Kokkos::deep_copy(acc,0.0);
  for (int d = 0; d < (pm->three_d ? 3 : 2); ++d) {
    force_[d]->Apply(solution_,work_);
    Kokkos::parallel_for("sg_store_acceleration",Kokkos::RangePolicy<DevExeSpace>(0,n),
    KOKKOS_LAMBDA(const int q) {
      int t = q;
      const int i = t%ind.nx1+ind.is; t /= ind.nx1;
      const int j = t%ind.nx2+ind.js; t /= ind.nx2;
      const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
      acc(t,d,k,j,i) = work(q);
    });
    if (slab_) {
      boundary_force_[d]->Apply(boundary_values_,work_);
      Kokkos::parallel_for("sg_add_slab_force",Kokkos::RangePolicy<DevExeSpace>(0,n),
      KOKKOS_LAMBDA(const int q) {
        int t = q;
        const int i = t%ind.nx1+ind.is; t /= ind.nx1;
        const int j = t%ind.nx2+ind.js; t /= ind.nx2;
        const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
        acc(t,d,k,j,i) += work(q);
      });
    }
  }
  ++solve_count;
  last_time = time;
  UpdateTimeStep();
  if (report_each_solve_ && global_variable::my_rank == 0) {
    std::cout << std::setprecision(16) << "# SELF_GRAVITY time=" << time
              << " cells=" << system_->volume.size() << " levels=" << mg_->Levels()
              << " iterations=" << last_solve.iterations << " residual="
              << last_solve.relative_residual << '\n';
  }
}

void SelfGravity::KickHydro(Real dt) {
  const auto ind = pack_->pmesh->mb_indcs;
  const int n = pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  auto u = pack_->phydro->u0;
  auto acc = acceleration;
  const bool energy = pack_->phydro->peos->eos_data.is_ideal;
  Kokkos::parallel_for("sg_hydro_kick",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    const Real rho = u(t,IDN,k,j,i);
    Real de = 0;
    for (int d = 0; d < 3; ++d) {
      const Real dv = dt*acc(t,d,k,j,i);
      de += dv*(u(t,IM1+d,k,j,i)+0.5*rho*dv);
      u(t,IM1+d,k,j,i) += rho*dv;
    }
    if (energy) u(t,IEN,k,j,i) += de;
  });
}

void SelfGravity::UpdateTimeStep() {
  const auto ind = pack_->pmesh->mb_indcs;
  const int ndim = pack_->pmesh->three_d ? 3 : 2;
  const int n = pack_->nmb_thispack*ind.nx1*ind.nx2*ind.nx3;
  auto u = pack_->phydro->u0;
  auto extra = extra_density_;
  auto acc = acceleration;
  auto size = pack_->pmb->mb_size.d_view;
  const Real fourpig = four_pi_G;
  Real limit = std::numeric_limits<Real>::max();
  Kokkos::parallel_reduce("sg_timestep",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q, Real &minimum) {
    int t = q;
    const int i = t%ind.nx1+ind.is; t /= ind.nx1;
    const int j = t%ind.nx2+ind.js; t /= ind.nx2;
    const int k = t%ind.nx3+ind.ks; t /= ind.nx3;
    const Real density = fabs(u(t,IDN,k,j,i)+extra(t,0,k,j,i));
    if (density > 0) minimum = fmin(minimum,1.0/sqrt(fourpig*density));
    const Real dx[3] = {size(t).dx1,size(t).dx2,size(t).dx3};
    for (int d = 0; d < ndim; ++d) {
      const Real a = fabs(acc(t,d,k,j,i));
      if (a > 0) minimum = fmin(minimum,sqrt(dx[d]/a));
    }
  },Kokkos::Min<Real>(limit));
  dtnew = gravity_cfl_*limit;
}
}  // namespace gravity
