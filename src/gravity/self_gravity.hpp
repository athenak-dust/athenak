//========================================================================================
// AthenaK: composite AMR self-gravity. BSD-3-Clause; see LICENSE.
//========================================================================================
#ifndef GRAVITY_SELF_GRAVITY_HPP_
#define GRAVITY_SELF_GRAVITY_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "gravity/sparse_mg.hpp"
#include "gravity/mesh_geometry.hpp"
#include "gravity/restart.hpp"

class Driver;
class MeshBlockPack;
class ParameterInput;

namespace gravity {
struct CompositeSystem;
class DistributedOperator;
class SlabBoundary;
class VacuumBoundary;

// A field solver independent of the source representation. Hydro contributes its
// active-cell density; a future dust implementation can supply an additional
// deposited density and gather the resulting acceleration through its own weights.
// Potential and acceleration have valid ACTIVE cells only, not hydro ghost values.
class SelfGravity {
 public:
  SelfGravity(MeshBlockPack *pack, ParameterInput *pin);
  ~SelfGravity();
  SelfGravity(const SelfGravity &) = delete;
  SelfGravity &operator=(const SelfGravity &) = delete;

  void Solve(Real time);
  // Symmetric half-kicks around the existing hydro/OA update. Density does not
  // change during a kick; ideal-gas internal energy is preserved exactly.
  void KickHydro(Real dt);
  void UpdateTimeStep();
  // Gravity-enabled 3D hydro uses symmetric orbital halfsteps around residual
  // RK hydro. This corrects operator splitting; limited PLM remapping can still
  // have spatial errors that depend on timestep at fixed mesh resolution.
  bool UsesOrbitalSplit() const { return orbital_split_; }
  void OrbitalStep(Real dt);
  void SetHydroBoundaryTime(Real time);
  void ClearHydroBoundaryTime() { hydro_boundary_time_valid_ = false; }
  Real HydroBoundaryTime(Real stage_time) const {
    return hydro_boundary_time_valid_ ? hydro_boundary_time_ : stage_time;
  }
  // Rebuild is automatic after topology/load-balance or shear-phase changes.
  void Invalidate();
  void LoadRestartState(GravityRestartState state) { restart_state_ = std::move(state); }
  void ResumeAmrAfterRestart(Driver *driver, ParameterInput *pin);
  bool amr_check_after_output = false;

  using ExtraDensitySource = std::function<void(Real, DvceArray5D<Real> &)>;
  void SetExtraDensitySource(ExtraDensitySource source) {
    extra_source_ = std::move(source);
  }

  DvceArray5D<Real> phi, acceleration;
  Real four_pi_G, dtnew;
  Real last_source_mean = 0, last_time = 0;
  SolveStats last_solve;
  std::uint64_t solve_count = 0, rebuild_count = 0;
  std::uint64_t cache_hit_count = 0;
  bool last_solve_was_cached = false;
  std::uint64_t phase_update_count = 0, preconditioner_refresh_count = 0;

 private:
  void Rebuild(Real shear_offset);
  void RebuildDistributed(Real shear_offset);
  void UpdatePhase(Real shear_offset);
  void RefreshHierarchy();
  void AllocateFields(const std::vector<Real> &local_volume,
                      const std::vector<int> &physical_rows, bool keep_guess);
  bool CachedSourceUnchanged(const DvceArray1D<Real> &physical_source);
  void RememberSource(const DvceArray1D<Real> &physical_source);
  MeshBlockPack *pack_;
  std::unique_ptr<CompositeSystem> system_;
  std::unique_ptr<SparseMultigrid> mg_;
  std::unique_ptr<MeshGeometry> geometry_;
  std::vector<GravityPatch> solver_patches_;
  CompositeDomain solver_domain_;
  bool solver_neumann_z_ = false, hierarchy_fresh_ = true;
  std::unique_ptr<DistributedOperator> force_[3];
  std::unique_ptr<SlabBoundary> slab_;
  std::unique_ptr<VacuumBoundary> vacuum_;
  std::unique_ptr<DistributedOperator> boundary_rhs_op_, boundary_force_[3];
  DvceArray1D<Real> boundary_values_;
  DvceArray1D<Real> rhs_, solution_, volume_, work_;
  DvceArray1D<Real> physical_source_;
  DvceArray1D<int> physical_rows_;
  DvceArray1D<Real> cached_source_;  // physical source cells, independent of MG padding
  DvceArray5D<Real> extra_density_;
  ExtraDensitySource extra_source_;
  std::vector<GlobalIndex> partition_;
  int topology_ = -1, max_iterations_, restart_;
  int slab_modes_x_ = 0, slab_modes_y_ = 0;
  Real phase_ = 0, tolerance_, gravity_cfl_;
  bool phase_valid_ = false, report_each_solve_;
  bool profile_ = false;
  bool field_cache_valid_ = false;
  Real cached_phase_ = 0, cached_four_pi_G_ = 0;
  int cached_topology_ = -1;
  std::uint64_t cached_operator_updates_ = 0;
  bool orbital_split_ = false, hydro_boundary_time_valid_ = false;
  Real hydro_boundary_time_ = 0;
  std::string vertical_bc_, setup_, isolated_method_;
  GravityRestartState restart_state_;
};
}  // namespace gravity
#endif  // GRAVITY_SELF_GRAVITY_HPP_
