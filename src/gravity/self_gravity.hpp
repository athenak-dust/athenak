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
class MeshBoundaryValuesCC;
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
  // Start the HOST assembly of the shear-phase rows for a later Solve(time) in a
  // background thread, overlapping the matter step. The next phase update uses it only if
  // its shift, topology and geometry are still exactly those of the request; otherwise it
  // is discarded and assembled again. The assembly is the same code on the same data, so
  // the operator coefficients are bitwise those of the synchronous path.
  void PrefetchPhase(Real time);
  // kdk_reuse helpers; public only because NVCC device lambdas require it
  bool KdkReuseAllowed(Real time, Real shift);
  void RememberReuseState();
  void ReuseCheck(Real time, long long regathered);
  bool ExtrapolateGuess(Real time, int solver_n);
  void RecordGuess(Real time, bool extrapolation_history);
  // Symmetric half-kicks around the existing hydro/OA update. Density does not
  // change during a kick; ideal-gas internal energy is preserved exactly.
  void KickHydro(Real dt);
  // Both species receive the field sampled by the immediately preceding Solve.
  void KickMatter(Real dt);
  // <gravity>/energy_source = kick (default) | flux | diag.  The half-kicks change
  // the gas energy by the kinetic-energy change of the cell-centred momentum kick, sum rho v.g dt;
  // the potential energy W = 1/2 sum rho phi V changes with the face mass fluxes, so total
  // K + U + W drifts (12.3% / 5.4% at 32^3 / 64^3, all of it this mismatch).  flux
  // replaces, at the end of each step, the gas energy of the two kicks by the face work
  //   -1/2 sum_d [S_{i+1/2}(phibar_{i+1} - phibar_i) + S_{i-1/2}(phibar_i - phibar_{i-1})]/dx_d,
  // S the time-integrated RK mass flux and phibar = (phi^n + phi^{n+1})/2 (the difference goes
  // to the internal energy; K + U + W then holds to the solver tolerance); diag only adds up the
  // difference (energy_mismatch_total).  For now: ideal gas, gas kicked, rk1/rk2/rk3, one
  // refinement level, periodic in every direction (no shearing box, no isolated x3).
  void BeginEnergyStep();
  void AccumulateMassFlux(Real gam0, Real beta_dt);
  void EndEnergyStep();
  int energy_mode = 0;               // 0 kick, 1 diag, 2 flux
  Real energy_mismatch_total = 0;    // sum over steps of sum_cells (face work - kick energy) V
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
  // Topology/load balance rebuilds automatically; shear updates only surface rows.
  void Invalidate();
  void LoadRestartState(GravityRestartState state) { restart_state_ = std::move(state); }
  const GravityRestartState &RestartState() const { return restart_state_; }
  void ClearRestartState() { restart_state_ = GravityRestartState{}; }
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
  struct PhaseAssembly;
  Real ShearShift(Real time) const;
  void AssemblePhase(Real shear_offset, PhaseAssembly &out);
  void JoinPrefetch();
  void Rebuild(Real shear_offset);
  void RebuildDistributed(Real shear_offset);
  void UpdatePhase(Real shear_offset);
  void RefreshHierarchy();
  void AllocateFields(const std::vector<Real> &local_volume,
                      const std::vector<int> &physical_rows, bool keep_guess);
  bool CachedSourceUnchanged(const DvceArray1D<Real> &physical_source);
  void RememberSource(const DvceArray1D<Real> &physical_source);
  void UpdateDustField(Real time);
  MeshBlockPack *pack_;
  std::unique_ptr<CompositeSystem> system_;
  std::unique_ptr<SparseMultigrid> mg_;
  std::unique_ptr<MeshGeometry> geometry_;
  std::vector<std::unique_ptr<MeshGeometry>> coarse_geometry_;
  std::vector<std::vector<int>> prolongation_phase_rows_;
  std::vector<GravityPatch> solver_patches_;
  std::unique_ptr<PhaseAssembly> prefetch_;
  // <gravity>/kdk_reuse (default off): the opening solve of a step keeps the
  // closing field of the previous step when nothing but exact representation changes
  // (split, lobe release, migration, reordering) happened in between; see KdkReuseAllowed.
  bool kdk_reuse_ = false, reuse_check_ = false, reuse_valid_ = false;
  std::uint64_t reuse_epoch_ = 0, reuse_count_ = 0, reuse_regathered_ = 0;
  Real reuse_check_max_residual_ = 0, reuse_check_max_source_ = 0, reuse_check_max_gather_ = 0;
  DvceArray1D<Real> reuse_gas_;  // gas density of the source cells (gas sources only)
  Real last_physical_mass_ = 0;
  // <gravity>/guess = extrapolate (default previous): linear extrapolation in time of
  // the last two accepted potentials as the initial guess of a solve at a new time
  bool guess_extrapolate_ = false, guess_prev_valid_ = false, guess_cur_valid_ = false;
  DvceArray1D<Real> guess_prev_, guess_tmp_;
  Real guess_t_prev_ = 0, guess_t_cur_ = 0;
  int guess_topology_ = -1;
  std::uint64_t guess_extrapolations_ = 0;
  bool prefetch_enabled_ = false;
  std::uint64_t geometry_epoch_ = 0, prefetch_used_ = 0, prefetch_discarded_ = 0;
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
  // the energy_source = flux / diag work arrays and the phibar ghost exchange
  DvceArray5D<Real> energy_phi_old_, energy_phibar_, energy_kick_, energy_mflx_[3], energy_cdummy_;
  MeshBoundaryValuesCC *energy_bval_ = nullptr;
  ParameterInput *pin_ = nullptr;
  DvceArray5D<Real> dust_density_;
  DvceArray2D<Real> particle_acceleration_;
  ExtraDensitySource extra_source_;
  std::vector<GlobalIndex> partition_;
  int topology_ = -1, max_iterations_, restart_;
  int slab_modes_x_ = 0, slab_modes_y_ = 0;
  int phase_threads_ = 0, setup_threads_ = 0;
  bool phase_serial_only_ = false;
  Real phase_ = 0, tolerance_, gravity_cfl_;
  bool phase_valid_ = false, report_each_solve_;
  bool profile_ = false;
  bool field_cache_valid_ = false;
  Real cached_phase_ = 0, cached_four_pi_G_ = 0;
  int cached_topology_ = -1;
  std::uint64_t cached_operator_updates_ = 0;
  bool orbital_split_ = false, hydro_boundary_time_valid_ = false;
  bool coupled_dust_ = false;
  // Gas only / dust only / gas + dust: which matter makes the field
  // (<gravity>/source) and which matter it acts on (<gravity>/kick, default = source)
  bool source_gas_ = true, source_dust_ = false, kick_gas_ = true, kick_dust_ = false;
  Real hydro_boundary_time_ = 0;
  std::string vertical_bc_, setup_, isolated_method_;
  GravityRestartState restart_state_;
};
}  // namespace gravity
#endif  // GRAVITY_SELF_GRAVITY_HPP_
