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

class Driver;
class MeshBlockPack;
class ParameterInput;

namespace gravity {
struct CompositeSystem;
class DistributedOperator;
class SlabBoundary;

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
  // Rebuild is automatic after topology/load-balance or shear-phase changes.
  void Invalidate();

  using ExtraDensitySource = std::function<void(Real, DvceArray5D<Real> &)>;
  void SetExtraDensitySource(ExtraDensitySource source) {
    extra_source_ = std::move(source);
  }

  DvceArray5D<Real> phi, acceleration;
  Real four_pi_G, dtnew;
  Real last_source_mean = 0, last_time = 0;
  SolveStats last_solve;
  std::uint64_t solve_count = 0, rebuild_count = 0;

 private:
  void Rebuild(Real shear_offset);
  MeshBlockPack *pack_;
  std::unique_ptr<CompositeSystem> system_;
  std::unique_ptr<SparseMultigrid> mg_;
  std::unique_ptr<DistributedOperator> force_[3];
  std::unique_ptr<SlabBoundary> slab_;
  std::unique_ptr<DistributedOperator> boundary_rhs_op_, boundary_force_[3];
  DvceArray1D<Real> boundary_values_;
  DvceArray1D<Real> rhs_, solution_, volume_, work_;
  DvceArray5D<Real> extra_density_;
  ExtraDensitySource extra_source_;
  std::vector<int> partition_;
  int topology_ = -1, max_iterations_, restart_;
  int slab_modes_x_ = 0, slab_modes_y_ = 0;
  Real phase_ = 0, tolerance_, gravity_cfl_;
  bool phase_valid_ = false, report_each_solve_;
  std::string vertical_bc_;
};
}  // namespace gravity
#endif  // GRAVITY_SELF_GRAVITY_HPP_
