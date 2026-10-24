//========================================================================================
// AthenaK: horizontally periodic/sheared, vertically isolated Poisson boundary.
// BSD-3-Clause; see LICENSE.
//========================================================================================
#ifndef GRAVITY_SLAB_BOUNDARY_HPP_
#define GRAVITY_SLAB_BOUNDARY_HPP_

#include <array>
#include <vector>
#include "gravity/composite.hpp"

namespace gravity {
// Direct horizontal Fourier sums provide a source-based reference boundary, not
// a substitute uniform-grid volume solve. The zero mode retains the full slab
// mass. Cost is O(Ncell * Nmode); this remains a bounded accuracy reference.
// The production isolated path uses the distributed, coarsened vacuum domain.
class SlabBoundary {
 public:
  SlabBoundary(const std::vector<CompositeCell> &cells, const CompositeDomain &domain,
               const std::vector<BoundarySample> &samples,
               const std::vector<int> &cell_partition,
               const std::vector<int> &sample_partition, int modes_x, int modes_y);
  // Owned-only reference constructor. min_width is GLOBAL, computed from patch
  // metadata or a reduction, so all ranks use the same Fourier coefficient set.
  // Empty owned cells/samples are valid; no global leaf array is required.
  SlabBoundary(const std::vector<CompositeCell> &owned_cells,
               const CompositeDomain &domain,
               const std::vector<BoundarySample> &owned_samples,
               const std::array<Real,2> &min_width, int modes_x, int modes_y);
  void Compute(const DvceArray1D<Real> &density, Real four_pi_G,
               const DvceArray1D<Real> &boundary_values);
  // Update only the horizontal mode wavevectors; retain owned source/sample data.
  // This direct boundary remains a reference backend, not the production vacuum path.
  void SetShearOffset(Real shear_offset);
  int Modes() const { return nmodes_; }

 private:
  void InitializeModes(const CompositeDomain &domain,
                       const std::array<Real,2> &min_width, int modes_x, int modes_y);
  DvceArray2D<Real> cells_, samples_, wave_, coefficients_;
  Real zlo_, zhi_, area_, lx_, ly_;
  int nmodes_, modes_x_, modes_y_;
};
}  // namespace gravity
#endif  // GRAVITY_SLAB_BOUNDARY_HPP_
