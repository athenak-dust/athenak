//========================================================================================
// AthenaK: horizontally periodic/sheared, vertically isolated Poisson boundary.
// BSD-3-Clause; see LICENSE.
//========================================================================================
#ifndef GRAVITY_SLAB_BOUNDARY_HPP_
#define GRAVITY_SLAB_BOUNDARY_HPP_

#include <vector>
#include "gravity/composite.hpp"

namespace gravity {
// Direct horizontal Fourier sums provide a source-based reference boundary, not
// a substitute uniform-grid volume solve. The zero mode retains the full slab
// mass. Cost is O(Ncell * Nmode); an FFT/FMM boundary backend is future work.
class SlabBoundary {
 public:
  SlabBoundary(const std::vector<CompositeCell> &cells, const CompositeDomain &domain,
               const std::vector<BoundarySample> &samples,
               const std::vector<int> &cell_partition,
               const std::vector<int> &sample_partition, int modes_x, int modes_y);
  void Compute(const DvceArray1D<Real> &density, Real four_pi_G,
               const DvceArray1D<Real> &boundary_values);
  int Modes() const { return nmodes_; }

 private:
  DvceArray2D<Real> cells_, samples_, wave_, coefficients_;
  Real zlo_, zhi_, area_;
  int nmodes_;
};
}  // namespace gravity
#endif  // GRAVITY_SLAB_BOUNDARY_HPP_
