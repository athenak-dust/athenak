#ifndef DUST_DUST_SPLIT_VELOCITY_HPP_
#define DUST_DUST_SPLIT_VELOCITY_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_split_velocity.hpp
//! \brief The velocities of the pieces of a P1 split or lobe release with
//! <dust>/cloud_split_velocity = mesh_gradient (dust_split_velocity.cpp): the helpers the
//! fill kernels of the device splits call.

#include "athena.hpp"

namespace dust {

// the centroid of a cloud along one direction relative to its position: a transition lobe
// (1 lower, 2 upper) is the shape (2/3) S_ell(x - x_p) + (1/3) S_ell(x - x_p -/+ ell), whose
// centroid is -/+ ell/3; no lobe and both lobes (the hat of half-width 2 ell) are symmetric
KOKKOS_INLINE_FUNCTION
Real SplitLobeShift(const int lobe, const Real ell) {
  return (lobe == 2) ? ell/3.0 : ((lobe == 1) ? -ell/3.0 : 0.0);
}

// piece velocity: v += G d for the event of row `row` of g (G_ij at g(row, 3 i + j), the
// parent's mean-flow gradient, zero where the split falls back to the parent's velocity);
// d the offset of the piece's centroid from the parent's; zsign = -1 for a piece mirrored at a
// reflecting x3 face (its vz was flipped).  Both the velocity and its RK register.
template <class PR, class GV>
KOKKOS_INLINE_FUNCTION
void SplitAddGradient(const PR &pr, const int s, const GV &g, const int row, const Real d0,
                      const Real d1, const Real d2, const Real zsign) {
  const Real dvx = g(row,0)*d0 + g(row,1)*d1 + g(row,2)*d2;
  const Real dvy = g(row,3)*d0 + g(row,4)*d1 + g(row,5)*d2;
  const Real dvz = zsign*(g(row,6)*d0 + g(row,7)*d1 + g(row,8)*d2);
  pr(IPVX,s) += dvx; pr(IPVX1,s) += dvx;
  pr(IPVY,s) += dvy; pr(IPVY1,s) += dvy;
  pr(IPVZ,s) += dvz; pr(IPVZ1,s) += dvz;
}

}  // namespace dust

#endif  // DUST_DUST_SPLIT_VELOCITY_HPP_
