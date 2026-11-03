#ifndef SHEARING_BOX_PULL_SLOPE_HPP_
#define SHEARING_BOX_PULL_SLOPE_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pull_slope.hpp
//! \brief One-sided slope of the collected shear ghost zones (ShearingBoxCC::PullSend) at a
//! coarser sender's y / z edge row whose ghost row comes from a still coarser neighbour.

#include "athena.hpp"
#include "mesh/prolongation.hpp"

//----------------------------------------------------------------------------------------
//! \fn Real PullOneSidedSlope()
//! \brief d1, d2: the two interior differences next to the edge, in the order of the
//! two-sided slope (+x2 / +x3 differences); u0: the edge cell; positive: a non-negative
//! variable.  Returns the slope in the units of the pull (4 PROLONG_CC_SLOPE: the change
//! across one sender cell).  For a non-negative variable the magnitude is capped at
//! max(u0, 0), the bound a centred minmod slope with any non-negative ghost satisfies: with
//! |offset| <= 1/4 per direction (one level) every child keeps at least u0/4.  Only the
//! slope is limited, so the mean of the children stays u0; below the cap the slope is
//! unchanged.

KOKKOS_INLINE_FUNCTION
Real PullOneSidedSlope(const Real d1, const Real d2, const Real u0, const bool positive) {
  Real m = 4.0*PROLONG_CC_SLOPE(d1, d2);
  if (positive) {
    const Real c = fmax(u0, 0.0);
    m = fmin(fmax(m, -c), c);
  }
  return m;
}

#endif  // SHEARING_BOX_PULL_SLOPE_HPP_
