//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_VACUUM_BOUNDARY_HPP_
#define GRAVITY_VACUUM_BOUNDARY_HPP_

#include <cstddef>

#include "gravity/vacuum_mesh.hpp"

namespace gravity {

// Source-free exterior boundary/gauge helper. Owns boundary-size DEVICE metadata
// only; no source modes, global cells, or direct source-by-boundary sums. The
// supplied communicator is borrowed and must remain valid for this object's life.
class VacuumBoundary {
 public:
#if MPI_PARALLEL_ENABLED
  VacuumBoundary(const VacuumMesh &mesh,const LocalCompositeSystem &fine,int rank,
                 MPI_Comm comm=MPI_COMM_WORLD);
#else
  VacuumBoundary(const VacuumMesh &mesh,const LocalCompositeSystem &fine,int rank);
#endif
  VacuumBoundary(const VacuumBoundary &)=delete;
  VacuumBoundary &operator=(const VacuumBoundary &)=delete;

  // global_physical_mass is the already-reduced integral of PHYSICAL density,
  // without mean subtraction. Every sample gets outward dphi/dn=q, where
  // q=four_pi_G*M/(2A). Empty owned sample vectors are valid. Kernel-bearing
  // methods are public so NVCC can instantiate their extended device lambdas.
  void GPUFill(Real four_pi_G,Real global_physical_mass,
               const DvceArray1D<Real> &local_boundary_values);
  // Collective; GPUFill must have supplied the current source mass first.
  // Return <phi_adj+q*h/2>_both_faces - q*Lextended/2. The caller subtracts
  // this scalar from ALL owned solver rows, including vacuum rows. This fixes
  // the Green zero-mode gauge without changing force or subtracting source mass.
  Real GaugeShift(const DvceArray1D<Real> &solution_local) const;

  int LocalSamples() const { return adjacent_.extent_int(0); }
  Real Flux() const { return q_; }
  Real Area() const { return area_; }
  std::size_t DeviceMetadataBytes() const;

 private:
  DvceArray1D<int> adjacent_;
  DvceArray1D<Real> area_weights_,widths_;
  Real area_=0,extended_length_=0,q_=0;
  int owned_rows_=0;
  bool filled_=false;
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_;
#endif
};

}  // namespace gravity
#endif  // GRAVITY_VACUUM_BOUNDARY_HPP_
