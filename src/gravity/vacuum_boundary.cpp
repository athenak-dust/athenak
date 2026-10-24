//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/vacuum_boundary.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "globals.hpp"

namespace gravity {

VacuumBoundary::VacuumBoundary(const VacuumMesh &mesh,const LocalCompositeSystem &fine,
    int rank
#if MPI_PARALLEL_ENABLED
    ,MPI_Comm comm
#endif
    ) {
#if MPI_PARALLEL_ENABLED
  comm_=comm;
  int actual_rank,nranks;
  MPI_Comm_rank(comm_,&actual_rank); MPI_Comm_size(comm_,&nranks);
  if (rank!=actual_rank || mesh.row_partition.size()!=static_cast<std::size_t>(nranks+1))
    throw std::invalid_argument("Gravity vacuum boundary communicator/partition mismatch");
#endif
  if (!mesh.neumann_z || !fine.neumann_z || !mesh.domain.isolated_z ||
      mesh.domain.ndim!=3 || rank<0 || rank+1>=static_cast<int>(mesh.row_partition.size()))
    throw std::invalid_argument("Gravity vacuum boundary requires owned3D Neumann metadata");
  const GlobalIndex begin=mesh.row_partition[rank],end=mesh.row_partition[rank+1];
  if (end<begin || end-begin>std::numeric_limits<int>::max() ||
      fine.cells.size()!=static_cast<std::size_t>(end-begin) ||
      (!fine.row_partition.empty() && fine.row_partition!=mesh.row_partition))
    throw std::invalid_argument("Gravity vacuum boundary owned row range mismatch");
  owned_rows_=static_cast<int>(end-begin);
  const std::size_t count=fine.boundary_samples.size();
  if (count>static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      fine.boundary_cells.size()!=count || fine.boundary_area.size()!=count ||
      fine.boundary_width.size()!=count)
    throw std::invalid_argument("Gravity vacuum boundary sample metadata mismatch");
  area_=(mesh.domain.hi[0]-mesh.domain.lo[0])*(mesh.domain.hi[1]-mesh.domain.lo[1]);
  extended_length_=mesh.domain.hi[2]-mesh.domain.lo[2];
  if (!(area_>0) || !(extended_length_>0) || !std::isfinite(area_) ||
      !std::isfinite(extended_length_))
    throw std::invalid_argument("Gravity vacuum boundary invalid domain area/length");
  adjacent_=DvceArray1D<int>("vacuum_boundary_adjacent",count);
  area_weights_=DvceArray1D<Real>("vacuum_boundary_area",count);
  widths_=DvceArray1D<Real>("vacuum_boundary_width",count);
  auto ah=Kokkos::create_mirror_view(adjacent_);
  auto wh=Kokkos::create_mirror_view(area_weights_);
  auto hh=Kokkos::create_mirror_view(widths_);
  double local_area=0.,compensation=0.;
  for (std::size_t i=0; i<count; ++i) {
    const GlobalIndex cell=fine.boundary_cells[i]-begin;
    const Real weight=fine.boundary_area[i],width=fine.boundary_width[i];
    if (cell<0 || cell>=owned_rows_ || !(weight>0) || !(width>0) ||
        !std::isfinite(weight) || !std::isfinite(width))
      throw std::invalid_argument("Gravity vacuum boundary invalid owned sample");
    ah(i)=static_cast<int>(cell); wh(i)=weight; hh(i)=width;
    const double y=static_cast<double>(weight)-compensation,next=local_area+y;
    compensation=(next-local_area)-y; local_area=next;
  }
  double total_area=local_area;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local_area,&total_area,1,MPI_DOUBLE,MPI_SUM,comm_);
#endif
  const double expected=2*static_cast<double>(area_);
  // Relative to the physical area: an absolute unit-area floor would silently
  // weaken coverage validation when the domain is small in the chosen units.
  const double tolerance=256*std::numeric_limits<Real>::epsilon()*expected
                         +std::numeric_limits<Real>::min();
  if (std::abs(total_area-expected)>tolerance)
    throw std::invalid_argument("Gravity vacuum boundary global face area is not2A");
  Kokkos::deep_copy(adjacent_,ah); Kokkos::deep_copy(area_weights_,wh);
  Kokkos::deep_copy(widths_,hh);
}

void VacuumBoundary::GPUFill(Real four_pi_G,Real global_physical_mass,
    const DvceArray1D<Real> &values) {
  if (values.extent(0)!=adjacent_.extent(0) || !std::isfinite(four_pi_G) ||
      !std::isfinite(global_physical_mass))
    throw std::invalid_argument("Gravity vacuum boundary invalid flux fill arguments");
  q_=four_pi_G*global_physical_mass/(2*area_);
  if (!std::isfinite(q_)) throw std::overflow_error("Gravity vacuum boundary nonfinite flux");
  const Real q=q_; const int count=adjacent_.extent_int(0);
  Kokkos::parallel_for("vacuum_boundary_flux",Kokkos::RangePolicy<>(0,count),
      KOKKOS_LAMBDA(int i) { values(i)=q; });
  filled_=true;
}

Real VacuumBoundary::GaugeShift(const DvceArray1D<Real> &solution) const {
  if (!filled_ || solution.extent(0)!=static_cast<std::size_t>(owned_rows_))
    throw std::invalid_argument("Gravity vacuum gauge requires current flux/owned solution");
  const auto adjacent=adjacent_; const auto weights=area_weights_,widths=widths_;
  const Real q=q_; const int count=adjacent.extent_int(0); Real local=0;
  Kokkos::parallel_reduce("vacuum_boundary_gauge",Kokkos::RangePolicy<>(0,count),
      KOKKOS_LAMBDA(int i,Real &sum) {
        sum+=weights(i)*(solution(adjacent(i))+q*widths(i)/2);
      },local);
  Real global=local;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(&local,&global,1,MPI_ATHENA_REAL,MPI_SUM,comm_);
#endif
  return global/(2*area_)-q_*extended_length_/2;
}

std::size_t VacuumBoundary::DeviceMetadataBytes() const {
  return adjacent_.size()*sizeof(int)+(area_weights_.size()+widths_.size())*sizeof(Real);
}

}  // namespace gravity
