//========================================================================================
// AthenaK: horizontally periodic/sheared, vertically isolated Poisson boundary.
// BSD-3-Clause; see LICENSE.
//========================================================================================
#include "gravity/slab_boundary.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "globals.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace gravity {
struct SlabMoments {
  Real pc, ps, mc, ms;
  KOKKOS_INLINE_FUNCTION SlabMoments() : pc(0), ps(0), mc(0), ms(0) {}
  KOKKOS_INLINE_FUNCTION SlabMoments &operator+=(const SlabMoments &b) {
    pc += b.pc; ps += b.ps; mc += b.mc; ms += b.ms;
    return *this;
  }
};
}  // namespace gravity
namespace Kokkos {
template<> struct reduction_identity<gravity::SlabMoments> {
  KOKKOS_FORCEINLINE_FUNCTION static gravity::SlabMoments sum() {
    return gravity::SlabMoments();
  }
};
}  // namespace Kokkos

namespace gravity {
namespace {
KOKKOS_INLINE_FUNCTION Real Sinc(Real x) {
  return fabs(x) < 1.e-7 ? 1-x*x/6 : sin(x)/x;
}
void GlobalSum(Real *data, int count) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,data,count,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
}
}  // namespace

SlabBoundary::SlabBoundary(const std::vector<CompositeCell> &cells,
    const CompositeDomain &domain, const std::vector<BoundarySample> &samples,
    const std::vector<int> &cell_partition, const std::vector<int> &sample_partition,
    int modes_x, int modes_y) {
  if (!domain.isolated_z || domain.ndim != 3 || samples.empty()) {
    throw std::invalid_argument("slab gravity boundary requires 3D isolated z samples");
  }
  const int rank = global_variable::my_rank;
  const int begin = cell_partition.at(rank), end = cell_partition.at(rank+1);
  const int sbegin = sample_partition.at(rank), send = sample_partition.at(rank+1);
  cells_ = DvceArray2D<Real>("slab_cells",end-begin,6);
  samples_ = DvceArray2D<Real>("slab_samples",send-sbegin,4);
  auto ch = Kokkos::create_mirror_view(cells_);
  auto sh = Kokkos::create_mirror_view(samples_);
  std::array<Real,2> hmin{{std::numeric_limits<Real>::max(),std::numeric_limits<Real>::max()}};
  for (const auto &cell : cells) {
    for (int d = 0; d < 2; ++d) hmin[d] = std::min(hmin[d],cell.width[d]);
  }
  for (int q = begin; q < end; ++q) for (int d = 0; d < 3; ++d) {
    ch(q-begin,d) = cells[q].center[d];
    ch(q-begin,d+3) = cells[q].width[d];
  }
  for (int q = sbegin; q < send; ++q) {
    for (int d = 0; d < 3; ++d) sh(q-sbegin,d) = samples[q].center[d];
    sh(q-sbegin,3) = samples[q].side;
  }
  Kokkos::deep_copy(cells_,ch);
  Kokkos::deep_copy(samples_,sh);
  InitializeModes(domain,hmin,modes_x,modes_y);
}

SlabBoundary::SlabBoundary(const std::vector<CompositeCell> &cells,
    const CompositeDomain &domain, const std::vector<BoundarySample> &samples,
    const std::array<Real,2> &min_width, int modes_x, int modes_y) {
  if (!domain.isolated_z || domain.ndim != 3) {
    throw std::invalid_argument("owned slab gravity boundary requires 3D isolated z");
  }
  cells_ = DvceArray2D<Real>("slab_cells",cells.size(),6);
  samples_ = DvceArray2D<Real>("slab_samples",samples.size(),4);
  auto ch = Kokkos::create_mirror_view(cells_);
  auto sh = Kokkos::create_mirror_view(samples_);
  for (size_t q = 0; q < cells.size(); ++q) for (int d = 0; d < 3; ++d) {
    ch(q,d) = cells[q].center[d];
    ch(q,d+3) = cells[q].width[d];
  }
  for (size_t q = 0; q < samples.size(); ++q) {
    for (int d = 0; d < 3; ++d) sh(q,d) = samples[q].center[d];
    sh(q,3) = samples[q].side;
  }
  Kokkos::deep_copy(cells_,ch);
  Kokkos::deep_copy(samples_,sh);
  InitializeModes(domain,min_width,modes_x,modes_y);
}

void SlabBoundary::InitializeModes(const CompositeDomain &domain,
    const std::array<Real,2> &hmin, int modes_x, int modes_y) {
  if (!(hmin[0] > 0) || !(hmin[1] > 0) || !std::isfinite(hmin[0]) ||
      !std::isfinite(hmin[1]) || modes_x < 0 || modes_y < 0) {
    throw std::invalid_argument("slab gravity invalid global spacing/mode cutoff");
  }
  const Real lx = domain.hi[0]-domain.lo[0], ly = domain.hi[1]-domain.lo[1];
  lx_ = lx; ly_ = ly;
  area_ = lx*ly; zlo_ = domain.lo[2]; zhi_ = domain.hi[2];
  // Default cutoff resolves the finest active horizontal spacing. Explicit smaller
  // cutoffs are convergence-study parameters, not silently adaptive approximations.
  if (modes_x <= 0) modes_x = std::max(1,static_cast<int>(std::lround(lx/hmin[0]))/2);
  if (modes_y <= 0) modes_y = std::max(1,static_cast<int>(std::lround(ly/hmin[1]))/2);
  const long long count = (2LL*modes_x+1)*modes_y+modes_x;
  if (count > std::numeric_limits<int>::max()/4) {
    throw std::runtime_error("slab gravity Fourier coefficient count exceeds MPI capacity");
  }
  nmodes_ = count;
  modes_x_ = modes_x; modes_y_ = modes_y;
  wave_ = DvceArray2D<Real>("slab_modes",nmodes_,3);
  coefficients_ = DvceArray2D<Real>("slab_coefficients",nmodes_,4);
  SetShearOffset(domain.shear_offset);
}

void SlabBoundary::SetShearOffset(Real shear_offset) {
  if (!std::isfinite(shear_offset)) {
    throw std::invalid_argument("slab gravity nonfinite shear offset");
  }
  auto kh = Kokkos::create_mirror_view(wave_);
  const Real twopi = 2*std::acos(Real(-1));
  const Real shift = std::remainder(shear_offset,ly_);
  int m = 0;
  for (int ny = 0; ny <= modes_y_; ++ny) {
    for (int nx = ny == 0 ? 1 : -modes_x_; nx <= modes_x_; ++nx) {
      const Real ky = twopi*ny/ly_, kx = twopi*nx/lx_-ky*shift/lx_;
      kh(m,0) = kx; kh(m,1) = ky; kh(m,2) = std::sqrt(kx*kx+ky*ky);
      ++m;
    }
  }
  Kokkos::deep_copy(wave_,kh);
}

void SlabBoundary::Compute(const DvceArray1D<Real> &density, Real four_pi_G,
                          const DvceArray1D<Real> &boundary_values) {
  const int n = cells_.extent_int(0), nb = samples_.extent_int(0);
  if (density.extent_int(0) != n || boundary_values.extent_int(0) != nb) {
    throw std::invalid_argument("slab gravity source/boundary local shape mismatch");
  }
  auto c = cells_; auto wave = wave_; auto coef = coefficients_;
  const Real zlo = zlo_, zhi = zhi_;
  Real mass = 0, z_moment = 0;
  Kokkos::parallel_reduce("slab_zero_mode_mass",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q, Real &sum) {
    sum += density(q)*c(q,3)*c(q,4)*c(q,5);
  },mass);
  Kokkos::parallel_reduce("slab_zero_mode_moment",Kokkos::RangePolicy<DevExeSpace>(0,n),
  KOKKOS_LAMBDA(const int q, Real &sum) {
    sum += density(q)*c(q,3)*c(q,4)*c(q,5)*c(q,2);
  },z_moment);
  Real zero[2] = {mass,z_moment}; GlobalSum(zero,2);
  const Real total_mass = zero[0], first_z = zero[1];
  Kokkos::parallel_for("slab_fourier_moments",
      Kokkos::TeamPolicy<DevExeSpace>(nmodes_,Kokkos::AUTO),
  KOKKOS_LAMBDA(const TeamMember_t &team) {
    const int m = team.league_rank();
    const Real kx = wave(m,0), ky = wave(m,1), k = wave(m,2);
    SlabMoments sum;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,n),
    [&](const int q, SlabMoments &s) {
      const Real horizontal = density(q)*c(q,3)*c(q,4)*
          Sinc(0.5*kx*c(q,3))*Sinc(0.5*ky*c(q,4));
      const Real phase = kx*c(q,0)+ky*c(q,1);
      const Real height = -expm1(-k*c(q,5))/k;
      const Real p = horizontal*height*exp(-k*(zhi-c(q,2)-0.5*c(q,5)));
      const Real b = horizontal*height*exp(-k*(c(q,2)-0.5*c(q,5)-zlo));
      const Real co = cos(phase), si = sin(phase);
      s.pc += p*co; s.ps += p*si; s.mc += b*co; s.ms += b*si;
    },sum);
    Kokkos::single(Kokkos::PerTeam(team),[&]() {
      coef(m,0) = sum.pc; coef(m,1) = sum.ps;
      coef(m,2) = sum.mc; coef(m,3) = sum.ms;
    });
  });
  auto coeff_host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),coef);
  GlobalSum(coeff_host.data(),4*nmodes_);
  Kokkos::deep_copy(coef,coeff_host);
  const Real factor = four_pi_G/(2*area_);
  const int nm = nmodes_;
  auto samples = samples_;
  Kokkos::parallel_for("slab_boundary_values",Kokkos::RangePolicy<DevExeSpace>(0,nb),
  KOKKOS_LAMBDA(const int q) {
    const Real x = samples(q,0), y = samples(q,1), z = samples(q,2);
    const bool upper = samples(q,3) > 0;
    const int col = upper ? 0 : 2;
    const Real distance = upper ? z-zhi : zlo-z;
    Real value = factor*(upper ? total_mass*z-first_z : first_z-total_mass*z);
    for (int m = 0; m < nm; ++m) {
      const Real phase = wave(m,0)*x+wave(m,1)*y, k = wave(m,2);
      value -= 2*factor/k*exp(-k*distance)*
               (coef(m,col)*cos(phase)+coef(m,col+1)*sin(phase));
    }
    boundary_values(q) = value;
  });
}
}  // namespace gravity
