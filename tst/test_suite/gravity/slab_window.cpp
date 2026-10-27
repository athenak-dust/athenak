// Direct isolated reference: physical-window continuity, remapping and the
// unsheared finite sum. Run gravity_slab_window_check outside the source tree.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "globals.hpp"
#include "gravity/slab_boundary.hpp"

namespace {
using namespace gravity;
constexpr int nxcell = 16, nzcell = 4;
const Real pi = std::acos(Real(-1));
Real Max(Real x) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&x,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  return x;
}
void Check(bool ok, const char *message) {
  if (!ok) throw std::runtime_error(message);
}
Real Sinc(Real x) { return std::abs(x) < 1.e-7 ? 1-x*x/6 : std::sin(x)/x; }
// Independent, original unsheared rectangular Fourier sum of cell integrals.
Real Unsheared(const std::vector<CompositeCell> &cells, const BoundarySample &sample,
               int mx, int my) {
  Real mass = 0, moment = 0;
  for (const auto &c : cells) {
    const Real rho = 1+.1*std::cos(2*pi*(2*c.center[0]+c.center[1]));
    const Real m = rho*c.width[0]*c.width[1]*c.width[2];
    mass += m; moment += m*c.center[2];
  }
  Real value = .5*(mass*sample.center[2]-moment);
  for (int ny = 0; ny <= my; ++ny) for (int nx = ny == 0 ? 1 : -mx; nx <= mx; ++nx) {
    const Real kx = 2*pi*nx, ky = 2*pi*ny, k = std::hypot(kx,ky);
    Real co = 0, si = 0;
    for (const auto &c : cells) {
      const Real rho = 1+.1*std::cos(2*pi*(2*c.center[0]+c.center[1]));
      const Real h = rho*c.width[0]*c.width[1]*Sinc(kx*c.width[0]/2)*
          Sinc(ky*c.width[1]/2)*(-std::expm1(-k*c.width[2]))/k*
          std::exp(-k*(1-c.center[2]-c.width[2]/2));
      co += h*std::cos(kx*c.center[0]+ky*c.center[1]);
      si += h*std::sin(kx*c.center[0]+ky*c.center[1]);
    }
    value -= std::exp(-k*(sample.center[2]-1))/k*
        (co*std::cos(kx*sample.center[0]+ky*sample.center[1])+
         si*std::sin(kx*sample.center[0]+ky*sample.center[1]));
  }
  return value;
}
void Run(int rank, int ranks) {
  CompositeDomain domain; domain.ndim = 3; domain.isolated_z = true;
  domain.lo = {0,0,0}; domain.hi = {1,1,1};
  std::vector<CompositeCell> all,cells;
  std::vector<BoundarySample> samples;
  for (int k = 0; k < nzcell; ++k) for (int j = 0; j < nxcell; ++j)
    for (int i = 0; i < nxcell; ++i) {
      CompositeCell c{{(i+.5)/nxcell,(j+.5)/nxcell,(k+.5)/nzcell},
                      {1./nxcell,1./nxcell,1./nzcell}};
      if (all.size()%ranks == static_cast<size_t>(rank)) cells.push_back(c);
      all.push_back(c);
    }
  for (int j = 0; j < 4; ++j) for (int i = 0; i < 4; ++i)
    if ((j*4+i)%ranks == rank) samples.push_back({{(i+.5)/4,(j+.5)/4,1.0625},1});
  const std::array<Real,2> spacing{{1./nxcell,1./nxcell}};
  DvceArray1D<Real> rho("rho",cells.size()),values("boundary",samples.size());
  auto rh = Kokkos::create_mirror_view(rho);
  auto set_source = [&](Real s, int ix, int iy) {
    for (size_t i = 0; i < cells.size(); ++i)
      rh(i) = 1+.1*std::cos(2*pi*((ix-iy*s)*cells[i].center[0]+iy*cells[i].center[1]));
    Kokkos::deep_copy(rho,rh);
  };
  auto evaluate = [&](SlabBoundary &b, Real phase) {
    b.SetShearOffset(phase); b.Compute(rho,1,values);
    auto host = Kokkos::create_mirror_view_and_copy(HostMemSpace(),values);
    return std::vector<Real>(host.data(),host.data()+host.size());
  };
  auto difference = [&](const std::vector<Real> &a, const std::vector<Real> &b) {
    Real error = 0;
    for (size_t i = 0; i < a.size(); ++i) error = std::max(error,std::abs(a[i]-b[i]));
    return Max(error);
  };
  SlabBoundary b(cells,domain,samples,spacing,2,3);
  Real max_jump = 0;
  // +/- halfwrap, fullwrap, and interior integer ny*phase crossings. Sources
  // cover flat-core edges and taper modes for all retained nonzero ky values.
  for (int iy : {1,2,3}) for (int ix : {-3,-2,2,3})
    for (Real crossing : {Real(-.5),Real(.5),Real(1),Real(1)/iy}) {
      Real previous = 0;
      for (Real epsilon : {Real(1.e-7),Real(1.e-10)}) {
        set_source(crossing-epsilon,ix,iy);
        auto left = evaluate(b,crossing-epsilon);
        set_source(crossing+epsilon,ix,iy);
        const Real jump = difference(left,evaluate(b,crossing+epsilon));
        Check(jump < 2.e-8,"slab boundary discontinuity at a phase crossing");
        if (previous > 0) Check(jump < .01*previous+2.e-13,
                                "slab phase jump does not vanish with epsilon");
        previous = jump; max_jump = std::max(max_jump,jump);
      }
    }
  Real periodic_error = 0, restart_error = 0;
  for (Real s : {Real(-.5),Real(-.27),Real(0),Real(.23),Real(.5)}) {
    set_source(s,3,3);
    const auto original = evaluate(b,s);
    for (int wraps : {-2,-1,1,2})
      periodic_error = std::max(periodic_error,difference(original,evaluate(b,s+wraps)));
    domain.shear_offset = s;
    SlabBoundary fresh(cells,domain,samples,spacing,2,3);
    restart_error = std::max(restart_error,difference(original,evaluate(fresh,s)));
  }
  Check(periodic_error < 2.e-13 && restart_error < 2.e-13,
        "slab window depends on remap count or update history");
  set_source(0,2,1);
  const auto zero = evaluate(b,0);
  Real zero_error = 0;
  for (size_t i = 0; i < samples.size(); ++i)
    zero_error = std::max(zero_error,std::abs(zero[i]-Unsheared(all,samples[i],2,3)));
  zero_error = Max(zero_error);
  Check(zero_error < 2.e-13,"zero-phase boundary changed its rectangular reference sum");
  // Fixed discretized source, jointly increasing physical-kx and ky cutoffs.
  // An off-integer sheared source has power outside the deliberately small core.
  set_source(.37,3,2);
  SlabBoundary reference(cells,domain,samples,spacing,10,10);
  const auto target = evaluate(reference,.37);
  std::vector<Real> errors;
  for (int cutoff : {2,4,6}) {
    SlabBoundary trial(cells,domain,samples,spacing,cutoff,cutoff);
    errors.push_back(difference(target,evaluate(trial,.37)));
  }
  Check(errors[2] < errors[0] && errors[2] < 2.e-6,
        "joint cutoff increase failed the smooth-source convergence control");
  if (rank == 0) std::cout << "slab window PASS max_phase_change=" << max_jump
      << " periodic_error=" << periodic_error << " restart_error=" << restart_error
      << " zero_reference_error=" << zero_error << " convergence="
      << errors[0] << ',' << errors[1] << ',' << errors[2] << '\n';
}
}  // namespace
int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  int provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
#else
  global_variable::my_rank = 0; global_variable::nranks = 1;
#endif
  Kokkos::initialize(argc,argv);
  int status = 0;
  try {
    std::cout << std::setprecision(17);
    Run(global_variable::my_rank,global_variable::nranks);
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n'; status = 1;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD,status);
#endif
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return status;
}
