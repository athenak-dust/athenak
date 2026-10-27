// Finite extreme RHS/guess scales, true residual acceptance, zero RHS and
// collective invalid-input rejection, including MPI ranks owning no rows.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "gravity/sparse_mg.hpp"

namespace {
using namespace gravity;
Real Max(Real x) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&x,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);
#endif
  return x;
}
void Check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void Run(int rank) {
  HostHierarchy h; h.constant_nullspace = true;
  HostCsr a; a.rows = a.cols = 2; a.rowptr = {0,2,4};
  a.colidx = {0,1,0,1}; a.values = {1,-1,-1,1}; h.matrices = {a};
  SparseMultigrid mg(h);
  DvceArray1D<Real> b("rhs",mg.LocalSize()),x("guess",mg.LocalSize());
  auto bh = Kokkos::create_mirror_view(b), xh = Kokkos::create_mirror_view(x);
  auto fill = [&](Real source, Real guess, bool constant = false) {
    for (int i = 0; i < mg.LocalSize(); ++i) {
      const Real sign = mg.LocalBegin()+i == 0 ? 1 : -1;
      bh(i) = constant ? source : sign*source;
      xh(i) = sign*guess;
    }
    Kokkos::deep_copy(b,bh); Kokkos::deep_copy(x,xh);
  };
  Real max_error = 0, max_residual = 0;
  int solves = 0;
  for (Real scale : {Real(1.e-300),Real(1.e-200),Real(1.e-100),Real(1),
                     Real(1.e160),Real(1.e300),Real(1.e308)}) {
    for (Real fraction : {Real(0),Real(.25),Real(.5),Real(1)}) {
      fill(scale,fraction*scale);
      const auto stats = mg.Solve(b,x,1.e-10,10,2);
      auto actual = Kokkos::create_mirror_view_and_copy(HostMemSpace(),x);
      Real error = 0;
      for (int i = 0; i < mg.LocalSize(); ++i) {
        const Real expected = mg.LocalBegin()+i == 0 ? .5 : -.5;
        error = std::max(error,std::abs(actual(i)/scale-expected));
      }
      error = Max(error);
      // For A=[1,-1;-1,1], gauge-fixed error is half the independently
      // evaluated relative residual after dividing all data by scale.
      Check(stats.converged && !stats.preconditioner_breakdown && error < 1.e-10,
            "FGMRES falsely converged or failed for a finite RHS/guess scale");
      Check(std::isfinite(stats.relative_residual) && stats.relative_residual <= 1.e-10,
            "FGMRES reported an invalid relative residual");
      max_error = std::max(max_error,error);
      max_residual = std::max(max_residual,2*error); ++solves;
    }
  }
  for (Real guess : {Real(0),Real(1.e-200),Real(1),Real(1.e160),Real(1.e300)}) {
    fill(0,guess);
    const auto stats = mg.Solve(b,x,1.e-10,10,2);
    auto actual = Kokkos::create_mirror_view_and_copy(HostMemSpace(),x);
    Real residual = 0;
    for (int i = 0; i < mg.LocalSize(); ++i) residual = std::max(residual,2*std::abs(actual(i)));
    Check(stats.converged && Max(residual) <= 1.e-10 && stats.relative_residual == 0,
          "zero RHS lost its absolute residual tolerance with a nonzero guess");
    ++solves;
  }
  for (Real scale : {Real(1.e-200),Real(1.e160),Real(1.e300)}) {
    fill(scale,scale);
    for (int i = 0; i < mg.LocalSize(); ++i) xh(i) += 2*scale;
    Kokkos::deep_copy(x,xh);
    const auto stats = mg.Solve(b,x,1.e-10,10,2);
    auto actual = Kokkos::create_mirror_view_and_copy(HostMemSpace(),x);
    Real error = 0;
    for (int i = 0; i < mg.LocalSize(); ++i)
      error = std::max(error,std::abs(actual(i)/scale-(mg.LocalBegin()+i == 0 ? .5 : -.5)));
    Check(stats.converged && Max(error) < 1.e-10,
          "warm-guess mean projection changed with system scale");
    ++solves;
  }
  fill(0,0);
  Kokkos::deep_copy(x,Real(1.7e308));
  Check(mg.Solve(b,x,1.e-10,10,2).converged,
        "extreme constant warm guess overflowed before mean projection");
  ++solves;
  // Validate before the mean projection: summing these raw finite components
  // would overflow. Their projected RHS is exactly zero, with the stated mean.
  fill(Real(1.7e308),0,true);
  const auto constant = mg.Solve(b,x,1.e-10,10,2);
  Check(constant.converged && constant.removed_rhs_mean == Real(1.7e308),
        "extreme constant RHS overflowed before mean projection");
  for (bool corrupt_rhs : {false,true})
    for (Real invalid : {std::numeric_limits<Real>::infinity(),
                        std::numeric_limits<Real>::quiet_NaN()}) {
      fill(1,0);
      if (mg.LocalBegin() == 0 && mg.LocalSize() > 0) {
        if (corrupt_rhs) bh(0) = invalid;
        else xh(0) = invalid;
      }
      Kokkos::deep_copy(b,bh); Kokkos::deep_copy(x,xh);
      bool rejected = false;
      try { mg.Solve(b,x,1.e-10,10,2); }
      catch (const std::runtime_error &) { rejected = true; }
      Check(rejected,"nonfinite FGMRES input was not rejected collectively");
    }
  fill(1.e-200,1.e200);
  bool range_rejected = false;
  try { mg.Solve(b,x,1.e-10,10,2); }
  catch (const std::runtime_error &) { range_rejected = true; }
  Check(range_rejected,"normalization silently erased a nonzero RHS");
  fill(1.e-200,0);
  const auto no_iterations = mg.Solve(b,x,1.e-10,0,2);
  Check(!no_iterations.converged && std::abs(no_iterations.relative_residual-1) < 1.e-14,
        "zero iteration budget certified an unsolved tiny RHS");
  if (rank == 0) std::cout << "solver scaling PASS solves=" << solves+1
      << " max_normalized_solution_error=" << max_error
      << " max_independent_relative_residual=" << max_residual
      << " nonfinite_and_scale_range_rejection=PASS\n";
}
}  // namespace
int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  int provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
#endif
  Kokkos::initialize(argc,argv);
  int rank = 0, status = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
  try { std::cout << std::setprecision(17); Run(rank); }
  catch (const std::exception &e) {
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
