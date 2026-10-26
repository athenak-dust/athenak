// Regression for full-period vacuum cells and terminal-operator rank checks.
// Build gravity_coarse_check explicitly; run outside the source tree, optionally
// under MPI. No files are written. The C1 case failed before the geometry repair.
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gravity/geometric_mg.hpp"
#include "gravity/vacuum_mesh.hpp"

namespace {
using namespace gravity;
using Vector = DvceArray1D<Real>;
void Check(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
Real Sum(Real value) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&value,1,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif
  return value;
}
HostCsr Dense(const std::vector<std::vector<Real>> &values, Real scale = 1) {
  HostCsr a;
  a.rows = a.cols = values.size(); a.rowptr.push_back(0);
  for (const auto &row : values) {
    for (int j = 0; j < a.cols; ++j) {
      a.colidx.push_back(j); a.values.push_back(scale*row[j]);
    }
    a.rowptr.push_back(a.values.size());
  }
  return a;
}
void RankChecks(int rank) {
  const std::vector<std::vector<Real>> connected{{2,-1,0,-1},{-1,2,-1,0},
                                               {0,-1,2,-1},{-1,0,-1,2}};
  for (Real scale : {Real(1.e-14),Real(1),Real(1.e14)}) {
    HostHierarchy h; h.constant_nullspace = true; h.matrices = {Dense(connected,scale)};
    SparseMultigrid solver(h);
    Vector phi("expected",solver.LocalSize()),rhs("rhs",solver.LocalSize()),
           x("solution",solver.LocalSize());
    auto host = Kokkos::create_mirror_view(phi);
    for (int i = 0; i < solver.LocalSize(); ++i) host(i) = solver.LocalBegin()+i-1.5;
    Kokkos::deep_copy(phi,host); Kokkos::deep_copy(x,0.);
    solver.ApplyMatrix(0,phi,rhs);
    const auto stats = solver.Solve(rhs,x,1.e-10,20,8);
    auto actual = Kokkos::create_mirror_view_and_copy(HostMemSpace(),x);
    Real error = 0;
    for (int i = 0; i < solver.LocalSize(); ++i) error += std::pow(actual(i)-host(i),2);
    Check(stats.converged && !stats.preconditioner_breakdown && Sum(error) < 1.e-20,
          "coarse rank guard depends on the physical matrix scale");
  }
  // Both invalid matrices have left/right constant nullspaces, but A+11^T/n
  // remains singular. The second has no zero columns: the pivot test is needed.
  const std::vector<std::vector<std::vector<Real>>> invalid{
      {{1,-1,0},{-1,1,0},{0,0,0}},
      {{1,-1,0,0},{-1,1,0,0},{0,0,1,-1},{0,0,-1,1}},
      {{1,-1,0,0},{-1,1+1.e-15,-1.e-15,0},
       {0,-1.e-15,1+1.e-15,-1},{0,0,-1,1}}};
  for (const auto &matrix : invalid) {
    bool refused = false;
    try {
      HostHierarchy h; h.constant_nullspace = true; h.matrices = {Dense(matrix)};
      SparseMultigrid solver(h);
    } catch (const std::runtime_error &error) {
      refused = std::string(error.what()).find("terminal") != std::string::npos;
    }
    Check(refused,"invalid coarse geometry did not fail collectively at setup");
  }
  // A single periodic cell has exactly the legitimate constant nullspace.
  HostHierarchy singleton; singleton.constant_nullspace = true;
  singleton.matrices = {Dense({{0}})};
  SparseMultigrid solver(singleton);
  if (rank == 0) std::cout << "terminal scale/nullspace guards PASS\n";
}
void VacuumMean(int rank, int ranks, bool shear, Real xlength) {
  CompositeDomain domain;
  domain.ndim = 3; domain.root_counts = {16,16,16};
  domain.hi = {xlength,1,1}; domain.isolated_z = true;
  domain.shear_phase_updates = shear;
  std::vector<GravityPatch> patches;
  for (int z = 0; z < 2; ++z) for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
    const int refine = x == 0 && y == 0 && z == 0 ? 2 : 1;
    for (int k = 0; k < refine; ++k) for (int j = 0; j < refine; ++j)
      for (int i = 0; i < refine; ++i) {
        GravityPatch p;
        p.lo = {(x+Real(i)/refine)*xlength/2,(y+Real(j)/refine)/2,
                (z+Real(k)/refine)/2};
        p.hi = {p.lo[0]+xlength/(2*refine),p.lo[1]+Real(1)/(2*refine),
                p.lo[2]+Real(1)/(2*refine)};
        p.cells = {8,8,8}; p.gid = patches.size(); p.owner = p.gid%ranks;
        patches.push_back(p);
      }
  }
  auto extended = BuildVacuumMesh(patches,domain,ranks);
  domain = extended.domain;
  auto system = BuildGeometricHierarchy(extended.patches,domain,rank,ranks,true,true);
  SparseMultigrid solver(system.hierarchy);
  std::vector<std::vector<int>> rows{system.fine_geometry->PhaseRows()};
  for (auto &g : system.coarse_geometry) rows.push_back(g->PhaseRows());
  solver.ConfigureHierarchyPhaseRows(rows,system.prolongation_phase_rows);
  for (Real phase : {Real(0),Real(-.0625+1.e-9),Real(-.0625),
                     Real(-.0625-1.e-9),Real(-1.02)}) {
    if (!shear && phase != 0) continue;
    domain.shear_offset = phase;
    std::vector<CsrRowUpdate> a,p;
    auto fine = system.fine_geometry->BuildPhaseUpdate(phase,true);
    a.push_back(fine.matrix);
    for (auto &g : system.coarse_geometry) a.push_back(g->BuildPhaseUpdate(phase,false).matrix);
    for (size_t l = 0; l < system.coarse_geometry.size(); ++l) {
      auto &f = l == 0 ? *system.fine_geometry : *system.coarse_geometry[l-1];
      p.push_back(BuildProlongationPhaseUpdate(f,*system.coarse_geometry[l],domain,rank,
                                               system.prolongation_phase_rows[l]));
    }
    solver.UpdateHierarchyPhaseRows(a,p);
    // Construct the actual physical mean-sheet RHS independently of A*phi:
    // rho=1 in the source domain, rho=0 outside, outward flux 0.05 at both walls.
    // Rebuild only the fine metadata/BC rows so this check uses the current phase.
    auto source = system.fine_geometry->BuildSystem(true);
    Vector rhs("mean_rhs",solver.LocalSize()),x("mean_phi",solver.LocalSize()),
           product("Aphi",solver.LocalSize());
    auto host = Kokkos::create_mirror_view(rhs);
    for (int i = 0; i < solver.LocalSize(); ++i) {
      const Real z = source.cells[i].center[2];
      host(i) = z > 0 && z < 1 ? -.1*source.volume[i] : 0.;
      for (int k = source.boundary_rhs.rowptr[i]; k < source.boundary_rhs.rowptr[i+1]; ++k)
        host(i) += .05*source.boundary_rhs.values[k];
    }
    Kokkos::deep_copy(rhs,host); Kokkos::deep_copy(x,0.);
    const auto stats = solver.Solve(rhs,x,1.e-10,150,24);
    solver.ApplyMatrix(0,x,product);
    auto actual = Kokkos::create_mirror_view_and_copy(HostMemSpace(),product);
    Real e2 = 0, b2 = 0;
    for (int i = 0; i < solver.LocalSize(); ++i) {
      e2 += std::pow(actual(i)-host(i),2); b2 += host(i)*host(i);
    }
    const Real residual = std::sqrt(Sum(e2)/Sum(b2));
    Check(stats.converged && !stats.preconditioner_breakdown && residual <= 1.e-10,
          "C1/vacuum mean solve failed or true residual disagrees");
    if (rank == 0) std::cout << "C1 mean shear=" << shear << " Lx=" << xlength
        << " phase=" << phase << " iterations=" << stats.iterations
        << " true_residual=" << residual << '\n';
  }
}
}  // namespace

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  Check(provided >= MPI_THREAD_FUNNELED,"MPI thread support unavailable");
#endif
  Kokkos::initialize(argc,argv);
  int rank = 0, ranks = 1, status = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
#endif
  try {
    std::cout << std::setprecision(17);
    RankChecks(rank);
    VacuumMean(rank,ranks,false,1);
    VacuumMean(rank,ranks,true,1);
    VacuumMean(rank,ranks,true,.25);
    if (rank == 0) std::cout << "gravity coarse-grid checks PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n'; status = 1;
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
