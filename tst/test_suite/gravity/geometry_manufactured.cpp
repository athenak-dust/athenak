// Independent manufactured checks of the shared physical operator. Build the
// explicit gravity_geometry_check target and run it outside the source tree.
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "gravity/mesh_geometry.hpp"

namespace {
using namespace gravity;
using Vec = std::array<Real,3>;
void Require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
std::vector<GravityPatch> Patches(int n, int nd, bool mixed) {
  if (n == 1) return std::vector<GravityPatch>(1);
  std::vector<GravityPatch> patches;
  for (int k = 0; k < (nd == 3 ? 2 : 1); ++k)
    for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) {
      const int refine = mixed && i == 1 && j == 1 && (nd == 2 || k == 1) ? 2 : 1;
      for (int kk = 0; kk < (nd == 3 ? refine : 1); ++kk)
        for (int jj = 0; jj < refine; ++jj) for (int ii = 0; ii < refine; ++ii) {
          GravityPatch p;
          p.lo = {(.5*i+.5*ii/refine),(.5*j+.5*jj/refine),
                  nd == 3 ? .5*k+.5*kk/refine : 0.};
          p.hi = {p.lo[0]+.5/refine,p.lo[1]+.5/refine,
                  nd == 3 ? p.lo[2]+.5/refine : 1.};
          p.cells = {n/2,n/2,nd == 3 ? n/2 : 1};
          patches.push_back(p);
        }
    }
  return patches;
}
CompositeDomain Domain(int n, int nd, bool isolated) {
  CompositeDomain d; d.ndim = nd; d.root_counts = {n,n,nd == 3 ? n : 1};
  d.isolated_z = isolated; return d;
}
template<typename Csr>
double Apply(const Csr &a, int i, const std::vector<double> &x) {
  double value = 0.;
  for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) value += a.values[k]*x[a.colidx[k]];
  return value;
}
template<typename A, typename B>
double RowDifference(const A &a, int i, const B &b, int j,
                     const std::vector<int> *remap = nullptr) {
  std::map<GlobalIndex,double> row;
  for (int k = a.rowptr[i]; k < a.rowptr[i+1]; ++k) row[a.colidx[k]] += a.values[k];
  for (int k = b.rowptr[j]; k < b.rowptr[j+1]; ++k)
    row[remap ? (*remap)[b.colidx[k]] : b.colidx[k]] -= b.values[k];
  double error = 0.;
  for (const auto &v : row) error = std::max(error,std::abs(v.second));
  return error;
}
void ConstantsAndFlux(const LocalCompositeSystem &s, bool isolated) {
  const int n = s.cells.size();
  std::vector<double> one(n,1.), boundary(s.boundary_samples.size(),s.neumann_z ? 0. : 1.),
      columns(n,0.);
  double error = 0.;
  for (int i = 0; i < n; ++i) {
    error = std::max(error,std::abs(Apply(s.matrix,i,one)-Apply(s.boundary_rhs,i,boundary)));
    for (int d = 0; d < 3; ++d)
      error = std::max(error,std::abs(Apply(s.acceleration[d],i,one)+
                                    Apply(s.boundary_acceleration[d],i,boundary)));
    for (int k = s.matrix.rowptr[i]; k < s.matrix.rowptr[i+1]; ++k)
      columns[s.matrix.colidx[k]] += s.matrix.values[k];
  }
  Require(error < 2.e-11,"constant potential gained a force or Laplacian");
  if (!isolated || s.neumann_z) for (double v : columns)
    Require(std::abs(v) < 2.e-12,"shared face flux lost the periodic left nullspace");
}
void Polynomials(int nd, bool neumann) {
  const int n = 16;
  auto domain = Domain(n,nd,nd == 3);
  MeshGeometry geometry(Patches(n,nd,true),domain,0,1,neumann);
  const auto s = geometry.BuildSystem();
  const int cells = s.cells.size();
  std::vector<double> phi(cells), boundary(s.boundary_samples.size());
  double force_error = 0., operator_error = 0.;
  // Six independent quadratic moments in 3D, including every mixed moment.
  for (int a = 0; a < nd; ++a) for (int b = a; b < nd; ++b) {
    auto value = [&](const Vec &x) { return (x[a]-.5)*(x[b]-.5); };
    for (int i = 0; i < cells; ++i) phi[i] = value(s.cells[i].center);
    for (size_t i = 0; i < boundary.size(); ++i) {
      const auto &sample = s.boundary_samples[i];
      boundary[i] = value(sample.center);
      if (neumann) {
        auto face = sample.center;
        face[2] -= sample.side*.5*s.boundary_width[i];
        boundary[i] = sample.side*((a == 2 ? face[b]-.5 : 0.)+
                                  (b == 2 ? face[a]-.5 : 0.));
      }
    }
    for (int i = 0; i < cells; ++i) {
      const auto &x = s.cells[i].center;
      bool interior = true;
      // A polynomial involving x/y is not periodic; exclude its wrap stencils.
      for (int d = 0; d < nd; ++d) if (d != 2 || nd != 3)
        interior = interior && x[d] > 3.01/n && x[d] < 1.-3.01/n;
      if (!interior) continue;
      const double laplacian = a == b ? 2. : 0.;
      operator_error = std::max(operator_error,
          std::abs((Apply(s.matrix,i,phi)-Apply(s.boundary_rhs,i,boundary))/
                   s.volume[i]+laplacian));
      for (int d = 0; d < nd; ++d) {
        const double derivative = (d == a ? x[b]-.5 : 0.)+(d == b ? x[a]-.5 : 0.);
        force_error = std::max(force_error,std::abs(Apply(s.acceleration[d],i,phi)+
            Apply(s.boundary_acceleration[d],i,boundary)+derivative));
      }
    }
  }
  std::cout << "polynomial nd=" << nd << " neumann=" << neumann
            << " A=" << operator_error << " G=" << force_error << '\n';
  Require(operator_error < 2.e-9 && force_error < 2.e-11,"quadratic reproduction failed");
}
double CubicForce(int n) {
  auto d = Domain(n,3,true);
  MeshGeometry geometry(Patches(n,3,true),d,0,1);
  const auto s = geometry.BuildSystem();
  std::vector<double> phi(s.cells.size()), boundary(s.boundary_samples.size());
  auto value = [](const Vec &x) { const double z = x[2]-.5; return z*z*z; };
  for (size_t i = 0; i < phi.size(); ++i) phi[i] = value(s.cells[i].center);
  for (size_t i = 0; i < boundary.size(); ++i) boundary[i] = value(s.boundary_samples[i].center);
  double error = 0.;
  for (size_t i = 0; i < phi.size(); ++i) for (int d = 0; d < 2; ++d)
    error = std::max(error,std::abs(Apply(s.acceleration[d],i,phi)+
                                  Apply(s.boundary_acceleration[d],i,boundary)));
  return error;
}
void CheckUpdate(const LocalCsr &initial, const CsrRowUpdate &update, const LocalCsr &fresh) {
  size_t next = 0; double error = 0.;
  for (int i = 0; i < initial.LocalRows(); ++i) {
    if (next < update.rows.size() && update.rows[next] == i) {
      error = std::max(error,RowDifference(fresh,i,update,next)); ++next;
    } else error = std::max(error,RowDifference(fresh,i,initial,i));
  }
  Require(next == update.rows.size() && error < 2.e-11,"phase update differs from full rebuild");
}
void Phases(int nd, bool mixed, bool isolated, bool neumann, int n = 8) {
  auto d = Domain(n,nd,isolated); const auto patches = Patches(n,nd,mixed);
  MeshGeometry geometry(patches,d,0,1,neumann); const auto initial = geometry.BuildSystem();
  MeshGeometry discovered(patches,d,0,1,neumann);
  Require(discovered.PhaseRows() == geometry.PhaseRows(),"standalone phase discovery differs");
  for (int threads : {1,2}) {
    geometry.SetPhaseThreads(threads);
    for (double phase : {.137,.49,.999999,1.,1.0625,-.17,0.}) {
      const auto update = geometry.BuildPhaseUpdate(phase);
      d.shear_offset = phase;
      MeshGeometry rebuilt(patches,d,0,1,neumann); const auto fresh = rebuilt.BuildSystem();
      CheckUpdate(initial.matrix,update.matrix,fresh.matrix);
      CheckUpdate(initial.boundary_rhs,update.boundary_rhs,fresh.boundary_rhs);
      for (int k = 0; k < 3; ++k) {
        CheckUpdate(initial.acceleration[k],update.acceleration[k],fresh.acceleration[k]);
        CheckUpdate(initial.boundary_acceleration[k],update.boundary_acceleration[k],
                    fresh.boundary_acceleration[k]);
      }
      ConstantsAndFlux(fresh,isolated);
    }
  }
  std::cout << "phase identities nd=" << nd << " mixed=" << mixed
            << " isolated=" << isolated << " neumann=" << neumann << " n=" << n << '\n';
}
void Reference(bool isolated) {
  auto d = Domain(8,3,isolated); d.shear_offset = .137;
  MeshGeometry geometry(Patches(8,3,true),d,0,1); const auto local = geometry.BuildSystem();
  const auto global = BuildCompositeSystem(local.cells,d);
  using Key = std::tuple<Real,Real,Real,int>;
  auto key = [](const BoundarySample &b) {
    return Key{b.center[0],b.center[1],b.center[2],b.side};
  };
  std::map<Key,int> samples;
  for (size_t i = 0; i < local.boundary_samples.size(); ++i) samples[key(local.boundary_samples[i])] = i;
  std::vector<int> map(global.boundary_samples.size());
  for (size_t i = 0; i < map.size(); ++i) map[i] = samples.at(key(global.boundary_samples[i]));
  double error = 0.;
  for (size_t i = 0; i < local.cells.size(); ++i) {
    error = std::max(error,RowDifference(local.matrix,i,global.hierarchy.matrices[0],i));
    error = std::max(error,RowDifference(local.boundary_rhs,i,global.boundary_rhs,i,&map));
    for (int k = 0; k < 3; ++k) {
      error = std::max(error,RowDifference(local.acceleration[k],i,global.acceleration[k],i));
      error = std::max(error,RowDifference(local.boundary_acceleration[k],i,
                                          global.boundary_acceleration[k],i,&map));
    }
  }
  std::cout << "reference isolated=" << isolated << " max_coefficient_error=" << error << '\n';
  Require(error < 2.e-11,"reference cell or boundary ordering changed the physical operator");
}
}  // namespace

int main() {
  Require(sizeof(Real) == sizeof(double),"gravity geometry checks require double precision");
  Polynomials(2,false); Polynomials(3,false); Polynomials(3,true);
  const double coarse = CubicForce(16), fine = CubicForce(32);
  std::cout << "cubic interface force errors " << coarse << " -> " << fine << '\n';
  Require(coarse/fine > 3.8,"interface force lost second-order convergence");
  for (int nd : {2,3}) for (bool mixed : {false,true}) Phases(nd,mixed,false,false);
  Phases(3,true,true,false); Phases(3,true,true,true);
  Phases(2,false,false,false,2); Phases(3,false,false,false,2);
  Phases(2,false,false,false,1); Phases(3,false,false,false,1);
  Reference(false); Reference(true);
  std::cout << "gravity geometry checks passed\n";
}
