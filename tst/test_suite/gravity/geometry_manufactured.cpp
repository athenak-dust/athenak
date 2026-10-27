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
#include "gravity/vacuum_mesh.hpp"

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
  std::array<std::vector<double>,3> force_columns;
  std::array<std::vector<double>,3> boundary_force_columns;
  for (auto &c : force_columns) c.assign(n,0.);
  for (auto &c : boundary_force_columns) c.assign(s.boundary_samples.size(),0.);
  double error = 0.;
  for (int i = 0; i < n; ++i) {
    error = std::max(error,std::abs(Apply(s.matrix,i,one)-Apply(s.boundary_rhs,i,boundary)));
    for (int d = 0; d < 3; ++d)
      error = std::max(error,std::abs(Apply(s.acceleration[d],i,one)+
                                    Apply(s.boundary_acceleration[d],i,boundary)));
    for (int k = s.matrix.rowptr[i]; k < s.matrix.rowptr[i+1]; ++k)
      columns[s.matrix.colidx[k]] += s.matrix.values[k];
    for (int d = 0; d < 3; ++d) {
      const auto &g = s.acceleration[d];
      for (int k = g.rowptr[i]; k < g.rowptr[i+1]; ++k)
        force_columns[d][g.colidx[k]] += s.volume[i]*g.values[k];
      const auto &b = s.boundary_acceleration[d];
      for (int k = b.rowptr[i]; k < b.rowptr[i+1]; ++k)
        boundary_force_columns[d][b.colidx[k]] += s.volume[i]*b.values[k];
    }
  }
  Require(error < 2.e-11,"constant potential gained a force or Laplacian");
  if (!isolated || s.neumann_z) for (double v : columns)
    Require(std::abs(v) < 2.e-12,"shared face flux lost the periodic left nullspace");
  for (int d = 0; d < (isolated ? 2 : 3); ++d) {
    for (double value : force_columns[d])
      Require(std::abs(value) < 2.e-12,"shared potential face average lost zero volume force");
    for (double value : boundary_force_columns[d])
      Require(std::abs(value) < 2.e-12,"boundary potential data gained a net lateral force");
  }
}
void Polynomials(int nd, bool neumann, bool c1 = false) {
  const int n = 16;
  auto domain = Domain(n,nd,nd == 3);
  if (c1) domain.c1_z_interfaces = {.25,.75};
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
  std::cout << "polynomial nd=" << nd << " neumann=" << neumann << " c1=" << c1
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
void PhaseFixture(const std::vector<GravityPatch> &patches, CompositeDomain d, bool neumann) {
  MeshGeometry geometry(patches,d,0,1,neumann); const auto initial = geometry.BuildSystem();
  MeshGeometry discovered(patches,d,0,1,neumann);
  Require(discovered.PhaseRows() == geometry.PhaseRows(),"standalone phase discovery differs");
  for (int threads : {1,2}) {
    geometry.SetPhaseThreads(threads);
    for (double phase : {.137,.49,.999999,1.,1.0625,-.17,0.}) {
      const auto update = geometry.BuildPhaseUpdate(phase);
      const auto matrix_only = geometry.BuildPhaseUpdate(phase,false);
      d.shear_offset = phase;
      MeshGeometry rebuilt(patches,d,0,1,neumann); const auto fresh = rebuilt.BuildSystem();
      CheckUpdate(initial.matrix,update.matrix,fresh.matrix);
      CheckUpdate(initial.matrix,matrix_only.matrix,fresh.matrix);
      Require(matrix_only.boundary_rhs.rows.empty(),"matrix-only phase update allocated boundary rows");
      for (int k = 0; k < 3; ++k)
        Require(matrix_only.acceleration[k].rows.empty() &&
                matrix_only.boundary_acceleration[k].rows.empty(),
                "matrix-only phase update allocated force rows");
      CheckUpdate(initial.boundary_rhs,update.boundary_rhs,fresh.boundary_rhs);
      for (int k = 0; k < 3; ++k) {
        CheckUpdate(initial.acceleration[k],update.acceleration[k],fresh.acceleration[k]);
        CheckUpdate(initial.boundary_acceleration[k],update.boundary_acceleration[k],
                    fresh.boundary_acceleration[k]);
      }
      ConstantsAndFlux(fresh,d.isolated_z);
    }
  }
  std::cout << "phase identities nd=" << d.ndim << " patches=" << patches.size()
            << " isolated=" << d.isolated_z << " neumann=" << neumann
            << " c1=" << d.c1_z_interfaces.size() << '\n';
}
void Phases(int nd, bool mixed, bool isolated, bool neumann, int n = 8, bool c1 = false) {
  auto d = Domain(n,nd,isolated);
  if (c1) d.c1_z_interfaces = {.25,.75};
  PhaseFixture(Patches(n,nd,mixed),d,neumann);
}
void E3Multilevel(const Vec &length) {
  // Same leaf boxes as review E3: all root blocks refined once, then the
  // x1max band and the two middle z bands refined again (28+32 blocks).
  // Cell counts are reduced by four to keep this a small geometry-only check.
  std::vector<GravityPatch> patches;
  for (int k = 0; k < 4; ++k) for (int j = 0; j < 2; ++j) for (int i = 0; i < 4; ++i) {
    const int r = i == 3 && (k == 1 || k == 2) ? 2 : 1;
    for (int kk = 0; kk < r; ++kk) for (int jj = 0; jj < r; ++jj)
      for (int ii = 0; ii < r; ++ii) {
        GravityPatch p;
        p.lo = {i/4.+ii/(4.*r),j/2.+jj/(2.*r),k/4.+kk/(4.*r)};
        p.hi = {p.lo[0]+1./(4*r),p.lo[1]+1./(2*r),p.lo[2]+1./(4*r)};
        for (int d = 0; d < 3; ++d) { p.lo[d] *= length[d]; p.hi[d] *= length[d]; }
        p.cells = {2,4,2}; patches.push_back(p);
      }
  }
  auto d = Domain(4,3,false); d.hi = length;
  PhaseFixture(patches,d,false);
  std::cout << "E3 multilevel aspect " << length[0] << ',' << length[1] << ',' << length[2] << '\n';
}
void PiecewiseVacuum() {
  auto d = Domain(16,3,true);
  const auto vacuum = BuildVacuumMesh(Patches(16,3,true),d,1);
  MeshGeometry geometry(vacuum.patches,vacuum.domain,0,1,true);
  const auto s = geometry.BuildSystem();
  std::vector<double> phi(s.cells.size()), boundary(s.boundary_samples.size(),.5);
  for (size_t i = 0; i < phi.size(); ++i) {
    const double z = s.cells[i].center[2];
    phi[i] = z < 0 ? .25-.5*z : z > 1 ? .25+.5*(z-1) : .5*((z-.5)*(z-.5)+.25);
  }
  double error = 0., force = 0.;
  for (size_t i = 0; i < phi.size(); ++i) {
    const double z = s.cells[i].center[2], rho = z > 0 && z < 1 ? 1. : 0.;
    error = std::max(error,std::abs((Apply(s.matrix,i,phi)-Apply(s.boundary_rhs,i,boundary))/
                                   s.volume[i]+rho));
    for (int d = 0; d < 3; ++d) {
      const double derivative = d != 2 ? 0. : z < 0 ? -.5 : z > 1 ? .5 : z-.5;
      force = std::max(force,std::abs(Apply(s.acceleration[d],i,phi)+
                                      Apply(s.boundary_acceleration[d],i,boundary)+derivative));
    }
  }
  std::cout << "piecewise vacuum A=" << error << " G=" << force << '\n';
  Require(error < 2.e-9 && force < 2.e-11 && s.quadratic_interface_fallbacks == 0 &&
          s.quadratic_x_image_fallbacks == 0,
          "piecewise vacuum quadratic reproduction failed");
  ConstantsAndFlux(s,true);
}
void CoarseC1Fallback() {
  auto d = Domain(2,3,true); d.c1_z_interfaces = {.5};
  MeshGeometry coarse(Patches(2,3,false),d,0,1,true,true);
  const auto system = coarse.BuildSystem();
  Require(system.quadratic_interface_fallbacks > 0 && system.quadratic_x_image_fallbacks > 0,
          "thin coarse reconstruction fallback was not counted");
  ConstantsAndFlux(system,true);
  bool rejected = false;
  try {
    MeshGeometry fine(Patches(2,3,false),d,0,1,true); fine.BuildSystem();
  } catch (const std::runtime_error &) { rejected = true; }
  Require(rejected,"rank-deficient fine one-sided reconstruction silently fell back");
  std::cout << "counted coarse C1 fallbacks=" << system.quadratic_interface_fallbacks
            << " x-image fallbacks=" << system.quadratic_x_image_fallbacks << '\n';
}
void VirtualImages() {
  auto d = Domain(8,3,true); d.shear_offset = .137;
  MeshGeometry geometry(Patches(8,3,true),d,0,1,true);
  bool images = false, shear = false;
  geometry.VirtualAverage({-.125,.25,.25},{0.,.375,.375},&images,&shear);
  Require(images && shear,"x image was not marked as a shear dependency");
  geometry.VirtualAverage({.25,-.125,.25},{.375,0.,.375},&images,&shear);
  Require(images && !shear,"y-only image was marked as a shear dependency");
  geometry.VirtualAverage({.25,.25,-.125},{.375,.375,0.},&images,&shear);
  Require(images && !shear,"z reflection was marked as a shear dependency");
  geometry.VirtualAverage({0.,.25,.25},{.0625,.3125,.3125},&images,&shear);
  Require(images && shear,"recursive x image was not marked as a shear dependency");
  std::cout << "virtual-average x/y/z and recursive shear dependency checks passed\n";
}
void OrdinaryPeriodicTranslation() {
  auto d = Domain(16,2,false); d.shear_phase_updates = false;
  MeshGeometry geometry(Patches(16,2,false),d,0,1); const auto s = geometry.BuildSystem();
  std::vector<double> phi(s.cells.size()), shifted(phi.size());
  std::vector<GlobalIndex> partner(phi.size());
  for (size_t i = 0; i < phi.size(); ++i) {
    const auto x = s.cells[i].center;
    phi[i] = std::sin(7.1*x[0]+3.7*x[1]+5.3*x[0]*x[1]);
    auto point = x; point[0] += 1./16.; partner[i] = geometry.Locate(point);
  }
  for (size_t i = 0; i < phi.size(); ++i) shifted[i] = phi[partner[i]];
  double error = 0.;
  for (size_t i = 0; i < phi.size(); ++i) {
    error = std::max(error,std::abs(Apply(s.matrix,i,shifted)-Apply(s.matrix,partner[i],phi)));
    for (int q = 0; q < 2; ++q)
      error = std::max(error,std::abs(Apply(s.acceleration[q],i,shifted)-
                                     Apply(s.acceleration[q],partner[i],phi)));
  }
  bool rejected = false;
  try { geometry.BuildPhaseUpdate(.01); } catch (const std::invalid_argument &) { rejected = true; }
  Require(error < 2.e-12 && rejected,"ordinary periodic translation/capability failed");
  std::cout << "ordinary periodic translation max_error=" << error << '\n';
}
void PhaseContinuity(bool mixed, bool isolated) {
  auto d = Domain(8,3,isolated);
  if (isolated) d.c1_z_interfaces = {.25,.75};
  auto sample = [&](double phase) {
    d.shear_offset = phase;
    MeshGeometry geometry(Patches(8,3,mixed),d,0,1,isolated);
    const auto s = geometry.BuildSystem();
    std::vector<double> phi(s.cells.size()), boundary(s.boundary_samples.size());
    for (size_t i = 0; i < phi.size(); ++i) {
      const auto x = s.cells[i].center;
      phi[i] = std::sin(2*M_PI*x[0]+.37)*std::cos(2*M_PI*x[1]+.21)+
               .13*std::cos(4*M_PI*x[2]+.11);
    }
    for (size_t i = 0; i < boundary.size(); ++i) {
      const auto &b = s.boundary_samples[i];
      const double wall = b.center[2]-b.side*.5*s.boundary_width[i];
      boundary[i] = b.side*(-.52*M_PI*std::sin(4*M_PI*wall+.11));
    }
    std::array<std::vector<double>,4> values;
    for (auto &v : values) v.resize(phi.size());
    for (size_t i = 0; i < phi.size(); ++i) {
      values[0][i] = Apply(s.matrix,i,phi)-Apply(s.boundary_rhs,i,boundary);
      for (int q = 0; q < 3; ++q)
        values[q+1][i] = Apply(s.acceleration[q],i,phi)+
                         Apply(s.boundary_acceleration[q],i,boundary);
    }
    return values;
  };
  for (double center : {0.,1./8.}) {
    const auto exact = sample(center);
    double previous = 1.e100;
    for (double epsilon : {1.e-6,1.e-8,1.e-10}) {
      double error = 0.;
      for (double sign : {-1.,1.}) {
        const auto shifted = sample(center+sign*epsilon);
        for (int q = 0; q < 4; ++q) for (size_t i = 0; i < exact[q].size(); ++i)
          error = std::max(error,std::abs(shifted[q][i]-exact[q][i]));
      }
      std::cout << "phase continuity mixed=" << mixed << " isolated=" << isolated
                << " center=" << center << " epsilon=" << epsilon << " error=" << error << '\n';
      Require(error < previous*.05+5.e-10,"A/G has a finite jump at an aligned shear phase");
      previous = error;
    }
  }
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
  Polynomials(2,false); Polynomials(3,false); Polynomials(3,true); Polynomials(3,true,true);
  const double coarse = CubicForce(16), fine = CubicForce(32);
  std::cout << "cubic interface force errors " << coarse << " -> " << fine << '\n';
  Require(coarse/fine > 3.8,"interface force lost second-order convergence");
  for (int nd : {2,3}) for (bool mixed : {false,true}) Phases(nd,mixed,false,false);
  Phases(3,true,true,false); Phases(3,true,true,true);
  Phases(3,true,true,true,8,true);
  Phases(2,false,false,false,2); Phases(3,false,false,false,2);
  Phases(2,false,false,false,1); Phases(3,false,false,false,1);
  Reference(false); Reference(true);
  PiecewiseVacuum(); CoarseC1Fallback();
  VirtualImages();
  OrdinaryPeriodicTranslation();
  PhaseContinuity(false,false); PhaseContinuity(true,false); PhaseContinuity(true,true);
  E3Multilevel({1.,1.,1.}); E3Multilevel({.2,.2,.4});
  std::cout << "gravity geometry checks passed\n";
}
