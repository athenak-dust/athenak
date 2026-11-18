//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
// Irregular rows on the device (see gpu_irregular.hpp).  This file is compiled with
// -ffp-contract=off (host compiler) and --fmad=false (CUDA): every rounding is the one written.
// The host compiler (GCC -O3, default -ffp-contract=fast) contracts a product whose only uses
// are additions; the host expressions below that it contracts are written with Kokkos::fma:
// the squared distances of the fit samples, the normal matrix accumulation, the Gauss-Jordan
// eliminations, the moment-coefficient dot products and the Faces() probe.  All other products
// feed divisions, products or stores, or are exact (by +-1, 0.5, 0.25, 0.125).
// Scope: this mirror of the host's contraction was verified bitwise (test_gpu_assembly = 2
// and the CPU suite) only for the tested configuration: GCC 11.5 (host and CUDA host compiler)
// with the Release flags (-O3, default contraction) for mesh_geometry.cpp, CUDA 12.5 (nvcc),
// on aarch64 (GH200).  Only this file pins its contraction flags; another compiler, a Debug
// build or other floating-point flags for the host files can contract differently, and the
// device rows then need not be bitwise equal to the host's (mode 2 reports it).
// Shared tables written by the row-parallel kernels (the fit and AddDiagonal marks) use
// atomics; every other store goes to the thread's own row, fit or entry.
#include "gravity/gpu_irregular.hpp"

#include <chrono>
#include <cstdint>

namespace gravity {
namespace gpu {
namespace {

constexpr int kS = 48;    // samples of a two-ring fit
constexpr int kD = 128;   // a face's potential difference / face potential
constexpr int kA = 192;   // a row component
enum Status { kOk = 0, kFailX = -1, kFailWall = -2, kFailGeometry = -3, kFailSamples = -4,
              kFailRank = -5, kFailDuplicate = -6, kFailBuffer = -7 };

struct Dev {
  Geometry g;
  Irregular t;
};

// The host's IndexedRow (+ Append): entries in first-appearance order with a hash index of the
// columns (open addressing, H >= 2 N slots), every column's contributions summed in the order
// they arrive; Sort orders the entries by column as Append does before it stores them.  A face's
// difference/potential row is added to the row components column by column (each column once
// per face), so the order of its entries does not matter.
template <int N, int H>
struct HRow {
  GlobalIndex col[N];
  double val[N];
  unsigned short slot[H];   // entry index + 1, 0 empty
  int n = 0;
  bool overflow = false;
  KOKKOS_INLINE_FUNCTION void Clear() {
    n = 0; overflow = false;
    for (int h = 0; h < H; ++h) slot[h] = 0;
  }
  KOKKOS_INLINE_FUNCTION static int Hash(GlobalIndex c) {
    const std::uint64_t x = static_cast<std::uint64_t>(c)*0x9e3779b97f4a7c15ULL;
    return static_cast<int>((x^(x >> 29))&(H-1));
  }
  KOKKOS_INLINE_FUNCTION void Add(GlobalIndex c, double v) {
    if (v == 0.) return;
    int h = Hash(c);
    while (slot[h]) {
      const int i = slot[h]-1;
      if (col[i] == c) { val[i] += v; return; }
      h = (h+1)&(H-1);
    }
    if (n == N) { overflow = true; return; }
    col[n] = c; val[n] = v; ++n;
    slot[h] = static_cast<unsigned short>(n);
  }
  KOKKOS_INLINE_FUNCTION int Nonzeros() const {
    int k = 0;
    for (int i = 0; i < n; ++i) k += val[i] != 0.;
    return k;
  }
  // by column (unique columns); the index is stale afterwards
  KOKKOS_INLINE_FUNCTION void Sort() {
    for (int gap = n/2; gap > 0; gap /= 2)
      for (int i = gap; i < n; ++i) {
        const GlobalIndex c = col[i]; const double v = val[i];
        int j = i;
        for (; j >= gap && col[j-gap] > c; j -= gap) { col[j] = col[j-gap]; val[j] = val[j-gap]; }
        col[j] = c; val[j] = v;
      }
  }
};
// MeshGeometry::Impl::Patch / Indices of a global real row
KOKKOS_INLINE_FUNCTION Ref RefOf(const Dev &v, GlobalIndex row) {
  int lo = 0, hi = static_cast<int>(v.t.rorder.extent(0));   // first p with row < row_begin(p)
  while (lo < hi) {
    const int mid = (lo+hi)/2;
    if (row < v.g.row_begin(v.t.rorder(mid))) hi = mid; else lo = mid+1;
  }
  Ref r;
  r.p = v.t.rorder(lo > 0 ? lo-1 : 0);
  GlobalIndex q = row-v.g.row_begin(r.p);
  r.i[0] = static_cast<int>(q%v.g.cells(r.p,0)); q /= v.g.cells(r.p,0);
  r.i[1] = static_cast<int>(q%v.g.cells(r.p,1)); q /= v.g.cells(r.p,1);
  r.i[2] = static_cast<int>(q);
  return r;
}

// MeshGeometry::Impl::Faces(row) in order, each face passed with the other cell to
// visit(face, other cell) (a nonzero return ends the enumeration with that status)
template <class Visit>
KOKKOS_INLINE_FUNCTION int ForFaces(const Dev &v, const Ref &x, const Cell3 &ci, Visit &&visit) {
  const auto &g = v.g;
  const GlobalIndex row = Id(g,x);
  double coverage[3] = {0., 0., 0.};
  for (int d = 0; d < g.nd; ++d) for (int side = -1; side <= 1; side += 2) {
    Ref y = x; y.i[d] += side;
    if (y.i[d] >= 0 && y.i[d] < g.cells(x.p,d)) {
      const Cell3 cj = GetCell(g,y);
      const double image[3] = {0., 0., 0.};
      const FaceG f = MakeFace(g,row,ci,Id(g,y),cj,image,d,side);
      if (f.area > 0. && !(f.distance > 0.)) return kFailGeometry;
      coverage[d] += f.area;
      const int s = visit(f,cj);
      if (s) return s;
      continue;
    }
    const double edge = ci.c[d]+side*.5*ci.w[d];
    const bool periodic = Kokkos::fabs(edge-(side < 0 ? g.dlo[d] : g.dhi[d])) < g.minwidth[d]*1.e-5;
    if (periodic && d == 2 && v.t.isolated_z) return kFailWall;   // wall samples: host
    if (periodic && d == 0) return kFailX;                          // x images (shear): host
    double image[3] = {0., 0., 0.};
    if (periodic) image[d] = side*g.length[d];
    double qlo[3] = {0., 0., 0.}, qhi[3] = {0., 0., 0.};
    for (int t = 0; t < g.nd; ++t) {
      qlo[t] = ci.c[t]-.5*ci.w[t]-image[t];
      qhi[t] = ci.c[t]+.5*ci.w[t]-image[t];
    }
    const double probe = Kokkos::fma(side*g.minwidth[d],1.e-6,edge-image[d]);
    qlo[d] = probe; qhi[d] = probe;
    const int k = 6*x.p+2*d+(side > 0);
    for (int c = v.t.cand_off(k); c < v.t.cand_off(k+1); ++c) {
      const int q = v.t.cand(c);
      bool hit = true;
      for (int t = 0; t < g.nd; ++t)
        if (v.t.phi(q,t) < qlo[t] || qhi[t] < v.t.plo(q,t)) hit = false;
      if (!hit) continue;
      int first[3] = {0, 0, 0}, last[3] = {0, 0, 0};
      bool empty = false;
      for (int t = 0; t < g.nd; ++t) {
        const double h = g.width(q,t);
        if (t == d) {
          first[t] = last[t] = static_cast<int>(Kokkos::floor((probe-v.t.plo(q,t))/h));
        } else {
          first[t] = static_cast<int>(Kokkos::floor((qlo[t]-v.t.plo(q,t))/h+1.e-8));
          last[t] = static_cast<int>(Kokkos::ceil((qhi[t]-v.t.plo(q,t))/h-1.e-8))-1;
        }
        if (first[t] < 0) first[t] = 0;
        if (last[t] > g.cells(q,t)-1) last[t] = g.cells(q,t)-1;
        if (first[t] > last[t]) empty = true;
      }
      if (empty) continue;
      for (int kk = first[2]; kk <= last[2]; ++kk)
        for (int j = first[1]; j <= last[1]; ++j)
          for (int i = first[0]; i <= last[0]; ++i) {
            Ref z; z.p = q; z.i[0] = i; z.i[1] = j; z.i[2] = kk;
            const Cell3 cj = GetCell(g,z);
            if (Kokkos::fabs(cj.c[d]-side*.5*cj.w[d]+image[d]-edge) > g.minwidth[d]*1.e-5) continue;
            const FaceG f = MakeFace(g,row,ci,Id(g,z),cj,image,d,side);
            if (f.area == 0.) continue;
            if (!(f.distance > 0.)) return kFailGeometry;
            coverage[d] += f.area;
            const int s = visit(f,cj);
            if (s) return s;
          }
    }
  }
  double volume = 1.;
  for (int d = 0; d < g.nd; ++d) volume *= ci.w[d];
  for (int d = 0; d < g.nd; ++d) {
    const double expected = 2.*volume/ci.w[d];
    if (Kokkos::fabs(coverage[d]-expected) > expected*1.e-5) return kFailGeometry;
  }
  return kOk;
}

struct Sample { GlobalIndex id; double delta[3]; };
// MeshGeometry::Impl::Quad of one cell (built by the fit kernels): the samples, the inverse
// normal matrix and (with coefficients) the center coefficient
struct QuadFit {
  GlobalIndex id = -1;
  int n = 0;
  Cell3 cell;
  Sample s[kS];
  double inv[9][9];
  double center[9];
};

KOKKOS_INLINE_FUNCTION void Basis(const Cell3 &cell, const double *delta, int nd, double *b) {
  for (int i = 0; i < 9; ++i) b[i] = 0.;
  for (int d = 0; d < nd; ++d) b[d] = delta[d]/cell.w[d];
  int k = nd;
  for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t) b[k++] = b[d]*b[t]*(d == t ? .5 : 1.);
}
// the coefficient (mode values) of sample si
KOKKOS_INLINE_FUNCTION void Coefficient(const QuadFit &q, int si, int nd, int modes, double *value) {
  double b[9];
  Basis(q.cell,q.s[si].delta,nd,b);
  double distance = 0.;
  for (int d = 0; d < nd; ++d) distance = Kokkos::fma(b[d],b[d],distance);
  for (int i = 0; i < 9; ++i) value[i] = 0.;
  for (int i = 0; i < modes; ++i)
    for (int j = 0; j < modes; ++j) value[i] += q.inv[i][j]*b[j]/distance;
}

// the samples and inverse of the fit of cell id (the center coefficient only with coefficients)
KOKKOS_INLINE_FUNCTION int BuildQuad(const Dev &v, GlobalIndex id, QuadFit &q, bool coefficients) {
  const auto &g = v.g;
  const int nd = g.nd;
  q.id = id; q.n = 0;
  q.cell = GetCell(g,RefOf(v,id));
  const Cell3 &cell = q.cell;
  const bool one_sided = g.nplanes > 0;
  auto add = [&](GlobalIndex sid, const double *delta) -> int {
    const double x = cell.c[0]+delta[0];
    if (x < g.dlo[0] || x >= g.dhi[0]) return kFailX;   // an x image: host
    if (one_sided) {
      for (int k = 0; k < g.nplanes; ++k)
        if ((cell.c[2] < g.planes[k]) != (cell.c[2]+delta[2] < g.planes[k])) return kOk;
    }
    double distance = 0.;
    for (int d = 0; d < nd; ++d) {
      const double u = delta[d]/cell.w[d];
      distance = Kokkos::fma(u,u,distance);
    }
    if (distance < 1.e-20) return kOk;
    for (int i = 0; i < q.n; ++i) if (q.s[i].id == sid) {
      double error = 0.;
      for (int d = 0; d < nd; ++d) error += Kokkos::fabs(q.s[i].delta[d]-delta[d])/cell.w[d];
      if (error < 1.e-8) return kOk;
    }
    if (q.n == kS) return kFailSamples;
    q.s[q.n].id = sid;
    for (int d = 0; d < 3; ++d) q.s[q.n].delta[d] = delta[d];
    ++q.n;
    return kOk;
  };
  auto neighbors = [&](GlobalIndex nid, const double *offset) -> int {
    const Ref rn = RefOf(v,nid);
    return ForFaces(v,rn,GetCell(g,rn),[&](const FaceG &f, const Cell3 &) -> int {
      const double sign = f.side;
      double delta[3] = {offset[0], offset[1], offset[2]};
      for (int d = 0; d < nd; ++d) delta[d] += sign*f.delta[d];
      return add(nid == f.left ? f.right : f.left,delta);
    });
  };
  const double zero[3] = {0., 0., 0.};
  int status = neighbors(id,zero);
  const int last = q.n;
  for (int i = 0; status == kOk && i < last; ++i) {
    const Sample s = q.s[i];
    if (s.id < g.n) status = neighbors(s.id,s.delta);
  }
  if (status) return status;
  // the coefficient list is the samples in order and the center last (repeated cells: host)
  for (int a = 0; a < q.n; ++a) {
    if (q.s[a].id == id) return kFailDuplicate;
    for (int b = 0; b < a; ++b) if (q.s[b].id == q.s[a].id) return kFailDuplicate;
  }
  // invert (full rank after two rings, else host)
  const int modes = nd+nd*(nd+1)/2;
  double normal[9][9];
  for (int i = 0; i < 9; ++i) for (int j = 0; j < 9; ++j) { normal[i][j] = 0.; q.inv[i][j] = 0.; }
  for (int s = 0; s < q.n; ++s) {
    double b[9];
    Basis(cell,q.s[s].delta,nd,b);
    double distance = 0.;
    for (int d = 0; d < nd; ++d) distance = Kokkos::fma(b[d],b[d],distance);
    const double weight = 1./distance;
    for (int i = 0; i < modes; ++i) {
      const double wb = weight*b[i];
      for (int j = 0; j < modes; ++j) normal[i][j] = Kokkos::fma(wb,b[j],normal[i][j]);
    }
  }
  for (int i = 0; i < modes; ++i) q.inv[i][i] = 1.;
  for (int i = 0; i < modes; ++i) {
    int pivot = i;
    for (int j = i+1; j < modes; ++j)
      if (Kokkos::fabs(normal[j][i]) > Kokkos::fabs(normal[pivot][i])) pivot = j;
    if (Kokkos::fabs(normal[pivot][i]) < 1.e-12) return kFailRank;
    if (pivot != i) {
      for (int j = 0; j < 9; ++j) {
        double a = normal[i][j]; normal[i][j] = normal[pivot][j]; normal[pivot][j] = a;
        a = q.inv[i][j]; q.inv[i][j] = q.inv[pivot][j]; q.inv[pivot][j] = a;
      }
    }
    const double scale = normal[i][i];
    for (int j = 0; j < modes; ++j) { normal[i][j] /= scale; q.inv[i][j] /= scale; }
    for (int j = 0; j < modes; ++j) if (j != i) {
      const double factor = normal[j][i];
      for (int k = 0; k < modes; ++k) {
        normal[j][k] = Kokkos::fma(-factor,normal[i][k],normal[j][k]);
        q.inv[j][k] = Kokkos::fma(-factor,q.inv[i][k],q.inv[j][k]);
      }
    }
  }
  if (!coefficients) return kOk;
  for (int i = 0; i < 9; ++i) q.center[i] = 0.;
  for (int s = 0; s < q.n; ++s) {
    double value[9];
    Coefficient(q,s,nd,modes,value);
    for (int i = 0; i < modes; ++i) q.center[i] -= value[i];
  }
  return kOk;
}

// a fit of the global table: -2 marked only (mark pass), -1 unavailable, else its index
struct FitRef { int q; };

// MeshGeometry::Impl::AddDiagonal's choice for cell id: the centered three-point branch (regular,
// with the face neighbors nb) or the fit
struct DiagInfo {
  int status = kOk;
  bool regular = false;
  GlobalIndex nb[3][2];
};
KOKKOS_INLINE_FUNCTION void Analyze(const Dev &v, GlobalIndex id, DiagInfo &info) {
  const auto &g = v.g;
  const int nd = g.nd;
  info.status = kOk;
  const Ref rr = RefOf(v,id);
  const Cell3 cell = GetCell(g,rr);
  auto &nb = info.nb;
  bool interior = true;
  for (int d = 0; d < nd; ++d) interior = interior && rr.i[d] > 0 && rr.i[d]+1 < g.cells(rr.p,d);
  bool regular = !NearPlane(g,cell) && !(g.shear && TouchesX(g,cell));
  for (int d = 0; d < 3; ++d) { nb[d][0] = 0; nb[d][1] = 0; }
  if (interior && regular) {
    for (int d = 0; d < nd; ++d) for (int side = 0; side < 2; ++side) {
      Ref y = rr; y.i[d] += 2*side-1; nb[d][side] = Id(g,y);
    }
  } else if (regular) {
    int count = 0;
    const int s = ForFaces(v,rr,cell,[&](const FaceG &f, const Cell3 &cj) -> int {
      ++count;
      const GlobalIndex other = f.side > 0 ? f.right : f.left;
      regular = regular && !HasOffset(f.ti,nd) && !HasOffset(f.tj,nd) &&
                !AtC1(g,f,f.side > 0 ? cell : cj);
      if (other < g.n)
        for (int d = 0; d < nd; ++d)
          regular = regular && Kokkos::fabs(cj.w[d]-cell.w[d]) < g.minwidth[d]*1.e-8;
      nb[f.normal][f.side > 0 ? 1 : 0] = other;
      return kOk;
    });
    if (s) { info.status = s; return; }
    regular = regular && count == 2*nd;
  }
  info.regular = regular;
}

// sum_i moment[i] coefficient[i] of every coefficient of fit q (AssembleRow's reconstruct,
// FacePotential's side), added to row
template <class Row>
KOKKOS_INLINE_FUNCTION void Apply(const Quads &f, int q, const double *moment, int k, Row &row) {
  for (GlobalIndex s = f.off(q); s < f.off(q+1); ++s) {
    double value = 0.;
    for (int i = 0; i < k; ++i) value = Kokkos::fma(moment[i],f.val(s,i),value);
    row.Add(f.col(s),value);
  }
  double value = 0.;
  for (int i = 0; i < k; ++i) value = Kokkos::fma(moment[i],f.center(q,i),value);
  row.Add(f.cell(q),value);
}

enum Pass { kMark = 0, kRows = 1 };

// MeshGeometry::Impl::AssembleRow of one owned row; emit(component, row) when a component is
// complete (1+d after the faces of normal d, 0 at the end).  kMark: only marks the fits the row
// needs (f.index(id) = 1).
template <int P, class Emit>
KOKKOS_INLINE_FUNCTION int AssembleIrregular(const Dev &v, const Quads &fits, const Ref &r,
                                             bool forces, Emit &emit, bool &interface) {
  const auto &g = v.g;
  const int nd = g.nd;
  const GlobalIndex row = Id(g,r);
  const Cell3 ci = GetCell(g,r);
  double volume = 1.;
  for (int d = 0; d < nd; ++d) volume *= ci.w[d];
  HRow<kA,512> a, acc;
  HRow<kD,256> tmp;
  a.Clear(); acc.Clear();
  interface = false;
  int normal = -1;
  auto quad = [&](GlobalIndex id) -> FitRef {
    // (neighboring rows mark the same cell: atomic, the value is the same)
    if (P == kMark) { Kokkos::atomic_exchange(&fits.index(id),1); return FitRef{-2}; }
    const int q = fits.index(id);
    if (q < 0 || fits.status(q) != kOk) return FitRef{-1};
    return FitRef{q};
  };
  auto failed = [&](GlobalIndex id) -> int {
    const int q = fits.index(id);
    return q < 0 ? kFailRank : fits.status(q);
  };
  // AddDiagonal(target, id, weights): the centered three-point branch or the fit, as the table of
  // analyses says (the mark pass only marks the cell for the table)
  auto diagonal = [&](GlobalIndex id, const double *weights) -> int {
    if (P == kMark) { Kokkos::atomic_exchange(&fits.dindex(id),1); return kOk; }
    const int e = fits.dindex(id);
    if (e < 0) return kFailRank;
    const int analysis = fits.dstatus(e);
    if (analysis < 0) return analysis;
    const Cell3 cell = GetCell(g,RefOf(v,id));
    if (analysis == 1) {
      for (int d = 0; d < nd; ++d) {
        const double coefficient = weights[d]/(cell.w[d]*cell.w[d]);
        tmp.Add(fits.dnb(e,2*d),coefficient); tmp.Add(fits.dnb(e,2*d+1),coefficient);
        tmp.Add(id,-2.*coefficient);
      }
      return kOk;
    }
    const FitRef q = quad(id);
    if (q.q == -2) return kOk;
    if (q.q < 0) return failed(id);
    for (GlobalIndex s = fits.off(q.q); s <= fits.off(q.q+1); ++s) {
      const bool center = s == fits.off(q.q+1);
      double coefficient = 0.;
      int k = nd;
      for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t) {
        if (d == t) {
          const double c = center ? fits.center(q.q,k) : fits.val(s,k);
          coefficient += weights[d]*c/(cell.w[d]*cell.w[d]);
        }
        ++k;
      }
      tmp.Add(center ? fits.cell(q.q) : fits.col(s),coefficient);
    }
    return kOk;
  };
  const int status = ForFaces(v,r,ci,[&](const FaceG &f, const Cell3 &cj) -> int {
    if (P != kMark && forces && f.normal != normal) {
      if (normal >= 0) {
        if (acc.overflow) return kFailBuffer;
        emit(1+normal,acc);
      }
      acc.Clear(); normal = f.normal;
    }
    const Cell3 &left = f.side > 0 ? ci : cj;
    const Cell3 &right = f.side > 0 ? cj : ci;
    const int n = f.normal;
    tmp.Clear();
    tmp.Add(f.right,1.); tmp.Add(f.left,-1.);
    const bool c1 = AtC1(g,f,left);
    const double shift = .25*(left.w[n]-right.w[n]);
    if (HasOffset(f.ti,nd) || HasOffset(f.tj,nd) || c1 || left.w[n] != right.w[n]) {
      auto reconstruct = [&](GlobalIndex id, const Cell3 &cell, const double *offset,
                             double sign) -> int {
        const FitRef q = quad(id);
        if (q.q == -2) return kOk;
        if (q.q < 0) return failed(id);
        double moment[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
        for (int d = 0; d < nd; ++d) moment[d] = sign*offset[d]/cell.w[d];
        int k = nd;
        for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t) {
          moment[k] = sign*offset[d]*offset[t]/(cell.w[d]*cell.w[t])*(d == t ? .5 : 1.);
          if (d == n && t == d) {
            if (c1) moment[k] -= sign*.125;
            else moment[k] += .5*f.distance*shift/(cell.w[d]*cell.w[d]);
          }
          ++k;
        }
        Apply(fits,q.q,moment,k,tmp);
        return kOk;
      };
      int s = reconstruct(f.right,right,f.tj,1.);
      if (s) return s;
      s = reconstruct(f.left,left,f.ti,-1.);
      if (s) return s;
    }
    if (tmp.overflow) return kFailBuffer;
    const double sign = f.side;
    if (P != kMark) {
      for (int e = 0; e < tmp.n; ++e) {
        const double flux = f.area/f.distance*tmp.val[e];
        if (f.left != f.right) a.Add(tmp.col[e],-sign*flux);
      }
    }
    if (forces) {
      // FacePotential (two real cells)
      tmp.Clear();
      const double weight_left = right.w[n]/(left.w[n]+right.w[n]);
      const bool transverse = HasOffset(f.ti,nd) || HasOffset(f.tj,nd);
      auto side = [&](GlobalIndex id, const Cell3 &cell, const double *offset, double weight) -> int {
        tmp.Add(id,weight);
        double diag[3] = {0., 0., 0.};
        for (int d = 0; d < nd; ++d)
          diag[d] = weight*(d == n ? -.125*cell.w[d]*cell.w[d] : f.width[d]*f.width[d]/24.);
        if (!transverse) return diagonal(id,diag);
        const FitRef q = quad(id);
        if (q.q == -2) return kOk;
        if (q.q < 0) return failed(id);
        double moment[9] = {0., 0., 0., 0., 0., 0., 0., 0., 0.};
        for (int d = 0; d < nd; ++d) moment[d] = weight*offset[d]/cell.w[d];
        int k = nd;
        for (int d = 0; d < nd; ++d) for (int t = d; t < nd; ++t) {
          moment[k] = weight*offset[d]*offset[t]/(cell.w[d]*cell.w[t])*(d == t ? .5 : 1.);
          if (d == t) moment[k] += diag[d]/(cell.w[d]*cell.w[d]);
          ++k;
        }
        Apply(fits,q.q,moment,k,tmp);
        return kOk;
      };
      int s = side(f.left,left,f.ti,weight_left);
      if (s) return s;
      s = side(f.right,right,f.tj,1.-weight_left);
      if (s) return s;
      if (tmp.overflow) return kFailBuffer;
      if (P != kMark)
        for (int e = 0; e < tmp.n; ++e) acc.Add(tmp.col[e],-sign*f.area*tmp.val[e]/volume);
    }
    const GlobalIndex other = row == f.left ? f.right : f.left;
    if (other < g.n && Kokkos::fabs(ci.w[0]-cj.w[0]) > g.minwidth[0]*1.e-5) interface = true;
    return kOk;
  });
  if (status) return status;
  if (P == kMark) return kOk;
  if (forces && normal >= 0) {
    if (acc.overflow) return kFailBuffer;
    emit(1+normal,acc);
  }
  if (a.overflow) return kFailBuffer;
  emit(0,a);
  return kOk;
}

struct MarkFits {
  Dev v;
  Quads fits;
  bool forces;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    auto emit = [](int, HRow<kA,512> &) {};
    bool interface = false;
    AssembleIrregular<kMark>(v,fits,LocalRef(v.g,v.t.rows(i)),forces,emit,interface);
  }
};
// per marked AddDiagonal cell: the analysis; the fits of the cells without the centered branch
struct AnalyzeDiagonals {
  Dev v;
  Quads fits;
  KOKKOS_INLINE_FUNCTION void operator()(const int e) const {
    const GlobalIndex id = fits.dcell(e);
    DiagInfo info;
    Analyze(v,id,info);
    fits.dstatus(e) = info.status < 0 ? info.status : (info.regular ? 1 : 0);
    for (int d = 0; d < 3; ++d) for (int side = 0; side < 2; ++side)
      fits.dnb(e,2*d+side) = info.nb[d][side];
    // (one entry per cell; atomic as the marks of the row kernels)
    if (info.status == kOk && !info.regular) Kokkos::atomic_exchange(&fits.index(id),1);
  }
};
// per fit: build it (without coefficients): its status and sample count
struct CountFits {
  Dev v;
  Quads fits;
  KOKKOS_INLINE_FUNCTION void operator()(const int q) const {
    QuadFit fit;
    const int s = BuildQuad(v,fits.cell(q),fit,false);
    fits.status(q) = s;
    fits.off(q) = s == kOk ? fit.n : 0;
  }
};
// per built fit: the sample cells and coefficients, the center coefficient
struct FillFits {
  Dev v;
  Quads fits;
  KOKKOS_INLINE_FUNCTION void operator()(const int q) const {
    if (fits.status(q) != kOk) return;
    QuadFit fit;
    BuildQuad(v,fits.cell(q),fit,true);
    const int nd = v.g.nd, modes = nd+nd*(nd+1)/2;
    const GlobalIndex k = fits.off(q);
    for (int s = 0; s < fit.n; ++s) {
      double value[9];
      Coefficient(fit,s,nd,modes,value);
      fits.col(k+s) = fit.s[s].id;
      for (int i = 0; i < 9; ++i) fits.val(k+s,i) = value[i];
    }
    for (int i = 0; i < 9; ++i) fits.center(q,i) = fit.center[i];
  }
};

struct CountIrregular {
  Dev v;
  Quads fits;
  bool forces;
  int components;
  DeviceIndex count0, count1, count2, count3;
  DeviceFlags status;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const int row = v.t.rows(i);
    GlobalIndex k[4] = {0, 0, 0, 0};
    auto emit = [&](int c, HRow<kA,512> &a) { k[c] = a.Nonzeros(); };
    bool interface = false;
    const int s = AssembleIrregular<kRows>(v,fits,LocalRef(v.g,row),forces,emit,interface);
    status(row) = s ? (s < 0 ? s : -s) : (interface ? 2 : 1);
    if (s) k[0] = k[1] = k[2] = k[3] = 0;
    count0(row) = k[0];
    if (components > 1) count1(row) = k[1];
    if (components > 2) count2(row) = k[2];
    if (components > 3) count3(row) = k[3];
  }
};

struct FillIrregular {
  Dev v;
  Quads fits;
  bool forces;
  int components;
  DeviceIndex offset0, offset1, offset2, offset3, col0, col1, col2, col3;
  Kokkos::View<double*, DevMemSpace> val0, val1, val2, val3;
  DeviceFlags status;
  KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
    const int row = v.t.rows(i);
    if (status(row) <= 0) return;
    auto emit = [&](int c, HRow<kA,512> &a) {
      a.Sort();
      const DeviceIndex &off = c == 0 ? offset0 : c == 1 ? offset1 : c == 2 ? offset2 : offset3;
      const DeviceIndex &col = c == 0 ? col0 : c == 1 ? col1 : c == 2 ? col2 : col3;
      const Kokkos::View<double*, DevMemSpace> &val = c == 0 ? val0 : c == 1 ? val1 : c == 2 ? val2 : val3;
      GlobalIndex k = off(row);
      for (int e = 0; e < a.n; ++e) if (a.val[e] != 0.) { col(k) = a.col[e]; val(k) = a.val[e]; ++k; }
    };
    bool interface = false;
    AssembleIrregular<kRows>(v,fits,LocalRef(v.g,row),forces,emit,interface);
  }
};

}  // namespace

Quads IrregularCount(const Geometry &g, const Irregular &t, bool forces, int components,
                     const DeviceIndex &count0, const DeviceIndex &count1,
                     const DeviceIndex &count2, const DeviceIndex &count3,
                     const DeviceFlags &status, std::array<double,4> *seconds) {
  using Clock = std::chrono::steady_clock;
  auto t0 = Clock::now();
  auto lap = [&](int i) {
    Kokkos::fence();
    const auto t1 = Clock::now();
    if (seconds) (*seconds)[i] = std::chrono::duration<double>(t1-t0).count();
    t0 = t1;
  };
  Quads fits;
  const int n = static_cast<int>(t.rows.extent(0));
  if (n == 0) return fits;
  const Dev v{g,t};
  // the fits the rows need (marks), their list, then each fit once: count, offsets, fill
  fits.index = Kokkos::View<int*, DevMemSpace>("sg_gpu_fit_index",static_cast<size_t>(g.n));
  fits.dindex = Kokkos::View<int*, DevMemSpace>("sg_gpu_diagonal_index",static_cast<size_t>(g.n));
  Kokkos::parallel_for("sg_gpu_mark_fits",Kokkos::RangePolicy<DevExeSpace>(0,n),
                       MarkFits{v,fits,forces});
  // the AddDiagonal cells: list, analyses (which mark their fits)
  {
    auto dindex = fits.dindex;
    int diagonals = 0;
    Kokkos::parallel_scan("sg_gpu_diagonal_list",
        Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<int64_t>>(0,static_cast<int64_t>(g.n)),
        KOKKOS_LAMBDA(const int64_t i, int &update, const bool final) {
          const int marked = dindex(i) == 1;
          if (final) dindex(i) = marked ? update : -1;
          update += marked;
        },diagonals);
    fits.diagonals = diagonals;
    fits.dcell = Kokkos::View<GlobalIndex*, DevMemSpace>("sg_gpu_diagonal_cell",diagonals);
    fits.dstatus = Kokkos::View<int*, DevMemSpace>("sg_gpu_diagonal_status",diagonals);
    fits.dnb = Kokkos::View<GlobalIndex*[6], Kokkos::LayoutRight, DevMemSpace>(
        "sg_gpu_diagonal_neighbors",diagonals);
    auto dcell = fits.dcell;
    Kokkos::parallel_for("sg_gpu_diagonal_cells",
        Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<int64_t>>(0,static_cast<int64_t>(g.n)),
        KOKKOS_LAMBDA(const int64_t i) { if (dindex(i) >= 0) dcell(dindex(i)) = i; });
    Kokkos::parallel_for("sg_gpu_analyse_diagonals",Kokkos::RangePolicy<DevExeSpace>(0,diagonals),
                         AnalyzeDiagonals{v,fits});
  }
  auto index = fits.index;
  int count = 0;
  Kokkos::parallel_scan("sg_gpu_fit_list",
      Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<int64_t>>(0,static_cast<int64_t>(g.n)),
      KOKKOS_LAMBDA(const int64_t i, int &update, const bool final) {
        const int marked = index(i) == 1;
        if (final) index(i) = marked ? update : -1;
        update += marked;
      },count);
  fits.count = count;
  fits.cell = Kokkos::View<GlobalIndex*, DevMemSpace>("sg_gpu_fit_cell",count);
  fits.status = Kokkos::View<int*, DevMemSpace>("sg_gpu_fit_status",count);
  fits.off = Kokkos::View<GlobalIndex*, DevMemSpace>("sg_gpu_fit_offsets",count+1);
  fits.center = Kokkos::View<double*[9], Kokkos::LayoutRight, DevMemSpace>("sg_gpu_fit_center",count);
  auto cell = fits.cell;
  Kokkos::parallel_for("sg_gpu_fit_cells",
      Kokkos::RangePolicy<DevExeSpace,Kokkos::IndexType<int64_t>>(0,static_cast<int64_t>(g.n)),
      KOKKOS_LAMBDA(const int64_t i) { if (index(i) >= 0) cell(index(i)) = i; });
  lap(0);
  Kokkos::parallel_for("sg_gpu_count_fits",Kokkos::RangePolicy<DevExeSpace>(0,count),
                       CountFits{v,fits});
  GlobalIndex samples = 0;
  Kokkos::parallel_scan("sg_gpu_fit_offsets",Kokkos::RangePolicy<DevExeSpace>(0,count+1),
                        ScanCounts{fits.off},samples);
  fits.samples = samples;
  fits.col = Kokkos::View<GlobalIndex*, DevMemSpace>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,"sg_gpu_fit_columns"),samples);
  fits.val = Kokkos::View<double*[9], Kokkos::LayoutRight, DevMemSpace>(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,"sg_gpu_fit_values"),samples);
  lap(1);
  Kokkos::parallel_for("sg_gpu_fill_fits",Kokkos::RangePolicy<DevExeSpace>(0,count),
                       FillFits{v,fits});
  lap(2);
  Kokkos::parallel_for("sg_gpu_count_irregular",Kokkos::RangePolicy<DevExeSpace>(0,n),
      CountIrregular{v,fits,forces,components,count0,count1,count2,count3,status});
  lap(3);
  return fits;
}

void IrregularFill(const Geometry &g, const Irregular &t, const Quads &fits, bool forces,
                   int components, const std::array<DeviceIndex,4> &offset,
                   const std::array<DeviceIndex,4> &col,
                   const std::array<Kokkos::View<double*, DevMemSpace>,4> &val,
                   const DeviceFlags &status) {
  const int n = static_cast<int>(t.rows.extent(0));
  if (n == 0) return;
  Kokkos::parallel_for("sg_gpu_fill_irregular",Kokkos::RangePolicy<DevExeSpace>(0,n),
      FillIrregular{Dev{g,t},fits,forces,components,offset[0],offset[1],offset[2],offset[3],
                    col[0],col[1],col[2],col[3],val[0],val[1],val[2],val[3],status});
}

}  // namespace gpu
}  // namespace gravity
