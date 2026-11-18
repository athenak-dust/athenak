//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_GPU_ASSEMBLY_HPP_
#define GRAVITY_GPU_ASSEMBLY_HPP_
// Device assembly of the regular rows of a MeshGeometry system (test switch
// <gravity>/test_gpu_assembly, see mesh_geometry.cpp). Included by mesh_geometry.cpp only.
// The predicates and coefficients are templates on the geometry tables: the host classifies
// the rows with HostGeometry (std::vector storage owned by the caller), the kernels use the
// uploaded Geometry.
//
// A row is regular when the host's MeshGeometry::Impl::AssembleRow takes only its plain
// branches for it: every face of the row is a single aligned face to a real cell of the same
// normal width (no transverse offset, no C1 plane, no wall sample), the row does not touch an
// x boundary, and AddDiagonal of the row and of its face neighbors is the centered
// three-point branch (no quadratic fit). The predicates below are those of the host, on the
// same cell centers and widths (computed by the host code and uploaded), and the
// coefficients are the same expressions in the same order: the only inexact products are
// w*w, (-0.125*w)*w, fw*fw, area*(hi-lo) and (-sign*area)*v, none of which feeds an
// addition, so floating-point contraction cannot change a result on either side; every
// other product is by 0.5, 0.125 or +-1 and exact. Each column of a row is accumulated in
// the host's first-contribution order (zero contributions skipped), the row is sorted by
// column and exact zeros are dropped, as Append does. Rows that are not regular go to the
// irregular kernels (gpu_irregular.hpp) or stay on the host (MeshGeometry::PrepareSplit).

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "athena.hpp"
#include "gravity/sparse_mg.hpp"

namespace gravity {
namespace gpu {

constexpr int kMaxPlanes = 4;

// Patch tables (all patches of the geometry) and the owned rows.
struct Geometry {
  int nd = 3, shear = 0, nplanes = 0;
  double dlo[3] = {0, 0, 0}, dhi[3] = {0, 0, 0}, length[3] = {0, 0, 0};
  double minwidth[3] = {0, 0, 0}, planes[kMaxPlanes] = {0, 0, 0, 0};
  GlobalIndex n = 0;                                 // global real rows
  Kokkos::View<double*[3], Kokkos::LayoutRight, DevMemSpace> width;
  Kokkos::View<int*[3], Kokkos::LayoutRight, DevMemSpace> cells, coff;
  Kokkos::View<double*, DevMemSpace> center;        // per patch and axis: the cell centers
  Kokkos::View<GlobalIndex*, DevMemSpace> row_begin;
  Kokkos::View<int*[6], Kokkos::LayoutRight, DevMemSpace> nb;     // face neighbor or -1
  Kokkos::View<int*[6], Kokkos::LayoutRight, DevMemSpace> wrap;   // across the periodic boundary
  // owned patches in row order: patch index and first local row (plus the end)
  Kokkos::View<int*, DevMemSpace> owned;
  Kokkos::View<int*, DevMemSpace> owned_first;
};

// Host tables with the access syntax of the device views (raw storage owned by the caller).
template <class T, int M>
struct HostTable {
  T *p = nullptr;
  std::size_t n = 0;
  KOKKOS_INLINE_FUNCTION T &operator()(std::size_t i, int j) const { return p[i*M+j]; }
  KOKKOS_INLINE_FUNCTION std::size_t extent(int) const { return n; }
};
template <class T>
struct HostArray {
  T *p = nullptr;
  std::size_t n = 0;
  KOKKOS_INLINE_FUNCTION T &operator()(std::size_t i) const { return p[i]; }
  KOKKOS_INLINE_FUNCTION std::size_t extent(int) const { return n; }
};
struct HostGeometry {
  int nd = 3, shear = 0, nplanes = 0;
  double dlo[3] = {0, 0, 0}, dhi[3] = {0, 0, 0}, length[3] = {0, 0, 0};
  double minwidth[3] = {0, 0, 0}, planes[kMaxPlanes] = {0, 0, 0, 0};
  GlobalIndex n = 0;
  HostTable<double,3> width;
  HostTable<int,3> cells, coff;
  HostArray<double> center;
  HostArray<GlobalIndex> row_begin;
  HostTable<int,6> nb, wrap;
  HostArray<int> owned, owned_first;
};

struct Ref { int p, i[3]; };
struct Cell3 { double c[3], w[3]; };
struct FaceG {
  GlobalIndex left, right;
  int normal, side;
  double area, distance;
  double delta[3], ti[3], tj[3], width[3];
};

template <int N>
struct LRow {
  GlobalIndex col[N];
  double val[N];
  int n = 0;
  bool overflow = false;
  // the host's IndexedRow::Add: zero skipped, sum into an existing column, else append
  KOKKOS_INLINE_FUNCTION void Add(GlobalIndex c, double v) {
    if (v == 0.) return;
    for (int i = 0; i < n; ++i) if (col[i] == c) { val[i] += v; return; }
    if (n == N) { overflow = true; return; }
    col[n] = c; val[n] = v; ++n;
  }
  // the host's Append: sorted by column (columns are unique), zeros dropped
  KOKKOS_INLINE_FUNCTION void Sort() {
    for (int i = 1; i < n; ++i) {
      const GlobalIndex c = col[i]; const double v = val[i];
      int j = i-1;
      while (j >= 0 && col[j] > c) { col[j+1] = col[j]; val[j+1] = val[j]; --j; }
      col[j+1] = c; val[j+1] = v;
    }
  }
  KOKKOS_INLINE_FUNCTION int Nonzeros() const {
    int k = 0;
    for (int i = 0; i < n; ++i) k += val[i] != 0.;
    return k;
  }
};

template <class G>
KOKKOS_INLINE_FUNCTION GlobalIndex Id(const G &g, const Ref &r) {
  return g.row_begin(r.p)+r.i[0]+static_cast<GlobalIndex>(g.cells(r.p,0))*
      (r.i[1]+static_cast<GlobalIndex>(g.cells(r.p,1))*r.i[2]);
}
template <class G>
KOKKOS_INLINE_FUNCTION Cell3 GetCell(const G &g, const Ref &r) {
  Cell3 c;
  for (int d = 0; d < 3; ++d) { c.w[d] = g.width(r.p,d); c.c[d] = g.center(g.coff(r.p,d)+r.i[d]); }
  return c;
}
KOKKOS_INLINE_FUNCTION double Max(double a, double b) { return (a < b) ? b : a; }  // std::max
KOKKOS_INLINE_FUNCTION double Min(double a, double b) { return (b < a) ? b : a; }  // std::min
KOKKOS_INLINE_FUNCTION bool HasOffset(const double *v, int nd) {
  for (int d = 0; d < nd; ++d) if (v[d] != 0.) return true;
  return false;
}
template <class G>
KOKKOS_INLINE_FUNCTION bool TouchesX(const G &g, const Cell3 &c) {
  return Kokkos::fabs(c.c[0]-.5*c.w[0]-g.dlo[0]) < g.minwidth[0]*1.e-5 ||
         Kokkos::fabs(c.c[0]+.5*c.w[0]-g.dhi[0]) < g.minwidth[0]*1.e-5;
}
template <class G>
KOKKOS_INLINE_FUNCTION bool NearPlane(const G &g, const Cell3 &c) {
  bool near = false;
  for (int k = 0; k < g.nplanes; ++k)
    near = near || Kokkos::fabs(Kokkos::fabs(c.c[2]-g.planes[k])-.5*c.w[2]) < g.minwidth[2]*1.e-7;
  return near;
}
// face neighbor of x across (d, s): in the patch, or in the aligned same-shape neighbor patch;
// image = the host Faces() image component d of a periodic crossing
template <class G>
KOKKOS_INLINE_FUNCTION bool Step(const G &g, const Ref &x, int d, int s, Ref &y,
                                 double &image) {
  y = x; y.i[d] += s; image = 0.;
  if (y.i[d] >= 0 && y.i[d] < g.cells(x.p,d)) return true;
  const int q = g.nb(x.p,2*d+(s > 0));
  if (q < 0) return false;
  y.p = q; y.i[d] = s > 0 ? 0 : g.cells(q,d)-1;
  if (g.wrap(x.p,2*d+(s > 0))) image = s*g.length[d];
  return true;
}
// MeshGeometry::Impl::MakeFace
template <class G>
KOKKOS_INLINE_FUNCTION FaceG MakeFace(const G &g, GlobalIndex row, const Cell3 &ci,
    GlobalIndex other, const Cell3 &cj, const double *image, int d, int side) {
  FaceG f;
  f.normal = d; f.side = side; f.area = 1.; f.distance = 0.;
  for (int t = 0; t < 3; ++t) { f.delta[t] = 0.; f.ti[t] = 0.; f.tj[t] = 0.; f.width[t] = 0.; }
  const Cell3 &left = side > 0 ? ci : cj;
  const Cell3 &right = side > 0 ? cj : ci;
  f.left = side > 0 ? row : other; f.right = side > 0 ? other : row;
  for (int t = 0; t < g.nd; ++t) {
    const double shift = side*image[t];
    f.delta[t] = right.c[t]+shift-left.c[t];
    if (t == d) continue;
    const double lo = Max(static_cast<double>(left.c[t]-.5*left.w[t]),
                          right.c[t]-.5*right.w[t]+shift);
    const double hi = Min(static_cast<double>(left.c[t]+.5*left.w[t]),
                          right.c[t]+.5*right.w[t]+shift);
    if (hi-lo <= g.minwidth[t]*1.e-8) { f.area = 0.; break; }
    f.area *= hi-lo;
    f.width[t] = hi-lo;
    f.ti[t] = .5*(hi+lo)-left.c[t];
    f.tj[t] = .5*(hi+lo)-right.c[t]-shift;
    if (Kokkos::fabs(f.ti[t]) < g.minwidth[t]*1.e-12) f.ti[t] = 0.;
    if (Kokkos::fabs(f.tj[t]) < g.minwidth[t]*1.e-12) f.tj[t] = 0.;
  }
  f.distance = f.delta[d];
  return f;
}
// MeshGeometry::Impl::AtC1Interface for a face between two real cells (left = Cell(f.left))
template <class G>
KOKKOS_INLINE_FUNCTION bool AtC1(const G &g, const FaceG &f, const Cell3 &left) {
  if (f.normal != 2) return false;
  const double z = left.c[2]+.5*left.w[2];
  for (int k = 0; k < g.nplanes; ++k)
    if (Kokkos::fabs(z-g.planes[k]) < g.minwidth[2]*1.e-7) return true;
  return false;
}
// Why a row is left to the host (the first reason met; RegularRows stores -reason)
enum Reason { kRegular = 0, kRowTouchesX = 1, kNearPlane = 2, kNeighborTouchesX = 3,
              kLevelJump = 4, kDomainWall = 5, kFaceGeometry = 6, kOffset = 7, kC1Face = 8,
              kWidth = 9, kSelfFace = 10, kOverflow = 11, kReasons = 12 };
// The face of x across (d, s) as the host's Faces(x) gives it, if it is one plain face to a real
// cell (in the patch or the aligned neighbor patch) with positive area and distance.
template <class G>
KOKKOS_INLINE_FUNCTION int FaceOf(const G &g, const Ref &x, const Cell3 &cx, int d,
                                  int s, Ref &y, Cell3 &cy, FaceG &f) {
  double im = 0.;
  if (!Step(g,x,d,s,y,im)) return g.nb(x.p,2*d+(s > 0)) == -2 ? kDomainWall : kLevelJump;
  double image[3] = {0., 0., 0.};
  image[d] = im;
  cy = GetCell(g,y);
  f = MakeFace(g,Id(g,x),cx,Id(g,y),cy,image,d,s);
  return f.area > 0. && f.distance > 0. ? kRegular : kFaceGeometry;
}
// AddDiagonal(x) takes the centered three-point branch: its neighbors nbid[d][side]
template <class G>
KOKKOS_INLINE_FUNCTION int DiagRegular(const G &g, const Ref &x, GlobalIndex nbid[3][2]) {
  const Cell3 cx = GetCell(g,x);
  bool interior = true;
  for (int d = 0; d < g.nd; ++d) interior = interior && x.i[d] > 0 && x.i[d]+1 < g.cells(x.p,d);
  if (NearPlane(g,cx)) return kNearPlane;
  if (g.shear && TouchesX(g,cx)) return kNeighborTouchesX;
  for (int d = 0; d < g.nd; ++d) for (int side = 0; side < 2; ++side) {
    const int s = 2*side-1;
    if (interior) {
      Ref y = x; y.i[d] += s; nbid[d][side] = Id(g,y);
      continue;
    }
    Ref y; Cell3 cy; FaceG f;
    const int why = FaceOf(g,x,cx,d,s,y,cy,f);
    if (why) return why;
    if (HasOffset(f.ti,g.nd) || HasOffset(f.tj,g.nd)) return kOffset;
    if (AtC1(g,f,s > 0 ? cx : cy)) return kC1Face;
    for (int t = 0; t < g.nd; ++t)
      if (!(Kokkos::fabs(cy.w[t]-cx.w[t]) < g.minwidth[t]*1.e-8)) return kWidth;
    nbid[d][side] = Id(g,y);
  }
  return kRegular;
}
// A face of the row r that AssembleRow treats without reconstruction
template <class G>
KOKKOS_INLINE_FUNCTION int PlainFace(const G &g, const Ref &r, const Cell3 &cr, int d,
                                     int s, Ref &y, Cell3 &cy, FaceG &f) {
  const int why = FaceOf(g,r,cr,d,s,y,cy,f);
  if (why) return why;
  if (f.left == f.right) return kSelfFace;
  const Cell3 &left = s > 0 ? cr : cy;
  const Cell3 &right = s > 0 ? cy : cr;
  if (HasOffset(f.ti,g.nd) || HasOffset(f.tj,g.nd)) return kOffset;
  if (AtC1(g,f,left)) return kC1Face;
  return left.w[d] != right.w[d] ? kWidth : kRegular;
}
// an irregular row far enough (margin cell widths) from the x boundary and isolated-z walls for
// the irregular kernels (gpu_irregular.hpp)
template <class G>
KOKKOS_INLINE_FUNCTION bool Far(const G &g, const Cell3 &c, double margin, bool isolated_z) {
  bool far = c.c[0]-g.dlo[0] > margin*c.w[0] && g.dhi[0]-c.c[0] > margin*c.w[0];
  if (isolated_z) far = far && c.c[2]-g.dlo[2] > margin*c.w[2] && g.dhi[2]-c.c[2] > margin*c.w[2];
  return far;
}
// the whole row r is regular (every component; forces: also the accelerations): kRegular, else
// the first reason met.  Without forces the row needs plain faces only (A takes no AddDiagonal).
template <class G>
KOKKOS_INLINE_FUNCTION int Regular(const G &g, const Ref &r, bool forces) {
  const Cell3 cr = GetCell(g,r);
  if (TouchesX(g,cr)) return kRowTouchesX;   // phase rows (and x images) stay on the host
  GlobalIndex nbid[3][2];
  int why = forces ? DiagRegular(g,r,nbid) : kRegular;
  if (why) return why;
  for (int d = 0; d < g.nd; ++d) for (int s = -1; s <= 1; s += 2) {
    Ref y; Cell3 cy; FaceG f;
    why = PlainFace(g,r,cr,d,s,y,cy,f);
    if (why) return why;
    if (!forces) continue;
    why = DiagRegular(g,y,nbid);
    if (why) return why;
  }
  return kRegular;
}
// FacePotential side(): AddPotential(row, weight) and AddDiagonal (regular branch)
template <int N>
KOKKOS_INLINE_FUNCTION void Side(const FaceG &f, GlobalIndex row, const Cell3 &cell, double weight,
                                 const GlobalIndex nb[3][2], int nd, LRow<N> &pot) {
  pot.Add(row,weight);
  double diagonal[3] = {0., 0., 0.};
  for (int d = 0; d < nd; ++d)
    diagonal[d] = weight*(d == f.normal ? -.125*cell.w[d]*cell.w[d] :
                                          f.width[d]*f.width[d]/24.);
  for (int d = 0; d < nd; ++d) {
    const double coefficient = diagonal[d]/(cell.w[d]*cell.w[d]);
    pot.Add(nb[d][0],coefficient); pot.Add(nb[d][1],coefficient);
    pot.Add(row,-2.*coefficient);
  }
}
// component 0: A; component 1+d: acceleration d (rows that are Regular only)
template <class G, int N>
KOKKOS_INLINE_FUNCTION void Component(const G &g, const Ref &r, int component, LRow<N> &out) {
  out.n = 0; out.overflow = false;
  const Cell3 cr = GetCell(g,r);
  const GlobalIndex id = Id(g,r);
  if (component == 0) {
    for (int d = 0; d < g.nd; ++d) for (int s = -1; s <= 1; s += 2) {
      Ref y; Cell3 cy; FaceG f;
      PlainFace(g,r,cr,d,s,y,cy,f);
      LRow<2> difference;
      difference.Add(f.right,1.); difference.Add(f.left,-1.);
      const double sign = f.side;
      for (int k = 0; k < difference.n; ++k) {
        const double flux = f.area/f.distance*difference.val[k];
        if (f.left != f.right) out.Add(difference.col[k],-sign*flux);
      }
    }
    return;
  }
  const int d = component-1;
  double volume = 1.;
  for (int t = 0; t < g.nd; ++t) volume *= cr.w[t];
  GlobalIndex nr[3][2];
  DiagRegular(g,r,nr);
  for (int s = -1; s <= 1; s += 2) {
    Ref y; Cell3 cy; FaceG f;
    PlainFace(g,r,cr,d,s,y,cy,f);
    GlobalIndex ny[3][2];
    DiagRegular(g,y,ny);
    const bool r_left = s > 0;
    const Cell3 &left = r_left ? cr : cy;
    const Cell3 &right = r_left ? cy : cr;
    LRow<16> pot;
    const double weight_left = right.w[f.normal]/(left.w[f.normal]+right.w[f.normal]);
    Side(f,f.left,left,weight_left,r_left ? nr : ny,g.nd,pot);
    Side(f,f.right,right,1.-weight_left,r_left ? ny : nr,g.nd,pot);
    if (pot.overflow) { out.overflow = true; return; }
    const double sign = f.side;
    for (int k = 0; k < pot.n; ++k) out.Add(pot.col[k],-sign*f.area*pot.val[k]/volume);
  }
}
// the owned local row -> (patch, cell indices)
template <class G>
KOKKOS_INLINE_FUNCTION Ref LocalRef(const G &g, int row) {
  int lo = 0, hi = static_cast<int>(g.owned.extent(0))-1;   // owned_first(lo) <= row
  while (lo < hi) {
    const int mid = (lo+hi+1)/2;
    if (g.owned_first(mid) <= row) lo = mid; else hi = mid-1;
  }
  Ref r; r.p = g.owned(lo);
  int q = row-g.owned_first(lo);
  r.i[0] = q%g.cells(r.p,0); q /= g.cells(r.p,0);
  r.i[1] = q%g.cells(r.p,1); q /= g.cells(r.p,1);
  r.i[2] = q;
  return r;
}

using DeviceIndex = Kokkos::View<GlobalIndex*, DevMemSpace>;
using DeviceFlags = Kokkos::View<int*, DevMemSpace>;
constexpr int kRowA = 8, kRowG = 24;   // per-thread row buffers (A: 2 nd+1, forces: <= 17 in 3-D)

// count[c](row): stored entries of component c (0: A, 1+d: acceleration d, d < nd) of a
// regular row (regular(row) == 1, else 0); a row whose buffers overflow is no longer regular
struct CountRows {
  Geometry g;
  DeviceFlags regular;
  int components;
  DeviceIndex count0, count1, count2, count3;
  KOKKOS_INLINE_FUNCTION void operator()(const int row) const {
    GlobalIndex k[4] = {0, 0, 0, 0};
    if (regular(row) == 1) {   // (2: the irregular kernels, gpu_irregular.hpp)
      const Ref r = LocalRef(g,row);
      bool overflow = false;
      for (int c = 0; c < components; ++c) {
        if (c == 0) {
          LRow<kRowA> a; Component(g,r,c,a);
          overflow = overflow || a.overflow; k[c] = a.Nonzeros();
        } else {
          LRow<kRowG> a; Component(g,r,c,a);
          overflow = overflow || a.overflow; k[c] = a.Nonzeros();
        }
      }
      if (overflow) { regular(row) = -kOverflow; k[0] = k[1] = k[2] = k[3] = 0; }
    }
    count0(row) = k[0];
    if (components > 1) count1(row) = k[1];
    if (components > 2) count2(row) = k[2];
    if (components > 3) count3(row) = k[3];
  }
};
// in-place exclusive scan of n+1 counts (the last one zero): offsets, the total at n
struct ScanCounts {
  DeviceIndex a;
  KOKKOS_INLINE_FUNCTION void operator()(const int i, GlobalIndex &update, const bool last) const {
    const GlobalIndex v = a(i);
    if (last) a(i) = update;
    update += v;
  }
};
// the entries of the regular rows at their offsets
struct FillRows {
  Geometry g;
  DeviceFlags regular;
  int components;
  DeviceIndex offset0, offset1, offset2, offset3;
  DeviceIndex col0, col1, col2, col3;
  Kokkos::View<double*, DevMemSpace> val0, val1, val2, val3;
  template <int N>
  KOKKOS_INLINE_FUNCTION void Write(LRow<N> &a, GlobalIndex k, const DeviceIndex &col,
                                    const Kokkos::View<double*, DevMemSpace> &val) const {
    a.Sort();
    for (int i = 0; i < a.n; ++i) if (a.val[i] != 0.) { col(k) = a.col[i]; val(k) = a.val[i]; ++k; }
  }
  KOKKOS_INLINE_FUNCTION void operator()(const int row) const {
    if (regular(row) != 1) return;
    const Ref r = LocalRef(g,row);
    for (int c = 0; c < components; ++c) {
      if (c == 0) {
        LRow<kRowA> a; Component(g,r,c,a); Write(a,offset0(row),col0,val0);
      } else {
        LRow<kRowG> a; Component(g,r,c,a);
        if (c == 1) Write(a,offset1(row),col1,val1);
        else if (c == 2) Write(a,offset2(row),col2,val2);
        else Write(a,offset3(row),col3,val3);
      }
    }
  }
};

}  // namespace gpu
}  // namespace gravity
#endif  // GRAVITY_GPU_ASSEMBLY_HPP_
