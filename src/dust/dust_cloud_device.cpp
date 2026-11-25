//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud_device.cpp
//! \brief Device (Kokkos) implementation of the moving-particle P1 split of refinable
//! clouds (the only one since the host split was retired) and of the merge pass  // retired-compat
//! G.  Flag and count the pieces per particle, exclusive scans for the slots and the new tags,
//! then one fill kernel.  The particle order: kept particles in their order, then the pieces of
//! the flagged particles in theirs.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"
#include "dust_split_velocity.hpp"
#include <Kokkos_Sort.hpp>
#if defined(KOKKOS_ENABLE_CUDA)
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#endif

namespace dust {

namespace {

struct DPiece {Real off; Real frac; int lobe;};

// pieces of a cloud along one direction (port of the host 'pieces' lambda): offset in
// child half-widths, fraction of the cloud's mass, lobe code; lobe_old = 3 for a plain
// parent (center child with both edge children attached)
KOKKOS_INLINE_FUNCTION
int P1Pieces(const int lobe_old, const bool sep_lo, const bool sep_hi, DPiece out[3]) {
  int n = 0;
  const Real total = 0.5 + 0.25*((lobe_old & 1) ? 1 : 0) + 0.25*((lobe_old & 2) ? 1 : 0);
  const bool lo = (lobe_old & 1) && sep_lo, hi = (lobe_old & 2) && sep_hi;
  const int lobe_new = lobe_old & ~((lo ? 1 : 0) | (hi ? 2 : 0));
  const Real keep = 0.5 + 0.25*((lobe_new & 1) ? 1 : 0) + 0.25*((lobe_new & 2) ? 1 : 0);
  out[n].off = 0.0; out[n].frac = keep/total; out[n].lobe = lobe_new; ++n;
  if (lo) {out[n].off = -1.0; out[n].frac = 0.25/total; out[n].lobe = 0; ++n;}
  if (hi) {out[n].off = 1.0; out[n].frac = 0.25/total; out[n].lobe = 0; ++n;}
  return n;
}

// the MeshBlock of a new or moved cloud at (x, y), searched from the block m of its parent on
// the actual leaves; the parent's block if no block holds the position, or if the block is on
// another rank (the particle migration then moves it)
KOKKOS_INLINE_FUNCTION
int NewCloudBlock(const LeafTraversal &lt, const Real x, const Real y, const int m,
                  const int ncell12) {
  const int leaf = lt.LeafOwner(x, y, m);
  const int b = (leaf < 0) ? m : leaf/ncell12;
  return (b < lt.nmb) ? b : m;
}

}  // namespace



namespace {

// the level-L cell of a position (the keys that sort the pieces of a merge level), and the
// merge-cell mask: the level-L cell and the B cells around it carry no dilated level above L.
// The mask is evaluated on the actual leaves from the block m of the
// particle asking (LeafTraversal::MergeOK), no longer read from a map built on the level-L grid.
struct MergeMask {
  LeafTraversal lt;
  int B = 0;
  static constexpr int kL = CloudGeometry::kCloudLevelMask + 1;
  int n1[kL], n2[kL];
  Real dx1[kL], dx2[kL];
  Real x1min, x2min, x1max, x2max;
  KOKKOS_INLINE_FUNCTION
  std::int64_t Cell(const Real x, const Real y, const int L) const {   // J n1 + I
    const int I = CloudGeometry::WrapIndex(static_cast<int>(Kokkos::floor(
        (CloudGeometry::WrapPosition(x, x1min, x1max) - x1min)/dx1[L])), n1[L]);
    const int J = CloudGeometry::WrapIndex(static_cast<int>(Kokkos::floor(
        (CloudGeometry::WrapPosition(y, x2min, x2max) - x2min)/dx2[L])), n2[L]);
    return static_cast<std::int64_t>(J)*n1[L] + I;
  }
  KOKKOS_INLINE_FUNCTION
  bool In(const Real x, const Real y, const int L, const int m) const {
    return lt.MergeOK(x, y, L, B, m);
  }
};

KOKKOS_INLINE_FUNCTION
int ReleasePieces(const int lobe_old, DPiece out[3]) {
  if (lobe_old == 0) {out[0].off = 0.0; out[0].frac = 1.0; out[0].lobe = 0; return 1;}
  int n = 0;
  const Real total = 0.5 + 0.25*((lobe_old & 1) ? 1 : 0) + 0.25*((lobe_old & 2) ? 1 : 0);
  out[n].off = 0.0; out[n].frac = 0.5/total; out[n].lobe = 0; ++n;
  if (lobe_old & 1) {out[n].off = -1.0; out[n].frac = 0.25/total; out[n].lobe = 0; ++n;}
  if (lobe_old & 2) {out[n].off = 1.0; out[n].frac = 0.25/total; out[n].lobe = 0; ++n;}
  return n;
}

// the level-L cell of a position and the merge-cell mask on the 3-D leaves (MergeMask in 3-D)
struct MergeMask3D {
  LeafTraversal3D lt;
  int B = 0;
  static constexpr int kL = CloudGeometry::kCloudLevelMask + 1;
  int n1[kL], n2[kL], n3[kL];
  Real dx1[kL], dx2[kL], dx3[kL];
  Real x1min, x2min, x3min, x1max, x2max, x3max;
  KOKKOS_INLINE_FUNCTION
  void Index(const Real x, const Real y, const Real z, const int L, int &I, int &J, int &K) const {
    using T = CloudGeometry;
    I = T::WrapIndex(static_cast<int>(Kokkos::floor((T::WrapPosition(x, x1min, x1max) - x1min)/
                                                     dx1[L])), n1[L]);
    J = T::WrapIndex(static_cast<int>(Kokkos::floor((T::WrapPosition(y, x2min, x2max) - x2min)/
                                                     dx2[L])), n2[L]);
    K = T::WrapIndex(static_cast<int>(Kokkos::floor((T::WrapPosition(z, x3min, x3max) - x3min)/
                                                     dx3[L])), n3[L]);
  }
  KOKKOS_INLINE_FUNCTION
  std::int64_t Cell(const Real x, const Real y, const Real z, const int L) const {
    int I, J, K;
    Index(x, y, z, L, I, J, K);
    return (static_cast<std::int64_t>(K)*n2[L] + J)*n1[L] + I;
  }
  KOKKOS_INLINE_FUNCTION
  bool In(const Real x, const Real y, const Real z, const int L, const int m) const {
    return lt.MergeOK(x, y, z, L, B, m);
  }
};

// the per-dimension steps of the merge pass's far-field release written once
// (ReleaseFarLobesT), one specialization per traversal with each dimension's own code: the pieces
// of a lobed level-l cloud in a merge cell of level l - 1 (ReleaseN: their number, 0 if the cloud
// stays; ReleaseFill: the pieces written to slots s0.., the first keeps the parent's tag).
template <class LTT> struct GDim;

template <>
struct GDim<LeafTraversal> {
  using MM = MergeMask;
  template <class PR, class PI>
  KOKKOS_INLINE_FUNCTION
  static int ReleaseN(const MM &mm, const LeafTraversal &lt, const PR &pr, const PI &pi,
                      const int p) {
    const int code = pi(PLEV,p);
    const int lev = CloudGeometry::CloudLevel(code);
    const int l1 = CloudGeometry::CloudLobe1(code);
    const int l2 = CloudGeometry::CloudLobe2(code);
    if ((l1 == 0 && l2 == 0) || lev == 0 || !mm.In(pr(IPX,p), pr(IPY,p), lev - 1, lt.Anchor(pi(PGID,p)))) {
      return 0;
    }
    DPiece a[3], b[3];
    return ReleasePieces(l1, a)*ReleasePieces(l2, b);
  }
  template <class PR, class PI, class TO>
  KOKKOS_INLINE_FUNCTION
  static void ReleaseFill(const CloudGeometry &transfer, const LeafTraversal &lt,
                          const PR &pr, const PI &pi, const PR &prn, const PI &pin, const int p,
                          const int s0, const int nrdata, const int nidata, const int gids,
                          const int ncell12, const std::uint64_t tag0,
                          const std::uint64_t tstride, const TO &toff) {
    const int code = pi(PLEV,p);
    const int lev = CloudGeometry::CloudLevel(code);
    DPiece a[3], b[3];
    const int n1 = ReleasePieces(CloudGeometry::CloudLobe1(code), a);
    const int n2 = ReleasePieces(CloudGeometry::CloudLobe2(code), b);
    const Real e1 = transfer.CloudHalfWidth1(lev), e2 = transfer.CloudHalfWidth2(lev);
    int k = 0;
    for (int jb=0; jb<n2; ++jb) {
      for (int ja=0; ja<n1; ++ja) {
        const int s = s0 + k;
        const Real cx = CloudGeometry::WrapPosition(pr(IPX,p) + a[ja].off*e1,
                                                           transfer.x1min, transfer.x1max);
        const Real cy = CloudGeometry::WrapPosition(pr(IPY,p) + b[jb].off*e2,
                                                           transfer.x2min, transfer.x2max);
        for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
        for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
        prn(IPX,s) = cx; prn(IPX1,s) = cx; prn(IPY,s) = cy; prn(IPY1,s) = cy;
        prn(IPM,s) = pr(IPM,p)*a[ja].frac*b[jb].frac;
        pin(PLEV,s) = lev;
        pin(PGID,s) = gids + NewCloudBlock(lt, cx, cy, lt.Anchor(pin(PGID,s)), ncell12);
        if (k > 0) SetTag(pin, s, tag0 + static_cast<std::uint64_t>(toff(p) + k - 1)*tstride);
        ++k;
      }
    }
  }
  // the structure of a leaf t (budget_structure): false for a ghost cell
  template <class LM>
  KOKKOS_INLINE_FUNCTION
  static bool Structure(const LeafTraversal &ltb, const CloudGeometry &tr, const LM &lmass,
                        const int t, const int n12, const int is_, const int ie_, const int js_,
                        const int je_, const int, const int, const Real rfloor, Real &s) {
    const int n1 = ltb.ncells1;
    const int b = t/n12, r = t % n12, j = r/n1, i = r % n1;
    if (i < is_ || i > ie_ || j < js_ || j > je_) return false;
    const Real dx = ltb.box(b,2), dy = ltb.box(b,3);
    const Real xc = ltb.box(b,0) + (i - is_ + 0.5)*dx, yc = ltb.box(b,1) + (j - js_ + 0.5)*dy;
    const Real rho = lmass(0,t)/(dx*dy);
    const Real den = Kokkos::fmax(rho, rfloor);
    for (int d=0; d<4; ++d) {
      const Real x = CloudGeometry::WrapPosition(xc + ((d == 0) ? dx : ((d == 1) ? -dx : 0.0)),
                                                        tr.x1min, tr.x1max);
      const Real y = CloudGeometry::WrapPosition(yc + ((d == 2) ? dy : ((d == 3) ? -dy : 0.0)),
                                                        tr.x2min, tr.x2max);
      const int tn = ltb.LeafOwner(x, y, b);
      if (tn < 0) continue;
      const int bn = tn/n12;
      const Real rn = lmass(0,tn)/(ltb.box(bn,2)*ltb.box(bn,3));
      s = Kokkos::fmax(s, Kokkos::fabs(rn - rho)/den);
    }
    return true;
  }
};

template <>
struct GDim<LeafTraversal3D> {
  using MM = MergeMask3D;
  template <class PR, class PI>
  KOKKOS_INLINE_FUNCTION
  static int ReleaseN(const MM &mm, const LeafTraversal3D &lt, const PR &pr, const PI &pi,
                      const int p) {
    using T = CloudGeometry;
    const int code = pi(PLEV,p);
    const int lev = T::CloudLevel(code);
    const int l1 = T::CloudLobe1(code), l2 = T::CloudLobe2(code), l3 = T::CloudLobe3(code);
    if ((l1 == 0 && l2 == 0 && l3 == 0) || lev == 0 ||
        !mm.In(pr(IPX,p), pr(IPY,p), pr(IPZ,p), lev - 1, lt.Anchor(pi(PGID,p)))) {
      return 0;
    }
    DPiece a[3], b[3], c[3];
    return ReleasePieces(l1, a)*ReleasePieces(l2, b)*ReleasePieces(l3, c);
  }
  // (shear-periodic x1; beyond a physical x3 face: mirrored)
  template <class PR, class PI, class TO>
  KOKKOS_INLINE_FUNCTION
  static void ReleaseFill(const CloudGeometry &, const LeafTraversal3D &lt,
                          const PR &pr, const PI &pi, const PR &prn, const PI &pin, const int p,
                          const int s0, const int nrdata, const int nidata, const int gids,
                          const int ncell123, const std::uint64_t tag0,
                          const std::uint64_t tstride, const TO &toff) {
    using T = CloudGeometry;
    const int code = pi(PLEV,p);
    const int lev = T::CloudLevel(code);
    DPiece a[3], b[3], cc[3];
    const int n1 = ReleasePieces(T::CloudLobe1(code), a);
    const int n2 = ReleasePieces(T::CloudLobe2(code), b);
    const int n3 = ReleasePieces(T::CloudLobe3(code), cc);
    const Real e1 = lt.CloudHalfWidth1(lev), e2 = lt.CloudHalfWidth2(lev);
    const Real e3 = lt.CloudHalfWidth3(lev);
    int k = 0;
    for (int jc=0; jc<n3; ++jc) {
      for (int jb=0; jb<n2; ++jb) {
        for (int ja=0; ja<n1; ++ja) {
          const int s = s0 + k;
          Real cx = pr(IPX,p) + a[ja].off*e1, cy = pr(IPY,p) + b[jb].off*e2;
          Real cz = pr(IPZ,p) + cc[jc].off*e3;
          const int zmir = lt.WrapPos(cx, cy, cz);
          for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
          for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
          prn(IPX,s) = cx; prn(IPX1,s) = cx; prn(IPY,s) = cy; prn(IPY1,s) = cy;
          prn(IPZ,s) = cz; prn(IPZ1,s) = cz;
          prn(IPM,s) = pr(IPM,p)*a[ja].frac*b[jb].frac*cc[jc].frac;
          pin(PLEV,s) = lev;
          if (zmir && lt.x3bc == 1) {prn(IPVZ,s) = -prn(IPVZ,s); prn(IPVZ1,s) = -prn(IPVZ1,s);}
          const int leaf = lt.LeafOwner(cx, cy, cz, lt.Anchor(pin(PGID,s)));
          const int blk = (leaf < 0) ? lt.Anchor(pin(PGID,s)) : leaf/ncell123;
          pin(PGID,s) = gids + ((blk < lt.nmb) ? blk : lt.Anchor(pin(PGID,s)));
          if (k > 0) SetTag(pin, s, tag0 + static_cast<std::uint64_t>(toff(p) + k - 1)*tstride);
          ++k;
        }
      }
    }
  }
  template <class LM>
  KOKKOS_INLINE_FUNCTION
  static bool Structure(const LeafTraversal3D &lt, const CloudGeometry &, const LM &lmass,
                        const int t, const int ncell123, const int is_, const int ie_,
                        const int js_, const int je_, const int ks_, const int ke_,
                        const Real rfloor, Real &s) {
    int b, k, j, i;
    lt.Decode(t, b, k, j, i);
    if (i < is_ || i > ie_ || j < js_ || j > je_ || k < ks_ || k > ke_) return false;
    const Real dx = lt.box(b,3), dy = lt.box(b,4), dz = lt.box(b,5);
    const Real xc = lt.box(b,0) + (i - is_ + 0.5)*dx, yc = lt.box(b,1) + (j - js_ + 0.5)*dy;
    const Real zc = lt.box(b,2) + (k - ks_ + 0.5)*dz;
    const Real rho = lmass(0,t)/(dx*dy*dz);
    const Real den = Kokkos::fmax(rho, rfloor);
    for (int d=0; d<6; ++d) {
      Real x = xc + ((d == 0) ? dx : ((d == 1) ? -dx : 0.0));
      Real y = yc + ((d == 2) ? dy : ((d == 3) ? -dy : 0.0));
      Real z = zc + ((d == 4) ? dz : ((d == 5) ? -dz : 0.0));
      lt.WrapPos(x, y, z);
      const int tn = lt.LeafOwner(x, y, z, b);
      if (tn < 0) continue;
      const int bn = tn/ncell123;
      const Real rn = lmass(0,tn)/(lt.box(bn,3)*lt.box(bn,4)*lt.box(bn,5));
      s = Kokkos::fmax(s, Kokkos::fabs(rn - rho)/den);
    }
    return true;
  }
};

// the structure-relative pass budget (cloud_merge_budget_kappa > 0) written once:
// the budget of leaf t is max(min(beta, kappa s_t), 1e-12), s_t the largest |rho_n - rho_t| /
// max(rho_t, rfloor) over the neighbor leaves at one cell width beside its center (GDim::Structure:
// four in 2-D, six in 3-D, each dimension's own code), with the leaf masses at the entry of the pass
// (the owners' totals also on the halo leaves).  Only this rank's leaves are judged.
template <class LTT>
void BudgetStructureT(DustGasDrag *pd, const LTT &lt, const CloudGeometry &tr,
                      const DvceArray2D<Real> &lent, const DvceArray1D<Real> &lbeta, const int nbt,
                      const int nown, const int ncell, const int bnd[6], const Real beta,
                      const Real bkappa, const Real rfloor) {
  using GD = GDim<LTT>;
  constexpr bool k3 = (LeafApi<LTT>::kDim == 3);
  DvceArray2D<Real> lmass(std::string(std::string(k3 ? "budget3d_lmass" : "budget_lmass")), 1, nbt);
  Kokkos::parallel_for(k3 ? "budget3d_lmass" : "budget_lmass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
  KOKKOS_LAMBDA(const int t) {lmass(0,t) = lent(0,t);});
  pd->HaloForwardFlat(lmass, 1);
  const int is_ = bnd[0], ie_ = bnd[1], js_ = bnd[2], je_ = bnd[3], ks_ = bnd[4], ke_ = bnd[5];
  const auto ltb = lt;
  const auto trb = tr;
  Kokkos::parallel_for(k3 ? "budget3d_structure" : "budget_structure",
                       Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
  KOKKOS_LAMBDA(const int t) {
    lbeta(t) = beta;
    // only this rank's leaves are judged (the halo leaves keep beta; their owner decides)
    if (t >= nown) return;
    Real s = 0.0;
    if (!GD::Structure(ltb, trb, lmass, t, ncell, is_, ie_, js_, je_, ks_, ke_, rfloor, s)) return;
    lbeta(t) = Kokkos::fmax(Kokkos::fmin(beta, bkappa*s), 1.0e-12);
  });
}

// the pieces of a merge level by a 64-bit cell key, without a table over all cells
// of the level (the dense counting sort above needed one entry per level-L cell): the
// distinct keys in ascending order (ukey), the members of ukey(u) in list(start(u) ..
// start(u+1) - 1) in ascending particle order; lookups by binary search.  Memory and work grow
// with the listed pieces only.
struct KeyCSR {
  DvceArray1D<std::int64_t> ukey;
  DvceArray1D<int> start, list;
  int nu = 0;
  KOKKOS_INLINE_FUNCTION
  int Find(const std::int64_t key) const {   // index u of key, -1 if not listed
    int lo = 0, hi = nu;
    while (lo < hi) {
      const int mid = (lo + hi)/2;
      if (ukey(mid) < key) lo = mid + 1; else hi = mid;
    }
    return (lo < nu && ukey(lo) == key) ? lo : -1;
  }
  KOKKOS_INLINE_FUNCTION
  int First(const std::int64_t key) const {const int u = Find(key); return (u < 0) ? 0 : start(u);}
  KOKKOS_INLINE_FUNCTION
  int Last(const std::int64_t key) const {const int u = Find(key); return (u < 0) ? 0 : start(u + 1);}
};

// the device buffers of the sparse key tables, grown when needed
// and kept across passes.  Allocating and freeing them in every pass (cudaMalloc and cudaFree
// synchronize the device; thrust's sort allocated its own storage in every call) made the first
// sparse version slower than the dense tables.  Released at Kokkos::finalize.
// Slots: KeyCSR 0 pieces, 3 q-stage walk (1 and 2 held the level-L and lobed clouds of the tent
// pass, retired).  A slot is rebuilt only after the tables built in it are out of  // retired-compat
// use (kernels on the one stream run in order).
struct KeyScratch {
  static constexpr int kSlots = 4;  // retired-compat
  DvceArray1D<int> pos, uidx, list[kSlots], start[kSlots];
  DvceArray1D<std::int64_t> comp, ukey[kSlots], skey;   // skey: SortParticles
  DvceArray1D<char> sort_tmp;
  bool sort_busy = false;
};
KeyScratch *key_scratch = nullptr;
KeyScratch &Scratch() {
  if (key_scratch == nullptr) {
    key_scratch = new KeyScratch;
    Kokkos::push_finalize_hook([]() {delete key_scratch; key_scratch = nullptr;});
  }
  return *key_scratch;
}
// v with at least n entries (contents not kept)
template <typename T>
DvceArray1D<T> Grow(DvceArray1D<T> &v, const std::size_t n, const char *label) {
  if (v.extent(0) < n) {
    v = DvceArray1D<T>();
    v = DvceArray1D<T>(Kokkos::view_alloc(Kokkos::WithoutInitializing, std::string(label)), n + n/4 + 64);
  }
  return v;
}
#if defined(KOKKOS_ENABLE_CUDA)
// thrust's temporary storage of a sort from KeyScratch::sort_tmp (a second request while it is
// in use gets its own allocation)
struct SortTemp {
  typedef char value_type;
  char *allocate(const std::ptrdiff_t n) {
    KeyScratch &s = Scratch();
    if (s.sort_busy) return static_cast<char*>(Kokkos::kokkos_malloc<DevMemSpace>("key_sort_tmp2", n));
    s.sort_busy = true;
    return Grow(s.sort_tmp, static_cast<std::size_t>(n), "key_sort_tmp").data();
  }
  void deallocate(char *p, std::size_t) {
    KeyScratch &s = Scratch();
    if (s.sort_busy && p == s.sort_tmp.data()) s.sort_busy = false;
    else Kokkos::kokkos_free<DevMemSpace>(p);
  }
};
#endif
// the first n entries of v in ascending order (Kokkos::sort on the device is the same thrust sort)
void SortFirst(const DvceArray1D<std::int64_t> &v, const int n) {
  if (n < 2) return;
#if defined(KOKKOS_ENABLE_CUDA)
  SortTemp alloc;
  std::int64_t *d = v.data();
  thrust::sort(thrust::cuda::par(alloc).on(DevExeSpace().cuda_stream()), d, d + n);
#else
  Kokkos::sort(Kokkos::subview(v, std::make_pair(0, n)));
#endif
}

// key < 0: not listed; listed keys below kbound; the members of a key in particle order
void BuildKeyCSR(const DvceArray1D<std::int64_t> &key, const std::int64_t kbound, const int slot,
                 KeyCSR &c) {
  const int n = key.extent_int(0);
  KeyScratch &s = Scratch();
  auto pos = Grow(s.pos, n > 0 ? n : 1, "kcsr_pos");
  int nl = 0;
  Kokkos::parallel_scan("kcsr_pos", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (key(p) < 0) return;
    if (final) pos(p) = upd;
    ++upd;
  }, nl);
  c.list = Grow(s.list[slot], nl > 0 ? nl : 1, "kcsr_list");
  c.nu = 0;
  if (nl == 0) {
    c.ukey = Grow(s.ukey[slot], 1, "kcsr_ukey");
    c.start = Grow(s.start[slot], 1, "kcsr_start");
    return;
  }
  // the composite key n + p must fit in 64 bits, checked only for a
  // non-empty list (an empty deep level was stopped by the bound of its whole grid), and against
  // the largest listed key when the caller's bound is too large
  if (kbound >= std::numeric_limits<std::int64_t>::max()/n - 1) {
    std::int64_t kmax = 0;
    Kokkos::parallel_reduce("kcsr_max", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
    KOKKOS_LAMBDA(const int p, std::int64_t &m) {if (key(p) > m) m = key(p);},
    Kokkos::Max<std::int64_t>(kmax));
    if (kmax >= std::numeric_limits<std::int64_t>::max()/n - 1) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "merge cell keys too large for the composite sort (largest key " << kmax
                << ", " << n << " particles)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  // (key, particle) as one value key n + p, sorted
  const std::int64_t nn = n;
  auto comp = Grow(s.comp, nl, "kcsr_comp");
  Kokkos::parallel_for("kcsr_comp", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const int p) {if (key(p) >= 0) comp(pos(p)) = key(p)*nn + p;});
  SortFirst(comp, nl);
  auto uidx = Grow(s.uidx, nl, "kcsr_uidx");
  int nu = 0;
  Kokkos::parallel_scan("kcsr_distinct", Kokkos::RangePolicy<>(DevExeSpace(), 0, nl),
  KOKKOS_LAMBDA(const int k, int &upd, const bool final) {
    const bool first = (k == 0) || (comp(k)/nn != comp(k - 1)/nn);
    if (final) uidx(k) = first ? upd : -1;
    if (first) ++upd;
  }, nu);
  c.ukey = Grow(s.ukey[slot], nu, "kcsr_ukey");
  c.start = Grow(s.start[slot], nu + 1, "kcsr_start");
  auto uk = c.ukey; auto st = c.start; auto ls = c.list;
  const int nl_ = nl, nu_ = nu;
  Kokkos::parallel_for("kcsr_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, nl),
  KOKKOS_LAMBDA(const int k) {
    ls(k) = static_cast<int>(comp(k) % nn);
    if (uidx(k) >= 0) {uk(uidx(k)) = comp(k)/nn; st(uidx(k)) = k;}
    if (k == nl_ - 1) st(nu_) = nl_;
  });
  c.nu = nu;
}

}  // namespace

namespace {

// sign * (mass, momentum) of the live clouds 0..n-1 of (r, i) on the leaves, with the
// production cloud weights; one kernel for both traversals (BudgetDeposit and
// BudgetDeposit3D, the same operations)
template <class LTT>
void BudgetDepositT(const LTT &lt, const DvceArray2D<Real> &r, const DvceArray2D<int> &i,
                    const int n, const Real sign, DvceArray2D<Real> &f) {
  using A = LeafApi<LTT>;
  if (n <= 0) return;
  auto f_ = f;
  auto tr = lt;
  Kokkos::parallel_for((A::kDim == 3) ? "budget_deposit3d" : "budget_deposit",
                       Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const int p) {
    if (r(IPM,p) <= 0.0) return;
    const Real m = sign*r(IPM,p);
    const Real v1 = r(IPVX,p), v2 = r(IPVY,p), v3 = r(IPVZ,p);
    Real x[3];
    A::Load(r, p, x);
    A::ForEachWeight(tr, x, i(PLEV,p), tr.Anchor(i(PGID,p)), [&](const int t, const Real w) {
      Kokkos::atomic_add(&f_(0,t), m*w); Kokkos::atomic_add(&f_(1,t), m*w*v1);
      Kokkos::atomic_add(&f_(2,t), m*w*v2); Kokkos::atomic_add(&f_(3,t), m*w*v3);
    });
  });
}

// the leaf change (dm, d1..d3) in units of the budget: returns |dm| / (beta Mf) and sets
// rmom = |dp|_inf / (beta Mf vref), Mf = max(M, mfloor), vref = max(vmin, |P|_inf / M)
KOKKOS_INLINE_FUNCTION
Real BudgetRatio(const Real beta, const Real mfloor, const Real vmin, const Real M, const Real p1,
                 const Real p2, const Real p3, const Real dm, const Real d1, const Real d2,
                 const Real d3, Real &rmom) {
  const Real Mf = Kokkos::fmax(M, mfloor);
  Real vref = vmin;
  if (M > 0.0) {
    vref = Kokkos::fmax(vref, Kokkos::fmax(Kokkos::fabs(p1), Kokkos::fmax(Kokkos::fabs(p2),
                                                                          Kokkos::fabs(p3)))/M);
  }
  const Real dp = Kokkos::fmax(Kokkos::fabs(d1), Kokkos::fmax(Kokkos::fabs(d2), Kokkos::fabs(d3)));
  rmom = (vref > 0.0) ? dp/(beta*Mf*vref) : ((dp > 0.0) ? 1.0e30 : 0.0);
  return Kokkos::fabs(dm)/(beta*Mf);
}

}  // namespace


//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ReleaseFarLobesT
//! \brief (merge pass, step 1) the far-field release of lobed level-l clouds in merge cells of
//! level l - 1, written once for both traversals: every such cloud is replaced by its pieces (ReleasePieces along each
//! direction, GDim), in place in the particle order; new tags for all pieces but the first.
//! Returns the released clouds of this rank.

template <class LTT, class MMT>
int DustGasDrag::ReleaseFarLobesT(const LTT &lt, const MMT &mm, const int ncell) {
  using GD = GDim<LTT>;
  constexpr bool k3 = (LeafApi<LTT>::kDim == 3);
  particles::Particles *ppar = pmy_pack->ppart;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const int gids = pmy_pack->gids;
  const auto transfer = cloud_geometry;
  int nrelease = 0;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  DvceArray1D<int> np(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                        std::string(k3 ? "rel3d_np" : "rel_np")), npart);
  Kokkos::parallel_reduce(k3 ? "rel3d_count" : "rel_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &sum) {
    const int n = GD::ReleaseN(mm, lt, pr, pi, p);
    np(p) = n;
    if (n > 0) sum += 1;
  }, Kokkos::Sum<int>(nrelease));
  if (nrelease == 0) return 0;
  DvceArray1D<int> off(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                         std::string(k3 ? "rel3d_off" : "rel_off")), npart);
  DvceArray1D<int> toff(std::string(std::string(k3 ? "rel3d_toff" : "rel_toff")), npart);
  // (64-bit sums: the particle count after the release must stay within the int range)
  long long nnew64 = 0, ntags64 = 0;
  Kokkos::parallel_scan(k3 ? "rel3d_scan" : "rel_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, long long &upd, const bool final) {
    if (final) off(p) = static_cast<int>(upd);
    upd += (np(p) == 0) ? 1 : np(p);
  }, nnew64);
  Kokkos::parallel_scan(k3 ? "rel3d_tscan" : "rel_tscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, long long &upd, const bool final) {
    if (final) toff(p) = static_cast<int>(upd);
    upd += (np(p) > 0) ? np(p) - 1 : 0;
  }, ntags64);
  if (nnew64 > static_cast<long long>(INT_MAX)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "the release of far lobes on rank " << global_variable::my_rank << ": "
              << nnew64 << " clouds after it exceed the int range of the particle arrays"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nnew = static_cast<int>(nnew64), ntags = static_cast<int>(ntags64);
  DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, nnew);
  DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, nnew);
  // tags unique over the ranks, 64-bit
  const std::uint64_t tag0 = IssueTags(static_cast<std::uint64_t>(ntags));
  const std::uint64_t tstride = static_cast<std::uint64_t>(global_variable::nranks);
  Kokkos::parallel_for(k3 ? "rel3d_fill" : "rel_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    const int s0 = off(p);
    if (np(p) == 0) {
      for (int c=0; c<nrdata; ++c) prn(c,s0) = pr(c,p);
      for (int c=0; c<nidata; ++c) pin(c,s0) = pi(c,p);
      return;
    }
    GD::ReleaseFill(transfer, lt, pr, pi, prn, pin, p, s0, nrdata, nidata, gids, ncell, tag0,
                    tstride, toff);
  });
  ppar->prtcl_rdata = prn;
  ppar->prtcl_idata = pin;
  ppar->nprtcl_thispack = nnew;
  return nrelease;
}


//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::EnsureRootCloudMass
//! \brief The root cloud mass of every species, once (cloud_merge_mroot): the largest plain
//! level-0 cloud, else four times the largest plain level-1 cloud.  Used by the merge pass and
//! by the resampling caps, whichever runs first.

void DustGasDrag::EnsureRootCloudMass() {
  particles::Particles *ppar = pmy_pack->ppart;
  if (cloud_merge_mroot.empty()) {
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    int nsp = 1;
    Kokkos::parallel_reduce("merge_nsp", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &mx) {if (pi(PSP,p) + 1 > mx) mx = pi(PSP,p) + 1;},
    Kokkos::Max<int>(nsp));
#if MPI_PARALLEL_ENABLED
    // the same quantum on every rank (called collectively by RefineClouds)
    MPI_Allreduce(MPI_IN_PLACE, &nsp, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    cloud_merge_mroot.assign(nsp, 0.0);
    for (int sp=0; sp<nsp; ++sp) {
      Real m0 = 0.0, m1 = 0.0;
      Kokkos::parallel_reduce("merge_mroot", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, Real &a, Real &b) {
        const int code = pi(PLEV,p);
        if (pi(PSP,p) != sp || CloudGeometry::CloudLobe1(code) ||
            CloudGeometry::CloudLobe2(code) ||
            CloudGeometry::CloudLobe3(code)) return;   // (x3 lobe: 0 in 2-D)
        const int l = CloudGeometry::CloudLevel(code);
        if (l == 0 && pr(IPM,p) > a) a = pr(IPM,p);
        if (l == 1 && pr(IPM,p) > b) b = pr(IPM,p);
      }, Kokkos::Max<Real>(m0), Kokkos::Max<Real>(m1));
#if MPI_PARALLEL_ENABLED
      Real mm[2] = {m0, m1};
      MPI_Allreduce(MPI_IN_PLACE, mm, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      m0 = mm[0]; m1 = mm[1];
#endif
      // a level-1 cloud carries 1/4 of a root cloud in 2-D, 1/8 in 3-D (with
      // cloud_split_max_level the value is recorded before the first instant switch, while the
      // levels are still those of the split lattice: CloudInstantSwitch)
      cloud_merge_mroot[sp] = (m0 > 0.0) ? m0 : (cloud_dim == 3 ? 8.0 : 4.0)*m1;
    }
  }
}

namespace {
// the same decision on every rank (one rank: the value itself)
bool AnyRank(const bool v) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    int x = v ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    return x != 0;
  }
#endif
  return v;
}
long long SumRanks(const long long v) {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    long long x = v;
    MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return x;
  }
#endif
  return v;
}
// the per-species root cloud masses of the merge pass on the device, nsp entries
// (nsp >= the species of every particle; zero where a species has none).  Was a fixed Real[4]
// captured by the kernels, which limited the pass to 4 species.
DvceArray1D<Real> SpeciesRootMass(const std::vector<Real> &mroot, const int nsp) {
  DvceArray1D<Real> d("merge_mroot_species", nsp);
  auto h = Kokkos::create_mirror_view(d);
  for (int sp=0; sp<nsp; ++sp) {
    h(sp) = (sp < static_cast<int>(mroot.size())) ? mroot[sp] : 0.0;
  }
  Kokkos::deep_copy(d, h);
  return d;
}
// the bound of the cell keys (sp ncl + cell) << shift of nsp species, stopped if
// a key could leave the 64-bit range
std::int64_t SpeciesKeyBound(const int nsp, const std::int64_t ncl, const int shift) {
  const std::int64_t lim = std::numeric_limits<std::int64_t>::max() >> shift;
  if (nsp < 1 || ncl < 1 || ncl > lim/nsp) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "merge cell keys out of the 64-bit range (" << nsp << " species, " << ncl
              << " cells, shift " << shift << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  return (static_cast<std::int64_t>(nsp)*ncl) << shift;
}
// the pass budget's decision on the proposed G groups k0..k1-1, written once for both
// traversals (the same operations): ldelg
// the cumulated leaf change of the proposals still standing (new cloud - members); a leaf touched
// by them violates when the change so far (lcum) plus ldelg exceeds the budget; every group whose
// new cloud or members touch a violating leaf is rejected; repeat (at most maxev rounds, then the
// rest is rejected).  Then the members of accepted groups are used up, the new clouds of rejected
// groups dropped, and lcum advanced.  Several ranks: the owners judge their leaves, the same number of rounds everywhere.
struct GroupBudgetResult {double rg = 0.0, rp = 0.0, rm = 0.0; int nev = 0; bool capped = false;};

template <class LTT>
GroupBudgetResult GroupBudgetRoundsT(DustGasDrag *pd, const LTT &lt, const int npart, const int k0,
    const int k1, const DvceArray1D<std::int64_t> &kq, const DvceArray1D<int> &gid,
    const DvceArray1D<int> &gknew, const DvceArray1D<int> &krej, const DvceArray2D<Real> &pr,
    const DvceArray2D<int> &pi, const DvceArray2D<Real> &nr, const DvceArray2D<int> &ni,
    DvceArray2D<Real> &ldelg, DvceArray1D<int> &lviolg, const DvceArray2D<Real> &lent,
    const DvceArray2D<Real> &lcum, const DvceArray1D<Real> &lbeta, const Real bkappa,
    const Real beta, const int maxev, const int nown, const int nbt, const int ncell,
    const Real gb_floor, const Real gb_vmin) {
  using A = LeafApi<LTT>;
  constexpr bool k3 = (A::kDim == 3);
  GroupBudgetResult res;
  int nev = 0;
  for (;;) {
    ++nev;
    Kokkos::deep_copy(ldelg, 0.0);
    Kokkos::parallel_for(k3 ? "q3b_members" : "qb_members", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (kq(p) < 0 || gid(p) < 0) return;
      const int k = gknew(gid(p));
      if (k < 0 || krej(k)) return;
      const Real m = pr(IPM,p);
      const Real v1 = pr(IPVX,p), v2 = pr(IPVY,p), v3 = pr(IPVZ,p);
      Real x[3];
      A::Load(pr, p, x);
      A::ForEachWeight(lt, x, pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int t, const Real w) {
        Kokkos::atomic_add(&ldelg(0,t), -m*w); Kokkos::atomic_add(&ldelg(1,t), -m*w*v1);
        Kokkos::atomic_add(&ldelg(2,t), -m*w*v2); Kokkos::atomic_add(&ldelg(3,t), -m*w*v3);
      });
    });
    Kokkos::parallel_for(k3 ? "q3b_new" : "qb_new", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
    KOKKOS_LAMBDA(const int k) {
      if (krej(k)) return;
      const Real m = nr(IPM,k);
      const Real v1 = nr(IPVX,k), v2 = nr(IPVY,k), v3 = nr(IPVZ,k);
      Real x[3];
      A::Load(nr, k, x);
      A::ForEachWeight(lt, x, ni(PLEV,k), lt.Anchor(ni(PGID,k)), [&](const int t, const Real w) {
        Kokkos::atomic_add(&ldelg(0,t), m*w); Kokkos::atomic_add(&ldelg(1,t), m*w*v1);
        Kokkos::atomic_add(&ldelg(2,t), m*w*v2); Kokkos::atomic_add(&ldelg(3,t), m*w*v3);
      });
    });
    pd->HaloReverseAddFlat(ldelg, 4);   // the owners judge their leaves
    int nviol = 0;
    Kokkos::parallel_reduce(k3 ? "q3b_viol" : "qb_viol", Kokkos::RangePolicy<>(DevExeSpace(), 0, nown),
    KOKKOS_LAMBDA(const int t, int &sum) {
      int bad = 0;
      if (ldelg(0,t) != 0.0 || ldelg(1,t) != 0.0 || ldelg(2,t) != 0.0 || ldelg(3,t) != 0.0) {
        const int tb_ = t/ncell;
        const Real vol = A::Measure(lt, tb_);
        Real rp = 0.0;
        const Real rm = BudgetRatio((bkappa > 0.0) ? lbeta(t) : beta, gb_floor*vol, gb_vmin,
            lent(0,t), lent(1,t), lent(2,t), lent(3,t),
            lcum(0,t) + ldelg(0,t), lcum(1,t) + ldelg(1,t), lcum(2,t) + ldelg(2,t),
            lcum(3,t) + ldelg(3,t), rp);
        bad = (rm > 1.0 || rp > 1.0) ? 1 : 0;
      }
      lviolg(t) = bad;
      sum += bad;
    }, nviol);
    pd->HaloForwardFlags(lviolg);                       // the owners' verdicts on the halo
    // the same number of rounds everywhere (the sum of all ranks is 64-bit: not narrowed)
    if (SumRanks(nviol) == 0) break;
    if (nev >= maxev) {   // no convergence: the rest of stage G is rejected
      Kokkos::parallel_for(k3 ? "q3b_rej_all" : "qb_rej_all", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
      KOKKOS_LAMBDA(const int k) {krej(k) = 1;});
      Kokkos::deep_copy(ldelg, 0.0);
      res.capped = true;
      break;
    }
    Kokkos::parallel_for(k3 ? "q3b_rej_new" : "qb_rej_new", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
    KOKKOS_LAMBDA(const int k) {
      if (krej(k)) return;
      int bad = 0;
      Real x[3];
      A::Load(nr, k, x);
      A::ForEachWeight(lt, x, ni(PLEV,k), lt.Anchor(ni(PGID,k)), [&](const int t, const Real w) {
        if (w != 0.0 && lviolg(t)) bad = 1;
      });
      if (bad) krej(k) = 1;
    });
    Kokkos::parallel_for(k3 ? "q3b_rej_members" : "qb_rej_members",
                         Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (kq(p) < 0 || gid(p) < 0) return;
      const int k = gknew(gid(p));
      if (k < 0 || krej(k)) return;
      int bad = 0;
      Real x[3];
      A::Load(pr, p, x);
      A::ForEachWeight(lt, x, pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int t, const Real w) {
        if (w != 0.0 && lviolg(t)) bad = 1;
      });
      if (bad) Kokkos::atomic_exchange(&krej(k), 1);
    });
  }
  // apply: the members of accepted groups are used up; rejected groups keep their members
  // exactly as they are and their new cloud is dropped
  double rp_ = 0.0, rm_ = 0.0, rg_ = 0.0;
  Kokkos::parallel_reduce(k3 ? "q3b_apply" : "qb_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, double &n, double &m) {
    if (kq(p) < 0 || gid(p) < 0) return;
    const int k = gknew(gid(p));
    if (k < 0) return;
    if (krej(k)) {n += 1.0; m += pr(IPM,p); return;}
    pr(IPM,p) = 0.0;
  }, rp_, rm_);
  Kokkos::parallel_reduce(k3 ? "q3b_dead" : "qb_dead", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
  KOKKOS_LAMBDA(const int k, double &n) {
    if (krej(k)) {nr(IPM,k) = 0.0; n += 1.0;}
  }, rg_);
  Kokkos::parallel_for(k3 ? "budget3d_advance" : "budget_advance",
                       Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
  KOKKOS_LAMBDA(const int t) {
    for (int c=0; c<4; ++c) lcum(c,t) += ldelg(c,t);
  });
  res.rg = rg_; res.rp = rp_; res.rm = rm_; res.nev = nev;
  return res;
}

// the compaction after a level of the quantum pass, both traversals: the pieces with
// mass are kept in order, the new clouds with mass appended (budget-rejected ones have mass 0).
// Returns false (nothing changed) when every piece is kept and no new cloud is.
template <bool k3>
bool CompactMergedT(particles::Particles *ppar, const DvceArray2D<Real> &nr,
                    const DvceArray2D<int> &ni, const int npart, const int ncommit) {
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  DvceArray1D<int> koff(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                          std::string(k3 ? "q3_koff" : "q_koff")),
                        npart);
  int nkeep = 0;
  Kokkos::parallel_scan(k3 ? "q3_kscan" : "q_kscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) koff(p) = upd;
    upd += (pr(IPM,p) > 0.0) ? 1 : 0;
  }, nkeep);
  DvceArray1D<int> noff(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                          std::string(k3 ? "q3_noff" : "q_noff")),
                        ncommit > 0 ? ncommit : 1);
  int nlive = 0;   // new clouds that are kept (budget-rejected ones have mass 0)
  if (ncommit > 0) {
    Kokkos::parallel_scan(k3 ? "q3_nscan" : "q_nscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
    KOKKOS_LAMBDA(const int k, int &upd, const bool final) {
      if (final) noff(k) = upd;
      upd += (nr(IPM,k) > 0.0) ? 1 : 0;
    }, nlive);
  }
  if (nkeep == npart && nlive == 0) return false;
  const int ntot = nkeep + nlive;
  DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, ntot);
  DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, ntot);
  Kokkos::parallel_for(k3 ? "q3_keep" : "q_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (pr(IPM,p) <= 0.0) return;
    const int s = koff(p);
    for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
    for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
  });
  Kokkos::parallel_for(k3 ? "q3_append" : "q_append", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
  KOKKOS_LAMBDA(const int k) {
    if (nr(IPM,k) <= 0.0) return;
    const int s = nkeep + noff(k);
    for (int c=0; c<nrdata; ++c) prn(c,s) = nr(c,k);
    for (int c=0; c<nidata; ++c) pin(c,s) = ni(c,k);
  });
  ppar->prtcl_rdata = prn;
  ppar->prtcl_idata = pin;
  ppar->nprtcl_thispack = ntot;
  return true;
}
}  // namespace

//----------------------------------------------------------------------------------------
// the quantum merge pass written once (DustGasDrag::MergeQuantumT, below): one
// driver for both traversals (pass budget, levels, cell keys, the walk's cell list, the group
// budget, counters, compaction).  GKern<LT> holds each dimension's own straight-line kernels where
// the rule's arithmetic lives: stage X (9 or 27 members; 3-D: no family across a shear-periodic
// x1 face or a physical x3 face), the walk's keys (Morton order of 4^d sub-cells), the walk
// (bounding boxes per direction; 3-D: cloud_merge_alpha_leaf), the accumulation and the new
// clouds' state (the 2-D X budget, X protection and position order were retired).  // retired-compat
// The counters of a level (qc) have one layout: 0 X families, 1 their mass, 2 G groups, 3 their
// pieces, 4 their mass, 5 deferred pieces, 6 residual pieces (the 2-D pass's X energy / variance
// and dK / dS sums were never reported anywhere and are gone).

namespace {

template <class LTT>
struct GLevel {
  using MM = typename GDim<LTT>::MM;
  LTT lt;
  MM mm;
  DvceArray2D<Real> pr;
  DvceArray2D<int> pi;
  int npart = 0, nmax = 1, nrdata = 0, nidata = 0, gids = 0, ncellb = 0, lev = 0, L = 0;
  Real H1 = 0.0, H2 = 0.0, H3 = 0.0, H = 0.0, ell1 = 0.0, ell2 = 0.0, ell3 = 0.0;
  int m1 = 1, m2 = 1, m3 = 1;
  std::int64_t ncl = 1;
  Real omega = 0.0, alpha = 0.0, fmin = 0.0;
  DvceArray1D<std::int64_t> kp;
  KeyCSR spc;
  DvceArray2D<Real> nr;
  DvceArray2D<int> ni;
  DvceArray1D<int> nnew;
  DvceArray1D<double> qc;
  DvceArray2D<double> acc;   // m, m dx, m dy (, m dz) of a G cloud about its reference point
  DvceArray2D<Real> aref;    // the reference point of a G cloud
};

template <class LTT> struct GKern;

//------------------------------- 2-D (LeafTraversal) -------------------------------------
template <>
struct GKern<LeafTraversal> {
  using LT = LeafTraversal;
  static constexpr int kNsub = 16;      // sub-cells of a merge cell (4 x 4), walk keys << 4
  static constexpr int kShift = 4;
  static constexpr double kFam = 9.0;   // members of an X family
  static int Nlev(const DustGasDrag *pd, const LT &) {return pd->cloud_geometry.cloud_nlev;}
  static int Ncell(const DustGasDrag *pd, const LT &) {
    return pd->qdep.extent_int(4)*pd->qdep.extent_int(3);
  }
  static MergeMask Mask(const DustGasDrag *pd, const Mesh *, const LT &lt, const int nlev) {
    const auto &transfer = pd->cloud_geometry;
    MergeMask mm;
    mm.lt = lt;
    mm.B = pd->cloud_merge_buffer;
    mm.x1min = transfer.x1min; mm.x2min = transfer.x2min;
    mm.x1max = transfer.x1max; mm.x2max = transfer.x2max;
    for (int L=0; L<nlev; ++L) {
      mm.dx1[L] = transfer.root_dx1/static_cast<Real>(1 << L);
      mm.dx2[L] = transfer.root_dx2/static_cast<Real>(1 << L);
      mm.n1[L] = transfer.ncoarse1 << L;
      mm.n2[L] = transfer.ncoarse2 << L;
    }
    return mm;
  }
  // the root cell area (the budgets' mass floor per root cell)
  static Real RootMeasure(const DustGasDrag *pd, const LT &) {
    return pd->cloud_geometry.root_dx1*pd->cloud_geometry.root_dx2;
  }
  static void Bounds(const Mesh *pmesh, const LT &, int bnd[6]) {
    bnd[0] = pmesh->mb_indcs.is; bnd[1] = pmesh->mb_indcs.ie;
    bnd[2] = pmesh->mb_indcs.js; bnd[3] = pmesh->mb_indcs.je; bnd[4] = 0; bnd[5] = 0;
  }
  static void Level(const DustGasDrag *pd, GLevel<LT> &g) {
    const auto &transfer = pd->cloud_geometry;
    g.H1 = g.mm.dx1[g.L]; g.H2 = g.mm.dx2[g.L]; g.H = Kokkos::min(g.H1, g.H2);
    g.ell1 = transfer.CloudHalfWidth1(g.lev); g.ell2 = transfer.CloudHalfWidth2(g.lev);
    g.m1 = g.mm.n1[g.L]; g.m2 = g.mm.n2[g.L];
    g.ncl = static_cast<std::int64_t>(g.m1)*g.m2;
  }
  static Real Qden(const int L) {return static_cast<Real>(1 << (2*L));}
  KOKKOS_INLINE_FUNCTION static bool Lobed(const int code) {
    return CloudGeometry::CloudLobe1(code) || CloudGeometry::CloudLobe2(code);
  }
  template <class PR>
  KOKKOS_INLINE_FUNCTION
  static std::int64_t Cell(const MergeMask &mm, const PR &pr, const int p, const int L) {
    return mm.Cell(pr(IPX,p), pr(IPY,p), L);
  }

  // stage X: complete P1 families (9 members) back into their level-L parent
  static void StageX(const DustGasDrag *pd, const GLevel<LT> &g, const Real dtol) {
    const auto transfer = pd->cloud_geometry;
    const auto lt = g.lt;
    const auto mm = g.mm;
    auto pr = g.pr;
    auto pi = g.pi;
    const int npart = g.npart, nmax = g.nmax, nrdata = g.nrdata, nidata = g.nidata;
    const int gids = g.gids, ncell12 = g.ncellb, L = g.L;
    const Real H = g.H, ell1 = g.ell1, ell2 = g.ell2;
    const std::int64_t ncl = g.ncl;
    const Real omega = g.omega, alpha = g.alpha;
    auto kp = g.kp;
    const KeyCSR spc = g.spc;
    auto nr = g.nr;
    auto ni = g.ni;
    auto nnew = g.nnew;
    auto qc = g.qc;
    DvceArray1D<int> owner("q_owner", npart);
    Kokkos::deep_copy(owner, -1);
    Kokkos::parallel_for("q_stencil", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int c) {
      if (kp(c) < 0 || pr(IPM,c) <= 0.0) return;
      const Real xc0 = pr(IPX,c), yc0 = pr(IPY,c), mc = pr(IPM,c), ts = pr(IPTS,c);
      if (!mm.In(xc0, yc0, L, lt.Anchor(pi(PGID,c)))) return;
      const int sp = pi(PSP,c);
      const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
      const Real tolv = alpha*H/tau;
      int mem[9]; int nm = 0; mem[nm++] = c;
      Real vlo[3] = {pr(IPVX,c), pr(IPVY,c), pr(IPVZ,c)};
      Real vhi[3] = {vlo[0], vlo[1], vlo[2]};
      for (int b=-1; b<=1; ++b) {
        for (int a=-1; a<=1; ++a) {
          if (a == 0 && b == 0) continue;
          const Real w = mc/static_cast<Real>((a != 0 ? 2 : 1)*(b != 0 ? 2 : 1));
          const Real tx = xc0 + a*ell1, ty = yc0 + b*ell2;
          const std::int64_t key = sp*ncl + mm.Cell(tx, ty, L);
          int best = -1; Real bd = 1.0e30;
          for (int k=spc.First(key), ke=spc.Last(key); k<ke; ++k) {
            const int q = spc.list(k);
            if (q == c || pr(IPM,q) <= 0.0 || owner(q) >= 0) continue;
            if (Kokkos::fabs(pr(IPM,q) - w) > 1.0e-6*w || pr(IPTS,q) != ts) continue;
            const Real dx = Kokkos::fabs(transfer.MinImage1(pr(IPX,q) - tx));
            const Real dy = Kokkos::fabs(transfer.MinImage2(pr(IPY,q) - ty));
            if (dx > dtol*ell1 || dy > dtol*ell2) continue;
            if (!mm.In(pr(IPX,q), pr(IPY,q), L, lt.Anchor(pi(PGID,q)))) continue;
            const Real d = Kokkos::fmax(dx/ell1, dy/ell2);
            if (d < bd) {bd = d; best = q;}
          }
          if (best < 0) return;
          mem[nm++] = best;
          const Real v[3] = {pr(IPVX,best), pr(IPVY,best), pr(IPVZ,best)};
          for (int d=0; d<3; ++d) {
            vlo[d] = Kokkos::fmin(vlo[d], v[d]); vhi[d] = Kokkos::fmax(vhi[d], v[d]);
            if (vhi[d] - vlo[d] > tolv) return;
          }
        }
      }
      for (int k=0; k<9; ++k) {   // claim; roll back on a conflict (retried next pass)
        if (Kokkos::atomic_compare_exchange(&owner(mem[k]), -1, c) != -1) {
          for (int j=0; j<k; ++j) Kokkos::atomic_exchange(&owner(mem[j]), -1);
          return;
        }
      }
      const int knew = Kokkos::atomic_fetch_add(&nnew(0), 1);
      double M = 0.0, sx = 0.0, sy = 0.0;
      for (int q=0; q<nrdata; ++q) nr(q,knew) = 0.0;
      for (int k=0; k<9; ++k) {
        const int t = mem[k]; const Real m = pr(IPM,t);
        const Real dx = transfer.MinImage1(pr(IPX,t) - xc0), dy = transfer.MinImage2(pr(IPY,t) - yc0);
        M += m; sx += m*dx; sy += m*dy;
        for (int q=0; q<nrdata; ++q) nr(q,knew) += m*pr(q,t);
      }
      for (int q=0; q<nrdata; ++q) nr(q,knew) /= M;
      nr(IPTS,knew) = ts;   // the family's common ts
      const Real px = CloudGeometry::WrapPosition(xc0 + sx/M, transfer.x1min, transfer.x1max);
      const Real py = CloudGeometry::WrapPosition(yc0 + sy/M, transfer.x2min, transfer.x2max);
      nr(IPM,knew) = M;
      nr(IPX,knew) = px; nr(IPX1,knew) = px; nr(IPY,knew) = py; nr(IPY1,knew) = py;
      for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,c);
      ni(PLEV,knew) = L;
      ni(PGID,knew) = gids + NewCloudBlock(lt, px, py, lt.Anchor(ni(PGID,knew)), ncell12);
      for (int k=0; k<9; ++k) pr(IPM,mem[k]) = 0.0;
      Kokkos::atomic_add(&qc(0), 1.0);
      Kokkos::atomic_add(&qc(1), M);
    });
  }

  // the walk's keys: plain level-lev pieces in level-L merge cells of the deep zone,
  // ((sp ncl + cell) << 4) + the Morton order of the 4 x 4 sub-cells
  static void GKeys(const DustGasDrag *pd, const GLevel<LT> &g, const int glag,
                    const DvceArray1D<std::int64_t> &kq) {
    const auto transfer = pd->cloud_geometry;
    const auto lt = g.lt;
    const auto mm = g.mm;
    auto pr = g.pr;
    auto pi = g.pi;
    const int npart = g.npart, L = g.L;
    const Real cs1 = g.H1, cs2 = g.H2;
    const int m1 = g.m1, m2 = g.m2;
    const std::int64_t nc = g.ncl;
    auto kp = g.kp;
    Kokkos::parallel_for("q_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      kq(p) = -1;
      if (kp(p) < 0 || pr(IPM,p) <= 0.0) return;          // plain level-lev pieces only
      const Real x = CloudGeometry::WrapPosition(pr(IPX,p), transfer.x1min, transfer.x1max);
      const Real y = CloudGeometry::WrapPosition(pr(IPY,p), transfer.x2min, transfer.x2max);
      const Real u = (x - transfer.x1min)/cs1, v = (y - transfer.x2min)/cs2;
      const int I = CloudGeometry::WrapIndex(static_cast<int>(Kokkos::floor(u)), m1);
      const int J = CloudGeometry::WrapIndex(static_cast<int>(Kokkos::floor(v)), m2);
      if (!mm.In(x, y, L, lt.Anchor(pi(PGID,p)))) return;
      for (int dj=-glag; dj<=glag; ++dj) {   // deep zone only: the outer band is stage X's
        for (int di=-glag; di<=glag; ++di) {
          const Real xn = transfer.x1min + (CloudGeometry::WrapIndex(I + di, m1) + 0.5)*cs1;
          const Real yn = transfer.x2min + (CloudGeometry::WrapIndex(J + dj, m2) + 0.5)*cs2;
          if (!mm.In(xn, yn, L, lt.Anchor(pi(PGID,p)))) return;
        }
      }
      const int si = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(u - Kokkos::floor(u)))));
      const int sj = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(v - Kokkos::floor(v)))));
      const int sub = ((sj >> 1) << 3) | ((si >> 1) << 2) | ((sj & 1) << 1) | (si & 1);   // Morton
      kq(p) = ((pi(PSP,p)*nc + static_cast<std::int64_t>(J)*m1 + I) << 4) + sub;
    });
  }
  // the walk's cell list (the pieces of a sub-cell in particle order)
  static void WalkCSR(const DustGasDrag *, const GLevel<LT> &g, const int nsp,
                      const DvceArray1D<std::int64_t> &kq, KeyCSR &qsc) {
    BuildKeyCSR(kq, SpeciesKeyBound(nsp, g.ncl, 4), 3, qsc);  // retired-compat
  }

  // the cell walk: the pieces of a level-L merge cell, sub-cell by sub-cell in the rotated Morton
  // order, into level-L clouds of the quantum (position and velocity bounding boxes, K open groups)
  static void Walk(const DustGasDrag *pd, const GLevel<LT> &g, const KeyCSR &qsc,
                   const DvceArray1D<int> &cfirst, const int ncell, const DvceArray1D<Real> &mrootsp,
                   const Real qden, const int K, const int rot, const int rev, const bool,
                   const DvceArray1D<int> &gid, const DvceArray1D<int> &gknew,
                   const DvceArray1D<int> &ngroup, const Real smax) {
    const auto transfer = pd->cloud_geometry;
    auto pr = g.pr;
    auto pi = g.pi;
    const int nidata = g.nidata, L = g.L;
    const Real cs1 = g.H1, cs2 = g.H2;
    const int m1 = g.m1;
    const std::int64_t nc = g.ncl;
    const Real omega = g.omega, alpha = g.alpha, fmin = g.fmin;
    auto nr = g.nr;
    auto ni = g.ni;
    auto nnew = g.nnew;
    auto qc = g.qc;
    auto aref = g.aref;
    const Real smax1 = smax*cs1;
    const Real smax2 = smax*cs2;
    Kokkos::parallel_for("q_walk", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(const int ic) {
      const int u0 = cfirst(ic), u1 = cfirst(ic + 1);
      const std::int64_t kcell = qsc.ukey(u0) >> 4;
      const int sp = static_cast<int>(kcell/nc);
      const std::int64_t cell = kcell % nc;
      const int I = static_cast<int>(cell % m1), J = static_cast<int>(cell/m1);
      const Real mq = mrootsp(sp)/qden;
      const Real xcc = transfer.x1min + (I + 0.5)*cs1, ycc = transfer.x2min + (J + 0.5)*cs2;
      Real gM[8], vlo[8][3], vhi[8][3];
      Real xlo[8], xhi[8], ylo[8], yhi[8], gts[8];
      int gg[8], gfirst[8], gn[8], nopen = 0;
      auto commit = [&](const int j) {
        const int knew = Kokkos::atomic_fetch_add(&nnew(0), 1);
        gknew(gg[j]) = knew;
        for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,gfirst[j]);
        ni(PLEV,knew) = L;
        nr(IPTS,knew) = gts[j];   // the group's common ts
        aref(0,knew) = xcc; aref(1,knew) = ycc;
        Kokkos::atomic_add(&qc(2), 1.0);
        Kokkos::atomic_add(&qc(3), static_cast<double>(gn[j]));
        Kokkos::atomic_add(&qc(4), static_cast<double>(gM[j]));
      };
      for (int s=0; s<16; ++s) {
        const int sub = rev ? (rot + 16 - s) % 16 : (rot + s) % 16;
        int kb = 0, ke = 0;   // the members of sub-cell sub
        for (int u=u0; u<u1; ++u) {
          if ((qsc.ukey(u) & 15) == sub) {kb = qsc.start(u); ke = qsc.start(u + 1); break;}
        }
        for (int k=kb; k<ke; ++k) {
          const int p = qsc.list(k);
          const Real m = pr(IPM,p);
          if (m <= 0.0) continue;
          const Real dx = transfer.MinImage1(pr(IPX,p) - xcc), dy = transfer.MinImage2(pr(IPY,p) - ycc);
          const Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
          const Real ts = pr(IPTS,p);
          const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
          const Real tolv = alpha*Kokkos::fmin(cs1, cs2)/tau;
          int j = -1;
          for (int o=0; o<nopen && j<0; ++o) {
            if (gts[o] != ts) continue;
            bool ok = (Kokkos::fmax(xhi[o], dx) - Kokkos::fmin(xlo[o], dx) <= smax1) &&
                      (Kokkos::fmax(yhi[o], dy) - Kokkos::fmin(ylo[o], dy) <= smax2);
            for (int d=0; d<3 && ok; ++d) ok = Kokkos::fmax(vhi[o][d], v[d]) - Kokkos::fmin(vlo[o][d], v[d]) <= tolv;
            if (ok) j = o;
          }
          if (j < 0) {
            // deferred (K <= 8 by the input check; 8 the size of the open-group arrays)
            if (nopen == K || nopen == 8) {Kokkos::atomic_add(&qc(5), 1.0); continue;}
            j = nopen++;
            gg[j] = Kokkos::atomic_fetch_add(&ngroup(0), 1);
            gM[j] = 0.0; gn[j] = 0;
            xlo[j] = xhi[j] = dx; ylo[j] = yhi[j] = dy; gts[j] = ts; gfirst[j] = p;
            for (int d=0; d<3; ++d) {vlo[j][d] = vhi[j][d] = v[d];}
          }
          gid(p) = gg[j];
          gM[j] += m; ++gn[j];
          xlo[j] = Kokkos::fmin(xlo[j], dx); xhi[j] = Kokkos::fmax(xhi[j], dx);
          ylo[j] = Kokkos::fmin(ylo[j], dy); yhi[j] = Kokkos::fmax(yhi[j], dy);
          for (int d=0; d<3; ++d) {vlo[j][d] = Kokkos::fmin(vlo[j][d], v[d]); vhi[j][d] = Kokkos::fmax(vhi[j][d], v[d]);}
          if (gM[j] >= mq*(1.0 - 1.0e-12)) {
            commit(j);
            --nopen;   // move the last open group into slot j
            if (j != nopen) {
              gM[j] = gM[nopen]; xlo[j] = xlo[nopen]; xhi[j] = xhi[nopen]; ylo[j] = ylo[nopen];
              yhi[j] = yhi[nopen]; gts[j] = gts[nopen]; gg[j] = gg[nopen]; gfirst[j] = gfirst[nopen];
              gn[j] = gn[nopen];
              for (int d=0; d<3; ++d) {vlo[j][d] = vlo[nopen][d]; vhi[j][d] = vhi[nopen][d];}
            }
          }
        }
      }
      for (int o=0; o<nopen; ++o) {
        if (gM[o] >= fmin*mq) commit(o);
        if (gknew(gg[o]) < 0) Kokkos::atomic_add(&qc(6), static_cast<double>(gn[o]));
      }
    });
  }
  // the members of committed groups into their new clouds (the pieces are used up)
  static void Accumulate(const DustGasDrag *pd, const GLevel<LT> &g,
                         const DvceArray1D<std::int64_t> &kq, const DvceArray1D<int> &gid,
                         const DvceArray1D<int> &gknew, const bool pbud) {
    const auto transfer = pd->cloud_geometry;
    auto pr = g.pr;
    const int npart = g.npart, nrdata = g.nrdata;
    auto nr = g.nr;
    auto acc = g.acc;
    auto aref = g.aref;
    Kokkos::parallel_for("q_accumulate", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (kq(p) < 0) return;
      const int g_ = gid(p);
      if (g_ < 0) return;
      const int k = gknew(g_);
      if (k < 0) return;
      const Real m = pr(IPM,p);
      const Real dx = transfer.MinImage1(pr(IPX,p) - aref(0,k)), dy = transfer.MinImage2(pr(IPY,p) - aref(1,k));
      Kokkos::atomic_add(&acc(0,k), static_cast<double>(m));
      Kokkos::atomic_add(&acc(1,k), static_cast<double>(m*dx));
      Kokkos::atomic_add(&acc(2,k), static_cast<double>(m*dy));
      for (int q=0; q<nrdata; ++q) {
        if (q == IPM || q == IPX || q == IPY || q == IPX1 || q == IPY1) continue;
        if (q == IPTS) continue;   // set at the commit
        Kokkos::atomic_add(&nr(q,k), m*pr(q,p));
      }
      if (!pbud) pr(IPM,p) = 0.0;
    });
  }
  // the state of the new G clouds k0..k1-1: means and centroid
  static void Finalize(const DustGasDrag *pd, const GLevel<LT> &g, const int k0, const int k1) {
    const auto transfer = pd->cloud_geometry;
    const auto lt = g.lt;
    const int nrdata = g.nrdata, gids = g.gids, ncell12 = g.ncellb;
    auto nr = g.nr;
    auto ni = g.ni;
    auto acc = g.acc;
    auto aref = g.aref;
    Kokkos::parallel_for("q_finalize_state", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
    KOKKOS_LAMBDA(const int k) {
      const double M = acc(0,k);
      for (int q=0; q<nrdata; ++q) {
        if (q == IPM || q == IPX || q == IPY || q == IPX1 || q == IPY1) continue;
        if (q == IPTS) continue;   // set at the commit
        nr(q,k) /= M;
      }
      const Real px = CloudGeometry::WrapPosition(aref(0,k) + acc(1,k)/M, transfer.x1min, transfer.x1max);
      const Real py = CloudGeometry::WrapPosition(aref(1,k) + acc(2,k)/M, transfer.x2min, transfer.x2max);
      nr(IPM,k) = M;
      nr(IPX,k) = px; nr(IPX1,k) = px; nr(IPY,k) = py; nr(IPY1,k) = py;
      ni(PGID,k) = gids + NewCloudBlock(lt, px, py, lt.Anchor(ni(PGID,k)), ncell12);
    });
  }
};

//------------------------------- 3-D (LeafTraversal3D) -----------------------------------
template <>
struct GKern<LeafTraversal3D> {
  using LT = LeafTraversal3D;
  using T = CloudGeometry;
  static constexpr int kNsub = 64;      // sub-cells of a merge cell (4 x 4 x 4), walk keys << 6
  static constexpr int kShift = 6;
  static constexpr double kFam = 27.0;  // members of an X family
  static int Nlev(const DustGasDrag *, const LT &lt) {return lt.nlev;}
  static int Ncell(const DustGasDrag *, const LT &lt) {return lt.ncells1*lt.ncells2*lt.ncells3;}
  static MergeMask3D Mask(const DustGasDrag *pd, const Mesh *pmesh, const LT &lt, const int nlev) {
    MergeMask3D mm;
    mm.lt = lt;
    mm.B = pd->cloud_merge_buffer;
    mm.x1min = lt.x1min; mm.x2min = lt.x2min; mm.x3min = lt.x3min;
    mm.x1max = lt.x1max; mm.x2max = lt.x2max; mm.x3max = lt.x3max;
    const int nc1 = pmesh->mesh_indcs.nx1, nc2 = pmesh->mesh_indcs.nx2, nc3 = pmesh->mesh_indcs.nx3;
    for (int L=0; L<nlev; ++L) {
      mm.dx1[L] = lt.root_dx1/static_cast<Real>(1 << L);
      mm.dx2[L] = lt.root_dx2/static_cast<Real>(1 << L);
      mm.dx3[L] = lt.root_dx3/static_cast<Real>(1 << L);
      mm.n1[L] = nc1 << L; mm.n2[L] = nc2 << L; mm.n3[L] = nc3 << L;
    }
    return mm;
  }
  static Real RootMeasure(const DustGasDrag *, const LT &lt) {
    return lt.root_dx1*lt.root_dx2*lt.root_dx3;
  }
  static void Bounds(const Mesh *, const LT &lt, int bnd[6]) {
    bnd[0] = lt.is; bnd[1] = lt.is + lt.nx1 - 1; bnd[2] = lt.js; bnd[3] = lt.js + lt.nx2 - 1;
    bnd[4] = lt.ks; bnd[5] = lt.ks + lt.nx3 - 1;
  }
  static void Level(const DustGasDrag *, GLevel<LT> &g) {
    const int L = g.L;
    g.H1 = g.mm.dx1[L]; g.H2 = g.mm.dx2[L]; g.H3 = g.mm.dx3[L];
    g.H = Kokkos::min(g.H1, Kokkos::min(g.H2, g.H3));
    g.ell1 = g.lt.CloudHalfWidth1(g.lev); g.ell2 = g.lt.CloudHalfWidth2(g.lev);
    g.ell3 = g.lt.CloudHalfWidth3(g.lev);
    g.m1 = g.mm.n1[L]; g.m2 = g.mm.n2[L]; g.m3 = g.mm.n3[L];
    g.ncl = static_cast<std::int64_t>(g.m1)*g.m2*g.m3;
  }
  static Real Qden(const int L) {return static_cast<Real>(std::int64_t(1) << (3*L));}
  KOKKOS_INLINE_FUNCTION static bool Lobed(const int code) {
    return T::CloudLobe1(code) || T::CloudLobe2(code) || T::CloudLobe3(code);
  }
  template <class PR>
  KOKKOS_INLINE_FUNCTION
  static std::int64_t Cell(const MergeMask3D &mm, const PR &pr, const int p, const int L) {
    return mm.Cell(pr(IPX,p), pr(IPY,p), pr(IPZ,p), L);
  }

  // stage X: complete P1 families (27 members) back into their level-L parent; none across a
  // shear-periodic x1 face (its members were wrapped with the shear offset) or a physical x3 face
  // (its members were mirrored).
  static void StageX(const DustGasDrag *, const GLevel<LT> &g, const Real dtol) {
    const auto lt = g.lt;
    const auto mm = g.mm;
    auto pr = g.pr;
    auto pi = g.pi;
    const int npart = g.npart, nrdata = g.nrdata, nidata = g.nidata, gids = g.gids;
    const int ncell123 = g.ncellb, L = g.L;
    const Real H = g.H, ell1 = g.ell1, ell2 = g.ell2, ell3 = g.ell3;
    const std::int64_t ncl = g.ncl;
    const Real omega = g.omega, alpha = g.alpha;
    auto kp = g.kp;
    const KeyCSR spc = g.spc;
    auto nr = g.nr;
    auto ni = g.ni;
    auto nnew = g.nnew;
    auto qc = g.qc;
    DvceArray1D<int> owner("q3_owner", npart);
    Kokkos::deep_copy(owner, -1);
    Kokkos::parallel_for("q3_stencil", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int c) {
      if (kp(c) < 0 || pr(IPM,c) <= 0.0) return;
      const Real xc0 = pr(IPX,c), yc0 = pr(IPY,c), zc0 = pr(IPZ,c);
      const Real mc = pr(IPM,c), ts = pr(IPTS,c);
      // shear-periodic x1: no family across an x1 face (its members were wrapped
      // with the shear offset; the template does not hold there)
      if (lt.shear && (xc0 - ell1 < lt.x1min || xc0 + ell1 >= lt.x1max)) return;
      // (nor across a physical x3 face: its members were mirrored)
      if (!lt.x3per && (zc0 - ell3 < lt.x3min || zc0 + ell3 >= lt.x3max)) return;
      if (!mm.In(xc0, yc0, zc0, L, lt.Anchor(pi(PGID,c)))) return;
      const int sp = pi(PSP,c);
      const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
      const Real tolv = alpha*H/tau;
      int mem[27]; int nm = 0; mem[nm++] = c;
      Real vlo[3] = {pr(IPVX,c), pr(IPVY,c), pr(IPVZ,c)};
      Real vhi[3] = {vlo[0], vlo[1], vlo[2]};
      for (int e=-1; e<=1; ++e) {
        for (int b=-1; b<=1; ++b) {
          for (int a=-1; a<=1; ++a) {
            if (a == 0 && b == 0 && e == 0) continue;
            const Real w = mc/static_cast<Real>((a != 0 ? 2 : 1)*(b != 0 ? 2 : 1)*
                                                (e != 0 ? 2 : 1));
            const Real tx = xc0 + a*ell1, ty = yc0 + b*ell2, tz = zc0 + e*ell3;
            const std::int64_t key = sp*ncl + mm.Cell(tx, ty, tz, L);
            int best = -1; Real bd = 1.0e30;
            for (int k=spc.First(key), ke=spc.Last(key); k<ke; ++k) {
              const int q = spc.list(k);
              if (q == c || pr(IPM,q) <= 0.0 || owner(q) >= 0) continue;
              if (Kokkos::fabs(pr(IPM,q) - w) > 1.0e-6*w || pr(IPTS,q) != ts) continue;
              const Real dx = Kokkos::fabs(T::MinImage(pr(IPX,q) - tx, lt.lx));
              const Real dy = Kokkos::fabs(T::MinImage(pr(IPY,q) - ty, lt.ly));
              const Real dz = Kokkos::fabs(T::MinImage(pr(IPZ,q) - tz, lt.lz));
              if (dx > dtol*ell1 || dy > dtol*ell2 || dz > dtol*ell3) continue;
              if (!mm.In(pr(IPX,q), pr(IPY,q), pr(IPZ,q), L, lt.Anchor(pi(PGID,q)))) continue;
              const Real d = Kokkos::fmax(dx/ell1, Kokkos::fmax(dy/ell2, dz/ell3));
              if (d < bd) {bd = d; best = q;}
            }
            if (best < 0) return;
            mem[nm++] = best;
            const Real v[3] = {pr(IPVX,best), pr(IPVY,best), pr(IPVZ,best)};
            for (int d=0; d<3; ++d) {
              vlo[d] = Kokkos::fmin(vlo[d], v[d]); vhi[d] = Kokkos::fmax(vhi[d], v[d]);
              if (vhi[d] - vlo[d] > tolv) return;
            }
          }
        }
      }
      for (int k=0; k<27; ++k) {   // claim; roll back on a conflict (retried next pass)
        if (Kokkos::atomic_compare_exchange(&owner(mem[k]), -1, c) != -1) {
          for (int j=0; j<k; ++j) Kokkos::atomic_exchange(&owner(mem[j]), -1);
          return;
        }
      }
      const int knew = Kokkos::atomic_fetch_add(&nnew(0), 1);
      double M = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
      for (int q=0; q<nrdata; ++q) nr(q,knew) = 0.0;
      for (int k=0; k<27; ++k) {
        const int t = mem[k]; const Real m = pr(IPM,t);
        M += m;
        sx += m*T::MinImage(pr(IPX,t) - xc0, lt.lx);
        sy += m*T::MinImage(pr(IPY,t) - yc0, lt.ly);
        sz += m*T::MinImage(pr(IPZ,t) - zc0, lt.lz);
        for (int q=0; q<nrdata; ++q) nr(q,knew) += m*pr(q,t);
      }
      for (int q=0; q<nrdata; ++q) nr(q,knew) /= M;
      nr(IPTS,knew) = ts;
      Real px = xc0 + sx/M, py = yc0 + sy/M, pz = zc0 + sz/M;
      lt.WrapPos(px, py, pz);
      nr(IPM,knew) = M;
      nr(IPX,knew) = px; nr(IPX1,knew) = px; nr(IPY,knew) = py; nr(IPY1,knew) = py;
      nr(IPZ,knew) = pz; nr(IPZ1,knew) = pz;
      for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,c);
      ni(PLEV,knew) = L;
      const int leaf = lt.LeafOwner(px, py, pz, lt.Anchor(pi(PGID,c)));
      const int blk = (leaf < 0) ? lt.Anchor(pi(PGID,c)) : leaf/ncell123;
      ni(PGID,knew) = gids + ((blk < lt.nmb) ? blk : lt.Anchor(pi(PGID,c)));
      for (int k=0; k<27; ++k) pr(IPM,mem[k]) = 0.0;
      Kokkos::atomic_add(&qc(0), 1.0);
      Kokkos::atomic_add(&qc(1), M);
    });
  }
  static void GKeys(const DustGasDrag *, const GLevel<LT> &g, const int glag,
                    const DvceArray1D<std::int64_t> &kq) {
    const auto lt = g.lt;
    const auto mm = g.mm;
    auto pr = g.pr;
    auto pi = g.pi;
    const int npart = g.npart, L = g.L;
    const Real H1 = g.H1, H2 = g.H2, H3 = g.H3;
    const int m1 = g.m1, m2 = g.m2, m3 = g.m3;
    const std::int64_t ncl = g.ncl;
    auto kp = g.kp;
    Kokkos::parallel_for("q3_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      kq(p) = -1;
      if (kp(p) < 0 || pr(IPM,p) <= 0.0) return;
      const Real x = T::WrapPosition(pr(IPX,p), lt.x1min, lt.x1max);
      const Real y = T::WrapPosition(pr(IPY,p), lt.x2min, lt.x2max);
      const Real z = T::WrapPosition(pr(IPZ,p), lt.x3min, lt.x3max);
      const Real u = (x - lt.x1min)/H1, v = (y - lt.x2min)/H2, w = (z - lt.x3min)/H3;
      const int I = T::WrapIndex(static_cast<int>(Kokkos::floor(u)), m1);
      const int J = T::WrapIndex(static_cast<int>(Kokkos::floor(v)), m2);
      const int Kc = T::WrapIndex(static_cast<int>(Kokkos::floor(w)), m3);
      const int am = lt.Anchor(pi(PGID,p));
      if (!mm.In(x, y, z, L, am)) return;
      for (int dk=-glag; dk<=glag; ++dk) {   // deep zone only: the outer band is stage X's
        for (int dj=-glag; dj<=glag; ++dj) {
          for (int di=-glag; di<=glag; ++di) {
            const Real xn = lt.x1min + (T::WrapIndex(I + di, m1) + 0.5)*H1;
            const Real yn = lt.x2min + (T::WrapIndex(J + dj, m2) + 0.5)*H2;
            const Real zn = lt.x3min + (T::WrapIndex(Kc + dk, m3) + 0.5)*H3;
            if (!mm.In(xn, yn, zn, L, am)) return;
          }
        }
      }
      const int si = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(u - Kokkos::floor(u)))));
      const int sj = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(v - Kokkos::floor(v)))));
      const int sk = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(w - Kokkos::floor(w)))));
      // Morton order of the 4 x 4 x 4 sub-cells
      const int sub = ((sk >> 1) << 5) | ((sj >> 1) << 4) | ((si >> 1) << 3) |
                      ((sk & 1) << 2) | ((sj & 1) << 1) | (si & 1);
      kq(p) = ((pi(PSP,p)*ncl + (static_cast<std::int64_t>(Kc)*m2 + J)*m1 + I) << 6) + sub;
    });
  }
  static void WalkCSR(const DustGasDrag *, const GLevel<LT> &g, const int nsp,
                      const DvceArray1D<std::int64_t> &kq, KeyCSR &qsc) {
    BuildKeyCSR(kq, SpeciesKeyBound(nsp, g.ncl, 6), 3, qsc);  // retired-compat
  }
  static void Walk(const DustGasDrag *, const GLevel<LT> &g, const KeyCSR &qsc,
                   const DvceArray1D<int> &cfirst, const int ncell, const DvceArray1D<Real> &mrootsp,
                   const Real qden, const int K, const int rot, const int rev, const bool alpha_leaf,
                   const DvceArray1D<int> &gid, const DvceArray1D<int> &gknew,
                   const DvceArray1D<int> &ngroup, const Real smax) {
    const auto lt = g.lt;
    auto pr = g.pr;
    auto pi = g.pi;
    const int nidata = g.nidata, L = g.L;
    const Real H1 = g.H1, H2 = g.H2, H3 = g.H3, H = g.H;
    const int m1 = g.m1, m2 = g.m2;
    const std::int64_t ncl = g.ncl;
    const Real omega = g.omega, alpha = g.alpha, fmin = g.fmin;
    auto nr = g.nr;
    auto ni = g.ni;
    auto nnew = g.nnew;
    auto qc = g.qc;
    auto aref = g.aref;
    const Real smax1 = smax*H1, smax2 = smax*H2;
    const Real smax3 = smax*H3;
    Kokkos::parallel_for("q3_walk", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncell),
    KOKKOS_LAMBDA(const int ic) {
      const int u0 = cfirst(ic), u1 = cfirst(ic + 1);
      const std::int64_t kcell = qsc.ukey(u0) >> 6;
      const int sp = static_cast<int>(kcell/ncl);
      const std::int64_t cell = kcell % ncl;
      const int I = static_cast<int>(cell % m1), J = static_cast<int>((cell/m1) % m2);
      const int Kc = static_cast<int>(cell/(static_cast<std::int64_t>(m1)*m2));
      const Real mq = mrootsp(sp)/qden;
      const Real xcc = lt.x1min + (I + 0.5)*H1, ycc = lt.x2min + (J + 0.5)*H2;
      const Real zcc = lt.x3min + (Kc + 0.5)*H3;
      // cloud_merge_alpha_leaf: the velocity tolerance on the cell size of the leaf
      // that holds this level-L cell (the gas resolution there), alpha H_leaf/tau, H_leaf = 2^(L -
      // leaf level) H; the default keeps alpha H/tau
      Real hfac = 1.0;
      if (alpha_leaf) {
        const int am0 = lt.Anchor(pi(PGID,qsc.list(qsc.start(u0))));
        for (int Lc=L-1; Lc>=0 && lt.MergeOK(xcc, ycc, zcc, Lc, 0, am0); --Lc) hfac *= 2.0;
      }
      Real gM[8], vlo[8][3], vhi[8][3];
      Real xlo[8], xhi[8], ylo[8], yhi[8], zlo[8], zhi[8], gts[8];
      int gg[8], gfirst[8], gn[8], nopen = 0;
      auto commit = [&](const int j) {
        const int knew = Kokkos::atomic_fetch_add(&nnew(0), 1);
        gknew(gg[j]) = knew;
        for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,gfirst[j]);
        ni(PLEV,knew) = L;
        nr(IPTS,knew) = gts[j];
        aref(0,knew) = xcc; aref(1,knew) = ycc; aref(2,knew) = zcc;
        Kokkos::atomic_add(&qc(2), 1.0);
        Kokkos::atomic_add(&qc(3), static_cast<double>(gn[j]));
        Kokkos::atomic_add(&qc(4), static_cast<double>(gM[j]));
      };
      for (int s=0; s<64; ++s) {
        const int sub = rev ? (rot + 64 - s) % 64 : (rot + s) % 64;
        int kb = 0, ke = 0;
        for (int u=u0; u<u1; ++u) {
          if ((qsc.ukey(u) & 63) == sub) {kb = qsc.start(u); ke = qsc.start(u + 1); break;}
        }
        for (int k=kb; k<ke; ++k) {
          const int p = qsc.list(k);
          const Real m = pr(IPM,p);
          if (m <= 0.0) continue;
          const Real dx = T::MinImage(pr(IPX,p) - xcc, lt.lx);
          const Real dy = T::MinImage(pr(IPY,p) - ycc, lt.ly);
          const Real dz = T::MinImage(pr(IPZ,p) - zcc, lt.lz);
          const Real v[3] = {pr(IPVX,p), pr(IPVY,p), pr(IPVZ,p)};
          const Real ts = pr(IPTS,p);
          const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
          const Real tolv = alpha*hfac*H/tau;
          int j = -1;
          for (int o=0; o<nopen && j<0; ++o) {
            if (gts[o] != ts) continue;
            bool ok = (Kokkos::fmax(xhi[o], dx) - Kokkos::fmin(xlo[o], dx) <= smax1) &&
                      (Kokkos::fmax(yhi[o], dy) - Kokkos::fmin(ylo[o], dy) <= smax2) &&
                      (Kokkos::fmax(zhi[o], dz) - Kokkos::fmin(zlo[o], dz) <= smax3);
            for (int d=0; d<3 && ok; ++d) {
              ok = Kokkos::fmax(vhi[o][d], v[d]) - Kokkos::fmin(vlo[o][d], v[d]) <= tolv;
            }
            if (ok) j = o;
          }
          if (j < 0) {
            if (nopen == K || nopen == 8) {Kokkos::atomic_add(&qc(5), 1.0); continue;}
            j = nopen++;
            gg[j] = Kokkos::atomic_fetch_add(&ngroup(0), 1);
            gM[j] = 0.0; gn[j] = 0;
            xlo[j] = xhi[j] = dx; ylo[j] = yhi[j] = dy; zlo[j] = zhi[j] = dz;
            gts[j] = ts; gfirst[j] = p;
            for (int d=0; d<3; ++d) {vlo[j][d] = vhi[j][d] = v[d];}
          }
          gid(p) = gg[j];
          gM[j] += m; ++gn[j];
          xlo[j] = Kokkos::fmin(xlo[j], dx); xhi[j] = Kokkos::fmax(xhi[j], dx);
          ylo[j] = Kokkos::fmin(ylo[j], dy); yhi[j] = Kokkos::fmax(yhi[j], dy);
          zlo[j] = Kokkos::fmin(zlo[j], dz); zhi[j] = Kokkos::fmax(zhi[j], dz);
          for (int d=0; d<3; ++d) {
            vlo[j][d] = Kokkos::fmin(vlo[j][d], v[d]); vhi[j][d] = Kokkos::fmax(vhi[j][d], v[d]);
          }
          if (gM[j] >= mq*(1.0 - 1.0e-12)) {
            commit(j);
            --nopen;
            if (j != nopen) {
              gM[j] = gM[nopen]; xlo[j] = xlo[nopen]; xhi[j] = xhi[nopen];
              ylo[j] = ylo[nopen]; yhi[j] = yhi[nopen]; zlo[j] = zlo[nopen];
              zhi[j] = zhi[nopen]; gts[j] = gts[nopen]; gg[j] = gg[nopen];
              gfirst[j] = gfirst[nopen]; gn[j] = gn[nopen];
              for (int d=0; d<3; ++d) {vlo[j][d] = vlo[nopen][d]; vhi[j][d] = vhi[nopen][d];}
            }
          }
        }
      }
      for (int o=0; o<nopen; ++o) {
        if (gM[o] >= fmin*mq) commit(o);
        if (gknew(gg[o]) < 0) Kokkos::atomic_add(&qc(6), static_cast<double>(gn[o]));
      }
    });
  }
  static void Accumulate(const DustGasDrag *, const GLevel<LT> &g,
                         const DvceArray1D<std::int64_t> &kq, const DvceArray1D<int> &gid,
                         const DvceArray1D<int> &gknew, const bool pbud) {
    const auto lt = g.lt;
    auto pr = g.pr;
    const int npart = g.npart, nrdata = g.nrdata;
    auto nr = g.nr;
    auto acc = g.acc;
    auto aref = g.aref;
    Kokkos::parallel_for("q3_accumulate", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (kq(p) < 0) return;
      const int g_ = gid(p);
      if (g_ < 0) return;
      const int k = gknew(g_);
      if (k < 0) return;
      const Real m = pr(IPM,p);
      Kokkos::atomic_add(&acc(0,k), static_cast<double>(m));
      Kokkos::atomic_add(&acc(1,k), static_cast<double>(m*T::MinImage(pr(IPX,p) - aref(0,k), lt.lx)));
      Kokkos::atomic_add(&acc(2,k), static_cast<double>(m*T::MinImage(pr(IPY,p) - aref(1,k), lt.ly)));
      Kokkos::atomic_add(&acc(3,k), static_cast<double>(m*T::MinImage(pr(IPZ,p) - aref(2,k), lt.lz)));
      for (int q=0; q<nrdata; ++q) {
        if (q == IPM || q == IPX || q == IPY || q == IPZ || q == IPX1 || q == IPY1 ||
            q == IPZ1) continue;
        if (q == IPTS) continue;
        Kokkos::atomic_add(&nr(q,k), m*pr(q,p));
      }
      if (!pbud) pr(IPM,p) = 0.0;
    });
  }
  static void Finalize(const DustGasDrag *, const GLevel<LT> &g, const int k0, const int k1) {
    const auto lt = g.lt;
    const int nrdata = g.nrdata, gids = g.gids, ncell123 = g.ncellb;
    auto nr = g.nr;
    auto ni = g.ni;
    auto acc = g.acc;
    auto aref = g.aref;
    Kokkos::parallel_for("q3_finalize_state", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
    KOKKOS_LAMBDA(const int k) {
      const double M = acc(0,k);
      for (int q=0; q<nrdata; ++q) {
        if (q == IPM || q == IPX || q == IPY || q == IPZ || q == IPX1 || q == IPY1 ||
            q == IPZ1) continue;
        if (q == IPTS) continue;
        nr(q,k) /= M;
      }
      Real px = aref(0,k) + acc(1,k)/M, py = aref(1,k) + acc(2,k)/M, pz = aref(2,k) + acc(3,k)/M;
      lt.WrapPos(px, py, pz);
      nr(IPM,k) = M;
      nr(IPX,k) = px; nr(IPX1,k) = px; nr(IPY,k) = py; nr(IPY1,k) = py;
      nr(IPZ,k) = pz; nr(IPZ1,k) = pz;
      const int a = lt.Anchor(ni(PGID,k));
      const int leaf = lt.LeafOwner(px, py, pz, a);
      const int blk = (leaf < 0) ? a : leaf/ncell123;
      ni(PGID,k) = gids + ((blk < lt.nmb) ? blk : a);
    });
  }
};

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MergeQuantumT
//! \brief The production merge pass (quantum mode) of the refinable clouds, both traversals
//! (the former 2-D quantum branch and the 3-D pass; the 2-D tent pass was
//! retired):  // retired-compat
//! (1) far-field release of lobed level-l clouds in merge cells of level l - 1; per level lev
//! (L = lev - 1): (X, with cloud_merge_stencil) the exact re-merge of complete P1 families
//! (3^d members, masses m_c / 2^(|a|+|b|(+|c|))) into level-L clouds; (G) the mass-quantum walk
//! of the plain level-lev pieces of the deep merge zone in each level-L cell (4^d sub-cells,
//! Morton order rotated by the pass number) into level-L clouds of the quantum
//! m_root / 2^(d L), position and velocity bounding boxes, K open groups; the pass budget (with
//! the structure-relative beta over the 2d neighbor leaves) accepts or rejects whole groups; the
//! compaction.  3-D only: cloud_merge_alpha_leaf (old 2-D restart files carry the key, which 2-D
//! never applied).  Returns the merges.

template <class LTT>
int DustGasDrag::MergeQuantumT(const LTT &lt_in) {
  using GK = GKern<LTT>;
  constexpr bool k3 = (LeafApi<LTT>::kDim == 3);
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart0 = ppar->nprtcl_thispack;
  const auto lt = lt_in;
  const int nlev = GK::Nlev(this, lt);
  // several ranks: the budget exchanges are collective, so a rank without
  // particles runs the pass too; it stops only when no rank has any
  if (nlev == 0 || SumRanks(npart0) == 0) return 0;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const int gids = pmy_pack->gids;
  const int ncellb = GK::Ncell(this, lt);   // leaf cells of a block slot
  EnsureNextTag();
  const auto mm = GK::Mask(this, pmesh, lt, nlev);
  // the pass budget: lent the entry leaf fields, lcum the change so far
  // leaves of the halo blocks included (several ranks); only the owner of a leaf judges it
  const bool pbud = cloud_merge_budget > 0.0;
  // (the leaf cells of all block slots, halo included, are indexed by int)
  if (pbud && static_cast<long long>(lt.nfld)*ncellb > static_cast<long long>(INT_MAX)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "the pass budget of G (cloud_merge_budget) on rank " << global_variable::my_rank
              << ": " << lt.nfld << " block slots of " << ncellb << " leaf cells exceed the int "
              << "range" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nbt = pbud ? lt.nfld*ncellb : 1;
  const int nown = pbud ? lt.nmb*ncellb : 1;
  DvceArray2D<Real> lent(std::string(k3 ? "budget3d_lent" : "budget_lent"), 4, nbt);
  DvceArray2D<Real> lcum(std::string(k3 ? "budget3d_lcum" : "budget_lcum"), 4, nbt);
  if (pbud) {
    BudgetDepositT(lt, ppar->prtcl_rdata, ppar->prtcl_idata, npart0, 1.0, lent);
    HaloReverseAddFlat(lent, 4);   // the owners' leaves hold every rank's clouds
  }
  bool bdirty = false;
  double gcb[4] = {0.0, 0.0, 0.0, 0.0};   // rejected groups, pieces, mass; evaluations

  // (1) far-field release of lobed level-l clouds in merge cells of level l-1
  const int nrelease = ReleaseFarLobesT(lt, mm, ncellb);
  if (nrelease > 0) bdirty = true;
  // root-level cloud mass per species (first pass; shared with the resampling).  Several ranks:
  // every rank gets here and the value is reduced over them
  EnsureRootCloudMass();
  // any number of species (the keys sp ncl + cell are 64-bit; the masses a View)
  const int nsp = std::max(1, std::max(static_cast<int>(cloud_merge_mroot.size()), nspecies));
  const auto mrootsp = SpeciesRootMass(cloud_merge_mroot, nsp);
  // the mass floor of the budgets (cloud_absorb_floor root clouds per root cell): the largest
  // root cloud mass of the species (species 0 may have none, e.g. eps_1 = 0)
  Real mroot_floor = 0.0;
  for (const Real m : cloud_merge_mroot) mroot_floor = std::max(mroot_floor, m);
  const Real rvol = GK::RootMeasure(this, lt);
  // cloud_merge_budget_kappa > 0: the budget of leaf t is max(min(beta, kappa s_t), 1e-12), s_t =
  // max over its 2d neighbor leaves of |rho_n - rho_t| / max(rho_t, floor) at the entry of the pass
  const Real bkappa = cloud_merge_budget_kappa;
  DvceArray1D<Real> lbeta(std::string(k3 ? "budget3d_lbeta" : "budget_lbeta"), (pbud && bkappa > 0.0) ? nbt : 1);
  if (pbud && bkappa > 0.0) {
    int bnd[6];
    GK::Bounds(pmesh, lt, bnd);
    const Real rfloor = cloud_absorb_floor*mroot_floor/rvol;
    BudgetStructureT(this, lt, cloud_geometry, lent, lbeta, nbt, nown, ncellb, bnd,
                     cloud_merge_budget, bkappa, rfloor);
  }
  const Real omega = is_shearing_box ? omega0 : 0.0;
  const Real alpha = cloud_merge_alpha, fmin = cloud_merge_min_mass;
  // the members of a G group and of an X family have bitwise equal ts;
  // the new cloud keeps that value instead of the mass-weighted mean (always)
  const bool alpha_leaf = k3 && cloud_merge_alpha_leaf;   // (never applied in 2-D)
  double qtot[8] = {0.0};   // X families, mass; G groups, pieces, mass; deferred, residual pieces
  // cloud_split_max_level = L: levels <= L only (a cloud above L is a level-L cloud in mass; it
  // widens back by the cloud instant switch when the mesh coarsens)
  const int lev_g = (cloud_split_max_level >= 0) ? std::min(nlev, cloud_split_max_level) : nlev;
  for (int lev=1; lev<=lev_g; ++lev) {
    GLevel<LTT> g;
    g.lt = lt; g.mm = mm;
    g.lev = lev; g.L = lev - 1;
    g.pr = ppar->prtcl_rdata;
    g.pi = ppar->prtcl_idata;
    g.npart = ppar->nprtcl_thispack;
    g.nrdata = nrdata; g.nidata = nidata; g.gids = gids; g.ncellb = ncellb;
    g.omega = omega; g.alpha = alpha; g.fmin = fmin;
    GK::Level(this, g);
    const int L = g.L, npart = g.npart;
    auto pr = g.pr;
    auto pi = g.pi;
    const std::int64_t ncl = g.ncl;
    // plain level-lev pieces with mass, by species and level-L cell
    DvceArray1D<std::int64_t> kp(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                    std::string(k3 ? "merge3d_kp" : "merge_kp")), npart);
    Kokkos::parallel_for(k3 ? "merge3d_keys" : "merge_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      const int code = pi(PLEV,p);
      kp(p) = -1;
      if (GK::Lobed(code)) return;
      if (CloudGeometry::CloudLevel(code) != lev || pr(IPM,p) <= 0.0) return;
      kp(p) = pi(PSP,p)*ncl + GK::Cell(mm, pr, p, L);
    });
    g.kp = kp;
    BuildKeyCSR(kp, SpeciesKeyBound(nsp, ncl, 0), 0, g.spc);   // keys sp ncl + cell
    const int nmax = (npart > 0) ? npart : 1;
    g.nmax = nmax;
    g.nr = DvceArray2D<Real>(std::string(k3 ? "q3_nr" : "q_nr"), nrdata, nmax);
    g.ni = DvceArray2D<int>(std::string(k3 ? "q3_ni" : "q_ni"), nidata, nmax);
    g.acc = DvceArray2D<double>(std::string(k3 ? "q3_acc" : "q_acc"), k3 ? 4 : 3, nmax);   // m, m dx, m dy (, m dz)
    g.aref = DvceArray2D<Real>(std::string(k3 ? "q3_ref" : "q_ref"), k3 ? 3 : 2, nmax);
    g.nnew = DvceArray1D<int>(std::string(k3 ? "q3_nnew" : "q_nnew"), 1);
    g.qc = DvceArray1D<double>(std::string(k3 ? "q3_counters" : "q_counters"), 8);
    auto nr = g.nr;
    auto ni = g.ni;
    auto nnew = g.nnew;
    const int K = cloud_merge_K;
    const int rot = cloud_merge_pass % GK::kNsub, rev = (cloud_merge_pass/GK::kNsub) % 2;
    // ---- stage X: complete P1 families back into their level-L parent
    if (cloud_merge_stencil) GK::StageX(this, g, cloud_merge_stencil_tol);
    // ---- the leaf change made so far in this pass (release, stage X, coarser levels)
    const Real gb_floor = cloud_absorb_floor*mroot_floor/rvol;
    const Real gb_vmin = alpha*g.H*((omega > 0.0) ? omega : 0.0);
    if (pbud) {
      int nx_now = 0;
      Kokkos::deep_copy(nx_now, Kokkos::subview(nnew, 0));
      if (nx_now > 0) bdirty = true;
      Kokkos::deep_copy(lcum, 0.0);
      if (AnyRank(bdirty)) {
        BudgetDepositT(lt, pr, pi, npart, 1.0, lcum);
        BudgetDepositT(lt, nr, ni, nx_now, 1.0, lcum);
        HaloReverseAddFlat(lcum, 4);   // the owners' leaves hold every rank's change
        Kokkos::parallel_for(k3 ? "budget3d_cum" : "budget_cum", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
        KOKKOS_LAMBDA(const int t) {
          for (int c=0; c<4; ++c) lcum(c,t) -= lent(c,t);
        });
      }
    }
    DvceArray1D<int> krej(std::string(k3 ? "q3_krej" : "q_krej"), nmax);   // new G cloud rejected by the budget
    DvceArray2D<Real> ldelg(std::string(k3 ? "q3_ldel" : "q_ldel"), 4, nbt);
    DvceArray1D<int> lviolg(std::string(k3 ? "q3_lviol" : "q_lviol"), nbt);
    // ---- stage G: the cell walk.  Pieces at level lev in level-L merge cells -> level-L clouds
    DvceArray1D<int> gid(std::string(k3 ? "q3_gid" : "q_gid"), npart), gknew(std::string(k3 ? "q3_gknew" : "q_gknew"), nmax);
    DvceArray1D<int> ngroup(std::string(k3 ? "q3_ngroup" : "q_ngroup"), 1);
    Kokkos::deep_copy(gid, -1);
    Kokkos::deep_copy(gknew, -1);
    DvceArray1D<std::int64_t> kq(std::string(k3 ? "q3_key" : "q_key"), npart);
    int k0 = 0;   // the new G clouds are k0..k1-1 (after those of stage X)
    Kokkos::deep_copy(k0, Kokkos::subview(nnew, 0));
    {
      GK::GKeys(this, g, cloud_merge_g_lag, kq);
      KeyCSR qsc;
      GK::WalkCSR(this, g, nsp, kq, qsc);
      // the distinct cells: their first entry in qsc.ukey (the 4^d sub-keys of a cell are adjacent)
      DvceArray1D<int> cfirst(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                 std::string(k3 ? "q3_cfirst" : "q_cfirst")), qsc.nu + 1);
      int ncell = 0;
      {
        auto uk = qsc.ukey; const int nu = qsc.nu;
        Kokkos::parallel_scan(k3 ? "q3_cells" : "q_cells", Kokkos::RangePolicy<>(DevExeSpace(), 0, nu),
        KOKKOS_LAMBDA(const int u, int &upd, const bool final) {
          const bool first = (u == 0) || ((uk(u) >> GK::kShift) != (uk(u - 1) >> GK::kShift));
          if (first && final) cfirst(upd) = u;
          if (first) ++upd;
          if (final && u == nu - 1) cfirst(upd) = nu;
        }, ncell);
      }
      GK::Walk(this, g, qsc, cfirst, ncell, mrootsp, GK::Qden(L), K, rot, rev, alpha_leaf, gid,
               gknew, ngroup, cloud_merge_smax);
      // the members of committed groups into their new clouds (the pieces are used up below)
      GK::Accumulate(this, g, kq, gid, gknew, pbud);
      int k1 = k0;
      Kokkos::deep_copy(k1, Kokkos::subview(nnew, 0));
      if (k1 > k0) GK::Finalize(this, g, k0, k1);
      // ---- accept or reject the proposed groups by the pass budget.  Several ranks: proposals
      // on any rank are evaluated on all of them (their leaves may be touched by a neighbor's)
      if (pbud && AnyRank(k1 > k0)) {
        const GroupBudgetResult gr = GroupBudgetRoundsT(this, lt, npart, k0, k1, kq, gid, gknew,
            krej, pr, pi, nr, ni, ldelg, lviolg, lent, lcum, lbeta, bkappa, cloud_merge_budget,
            cloud_merge_budget_evals, nown, nbt, ncellb, gb_floor, gb_vmin);
        gcb[0] += gr.rg; gcb[1] += gr.rp; gcb[2] += gr.rm; gcb[3] += gr.nev;
        if (gr.rg < static_cast<double>(k1 - k0)) bdirty = true;
      }
    }
    auto qch = Kokkos::create_mirror_view_and_copy(HostMemSpace(), g.qc);
    for (int c=0; c<8; ++c) qtot[c] += qch(c);
    int ncommit = 0;
    Kokkos::deep_copy(ncommit, Kokkos::subview(nnew, 0));
    // compaction: drop the used-up pieces, append the new clouds that are kept
    if (CompactMergedT<k3>(ppar, nr, ni, npart, ncommit)) bdirty = true;
  }
  const int nfinal = ppar->nprtcl_thispack;
  Kokkos::fence();
  pmesh->nprtcl_thisrank = nfinal;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nfinal;
  pmesh->nprtcl_total = nfinal;
  ++cloud_merge_pass;
  const unsigned long long nevents = static_cast<unsigned long long>(qtot[0] + qtot[2] - gcb[0]);
  cloud_coarsen_events += nevents;
  if (phst_track) phst_g_events += nevents;
  cloud_coarsen_in += static_cast<unsigned long long>(GK::kFam*qtot[0] + qtot[3] - gcb[1]);
  cloud_far_release += nrelease;
  return static_cast<int>(nevents);
}
template int DustGasDrag::MergeQuantumT<LeafTraversal>(const LeafTraversal &);
template int DustGasDrag::MergeQuantumT<LeafTraversal3D>(const LeafTraversal3D &);


//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SortParticles
//! \brief Reorders the particle arrays by (block, cell of the block, old position): the cells
//! of a block row by row (2-D: (j, i); 3-D clouds: (k, j, i)), stable inside a cell;
//! cloud_sort_order = random (test) takes a random permutation instead.  Every nrdata / nidata
//! column moves (tags with their particles); the arrays are swapped with spare arrays of the
//! same extent (kept for the next call).  Runs at the start of a cycle (RefineClouds), after
//! the split, R, G and their migration, where no send list is pending.  State kept per particle
//! slot across this point: the cell-order index (set to the identity here: a rebuild would give
//! the same), and with self-gravity the particle accelerations of the last gather with their
//! per-slot snapshot (x, y, z, m, PLEV, PGID; dust_gravity.cpp), which the opening solve
//! compares slot by slot and regathers where they differ: a moved particle is regathered, to
//! the same value (the field and the gather inputs are unchanged).  Conservation does not
//! depend on the order.  The resampling and the merge pass take their candidates in array
//! order, so their results change with the order (statistically the same; not bitwise equal
//! to an unsorted run), as do the round-off of the deposits.

void DustGasDrag::SortParticles() {
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  if (npart < 2) return;
  Kokkos::fence();
  Kokkos::Timer t;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const int nx1 = pmy_pack->pmesh->mb_indcs.nx1, nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  // 3-D keys (block, k, j, i) as the cell-order index (UpdateCloudOrder); in 2-D nx3 = 1, k = 0
  // (point particles, <dust>/point_sort_interval: 3-D keys on any 3-D mesh; cloud_dim == 3 implies
  // three_d, so the cloud keys are unchanged)
  const bool z3 = cloud_dim == 3 || pmy_pack->pmesh->three_d;
  const int nx3 = z3 ? pmy_pack->pmesh->mb_indcs.nx3 : 1;
  const int gids = pmy_pack->gids, nmb = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool randomize = cloud_sort_random;
  const int cyc = pmy_pack->pmesh->ncycle;
  const std::int64_t ncell = static_cast<std::int64_t>(nx1)*nx2*nx3, np = npart;
  KeyScratch &s = Scratch();
  auto key = Grow(s.skey, npart, "sort_key");
  Kokkos::parallel_for("sort_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    m = (m < 0) ? 0 : ((m >= nmb) ? nmb - 1 : m);
    int i = static_cast<int>(Kokkos::floor((pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1));
    int j = static_cast<int>(Kokkos::floor((pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2));
    i = (i < 0) ? 0 : ((i >= nx1) ? nx1 - 1 : i);
    j = (j < 0) ? 0 : ((j >= nx2) ? nx2 - 1 : j);
    int k = 0;
    if (z3) {
      k = static_cast<int>(Kokkos::floor((pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3));
      k = (k < 0) ? 0 : ((k >= nx3) ? nx3 - 1 : k);
    }
    const std::int64_t cell = (static_cast<std::int64_t>(k)*nx2 + j)*nx1 + i;
    if (randomize) {   // (diagnostic) a random permutation instead (splitmix64 of p, cycle)
      std::uint64_t z = static_cast<std::uint64_t>(p) + 0x9E3779B97F4A7C15ull*(static_cast<std::uint64_t>(cyc) + 1);
      z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27))*0x94D049BB133111EBull; z ^= z >> 31;
      key(p) = static_cast<std::int64_t>(z >> 34)*np + p;
      return;
    }
    key(p) = (m*ncell + cell)*np + p;
  });
  Kokkos::fence();
  const double t_key = t.seconds();
  t.reset();
  SortFirst(key, npart);
  Kokkos::fence();
  const double t_sort = t.seconds();
  t.reset();
  if (sort_spare_r.extent(0) != pr.extent(0) || sort_spare_r.extent(1) != pr.extent(1) ||
      sort_spare_i.extent(0) != pi.extent(0) || sort_spare_i.extent(1) != pi.extent(1)) {
    sort_spare_r = DvceArray2D<Real>();
    sort_spare_i = DvceArray2D<int>();
    sort_spare_r = DvceArray2D<Real>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"),
                                     pr.extent(0), pr.extent(1));
    sort_spare_i = DvceArray2D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"),
                                    pi.extent(0), pi.extent(1));
  }
  auto nr = sort_spare_r;
  auto ni = sort_spare_i;
  int moved = 0;
  Kokkos::parallel_reduce("sort_move", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int k, int &mv) {
    const int p = static_cast<int>(key(k) % np);
    for (int c=0; c<nrdata; ++c) nr(c,k) = pr(c,p);
    for (int c=0; c<nidata; ++c) ni(c,k) = pi(c,p);
    if (p != k) ++mv;
  }, moved);
  ppar->prtcl_rdata = nr;
  ppar->prtcl_idata = ni;
  sort_spare_r = pr;
  sort_spare_i = pi;
  ++cloud_sort_count;
  // the cell-order index: after a sort in cell order it is the identity (what a rebuild would
  // give, the same keys); after a random permutation it is rebuilt (UpdateCloudOrder)
  if (cloud_order_interval > 0) {
    if (randomize) {
      cloud_ord_n = 0;
    } else {
      if (cloud_ord.extent_int(0) < npart) {
        cloud_ord = DvceArray1D<int>();
        cloud_ord_tag = DvceArray1D<std::uint64_t>();
        cloud_ord = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "cloud_ord"),
                                     npart + npart/4);
        cloud_ord_tag = DvceArray1D<std::uint64_t>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                   "cloud_ord_tag"), npart + npart/4);
      }
      auto ord = cloud_ord; auto tg = cloud_ord_tag;
      Kokkos::parallel_for("cloud_order_identity", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int k) {ord(k) = k; tg(k) = GetTag(ni,k);});
      cloud_ord_n = npart;
      cloud_sort_cycle = cyc;
    }
  }
  Kokkos::fence();
  const double t_move = t.seconds();
  (void) t_key; (void) t_sort; (void) t_move; (void) moved;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::UpdateCloudOrder
//! \brief cloud_order_interval = N: an index of the particles in (block, cell,
//! position) order, without moving them; the two production cloud kernels run in this order
//! (CloudParForOrdered), the particles appended since the index was built in their own.  Every
//! N cycles it is rebuilt; in between it is checked at the end of RefineClouds: each indexed
//! slot must still hold the particle (PTAG) it held, which fails after any compaction or
//! reordering (resampling, merge pass, restart, ...).  The storage order, and with it every
//! resampling and merge decision, is that of a run without the index.

void DustGasDrag::UpdateCloudOrder() {
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  // the sort of this cycle (cell order) left the identity index: nothing to rebuild
  if (cloud_sort_cycle == pmy_pack->pmesh->ncycle && cloud_ord_n == npart) return;
  bool rebuild = (pmy_pack->pmesh->ncycle % cloud_order_interval == 0) || cloud_ord_n == 0 ||
                 cloud_ord_n > npart;
  if (!rebuild) {
    auto ord = cloud_ord; auto tg = cloud_ord_tag;
    int bad = 0;
    Kokkos::parallel_reduce("cloud_order_check", Kokkos::RangePolicy<>(DevExeSpace(), 0, cloud_ord_n),
    KOKKOS_LAMBDA(const int k, int &b) {
      const int p = ord(k);
      if (p < 0 || p >= npart || GetTag(pi,p) != tg(k)) ++b;
    }, bad);
    rebuild = bad > 0;
  }
  if (!rebuild) return;
  if (npart < 2) {cloud_ord_n = 0; return;}
  Kokkos::Timer t;
  const int nx1 = pmy_pack->pmesh->mb_indcs.nx1, nx2 = pmy_pack->pmesh->mb_indcs.nx2;
  // 3-D keys (block, k, j, i); in 2-D nx3 = 1, k = 0
  const int nx3 = cloud_dim == 3 ? pmy_pack->pmesh->mb_indcs.nx3 : 1;
  const bool z3 = cloud_dim == 3;
  const int gids = pmy_pack->gids, nmb = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const std::int64_t ncell = static_cast<std::int64_t>(nx1)*nx2*nx3, np = npart;
  KeyScratch &s = Scratch();
  auto key = Grow(s.skey, npart, "sort_key");
  Kokkos::parallel_for("cloud_order_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    m = (m < 0) ? 0 : ((m >= nmb) ? nmb - 1 : m);
    int i = static_cast<int>(Kokkos::floor((pr(IPX,p) - mbsize.d_view(m).x1min)/mbsize.d_view(m).dx1));
    int j = static_cast<int>(Kokkos::floor((pr(IPY,p) - mbsize.d_view(m).x2min)/mbsize.d_view(m).dx2));
    i = (i < 0) ? 0 : ((i >= nx1) ? nx1 - 1 : i);
    j = (j < 0) ? 0 : ((j >= nx2) ? nx2 - 1 : j);
    int k = 0;
    if (z3) {
      k = static_cast<int>(Kokkos::floor((pr(IPZ,p) - mbsize.d_view(m).x3min)/mbsize.d_view(m).dx3));
      k = (k < 0) ? 0 : ((k >= nx3) ? nx3 - 1 : k);
    }
    key(p) = (m*ncell + (static_cast<std::int64_t>(k)*nx2 + j)*nx1 + i)*np + p;
  });
  SortFirst(key, npart);
  if (cloud_ord.extent_int(0) < npart) {
    cloud_ord = DvceArray1D<int>();
    cloud_ord_tag = DvceArray1D<std::uint64_t>();
    cloud_ord = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "cloud_ord"), npart + npart/4);
    cloud_ord_tag = DvceArray1D<std::uint64_t>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                  "cloud_ord_tag"), npart + npart/4);
  }
  auto ord = cloud_ord; auto tg = cloud_ord_tag;
  Kokkos::parallel_for("cloud_order_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int k) {
    const int p = static_cast<int>(key(k) % np);
    ord(k) = p; tg(k) = GetTag(pi,p);
  });
  cloud_ord_n = npart;
  ++cloud_ord_builds;
  Kokkos::fence();
}

}  // namespace dust
