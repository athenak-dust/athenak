//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud_device.cpp
//! \brief Device (Kokkos) implementation of the moving-particle P1 split of refinable
//! clouds.  Same rules as the host SplitCloudsP1, without its
//! diagnostics (split log, split oracle), which keep the
//! host path.  Flag and count the pieces per particle, exclusive scans for the slots and
//! the new tags, then one fill kernel.  The particle order is the host version's: kept
//! particles in their order, then the pieces of the flagged particles in theirs.

#include <algorithm>
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
// parent (centre child with both edge children attached)
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

// is the particle split (or a lobe released) now?  Port of the host flag kernel; the
// eligibility on the actual leaves, from the particle's block m.
KOKKOS_INLINE_FUNCTION
bool P1Flag(const LeafTraversal &t, const int m, const int code, const Real x, const Real y,
            const Real slack) {
  const int lev = SMRCompositeTransfer::CloudLevel(code);
  const int l1 = SMRCompositeTransfer::CloudLobe1(code);
  const int l2 = SMRCompositeTransfer::CloudLobe2(code);
  if (l1 == 0 && l2 == 0) return t.CloudInsideFiner(x, y, lev, slack, m);
  if (lev > 0 && t.CloudInsideFiner(x, y, lev - 1, slack, m)) {
    const Real h1 = t.CloudHalfWidth1(lev - 1), h2 = t.CloudHalfWidth2(lev - 1);
    return ((l1 & 1) && t.CloudInsideFiner(x - h1, y, lev - 1, slack, m)) ||
           ((l1 & 2) && t.CloudInsideFiner(x + h1, y, lev - 1, slack, m)) ||
           ((l2 & 1) && t.CloudInsideFiner(x, y - h2, lev - 1, slack, m)) ||
           ((l2 & 2) && t.CloudInsideFiner(x, y + h2, lev - 1, slack, m));
  }
  return false;
}

// pieces of a flagged particle along both directions (port of the host event code)
KOKKOS_INLINE_FUNCTION
void P1Event(const LeafTraversal &t, const int m, const int code, const Real x, const Real y,
             const Real slack, DPiece p1[3],
             DPiece p2[3], int &n1, int &n2, int &clev, bool &parent) {
  const int lev = SMRCompositeTransfer::CloudLevel(code);
  const int l1 = SMRCompositeTransfer::CloudLobe1(code);
  const int l2 = SMRCompositeTransfer::CloudLobe2(code);
  if (l1 == 0 && l2 == 0) {
    const Real h1 = t.CloudHalfWidth1(lev), h2 = t.CloudHalfWidth2(lev);
    n1 = P1Pieces(3, t.CloudInsideFiner(x - h1, y, lev, slack, m),
                  t.CloudInsideFiner(x + h1, y, lev, slack, m), p1);
    n2 = P1Pieces(3, t.CloudInsideFiner(x, y - h2, lev, slack, m),
                  t.CloudInsideFiner(x, y + h2, lev, slack, m), p2);
    clev = lev + 1;
    parent = true;
  } else {
    const Real h1 = t.CloudHalfWidth1(lev - 1), h2 = t.CloudHalfWidth2(lev - 1);
    n1 = P1Pieces(l1, t.CloudInsideFiner(x - h1, y, lev - 1, slack, m),
                  t.CloudInsideFiner(x + h1, y, lev - 1, slack, m), p1);
    n2 = P1Pieces(l2, t.CloudInsideFiner(x, y - h2, lev - 1, slack, m),
                  t.CloudInsideFiner(x, y + h2, lev - 1, slack, m), p2);
    clev = lev;
    parent = false;
  }
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SplitCloudsP1Device
//! \brief Device P1 split; returns the number of split/released particles.

int DustGasDrag::SplitCloudsP1Device() {
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart = ppar->nprtcl_thispack;
  if (npart == 0) return 0;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const auto transfer = smr_transfer;
  const auto lt = leaf_traversal;   // the actual leaves
  const Real slack = cloud_split_slack;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
  const int gids = pmy_pack->gids;
  const int ncell12 = qdep.extent_int(4)*qdep.extent_int(3);

  // pieces per particle (0 = not flagged), and the numbers of parents / lobed events.
  // The work arrays persist between cycles (per-cycle allocations and frees dominated the
  // host side of a mixed-mesh cycle).
  if (split_np.extent_int(0) < npart) {
    split_np = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                   "dust_split_np"), npart + npart/4);
    split_off = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                    "dust_split_off"), npart + npart/4);
  }
  auto np = split_np;
  auto tagoff = split_off;
  int nflag = 0, nparent = 0;
  Kokkos::parallel_reduce("dust_split_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &sf, int &sp) {
    const int code = pi(PLEV,p);
    const Real x = pr(IPX,p), y = pr(IPY,p);
    const int m = lt.Anchor(pi(PGID,p));   // anchor block
    if (!P1Flag(lt, m, code, x, y, slack)) {np(p) = 0; return;}
    DPiece p1[3], p2[3]; int n1, n2, clev; bool parent;
    P1Event(lt, m, code, x, y, slack, p1, p2, n1, n2, clev, parent);
    np(p) = n1*n2;
    sf += 1;
    if (parent) sp += 1;
  }, Kokkos::Sum<int>(nflag), Kokkos::Sum<int>(nparent));
  if (nflag == 0) return 0;
  EnsureNextTag();

  // In place: the centre piece (k = 0, which keeps the tag) overwrites its parent's slot,
  // and the other pieces are appended after the npart particles, at the offsets (and with
  // the tags) of the exclusive scan of the extra pieces.  The particle arrays keep 25%
  // headroom so that the append rarely reallocates.
  int ntags = 0;
  Kokkos::parallel_scan("dust_split_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (final) tagoff(p) = upd;
    upd += (np(p) > 0) ? np(p) - 1 : 0;
  }, ntags);
  const int nnew = npart + ntags;
  if (ppar->prtcl_rdata.extent_int(1) < nnew) {
    Kokkos::resize(ppar->prtcl_rdata, nrdata, nnew + nnew/4);
    Kokkos::resize(ppar->prtcl_idata, nidata, nnew + nnew/4);
  }
  auto prv = ppar->prtcl_rdata;
  auto piv = ppar->prtcl_idata;
  // tags of this rank: tag0, tag0 + nranks, ... (unique over the ranks; 64-bit, the counter is
  // moved on, with a check, before the fill)
  const std::uint64_t tag0 = IssueTags(static_cast<std::uint64_t>(ntags));
  const std::uint64_t tstride = static_cast<std::uint64_t>(global_variable::nranks);
  Kokkos::parallel_for("dust_split_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int p) {
    if (np(p) == 0) return;
    const int code = piv(PLEV,p);
    const Real x = prv(IPX,p), y = prv(IPY,p), mass = prv(IPM,p);
    DPiece p1[3], p2[3]; int n1, n2, clev; bool parent;
    P1Event(lt, lt.Anchor(piv(PGID,p)), code, x, y, slack, p1, p2, n1, n2,
            clev, parent);
    const Real e1 = transfer.CloudHalfWidth1(clev), e2 = transfer.CloudHalfWidth2(clev);
    // the parent's slot is written last, after the appended pieces have copied it
    for (int k=n1*n2-1; k>=0; --k) {
      const int a = k % n1, b = k/n1;
      const int s = (k == 0) ? p : npart + tagoff(p) + k - 1;
      const Real cx = SMRCompositeTransfer::WrapPosition(x + p1[a].off*e1, transfer.x1min,
                                                         transfer.x1max);
      const Real cy = SMRCompositeTransfer::WrapPosition(y + p2[b].off*e2, transfer.x2min,
                                                         transfer.x2max);
      if (k > 0) {
        for (int c=0; c<nrdata; ++c) prv(c,s) = prv(c,p);
        for (int c=0; c<nidata; ++c) piv(c,s) = piv(c,p);
        // the centre keeps the tag
        SetTag(piv, s, tag0 + static_cast<std::uint64_t>(tagoff(p) + k - 1)*tstride);
      }
      prv(IPX,s) = cx; prv(IPX1,s) = cx;
      prv(IPY,s) = cy; prv(IPY1,s) = cy;
      prv(IPM,s) = mass*p1[a].frac*p2[b].frac;
      piv(PLEV,s) = clev | (p1[a].lobe << sh1) | (p2[b].lobe << sh2);
      piv(PGID,s) = gids + NewCloudBlock(lt, cx, cy, lt.Anchor(piv(PGID,s)), ncell12);
    }
  });
  cloud_split_count += nparent;
  cloud_lobe_split_count += nflag - nparent;
  ppar->nprtcl_thispack = nnew;
  pmesh->nprtcl_thisrank = nnew;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nnew;
  pmesh->nprtcl_total = nnew;
  return nflag;
}


namespace {

constexpr int kNbMax = 96;     // pieces in a centre's tent support
constexpr int kNodeMax = 64;   // nodes of a centre (regular offsets and irregular pieces)

KOKKOS_INLINE_FUNCTION
Real Snap(const Real d, const Real ell) {
  const Real r = Kokkos::round(d/ell);
  return (Kokkos::fabs(d - r*ell) < 0.1*ell) ? r*ell : d;
}

// the level-L cell of a position (the keys that sort the pieces of a merge level), and the
// merge-cell mask: the level-L cell and the B cells around it carry no dilated level above L.
// The mask is evaluated on the actual leaves from the block m of the
// particle asking (LeafTraversal::MergeOK), no longer read from a map built on the level-L grid.
struct MergeMask {
  LeafTraversal lt;
  int B = 0;
  static constexpr int kL = SMRCompositeTransfer::kCloudLevelMask + 1;
  int n1[kL], n2[kL];
  Real dx1[kL], dx2[kL];
  Real x1min, x2min, x1max, x2max;
  KOKKOS_INLINE_FUNCTION
  std::int64_t Cell(const Real x, const Real y, const int L) const {   // J n1 + I
    const int I = SMRCompositeTransfer::WrapIndex(static_cast<int>(Kokkos::floor(
        (SMRCompositeTransfer::WrapPosition(x, x1min, x1max) - x1min)/dx1[L])), n1[L]);
    const int J = SMRCompositeTransfer::WrapIndex(static_cast<int>(Kokkos::floor(
        (SMRCompositeTransfer::WrapPosition(y, x2min, x2max) - x2min)/dx2[L])), n2[L]);
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
// synchronise the device; thrust's sort allocated its own storage in every call) made the first
// sparse version 1-3% slower than the dense tables at BA 128/256.  Released at Kokkos::finalize.
// Slots: KeyCSR 0 pieces, 1 level-L clouds, 2 lobed clouds, 3 q-stage walk; CellLists 1 tent
// commits (slot 0 held the commits of the removed quantum absorption).  A slot is
// rebuilt only after the tables built in it are out of use (kernels on the one stream run in
// order).
struct KeyScratch {
  static constexpr int kSlots = 4;
  DvceArray1D<int> pos, uidx, list[kSlots], start[kSlots], hhead[kSlots];
  DvceArray1D<std::int64_t> comp, ukey[kSlots], hkey[kSlots], skey;   // skey: SortParticles
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

// key < 0: not listed; listed keys below kbound.  rank / inv (optional): the members of
// a key in the order of rank(p) (a permutation of 0..n-1 with inverse inv) instead of particle order
void BuildKeyCSR(const DvceArray1D<std::int64_t> &key, const std::int64_t kbound, const int slot,
                 KeyCSR &c, const DvceArray1D<int> &rank, const DvceArray1D<int> &inv) {
  const bool ranked = rank.extent(0) > 0;
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
  KOKKOS_LAMBDA(const int p) {if (key(p) >= 0) comp(pos(p)) = key(p)*nn + (ranked ? rank(p) : p);});
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
    ls(k) = ranked ? inv(static_cast<int>(comp(k) % nn)) : static_cast<int>(comp(k) % nn);
    if (uidx(k) >= 0) {uk(uidx(k)) = comp(k)/nn; st(uidx(k)) = k;}
    if (k == nl_ - 1) st(nu_) = nl_;
  });
  c.nu = nu;
}
void BuildKeyCSR(const DvceArray1D<std::int64_t> &key, const std::int64_t kbound, const int slot,
                 KeyCSR &c) {
  BuildKeyCSR(key, kbound, slot, c, DvceArray1D<int>(), DvceArray1D<int>());
}

// per-cell linked lists of the new clouds of a pass (the dense head array needed one
// entry per level-L cell): an open-addressing hash table from the 64-bit cell key to the first
// cloud of the cell (capacity a power of two, at least twice the clouds), nnext as before.
// Push links cloud k in front of its cell's list and returns the previous first cloud; Head
// returns the first cloud of a cell, -1 if none.  As with the dense heads, a lookup only asks
// for cells filled by earlier kernels (a probe never stops at a slot filled later).
struct CellLists {
  DvceArray1D<std::int64_t> hkey;   // -1: empty
  DvceArray1D<int> hhead;
  unsigned mask = 0;
  void Allocate(const int nmax, const int slot) {   // buffers of KeyScratch slot
    int cap = 16;
    while (cap < 2*nmax + 16) cap *= 2;
    KeyScratch &s = Scratch();
    hkey = Grow(s.hkey[slot], cap, "cl_key");
    hhead = Grow(s.hhead[slot], cap, "cl_head");
    Kokkos::deep_copy(Kokkos::subview(hkey, std::make_pair(0, cap)), static_cast<std::int64_t>(-1));
    Kokkos::deep_copy(Kokkos::subview(hhead, std::make_pair(0, cap)), -1);
    mask = static_cast<unsigned>(cap - 1);
  }
  KOKKOS_INLINE_FUNCTION
  static unsigned Hash(const std::int64_t key) {
    return static_cast<unsigned>((static_cast<std::uint64_t>(key)*0x9E3779B97F4A7C15ull) >> 32);
  }
  KOKKOS_INLINE_FUNCTION
  int Push(const std::int64_t key, const int k) const {
    unsigned h = Hash(key) & mask;
    while (true) {
      const std::int64_t prev = Kokkos::atomic_compare_exchange(&hkey(h), static_cast<std::int64_t>(-1), key);
      if (prev == -1 || prev == key) return Kokkos::atomic_exchange(&hhead(h), k);
      h = (h + 1) & mask;
    }
  }
  KOKKOS_INLINE_FUNCTION
  int Head(const std::int64_t key) const {
    unsigned h = Hash(key) & mask;
    while (true) {
      const std::int64_t s = hkey(h);
      if (s == key) return hhead(h);
      if (s == -1) return -1;
      h = (h + 1) & mask;
    }
  }
};


}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MergeCloudsDevice
//! \brief Device version of MergeClouds: far-field lobe release and the P1
//! tent merge with the node rule and the merge-side node convention, as the host prototype.
//! Differences, by construction of the parallel algorithm: centres are the local maxima
//! (mass, then x, then y) among unblocked zone pieces within the blocking radius, in two
//! rounds (the host designates greedily, heaviest first); commits run in 16 colour rounds of
//! the centres' level-L cells (I mod 4, J mod 4), so that the supports, the nodes' implicit
//! masses and the new clouds of concurrent centres never overlap.  On a quiet lattice the
//! centres and the commits are the host's; for irregular particles the order differs.

namespace {

// sign * (mass, momentum) of the live clouds 0..n-1 of (r, i) on the leaves, with the
// production cloud weights
void BudgetDeposit(const LeafTraversal &lt, const DvceArray2D<Real> &r,
                   const DvceArray2D<int> &i, const int n, const Real sign, DvceArray2D<Real> &f) {
  if (n <= 0) return;
  auto f_ = f;
  auto tr = lt;
  Kokkos::parallel_for("budget_deposit", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const int p) {
    if (r(IPM,p) <= 0.0) return;
    const Real m = sign*r(IPM,p);
    const Real v1 = r(IPVX,p), v2 = r(IPVY,p), v3 = r(IPVZ,p);
    tr.ForEachCloudWeight(r(IPX,p), r(IPY,p), i(PLEV,p), tr.Anchor(i(PGID,p)), [&](const int t, const Real w) {
      Kokkos::atomic_add(&f_(0,t), m*w); Kokkos::atomic_add(&f_(1,t), m*w*v1);
      Kokkos::atomic_add(&f_(2,t), m*w*v2); Kokkos::atomic_add(&f_(3,t), m*w*v3);
    });
  });
}

// a G group's hash in [0, 1) from its new cloud's tag (splitmix64; the same on every
// rank and in every run), for cloud_merge_budget_thin
KOKKOS_INLINE_FUNCTION
Real GroupHash01(const std::uint64_t tag) {   // the 64-bit tag (the same value below 2^31)
  std::uint64_t z = tag + 0x9E3779B97F4A7C15ULL;
  z = (z ^ (z >> 30))*0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27))*0x94D049BB133111EBULL;
  z ^= z >> 31;
  return static_cast<Real>(z >> 11)*(1.0/9007199254740992.0);
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
        if (pi(PSP,p) != sp || SMRCompositeTransfer::CloudLobe1(code) ||
            SMRCompositeTransfer::CloudLobe2(code) ||
            SMRCompositeTransfer::CloudLobe3(code)) return;   // (x3 lobe: 0 in 2-D)
        const int l = SMRCompositeTransfer::CloudLevel(code);
        if (l == 0 && pr(IPM,p) > a) a = pr(IPM,p);
        if (l == 1 && pr(IPM,p) > b) b = pr(IPM,p);
      }, Kokkos::Max<Real>(m0), Kokkos::Max<Real>(m1));
#if MPI_PARALLEL_ENABLED
      Real mm[2] = {m0, m1};
      MPI_Allreduce(MPI_IN_PLACE, mm, 2, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
      m0 = mm[0]; m1 = mm[1];
#endif
      // a level-1 cloud carries 1/4 of a root cloud in 2-D, 1/8 in 3-D
      cloud_merge_mroot[sp] = (m0 > 0.0) ? m0 : (clouds3d ? 8.0 : 4.0)*m1;
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
}  // namespace

int DustGasDrag::MergeCloudsDevice() {
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart0 = ppar->nprtcl_thispack;
  const auto transfer = smr_transfer;
  const auto lt = leaf_traversal;   // the actual leaves
  const int nlev = transfer.cloud_nlev;
  // several ranks: the budget exchanges are collective, so a rank without
  // particles runs the pass too; it stops only when no rank has any
  if (nlev == 0 || SumRanks(npart0) == 0) return 0;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const int sh1 = SMRCompositeTransfer::kCloudLobeShift1;
  const int sh2 = SMRCompositeTransfer::kCloudLobeShift2;
  const int gids = pmy_pack->gids;
  const int ncell12 = qdep.extent_int(4)*qdep.extent_int(3);
  const int B = cloud_merge_buffer;
  EnsureNextTag();

  // level-L cell keys and the merge-cell mask on the actual leaves
  MergeMask mm;
  mm.lt = lt;
  mm.B = B;
  mm.x1min = transfer.x1min; mm.x2min = transfer.x2min;
  mm.x1max = transfer.x1max; mm.x2max = transfer.x2max;
  for (int L=0; L<nlev; ++L) {
    mm.dx1[L] = transfer.root_dx1/static_cast<Real>(1 << L);
    mm.dx2[L] = transfer.root_dx2/static_cast<Real>(1 << L);
    mm.n1[L] = transfer.ncoarse1 << L;
    mm.n2[L] = transfer.ncoarse2 << L;
  }


  // one field-error budget for the whole pass (stage G, and X with
  // cloud_merge_budget_x).  lent:
  // leaf mass and momentum at the entry of the pass; lcum: the change made so far (recomputed from
  // the actual state after stage X of each level, then advanced by what each stage accepts)
  const bool pbud = cloud_merge_mode == 1 && cloud_merge_budget > 0.0;
  // leaves of the halo blocks included (one rank: nfld = nmb)
  const int nbt = pbud ? leaf_traversal.nfld*qdep.extent_int(3)*qdep.extent_int(4) : 1;
  DvceArray2D<Real> lent("budget_lent", 4, nbt), lcum("budget_lcum", 4, nbt);
  Kokkos::Timer tpass;
  double tbud = 0.0;                 // seconds in the budget kernels
  bool bdirty = false;               // the pass has changed the particle set
  // G / C: rejected groups, pieces, mass; evaluations (stage C was removed: zero, kept for the
  // log columns)
  double gcb[2][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
  double bcap = 0.0;                 // stages that hit the evaluation limit
  const bool xgate = pbud && cloud_merge_budget_x;
  double xgb[4] = {0.0, 0.0, 0.0, 0.0};   // X: rejected families, pieces, mass; evaluations
  // the leaves of this rank's blocks: only their owner judges them (one rank: all)
  const int nown = pbud ? leaf_traversal.nmb*qdep.extent_int(3)*qdep.extent_int(4) : 1;
  if (pbud) {
    Kokkos::Timer btimer;
    BudgetDeposit(lt, ppar->prtcl_rdata, ppar->prtcl_idata, npart0, 1.0, lent);
    HaloReverseAddFlat(lent, 4);   // the owners' leaves hold every rank's clouds
    Kokkos::fence();
    tbud += btimer.seconds();
  }

  // (1) far-field release of lobed level-l clouds in merge cells of level l-1
  int nrelease = 0;
  {
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    DvceArray1D<int> np(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rel_np"), npart);
    Kokkos::parallel_reduce("rel_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &sum) {
      const int code = pi(PLEV,p);
      const int lev = SMRCompositeTransfer::CloudLevel(code);
      const int l1 = SMRCompositeTransfer::CloudLobe1(code);
      const int l2 = SMRCompositeTransfer::CloudLobe2(code);
      if ((l1 == 0 && l2 == 0) || lev == 0 || !mm.In(pr(IPX,p), pr(IPY,p), lev - 1, lt.Anchor(pi(PGID,p)))) {
        np(p) = 0; return;
      }
      DPiece a[3], b[3];
      np(p) = ReleasePieces(l1, a)*ReleasePieces(l2, b);
      sum += 1;
    }, Kokkos::Sum<int>(nrelease));
    if (nrelease > 0) {
      DvceArray1D<int> off(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rel_off"), npart), toff("rel_toff", npart);
      int nnew = 0, ntags = 0;
      Kokkos::parallel_scan("rel_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
        if (final) off(p) = upd;
        upd += (np(p) == 0) ? 1 : np(p);
      }, nnew);
      Kokkos::parallel_scan("rel_tscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
        if (final) toff(p) = upd;
        upd += (np(p) > 0) ? np(p) - 1 : 0;
      }, ntags);
      DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, nnew);
      DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, nnew);
      // tags unique over the ranks, 64-bit
      const std::uint64_t tag0 = IssueTags(static_cast<std::uint64_t>(ntags));
      const std::uint64_t tstride = static_cast<std::uint64_t>(global_variable::nranks);
      Kokkos::parallel_for("rel_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        const int s0 = off(p);
        if (np(p) == 0) {
          for (int c=0; c<nrdata; ++c) prn(c,s0) = pr(c,p);
          for (int c=0; c<nidata; ++c) pin(c,s0) = pi(c,p);
          return;
        }
        const int code = pi(PLEV,p);
        const int lev = SMRCompositeTransfer::CloudLevel(code);
        DPiece a[3], b[3];
        const int n1 = ReleasePieces(SMRCompositeTransfer::CloudLobe1(code), a);
        const int n2 = ReleasePieces(SMRCompositeTransfer::CloudLobe2(code), b);
        const Real e1 = transfer.CloudHalfWidth1(lev), e2 = transfer.CloudHalfWidth2(lev);
        int k = 0;
        for (int jb=0; jb<n2; ++jb) {
          for (int ja=0; ja<n1; ++ja) {
            const int s = s0 + k;
            const Real cx = SMRCompositeTransfer::WrapPosition(pr(IPX,p) + a[ja].off*e1,
                                                               transfer.x1min, transfer.x1max);
            const Real cy = SMRCompositeTransfer::WrapPosition(pr(IPY,p) + b[jb].off*e2,
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
      });
      ppar->prtcl_rdata = prn;
      ppar->prtcl_idata = pin;
      ppar->nprtcl_thispack = nnew;
    }
  }

  if (nrelease > 0) bdirty = true;
  // root-level cloud mass per species (first pass, as the host; shared with the resampling).
  // Several ranks: every rank gets here (the pass returns early only when no rank has particles)
  // and the value is reduced over them
  EnsureRootCloudMass();
  // any number of species (the keys sp ncl + cell are 64-bit; the masses a View)
  const int nsp = std::max(1, std::max(static_cast<int>(cloud_merge_mroot.size()), nspecies));
  const auto mrootsp = SpeciesRootMass(cloud_merge_mroot, nsp);
  // the mass floor of the budgets (cloud_absorb_floor root clouds per root cell): the largest
  // root cloud mass of the species (species 0 may have none, e.g. eps_1 = 0)
  Real mroot_floor = 0.0;
  for (const Real m : cloud_merge_mroot) mroot_floor = std::max(mroot_floor, m);
  // cloud_merge_budget_kappa > 0: the budget of leaf t is
  // max(min(beta, kappa s_t), 1e-12), s_t = max over its four neighbour leaves (the leaves at one
  // cell width beside its centre, of any level) of |rho_n - rho_t| / max(rho_t, floor), with the
  // dust densities at the entry of the pass.  A lossy merge may change a leaf by at most the
  // structure it resolves: none on a quiet state (LinA, a dusty wave), as before where the dust
  // has structure of order one (BA's clumps and outflows); exact merges (round-off) still pass.
  const Real bkappa = cloud_merge_budget_kappa;
  DvceArray1D<Real> lbeta("budget_lbeta", (pbud && bkappa > 0.0) ? nbt : 1);
  if (pbud && bkappa > 0.0) {
    const Real beta = cloud_merge_budget;
    const int n1 = qdep.extent_int(4), n2 = qdep.extent_int(3);
    const int is_ = pmy_pack->pmesh->mb_indcs.is, ie_ = pmy_pack->pmesh->mb_indcs.ie;
    const int js_ = pmy_pack->pmesh->mb_indcs.js, je_ = pmy_pack->pmesh->mb_indcs.je;
    const Real rfloor = cloud_absorb_floor*mroot_floor/(transfer.root_dx1*transfer.root_dx2);
    const auto ltb = lt;
    const auto tr = transfer;
    const int nown_ = nown;
    // the leaf masses with the owners' totals also on the halo leaves (several
    // ranks: lent holds this rank's share there; one rank: a copy)
    DvceArray2D<Real> lmass("budget_lmass", 1, nbt);
    Kokkos::parallel_for("budget_lmass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
    KOKKOS_LAMBDA(const int t) {lmass(0,t) = lent(0,t);});
    HaloForwardFlat(lmass, 1);
    Kokkos::parallel_for("budget_structure", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
    KOKKOS_LAMBDA(const int t) {
      lbeta(t) = beta;
      // only this rank's leaves are judged (the halo leaves keep beta; their owner decides)
      if (t >= nown_) return;
      const int b = t/(n1*n2), r = t % (n1*n2), j = r/n1, i = r % n1;
      if (i < is_ || i > ie_ || j < js_ || j > je_) return;
      const Real dx = ltb.box(b,2), dy = ltb.box(b,3);
      const Real xc = ltb.box(b,0) + (i - is_ + 0.5)*dx, yc = ltb.box(b,1) + (j - js_ + 0.5)*dy;
      const Real rho = lmass(0,t)/(dx*dy);
      const Real den = Kokkos::fmax(rho, rfloor);
      Real s = 0.0;
      for (int d=0; d<4; ++d) {
        const Real x = SMRCompositeTransfer::WrapPosition(xc + ((d == 0) ? dx : ((d == 1) ? -dx : 0.0)),
                                                          tr.x1min, tr.x1max);
        const Real y = SMRCompositeTransfer::WrapPosition(yc + ((d == 2) ? dy : ((d == 3) ? -dy : 0.0)),
                                                          tr.x2min, tr.x2max);
        const int tn = ltb.LeafOwner(x, y, b);
        if (tn < 0) continue;
        const int bn = tn/(n1*n2);
        const Real rn = lmass(0,tn)/(ltb.box(bn,2)*ltb.box(bn,3));
        s = Kokkos::fmax(s, Kokkos::fabs(rn - rho)/den);
      }
      lbeta(t) = Kokkos::fmax(Kokkos::fmin(beta, bkappa*s), 1.0e-12);
    });
  }

  // counters: 0 commits, 1 contributions, 2 rejected, 3 pending, 4 nc shifts, 5 nc skipped
  // 6..10 (pending log): not ready (mask), lobe wait, over limit, tolerance, min mass
  DvceArray1D<long long> icnt("merge_icnt", 11);
  DvceArray1D<Real> dk("merge_dk", 2);   // dK_grad, dK_res
  const Real omega = is_shearing_box ? omega0 : 0.0;
  const Real alpha = cloud_merge_alpha, fmin = cloud_merge_min_mass;
  // the members of a G group and of an X family have bitwise equal ts;
  // with cloud_merge_exact_ts the new cloud keeps that value instead of the mass-weighted mean,
  // whose last bits follow the order of the atomic sums (and so the storage order)
  const bool exact_ts = cloud_merge_exact_ts;
  const bool nc = cloud_merge_nc;
  const bool lobe_wait = cloud_merge_lobe_wait > 0, lobe_wait_dir = cloud_merge_lobe_wait == 2;
  // the cell lists of unreleased lobed clouds are read by the tent pass with
  // cloud_merge_lobe_wait only; quantum G (production) does not build them, whatever
  // cloud_merge_lobe_wait says (old inputs and restarts carry 1)
  const bool lobe_keys = cloud_merge_mode != 1 && lobe_wait;

  // quantum mode: counters over levels (the slots 13..21 of the removed stage C stay zero and
  // keep the log columns)
  double qtot[24] = {0.0};
  unsigned long long q_events = 0, q_in = 0;
  double xwtot[2] = {0.0, 0.0};     // X protection: waiting pieces, their mass
  // absorptions rejected by the budget, their mass, evaluations (the absorption was removed:
  // zero, kept for the log columns)
  const double budtot[3] = {0.0, 0.0, 0.0};

  for (int lev=1; lev<=nlev; ++lev) {
    const int L = lev - 1;
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    const Real H1 = mm.dx1[L], H2 = mm.dx2[L], H = Kokkos::min(H1, H2);
    const Real ell1 = transfer.CloudHalfWidth1(lev), ell2 = transfer.CloudHalfWidth2(lev);
    const int n1 = mm.n1[L], n2 = mm.n2[L];
    const std::int64_t ncl = static_cast<std::int64_t>(n1)*n2;   // level-L cells (keys: sp ncl + cell)
    // cell keys: plain level-lev pieces (m > 0) and plain level-L clouds, by species
    DvceArray1D<std::int64_t> kp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_kp"), npart);
    DvceArray1D<std::int64_t> kc("merge_kc", npart);
    DvceArray1D<std::int64_t> kl("merge_kl", npart);   // unreleased lobed level-lev clouds (lobe wait)
    Kokkos::parallel_for("merge_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      const int code = pi(PLEV,p);
      kp(p) = -1; kc(p) = -1; kl(p) = -1;
      if (SMRCompositeTransfer::CloudLobe1(code) || SMRCompositeTransfer::CloudLobe2(code)) {
        if (lobe_keys && SMRCompositeTransfer::CloudLevel(code) == lev && pr(IPM,p) > 0.0) {
          kl(p) = pi(PSP,p)*ncl + mm.Cell(pr(IPX,p), pr(IPY,p), L);
        }
        return;
      }
      const int l = SMRCompositeTransfer::CloudLevel(code);
      const std::int64_t key = pi(PSP,p)*ncl + mm.Cell(pr(IPX,p), pr(IPY,p), L);
      if (l == lev && pr(IPM,p) > 0.0) kp(p) = key;
      if (l == L) kc(p) = key;
    });
    // sparse cell lists (KeyCSR), no table over the level-L grid
    KeyCSR spc, scc, slc;
    const std::int64_t kbnd = SpeciesKeyBound(nsp, ncl, 0);   // keys sp ncl + cell
    BuildKeyCSR(kp, kbnd, 0, spc);
    // the level-L clouds are looked up by the tent pass only (in quantum mode they were read by
    // the absorption, since removed)
    if (cloud_merge_mode != 1) BuildKeyCSR(kc, kbnd, 1, scc);
    if (lobe_keys) BuildKeyCSR(kl, kbnd, 2, slc);
    if (cloud_merge_mode == 1) {
      // ===== quantum mode =====
      // X: exact P1-stencil re-merge (geometric, strict tolerance, velocity bounding box);
      // G: mass-quantum grouping of plain level-lev pieces in level-L merge cells into level-L
      //    clouds (true-velocity and position bounding boxes, K open groups, rotated walks);
      // then the compaction.  (Stage C, the same walk at level lev in the fine interior, and the
      // absorption of G residuals were removed.)
      const int nmax = (npart > 0) ? npart : 1;
      DvceArray2D<Real> nr("q_nr", nrdata, nmax);
      DvceArray2D<int> ni("q_ni", nidata, nmax);
      DvceArray1D<int> nkind("q_nkind", nmax);             // 0 X (final), 1 G
      DvceArray2D<double> acc("q_acc", 8, nmax);           // m, m dx, m dy, m dx^2, m dy^2, m v^2, ref x, ref y
      DvceArray1D<int> nnew("q_nnew", 1);
      DvceArray1D<double> qc("q_counters", 24);
      const int K = cloud_merge_K, rot = cloud_merge_pass % 16, rev = (cloud_merge_pass/16) % 2;
      const Real ell = Kokkos::min(ell1, ell2);
      // ---- stage X
      if (cloud_merge_stencil) {
        DvceArray1D<int> owner("q_owner", npart);
        Kokkos::deep_copy(owner, -1);
        DvceArray1D<int> xknew("dust_xknew", xgate ? npart : 1);
        if (xgate) Kokkos::deep_copy(xknew, -1);
        const bool aud = xgate;
        DvceArray2D<double> xfam("q_xfam", 4, xgate ? nmax : 1);
        const Real dtol = cloud_merge_stencil_tol;
        const Real vn1 = H1*H1/6.0, vn2 = H2*H2/6.0, vo1 = ell1*ell1/6.0, vo2 = ell2*ell2/6.0;
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
          if (aud) xknew(c) = knew;
          double M = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, svv = 0.0;
          for (int q=0; q<nrdata; ++q) nr(q,knew) = 0.0;
          for (int k=0; k<9; ++k) {
            const int t = mem[k]; const Real m = pr(IPM,t);
            const Real dx = transfer.MinImage1(pr(IPX,t) - xc0), dy = transfer.MinImage2(pr(IPY,t) - yc0);
            M += m; sx += m*dx; sy += m*dy; sxx += m*dx*dx; syy += m*dy*dy;
            svv += m*(pr(IPVX,t)*pr(IPVX,t) + pr(IPVY,t)*pr(IPVY,t) + pr(IPVZ,t)*pr(IPVZ,t));
            for (int q=0; q<nrdata; ++q) nr(q,knew) += m*pr(q,t);
          }
          for (int q=0; q<nrdata; ++q) nr(q,knew) /= M;
          if (exact_ts) nr(IPTS,knew) = ts;   // the family's common ts
          const Real px = SMRCompositeTransfer::WrapPosition(xc0 + sx/M, transfer.x1min, transfer.x1max);
          const Real py = SMRCompositeTransfer::WrapPosition(yc0 + sy/M, transfer.x2min, transfer.x2max);
          nr(IPM,knew) = M;
          nr(IPX,knew) = px; nr(IPX1,knew) = px; nr(IPY,knew) = py; nr(IPY1,knew) = py;
          for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,c);
          ni(PLEV,knew) = L;
          {
            ni(PGID,knew) = gids + NewCloudBlock(lt, px, py, lt.Anchor(ni(PGID,knew)), ncell12);
          }
          nkind(knew) = 0;
          const double vb2 = nr(IPVX,knew)*nr(IPVX,knew) + nr(IPVY,knew)*nr(IPVY,knew) +
                             nr(IPVZ,knew)*nr(IPVZ,knew);
          if (xgate) {   // decided after the kernel
            xfam(0,knew) = M; xfam(1,knew) = 0.5*(svv - M*vb2);
            xfam(2,knew) = M*vn1 - (sxx - sx*sx/M + M*vo1);
            xfam(3,knew) = M*vn2 - (syy - sy*sy/M + M*vo2);
            return;
          }
          for (int k=0; k<9; ++k) pr(IPM,mem[k]) = 0.0;
          Kokkos::atomic_add(&qc(0), 1.0);
          Kokkos::atomic_add(&qc(1), M);
          Kokkos::atomic_add(&qc(2), 0.5*(svv - M*vb2));
          Kokkos::atomic_add(&qc(3), M*vn1 - (sxx - sx*sx/M + M*vo1));
          Kokkos::atomic_add(&qc(4), M*vn2 - (syy - sy*sy/M + M*vo2));
        });
        // ---- stage X is decided by the pass budget like stage G.  The kernel
        // above only claimed and built the new clouds; the cumulated leaf change of the standing
        // families (new cloud - nine members) is tested on top of the change made so far, families
        // touching a violating leaf are rejected with their members kept exactly as they are.
        DvceArray1D<int> xrej("q_xrej", xgate ? nmax : 1);
        if (xgate) {
          int nx = 0;
          Kokkos::deep_copy(nx, Kokkos::subview(nnew, 0));
          if (nx > 0) {
            Kokkos::Timer btimer;
            const Real beta = cloud_merge_budget;
            const int maxev = cloud_merge_budget_evals;
            const int xn12 = qdep.extent_int(4)*qdep.extent_int(3);
            const auto xmbsz = pmy_pack->pmb->mb_size;
            const Real xfloor = cloud_absorb_floor*mroot_floor/(transfer.root_dx1*transfer.root_dx2);
            const Real xvmin = alpha*H*((omega > 0.0) ? omega : 0.0);
            Kokkos::deep_copy(lcum, 0.0);
            if (bdirty) {
              BudgetDeposit(lt, pr, pi, npart, 1.0, lcum);
              Kokkos::parallel_for("budget_cum_x", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
              KOKKOS_LAMBDA(const int t) {
                for (int c=0; c<4; ++c) lcum(c,t) -= lent(c,t);
              });
            }
            DvceArray2D<Real> ldelx("q_ldelx", 4, nbt);
            DvceArray1D<int> lviolx("q_lviolx", nbt);
            int nev = 0;
            for (;;) {
              ++nev;
              Kokkos::deep_copy(ldelx, 0.0);
              Kokkos::parallel_for("qx_members", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
              KOKKOS_LAMBDA(const int p) {
                if (owner(p) < 0) return;
                const int k = xknew(owner(p));
                if (k < 0 || xrej(k)) return;
                const Real m = pr(IPM,p);
                const Real v1 = pr(IPVX,p), v2 = pr(IPVY,p), v3 = pr(IPVZ,p);
                lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int t, const Real w) {
                  Kokkos::atomic_add(&ldelx(0,t), -m*w); Kokkos::atomic_add(&ldelx(1,t), -m*w*v1);
                  Kokkos::atomic_add(&ldelx(2,t), -m*w*v2); Kokkos::atomic_add(&ldelx(3,t), -m*w*v3);
                });
              });
              Kokkos::parallel_for("qx_new", Kokkos::RangePolicy<>(DevExeSpace(), 0, nx),
              KOKKOS_LAMBDA(const int k) {
                if (xrej(k)) return;
                const Real m = nr(IPM,k);
                const Real v1 = nr(IPVX,k), v2 = nr(IPVY,k), v3 = nr(IPVZ,k);
                lt.ForEachCloudWeight(nr(IPX,k), nr(IPY,k), ni(PLEV,k), lt.Anchor(ni(PGID,k)), [&](const int t, const Real w) {
                  Kokkos::atomic_add(&ldelx(0,t), m*w); Kokkos::atomic_add(&ldelx(1,t), m*w*v1);
                  Kokkos::atomic_add(&ldelx(2,t), m*w*v2); Kokkos::atomic_add(&ldelx(3,t), m*w*v3);
                });
              });
              int nviol = 0;
              Kokkos::parallel_reduce("qx_viol", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
              KOKKOS_LAMBDA(const int t, int &sum) {
                int bad = 0;
                if (ldelx(0,t) != 0.0 || ldelx(1,t) != 0.0 || ldelx(2,t) != 0.0 || ldelx(3,t) != 0.0) {
                  const int tb_ = t/xn12;
                  const Real vol = xmbsz.d_view(tb_).dx1*xmbsz.d_view(tb_).dx2;
                  Real rp = 0.0;
                  const Real rm = BudgetRatio((bkappa > 0.0) ? lbeta(t) : beta, xfloor*vol, xvmin,
                      lent(0,t), lent(1,t), lent(2,t),
                      lent(3,t), lcum(0,t) + ldelx(0,t), lcum(1,t) + ldelx(1,t), lcum(2,t) + ldelx(2,t),
                      lcum(3,t) + ldelx(3,t), rp);
                  bad = (rm > 1.0 || rp > 1.0) ? 1 : 0;
                }
                lviolx(t) = bad;
                sum += bad;
              }, nviol);
              if (nviol == 0) break;
              if (nev >= maxev) {   // no convergence: the rest of stage X is rejected
                Kokkos::parallel_for("qx_rej_all", Kokkos::RangePolicy<>(DevExeSpace(), 0, nx),
                KOKKOS_LAMBDA(const int k) {xrej(k) = 1;});
                bcap += 1.0;
                break;
              }
              Kokkos::parallel_for("qx_rej_new", Kokkos::RangePolicy<>(DevExeSpace(), 0, nx),
              KOKKOS_LAMBDA(const int k) {
                if (xrej(k)) return;
                int bad = 0;
                lt.ForEachCloudWeight(nr(IPX,k), nr(IPY,k), ni(PLEV,k), lt.Anchor(ni(PGID,k)), [&](const int t, const Real w) {
                  if (w != 0.0 && lviolx(t)) bad = 1;
                });
                if (bad) xrej(k) = 1;
              });
              Kokkos::parallel_for("qx_rej_members", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
              KOKKOS_LAMBDA(const int p) {
                if (owner(p) < 0) return;
                const int k = xknew(owner(p));
                if (k < 0 || xrej(k)) return;
                int bad = 0;
                lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int t, const Real w) {
                  if (w != 0.0 && lviolx(t)) bad = 1;
                });
                if (bad) Kokkos::atomic_exchange(&xrej(k), 1);
              });
            }
            double rp_ = 0.0, rm_ = 0.0, rg_ = 0.0;
            Kokkos::parallel_reduce("qx_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
            KOKKOS_LAMBDA(const int p, double &n, double &m) {
              if (owner(p) < 0) return;
              const int k = xknew(owner(p));
              if (k < 0) return;
              if (xrej(k)) {n += 1.0; m += pr(IPM,p); return;}
              pr(IPM,p) = 0.0;
            }, rp_, rm_);
            Kokkos::parallel_reduce("qx_dead", Kokkos::RangePolicy<>(DevExeSpace(), 0, nx),
            KOKKOS_LAMBDA(const int k, double &n) {
              if (xrej(k)) {nr(IPM,k) = 0.0; n += 1.0; return;}
              Kokkos::atomic_add(&qc(0), 1.0);
              Kokkos::atomic_add(&qc(1), xfam(0,k));
              Kokkos::atomic_add(&qc(2), xfam(1,k));
              Kokkos::atomic_add(&qc(3), xfam(2,k));
              Kokkos::atomic_add(&qc(4), xfam(3,k));
            }, rg_);
            Kokkos::fence();
            tbud += btimer.seconds();
            xgb[0] += rg_; xgb[1] += rp_; xgb[2] += rm_; xgb[3] += nev;
          }
        }
      }
      // ---- protect stage X's waiting candidates.  A read-only pass over the
      // pieces that survived stage X: a centre whose 9-point stencil is complete by stage X's own
      // rule (nearest candidates for the velocity range, claims ignored) marks itself and every
      // candidate of every slot as waiting.  Waiting pieces are skipped by the G walk, so they
      // cannot become G residuals before stage X retries them.
      const bool xprot = cloud_merge_xprotect && cloud_merge_stencil;
      DvceArray1D<int> xwait("q_xwait", xprot ? npart : 1);
      if (xprot) {
        const Real dtol = cloud_merge_stencil_tol;
        Kokkos::parallel_for("q_xwait", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int c) {
          if (kp(c) < 0 || pr(IPM,c) <= 0.0) return;
          const Real xc0 = pr(IPX,c), yc0 = pr(IPY,c), mc = pr(IPM,c), ts = pr(IPTS,c);
          if (!mm.In(xc0, yc0, L, lt.Anchor(pi(PGID,c)))) return;
          const int sp = pi(PSP,c);
          const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
          const Real tolv = alpha*H/tau;
          for (int ps=0; ps<2; ++ps) {   // 0: is the stencil complete?  1: mark it
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
                  if (q == c || pr(IPM,q) <= 0.0) continue;
                  if (Kokkos::fabs(pr(IPM,q) - w) > 1.0e-6*w || pr(IPTS,q) != ts) continue;
                  const Real dx = Kokkos::fabs(transfer.MinImage1(pr(IPX,q) - tx));
                  const Real dy = Kokkos::fabs(transfer.MinImage2(pr(IPY,q) - ty));
                  if (dx > dtol*ell1 || dy > dtol*ell2) continue;
                  if (!mm.In(pr(IPX,q), pr(IPY,q), L, lt.Anchor(pi(PGID,q)))) continue;
                  if (ps == 1) {Kokkos::atomic_exchange(&xwait(q), 1); continue;}
                  const Real d = Kokkos::fmax(dx/ell1, dy/ell2);
                  if (d < bd) {bd = d; best = q;}
                }
                if (ps == 1) continue;
                if (best < 0) return;
                const Real v[3] = {pr(IPVX,best), pr(IPVY,best), pr(IPVZ,best)};
                for (int d=0; d<3; ++d) {
                  vlo[d] = Kokkos::fmin(vlo[d], v[d]); vhi[d] = Kokkos::fmax(vhi[d], v[d]);
                  if (vhi[d] - vlo[d] > tolv) return;
                }
              }
            }
            if (ps == 1) Kokkos::atomic_exchange(&xwait(c), 1);
          }
        });
        double wn = 0.0, wm = 0.0;
        Kokkos::parallel_reduce("q_xwait_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int p, double &n, double &m) {
          if (xwait(p)) {n += 1.0; m += pr(IPM,p);}
        }, wn, wm);
        xwtot[0] += wn; xwtot[1] += wm;
      }
      // ---- the leaf change made so far in this pass (release, stage X, coarser
      // levels), from the actual state: live pieces and the new clouds of stage X, minus the entry
      const int gb_n12 = qdep.extent_int(4)*qdep.extent_int(3);
      const Real gb_floor = cloud_absorb_floor*mroot_floor/(transfer.root_dx1*transfer.root_dx2);
      const Real gb_vmin = alpha*H*((omega > 0.0) ? omega : 0.0);
      if (pbud) {
        Kokkos::Timer btimer;
        int nx_now = 0;
        Kokkos::deep_copy(nx_now, Kokkos::subview(nnew, 0));
        if (nx_now > 0) bdirty = true;
        Kokkos::deep_copy(lcum, 0.0);
        if (AnyRank(bdirty)) {
          BudgetDeposit(lt, pr, pi, npart, 1.0, lcum);
          BudgetDeposit(lt, nr, ni, nx_now, 1.0, lcum);
          HaloReverseAddFlat(lcum, 4);   // the owners' leaves hold every rank's change
          Kokkos::parallel_for("budget_cum", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
          KOKKOS_LAMBDA(const int t) {
            for (int c=0; c<4; ++c) lcum(c,t) -= lent(c,t);
          });
        }
        Kokkos::fence();
        tbud += btimer.seconds();
      }
      DvceArray1D<int> krej("q_krej", nmax);   // new G cloud rejected by the budget
      DvceArray1D<int> gbrej("q_gbrej", pbud ? npart : 1);   // piece of such a group: 1
      DvceArray2D<Real> ldelg("q_ldel", 4, nbt);
      DvceArray1D<int> lviolg("q_lviol", nbt);
      double lrej[4] = {0.0, 0.0, 0.0, 0.0};
      // ---- stage G: the cell walk.  Pieces at level lev in level-L merge cells -> level-L clouds
      DvceArray1D<int> gid("q_gid", npart), gknew("q_gknew", nmax), gkey("q_gkey", nmax);
      DvceArray1D<int> ngroup("q_ngroup", 1);
      DvceArray1D<char> gres("q_gres", npart);   // a G walk left the piece in an uncommitted group
      Kokkos::deep_copy(gid, -1);
      Kokkos::deep_copy(gknew, -1);
      {
        const Real cs1 = H1, cs2 = H2;                           // the target (level-L) cells
        const int m1 = n1, m2 = n2;
        const std::int64_t nc = static_cast<std::int64_t>(m1)*m2;
        const Real smax1 = cloud_merge_smax*cs1;
        const Real smax2 = cloud_merge_smax*cs2;
        const Real qden = static_cast<Real>(1 << (2*L));
        const int q0 = 5;                                        // counter block
        const int glag = cloud_merge_g_lag;
        DvceArray1D<std::int64_t> kq("q_key", npart);
        int k0 = 0;   // the new G clouds are k0..k1-1 (after those of stage X)
        Kokkos::deep_copy(k0, Kokkos::subview(nnew, 0));
        Kokkos::parallel_for("q_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int p) {
          kq(p) = -1;
          if (kp(p) < 0 || pr(IPM,p) <= 0.0) return;          // plain level-lev pieces only
          if (xprot && xwait(p)) return;                        // waiting for stage X
          const Real x = SMRCompositeTransfer::WrapPosition(pr(IPX,p), transfer.x1min, transfer.x1max);
          const Real y = SMRCompositeTransfer::WrapPosition(pr(IPY,p), transfer.x2min, transfer.x2max);
          const Real u = (x - transfer.x1min)/cs1, v = (y - transfer.x2min)/cs2;
          const int I = SMRCompositeTransfer::WrapIndex(static_cast<int>(Kokkos::floor(u)), m1);
          const int J = SMRCompositeTransfer::WrapIndex(static_cast<int>(Kokkos::floor(v)), m2);
          if (!mm.In(x, y, L, lt.Anchor(pi(PGID,p)))) return;
          for (int dj=-glag; dj<=glag; ++dj) {   // deep zone only: the outer band is stage X's
            for (int di=-glag; di<=glag; ++di) {
              const Real xn = transfer.x1min + (SMRCompositeTransfer::WrapIndex(I + di, m1) + 0.5)*cs1;
              const Real yn = transfer.x2min + (SMRCompositeTransfer::WrapIndex(J + dj, m2) + 0.5)*cs2;
              if (!mm.In(xn, yn, L, lt.Anchor(pi(PGID,p)))) return;
            }
          }
          const int si = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(u - Kokkos::floor(u)))));
          const int sj = Kokkos::min(3, static_cast<int>(Kokkos::floor(4.0*(v - Kokkos::floor(v)))));
          const int sub = ((sj >> 1) << 3) | ((si >> 1) << 2) | ((sj & 1) << 1) | (si & 1);   // Morton
          kq(p) = ((pi(PSP,p)*nc + static_cast<std::int64_t>(J)*m1 + I) << 4) + sub;
        });
        // sparse keys (cell << 4 | sub), the walk over the cells that hold pieces
        KeyCSR qsc;
        if (cloud_merge_position_order) {
          // (diagnostic) the walk takes the pieces of a sub-cell in the order of
          // their positions (rows of y, then x, 2^20 per side of the domain; particle order
          // among equal ones) instead of the particle order, which carries the storage history
          // (the split appends children in the scan order of their parents)
          const int np_ = npart;
          DvceArray1D<std::int64_t> sk(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_poskey"), np_);
          DvceArray1D<int> qrank(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_rank"), np_);
          DvceArray1D<int> qinv(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_inv"), np_);
          const std::int64_t nn = np_, side = std::int64_t(1) << 20;
          if (static_cast<double>(side)*side*nn >= 9.0e18) {
            std::cout << "### FATAL ERROR: cloud_merge_position_order: too many particles" << std::endl;
            std::exit(EXIT_FAILURE);
          }
          const Real lx1 = transfer.x1max - transfer.x1min, lx2 = transfer.x2max - transfer.x2min;
          Kokkos::parallel_for("q_poskey", Kokkos::RangePolicy<>(DevExeSpace(), 0, np_),
          KOKKOS_LAMBDA(const int p) {
            const Real x = SMRCompositeTransfer::WrapPosition(pr(IPX,p), transfer.x1min, transfer.x1max);
            const Real y = SMRCompositeTransfer::WrapPosition(pr(IPY,p), transfer.x2min, transfer.x2max);
            std::int64_t ix = static_cast<std::int64_t>((x - transfer.x1min)/lx1*side);
            std::int64_t iy = static_cast<std::int64_t>((y - transfer.x2min)/lx2*side);
            ix = (ix < 0) ? 0 : ((ix >= side) ? side - 1 : ix);
            iy = (iy < 0) ? 0 : ((iy >= side) ? side - 1 : iy);
            sk(p) = (iy*side + ix)*nn + p;
          });
          Kokkos::sort(sk);
          Kokkos::parallel_for("q_rank", Kokkos::RangePolicy<>(DevExeSpace(), 0, np_),
          KOKKOS_LAMBDA(const int r) {
            const int p = static_cast<int>(sk(r) % nn);
            qrank(p) = r; qinv(r) = p;
          });
          BuildKeyCSR(kq, SpeciesKeyBound(nsp, nc, 4), 3, qsc, qrank, qinv);
        } else {
          BuildKeyCSR(kq, SpeciesKeyBound(nsp, nc, 4), 3, qsc);
        }
        // the distinct cells: their first entry in qsc.ukey (the 16 sub-keys of a cell are adjacent)
        DvceArray1D<int> cfirst(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_cfirst"), qsc.nu + 1);
        int ncell = 0;
        {
          auto uk = qsc.ukey; const int nu = qsc.nu;
          Kokkos::parallel_scan("q_cells", Kokkos::RangePolicy<>(DevExeSpace(), 0, nu),
          KOKKOS_LAMBDA(const int u, int &upd, const bool final) {
            const bool first = (u == 0) || ((uk(u) >> 4) != (uk(u - 1) >> 4));
            if (first && final) cfirst(upd) = u;
            if (first) ++upd;
            if (final && u == nu - 1) cfirst(upd) = nu;
          }, ncell);
        }
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
            nkind(knew) = 1;
            if (exact_ts) nr(IPTS,knew) = gts[j];   // the group's common ts
            acc(6,knew) = xcc; acc(7,knew) = ycc;
            Kokkos::atomic_add(&qc(q0), 1.0);
            Kokkos::atomic_add(&qc(q0 + 1), static_cast<double>(gn[j]));
            Kokkos::atomic_add(&qc(q0 + 2), static_cast<double>(gM[j]));
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
                if (nopen == K) {Kokkos::atomic_add(&qc(8), 1.0); continue;}   // deferred
                j = nopen++;
                gg[j] = Kokkos::atomic_fetch_add(&ngroup(0), 1);
                gkey(gg[j]) = kcell;
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
            if (gknew(gg[o]) < 0) Kokkos::atomic_add(&qc(9), static_cast<double>(gn[o]));
          }
        });
        // accumulate the members of committed groups into their new clouds; the pieces are used up
        Kokkos::parallel_for("q_accumulate", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int p) {
          if (kq(p) < 0) return;
          const int g = gid(p);
          if (g < 0) return;
          const int k = gknew(g);
          if (k < 0) {gres(p) = 1; return;}
          const Real m = pr(IPM,p);
          const Real dx = transfer.MinImage1(pr(IPX,p) - acc(6,k)), dy = transfer.MinImage2(pr(IPY,p) - acc(7,k));
          Kokkos::atomic_add(&acc(0,k), static_cast<double>(m));
          Kokkos::atomic_add(&acc(1,k), static_cast<double>(m*dx));
          Kokkos::atomic_add(&acc(2,k), static_cast<double>(m*dy));
          Kokkos::atomic_add(&acc(3,k), static_cast<double>(m*dx*dx));
          Kokkos::atomic_add(&acc(4,k), static_cast<double>(m*dy*dy));
          Kokkos::atomic_add(&acc(5,k), static_cast<double>(m*(pr(IPVX,p)*pr(IPVX,p) +
                             pr(IPVY,p)*pr(IPVY,p) + pr(IPVZ,p)*pr(IPVZ,p))));
          for (int q=0; q<nrdata; ++q) {
            if (q == IPM || q == IPX || q == IPY || q == IPX1 || q == IPY1) continue;
            if (exact_ts && q == IPTS) continue;   // set at the commit
            Kokkos::atomic_add(&nr(q,k), m*pr(q,p));
          }
          if (!pbud) pr(IPM,p) = 0.0;
        });
        // the state of the new G clouds: means and centroid (the dK, dS counters follow below,
        // for the clouds that are kept)
        int k1 = k0;
        Kokkos::deep_copy(k1, Kokkos::subview(nnew, 0));
        if (k1 > k0) {
          Kokkos::parallel_for("q_finalize_state", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
          KOKKOS_LAMBDA(const int k) {
            const double M = acc(0,k);
            for (int q=0; q<nrdata; ++q) {
              if (q == IPM || q == IPX || q == IPY || q == IPX1 || q == IPY1) continue;
              if (exact_ts && q == IPTS) continue;   // set at the commit
              nr(q,k) /= M;
            }
            const Real px = SMRCompositeTransfer::WrapPosition(acc(6,k) + acc(1,k)/M, transfer.x1min, transfer.x1max);
            const Real py = SMRCompositeTransfer::WrapPosition(acc(7,k) + acc(2,k)/M, transfer.x2min, transfer.x2max);
            nr(IPM,k) = M;
            nr(IPX,k) = px; nr(IPX1,k) = px; nr(IPY,k) = py; nr(IPY1,k) = py;
            ni(PGID,k) = gids + NewCloudBlock(lt, px, py, lt.Anchor(ni(PGID,k)), ncell12);
          });
        }
        // ---- accept or reject the proposed groups.  ldelg: cumulated leaf change of
        // the proposals still standing (new cloud - members); a leaf touched by them violates when
        // the change so far plus ldelg exceeds the budget; every group whose new cloud or members
        // touch a violating leaf is rejected; repeat.  The decision depends on the proposals and
        // the leaf fields only, not on any order of commits.
        // several ranks: proposals on any rank are evaluated on all of them (their
        // leaves may be touched by a neighbour's proposals)
        if (pbud && AnyRank(k1 > k0)) {
          Kokkos::Timer btimer;
          const Real beta = cloud_merge_budget;
          const int maxev = cloud_merge_budget_evals;
          int nev = 0;
          for (;;) {
            ++nev;
            Kokkos::deep_copy(ldelg, 0.0);
            Kokkos::parallel_for("qb_members", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
            KOKKOS_LAMBDA(const int p) {
              if (kq(p) < 0 || gid(p) < 0) return;
              const int k = gknew(gid(p));
              if (k < 0 || krej(k)) return;
              const Real m = pr(IPM,p);
              const Real v1 = pr(IPVX,p), v2 = pr(IPVY,p), v3 = pr(IPVZ,p);
              lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int t, const Real w) {
                Kokkos::atomic_add(&ldelg(0,t), -m*w); Kokkos::atomic_add(&ldelg(1,t), -m*w*v1);
                Kokkos::atomic_add(&ldelg(2,t), -m*w*v2); Kokkos::atomic_add(&ldelg(3,t), -m*w*v3);
              });
            });
            Kokkos::parallel_for("qb_new", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
            KOKKOS_LAMBDA(const int k) {
              if (krej(k)) return;
              const Real m = nr(IPM,k);
              const Real v1 = nr(IPVX,k), v2 = nr(IPVY,k), v3 = nr(IPVZ,k);
              lt.ForEachCloudWeight(nr(IPX,k), nr(IPY,k), ni(PLEV,k), lt.Anchor(ni(PGID,k)), [&](const int t, const Real w) {
                Kokkos::atomic_add(&ldelg(0,t), m*w); Kokkos::atomic_add(&ldelg(1,t), m*w*v1);
                Kokkos::atomic_add(&ldelg(2,t), m*w*v2); Kokkos::atomic_add(&ldelg(3,t), m*w*v3);
              });
            });
            HaloReverseAddFlat(ldelg, 4);   // the owners judge their leaves
            int nviol = 0;
            Kokkos::parallel_reduce("qb_viol", Kokkos::RangePolicy<>(DevExeSpace(), 0, nown),
            KOKKOS_LAMBDA(const int t, int &sum) {
              int bad = 0;
              if (ldelg(0,t) != 0.0 || ldelg(1,t) != 0.0 || ldelg(2,t) != 0.0 || ldelg(3,t) != 0.0) {
                const int tb_ = t/gb_n12;
                const Real vol = lt.box(tb_,2)*lt.box(tb_,3);
                Real rp = 0.0;
                const Real rm = BudgetRatio((bkappa > 0.0) ? lbeta(t) : beta, gb_floor*vol, gb_vmin,
                    lent(0,t), lent(1,t), lent(2,t),
                    lent(3,t), lcum(0,t) + ldelg(0,t), lcum(1,t) + ldelg(1,t), lcum(2,t) + ldelg(2,t),
                    lcum(3,t) + ldelg(3,t), rp);
                bad = (rm > 1.0 || rp > 1.0) ? 1 : 0;
              }
              lviolg(t) = bad;
              sum += bad;
            }, nviol);
            HaloForwardFlags(lviolg);                       // the owners' verdicts on the halo
            nviol = static_cast<int>(SumRanks(nviol));      // the same number of rounds everywhere
            if (nviol == 0) break;
            if (nev >= maxev) {   // no convergence: the rest of stage G is rejected
              Kokkos::parallel_for("qb_rej_all", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
              KOKKOS_LAMBDA(const int k) {krej(k) = 1;});
              Kokkos::deep_copy(ldelg, 0.0);
              bcap += 1.0;
              break;
            }
            Kokkos::parallel_for("qb_rej_new", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
            KOKKOS_LAMBDA(const int k) {
              if (krej(k)) return;
              int bad = 0;
              lt.ForEachCloudWeight(nr(IPX,k), nr(IPY,k), ni(PLEV,k), lt.Anchor(ni(PGID,k)), [&](const int t, const Real w) {
                if (w != 0.0 && lviolg(t)) bad = 1;
              });
              if (bad) krej(k) = 1;
            });
            Kokkos::parallel_for("qb_rej_members", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
            KOKKOS_LAMBDA(const int p) {
              if (kq(p) < 0 || gid(p) < 0) return;
              const int k = gknew(gid(p));
              if (k < 0 || krej(k)) return;
              int bad = 0;
              lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pi(PLEV,p), lt.Anchor(pi(PGID,p)), [&](const int t, const Real w) {
                if (w != 0.0 && lviolg(t)) bad = 1;
              });
              if (bad) Kokkos::atomic_exchange(&krej(k), 1);
            });
          }
          // apply: the members of accepted groups are used up; rejected groups keep their members
          // exactly as they are and their new cloud is dropped
          double rp_ = 0.0, rm_ = 0.0, rg_ = 0.0;
          Kokkos::parallel_reduce("qb_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
          KOKKOS_LAMBDA(const int p, double &n, double &m) {
            if (kq(p) < 0 || gid(p) < 0) return;
            const int k = gknew(gid(p));
            if (k < 0) return;
            if (krej(k)) {gbrej(p) = 1; n += 1.0; m += pr(IPM,p); return;}
            pr(IPM,p) = 0.0;
          }, rp_, rm_);
          Kokkos::parallel_reduce("qb_dead", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
          KOKKOS_LAMBDA(const int k, double &n) {
            if (krej(k)) {nr(IPM,k) = 0.0; n += 1.0;}
          }, rg_);
          Kokkos::parallel_for("budget_advance", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
          KOKKOS_LAMBDA(const int t) {
            for (int c=0; c<4; ++c) lcum(c,t) += ldelg(c,t);
          });
          Kokkos::fence();
          tbud += btimer.seconds();
          lrej[0] += rg_; lrej[1] += rp_; lrej[2] += rm_; lrej[3] += nev;
          if (rg_ < static_cast<double>(k1 - k0)) bdirty = true;
        }
      }
      int ncommit = 0;
      Kokkos::deep_copy(ncommit, Kokkos::subview(nnew, 0));
      // the dK and dS counters of the G clouds that are kept (their state was finalized above)
      if (ncommit > 0) {
        const Real vG1 = H1*H1/6.0, vG2 = H2*H2/6.0, vF1 = ell1*ell1/6.0, vF2 = ell2*ell2/6.0;
        Kokkos::parallel_for("q_finalize", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
        KOKKOS_LAMBDA(const int k) {
          if (nkind(k) == 0 || nr(IPM,k) <= 0.0) return;
          const double M = acc(0,k);
          const double vb2 = nr(IPVX,k)*nr(IPVX,k) + nr(IPVY,k)*nr(IPVY,k) + nr(IPVZ,k)*nr(IPVZ,k);
          const double vn1 = vG1, vn2 = vG2;
          Kokkos::atomic_add(&qc(10), 0.5*(acc(5,k) - M*vb2));
          Kokkos::atomic_add(&qc(11), M*vn1 - (acc(3,k) - acc(1,k)*acc(1,k)/M + M*vF1));
          Kokkos::atomic_add(&qc(12), M*vn2 - (acc(4,k) - acc(2,k)*acc(2,k)/M + M*vF2));
        });
      }
      auto qch = Kokkos::create_mirror_view_and_copy(HostMemSpace(), qc);
      for (int c=0; c<24; ++c) qtot[c] += qch(c);
      // the walks counted the proposals: take the budget-rejected groups out
      for (int c=0; c<3; ++c) qtot[5 + c] -= lrej[c];
      for (int c=0; c<4; ++c) gcb[0][c] += lrej[c];
      q_events += static_cast<unsigned long long>(qch(0) + qch(5) - lrej[0]);
      q_in += static_cast<unsigned long long>(9.0*qch(0) + qch(6) - lrej[1]);
      // compaction: drop the used-up pieces, append the new clouds
      DvceArray1D<int> koff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_koff"), npart);
      int nkeep = 0;
      Kokkos::parallel_scan("q_kscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
        if (final) koff(p) = upd;
        upd += (pr(IPM,p) > 0.0) ? 1 : 0;
      }, nkeep);
      DvceArray1D<int> noff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q_noff"), ncommit > 0 ? ncommit : 1);
      int nlive = 0;   // new clouds that are kept (budget-rejected ones have mass 0)
      if (ncommit > 0) {
        Kokkos::parallel_scan("q_nscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
        KOKKOS_LAMBDA(const int k, int &upd, const bool final) {
          if (final) noff(k) = upd;
          upd += (nr(IPM,k) > 0.0) ? 1 : 0;
        }, nlive);
      }
      if (nkeep == npart && nlive == 0) continue;
      bdirty = true;
      const int ntot = nkeep + nlive;
      DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, ntot);
      DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, ntot);
      Kokkos::parallel_for("q_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        if (pr(IPM,p) <= 0.0) return;
        const int s = koff(p);
        for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
        for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
      });
      Kokkos::parallel_for("q_append", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
      KOKKOS_LAMBDA(const int k) {
        if (nr(IPM,k) <= 0.0) return;
        const int s = nkeep + noff(k);
        for (int c=0; c<nrdata; ++c) prn(c,s) = nr(c,k);
        for (int c=0; c<nidata; ++c) pin(c,s) = ni(c,k);
      });
      ppar->prtcl_rdata = prn;
      ppar->prtcl_idata = pin;
      ppar->nprtcl_thispack = ntot;
      continue;
    }
    auto tent = [=] KOKKOS_FUNCTION (const Real dx, const Real dy) {
      return Kokkos::fmax(0.0, 1.0 - Kokkos::fabs(Snap(dx, ell1))/H1)*
             Kokkos::fmax(0.0, 1.0 - Kokkos::fabs(Snap(dy, ell2))/H2);
    };
    // centres: two rounds of local maxima among unblocked zone pieces
    DvceArray1D<int> state(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_state"), npart);   // 0 none, 1 candidate, 2 centre, 3 blocked
    const Real b1 = H1 - 0.5*ell1, b2 = H2 - 0.5*ell2;
    Kokkos::parallel_for("merge_cand", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      state(p) = 0;
      if (kp(p) < 0 || !mm.In(pr(IPX,p), pr(IPY,p), L, lt.Anchor(pi(PGID,p)))) return;
      const int sp = pi(PSP,p); const std::int64_t cell = kp(p) - sp*ncl;
      const int I = static_cast<int>(cell % n1), J = static_cast<int>(cell/n1);
      for (int dj=-1; dj<=1; ++dj) {
        for (int di=-1; di<=1; ++di) {
          const std::int64_t key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                          SMRCompositeTransfer::WrapIndex(I + di, n1);
          for (int k=scc.First(key), ke=scc.Last(key); k<ke; ++k) {
            const int q = scc.list(k);
            if (Kokkos::fabs(transfer.MinImage1(pr(IPX,p) - pr(IPX,q))) < b1 &&
                Kokkos::fabs(transfer.MinImage2(pr(IPY,p) - pr(IPY,q))) < b2) {state(p) = 3; return;}
          }
        }
      }
      state(p) = 1;
    });
    for (int round=0; round<2; ++round) {
      DvceArray1D<int> win(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_win"), npart);
      Kokkos::parallel_for("merge_mis", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        win(p) = 0;
        if (state(p) != 1) return;
        const int sp = pi(PSP,p); const std::int64_t cell = kp(p) - sp*ncl;
      const int I = static_cast<int>(cell % n1), J = static_cast<int>(cell/n1);
        const Real mp = pr(IPM,p), xp = pr(IPX,p), yp = pr(IPY,p);
        for (int dj=-1; dj<=1; ++dj) {
          for (int di=-1; di<=1; ++di) {
            const std::int64_t key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                            SMRCompositeTransfer::WrapIndex(I + di, n1);
            for (int k=spc.First(key), ke=spc.Last(key); k<ke; ++k) {
              const int q = spc.list(k);
              if (q == p || (state(q) != 1 && state(q) != 2)) continue;
              if (Kokkos::fabs(transfer.MinImage1(xp - pr(IPX,q))) >= b1 ||
                  Kokkos::fabs(transfer.MinImage2(yp - pr(IPY,q))) >= b2) continue;
              if (state(q) == 2) return;   // within the radius of an earlier centre
              const Real mq = pr(IPM,q), xq = pr(IPX,q), yq = pr(IPY,q);
              const bool before = (mq > mp) || (mq == mp && (xq < xp || (xq == xp && yq < yp)));
              if (before) return;
            }
          }
        }
        win(p) = 1;
      });
      Kokkos::parallel_for("merge_mis_set", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {if (win(p)) state(p) = 2;});
    }
    int ncentre = 0;
    Kokkos::parallel_reduce("merge_ncentre", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &sum) {if (state(p) == 2) sum += 1;}, Kokkos::Sum<int>(ncentre));
    if (ncentre == 0) continue;
    // rank of a centre among the centres of its own cell (two centres may share a cell when
    // they are more than the blocking radius apart in one direction): sub-rounds by rank
    DvceArray1D<int> rank(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_rank"), npart);
    int maxrank = 0;
    Kokkos::parallel_reduce("merge_rank", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &mx) {
      rank(p) = -1;
      if (state(p) != 2) return;
      const Real mp = pr(IPM,p), xp = pr(IPX,p), yp = pr(IPY,p);
      int r = 0;
      for (int k=spc.First(kp(p)), ke=spc.Last(kp(p)); k<ke; ++k) {
        const int q = spc.list(k);
        if (q == p || state(q) != 2) continue;
        const Real mq = pr(IPM,q), xq = pr(IPX,q), yq = pr(IPY,q);
        if ((mq > mp) || (mq == mp && (xq < xp || (xq == xp && yq < yp)))) ++r;
      }
      rank(p) = r;
      if (r > mx) mx = r;
    }, Kokkos::Max<int>(maxrank));
    // new level-L clouds of this pass, and their per-cell linked lists
    DvceArray2D<Real> nr(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_nr"), nrdata, ncentre);
    DvceArray2D<int> ni(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_ni"), nidata, ncentre);
    CellLists nhead;   // hash of the cells, no level-L table
    nhead.Allocate(ncentre, 1);
    DvceArray1D<int> nnext("merge_nnext", ncentre);
    DvceArray1D<int> nnew("merge_nnew", 1);
    for (int round=0; round<16*(maxrank + 1); ++round) {
      const int col = round % 16, sub = round/16;
      Kokkos::parallel_for("merge_commit", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int c) {
        if (state(c) != 2 || rank(c) != sub) return;
        const int sp = pi(PSP,c); const std::int64_t cell = kp(c) - sp*ncl;
      const int I = static_cast<int>(cell % n1), J = static_cast<int>(cell/n1);
        if ((I % 4) + 4*(J % 4) != col) return;
        if (pr(IPM,c) <= 0.0) return;
        const Real mref = mrootsp(sp)/static_cast<Real>(1 << (2*L)), mnode = 0.25*mref;
        const Real xc0 = pr(IPX,c), yc0 = pr(IPY,c);
        // neighbours with a positive tent weight
        int nb[kNbMax]; int nnb = 0; bool ready = true, over = false; int why = 6;
        for (int dj=-1; dj<=1 && ready; ++dj) {
          for (int di=-1; di<=1 && ready; ++di) {
            const std::int64_t key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                            SMRCompositeTransfer::WrapIndex(I + di, n1);
            for (int k=spc.First(key), ke=spc.Last(key); k<ke; ++k) {
              const int t = spc.list(k);
              if (t == c || pr(IPM,t) <= 0.0) continue;
              // with exact_ts the new cloud takes the centre's ts: only pieces with that same ts
              // (bitwise) join (G and X compare ts the same way)
              if (exact_ts && pr(IPTS,t) != pr(IPTS,c)) continue;
              const Real dx = transfer.MinImage1(pr(IPX,t) - xc0);
              const Real dy = transfer.MinImage2(pr(IPY,t) - yc0);
              if (Kokkos::fabs(dx) >= H1*(1.0 - 1.0e-12) || Kokkos::fabs(dy) >= H2*(1.0 - 1.0e-12)) continue;
              if (tent(dx, dy) <= 0.0) continue;
              if (!mm.In(pr(IPX,t), pr(IPY,t), L, lt.Anchor(pi(PGID,t)))) {ready = false; break;}
              if (nnb == kNbMax) {over = true; ready = false; why = 8; break;}
              nb[nnb++] = t;
            }
          }
        }
        // lobe wait: an unreleased lobed cloud whose released pieces could fall in this
        // centre's tent makes the centre wait: reach H + ell in both directions
        // (cloud_merge_lobe_wait = 1), or H plus ell only along the cloud's lobes (= 2)
        if (ready && lobe_wait) {
          // the reach (up to H + ell) can span two level-L cells: search +-2 cells
          for (int dj=-2; dj<=2 && ready; ++dj) {
            for (int di=-2; di<=2 && ready; ++di) {
              const std::int64_t key = sp*ncl + SMRCompositeTransfer::WrapIndex(J + dj, n2)*n1 +
                              SMRCompositeTransfer::WrapIndex(I + di, n1);
              for (int k=slc.First(key), ke=slc.Last(key); k<ke; ++k) {
                const int t = slc.list(k);
                const int ct = pi(PLEV,t);
                const Real r1 = H1 + (lobe_wait_dir ? (SMRCompositeTransfer::CloudLobe1(ct) ? ell1 : 0.0) : ell1);
                const Real r2 = H2 + (lobe_wait_dir ? (SMRCompositeTransfer::CloudLobe2(ct) ? ell2 : 0.0) : ell2);
                if (Kokkos::fabs(transfer.MinImage1(pr(IPX,t) - xc0)) < r1*(1.0 - 1.0e-12) &&
                    Kokkos::fabs(transfer.MinImage2(pr(IPY,t) - yc0)) < r2*(1.0 - 1.0e-12)) {ready = false; why = 7; break;}
              }
            }
          }
        }
        if (!ready) {Kokkos::atomic_increment(&icnt(3)); Kokkos::atomic_increment(&icnt(why)); return;}
        (void) over;
        // nodes: key (r1, r2) for regular offsets (within 0.1 ell), an own node otherwise
        int nkey1[kNodeMax], nkey2[kNodeMax], nq0[kNodeMax], nnode = 0;
        int pnode[kNbMax + 1], pidx[kNbMax + 1]; int npc = 0;
        auto add_piece = [&](const int t) -> bool {
          const Real dx = transfer.MinImage1(pr(IPX,t) - xc0);
          const Real dy = transfer.MinImage2(pr(IPY,t) - yc0);
          const int r1 = static_cast<int>(Kokkos::round(dx/ell1));
          const int r2 = static_cast<int>(Kokkos::round(dy/ell2));
          const bool reg = Kokkos::fabs(dx - r1*ell1) < 0.1*ell1 && Kokkos::fabs(dy - r2*ell2) < 0.1*ell2;
          int k = -1;
          if (reg) {
            for (int n=0; n<nnode; ++n) if (nkey1[n] == r1 && nkey2[n] == r2 && nq0[n] >= 0) {k = n; break;}
          }
          if (k < 0) {
            if (nnode == kNodeMax) return false;
            k = nnode++;
            nkey1[k] = reg ? r1 : 1000000 + t; nkey2[k] = reg ? r2 : 0; nq0[k] = t;
          }
          pnode[npc] = k; pidx[npc] = t; ++npc;
          return true;
        };
        if (!add_piece(c)) {Kokkos::atomic_increment(&icnt(3)); Kokkos::atomic_increment(&icnt(8)); return;}
        for (int k=0; k<nnb; ++k) if (!add_piece(nb[k])) {Kokkos::atomic_increment(&icnt(3)); Kokkos::atomic_increment(&icnt(8)); return;}
        // node sums, implicit content, shares and effective positions
        Real E[kNodeMax], Ex[kNodeMax], Ey[kNodeMax], Im[kNodeMax], Ix[kNodeMax], Iy[kNodeMax];
        Real share[kNodeMax], ex[kNodeMax], ey[kNodeMax], ox0[kNodeMax], oy0[kNodeMax];
        for (int n=0; n<nnode; ++n) {E[n] = Ex[n] = Ey[n] = 0.0;}
        for (int k=0; k<npc; ++k) {
          const int t = pidx[k], n = pnode[k], q0 = nq0[n];
          E[n] += pr(IPM,t);
          Ex[n] += pr(IPM,t)*transfer.MinImage1(pr(IPX,t) - pr(IPX,q0));
          Ey[n] += pr(IPM,t)*transfer.MinImage2(pr(IPY,t) - pr(IPY,q0));
        }
        for (int n=0; n<nnode; ++n) {
          const int q0 = nq0[n];
          ox0[n] = transfer.MinImage1(pr(IPX,q0) - xc0);
          oy0[n] = transfer.MinImage2(pr(IPY,q0) - yc0);
          Im[n] = Ix[n] = Iy[n] = 0.0;
          const Real xn = pr(IPX,q0), yn = pr(IPY,q0);
          const std::int64_t ncell = mm.Cell(xn, yn, L);
          const int In_ = static_cast<int>(ncell % n1), Jn = static_cast<int>(ncell/n1);
          for (int dj=-1; dj<=1; ++dj) {
            for (int di=-1; di<=1; ++di) {
              const std::int64_t key = sp*ncl + SMRCompositeTransfer::WrapIndex(Jn + dj, n2)*n1 +
                              SMRCompositeTransfer::WrapIndex(In_ + di, n1);
              for (int k=scc.First(key), ke=scc.Last(key); k<ke; ++k) {
                const int q = scc.list(k);
                const Real dx = transfer.MinImage1(xn - pr(IPX,q)), dy = transfer.MinImage2(yn - pr(IPY,q));
                const Real w = 0.25*pr(IPM,q)*tent(dx, dy);
                if (w <= 0.0) continue;
                Im[n] += w; Ix[n] += w*(Snap(dx, ell1) - dx); Iy[n] += w*(Snap(dy, ell2) - dy);
              }
              for (int q=nhead.Head(key); q>=0; q=nnext(q)) {
                const Real dx = transfer.MinImage1(xn - nr(IPX,q)), dy = transfer.MinImage2(yn - nr(IPY,q));
                const Real w = 0.25*nr(IPM,q)*tent(dx, dy);
                if (w <= 0.0) continue;
                Im[n] += w; Ix[n] += w*(Snap(dx, ell1) - dx); Iy[n] += w*(Snap(dy, ell2) - dy);
              }
            }
          }
          const Real w = tent(ox0[n], oy0[n]);
          share[n] = (E[n] > 0.0) ? Kokkos::fmin(E[n], w*(E[n] + Im[n])) : 0.0;
          if (E[n] > 0.0 && E[n] - share[n] < 1.0e-6*mnode) share[n] = E[n];
          if (E[n] > 0.0) {
            if (nc) {ex[n] = (Ix[n] + Ex[n])/(Im[n] + E[n]); ey[n] = (Iy[n] + Ey[n])/(Im[n] + E[n]);}
            else {ex[n] = Ex[n]/E[n]; ey[n] = Ey[n]/E[n];}
          }
        }
        // contributions and the velocity tolerance (drop the worst until all fit)
        Real cm[kNbMax + 1], cox[kNbMax + 1], coy[kNbMax + 1]; bool act[kNbMax + 1];
        int ncon = 0;
        for (int k=0; k<npc; ++k) {
          const int t = pidx[k], n = pnode[k];
          const Real s = (E[n] > 0.0) ? share[n]*pr(IPM,t)/E[n] : 0.0;
          cm[k] = s; cox[k] = ox0[n] + ex[n]; coy[k] = oy0[n] + ey[n]; act[k] = s > 0.0;
          if (act[k]) ++ncon;
        }
        Real M = 0.0, xc = 0.0, yc = 0.0, vm[3] = {0.0, 0.0, 0.0}, G[3][2];
        bool ok = false;
        while (ncon > 0) {
          M = 0.0; Real mx = 0.0, my = 0.0, mts = 0.0;
          vm[0] = vm[1] = vm[2] = 0.0;
          for (int k=0; k<npc; ++k) {
            if (!act[k]) continue;
            const int s = pidx[k]; const Real m = cm[k];
            M += m; mx += m*cox[k]; my += m*coy[k];
            vm[0] += m*pr(IPVX,s); vm[1] += m*pr(IPVY,s); vm[2] += m*pr(IPVZ,s);
            mts += m*pr(IPTS,s);
          }
          xc = mx/M; yc = my/M;
          for (int q=0; q<3; ++q) vm[q] /= M;
          const Real ts = mts/M;
          const Real tau = (omega > 0.0) ? Kokkos::fmin(ts, 1.0/omega) : ts;
          const Real tol = alpha*H/tau;
          for (int q=0; q<3; ++q) {G[q][0] = 0.0; G[q][1] = 0.0;}
          if (ncon >= 5) {
            Real cxx = 0.0, cxy = 0.0, cyy = 0.0, bx[3] = {0.0, 0.0, 0.0}, by[3] = {0.0, 0.0, 0.0};
            for (int k=0; k<npc; ++k) {
              if (!act[k]) continue;
              const int s = pidx[k]; const Real m = cm[k];
              const Real dx = cox[k] - xc, dy = coy[k] - yc;
              const Real v[3] = {pr(IPVX,s) - vm[0], pr(IPVY,s) - vm[1], pr(IPVZ,s) - vm[2]};
              cxx += m*dx*dx; cxy += m*dx*dy; cyy += m*dy*dy;
              for (int q=0; q<3; ++q) {bx[q] += m*dx*v[q]; by[q] += m*dy*v[q];}
            }
            const Real det = cxx*cyy - cxy*cxy;
            if (det > 1.0e-6*M*M*H*H*H*H) {
              for (int q=0; q<3; ++q) {
                G[q][0] = (cyy*bx[q] - cxy*by[q])/det;
                G[q][1] = (cxx*by[q] - cxy*bx[q])/det;
              }
            }
          }
          int worst = -1; Real rmax = 0.0;
          for (int k=0; k<npc; ++k) {
            if (!act[k]) continue;
            const int s = pidx[k];
            const Real dx = cox[k] - xc, dy = coy[k] - yc;
            const Real v[3] = {pr(IPVX,s), pr(IPVY,s), pr(IPVZ,s)};
            Real r2 = 0.0;
            for (int q=0; q<3; ++q) {const Real d = v[q] - vm[q] - G[q][0]*dx - G[q][1]*dy; r2 += d*d;}
            if (r2 > rmax) {rmax = r2; worst = k;}
          }
          if (rmax <= tol*tol) {ok = true; break;}
          if (pidx[worst] == c) break;
          act[worst] = false; --ncon;
        }
        if (!ok || M < fmin*mref) {
          Kokkos::atomic_increment(&icnt(2)); Kokkos::atomic_increment(&icnt(ok ? 10 : 9)); return;
        }
        // commit
        const int knew = Kokkos::atomic_fetch_add(&nnew(0), 1);
        for (int q=0; q<nrdata; ++q) nr(q,knew) = 0.0;
        Real dkg = 0.0, dkr = 0.0; int nin = 0;
        for (int k=0; k<npc; ++k) {
          if (!act[k]) continue;
          const int s = pidx[k]; const Real m = cm[k];
          for (int q=0; q<nrdata; ++q) nr(q,knew) += m*pr(q,s);
          const Real dx = cox[k] - xc, dy = coy[k] - yc;
          const Real v[3] = {pr(IPVX,s) - vm[0], pr(IPVY,s) - vm[1], pr(IPVZ,s) - vm[2]};
          for (int q=0; q<3; ++q) {
            const Real g = G[q][0]*dx + G[q][1]*dy;
            dkg += 0.5*m*g*g; dkr += 0.5*m*(v[q] - g)*(v[q] - g);
          }
          ++nin;
        }
        for (int q=0; q<nrdata; ++q) nr(q,knew) /= M;
        const Real px = SMRCompositeTransfer::WrapPosition(xc0 + xc, transfer.x1min, transfer.x1max);
        const Real py = SMRCompositeTransfer::WrapPosition(yc0 + yc, transfer.x2min, transfer.x2max);
        nr(IPM,knew) = M;
        nr(IPX,knew) = px; nr(IPX1,knew) = px; nr(IPY,knew) = py; nr(IPY1,knew) = py;
        nr(IPVX,knew) = vm[0]; nr(IPVY,knew) = vm[1]; nr(IPVZ,knew) = vm[2];
        if (exact_ts) nr(IPTS,knew) = pr(IPTS,c);   // the common ts of the pieces
        for (int q=0; q<nidata; ++q) ni(q,knew) = pi(q,c);
        ni(PLEV,knew) = L;
        {
          ni(PGID,knew) = gids + NewCloudBlock(lt, px, py, lt.Anchor(ni(PGID,knew)), ncell12);
        }
        const std::int64_t key = sp*ncl + mm.Cell(px, py, L);
        nnext(knew) = nhead.Push(key, knew);
        // take the shares; the node convention shifts what is left of each node
        Real taken[kNodeMax];
        for (int n=0; n<nnode; ++n) taken[n] = 0.0;
        for (int k=0; k<npc; ++k) {
          if (!act[k]) continue;
          const int s = pidx[k];
          taken[pnode[k]] += cm[k];
          pr(IPM,s) -= cm[k];
          if (s == c || pr(IPM,s) <= 0.0) pr(IPM,s) = 0.0;
        }
        long long nsh = 0, nsk = 0;
        if (nc) {
          for (int n=0; n<nnode; ++n) {
            const Real rem = E[n] - taken[n];
            if (rem <= 1.0e-6*mnode || taken[n] <= 0.0) continue;
            const int q0 = nq0[n];
            const Real dx = transfer.MinImage1(pr(IPX,q0) - px), dy = transfer.MinImage2(pr(IPY,q0) - py);
            const Real In = 0.25*M*tent(dx, dy);
            const Real xr = (ex[n]*(Im[n] + In + rem) - Ix[n] - In*(Snap(dx, ell1) - dx))/rem;
            const Real yr = (ey[n]*(Im[n] + In + rem) - Iy[n] - In*(Snap(dy, ell2) - dy))/rem;
            const Real sx = xr - Ex[n]/E[n], sy = yr - Ey[n]/E[n];
            if (Kokkos::fabs(sx) >= 0.1*ell1 || Kokkos::fabs(sy) >= 0.1*ell2) {++nsk; continue;}
            for (int k=0; k<npc; ++k) {
              if (pnode[k] != n) continue;
              const int t = pidx[k];
              if (pr(IPM,t) <= 0.0) continue;
              const Real nx = SMRCompositeTransfer::WrapPosition(pr(IPX,t) + sx, transfer.x1min, transfer.x1max);
              const Real ny = SMRCompositeTransfer::WrapPosition(pr(IPY,t) + sy, transfer.x2min, transfer.x2max);
              pr(IPX,t) = nx; pr(IPX1,t) = nx; pr(IPY,t) = ny; pr(IPY1,t) = ny;
              pi(PGID,t) = gids + NewCloudBlock(lt, nx, ny, lt.Anchor(pi(PGID,t)), ncell12);
            }
            ++nsh;
          }
        }
        Kokkos::atomic_increment(&icnt(0));
        Kokkos::atomic_add(&icnt(1), static_cast<long long>(nin));
        Kokkos::atomic_add(&icnt(4), nsh);
        Kokkos::atomic_add(&icnt(5), nsk);
        Kokkos::atomic_add(&dk(0), dkg);
        Kokkos::atomic_add(&dk(1), dkr);
      });
    }
    // compaction: drop the used-up pieces, append the new level-L clouds
    int ncommit = 0;
    Kokkos::deep_copy(ncommit, Kokkos::subview(nnew, 0));
    DvceArray1D<int> koff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge_koff"), npart);
    int nkeep = 0;
    Kokkos::parallel_scan("merge_kscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
      if (final) koff(p) = upd;
      upd += (pr(IPM,p) > 0.0) ? 1 : 0;
    }, nkeep);
    const int ntot = nkeep + ncommit;
    DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"), nrdata, ntot);
    DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"), nidata, ntot);
    Kokkos::parallel_for("merge_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (pr(IPM,p) <= 0.0) return;
      const int s = koff(p);
      for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
      for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
    });
    Kokkos::parallel_for("merge_append", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
    KOKKOS_LAMBDA(const int k) {
      for (int c=0; c<nrdata; ++c) prn(c,nkeep + k) = nr(c,k);
      for (int c=0; c<nidata; ++c) pin(c,nkeep + k) = ni(c,k);
    });
    ppar->prtcl_rdata = prn;
    ppar->prtcl_idata = pin;
    ppar->nprtcl_thispack = ntot;
  }

  const int nfinal = ppar->nprtcl_thispack;
  Kokkos::fence();
  pmesh->nprtcl_thisrank = nfinal;
  pmesh->nprtcl_eachrank[global_variable::my_rank] = nfinal;
  pmesh->nprtcl_total = nfinal;
  auto ic = Kokkos::create_mirror_view_and_copy(HostMemSpace(), icnt);
  auto dkh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), dk);
  const unsigned long long nevents = ic(0) + q_events, nin = ic(1) + q_in, nreject = ic(2);
  ++cloud_merge_pass;
  cloud_coarsen_events += nevents;
  if (phst_track) phst_g_events += nevents;
  cloud_coarsen_in += nin;
  cloud_coarsen_reject += nreject;
  cloud_far_release += nrelease;
  cloud_coarsen_dk_grad += dkh(0);
  cloud_coarsen_dk_res += dkh(1);
  (void) sh1; (void) sh2;
  return static_cast<int>(nevents);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::SortParticles
//! \brief Timing diagnostic: reorders the
//! particle arrays by (block, cell of the block, old position): the cells of a block row by row,
//! stable inside a
//! cell; cloud_sort_order = random (test) takes a random permutation instead.  Every nrdata / nidata column moves; the arrays are swapped with spare arrays
//! of the same extent (kept for the next call).  Runs at the start of a cycle (RefineClouds),
//! where no send list is pending.  The resampling and the merge pass take their candidates in
//! array order, so their results change with the order (statistically the same).  One line per
//! call in cloud_sort_log.txt (fenced stage times).

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
  const int gids = pmy_pack->gids, nmb = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  const bool randomize = cloud_sort_random;
  const int cyc = pmy_pack->pmesh->ncycle;
  const std::int64_t ncell = static_cast<std::int64_t>(nx1)*nx2, np = npart;
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
    const std::int64_t cell = static_cast<std::int64_t>(j)*nx1 + i;
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
  Kokkos::fence();
  const double t_move = t.seconds();
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
  const int nx3 = clouds3d ? pmy_pack->pmesh->mb_indcs.nx3 : 1;
  const bool z3 = clouds3d;
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

//========================================================================================
// The production merge pass for 3-D refinable clouds
//========================================================================================

namespace {

// sign * (mass, momentum) of the live clouds 0..n-1 of (r, i) on the 3-D leaves
void BudgetDeposit3D(const LeafTraversal3D &lt, const DvceArray2D<Real> &r,
                     const DvceArray2D<int> &i, const int n, const Real sign,
                     DvceArray2D<Real> &f) {
  if (n <= 0) return;
  auto f_ = f;
  auto tr = lt;
  Kokkos::parallel_for("budget_deposit3d", Kokkos::RangePolicy<>(DevExeSpace(), 0, n),
  KOKKOS_LAMBDA(const int p) {
    if (r(IPM,p) <= 0.0) return;
    const Real m = sign*r(IPM,p);
    const Real v1 = r(IPVX,p), v2 = r(IPVY,p), v3 = r(IPVZ,p);
    tr.ForEachCloudWeight(r(IPX,p), r(IPY,p), r(IPZ,p), i(PLEV,p), tr.Anchor(i(PGID,p)),
    [&](const int t, const Real w) {
      Kokkos::atomic_add(&f_(0,t), m*w); Kokkos::atomic_add(&f_(1,t), m*w*v1);
      Kokkos::atomic_add(&f_(2,t), m*w*v2); Kokkos::atomic_add(&f_(3,t), m*w*v3);
    });
  });
}

// the level-L cell of a position and the merge-cell mask on the 3-D leaves (MergeMask in 3-D)
struct MergeMask3D {
  LeafTraversal3D lt;
  int B = 0;
  static constexpr int kL = SMRCompositeTransfer::kCloudLevelMask + 1;
  int n1[kL], n2[kL], n3[kL];
  Real dx1[kL], dx2[kL], dx3[kL];
  Real x1min, x2min, x3min, x1max, x2max, x3max;
  KOKKOS_INLINE_FUNCTION
  void Index(const Real x, const Real y, const Real z, const int L, int &I, int &J, int &K) const {
    using T = SMRCompositeTransfer;
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

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::MergeCloudsDevice3D
//! \brief The production merge pass (quantum mode) for 3-D refinable clouds,
//! the rules of MergeCloudsDevice along three directions: (1) far-field release of lobed level-l
//! clouds in merge cells of level l - 1; per level lev (L = lev - 1): (X, with
//! cloud_merge_stencil) the exact re-merge of complete P1 families (27 members, masses
//! m_c / 2^(|a|+|b|+|c|)) into level-L clouds; (G) the mass-quantum walk of the plain level-lev
//! pieces of the deep merge zone in each level-L cell (4 x 4 x 4 sub-cells, Morton order rotated
//! by the pass number) into level-L clouds of the quantum m_root / 8^L, position and velocity
//! bounding boxes, K open groups; the pass budget (with the structure-relative beta over the
//! six neighbour leaves) accepts or rejects whole groups; the compaction.
//! No stage-X budget, no X protection.  Returns the merges.

int DustGasDrag::MergeCloudsDevice3D() {
  particles::Particles *ppar = pmy_pack->ppart;
  Mesh *pmesh = pmy_pack->pmesh;
  const int npart0 = ppar->nprtcl_thispack;
  const auto lt = leaf_traversal3d;
  const int nlev = lt.nlev;
  // several ranks: the budget exchanges are collective, so a rank without
  // particles runs the pass too; it stops only when no rank has any
  if (nlev == 0 || SumRanks(npart0) == 0) return 0;
  using T = SMRCompositeTransfer;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const int gids = pmy_pack->gids;
  const int ncell123 = lt.ncells1*lt.ncells2*lt.ncells3;
  EnsureNextTag();
  MergeMask3D mm;
  mm.lt = lt;
  mm.B = cloud_merge_buffer;
  mm.x1min = lt.x1min; mm.x2min = lt.x2min; mm.x3min = lt.x3min;
  mm.x1max = lt.x1max; mm.x2max = lt.x2max; mm.x3max = lt.x3max;
  const int nc1 = pmesh->mesh_indcs.nx1, nc2 = pmesh->mesh_indcs.nx2, nc3 = pmesh->mesh_indcs.nx3;
  for (int L=0; L<nlev; ++L) {
    mm.dx1[L] = lt.root_dx1/static_cast<Real>(1 << L);
    mm.dx2[L] = lt.root_dx2/static_cast<Real>(1 << L);
    mm.dx3[L] = lt.root_dx3/static_cast<Real>(1 << L);
    mm.n1[L] = nc1 << L; mm.n2[L] = nc2 << L; mm.n3[L] = nc3 << L;
  }
  Kokkos::Timer tpass;
  // the pass budget: lent the entry leaf fields, lcum the change so far
  // leaves of the halo blocks included (several ranks); only the owner of a leaf judges it
  const bool pbud = cloud_merge_budget > 0.0;
  const int nbt = pbud ? lt.nfld*ncell123 : 1;
  const int nown = pbud ? lt.nmb*ncell123 : 1;
  DvceArray2D<Real> lent("budget3d_lent", 4, nbt), lcum("budget3d_lcum", 4, nbt);
  if (pbud) {
    BudgetDeposit3D(lt, ppar->prtcl_rdata, ppar->prtcl_idata, npart0, 1.0, lent);
    HaloReverseAddFlat(lent, 4);   // the owners' leaves hold every rank's clouds
  }
  bool bdirty = false;
  double gcb[4] = {0.0, 0.0, 0.0, 0.0};   // rejected groups, pieces, mass; evaluations
  double bcap = 0.0;

  // (1) far-field release of lobed level-l clouds in merge cells of level l-1
  int nrelease = 0;
  {
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    DvceArray1D<int> np(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rel3d_np"), npart);
    Kokkos::parallel_reduce("rel3d_count", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &sum) {
      const int code = pi(PLEV,p);
      const int lev = T::CloudLevel(code);
      const int l1 = T::CloudLobe1(code), l2 = T::CloudLobe2(code), l3 = T::CloudLobe3(code);
      if ((l1 == 0 && l2 == 0 && l3 == 0) || lev == 0 ||
          !mm.In(pr(IPX,p), pr(IPY,p), pr(IPZ,p), lev - 1, lt.Anchor(pi(PGID,p)))) {
        np(p) = 0; return;
      }
      DPiece a[3], b[3], c[3];
      np(p) = ReleasePieces(l1, a)*ReleasePieces(l2, b)*ReleasePieces(l3, c);
      sum += 1;
    }, Kokkos::Sum<int>(nrelease));
    if (nrelease > 0) {
      DvceArray1D<int> off(Kokkos::view_alloc(Kokkos::WithoutInitializing, "rel3d_off"), npart);
      DvceArray1D<int> toff("rel3d_toff", npart);
      int nnew = 0, ntags = 0;
      Kokkos::parallel_scan("rel3d_scan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
        if (final) off(p) = upd;
        upd += (np(p) == 0) ? 1 : np(p);
      }, nnew);
      Kokkos::parallel_scan("rel3d_tscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
        if (final) toff(p) = upd;
        upd += (np(p) > 0) ? np(p) - 1 : 0;
      }, ntags);
      DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"),
                            nrdata, nnew);
      DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"),
                           nidata, nnew);
      const std::uint64_t tag0 = IssueTags(static_cast<std::uint64_t>(ntags));   // 64-bit tags
      const std::uint64_t tstride = static_cast<std::uint64_t>(global_variable::nranks);
      Kokkos::parallel_for("rel3d_fill", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        const int s0 = off(p);
        if (np(p) == 0) {
          for (int c=0; c<nrdata; ++c) prn(c,s0) = pr(c,p);
          for (int c=0; c<nidata; ++c) pin(c,s0) = pi(c,p);
          return;
        }
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
              // (shear-periodic x1; beyond a physical x3 face: mirrored)
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
      });
      ppar->prtcl_rdata = prn;
      ppar->prtcl_idata = pin;
      ppar->nprtcl_thispack = nnew;
      bdirty = true;
    }
  }
  EnsureRootCloudMass();
  // any number of species (the keys sp ncl + cell are 64-bit; the masses a View)
  const int nsp = std::max(1, std::max(static_cast<int>(cloud_merge_mroot.size()), nspecies));
  const auto mrootsp = SpeciesRootMass(cloud_merge_mroot, nsp);
  Real mroot_floor = 0.0;
  for (const Real m : cloud_merge_mroot) mroot_floor = std::max(mroot_floor, m);
  const Real rvol = lt.root_dx1*lt.root_dx2*lt.root_dx3;
  // the structure-relative budget: s_t = max over the six neighbour leaves
  const Real bkappa = cloud_merge_budget_kappa;
  DvceArray1D<Real> lbeta("budget3d_lbeta", (pbud && bkappa > 0.0) ? nbt : 1);
  if (pbud && bkappa > 0.0) {
    const Real beta = cloud_merge_budget;
    const int is_ = lt.is, js_ = lt.js, ks_ = lt.ks;
    const int ie_ = lt.is + lt.nx1 - 1, je_ = lt.js + lt.nx2 - 1, ke_ = lt.ks + lt.nx3 - 1;
    const Real rfloor = cloud_absorb_floor*mroot_floor/rvol;
    // the leaf masses with the owners' totals also on the halo leaves (one rank: a copy)
    DvceArray2D<Real> lmass("budget3d_lmass", 1, nbt);
    Kokkos::parallel_for("budget3d_lmass", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
    KOKKOS_LAMBDA(const int t) {lmass(0,t) = lent(0,t);});
    HaloForwardFlat(lmass, 1);
    const int nown_ = nown;
    Kokkos::parallel_for("budget3d_structure", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
    KOKKOS_LAMBDA(const int t) {
      lbeta(t) = beta;
      if (t >= nown_) return;   // the halo leaves keep beta; their owner decides
      int b, k, j, i;
      lt.Decode(t, b, k, j, i);
      if (i < is_ || i > ie_ || j < js_ || j > je_ || k < ks_ || k > ke_) return;
      const Real dx = lt.box(b,3), dy = lt.box(b,4), dz = lt.box(b,5);
      const Real xc = lt.box(b,0) + (i - is_ + 0.5)*dx, yc = lt.box(b,1) + (j - js_ + 0.5)*dy;
      const Real zc = lt.box(b,2) + (k - ks_ + 0.5)*dz;
      const Real rho = lmass(0,t)/(dx*dy*dz);
      const Real den = Kokkos::fmax(rho, rfloor);
      Real s = 0.0;
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
      lbeta(t) = Kokkos::fmax(Kokkos::fmin(beta, bkappa*s), 1.0e-12);
    });
  }
  const Real omega = is_shearing_box ? omega0 : 0.0;
  const Real alpha = cloud_merge_alpha, fmin = cloud_merge_min_mass;
  const bool exact_ts = cloud_merge_exact_ts;
  const bool alpha_leaf = cloud_merge_alpha_leaf;
  double qtot[8] = {0.0};   // X families, mass; G groups, pieces, mass; deferred, residual pieces
  for (int lev=1; lev<=nlev; ++lev) {
    const int L = lev - 1;
    auto pr = ppar->prtcl_rdata;
    auto pi = ppar->prtcl_idata;
    const int npart = ppar->nprtcl_thispack;
    const Real H1 = mm.dx1[L], H2 = mm.dx2[L], H3 = mm.dx3[L];
    const Real H = Kokkos::min(H1, Kokkos::min(H2, H3));
    const Real ell1 = lt.CloudHalfWidth1(lev), ell2 = lt.CloudHalfWidth2(lev);
    const Real ell3 = lt.CloudHalfWidth3(lev);
    const int m1 = mm.n1[L], m2 = mm.n2[L], m3 = mm.n3[L];
    const std::int64_t ncl = static_cast<std::int64_t>(m1)*m2*m3;
    // plain level-lev pieces with mass, by species and level-L cell
    DvceArray1D<std::int64_t> kp(Kokkos::view_alloc(Kokkos::WithoutInitializing, "merge3d_kp"),
                                 npart);
    Kokkos::parallel_for("merge3d_keys", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      const int code = pi(PLEV,p);
      kp(p) = -1;
      if (T::CloudLobe1(code) || T::CloudLobe2(code) || T::CloudLobe3(code)) return;
      if (T::CloudLevel(code) != lev || pr(IPM,p) <= 0.0) return;
      kp(p) = pi(PSP,p)*ncl + mm.Cell(pr(IPX,p), pr(IPY,p), pr(IPZ,p), L);
    });
    KeyCSR spc;
    BuildKeyCSR(kp, SpeciesKeyBound(nsp, ncl, 0), 0, spc);   // keys sp ncl + cell
    const int nmax = (npart > 0) ? npart : 1;
    DvceArray2D<Real> nr("q3_nr", nrdata, nmax);
    DvceArray2D<int> ni("q3_ni", nidata, nmax);
    DvceArray1D<int> nkind("q3_nkind", nmax);            // 0 X, 1 G
    DvceArray2D<double> acc("q3_acc", 4, nmax);          // m, m dx, m dy, m dz (G, about ref)
    DvceArray2D<Real> aref("q3_ref", 3, nmax);          // the reference point of a G cloud
    DvceArray1D<int> nnew("q3_nnew", 1);
    DvceArray1D<double> qc("q3_counters", 8);
    const int K = cloud_merge_K;
    const int rot = cloud_merge_pass % 64, rev = (cloud_merge_pass/64) % 2;
    // ---- stage X: complete P1 families (27 members) back into their level-L parent
    if (cloud_merge_stencil) {
      DvceArray1D<int> owner("q3_owner", npart);
      Kokkos::deep_copy(owner, -1);
      const Real dtol = cloud_merge_stencil_tol;
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
        if (exact_ts) nr(IPTS,knew) = ts;
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
        nkind(knew) = 0;
        for (int k=0; k<27; ++k) pr(IPM,mem[k]) = 0.0;
        Kokkos::atomic_add(&qc(0), 1.0);
        Kokkos::atomic_add(&qc(1), M);
      });
    }
    // ---- the leaf change made so far in this pass (release, stage X, coarser levels)
    const Real gb_floor = cloud_absorb_floor*mroot_floor/rvol;
    const Real gb_vmin = alpha*H*((omega > 0.0) ? omega : 0.0);
    if (pbud) {
      int nx_now = 0;
      Kokkos::deep_copy(nx_now, Kokkos::subview(nnew, 0));
      if (nx_now > 0) bdirty = true;
      Kokkos::deep_copy(lcum, 0.0);
      if (AnyRank(bdirty)) {
        BudgetDeposit3D(lt, pr, pi, npart, 1.0, lcum);
        BudgetDeposit3D(lt, nr, ni, nx_now, 1.0, lcum);
        HaloReverseAddFlat(lcum, 4);
        Kokkos::parallel_for("budget3d_cum", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
        KOKKOS_LAMBDA(const int t) {
          for (int c=0; c<4; ++c) lcum(c,t) -= lent(c,t);
        });
      }
    }
    DvceArray1D<int> krej("q3_krej", nmax);
    DvceArray2D<Real> ldelg("q3_ldel", 4, nbt);
    DvceArray1D<int> lviolg("q3_lviol", nbt);
    const bool bthin = cloud_merge_budget_thin;
    DvceArray2D<Real> lpacc("q3_lpacc", 1, bthin ? nbt : 1);   // accepted share
    // ---- stage G: the cell walk.  Pieces at level lev in level-L merge cells -> level-L clouds
    DvceArray1D<int> gid("q3_gid", npart), gknew("q3_gknew", nmax);
    DvceArray1D<int> ngroup("q3_ngroup", 1);
    Kokkos::deep_copy(gid, -1);
    Kokkos::deep_copy(gknew, -1);
    DvceArray1D<std::int64_t> kq("q3_key", npart);
    int k0 = 0;
    Kokkos::deep_copy(k0, Kokkos::subview(nnew, 0));
    {
      const Real smax1 = cloud_merge_smax*H1, smax2 = cloud_merge_smax*H2;
      const Real smax3 = cloud_merge_smax*H3;
      const Real qden = static_cast<Real>(std::int64_t(1) << (3*L));
      const int glag = cloud_merge_g_lag;
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
      KeyCSR qsc;
      BuildKeyCSR(kq, SpeciesKeyBound(nsp, ncl, 6), 3, qsc);
      DvceArray1D<int> cfirst(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q3_cfirst"),
                              qsc.nu + 1);
      int ncell = 0;
      {
        auto uk = qsc.ukey; const int nu = qsc.nu;
        Kokkos::parallel_scan("q3_cells", Kokkos::RangePolicy<>(DevExeSpace(), 0, nu),
        KOKKOS_LAMBDA(const int u, int &upd, const bool final) {
          const bool first = (u == 0) || ((uk(u) >> 6) != (uk(u - 1) >> 6));
          if (first && final) cfirst(upd) = u;
          if (first) ++upd;
          if (final && u == nu - 1) cfirst(upd) = nu;
        }, ncell);
      }
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
          nkind(knew) = 1;
          if (exact_ts) nr(IPTS,knew) = gts[j];
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
      // the members of committed groups into their new clouds (the pieces are used up below)
      Kokkos::parallel_for("q3_accumulate", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
      KOKKOS_LAMBDA(const int p) {
        if (kq(p) < 0) return;
        const int g = gid(p);
        if (g < 0) return;
        const int k = gknew(g);
        if (k < 0) return;
        const Real m = pr(IPM,p);
        Kokkos::atomic_add(&acc(0,k), static_cast<double>(m));
        Kokkos::atomic_add(&acc(1,k), static_cast<double>(m*T::MinImage(pr(IPX,p) - aref(0,k), lt.lx)));
        Kokkos::atomic_add(&acc(2,k), static_cast<double>(m*T::MinImage(pr(IPY,p) - aref(1,k), lt.ly)));
        Kokkos::atomic_add(&acc(3,k), static_cast<double>(m*T::MinImage(pr(IPZ,p) - aref(2,k), lt.lz)));
        for (int q=0; q<nrdata; ++q) {
          if (q == IPM || q == IPX || q == IPY || q == IPZ || q == IPX1 || q == IPY1 ||
              q == IPZ1) continue;
          if (exact_ts && q == IPTS) continue;
          Kokkos::atomic_add(&nr(q,k), m*pr(q,p));
        }
        if (!pbud) pr(IPM,p) = 0.0;
      });
      int k1 = k0;
      Kokkos::deep_copy(k1, Kokkos::subview(nnew, 0));
      if (k1 > k0) {
        Kokkos::parallel_for("q3_finalize_state", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
        KOKKOS_LAMBDA(const int k) {
          const double M = acc(0,k);
          for (int q=0; q<nrdata; ++q) {
            if (q == IPM || q == IPX || q == IPY || q == IPZ || q == IPX1 || q == IPY1 ||
                q == IPZ1) continue;
            if (exact_ts && q == IPTS) continue;
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
      // ---- accept or reject the proposed groups by the pass budget
      // several ranks: proposals on any rank are evaluated on all of them (their leaves may be
      // touched by a neighbour's proposals)
      if (pbud && AnyRank(k1 > k0)) {
        const Real beta = cloud_merge_budget;
        const int maxev = cloud_merge_budget_evals;
        int nev = 0;
        for (;;) {
          ++nev;
          Kokkos::deep_copy(ldelg, 0.0);
          Kokkos::parallel_for("q3b_members", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
          KOKKOS_LAMBDA(const int p) {
            if (kq(p) < 0 || gid(p) < 0) return;
            const int k = gknew(gid(p));
            if (k < 0 || krej(k)) return;
            const Real m = pr(IPM,p);
            const Real v1 = pr(IPVX,p), v2 = pr(IPVY,p), v3 = pr(IPVZ,p);
            lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pr(IPZ,p), pi(PLEV,p),
                                  lt.Anchor(pi(PGID,p)), [&](const int t, const Real w) {
              Kokkos::atomic_add(&ldelg(0,t), -m*w); Kokkos::atomic_add(&ldelg(1,t), -m*w*v1);
              Kokkos::atomic_add(&ldelg(2,t), -m*w*v2); Kokkos::atomic_add(&ldelg(3,t), -m*w*v3);
            });
          });
          Kokkos::parallel_for("q3b_new", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
          KOKKOS_LAMBDA(const int k) {
            if (krej(k)) return;
            const Real m = nr(IPM,k);
            const Real v1 = nr(IPVX,k), v2 = nr(IPVY,k), v3 = nr(IPVZ,k);
            lt.ForEachCloudWeight(nr(IPX,k), nr(IPY,k), nr(IPZ,k), ni(PLEV,k),
                                  lt.Anchor(ni(PGID,k)), [&](const int t, const Real w) {
              Kokkos::atomic_add(&ldelg(0,t), m*w); Kokkos::atomic_add(&ldelg(1,t), m*w*v1);
              Kokkos::atomic_add(&ldelg(2,t), m*w*v2); Kokkos::atomic_add(&ldelg(3,t), m*w*v3);
            });
          });
          HaloReverseAddFlat(ldelg, 4);   // the owners judge their leaves
          int nviol = 0;
          Kokkos::parallel_reduce("q3b_viol", Kokkos::RangePolicy<>(DevExeSpace(), 0, nown),
          KOKKOS_LAMBDA(const int t, int &sum) {
            int bad = 0;
            if (ldelg(0,t) != 0.0 || ldelg(1,t) != 0.0 || ldelg(2,t) != 0.0 || ldelg(3,t) != 0.0) {
              const int tb_ = t/ncell123;
              const Real vol = lt.box(tb_,3)*lt.box(tb_,4)*lt.box(tb_,5);
              Real rp = 0.0;
              const Real rm = BudgetRatio((bkappa > 0.0) ? lbeta(t) : beta, gb_floor*vol, gb_vmin,
                  lent(0,t), lent(1,t), lent(2,t), lent(3,t),
                  lcum(0,t) + ldelg(0,t), lcum(1,t) + ldelg(1,t), lcum(2,t) + ldelg(2,t),
                  lcum(3,t) + ldelg(3,t), rp);
              bad = (rm > 1.0 || rp > 1.0) ? 1 : 0;
              if (bthin) lpacc(0,t) = bad ? 0.9/Kokkos::fmax(rm, rp) : 1.0;
            } else if (bthin) {
              lpacc(0,t) = 1.0;
            }
            lviolg(t) = bad;
            sum += bad;
          }, nviol);
          HaloForwardFlags(lviolg);                       // the owners' verdicts on the halo
          if (bthin) HaloForwardFlat(lpacc, 1);
          nviol = static_cast<int>(SumRanks(nviol));      // the same number of rounds everywhere
          if (nviol == 0) break;
          // thinning in the rounds before the last one, the strict rule in the last
          const bool thin_now = bthin && nev < maxev - 1;
          if (nev >= maxev) {
            Kokkos::parallel_for("q3b_rej_all", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
            KOKKOS_LAMBDA(const int k) {krej(k) = 1;});
            Kokkos::deep_copy(ldelg, 0.0);
            bcap += 1.0;
            break;
          }
          Kokkos::parallel_for("q3b_rej_new", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
          KOKKOS_LAMBDA(const int k) {
            if (krej(k)) return;
            int bad = 0;
            const Real u = thin_now ? GroupHash01(GetTag(ni,k)) : 0.0;
            lt.ForEachCloudWeight(nr(IPX,k), nr(IPY,k), nr(IPZ,k), ni(PLEV,k),
                                  lt.Anchor(ni(PGID,k)), [&](const int t, const Real w) {
              if (w != 0.0 && lviolg(t) && (!thin_now || u >= lpacc(0,t))) bad = 1;
            });
            if (bad) krej(k) = 1;
          });
          Kokkos::parallel_for("q3b_rej_members", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
          KOKKOS_LAMBDA(const int p) {
            if (kq(p) < 0 || gid(p) < 0) return;
            const int k = gknew(gid(p));
            if (k < 0 || krej(k)) return;
            int bad = 0;
            const Real u = thin_now ? GroupHash01(GetTag(ni,k)) : 0.0;
            lt.ForEachCloudWeight(pr(IPX,p), pr(IPY,p), pr(IPZ,p), pi(PLEV,p),
                                  lt.Anchor(pi(PGID,p)), [&](const int t, const Real w) {
              if (w != 0.0 && lviolg(t) && (!thin_now || u >= lpacc(0,t))) bad = 1;
            });
            if (bad) Kokkos::atomic_exchange(&krej(k), 1);
          });
        }
        double rp_ = 0.0, rm_ = 0.0, rg_ = 0.0;
        Kokkos::parallel_reduce("q3b_apply", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
        KOKKOS_LAMBDA(const int p, double &n, double &m) {
          if (kq(p) < 0 || gid(p) < 0) return;
          const int k = gknew(gid(p));
          if (k < 0) return;
          if (krej(k)) {n += 1.0; m += pr(IPM,p); return;}
          pr(IPM,p) = 0.0;
        }, rp_, rm_);
        Kokkos::parallel_reduce("q3b_dead", Kokkos::RangePolicy<>(DevExeSpace(), k0, k1),
        KOKKOS_LAMBDA(const int k, double &n) {
          if (krej(k)) {nr(IPM,k) = 0.0; n += 1.0;}
        }, rg_);
        Kokkos::parallel_for("budget3d_advance", Kokkos::RangePolicy<>(DevExeSpace(), 0, nbt),
        KOKKOS_LAMBDA(const int t) {
          for (int c=0; c<4; ++c) lcum(c,t) += ldelg(c,t);
        });
        gcb[0] += rg_; gcb[1] += rp_; gcb[2] += rm_; gcb[3] += nev;
        if (rg_ < static_cast<double>(k1 - k0)) bdirty = true;
      }
    }
    auto qch = Kokkos::create_mirror_view_and_copy(HostMemSpace(), qc);
    for (int c=0; c<8; ++c) qtot[c] += qch(c);
    int ncommit = 0;
    Kokkos::deep_copy(ncommit, Kokkos::subview(nnew, 0));
    // compaction: drop the used-up pieces, append the new clouds that are kept
    DvceArray1D<int> koff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q3_koff"), npart);
    int nkeep = 0;
    Kokkos::parallel_scan("q3_kscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
      if (final) koff(p) = upd;
      upd += (pr(IPM,p) > 0.0) ? 1 : 0;
    }, nkeep);
    DvceArray1D<int> noff(Kokkos::view_alloc(Kokkos::WithoutInitializing, "q3_noff"),
                          ncommit > 0 ? ncommit : 1);
    int nlive = 0;
    if (ncommit > 0) {
      Kokkos::parallel_scan("q3_nscan", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
      KOKKOS_LAMBDA(const int k, int &upd, const bool final) {
        if (final) noff(k) = upd;
        upd += (nr(IPM,k) > 0.0) ? 1 : 0;
      }, nlive);
    }
    if (nkeep == npart && nlive == 0) continue;
    bdirty = true;
    const int ntot = nkeep + nlive;
    DvceArray2D<Real> prn(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_rdata"),
                          nrdata, ntot);
    DvceArray2D<int> pin(Kokkos::view_alloc(Kokkos::WithoutInitializing, "prtcl_idata"),
                         nidata, ntot);
    Kokkos::parallel_for("q3_keep", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p) {
      if (pr(IPM,p) <= 0.0) return;
      const int s = koff(p);
      for (int c=0; c<nrdata; ++c) prn(c,s) = pr(c,p);
      for (int c=0; c<nidata; ++c) pin(c,s) = pi(c,p);
    });
    Kokkos::parallel_for("q3_append", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncommit),
    KOKKOS_LAMBDA(const int k) {
      if (nr(IPM,k) <= 0.0) return;
      const int s = nkeep + noff(k);
      for (int c=0; c<nrdata; ++c) prn(c,s) = nr(c,k);
      for (int c=0; c<nidata; ++c) pin(c,s) = ni(c,k);
    });
    ppar->prtcl_rdata = prn;
    ppar->prtcl_idata = pin;
    ppar->nprtcl_thispack = ntot;
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
  cloud_coarsen_in += static_cast<unsigned long long>(27.0*qtot[0] + qtot[3] - gcb[1]);
  cloud_far_release += nrelease;
  return static_cast<int>(nevents);
}

}  // namespace dust
