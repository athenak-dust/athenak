#ifndef DUST_DUST_LEAF3D_HPP_
#define DUST_DUST_LEAF3D_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_leaf3d.hpp
//! \brief S3b: the block-local traversal of the actual leaves
//! under a refinable cloud on a 3-D mesh.  The 2-D LeafTraversal (dust.hpp) stays untouched;
//! this is its 3-D counterpart with the same rules: per direction the integral of the cloud
//! shape over a leaf interval is a difference of the hat CDF (and of its first moment), from
//! the one-dimensional helpers of LeafTraversal (MakeShape, FaceAt, Cell); the weight over a
//! leaf cell is the product of the three, the first moments the products with one moment.
//! Included by dust.hpp after LeafTraversal.
//!
//! Leaf cells are encoded ((b*ncells3 + k)*ncells2 + j)*ncells1 + i with b the block slot.
//! Block slots as in 2-D: [0, nmb) the blocks of this rank, [nmb, nfld) halo
//! blocks, [nfld, nall) every other block.  Stage 1 (one rank): nfld = nall = nmb.

struct LeafTraversal3D {
  static constexpr int kMaxNb = 64;    // distinct neighbours of a 3-D block (56 at most)
  int nmb = 0, gids = 0;
  int nfld = 0, nall = 0;
  int nx1 = 0, nx2 = 0, nx3 = 0, is = 0, js = 0, ks = 0;
  int ncells1 = 0, ncells2 = 0, ncells3 = 0;
  Real lx = 0.0, ly = 0.0, lz = 0.0;
  Real x1min = 0.0, x2min = 0.0, x3min = 0.0, x1max = 0.0, x2max = 0.0, x3max = 0.0;
  Real ilx = 0.0, ily = 0.0, ilz = 0.0;
  Real root_dx1 = 0.0, root_dx2 = 0.0, root_dx3 = 0.0;
  Real iroot_dx1 = 0.0, iroot_dx2 = 0.0, iroot_dx3 = 0.0;
  // (b, 0..8): x1min, x2min, x3min, dx1, dx2, dx3, 1/dx1, 1/dx2, 1/dx3 of slot b
  DvceArray2D<Real> box;
  DvceArray1D<int> nnb;                // number of neighbours of local block m
  DvceArray2D<int> nb;                 // (m, k): slot of neighbour k
  DvceArray3D<int> nbs;                // (m, k, 0..2): periodic shift (in domain lengths)
  static constexpr int kMaxLev = 16;
  int nlev = 0, nfine1 = 0, nfine2 = 0, nfine3 = 0;
  Real fine_dx1 = 0.0, fine_dx2 = 0.0, fine_dx3 = 0.0;
  int rdil[kMaxLev + 1] = {0};
  int cloud_max_level = 0;
  bool split_everywhere = false, split_overlap = false;
  DvceArray1D<int> blev;
  DvceArray2D<int> fbox;               // (b, 0..5): finest-cell box [x0,x1) x [y0,y1) x [z0,z1)
  DvceArray1D<int> nbmax;              // finest level among local block m and its neighbours
  DvceArray2D<int> safe;               // (m, 0..2): the ring its neighbours certainly hold
  DvceArray1D<std::int64_t> lockey;    // sorted logical-location keys (Locate)
  DvceArray1D<int> locblk;
  int nloc = 0;

  using LT = LeafTraversal;
  using T = SMRCompositeTransfer;

  KOKKOS_INLINE_FUNCTION
  int Leaf(const int b, const int k, const int j, const int i) const {
    return ((b*ncells3 + k)*ncells2 + j)*ncells1 + i;
  }
  KOKKOS_INLINE_FUNCTION
  void Decode(const int t, int &b, int &k, int &j, int &i) const {
    i = t % ncells1;
    int r = t/ncells1;
    j = r % ncells2;
    r /= ncells2;
    k = r % ncells3;
    b = r/ncells3;
  }

  // the periodic image of (x1, x2, x3) next to block m
  KOKKOS_INLINE_FUNCTION
  void Frame(const int m, Real &x1, Real &x2, Real &x3) const {
    const Real cx = box(m, 0) + 0.5*nx1*box(m, 3);
    const Real cy = box(m, 1) + 0.5*nx2*box(m, 4);
    const Real cz = box(m, 2) + 0.5*nx3*box(m, 5);
    x1 -= lx*round((x1 - cx)*ilx);
    x2 -= ly*round((x2 - cy)*ily);
    x3 -= lz*round((x3 - cz)*ilz);
  }

  // candidates as in LeafTraversal::ForCandidates, with the third direction:
  // f(b, sx, sy, sz): block b shifted by -(sx, sy, sz) domain lengths
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForCandidates(const int m, const bool global, const int ia, const int ib, const int ja,
                     const int jb, const int ka, const int kb, const int r,
                     const Function &f) const {
    if (!global) {
      f(m, 0, 0, 0);
      for (int k=0; k<nnb(m); ++k) f(nb(m, k), nbs(m, k, 0), nbs(m, k, 1), nbs(m, k, 2));
      return;
    }
    for (int b=0; b<nall; ++b) {
      int kx0, kx1, ky0, ky1, kz0, kz1;
      LT::ImageRange(fbox(b, 0), fbox(b, 1), r, nfine1, ia, ib, kx0, kx1);
      LT::ImageRange(fbox(b, 2), fbox(b, 3), r, nfine2, ja, jb, ky0, ky1);
      LT::ImageRange(fbox(b, 4), fbox(b, 5), r, nfine3, ka, kb, kz0, kz1);
      for (int kz=kz0; kz<=kz1; ++kz) {
        for (int ky=ky0; ky<=ky1; ++ky) {
          for (int kx=kx0; kx<=kx1; ++kx) f(b, -kx, -ky, -kz);
        }
      }
    }
  }
  KOKKOS_INLINE_FUNCTION
  void ImageRect(const int b, const int sx, const int sy, const int sz, const int r, int &x0,
                 int &x1, int &y0, int &y1, int &z0, int &z1) const {
    x0 = fbox(b, 0) - sx*nfine1 - r; x1 = fbox(b, 1) - sx*nfine1 + r;
    y0 = fbox(b, 2) - sy*nfine2 - r; y1 = fbox(b, 3) - sy*nfine2 + r;
    z0 = fbox(b, 4) - sz*nfine3 - r; z1 = fbox(b, 5) - sz*nfine3 + r;
  }
  // does the finest-cell box (inclusive bounds) meet a candidate block of level >= minlev
  // widened by r?
  KOKKOS_INLINE_FUNCTION
  bool Meets(const int m, const int ia, const int ib, const int ja, const int jb, const int ka,
             const int kb, const int minlev, const int r, const bool global) const {
    bool hit = false;
    ForCandidates(m, global, ia, ib, ja, jb, ka, kb, r,
                  [&](const int b, const int sx, const int sy, const int sz) {
      if (hit || blev(b) < minlev) return;
      int x0, x1, y0, y1, z0, z1;
      ImageRect(b, sx, sy, sz, r, x0, x1, y0, y1, z0, z1);
      if (ib < x0 || ia >= x1 || jb < y0 || ja >= y1 || kb < z0 || ka >= z1) return;
      hit = true;
    });
    return hit;
  }
  // is that box entirely covered by them?  The 2-D band scan (LeafTraversal::Covered) inside
  // slabs of z: every box taken contains the cell it was taken for and reaches some z above it;
  // the whole x-y rectangle is then covered up to the lowest such z top, where the next slab
  // starts.  A cell held by no box: not covered.
  KOKKOS_INLINE_FUNCTION
  bool Covered(const int m, const int ia, const int ib, const int ja, const int jb, const int ka,
               const int kb, const int minlev, const int r, const bool global) const {
    int z = ka;
    while (z <= kb) {
      int ztop = kb + 1;
      int y = ja;
      while (y <= jb) {
        int x = ia, top = jb + 1;
        while (x <= ib) {
          int xe = x, ye = y, ze = z;
          ForCandidates(m, global, x, x, y, y, z, z, r,
                        [&](const int b, const int sx, const int sy, const int sz) {
            if (blev(b) < minlev) return;
            int x0, x1, y0, y1, z0, z1;
            ImageRect(b, sx, sy, sz, r, x0, x1, y0, y1, z0, z1);
            if (x < x0 || x >= x1 || y < y0 || y >= y1 || z < z0 || z >= z1) return;
            if (x1 > xe || (x1 == xe && y1 > ye) || (x1 == xe && y1 == ye && z1 > ze)) {
              xe = x1; ye = y1; ze = z1;
            }
          });
          if (xe == x) return false;
          top = (ye < top) ? ye : top;
          ztop = (ze < ztop) ? ze : ztop;
          x = xe;
        }
        y = top;
      }
      z = ztop;
    }
    return true;
  }
  KOKKOS_INLINE_FUNCTION
  bool NeedGlobal(const int m, const int ia, const int ib, const int ja, const int jb,
                  const int ka, const int kb, const int r) const {
    int x0, x1, y0, y1, z0, z1;
    ImageRect(m, 0, 0, 0, 0, x0, x1, y0, y1, z0, z1);
    const int s1 = safe(m, 0), s2 = safe(m, 1), s3 = safe(m, 2);
    if (x0 - s1 <= ia - r && x1 + s1 > ib + r && y0 - s2 <= ja - r && y1 + s2 > jb + r &&
        z0 - s3 <= ka - r && z1 + s3 > kb + r) {
      return false;
    }
    return !Covered(m, ia - r, ib + r, ja - r, jb + r, ka - r, kb + r, 0, 0, false);
  }

  // candidate blocks of a support that does not lie inside the anchor block (see
  // LeafTraversal::Candidates); not inlined, the hot kernels only loop over the batch
  static constexpr int kBatch = 64;
  // a candidate as one int: block << 12 | (sx + 8) << 8 | (sy + 8) << 4 | (sz + 8)
  KOKKOS_INLINE_FUNCTION
  static int Pack(const int b, const int sx, const int sy, const int sz) {
    return (b << 12) | ((sx + 8) << 8) | ((sy + 8) << 4) | (sz + 8);
  }
  KOKKOS_INLINE_FUNCTION
  static int UnpackSx(const int c) {return ((c >> 8) & 15) - 8;}
  KOKKOS_INLINE_FUNCTION
  static int UnpackSy(const int c) {return ((c >> 4) & 15) - 8;}
  KOKKOS_INLINE_FUNCTION
  static int UnpackSz(const int c) {return (c & 15) - 8;}
  KOKKOS_FUNCTION DUST_NOINLINE
  int Candidates(const Real lo1, const Real hi1, const Real lo2, const Real hi2, const Real lo3,
                 const Real hi3, const int m, int &cursor, bool &global, int *cand) const {
    const int ia = static_cast<int>(floor((lo1 - x1min)/fine_dx1));
    const int ib = static_cast<int>(ceil((hi1 - x1min)/fine_dx1)) - 1;
    const int ja = static_cast<int>(floor((lo2 - x2min)/fine_dx2));
    const int jb = static_cast<int>(ceil((hi2 - x2min)/fine_dx2)) - 1;
    const int ka = static_cast<int>(floor((lo3 - x3min)/fine_dx3));
    const int kb = static_cast<int>(ceil((hi3 - x3min)/fine_dx3)) - 1;
    auto meets = [&](const int b, const int sx, const int sy, const int sz) {
      const Real bx0 = box(b, 0) - sx*lx, by0 = box(b, 1) - sy*ly, bz0 = box(b, 2) - sz*lz;
      return !(hi1 <= bx0 || lo1 >= bx0 + nx1*box(b, 3) ||
               hi2 <= by0 || lo2 >= by0 + nx2*box(b, 4) ||
               hi3 <= bz0 || lo3 >= bz0 + nx3*box(b, 5));
    };
    int n = 0;
    if (cursor == 0) {
      global = NeedGlobal(m, ia, ib, ja, jb, ka, kb, 0);
      if (!global) {
        ForCandidates(m, false, ia, ib, ja, jb, ka, kb, 0,
                      [&](const int b, const int sx, const int sy, const int sz) {
          if (!meets(b, sx, sy, sz)) return;
          cand[n++] = Pack(b, sx, sy, sz);   // at most 1 + kMaxNb <= kBatch + 1: checked below
        });
        cursor = -1;
        return n;
      }
    }
    int b = (cursor > 0) ? cursor - 1 : 0;
    for (; b<nall; ++b) {
      int kx0, kx1, ky0, ky1, kz0, kz1;
      LT::ImageRange(fbox(b, 0), fbox(b, 1), 0, nfine1, ia - 1, ib + 1, kx0, kx1);
      LT::ImageRange(fbox(b, 2), fbox(b, 3), 0, nfine2, ja - 1, jb + 1, ky0, ky1);
      LT::ImageRange(fbox(b, 4), fbox(b, 5), 0, nfine3, ka - 1, kb + 1, kz0, kz1);
      const int nimg = (kx1 >= kx0 && ky1 >= ky0 && kz1 >= kz0)
                     ? (kx1 - kx0 + 1)*(ky1 - ky0 + 1)*(kz1 - kz0 + 1) : 0;
      if (nimg > kBatch || kx0 < -7 || kx1 > 8 || ky0 < -7 || ky1 > 8 || kz0 < -7 || kz1 > 8) {
        Kokkos::abort("LeafTraversal3D::Candidates: a support meets too many periodic images");
      }
      if (n + nimg > kBatch) break;
      for (int kz=kz0; kz<=kz1; ++kz) {
        for (int ky=ky0; ky<=ky1; ++ky) {
          for (int kx=kx0; kx<=kx1; ++kx) {
            if (!meets(b, -kx, -ky, -kz)) continue;
            if (b >= nfld) Kokkos::abort("LeafTraversal3D::Candidates: a cloud reaches beyond the halo");
            cand[n++] = Pack(b, -kx, -ky, -kz);
          }
        }
      }
    }
    cursor = (b < nall) ? b + 1 : -1;
    return n;
  }

  // every leaf cell under the cloud `code` at (x1, x2, x3), anchor block m:
  // function(target, w, m1x, m1y, m1z) with kMoments, function(target, w) without.  Returns
  // the sum of the weights.  The callback appears once, in loops that are not unrolled.
  template <bool kMoments, typename Function>
  KOKKOS_INLINE_FUNCTION
  Real ForEach(const Real px1, const Real px2, const Real px3, const int code, const int m,
               const Function &function, bool *reach = nullptr) const {
    Real x1 = px1, x2 = px2, x3 = px3;
    Frame(m, x1, x2, x3);
    const int lev = T::CloudLevel(code);
    const bool point = kMoments && (code & T::kCloudGatherPoint) != 0;
    const LT::Shape s1 = LT::MakeShape(x1, ldexp(root_dx1, -lev), ldexp(iroot_dx1, lev),
                                       T::CloudLobe1(code), point);
    const LT::Shape s2 = LT::MakeShape(x2, ldexp(root_dx2, -lev), ldexp(iroot_dx2, lev),
                                       T::CloudLobe2(code), point);
    const LT::Shape s3 = LT::MakeShape(x3, ldexp(root_dx3, -lev), ldexp(iroot_dx3, lev),
                                       T::CloudLobe3(code), point);
    const Real ax0 = box(m, 0), ay0 = box(m, 1), az0 = box(m, 2);
    const bool inside = (s1.lo >= ax0 && s1.hi <= ax0 + nx1*box(m, 3) &&
                         s2.lo >= ay0 && s2.hi <= ay0 + nx2*box(m, 4) &&
                         s3.lo >= az0 && s3.hi <= az0 + nx3*box(m, 5));
    int cand[kBatch + 1];
    int ncand = 1, cursor = -1;
    bool global = false;
    cand[0] = Pack(m, 0, 0, 0);
    if (!inside) {
      cursor = 0;
      ncand = Candidates(s1.lo, s1.hi, s2.lo, s2.hi, s3.lo, s3.hi, m, cursor, global, cand);
    }
    if (reach != nullptr) *reach = !global;
    Real wsum = 0.0;
    while (true) {
#pragma unroll 1
      for (int c=0; c<ncand; ++c) {
        const int b = cand[c] >> 12;
        const Real dx = box(b, 3), dy = box(b, 4), dz = box(b, 5);
        const Real bx0 = box(b, 0) - UnpackSx(cand[c])*lx;
        const Real by0 = box(b, 1) - UnpackSy(cand[c])*ly;
        const Real bz0 = box(b, 2) - UnpackSz(cand[c])*lz;
        const Real idx = box(b, 6), idy = box(b, 7), idz = box(b, 8);
        int i0 = static_cast<int>(floor((s1.lo - bx0)*idx));
        int i1 = static_cast<int>(ceil((s1.hi - bx0)*idx)) - 1;
        int j0 = static_cast<int>(floor((s2.lo - by0)*idy));
        int j1 = static_cast<int>(ceil((s2.hi - by0)*idy)) - 1;
        int k0 = static_cast<int>(floor((s3.lo - bz0)*idz));
        int k1 = static_cast<int>(ceil((s3.hi - bz0)*idz)) - 1;
        i0 = (i0 < 0) ? 0 : i0; i1 = (i1 > nx1 - 1) ? nx1 - 1 : i1;
        j0 = (j0 < 0) ? 0 : j0; j1 = (j1 > nx2 - 1) ? nx2 - 1 : j1;
        k0 = (k0 < 0) ? 0 : k0; k1 = (k1 > nx3 - 1) ? nx3 - 1 : k1;
        constexpr int kRun = 4;
#pragma unroll 1
        for (int ic=i0; ic<=i1; ic+=kRun) {
          const int nr = (i1 - ic + 1 < kRun) ? i1 - ic + 1 : kRun;
          Real w1[kRun], g1[kRun];
          LT::Face f1 = LT::FaceAt<kMoments>(s1, bx0 + ic*dx);
#pragma unroll
          for (int a=0; a<kRun; ++a) {
            w1[a] = 0.0; g1[a] = 0.0;
            if (a < nr) {
              const LT::Face fn = LT::FaceAt<kMoments>(s1, bx0 + (ic + a + 1)*dx);
              LT::Cell<kMoments>(s1, f1, fn, w1[a], g1[a]);
              f1 = fn;
            }
          }
          LT::Face f3 = LT::FaceAt<kMoments>(s3, bz0 + k0*dz);
#pragma unroll 1
          for (int k=k0; k<=k1; ++k) {
            const LT::Face fn3 = LT::FaceAt<kMoments>(s3, bz0 + (k + 1)*dz);
            Real w3, g3;
            LT::Cell<kMoments>(s3, f3, fn3, w3, g3);
            f3 = fn3;
            if (w3 == 0.0 && (!kMoments || g3 == 0.0)) continue;
            LT::Face f2 = LT::FaceAt<kMoments>(s2, by0 + j0*dy);
#pragma unroll 1
            for (int j=j0; j<=j1; ++j) {
              const LT::Face fn2 = LT::FaceAt<kMoments>(s2, by0 + (j + 1)*dy);
              Real w2, g2;
              LT::Cell<kMoments>(s2, f2, fn2, w2, g2);
              f2 = fn2;
              if (w2 == 0.0 && (!kMoments || g2 == 0.0)) continue;
              const Real w23 = w2*w3;
              const int row = ((b*ncells3 + k + ks)*ncells2 + j + js)*ncells1 + ic + is;
#pragma unroll 1
              for (int a=0; a<nr; ++a) {
                const Real wa = (a == 0) ? w1[0] : ((a == 1) ? w1[1] : ((a == 2) ? w1[2] : w1[3]));
                const Real w = w23*wa;
                wsum += w;
                if constexpr (kMoments) {
                  const Real ga = (a == 0) ? g1[0] : ((a == 1) ? g1[1] : ((a == 2) ? g1[2] : g1[3]));
                  const Real m1x = w23*ga, m1y = wa*g2*w3, m1z = wa*w2*g3;
                  if (w == 0.0 && m1x == 0.0 && m1y == 0.0 && m1z == 0.0) continue;
                  function(row + a, w, m1x, m1y, m1z);
                } else {
                  if (w == 0.0) continue;
                  function(row + a, w);
                }
              }
            }
          }
        }
      }
      if (cursor < 0) break;
      ncand = Candidates(s1.lo, s1.hi, s2.lo, s2.hi, s3.lo, s3.hi, m, cursor, global, cand);
    }
    return wsum;
  }

  // logical-location key (l, lx1, lx2, lx3) of a block, 19 bits per index
  KOKKOS_INLINE_FUNCTION
  static std::int64_t LocKey(const int l, const int lx1, const int lx2, const int lx3) {
    return (static_cast<std::int64_t>(l) << 57) | (static_cast<std::int64_t>(lx3) << 38) |
           (static_cast<std::int64_t>(lx2) << 19) | static_cast<std::int64_t>(lx1);
  }
  // encoded leaf cell holding the finest cell (ic, jc, kc) (periodic), -1 if none
  KOKKOS_INLINE_FUNCTION
  int Locate(const int ic_in, const int jc_in, const int kc_in, int *bout = nullptr) const {
    const int ic = ((ic_in % nfine1) + nfine1) % nfine1;
    const int jc = ((jc_in % nfine2) + nfine2) % nfine2;
    const int kc = ((kc_in % nfine3) + nfine3) % nfine3;
    for (int l=0; l<=nlev; ++l) {
      const int sh = nlev - l, w1 = nx1 << sh, w2 = nx2 << sh, w3 = nx3 << sh;
      const std::int64_t key = LocKey(l, ic/w1, jc/w2, kc/w3);
      int lo = 0, hi = nloc;
      while (lo < hi) {
        const int mid = (lo + hi)/2;
        if (lockey(mid) < key) lo = mid + 1; else hi = mid;
      }
      if (lo < nloc && lockey(lo) == key) {
        const int b = locblk(lo);
        if (bout != nullptr) *bout = b;
        const int i = (ic - fbox(b, 0)) >> sh, j = (jc - fbox(b, 2)) >> sh;
        const int k = (kc - fbox(b, 4)) >> sh;
        return Leaf(b, k + ks, j + js, i + is);
      }
    }
    return -1;
  }
  KOKKOS_INLINE_FUNCTION
  int Owner(const Real px1, const Real px2, const Real px3, const int m,
            int *bout = nullptr) const {
    Real x1 = px1, x2 = px2, x3 = px3;
    Frame(m, x1, x2, x3);
    int res = -1, bb = -1;
    const int ic = static_cast<int>(floor((x1 - x1min)/fine_dx1));
    const int jc = static_cast<int>(floor((x2 - x2min)/fine_dx2));
    const int kc = static_cast<int>(floor((x3 - x3min)/fine_dx3));
    for (int pass=0; pass<3 && res < 0; ++pass) {
      if (pass == 1 && nloc > 0) {
        res = Locate(ic, jc, kc, &bb);
        break;
      }
      const Real tol = (pass == 2) ? 1.0e-9 : 0.0;
      ForCandidates(m, pass >= 1, ic - 1, ic + 1, jc - 1, jc + 1, kc - 1, kc + 1, 0,
                    [&](const int b, const int sx, const int sy, const int sz) {
        if (res >= 0) return;
        const Real fi = (x1 - (box(b, 0) - sx*lx))/box(b, 3);
        const Real fj = (x2 - (box(b, 1) - sy*ly))/box(b, 4);
        const Real fk = (x3 - (box(b, 2) - sz*lz))/box(b, 5);
        if (!(fi >= -tol && fi < nx1 + tol && fj >= -tol && fj < nx2 + tol &&
              fk >= -tol && fk < nx3 + tol)) return;
        int i = static_cast<int>(floor(fi)), j = static_cast<int>(floor(fj));
        int k = static_cast<int>(floor(fk));
        i = (i < 0) ? 0 : ((i > nx1 - 1) ? nx1 - 1 : i);
        j = (j < 0) ? 0 : ((j > nx2 - 1) ? nx2 - 1 : j);
        k = (k < 0) ? 0 : ((k > nx3 - 1) ? nx3 - 1 : k);
        res = Leaf(b, k + ks, j + js, i + is);
        bb = b;
      });
    }
    if (bout != nullptr) *bout = bb;
    return res;
  }

  // ---- eligibility (LeafTraversal) ----------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  bool AnyAbove(const int m, const int ia, const int ib, const int ja, const int jb,
                const int ka, const int kb, const int lev, bool *reach = nullptr) const {
    if (lev + 1 > nlev) return false;
    const int r = rdil[lev + 1];
    const bool g = NeedGlobal(m, ia, ib, ja, jb, ka, kb, r);
    if (reach != nullptr) *reach = !g;
    if (!g && m < nmb && lev + 1 > nbmax(m) && ia <= ib && ja <= jb && ka <= kb) return false;
    return Meets(m, ia, ib, ja, jb, ka, kb, lev + 1, r, g);
  }
  KOKKOS_INLINE_FUNCTION
  bool AllAbove(const int m, const int ia, const int ib, const int ja, const int jb,
                const int ka, const int kb, const int lev, bool *reach = nullptr) const {
    if (lev + 1 > nlev) return false;
    const int r = rdil[lev + 1];
    const bool g = NeedGlobal(m, ia, ib, ja, jb, ka, kb, r);
    if (reach != nullptr) *reach = !g;
    if (!g && m < nmb && lev + 1 > nbmax(m) && ia <= ib && ja <= jb && ka <= kb) return false;
    return Covered(m, ia, ib, ja, jb, ka, kb, lev + 1, r, g);
  }
  // the finest-cell box of the support of a plain level-lev cloud shrunk by slack (InsideFiner)
  // or of its full support (OverlapsFiner)
  KOKKOS_INLINE_FUNCTION
  void SupportBox(const Real x1, const Real x2, const Real x3, const Real f, const int lev,
                  int &ia, int &ib, int &ja, int &jb, int &ka, int &kb) const {
    const Real r1 = f*root_dx1/static_cast<Real>(1 << lev);
    const Real r2 = f*root_dx2/static_cast<Real>(1 << lev);
    const Real r3 = f*root_dx3/static_cast<Real>(1 << lev);
    ia = static_cast<int>(floor((x1 - r1 - x1min)/fine_dx1));
    ib = static_cast<int>(ceil((x1 + r1 - x1min)/fine_dx1)) - 1;
    ja = static_cast<int>(floor((x2 - r2 - x2min)/fine_dx2));
    jb = static_cast<int>(ceil((x2 + r2 - x2min)/fine_dx2)) - 1;
    ka = static_cast<int>(floor((x3 - r3 - x3min)/fine_dx3));
    kb = static_cast<int>(ceil((x3 + r3 - x3min)/fine_dx3)) - 1;
  }
  KOKKOS_INLINE_FUNCTION
  bool InsideFiner(const Real px1, const Real px2, const Real px3, const int code,
                   const Real slack, const int m, bool *reach = nullptr) const {
    if (reach != nullptr) *reach = true;
    if (T::CloudLobe1(code) || T::CloudLobe2(code) || T::CloudLobe3(code)) return false;
    const int lev = code & T::kCloudLevelMask;
    if (lev >= nlev || lev >= cloud_max_level) return false;
    if (split_everywhere) return true;
    Real x1 = px1, x2 = px2, x3 = px3;
    Frame(m, x1, x2, x3);
    int ia, ib, ja, jb, ka, kb;
    SupportBox(x1, x2, x3, 1.0 - slack, lev, ia, ib, ja, jb, ka, kb);
    return split_overlap ? AnyAbove(m, ia, ib, ja, jb, ka, kb, lev, reach)
                         : AllAbove(m, ia, ib, ja, jb, ka, kb, lev, reach);
  }
  KOKKOS_INLINE_FUNCTION
  bool OverlapsFiner(const Real px1, const Real px2, const Real px3, const int code,
                     const int m, bool *reach = nullptr) const {
    if (reach != nullptr) *reach = true;
    const int lev = T::CloudLevel(code);
    if (lev >= nlev) return false;
    Real x1 = px1, x2 = px2, x3 = px3;
    Frame(m, x1, x2, x3);
    int ia, ib, ja, jb, ka, kb;
    SupportBox(x1, x2, x3, 1.0, lev, ia, ib, ja, jb, ka, kb);
    return AnyAbove(m, ia, ib, ja, jb, ka, kb, lev, reach);
  }
  // the merge-cell mask: the level-L cell holding (x1, x2, x3) and the B cells around it carry
  // no dilated level above L
  KOKKOS_INLINE_FUNCTION
  bool MergeOK(const Real px1, const Real px2, const Real px3, const int L, const int B,
               const int m, bool *reach = nullptr) const {
    Real x1 = px1, x2 = px2, x3 = px3;
    Frame(m, x1, x2, x3);
    const int r = 1 << (nlev - L);
    const int I = static_cast<int>(floor((x1 - x1min)/(root_dx1/static_cast<Real>(1 << L))));
    const int J = static_cast<int>(floor((x2 - x2min)/(root_dx2/static_cast<Real>(1 << L))));
    const int K = static_cast<int>(floor((x3 - x3min)/(root_dx3/static_cast<Real>(1 << L))));
    if (reach != nullptr) *reach = true;
    return !AnyAbove(m, (I - B)*r, (I + B + 1)*r - 1, (J - B)*r, (J + B + 1)*r - 1,
                     (K - B)*r, (K + B + 1)*r - 1, L, reach);
  }

  // ---- the cloud interface (LeafTraversal) --------------------------------------------
  KOKKOS_INLINE_FUNCTION
  Real CloudHalfWidth1(const int lev) const {
    return root_dx1/static_cast<Real>(1 << (lev & T::kCloudLevelMask));
  }
  KOKKOS_INLINE_FUNCTION
  Real CloudHalfWidth2(const int lev) const {
    return root_dx2/static_cast<Real>(1 << (lev & T::kCloudLevelMask));
  }
  KOKKOS_INLINE_FUNCTION
  Real CloudHalfWidth3(const int lev) const {
    return root_dx3/static_cast<Real>(1 << (lev & T::kCloudLevelMask));
  }
  KOKKOS_INLINE_FUNCTION
  int Anchor(const int pgid) const {
    const int m = pgid - gids;
    return (m >= 0 && m < nmb) ? m : 0;
  }
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachCloudWeight(const Real x1, const Real x2, const Real x3, const int code,
                          const int m, const Function &function) const {
    ForEach<false>(x1, x2, x3, code, m, function);
  }
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForEachCloudMoment(const Real x1, const Real x2, const Real x3, const int code,
                          const int m, const Function &function) const {
    ForEach<true>(x1, x2, x3, code, m, function);
  }
  KOKKOS_INLINE_FUNCTION
  bool CloudInsideFiner(const Real x1, const Real x2, const Real x3, const int code,
                        const Real slack, const int m) const {
    return InsideFiner(x1, x2, x3, code, slack, m);
  }
  KOKKOS_INLINE_FUNCTION
  bool CloudOverlapsFiner(const Real x1, const Real x2, const Real x3, const int code,
                          const int m) const {
    return OverlapsFiner(x1, x2, x3, code, m);
  }
  KOKKOS_INLINE_FUNCTION
  int LeafOwner(const Real x1, const Real x2, const Real x3, const int m) const {
    return Owner(x1, x2, x3, m);
  }
};

#endif // DUST_DUST_LEAF3D_HPP_
