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
//! blocks, [nfld, nall) every other block.  One rank: nfld = nall = nmb.

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
  // shear-periodic x1.  A point (x, y) beyond the outer x1 face is the
  // point (x - lx, y + ysh) of the domain, one beyond the inner face (x + lx, y - ysh)
  // (bvals_part.cpp); ysh at the time of the positions traversed (DustGasDrag::UpdateShear3D).
  // A support across an x1 face is traversed in two parts (ForEach): the part in the domain
  // (candidates without an x shift, mode 1) and the cloud translated to the other side (mode 2:
  // across the inner face, translated by (+lx, -ysh); mode 3: across the outer face, (-lx, +ysh)),
  // whose candidates are the shear neighbours of the anchor: (m, side, k) packed candidates
  // without an x shift, side 0 / 1 = mode 2 / 3, nsnb(m, side) of them, every block that meets
  // the translated box of m widened by sreach(m) finest cells
  bool shear = false;
  Real ysh = 0.0;
  DvceArray2D<int> nsnb;
  DvceArray3D<int> snb;
  DvceArray1D<int> sreach;

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

  // the periodic image of (x1, x2, x3) next to block m; with shear-periodic x1 a shift by n
  // domain lengths in x1 moves x2 by n ysh.  Returns n.
  KOKKOS_INLINE_FUNCTION
  int Frame(const int m, Real &x1, Real &x2, Real &x3) const {
    const Real cx = box(m, 0) + 0.5*nx1*box(m, 3);
    const Real cy = box(m, 1) + 0.5*nx2*box(m, 4);
    const Real cz = box(m, 2) + 0.5*nx3*box(m, 5);
    const Real n1 = round((x1 - cx)*ilx);
    x1 -= lx*n1;
    if (shear) x2 += n1*ysh;
    x2 -= ly*round((x2 - cy)*ily);
    x3 -= lz*round((x3 - cz)*ilz);
    return static_cast<int>(n1);
  }
  // a position into the domain (every direction); with shear-periodic x1 a wrap in x1 by n domain
  // lengths moves x2 by n ysh (the rule of the particle routing)
  KOKKOS_INLINE_FUNCTION
  void WrapPos(Real &x1, Real &x2, Real &x3) const {
    const Real w1 = LT::Wrap(x1, x1min, x1max);
    if (shear && w1 != x1) x2 += round((x1 - w1)*ilx)*ysh;
    x1 = w1;
    x2 = LT::Wrap(x2, x2min, x2max);
    x3 = LT::Wrap(x3, x3min, x3max);
  }
  // the side of an x1 face a finest-cell range [ia, ib] crosses (-1: none; 0 inner; 1 outer)
  KOKKOS_INLINE_FUNCTION
  int ShearSide(const int ia, const int ib) const {
    if (!shear) return -1;
    return (ib >= nfine1) ? 1 : ((ia < 0) ? 0 : -1);
  }

  // candidates as in LeafTraversal::ForCandidates, with the third direction:
  // f(b, sx, sy, sz): block b shifted by -(sx, sy, sz) domain lengths.  mode: 0 all,
  // 1 none shifted in x1, 2 / 3 the shear neighbours of m across the inner / outer x1 face
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  void ForCandidates(const int m, const bool global, const int ia, const int ib, const int ja,
                     const int jb, const int ka, const int kb, const int r,
                     const Function &f, const int mode = 0) const {
    if (!global) {
      if (mode >= 2) {
        const int sd = mode - 2;
        for (int k=0; k<nsnb(m, sd); ++k) {
          const int c = snb(m, sd, k);
          f(c >> 12, 0, UnpackSy(c), UnpackSz(c));
        }
        return;
      }
      f(m, 0, 0, 0);
      for (int k=0; k<nnb(m); ++k) {
        if (mode == 1 && nbs(m, k, 0) != 0) continue;
        f(nb(m, k), nbs(m, k, 0), nbs(m, k, 1), nbs(m, k, 2));
      }
      return;
    }
    for (int b=0; b<nall; ++b) {
      int kx0, kx1, ky0, ky1, kz0, kz1;
      LT::ImageRange(fbox(b, 0), fbox(b, 1), r, nfine1, ia, ib, kx0, kx1);
      LT::ImageRange(fbox(b, 2), fbox(b, 3), r, nfine2, ja, jb, ky0, ky1);
      LT::ImageRange(fbox(b, 4), fbox(b, 5), r, nfine3, ka, kb, kz0, kz1);
      if (mode != 0) {   // only the image without an x1 shift
        if (kx0 > 0 || kx1 < 0) continue;
        kx0 = 0; kx1 = 0;
      }
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
             const int kb, const int minlev, const int r, const bool global,
             const int mode = 0) const {
    bool hit = false;
    ForCandidates(m, global, ia, ib, ja, jb, ka, kb, r,
                  [&](const int b, const int sx, const int sy, const int sz) {
      if (hit || blev(b) < minlev) return;
      int x0, x1, y0, y1, z0, z1;
      ImageRect(b, sx, sy, sz, r, x0, x1, y0, y1, z0, z1);
      if (ib < x0 || ia >= x1 || jb < y0 || ja >= y1 || kb < z0 || ka >= z1) return;
      hit = true;
    }, mode);
    return hit;
  }
  // is that box entirely covered by them?  The 2-D band scan (LeafTraversal::Covered) inside
  // slabs of z: every box taken contains the cell it was taken for and reaches some z above it;
  // the whole x-y rectangle is then covered up to the lowest such z top, where the next slab
  // starts.  A cell held by no box: not covered.
  KOKKOS_INLINE_FUNCTION
  bool Covered(const int m, const int ia, const int ib, const int ja, const int jb, const int ka,
               const int kb, const int minlev, const int r, const bool global,
               const int mode = 0) const {
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
          }, mode);
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
  // mode 2 / 3: inside the translated box of m that the shear neighbours were found for (one
  // finest cell less, against round-off)
  KOKKOS_INLINE_FUNCTION
  bool InShearBox(const int m, const int mode, const int ia, const int ib, const int ja,
                  const int jb, const int ka, const int kb) const {
    const int s = 2*(mode - 2) - 1;
    const int R = sreach(m) - 1;
    const Real yf = s*ysh/fine_dx2;
    const int x0 = fbox(m, 0) - s*nfine1 - R, x1 = fbox(m, 1) - s*nfine1 + R;
    const int y0 = static_cast<int>(ceil(fbox(m, 2) + yf)) - R;
    const int y1 = static_cast<int>(floor(fbox(m, 3) + yf)) + R;
    const int z0 = fbox(m, 4) - R, z1 = fbox(m, 5) + R;
    return ia >= x0 && ib < x1 && ja >= y0 && jb < y1 && ka >= z0 && kb < z1;
  }
  KOKKOS_INLINE_FUNCTION
  bool NeedGlobal(const int m, const int ia, const int ib, const int ja, const int jb,
                  const int ka, const int kb, const int r, const int mode = 0) const {
    // modes 1-3 (shear-periodic x1, one part of a support): the box in the domain in x1
    int a1 = ia - r, b1 = ib + r;
    if (mode != 0) {a1 = (a1 < 0) ? 0 : a1; b1 = (b1 > nfine1 - 1) ? nfine1 - 1 : b1;}
    if (mode >= 2) return !InShearBox(m, mode, a1, b1, ja - r, jb + r, ka - r, kb + r);
    int x0, x1, y0, y1, z0, z1;
    ImageRect(m, 0, 0, 0, 0, x0, x1, y0, y1, z0, z1);
    const int s1 = safe(m, 0), s2 = safe(m, 1), s3 = safe(m, 2);
    if (x0 - s1 <= a1 && x1 + s1 > b1 && y0 - s2 <= ja - r && y1 + s2 > jb + r &&
        z0 - s3 <= ka - r && z1 + s3 > kb + r) {
      return false;
    }
    return !Covered(m, a1, b1, ja - r, jb + r, ka - r, kb + r, 0, 0, false, mode);
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
                 const Real hi3, const int m, int &cursor, bool &global, int *cand,
                 const int mode = 0) const {
    int ia = static_cast<int>(floor((lo1 - x1min)/fine_dx1));
    int ib = static_cast<int>(ceil((hi1 - x1min)/fine_dx1)) - 1;
    if (mode != 0) {   // one part of a support across a shear-periodic x1 face: in the domain
      ia = (ia < 0) ? 0 : ia; ib = (ib > nfine1 - 1) ? nfine1 - 1 : ib;
    }
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
      global = NeedGlobal(m, ia, ib, ja, jb, ka, kb, 0, mode);
      if (!global) {
        ForCandidates(m, false, ia, ib, ja, jb, ka, kb, 0,
                      [&](const int b, const int sx, const int sy, const int sz) {
          if (!meets(b, sx, sy, sz) || n > kBatch) return;
          cand[n++] = Pack(b, sx, sy, sz);   // at most 1 + kMaxNb <= kBatch + 1: checked below
        }, mode);
        if (n > kBatch) Kokkos::abort("LeafTraversal3D::Candidates: too many shear neighbours");
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
      if (mode != 0) {   // only the image without an x1 shift
        if (kx0 > 0 || kx1 < 0) continue;
        kx0 = 0; kx1 = 0;
      }
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
  // Shear-periodic x1: a support across an x1 face in two parts, the cloud in the
  // domain and the cloud translated across the face; the callers add MinImage(p - c) w to the
  // moments with the position p they passed, so in a part whose position differs from p by a
  // shear shift (the translated part, or a cloud framed across x1) m1y carries the difference.
  template <bool kMoments, typename Function>
  KOKKOS_INLINE_FUNCTION
  Real ForEach(const Real px1, const Real px2, const Real px3, const int code, const int m,
               const Function &function, bool *reach = nullptr) const {
    Real x1 = px1, x2 = px2, x3 = px3;
    const int nw = Frame(m, x1, x2, x3);
    const int lev = T::CloudLevel(code);
    const bool point = kMoments && (code & T::kCloudGatherPoint) != 0;
    const Real h1 = ldexp(root_dx1, -lev), ih1 = ldexp(iroot_dx1, lev);
    const Real h2 = ldexp(root_dx2, -lev), ih2 = ldexp(iroot_dx2, lev);
    LT::Shape s1 = LT::MakeShape(x1, h1, ih1, T::CloudLobe1(code), point);
    LT::Shape s2 = LT::MakeShape(x2, h2, ih2, T::CloudLobe2(code), point);
    const LT::Shape s3 = LT::MakeShape(x3, ldexp(root_dx3, -lev), ldexp(iroot_dx3, lev),
                                       T::CloudLobe3(code), point);
    int npart = 1, side = 0;
    if (shear && (s1.lo < x1min || s1.hi > x1max)) {npart = 2; side = (s1.hi > x1max) ? 1 : 0;}
    Real wsum = 0.0;
    bool global_any = false;
#pragma unroll 1
    for (int part=0; part<npart; ++part) {
      int mode = (npart == 2) ? 1 : 0;
      Real yp = x2;   // the y of this part's cloud
      if (part == 1) {
        const Real sg = (side == 1) ? 1.0 : -1.0;
        yp = x2 + sg*ysh;
        s1 = LT::MakeShape(x1 - sg*lx, h1, ih1, T::CloudLobe1(code), point);
        s2 = LT::MakeShape(yp, h2, ih2, T::CloudLobe2(code), point);
        mode = 2 + side;
      }
      const bool ycorr = kMoments && shear && (part == 1 || nw != 0);
      const Real ax0 = box(m, 0), ay0 = box(m, 1), az0 = box(m, 2);
      const bool inside = (mode == 0 && s1.lo >= ax0 && s1.hi <= ax0 + nx1*box(m, 3) &&
                           s2.lo >= ay0 && s2.hi <= ay0 + nx2*box(m, 4) &&
                           s3.lo >= az0 && s3.hi <= az0 + nx3*box(m, 5));
      int cand[kBatch + 1];
      int ncand = 1, cursor = -1;
      bool global = false;
      cand[0] = Pack(m, 0, 0, 0);
      if (!inside) {
        cursor = 0;
        ncand = Candidates(s1.lo, s1.hi, s2.lo, s2.hi, s3.lo, s3.hi, m, cursor, global, cand, mode);
      }
      global_any = global_any || global;
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
                // the shear shift of this part against the caller's position (see above)
                Real cyj = 0.0;
                if (ycorr) {
                  const Real yc = by0 + (j + 0.5)*dy;
                  cyj = (yp - yc) - T::MinImage(px2 - yc, ly);
                }
                const int row = ((b*ncells3 + k + ks)*ncells2 + j + js)*ncells1 + ic + is;
#pragma unroll 1
                for (int a=0; a<nr; ++a) {
                  const Real wa = (a == 0) ? w1[0] : ((a == 1) ? w1[1] : ((a == 2) ? w1[2] : w1[3]));
                  const Real w = w23*wa;
                  wsum += w;
                  if constexpr (kMoments) {
                    const Real ga = (a == 0) ? g1[0] : ((a == 1) ? g1[1] : ((a == 2) ? g1[2] : g1[3]));
                    const Real m1x = w23*ga, m1z = wa*w2*g3;
                    Real m1y = wa*g2*w3;
                    if (ycorr) m1y += cyj*w;
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
        ncand = Candidates(s1.lo, s1.hi, s2.lo, s2.hi, s3.lo, s3.hi, m, cursor, global, cand, mode);
      }
    }
    if (reach != nullptr) *reach = !global_any;
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
    // shear-periodic x1: a point beyond an x1 face into the domain, then the
    // shear neighbours of m
    int mode = 0;
    if (shear && (x1 < x1min || x1 >= x1max)) {
      const int side = (x1 >= x1max) ? 1 : 0;
      const Real sg = (side == 1) ? 1.0 : -1.0;
      x1 -= sg*lx; x2 += sg*ysh;
      mode = 2 + side;
    }
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
      }, mode);
    }
    if (bout != nullptr) *bout = bb;
    return res;
  }

  // ---- eligibility (LeafTraversal) ----------------------------------------------------
  KOKKOS_INLINE_FUNCTION
  bool AnyAbove(const int m, const int ia, const int ib, const int ja, const int jb,
                const int ka, const int kb, const int lev, bool *reach = nullptr,
                const int mode = 0) const {
    if (lev + 1 > nlev) return false;
    const int r = rdil[lev + 1];
    const bool g = NeedGlobal(m, ia, ib, ja, jb, ka, kb, r, mode);
    if (reach != nullptr) *reach = !g;
    if (mode <= 1 && !g && m < nmb && lev + 1 > nbmax(m) && ia <= ib && ja <= jb && ka <= kb) {
      return false;
    }
    return Meets(m, ia, ib, ja, jb, ka, kb, lev + 1, r, g, mode);
  }
  KOKKOS_INLINE_FUNCTION
  bool AllAbove(const int m, const int ia, const int ib, const int ja, const int jb,
                const int ka, const int kb, const int lev, bool *reach = nullptr,
                const int mode = 0) const {
    if (lev + 1 > nlev) return false;
    const int r = rdil[lev + 1];
    const bool g = NeedGlobal(m, ia, ib, ja, jb, ka, kb, r, mode);
    if (reach != nullptr) *reach = !g;
    if (mode <= 1 && !g && m < nmb && lev + 1 > nbmax(m) && ia <= ib && ja <= jb && ka <= kb) {
      return false;
    }
    return Covered(m, ia, ib, ja, jb, ka, kb, lev + 1, r, g, mode);
  }
  // shear-periodic x1: AnyAbove (kAll false) or AllAbove (true) of a box across an
  // x1 face (side 0 inner, 1 outer) in two parts: box A in the domain (mode 1) and box B, the
  // same query around the position translated across the face, in the domain (mode 2 / 3);
  // Any: either part, All: both
  template <bool kAll>
  KOKKOS_INLINE_FUNCTION
  bool AboveParts(const int m, const int side, const int lev, bool *reach, int ia, int ib,
                  const int ja, const int jb, const int ka, const int kb, int ia2, int ib2,
                  const int ja2, const int jb2, const int ka2, const int kb2) const {
    ia = (ia < 0) ? 0 : ia; ib = (ib > nfine1 - 1) ? nfine1 - 1 : ib;
    ia2 = (ia2 < 0) ? 0 : ia2; ib2 = (ib2 > nfine1 - 1) ? nfine1 - 1 : ib2;
    bool ra = true, rb = true;
    const bool qa = kAll ? AllAbove(m, ia, ib, ja, jb, ka, kb, lev, &ra, 1)
                         : AnyAbove(m, ia, ib, ja, jb, ka, kb, lev, &ra, 1);
    const bool qb = kAll ? AllAbove(m, ia2, ib2, ja2, jb2, ka2, kb2, lev, &rb, 2 + side)
                         : AnyAbove(m, ia2, ib2, ja2, jb2, ka2, kb2, lev, &rb, 2 + side);
    if (reach != nullptr) *reach = ra && rb;
    return kAll ? (qa && qb) : (qa || qb);
  }
  // the query of the support box (scale f) of a plain level-lev cloud at the framed position
  template <bool kAll>
  KOKKOS_INLINE_FUNCTION
  bool SupportAbove(const int m, const Real x1, const Real x2, const Real x3, const Real f,
                    const int lev, bool *reach) const {
    int ia, ib, ja, jb, ka, kb;
    SupportBox(x1, x2, x3, f, lev, ia, ib, ja, jb, ka, kb);
    const int side = ShearSide(ia, ib);
    if (side < 0) {
      return kAll ? AllAbove(m, ia, ib, ja, jb, ka, kb, lev, reach)
                  : AnyAbove(m, ia, ib, ja, jb, ka, kb, lev, reach);
    }
    const Real sg = (side == 1) ? 1.0 : -1.0;
    int ia2, ib2, ja2, jb2, ka2, kb2;
    SupportBox(x1 - sg*lx, x2 + sg*ysh, x3, f, lev, ia2, ib2, ja2, jb2, ka2, kb2);
    return AboveParts<kAll>(m, side, lev, reach, ia, ib, ja, jb, ka, kb, ia2, ib2, ja2, jb2, ka2,
                            kb2);
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
    return split_overlap ? SupportAbove<false>(m, x1, x2, x3, 1.0 - slack, lev, reach)
                         : SupportAbove<true>(m, x1, x2, x3, 1.0 - slack, lev, reach);
  }
  KOKKOS_INLINE_FUNCTION
  bool OverlapsFiner(const Real px1, const Real px2, const Real px3, const int code,
                     const int m, bool *reach = nullptr) const {
    if (reach != nullptr) *reach = true;
    const int lev = T::CloudLevel(code);
    if (lev >= nlev) return false;
    Real x1 = px1, x2 = px2, x3 = px3;
    Frame(m, x1, x2, x3);
    return SupportAbove<false>(m, x1, x2, x3, 1.0, lev, reach);
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
    const int side = ShearSide((I - B)*r, (I + B + 1)*r - 1);
    if (side < 0) {
      return !AnyAbove(m, (I - B)*r, (I + B + 1)*r - 1, (J - B)*r, (J + B + 1)*r - 1,
                       (K - B)*r, (K + B + 1)*r - 1, L, reach);
    }
    // across a shear-periodic x1 face: also the cells around the translated position
    const Real sg = (side == 1) ? 1.0 : -1.0;
    const int I2 = static_cast<int>(floor((x1 - sg*lx - x1min)/(root_dx1/static_cast<Real>(1 << L))));
    const int J2 = static_cast<int>(floor((x2 + sg*ysh - x2min)/(root_dx2/static_cast<Real>(1 << L))));
    return !AboveParts<false>(m, side, L, reach, (I - B)*r, (I + B + 1)*r - 1, (J - B)*r,
                              (J + B + 1)*r - 1, (K - B)*r, (K + B + 1)*r - 1, (I2 - B)*r,
                              (I2 + B + 1)*r - 1, (J2 - B)*r, (J2 + B + 1)*r - 1, (K - B)*r,
                              (K + B + 1)*r - 1);
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
