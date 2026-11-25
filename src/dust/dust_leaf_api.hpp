#ifndef DUST_DUST_LEAF_API_HPP_
#define DUST_DUST_LEAF_API_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_leaf_api.hpp
//! \brief One compile-time interface over the two leaf traversals of the refinable clouds,
//! LeafTraversal (2-D, dust.hpp) and LeafTraversal3D (dust_leaf3d.hpp), so that a caller can be
//! written once as a template on the traversal type (the first step of the
//! unification; each traversal keeps its own geometry, keys, slot layout and cache size).
//!
//! LeafApi<LT> gives, for LT = LeafTraversal or LeafTraversal3D:
//!  - kDim, kCache (the leaf weights a kernel keeps from its gather: 25 in 2-D, 27 in 3-D);
//!  - Load(pr, p, x): the cloud position x[0..kDim-1] from the particle arrays (IPX, IPY, IPZ);
//!  - Owner(lt, x, m, &b), ForEachWeight(lt, x, code, m, f): the traversal's own calls;
//!  - Decode(lt, t, b, k, j, i): the leaf cell of an encoded target (2-D: k = 0);
//!  - Measure(lt, b): the cell measure of block slot b in each dimension's own expression (2-D the
//!    area dx1 dx2, 3-D the volume dx1 dx2 dx3), so that a deposit written on the interface
//!    evaluates exactly the same operations as before (CPU bitwise);
//!  - Slot(lt, leaf): the block slot of an encoded leaf cell; WeightSum(lt, x, code, m, f): the
//!    weight loop without moments, returning the sum of the weights (the block checks);
//!  - per direction d: Lo, Dx, Nx (the block box), HalfWidth, FineDx, Lobe; Frame(lt, m, x).
//! Included by dust.hpp inside namespace dust, after the two traversals and the deposit helpers.

// (included by dust.hpp inside namespace dust, after DepositAdd / DepositAddWarp)

template <class LTT> struct LeafApi;

template <>
struct LeafApi<LeafTraversal> {
  using LT = LeafTraversal;
  static constexpr int kDim = 2;
  static constexpr int kCache = CloudGeometry::kCloudWeightCache;
  template <class PR>
  KOKKOS_INLINE_FUNCTION
  static void Load(const PR &pr, const int p, Real x[3]) {
    x[0] = pr(IPX,p); x[1] = pr(IPY,p); x[2] = 0.0;
  }
  KOKKOS_INLINE_FUNCTION
  static int Owner(const LT &lt, const Real x[3], const int m, int *bout = nullptr) {
    return lt.Owner(x[0], x[1], m, bout);
  }
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  static void ForEachWeight(const LT &lt, const Real x[3], const int code, const int m,
                            const Function &function) {
    lt.ForEachCloudWeight(x[0], x[1], code, m, function);
  }
  KOKKOS_INLINE_FUNCTION
  static void Decode(const LT &lt, const int t, int &b, int &k, int &j, int &i) {
    i = t % lt.ncells1;
    const int r = t/lt.ncells1;
    j = r % lt.ncells2;
    b = r/lt.ncells2;
    k = 0;
  }
  KOKKOS_INLINE_FUNCTION
  static Real Measure(const LT &lt, const int b) {return lt.box(b,2)*lt.box(b,3);}
  // the block slot of an encoded leaf cell
  KOKKOS_INLINE_FUNCTION
  static int Slot(const LT &lt, const int leaf) {return leaf/(lt.ncells1*lt.ncells2);}
  // the traversal's weight loop without moments; returns the sum of the weights
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  static Real WeightSum(const LT &lt, const Real x[3], const int code, const int m,
                        const Function &function) {
    return lt.template ForEach<false>(x[0], x[1], code, m, function);
  }
  // per direction d (0-based): the block box (lower face, cell width, cells), the cloud half
  // width of a level, the finest cell width, the lobe of a cloud code
  KOKKOS_INLINE_FUNCTION static Real Lo(const LT &lt, const int b, const int d) {return lt.box(b, d);}
  KOKKOS_INLINE_FUNCTION static Real Dx(const LT &lt, const int b, const int d) {return lt.box(b, 2 + d);}
  KOKKOS_INLINE_FUNCTION static int Nx(const LT &lt, const int d) {return d == 0 ? lt.nx1 : lt.nx2;}
  KOKKOS_INLINE_FUNCTION
  static Real HalfWidth(const LT &lt, const int lev, const int d) {
    return d == 0 ? lt.CloudHalfWidth1(lev) : lt.CloudHalfWidth2(lev);
  }
  KOKKOS_INLINE_FUNCTION
  static Real FineDx(const LT &lt, const int d) {return d == 0 ? lt.fine_dx1 : lt.fine_dx2;}
  KOKKOS_INLINE_FUNCTION
  static int Lobe(const int code, const int d) {
    return d == 0 ? CloudGeometry::CloudLobe1(code) : CloudGeometry::CloudLobe2(code);
  }
  // the periodic image of x next to block m
  KOKKOS_INLINE_FUNCTION
  static void Frame(const LT &lt, const int m, Real x[3]) {lt.Frame(m, x[0], x[1]);}
  // the moment traversal with a callback f(target, w, m1x, m1y, m1z) (2-D: m1z = 0)
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  static void ForEachMoment(const LT &lt, const Real x[3], const int code, const int m,
                            const Function &function) {
    lt.ForEachCloudMoment(x[0], x[1], code, m,
    [&](const int target, const Real w, const Real m1x, const Real m1y) {
      function(target, w, m1x, m1y, 0.0);
    });
  }
  // one leaf's term of the linear-moment gather of the packed field gp (value and central
  // differences, 9 components in 2-D) at the position x, added to u[0..2]; the expressions and
  // their order of the 2-D kernels (PC2 predictor and advance)
  template <class GP>
  KOKKOS_INLINE_FUNCTION
  static void GatherLinear(const LT &lt, const GP &gp, const int target, const Real w,
                           const Real m1x, const Real m1y, const Real, const Real x[3],
                           const int is, const int js, const int ks, Real u[3]) {
    const int ti = target % lt.ncells1;
    const int remainder = target/lt.ncells1;
    const int tj = remainder % lt.ncells2;
    const int tm = remainder/lt.ncells2;
    // the leaf's block slot may be a halo block: its box from lt
    const Real dx1 = lt.box(tm,2), dx2 = lt.box(tm,3);
    const Real xc = lt.box(tm,0) + (ti - is + 0.5)*dx1;
    const Real yc = lt.box(tm,1) + (tj - js + 0.5)*dx2;
    const Real cx = 0.5*(m1x + CloudGeometry::MinImage(x[0] - xc, lt.x1max - lt.x1min)*w)/dx1;
    const Real cy = 0.5*(m1y + CloudGeometry::MinImage(x[1] - yc, lt.x2max - lt.x2min)*w)/dx2;
    u[0] += w*gp(tm,ks,tj,ti,0) + cx*gp(tm,ks,tj,ti,3) + cy*gp(tm,ks,tj,ti,6);
    u[1] += w*gp(tm,ks,tj,ti,1) + cx*gp(tm,ks,tj,ti,4) + cy*gp(tm,ks,tj,ti,7);
    u[2] += w*gp(tm,ks,tj,ti,2) + cx*gp(tm,ks,tj,ti,5) + cy*gp(tm,ks,tj,ti,8);
  }
  // IMEX: the coupled mass m c_p, c_p = h/(t_s + h) (the 2-D expression)
  KOKKOS_INLINE_FUNCTION
  static Real CoupledMass(const Real m, const Real ts, const Real a_dt) {
    return m*(a_dt/(ts + a_dt));
  }
  // IMEX: the position of the stage's gather and deposit (2-D: no orbital-advection shift)
  template <class PR>
  KOKKOS_INLINE_FUNCTION
  static void LoadImplicit(const PR &pr, const int p, const Real, Real x[3]) {
    x[0] = pr(IPX,p); x[1] = pr(IPY,p); x[2] = 0.0;
  }
  // the deposit of n = 3 or 4 components into f(b, 0..n-1, k, j, i): one atomic per component
  template <int N, class F>
  KOKKOS_INLINE_FUNCTION
  static void DepositN(const F &f, const std::size_t, const int, const int b, const int k,
                       const int j, const int i, Real (&v)[N]) {
    for (int c=0; c<N; ++c) DepositAdd(&f(b,c,k,j,i), v[c]);
  }
};

template <>
struct LeafApi<LeafTraversal3D> {
  using LT = LeafTraversal3D;
  static constexpr int kDim = 3;
  static constexpr int kCache = 27;   // (3 x 3 x 3: a same-level cloud)
  template <class PR>
  KOKKOS_INLINE_FUNCTION
  static void Load(const PR &pr, const int p, Real x[3]) {
    x[0] = pr(IPX,p); x[1] = pr(IPY,p); x[2] = pr(IPZ,p);
  }
  KOKKOS_INLINE_FUNCTION
  static int Owner(const LT &lt, const Real x[3], const int m, int *bout = nullptr) {
    return lt.Owner(x[0], x[1], x[2], m, bout);
  }
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  static void ForEachWeight(const LT &lt, const Real x[3], const int code, const int m,
                            const Function &function) {
    lt.ForEachCloudWeight(x[0], x[1], x[2], code, m, function);
  }
  KOKKOS_INLINE_FUNCTION
  static void Decode(const LT &lt, const int t, int &b, int &k, int &j, int &i) {
    lt.Decode(t, b, k, j, i);
  }
  KOKKOS_INLINE_FUNCTION
  static Real Measure(const LT &lt, const int b) {return lt.box(b,3)*lt.box(b,4)*lt.box(b,5);}
  KOKKOS_INLINE_FUNCTION
  static int Slot(const LT &lt, const int leaf) {return leaf/(lt.ncells1*lt.ncells2*lt.ncells3);}
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  static Real WeightSum(const LT &lt, const Real x[3], const int code, const int m,
                        const Function &function) {
    return lt.template ForEach<false>(x[0], x[1], x[2], code, m, function);
  }
  KOKKOS_INLINE_FUNCTION static Real Lo(const LT &lt, const int b, const int d) {return lt.box(b, d);}
  KOKKOS_INLINE_FUNCTION static Real Dx(const LT &lt, const int b, const int d) {return lt.box(b, 3 + d);}
  KOKKOS_INLINE_FUNCTION
  static int Nx(const LT &lt, const int d) {return d == 0 ? lt.nx1 : (d == 1 ? lt.nx2 : lt.nx3);}
  KOKKOS_INLINE_FUNCTION
  static Real HalfWidth(const LT &lt, const int lev, const int d) {
    return d == 0 ? lt.CloudHalfWidth1(lev) : (d == 1 ? lt.CloudHalfWidth2(lev) : lt.CloudHalfWidth3(lev));
  }
  KOKKOS_INLINE_FUNCTION
  static Real FineDx(const LT &lt, const int d) {
    return d == 0 ? lt.fine_dx1 : (d == 1 ? lt.fine_dx2 : lt.fine_dx3);
  }
  KOKKOS_INLINE_FUNCTION
  static int Lobe(const int code, const int d) {
    using T = CloudGeometry;
    return d == 0 ? T::CloudLobe1(code) : (d == 1 ? T::CloudLobe2(code) : T::CloudLobe3(code));
  }
  // (with shear-periodic x1 a shift by n domain lengths in x1 moves x2 by n ysh)
  KOKKOS_INLINE_FUNCTION
  static void Frame(const LT &lt, const int m, Real x[3]) {lt.Frame(m, x[0], x[1], x[2]);}
  template <typename Function>
  KOKKOS_INLINE_FUNCTION
  static void ForEachMoment(const LT &lt, const Real x[3], const int code, const int m,
                            const Function &function) {
    lt.ForEachCloudMoment(x[0], x[1], x[2], code, m, function);
  }
  // (the expressions and their order of the 3-D kernels, 12 packed components)
  template <class GP>
  KOKKOS_INLINE_FUNCTION
  static void GatherLinear(const LT &lt, const GP &gp, const int target, const Real w,
                           const Real m1x, const Real m1y, const Real m1z, const Real x[3],
                           const int is, const int js, const int ks, Real u[3]) {
    using T = CloudGeometry;
    int tb, tk, tj, ti;
    lt.Decode(target, tb, tk, tj, ti);
    const Real dx1 = lt.box(tb,3), dx2 = lt.box(tb,4), dx3 = lt.box(tb,5);
    const Real xc = lt.box(tb,0) + (ti - is + 0.5)*dx1;
    const Real yc = lt.box(tb,1) + (tj - js + 0.5)*dx2;
    const Real zc = lt.box(tb,2) + (tk - ks + 0.5)*dx3;
    const Real cx = 0.5*(m1x + T::MinImage(x[0] - xc, lt.lx)*w)/dx1;
    const Real cy = 0.5*(m1y + T::MinImage(x[1] - yc, lt.ly)*w)/dx2;
    const Real cz = 0.5*(m1z + T::MinImage(x[2] - zc, lt.lz)*w)/dx3;
    u[0] += w*gp(tb,tk,tj,ti,0) + cx*gp(tb,tk,tj,ti,3) + cy*gp(tb,tk,tj,ti,6) + cz*gp(tb,tk,tj,ti,9);
    u[1] += w*gp(tb,tk,tj,ti,1) + cx*gp(tb,tk,tj,ti,4) + cy*gp(tb,tk,tj,ti,7) + cz*gp(tb,tk,tj,ti,10);
    u[2] += w*gp(tb,tk,tj,ti,2) + cx*gp(tb,tk,tj,ti,5) + cy*gp(tb,tk,tj,ti,8) + cz*gp(tb,tk,tj,ti,11);
  }
  // (the 3-D expression)
  KOKKOS_INLINE_FUNCTION
  static Real CoupledMass(const Real m, const Real ts, const Real a_dt) {
    return m*a_dt/(ts + a_dt);
  }
  // (3-D: y + sh x, the gas in its t^n orbital-advection frame)
  template <class PR>
  KOKKOS_INLINE_FUNCTION
  static void LoadImplicit(const PR &pr, const int p, const Real sh, Real x[3]) {
    x[0] = pr(IPX,p); x[1] = pr(IPY,p) + sh*pr(IPX,p); x[2] = pr(IPZ,p);
  }
  // (3-D: the warp-aggregated atomics, DepositAddWarp, keyed by the leaf)
  template <int N, class F>
  KOKKOS_INLINE_FUNCTION
  static void DepositN(const F &f, const std::size_t stride, const int key, const int b,
                       const int k, const int j, const int i, Real (&v)[N]) {
    DepositAddWarp(&f(b,0,k,j,i), stride, key, v);
  }
};


//----------------------------------------------------------------------------------------
// the leaf deposit of the clouds' density (component 0) and momentum density (1-3)
// written once for both traversals: the science fields (DepositScienceFields), the 3-D
// DepositCloudDensity3D and the density triggers (2-D DensityRefinement, 3-D with the capped
// kernel).  The per-leaf value d = m w / (cell measure) in each dimension's own expression
// (LeafApi::Measure); f the dimension's field: a flat leaf field f(c, leaf) (2-D) or f(b,c,k,j,i).
// skip_m0 (2-D science): massless slots skipped; kSkipW0 (2-D trigger): zero weights skipped;
// species >= 0: that species alone; nc 1 (density only) or 4; codef: the cloud code of the
// deposit (CloudCodeSame, or CloudCodeCapped for the trigger's capped kernel).

struct CloudCodeSame {
  KOKKOS_INLINE_FUNCTION int operator()(const int code) const {return code;}
};
struct CloudCodeCapped {
  int lcap;
  KOKKOS_INLINE_FUNCTION int operator()(const int code) const {return TriggerCloudCode(code, lcap);}
};
KOKKOS_INLINE_FUNCTION
void CloudFieldAdd(const DvceArray2D<Real> &f, const int c, const int t, const int, const int,
                   const int, const int, const Real v) {
  DepositAdd(&f(c,t), v);
}
KOKKOS_INLINE_FUNCTION
void CloudFieldAdd(const DvceArray5D<Real> &f, const int c, const int, const int b, const int k,
                   const int j, const int i, const Real v) {
  DepositAdd(&f(b,c,k,j,i), v);
}

template <class LTT, bool kSkipW0, class F, class CF>
void CloudFieldDepositT(const std::string &label, const LTT &lt, const DvceArray2D<Real> &pr,
                        const DvceArray2D<int> &pi, const int np, const int species,
                        const bool skip_m0, const int nc, const F &f, const CF &codef) {
  using A = LeafApi<LTT>;
  const LTT ltc = lt;
  auto ff = f;
  Kokkos::parallel_for(label, Kokkos::RangePolicy<>(DevExeSpace(), 0, np),
  KOKKOS_LAMBDA(const int p) {
    const Real m = pr(IPM,p);
    if (skip_m0 && m == 0.0) return;
    if (species >= 0 && pi(PSP,p) != species) return;
    const Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
    Real x[3];
    A::Load(pr, p, x);
    A::ForEachWeight(ltc, x, codef(pi(PLEV,p)), ltc.Anchor(pi(PGID,p)),
    [&](const int target, const Real w) {
      if (kSkipW0 && w == 0.0) return;
      int tb, tk, tj, ti;
      A::Decode(ltc, target, tb, tk, tj, ti);
      const Real d = m*w/A::Measure(ltc, tb);
      CloudFieldAdd(ff, 0, target, tb, tk, tj, ti, d);
      if (nc == 1) return;
      CloudFieldAdd(ff, 1, target, tb, tk, tj, ti, d*vx);
      CloudFieldAdd(ff, 2, target, tb, tk, tj, ti, d*vy);
      CloudFieldAdd(ff, 3, target, tb, tk, tj, ti, d*vz);
    });
  });
}


#endif  // DUST_DUST_LEAF_API_HPP_
