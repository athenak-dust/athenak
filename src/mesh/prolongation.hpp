#ifndef MESH_PROLONGATION_HPP_
#define MESH_PROLONGATION_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file prolongation.hpp
//! \brief prolongation operators for cell-centered and face-centered variables,
//! implemented as inline functions so they can be used both in Bval and AMR functions.

#include "z4c/z4c.hpp"

// Diagnostic build options: -DATHENAK_PROLONG_UNLIMITED replaces the min-mod limited slope
// of ProlongCC by the unlimited central slope (used to attribute SMR interface errors);
// -DATHENAK_PROLONG_MC uses the monotonized-central slope minmod(2dl, 2dr, (dl+dr)/2).
// Two diagnostic splits of MC's correction relative to the central slope:
// -DATHENAK_PROLONG_MC_EXTCENTRAL: central slope where dl*dr <= 0 (extremum stencils),
//   MC elsewhere; -DATHENAK_PROLONG_MC_MONOCENTRAL: central slope where dl*dr > 0
//   (monotone stencils), zero slope elsewhere.
#define PROLONG_CC_MC(dl, dr) (0.125*(SIGN(dl) + SIGN(dr)) \
                               *fmin(0.5*fabs((dl) + (dr)), 2.0*fmin(fabs(dl), fabs(dr))))
#ifdef ATHENAK_PROLONG_UNLIMITED
#define PROLONG_CC_SLOPE(dl, dr) (0.125*((dl) + (dr)))
#elif defined(ATHENAK_PROLONG_MC)
#define PROLONG_CC_SLOPE(dl, dr) PROLONG_CC_MC(dl, dr)
#elif defined(ATHENAK_PROLONG_MC_EXTCENTRAL)
#define PROLONG_CC_SLOPE(dl, dr) (((dl)*(dr) <= 0.0) ? 0.125*((dl) + (dr)) \
                                                     : PROLONG_CC_MC(dl, dr))
#elif defined(ATHENAK_PROLONG_MC_MONOCENTRAL)
#define PROLONG_CC_SLOPE(dl, dr) (((dl)*(dr) > 0.0) ? 0.125*((dl) + (dr)) : 0.0)
#else
#define PROLONG_CC_SLOPE(dl, dr) (0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr)))
#endif

//----------------------------------------------------------------------------------------
//! \fn ProlongCC()
//! \brief 2nd-order (piecewise-linear) prolongation operator for cell-centered variables

KOKKOS_INLINE_FUNCTION
void ProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi,
               const bool multi_d, const bool three_d,
               const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a) {
  // calculate x1-gradient using the min-mod limiter
  Real dl = ca(m,v,k,j,i  ) - ca(m,v,k,j,i-1);
  Real dr = ca(m,v,k,j,i+1) - ca(m,v,k,j,i  );
  Real dvar1 = PROLONG_CC_SLOPE(dl, dr);

  // calculate x2-gradient using the min-mod limiter
  Real dvar2 = 0.0;
  if (multi_d) {
    dl = ca(m,v,k,j  ,i) - ca(m,v,k,j-1,i);
    dr = ca(m,v,k,j+1,i) - ca(m,v,k,j  ,i);
    dvar2 = PROLONG_CC_SLOPE(dl, dr);
  }

  // calculate x1-gradient using the min-mod limiter
  Real dvar3 = 0.0;
  if (three_d) {
    dl = ca(m,v,k  ,j,i) - ca(m,v,k-1,j,i);
    dr = ca(m,v,k+1,j,i) - ca(m,v,k  ,j,i);
    dvar3 = PROLONG_CC_SLOPE(dl, dr);
  }

  // interpolate to the finer grid
  a(m,v,fk,fj,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
  a(m,v,fk,fj,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
  if (multi_d) {
    a(m,v,fk,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
    a(m,v,fk,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
  }
  if (three_d) {
    a(m,v,fk+1,fj  ,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj  ,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongCCAmplitudeAware()
//! \brief Hydro-only diagnostic candidate (<mesh_refinement>/
//! prolong_amplitude_eps > 0): amplitude-aware blend of the central and MC slopes for the
//! conserved hydro variables of an isothermal static mesh.  Per coarse cell and direction
//! the blend weight w = 1 - S((D - eps)/eps) on [eps, 2 eps] (S the C1 smoothstep, w = 1
//! below, 0 above) uses the Galilean-invariant one-sided jumps D = max over both sides of
//! max(|drho|/rho, |du|_inf/cs), u = M/rho cell by cell; the weight is shared by all
//! variables of the cell.  If a child density would fall below 0.1 rho, all child
//! increments of the cell are scaled by one common factor (coarse mean kept).  With w = 1
//! everywhere the result is bitwise the -DATHENAK_PROLONG_UNLIMITED central slope.
//! Counters (cnt, 3 directions x 10): 0 cell-directions, 1 D <= eps, 2 mixed, 3 D >= 2 eps,
//! 4 variable stencils in the mixed band, 5 of which MC != central, 6 variable stencils in
//! the MC band, 7 of which MC != central, 8 (direction 0 only) cells scaled for positivity.

KOKKOS_INLINE_FUNCTION
Real ProlongAmpJump(const DvceArray5D<Real> &ca, const int m, const int k, const int j,
                    const int i, const int dk, const int dj, const int di, const Real cs) {
  const Real d0 = ca(m,IDN,k,j,i);
  const Real dlo = ca(m,IDN,k-dk,j-dj,i-di);
  const Real dhi = ca(m,IDN,k+dk,j+dj,i+di);
  Real dmax = fmax(fabs(d0 - dlo), fabs(dhi - d0))/d0;
  for (int n=IM1; n<=IM3; ++n) {
    const Real u0 = ca(m,n,k,j,i)/d0;
    const Real ulo = ca(m,n,k-dk,j-dj,i-di)/dlo;
    const Real uhi = ca(m,n,k+dk,j+dj,i+di)/dhi;
    dmax = fmax(dmax, fmax(fabs(u0 - ulo), fabs(uhi - u0))/cs);
  }
  return dmax;
}

KOKKOS_INLINE_FUNCTION
Real ProlongAmpWeight(const Real dj, const Real eps) {
  if (dj <= eps) return 1.0;
  if (dj >= 2.0*eps) return 0.0;
  const Real x = (dj - eps)/eps;
  return 1.0 - x*x*(3.0 - 2.0*x);
}

KOKKOS_INLINE_FUNCTION
void ProlongCCAmplitudeAware(const int m, const int v, const int k, const int j,
                             const int i, const int fk, const int fj, const int fi,
                             const bool multi_d, const bool three_d,
                             const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a,
                             const Real eps, const Real cs,
                             const DvceArray1D<int64_t> &cnt) {
  // blend weights per direction (shared by all variables of the cell)
  const Real dj1 = ProlongAmpJump(ca, m, k, j, i, 0, 0, 1, cs);
  const Real w1 = ProlongAmpWeight(dj1, eps);
  Real dj2 = 0.0, w2 = 1.0, dj3 = 0.0, w3 = 1.0;
  if (multi_d) {dj2 = ProlongAmpJump(ca, m, k, j, i, 0, 1, 0, cs);
                w2 = ProlongAmpWeight(dj2, eps);}
  if (three_d) {dj3 = ProlongAmpJump(ca, m, k, j, i, 1, 0, 0, cs);
                w3 = ProlongAmpWeight(dj3, eps);}

  // blended slope of variable n along (dk,dj,di) with weight w; dvar = slope/4 as in
  // ProlongCC (child offset 1/4 of the coarse cell)
  auto blend = [&](const int n, const int dk, const int dj, const int di, const Real w,
                   const int d, const bool count) {
    const Real dl = ca(m,n,k,j,i) - ca(m,n,k-dk,j-dj,i-di);
    const Real dr = ca(m,n,k+dk,j+dj,i+di) - ca(m,n,k,j,i);
    const Real cen = 0.125*((dl) + (dr));
    const Real mc = PROLONG_CC_MC(dl, dr);
    if (count && w < 1.0) {
      const int b = 10*d + ((w > 0.0) ? 4 : 6);
      Kokkos::atomic_increment(&cnt(b));
      if (mc != cen) Kokkos::atomic_increment(&cnt(b + 1));
    }
    return w*cen + (1.0 - w)*mc;
  };
  Real dvar1 = blend(v, 0, 0, 1, w1, 0, true);
  Real dvar2 = 0.0, dvar3 = 0.0;
  if (multi_d) dvar2 = blend(v, 0, 1, 0, w2, 1, true);
  if (three_d) dvar3 = blend(v, 1, 0, 0, w3, 2, true);

  // positivity: smallest child density with the blended density slopes
  const Real d0 = ca(m,IDN,k,j,i);
  Real ddsum = fabs(blend(IDN, 0, 0, 1, w1, 0, false));
  if (multi_d) ddsum += fabs(blend(IDN, 0, 1, 0, w2, 1, false));
  if (three_d) ddsum += fabs(blend(IDN, 1, 0, 0, w3, 2, false));
  constexpr Real beta = 0.1;
  if (d0 - ddsum < beta*d0) {
    const Real theta = (1.0 - beta)*d0/ddsum;
    dvar1 *= theta; dvar2 *= theta; dvar3 *= theta;
    if (v == IDN) Kokkos::atomic_increment(&cnt(8));
  }
  if (v == IDN) {
    const Real dj[3] = {dj1, dj2, dj3};
    for (int d=0; d<(three_d ? 3 : (multi_d ? 2 : 1)); ++d) {
      Kokkos::atomic_increment(&cnt(10*d));
      Kokkos::atomic_increment(&cnt(10*d + ((dj[d] <= eps) ? 1 : ((dj[d] >= 2.0*eps) ? 3 : 2))));
    }
  }

  // interpolate to the finer grid (same operation order as ProlongCC)
  a(m,v,fk,fj,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 - dvar3;
  a(m,v,fk,fj,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 - dvar3;
  if (multi_d) {
    a(m,v,fk,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 - dvar3;
    a(m,v,fk,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 - dvar3;
  }
  if (three_d) {
    a(m,v,fk+1,fj  ,fi  ) = ca(m,v,k,j,i) - dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj  ,fi+1) = ca(m,v,k,j,i) + dvar1 - dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi  ) = ca(m,v,k,j,i) - dvar1 + dvar2 + dvar3;
    a(m,v,fk+1,fj+1,fi+1) = ca(m,v,k,j,i) + dvar1 + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX1Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X1-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX1Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d, const bool three_d,
                   const DvceArray4D<Real> &cbx1f, const DvceArray4D<Real> &bx1f) {
  // Prolongate b.x1f (v=0) by interpolating in x2/x3
  Real dvar2 = 0.0;
  if (multi_d) {
    Real dl = cbx1f(m,k,j  ,i) - cbx1f(m,k,j-1,i);
    Real dr = cbx1f(m,k,j+1,i) - cbx1f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  Real dvar3 = 0.0;
  if (three_d) {
    Real dl = cbx1f(m,k  ,j,i) - cbx1f(m,k-1,j,i);
    Real dr = cbx1f(m,k+1,j,i) - cbx1f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx1f(m,fk,fj,fi) = cbx1f(m,k,j,i) - dvar2 - dvar3;
  if (multi_d) {
    bx1f(m,fk,fj+1,fi) = cbx1f(m,k,j,i) + dvar2 - dvar3;
  }
  if (three_d) {
    bx1f(m,fk+1,fj  ,fi) = cbx1f(m,k,j,i) - dvar2 + dvar3;
    bx1f(m,fk+1,fj+1,fi) = cbx1f(m,k,j,i) + dvar2 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX2Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X2-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX2Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool three_d,
                   const DvceArray4D<Real> &cbx2f, const DvceArray4D<Real> &bx2f) {
  // Prolongate b.x2f (v=1) by interpolating in x1/x3
  Real dl = cbx2f(m,k,j,i  ) - cbx2f(m,k,j,i-1);
  Real dr = cbx2f(m,k,j,i+1) - cbx2f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar3 = 0.0;
  if (three_d) {
    dl = cbx2f(m,k  ,j,i) - cbx2f(m,k-1,j,i);
    dr = cbx2f(m,k+1,j,i) - cbx2f(m,k  ,j,i);
    dvar3 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx2f(m,fk  ,fj,fi  ) = cbx2f(m,k,j,i) - dvar1 - dvar3;
  bx2f(m,fk  ,fj,fi+1) = cbx2f(m,k,j,i) + dvar1 - dvar3;
  if (three_d) {
    bx2f(m,fk+1,fj,fi  ) = cbx2f(m,k,j,i) - dvar1 + dvar3;
    bx2f(m,fk+1,fj,fi+1) = cbx2f(m,k,j,i) + dvar1 + dvar3;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongFCSharedX3Face()
//! \brief 2nd-order (piecewise-linear) prolongation operator for face-centered variables
//! on shared X3-faces between fine and coarse cells

KOKKOS_INLINE_FUNCTION
void ProlongFCSharedX3Face(const int m, const int k, const int j, const int i,
                   const int fk, const int fj, const int fi,
                   const bool multi_d,
                   const DvceArray4D<Real> &cbx3f, const DvceArray4D<Real> &bx3f) {
  // Prolongate b.x3f (v=2) by interpolating in x1/x2
  Real dl = cbx3f(m,k,j,i  ) - cbx3f(m,k,j,i-1);
  Real dr = cbx3f(m,k,j,i+1) - cbx3f(m,k,j,i  );
  Real dvar1 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));

  Real dvar2 = 0.0;
  if (multi_d) {
    dl = cbx3f(m,k,j  ,i) - cbx3f(m,k,j-1,i);
    dr = cbx3f(m,k,j+1,i) - cbx3f(m,k,j  ,i);
    dvar2 = 0.125*(SIGN(dl) + SIGN(dr))*fmin(fabs(dl), fabs(dr));
  }

  bx3f(m,fk,fj  ,fi  ) = cbx3f(m,k,j,i) - dvar1 - dvar2;
  bx3f(m,fk,fj  ,fi+1) = cbx3f(m,k,j,i) + dvar1 - dvar2;
  if (multi_d) {
    bx3f(m,fk,fj+1,fi  ) = cbx3f(m,k,j,i) - dvar1 + dvar2;
    bx3f(m,fk,fj+1,fi+1) = cbx3f(m,k,j,i) + dvar1 + dvar2;
  }
  return;
}

//----------------------------------------------------------------------------------------
//! \fn ProlongInternalFC()
//! \brief 2nd-order prolongation operator for face-centered variables on internal edges
//! of new fine cells within one coarse cell using divergence-preserving interpolation
//! scheme of Toth & Roe, JCP 180, 736 (2002).

KOKKOS_INLINE_FUNCTION
void ProlongFCInternal(const int m, const int fk, const int fj, const int fi,
                       const bool three_d, const DvceFaceFld4D<Real> &b) {
  // Prolongate internal fields in 3D
  if (three_d) {
    Real Uxx  = 0.0, Vyy  = 0.0, Wzz  = 0.0;
    Real Uxyz = 0.0, Vxyz = 0.0, Wxyz = 0.0;
    for (int jj=0; jj<2; jj++) {
      int jsgn = 2*jj - 1;
      int fjj  = fj + jj, fjp = fj + 2*jj;
      for (int ii=0; ii<2; ii++) {
        int isgn = 2*ii - 1;
        int fii = fi + ii, fip = fi + 2*ii;
        Uxx += isgn*(jsgn*(b.x2f(m,fk  ,fjp,fii) + b.x2f(m,fk+1,fjp,fii)) +
                          (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)));

        Vyy += jsgn*(     (b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii)) +
                     isgn*(b.x1f(m,fk  ,fjj,fip) + b.x1f(m,fk+1,fjj,fip)));

        Wzz +=       isgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip)) +
                     jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));

        Uxyz += isgn*jsgn*(b.x1f(m,fk+1,fjj,fip) - b.x1f(m,fk  ,fjj,fip));
        Vxyz += isgn*jsgn*(b.x2f(m,fk+1,fjp,fii) - b.x2f(m,fk  ,fjp,fii));
        Wxyz += isgn*jsgn*(b.x3f(m,fk+2,fjj,fii) - b.x3f(m,fk  ,fjj,fii));
      }
    }
    Uxx *= 0.125;  Vyy *= 0.125;  Wzz *= 0.125;
    Uxyz *= 0.0625; Vxyz *= 0.0625; Wxyz *= 0.0625;

    b.x1f(m,fk  ,fj  ,fi+1) = 0.5*(b.x1f(m,fk  ,fj  ,fi  ) + b.x1f(m,fk  ,fj  ,fi+2))
                            + Uxx - Vxyz - Wxyz;
    b.x1f(m,fk  ,fj+1,fi+1) = 0.5*(b.x1f(m,fk  ,fj+1,fi  ) + b.x1f(m,fk  ,fj+1,fi+2))
                            + Uxx - Vxyz + Wxyz;
    b.x1f(m,fk+1,fj  ,fi+1) = 0.5*(b.x1f(m,fk+1,fj  ,fi  ) + b.x1f(m,fk+1,fj  ,fi+2))
                            + Uxx + Vxyz - Wxyz;
    b.x1f(m,fk+1,fj+1,fi+1) = 0.5*(b.x1f(m,fk+1,fj+1,fi  ) + b.x1f(m,fk+1,fj+1,fi+2))
                            + Uxx + Vxyz + Wxyz;
    b.x2f(m,fk  ,fj+1,fi  ) = 0.5*(b.x2f(m,fk  ,fj  ,fi  ) + b.x2f(m,fk  ,fj+2,fi  ))
                            + Vyy - Uxyz - Wxyz;
    b.x2f(m,fk  ,fj+1,fi+1) = 0.5*(b.x2f(m,fk  ,fj  ,fi+1) + b.x2f(m,fk  ,fj+2,fi+1))
                            + Vyy - Uxyz + Wxyz;
    b.x2f(m,fk+1,fj+1,fi  ) = 0.5*(b.x2f(m,fk+1,fj  ,fi  ) + b.x2f(m,fk+1,fj+2,fi  ))
                            + Vyy + Uxyz - Wxyz;
    b.x2f(m,fk+1,fj+1,fi+1) = 0.5*(b.x2f(m,fk+1,fj  ,fi+1) + b.x2f(m,fk+1,fj+2,fi+1))
                            + Vyy + Uxyz + Wxyz;
    b.x3f(m,fk+1,fj  ,fi  ) = 0.5*(b.x3f(m,fk+2,fj  ,fi  ) + b.x3f(m,fk  ,fj  ,fi  ))
                            + Wzz - Uxyz - Vxyz;
    b.x3f(m,fk+1,fj  ,fi+1) = 0.5*(b.x3f(m,fk+2,fj  ,fi+1) + b.x3f(m,fk  ,fj  ,fi+1))
                            + Wzz - Uxyz + Vxyz;
    b.x3f(m,fk+1,fj+1,fi  ) = 0.5*(b.x3f(m,fk+2,fj+1,fi  ) + b.x3f(m,fk  ,fj+1,fi  ))
                            + Wzz + Uxyz - Vxyz;
    b.x3f(m,fk+1,fj+1,fi+1) = 0.5*(b.x3f(m,fk+2,fj+1,fi+1) + b.x3f(m,fk  ,fj+1,fi+1))
                            + Wzz + Uxyz + Vxyz;

  // Prolongate internal fields in 2D
  } else {
    Real tmp1 = 0.25*(b.x2f(m,fk,fj+2,fi+1) - b.x2f(m,fk,fj,  fi+1)
                    - b.x2f(m,fk,fj+2,fi  ) + b.x2f(m,fk,fj,  fi  ));
    Real tmp2 = 0.25*(b.x1f(m,fk,fj,  fi  ) - b.x1f(m,fk,fj,  fi+2)
                    - b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2));
    b.x1f(m,fk,fj  ,fi+1) = 0.5*(b.x1f(m,fk,fj,  fi  ) + b.x1f(m,fk,fj,  fi+2)) + tmp1;
    b.x1f(m,fk,fj+1,fi+1) = 0.5*(b.x1f(m,fk,fj+1,fi  ) + b.x1f(m,fk,fj+1,fi+2)) + tmp1;
    b.x2f(m,fk,fj+1,fi  ) = 0.5*(b.x2f(m,fk,fj,  fi  ) + b.x2f(m,fk,fj+2,fi  )) + tmp2;
    b.x2f(m,fk,fj+1,fi+1) = 0.5*(b.x2f(m,fk,fj,  fi+1) + b.x2f(m,fk,fj+2,fi+1)) + tmp2;
  }
  return;
}

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
Real ProlongInterpolation(const int m, const int v, int k, int j, int i,
                            const int nx1, const int nx2, const int nx3,
                            const bool offsetk, const bool offsetj, const bool offseti,
                        const DvceArray5D<Real> &ca, const DualArray3D<Real> &weights) {
  // interpolated value at new grid point
  Real ivals = 0;

  for (int kk=0; kk<NGHOST+1; kk++) {
    for (int jj=0; jj<NGHOST+1; jj++) {
      for (int ii=0; ii<NGHOST+1; ii++) {
        int wghti = (offseti) ? NGHOST-ii : ii;
        int wghtj = (offsetj) ? NGHOST-jj : jj;
        int wghtk = (offsetk) ? NGHOST-kk : kk;
        ivals += weights.d_view(wghtk,wghtj,wghti)*ca(m,v,
                    k-NGHOST/2+kk,j-NGHOST/2+jj,i-NGHOST/2+ii);
      }
    }
  }

  return ivals;
}

//----------------------------------------------------------------------------------------
//! \fn HighOrderProlongCC()
//! \brief high-order prolongation operator for cell-centered variables

template <int NGHOST>
KOKKOS_INLINE_FUNCTION
void HighOrderProlongCC(const int m, const int v, const int k, const int j, const int i,
               const int fk, const int fj, const int fi, const int nx1, const int nx2,
               const int nx3, const DvceArray5D<Real> &ca, const DvceArray5D<Real> &a,
               const DualArray3D<Real> &weights) {
  // stencil size for interpolator
  a(m,v,fk  ,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false,false, ca, weights);
  a(m,v,fk  ,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false,false, true, ca, weights);
  a(m,v,fk  ,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true,false, ca, weights);
  a(m,v,fk  ,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                        false, true, true, ca, weights);
  a(m,v,fk+1,fj  ,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true,false,false, ca, weights);
  a(m,v,fk+1,fj  ,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true,false, true, ca, weights);
  a(m,v,fk+1,fj+1,fi  ) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true, true,false, ca, weights);
  a(m,v,fk+1,fj+1,fi+1) = ProlongInterpolation<NGHOST>(m,v,k,j,i, nx1, nx2, nx3,
                                                         true, true, true, ca, weights);
  return;
}

#endif // MESH_PROLONGATION_HPP_

