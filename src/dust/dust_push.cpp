//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_push.cpp
//! \brief Explicit (non-drag) part of the dust particle update: RK register copies at
//! the start of each cycle, and the per-stage rotation/shear kick + position drift.
//! All velocities are measured relative to the background shear flow -q*Omega*x, so the
//! rotational force is (2*Omega*v_y', -(2-q)*Omega*v_x, -Omega^2*z) — identical in form
//! to the gas source terms in ShearingBoxCC::SourceTermsCC — and the position drift is
//! dx/dt = v - q*Omega*x in the azimuthal direction. In the 2D r-z shearing box the
//! azimuthal velocity component is IPVZ (matching gas IM3) and there is no vertical
//! gravity, again matching the gas source terms.

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::FirstTwoImpRK
//! \brief On stage 1, copies the gas conserved variables into the RK register (u1 <- u0,
//! replacing Hydro::CopyCons in the combined task list) and the particle phase-space
//! coordinates into the particle registers ((x1,v1) <- (x,v)). Under imex2+ the two
//! fully-implicit pre-stages of the tableau are dormant (their a_twid rows are zero), so
//! unlike IonNeutral::FirstTwoImpRK no implicit solves are performed here.

TaskStatus DustGasDrag::FirstTwoImpRK(Driver *pdrive, int stage) {
  hydro::Hydro *phyd = pmy_pack->phydro;
  if (coupling == DustCoupling::hybrid && stage == 2) {
    // Stage 1 has completed its full boundary tail.  Form the conserved midpoint gas
    // velocity before RKUpdate(2) overwrites u0.  Include ghosts so PC2 needs no second
    // mesh exchange for its midpoint gather.
    if (hybrid_mode == HybridMode::pc2) {
      int nmb1 = pmy_pack->nmb_thispack - 1;
      int ni = phyd->u0.extent_int(4);
      int nj = phyd->u0.extent_int(3);
      int nk = phyd->u0.extent_int(2);
      auto &u0 = phyd->u0;
      auto &u1 = phyd->u1;
      auto &ustar_ = ustar;
      par_for("dust_gas_midpoint",DevExeSpace(),0,nmb1,0,nk-1,0,nj-1,0,ni-1,
      KOKKOS_LAMBDA(const int m, const int k, const int j, const int i) {
        Real inv_rho_sum = 1.0/(u1(m,IDN,k,j,i) + u0(m,IDN,k,j,i));
        ustar_(m,0,k,j,i) = (u1(m,IM1,k,j,i) + u0(m,IM1,k,j,i))*inv_rho_sum;
        ustar_(m,1,k,j,i) = (u1(m,IM2,k,j,i) + u0(m,IM2,k,j,i))*inv_rho_sum;
        ustar_(m,2,k,j,i) = (u1(m,IM3,k,j,i) + u0(m,IM3,k,j,i))*inv_rho_sum;
      });
      ++hybrid_pc2_cycles;
    } else {
      ++hybrid_split_be_cycles;
    }
    return TaskStatus::complete;
  }

  if (stage != 1) {return TaskStatus::complete;}  // register save is stage 1 only

  Kokkos::deep_copy(DevExeSpace(), phyd->u1, phyd->u0);

  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  par_for("dust_reg_copy",DevExeSpace(),0,(ppar->nprtcl_thispack-1),
  KOKKOS_LAMBDA(const int p) {
    pr(IPX1,p)  = pr(IPX,p);
    pr(IPVX1,p) = pr(IPVX,p);
    pr(IPY1,p)  = pr(IPY,p);
    pr(IPVY1,p) = pr(IPVY,p);
    pr(IPZ1,p)  = pr(IPZ,p);
    pr(IPVZ1,p) = pr(IPVZ,p);
  });

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ExplicitPush
//! \brief Low-storage RK update of particle velocities and positions with the explicit
//! forces, evaluated from the pre-stage state:
//!   v <- gam0*v + gam1*v1 + beta*dt*F_rot(v_old) + a_twid[2][2]*dt*R_j  (stage 2 only)
//!   x <- gam0*x + gam1*x1 + beta*dt*(v_old - q*Omega*x_old yhat)
//! The a_twid term applies the drag rate recorded at the previous implicit stage, which
//! is the only nonzero history contribution of the imex2+ tableau.

TaskStatus DustGasDrag::ExplicitPush(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}

  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;

  if (coupling == DustCoupling::hybrid) {
    // Both hybrid branches update particles exactly once, during RK stage 2.  PC2 does
    // the midpoint gather/update/actual-impulse scatter here.  Split-BE prepares v* and
    // Q/P in the same traversal; its relaxed kick and drift occur later.
    if (stage != 2) return TaskStatus::complete;

    Real dt = pmy_pack->pmesh->dt;
    bool three_d = pmy_pack->pmesh->three_d;
    bool br = back_reaction;
    bool is_sbox = is_shearing_box;
    bool is_strat = is_stratified;
    Real qshear_ = qshear;
    Real omega0_ = omega0;
    auto &indcs = pmy_pack->pmesh->mb_indcs;
    int is = indcs.is, js = indcs.js, ks = indcs.ks;
    int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
    auto &mbsize = pmy_pack->pmb->mb_size;
    auto gids = pmy_pack->gids;
    int scheme = static_cast<int>(deposit);

    if (hybrid_mode == HybridMode::pc2) {
      if (br) Kokkos::deep_copy(DevExeSpace(), dmom, 0.0);
      auto &ustar_ = ustar;
      auto &dmom_ = dmom;
      par_for("dust_pc2_advance",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
        int m = pi(PGID,p) - gids;
        Real x0 = pr(IPX1,p), y0 = pr(IPY1,p), z0 = pr(IPZ1,p);
        Real vx0 = pr(IPVX1,p), vy0 = pr(IPVY1,p), vz0 = pr(IPVZ1,p);

        // First-order midpoint position predictor, including background shear transport.
        Real xh = x0 + 0.5*dt*vx0;
        Real yh = y0 + 0.5*dt*(vy0 - ((is_sbox && three_d) ?
                                      qshear_*omega0_*x0 : 0.0));
        Real zh = three_d ? z0 + 0.5*dt*vz0 : z0;

        int ip, jp, kp;
        Real wx[3], wy[3], wz[3];
        PMWeights(xh, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
                  scheme, ip, wx);
        PMWeights(yh, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
                  scheme, jp, wy);
        if (three_d) {
          PMWeights(zh, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks,
                    scheme, kp, wz);
        } else {
          kp = ks;
          wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
        }

        Real ugx = 0.0, ugy = 0.0, ugz = 0.0;
        int clo = three_d ? 0 : 1, chi = three_d ? 2 : 1;
        for (int c=clo; c<=chi; ++c) {
          for (int b=0; b<3; ++b) {
            Real wcb = wz[c]*wy[b];
            if (wcb == 0.0) continue;
            for (int a=0; a<3; ++a) {
              Real w = wcb*wx[a];
              int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
              ugx += w*ustar_(m,0,kk,jj,ii);
              ugy += w*ustar_(m,1,kk,jj,ii);
              ugz += w*ustar_(m,2,kk,jj,ii);
            }
          }
        }

        // Solve the implicit-midpoint velocity equations.  The shearing-box Coriolis
        // pair is a 2x2 linear system; vertical gravity uses the predicted midpoint.
        Real h = 0.5*dt;
        Real lambda = h/pr(IPTS,p);
        Real aa = 1.0 + lambda;
        Real rx = vx0 + lambda*ugx;
        Real ry = vy0 + lambda*ugy;
        Real rz = vz0 + lambda*ugz;
        Real vxh, vyh, vzh;
        if (is_sbox && three_d) {
          Real cxy = 2.0*h*omega0_;
          Real cyx = h*(2.0-qshear_)*omega0_;
          Real det = aa*aa + cxy*cyx;
          vxh = (aa*rx + cxy*ry)/det;
          vyh = (aa*ry - cyx*rx)/det;
          if (is_strat) rz -= h*SQR(omega0_)*zh;
          vzh = rz/aa;
        } else if (is_sbox) {
          Real cxz = 2.0*h*omega0_;
          Real czx = h*(2.0-qshear_)*omega0_;
          Real det = aa*aa + cxz*czx;
          vxh = (aa*rx + cxz*rz)/det;
          vzh = (aa*rz - czx*rx)/det;
          vyh = ry/aa;
        } else {
          vxh = rx/aa;
          vyh = ry/aa;
          vzh = rz/aa;
        }

        pr(IPVX,p) = 2.0*vxh - vx0;
        pr(IPVY,p) = 2.0*vyh - vy0;
        pr(IPVZ,p) = 2.0*vzh - vz0;
        pr(IPX,p) = x0 + dt*vxh;
        pr(IPY,p) = y0 + dt*(vyh - ((is_sbox && three_d) ?
                                    qshear_*omega0_*xh : 0.0));
        if (three_d) pr(IPZ,p) = z0 + dt*vzh;

        // Preserve the midpoint for diagnostics and deposit the direct actual drag
        // impulse, never a residual inferred from the total velocity change.
        pr(IPX1,p) = xh; pr(IPY1,p) = yh; pr(IPZ1,p) = zh;
        Real dvx_drag = dt*(ugx-vxh)/pr(IPTS,p);
        Real dvy_drag = dt*(ugy-vyh)/pr(IPTS,p);
        Real dvz_drag = dt*(ugz-vzh)/pr(IPTS,p);
        pr(IPRX,p) = dvx_drag;
        pr(IPRY,p) = dvy_drag;
        pr(IPRZ,p) = dvz_drag;

        if (br) {
          Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
          if (three_d) vol *= mbsize.d_view(m).dx3;
          Real fac = -pr(IPM,p)/vol;
          for (int c=clo; c<=chi; ++c) {
            for (int b=0; b<3; ++b) {
              Real wcb = wz[c]*wy[b]*fac;
              if (wcb == 0.0) continue;
              for (int a=0; a<3; ++a) {
                Real w = wcb*wx[a];
                int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
                DepositAdd(&dmom_(m,0,kk,jj,ii), w*dvx_drag);
                DepositAdd(&dmom_(m,1,kk,jj,ii), w*dvy_drag);
                DepositAdd(&dmom_(m,2,kk,jj,ii), w*dvz_drag);
              }
            }
          }
        }
      });
    } else {
      if (br) Kokkos::deep_copy(DevExeSpace(), qdep, 0.0);
      auto &qdep_ = qdep;
      par_for("dust_be_prepare",DevExeSpace(),0,(npart-1),
      KOKKOS_LAMBDA(const int p) {
        Real x0 = pr(IPX1,p), y0 = pr(IPY1,p), z0 = pr(IPZ1,p);
        Real vx0 = pr(IPVX1,p), vy0 = pr(IPVY1,p), vz0 = pr(IPVZ1,p);
        Real fx = 0.0, fy = 0.0, fz = 0.0;
        if (is_sbox) {
          if (three_d) {
            fx = 2.0*omega0_*vy0;
            fy = -(2.0-qshear_)*omega0_*vx0;
            if (is_strat) fz = -SQR(omega0_)*z0;
          } else {
            fx = 2.0*omega0_*vz0;
            fz = -(2.0-qshear_)*omega0_*vx0;
          }
        }
        pr(IPVX,p) = vx0 + dt*fx;
        pr(IPVY,p) = vy0 + dt*fy;
        pr(IPVZ,p) = vz0 + dt*fz;
        pr(IPX,p) = x0; pr(IPY,p) = y0; pr(IPZ,p) = z0;

        if (!br) return;
        int m = pi(PGID,p) - gids;
        int ip, jp, kp;
        Real wx[3], wy[3], wz[3];
        PMWeights(x0, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
                  scheme, ip, wx);
        PMWeights(y0, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
                  scheme, jp, wy);
        if (three_d) {
          PMWeights(z0, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks,
                    scheme, kp, wz);
        } else {
          kp = ks;
          wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
        }
        Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
        if (three_d) vol *= mbsize.d_view(m).dx3;
        Real cj = dt/(pr(IPTS,p) + dt);
        Real muc = pr(IPM,p)*cj/vol;
        Real rate = pr(IPM,p)/(pr(IPTS,p)*vol);
        int clo = three_d ? 0 : 1, chi = three_d ? 2 : 1;
        for (int c=clo; c<=chi; ++c) {
          for (int b=0; b<3; ++b) {
            Real wcb = wz[c]*wy[b];
            if (wcb == 0.0) continue;
            for (int a=0; a<3; ++a) {
              Real w = wcb*wx[a];
              int kk = kp+c-1, jj = jp+b-1, ii = ip+a-1;
              DepositAdd(&qdep_(m,0,kk,jj,ii), w*muc);
              DepositAdd(&qdep_(m,1,kk,jj,ii), w*muc*pr(IPVX,p));
              DepositAdd(&qdep_(m,2,kk,jj,ii), w*muc*pr(IPVY,p));
              DepositAdd(&qdep_(m,3,kk,jj,ii), w*muc*pr(IPVZ,p));
              DepositAdd(&qdep_(m,4,kk,jj,ii), w*rate);
            }
          }
        }
      });
    }
    return TaskStatus::complete;
  }

  Real dt = pmy_pack->pmesh->dt;
  Real gam0 = pdrive->gam0[stage-1];
  Real gam1 = pdrive->gam1[stage-1];
  Real beta_dt = (pdrive->beta[stage-1])*dt;
  // history term: only a_twid[2][2] is nonzero for imex2+, consumed at explicit stage 2
  Real atw_dt = (stage == 2) ? (pdrive->a_twid[2][2])*dt : 0.0;

  bool three_d = pmy_pack->pmesh->three_d;
  bool is_sbox = is_shearing_box;
  bool is_strat = is_stratified;
  Real qshear_ = qshear;
  Real omega0_ = omega0;

  par_for("dust_push",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    Real x_old  = pr(IPX,p);
    Real y_old  = pr(IPY,p);
    Real z_old  = pr(IPZ,p);
    Real vx_old = pr(IPVX,p);
    Real vy_old = pr(IPVY,p);
    Real vz_old = pr(IPVZ,p);

    // rotational/shear forces from pre-stage velocities (shear-relative frame)
    Real fx = 0.0, fy = 0.0, fz = 0.0;
    if (is_sbox) {
      if (three_d) {
        // 3D: azimuthal = y
        fx = 2.0*omega0_*vy_old;
        fy = -(2.0-qshear_)*omega0_*vx_old;
        if (is_strat) {fz = -SQR(omega0_)*z_old;}
      } else {
        // 2D r-z: azimuthal = z (matching gas IM3), no vertical gravity
        fx = 2.0*omega0_*vz_old;
        fz = -(2.0-qshear_)*omega0_*vx_old;
      }
    }

    pr(IPVX,p) = gam0*vx_old + gam1*pr(IPVX1,p) + beta_dt*fx + atw_dt*pr(IPRX,p);
    pr(IPVY,p) = gam0*vy_old + gam1*pr(IPVY1,p) + beta_dt*fy + atw_dt*pr(IPRY,p);
    pr(IPVZ,p) = gam0*vz_old + gam1*pr(IPVZ1,p) + beta_dt*fz + atw_dt*pr(IPRZ,p);

    // position drift: azimuthal transport includes the background shear -q*Omega*x
    Real ydot = vy_old;
    if (is_sbox && three_d) {ydot -= qshear_*omega0_*x_old;}
    pr(IPX,p) = gam0*x_old + gam1*pr(IPX1,p) + beta_dt*vx_old;
    pr(IPY,p) = gam0*y_old + gam1*pr(IPY1,p) + beta_dt*ydot;
    if (three_d) {
      pr(IPZ,p) = gam0*z_old + gam1*pr(IPZ1,p) + beta_dt*vz_old;
    }
  });

  return TaskStatus::complete;
}

} // namespace dust
