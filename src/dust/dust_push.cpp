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
//! replacing Hydro::CopyCons in the combined task list). The particle register copy is
//! fused into ExplicitPush, eliminating a separate full-particle pass. Under imex2+ the
//! two fully-implicit pre-stages of the tableau are dormant (their a_twid rows are zero),
//! so unlike IonNeutral::FirstTwoImpRK no implicit solves are performed here.

TaskStatus DustGasDrag::FirstTwoImpRK(Driver *pdrive, int stage) {
  if (stage != 1) {return TaskStatus::complete;}  // only execute on first stage

  hydro::Hydro *phyd = pmy_pack->phydro;
  Kokkos::deep_copy(DevExeSpace(), phyd->u1, phyd->u0);

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
  int npart = ppar->nprtcl_thispack;

  Real dt = pmy_pack->pmesh->dt;
  Real gam0 = pdrive->gam0[stage-1];
  Real gam1 = pdrive->gam1[stage-1];
  Real beta_dt = (pdrive->beta[stage-1])*dt;
  // history term: only a_twid[2][2] is nonzero for imex2+, consumed at explicit stage 2
  Real atw_dt = (stage == 2) ? (pdrive->a_twid[2][2])*dt : 0.0;
  bool first_stage = (stage == 1);

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

    // Save the beginning-of-cycle state while it is already resident in registers.
    if (first_stage) {
      pr(IPX1,p) = x_old;
      pr(IPY1,p) = y_old;
      pr(IPZ1,p) = z_old;
      pr(IPVX1,p) = vx_old;
      pr(IPVY1,p) = vy_old;
      pr(IPVZ1,p) = vz_old;
    }

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

    // On stage 1, use the old values directly instead of reloading the just-written
    // registers.  Stage 2 reads the persistent beginning-of-cycle state as before.
    Real vx1 = first_stage ? vx_old : pr(IPVX1,p);
    Real vy1 = first_stage ? vy_old : pr(IPVY1,p);
    Real vz1 = first_stage ? vz_old : pr(IPVZ1,p);
    pr(IPVX,p) = gam0*vx_old + gam1*vx1 + beta_dt*fx + atw_dt*pr(IPRX,p);
    pr(IPVY,p) = gam0*vy_old + gam1*vy1 + beta_dt*fy + atw_dt*pr(IPRY,p);
    pr(IPVZ,p) = gam0*vz_old + gam1*vz1 + beta_dt*fz + atw_dt*pr(IPRZ,p);

    // position drift: azimuthal transport includes the background shear -q*Omega*x
    Real ydot = vy_old;
    if (is_sbox && three_d) {ydot -= qshear_*omega0_*x_old;}
    Real x1 = first_stage ? x_old : pr(IPX1,p);
    Real y1 = first_stage ? y_old : pr(IPY1,p);
    pr(IPX,p) = gam0*x_old + gam1*x1 + beta_dt*vx_old;
    pr(IPY,p) = gam0*y_old + gam1*y1 + beta_dt*ydot;
    if (three_d) {
      Real z1 = first_stage ? z_old : pr(IPZ1,p);
      pr(IPZ,p) = gam0*z_old + gam1*z1 + beta_dt*vz_old;
    }
  });

  return TaskStatus::complete;
}

} // namespace dust
