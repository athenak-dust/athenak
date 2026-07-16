//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_newdt.cpp
//! \brief Dust particle transport timestep and the gamma-switch of Krapp et al. (2024)
//! Eq. (18). There is NO timestep constraint from the drag force itself for any stopping
//! time or dust-to-gas ratio; only particle transport (so that deposit stencils stay
//! within the ghost zones) limits dt.

#include <math.h>

#include <limits>
#include <algorithm>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::NewTimeStep
//! \brief Computes the minimum particle transport timestep min(dx/|v_transport|) over
//! all particles. In the 3D shearing box the azimuthal transport velocity includes the
//! background shear (particles are not orbital-advected): vy_transport = vy - q*Omega*x.
//! The particle CFL number dt_cfl is folded in here since Mesh::NewTimeStep applies no
//! CFL factor to ppart->dtnew.

TaskStatus DustGasDrag::NewTimeStep(Driver *pdrive, int stage) {
  if (stage != (pdrive->nexp_stages)) {
    return TaskStatus::complete;  // only execute on last stage
  }

  particles::Particles *ppar = pmy_pack->ppart;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) {
    ppar->dtnew = std::numeric_limits<float>::max();
    return TaskStatus::complete;
  }

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto gids = pmy_pack->gids;
  bool three_d = pmy_pack->pmesh->three_d;
  bool shear3d = is_shearing_box && three_d;
  Real qo = qshear*omega0;

  Real dtp = std::numeric_limits<float>::max();
  Kokkos::parallel_reduce("dust_newdt",Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
  KOKKOS_LAMBDA(const int &p, Real &min_dt) {
    int m = pi(PGID,p) - gids;
    Real vy = pr(IPVY,p);
    if (shear3d) {vy -= qo*pr(IPX,p);}
    min_dt = fmin(mbsize.d_view(m).dx1/fmax(fabs(pr(IPVX,p)), 1.0e-30), min_dt);
    min_dt = fmin(mbsize.d_view(m).dx2/fmax(fabs(vy), 1.0e-30), min_dt);
    if (three_d) {
      min_dt = fmin(mbsize.d_view(m).dx3/fmax(fabs(pr(IPVZ,p)), 1.0e-30), min_dt);
    }
  }, Kokkos::Min<Real>(dtp));

  ppar->dtnew = dt_cfl*dtp;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::GammaSwitch
//! \brief Once per cycle (before the time integrator), switches the imex2+ singly-
//! diagonal coefficient between gamma = 1+1/sqrt(2) (monotone, most accurate for
//! resolved drag, dt <~ t_s) and gamma = 1/2 (second-order asymptotic convergence in the
//! stiff regime dt >> t_s), per Krapp et al. (2024) Eq. (18). Host-side coefficient
//! rebuild is safe because all tasks read the Driver members per invocation, and the
//! recorded drag rates are consumed strictly within one cycle.

TaskStatus DustGasDrag::GammaSwitch(Driver *pdrive, int stage) {
  if (!gamma_switch) {return TaskStatus::complete;}

  Real gnew = (pmy_pack->pmesh->dt > taus_max) ? 0.5 : 1.0 + 1.0/sqrt(2.0);
  if (gnew != pdrive->gamma) {
    pdrive->SetImEx2PlusCoefficients(gnew);
  }
  return TaskStatus::complete;
}

} // namespace dust
