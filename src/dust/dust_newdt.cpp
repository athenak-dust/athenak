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
//! \fn DustGasDrag::RefreshStoppingTimeMaximum
//! \brief Validates the post-pgen particle stopping times and, for particle_static or
//! dynamic mode, obtains their maximum across all ranks.  A maximum-finite-Real sentinel
//! folds validation into the same MAX reduction, so dynamic mode needs only one MPI
//! collective per cycle.  species_fixed and particle_static call this only once.

void DustGasDrag::RefreshStoppingTimeMaximum() {
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &taus_ = taus;
  int npart = ppar->nprtcl_thispack;
  int nspec = nspecies;
  int mode = static_cast<int>(stopping_time_mode);
  int fixed_mode = static_cast<int>(DustStoppingTimeMode::species_fixed);
  Real bad_value = std::numeric_limits<Real>::max();
  Real rel_tol = 64.0*std::numeric_limits<Real>::epsilon();

  Real local_max = 0.0;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_validate_tstop",
    Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, Real &max_tstop) {
      int s = pi(PSP,p);
      Real ts = pr(IPTS,p);
      bool bad = (s < 0 || s >= nspec || !Kokkos::isfinite(ts) || !(ts > 0.0));
      if (!bad && mode == fixed_mode) {
        Real ts_ref = taus_.d_view(s);
        bad = (fabs(ts - ts_ref) > rel_tol*fabs(ts_ref));
      }
      Real candidate = bad ? bad_value : ts;
      max_tstop = fmax(max_tstop, candidate);
    }, Kokkos::Max<Real>(local_max));
  }

#if MPI_PARALLEL_ENABLED
  int ierr = MPI_Allreduce(MPI_IN_PLACE, &local_max, 1, MPI_ATHENA_REAL, MPI_MAX,
                           MPI_COMM_WORLD);
  if (ierr != MPI_SUCCESS) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error while reducing dust stopping times" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  if (local_max == bad_value) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid dust particle stopping-time state: PSP must be in "
              << "[0, nspecies), IPTS must be finite and positive, and species_fixed "
              << "requires IPTS = taus_[PSP]" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (stopping_time_mode != DustStoppingTimeMode::species_fixed && local_max > 0.0) {
    taus_max = local_max;
  }
  stopping_times_initialized = true;
}

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
  if (!stopping_times_initialized ||
      stopping_time_mode == DustStoppingTimeMode::dynamic) {
    RefreshStoppingTimeMaximum();
  }
  if (!gamma_switch) {return TaskStatus::complete;}

  Real gnew = (pmy_pack->pmesh->dt > taus_max) ? 0.5 : 1.0 + 1.0/sqrt(2.0);
  if (gnew != pdrive->gamma) {
    pdrive->SetImEx2PlusCoefficients(gnew);
  }
  return TaskStatus::complete;
}

} // namespace dust
