//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_density.cpp
//! \brief The dust mass density as the module deposits it: rho_dust = sum_p m_p W(x_p)/V
//! with the module's PM kernel, the ghost deposits added into the neighbours and the
//! ghost layer of the result filled by a copy exchange (with the shear-periodic remap in
//! a 3D shearing box).  This is the density the drag coupling sees, assembled on demand
//! for diagnostics and outputs (dust_dpm, phst, user histories) outside the stage task
//! lists; nothing in the time step reads it.
//!
//! Ported from the dust-multigrid branch of github.com/jhlim0918/athenak (where it is the
//! Poisson source of the dust self-gravity, dust_gravity.cpp).  Ghost deposits that land
//! across a shear-periodic x1 face are NOT folded back onto the sheared neighbour here
//! (that fold needs the conservative y-remap of the shear-fold work on that branch); they
//! are dropped, so in a 3D shear-periodic box the deposited density is low in the first
//! kernel half-width of the x1 faces.  Periodic and 2D r-z boxes are exact.

#include <iostream>
#include <cstdlib>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box/shearing_box.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

namespace {
void RequireDone(TaskStatus status, const char *operation) {
  if (status == TaskStatus::fail) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Dust density exchange failed during " << operation
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::DepositMass
//! \brief rho_dust = sum_p m_p W(x_p)/V on the active zone plus the deposit halo.  The
//! particles sit inside their owning MeshBlocks (last stage's migration), so the TSC
//! stencil reaches at most one ghost layer.

void DustGasDrag::DepositMass() {
  Kokkos::deep_copy(DevExeSpace(), rho_dust, 0.0);

  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  int npart = ppar->nprtcl_thispack;
  if (npart == 0) {return;}

  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int is = indcs.is, js = indcs.js, ks = indcs.ks;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  bool three_d = pmy_pack->pmesh->three_d;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto gids = pmy_pack->gids;
  int scheme = static_cast<int>(deposit);
  auto &rho = rho_dust;

  par_for("dust_dens_deposit",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int ip, jp, kp;
    Real wx[3], wy[3], wz[3];
    PMWeights(pr(IPX,p), mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, nx1, is,
              scheme, ip, wx);
    PMWeights(pr(IPY,p), mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, nx2, js,
              scheme, jp, wy);
    if (three_d) {
      PMWeights(pr(IPZ,p), mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, nx3, ks,
                scheme, kp, wz);
    } else {
      kp = ks;
      wz[0] = 0.0; wz[1] = 1.0; wz[2] = 0.0;
    }
    Real vol = mbsize.d_view(m).dx1*mbsize.d_view(m).dx2;
    if (three_d) {vol *= mbsize.d_view(m).dx3;}
    Real minv = pr(IPM,p)/vol;
    int clo = three_d ? 0 : 1, chi = three_d ? 2 : 1;
    for (int c=clo; c<=chi; ++c) {
      for (int b=0; b<3; ++b) {
        Real wcb = wz[c]*wy[b]*minv;
        if (wcb == 0.0) continue;
        for (int a=0; a<3; ++a) {
          DepositAdd(&rho(m,0,kp+c-1,jp+b-1,ip+a-1), wcb*wx[a]);
        }
      }
    }
  });
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::AssembleDustDensityNow
//! \brief Synchronous deposit + additive exchange of the ghost deposits + copy exchange
//! of the ghost layer (+ shear remap of the x1 ghosts in a 3D shearing box).  Collective:
//! every rank must call it.

void DustGasDrag::AssembleDustDensityNow() {
  Real time = pmy_pack->pmesh->time;
  DepositMass();
  RequireDone(pbval_rd->InitRecv(1), "density InitRecv");
  RequireDone(pbval_rd->PackAndSendDeposit(rho_dust), "density send");
  TaskStatus status;
  do {
    status = pbval_rd->RecvAndSumDeposit(rho_dust);
    RequireDone(status, "density receive");
  } while (status == TaskStatus::incomplete);
  RequireDone(pbval_rd->ClearSend(), "density ClearSend");
  RequireDone(pbval_rd->ClearRecv(), "density ClearRecv");

  RequireDone(pbval_rc->InitRecv(1), "density-copy InitRecv");
  RequireDone(pbval_rc->PackAndSendCC(rho_dust, cdummy), "density-copy send");
  do {
    status = pbval_rc->RecvAndUnpackCC(rho_dust, cdummy);
    RequireDone(status, "density-copy receive");
  } while (status == TaskStatus::incomplete);
  RequireDone(pbval_rc->ClearSend(), "density-copy ClearSend");
  RequireDone(pbval_rc->ClearRecv(), "density-copy ClearRecv");

  if (psbox_rc != nullptr) {
    RequireDone(psbox_rc->InitRecv(time), "density-shear InitRecv");
    RequireDone(psbox_rc->PackAndSendCC(rho_dust, ReconstructionMethod::plm),
                "density-shear send");
    do {
      status = psbox_rc->RecvAndUnpackCC(rho_dust);
      RequireDone(status, "density-shear receive");
    } while (status == TaskStatus::incomplete);
    RequireDone(psbox_rc->ClearSend(), "density-shear ClearSend");
    RequireDone(psbox_rc->ClearRecv(), "density-shear ClearRecv");
  }
}

}  // namespace dust
