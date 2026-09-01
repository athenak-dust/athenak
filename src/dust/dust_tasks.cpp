//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_tasks.cpp
//! \brief functions that control DustGasDrag tasks stored in tasklists in MeshBlockPack.
//! Like IonNeutral, this module assembles the COMBINED gas+dust task graph: when a
//! <dust> block is present, Hydro::AssembleHydroTasks and Particles::AssembleTasks are
//! not called, and every Hydro/Particles task is inserted here instead. Particle
//! migration runs INSIDE each stage (positions change every stage), and the implicit
//! drag solve (deposit -> halo add -> cell solve -> u* ghost fill -> gather/kick/PMBR ->
//! halo add -> apply) sits between the explicit gas update and the boundary
//! communication tail, mirroring the placement of IonNeutral::ImpRKUpdate.

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "shearing_box/shearing_box.hpp"
#include "dust.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn void DustGasDrag::AssembleDustGasDragTasks
//! \brief Adds dust+hydro tasks to appropriate task lists used by time integrators.
//! Called by MeshBlockPack::AddPhysics() function directly after DustGasDrag constructor.

void DustGasDrag::AssembleDustGasDragTasks(
    std::map<std::string, std::shared_ptr<TaskList>> tl) {
  TaskID none(0);
  using hydro::Hydro;
  Hydro *phyd = pmy_pack->phydro;

  // assemble "before_timeintegrator" task list
  id.gswitch = tl["before_timeintegrator"]->AddTask(&DustGasDrag::GammaSwitch,this,none);

  // assemble "before_stagen" task list
  id.h_irecv = tl["before_stagen"]->AddTask(&Hydro::InitRecv, phyd, none);
  id.irecvd  = tl["before_stagen"]->AddTask(&DustGasDrag::InitRecvDep, this, none);

  // assemble "stagen" task list
  // register copies (replaces Hydro::CopyCons), then the explicit gas chain
  id.first2    = tl["stagen"]->AddTask(&DustGasDrag::FirstTwoImpRK, this, none);
  id.h_flux    = tl["stagen"]->AddTask(&Hydro::Fluxes, phyd, id.first2);
  id.h_sendf   = tl["stagen"]->AddTask(&Hydro::SendFlux, phyd, id.h_flux);
  id.h_recvf   = tl["stagen"]->AddTask(&Hydro::RecvFlux, phyd, id.h_sendf);
  id.h_rkupdt  = tl["stagen"]->AddTask(&Hydro::RKUpdate, phyd, id.h_recvf);
  id.h_srctrms = tl["stagen"]->AddTask(&Hydro::HydroSrcTerms, phyd, id.h_rkupdt);
  // gas drag-rate history (stage 2 only); must precede the implicit solve
  id.gatwid    = tl["stagen"]->AddTask(&DustGasDrag::AddDragHistoryGas, this,
                                       id.h_srctrms);
  // explicit particle push + per-stage migration; overlaps with the gas chain
  id.push      = tl["stagen"]->AddTask(&DustGasDrag::ExplicitPush, this, id.first2);
  id.p_newgid  = tl["stagen"]->AddTask(&DustGasDrag::UpdateParticleGIDs, this, id.push);
  id.p_cnt     = tl["stagen"]->AddTask(&DustGasDrag::CountParticleSends, this,
                                        id.p_newgid);
  id.p_irecv   = tl["stagen"]->AddTask(&DustGasDrag::InitParticleRecv, this, id.p_cnt);
  id.p_sendp   = tl["stagen"]->AddTask(&DustGasDrag::SendParticles, this, id.p_irecv);
  id.p_recvp   = tl["stagen"]->AddTask(&DustGasDrag::RecvParticles, this, id.p_sendp);
  // implicit drag solve
  id.scat      = tl["stagen"]->AddTask(&DustGasDrag::DepositDrag, this, id.p_recvp);
  id.sendd     = tl["stagen"]->AddTask(&DustGasDrag::SendDepQP, this, id.scat);
  id.recvd     = tl["stagen"]->AddTask(&DustGasDrag::RecvDepQP, this, id.sendd);
  TaskID dep_solve = (id.recvd | id.gatwid);
  id.solve     = tl["stagen"]->AddTask(&DustGasDrag::GasImplicitSolve, this, dep_solve);
  id.sendus    = tl["stagen"]->AddTask(&DustGasDrag::SendUstar, this, id.solve);
  id.recvus    = tl["stagen"]->AddTask(&DustGasDrag::RecvUstar, this, id.sendus);
  // shear-periodic remap of the u* x1 ghost zones replaces the plain-periodic fill
  // (no-ops unless a 3D shearing box with shear-periodic x1 is active)
  id.sendus_shr = tl["stagen"]->AddTask(&DustGasDrag::SendUstarShr, this, id.recvus);
  id.recvus_shr = tl["stagen"]->AddTask(&DustGasDrag::RecvUstarShr, this,
                                        id.sendus_shr);
  id.gkp       = tl["stagen"]->AddTask(&DustGasDrag::GatherKickPMBR, this,
                                       id.recvus_shr);
  id.sendbr    = tl["stagen"]->AddTask(&DustGasDrag::SendPMBR, this, id.gkp);
  id.recvbr    = tl["stagen"]->AddTask(&DustGasDrag::RecvPMBR, this, id.sendbr);
  id.apply     = tl["stagen"]->AddTask(&DustGasDrag::ApplyPMBR, this, id.recvbr);
  // standard hydro tail (boundary comms, BCs, cons-to-prim, timestep)
  id.h_sendu_oa  = tl["stagen"]->AddTask(&Hydro::SendU_OA, phyd, id.apply);
  id.h_recvu_oa  = tl["stagen"]->AddTask(&Hydro::RecvU_OA, phyd, id.h_sendu_oa);
  id.h_restu     = tl["stagen"]->AddTask(&Hydro::RestrictU, phyd, id.h_recvu_oa);
  id.h_sendu     = tl["stagen"]->AddTask(&Hydro::SendU, phyd, id.h_restu);
  id.h_recvu     = tl["stagen"]->AddTask(&Hydro::RecvU, phyd, id.h_sendu);
  id.h_sendu_shr = tl["stagen"]->AddTask(&Hydro::SendU_Shr, phyd, id.h_recvu);
  id.h_recvu_shr = tl["stagen"]->AddTask(&Hydro::RecvU_Shr, phyd, id.h_sendu_shr);
  id.h_bcs       = tl["stagen"]->AddTask(&Hydro::ApplyPhysicalBCs, phyd, id.h_recvu_shr);
  id.h_prol      = tl["stagen"]->AddTask(&Hydro::Prolongate, phyd, id.h_bcs);
  id.h_c2p       = tl["stagen"]->AddTask(&Hydro::ConToPrim, phyd, id.h_prol);
  id.h_newdt     = tl["stagen"]->AddTask(&Hydro::NewTimeStep, phyd, id.h_c2p);
  id.newdt       = tl["stagen"]->AddTask(&DustGasDrag::NewTimeStep, this, id.gkp);

  // assemble "after_stagen" task list
  id.h_csend = tl["after_stagen"]->AddTask(&Hydro::ClearSend, phyd, none);
  id.h_crecv = tl["after_stagen"]->AddTask(&Hydro::ClearRecv, phyd, id.h_csend);
  id.p_csend = tl["after_stagen"]->AddTask(&DustGasDrag::ClearParticleSend, this, none);
  id.p_crecv = tl["after_stagen"]->AddTask(&DustGasDrag::ClearParticleRecv, this,
                                            id.p_csend);
  id.cleard  = tl["after_stagen"]->AddTask(&DustGasDrag::ClearDep, this, none);

  return;
}

//----------------------------------------------------------------------------------------
//! \brief Particle migration wrappers for the combined dust task graph.  The final
//! imex2+ stage only assembles the explicit solution and does not move particles, so a
//! third full GID scan and its MPI bookkeeping have no matching work to perform.

TaskStatus DustGasDrag::UpdateParticleGIDs(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pmy_pack->ppart->NewGID(pdrive, stage);
}

TaskStatus DustGasDrag::CountParticleSends(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pmy_pack->ppart->SendCnt(pdrive, stage);
}

TaskStatus DustGasDrag::InitParticleRecv(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pmy_pack->ppart->InitRecv(pdrive, stage);
}

TaskStatus DustGasDrag::SendParticles(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pmy_pack->ppart->SendP(pdrive, stage);
}

TaskStatus DustGasDrag::RecvParticles(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pmy_pack->ppart->RecvP(pdrive, stage);
}

TaskStatus DustGasDrag::ClearParticleSend(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pmy_pack->ppart->ClearSend(pdrive, stage);
}

TaskStatus DustGasDrag::ClearParticleRecv(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pmy_pack->ppart->ClearRecv(pdrive, stage);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::InitRecvDep
//! \brief Posts non-blocking receives for the three per-stage dust exchanges. No-op on
//! the dormant assembly stage (no matching sends would be posted).

TaskStatus DustGasDrag::InitRecvDep(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}

  TaskStatus tstat = TaskStatus::complete;
  if (back_reaction) {
    tstat = pbval_qp->InitRecv(4);
    if (tstat != TaskStatus::complete) return tstat;
  }
  tstat = pbval_us->InitRecv(3);
  if (tstat != TaskStatus::complete) return tstat;
  if (back_reaction) {
    tstat = pbval_dm->InitRecv(3);
    if (tstat != TaskStatus::complete) return tstat;
  }
  if (psbox_us != nullptr) {
    // also computes the shear offset used by the u* remap this stage; the O(dt)
    // offset between stages is second-order consistent
    tstat = psbox_us->InitRecv(pmy_pack->pmesh->time);
  }
  return tstat;
}

//----------------------------------------------------------------------------------------
// Send/receive wrappers for the three exchanges of the implicit drag solve

TaskStatus DustGasDrag::SendDepQP(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}
  return pbval_qp->PackAndSendDeposit(qdep);
}

TaskStatus DustGasDrag::RecvDepQP(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}
  return pbval_qp->RecvAndSumDeposit(qdep);
}

TaskStatus DustGasDrag::SendUstar(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pbval_us->PackAndSendCC(ustar, cdummy);
}

TaskStatus DustGasDrag::RecvUstar(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  return pbval_us->RecvAndUnpackCC(ustar, cdummy);
}

TaskStatus DustGasDrag::SendUstarShr(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || psbox_us == nullptr) {return TaskStatus::complete;}
  return psbox_us->PackAndSendCC(ustar, ReconstructionMethod::plm);
}

TaskStatus DustGasDrag::RecvUstarShr(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || psbox_us == nullptr) {return TaskStatus::complete;}
  return psbox_us->RecvAndUnpackCC(ustar);
}

TaskStatus DustGasDrag::SendPMBR(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}
  return pbval_dm->PackAndSendDeposit(dmom);
}

TaskStatus DustGasDrag::RecvPMBR(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage) || !back_reaction) {return TaskStatus::complete;}
  return pbval_dm->RecvAndSumDeposit(dmom);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ClearDep
//! \brief Waits for all sends/receives of the dust exchanges to complete.  The dormant
//! assembly stage returns early: the preceding active stage drained its requests and
//! the dormant stage posts no new dust exchanges.

TaskStatus DustGasDrag::ClearDep(Driver *pdrive, int stage) {
  if (!ActiveStage(pdrive, stage)) {return TaskStatus::complete;}
  TaskStatus tstat = pbval_qp->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_qp->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_us->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_us->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_dm->ClearSend();
  if (tstat != TaskStatus::complete) return tstat;
  tstat = pbval_dm->ClearRecv();
  if (tstat != TaskStatus::complete) return tstat;
  if (psbox_us != nullptr) {
    tstat = psbox_us->ClearSend();
    if (tstat != TaskStatus::complete) return tstat;
    tstat = psbox_us->ClearRecv();
  }
  return tstat;
}

} // namespace dust
