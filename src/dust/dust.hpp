#ifndef DUST_DUST_HPP_
#define DUST_DUST_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust.hpp
//! \brief definitions for DustGasDrag class: Lagrangian dust particles coupled to a
//! Hydro gas by stiff mutual drag, integrated with the imex2+ IMEX(4,3,2) integrator
//! (Krapp et al. 2024) using the particle-mesh scatter/gather method of Yang & Johansen
//! (2016) with exact momentum-conserving back-reaction (PMBR), and the O(N) closed-form
//! per-cell drag solve of Benitez-Llambay, Krapp & Pessah (2019) generalized to
//! per-particle stopping times.
//!
//! This module couples the Particles and Hydro modules the same way IonNeutral couples
//! Hydro and MHD: it owns the combined task graph, the deposited fields, and the drag
//! kernels, while particle storage/migration stays in Particles and fluxes/BCs in Hydro.

#include <map>
#include <memory>
#include <string>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"
#include "bvals/bvals.hpp"

// forward declarations
class Driver;
class MeshBlockPack;
class ShearingBoxCC;

// constants that enumerate particle-mesh deposit schemes
enum class DustDeposit {ngp=0, cic=1, tsc=2};

//----------------------------------------------------------------------------------------
//! \struct DustGasDragTaskIDs
//  \brief container to hold TaskIDs of all dust+hydro tasks

struct DustGasDragTaskIDs {
  // "before_timeintegrator" tasks
  TaskID gswitch;
  // "before_stagen" tasks
  TaskID h_irecv, irecvd;
  // "stagen" tasks
  TaskID first2;
  TaskID h_flux, h_sendf, h_recvf, h_rkupdt, h_srctrms;
  TaskID gatwid, push;
  TaskID p_newgid, p_cnt, p_irecv, p_sendp, p_recvp;
  TaskID scat, sendd, recvd, solve, sendus, recvus, sendus_shr, recvus_shr;
  TaskID gkp, sendbr, recvbr, apply;
  TaskID h_sendu_oa, h_recvu_oa, h_restu, h_sendu, h_recvu, h_sendu_shr, h_recvu_shr;
  TaskID h_bcs, h_prol, h_c2p, h_newdt, newdt;
  // "after_stagen" tasks
  TaskID h_csend, h_crecv, p_csend, p_crecv, cleard;
};

namespace dust {

//----------------------------------------------------------------------------------------
//! \class DustGasDrag

class DustGasDrag {
 public:
  DustGasDrag(MeshBlockPack *ppack, ParameterInput *pin);
  ~DustGasDrag();

  // data
  int nspecies;              // number of dust species (per-species stopping times)
  bool back_reaction;        // dust exerts drag on gas (PMBR); false = test particles
  bool gamma_switch;         // switch gamma to 1/2 when dt > max stopping time (Krapp24)
  bool is_shearing_box;      // <shearing_box> block present
  bool is_stratified;        // vertical gravity (3D shearing box only)
  Real qshear, omega0;       // shearing box parameters (0 if no shearing box)
  Real dt_cfl;               // particle CFL number for transport timestep
  Real taus_max;             // largest stopping time over all species
  Real dust_to_gas;          // total dust/gas mass ratio for default mass normalization
  DualArray1D<Real> taus;    // per-species stopping times
  DustDeposit deposit;       // particle-mesh deposit scheme (tsc default)

  // deposited fields, dimensioned (nmb, nvar, ncells3, ncells2, ncells1)
  DvceArray5D<Real> qdep;    // nvar=4: [0]=Q (drag-weighted density sum), [1-3]=P
  DvceArray5D<Real> ustar;   // nvar=3: provisional drag-corrected gas velocity u*
  DvceArray5D<Real> dmom;    // nvar=3: PMBR momentum deposit; becomes R_g after apply
  DvceArray5D<Real> cdummy;  // 1-element dummy coarse array for ustar copy exchange

  // Boundary communication objects
  MeshBoundaryValuesDep *pbval_qp;  // additive exchange of (Q,P) ghost deposits
  MeshBoundaryValuesDep *pbval_dm;  // additive exchange of PMBR ghost deposits
  MeshBoundaryValuesCC  *pbval_us;  // copy exchange to fill u* ghost zones
  ShearingBoxCC *psbox_us = nullptr;  // shear-periodic remap of u* x1 ghost zones (3D)

  // container to hold names of TaskIDs
  DustGasDragTaskIDs id;

  // functions...
  void AssembleDustGasDragTasks(std::map<std::string, std::shared_ptr<TaskList>> tl);
  // dust tasks are dormant on the trivial assembly stage of imex2+ (beta=0, gam0=1)
  bool ActiveStage(Driver *pdrive, int stage) const;
  // default particle mass/species/stopping-time initialization (pgen may override)
  void SetDefaultMasses(ParameterInput *pin);

  // ...in "before_timeintegrator" list
  TaskStatus GammaSwitch(Driver *pdrive, int stage);
  // ...in "before_stagen" list
  TaskStatus InitRecvDep(Driver *pdrive, int stage);
  // ...in "stagen" list
  TaskStatus FirstTwoImpRK(Driver *pdrive, int stage);     // stage 1 register copies
  TaskStatus ExplicitPush(Driver *pdrive, int stage);      // rotation kick + drift
  TaskStatus AddDragHistoryGas(Driver *pdrive, int stage); // u0 += a_twid*dt*R_g
  TaskStatus DepositDrag(Driver *pdrive, int stage);       // scatter Q,P
  TaskStatus SendDepQP(Driver *pdrive, int stage);
  TaskStatus RecvDepQP(Driver *pdrive, int stage);
  TaskStatus GasImplicitSolve(Driver *pdrive, int stage);  // u* = (rho*u+P)/(rho+Q)
  TaskStatus SendUstar(Driver *pdrive, int stage);
  TaskStatus RecvUstar(Driver *pdrive, int stage);
  TaskStatus SendUstarShr(Driver *pdrive, int stage);
  TaskStatus RecvUstarShr(Driver *pdrive, int stage);
  TaskStatus GatherKickPMBR(Driver *pdrive, int stage);    // gather+kick+record+scatter
  TaskStatus SendPMBR(Driver *pdrive, int stage);
  TaskStatus RecvPMBR(Driver *pdrive, int stage);
  TaskStatus ApplyPMBR(Driver *pdrive, int stage);         // u0 += dmom; dmom -> R_g
  TaskStatus NewTimeStep(Driver *pdrive, int stage);
  // ...in "after_stagen" list
  TaskStatus ClearDep(Driver *pdrive, int stage);

 private:
  MeshBlockPack *pmy_pack;  // ptr to MeshBlockPack containing this DustGasDrag
};

} // namespace dust
#endif // DUST_DUST_HPP_
