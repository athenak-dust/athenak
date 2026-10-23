#ifndef SHEARING_BOX_ORBITAL_ADVECTION_HPP_
#define SHEARING_BOX_ORBITAL_ADVECTION_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file orbital_advection.hpp
//! \brief definitions for classes that implement orbital advection abstract base and
//! derived classes (for CC and FC variables).

#include "athena.hpp"
#include "parameter_input.hpp"
#include "shearing_box/shearing_box.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvection
//  \brief Abstract base class for orbital advection of CC and FC variables

class OrbitalAdvection {
 public:
  OrbitalAdvection(MeshBlockPack *ppack, ParameterInput *pin);
  ~OrbitalAdvection();

  // data
  int maxjshift;            // maximum integer shift of any cell in orbital advection
  Real qshear, omega0;      // Copies needed for all OA functions
  bool shearing_box_r_phi;  // NOT YET IMPLEMENTED

  // data buffers for orbital advection. Only two x2-faces communicate
  ShearingBoxBoundaryBuffer sendbuf[2], recvbuf[2];

#if MPI_PARALLEL_ENABLED
  // unique MPI communicator for orbital advection
  MPI_Comm comm_orb_advect;
#endif

  // x2-face neighbours on another level (stage S2).  xlev(m,n) for the lower
  // (n=0) and upper (n=1) x2 face: 0 a neighbour on the same level, 1 a coarser neighbour,
  // 2 four finer neighbours.  A coarser sender prolongates (limited linear, as ProlongCC) and
  // a finer sender restricts the rows into the receiver's buffer, so the remap itself is the
  // same-level one.  The coarse side then takes the finer side's remap fluxes through the face
  // (sum of the 2x2 fine fluxes / 8): conservative to round-off.
  DualArray2D<int> xlev;
  DualArray2D<int> xslot;   // (m,n): the slot (nghbr index) of the coarser neighbour
  bool any_mixed;           // on this rank
  bool oa_flux_correct;     // <shearing_box>/experimental_oa_flux_correct (default true)
  int nvar_oa;              // variables (set by the CC class)
  int nrow;                 // ng + maxjshift: rows exchanged at the receiver's resolution
  DvceArray3D<Real> msend[2], mrecv[2];   // mixed-level data (m, subface, flat) for MPI
  DvceArray3D<Real> fsend[2], frecv[2];   // remap fluxes through mixed faces (m, subface, flat)
  DvceArray5D<Real> fown;                 // (m, v, n, k, i): own remap flux at mixed faces
#if MPI_PARALLEL_ENABLED
  MPI_Request *msreq[2], *mrreq[2], *fsreq[2], *frreq[2];   // index 4*m + subface
  int nreq_mixed;
#endif
  void BuildMixedLists();   // in the constructor and after every regrid

  // functions
  TaskStatus InitRecv();
  TaskStatus ClearRecv();
  TaskStatus ClearSend();

 protected:
  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, etc.)
  MeshBlockPack *pmy_pack;
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionCC
//  \brief Derived class implementing orbital advection of CC variables

class OrbitalAdvectionCC : public OrbitalAdvection {
 public:
  OrbitalAdvectionCC(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  // functions to communicate CC data with orbital advection
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, ReconstructionMethod rcon);
  // the fine side sends its remap fluxes through faces with a coarser neighbour;
  // the coarse side replaces its own by them
  TaskStatus SendFluxCC();
  TaskStatus RecvFluxCC(DvceArray5D<Real> &a);
  // public for CUDA (extended lambdas): pack of the mixed-level rows
  void PackMixedCC(DvceArray5D<Real> &a);
  void UnpackMixedCC();
};

//----------------------------------------------------------------------------------------
//! \class OrbitalAdvectionFC
//  \brief Derived class implementing orbital advection of FC variables

class OrbitalAdvectionFC : public OrbitalAdvection {
 public:
  OrbitalAdvectionFC(MeshBlockPack *ppack, ParameterInput *pin);

  // containers for Ex and Ey used to remap magnetic fields
  DvceArray4D<Real> emfx, emfz;

  // functions to communicate FC data with orbital advection
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b0, ReconstructionMethod rcon);
};

#endif // SHEARING_BOX_ORBITAL_ADVECTION_HPP_
