#ifndef SHEARING_BOX_SHEARING_BOX_HPP_
#define SHEARING_BOX_SHEARING_BOX_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file shearing_box.hpp
//! \brief definitions for classes that implement shearing box abstract base and derived
//! classes (for CC and FC variables).

#include <utility>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "tasklist/task_list.hpp"

//----------------------------------------------------------------------------------------
//! \struct ShearingBoxTaskIDs
//  \brief container to hold TaskIDs of all shearing_box tasks

struct ShearingBoxTaskIDs {
  TaskID irecv;
  TaskID copyu;
  TaskID flux;
  TaskID sendf;
  TaskID recvf;
  TaskID csend;
  TaskID crecv;
};

//----------------------------------------------------------------------------------------
//! \struct ShearingBoxBoundaryBuffer
//! \brief container for storing boundary buffers. Used by both the orbital advection and
//! shearing box methods for both CC and FC variables.
//! Basically a much simplified version of the MeshBoundaryBuffer struct.

struct ShearingBoxBoundaryBuffer {
  // Views that store buffer data and fluxes on device
  DvceArray5D<Real> vars, flux;
#if MPI_PARALLEL_ENABLED
  // vectors of length (number of MBs) to hold MPI requests
  // Using STL vector causes problems with some GPU compilers, so just use plain C array
  MPI_Request *vars_req, *flux_req;
#endif
};

//----------------------------------------------------------------------------------------
//! \class ShearingBox
//  \brief Abstract base class for shearing box boundaries with CC and FC vars

class ShearingBox {
 public:
  ShearingBox(MeshBlockPack *ppack, ParameterInput *pin);
  ~ShearingBox();

  // data
  HostArray1D<int> nmb_x1bndry;    // number of MBs that touch x1 boundaries
  DualArray2D<int> x1bndry_mbgid;  // GIDs of MBs at x1 boundaries
  Real yshear;                     // x2-distance x1-boundaries have sheared
  Real qshear, omega0;             // Copies needed for all SB funcs
  bool shearing_box_r_phi;         // NOT YET IMPLEMENTED
  bool is_stratified;              // true for stratified shearing box

  // data buffers for shearing box BCs.  Only two x1-faces get sheared
  // Use separate variables for ix1/ox1 since number of MBs on each face can be different
  ShearingBoxBoundaryBuffer sendbuf[2], recvbuf[2];
  // (S1') with mesh refinement the same exchange also shifts the x1 ghost
  // zones of the coarse arrays: a fine x1-face block whose partner across the boundary (or a
  // diagonal neighbour there) is coarser gets those ghost cells by prolongation from its coarse
  // array, after the shear exchange; its coarse array must be sheared first
  bool coarse_on = false;
  ShearingBoxBoundaryBuffer csendbuf[2], crecvbuf[2];

#if MPI_PARALLEL_ENABLED
  // unique MPI communicator for shearing box
  MPI_Comm comm_sbox;
#endif

  // functions
  TaskStatus InitRecv(Real time);
  TaskStatus ClearRecv();
  TaskStatus ClearSend();
  // the lists of x1-face MeshBlocks and the MPI requests, again after a regrid; and the
  // check that the levels are constant along x2 (the exchange is between blocks of one level)
  void BuildBoundaryLists();
  void CheckLevelsAlongX2() const;
  void CheckSupportedSetup() const;   // block widths, periodic x3
  // function to find target MB offset by shear.  Returns GID and rank
  void FindTargetMB(const int igid, const int jshift, int &gid, int &rank);
  // function to find index in x1bndry array of MB with input GID
  int TargetIndex(const int n, const int tgid) {
    for (int m=0; m<nmb_x1bndry(n); ++m) {
      if (x1bndry_mbgid.h_view(n,m) == tgid) return m;
    }
    return -1;
  }

 protected:
  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, etc.)
  MeshBlockPack *pmy_pack;
  // geometry of one exchange: the MeshBlock arrays (coarse = false) or the coarse
  // arrays; rows js..je of x2, rows kl..ku of x3 (with ghosts in 3-D), first outer ghost column
  // ie+1, active x2 cells nx2, all x2 cells nj, cell size factor, MPI buffer id offset
  struct ShearSet {int js, je, kl, ku, ie, nx2, nj; Real dxfac; int tagbase;};
  ShearSet Set(const bool coarse) const;
  // the pieces of the integer shift of boundary block m of face n: their number (2 or 3), rows of
  // the send buffer (jsrc) and of the receive buffer (jdst), and the block offset of the target
  // of each piece (its sender, seen from the receiving block, is at the negated offset)
  int Pieces(const int n, const int m, const ShearSet &st, std::pair<int,int> jsrc[3],
             std::pair<int,int> jdst[3], int jsh[3]) const;
  bool PostRecvs(const bool coarse);
  bool TestRecvs(const bool coarse, bool &done);
  bool WaitRecvs(const bool coarse);
  bool WaitSends(const bool coarse);
  void AllocateRequests();
  void FreeRequests();
  int nreq_[2] = {0, 0};   // length of the request arrays (3 per boundary block)
};

//----------------------------------------------------------------------------------------
//! \class ShearingBoxCC
//! \brief Derived class implementing shearing box boundary conditions and source terms
//! for CC vars

class ShearingBoxCC : public ShearingBox {
 public:
  ShearingBoxCC(MeshBlockPack *ppack, ParameterInput *pin, int nvar);
  // functions to communicate CC data with shearing box BCs
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, ReconstructionMethod rcon);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a);
  // the same for the coarse arrays (mesh refinement)
  TaskStatus PackAndSendCoarseCC(DvceArray5D<Real> &ca, ReconstructionMethod rcon);
  TaskStatus RecvAndUnpackCoarseCC(DvceArray5D<Real> &ca);
  // buffers for the current lists of x1-face blocks (after BuildBoundaryLists)
  void AllocateBuffers();
  // shearing box source terms for Hydro CC variables
  void SourceTermsCC(const DvceArray5D<Real> &w0, const EOS_Data &eos_data,
                     const Real bdt, DvceArray5D<Real> &u0);
  // shearing box source terms for MHD CC variables
  void SourceTermsCC(const DvceArray5D<Real> &w0, const DvceArray5D<Real> &bcc0,
                     const EOS_Data &eos_data, const Real bdt, DvceArray5D<Real> &u0);

  // (public: CUDA does not allow extended device lambdas in private member functions)
  TaskStatus PackAndSend(DvceArray5D<Real> &a, ReconstructionMethod rcon, const bool coarse);
  TaskStatus RecvAndUnpack(DvceArray5D<Real> &a, const bool coarse);

 private:
  int nvar_;
};

//----------------------------------------------------------------------------------------
//! \class ShearingBoxFC
//!  \brief Derived class implementing shearing box boundary conditions and source terms
//! (if needed) for FC vars

class ShearingBoxFC : public ShearingBox {
 public:
  ShearingBoxFC(MeshBlockPack *ppack, ParameterInput *pin);
  // functions to communicate CC data with shearing box BCs
  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, ReconstructionMethod rcon);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b);
  // shearing box source terms for FC variables
  void SourceTermsFC(const DvceFaceFld4D<Real> &b0, DvceEdgeFld4D<Real> &efld);
};

#endif // SHEARING_BOX_SHEARING_BOX_HPP_
