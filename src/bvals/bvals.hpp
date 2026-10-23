#ifndef BVALS_BVALS_HPP_
#define BVALS_BVALS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file bvals.hpp
//! \brief defines classes for handling boundary values for both particles as well as all
//! types of Mesh variables. For Mesh variables, methods for cell-centered and
//! face-centered fields are currently implemented, based on derived classes from the
//! generic MeshBoundaryValue class.  A separate ParticlesBoundaryValues class is
//! implemented for particles.

// identifiers for all 6 faces of a MeshBlock
enum BoundaryFace {undef=-1, inner_x1, outer_x1, inner_x2, outer_x2, inner_x3, outer_x3};

// identifiers for boundary conditions
enum class BoundaryFlag {undef=-1,block, reflect, inflow, outflow, diode, user, periodic,
                         shear_periodic, vacuum};

#include <algorithm>
#include <vector>

#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "coordinates/coordinates.hpp"
#include "tasklist/task_list.hpp"
//#include "particles/particles.hpp"

// Forward declarations
class MeshBlockPack;
namespace particles {
class Particles;
}

//----------------------------------------------------------------------------------------
//! \fn int CreateBvals_MPI_Tag(int lid, int bufid)
//! \brief calculate an MPI tag for boundary buffer communications.  Note maximum size of
//! lid that can be encoded is set by (NUM_BITS_LID) macro defined in athena.hpp.
//! The convention in AthenaK is lid and bufid are both for the *receiving* process.
//! \fn int ShearSlotOx1(int n)
//! \brief x1 direction (-1, 0, +1) of neighbour slot n (NeighborIndex layout, 3-D)
KOKKOS_INLINE_FUNCTION
int ShearSlotOx1(const int n) {
  if (n < 4) return -1;
  if (n < 8) return 1;
  if (n < 16) return 0;
  if (n < 24) return ((n - 16) & 2) ? 1 : -1;
  if (n < 32) return 0;
  if (n < 40) return ((n - 32) & 2) ? 1 : -1;
  if (n < 48) return 0;
  return ((n - 48) & 1) ? 1 : -1;
}

inline int CreateBvals_MPI_Tag(int lid, int bufid) {
  return (bufid << (NUM_BITS_LID)) | lid;
}

//----------------------------------------------------------------------------------------
//! \struct BufferIndcs
//! \brief indices for range of cells packed/unpacked into boundary buffers

struct MeshBufferIndcs {
  int bis,bie,bjs,bje,bks,bke;  // start/end buffer ("b") indices in each dir
  MeshBufferIndcs() :
    bis(0), bie(0), bjs(0), bje(0), bks(0), bke(0) {}
};

//----------------------------------------------------------------------------------------
//! \struct MeshBoundaryBuffer
//! \brief container for index ranges, storage, and flags for boundary buffers

struct MeshBoundaryBuffer {
  // fixed-length-3 arrays used to store indices of each buffer for cell-centered vars, or
  // each component of a face-centered vector field ([0,1,2] --> [x1f, x2f, x3f]). For
  // cell-centered variables only first [0] component of index arrays are needed.
  MeshBufferIndcs isame[3];  // indices for pack/unpack when dest/src at same level
  MeshBufferIndcs icoar[3];  // indices for pack/unpack when dest/src at coarser level
  MeshBufferIndcs ifine[3];  // indices for pack/unpack when dest/src at finer level
  MeshBufferIndcs iprol[3];  // indices for prolongation (only used for receives)
  MeshBufferIndcs iflux_same[3];  // indices for pack/unpack for flux correction
  MeshBufferIndcs iflux_coar[3];  // indices for pack/unpack for flux correction
  // With Z4c higher-order prolongation/rstriction, must also send coarse data between
  // MeshBlocks at the same level, which requires an additional indices array
  MeshBufferIndcs isame_z4c;  // indices for pack/unpack with z4c when dst/src at same lvl

  // Maximum number of data elements (bie-bis+1) across 3 components of above
  int isame_ndat, isame_z4c_ndat, icoar_ndat, ifine_ndat, iflxs_ndat, iflxc_ndat;

  // 2D Views that store buffer data on device, dimensioned (nmb, ndata)
  DvceArray2D<Real> vars, flux;

#if MPI_PARALLEL_ENABLED
  // vectors of length (number of MBs) to hold MPI requests
  // Using STL vector causes problems with some GPU compilers, so just use plain C array
  MPI_Request *vars_req, *flux_req;
#endif

  // function to allocate memory for buffers for variables and their fluxes
  // Must only be called after BufferIndcs above are initialized
  void AllocateBuffers(int nmb, int nvars, bool is_z4c) {
    // With Z4c, buffers may contain BOTH same and coarse data
    if (is_z4c) {
      int nmax = std::max(isame_z4c_ndat, std::max(icoar_ndat, ifine_ndat) );
      Kokkos::realloc(vars, nmb, (nvars*nmax));
    } else {
      int nmax = std::max(isame_ndat, std::max(icoar_ndat, ifine_ndat) );
      Kokkos::realloc(vars, nmb, (nvars*nmax));
    }
    int nmax = std::max(iflxs_ndat, iflxc_ndat);
    Kokkos::realloc(flux, nmb, (nvars*nmax));
  }
};

//----------------------------------------------------------------------------------------
//! \struct RankPackedVarEntry
//! \brief metadata for one (MeshBlock,neighbor) var-payload in a rank-packed message

struct RankPackedVarEntry {
  int m;
  int n;
  int lid;
  int dn;
  int data_size;
  int offset;
};

//----------------------------------------------------------------------------------------
//! \struct RankPackedVarMessage
//! \brief metadata for one aggregated MPI vars message between ranks

struct RankPackedVarMessage {
  int rank;
  int nentries;
  int entry_offset;
  int hdr_offset;
  int offset;
  int data_size;
};

// Forward declarations
class MeshBlockPack;

//----------------------------------------------------------------------------------------
//! \class MeshBoundaryValues
//  \brief Abstract base class for boundary values for different kinds of Mesh variables

class MeshBoundaryValues {
 public:
  // S1': no flux correction across the shear-periodic x1 boundaries (the
  // neighbour of the unsheared periodic topology is not the sheared one; the uniform shearing box
  // does no flux correction there either).  Set by Hydro with mesh refinement in 3-D.
  bool skip_shear_x1_flux=false;
  // No prolongation or same-level restriction through neighbour slots across a
  // shear-periodic x1 face: the shearing box collects those ghost zones itself
  bool skip_shear_x1_prol = false;
  MeshBoundaryValues(MeshBlockPack *ppack, ParameterInput *pin, bool z4c);
  virtual ~MeshBoundaryValues();

  // data for all 56 buffers in most general 3D case. Not all elements used in most cases.
  // However each MeshBoundaryBuffer is lightweight, so the convenience of fixed array
  // sizes and index values for array elements outweighs cost of extra memory.
  MeshBoundaryBuffer sendbuf[56], recvbuf[56];

  // constant inflow states at each face, initialized in problem generator
  DualArray2D<Real> u_in, b_in, i_in;

#if MPI_PARALLEL_ENABLED
  // unique MPI communicators for each case (variables/fluxes)
  MPI_Comm comm_vars, comm_flux;

  // rank-packed vars communication path
  int rank_packed_bvals_nvars_;
  int rank_packed_mesh_seq_;
  std::vector<RankPackedVarEntry> send_var_entries_, recv_var_entries_;
  std::vector<RankPackedVarMessage> send_var_msgs_, recv_var_msgs_;
  std::vector<MPI_Request> send_var_reqs_, recv_var_reqs_;
  DvceArray1D<Real> rank_sendbuf_vars_, rank_recvbuf_vars_;
  // Scratch host buffers holding the (lid,dn,data_size) triples per entry, used
  // only for the one-shot header exchange in BuildRankPackedVarMetadata that
  // teaches each receiver its peers' pack order.
  HostArray1D<int> rank_sendhdr_vars_, rank_recvhdr_vars_;
  // Per-(MeshBlock,neighbour) base offsets into the rank-packed aggregate
  // buffers, dimensioned (nmb*nnghbr). For off-rank neighbours these hold the
  // entry's offset in rank_{send,recv}buf_vars_; on-rank/non-existent entries
  // are -1. Built once in BuildRankPackedVarMetadata so the pack/unpack kernels
  // read/write the aggregate buffer directly (fusing the former
  // RankPackAgg/RankUnpackScatter kernels into SendBuff/RecvBuff).
  DvceArray1D<int> send_agg_offset_, recv_agg_offset_;
#endif

  //functions
  virtual void InitSendIndices(MeshBoundaryBuffer &buf,int x,int y,int z,int a,int b)=0;
  virtual void InitRecvIndices(MeshBoundaryBuffer &buf,int x,int y,int z,int a,int b)=0;
  void InitializeBuffers(const int nvar);

  TaskStatus InitRecv(const int nvar);
  virtual TaskStatus InitFluxRecv(const int nvar)=0;
  TaskStatus ClearRecv();
  TaskStatus ClearSend();
  TaskStatus ClearFluxRecv();
  TaskStatus ClearFluxSend();

  // BCs associated with various physics modules
  static void HydroBCs(MeshBlockPack *pp, DualArray2D<Real> uin, DvceArray5D<Real> u0,
                       const bool coarse=false);
  static void BFieldBCs(MeshBlockPack *pp, DualArray2D<Real> bin, DvceFaceFld4D<Real> b0);
  static void RadiationBCs(MeshBlockPack *pp,DualArray2D<Real> iin,DvceArray5D<Real> i0);
  static void Z4cBCs(MeshBlockPack *pp, DualArray2D<Real> uin, DvceArray5D<Real> u0,
                     DvceArray5D<Real> coarse_u0);

 protected:
  // must use pointer to MBPack and not parent physics module since parent can be one of
  // many types (Hydro, MHD, Radiation, Z4c, etc.)
  MeshBlockPack* pmy_pack;
  bool is_z4c_;   // flag to denote if this BoundaryValues is for Z4c module

#if MPI_PARALLEL_ENABLED
  int GetVarDataSize(const MeshBoundaryBuffer &buf, int m, int n, int nvars) const;
  void BuildRankPackedVarMetadata(const int nvars);
#endif
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValuesCC
//  \brief Derived class implementing boundary values for cell-centered variables

class MeshBoundaryValuesCC : public MeshBoundaryValues {
 public:
  MeshBoundaryValuesCC(MeshBlockPack *ppack, ParameterInput *pin, bool z4c);

  //functions
  void InitSendIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  TaskStatus InitFluxRecv(const int nvar) override;

  // functions to communicate CC data
  TaskStatus PackAndSendCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  TaskStatus RecvAndUnpackCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca);
  // functions to communicate fluxes of CC data
  TaskStatus PackAndSendFluxCC(DvceFaceFld5D<Real> &flx);
  TaskStatus RecvAndUnpackFluxCC(DvceFaceFld5D<Real> &flx);

  // functions to prolongate conserved and primitive CC variables
  void FillCoarseInBndryCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca,
       bool is_z4c=false);
  void ProlongateCC(DvceArray5D<Real> &a, DvceArray5D<Real> &ca, bool is_z4c=false,
                    Real amp_eps=0.0, Real amp_cs=0.0);
  void ConsToPrimCoarseBndry(const DvceArray5D<Real> &cons, DvceArray5D<Real> &prim);
  void PrimToConsFineBndry(const DvceArray5D<Real> &prim, DvceArray5D<Real> &cons);
  void ConsToPrimCoarseBndry(const DvceArray5D<Real> &cons, const DvceFaceFld4D<Real> &b,
                             DvceArray5D<Real> &prim);
  void PrimToConsFineBndry(const DvceArray5D<Real> &prim, const DvceFaceFld4D<Real> &b,
                           DvceArray5D<Real> &cons);
};

//----------------------------------------------------------------------------------------
//! \class MeshBoundaryValuesDep
//  \brief Derived class implementing an ADDITIVE ghost-zone exchange for cell-centered
//  fields built by particle deposition: ghost-cell deposits are packed and *added* into
//  the overlapping active cells of the neighboring MeshBlock (the reverse data flow of
//  the ordinary copy exchange). The experimental single-rank static-SMR path remaps
//  ghost-cell densities by geometric overlap and receiving-cell volume.

//----------------------------------------------------------------------------------------
//! \struct DepRemapMap
//! \brief S3a: one part of the general deposit remap: the ghost
//! deposits added to the overlapping active cells of the blocks of this rank, as a list per
//! target cell (CSR, sorted by source block gid and source cell: the sums are independent of
//! the rank layout), and the source ghost cells this rank sends to every other rank.

struct DepRemapMap {
  int nseg = 0, nent = 0;
  DvceArray1D<int> seg_tgt;    // target cell m*ncell + cell (active cell of a block here)
  DvceArray1D<int> seg_ptr;    // nseg+1 offsets into the entries
  DvceArray1D<int> ent_src;    // >= 0: local source cell m*ncell + cell; < 0: -(recv entry+1)
  DvceArray1D<Real> ent_w;     // overlap volume / target cell volume
  int nsend = 0, nrecv = 0;
  DvceArray1D<int> send_cell;  // local source cells packed for the other ranks (by rank)
  std::vector<int> send_rank, send_off, send_cnt;   // per peer rank: offset, count (entries)
  std::vector<int> recv_rank, recv_off, recv_cnt;
  // Ghost cells beyond a physical x3 face that are dropped (x3 mode drop): local
  // source cells and their volumes, for the record of the dropped deposits
  int ndrop = 0;
  DvceArray1D<int> drop_cell;
  DvceArray1D<Real> drop_vol;
};

//----------------------------------------------------------------------------------------
//! \class DepRemapPlan
//! \brief S3a: the additive ghost-deposit exchange of a 3-D mesh with
//! any levels (static or adaptive refinement), multi-rank, and shear-periodic x1 faces.  Every
//! ghost cell deposit goes to the active cells it overlaps (volume overlap in the integer cell
//! units of the finest level: exact, conservative); ghost cells beyond a shear-periodic x1 face
//! go to the opposite face shifted in x2 by the shear offset (the particles' convention,
//! bvals_part.cpp).  Built on the host from the global block list (no finest-grid table): the
//! static part once per mesh, the shear part at every stage.  Shared by the deposit exchanges.

class DepRemapPlan {
 public:
  // x3mode (a 3-D mesh with physical x3 faces): 0 periodic x3, 1 the ghost cells beyond
  // a physical x3 face fold back onto their mirror cells, 2 they are dropped (and recorded)
  DepRemapPlan(MeshBlockPack *pp, bool shear_x1, Real qshear, Real omega0, int x3mode = 0);
  void BuildStatic();              // after construction and after every regrid
  void UpdateShear(Real time);     // the shear part for the offset at time (no-op without shear)
  bool Shear() const {return shear_x1_;}
  DepRemapMap stat, shr;
  int build_seq = 0;               // increases with every BuildStatic
  Real shear_time = -1.0;
  // Public because CUDA extended lambdas cannot be enclosed by a private member.
  void BuildPart(const int part, const double shR, DepRemapMap &map);

 private:
  MeshBlockPack *pmy_pack;
  bool shear_x1_;
  int x3mode_ = 0;
  Real qshear_, omega0_;
  std::vector<int> face_gids_[2];  // blocks at the inner / outer x1 face (all ranks)
};

class MeshBoundaryValuesDep : public MeshBoundaryValues {
 public:
  MeshBoundaryValuesDep(MeshBlockPack *ppack, ParameterInput *pin);
  ~MeshBoundaryValuesDep();

  // The general 3-D path: the dust module creates the plan and gives it to its
  // deposit exchanges; then these replace InitRecv / ClearSend / ClearRecv of the base class
  void SetGeneralPlan(DepRemapPlan *p) {plan_ = p;}
  bool General() const {return plan_ != nullptr;}
  TaskStatus InitRecvDep(const int nvar);
  TaskStatus ClearSendDep();
  TaskStatus ClearRecvDep();
  // Public because CUDA extended lambdas cannot be enclosed by a private member.
  void GeneralPack(DvceArray5D<Real> &a, const DepRemapMap &map, DvceArray1D<Real> &buf);
  void GeneralApply(DvceArray5D<Real> &a, const DepRemapMap &map,
                    const DvceArray1D<Real> &buf);
  void GeneralDropRecord(DvceArray5D<Real> &a, const DepRemapMap &map);
  std::vector<double> dropped_sum;   // dropped deposits, integrated, per variable

  //functions
  void InitSendIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  TaskStatus InitFluxRecv(const int nvar) override {return TaskStatus::complete;}

  // pack ghost-region deposits and send; receive and sum into active cells
  TaskStatus PackAndSendDeposit(DvceArray5D<Real> &a);
  TaskStatus RecvAndSumDeposit(DvceArray5D<Real> &a);
  // Public because CUDA extended lambdas cannot be enclosed by a private member.
  TaskStatus RemapLocalSMRDeposit(DvceArray5D<Real> &a);
  // the remap of the owner_overlap path from the current mesh; called again by the dust module
  // after every regrid of a dynamic mesh
  void BuildLocalSMRMap();
  bool LocalSMRRemap() const {return local_smr_remap_;}

 private:
  static constexpr int max_smr_targets_ = 8;
  bool local_smr_remap_ = false;
  bool local_smr_direct_ = false;
  // General path: buffers and requests of this exchange (static, shear part)
  DepRemapPlan *plan_ = nullptr;
  int gen_nvar_ = 0;
  DvceArray1D<Real> gsbuf_[2], grbuf_[2];   // device buffers given to MPI (as the base class)
#if MPI_PARALLEL_ENABLED
  MPI_Comm comm_dep_;
  std::vector<MPI_Request> gsreq_, grreq_;
#endif
  DvceArray4D<int> smr_target_count_;
  DvceArray5D<int> smr_target_index_;
  DvceArray5D<Real> smr_target_weight_;
};

//----------------------------------------------------------------------------------------
//! \class BoundaryValuesFC
//  \brief Derived class implementing boundary values for face-centered vector fields

class MeshBoundaryValuesFC : public MeshBoundaryValues {
 public:
  MeshBoundaryValuesFC(MeshBlockPack *ppack, ParameterInput *pin);

  //functions
  void InitSendIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  void InitRecvIndices(MeshBoundaryBuffer &b,int o1,int o2,int o3,int f1,int f2) override;
  TaskStatus InitFluxRecv(const int nvar) override;

  TaskStatus PackAndSendFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  TaskStatus RecvAndUnpackFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void FillCoarseInBndryFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);
  void ProlongateFC(DvceFaceFld4D<Real> &b, DvceFaceFld4D<Real> &cb);

  TaskStatus PackAndSendFluxFC(DvceEdgeFld4D<Real> &flx);
  TaskStatus RecvAndUnpackFluxFC(DvceEdgeFld4D<Real> &flx);
  void SumBoundaryFluxes(DvceEdgeFld4D<Real> &flx, const bool same_level,
                         DvceArray2D<int> &nflx);
  void ZeroFluxesAtBoundaryWithFiner(DvceEdgeFld4D<Real> &flx, DvceArray2D<int> &nflx);
  void AverageBoundaryFluxes(DvceEdgeFld4D<Real> &flx, DvceArray2D<int> &nflx);
};

//----------------------------------------------------------------------------------------
//! \struct ParticleLocationData
//! \brief data describing location of data for particles communicated with MPI

struct ParticleLocationData {
  int prtcl_indx;   // index in particle array
  int dest_gid;     // GID of target MeshBlock
  int dest_rank;    // rank of target MeshBlock
};

// Custom operators to sort ParticleLocationData array by dest_rank or prtcl_indx
struct {
  bool operator()(ParticleLocationData a, ParticleLocationData b)
    const { return a.dest_rank < b.dest_rank; }
} SortByRank;
struct {
  bool operator()(ParticleLocationData a, ParticleLocationData b)
    const { return a.prtcl_indx < b.prtcl_indx; }
} SortByIndex;

//----------------------------------------------------------------------------------------
//! \struct ParticleMessageData
//! \brief Data describing MPI messages containing particles

struct ParticleMessageData {
  int sendrank;  // rank of sender
  int recvrank;  // rank of receiver
  int nprtcls;   // number of particles in message
  ParticleMessageData(int a, int b, int c) :
    sendrank(a), recvrank(b), nprtcls(c) {}
};

//----------------------------------------------------------------------------------------
//! \class ParticlesBoundaryValues
//  \brief Defines boundary values class for particles

namespace particles {
class ParticlesBoundaryValues {
 public:
  ParticlesBoundaryValues(particles::Particles *ppart, ParameterInput *pin);
  ~ParticlesBoundaryValues();

  int nprtcl_send, nprtcl_recv;
  // sendlist allocation is persistent: nprtcl_send is the valid-prefix length, while
  // sendlist.extent(0) is capacity.  Never shrink the allocation after a migration.
  DualArray1D<ParticleLocationData> sendlist;
  DualArray1D<int> send_count;  // device-side append counter (length one)

  // shear-periodic x1 boundary support (uniform grids only). Particles crossing the
  // radial mesh boundaries are shifted azimuthally (positions only; velocities are
  // shear-relative), so their destination MeshBlock is generally not the x1-neighbor:
  // it is found from a (side, lx3, lx2) -> gid/rank map over the boundary MeshBlocks.
  bool shear_periodic_x1=false;
  Real qshear_sp=0.0, omega0_sp=0.0;   // copies of <shearing_box> parameters
  int nmb_sp_x2=1, nmb_sp_x3=1;        // root-grid MeshBlock counts in x2/x3
  DualArray3D<int> sgid_map;           // destination GID:  (side, lx3, lx2)
  DualArray3D<int> srank_map;          // destination rank: (side, lx3, lx2)
  void BuildShearMaps();               // also after every regrid
  // Dust particles at the physical x3 faces of a 3-D mesh (S3c): 0 periodic,
  // 1 reflect (mirror position and velocity), 2 outflow (the particle leaves: mass and count added
  // to x3_removed, mass set to 0, parked at rest in its block; DustGasDrag removes it at the start
  // of the next cycle)
  int x3bc_dust = 0;
  DualArray1D<Real> x3_removed;        // (mass, count) since the last removal

  // Data needed to count number of messages and particles to send between ranks
  int nsends; // number of MPI sends to neighboring ranks on this rank
  int nrecvs; // number of MPI recvs from neighboring ranks on this rank
  std::vector<int> nsends_eachrank;                // length nranks
  std::vector<ParticleMessageData> sends_thisrank; // length nsends
  std::vector<ParticleMessageData> recvs_thisrank; // length nrecvs
  std::vector<ParticleMessageData> sends_allranks; // length ncounts summed over ranks

#if MPI_PARALLEL_ENABLED
  DvceArray1D<Real> prtcl_rsendbuf, prtcl_rrecvbuf;
  DvceArray1D<int>  prtcl_isendbuf, prtcl_irecvbuf;
  std::vector<MPI_Request> rrecv_req, rsend_req;  // vectors of requests for Reals
  std::vector<MPI_Request> irecv_req, isend_req;  // vectors of requests for ints
  MPI_Comm mpi_comm_part;                       // unique MPI communicators for particles
#endif

  //functions
  TaskStatus SetNewPrtclGID();
  TaskStatus CountSendsAndRecvs();
  TaskStatus InitPrtclRecv();
  TaskStatus ClearPrtclRecv();
  TaskStatus PackAndSendPrtcls();
  TaskStatus ClearPrtclSend();
  TaskStatus RecvAndUnpackPrtcls();

 protected:
  particles::Particles* pmy_part;
};
} // namespace particles

#endif // BVALS_BVALS_HPP_
