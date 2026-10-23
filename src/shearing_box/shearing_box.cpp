//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box.cpp
//! \brief constructor for ShearingBox abstract base class, and utility functions

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "mhd/mhd.hpp"
#include "shearing_box.hpp"

//----------------------------------------------------------------------------------------
//! ShearingBox base class constructor

ShearingBox::ShearingBox(MeshBlockPack *ppack, ParameterInput *pin) :
    nmb_x1bndry("nmbx1",2),
    x1bndry_mbgid("x1gid",1,1),
    shearing_box_r_phi(false),     // 2D r-phi not yet implemented
    pmy_pack(ppack) {
  // Read shear rate and orbital frequency
  qshear = pin->GetReal("shearing_box","qshear");
  omega0 = pin->GetReal("shearing_box","omega0");
  is_stratified = pin->GetOrAddBoolean("shearing_box","stratified",false);
  // (S2b) blocks on the x1 faces may change level along x2
  face_levels = pin->GetOrAddBoolean("shearing_box", "experimental_face_levels", false);
  report_timing = pin->GetOrAddBoolean("shearing_box", "report_timing", false);
  // (S2c) non-periodic x3 needs the coarse-array physical
  // boundaries for the corners the collected ghost zones leave to them; dust
  // (u* through the same collection, deposits through the general remap, particle routing by the
  // face-block table) is covered
  if (face_levels && ppack->pmesh->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic &&
      pin->DoesBlockExist("mesh_refinement") &&
      !pin->GetOrAddBoolean("mesh_refinement", "coarse_physical_bcs", true)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<shearing_box>/experimental_face_levels with non-periodic x3 needs "
              << "<mesh_refinement>/coarse_physical_bcs = true" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  BuildBoundaryLists();
  CheckLevelsAlongX2();
  // MeshRefinement is built after the physics: read its key here
  CheckSupportedSetup(!pin->DoesParameterExist("mesh_refinement", "coarse_physical_bcs") ||
                      pin->GetBoolean("mesh_refinement", "coarse_physical_bcs"));

#if MPI_PARALLEL_ENABLED
  // create unique communicators for shearing box
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_sbox);
  MPI_Comm_dup(MPI_COMM_WORLD, &comm_pull);
#endif
}

//----------------------------------------------------------------------------------------
// ShearingBox base class destructor

ShearingBox::~ShearingBox() {
  if (report_timing) {   // per rank
    std::cout << "# shearing box timing rank " << global_variable::my_rank << ": PlanPull "
              << n_plan_pull << " calls " << t_plan_pull << " s, flux plans " << n_plan_flux
              << " calls " << t_plan_flux << " s (host)" << std::endl;
  }
  FreeRequests();
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::BuildBoundaryLists()
//! \brief GIDs of the MeshBlocks of this rank at the ix1/ox1 shearing-box boundaries and the
//! MPI requests (each boundary block exchanges with up to 3 blocks).  Called by the
//! constructor and again after every regrid.

void ShearingBox::BuildBoundaryLists() {
  pull_ghost_valid = false;   // new or moved blocks: x1 ghosts not collected yet
  // Create vector with GID of every MBs on this rank at ix1/ox1 shearing-box boundaries
  std::vector<int> tmp_ix1bndry_gid, tmp_ox1bndry_gid;
  auto &mbbcs = pmy_pack->pmb->mb_bcs;
  for (int m=0; m<(pmy_pack->nmb_thispack); ++m) {
    if (mbbcs.h_view(m,BoundaryFace::inner_x1) == BoundaryFlag::shear_periodic) {
      tmp_ix1bndry_gid.push_back(m + pmy_pack->gids);
    }
    if (mbbcs.h_view(m,BoundaryFace::outer_x1) == BoundaryFlag::shear_periodic) {
      tmp_ox1bndry_gid.push_back(m + pmy_pack->gids);
    }
  }
  // number of MBs at ix1/ox1 boundaries is size of vectors
  nmb_x1bndry(0) = tmp_ix1bndry_gid.size();
  nmb_x1bndry(1) = tmp_ox1bndry_gid.size();

  // allocate mbgid array and initialize GIDs to -1
  // Ensure nmb is at least 1 to avoid zero-sized allocations
  int nmb = std::max(1, std::max(nmb_x1bndry(0),nmb_x1bndry(1)));
  Kokkos::realloc(x1bndry_mbgid, 2, nmb);
  for (int n=0; n<2; ++n) {
    for (int m=0; m<nmb; ++m) {
      x1bndry_mbgid.h_view(n,m) = -1;
    }
  }
  // load GIDs of meshblocks at x1 boundaries into DualArray
  for (int m=0; m<nmb_x1bndry(0); ++m) {
    x1bndry_mbgid.h_view(0,m) = tmp_ix1bndry_gid[m];
  }
  for (int m=0; m<nmb_x1bndry(1); ++m) {
    x1bndry_mbgid.h_view(1,m) = tmp_ox1bndry_gid[m];
  }
  // sync with device
  x1bndry_mbgid.template modify<HostMemSpace>();
  x1bndry_mbgid.template sync<DevExeSpace>();

  FreeRequests();
  AllocateRequests();
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::CheckLevelsAlongX2()
//! \brief With mesh refinement in 3-D (S1') the shear exchange (and the
//! orbital advection) shifts data between MeshBlocks of one level: FindTargetMB looks for the
//! shifted block at the sender's level and x3 location.  Every MeshBlock on a shear-periodic x1
//! face must have its x2 neighbours on its level (refinement in columns along x2 there).  Coarser partners across the x1 boundary
//! are allowed: their data reach the fine block's coarse array, which is sheared as well.

void ShearingBox::CheckLevelsAlongX2() const {
  if (face_levels) return;   // the collected ghost zones need no uniform columns
  Mesh *pm = pmy_pack->pmesh;
  if (!(pm->multilevel && pm->three_d &&
        pm->mesh_bcs[BoundaryFace::inner_x1] == BoundaryFlag::shear_periodic)) return;
  // (S2) only the blocks on the shear-periodic x1 faces (the orbital
  // advection handles x2 neighbours on other levels in the interior); all x2 face slots
  auto &nghbr = pmy_pack->pmb->nghbr;
  auto &lev = pmy_pack->pmb->mb_lev;
  auto &mbbcs = pmy_pack->pmb->mb_bcs;
  bool ok = true;
  for (int m=0; m<pmy_pack->nmb_thispack; ++m) {
    if (mbbcs.h_view(m,BoundaryFace::inner_x1) != BoundaryFlag::shear_periodic &&
        mbbcs.h_view(m,BoundaryFace::outer_x1) != BoundaryFlag::shear_periodic) continue;
    for (int n=8; n<16; ++n) {
      if (nghbr.h_view(m,n).gid >= 0 && nghbr.h_view(m,n).lev != lev.h_view(m)) ok = false;
    }
  }
  if (!ok) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shearing box with mesh refinement (experimental_smr): the x2 neighbours of "
              << "every MeshBlock on a shear-periodic x1 face must be on its level (the "
              << "blocks there refine in columns along x2)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::CheckSupportedSetup()
//! \brief Setup checks (Codex review): Pieces splits a block's x2 rows into at most
//! three pieces, which needs nx2 >= 2 ng; the coarse arrays (3-D with refinement) have half the
//! rows and the same ng, so there the fine blocks need nx2 >= 4 ng.  Non-periodic x3 needs the
//! physical boundary values of the coarse arrays (on by default).

void ShearingBox::CheckSupportedSetup(const bool coarse_bcs) const {
  Mesh *pm = pmy_pack->pmesh;
  if (pm->mesh_bcs[BoundaryFace::inner_x1] != BoundaryFlag::shear_periodic) return;
  const auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng;
  if (indcs.nx2 < 2*ng) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shearing box: meshblock/nx2 = " << indcs.nx2 << " must be at least 2*nghost"
              << " = " << 2*ng << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!(pm->multilevel && pm->three_d)) return;
  if (indcs.cnx2 < 2*ng) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Shearing box with mesh refinement: the coarse arrays are sheared too, so "
              << "meshblock/nx2 = " << indcs.nx2 << " must be at least 4*nghost = " << 4*ng
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // the coarse arrays get physical boundary values in Hydro::Prolongate
  if (pm->mesh_bcs[BoundaryFace::inner_x3] != BoundaryFlag::periodic &&
      !coarse_bcs && global_variable::my_rank == 0) {
    std::cout << "# WARNING: shearing box with mesh refinement in 3-D and non-periodic x3 "
              << "without <mesh_refinement>/coarse_physical_bcs (control only): the "
              << "prolongation next to the x3 faces reads unset coarse cells"
              << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::AllocateRequests(), FreeRequests()
//! \brief MPI requests of the fine and the coarse exchange: 3 per boundary block and face

void ShearingBox::AllocateRequests() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    nreq_[n] = 3*nmb_x1bndry(n);
    if (nreq_[n] == 0) continue;
    sendbuf[n].vars_req = new MPI_Request[nreq_[n]];
    recvbuf[n].vars_req = new MPI_Request[nreq_[n]];
    csendbuf[n].vars_req = new MPI_Request[nreq_[n]];
    crecvbuf[n].vars_req = new MPI_Request[nreq_[n]];
    for (int r=0; r<nreq_[n]; ++r) {
      sendbuf[n].vars_req[r] = MPI_REQUEST_NULL;
      recvbuf[n].vars_req[r] = MPI_REQUEST_NULL;
      csendbuf[n].vars_req[r] = MPI_REQUEST_NULL;
      crecvbuf[n].vars_req[r] = MPI_REQUEST_NULL;
    }
  }
#endif
}

void ShearingBox::FreeRequests() {
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<2; ++n) {
    if (nreq_[n] > 0) {
      delete [] sendbuf[n].vars_req;
      delete [] recvbuf[n].vars_req;
      delete [] csendbuf[n].vars_req;
      delete [] crecvbuf[n].vars_req;
    }
    nreq_[n] = 0;
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn ShearingBox::Set()
//! \brief Index ranges of one exchange: the MeshBlock arrays or the coarse arrays

ShearingBox::ShearSet ShearingBox::Set(const bool coarse) const {
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  ShearSet st;
  const int ng = indcs.ng;
  if (!coarse) {
    st.js = indcs.js; st.je = indcs.je; st.kl = indcs.ks; st.ku = indcs.ke;
    st.ie = indcs.ie; st.nx2 = indcs.nx2; st.dxfac = 1.0; st.tagbase = 0;
  } else {
    st.js = indcs.cjs; st.je = indcs.cje; st.kl = indcs.cks; st.ku = indcs.cke;
    st.ie = indcs.cie; st.nx2 = indcs.cnx2; st.dxfac = 2.0; st.tagbase = 8;
  }
  if (pmy_pack->pmesh->three_d) {st.kl -= ng; st.ku += ng;}
  st.nj = st.nx2 + 2*ng;
  return st;
}

//----------------------------------------------------------------------------------------
//! \fn int ShearingBox::Pieces()
//! \brief The integer shift of boundary block m of face n, broken into pieces as in the original
//! algorithm (case 1 and 3: the integer shift jr < ng or jr >= nx2-ng, three target blocks;
//! case 2: two).  Rows jsrc of the send buffer go to rows jdst of the receive buffer of the block
//! at offset jsh in x2 (the receive of block m from its sender at offset -jsh lands in the same
//! rows jdst).

int ShearingBox::Pieces(const int n, const int m, const ShearSet &st,
                        std::pair<int,int> jsrc[3], std::pair<int,int> jdst[3],
                        int jsh[3]) const {
  const int ng = pmy_pack->pmesh->mb_indcs.ng;
  const int js = st.js, je = st.je, nx2 = st.nx2;
  const int mm = x1bndry_mbgid.h_view(n,m) - pmy_pack->gids;
  // Find integer and fractional number of grids over which offset extends.
  // This assumes every grid has same number of cells in x2-direction!
  const int joffset =
      static_cast<int>(yshear/(st.dxfac*(pmy_pack->pmb->mb_size.h_view(mm).dx2)));
  const int ji = joffset/nx2;
  const int jr = joffset - ji*nx2;
  if (jr < ng) {               //--- CASE 1 (in my nomenclature)
    if (n==0) {
      jsrc[0] = std::make_pair(js,js+ng-jr);
      jsrc[1] = std::make_pair(js,je+1);
      jsrc[2] = std::make_pair(je-(ng-1)-jr,je+1);
      jdst[0] = std::make_pair(je+1+jr,je+ng+1);
      jdst[1] = std::make_pair(js+jr,je+jr+1);
      jdst[2] = std::make_pair(js-ng,js+jr);
    } else {
      jsrc[0] = std::make_pair(js,js+ng+jr);
      jsrc[1] = std::make_pair(js,je+1);
      jsrc[2] = std::make_pair(je-(ng-1)+jr,je+1);
      jdst[0] = std::make_pair(je+1-jr,je+ng+1);
      jdst[1] = std::make_pair(js-jr,je-jr+1);
      jdst[2] = std::make_pair(js-ng,js-jr);
    }
    // ix1 boundary: send to (target-1) through (target+1)
    // ox1 boundary: send to (target-1) through (target+1)
    for (int l=0; l<3; ++l) {jsh[l] = (n==0) ? (ji+l-1) : (l-1-ji);}
    return 3;
  } else if (jr < (nx2-ng)) {  //--- CASE 2
    if (n==0) {
      jsrc[0] = std::make_pair(js,je+ng-jr+1);
      jsrc[1] = std::make_pair(je-(ng-1)-jr,je+1);
      jdst[0] = std::make_pair(js+jr,je+ng+1);
      jdst[1] = std::make_pair(js-ng,js+jr);
    } else {
      jsrc[0] = std::make_pair(js,js+ng+jr);
      jsrc[1] = std::make_pair(js-ng+jr,je+1);
      jdst[0] = std::make_pair(je-jr+1,je+ng+1);
      jdst[1] = std::make_pair(js-ng,je-jr+1);
    }
    // ix1 boundary: send to (target  ) through (target+1)
    // ox1 boundary: send to (target-1) through (target  )
    for (int l=0; l<2; ++l) {jsh[l] = (n==0) ? (ji+l) : (l-1-ji);}
    return 2;
  }
  //--- CASE 3
  if (n==0) {
    jsrc[0] = std::make_pair(js,js+ng+(nx2-jr));
    jsrc[1] = std::make_pair(js,je+1);
    jsrc[2] = std::make_pair(je-(ng-1)+(nx2-jr),je+1);
    jdst[0] = std::make_pair(je+1-(nx2-jr),je+ng+1);
    jdst[1] = std::make_pair(js-(nx2-jr),je-(nx2-jr)+1);
    jdst[2] = std::make_pair(js-ng,js-(nx2-jr));
  } else {
    jsrc[0] = std::make_pair(js,js+ng-(nx2-jr));
    jsrc[1] = std::make_pair(js,je+1);
    jsrc[2] = std::make_pair(je-(ng-1)-(nx2-jr),je+1);
    jdst[0] = std::make_pair(je+1+(nx2-jr),je+ng+1);
    jdst[1] = std::make_pair(js+(nx2-jr),je+(nx2-jr)+1);
    jdst[2] = std::make_pair(js-ng,js+(nx2-jr));
  }
  // ix1 boundary: send to (target  ) through (target+2)
  // ox1 boundary: send to (target-2) through (target  )
  for (int l=0; l<3; ++l) {jsh[l] = (n==0) ? (ji+l) : (l-2-ji);}
  return 3;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::FindTargetMB()
//! \brief  function to find target MB offset by shear.  Returns GID and rank

void ShearingBox::FindTargetMB(const int igid, const int jshift, int &gid,
                                       int &rank) {
  Mesh *pm = pmy_pack->pmesh;
  // find lloc of input MB
  LogicalLocation lloc = pm->lloc_eachmb[igid];
  // find number of MBs in x2 direction at this level
  std::int32_t nmbx2 = pm->nmb_rootx2 << (lloc.level - pm->root_level);
  // apply shift by input number of blocks
  lloc.lx2 = static_cast<std::int32_t>((lloc.lx2 + jshift) % nmbx2);
  if (lloc.lx2 < 0) lloc.lx2 += nmbx2;
  // find target GID and rank
  gid = (pm->ptree->FindMeshBlock(lloc))->GetGID();
  rank = pm->rank_eachmb[gid];
  return;
}
