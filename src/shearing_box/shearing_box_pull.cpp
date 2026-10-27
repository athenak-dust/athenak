//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box_pull.cpp
//! \brief x1 ghost zones of the shearing box when the blocks on
//! the x1 faces change level along x2 (ShearingBox::face_levels, on with shear-periodic x1).
//!
//! Every block on a shear-periodic x1 face has up to two receivers per face: its x1 ghost
//! columns (level of the block) and those of its coarse array (one level coarser; only with
//! coarse_on and above the root level).  A receiver at level Lr needs the cells just inside
//! the opposite face, depth 0..ng-1 at Lr, rows J0-ng-joff-3 .. J0+ny+ng-joff+2 and x3 rows
//! K0-ng .. K0+nz+ng-1 (periodic), with joff + eps the shear shift in Lr cells.  Every opposite
//! block that overlaps this rectangle sends its part converted to Lr: copied (same level),
//! averaged over 2^d cells per direction (finer by d), or prolongated with limited slopes
//! (coarser; slopes from its own cells and ghosts, none in x at the face column: its own x1
//! ghosts are being exchanged).  The receiver remaps the rows by eps as the two-step exchange
//! does (DC/PLM/PPMX_RemapFlx) and writes all its x1 ghost cells.  The pieces are planned
//! every stage from the global block list (all ranks derive the same list); one MPI message
//! per (receiving block, sending block) pair.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/prolongation.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"
#include "remap_fluxes.hpp"

namespace {
// floor division for possibly negative numerators (positive d)
long long FloorDiv(long long a, long long d) {
  return (a >= 0) ? (a/d) : -((-a + d - 1)/d);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ShearingBox::PlanPull()

void ShearingBox::PlanPull() {
  const auto tp0 = std::chrono::steady_clock::now();
  struct Tm {   // host time of the plan, on every return path
    ShearingBox *b; std::chrono::steady_clock::time_point t0;
    ~Tm() {b->t_plan_pull += std::chrono::duration<double>(std::chrono::steady_clock::now()
                                                           - t0).count(); b->n_plan_pull++;}
  } tm_{this, tp0};
  nppiece = 0;
  nprecv = 0;
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int ng = indcs.ng, nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int root = pm->root_level;
  const int my_rank = global_variable::my_rank;
  const int gids = pmy_pack->gids;
  const int nmb = pmy_pack->nmb_thispack;
  const Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  auto nxl = [pm, root](int l) {return pm->nmb_rootx1 << (l - root);};
  const bool zper = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
  // a receiver: the x1 ghosts of a face block (type 0) or of its coarse array
  // (type 1); the same function on the receiving and on the sending rank
  struct Recv {int gid, type, f, lr, ny, nz; long long j0, k0, joff; Real eps;};
  auto make_recv = [&](int gid, int level, std::int32_t lx2, std::int32_t lx3, int type,
                       int f) {
    Recv r;
    r.gid = gid; r.type = type; r.f = f;
    r.lr = (type == 0) ? level : level - 1;
    r.ny = (type == 0) ? nx2 : nx2/2;
    r.nz = (type == 0) ? nx3 : nx3/2;
    r.j0 = static_cast<long long>(lx2)*r.ny;
    r.k0 = static_cast<long long>(lx3)*r.nz;
    const long long ny_l = static_cast<long long>(pm->nmb_rootx2)*nx2 << (r.lr - root);
    const Real q = ((f == 0) ? yshear : -yshear)/(ly/static_cast<Real>(ny_l));
    const Real fl = std::floor(q);
    r.eps = q - fl;
    r.joff = static_cast<long long>(fl) % ny_l;
    if (r.joff < 0) r.joff += ny_l;
    return r;
  };
  bool bad = false;
  int maxny = 1, maxnz = 1;
  std::vector<std::array<int, kPNField>> pcs;
  // the pieces of (receiver r, sending block s), in the same order on both sides
  auto pieces = [&](const Recv &r, const ShearCand &s, int ridx) {
    const int lr = r.lr;
    const long long ny_l = static_cast<long long>(pm->nmb_rootx2)*nx2 << (lr - root);
    const long long nz_l = static_cast<long long>(pm->nmb_rootx3)*nx3 << (lr - root);
    const int nyb = r.ny + 2*ng + 6, nzb = r.nz + 2*ng;
    maxny = std::max(maxny, nyb); maxnz = std::max(maxnz, nzb);
    const long long y0 = r.j0 - ng - r.joff - 3, y1 = y0 + nyb;
    const long long z0 = r.k0 - ng, z1 = z0 + nzb;
    const int d = s.level - lr;
    long long sy0, sy1, sz0, sz1;
    if (d >= 0) {
      sy0 = static_cast<long long>(s.lx2)*(nx2 >> d); sy1 = sy0 + (nx2 >> d);
      sz0 = static_cast<long long>(s.lx3)*(nx3 >> d); sz1 = sz0 + (nx3 >> d);
    } else {
      sy0 = static_cast<long long>(s.lx2)*(static_cast<long long>(nx2) << (-d));
      sy1 = sy0 + (static_cast<long long>(nx2) << (-d));
      sz0 = static_cast<long long>(s.lx3)*(static_cast<long long>(nx3) << (-d));
      sz1 = sz0 + (static_cast<long long>(nx3) << (-d));
    }
    long long kya = FloorDiv(y0 - sy1, ny_l) + 1, kyb = FloorDiv(y1 - 1 - sy0, ny_l);
    long long kza = FloorDiv(z0 - sz1, nz_l) + 1, kzb = FloorDiv(z1 - 1 - sz0, nz_l);
    // non-periodic x3: no images; the x3 ghost rows beyond the boundary are
    // left to the physical boundary conditions (they run after this exchange)
    if (!zper) {kza = std::max(kza, 0LL); kzb = std::min(kzb, 0LL);}
    int seq = 0;
    for (long long ky=kya; ky<=kyb; ++ky) {
      const long long a0 = std::max(y0, sy0 + ky*ny_l), a1 = std::min(y1, sy1 + ky*ny_l);
      if (a1 <= a0) continue;
      for (long long kz=kza; kz<=kzb; ++kz) {
        const long long b0 = std::max(z0, sz0 + kz*nz_l), b1 = std::min(z1, sz1 + kz*nz_l);
        if (b1 <= b0) continue;
        if (d >= 0 && ((nx2 % (1 << d)) != 0 || (nx3 % (1 << d)) != 0 ||
                       ng*(1 << d) > nx1)) {
          bad = true;
          continue;
        }
        std::array<int, kPNField> p{};
        p[kPRGid] = r.gid; p[kPRType] = r.type; p[kPRFace] = r.f; p[kPRIdx] = ridx;
        p[kPSGid] = s.gid; p[kPSLoc] = (s.rank == my_rank) ? 1 : 0; p[kPSFace] = 1 - r.f;
        p[kPD] = d;
        p[kPY0] = static_cast<int>(a0 - y0); p[kPNy] = static_cast<int>(a1 - a0);
        p[kPYS] = static_cast<int>(a0 - (sy0 + ky*ny_l));
        p[kPZ0] = static_cast<int>(b0 - z0); p[kPNz] = static_cast<int>(b1 - b0);
        p[kPZS] = static_cast<int>(b0 - (sz0 + kz*nz_l));
        p[kPOff] = -1; p[kPSeq] = seq++;
        pcs.push_back(p);
      }
    }
  };
  // receivers of this rank (every candidate: local or remote sender)
  std::vector<std::tuple<int, int, int>> rl;   // local receivers (m, type, face)
  std::vector<Real> reps;
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[gids + m];
    for (int f=0; f<2; ++f) {
      if (ll.lx1 != ((f == 0) ? 0 : nxl(ll.level) - 1)) continue;
      for (int type=0; type<2; ++type) {
        if (type == 1 && !(coarse_on && ll.level > root)) continue;
        const Recv r = make_recv(gids + m, ll.level, ll.lx2, ll.lx3, type, f);
        const int ridx = static_cast<int>(rl.size());
        rl.emplace_back(m, type, f);
        reps.push_back(r.eps);
        for (const ShearCand &s : cand_[f][m]) pieces(r, s, ridx);
      }
    }
  }
  // senders of this rank for receivers on other ranks
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[gids + m];
    const ShearCand self{gids + m, my_rank, ll.level, ll.lx2, ll.lx3};
    for (int f=0; f<2; ++f) {
      if (ll.lx1 != ((f == 0) ? 0 : nxl(ll.level) - 1)) continue;
      for (const ShearCand &rc : cand_[f][m]) {
        if (rc.rank == my_rank) continue;
        for (int type=0; type<2; ++type) {
          if (type == 1 && !(coarse_on && rc.level > root)) continue;
          pieces(make_recv(rc.gid, rc.level, rc.lx2, rc.lx3, type, 1 - f), self, -1);
        }
      }
    }
  }
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "shearing box (x1-face levels along x2): the opposite face is too many "
              << "levels finer (nghost*2^d must fit in meshblock/nx1, nx2 and nx3 divisible "
              << "by 2^d)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // one message per peer rank and direction
  std::vector<std::array<long long, 8>> key;
  std::vector<int> rem;
  for (int n=0; n<static_cast<int>(pcs.size()); ++n) {
    const auto &p = pcs[n];
    const bool rloc = (p[kPRIdx] >= 0), sloc = (p[kPSLoc] == 1);
    if (rloc == sloc) continue;
    const int peer = rloc ? pm->rank_eachmb[p[kPSGid]] : pm->rank_eachmb[p[kPRGid]];
    key.push_back({rloc ? 0LL : 1LL, peer, p[kPRGid], p[kPRType], p[kPRFace], p[kPSGid],
                   p[kPSeq], static_cast<long long>(nvar_pull)*ng*p[kPNz]*p[kPNy]});
    rem.push_back(n);
  }
  std::vector<int> offset;
  int soff = 0, roff = 0;
#if MPI_PARALLEL_ENABLED
  LayoutPeerMessages(key, offset, pmsg_rank, pmsg_off, pmsg_size, pmsg_send, soff, roff);
#else
  std::vector<int> mr, mo, ms, md;
  LayoutPeerMessages(key, offset, mr, mo, ms, md, soff, roff);
#endif
  for (int i=0; i<static_cast<int>(rem.size()); ++i) pcs[rem[i]][kPOff] = offset[i];
  nppiece = static_cast<int>(pcs.size());
  nprecv = static_cast<int>(rl.size());
  npymax = maxny; npzmax = maxnz;
  if (static_cast<int>(ppiece.extent(0)) < std::max(1, nppiece)) {
    Kokkos::realloc(ppiece, std::max(1, nppiece), static_cast<int>(kPNField));
  }
  for (int n=0; n<nppiece; ++n) {
    for (int c=0; c<kPNField; ++c) ppiece.h_view(n,c) = pcs[n][c];
  }
  ppiece.template modify<HostMemSpace>();
  ppiece.template sync<DevExeSpace>();
  if (static_cast<int>(precv.extent(0)) < std::max(1, nprecv)) {
    Kokkos::realloc(precv, std::max(1, nprecv), 3);
    Kokkos::realloc(precv_eps, std::max(1, nprecv));
  }
  for (int r=0; r<nprecv; ++r) {
    precv.h_view(r,0) = std::get<0>(rl[r]);
    precv.h_view(r,1) = std::get<1>(rl[r]);
    precv.h_view(r,2) = std::get<2>(rl[r]);
    precv_eps.h_view(r) = reps[r];
  }
  precv.template modify<HostMemSpace>();
  precv.template sync<DevExeSpace>();
  precv_eps.template modify<HostMemSpace>();
  precv_eps.template sync<DevExeSpace>();
  const int nb1 = indcs.nx2 + 2*ng + 6, nb3 = indcs.nx3 + 2*ng;
  if (static_cast<int>(pbuf.extent(0)) < std::max(1, nprecv) ||
      static_cast<int>(pbuf.extent(1)) != nvar_pull) {
    Kokkos::realloc(pbuf, std::max(1, nprecv), nvar_pull, nb3, ng, nb1);
  }
  if (static_cast<int>(psendflat.extent(0)) < std::max(1, soff)) {
    Kokkos::realloc(psendflat, std::max(1, soff));
  }
  if (static_cast<int>(precvflat.extent(0)) < std::max(1, roff)) {
    Kokkos::realloc(precvflat, std::max(1, roff));
  }
#if MPI_PARALLEL_ENABLED
  const int nmsg = static_cast<int>(pmsg_rank.size());
  if (npreq_ < nmsg) {
    delete [] psreq_; delete [] prreq_;
    npreq_ = std::max(16, 2*nmsg);
    psreq_ = new MPI_Request[npreq_];
    prreq_ = new MPI_Request[npreq_];
  }
  for (int n=0; n<npreq_; ++n) {psreq_[n] = MPI_REQUEST_NULL; prreq_[n] = MPI_REQUEST_NULL;}
  pmsg_tag.assign(nmsg, 0);   // one message per (peer, direction): the order identifies it
  bool no_errors = true;
  for (int n=0; n<nmsg; ++n) {
    if (pmsg_send[n] == 1) continue;
    if (MPI_Irecv(precvflat.data() + pmsg_off[n], pmsg_size[n], MPI_ATHENA_REAL,
                  pmsg_rank[n], pmsg_tag[n], comm_pull, &(prreq_[n])) != MPI_SUCCESS) {
      no_errors = false;
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in posting receives (shearing box, face levels)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn bool ShearingBox::WaitPull()

bool ShearingBox::WaitPull() {
  bool no_errors = true;
#if MPI_PARALLEL_ENABLED
  for (int n=0; n<npreq_; ++n) {
    if (MPI_Wait(&(psreq_[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
    if (MPI_Wait(&(prreq_[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
  }
#endif
  return no_errors;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ShearingBoxCC::PullSend()
//! \brief The local senders convert their cells next to the face to each receiver's level

TaskStatus ShearingBoxCC::PullSend(DvceArray5D<Real> &a) {
  if (nppiece == 0) return TaskStatus::complete;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks, ng = indcs.ng;
  const int gids = pmy_pack->gids;
  const int nv = nvar_;
  auto pp = ppiece;
  auto pb = pbuf;
  auto sf = psendflat;
  const int nymax = npymax, nzmax = npzmax;
  const bool xslope = pull_ghost_valid;   // x slope at the face column only once the x1
                                          // ghosts hold collected values
  par_for("shr-pull-pack", DevExeSpace(), 0, nppiece-1, 0, nv*ng-1, 0, nzmax-1, 0, nymax-1,
  KOKKOS_LAMBDA(const int n, const int vd, const int zz, const int yy) {
    if (pp.d_view(n,kPSLoc) == 0) return;
    const int ny = pp.d_view(n,kPNy), nz = pp.d_view(n,kPNz);
    if (yy >= ny || zz >= nz) return;
    const int v = vd / ng, dep = vd - v*ng;
    const int m = pp.d_view(n,kPSGid) - gids;
    const int sface = pp.d_view(n,kPSFace);
    const int d = pp.d_view(n,kPD);
    const int r = pp.d_view(n,kPYS) + yy;     // source rows / x3 rows at the receiver's level
    const int kz = pp.d_view(n,kPZS) + zz;
    Real val;
    if (d >= 0) {
      const int f = 1 << d;
      const int i0 = (sface == 1) ? (ie - (dep + 1)*f + 1) : (is + dep*f);
      const int j0 = js + r*f, k0 = ks + kz*f;
      Real sum = 0.0;
      for (int k=0; k<f; ++k) {
        for (int j=0; j<f; ++j) {
          for (int i=0; i<f; ++i) sum += a(m, v, k0+k, j0+j, i0+i);
        }
      }
      val = sum/static_cast<Real>(f*f*f);
    } else {
      const int f = 1 << (-d);
      const Real rf = 1.0/static_cast<Real>(f);
      const int jp = js + r/f, kp = ks + kz/f, cd = dep/f;
      const int ip = (sface == 1) ? (ie - cd) : (is + cd);
      const Real oy = ((r % f) + 0.5)*rf - 0.5;
      const Real oz = ((kz % f) + 0.5)*rf - 0.5;
      const Real ox = ((sface == 1) ? -1.0 : 1.0)*(((dep % f) + 0.5)*rf - 0.5);
      const Real u0 = a(m, v, kp, jp, ip);
      Real dl = u0 - a(m, v, kp, jp-1, ip), dr = a(m, v, kp, jp+1, ip) - u0;
      const Real my = 4.0*PROLONG_CC_SLOPE(dl, dr);
      dl = u0 - a(m, v, kp-1, jp, ip); dr = a(m, v, kp+1, jp, ip) - u0;
      const Real mz = 4.0*PROLONG_CC_SLOPE(dl, dr);
      // the face column: its x1 ghosts hold the last collected values (one stage old), except
      // right after a regrid (no slope then)
      Real mx = 0.0;
      if (cd > 0 || xslope) {
        dl = u0 - a(m, v, kp, jp, ip-1); dr = a(m, v, kp, jp, ip+1) - u0;
        mx = 4.0*PROLONG_CC_SLOPE(dl, dr);
      }
      val = u0 + ox*mx + oy*my + oz*mz;
    }
    const int ridx = pp.d_view(n,kPRIdx);
    if (ridx >= 0) {
      pb(ridx, v, pp.d_view(n,kPZ0) + zz, dep, pp.d_view(n,kPY0) + yy) = val;
    } else {
      sf(pp.d_view(n,kPOff) + ((v*ng + dep)*nz + zz)*ny + yy) = val;
    }
  });
#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  bool no_errors = true;
  for (size_t n=0; n<pmsg_rank.size(); ++n) {
    if (pmsg_send[n] == 0) continue;
    if (MPI_Isend(psendflat.data() + pmsg_off[n], pmsg_size[n], MPI_ATHENA_REAL,
                  pmsg_rank[n], pmsg_tag[n], comm_pull, &(psreq_[n])) != MPI_SUCCESS) {
      no_errors = false;
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in posting sends (shearing box, face levels)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ShearingBoxCC::PullRecv()
//! \brief (coarse = false) the pieces from other ranks, then the remap of the fine receivers
//! into a; (coarse = true) the remap of the coarse receivers into the coarse array a

TaskStatus ShearingBoxCC::PullRecv(DvceArray5D<Real> &a, const bool coarse) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int ng = indcs.ng;
  const int nv = nvar_;
  auto pp = ppiece;
  auto pb = pbuf;
  if (!coarse) {
#if MPI_PARALLEL_ENABLED
    bool done = true, no_errors = true;
    for (size_t n=0; n<pmsg_rank.size(); ++n) {
      int test;
      if (MPI_Test(&(prreq_[n]), &test, MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
      if (!(static_cast<bool>(test))) done = false;
    }
    if (!no_errors) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "MPI error in testing receives (shearing box, face levels)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!done) return TaskStatus::incomplete;
    if (nppiece > 0) {
      auto rf = precvflat;
      const int nymax = npymax, nzmax = npzmax;
      par_for("shr-pull-unpk", DevExeSpace(), 0, nppiece-1, 0, nv*ng-1, 0, nzmax-1, 0,
              nymax-1,
      KOKKOS_LAMBDA(const int n, const int vd, const int zz, const int yy) {
        if (pp.d_view(n,kPSLoc) == 1 || pp.d_view(n,kPRIdx) < 0) return;
        const int ny = pp.d_view(n,kPNy), nz = pp.d_view(n,kPNz);
        if (yy >= ny || zz >= nz) return;
        const int v = vd / ng, dep = vd - v*ng;
        pb(pp.d_view(n,kPRIdx), v, pp.d_view(n,kPZ0) + zz, dep, pp.d_view(n,kPY0) + yy) =
            rf(pp.d_view(n,kPOff) + ((v*ng + dep)*nz + zz)*ny + yy);
      });
    }
#endif
  }
  if (!coarse) pull_ghost_valid = true;   // after this the x1 ghosts hold collected values
  if (nprecv == 0) return TaskStatus::complete;
  const int type = coarse ? 1 : 0;
  const int nyr = coarse ? indcs.cnx2 : indcs.nx2;
  const int nzr = coarse ? indcs.cnx3 : indcs.nx3;
  const int jb = (coarse ? indcs.cjs : indcs.js) - ng;
  const int kb = (coarse ? indcs.cks : indcs.ks) - ng;
  const int il = coarse ? indcs.cis : indcs.is;
  const int iu = coarse ? indcs.cie : indcs.ie;
  const int nrow = nyr + 2*ng, nb = nrow + 6, nzb = nzr + 2*ng;
  auto rlist = precv;
  auto reps = precv_eps;
  const ReconstructionMethod rcon = rcon_pull;
  int scr_lvl = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nb)*2;
  par_for_outer("shr-pull-remap", DevExeSpace(), scr_size, scr_lvl, 0, nprecv-1, 0, nv-1,
                0, nzb-1, 0, ng-1,
  KOKKOS_LAMBDA(TeamMember_t member, const int r, const int v, const int zz, const int dep) {
    if (rlist.d_view(r,1) != type) return;
    ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nb);
    ScrArray1D<Real> ust(member.team_scratch(scr_lvl), nb);
    par_for_inner(member, 0, nb-1, [&](const int b) {
      a_(b) = pb(r, v, zz, dep, b);
      ust(b) = 0.0;
    });
    member.team_barrier();
    const Real eps = reps.d_view(r);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, 3, nrow+3, eps, a_, ust);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, 3, nrow+3, eps, a_, ust);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
      case ReconstructionMethod::wenoz:
        PPMX_RemapFlx(member, 3, nrow+3, eps, a_, ust);
        break;
      default:
        break;
    }
    member.team_barrier();
    const int m = rlist.d_view(r,0);
    const int icol = (rlist.d_view(r,2) == 0) ? (il - 1 - dep) : (iu + 1 + dep);
    par_for_inner(member, 0, nrow-1, [&](const int jl) {
      a(m, v, kb + zz, jb + jl, icol) = a_(jl+3) - (ust(jl+4) - ust(jl+3));
    });
  });
  return TaskStatus::complete;
}
