//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file shearing_box_flux.cpp
//! \brief Matching of the x1 fluxes across the shear-periodic boundary.
//! The two x1 faces of the box are one surface, sheared by yshear.  Each side computes its
//! face fluxes from its own sheared ghost cells, so the mass leaving through one face differs
//! from the mass entering through the other (uniform meshes: small; across a level change:
//! large).  Here, on every column of the faces (one level along x2) the
//! finer side is authoritative (equal levels: the outer face); every block on the other side
//! gathers the authoritative face fluxes over its rows shifted back by the shear offset (plus
//! 3 rows below and 2 above for the reconstruction), restricted 2x2 to its level when they are
//! finer, remaps them by the fractional shift as the ghost cells are remapped
//! (DC/PLM/PPMX_RemapFlx), and replaces its own x1 face fluxes.  The sums are conserved, so
//! mass and momentum leave one face exactly as they enter the other (up to round-off).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "shearing_box.hpp"
#include "remap_fluxes.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxCC::PlanFluxMatch()
//! \brief With the current yshear: the receiving face blocks of this rank, the pieces that
//! this rank sends or receives, the buffers; posts the MPI receives.

void ShearingBoxCC::PlanFluxMatch() {
  struct Tm {   // host time of the plan (both kinds), on every return path
    ShearingBox *b; std::chrono::steady_clock::time_point t0;
    ~Tm() {b->t_plan_flux += std::chrono::duration<double>(std::chrono::steady_clock::now()
                                                           - t0).count(); b->n_plan_flux++;}
  } tm_{this, std::chrono::steady_clock::now()};
  nfpiece = 0;
  nfrecv = 0;
  if (!flux_on) return;
  if (face_levels) {PlanFluxPair(); return;}
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int nx2 = indcs.nx2, nk = indcs.nx3;
  const int root = pm->root_level;
  const int my_rank = global_variable::my_rank;
  const int gids = pmy_pack->gids;
  const Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  const int nbuf = nx2 + 6;
  std::map<std::tuple<int, int, int, int>, int> loc;
  for (int g=0; g<pm->nmb_total; ++g) {
    const auto &ll = pm->lloc_eachmb[g];
    loc[std::make_tuple(ll.level, ll.lx1, ll.lx2, ll.lx3)] = g;
  }
  auto find = [&loc](int l, int i, int j, int k) {
    auto it = loc.find(std::make_tuple(l, i, j, k));
    return (it == loc.end()) ? -1 : it->second;
  };
  auto nxl = [pm, root](int l) {return pm->nmb_rootx1 << (l - root);};
  std::vector<std::vector<int>> pcs;
  std::vector<std::pair<int, int>> rl;
  std::vector<Real> reps;
  int soff = 0, roff = 0;
  bool bad = false;
  for (int g=0; g<pm->nmb_total; ++g) {
    const auto &ll = pm->lloc_eachmb[g];
    const int l = ll.level;
    for (int f=0; f<2; ++f) {
      if (ll.lx1 != ((f == 0) ? 0 : nxl(l) - 1)) continue;
      const int fo = 1 - f;
      int lo = -1;
      if (find(l, (fo == 0) ? 0 : nxl(l) - 1, ll.lx2, ll.lx3) >= 0) {
        lo = l;
      } else if (l > root &&
                 find(l-1, (fo == 0) ? 0 : nxl(l-1) - 1, ll.lx2 >> 1, ll.lx3 >> 1) >= 0) {
        lo = l - 1;
      } else if (find(l+1, (fo == 0) ? 0 : nxl(l+1) - 1, 2*ll.lx2, 2*ll.lx3) >= 0) {
        lo = l + 1;
      }
      if (lo < 0) {bad = true; continue;}
      if (!((lo > l) || (lo == l && f == 0))) continue;   // authoritative side: sends
      const bool rloc = (pm->rank_eachmb[g] == my_rank);
      const int ls = lo;
      const int rb = (ls == l) ? nx2 : nx2/2;
      const long long ny = static_cast<long long>(pm->nmb_rootx2 << (l - root))*nx2;
      const Real q = ((f == 0) ? yshear : -yshear)/(ly/static_cast<Real>(ny));
      const Real fl = std::floor(q);
      const Real eps = q - fl;
      long long joff = static_cast<long long>(fl) % ny;
      if (joff < 0) joff += ny;
      int ridx = -1;
      if (rloc) {
        ridx = static_cast<int>(rl.size());
        rl.emplace_back(g - gids, f);
        reps.push_back(eps);
      }
      long long pos = static_cast<long long>(ll.lx2)*nx2 - joff - 3;
      int dst = 0, npc = 0;
      while (dst < nbuf) {
        const long long gr = ((pos % ny) + ny) % ny;
        const int by = static_cast<int>(gr/rb);
        const int r0 = static_cast<int>(gr - static_cast<long long>(by)*rb);
        const int len = std::min(rb - r0, nbuf - dst);
        const int nzh = (ls == l) ? 1 : 2;
        for (int zh=0; zh<nzh; ++zh) {
          const int sg = find(ls, (fo == 0) ? 0 : nxl(ls) - 1, by,
                              (ls == l) ? ll.lx3 : 2*ll.lx3 + zh);
          if (sg < 0) {bad = true; continue;}
          const bool sloc = (pm->rank_eachmb[sg] == my_rank);
          const int nkp = (ls == l) ? nk : nk/2;
          if (rloc || sloc) {
            std::vector<int> p(kNField, 0);
            p[kRGid] = g; p[kSGid] = sg; p[kSFace] = fo; p[kR0] = r0; p[kLen] = len;
            p[kDst] = dst; p[kKofs] = (ls == l) ? 0 : zh*(nk/2); p[kNk] = nkp;
            p[kRestr] = (ls != l) ? 1 : 0; p[kQ] = npc; p[kRIdx] = ridx;
            p[kSLoc] = sloc ? 1 : 0;
            p[kRRank] = pm->rank_eachmb[g]; p[kSRank] = pm->rank_eachmb[sg];
            const int size = nvar_*nkp*len;
            p[kOff] = -1;
            if (rloc && !sloc) {p[kOff] = roff; roff += size;}
            if (sloc && !rloc) {p[kOff] = soff; soff += size;}
            pcs.push_back(p);
          }
          npc++;
        }
        pos += len;
        dst += len;
      }
      if (npc > 22) bad = true;    // MPI tags 40 + piece
    }
  }
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "shear flux matching: unexpected face levels or too many pieces"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nfpiece = static_cast<int>(pcs.size());
  nfrecv = static_cast<int>(rl.size());
  if (static_cast<int>(fpiece.extent(0)) < std::max(1, nfpiece)) {
    Kokkos::realloc(fpiece, std::max(1, nfpiece), static_cast<int>(kNField));
  }
  for (int n=0; n<nfpiece; ++n) {
    for (int c=0; c<kNField; ++c) fpiece.h_view(n,c) = pcs[n][c];
  }
  fpiece.template modify<HostMemSpace>();
  fpiece.template sync<DevExeSpace>();
  if (static_cast<int>(frecv_list.extent(0)) < std::max(1, nfrecv)) {
    Kokkos::realloc(frecv_list, std::max(1, nfrecv), 2);
    Kokkos::realloc(frecv_eps, std::max(1, nfrecv));
  }
  for (int r=0; r<nfrecv; ++r) {
    frecv_list.h_view(r,0) = rl[r].first;
    frecv_list.h_view(r,1) = rl[r].second;
    frecv_eps.h_view(r) = reps[r];
  }
  frecv_list.template modify<HostMemSpace>();
  frecv_list.template sync<DevExeSpace>();
  frecv_eps.template modify<HostMemSpace>();
  frecv_eps.template sync<DevExeSpace>();
  if (static_cast<int>(fbuf.extent(0)) < std::max(1, nfrecv) ||
      static_cast<int>(fbuf.extent(1)) != nvar_) {
    Kokkos::realloc(fbuf, std::max(1, nfrecv), nvar_, nk, nbuf);
  }
  if (static_cast<int>(fsendflat.extent(0)) < std::max(1, soff)) {
    Kokkos::realloc(fsendflat, std::max(1, soff));
  }
  if (static_cast<int>(frecvflat.extent(0)) < std::max(1, roff)) {
    Kokkos::realloc(frecvflat, std::max(1, roff));
  }
#if MPI_PARALLEL_ENABLED
  if (nfreq_ < nfpiece) {
    delete [] fsreq_; delete [] frreq_;
    nfreq_ = std::max(16, 2*nfpiece);
    fsreq_ = new MPI_Request[nfreq_];
    frreq_ = new MPI_Request[nfreq_];
  }
  for (int n=0; n<nfreq_; ++n) {fsreq_[n] = MPI_REQUEST_NULL; frreq_[n] = MPI_REQUEST_NULL;}
  bool no_errors = true;
  for (int n=0; n<nfpiece; ++n) {
    const auto &p = pcs[n];
    if (p[kRIdx] < 0 || p[kSLoc] == 1) continue;   // receive only: local receiver, remote sender
    const int size = nvar_*p[kNk]*p[kLen];
    const int tag = CreateBvals_MPI_Tag(p[kRGid] - gids, 40 + p[kQ]);
    if (MPI_Irecv(frecvflat.data() + p[kOff], size, MPI_ATHENA_REAL, p[kSRank], tag,
                  comm_sbox, &(frreq_[n])) != MPI_SUCCESS) {
      no_errors = false;
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in posting receives (shear flux matching)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ShearingBoxCC::SendFluxMatch()
//! \brief The authoritative face fluxes of the local senders: into the strips of local
//! receivers, or packed and sent

TaskStatus ShearingBoxCC::SendFluxMatch(DvceFaceFld5D<Real> &flx) {
  if (flux_on && face_levels) return SendFluxPair(flx);
  if (!flux_on || nfpiece == 0) return TaskStatus::complete;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks;
  const int nk = indcs.nx3, nx2 = indcs.nx2;
  const int gids = pmy_pack->gids;
  const int nv = nvar_;
  auto fp = fpiece;
  auto fb = fbuf;
  auto sf = fsendflat;
  auto f1 = flx.x1f;
  par_for("shr-flx-pack", DevExeSpace(), 0, nfpiece-1, 0, nv-1, 0, nk-1, 0, nx2-1,
  KOKKOS_LAMBDA(const int n, const int v, const int kk, const int rr) {
    if (fp.d_view(n,kSLoc) == 0) return;
    const int nkp = fp.d_view(n,kNk), len = fp.d_view(n,kLen);
    if (kk >= nkp || rr >= len) return;
    const int m = fp.d_view(n,kSGid) - gids;
    const int iface = (fp.d_view(n,kSFace) == 0) ? is : (ie + 1);
    const int r = fp.d_view(n,kR0) + rr;
    Real val;
    if (fp.d_view(n,kRestr) == 1) {
      const int k0 = ks + 2*kk, j0 = js + 2*r;
      val = 0.25*((f1(m,v,k0,j0,iface) + f1(m,v,k0,j0+1,iface)) +
                  (f1(m,v,k0+1,j0,iface) + f1(m,v,k0+1,j0+1,iface)));
    } else {
      val = f1(m,v,ks+kk,js+r,iface);
    }
    const int ridx = fp.d_view(n,kRIdx);
    if (ridx >= 0) {
      fb(ridx, v, fp.d_view(n,kKofs) + kk, fp.d_view(n,kDst) + rr) = val;
    } else {
      sf(fp.d_view(n,kOff) + (v*nkp + kk)*len + rr) = val;
    }
  });
#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  bool no_errors = true;
  Mesh *pm = pmy_pack->pmesh;
  for (int n=0; n<nfpiece; ++n) {
    if (fpiece.h_view(n,kSLoc) == 0 || fpiece.h_view(n,kRIdx) >= 0) continue;
    const int rrank = fpiece.h_view(n,kRRank);
    const int size = nvar_*fpiece.h_view(n,kNk)*fpiece.h_view(n,kLen);
    const int lid = fpiece.h_view(n,kRGid) - pm->gids_eachrank[rrank];
    const int tag = CreateBvals_MPI_Tag(lid, 40 + fpiece.h_view(n,kQ));
    if (MPI_Isend(fsendflat.data() + fpiece.h_view(n,kOff), size, MPI_ATHENA_REAL, rrank,
                  tag, comm_sbox, &(fsreq_[n])) != MPI_SUCCESS) {
      no_errors = false;
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in posting sends (shear flux matching)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ShearingBoxCC::RecvFluxMatch()
//! \brief Strips from other ranks, then the remap and the replacement of the x1 face fluxes
//! of the local receivers

TaskStatus ShearingBoxCC::RecvFluxMatch(DvceFaceFld5D<Real> &flx, ReconstructionMethod rcon) {
  if (flux_on && face_levels) return RecvFluxPair(flx);
  if (!flux_on || nfpiece == 0) return TaskStatus::complete;
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks;
  const int nk = indcs.nx3, nx2 = indcs.nx2;
  const int nv = nvar_;
  auto fp = fpiece;
  auto fb = fbuf;
#if MPI_PARALLEL_ENABLED
  bool done = true, no_errors = true;
  for (int n=0; n<nfpiece; ++n) {
    int test;
    if (MPI_Test(&(frreq_[n]), &test, MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
    if (!(static_cast<bool>(test))) done = false;
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in testing receives (shear flux matching)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!done) return TaskStatus::incomplete;
  auto rf = frecvflat;
  par_for("shr-flx-unpk", DevExeSpace(), 0, nfpiece-1, 0, nv-1, 0, nk-1, 0, nx2-1,
  KOKKOS_LAMBDA(const int n, const int v, const int kk, const int rr) {
    if (fp.d_view(n,kSLoc) == 1 || fp.d_view(n,kRIdx) < 0) return;
    const int nkp = fp.d_view(n,kNk), len = fp.d_view(n,kLen);
    if (kk >= nkp || rr >= len) return;
    fb(fp.d_view(n,kRIdx), v, fp.d_view(n,kKofs) + kk, fp.d_view(n,kDst) + rr) =
        rf(fp.d_view(n,kOff) + (v*nkp + kk)*len + rr);
  });
#endif
  if (nfrecv == 0) return TaskStatus::complete;
  auto rlist = frecv_list;
  auto reps = frecv_eps;
  auto f1 = flx.x1f;
  const int nb = nx2 + 6;
  int scr_lvl = 0;
  size_t scr_size = ScrArray1D<Real>::shmem_size(nb)*2;
  par_for_outer("shr-flx-remap", DevExeSpace(), scr_size, scr_lvl, 0, nfrecv-1, 0, nv-1,
                0, nk-1,
  KOKKOS_LAMBDA(TeamMember_t member, const int r, const int v, const int kk) {
    ScrArray1D<Real> a_(member.team_scratch(scr_lvl), nb);
    ScrArray1D<Real> ust(member.team_scratch(scr_lvl), nb);
    par_for_inner(member, 0, nb-1, [&](const int b) {
      a_(b) = fb(r, v, kk, b);
      ust(b) = 0.0;
    });
    member.team_barrier();
    const Real eps = reps.d_view(r);
    switch (rcon) {
      case ReconstructionMethod::dc:
        DC_RemapFlx(member, 3, nx2+3, eps, a_, ust);
        break;
      case ReconstructionMethod::plm:
        PLM_RemapFlx(member, 3, nx2+3, eps, a_, ust);
        break;
      case ReconstructionMethod::ppm4:
      case ReconstructionMethod::ppmx:
      case ReconstructionMethod::wenoz:
        PPMX_RemapFlx(member, 3, nx2+3, eps, a_, ust);
        break;
      default:
        break;
    }
    member.team_barrier();
    const int m = rlist.d_view(r,0);
    const int iface = (rlist.d_view(r,1) == 0) ? is : (ie + 1);
    par_for_inner(member, 0, nx2-1, [&](const int jl) {
      f1(m, v, ks+kk, js+jl, iface) = a_(jl+3) - (ust(jl+4) - ust(jl+3));
    });
  });
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ShearingBoxCC::ClearFluxMatch()
//! \brief Waits for the sends and receives of this stage

TaskStatus ShearingBoxCC::ClearFluxMatch() {
#if MPI_PARALLEL_ENABLED
  if (!flux_on) return TaskStatus::complete;
  bool no_errors = true;
  for (int n=0; n<nqreq_; ++n) {
    if (MPI_Wait(&(qsreq_[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
    if (MPI_Wait(&(qrreq_[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
  }
  for (int n=0; n<nfreq_; ++n) {
    if (MPI_Wait(&(fsreq_[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
    if (MPI_Wait(&(frreq_[n]), MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in clearing (shear flux matching)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ShearingBoxCC::PlanFluxPair()
//! \brief Face levels: for every block Y on a face and every block X on the
//! opposite face that overlaps it after the shear shift and is authoritative there (finer, or
//! the same level and on the outer face), a piece: the rows / x3 rows of Y it covers, the
//! offsets that place X's rows in Y's frame (in X cells: X row jl occupies [jl + base + frac,
//! jl + base + frac + 1) and Y row r occupies [r 2^d, (r+1) 2^d)), and the MPI message.

void ShearingBoxCC::PlanFluxPair() {
  nqpiece = 0;
  Mesh *pm = pmy_pack->pmesh;
  const bool zper = (pm->mesh_bcs[BoundaryFace::inner_x3] == BoundaryFlag::periodic);
  auto &indcs = pm->mb_indcs;
  const int nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int root = pm->root_level;
  const int my_rank = global_variable::my_rank;
  const Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  auto nxl = [pm, root](int l) {return pm->nmb_rootx1 << (l - root);};
  std::vector<int> face[2];
  for (int g=0; g<pm->nmb_total; ++g) {
    const auto &ll = pm->lloc_eachmb[g];
    if (ll.lx1 == 0) face[0].push_back(g);
    if (ll.lx1 == nxl(ll.level) - 1) face[1].push_back(g);
  }
  std::vector<std::vector<int>> pcs;
  std::vector<Real> fr;
  std::map<std::pair<int, int>, std::vector<int>> bypair;   // (Y gid, X gid) -> piece ids
  std::vector<std::pair<int, int>> pair_order;
  bool bad = false;
  // receivers in gid order, then face
  std::vector<std::pair<int, int>> ys;
  for (int f=0; f<2; ++f) for (int g : face[f]) ys.emplace_back(g, f);
  std::stable_sort(ys.begin(), ys.end(),
                   [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                     return a.first < b.first;});
  int cur = -1;
  std::map<int, int> pair_of;
  for (const auto &yf : ys) {
    const int yg = yf.first, fy = yf.second;
    if (yg != cur) {cur = yg; pair_of.clear();}
    const auto &yl = pm->lloc_eachmb[yg];
    const bool rloc = (pm->rank_eachmb[yg] == my_rank);
    for (int xg : face[1 - fy]) {
      const auto &xl = pm->lloc_eachmb[xg];
      const int d = xl.level - yl.level;
      if (d < 0 || (d == 0 && fy == 1)) continue;   // Y is authoritative there
      const long long f2 = 1LL << d;
      if ((nx2 % f2) != 0 || (nx3 % f2) != 0) {bad = true; continue;}
      const long long nyx = static_cast<long long>(pm->nmb_rootx2)*nx2 << (xl.level - root);
      const long long nzx = static_cast<long long>(pm->nmb_rootx3)*nx3 << (xl.level - root);
      // x3: X k row kx sits at kx + bz in Y's frame (X cells); Y spans [0, nx3 f2)
      const long long zy0 = static_cast<long long>(yl.lx3)*nx3*f2;
      const long long zx0 = static_cast<long long>(xl.lx3)*nx3;
      // y: shift of X's rows in Y's frame, in X cells
      const Real sig = yshear/(ly/static_cast<Real>(nyx));
      const Real del = (fy == 0) ? sig : -sig;
      const Real dfl = std::floor(del);
      const Real dfr = del - dfl;
      long long dint = static_cast<long long>(dfl) % nyx;
      if (dint < 0) dint += nyx;
      const long long yy0 = static_cast<long long>(yl.lx2)*nx2*f2;
      const long long yx0 = static_cast<long long>(xl.lx2)*nx2 + dint;
      const long long spany = static_cast<long long>(nx2)*f2, spanz = static_cast<long long>(nx3)*f2;
      for (long long kz=(zper ? -2 : 0); kz<=(zper ? 2 : 0); ++kz) {   // (S2c) images only in a
                                                                      // periodic x3
        const long long bz = zx0 + kz*nzx - zy0;
        if (!(bz < spanz && bz + nx3 > 0)) continue;
        const int k0 = static_cast<int>(std::max(0LL, bz/f2));
        const int k1 = static_cast<int>(std::min(static_cast<long long>(nx3),
                                                 (bz + nx3 + f2 - 1)/f2));
        if (k1 <= k0) continue;
        for (long long ky=-2; ky<=2; ++ky) {
          const long long by = yx0 + ky*nyx - yy0;    // X row 0 at [by + dfr, by + dfr + 1)
          const Real lo = static_cast<Real>(by) + dfr, hi = lo + nx2;
          if (!(lo < static_cast<Real>(spany) && hi > 0.0)) continue;
          const int r0 = std::max(0, static_cast<int>(std::floor(lo/static_cast<Real>(f2))));
          const int r1 = std::min(nx2, static_cast<int>(std::ceil(hi/static_cast<Real>(f2))));
          if (r1 <= r0) continue;
          if (pair_of.find(xg) == pair_of.end()) {
            pair_of.emplace(xg, static_cast<int>(pair_of.size()));
          }
          const bool sloc = (pm->rank_eachmb[xg] == my_rank);
          if (!(rloc || sloc)) continue;
          std::vector<int> p(kQNField, 0);
          p[kQRGid] = yg; p[kQRIdx] = rloc ? 1 : -1; p[kQRFace] = fy;
          p[kQSGid] = xg; p[kQSLoc] = sloc ? 1 : 0; p[kQSFace] = 1 - fy; p[kQD] = d;
          p[kQR0] = r0; p[kQNy] = r1 - r0; p[kQK0] = k0; p[kQNz] = k1 - k0;
          p[kQBase] = static_cast<int>(by); p[kQBaseZ] = static_cast<int>(bz);
          p[kQOff] = -1; p[kQPair] = pair_of[xg];
          auto key = std::make_pair(yg, xg);
          if (bypair.find(key) == bypair.end()) pair_order.push_back(key);
          bypair[key].push_back(static_cast<int>(pcs.size()));
          pcs.push_back(p);
          fr.push_back(dfr);
        }
      }
    }
  }
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "shear flux matching (face levels): level difference too large for the "
              << "block size" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // flat buffers per message (pieces of one pair are contiguous)
  std::vector<int> order;
  int soff = 0, roff = 0;
#if MPI_PARALLEL_ENABLED
  qmsg_rank.clear(); qmsg_off.clear(); qmsg_size.clear(); qmsg_tag.clear(); qmsg_send.clear();
#endif
  for (const auto &key : pair_order) {
    const auto &ids = bypair[key];
    const bool rloc = (pm->rank_eachmb[key.first] == my_rank);
    const bool sloc = (pm->rank_eachmb[key.second] == my_rank);
    if (rloc != sloc) {
      int &off = rloc ? roff : soff;
      const int start = off;
      for (int id : ids) {
        pcs[id][kQOff] = off;
        off += (nvar_ + 1)*pcs[id][kQNz]*pcs[id][kQNy];
      }
#if MPI_PARALLEL_ENABLED
      qmsg_rank.push_back(rloc ? pm->rank_eachmb[key.second] : pm->rank_eachmb[key.first]);
      qmsg_off.push_back(start);
      qmsg_size.push_back(off - start);
      const int lid = key.first - pm->gids_eachrank[pm->rank_eachmb[key.first]];
      qmsg_tag.push_back(CreateBvals_MPI_Tag(lid, pcs[ids[0]][kQPair]));
      qmsg_send.push_back(sloc ? 1 : 0);
#endif
    }
    for (int id : ids) order.push_back(id);
  }
  nqpiece = static_cast<int>(order.size());
  if (static_cast<int>(qpiece.extent(0)) < std::max(1, nqpiece)) {
    Kokkos::realloc(qpiece, std::max(1, nqpiece), static_cast<int>(kQNField));
    Kokkos::realloc(qfrac, std::max(1, nqpiece));
  }
  for (int n=0; n<nqpiece; ++n) {
    for (int c=0; c<kQNField; ++c) qpiece.h_view(n,c) = pcs[order[n]][c];
    qfrac.h_view(n) = fr[order[n]];
  }
  qpiece.template modify<HostMemSpace>();
  qpiece.template sync<DevExeSpace>();
  qfrac.template modify<HostMemSpace>();
  qfrac.template sync<DevExeSpace>();
  const int nmb = pmy_pack->nmb_thispack;
  if (static_cast<int>(qacc.extent(0)) < std::max(1, nmb) ||
      static_cast<int>(qacc.extent(2)) != nvar_ + 1) {
    Kokkos::realloc(qacc, std::max(1, nmb), 2, nvar_ + 1, nx3, nx2);
  }
  if (static_cast<int>(qsendflat.extent(0)) < std::max(1, soff)) {
    Kokkos::realloc(qsendflat, std::max(1, soff));
  }
  if (static_cast<int>(qrecvflat.extent(0)) < std::max(1, roff)) {
    Kokkos::realloc(qrecvflat, std::max(1, roff));
  }
#if MPI_PARALLEL_ENABLED
  const int nmsg = static_cast<int>(qmsg_rank.size());
  if (nqreq_ < nmsg) {
    delete [] qsreq_; delete [] qrreq_;
    nqreq_ = std::max(16, 2*nmsg);
    qsreq_ = new MPI_Request[nqreq_];
    qrreq_ = new MPI_Request[nqreq_];
  }
  for (int n=0; n<nqreq_; ++n) {qsreq_[n] = MPI_REQUEST_NULL; qrreq_[n] = MPI_REQUEST_NULL;}
  bool no_errors = true;
  for (int n=0; n<nmsg; ++n) {
    if (qmsg_send[n] == 1) continue;
    if (MPI_Irecv(qrecvflat.data() + qmsg_off[n], qmsg_size[n], MPI_ATHENA_REAL, qmsg_rank[n],
                  qmsg_tag[n], comm_qflx, &(qrreq_[n])) != MPI_SUCCESS) {
      no_errors = false;
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in posting receives (flux pairs)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ShearingBoxCC::SendFluxPair()
//! \brief The authoritative blocks integrate their x1 face fluxes (piecewise linear in x2,
//! limited, none at the block's own first and last rows) over each receiving cell; the sums
//! and the covered fraction go to the receiver's accumulators or to the flat buffer.

TaskStatus ShearingBoxCC::SendFluxPair(DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks;
  const int nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int gids = pmy_pack->gids;
  const int nv = nvar_;
  Kokkos::deep_copy(DevExeSpace(), qacc, 0.0);
  if (nqpiece == 0) return TaskStatus::complete;
  auto qp = qpiece;
  auto qf = qfrac;
  auto acc = qacc;
  auto sf = qsendflat;
  auto f1 = flx.x1f;
  par_for("shr-qflx-pack", DevExeSpace(), 0, nqpiece-1, 0, nv, 0, nx3-1, 0, nx2-1,
  KOKKOS_LAMBDA(const int n, const int vv, const int kk, const int rr) {
    if (qp.d_view(n,kQSLoc) == 0) return;
    const int ny = qp.d_view(n,kQNy), nz = qp.d_view(n,kQNz);
    if (rr >= ny || kk >= nz) return;
    const int r = qp.d_view(n,kQR0) + rr, k = qp.d_view(n,kQK0) + kk;
    const int f2 = 1 << qp.d_view(n,kQD);
    const int mx = qp.d_view(n,kQSGid) - gids;
    const int ix = (qp.d_view(n,kQSFace) == 0) ? is : (ie + 1);
    const Real base = static_cast<Real>(qp.d_view(n,kQBase)) + qf.d_view(n);
    const int bz = qp.d_view(n,kQBaseZ);
    const Real ylo = static_cast<Real>(r*f2), yhi = static_cast<Real>((r + 1)*f2);
    const int jlo = Kokkos::max(0, static_cast<int>(floor(ylo - base)) - 1);
    const int jhi = Kokkos::min(nx2 - 1, static_cast<int>(ceil(yhi - base)));
    const int kxlo = Kokkos::max(0, k*f2 - bz);
    const int kxhi = Kokkos::min(nx3 - 1, (k + 1)*f2 - 1 - bz);
    Real sum = 0.0;
    for (int kx=kxlo; kx<=kxhi; ++kx) {
      for (int jl=jlo; jl<=jhi; ++jl) {
        const Real c0 = base + jl;
        const Real lo = fmax(c0, ylo), hi = fmin(c0 + 1.0, yhi);
        if (hi <= lo) continue;
        if (vv == nv) {
          sum += hi - lo;
        } else {
          const Real fc = f1(mx, vv, ks+kx, js+jl, ix);
          Real slope = 0.0;
          if (jl > 0 && jl < nx2 - 1) {
            const Real dql = fc - f1(mx, vv, ks+kx, js+jl-1, ix);
            const Real dqr = f1(mx, vv, ks+kx, js+jl+1, ix) - fc;
            if (dql*dqr > 0.0) slope = 2.0*dql*dqr/(dql + dqr);
          }
          sum += (hi - lo)*(fc + slope*(0.5*(lo + hi) - (c0 + 0.5)));
        }
      }
    }
    const Real val = sum/static_cast<Real>(f2*f2);
    if (qp.d_view(n,kQRIdx) >= 0) {
      Kokkos::atomic_add(&acc(qp.d_view(n,kQRGid) - gids, qp.d_view(n,kQRFace), vv, k, r), val);
    } else {
      sf(qp.d_view(n,kQOff) + (vv*nz + kk)*ny + rr) = val;
    }
  });
#if MPI_PARALLEL_ENABLED
  Kokkos::fence();
  bool no_errors = true;
  for (size_t n=0; n<qmsg_rank.size(); ++n) {
    if (qmsg_send[n] == 0) continue;
    if (MPI_Isend(qsendflat.data() + qmsg_off[n], qmsg_size[n], MPI_ATHENA_REAL, qmsg_rank[n],
                  qmsg_tag[n], comm_qflx, &(qsreq_[n])) != MPI_SUCCESS) {
      no_errors = false;
    }
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in posting sends (flux pairs)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn TaskStatus ShearingBoxCC::RecvFluxPair()
//! \brief Sums from other ranks, then every face cell: F = F_own (1 - covered) + received

TaskStatus ShearingBoxCC::RecvFluxPair(DvceFaceFld5D<Real> &flx) {
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, ie = indcs.ie, js = indcs.js, ks = indcs.ks;
  const int nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int gids = pmy_pack->gids;
  const int nv = nvar_;
  auto qp = qpiece;
  auto acc = qacc;
#if MPI_PARALLEL_ENABLED
  bool done = true, no_errors = true;
  for (size_t n=0; n<qmsg_rank.size(); ++n) {
    int test;
    if (MPI_Test(&(qrreq_[n]), &test, MPI_STATUS_IGNORE) != MPI_SUCCESS) no_errors = false;
    if (!(static_cast<bool>(test))) done = false;
  }
  if (!no_errors) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "MPI error in testing receives (flux pairs)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (!done) return TaskStatus::incomplete;
  if (nqpiece > 0) {
    auto rf = qrecvflat;
    par_for("shr-qflx-unpk", DevExeSpace(), 0, nqpiece-1, 0, nv, 0, nx3-1, 0, nx2-1,
    KOKKOS_LAMBDA(const int n, const int vv, const int kk, const int rr) {
      if (qp.d_view(n,kQSLoc) == 1 || qp.d_view(n,kQRIdx) < 0) return;
      const int ny = qp.d_view(n,kQNy), nz = qp.d_view(n,kQNz);
      if (rr >= ny || kk >= nz) return;
      Kokkos::atomic_add(&acc(qp.d_view(n,kQRGid) - gids, qp.d_view(n,kQRFace), vv,
                              qp.d_view(n,kQK0) + kk, qp.d_view(n,kQR0) + rr),
                         rf(qp.d_view(n,kQOff) + (vv*nz + kk)*ny + rr));
    });
  }
#endif
  const int nmb = pmy_pack->nmb_thispack;
  auto f1 = flx.x1f;
  auto &mbbcs = pmy_pack->pmb->mb_bcs;
  par_for("shr-qflx-apply", DevExeSpace(), 0, nmb-1, 0, 1, 0, nv-1, 0, nx3-1, 0, nx2-1,
  KOKKOS_LAMBDA(const int m, const int f, const int v, const int k, const int j) {
    const auto face = (f == 0) ? BoundaryFace::inner_x1 : BoundaryFace::outer_x1;
    if (mbbcs.d_view(m,face) != BoundaryFlag::shear_periodic) return;
    const Real cov = acc(m, f, nv, k, j);
    if (cov <= 0.0) return;
    const int ix = (f == 0) ? is : (ie + 1);
    f1(m, v, ks+k, js+j, ix) = f1(m, v, ks+k, js+j, ix)*(1.0 - cov) + acc(m, f, v, k, j);
  });
  return TaskStatus::complete;
}
