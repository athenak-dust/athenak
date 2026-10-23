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
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
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
  const int nmb = pmy_pack->nmb_thispack;
  const Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  const int nbuf = nx2 + 6;
  auto nxl = [pm, root](int l) {return pm->nmb_rootx1 << (l - root);};
  std::vector<std::array<int, kNField>> pcs;
  std::vector<std::pair<int, int>> rl;
  std::vector<Real> reps;
  bool bad = false;
  // the walk of receiver r (level l, face f) over its rows shifted back, with
  // sources at level ls; find(level, lx2, lx3) looks up the opposite-face block among r's
  // candidates (or, on the sending side, is the sending block itself); keep(sg) selects the
  // pieces to record
  auto walk = [&](int rg, int l, std::int32_t lx2, std::int32_t lx3, int f, int ls, int ridx,
                  const auto &find, const auto &keep, Real &eps_out) {
    const int fo = 1 - f;
    const int rb = (ls == l) ? nx2 : nx2/2;
    const long long ny = static_cast<long long>(pm->nmb_rootx2 << (l - root))*nx2;
    const Real q = ((f == 0) ? yshear : -yshear)/(ly/static_cast<Real>(ny));
    const Real fl = std::floor(q);
    eps_out = q - fl;
    long long joff = static_cast<long long>(fl) % ny;
    if (joff < 0) joff += ny;
    long long pos = static_cast<long long>(lx2)*nx2 - joff - 3;
    int dst = 0, npc = 0;
    while (dst < nbuf) {
      const long long gr = ((pos % ny) + ny) % ny;
      const int by = static_cast<int>(gr/rb);
      const int r0 = static_cast<int>(gr - static_cast<long long>(by)*rb);
      const int len = std::min(rb - r0, nbuf - dst);
      const int nzh = (ls == l) ? 1 : 2;
      for (int zh=0; zh<nzh; ++zh) {
        const ShearCand *sc = find(ls, by, (ls == l) ? lx3 : 2*lx3 + zh);
        if (sc == nullptr) {
          if (ridx >= 0) bad = true;   // the receiving side must find every source
          npc++;
          continue;
        }
        if (keep(sc->gid)) {
          const int nkp = (ls == l) ? nk : nk/2;
          std::array<int, kNField> p{};
          p[kRGid] = rg; p[kSGid] = sc->gid; p[kSFace] = fo; p[kR0] = r0; p[kLen] = len;
          p[kDst] = dst; p[kKofs] = (ls == l) ? 0 : zh*(nk/2); p[kNk] = nkp;
          p[kRestr] = (ls != l) ? 1 : 0; p[kQ] = npc; p[kRIdx] = ridx;
          p[kSLoc] = (sc->rank == my_rank) ? 1 : 0;
          p[kRRank] = pm->rank_eachmb[rg]; p[kSRank] = sc->rank;
          p[kOff] = -1;
          pcs.push_back(p);
        }
        npc++;
      }
      pos += len;
      dst += len;
    }
  };
  // receivers of this rank: the old rule (the opposite level at the block's own position;
  // the finer side, equal levels: the outer face, is authoritative)
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[gids + m];
    const int l = ll.level;
    for (int f=0; f<2; ++f) {
      if (ll.lx1 != ((f == 0) ? 0 : nxl(l) - 1)) continue;
      const auto &cv = cand_[f][m];
      const auto &cs = candsort_[f][m];
      auto find = [&cv, &cs](int lev, std::int32_t j, std::int32_t k) -> const ShearCand* {
        auto it = std::lower_bound(cs.begin(), cs.end(), 0, [&](int a, int) {
          const ShearCand &c = cv[a];
          return std::make_tuple(c.level, c.lx3, c.lx2) < std::make_tuple(lev, k, j);});
        if (it == cs.end()) return nullptr;
        const ShearCand &c = cv[*it];
        return (c.level == lev && c.lx3 == k && c.lx2 == j) ? &c : nullptr;
      };
      int lo = -1;
      if (find(l, ll.lx2, ll.lx3) != nullptr) {
        lo = l;
      } else if (l > root && find(l-1, ll.lx2 >> 1, ll.lx3 >> 1) != nullptr) {
        lo = l - 1;
      } else if (find(l+1, 2*ll.lx2, 2*ll.lx3) != nullptr) {
        lo = l + 1;
      }
      if (lo < 0) {bad = true; continue;}
      if (!((lo > l) || (lo == l && f == 0))) continue;   // authoritative side: sends
      Real eps;
      const int ridx = static_cast<int>(rl.size());
      walk(gids + m, l, ll.lx2, ll.lx3, f, lo, ridx, find, [](int) {return true;}, eps);
      rl.emplace_back(m, f);
      reps.push_back(eps);
    }
  }
  // senders of this rank for receivers on other ranks: a receiving block r on the opposite
  // face whose x3 range this block's overlaps; its sources are at this block's level when
  // this block is the finer side there, or the outer face at equal levels
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[gids + m];
    const ShearCand self{gids + m, my_rank, ll.level, ll.lx2, ll.lx3};
    for (int f=0; f<2; ++f) {
      if (ll.lx1 != ((f == 0) ? 0 : nxl(ll.level) - 1)) continue;
      for (const ShearCand &rc : cand_[f][m]) {
        if (rc.rank == my_rank) continue;
        const int l = rc.level, fr = 1 - f;
        int lo = -1;
        if (ll.level == l && ll.lx3 == rc.lx3) lo = l;
        if (ll.level == l + 1 && (ll.lx3 >> 1) == rc.lx3) lo = l + 1;
        if (lo < 0) continue;   // not overlapping, or this block is the coarser side
        if (!((lo > l) || (lo == l && fr == 0))) continue;
        auto find = [&self](int lev, std::int32_t j, std::int32_t k) -> const ShearCand* {
          return (lev == self.level && j == self.lx2 && k == self.lx3) ? &self : nullptr;
        };
        Real eps;
        walk(rc.gid, l, rc.lx2, rc.lx3, fr, lo, -1, find,
             [&self](int g) {return g == self.gid;}, eps);
      }
    }
  }
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "shear flux matching: unexpected face levels" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // one message per peer rank and direction
  std::vector<std::array<long long, 8>> key;
  std::vector<int> rem;
  for (int n=0; n<static_cast<int>(pcs.size()); ++n) {
    const auto &p = pcs[n];
    const bool rloc = (p[kRIdx] >= 0), sloc = (p[kSLoc] == 1);
    if (rloc == sloc) continue;
    key.push_back({rloc ? 0LL : 1LL, rloc ? p[kSRank] : p[kRRank], p[kRGid], p[kSGid], p[kQ],
                   0, 0, static_cast<long long>(nvar_)*p[kNk]*p[kLen]});
    rem.push_back(n);
  }
  std::vector<int> offset;
  int soff = 0, roff = 0;
#if MPI_PARALLEL_ENABLED
  LayoutPeerMessages(key, offset, fmsg_rank, fmsg_off, fmsg_size, fmsg_send, soff, roff);
#else
  std::vector<int> mr, mo, ms, md;
  LayoutPeerMessages(key, offset, mr, mo, ms, md, soff, roff);
#endif
  for (int i=0; i<static_cast<int>(rem.size()); ++i) pcs[rem[i]][kOff] = offset[i];
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
  const int nmsg = static_cast<int>(fmsg_rank.size());
  if (nfreq_ < nmsg) {
    delete [] fsreq_; delete [] frreq_;
    nfreq_ = std::max(16, 2*nmsg);
    fsreq_ = new MPI_Request[nfreq_];
    frreq_ = new MPI_Request[nfreq_];
  }
  for (int n=0; n<nfreq_; ++n) {fsreq_[n] = MPI_REQUEST_NULL; frreq_[n] = MPI_REQUEST_NULL;}
  // one message per piece here (not per peer): on two GB200 nodes the merged
  // messages made the band case 6% slower (large messages between nodes); tag (receiver lid,
  // 40 + its piece number in the receiver's walk), as before S2d
  fmsg_rank.clear(); fmsg_off.clear(); fmsg_size.clear(); fmsg_send.clear(); fmsg_tag.clear();
  for (int n=0; n<nfpiece; ++n) {
    const auto &p = pcs[n];
    const bool rloc = (p[kRIdx] >= 0), sloc = (p[kSLoc] == 1);
    if (rloc == sloc) continue;
    fmsg_rank.push_back(rloc ? p[kSRank] : p[kRRank]);
    fmsg_off.push_back(p[kOff]);
    fmsg_size.push_back(nvar_*p[kNk]*p[kLen]);
    fmsg_send.push_back(rloc ? 0 : 1);
    const int lid = p[kRGid] - pm->gids_eachrank[p[kRRank]];
    fmsg_tag.push_back(CreateBvals_MPI_Tag(lid, 40 + p[kQ]));
    if (p[kQ] > 22) bad = true;
  }
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "shear flux matching: too many pieces for the MPI tags" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const int nmsg1 = static_cast<int>(fmsg_rank.size());
  if (nfreq_ < nmsg1) {
    delete [] fsreq_; delete [] frreq_;
    nfreq_ = std::max(16, 2*nmsg1);
    fsreq_ = new MPI_Request[nfreq_];
    frreq_ = new MPI_Request[nfreq_];
    for (int r=0; r<nfreq_; ++r) {fsreq_[r] = MPI_REQUEST_NULL; frreq_[r] = MPI_REQUEST_NULL;}
  }
  bool no_errors = true;
  for (int n=0; n<nmsg1; ++n) {
    if (fmsg_send[n] == 1) continue;
    if (MPI_Irecv(frecvflat.data() + fmsg_off[n], fmsg_size[n], MPI_ATHENA_REAL, fmsg_rank[n],
                  fmsg_tag[n], comm_sbox, &(frreq_[n])) != MPI_SUCCESS) {
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
  for (size_t n=0; n<fmsg_rank.size(); ++n) {   // one message per peer
    if (fmsg_send[n] == 0) continue;
    if (MPI_Isend(fsendflat.data() + fmsg_off[n], fmsg_size[n], MPI_ATHENA_REAL, fmsg_rank[n],
                  fmsg_tag[n], comm_sbox, &(fsreq_[n])) != MPI_SUCCESS) {
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
  for (size_t n=0; n<fmsg_rank.size(); ++n) {
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
  const int gids = pmy_pack->gids;
  const int nmb = pmy_pack->nmb_thispack;
  const Real ly = pm->mesh_size.x2max - pm->mesh_size.x2min;
  auto nxl = [pm, root](int l) {return pm->nmb_rootx1 << (l - root);};
  std::vector<std::array<int, kQNField>> pcs;
  std::vector<Real> fr;
  bool bad = false;
  // the pieces of (receiving block y on face fy, authoritative block x), the same
  // function on both sides; ridx >= 0: y is local
  auto pieces = [&](const ShearCand &y, int fy, const ShearCand &x, int ridx) {
    const int d = x.level - y.level;
    if (d < 0 || (d == 0 && fy == 1)) return;   // y is authoritative there
    const long long f2 = 1LL << d;
    if ((nx2 % f2) != 0 || (nx3 % f2) != 0) {bad = true; return;}
    const long long nyx = static_cast<long long>(pm->nmb_rootx2)*nx2 << (x.level - root);
    const long long nzx = static_cast<long long>(pm->nmb_rootx3)*nx3 << (x.level - root);
    // x3: x's k row kx sits at kx + bz in y's frame (x cells); y spans [0, nx3 f2)
    const long long zy0 = static_cast<long long>(y.lx3)*nx3*f2;
    const long long zx0 = static_cast<long long>(x.lx3)*nx3;
    // x2: shift of x's rows in y's frame, in x cells
    const Real sig = yshear/(ly/static_cast<Real>(nyx));
    const Real del = (fy == 0) ? sig : -sig;
    const Real dfl = std::floor(del);
    const Real dfr = del - dfl;
    long long dint = static_cast<long long>(dfl) % nyx;
    if (dint < 0) dint += nyx;
    const long long yy0 = static_cast<long long>(y.lx2)*nx2*f2;
    const long long yx0 = static_cast<long long>(x.lx2)*nx2 + dint;
    const long long spany = static_cast<long long>(nx2)*f2;
    const long long spanz = static_cast<long long>(nx3)*f2;
    int seq = 0;
    for (long long kz=(zper ? -2 : 0); kz<=(zper ? 2 : 0); ++kz) {   // (S2c) images only in a
                                                                      // periodic x3
      const long long bz = zx0 + kz*nzx - zy0;
      if (!(bz < spanz && bz + nx3 > 0)) continue;
      const int k0 = static_cast<int>(std::max(0LL, bz/f2));
      const int k1 = static_cast<int>(std::min(static_cast<long long>(nx3),
                                               (bz + nx3 + f2 - 1)/f2));
      if (k1 <= k0) continue;
      for (long long ky=-2; ky<=2; ++ky) {
        const long long by = yx0 + ky*nyx - yy0;    // x row 0 at [by + dfr, by + dfr + 1)
        const Real lo = static_cast<Real>(by) + dfr, hi = lo + nx2;
        if (!(lo < static_cast<Real>(spany) && hi > 0.0)) continue;
        const int r0 = std::max(0, static_cast<int>(std::floor(lo/static_cast<Real>(f2))));
        const int r1 = std::min(nx2, static_cast<int>(std::ceil(hi/static_cast<Real>(f2))));
        if (r1 <= r0) continue;
        std::array<int, kQNField> p{};
        p[kQRGid] = y.gid; p[kQRIdx] = (ridx >= 0) ? 1 : -1; p[kQRFace] = fy;
        p[kQSGid] = x.gid; p[kQSLoc] = (x.rank == my_rank) ? 1 : 0; p[kQSFace] = 1 - fy;
        p[kQD] = d;
        p[kQR0] = r0; p[kQNy] = r1 - r0; p[kQK0] = k0; p[kQNz] = k1 - k0;
        p[kQBase] = static_cast<int>(by); p[kQBaseZ] = static_cast<int>(bz);
        p[kQOff] = -1; p[kQPair] = seq++;
        pcs.push_back(p);
        fr.push_back(dfr);
      }
    }
  };
  // receiving blocks of this rank (in gid order, then face), every authoritative candidate
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[gids + m];
    const ShearCand self{gids + m, my_rank, ll.level, ll.lx2, ll.lx3};
    for (int f=0; f<2; ++f) {
      if (ll.lx1 != ((f == 0) ? 0 : nxl(ll.level) - 1)) continue;
      for (const ShearCand &x : cand_[f][m]) pieces(self, f, x, m);
    }
  }
  // authoritative blocks of this rank for receiving blocks on other ranks
  for (int m=0; m<nmb; ++m) {
    const auto &ll = pm->lloc_eachmb[gids + m];
    const ShearCand self{gids + m, my_rank, ll.level, ll.lx2, ll.lx3};
    for (int f=0; f<2; ++f) {
      if (ll.lx1 != ((f == 0) ? 0 : nxl(ll.level) - 1)) continue;
      for (const ShearCand &y : cand_[f][m]) {
        if (y.rank == my_rank) continue;
        pieces(y, 1 - f, self, -1);
      }
    }
  }
  if (bad) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "shear flux matching (face levels): level difference too large for the "
              << "block size" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // one message per peer rank and direction
  std::vector<std::array<long long, 8>> key;
  std::vector<int> rem;
  for (int n=0; n<static_cast<int>(pcs.size()); ++n) {
    const auto &p = pcs[n];
    const bool rloc = (p[kQRIdx] >= 0), sloc = (p[kQSLoc] == 1);
    if (rloc == sloc) continue;
    const int peer = rloc ? pm->rank_eachmb[p[kQSGid]] : pm->rank_eachmb[p[kQRGid]];
    key.push_back({rloc ? 0LL : 1LL, peer, p[kQRGid], p[kQRFace], p[kQSGid], p[kQPair], 0,
                   static_cast<long long>(nvar_ + 1)*p[kQNz]*p[kQNy]});
    rem.push_back(n);
  }
  std::vector<int> offset;
  int soff = 0, roff = 0;
#if MPI_PARALLEL_ENABLED
  LayoutPeerMessages(key, offset, qmsg_rank, qmsg_off, qmsg_size, qmsg_send, soff, roff);
#else
  std::vector<int> mr, mo, ms, md;
  LayoutPeerMessages(key, offset, mr, mo, ms, md, soff, roff);
#endif
  for (int i=0; i<static_cast<int>(rem.size()); ++i) pcs[rem[i]][kQOff] = offset[i];
  nqpiece = static_cast<int>(pcs.size());
  if (static_cast<int>(qpiece.extent(0)) < std::max(1, nqpiece)) {
    Kokkos::realloc(qpiece, std::max(1, nqpiece), static_cast<int>(kQNField));
    Kokkos::realloc(qfrac, std::max(1, nqpiece));
  }
  for (int n=0; n<nqpiece; ++n) {
    for (int c=0; c<kQNField; ++c) qpiece.h_view(n,c) = pcs[n][c];
    qfrac.h_view(n) = fr[n];
  }
  qpiece.template modify<HostMemSpace>();
  qpiece.template sync<DevExeSpace>();
  qfrac.template modify<HostMemSpace>();
  qfrac.template sync<DevExeSpace>();
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
  qmsg_tag.assign(nmsg, 0);
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
