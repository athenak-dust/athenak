//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_halo.cpp
//! \brief The dust halo: the leaf fields of whole
//! MeshBlocks are exchanged with the ranks that own the neighbours: forward, the owner sends
//! the values of its blocks (gas velocity and its differences, computed by the owner) to the
//! halo slots of the ranks next to them; reverse, a rank sends what it deposited into its halo
//! slots back to the owner, which adds it.  One aggregated message per peer rank and
//! direction; the lists are built with the block slots (BuildLeafTraversal) and are the same
//! on both sides of every pair (a block of this rank goes to rank r when it has a neighbour on
//! r; rank r receives exactly its neighbours on this rank).  The reverse sums are added peer
//! by peer in rank order, without atomics, so the result does not depend on message timing.
//! One rank: no peers, every exchange returns at once.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "dust.hpp"

namespace dust {

namespace {

// accessors of the three layouts of the leaf fields: (block, component, k, j, i),
// (block, k, j, i, component) and (component, flat leaf index)
struct Acc5D {
  DvceArray5D<Real> f;
  int c0 = 0, ks = 0;
  KOKKOS_INLINE_FUNCTION Real get(const int c, const int m, const int j, const int i) const {
    return f(m, c0 + c, ks, j, i);
  }
  KOKKOS_INLINE_FUNCTION void set(const int c, const int m, const int j, const int i,
                                  const Real v) const {f(m, c0 + c, ks, j, i) = v;}
  KOKKOS_INLINE_FUNCTION void add(const int c, const int m, const int j, const int i,
                                  const Real v) const {f(m, c0 + c, ks, j, i) += v;}
};
struct AccPack {
  DvceArray5D<Real> f;
  int ks = 0;
  KOKKOS_INLINE_FUNCTION Real get(const int c, const int m, const int j, const int i) const {
    return f(m, ks, j, i, c);
  }
  KOKKOS_INLINE_FUNCTION void set(const int c, const int m, const int j, const int i,
                                  const Real v) const {f(m, ks, j, i, c) = v;}
  KOKKOS_INLINE_FUNCTION void add(const int c, const int m, const int j, const int i,
                                  const Real v) const {f(m, ks, j, i, c) += v;}
};
struct AccFlat {
  DvceArray2D<Real> f;
  int ncells1 = 0, ncells2 = 0;
  KOKKOS_INLINE_FUNCTION int T(const int m, const int j, const int i) const {
    return (m*ncells2 + j)*ncells1 + i;
  }
  KOKKOS_INLINE_FUNCTION Real get(const int c, const int m, const int j, const int i) const {
    return f(c, T(m, j, i));
  }
  KOKKOS_INLINE_FUNCTION void set(const int c, const int m, const int j, const int i,
                                  const Real v) const {f(c, T(m, j, i)) = v;}
  KOKKOS_INLINE_FUNCTION void add(const int c, const int m, const int j, const int i,
                                  const Real v) const {f(c, T(m, j, i)) += v;}
};
struct AccFlatInt {
  DvceArray1D<int> f;
  int ncells1 = 0, ncells2 = 0;
  KOKKOS_INLINE_FUNCTION int T(const int m, const int j, const int i) const {
    return (m*ncells2 + j)*ncells1 + i;
  }
  KOKKOS_INLINE_FUNCTION Real get(const int, const int m, const int j, const int i) const {
    return static_cast<Real>(f(T(m, j, i)));
  }
  KOKKOS_INLINE_FUNCTION void set(const int, const int m, const int j, const int i,
                                  const Real v) const {f(T(m, j, i)) = static_cast<int>(v);}
  KOKKOS_INLINE_FUNCTION void add(const int, const int m, const int j, const int i,
                                  const Real v) const {f(T(m, j, i)) += static_cast<int>(v);}
};

// copy (forward: owner -> halo slots) or add back (reverse: halo slots -> owner) nc components
// of the interior cells of whole blocks
template <typename Acc>
void Exchange(DustHalo &h, const int nc, const bool reverse, const Acc &acc) {
#if MPI_PARALLEL_ENABLED
  if (h.npeer == 0) return;
  const int nx1 = h.nx1, nx2 = h.nx2, is = h.is, js = h.js;
  const int bs = nc*nx1*nx2;                         // values per block
  // forward: send the owner list, receive into the halo slots; reverse: the other way
  const std::vector<int> &out_off = reverse ? h.roff : h.soff;
  const std::vector<int> &in_off = reverse ? h.soff : h.roff;
  const DvceArray1D<int> out_blk = reverse ? h.rblk : h.sblk;
  const DvceArray1D<int> in_blk = reverse ? h.sblk : h.rblk;
  const int nout = out_off[h.npeer], nin = in_off[h.npeer];
  if (h.sbuf.extent_int(0) < nout*bs) Kokkos::realloc(h.sbuf, nout*bs);
  if (h.rbuf.extent_int(0) < nin*bs) Kokkos::realloc(h.rbuf, nin*bs);
  auto sbuf = h.sbuf;
  auto rbuf = h.rbuf;
  const int tag = reverse ? 2 : 1;
  std::vector<MPI_Request> rreq(h.npeer, MPI_REQUEST_NULL), sreq(h.npeer, MPI_REQUEST_NULL);
  for (int p=0; p<h.npeer; ++p) {
    const int n = (in_off[p + 1] - in_off[p])*bs;
    if (n == 0) continue;
    MPI_Irecv(rbuf.data() + static_cast<std::size_t>(in_off[p])*bs, n, MPI_ATHENA_REAL, h.peer[p],
              tag, h.comm, &rreq[p]);
  }
  if (nout > 0) {
    Kokkos::parallel_for("dust_halo_pack", Kokkos::RangePolicy<>(DevExeSpace(), 0, nout*bs),
    KOKKOS_LAMBDA(const int q) {
      const int i = q % nx1;
      const int j = (q/nx1) % nx2;
      const int c = (q/(nx1*nx2)) % nc;
      const int e = q/bs;
      sbuf(q) = acc.get(c, out_blk(e), js + j, is + i);
    });
    Kokkos::fence();
  }
  for (int p=0; p<h.npeer; ++p) {
    const int n = (out_off[p + 1] - out_off[p])*bs;
    if (n == 0) continue;
    MPI_Isend(sbuf.data() + static_cast<std::size_t>(out_off[p])*bs, n, MPI_ATHENA_REAL, h.peer[p],
              tag, h.comm, &sreq[p]);
  }
  MPI_Waitall(h.npeer, rreq.data(), MPI_STATUSES_IGNORE);
  if (!reverse) {
    if (nin > 0) {
      Kokkos::parallel_for("dust_halo_unpack", Kokkos::RangePolicy<>(DevExeSpace(), 0, nin*bs),
      KOKKOS_LAMBDA(const int q) {
        const int i = q % nx1;
        const int j = (q/nx1) % nx2;
        const int c = (q/(nx1*nx2)) % nc;
        const int e = q/bs;
        acc.set(c, in_blk(e), js + j, is + i, rbuf(q));
      });
    }
  } else {
    // peer by peer in rank order: a block of this rank may be in the halo of several ranks
    for (int p=0; p<h.npeer; ++p) {
      const int q0 = in_off[p]*bs, q1 = in_off[p + 1]*bs;
      if (q1 == q0) continue;
      Kokkos::parallel_for("dust_halo_add", Kokkos::RangePolicy<>(DevExeSpace(), q0, q1),
      KOKKOS_LAMBDA(const int q) {
        const int i = q % nx1;
        const int j = (q/nx1) % nx2;
        const int c = (q/(nx1*nx2)) % nc;
        const int e = q/bs;
        acc.add(c, in_blk(e), js + j, is + i, rbuf(q));
      });
    }
  }
  Kokkos::fence();
  MPI_Waitall(h.npeer, sreq.data(), MPI_STATUSES_IGNORE);
#else
  (void)h; (void)nc; (void)reverse; (void)acc;
#endif
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildHalo
//! \brief The exchange lists of the halo from the block slots of BuildLeafTraversal: per
//! peer rank (ascending), the blocks of this rank that have a neighbour on it (ascending gid)
//! and the halo slots of its blocks (ascending gid).  The counts are checked against the
//! peer's.

void DustGasDrag::BuildHalo(const std::vector<int> &slot_gid) {
  Mesh *pm = pmy_pack->pmesh;
  auto &h = halo_;
  const auto &lt = leaf_traversal;
  h.nx1 = lt.nx1; h.nx2 = lt.nx2; h.is = lt.is; h.js = lt.js;
  h.peer.clear(); h.soff.assign(1, 0); h.roff.assign(1, 0); h.npeer = 0;
  if (global_variable::nranks == 1) return;
#if MPI_PARALLEL_ENABLED
  if (!h.comm_set) {
    MPI_Comm_dup(MPI_COMM_WORLD, &h.comm);
    h.comm_set = true;
  }
  const int nmb = lt.nmb, nfld = lt.nfld, gids = lt.gids;
  const int myrank = global_variable::my_rank;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const int nslot = pmy_pack->pmb->nnghbr;
  // send: (peer, local block) pairs; receive: (owner, halo slot)
  std::vector<std::pair<int, int>> snd, rcv;
  for (int m=0; m<nmb; ++m) {
    std::vector<int> to;
    for (int n=0; n<nslot; ++n) {
      const int g = nghbr.h_view(m, n).gid;
      if (g < 0) continue;
      const int r = pm->rank_eachmb[g];
      if (r != myrank) to.push_back(r);
    }
    std::sort(to.begin(), to.end());
    to.erase(std::unique(to.begin(), to.end()), to.end());
    for (const int r : to) snd.push_back({r, gids + m});
  }
  for (int b=nmb; b<nfld; ++b) rcv.push_back({pm->rank_eachmb[slot_gid[b]], slot_gid[b]});
  std::sort(snd.begin(), snd.end());
  std::sort(rcv.begin(), rcv.end());
  for (const auto &e : snd) h.peer.push_back(e.first);
  for (const auto &e : rcv) h.peer.push_back(e.first);
  std::sort(h.peer.begin(), h.peer.end());
  h.peer.erase(std::unique(h.peer.begin(), h.peer.end()), h.peer.end());
  h.npeer = static_cast<int>(h.peer.size());
  std::vector<int> sh, rh;
  h.soff.assign(h.npeer + 1, 0); h.roff.assign(h.npeer + 1, 0);
  {
    std::size_t a = 0, c = 0;
    for (int p=0; p<h.npeer; ++p) {
      while (a < snd.size() && snd[a].first == h.peer[p]) {sh.push_back(snd[a].second - gids); ++a;}
      while (c < rcv.size() && rcv[c].first == h.peer[p]) {
        rh.push_back(nmb + static_cast<int>(std::lower_bound(slot_gid.begin() + nmb,
                     slot_gid.begin() + nfld, rcv[c].second) - (slot_gid.begin() + nmb)));
        ++c;
      }
      h.soff[p + 1] = static_cast<int>(sh.size());
      h.roff[p + 1] = static_cast<int>(rh.size());
    }
  }
  for (std::size_t k=0; k<rh.size(); ++k) {
    if (slot_gid[rh[k]] != rcv[k].second) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "dust halo: halo slots are not in ascending gid order" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  h.sblk = DvceArray1D<int>("dust_halo_sblk", std::max<std::size_t>(sh.size(), 1));
  h.rblk = DvceArray1D<int>("dust_halo_rblk", std::max<std::size_t>(rh.size(), 1));
  auto sb = Kokkos::create_mirror_view(h.sblk);
  auto rb = Kokkos::create_mirror_view(h.rblk);
  for (std::size_t k=0; k<sh.size(); ++k) sb(k) = sh[k];
  for (std::size_t k=0; k<rh.size(); ++k) rb(k) = rh[k];
  Kokkos::deep_copy(h.sblk, sb);
  Kokkos::deep_copy(h.rblk, rb);
  // the lists of the two sides of every pair must agree (gids, in order)
  {
    std::vector<MPI_Request> rq(2*h.npeer);
    std::vector<int> nrecv(h.npeer), nsend(h.npeer);
    for (int p=0; p<h.npeer; ++p) {
      nsend[p] = h.soff[p + 1] - h.soff[p];
      MPI_Irecv(&nrecv[p], 1, MPI_INT, h.peer[p], 3, h.comm, &rq[2*p]);
      MPI_Isend(&nsend[p], 1, MPI_INT, h.peer[p], 3, h.comm, &rq[2*p + 1]);
    }
    MPI_Waitall(2*h.npeer, rq.data(), MPI_STATUSES_IGNORE);
    std::vector<std::vector<int>> gin(h.npeer), gout(h.npeer);
    for (int p=0; p<h.npeer; ++p) {
      if (nrecv[p] != h.roff[p + 1] - h.roff[p]) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "dust halo: rank " << myrank << " expects " << h.roff[p + 1] - h.roff[p]
                  << " blocks from rank " << h.peer[p] << ", which sends " << nrecv[p] << std::endl;
        std::exit(EXIT_FAILURE);
      }
      gin[p].resize(std::max(nrecv[p], 1));
      gout[p].resize(std::max(nsend[p], 1));
      for (int k=0; k<nsend[p]; ++k) gout[p][k] = gids + sh[h.soff[p] + k];
    }
    for (int p=0; p<h.npeer; ++p) {
      MPI_Irecv(gin[p].data(), nrecv[p], MPI_INT, h.peer[p], 4, h.comm, &rq[2*p]);
      MPI_Isend(gout[p].data(), nsend[p], MPI_INT, h.peer[p], 4, h.comm, &rq[2*p + 1]);
    }
    MPI_Waitall(2*h.npeer, rq.data(), MPI_STATUSES_IGNORE);
    for (int p=0; p<h.npeer; ++p) {
      for (int k=0; k<nrecv[p]; ++k) {
        if (gin[p][k] != slot_gid[rh[h.roff[p] + k]]) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                    << "dust halo: block lists of ranks " << myrank << " and " << h.peer[p]
                    << " disagree" << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }
  ++h.builds;
#else
  (void)pm; (void)slot_gid;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::HaloForwardPack / HaloReverseAdd / HaloForwardFlat / HaloReverseAddFlat
//! \brief The exchanges of the leaf fields: the packed gather field gpack (9 components,
//! forward), the deposits qdep / dmom (components c0 .. c0 + nc - 1, reverse), the merge
//! budget fields (c, leaf) and the per-leaf flags (forward and reverse).

void DustGasDrag::HaloForwardPack(DvceArray5D<Real> &gp) {
  if (halo_.npeer == 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  AccPack a; a.f = gp; a.ks = pmy_pack->pmesh->mb_indcs.ks;
  Exchange(halo_, gp.extent_int(4), false, a);
  mpi_time[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++mpi_count[2];
}

void DustGasDrag::HaloReverseAdd(DvceArray5D<Real> &f, const int c0, const int nc) {
  if (halo_.npeer == 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  Acc5D a; a.f = f; a.c0 = c0; a.ks = pmy_pack->pmesh->mb_indcs.ks;
  Exchange(halo_, nc, true, a);
  mpi_time[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++mpi_count[3];
}

void DustGasDrag::HaloForwardFlat(DvceArray2D<Real> &f, const int nc) {
  if (halo_.npeer == 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  AccFlat a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, nc, false, a);
  mpi_time[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++mpi_count[2];
}

void DustGasDrag::HaloReverseAddFlat(DvceArray2D<Real> &f, const int nc) {
  if (halo_.npeer == 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  AccFlat a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, nc, true, a);
  mpi_time[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++mpi_count[3];
}

void DustGasDrag::HaloForwardFlags(DvceArray1D<int> &f) {
  if (halo_.npeer == 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  AccFlatInt a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, 1, false, a);
  mpi_time[2] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++mpi_count[2];
}

void DustGasDrag::HaloReverseAddFlags(DvceArray1D<int> &f) {
  if (halo_.npeer == 0) return;
  const auto t0 = std::chrono::steady_clock::now();
  AccFlatInt a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, 1, true, a);
  mpi_time[3] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  ++mpi_count[3];
}

}  // namespace dust
