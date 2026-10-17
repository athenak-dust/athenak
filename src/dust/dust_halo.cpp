//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_halo.cpp
//! \brief The dust halo: the leaf fields of whole
//! MeshBlocks are exchanged with the ranks that own the blocks a cloud can reach (the halo
//! blocks of BuildLeafTraversal: every block of another rank within kHaloRootCells root cells):
//! forward, the owner sends the values of its blocks (gas velocity and its differences,
//! computed by the owner) to the halo slots of the ranks that hold them; reverse, a rank sends
//! what it deposited into its halo slots back to the owner, which adds it.  One aggregated
//! message per peer rank and direction; the lists come from the requests of the receiving
//! ranks (BuildHalo).  The reverse sums are added peer by peer in rank order, without atomics,
//! so they do not depend on the arrival order of the messages.  One rank: no peers, every
//! exchange returns at once.

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
  using Clock = std::chrono::steady_clock;
  auto since = [](const Clock::time_point &a) {
    return std::chrono::duration<double>(Clock::now() - a).count();
  };
  double *ts = h.tsec + (reverse ? 4 : 0);
  ++h.nexch[reverse ? 1 : 0];
  Clock::time_point tc = Clock::now();
  Kokkos::fence();                     // the kernels before the exchange (their time, not MPI)
  ts[0] += since(tc);
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
  tc = Clock::now();
  for (int p=0; p<h.npeer; ++p) {
    const int n = (in_off[p + 1] - in_off[p])*bs;
    if (n == 0) continue;
    MPI_Irecv(rbuf.data() + static_cast<std::size_t>(in_off[p])*bs, n, MPI_ATHENA_REAL, h.peer[p],
              tag, h.comm, &rreq[p]);
  }
  ts[2] += since(tc);
  tc = Clock::now();
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
  ts[1] += since(tc);
  tc = Clock::now();
  for (int p=0; p<h.npeer; ++p) {
    const int n = (out_off[p + 1] - out_off[p])*bs;
    if (n == 0) continue;
    MPI_Isend(sbuf.data() + static_cast<std::size_t>(out_off[p])*bs, n, MPI_ATHENA_REAL, h.peer[p],
              tag, h.comm, &sreq[p]);
  }
  MPI_Waitall(h.npeer, rreq.data(), MPI_STATUSES_IGNORE);
  ts[2] += since(tc);
  tc = Clock::now();
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
  ts[3] += since(tc);
  tc = Clock::now();
  MPI_Waitall(h.npeer, sreq.data(), MPI_STATUSES_IGNORE);
  ts[2] += since(tc);
#else
  (void)h; (void)nc; (void)reverse; (void)acc;
#endif
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildHalo
//! \brief The exchange lists of the halo from the block slots of BuildLeafTraversal.  Every
//! rank asks the owners of its halo slots for those blocks (ascending gid; one all-to-all of
//! the counts and one of the gid lists), so that both sides of every pair hold the same lists:
//! per peer rank (ascending), the blocks of this rank asked for by it and the halo slots it
//! fills.  (The halo may hold blocks beyond the neighbours, so it is not symmetric in general.)

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
  const int myrank = global_variable::my_rank, nranks = global_variable::nranks;
  // what this rank needs, by owner (halo slots are in ascending gid, so is each list)
  std::vector<std::vector<int>> want(nranks), wslot(nranks);
  for (int b=nmb; b<nfld; ++b) {
    const int r = pm->rank_eachmb[slot_gid[b]];
    if (r == myrank) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "dust halo: halo block " << slot_gid[b] << " is on this rank" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    want[r].push_back(slot_gid[b]);
    wslot[r].push_back(b);
  }
  std::vector<int> nwant(nranks), ngive(nranks), sdisp(nranks + 1, 0), rdisp(nranks + 1, 0);
  for (int r=0; r<nranks; ++r) nwant[r] = static_cast<int>(want[r].size());
  MPI_Alltoall(nwant.data(), 1, MPI_INT, ngive.data(), 1, MPI_INT, h.comm);
  for (int r=0; r<nranks; ++r) {sdisp[r + 1] = sdisp[r] + nwant[r]; rdisp[r + 1] = rdisp[r] + ngive[r];}
  std::vector<int> wflat(std::max(sdisp[nranks], 1)), gflat(std::max(rdisp[nranks], 1));
  for (int r=0; r<nranks; ++r) std::copy(want[r].begin(), want[r].end(), wflat.begin() + sdisp[r]);
  MPI_Alltoallv(wflat.data(), nwant.data(), sdisp.data(), MPI_INT, gflat.data(), ngive.data(),
                rdisp.data(), MPI_INT, h.comm);
  std::vector<int> sh, rh;
  for (int r=0; r<nranks; ++r) {
    if (nwant[r] == 0 && ngive[r] == 0) continue;
    h.peer.push_back(r);
    for (int k=rdisp[r]; k<rdisp[r + 1]; ++k) {
      const int g = gflat[k];
      if (g < gids || g >= gids + nmb) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "dust halo: rank " << r << " asks rank " << myrank << " for block " << g
                  << ", which is not on it" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      sh.push_back(g - gids);
    }
    rh.insert(rh.end(), wslot[r].begin(), wslot[r].end());
    h.soff.push_back(static_cast<int>(sh.size()));
    h.roff.push_back(static_cast<int>(rh.size()));
  }
  h.npeer = static_cast<int>(h.peer.size());
  h.sblk = DvceArray1D<int>("dust_halo_sblk", std::max<std::size_t>(sh.size(), 1));
  h.rblk = DvceArray1D<int>("dust_halo_rblk", std::max<std::size_t>(rh.size(), 1));
  auto sb = Kokkos::create_mirror_view(h.sblk);
  auto rb = Kokkos::create_mirror_view(h.rblk);
  for (std::size_t k=0; k<sh.size(); ++k) sb(k) = sh[k];
  for (std::size_t k=0; k<rh.size(); ++k) rb(k) = rh[k];
  Kokkos::deep_copy(h.sblk, sb);
  Kokkos::deep_copy(h.rblk, rb);
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
  AccPack a; a.f = gp; a.ks = pmy_pack->pmesh->mb_indcs.ks;
  Exchange(halo_, gp.extent_int(4), false, a);
}

void DustGasDrag::HaloReverseAdd(DvceArray5D<Real> &f, const int c0, const int nc) {
  if (halo_.npeer == 0) return;
  Acc5D a; a.f = f; a.c0 = c0; a.ks = pmy_pack->pmesh->mb_indcs.ks;
  Exchange(halo_, nc, true, a);
}

void DustGasDrag::HaloForwardFlat(DvceArray2D<Real> &f, const int nc) {
  if (halo_.npeer == 0) return;
  AccFlat a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, nc, false, a);
}

void DustGasDrag::HaloReverseAddFlat(DvceArray2D<Real> &f, const int nc) {
  if (halo_.npeer == 0) return;
  AccFlat a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, nc, true, a);
}

void DustGasDrag::HaloForwardFlags(DvceArray1D<int> &f) {
  if (halo_.npeer == 0) return;
  AccFlatInt a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, 1, false, a);
}

void DustGasDrag::HaloReverseAddFlags(DvceArray1D<int> &f) {
  if (halo_.npeer == 0) return;
  AccFlatInt a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, 1, true, a);
}

}  // namespace dust
