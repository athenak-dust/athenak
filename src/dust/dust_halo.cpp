//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_halo.cpp
//! \brief The dust halo: the leaf fields of whole
//! MeshBlocks are exchanged with the ranks that own the blocks a cloud can reach (the halo
//! blocks of BuildLeafTraversal: every block of another rank within the reach of a block of this
//! rank):
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
#include <limits>
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
// of the cells in the lists (whole blocks, or the strips a cloud can reach).  can_nan: the field
// takes NaN (the strip guard fills the halo slots with it before the forward copy)
template <typename Acc>
void Exchange(DustHalo &h, const int nc, const bool reverse, const Acc &acc, const bool can_nan) {
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
  const int nc1 = h.ncells1, nc2 = h.ncells2, n12 = h.ncells1*h.ncells2;
  // forward: send the owner list, receive into the halo slots; reverse: the other way
  const std::vector<int> &out_off = reverse ? h.roff : h.soff;
  const std::vector<int> &in_off = reverse ? h.soff : h.roff;
  const DvceArray1D<int> out_cell = reverse ? h.rcell : h.scell;
  const DvceArray1D<int> in_cell = reverse ? h.scell : h.rcell;
  const int nout = out_off[h.npeer], nin = in_off[h.npeer];
  if (h.sbuf.extent_int(0) < nout*nc) Kokkos::realloc(h.sbuf, nout*nc);
  if (h.rbuf.extent_int(0) < nin*nc) Kokkos::realloc(h.rbuf, nin*nc);
  auto sbuf = h.sbuf;
  auto rbuf = h.rbuf;
  // the strip guard (test): before the reverse copy, nothing may lie outside the strips of the
  // halo slots; before the forward copy, the halo slots hold NaN where nothing is received
  const int nh = h.hslot1 - h.hslot0, hs0 = h.hslot0, nx1 = h.nx1, nx2 = h.nx2, is = h.is, js = h.js;
  if (h.guard && nh > 0) {
    auto rect = h.rect_d;
    if (reverse) {
      int nbad = 0;
      Kokkos::parallel_reduce("dust_halo_guard", Kokkos::RangePolicy<>(DevExeSpace(), 0, nh*nx2*nx1*nc),
      KOKKOS_LAMBDA(const int q, int &bad) {
        const int i = q % nx1, j = (q/nx1) % nx2, c = (q/(nx1*nx2)) % nc, e = q/(nx1*nx2*nc);
        const bool in = (i >= rect(e,0) && i < rect(e,1) && j >= rect(e,2) && j < rect(e,3));
        if (!in && acc.get(c, hs0 + e, js + j, is + i) != 0.0) ++bad;
      }, nbad);
      if (nbad > 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "dust halo: " << nbad << " deposits outside the strips of the halo blocks"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    } else if (can_nan) {
      const Real qnan = std::numeric_limits<Real>::quiet_NaN();
      Kokkos::parallel_for("dust_halo_nan", Kokkos::RangePolicy<>(DevExeSpace(), 0, nh*nx2*nx1*nc),
      KOKKOS_LAMBDA(const int q) {
        const int i = q % nx1, j = (q/nx1) % nx2, c = (q/(nx1*nx2)) % nc, e = q/(nx1*nx2*nc);
        acc.set(c, hs0 + e, js + j, is + i, qnan);
      });
    }
  }
  const int tag = reverse ? 2 : 1;
  std::vector<MPI_Request> rreq(h.npeer, MPI_REQUEST_NULL), sreq(h.npeer, MPI_REQUEST_NULL);
  tc = Clock::now();
  for (int p=0; p<h.npeer; ++p) {
    const int n = (in_off[p + 1] - in_off[p])*nc;
    if (n == 0) continue;
    MPI_Irecv(rbuf.data() + static_cast<std::size_t>(in_off[p])*nc, n, MPI_ATHENA_REAL, h.peer[p],
              tag, h.comm, &rreq[p]);
  }
  ts[2] += since(tc);
  tc = Clock::now();
  if (nout > 0) {
    Kokkos::parallel_for("dust_halo_pack", Kokkos::RangePolicy<>(DevExeSpace(), 0, nout*nc),
    KOKKOS_LAMBDA(const int q) {
      const int c = q % nc, t = out_cell(q/nc);
      const int m = t/n12, r = t - m*n12;
      sbuf(q) = acc.get(c, m, r/nc1, r % nc1);
    });
    Kokkos::fence();
  }
  ts[1] += since(tc);
  tc = Clock::now();
  for (int p=0; p<h.npeer; ++p) {
    const int n = (out_off[p + 1] - out_off[p])*nc;
    if (n == 0) continue;
    MPI_Isend(sbuf.data() + static_cast<std::size_t>(out_off[p])*nc, n, MPI_ATHENA_REAL, h.peer[p],
              tag, h.comm, &sreq[p]);
  }
  MPI_Waitall(h.npeer, rreq.data(), MPI_STATUSES_IGNORE);
  ts[2] += since(tc);
  tc = Clock::now();
  if (!reverse) {
    if (nin > 0) {
      Kokkos::parallel_for("dust_halo_unpack", Kokkos::RangePolicy<>(DevExeSpace(), 0, nin*nc),
      KOKKOS_LAMBDA(const int q) {
        const int c = q % nc, t = in_cell(q/nc);
        const int m = t/n12, r = t - m*n12;
        acc.set(c, m, r/nc1, r % nc1, rbuf(q));
      });
    }
  } else {
    // peer by peer in rank order: a block of this rank may be in the halo of several ranks
    for (int p=0; p<h.npeer; ++p) {
      const int q0 = in_off[p]*nc, q1 = in_off[p + 1]*nc;
      if (q1 == q0) continue;
      Kokkos::parallel_for("dust_halo_add", Kokkos::RangePolicy<>(DevExeSpace(), q0, q1),
      KOKKOS_LAMBDA(const int q) {
        const int c = q % nc, t = in_cell(q/nc);
        const int m = t/n12, r = t - m*n12;
        acc.add(c, m, r/nc1, r % nc1, rbuf(q));
      });
    }
  }
  Kokkos::fence();
  ts[3] += since(tc);
  tc = Clock::now();
  MPI_Waitall(h.npeer, sreq.data(), MPI_STATUSES_IGNORE);
  ts[2] += since(tc);
#else
  (void)h; (void)nc; (void)reverse; (void)acc; (void)can_nan;
#endif
}

}  // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildHalo
//! \brief The exchange lists of the halo from the block slots of BuildLeafTraversal.  Every
//! rank asks the owners of its halo slots for those blocks (ascending gid), each with the
//! rectangle of its cells that it needs (halo_.rect: the whole block, or a strip of it);
//! one all-to-all of the counts and one of the requests, so that both sides of every pair hold
//! the same lists: per peer rank (ascending), the cells of this rank's blocks asked for by it
//! and the cells of the halo slots it fills, block by block, row by row.  (The halo may hold
//! blocks beyond the neighbours, so it is not symmetric in general.)

void DustGasDrag::BuildHalo(const std::vector<int> &slot_gid) {
  Mesh *pm = pmy_pack->pmesh;
  auto &h = halo_;
  const auto &lt = leaf_traversal;
  h.nx1 = lt.nx1; h.nx2 = lt.nx2; h.is = lt.is; h.js = lt.js;
  h.ncells1 = lt.ncells1; h.ncells2 = lt.ncells2;
  h.hslot0 = lt.nmb; h.hslot1 = lt.nfld;
  h.guard = cloud_halo_strip_guard;
  h.peer.clear(); h.soff.assign(1, 0); h.roff.assign(1, 0); h.npeer = 0;
  h.nblk_send = h.nblk_recv = 0;
  if (global_variable::nranks == 1) return;
#if MPI_PARALLEL_ENABLED
  if (!h.comm_set) {
    MPI_Comm_dup(MPI_COMM_WORLD, &h.comm);
    h.comm_set = true;
  }
  const int nmb = lt.nmb, nfld = lt.nfld, gids = lt.gids;
  const int myrank = global_variable::my_rank, nranks = global_variable::nranks;
  const int nc1 = lt.ncells1, nc2 = lt.ncells2, is = lt.is, js = lt.js;
  auto flat = [&](const int b, const int j, const int i) {return (b*nc2 + js + j)*nc1 + is + i;};
  // what this rank needs, by owner (halo slots are in ascending gid, so is each list): gid and
  // rectangle, five ints per block
  std::vector<std::vector<int>> want(nranks), wslot(nranks);
  for (int b=nmb; b<nfld; ++b) {
    const int r = pm->rank_eachmb[slot_gid[b]];
    if (r == myrank) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "dust halo: halo block " << slot_gid[b] << " is on this rank" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const auto &c = h.rect[b - nmb];
    want[r].insert(want[r].end(), {slot_gid[b], c[0], c[1], c[2], c[3]});
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
    for (int k=rdisp[r]; k<rdisp[r + 1]; k+=5) {
      const int g = gflat[k];
      const int i0 = gflat[k + 1], i1 = gflat[k + 2], j0 = gflat[k + 3], j1 = gflat[k + 4];
      if (g < gids || g >= gids + nmb || i0 < 0 || i1 > lt.nx1 || j0 < 0 || j1 > lt.nx2 ||
          i0 >= i1 || j0 >= j1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "dust halo: rank " << r << " asks rank " << myrank << " for block " << g
                  << " cells [" << i0 << ", " << i1 << ") x [" << j0 << ", " << j1 << ")"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
      for (int j=j0; j<j1; ++j) for (int i=i0; i<i1; ++i) sh.push_back(flat(g - gids, j, i));
      ++h.nblk_send;
    }
    for (const int b : wslot[r]) {
      const auto &c = h.rect[b - nmb];
      for (int j=c[2]; j<c[3]; ++j) for (int i=c[0]; i<c[1]; ++i) rh.push_back(flat(b, j, i));
      ++h.nblk_recv;
    }
    h.soff.push_back(static_cast<int>(sh.size()));
    h.roff.push_back(static_cast<int>(rh.size()));
  }
  h.npeer = static_cast<int>(h.peer.size());
  h.scell = DvceArray1D<int>("dust_halo_scell", std::max<std::size_t>(sh.size(), 1));
  h.rcell = DvceArray1D<int>("dust_halo_rcell", std::max<std::size_t>(rh.size(), 1));
  auto sb = Kokkos::create_mirror_view(h.scell);
  auto rb = Kokkos::create_mirror_view(h.rcell);
  for (std::size_t k=0; k<sh.size(); ++k) sb(k) = sh[k];
  for (std::size_t k=0; k<rh.size(); ++k) rb(k) = rh[k];
  Kokkos::deep_copy(h.scell, sb);
  Kokkos::deep_copy(h.rcell, rb);
  const int nh = std::max(nfld - nmb, 1);
  h.rect_d = DvceArray2D<int>("dust_halo_rect", nh, 4);
  auto rr = Kokkos::create_mirror_view(h.rect_d);
  for (int b=nmb; b<nfld; ++b) for (int q=0; q<4; ++q) rr(b - nmb, q) = h.rect[b - nmb][q];
  Kokkos::deep_copy(h.rect_d, rr);
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
  Exchange(halo_, gp.extent_int(4), false, a, true);
}

void DustGasDrag::HaloReverseAdd(DvceArray5D<Real> &f, const int c0, const int nc) {
  if (halo_.npeer == 0) return;
  Acc5D a; a.f = f; a.c0 = c0; a.ks = pmy_pack->pmesh->mb_indcs.ks;
  Exchange(halo_, nc, true, a, true);
}

void DustGasDrag::HaloForwardFlat(DvceArray2D<Real> &f, const int nc) {
  if (halo_.npeer == 0) return;
  AccFlat a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, nc, false, a, true);
}

void DustGasDrag::HaloReverseAddFlat(DvceArray2D<Real> &f, const int nc) {
  if (halo_.npeer == 0) return;
  AccFlat a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, nc, true, a, true);
}

void DustGasDrag::HaloForwardFlags(DvceArray1D<int> &f) {
  if (halo_.npeer == 0) return;
  AccFlatInt a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, 1, false, a, false);
}

void DustGasDrag::HaloReverseAddFlags(DvceArray1D<int> &f) {
  if (halo_.npeer == 0) return;
  AccFlatInt a; a.f = f; a.ncells1 = leaf_traversal.ncells1; a.ncells2 = leaf_traversal.ncells2;
  Exchange(halo_, 1, true, a, false);
}

}  // namespace dust
