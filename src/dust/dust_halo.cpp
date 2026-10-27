//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_halo.cpp
//! \brief The dust halo for multi-rank runs.  The leaf fields of whole
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

// accessors of the layouts of the leaf fields by flat leaf index t: (block, component, k, j, i),
// (block, k, j, i, component) and (component, t).  2-D: t = (m ncells2 + j) ncells1 + i at the
// single k = ks; 3-D: t = ((m ncells3 + k) ncells2 + j) ncells1 + i
struct Acc5D {
  DvceArray5D<Real> f;
  int c0 = 0, ks = 0, nc1 = 1, n12 = 1;
  KOKKOS_INLINE_FUNCTION Real &at(const int c, const int t) const {
    const int m = t/n12, r = t - m*n12;
    return f(m, c0 + c, ks, r/nc1, r % nc1);
  }
  KOKKOS_INLINE_FUNCTION Real get(const int c, const int t) const {return at(c, t);}
  KOKKOS_INLINE_FUNCTION void set(const int c, const int t, const Real v) const {at(c, t) = v;}
  KOKKOS_INLINE_FUNCTION void add(const int c, const int t, const Real v) const {at(c, t) += v;}
};
struct Acc5D3 {
  DvceArray5D<Real> f;
  int c0 = 0, nc1 = 1, n12 = 1, n123 = 1;
  KOKKOS_INLINE_FUNCTION Real &at(const int c, const int t) const {
    const int m = t/n123, r = t - m*n123, k = r/n12, q = r - k*n12;
    return f(m, c0 + c, k, q/nc1, q % nc1);
  }
  KOKKOS_INLINE_FUNCTION Real get(const int c, const int t) const {return at(c, t);}
  KOKKOS_INLINE_FUNCTION void set(const int c, const int t, const Real v) const {at(c, t) = v;}
  KOKKOS_INLINE_FUNCTION void add(const int c, const int t, const Real v) const {at(c, t) += v;}
};
struct AccPack {
  DvceArray5D<Real> f;
  int ks = 0, nc1 = 1, n12 = 1;
  KOKKOS_INLINE_FUNCTION Real &at(const int c, const int t) const {
    const int m = t/n12, r = t - m*n12;
    return f(m, ks, r/nc1, r % nc1, c);
  }
  KOKKOS_INLINE_FUNCTION Real get(const int c, const int t) const {return at(c, t);}
  KOKKOS_INLINE_FUNCTION void set(const int c, const int t, const Real v) const {at(c, t) = v;}
  KOKKOS_INLINE_FUNCTION void add(const int c, const int t, const Real v) const {at(c, t) += v;}
};
struct AccPack3 {
  DvceArray5D<Real> f;
  int nc1 = 1, n12 = 1, n123 = 1;
  KOKKOS_INLINE_FUNCTION Real &at(const int c, const int t) const {
    const int m = t/n123, r = t - m*n123, k = r/n12, q = r - k*n12;
    return f(m, k, q/nc1, q % nc1, c);
  }
  KOKKOS_INLINE_FUNCTION Real get(const int c, const int t) const {return at(c, t);}
  KOKKOS_INLINE_FUNCTION void set(const int c, const int t, const Real v) const {at(c, t) = v;}
  KOKKOS_INLINE_FUNCTION void add(const int c, const int t, const Real v) const {at(c, t) += v;}
};
struct AccFlat {
  DvceArray2D<Real> f;
  KOKKOS_INLINE_FUNCTION Real get(const int c, const int t) const {return f(c, t);}
  KOKKOS_INLINE_FUNCTION void set(const int c, const int t, const Real v) const {f(c, t) = v;}
  KOKKOS_INLINE_FUNCTION void add(const int c, const int t, const Real v) const {f(c, t) += v;}
};
struct AccFlatInt {
  DvceArray1D<int> f;
  KOKKOS_INLINE_FUNCTION Real get(const int, const int t) const {
    return static_cast<Real>(f(t));
  }
  KOKKOS_INLINE_FUNCTION void set(const int, const int t, const Real v) const {
    f(t) = static_cast<int>(v);
  }
  KOKKOS_INLINE_FUNCTION void add(const int, const int t, const Real v) const {
    f(t) += static_cast<int>(v);
  }
};

// copy (forward: owner -> halo slots) or add back (reverse: halo slots -> owner) nc components
// of the cells in the lists (whole blocks, or with <dust>/cloud_halo_strips their reachable strips)
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
  // the kernels before the exchange (their time, not MPI): a fence for the timing alone (the pack
  // below fences before the sends), so only with <dust>/amr_profile
  if (ProfOn()) {
    Kokkos::fence();
    ts[0] += since(tc);
  }
  const int nc1 = h.ncells1, nc2 = h.ncells2;
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
      sbuf(q) = acc.get(q % nc, out_cell(q/nc));
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
        acc.set(q % nc, in_cell(q/nc), rbuf(q));
      });
    }
  } else {
    // peer by peer in rank order: a block of this rank may be in the halo of several ranks
    for (int p=0; p<h.npeer; ++p) {
      const int q0 = in_off[p]*nc, q1 = in_off[p + 1]*nc;
      if (q1 == q0) continue;
      Kokkos::parallel_for("dust_halo_add", Kokkos::RangePolicy<>(DevExeSpace(), q0, q1),
      KOKKOS_LAMBDA(const int q) {
        acc.add(q % nc, in_cell(q/nc), rbuf(q));
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
//! rank asks the owners of its halo slots for those blocks (ascending gid), each with the
//! rectangle of its cells that it needs (halo_.rect: the whole block, or its reachable strip);
//! one all-to-all of the counts and one of the requests, so that both sides of every pair hold
//! the same lists: per peer rank (ascending), the cells of this rank's blocks asked for by it
//! and the cells of the halo slots it fills, block by block, row by row.  (The halo may hold
//! blocks beyond the neighbours, so it is not symmetric in general.)

void DustGasDrag::BuildHalo(const std::vector<int> &slot_gid) {
  Mesh *pm = pmy_pack->pmesh;
  auto &h = halo_;
  // the slot geometry: the 2-D leaf traversal or the 3-D one
  int nmb = 0, nfld = 0, gids = 0;
  h.three = clouds3d;
  if (h.three) {
    const auto &l3 = leaf_traversal3d;
    h.nx1 = l3.nx1; h.nx2 = l3.nx2; h.nx3 = l3.nx3; h.is = l3.is; h.js = l3.js; h.ks = l3.ks;
    h.ncells1 = l3.ncells1; h.ncells2 = l3.ncells2; h.ncells3 = l3.ncells3;
    nmb = l3.nmb; nfld = l3.nfld; gids = l3.gids;
  } else {
    const auto &lt = leaf_traversal;
    h.nx1 = lt.nx1; h.nx2 = lt.nx2; h.nx3 = 1; h.is = lt.is; h.js = lt.js; h.ks = 0;
    h.ncells1 = lt.ncells1; h.ncells2 = lt.ncells2; h.ncells3 = 1;
    nmb = lt.nmb; nfld = lt.nfld; gids = lt.gids;
  }
  h.hslot0 = nmb; h.hslot1 = nfld;
  h.peer.clear(); h.soff.assign(1, 0); h.roff.assign(1, 0); h.npeer = 0;
  h.nblk_send = h.nblk_recv = 0;
  if (global_variable::nranks == 1) return;
#if MPI_PARALLEL_ENABLED
  if (!h.comm_set) {
    MPI_Comm_dup(MPI_COMM_WORLD, &h.comm);
    h.comm_set = true;
  }
  const int myrank = global_variable::my_rank, nranks = global_variable::nranks;
  const int nc1 = h.ncells1, nc2 = h.ncells2, nc3 = h.ncells3, is = h.is, js = h.js, ks = h.ks;
  auto flat = [&](const int b, const int k, const int j, const int i) {
    return ((b*nc3 + ks + k)*nc2 + js + j)*nc1 + is + i;
  };
  // what this rank needs, by owner (halo slots are in ascending gid, so is each list): gid and
  // box of cells, seven ints per block
  constexpr int kReq = 7;
  std::vector<std::vector<int>> want(nranks), wslot(nranks);
  for (int b=nmb; b<nfld; ++b) {
    const int r = pm->rank_eachmb[slot_gid[b]];
    if (r == myrank) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "dust halo: halo block " << slot_gid[b] << " is on this rank" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    const auto &c = h.rect[b - nmb];
    want[r].insert(want[r].end(), {slot_gid[b], c[0], c[1], c[2], c[3], c[4], c[5]});
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
    for (int q=rdisp[r]; q<rdisp[r + 1]; q+=kReq) {
      const int g = gflat[q];
      const int i0 = gflat[q + 1], i1 = gflat[q + 2], j0 = gflat[q + 3], j1 = gflat[q + 4];
      const int k0 = gflat[q + 5], k1 = gflat[q + 6];
      if (g < gids || g >= gids + nmb || i0 < 0 || i1 > h.nx1 || j0 < 0 || j1 > h.nx2 ||
          k0 < 0 || k1 > h.nx3 || i0 >= i1 || j0 >= j1 || k0 >= k1) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "dust halo: rank " << r << " asks rank " << myrank << " for block " << g
                  << " cells [" << i0 << ", " << i1 << ") x [" << j0 << ", " << j1 << ") x ["
                  << k0 << ", " << k1 << ")" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      for (int k=k0; k<k1; ++k) {
        for (int j=j0; j<j1; ++j) for (int i=i0; i<i1; ++i) sh.push_back(flat(g - gids, k, j, i));
      }
      ++h.nblk_send;
    }
    for (const int b : wslot[r]) {
      const auto &c = h.rect[b - nmb];
      for (int k=c[4]; k<c[5]; ++k) {
        for (int j=c[2]; j<c[3]; ++j) for (int i=c[0]; i<c[1]; ++i) rh.push_back(flat(b, k, j, i));
      }
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
  ++h.builds;
#else
  (void)pm; (void)slot_gid; (void)gids;
#endif
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::HaloForwardPack / HaloReverseAdd / HaloForwardFlat / HaloReverseAddFlat
//! \brief The exchanges of the leaf fields: the packed gather field gpack (9 components,
//! forward), the deposits qdep / dmom (components c0 .. c0 + nc - 1, reverse), the merge
//! budget fields (c, leaf) and the per-leaf flags (forward and reverse).

void DustGasDrag::HaloForwardPack(DvceArray5D<Real> &gp) {
  if (halo_.npeer == 0) return;
  const int nc1 = halo_.ncells1, n12 = halo_.ncells1*halo_.ncells2;
  if (halo_.three) {   // 3-D clouds: gpack3 (12 components)
    AccPack3 a; a.f = gp; a.nc1 = nc1; a.n12 = n12; a.n123 = n12*halo_.ncells3;
    Exchange(halo_, gp.extent_int(4), false, a);
    return;
  }
  AccPack a; a.f = gp; a.ks = pmy_pack->pmesh->mb_indcs.ks; a.nc1 = nc1; a.n12 = n12;
  Exchange(halo_, gp.extent_int(4), false, a);
}

void DustGasDrag::HaloReverseAdd(DvceArray5D<Real> &f, const int c0, const int nc) {
  if (halo_.npeer == 0) return;
  const int nc1 = halo_.ncells1, n12 = halo_.ncells1*halo_.ncells2;
  if (halo_.three) {
    Acc5D3 a; a.f = f; a.c0 = c0; a.nc1 = nc1; a.n12 = n12; a.n123 = n12*halo_.ncells3;
    Exchange(halo_, nc, true, a);
    return;
  }
  Acc5D a; a.f = f; a.c0 = c0; a.ks = pmy_pack->pmesh->mb_indcs.ks; a.nc1 = nc1; a.n12 = n12;
  Exchange(halo_, nc, true, a);
}

void DustGasDrag::HaloForward5D(DvceArray5D<Real> &f, const int c0, const int nc) {
  if (halo_.npeer == 0) return;
  const int nc1 = halo_.ncells1, n12 = halo_.ncells1*halo_.ncells2;
  if (halo_.three) {
    Acc5D3 a; a.f = f; a.c0 = c0; a.nc1 = nc1; a.n12 = n12; a.n123 = n12*halo_.ncells3;
    Exchange(halo_, nc, false, a);
    return;
  }
  Acc5D a; a.f = f; a.c0 = c0; a.ks = pmy_pack->pmesh->mb_indcs.ks; a.nc1 = nc1; a.n12 = n12;
  Exchange(halo_, nc, false, a);
}

void DustGasDrag::HaloForwardFlat(DvceArray2D<Real> &f, const int nc) {
  if (halo_.npeer == 0) return;
  AccFlat a; a.f = f;
  Exchange(halo_, nc, false, a);
}

void DustGasDrag::HaloReverseAddFlat(DvceArray2D<Real> &f, const int nc) {
  if (halo_.npeer == 0) return;
  AccFlat a; a.f = f;
  Exchange(halo_, nc, true, a);
}

void DustGasDrag::HaloForwardFlags(DvceArray1D<int> &f) {
  if (halo_.npeer == 0) return;
  AccFlatInt a; a.f = f;
  Exchange(halo_, 1, false, a);
}

void DustGasDrag::HaloReverseAddFlags(DvceArray1D<int> &f) {
  if (halo_.npeer == 0) return;
  AccFlatInt a; a.f = f;
  Exchange(halo_, 1, true, a);
}

}  // namespace dust
