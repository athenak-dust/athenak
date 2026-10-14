//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_restart.cpp
//! \brief Particles and refinable-cloud state in the native restart file.
//!
//! The rst file of AthenaK holds the input, the mesh tree and the cell data of every
//! MeshBlock; it has no particles.  One section is appended after the MeshBlock data:
//!
//!   char[8]  "AKPRST01"
//!   int32    version (1), sizeof(Real), sizeof(int), ranks written (1), nrdata, nidata
//!   uint64   bytes of the whole section (this head and the end marker included)
//!   int64    number of particles
//!   Real     particle time step (Particles::dtnew)
//!   int32    bytes of the dust state, then that many bytes of records
//!            (char[16] name, int32 bytes, data); empty without a <dust> block
//!   Real     prtcl_rdata, row by row (nrdata rows of N)
//!   int      prtcl_idata, row by row (nidata rows of N)
//!   char[8]  "AKPRST01"
//!
//! Every real and integer column is written, so tag, level code and lobes, MeshBlock id,
//! stopping time and the predictor copies come back bit for bit.  The dust records hold what
//! changes during a run and cannot be rebuilt from the particles: the next free tag, the
//! merge-pass counter (it rotates the walks), the root cloud masses, the resampling and the
//! cumulative split / merge counters of the history, the state of the hybrid coupling.  The
//! leaf maps, masks and per-cycle fields are rebuilt as in a fresh run; the merge schedule
//! and the cycle-mode resampling schedule follow from the cycle number of the file, the
//! time-mode resampling schedule from the record resample_clk (origin,
//! interval, next event).  Records are found by name, so a file written by a code with more
//! (or fewer) records is still read.
//!
//! One MPI rank only: the head has room for a per-rank layout, which is not implemented.
//! Files without the section (written before this, or by runs without particles) are still
//! readable by runs without particles.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock_pack.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

namespace {
const char kMagic[9] = "AKPRST01";
constexpr std::int32_t kVersion = 1;
constexpr std::size_t kHeadBytes = 8 + 6*sizeof(std::int32_t) + sizeof(std::uint64_t);

void Put(std::vector<char> &b, const void *p, std::size_t n) {
  const char *c = static_cast<const char*>(p);
  b.insert(b.end(), c, c + n);
}
template <typename T>
void PutValue(std::vector<char> &b, const T &v) {Put(b, &v, sizeof(T));}
void PutRecord(std::vector<char> &b, const char *name, const void *p, std::size_t n) {
  char key[16] = {0};
  std::strncpy(key, name, 15);
  Put(b, key, 16);
  PutValue(b, static_cast<std::int32_t>(n));
  Put(b, p, n);
}
[[noreturn]] void Broken(const std::string &what) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
            << "particle section of the restart file: " << what << std::endl;
  std::exit(EXIT_FAILURE);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn dust::ParticleRestartHeadBytes, dust::ParticleRestartSectionBytes
//! \brief Size of the fixed head of the section, and the size of the whole section read
//! from such a head (0 if the bytes are not a head: no section in the file).

std::size_t ParticleRestartHeadBytes() {return kHeadBytes;}

std::uint64_t ParticleRestartSectionBytes(const char *head) {
  if (std::memcmp(head, kMagic, 8) != 0) return 0;
  std::uint64_t n = 0;
  std::memcpy(&n, head + 8 + 6*sizeof(std::int32_t), sizeof(n));
  return n;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::PackRestartState, DustGasDrag::UnpackRestartState
//! \brief The dust records of the restart section (see the file header).

void DustGasDrag::PackRestartState(std::vector<char> &b) const {
  const std::int32_t tag = cloud_next_tag, pass = cloud_merge_pass;
  PutRecord(b, "next_tag", &tag, sizeof(tag));
  PutRecord(b, "merge_pass", &pass, sizeof(pass));
  if (!cloud_merge_mroot.empty()) {
    PutRecord(b, "mroot", cloud_merge_mroot.data(), cloud_merge_mroot.size()*sizeof(Real));
  }
  const std::uint64_t rs[2] = {cloud_resample_pass, cloud_resample_removed};
  PutRecord(b, "resample", rs, sizeof(rs));
  // the time-mode resampling schedule (by_time 0 in cycle mode; next -1 before
  // the first check)
  const double rc[2] = {static_cast<double>(cloud_resample_t0), static_cast<double>(cloud_resample_dt)};
  const std::int64_t rn[4] = {cloud_resample_by_time ? 1 : 0, cloud_resample_next,
                              static_cast<std::int64_t>(cloud_resample_events),
                              static_cast<std::int64_t>(cloud_resample_skipped)};
  char clk[sizeof(rc) + sizeof(rn)];
  std::memcpy(clk, rc, sizeof(rc));
  std::memcpy(clk + sizeof(rc), rn, sizeof(rn));
  PutRecord(b, "resample_clk", clk, sizeof(clk));
  const std::uint64_t sp[4] = {cloud_split_event, cloud_split_count, cloud_merge_count,
                               cloud_lobe_split_count};
  PutRecord(b, "split", sp, sizeof(sp));
  const std::uint64_t co[4] = {cloud_coarsen_events, cloud_coarsen_in, cloud_coarsen_reject,
                               cloud_far_release};
  PutRecord(b, "coarsen_n", co, sizeof(co));
  const Real dk[2] = {cloud_coarsen_dk_grad, cloud_coarsen_dk_res};
  PutRecord(b, "coarsen_dk", dk, sizeof(dk));
  const std::int32_t hi[2] = {static_cast<std::int32_t>(hybrid_mode), hybrid_have_metric ? 1 : 0};
  PutRecord(b, "hybrid_i", hi, sizeof(hi));
  const Real hr[3] = {hybrid_feedback_rate_max, hybrid_last_zeta, hybrid_last_chi};
  PutRecord(b, "hybrid_r", hr, sizeof(hr));
  const std::uint64_t hn[2] = {hybrid_pc2_cycles, hybrid_split_be_cycles};
  PutRecord(b, "hybrid_n", hn, sizeof(hn));
}

void DustGasDrag::UnpackRestartState(const char *p, std::size_t nbytes) {
  std::size_t at = 0;
  cloud_resample_restarted = true;
  while (at + 16 + sizeof(std::int32_t) <= nbytes) {
    char key[17] = {0};
    std::memcpy(key, p + at, 16);
    std::int32_t n = 0;
    std::memcpy(&n, p + at + 16, sizeof(n));
    const char *d = p + at + 16 + sizeof(n);
    if (n < 0 || at + 16 + sizeof(n) + static_cast<std::size_t>(n) > nbytes) Broken("dust record runs past the end");
    const std::string name(key);
    auto want = [&](std::size_t size) {
      if (static_cast<std::size_t>(n) != size) Broken("dust record " + name + " has an unexpected size");
    };
    if (name == "next_tag") {
      std::int32_t v; want(sizeof(v)); std::memcpy(&v, d, sizeof(v)); cloud_next_tag = v;
    } else if (name == "merge_pass") {
      std::int32_t v; want(sizeof(v)); std::memcpy(&v, d, sizeof(v)); cloud_merge_pass = v;
    } else if (name == "mroot") {
      if (n % sizeof(Real) != 0) Broken("dust record mroot has an unexpected size");
      cloud_merge_mroot.resize(n/sizeof(Real));
      std::memcpy(cloud_merge_mroot.data(), d, n);
    } else if (name == "resample") {
      std::uint64_t v[2]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      cloud_resample_pass = v[0]; cloud_resample_removed = v[1];
    } else if (name == "resample_clk") {
      double rc[2]; std::int64_t rn[4]; want(sizeof(rc) + sizeof(rn));
      std::memcpy(rc, d, sizeof(rc)); std::memcpy(rn, d + sizeof(rc), sizeof(rn));
      // only a time-mode schedule that had started is continued (see ResampleDue)
      if (rn[0] == 1 && rn[1] >= 0) {
        cloud_resample_clock_restored = true;
        cloud_resample_t0_file = rc[0]; cloud_resample_dt_file = rc[1];
        cloud_resample_next_file = rn[1];
        cloud_resample_events = static_cast<unsigned long long>(rn[2]);
        cloud_resample_skipped = static_cast<unsigned long long>(rn[3]);
      }
    } else if (name == "split") {
      std::uint64_t v[4]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      cloud_split_event = v[0]; cloud_split_count = v[1]; cloud_merge_count = v[2];
      cloud_lobe_split_count = v[3];
    } else if (name == "coarsen_n") {
      std::uint64_t v[4]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      cloud_coarsen_events = v[0]; cloud_coarsen_in = v[1]; cloud_coarsen_reject = v[2];
      cloud_far_release = v[3];
    } else if (name == "coarsen_dk") {
      Real v[2]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      cloud_coarsen_dk_grad = v[0]; cloud_coarsen_dk_res = v[1];
    } else if (name == "hybrid_i") {
      std::int32_t v[2]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      hybrid_mode = static_cast<HybridMode>(v[0]); hybrid_have_metric = (v[1] != 0);
    } else if (name == "hybrid_r") {
      Real v[3]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      hybrid_feedback_rate_max = v[0]; hybrid_last_zeta = v[1]; hybrid_last_chi = v[2];
    } else if (name == "hybrid_n") {
      std::uint64_t v[2]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      hybrid_pc2_cycles = v[0]; hybrid_split_be_cycles = v[1];
    } else if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust restart): record '" << name << "' is not known to this code "
                << "and is ignored" << std::endl;
    }
    at += 16 + sizeof(n) + static_cast<std::size_t>(n);
  }
}

//----------------------------------------------------------------------------------------
//! \fn dust::PackParticleRestart
//! \brief The whole restart section of this pack's particles as bytes.

void PackParticleRestart(MeshBlockPack *pmbp, std::vector<char> &buf) {
  particles::Particles *ppar = pmbp->ppart;
  if (global_variable::nranks > 1) {
    Broken("particles in the restart file are implemented for one MPI rank only");
  }
  const std::int64_t np = ppar->nprtcl_thispack;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  std::vector<char> state;
  if (pmbp->pdust != nullptr) pmbp->pdust->PackRestartState(state);
  buf.clear();
  buf.reserve(kHeadBytes + 64 + state.size() +
              static_cast<std::size_t>(np)*(nrdata*sizeof(Real) + nidata*sizeof(int)));
  Put(buf, kMagic, 8);
  const std::int32_t head[6] = {kVersion, static_cast<std::int32_t>(sizeof(Real)),
                                static_cast<std::int32_t>(sizeof(int)), 1, nrdata, nidata};
  Put(buf, head, sizeof(head));
  const std::uint64_t total = kHeadBytes + sizeof(std::int64_t) + sizeof(Real) + sizeof(std::int32_t) +
      state.size() + static_cast<std::uint64_t>(np)*(nrdata*sizeof(Real) + nidata*sizeof(int)) + 8;
  PutValue(buf, total);
  PutValue(buf, np);
  PutValue(buf, ppar->dtnew);
  PutValue(buf, static_cast<std::int32_t>(state.size()));
  Put(buf, state.data(), state.size());
  auto pr = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view_and_copy(HostMemSpace(), ppar->prtcl_idata);
  std::vector<Real> row(static_cast<std::size_t>(np));
  for (int c=0; c<nrdata; ++c) {
    for (std::int64_t p=0; p<np; ++p) row[p] = pr(c,p);
    Put(buf, row.data(), row.size()*sizeof(Real));
  }
  std::vector<int> irow(static_cast<std::size_t>(np));
  for (int c=0; c<nidata; ++c) {
    for (std::int64_t p=0; p<np; ++p) irow[p] = pi(c,p);
    Put(buf, irow.data(), irow.size()*sizeof(int));
  }
  Put(buf, kMagic, 8);
  if (buf.size() != total) Broken("internal size mismatch while packing");
}

//----------------------------------------------------------------------------------------
//! \fn dust::UnpackParticleRestart
//! \brief Restore the particles (and the dust records) of this pack from a section read from
//! a restart file.  The arrays are sized to the particles of the file, with the headroom of
//! the device split when clouds are refinable.

void UnpackParticleRestart(MeshBlockPack *pmbp, const std::vector<char> &buf) {
  particles::Particles *ppar = pmbp->ppart;
  Mesh *pm = pmbp->pmesh;
  if (global_variable::nranks > 1) {
    Broken("particles in the restart file are implemented for one MPI rank only");
  }
  if (buf.size() < kHeadBytes + 8 || std::memcmp(buf.data(), kMagic, 8) != 0 ||
      std::memcmp(buf.data() + buf.size() - 8, kMagic, 8) != 0) {
    Broken("missing or truncated (no head or end marker)");
  }
  std::int32_t head[6];
  std::memcpy(head, buf.data() + 8, sizeof(head));
  if (head[0] != kVersion || head[1] != static_cast<std::int32_t>(sizeof(Real)) ||
      head[2] != static_cast<std::int32_t>(sizeof(int)) || head[3] != 1 ||
      head[4] != ppar->nrdata || head[5] != ppar->nidata) {
    Broken("version, type sizes, rank count or the number of particle columns do not match "
           "this run (file: version " + std::to_string(head[0]) + ", Real " +
           std::to_string(head[1]) + " bytes, int " + std::to_string(head[2]) + " bytes, ranks " +
           std::to_string(head[3]) + ", nrdata " + std::to_string(head[4]) + ", nidata " +
           std::to_string(head[5]) + "; run: nrdata " + std::to_string(ppar->nrdata) +
           ", nidata " + std::to_string(ppar->nidata) + ")");
  }
  std::size_t at = kHeadBytes;
  std::int64_t np = 0;
  std::memcpy(&np, buf.data() + at, sizeof(np)); at += sizeof(np);
  Real dtnew = 0.0;
  std::memcpy(&dtnew, buf.data() + at, sizeof(dtnew)); at += sizeof(dtnew);
  std::int32_t nstate = 0;
  std::memcpy(&nstate, buf.data() + at, sizeof(nstate)); at += sizeof(nstate);
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  const std::size_t need = at + static_cast<std::size_t>(nstate) +
      static_cast<std::size_t>(np)*(nrdata*sizeof(Real) + nidata*sizeof(int)) + 8;
  if (np < 0 || nstate < 0 || need != buf.size()) Broken("sizes inside the section are inconsistent");
  if (pmbp->pdust != nullptr) pmbp->pdust->UnpackRestartState(buf.data() + at, nstate);
  at += nstate;

  const bool clouds = pmbp->pdust != nullptr &&
                      pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud;
  const std::int64_t cap = std::max<std::int64_t>(1, clouds ? np + np/4 : np);
  Kokkos::realloc(ppar->prtcl_rdata, nrdata, cap);
  Kokkos::realloc(ppar->prtcl_idata, nidata, cap);
  auto pr = Kokkos::create_mirror_view(ppar->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view(ppar->prtcl_idata);
  Kokkos::deep_copy(pr, 0.0);
  Kokkos::deep_copy(pi, 0);
  std::vector<Real> row(static_cast<std::size_t>(np));
  for (int c=0; c<nrdata; ++c) {
    std::memcpy(row.data(), buf.data() + at, row.size()*sizeof(Real)); at += row.size()*sizeof(Real);
    for (std::int64_t p=0; p<np; ++p) pr(c,p) = row[p];
  }
  std::vector<int> irow(static_cast<std::size_t>(np));
  const int gids = pmbp->gids, nmb = pmbp->nmb_thispack;
  for (int c=0; c<nidata; ++c) {
    std::memcpy(irow.data(), buf.data() + at, irow.size()*sizeof(int)); at += irow.size()*sizeof(int);
    for (std::int64_t p=0; p<np; ++p) pi(c,p) = irow[p];
    if (c == PGID) {
      for (std::int64_t p=0; p<np; ++p) {
        if (irow[p] < gids || irow[p] >= gids + nmb) Broken("a particle names a MeshBlock that does not exist");
      }
    }
  }
  Kokkos::deep_copy(ppar->prtcl_rdata, pr);
  Kokkos::deep_copy(ppar->prtcl_idata, pi);
  ppar->nprtcl_thispack = static_cast<int>(np);
  ppar->dtnew = dtnew;
  pm->nprtcl_thisrank = static_cast<int>(np);
  pm->nprtcl_eachrank[global_variable::my_rank] = static_cast<int>(np);
  pm->nprtcl_total = static_cast<int>(np);
  if (global_variable::my_rank == 0) {
    std::cout << "# restart: " << np << " particles restored (" << nrdata << " real and " << nidata
              << " integer columns), particle dt " << dtnew;
    if (pmbp->pdust != nullptr) {
      std::cout << ", next cloud tag " << pmbp->pdust->cloud_next_tag << ", merge pass "
                << pmbp->pdust->cloud_merge_pass;
    }
    std::cout << std::endl;
  }
}

}  // namespace dust
