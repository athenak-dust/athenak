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
//! interval, next event); a regrid whose resampling has not run yet
//! from the record regrid_evt.  Records are found by name, so a file written by a code with
//! more (or fewer) records is still read.
//!
//! Several MPI ranks: every rank writes one such section with its particles,
//! one after the other in rank order; "ranks written" in the head of every section is their
//! number.  The dust records of a section are its rank's (the tag counter differs by rank) and
//! include rank_blocks (rank, first gid, number of blocks).  A run on as many ranks with the
//! same blocks per rank takes its own section back exactly; any other run reads all sections,
//! keeps the particles of its blocks, takes the records of section 0 and starts the tag counter
//! anew.  Files without the section (written before this, or by runs without particles) are
//! still readable by runs without particles.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "athena.hpp"
#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif
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
  // the blocks of the rank that writes this section
  const std::int32_t rb[3] = {global_variable::my_rank, pmy_pack->gids, pmy_pack->nmb_thispack};
  PutRecord(b, "rank_blocks", rb, sizeof(rb));
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
  // merge clock and burst
  {
    const double mc[3] = {static_cast<double>(cloud_merge_t0), static_cast<double>(cloud_merge_dt),
                          static_cast<double>(cloud_merge_burst_until)};
    const std::int64_t mn[2] = {cloud_merge_by_time ? 1 : 0, cloud_merge_next};
    char mclk[sizeof(mc) + sizeof(mn)];
    std::memcpy(mclk, mc, sizeof(mc));
    std::memcpy(mclk + sizeof(mc), mn, sizeof(mn));
    PutRecord(b, "merge_clk", mclk, sizeof(mclk));
  }
  // dynamic mesh: pending mesh check after these outputs, cooldown age per block
  {
    Mesh *pm = pmy_pack->pmesh;
    if (experimental_amr && pm->adaptive && pm->pmr != nullptr) {
      std::vector<std::int32_t> v(2 + pm->nmb_total);
      v[0] = amr_check_after_output ? 1 : 0;
      v[1] = pm->nmb_total;
      for (int m=0; m<pm->nmb_total; ++m) v[2 + m] = pm->pmr->ncyc_since_ref(m);
      PutRecord(b, "amr_state", v.data(), v.size()*sizeof(std::int32_t));
    }
  }
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
  const std::int32_t am = amr_regrids;
  PutRecord(b, "amr_regrids", &am, sizeof(am));
  // a regrid whose resampling is still to come (a file that Finalize writes
  // right after the last regrid of a run) and the event counters
  const std::int64_t ev[6] = {amr_event_pending ? 1 : 0, amr_event_nnew, amr_event_ndel,
                              amr_event_regrid, amr_event_handled, amr_event_merged};
  PutRecord(b, "regrid_evt", ev, sizeof(ev));
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
    } else if (name == "rank_blocks") {
      std::int32_t v[3]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));   // read by the caller
    } else if (name == "merge_pass") {
      std::int32_t v; want(sizeof(v)); std::memcpy(&v, d, sizeof(v)); cloud_merge_pass = v;
    } else if (name == "mroot") {
      if (n % sizeof(Real) != 0) Broken("dust record mroot has an unexpected size");
      cloud_merge_mroot.resize(n/sizeof(Real));
      std::memcpy(cloud_merge_mroot.data(), d, n);
    } else if (name == "resample") {
      std::uint64_t v[2]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      cloud_resample_pass = v[0]; cloud_resample_removed = v[1];
    } else if (name == "merge_clk") {
      double mc[3]; std::int64_t mn[2]; want(sizeof(mc) + sizeof(mn));
      std::memcpy(mc, d, sizeof(mc)); std::memcpy(mn, d + sizeof(mc), sizeof(mn));
      cloud_merge_burst_until = mc[2];
      // the merge clock continues only with the same origin and interval
      if (mn[0] == 1 && cloud_merge_by_time && mc[0] == cloud_merge_t0 && mc[1] == cloud_merge_dt) {
        cloud_merge_next = mn[1]; cloud_merge_clock_restored = true;
      }
    } else if (name == "amr_state") {
      if (n < static_cast<std::int32_t>(2*sizeof(std::int32_t)) || n % sizeof(std::int32_t) != 0) {
        Broken("dust record amr_state has an unexpected size");
      }
      std::vector<std::int32_t> v(n/sizeof(std::int32_t));
      std::memcpy(v.data(), d, n);
      if (static_cast<std::size_t>(v[1]) + 2 != v.size()) Broken("dust record amr_state is inconsistent");
      amr_restart_state_found = true;
      amr_restart_pending = (v[0] != 0);
      amr_restart_ages.assign(v.begin() + 2, v.end());
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
    } else if (name == "amr_regrids") {
      std::int32_t v; want(sizeof(v)); std::memcpy(&v, d, sizeof(v)); amr_regrids = v;
    } else if (name == "regrid_evt") {
      std::int64_t v[6]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      amr_event_pending = (v[0] != 0);
      amr_event_nnew = static_cast<int>(v[1]); amr_event_ndel = static_cast<int>(v[2]);
      amr_event_regrid = static_cast<int>(v[3]);
      amr_event_handled = v[4]; amr_event_merged = v[5];
    } else if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust restart): record '" << name << "' is not known to this code "
                << "and is ignored" << std::endl;
    }
    at += 16 + sizeof(n) + static_cast<std::size_t>(n);
  }
  // a fixed PC2 / split-BE run does not keep the switching metric of the checkpoint
  if (!NeedsHybridMetric()) InvalidateHybridMetric();
}

//----------------------------------------------------------------------------------------
//! \fn dust::PackParticleRestart
//! \brief The whole restart section of this pack's particles as bytes.

void PackParticleRestart(MeshBlockPack *pmbp, std::vector<char> &buf) {
  particles::Particles *ppar = pmbp->ppart;
  if (global_variable::nranks > 1 && pmbp->pdust == nullptr) {
    Broken("particles without dust in the restart file are implemented for one MPI rank only");
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
                                static_cast<std::int32_t>(sizeof(int)), global_variable::nranks,
                                nrdata, nidata};
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
//! \fn dust::ParticleRestartSections
//! \brief The number of sections (ranks that wrote the file) from the head of the first one.

int ParticleRestartSections(const char *head) {
  std::int32_t h[6];
  std::memcpy(h, head + 8, sizeof(h));
  return h[3];
}

namespace {
// the parts of one section, after checking its head and sizes against this run
struct Section {
  std::int64_t np = 0;
  Real dtnew = 0.0;
  const char *state = nullptr;
  std::int32_t nstate = 0;
  const char *rows = nullptr;   // nrdata rows of Real, then nidata rows of int
};
Section ParseSection(const std::vector<char> &buf, const int nsec, const int nrdata, const int nidata) {
  if (buf.size() < kHeadBytes + 8 || std::memcmp(buf.data(), kMagic, 8) != 0 ||
      std::memcmp(buf.data() + buf.size() - 8, kMagic, 8) != 0) {
    Broken("missing or truncated (no head or end marker)");
  }
  std::int32_t head[6];
  std::memcpy(head, buf.data() + 8, sizeof(head));
  if (head[0] != kVersion || head[1] != static_cast<std::int32_t>(sizeof(Real)) ||
      head[2] != static_cast<std::int32_t>(sizeof(int)) || head[3] != nsec ||
      head[4] != nrdata || head[5] != nidata) {
    Broken("version, type sizes, rank count or the number of particle columns do not match "
           "this run (file: version " + std::to_string(head[0]) + ", Real " +
           std::to_string(head[1]) + " bytes, int " + std::to_string(head[2]) + " bytes, ranks " +
           std::to_string(head[3]) + ", nrdata " + std::to_string(head[4]) + ", nidata " +
           std::to_string(head[5]) + "; run: nrdata " + std::to_string(nrdata) +
           ", nidata " + std::to_string(nidata) + ")");
  }
  Section s;
  std::size_t at = kHeadBytes;
  std::memcpy(&s.np, buf.data() + at, sizeof(s.np)); at += sizeof(s.np);
  std::memcpy(&s.dtnew, buf.data() + at, sizeof(s.dtnew)); at += sizeof(s.dtnew);
  std::memcpy(&s.nstate, buf.data() + at, sizeof(s.nstate)); at += sizeof(s.nstate);
  const std::size_t need = at + static_cast<std::size_t>(s.nstate) +
      static_cast<std::size_t>(s.np)*(nrdata*sizeof(Real) + nidata*sizeof(int)) + 8;
  if (s.np < 0 || s.nstate < 0 || need != buf.size()) Broken("sizes inside the section are inconsistent");
  s.state = buf.data() + at;
  s.rows = buf.data() + at + s.nstate;
  return s;
}
// the record rank_blocks of a section's dust records (false if there is none)
bool RankBlocks(const char *p, const std::int32_t nbytes, std::int32_t rb[3]) {
  std::size_t at = 0;
  while (at + 16 + sizeof(std::int32_t) <= static_cast<std::size_t>(nbytes)) {
    char key[17] = {0};
    std::memcpy(key, p + at, 16);
    std::int32_t n = 0;
    std::memcpy(&n, p + at + 16, sizeof(n));
    if (n < 0) return false;
    if (std::string(key) == "rank_blocks" && n == static_cast<std::int32_t>(3*sizeof(std::int32_t))) {
      std::memcpy(rb, p + at + 16 + sizeof(n), 3*sizeof(std::int32_t));
      return true;
    }
    at += 16 + sizeof(n) + static_cast<std::size_t>(n);
  }
  return false;
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn dust::UnpackParticleRestart
//! \brief Restore the particles (and the dust records) of this pack from the nsec sections of
//! a restart file (read(k, buf) fills buf with section k).  The arrays are sized to the
//! particles of the file, with the headroom of the device split when clouds are refinable.
//! As many ranks with the same blocks as the writer: this rank's own section, exactly.  Else
//! the particles of this rank's blocks from all sections, the records of
//! section 0, a new tag counter.

void UnpackParticleRestart(MeshBlockPack *pmbp, const int nsec,
                           const std::function<void(int, std::vector<char>&)> &read) {
  particles::Particles *ppar = pmbp->ppart;
  Mesh *pm = pmbp->pmesh;
  const int nranks = global_variable::nranks, me = global_variable::my_rank;
  const int gids = pmbp->gids, nmb = pmbp->nmb_thispack;
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  if (nsec < 1) Broken("no section");
  std::vector<char> own;
  int exact = 0;
  if (nsec == nranks) {
    read(me, own);
    const Section s = ParseSection(own, nsec, nrdata, nidata);
    std::int32_t rb[3];
    // one rank: also files written before rank_blocks existed
    if (nranks == 1 || (RankBlocks(s.state, s.nstate, rb) && rb[0] == me && rb[1] == gids &&
                        rb[2] == nmb)) {
      exact = 1;
    }
  }
#if MPI_PARALLEL_ENABLED
  if (nranks > 1) MPI_Allreduce(MPI_IN_PLACE, &exact, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
#endif
  // the particles kept, column by column
  std::vector<std::vector<Real>> rc(nrdata);
  std::vector<std::vector<int>> ic(nidata);
  Real dtnew = 0.0;
  std::int64_t nread = 0;
  auto take = [&](const Section &s, const bool all) {
    const Real *r = reinterpret_cast<const Real*>(s.rows);
    std::vector<Real> row(static_cast<std::size_t>(s.np));
    std::vector<int> irow(static_cast<std::size_t>(s.np));
    // which particles: all (exact) or those of this rank's blocks
    std::vector<std::int64_t> sel;
    const char *ip = s.rows + static_cast<std::size_t>(s.np)*nrdata*sizeof(Real);
    std::memcpy(irow.data(), ip + static_cast<std::size_t>(PGID)*s.np*sizeof(int), s.np*sizeof(int));
    for (std::int64_t p=0; p<s.np; ++p) {
      const bool mine = (irow[p] >= gids && irow[p] < gids + nmb);
      if (all && !mine) Broken("a particle names a MeshBlock that does not exist");
      if (mine) sel.push_back(p);
    }
    for (int c=0; c<nrdata; ++c) {
      std::memcpy(row.data(), reinterpret_cast<const char*>(r) + static_cast<std::size_t>(c)*s.np*sizeof(Real),
                  s.np*sizeof(Real));
      for (const auto p : sel) rc[c].push_back(row[p]);
    }
    for (int c=0; c<nidata; ++c) {
      std::memcpy(irow.data(), ip + static_cast<std::size_t>(c)*s.np*sizeof(int), s.np*sizeof(int));
      for (const auto p : sel) ic[c].push_back(irow[p]);
    }
    nread += s.np;
  };
  if (exact) {
    const Section s = ParseSection(own, nsec, nrdata, nidata);
    if (pmbp->pdust != nullptr) pmbp->pdust->UnpackRestartState(s.state, s.nstate);
    dtnew = s.dtnew;
    take(s, true);
  } else {
    std::vector<char> buf;
    for (int k=0; k<nsec; ++k) {
      if (k == me && !own.empty()) {buf.swap(own);} else {read(k, buf);}
      const Section s = ParseSection(buf, nsec, nrdata, nidata);
      if (k == 0) {
        if (pmbp->pdust != nullptr) pmbp->pdust->UnpackRestartState(s.state, s.nstate);
        dtnew = s.dtnew;
      }
      take(s, false);
    }
    // the counters of the sections were per rank: the tag counter starts again from the largest
    // tag (EnsureNextTag, collectively at the first cycle)
    if (pmbp->pdust != nullptr) pmbp->pdust->cloud_next_tag = -1;
  }
  const std::int64_t np = rc.empty() ? 0 : static_cast<std::int64_t>(rc[0].size());
  const bool clouds = pmbp->pdust != nullptr &&
                      pmbp->pdust->smr_transfer_mode == DustSMRTransfer::refinable_cloud;
  const std::int64_t cap = std::max<std::int64_t>(1, clouds ? np + np/4 : np);
  Kokkos::realloc(ppar->prtcl_rdata, nrdata, cap);
  Kokkos::realloc(ppar->prtcl_idata, nidata, cap);
  auto pr = Kokkos::create_mirror_view(ppar->prtcl_rdata);
  auto pi = Kokkos::create_mirror_view(ppar->prtcl_idata);
  Kokkos::deep_copy(pr, 0.0);
  Kokkos::deep_copy(pi, 0);
  for (int c=0; c<nrdata; ++c) for (std::int64_t p=0; p<np; ++p) pr(c,p) = rc[c][p];
  for (int c=0; c<nidata; ++c) for (std::int64_t p=0; p<np; ++p) pi(c,p) = ic[c][p];
  Kokkos::deep_copy(ppar->prtcl_rdata, pr);
  Kokkos::deep_copy(ppar->prtcl_idata, pi);
  ppar->nprtcl_thispack = static_cast<int>(np);
  ppar->dtnew = dtnew;
  pm->nprtcl_thisrank = static_cast<int>(np);
  pm->nprtcl_eachrank[me] = static_cast<int>(np);
  long long tot = np;
#if MPI_PARALLEL_ENABLED
  if (nranks > 1) {
    int n32 = static_cast<int>(np);
    MPI_Allgather(&n32, 1, MPI_INT, pm->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &tot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  pm->nprtcl_total = static_cast<int>(tot);
  // every rank read all sections: each particle of the file is kept by exactly one rank
  if (!exact && tot != nread) {
    Broken("the particles of the file do not fall into the blocks of this run (" +
           std::to_string(tot) + " kept, " + std::to_string(nread) + " in the file)");
  }
  if (me == 0) {
    std::cout << "# restart: " << tot << " particles restored from " << nsec << " section(s) "
              << (exact ? "(each rank its own)" : "(sorted by block)") << " (" << nrdata
              << " real and " << nidata << " integer columns), particle dt " << dtnew;
    if (pmbp->pdust != nullptr) {
      std::cout << ", next cloud tag " << pmbp->pdust->cloud_next_tag << ", merge pass "
                << pmbp->pdust->cloud_merge_pass;
    }
    std::cout << std::endl;
  }
}

}  // namespace dust
