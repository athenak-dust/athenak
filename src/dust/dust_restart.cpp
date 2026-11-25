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
//!   int32    version (2), sizeof(Real), sizeof(int), ranks written (1), nrdata, nidata
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
//! cumulative split / merge counters of the history.  The
//! leaf maps, masks and per-cycle fields are rebuilt as in a fresh run; the merge schedule
//! and the cycle-mode resampling schedule follow from the cycle number of the file, the
//! time-mode resampling schedule from the record resample_clk (origin, interval, next
//! event); a regrid whose resampling has not run yet from the record regrid_evt.  Records
//! are found by name, so a file written by a code with more (or fewer) records is still
//! read.  The record species_table holds the resolved species table of the run as ASCII
//! lines (SpeciesTable::Record); a restart resolves the table of its input again and stops on any
//! difference (nspecies, kind of abundance, normalization, Omega, and per species tau_s,
//! stokes, epsilon / Z, ppc, H_p); a file without the record is read with a note.
//!
//! The optional records cloud_ord_n, cloud_ord and cloud_ord_tag follow the others: the
//! cell-order index of the cloud kernels (UpdateCloudOrder) as last built, which a restart
//! cannot rebuild from the particles (the kernels between two rebuilds run in the order of
//! the last build; the sums of the deposits follow it).  A restart on the same ranks and
//! blocks (each rank its own section: the same storage order) takes it back after checking
//! that it is a permutation; otherwise, without the records (older files), or when the writer
//! left them out (the int32 length of the dust state, dust_restart_order.hpp), the index is
//! rebuilt at the first cycle as before, and rank 0 reports the counts.  Each of the three
//! records may occur once per section, in the combinations of its status (stored: all three;
//! none or omitted: cloud_ord_n alone), with a sort cycle in [-1, cycle of the file]; any
//! other group is invalid and discarded the same way.  (Files of an unreleased development
//! version, which wrote an 8-byte cloud_ord_n, read as invalid; that version stops on the
//! 12-byte record of these files.)
//!
//! Several MPI ranks: every rank writes one such section with its particles,
//! one after the other in rank order; "ranks written" in the head of every section is their
//! number.  The dust records of a section are its rank's (the tag counter differs by rank) and
//! include rank_blocks (rank, first gid, number of blocks).  A run on as many ranks with the
//! same blocks per rank takes its own section back exactly; any other run reads all sections,
//! keeps the particles of its blocks, takes the records of section 0 and starts the tag counter
//! anew.  Files without the section (written before this, or by runs without particles) are
//! still readable by runs without particles.
//!
//! Version 2: the tag of a dust particle is an unsigned 64-bit
//! integer in two int columns, PTAG (bits 0-31) and PTAGH (bits 32-63, the last column), so a
//! dust section has nidata = 5 (version 1: 4).  The int32 record next_tag is replaced by
//! tag_state, uint64[3] = {counter set (0/1), next tag of the writing rank, bound}, bound being
//! above every tag issued in the whole history of the run (the largest of the counters of all
//! ranks, the live tags + 1 and the bound the run started from; the same in every section).  A
//! run that does not take its own section back (another rank count or block layout) starts
//! its new tags at the bound, so that no tag of a deleted particle is issued again.  Version 1
//! files are still read: the high words are 0 and the counter is the int32 record next_tag;
//! such a run on other ranks starts again from the largest live tag.

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
#include "dust_species.hpp"
#include "dust_restart_order.hpp"

namespace dust {

namespace {
// retired-compat: the records of retired options that older files carry (the hybrid
// coupling's state hybrid_i = int32[2], hybrid_r = Real[3], hybrid_n = uint64[2]; the
// kinetic-energy sums coarsen_dk = Real[2] of the tent pass and the host merge): their size in
// those files, 0 for any other name.  The reader checks the size as before and drops the record.
std::size_t RetiredRestartRecord(const std::string &name) {
  if (name == "hybrid_i") return 2*sizeof(std::int32_t);
  if (name == "hybrid_r") return 3*sizeof(Real);
  if (name == "hybrid_n") return 2*sizeof(std::uint64_t);
  if (name == "coarsen_dk") return 2*sizeof(Real);
  return 0;
}
const char kMagic[9] = "AKPRST01";
constexpr std::int32_t kVersion = 2;   // 64-bit tags (version 1: int32 tags)
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

// the versions this code reads (1: int32 tags, 2: 64-bit tags)
bool ParticleRestartVersionKnown(const std::int32_t version) {return version == 1 || version == 2;}

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
  const std::int32_t pass = cloud_merge_pass;
  // the 64-bit tag counter of this rank and the bound of all tags issued (collective)
  const std::uint64_t ts[3] = {cloud_next_tag_set ? 1ULL : 0ULL, cloud_next_tag, IssuedTagBound()};
  PutRecord(b, "tag_state", ts, sizeof(ts));
  // the blocks of the rank that writes this section
  const std::int32_t rb[3] = {global_variable::my_rank, pmy_pack->gids, pmy_pack->nmb_thispack};
  PutRecord(b, "rank_blocks", rb, sizeof(rb));
  PutRecord(b, "merge_pass", &pass, sizeof(pass));
  // the resolved species table (ASCII, dust_species.cpp)
  PutRecord(b, "species_table", species_record.data(), species_record.size());
  if (!cloud_merge_mroot.empty()) {
    PutRecord(b, "mroot", cloud_merge_mroot.data(), cloud_merge_mroot.size()*sizeof(Real));
  }
  // the cloud residual rule's own root-mass estimate (made only while the G / R passes had none)
  if (!imex_mown.empty()) {
    PutRecord(b, "imex_mown", imex_mown.data(), imex_mown.size()*sizeof(Real));
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
    if (pm->adaptive && pm->pmr != nullptr) {
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
  const std::uint64_t co[4] = {cloud_coarsen_events, cloud_coarsen_in, cloud_coarsen_reject,  // retired-compat
                               cloud_far_release};
  PutRecord(b, "coarsen_n", co, sizeof(co));
  const std::int32_t am = amr_regrids;
  PutRecord(b, "amr_regrids", &am, sizeof(am));
  // a regrid whose resampling is still to come (a file that Finalize writes
  // right after the last regrid of a run) and the event counters
  const std::int64_t ev[6] = {amr_event_pending ? 1 : 0, amr_event_nnew, amr_event_ndel,
                              amr_event_regrid, amr_event_handled, amr_event_merged};
  PutRecord(b, "regrid_evt", ev, sizeof(ev));
  // the first regrid of a pending event (time, cycle), used by cloud_event_dt
  const double et[2] = {amr_event_first_time, static_cast<double>(amr_event_first_ncycle)};
  PutRecord(b, "regrid_evt_t", et, sizeof(et));
  // the physical-x3 ledger: the run totals of the outflow (the same on
  // every rank), the particles parked for removal and the dropped deposits (summed over the ranks
  // here, restored on rank 0 only, so a restart on any number of ranks counts them once)
  {
    auto *pb = pmy_pack->ppart->pbval_part;
    Kokkos::deep_copy(pb->x3_removed.h_view, pb->x3_removed.d_view);
    const auto &ds = pbval_dm->dropped_sum;
    const int nd = static_cast<int>(ds.size());
    std::vector<double> led(5 + nd, 0.0);
    led[0] = x3_outflow_mass; led[1] = static_cast<double>(x3_outflow_count);
    led[2] = pb->x3_removed.h_view(0); led[3] = pb->x3_removed.h_view(1);
    led[4] = static_cast<double>(nd);
    for (int v=0; v<nd; ++v) led[5 + v] = ds[v];
#if MPI_PARALLEL_ENABLED
    if (global_variable::nranks > 1) {
      MPI_Allreduce(MPI_IN_PLACE, led.data() + 2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if (nd > 0) MPI_Allreduce(MPI_IN_PLACE, led.data() + 5, nd, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
#endif
    PutRecord(b, "x3_ledger", led.data(), led.size()*sizeof(double));
  }
}

void DustGasDrag::UnpackRestartState(const char *p, std::size_t nbytes, const bool own_section) {
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
    if (name == "next_tag") {   // version 1: the int32 counter (-1: not set)
      std::int32_t v; want(sizeof(v)); std::memcpy(&v, d, sizeof(v));
      cloud_next_tag_set = (v >= 0);
      cloud_next_tag = (v >= 0) ? static_cast<std::uint64_t>(v) : 0;
    } else if (name == "tag_state") {   // version 2
      std::uint64_t v[3]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      cloud_next_tag_set = (v[0] != 0);
      cloud_next_tag = v[1];
      cloud_tag_floor = v[2];
    } else if (name == "species_table") {   // the species may not change
      species_record_found = true;
      const std::string diff = CompareSpeciesRecords(std::string(d, static_cast<std::size_t>(n)),
                                                     species_record);
      if (!diff.empty()) {
        if (global_variable::my_rank == 0) {
          std::cout << "### FATAL ERROR in " << __FILE__ << std::endl
                    << "the dust species of the input differ from those of the restart file "
                    << "(record species_table); the species cannot change at a "
                    << "restart:" << std::endl << diff << std::flush;
        }
        std::exit(EXIT_FAILURE);
      }
    } else if (name == "rank_blocks") {
      std::int32_t v[3]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));   // read by the caller
    } else if (name == "merge_pass") {
      std::int32_t v; want(sizeof(v)); std::memcpy(&v, d, sizeof(v)); cloud_merge_pass = v;
    } else if (name == "mroot") {
      if (n % sizeof(Real) != 0) Broken("dust record mroot has an unexpected size");
      cloud_merge_mroot.resize(n/sizeof(Real));
      std::memcpy(cloud_merge_mroot.data(), d, n);
    } else if (name == "imex_mown") {
      if (n % sizeof(Real) != 0) Broken("dust record imex_mown has an unexpected size");
      imex_mown.resize(n/sizeof(Real));
      std::memcpy(imex_mown.data(), d, n);
      imex_mown_restored = true;
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
    } else if (name == "shear_until") {
      want(sizeof(double));   // (written by an interim version; the shear-face window needs no state)
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
    } else if (name == "cloud_ord_n") {   // (cell-order index: checked in ApplyRestartOrder)
      // (each of the three records once per section; a repeated one makes the group invalid)
      ++rst_ord_seen[0];
      rst_ord_info = true;
      if (n == static_cast<std::int32_t>(3*sizeof(std::int32_t))) {
        std::int32_t v[3]; std::memcpy(v, d, sizeof(v));
        rst_ord_status = v[0]; rst_ord_n = v[1]; rst_ord_sort_cycle = v[2];
      } else {
        rst_ord_info_bad = true;
      }
    } else if (name == "cloud_ord") {
      ++rst_ord_seen[1];
      if (own_section) {rst_ord_p = d; rst_ord_bytes = static_cast<std::size_t>(n);}
    } else if (name == "cloud_ord_tag") {
      ++rst_ord_seen[2];
      if (own_section) {rst_ord_tag_p = d; rst_ord_tag_bytes = static_cast<std::size_t>(n);}
    } else if (name == "coarsen_n") {
      std::uint64_t v[4]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      cloud_coarsen_events = v[0]; cloud_coarsen_in = v[1]; cloud_coarsen_reject = v[2];  // retired-compat
      cloud_far_release = v[3];
    } else if (const std::size_t rsize = RetiredRestartRecord(name)) {   // retired-compat
      want(rsize);   // (as the code before the retirement, then dropped)
    } else if (name == "amr_regrids") {
      std::int32_t v; want(sizeof(v)); std::memcpy(&v, d, sizeof(v)); amr_regrids = v;
    } else if (name == "regrid_evt") {
      std::int64_t v[6]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      amr_event_pending = (v[0] != 0);
      amr_event_nnew = static_cast<int>(v[1]); amr_event_ndel = static_cast<int>(v[2]);
      amr_event_regrid = static_cast<int>(v[3]);
      amr_event_handled = v[4]; amr_event_merged = v[5];
    } else if (name == "regrid_evt_t") {
      double v[2]; want(sizeof(v)); std::memcpy(v, d, sizeof(v));
      amr_event_first_time = v[0]; amr_event_first_ncycle = static_cast<long long>(v[1]);
    } else if (name == "x3_ledger") {
      if (n < static_cast<std::int32_t>(5*sizeof(double)) || n % sizeof(double) != 0) {
        Broken("dust record x3_ledger has an unexpected size");
      }
      std::vector<double> led(n/sizeof(double));
      std::memcpy(led.data(), d, n);
      const int nd = static_cast<int>(led[4]);
      if (static_cast<std::size_t>(5 + nd) != led.size()) Broken("dust record x3_ledger is inconsistent");
      x3_outflow_mass = led[0];
      x3_outflow_count = static_cast<long long>(led[1] + 0.5);
      const bool r0 = (global_variable::my_rank == 0);
      auto *pb = pmy_pack->ppart->pbval_part;
      pb->x3_removed.h_view(0) = r0 ? led[2] : 0.0;
      pb->x3_removed.h_view(1) = r0 ? led[3] : 0.0;
      pb->x3_removed.template modify<HostMemSpace>();
      pb->x3_removed.template sync<DevExeSpace>();
      pbval_dm->dropped_sum.assign(nd, 0.0);
      if (r0) for (int v=0; v<nd; ++v) pbval_dm->dropped_sum[v] = led[5 + v];
    } else if (global_variable::my_rank == 0) {
      std::cout << "# WARNING (dust restart): record '" << name << "' is not known to this code "
                << "and is ignored" << std::endl;
    }
    at += 16 + sizeof(n) + static_cast<std::size_t>(n);
  }
  if (!species_record_found && global_variable::my_rank == 0) {
    std::cout << "# NOTE (dust restart): the file has no record species_table (written by "
              << "an older version): the species of the input are not checked against it" << std::endl;
  }
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
  if (pmbp->pdust != nullptr) {
    auto *pd = pmbp->pdust;
    pd->PackRestartState(state);   // the required records
    std::size_t limit = kDustStateMaxBytes;
    if (pd->test_cloud_ord_budget >= 0) {
      limit = std::min<std::size_t>(limit, static_cast<std::size_t>(pd->test_cloud_ord_budget));
    }
    if (state.size() > kDustStateMaxBytes) {
      Broken("the dust records of rank " + std::to_string(global_variable::my_rank) + " (" +
             std::to_string(state.size()) + " bytes) exceed the int32 length of the dust state");
    }
    pd->PackRestartOrder(state, limit);   // optional, within the limit (collective)
  }
  // the dust state length is written as int32 below
  if (state.size() > kDustStateMaxBytes) Broken("internal: dust state above the int32 length");
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
  int nidata = 0;               // int rows of the file (version 1 dust: PTAGH missing)
};
Section ParseSection(const std::vector<char> &buf, const int nsec, const int nrdata, const int nidata) {
  if (buf.size() < kHeadBytes + 8 || std::memcmp(buf.data(), kMagic, 8) != 0 ||
      std::memcmp(buf.data() + buf.size() - 8, kMagic, 8) != 0) {
    Broken("missing or truncated (no head or end marker)");
  }
  std::int32_t head[6];
  std::memcpy(head, buf.data() + 8, sizeof(head));
  // version 2 has this run's columns; version 1 too, or (dust) all but the last one,
  // PTAGH, whose high words are then 0
  const bool columns_ok = (head[5] == nidata) ||
      (head[0] == 1 && nidata == static_cast<int>(PTAGH) + 1 && head[5] == static_cast<int>(PTAGH));
  if (!ParticleRestartVersionKnown(head[0]) || head[1] != static_cast<std::int32_t>(sizeof(Real)) ||
      head[2] != static_cast<std::int32_t>(sizeof(int)) || head[3] != nsec ||
      head[4] != nrdata || !columns_ok) {
    Broken("version, type sizes, rank count or the number of particle columns do not match "
           "this run (file: version " + std::to_string(head[0]) + ", Real " +
           std::to_string(head[1]) + " bytes, int " + std::to_string(head[2]) + " bytes, ranks " +
           std::to_string(head[3]) + ", nrdata " + std::to_string(head[4]) + ", nidata " +
           std::to_string(head[5]) + "; run: nrdata " + std::to_string(nrdata) +
           ", nidata " + std::to_string(nidata) + ")");
  }
  Section s;
  s.nidata = head[5];
  std::size_t at = kHeadBytes;
  std::memcpy(&s.np, buf.data() + at, sizeof(s.np)); at += sizeof(s.np);
  std::memcpy(&s.dtnew, buf.data() + at, sizeof(s.dtnew)); at += sizeof(s.dtnew);
  std::memcpy(&s.nstate, buf.data() + at, sizeof(s.nstate)); at += sizeof(s.nstate);
  const std::size_t need = at + static_cast<std::size_t>(s.nstate) +
      static_cast<std::size_t>(s.np)*(nrdata*sizeof(Real) + s.nidata*sizeof(int)) + 8;
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
      if (c < s.nidata) {
        std::memcpy(irow.data(), ip + static_cast<std::size_t>(c)*s.np*sizeof(int), s.np*sizeof(int));
      } else {   // a column the file does not have (PTAGH of version 1): 0
        std::fill(irow.begin(), irow.end(), 0);
      }
      for (const auto p : sel) ic[c].push_back(irow[p]);
    }
    nread += s.np;
  };
  if (exact) {
    const Section s = ParseSection(own, nsec, nrdata, nidata);
    if (pmbp->pdust != nullptr) pmbp->pdust->UnpackRestartState(s.state, s.nstate, true);
    dtnew = s.dtnew;
    take(s, true);
    // the index points into `own`, which lives until the end of this function
    if (pmbp->pdust != nullptr) pmbp->pdust->ApplyRestartOrder(true);
  } else {
    std::vector<char> buf;
    for (int k=0; k<nsec; ++k) {
      if (k == me && !own.empty()) {buf.swap(own);} else {read(k, buf);}
      const Section s = ParseSection(buf, nsec, nrdata, nidata);
      if (k == 0) {
        if (pmbp->pdust != nullptr) pmbp->pdust->UnpackRestartState(s.state, s.nstate, false);
        dtnew = s.dtnew;
      }
      take(s, false);
    }
    // the counters of the sections were per rank: the tag counter starts again (EnsureNextTag,
    // collectively at the first cycle): at the bound of all tags issued before
    // (version 2, record tag_state of section 0, the same in all), version 1 from the largest
    // live tag
    if (pmbp->pdust != nullptr) {
      pmbp->pdust->cloud_next_tag_set = false;
      pmbp->pdust->cloud_next_tag = 0;
      pmbp->pdust->ApplyRestartOrder(false);   // (another layout: the index is rebuilt)
    }
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
  // 2-D refinable clouds never carry the x3 lobe bits: a file whose
  // 2-D cloud codes have them was not written by a 2-D cloud run
  if (clouds && !pm->three_d) {
    for (std::int64_t p=0; p<np; ++p) {
      if (CloudGeometry::CloudLobe3(pi(PLEV,p)) != 0) {
        Broken("a 2-D cloud with x3 lobe bits in its code (" + std::to_string(pi(PLEV,p)) +
               ", particle " + std::to_string(p) + " of this rank)");
      }
    }
  }
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
  pm->nprtcl_total = tot;
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
      std::cout << ", next cloud tag ";
      if (pmbp->pdust->cloud_next_tag_set) {std::cout << pmbp->pdust->cloud_next_tag;}
      else {std::cout << "unset";}
      std::cout << " (tag bound " << pmbp->pdust->cloud_tag_floor << "), merge pass "
                << pmbp->pdust->cloud_merge_pass;
    }
    std::cout << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::PackRestartOrder
//! \brief The optional records of the cell-order index (see the file header), appended to
//! the required ones in b: cloud_ord_n = {status, n, cycle of the last cell-order sort}, then
//! with status stored the n entries of cloud_ord and of cloud_ord_tag (only [0, n) of the
//! arrays).  If they would take the dust state past `limit` bytes, only cloud_ord_n is written
//! (status omitted).  Collective: rank 0 reports the ranks that left the index out.

bool DustGasDrag::PackRestartOrder(std::vector<char> &b, const std::size_t limit) {
  if (cloud_order_interval <= 0) return false;
  const std::size_t n = static_cast<std::size_t>(cloud_ord_n > 0 ? cloud_ord_n : 0);
  std::int32_t status = kCloudOrderNone;
  if (n > 0) status = CloudOrderFits(b.size(), n, limit) ? kCloudOrderStored : kCloudOrderOmitted;
  // the info record tells the reader that the index was left out; it may pass a test limit,
  // never the int32 length (then nothing is written and the reader finds no record)
  if (b.size() + kCloudOrderInfoBytes <= kDustStateMaxBytes) {
    const std::int32_t info[3] = {status, static_cast<std::int32_t>(n), cloud_sort_cycle};
    PutRecord(b, "cloud_ord_n", info, sizeof(info));
  } else {
    status = kCloudOrderOmitted;
  }
  if (status == kCloudOrderStored) {
    auto o = Kokkos::create_mirror_view_and_copy(
        HostMemSpace(), Kokkos::subview(cloud_ord, std::make_pair(std::size_t(0), n)));
    auto t = Kokkos::create_mirror_view_and_copy(
        HostMemSpace(), Kokkos::subview(cloud_ord_tag, std::make_pair(std::size_t(0), n)));
    PutRecord(b, "cloud_ord", o.data(), n*sizeof(int));
    PutRecord(b, "cloud_ord_tag", t.data(), n*sizeof(std::uint64_t));
  }
  int omitted = (status == kCloudOrderOmitted) ? 1 : 0;
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, &omitted, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  if (omitted > 0 && global_variable::my_rank == 0) {
    std::cout << "# dust restart file: the cell-order index of the cloud kernels left out on "
              << omitted << " rank(s) (the dust state would pass "
              << limit << " bytes); a restart from this file rebuilds it there at its first "
              << "cycle (sums in another order until the next rebuild)" << std::endl;
  }
  return status == kCloudOrderOmitted;
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::ApplyRestartOrder
//! \brief After the particles of the file are taken (UnpackParticleRestart, collective): the
//! cell-order index of the records back when this rank took its own section (exact: the
//! storage order of the writer) and the records hold a permutation of [0, n) with its n tags.
//! The index may be longer or shorter than the particles and its tags may be stale, as in the
//! run between two rebuilds (the kernels and UpdateCloudOrder handle that).  Otherwise the
//! index is dropped (cloud_ord_n = 0, cloud_sort_cycle = -1) and built at the first cycle.
//! Rank 0 reports how many ranks restored, lacked (older file), had it left out by the
//! writer, had an invalid record (discarded) or another layout.

void DustGasDrag::ApplyRestartOrder(const bool exact) {
  enum {kRestored = 0, kAbsent, kOmitted, kInvalid, kRelayout, kCount};
  int what = kAbsent;
  std::string why;
  const std::int32_t n = rst_ord_n;
  const int nseen = rst_ord_seen[0] + rst_ord_seen[1] + rst_ord_seen[2];
  const bool arrays = rst_ord_seen[1] > 0 || rst_ord_seen[2] > 0;
  const int ncycle = pmy_pack->pmesh->ncycle;   // (the cycle of the checkpoint)
  bool stored = false;   // status stored, group consistent: check the entries
  if (cloud_order_interval <= 0) {
    what = -1;   // the index is off in this run: nothing to report
  } else if (!exact) {
    what = (nseen > 0) ? kRelayout : kAbsent;
  } else if (nseen == 0) {
    what = kAbsent;   // a file without the records (written before them)
  } else if (rst_ord_seen[0] > 1 || rst_ord_seen[1] > 1 || rst_ord_seen[2] > 1) {
    what = kInvalid;
    why = "records repeated (cloud_ord_n " + std::to_string(rst_ord_seen[0]) + "x, cloud_ord " +
          std::to_string(rst_ord_seen[1]) + "x, cloud_ord_tag " + std::to_string(rst_ord_seen[2]) +
          "x)";
  } else if (rst_ord_seen[0] == 0) {
    what = kInvalid; why = "cloud_ord / cloud_ord_tag without record cloud_ord_n";
  } else if (rst_ord_info_bad) {
    // (also the 8-byte head of an unreleased development version: its files fall back here)
    what = kInvalid; why = "record cloud_ord_n has an unexpected size";
  } else if (rst_ord_sort_cycle < -1 || rst_ord_sort_cycle > ncycle) {
    what = kInvalid; why = "sort cycle " + std::to_string(rst_ord_sort_cycle) + " outside [-1, " +
                           std::to_string(ncycle) + "]";
  } else if (rst_ord_status == kCloudOrderOmitted) {
    what = arrays ? kInvalid : kOmitted;
    if (arrays) why = "status omitted with cloud_ord / cloud_ord_tag";
  } else if (rst_ord_status == kCloudOrderNone) {
    what = (n == 0 && !arrays) ? kRestored : kInvalid;
    if (what == kInvalid) why = "status none with n = " + std::to_string(n) + " or entries";
  } else if (rst_ord_status != kCloudOrderStored || n <= 0) {
    what = kInvalid; why = "unknown status " + std::to_string(rst_ord_status) + " or n = " +
                           std::to_string(n);
  } else if (rst_ord_seen[1] != 1 || rst_ord_seen[2] != 1 ||
             rst_ord_bytes != static_cast<std::size_t>(n)*sizeof(int) ||
             rst_ord_tag_bytes != static_cast<std::size_t>(n)*sizeof(std::uint64_t)) {
    what = kInvalid; why = "cloud_ord / cloud_ord_tag missing or not of n = " + std::to_string(n) +
                           " entries";
  } else {
    stored = true;
  }
  if (stored) {
    // the entries straight into the host mirror of [0, n) of the new index, checked there to be
    // a permutation of [0, n) (every entry in range, none twice), then copied to the device
    cloud_ord = DvceArray1D<int>();
    cloud_ord_tag = DvceArray1D<std::uint64_t>();
    cloud_ord = DvceArray1D<int>(Kokkos::view_alloc(Kokkos::WithoutInitializing, "cloud_ord"),
                                 n + n/4);
    const auto range = std::make_pair(0, static_cast<int>(n));
    auto od = Kokkos::subview(cloud_ord, range);
    auto oh = Kokkos::create_mirror_view(od);
    std::memcpy(oh.data(), rst_ord_p, rst_ord_bytes);
    std::vector<bool> seen(static_cast<std::size_t>(n), false);
    what = kRestored;
    for (std::int32_t k=0; k<n; ++k) {
      const int p = oh(k);
      if (p < 0 || p >= n || seen[p]) {
        what = kInvalid;
        why = "entry " + std::to_string(k) + " = " + std::to_string(p) +
              ((p < 0 || p >= n) ? " outside [0, n)" : " repeated") +
              " (n = " + std::to_string(n) + ")";
        break;
      }
      seen[p] = true;
    }
    if (what == kRestored) {
      cloud_ord_tag = DvceArray1D<std::uint64_t>(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                                                    "cloud_ord_tag"), n + n/4);
      auto td = Kokkos::subview(cloud_ord_tag, range);
      auto th = Kokkos::create_mirror_view(td);
      std::memcpy(th.data(), rst_ord_tag_p, rst_ord_tag_bytes);
      Kokkos::deep_copy(od, oh);
      Kokkos::deep_copy(td, th);
    } else {
      cloud_ord = DvceArray1D<int>();   // (on a host build the mirror was the index itself)
    }
  }
  if (what == kRestored) {
    cloud_ord_n = n;
    cloud_sort_cycle = rst_ord_sort_cycle;
  } else {
    cloud_ord_n = 0;
    cloud_sort_cycle = -1;
  }
  if (what == kInvalid) {
    std::cout << "# WARNING (dust restart, rank " << global_variable::my_rank << "): the "
              << "cell-order index of the file is discarded (" << why << "); it is rebuilt at "
              << "the first cycle" << std::endl;
  }
  int count[kCount] = {0, 0, 0, 0, 0};
  if (what >= 0) count[what] = 1;
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    MPI_Allreduce(MPI_IN_PLACE, count, kCount, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  if (global_variable::my_rank == 0 && cloud_order_interval > 0) {
    std::cout << "# dust restart: cell-order index of the cloud kernels restored on "
              << count[kRestored] << " rank(s); rebuilt at the first cycle on: " << count[kAbsent]
              << " (file without the records), " << count[kOmitted] << " (left out by the "
              << "writer), " << count[kInvalid] << " (invalid records, discarded), "
              << count[kRelayout] << " (other ranks or blocks)" << std::endl;
  }
  rst_ord_info = rst_ord_info_bad = false;
  rst_ord_seen[0] = rst_ord_seen[1] = rst_ord_seen[2] = 0;
  rst_ord_status = 0; rst_ord_n = 0; rst_ord_sort_cycle = -1;
  rst_ord_p = rst_ord_tag_p = nullptr;
  rst_ord_bytes = rst_ord_tag_bytes = 0;
}

}  // namespace dust
