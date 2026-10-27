//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pbin.cpp
//! \brief The particle science file (file_type = pbin): the active dust
//! particles of one output time as typed columns, with everything needed to read them alone.
//!
//!   <output2>
//!   file_type = pbin        # no variable key: always the complete particle science state
//!   dt        = 6.283185307179586
//!   id        = particles   # optional, the default
//!
//! writes pbin/<basename>.<id>.<NNNNN>.pbin.  Format version 2 (vis/python/pbin.py reads 1 and 2):
//!
//!   preamble, 64 bytes
//!     char[8] "AKPBIN01"; uint32 0x01020304 (byte-order mark); uint32 format version (2);
//!     uint64 text_bytes; uint64 data_start (multiple of 64); uint64 data_bytes;
//!     uint64 n_particles; uint64 n_blocks; uint64 0 (reserved)
//!   text header, text_bytes of UTF-8 text (ASCII except in input comments): "key = value"
//!     lines in [sections]; [columns] lists
//!     every column as "column = <table> <name> <dtype> <count> <offset> <nbytes>" (numpy
//!     dtype, offset from data_start); [constants] (version 2) lists the particle quantities
//!     stored once instead of as a column, "constant = <table> <name> <dtype> <value>"; the
//!     last section, [input], is the complete parameter dump of the run, as in the mesh .bin
//!     files
//!   zero padding up to data_start
//!   data section, data_bytes: the columns one after the other, each 64-byte aligned; in a
//!     particle column the rows of rank 0 come first, then those of rank 1, ...; the block
//!     columns are in gid order
//!   footer, 16 bytes: char[8] "AKPBEND1"; uint64 size of the whole file
//!
//! Only the active rows are written: the capacity tail of the particle arrays and the
//! zero-mass rows (outflow tombstones, removed at the start of the next cycle) are left out;
//! the restart-only columns (RK registers, drag rates) are not part of the science state.  The
//! 64-bit tag (PTAG low word, PTAGH high word) is written as uint64.  Version 2:
//! with one species and species_fixed stopping times the species and
//! stopping_time columns are constants of the header, provided every active row holds exactly
//! species 0 and tau_s of species 0 (checked; otherwise both stay columns); with several
//! species or per-particle stopping times both are always columns.  Every rank
//! writes its own rows at a global prefix offset with independent MPI-IO writes of at most
//! 1 GiB (no gather on rank 0; ranks without particles write nothing; all offsets 64-bit and
//! checked).  The file is written as <name>.tmp and renamed when it is complete.  Writing reads
//! the particle arrays only: the run state is not changed.

#include <sys/stat.h>  // mkdir

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "outputs.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace {

constexpr char kMagic[9] = "AKPBIN01";
constexpr char kEnd[9] = "AKPBEND1";
constexpr std::uint32_t kVersion = 2;   // 1: no [constants] (2026-10-27, first files)
constexpr std::uint64_t kAlign = 64;
constexpr std::uint64_t kPreamble = 64;
constexpr std::uint64_t kFooter = 16;
constexpr std::uint64_t kChunk = std::uint64_t(1) << 30;   // bytes per MPI-IO call

[[noreturn]] void PbinFail(const std::string &message) {
  std::cerr << "### FATAL ERROR in pbin output: " << message << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}

// checked 64-bit arithmetic (offsets must also fit a signed MPI_Offset)
std::uint64_t Add(const std::uint64_t a, const std::uint64_t b) {
  const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (a > limit || b > limit - a) PbinFail("file offset exceeds the signed 64-bit range");
  return a + b;
}
std::uint64_t Mul(const std::uint64_t a, const std::uint64_t b) {
  const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (a != 0 && b > limit/a) PbinFail("column size exceeds the signed 64-bit range");
  return a*b;
}
std::uint64_t Align(const std::uint64_t a) {
  return Add(a, (kAlign - a % kAlign) % kAlign);
}

void WriteAt(IOWrapper &file, const void *data, const std::uint64_t bytes,
             const std::uint64_t offset) {
  Add(offset, bytes);
  const char *p = static_cast<const char*>(data);
  for (std::uint64_t done = 0; done < bytes;) {
    const std::uint64_t n = std::min(kChunk, bytes - done);
    if (file.Write_any_type_at(p + done, n, offset + done, "byte") != n) {
      PbinFail("write failed (" + std::to_string(n) + " bytes at offset " +
               std::to_string(offset + done) + ")");
    }
    done += n;
  }
}

bool LittleEndian() {
  const std::uint32_t one = 1;
  unsigned char c;
  std::memcpy(&c, &one, 1);
  return c == 1;
}

// FNV-1a (64 bit) over (level - root_level, lx1, lx2, lx3) of every MeshBlock in gid order, as
// little-endian int64: the same blocks in the same order (as in a mesh .bin file) give the
// same value (vis/python/pbin.py: topology_hash)
std::uint64_t TopologyHash(const Mesh *pm) {
  std::uint64_t h = 1469598103934665603ULL;
  for (int g=0; g<pm->nmb_total; ++g) {
    const LogicalLocation &ll = pm->lloc_eachmb[g];
    const std::int64_t v[4] = {ll.level - pm->root_level, ll.lx1, ll.lx2, ll.lx3};
    for (int q=0; q<4; ++q) {
      for (int b=0; b<8; ++b) {
        h ^= static_cast<std::uint64_t>((static_cast<std::uint64_t>(v[q]) >> (8*b)) & 0xffu);
        h *= 1099511628211ULL;
      }
    }
  }
  return h;
}

struct Column {
  std::string table, name, dtype, meaning;
  std::uint64_t itemsize = 0, count = 0, offset = 0, bytes = 0;
};

}  // namespace

//----------------------------------------------------------------------------------------
// ctor: the dust module is required (the columns are the dust particle state)

ParticleBinaryOutput::ParticleBinaryOutput(ParameterInput *pin, Mesh *pm,
                                           OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  if (pm->pmb_pack->ppart == nullptr || pm->pmb_pack->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Output block '" << op.block_name << "': file_type = pbin needs dust "
              << "particles (<particles> and <dust> blocks)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pm->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Output block '" << op.block_name << "': file_type = pbin needs a 2-D or "
              << "3-D mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  mkdir("pbin", 0775);
}

//----------------------------------------------------------------------------------------
//! \fn ParticleBinaryOutput::LoadOutputData
//! \brief The active rows of this rank (mass != 0), in array order, and the counts of rows
//! left out or suspicious.

void ParticleBinaryOutput::LoadOutputData(Mesh *pm) {
  particles::Particles *pp = pm->pmb_pack->ppart;
  const int np = pp->nprtcl_thispack;
  auto pr = pp->prtcl_rdata;
  auto pi = pp->prtcl_idata;
  if (active_.extent_int(0) < std::max(np, 1)) {
    active_ = DvceArray1D<int>("pbin_active", std::max(np, 1));
  }
  auto idx = active_;
  int nact = 0;
  Kokkos::parallel_scan("pbin_active_rows", Kokkos::RangePolicy<DevExeSpace>(0, np),
  KOKKOS_LAMBDA(const int p, int &upd, const bool final) {
    if (pr(IPM,p) != 0.0) {
      if (final) idx(upd) = p;
      ++upd;
    }
  }, nact);
  nactive_ = nact;
  long long nzero = 0, nbad = 0, nneg = 0;
  Kokkos::parallel_reduce("pbin_row_checks", Kokkos::RangePolicy<DevExeSpace>(0, np),
  KOKKOS_LAMBDA(const int p, long long &z, long long &b, long long &t) {
    const Real m = pr(IPM,p);
    if (m == 0.0) {++z; return;}
    if (!(m > 0.0) || !Kokkos::isfinite(m)) ++b;
    if (pi(PTAGH,p) < 0) ++t;   // a tag >= 2^63: never issued
  }, Kokkos::Sum<long long>(nzero), Kokkos::Sum<long long>(nbad), Kokkos::Sum<long long>(nneg));
  nzero_ = nzero;
  nbad_mass_ = nbad;
  nneg_tag_ = nneg;
}

//----------------------------------------------------------------------------------------
//! \fn ParticleBinaryOutput::WriteOutputFile
//! \brief See the file header.

void ParticleBinaryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  const int nranks = global_variable::nranks, me = global_variable::my_rank;
  particles::Particles *pp = pm->pmb_pack->ppart;
  dust::DustGasDrag *pd = pm->pmb_pack->pdust;
  const bool clouds = (pd->smr_transfer_mode == DustSMRTransfer::refinable_cloud);
  const bool little = LittleEndian();
  const std::string bo = little ? "<" : ">";

  // ---- global counts and this rank's offsets
  std::vector<std::uint64_t> counts(nranks, 0);
  counts[me] = static_cast<std::uint64_t>(nactive_);
  long long anomalies[4] = {nzero_, nbad_mass_, nneg_tag_, 0};
  // refinable clouds: PLEV bits outside the level and the three lobes (none expected)
  const int nidx = nactive_;
  auto idx = active_;
  auto pr = pp->prtcl_rdata;
  auto pi = pp->prtcl_idata;
  if (clouds) {
    using T = SMRCompositeTransfer;
    const int known = T::kCloudLevelMask | (3 << T::kCloudLobeShift1) |
                      (3 << T::kCloudLobeShift2) | (3 << T::kCloudLobeShift3);
    long long nunknown = 0;
    Kokkos::parallel_reduce("pbin_shape_check", Kokkos::RangePolicy<DevExeSpace>(0, nidx),
    KOKKOS_LAMBDA(const int q, long long &u) {
      if (pi(PLEV, idx(q)) & ~known) ++u;
    }, Kokkos::Sum<long long>(nunknown));
    anomalies[3] = nunknown;
  }
#if MPI_PARALLEL_ENABLED
  {
    std::uint64_t mine = counts[me];
    MPI_Allgather(&mine, 1, MPI_UINT64_T, counts.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, anomalies, 4, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  }
#endif
  std::uint64_t ntotal = 0, prefix = 0;
  for (int r=0; r<nranks; ++r) {
    if (r == me) prefix = ntotal;
    ntotal = Add(ntotal, counts[r]);
  }
  // version 2: one species with species_fixed stopping times stores species and stopping time
  // once in the header, if every active row holds exactly those values (else as columns)
  auto taus_h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), pd->taus.d_view);
  bool constant_sp_ts = (pd->nspecies == 1 &&
                         pd->stopping_time_mode == DustStoppingTimeMode::species_fixed &&
                         taus_h.extent_int(0) >= 1);
  long long ndiffer = 0;
  const Real ts0 = (taus_h.extent_int(0) >= 1) ? taus_h(0) : Real(0);
  if (constant_sp_ts) {
    Kokkos::parallel_reduce("pbin_constant_check", Kokkos::RangePolicy<DevExeSpace>(0, nidx),
    KOKKOS_LAMBDA(const int q, long long &d) {
      const int p = idx(q);
      if (pi(PSP,p) != 0 || pr(IPTS,p) != ts0) ++d;
    }, Kokkos::Sum<long long>(ndiffer));
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE, &ndiffer, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (ndiffer > 0) constant_sp_ts = false;
  }
  const std::uint64_t nblocks = static_cast<std::uint64_t>(pm->nmb_total);
  const std::uint64_t bprefix = static_cast<std::uint64_t>(pm->gids_eachrank[me]);
  const int nmb = pm->pmb_pack->nmb_thispack;

  // ---- the columns and their place in the data section
  std::vector<Column> cols;
  auto col = [&](const char *table, const char *name, const char *dt, std::uint64_t size,
                 const std::string &meaning) {
    Column c;
    c.table = table; c.name = name; c.dtype = bo + dt; c.itemsize = size; c.meaning = meaning;
    c.count = (c.table == "particles") ? ntotal : nblocks;
    cols.push_back(c);
  };
  const bool sbox = pd->is_shearing_box;
  col("particles", "id", "u8", 8, "numerical sample identity: the 64-bit tag; "
      "split / merge / resampling create and remove samples, an id is not a grain lineage");
  const char *species_meaning = "species_id (0-based; input index N = species_id + 1)";
  const char *ts_meaning = "stopping time t_s (code time)";
  if (!constant_sp_ts) col("particles", "species", "u4", 4, species_meaning);
  col("particles", "x1", "f8", 8, "position x1 (code length)");
  col("particles", "x2", "f8", 8, "position x2 (code length)");
  col("particles", "x3", "f8", 8, pm->three_d ? "position x3 (code length)" :
      "x3 column of a 2-D run (not a mesh direction; as stored)");
  col("particles", "v1", "f8", 8, "velocity component 1 (see [frame])");
  col("particles", "v2", "f8", 8, "velocity component 2 (see [frame])");
  col("particles", "v3", "f8", 8, "velocity component 3 (see [frame])");
  col("particles", "mass", "f8", 8, "macro-particle mass (code units; not a grain mass)");
  if (!constant_sp_ts) col("particles", "stopping_time", "f8", 8, ts_meaning);
  if (clouds) {
    col("particles", "shape", "u2", 2, "cloud shape code v1: bits 0-3 cloud level, bits 4-5 "
        "lobe along x1, 6-7 lobe along x2, 8-9 lobe along x3 (0 none, 1 lower, 2 upper, "
        "3 both)");
  }
  col("blocks", "gid", "i8", 8, "MeshBlock global id");
  col("blocks", "rank", "i4", 4, "MPI rank that wrote the block");
  col("blocks", "level", "i4", 4, "refinement level above the root level");
  col("blocks", "lx1", "i8", 8, "logical location x1 (at the block's level)");
  col("blocks", "lx2", "i8", 8, "logical location x2");
  col("blocks", "lx3", "i8", 8, "logical location x3");
  col("blocks", "x1min", "f8", 8, "block box x1min");
  col("blocks", "x1max", "f8", 8, "block box x1max");
  col("blocks", "x2min", "f8", 8, "block box x2min");
  col("blocks", "x2max", "f8", 8, "block box x2max");
  col("blocks", "x3min", "f8", 8, "block box x3min");
  col("blocks", "x3max", "f8", 8, "block box x3max");
  std::uint64_t at = 0;
  for (auto &c : cols) {
    c.offset = at;
    c.bytes = Mul(c.count, c.itemsize);
    at = Align(Add(at, c.bytes));
  }
  const std::uint64_t data_bytes = at;

  // ---- the text header (rank 0) and the data offset (all ranks)
  std::string text;
  std::uint64_t data_start = 0;
  if (me == 0) {
    std::ostringstream h;
    h << std::setprecision(17);
    h << "AthenaK particle science file (pbin) format_version = " << kVersion << "\n";
    h << "[file]\n";
    h << "format = AKPBIN\nformat_version = " << kVersion << "\n";
    h << "byte_order = " << (little ? "little" : "big") << "\n";
    h << "n_particles = " << ntotal << "\nn_blocks = " << nblocks << "\n";
    h << "written_by_ranks = " << nranks << "\n";
    h << "completeness = footer AKPBEND1 + file size; written as .tmp, renamed when complete\n";
    h << "[snapshot]\n";
    h << "time = " << pm->time << "\ncycle = " << pm->ncycle << "\n";
    h << "dt = " << pm->dt << "\n";
    h << "basename = " << out_params.file_basename << "\n";
    h << "output_block = " << out_params.block_name << "\n";
    h << "file_number = " << out_params.file_number << "\n";
    h << "source_commit = " << ATHENA_SOURCE_COMMIT << "\n";
    h << "mesh_update_seq = " << pm->GetAMRLoadBalanceUpdateSeq() << "\n";
    {
      std::ostringstream hx;
      hx << "0x" << std::hex << std::setw(16) << std::setfill('0') << TopologyHash(pm);
      h << "block_topology_hash = " << hx.str() << "  # FNV-1a 64 of (level, lx1, lx2, lx3) "
        << "as int64 for every block in gid order\n";
    }
    h << "output_point = end of a cycle, before that cycle's mesh refinement check (the first "
      << "output: after the problem generator; the last: after the final check)\n";
    h << "[mesh]\n";
    h << "ndim = " << (pm->three_d ? 3 : 2) << "\n";
    h << "nx1 = " << pm->mesh_indcs.nx1 << "\nnx2 = " << pm->mesh_indcs.nx2
      << "\nnx3 = " << pm->mesh_indcs.nx3 << "\n";
    h << "x1min = " << pm->mesh_size.x1min << "\nx1max = " << pm->mesh_size.x1max << "\n";
    h << "x2min = " << pm->mesh_size.x2min << "\nx2max = " << pm->mesh_size.x2max << "\n";
    h << "x3min = " << pm->mesh_size.x3min << "\nx3max = " << pm->mesh_size.x3max << "\n";
    h << "meshblock_nx1 = " << pm->mb_indcs.nx1 << "\nmeshblock_nx2 = " << pm->mb_indcs.nx2
      << "\nmeshblock_nx3 = " << pm->mb_indcs.nx3 << "\n";
    h << "levels = " << (pm->max_level - pm->root_level) << "  # finest level above the root\n";
    h << "multilevel = " << (pm->multilevel ? 1 : 0) << "\nadaptive = "
      << (pm->adaptive ? 1 : 0) << "\n";
    const char *face[6] = {"ix1", "ox1", "ix2", "ox2", "ix3", "ox3"};
    for (int f=0; f<6; ++f) {
      h << "bc_" << face[f] << " = " << pm->GetBoundaryString(pm->mesh_bcs[f]) << "\n";
    }
    h << "[particles]\n";
    h << "rows = active rows only (the capacity tail and zero-mass rows are left out)\n";
    h << "row_order = by writing rank, then the rank's array order (unsorted; neither the "
      << "order nor the rank layout is stable between outputs)\n";
    h << "zero_mass_rows_excluded = " << anomalies[0] << "\n";
    h << "invalid_mass_rows = " << anomalies[1] << "  # negative or non-finite mass "
      << "(written)\n";
    h << "negative_tag_rows = " << anomalies[2] << "  # tags >= 2^63 (high word below zero; "
      << "never issued; the key keeps its version-1 name)\n";
    if (clouds) h << "shape_unknown_bit_rows = " << anomalies[3] << "\n";
    h << "owner = not stored: the MeshBlock whose half-open box [min, max) holds the position "
      << "(blocks table)\n";
    if (constant_sp_ts) {
      h << "constant_columns = species stopping_time  # one species, species_fixed: one value "
        << "for every row, in [constants]\n";
    } else if (ndiffer > 0) {
      h << "constant_columns = none  # one species, species_fixed, but " << ndiffer
        << " rows differ from species 0 / its tau_s: kept as columns\n";
    } else {
      h << "constant_columns = none\n";
    }
    h << "[species]\n";
    {
      const int ns = pd->nspecies;
      h << "nspecies = " << ns << "\n";
      const char *mode = (pd->stopping_time_mode == DustStoppingTimeMode::species_fixed) ?
          "species_fixed" : (pd->stopping_time_mode == DustStoppingTimeMode::particle_static) ?
          "particle_static" : "dynamic";
      h << "stopping_time_mode = " << mode << "  # stopping_time (column or constant) is the "
        << "value each particle uses\n";
      h << "dust_to_gas = " << pd->dust_to_gas << "  # the <dust>/dust_to_gas key only; a "
        << "problem generator may set its own abundances (e.g. <problem>/eps_N, see [input])\n";
      h << "back_reaction = " << (pd->back_reaction ? 1 : 0) << "\n";
      auto &taus = taus_h;
      for (int s=0; s<ns; ++s) {
        h << "species." << s << " = input_index " << (s + 1) << ", tau_s "
          << ((s < taus.extent_int(0)) ? taus(s) : std::nan("")) << "\n";
      }
      if (!pd->cloud_merge_mroot.empty()) {
        for (std::size_t s=0; s<pd->cloud_merge_mroot.size(); ++s) {
          h << "cloud_root_mass." << s << " = " << pd->cloud_merge_mroot[s]
            << "  # root-level cloud mass of the merge pass (G/R reference)\n";
        }
      }
    }
    h << "[representation]\n";
    pd->ScienceRepresentation(h);
    h << "[frame]\n";
    h << "positions = lab frame, code length\n";
    if (sbox) {
      h << "velocity_frame = shear_relative  # velocities relative to the background shear "
        << "-q Omega x1\n";
      h << "shear_q = " << pd->qshear << "\nshear_omega = " << pd->omega0 << "\n";
      if (pm->three_d) {
        h << "axes = x1 radial, x2 azimuthal, x3 vertical\n";
        h << "azimuthal_velocity = v2  # v2_phys = v2 - q*Omega*x1\n";
      } else {
        h << "axes = x1 radial, x2 vertical (2-D r-z shearing box), azimuth not a mesh "
          << "direction\n";
        h << "azimuthal_velocity = v3  # v3_phys = v3 - q*Omega*x1; v2 is vertical\n";
      }
    } else {
      h << "velocity_frame = lab\n";
    }
    h << "[columns]\n";
    h << "# column = table name dtype count offset bytes (offset from data_start)\n";
    for (const auto &c : cols) {
      h << "column = " << c.table << " " << c.name << " " << c.dtype << " " << c.count << " "
        << c.offset << " " << c.bytes << "\n";
    }
    h << "[constants]\n";
    h << "# constant = table name dtype value (the same value in every row of the table)\n";
    if (constant_sp_ts) {
      h << "constant = particles species " << bo << "u4 0\n";
      h << "constant = particles stopping_time " << bo << "f8 " << static_cast<double>(ts0)
        << "\n";
    }
    h << "[meaning]\n";
    for (const auto &c : cols) h << c.table << "." << c.name << " = " << c.meaning << "\n";
    if (constant_sp_ts) {
      h << "particles.species = " << species_meaning << "\n";
      h << "particles.stopping_time = " << ts_meaning << "\n";
    }
    h << "[input]\n";
    std::ostringstream ost;
    pin->ParameterDump(ost);
    h << ost.str();
    text = h.str();
    data_start = Align(Add(kPreamble, text.size()));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&data_start, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
#endif
  const std::uint64_t file_bytes = Add(Add(data_start, data_bytes), kFooter);

  // ---- file name; written as .tmp, renamed when complete
  char number[16];
  std::snprintf(number, sizeof(number), "%05d", out_params.file_number);
  const std::string fname = std::string("pbin/") + out_params.file_basename + "." +
                            out_params.file_id + "." + number + ".pbin";
  const std::string tmpname = fname + ".tmp";
  IOWrapper file;
  file.Open(tmpname.c_str(), IOWrapper::FileMode::write);

  if (me == 0) {
    std::vector<char> head(static_cast<std::size_t>(data_start), 0);
    std::memcpy(head.data(), kMagic, 8);
    const std::uint32_t mark = 0x01020304u, version = kVersion;
    std::memcpy(head.data() + 8, &mark, 4);
    std::memcpy(head.data() + 12, &version, 4);
    const std::uint64_t pre[6] = {static_cast<std::uint64_t>(text.size()), data_start,
                                  data_bytes, ntotal, nblocks, 0};
    std::memcpy(head.data() + 16, pre, sizeof(pre));
    std::memcpy(head.data() + kPreamble, text.data(), text.size());
    WriteAt(file, head.data(), head.size(), 0);
  }

  // ---- particle columns: this rank's active rows, packed on the device
  for (const auto &c : cols) {
    if (c.table != "particles") continue;
    const std::uint64_t off = Add(Add(data_start, c.offset), Mul(prefix, c.itemsize));
    const std::uint64_t nbytes = Mul(static_cast<std::uint64_t>(nidx), c.itemsize);
    if (nidx == 0) continue;
    if (c.name == "id") {
      DvceArray1D<std::uint64_t> d("pbin_id", nidx);
      Kokkos::parallel_for("pbin_pack_id", Kokkos::RangePolicy<DevExeSpace>(0, nidx),
      KOKKOS_LAMBDA(const int q) {
        d(q) = GetTag(pi, idx(q));
      });
      auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d);
      WriteAt(file, h.data(), nbytes, off);
    } else if (c.name == "species") {
      DvceArray1D<std::uint32_t> d("pbin_species", nidx);
      Kokkos::parallel_for("pbin_pack_species", Kokkos::RangePolicy<DevExeSpace>(0, nidx),
      KOKKOS_LAMBDA(const int q) {
        d(q) = static_cast<std::uint32_t>(pi(PSP, idx(q)));
      });
      auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d);
      WriteAt(file, h.data(), nbytes, off);
    } else if (c.name == "shape") {
      using T = SMRCompositeTransfer;
      DvceArray1D<std::uint16_t> d("pbin_shape", nidx);
      Kokkos::parallel_for("pbin_pack_shape", Kokkos::RangePolicy<DevExeSpace>(0, nidx),
      KOKKOS_LAMBDA(const int q) {
        const int code = pi(PLEV, idx(q));
        d(q) = static_cast<std::uint16_t>(T::CloudLevel(code) | (T::CloudLobe1(code) << 4) |
                                          (T::CloudLobe2(code) << 6) |
                                          (T::CloudLobe3(code) << 8));
      });
      auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d);
      WriteAt(file, h.data(), nbytes, off);
    } else {
      int rc = -1;
      if (c.name == "x1") rc = IPX;
      if (c.name == "x2") rc = IPY;
      if (c.name == "x3") rc = IPZ;
      if (c.name == "v1") rc = IPVX;
      if (c.name == "v2") rc = IPVY;
      if (c.name == "v3") rc = IPVZ;
      if (c.name == "mass") rc = IPM;
      if (c.name == "stopping_time") rc = IPTS;
      if (rc < 0 || rc >= pp->nrdata) PbinFail("internal: column " + c.name);
      DvceArray1D<double> d("pbin_real", nidx);
      Kokkos::parallel_for("pbin_pack_real", Kokkos::RangePolicy<DevExeSpace>(0, nidx),
      KOKKOS_LAMBDA(const int q) {
        d(q) = static_cast<double>(pr(rc, idx(q)));
      });
      auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), d);
      WriteAt(file, h.data(), nbytes, off);
    }
  }

  // ---- block columns: this rank's blocks at their gids
  if (nmb > 0) {
    auto &size = pm->pmb_pack->pmb->mb_size;
    std::vector<std::int64_t> i8(nmb);
    std::vector<std::int32_t> i4(nmb);
    std::vector<double> f8(nmb);
    for (const auto &c : cols) {
      if (c.table != "blocks") continue;
      const std::uint64_t off = Add(Add(data_start, c.offset), Mul(bprefix, c.itemsize));
      const std::uint64_t nbytes = Mul(static_cast<std::uint64_t>(nmb), c.itemsize);
      const void *src = nullptr;
      for (int m=0; m<nmb; ++m) {
        const int gid = pm->pmb_pack->gids + m;
        const LogicalLocation &ll = pm->lloc_eachmb[gid];
        const auto &s = size.h_view(m);
        if (c.name == "gid") i8[m] = gid;
        if (c.name == "rank") i4[m] = me;
        if (c.name == "level") i4[m] = ll.level - pm->root_level;
        if (c.name == "lx1") i8[m] = ll.lx1;
        if (c.name == "lx2") i8[m] = ll.lx2;
        if (c.name == "lx3") i8[m] = ll.lx3;
        if (c.name == "x1min") f8[m] = s.x1min;
        if (c.name == "x1max") f8[m] = s.x1max;
        if (c.name == "x2min") f8[m] = s.x2min;
        if (c.name == "x2max") f8[m] = s.x2max;
        if (c.name == "x3min") f8[m] = s.x3min;
        if (c.name == "x3max") f8[m] = s.x3max;
      }
      if (c.dtype == bo + "i8") src = i8.data();
      if (c.dtype == bo + "i4") src = i4.data();
      if (c.dtype == bo + "f8") src = f8.data();
      if (src == nullptr) PbinFail("internal: block column " + c.name);
      WriteAt(file, src, nbytes, off);
    }
  }

  if (me == 0) {
    char foot[kFooter];
    std::memcpy(foot, kEnd, 8);
    std::memcpy(foot + 8, &file_bytes, 8);
    WriteAt(file, foot, kFooter, Add(data_start, data_bytes));
  }
  file.Close();
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  if (me == 0) {
    if (std::rename(tmpname.c_str(), fname.c_str()) != 0) {
      std::cout << "### WARNING (pbin): could not rename " << tmpname << " to " << fname
                << std::endl;
    }
    if (anomalies[1] > 0 || anomalies[2] > 0 || anomalies[3] > 0) {
      std::cout << "### WARNING (pbin " << fname << "): " << anomalies[1]
                << " rows with a negative or non-finite mass, " << anomalies[2]
                << " tags >= 2^63, " << anomalies[3] << " unknown shape bits" << std::endl;
    }
  }

  // increment counters
  out_params.file_number++;
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetInteger(out_params.block_name, "file_number", out_params.file_number);
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
