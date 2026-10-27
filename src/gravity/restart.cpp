// AthenaK self-gravity restart metadata. BSD-3-Clause; see LICENSE.
// Framing (native endianness, as in the surrounding native restart): magic[8],
// uint32 version/flags, uint64 bytes/nmb, double time, int64 cycle, uint32
// sizeof(Real)/reserved, nmb*(int32 lx1,lx2,lx3,level,age), magic[8].
#include "gravity/restart.hpp"
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_refinement.hpp"
#include "driver/driver.hpp"
#include "gravity/self_gravity.hpp"
#include "outputs/io_wrapper.hpp"
#include "parameter_input.hpp"
#include "dust/dust.hpp"

namespace gravity {
namespace {
constexpr char kMagic[9] = "AKSGRST1";
constexpr std::uint32_t kVersion = 1;
constexpr std::size_t kFixed = 64, kBlock = 20;
[[noreturn]] void Broken(const std::string &message) {
  std::cerr << "### FATAL ERROR: gravity restart: " << message << '\n';
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
template <typename T>
void Put(std::vector<char> &buffer, T value) {
  const char *p = reinterpret_cast<const char *>(&value);
  buffer.insert(buffer.end(), p, p+sizeof(T));
}
template <typename T>
T Get(const std::vector<char> &buffer, std::size_t &position) {
  if (position > buffer.size() || sizeof(T) > buffer.size()-position) {
    Broken("truncated metadata field");
  }
  T result;
  std::memcpy(&result, buffer.data()+position, sizeof(T));
  position += sizeof(T);
  return result;
}
std::size_t Bytes(const Mesh *pm) {
  if (pm->nmb_total < 1 || static_cast<std::uint64_t>(pm->nmb_total) >
      (std::numeric_limits<std::size_t>::max()-kFixed)/kBlock) {
    Broken("global block count exceeds metadata capacity");
  }
  return kFixed+kBlock*static_cast<std::size_t>(pm->nmb_total);
}
void Guard(const Mesh *pm) {
  if (pm->pmb_pack->pgrav == nullptr) Broken("gravity metadata writer has no gravity module");
}
std::uint64_t Add(std::uint64_t a, std::uint64_t b) {
  const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (a > limit || b > limit-a) Broken("extension offset exceeds signed 64-bit file extent");
  return a+b;
}
void Transfer(IOWrapper &file, char *data, std::uint64_t bytes, std::uint64_t offset,
              bool write) {
  Add(offset, bytes);
  constexpr std::uint64_t chunk = std::uint64_t(1) << 30;
  for (std::uint64_t done = 0; done < bytes;) {
    const auto n = std::min(chunk, bytes-done);
    const auto actual = write ? file.Write_any_type_at(data+done, n, offset+done, "byte", false)
                             : file.Read_bytes_at(data+done, 1, n, offset+done, false);
    if (actual != n) Broken(write ? "extension write failed" : "truncated extension record");
    done += n;
  }
}
}  // namespace

void WriteRestartExtensions(IOWrapper &file, std::uint64_t offset, Mesh *pm,
                            bool single_file_per_rank) {
  if (pm->pmb_pack->ppart == nullptr && pm->pmb_pack->pgrav == nullptr) return;
  if (single_file_per_rank) Broken("particles/self-gravity require a shared restart file");
  if (pm->pmb_pack->ppart != nullptr) {
    std::vector<char> buffer;
    dust::PackParticleRestart(pm->pmb_pack, buffer);
    std::vector<std::uint64_t> lengths(global_variable::nranks, buffer.size());
#if MPI_PARALLEL_ENABLED
    const std::uint64_t mine = buffer.size();
    MPI_Allgather(&mine, 1, MPI_UINT64_T, lengths.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
#endif
    std::uint64_t own = offset, end = offset;
    for (int rank = 0; rank < global_variable::nranks; ++rank) {
      if (rank == global_variable::my_rank) own = end;
      end = Add(end, lengths[rank]);
    }
    Transfer(file, buffer.data(), buffer.size(), own, true);
    offset = end;
  }
  if (pm->pmb_pack->pgrav != nullptr) WriteRestartState(file, offset, pm);
}

void ReadRestartExtensions(IOWrapper &file, std::uint64_t offset, Mesh *pm,
                           ParameterInput *pin, bool single_file_per_rank) {
  // Per-rank fluid-only restarts retain their original layout and behavior.
  if (single_file_per_rank) {
    if (pm->pmb_pack->ppart != nullptr || pm->pmb_pack->pgrav != nullptr) {
      Broken("particles/self-gravity require a shared restart file");
    }
    return;
  }
  const auto extent = file.GetSize(false);
  Add(0, extent);
  if (offset > extent) Broken("native fluid arrays run past EOF");
  auto read = [&](char *data, std::uint64_t bytes, std::uint64_t at) {
    if (at > extent || bytes > extent-at) Broken("extension runs past EOF");
    Transfer(file, data, bytes, at, false);
  };
  std::vector<std::uint64_t> offsets, lengths;
  std::vector<char> global_amr;
  if (extent-offset >= 8) {
    char magic[8];
    read(magic, 8, offset);
    if (std::memcmp(magic, "AKPRST01", 8) == 0) {
      std::vector<char> head(dust::ParticleRestartHeadBytes());
      read(head.data(), head.size(), offset);
      const int sections = dust::ParticleRestartSections(head.data());
      // Every rank, including an empty particle rank, has a fixed head and footer.
      if (sections < 1 || static_cast<std::uint64_t>(sections) >
          (extent-offset)/(head.size()+8)) Broken("invalid particle rank-section count");
      for (int rank = 0; rank < sections; ++rank) {
        read(head.data(), head.size(), offset);
        const auto bytes = dust::ParticleRestartSectionBytes(head.data());
        if (bytes < head.size()+sizeof(std::int64_t)+sizeof(Real)+sizeof(std::int32_t)+8 ||
            bytes > extent-offset || bytes > std::numeric_limits<std::size_t>::max() ||
            dust::ParticleRestartSections(head.data()) != sections) {
          Broken("invalid particle section extent or inconsistent rank count");
        }
        char footer[8];
        read(footer, 8, Add(offset, bytes-8));
        if (std::memcmp(footer, "AKPRST01", 8) != 0) Broken("invalid particle section footer");
        // Validate size arithmetic before the legacy unpacker allocates buffers.
        // Do this for every section, even when the MPI decomposition is unchanged.
        std::int32_t columns[6];
        std::memcpy(columns, head.data()+8, sizeof(columns));
        // (dust_restart.cpp: version 1 int32 tags, version 2 64-bit tags)
        if (!dust::ParticleRestartVersionKnown(columns[0]) || columns[1] != sizeof(Real) ||
            columns[2] != sizeof(int) || columns[4] < 1 || columns[5] < 1) {
          Broken("invalid particle section types");
        }
        std::vector<char> fixed(sizeof(std::int64_t)+sizeof(Real)+sizeof(std::int32_t));
        read(fixed.data(), fixed.size(), Add(offset, head.size()));
        std::int64_t count;
        std::int32_t state_bytes;
        std::memcpy(&count, fixed.data(), sizeof(count));
        std::memcpy(&state_bytes, fixed.data()+sizeof(count)+sizeof(Real), sizeof(state_bytes));
        const auto row_bytes = std::uint64_t(columns[4])*sizeof(Real) +
                               std::uint64_t(columns[5])*sizeof(int);
        const auto available = bytes-head.size()-fixed.size()-8;
        if (count < 0 || state_bytes < 0 || std::uint64_t(state_bytes) > available ||
            std::uint64_t(count) > (available-state_bytes)/row_bytes ||
            std::uint64_t(count)*row_bytes != available-state_bytes) {
          Broken("inconsistent particle/state payload sizes");
        }
        auto record_at = Add(offset, head.size()+fixed.size());
        const auto state_end = Add(record_at, state_bytes);
        while (record_at < state_end) {
          char record[20];
          if (state_end-record_at < sizeof(record)) Broken("truncated dust state record header");
          read(record, sizeof(record), record_at);
          std::int32_t n;
          std::memcpy(&n, record+16, sizeof(n));
          record_at = Add(record_at, sizeof(record));
          if (n < 0 || std::uint64_t(n) > state_end-record_at) Broken("truncated dust state record");
          if (std::memcmp(record, "amr_state", 10) == 0) {
            if (std::uint64_t(n) != (std::uint64_t(pm->nmb_total)+2)*sizeof(std::int32_t)) {
              Broken("dust AMR state block count mismatch");
            }
            std::vector<char> amr(n);
            read(amr.data(), amr.size(), record_at);
            for (std::size_t j = 0; j < amr.size()/sizeof(std::int32_t); ++j) {
              std::int32_t value;
              std::memcpy(&value, amr.data()+j*sizeof(value), sizeof(value));
              if (value < 0 || (j == 0 && value > 1) ||
                  (j == 1 && value != pm->nmb_total)) Broken("invalid dust AMR state");
            }
            if (!global_amr.empty() && amr != global_amr) {
              Broken("particle rank sections disagree on global AMR state");
            }
            global_amr = std::move(amr);
          }
          record_at = Add(record_at, n);
        }
        offsets.push_back(offset);
        lengths.push_back(bytes);
        offset = Add(offset, bytes);
      }
    }
  }
  if (pm->pmb_pack->ppart != nullptr) {
    if (offsets.empty()) Broken("this run has particles but the checkpoint has none");
    auto section = [&](int rank, std::vector<char> &buffer) {
      buffer.resize(static_cast<std::size_t>(lengths[rank]));
      read(buffer.data(), buffer.size(), offsets[rank]);
    };
    dust::UnpackParticleRestart(pm->pmb_pack, offsets.size(), section);
  } else if (!offsets.empty()) {
    Broken("checkpoint contains live particle sections but this run has no particles");
  }
  // Always validate the SG tail, including continuations with gravity disabled.
  ReadRestartState(file, offset, pm, pin, false);
}

void ResumeRestart(Mesh *pm, Driver *driver, ParameterInput *pin) {
  auto *dust = pm->pmb_pack->pdust;
  auto *grav = pm->pmb_pack->pgrav;
  if (dust == nullptr) {
    if (grav != nullptr) grav->ResumeAmrAfterRestart(driver, pin);
    return;
  }
  if (grav != nullptr) {
    const auto &state = grav->RestartState();
    if (state.found && state.adaptive) {
      if (dust->amr_restart_state_found) {
        if (dust->amr_restart_pending != state.pending || dust->amr_restart_ages != state.ages) {
          Broken("dust and gravity AMR replay metadata disagree");
        }
      } else if (dust->experimental_amr) {
        dust->amr_restart_state_found = true;
        dust->amr_restart_pending = state.pending;
        dust->amr_restart_ages = state.ages;
      }
    }
  }
  // This also maps/routs clouds before any pending topology change. The dust
  // route owns replay when active; otherwise gravity handles a hydro AMR replay.
  const bool dust_replays = dust->experimental_amr && pm->adaptive && pm->pmr != nullptr;
  dust->ResumeAmrAfterRestart(driver, pin);
  if (grav != nullptr) {
    if (dust_replays) grav->ClearRestartState();
    else grav->ResumeAmrAfterRestart(driver, pin);
    grav->amr_check_after_output = false;
  }
  dust->amr_restart_state_found = false;
  dust->amr_restart_pending = false;
  dust->amr_restart_ages.clear();
  dust->amr_check_after_output = false;
}

void MarkRestartHeader(ParameterInput *pin, bool single_file_per_rank) {
  if (single_file_per_rank) Broken("self-gravity requires a shared restart file");
  pin->SetInteger("gravity", "restart_record_version", kVersion);
}

void WriteRestartState(IOWrapper &file, std::uint64_t offset, Mesh *pm) {
  Guard(pm);
  Bytes(pm);
#if MPI_PARALLEL_ENABLED
  // Ages are global metadata in MeshRefinement: every rank increments every age,
  // and regridding remaps all global entries. Check that invariant at checkpoint
  // time so a future local-only age update cannot silently serialize root zeros.
  if (pm->adaptive) {
    std::vector<int> low(pm->nmb_total), high(pm->nmb_total);
    for (int m = 0; m < pm->nmb_total; ++m) low[m] = pm->pmr->ncyc_since_ref(m);
    MPI_Allreduce(low.data(), high.data(), pm->nmb_total, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, low.data(), pm->nmb_total, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (low != high) Broken("global cooldown ages disagree between MPI ranks");
  }
#endif
  if (global_variable::my_rank != 0) return;
  std::vector<char> buffer(kMagic, kMagic+8);
  Put(buffer, kVersion);
  const std::uint32_t flags = (pm->adaptive && pm->pmb_pack->pgrav->amr_check_after_output ? 1u : 0u)
                            | (pm->adaptive ? 2u : 0u);
  Put(buffer, flags);
  Put(buffer, static_cast<std::uint64_t>(Bytes(pm)));
  Put(buffer, static_cast<std::uint64_t>(pm->nmb_total));
  Put(buffer, static_cast<double>(pm->time));
  Put(buffer, static_cast<std::int64_t>(pm->ncycle));
  Put(buffer, static_cast<std::uint32_t>(sizeof(Real)));
  Put(buffer, std::uint32_t(0));
  for (int m = 0; m < pm->nmb_total; ++m) {
    const auto loc = pm->lloc_eachmb[m];
    Put(buffer, static_cast<std::int32_t>(loc.lx1));
    Put(buffer, static_cast<std::int32_t>(loc.lx2));
    Put(buffer, static_cast<std::int32_t>(loc.lx3));
    Put(buffer, static_cast<std::int32_t>(loc.level));
    const int age = pm->adaptive ? pm->pmr->ncyc_since_ref(m) : 0;
    if (age < 0) Broken("negative mesh cooldown age");
    Put(buffer, static_cast<std::int32_t>(age));
  }
  buffer.insert(buffer.end(), kMagic, kMagic+8);
  if (buffer.size() != Bytes(pm)) Broken("metadata size mismatch");
  Transfer(file, buffer.data(), buffer.size(), offset, true);
}

void ReadRestartState(IOWrapper &file, std::uint64_t offset, Mesh *pm,
                      ParameterInput *pin, bool single_file_per_rank) {
  if (single_file_per_rank) Broken("self-gravity requires a shared restart file");
  const int marker = pin->DoesParameterExist("gravity", "restart_record_version") ?
                     pin->GetInteger("gravity", "restart_record_version") : 0;
  if (marker != 0 && marker != kVersion) Broken("unsupported header record version");
  int found = 0;
  std::vector<char> buffer(Bytes(pm));
  if (global_variable::my_rank == 0) {
    char magic[8];
    const auto n = file.Read_bytes_at(magic, 1, 8, offset, false);
    if (marker == 0) {
      if (n != 0) Broken("trailing metadata exists without its input-header marker");
    } else {
      if (n != 8 || std::memcmp(magic, kMagic, 8) != 0) {
        Broken("marked checkpoint has missing, truncated or invalid metadata magic");
      }
      if (offset > file.GetSize() || buffer.size() > file.GetSize()-offset) {
        Broken("truncated metadata record");
      }
      Transfer(file, buffer.data(), buffer.size(), offset, false);
      char extra;
      if (file.Read_bytes_at(&extra, 1, 1, Add(offset, buffer.size()), false) != 0) {
        Broken("unexpected bytes after metadata record");
      }
      found = 1;
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&found, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (found) {
    constexpr std::size_t chunk = std::size_t(1) << 30;
    for (std::size_t done = 0; done < buffer.size();) {
      const auto n = std::min(chunk, buffer.size()-done);
      MPI_Bcast(buffer.data()+done, static_cast<int>(n), MPI_BYTE, 0, MPI_COMM_WORLD);
      done += n;
    }
  }
#endif
  GravityRestartState state;
  if (!found) {
    if (pm->pmb_pack->pgrav != nullptr && global_variable::my_rank == 0) {
      std::cout << "# gravity restart LEGACY: no gravity restart metadata; ";
      if (pm->adaptive) {
        const auto *dust = pm->pmb_pack->pdust;
        if (dust != nullptr && dust->amr_restart_state_found) {
          std::cout << "dust AMR metadata supplies cooldown ages and the pending mesh check\n";
        } else {
          std::cout << "cooldown ages start at zero and the pending mesh check cannot be "
                       "recovered; exact AMR continuation is not guaranteed\n";
        }
      } else {
        std::cout << "static mesh needs no AMR replay\n";
      }
    }
    if (pm->pmb_pack->pgrav != nullptr) pm->pmb_pack->pgrav->LoadRestartState(std::move(state));
    return;
  }
  if (std::memcmp(buffer.data(), kMagic, 8) != 0 ||
      std::memcmp(buffer.data()+buffer.size()-8, kMagic, 8) != 0) Broken("bad framing magic");
  std::size_t position = 8;
  if (Get<std::uint32_t>(buffer, position) != kVersion) Broken("unsupported record version");
  const auto flags = Get<std::uint32_t>(buffer, position);
  if (flags > 3) Broken("invalid state flags");
  if (Get<std::uint64_t>(buffer, position) != buffer.size()) Broken("incorrect byte count");
  if (Get<std::uint64_t>(buffer, position) != static_cast<std::uint64_t>(pm->nmb_total)) {
    Broken("global block count mismatch");
  }
  const double time = Get<double>(buffer, position);
  const auto cycle = Get<std::int64_t>(buffer, position);
  if (!std::isfinite(time) || time != static_cast<double>(pm->time) || cycle != pm->ncycle) {
    Broken("stored time/cycle disagrees with native mesh header");
  }
  if (Get<std::uint32_t>(buffer, position) != sizeof(Real) ||
      Get<std::uint32_t>(buffer, position) != 0) Broken("precision/reserved field mismatch");
  state.found = true;
  state.pending = flags & 1u;
  state.adaptive = flags & 2u;
  if (state.adaptive != pm->adaptive) Broken("adaptive-mesh configuration changed on restart");
  state.ages.resize(pm->nmb_total);
  for (int m = 0; m < pm->nmb_total; ++m) {
    const auto loc = pm->lloc_eachmb[m];
    if (Get<std::int32_t>(buffer, position) != loc.lx1 ||
        Get<std::int32_t>(buffer, position) != loc.lx2 ||
        Get<std::int32_t>(buffer, position) != loc.lx3 ||
        Get<std::int32_t>(buffer, position) != loc.level) Broken("global block ordering mismatch");
    state.ages[m] = Get<std::int32_t>(buffer, position);
    if (state.ages[m] < 0) Broken("negative mesh cooldown age");
  }
  // Duplicate records are a file-integrity invariant, independent of whether
  // the user enables gravity for this continuation. Never hide malformed joint
  // metadata by switching a physics module off.
  const auto *dust = pm->pmb_pack->pdust;
  if (dust != nullptr && dust->amr_restart_state_found &&
      (dust->amr_restart_pending != state.pending || dust->amr_restart_ages != state.ages)) {
    Broken("dust and gravity AMR replay metadata disagree");
  }
  if (pm->pmb_pack->pgrav != nullptr) pm->pmb_pack->pgrav->LoadRestartState(std::move(state));
}

void SelfGravity::ResumeAmrAfterRestart(Driver *driver, ParameterInput *pin) {
  if (!restart_state_.found) return;
  auto *pm = pack_->pmesh;
  if (restart_state_.adaptive) {
    if (restart_state_.ages.size() != static_cast<std::size_t>(pm->nmb_total)) {
      Broken("loaded cooldown count does not match mesh");
    }
    for (int m = 0; m < pm->nmb_total; ++m) pm->pmr->ncyc_since_ref(m) = restart_state_.ages[m];
    if (global_variable::my_rank == 0) {
      std::cout << "# gravity AMR restart: restored " << pm->nmb_total << " cooldown ages; "
                << "pending check of cycle " << pm->ncycle << ": "
                << (restart_state_.pending ? "done now" : "none") << '\n';
    }
    const bool pending = restart_state_.pending;
    restart_state_.found = false;
    restart_state_.ages.clear();
    if (pending) pm->pmr->AdaptiveMeshRefinement(driver, pin);
  } else {
    restart_state_.found = false;
  }
  amr_check_after_output = false;
}
}  // namespace gravity
