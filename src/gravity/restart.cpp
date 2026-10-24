// AthenaK self-gravity restart metadata. BSD-3-Clause; see LICENSE.
// Framing (native endianness, as in the surrounding native restart): magic[8],
// uint32 version/flags, uint64 bytes/nmb, double time, int64 cycle, uint32
// sizeof(Real)/reserved, nmb*(int32 lx1,lx2,lx3,level,age), magic[8].
#include "gravity/restart.hpp"
#include <cmath>
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
      (static_cast<std::uint64_t>(std::numeric_limits<int>::max())-kFixed)/kBlock) {
    Broken("global block count exceeds metadata capacity");
  }
  return kFixed+kBlock*pm->nmb_total;
}
void Guard(const Mesh *pm) {
  if (pm->pmb_pack->ppart != nullptr || pm->pmb_pack->pdust != nullptr) {
    Broken("combined gravity/particle restart sections are not implemented");
  }
}
}  // namespace

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
  if (buffer.size() != Bytes(pm) || file.Write_any_type_at(buffer.data(), buffer.size(),
      offset, "byte", false) != buffer.size()) Broken("metadata write failed");
}

void ReadRestartState(IOWrapper &file, std::uint64_t offset, Mesh *pm,
                      ParameterInput *pin, bool single_file_per_rank) {
  Guard(pm);
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
      if (file.Read_bytes_at(buffer.data(), 1, buffer.size(), offset, false) != buffer.size()) {
        Broken("truncated metadata record");
      }
      char extra;
      if (file.Read_bytes_at(&extra, 1, 1, offset+buffer.size(), false) != 0) {
        Broken("unexpected bytes after metadata record");
      }
      found = 1;
    }
  }
#if MPI_PARALLEL_ENABLED
  MPI_Bcast(&found, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (found) MPI_Bcast(buffer.data(), buffer.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
#endif
  GravityRestartState state;
  if (!found) {
    if (global_variable::my_rank == 0) {
      std::cout << "# gravity restart LEGACY: no gravity restart metadata; ";
      if (pm->adaptive) {
        std::cout << "cooldown ages start at zero and the pending mesh check cannot be "
                     "recovered; exact AMR continuation is not guaranteed\n";
      } else {
        std::cout << "static mesh needs no AMR replay\n";
      }
    }
    pm->pmb_pack->pgrav->LoadRestartState(std::move(state));
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
  pm->pmb_pack->pgrav->LoadRestartState(std::move(state));
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
