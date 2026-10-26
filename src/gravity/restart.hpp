// AthenaK self-gravity restart metadata. BSD-3-Clause; see LICENSE.
#ifndef GRAVITY_RESTART_HPP_
#define GRAVITY_RESTART_HPP_
#include <cstdint>
#include <vector>
class IOWrapper;
class Mesh;
class ParameterInput;
namespace gravity {
struct GravityRestartState {
  bool found = false, pending = false, adaptive = false;
  std::vector<int> ages;
};
// Shared-file metadata lives immediately after native fluid arrays. SelfGravity
// currently excludes particles/dust, whose existing trailing sections are unchanged.
void MarkRestartHeader(ParameterInput *pin, bool single_file_per_rank);
void WriteRestartState(IOWrapper &file, std::uint64_t offset, Mesh *mesh);
void ReadRestartState(IOWrapper &file, std::uint64_t offset, Mesh *mesh,
                      ParameterInput *pin, bool single_file_per_rank);
}  // namespace gravity
#endif  // GRAVITY_RESTART_HPP_
