// AthenaK self-gravity restart metadata. BSD-3-Clause; see LICENSE.
#ifndef GRAVITY_RESTART_HPP_
#define GRAVITY_RESTART_HPP_
#include <cstdint>
#include <vector>
class IOWrapper;
class Mesh;
class ParameterInput;
class Driver;
namespace gravity {
struct GravityRestartState {
  bool found = false, pending = false, adaptive = false;
  std::vector<int> ages;
};
// Extension chain: native fluid arrays, optional legacy AKPRST01 rank sections,
// optional AKSGRST1 record. Each module receives its own bounded byte range.
void WriteRestartExtensions(IOWrapper &file, std::uint64_t offset, Mesh *mesh,
                            bool single_file_per_rank);
void ReadRestartExtensions(IOWrapper &file, std::uint64_t offset, Mesh *mesh,
                           ParameterInput *pin, bool single_file_per_rank);
void ResumeRestart(Mesh *mesh, Driver *driver, ParameterInput *pin);
void MarkRestartHeader(ParameterInput *pin, bool single_file_per_rank);
void WriteRestartState(IOWrapper &file, std::uint64_t offset, Mesh *mesh);
void ReadRestartState(IOWrapper &file, std::uint64_t offset, Mesh *mesh,
                      ParameterInput *pin, bool single_file_per_rank);
}  // namespace gravity
#endif  // GRAVITY_RESTART_HPP_
