//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone and the AthenaK collaboration
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_snapshot_clump.cpp
//! \brief Snapshot-seeded synthetic dust-clump benchmark.
//!
//! This custom problem generator reads an AthenaK legacy binary particle-VTK file and
//! uses its particle positions as an empirical spatial template.  It is deliberately
//! not a restart: particle mass, stopping time, species, mean slip, and the uniform gas
//! state are supplied by the new input.  The intended use is a deterministic clumped
//! workload for drag/deposition benchmarks while particle+gas restart is unavailable.
//!
//! Configure with `-D PROBLEM=tests/dust_snapshot_clump`.  Every MPI rank reads the same
//! small source file, chooses the same deterministic subset, and retains only particles
//! owned by its local uniform-grid MeshBlocks.  This avoids assuming that a clumped
//! state has equal particle counts on every rank.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "dust/dust.hpp"
#include "globals.hpp"
#include "hydro/hydro.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"
#include "parameter_input.hpp"
#include "particles/particles.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

void DustSnapshotClumpHistory(HistoryData *pdata, Mesh *pm);

namespace {

struct ParticleSnapshot {
  int count = 0;
  std::vector<float> position;
  std::vector<float> velocity;
  std::vector<int> tag;
};

struct LocalParticle {
  int source_index;
  int selected_index;
  int local_block;
  Real x;
  Real y;
};

[[noreturn]] void Fatal(const std::string &message) {
  std::cout << "### FATAL ERROR in " << __FILE__ << std::endl << message << std::endl;
  std::exit(EXIT_FAILURE);
}

bool HostIsBigEndian() {
  std::uint32_t value = 1;
  return *(reinterpret_cast<unsigned char *>(&value)) == 0;
}

void SwapFloatBytes(float &value) {
  unsigned char *data = reinterpret_cast<unsigned char *>(&value);
  std::swap(data[0], data[3]);
  std::swap(data[1], data[2]);
}

std::string NextNonemptyLine(std::ifstream &stream) {
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) return line;
  }
  return "";
}

std::vector<float> ReadFloatArray(std::ifstream &stream, std::size_t count) {
  std::vector<float> values(count);
  stream.read(reinterpret_cast<char *>(values.data()), count*sizeof(float));
  if (stream.gcount() != static_cast<std::streamsize>(count*sizeof(float))) {
    Fatal("Particle VTK ended while reading a binary float array");
  }
  if (!HostIsBigEndian()) {
    for (float &value : values) SwapFloatBytes(value);
  }
  return values;
}

ParticleSnapshot ReadParticleVTK(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) Fatal("Unable to open <problem>/snapshot_file = " + path);

  std::string line;
  std::getline(stream, line);
  if (line.find("# vtk DataFile Version") != 0) {
    Fatal("Snapshot is not a legacy VTK file: " + path);
  }
  std::getline(stream, line);  // AthenaK description, time, cycle
  std::getline(stream, line);
  if (line != "BINARY") Fatal("Particle snapshot must be a binary legacy VTK file");
  std::getline(stream, line);
  if (line != "DATASET UNSTRUCTURED_GRID") {
    Fatal("Particle snapshot must use DATASET UNSTRUCTURED_GRID");
  }

  line = NextNonemptyLine(stream);
  std::istringstream point_header(line);
  std::string keyword, data_type;
  int count = 0;
  point_header >> keyword >> count >> data_type;
  if (keyword != "POINTS" || count <= 0 || data_type != "float") {
    Fatal("Unexpected POINTS header in particle VTK: " + line);
  }

  ParticleSnapshot snapshot;
  snapshot.count = count;
  snapshot.position = ReadFloatArray(stream, 3*static_cast<std::size_t>(count));
  snapshot.tag.resize(count);
  std::iota(snapshot.tag.begin(), snapshot.tag.end(), 0);

  line = NextNonemptyLine(stream);
  std::istringstream point_data_header(line);
  int point_data_count = 0;
  point_data_header >> keyword >> point_data_count;
  if (keyword != "POINT_DATA" || point_data_count != count) {
    Fatal("Unexpected POINT_DATA header in particle VTK: " + line);
  }

  while (stream.peek() != std::char_traits<char>::eof()) {
    line = NextNonemptyLine(stream);
    if (line.empty()) break;
    std::istringstream section(line);
    std::string name;
    section >> keyword >> name >> data_type;
    if (keyword == "SCALARS") {
      std::string lookup = NextNonemptyLine(stream);
      if (lookup != "LOOKUP_TABLE default") {
        Fatal("Unexpected scalar lookup-table header in particle VTK: " + lookup);
      }
      std::vector<float> values = ReadFloatArray(stream, count);
      if (name == "ptag") {
        for (int p=0; p<count; ++p) snapshot.tag[p] = static_cast<int>(std::lround(values[p]));
      }
    } else if (keyword == "VECTORS") {
      std::vector<float> values = ReadFloatArray(stream, 3*static_cast<std::size_t>(count));
      if (name == "pvel") snapshot.velocity = std::move(values);
    } else {
      Fatal("Unexpected data section in particle VTK: " + line);
    }
  }
  if (snapshot.velocity.size() != snapshot.position.size()) {
    Fatal("Particle snapshot does not contain VECTORS pvel");
  }
  return snapshot;
}

std::uint64_t SplitMix64(std::uint64_t value) {
  value += UINT64_C(0x9e3779b97f4a7c15);
  value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
  return value ^ (value >> 31);
}

int FindLocalBlock(MeshBlockPack *pmbp, Real x, Real y) {
  auto &mbsize = pmbp->pmb->mb_size;
  auto &meshsize = pmbp->pmesh->mesh_size;
  for (int m=0; m<pmbp->nmb_thispack; ++m) {
    bool in_x = (x >= mbsize.h_view(m).x1min && x < mbsize.h_view(m).x1max) ||
                (x == meshsize.x1max && mbsize.h_view(m).x1max == meshsize.x1max);
    bool in_y = (y >= mbsize.h_view(m).x2min && y < mbsize.h_view(m).x2max) ||
                (y == meshsize.x2max && mbsize.h_view(m).x2max == meshsize.x2max);
    if (in_x && in_y) return m;
  }
  return -1;
}

Real MapCoordinate(float source, Real source_min, Real source_max,
                   Real target_min, Real target_max) {
  if (!(source_max > source_min)) Fatal("Snapshot source coordinate bounds are invalid");
  Real fraction = (static_cast<Real>(source) - source_min)/(source_max - source_min);
  if (fraction < -1.0e-6 || fraction > 1.0 + 1.0e-6) {
    Fatal("A snapshot particle lies outside the configured source coordinate bounds");
  }
  fraction = std::max(static_cast<Real>(0.0), std::min(static_cast<Real>(1.0), fraction));
  Real target = target_min + fraction*(target_max - target_min);
  if (target == target_max) target = std::nextafter(target_max, target_min);
  return target;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn ProblemGenerator::UserProblem()
//! \brief Initialize a controllable dust benchmark from a particle-snapshot template.

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = DustSnapshotClumpHistory;
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr) {
    Fatal("dust_snapshot_clump requires <hydro>, <particles>, and <dust> blocks");
  }
  if (pmy_mesh_->three_d || pmy_mesh_->one_d || pmy_mesh_->multilevel) {
    Fatal("dust_snapshot_clump currently requires a uniform 2-D mesh");
  }

  std::string path = pin->GetString("problem", "snapshot_file");
  ParticleSnapshot snapshot = ReadParticleVTK(path);
  int target_count = pmy_mesh_->nprtcl_total;
  if (target_count <= 0 || target_count > snapshot.count) {
    Fatal("The ppc-derived target particle count must be in [1, snapshot particle count]");
  }

  // Select an exact-size, deterministic, unbiased subset by sorting source indices by a
  // stable integer hash of their original particle tags.  The same result is obtained
  // on every rank and backend; no device RNG or execution-order dependence is involved.
  std::uint64_t seed = static_cast<std::uint64_t>(
      pin->GetOrAddInteger("problem", "snapshot_seed", 1));
  std::vector<int> selected(snapshot.count);
  std::iota(selected.begin(), selected.end(), 0);
  std::sort(selected.begin(), selected.end(), [&](int a, int b) {
    std::uint64_t ha = SplitMix64(static_cast<std::uint64_t>(snapshot.tag[a]) + seed);
    std::uint64_t hb = SplitMix64(static_cast<std::uint64_t>(snapshot.tag[b]) + seed);
    return (ha < hb) || (ha == hb && a < b);
  });
  selected.resize(target_count);

  auto &meshsize = pmy_mesh_->mesh_size;
  Real sx1min = pin->GetOrAddReal("problem", "snapshot_x1min", meshsize.x1min);
  Real sx1max = pin->GetOrAddReal("problem", "snapshot_x1max", meshsize.x1max);
  Real sx2min = pin->GetOrAddReal("problem", "snapshot_x2min", meshsize.x2min);
  Real sx2max = pin->GetOrAddReal("problem", "snapshot_x2max", meshsize.x2max);

  std::vector<LocalParticle> local;
  local.reserve(pmbp->ppart->nprtcl_thispack);
  for (int q=0; q<target_count; ++q) {
    int source_index = selected[q];
    Real x = MapCoordinate(snapshot.position[3*source_index], sx1min, sx1max,
                           meshsize.x1min, meshsize.x1max);
    Real y = MapCoordinate(snapshot.position[3*source_index + 1], sx2min, sx2max,
                           meshsize.x2min, meshsize.x2max);
    int m = FindLocalBlock(pmbp, x, y);
    if (m >= 0) local.push_back({source_index, q, m, x, y});
  }

  // A clumped template generally produces unequal MPI-rank counts.  Resize to the true
  // local ownership, then refresh Mesh's count metadata before outputs or migration use
  // it.  The global count must remain exactly the ppc-derived target count.
  particles::Particles *ppar = pmbp->ppart;
  int local_count = static_cast<int>(local.size());
  Kokkos::realloc(ppar->prtcl_rdata, ppar->nrdata, local_count);
  Kokkos::realloc(ppar->prtcl_idata, ppar->nidata, local_count);
  ppar->nprtcl_thispack = local_count;
  pmy_mesh_->nprtcl_thisrank = local_count;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&local_count, 1, MPI_INT, pmy_mesh_->nprtcl_eachrank, 1, MPI_INT,
                MPI_COMM_WORLD);
#else
  pmy_mesh_->nprtcl_eachrank[0] = local_count;
#endif
  pmy_mesh_->nprtcl_total = 0;
  for (int r=0; r<global_variable::nranks; ++r) {
    pmy_mesh_->nprtcl_total += pmy_mesh_->nprtcl_eachrank[r];
  }
  if (pmy_mesh_->nprtcl_total != target_count) {
    Fatal("Snapshot particles are not owned exactly once by the distributed uniform mesh");
  }

  int nspecies = pmbp->pdust->nspecies;
  if (target_count < nspecies) Fatal("Target particle count is smaller than nspecies");
  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real gvx = pin->GetOrAddReal("problem", "gas_vx", 0.0);
  Real gvy = pin->GetOrAddReal("problem", "gas_vy", 0.0);
  Real gvz = pin->GetOrAddReal("problem", "gas_vz", 0.0);
  Real slip_x = pin->GetOrAddReal("problem", "slip_vx", 0.1);
  Real slip_y = pin->GetOrAddReal("problem", "slip_vy", 0.0);
  Real slip_z = pin->GetOrAddReal("problem", "slip_vz", 0.0);
  std::string velocity_mode = pin->GetOrAddString("problem", "velocity_mode", "uniform");
  if (velocity_mode != "uniform" && velocity_mode != "snapshot_centered" &&
      velocity_mode != "snapshot_raw") {
    Fatal("<problem>/velocity_mode must be uniform, snapshot_centered, or snapshot_raw");
  }

  Real source_mean[3] = {0.0, 0.0, 0.0};
  for (int source_index : selected) {
    for (int d=0; d<3; ++d) source_mean[d] += snapshot.velocity[3*source_index + d];
  }
  for (int d=0; d<3; ++d) source_mean[d] /= static_cast<Real>(target_count);

  std::vector<Real> eps(nspecies), particle_mass(nspecies);
  std::vector<int> species_count(nspecies, 0);
  for (int q=0; q<target_count; ++q) ++species_count[q % nspecies];
  Real domain_volume = (meshsize.x1max - meshsize.x1min)*(meshsize.x2max - meshsize.x2min);
  bool has_eps = pin->DoesParameterExist("problem", "eps_1");
  for (int s=0; s<nspecies; ++s) {
    eps[s] = has_eps ? pin->GetReal("problem", "eps_" + std::to_string(s+1)) :
                       pmbp->pdust->dust_to_gas/static_cast<Real>(nspecies);
    if (eps[s] < 0.0) Fatal("Per-species dust-to-gas ratios must be nonnegative");
    particle_mass[s] = eps[s]*rho0*domain_volume/static_cast<Real>(species_count[s]);
  }

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto hpr = Kokkos::create_mirror_view(pr);
  auto hpi = Kokkos::create_mirror_view(pi);
  Real vmax = 1.0e-30;
  for (int p=0; p<local_count; ++p) {
    const LocalParticle &lp = local[p];
    int s = lp.selected_index % nspecies;
    Real pv[3] = {gvx + slip_x, gvy + slip_y, gvz + slip_z};
    if (velocity_mode == "snapshot_raw") {
      for (int d=0; d<3; ++d) pv[d] = snapshot.velocity[3*lp.source_index + d];
    } else if (velocity_mode == "snapshot_centered") {
      Real gas_v[3] = {gvx, gvy, gvz};
      Real slip[3] = {slip_x, slip_y, slip_z};
      for (int d=0; d<3; ++d) {
        pv[d] = snapshot.velocity[3*lp.source_index + d] - source_mean[d]
              + gas_v[d] + slip[d];
      }
    }
    for (int d=0; d<3; ++d) vmax = std::max(vmax, fabs(pv[d]));
    for (int n=0; n<ppar->nrdata; ++n) hpr(n,p) = 0.0;
    hpi(PGID,p) = pmbp->gids + lp.local_block;
    hpi(PTAG,p) = snapshot.tag[lp.source_index];
    hpi(PSP,p) = s;
    hpr(IPX,p) = hpr(IPX1,p) = lp.x;
    hpr(IPY,p) = hpr(IPY1,p) = lp.y;
    hpr(IPZ,p) = hpr(IPZ1,p) = 0.0;
    hpr(IPVX,p) = hpr(IPVX1,p) = pv[0];
    hpr(IPVY,p) = hpr(IPVY1,p) = pv[1];
    hpr(IPVZ,p) = hpr(IPVZ1,p) = pv[2];
    hpr(IPRX,p) = hpr(IPRY,p) = hpr(IPRZ,p) = 0.0;
    hpr(IPTS,p) = pmbp->pdust->taus.h_view(s);
    hpr(IPM,p) = particle_mass[s];
  }
  Kokkos::deep_copy(pr, hpr);
  Kokkos::deep_copy(pi, hpi);

  // Uniform synthetic gas.  This intentionally does not claim to reconstruct the gas
  // state that accompanied the source particle snapshot.
  auto &indcs = pmy_mesh_->mb_indcs;
  int n1 = indcs.nx1 + 2*indcs.ng;
  int n2 = indcs.nx2 + 2*indcs.ng;
  auto &u0 = pmbp->phydro->u0;
  int nmb = pmbp->nmb_thispack;
  par_for("snapshot_clump_gas", DevExeSpace(), 0, nmb-1, 0, 0, 0, n2-1, 0, n1-1,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m,IDN,k,j,i) = rho0;
    u0(m,IM1,k,j,i) = rho0*gvx;
    u0(m,IM2,k,j,i) = rho0*gvy;
    u0(m,IM3,k,j,i) = rho0*gvz;
  });

  auto &mbsize = pmbp->pmb->mb_size;
  ppar->dtnew = pmbp->pdust->dt_cfl*
      std::min(mbsize.h_view(0).dx1, mbsize.h_view(0).dx2)/vmax;

  if (global_variable::my_rank == 0) {
    int count_min = pmy_mesh_->nprtcl_eachrank[0];
    int count_max = count_min;
    for (int r=1; r<global_variable::nranks; ++r) {
      count_min = std::min(count_min, pmy_mesh_->nprtcl_eachrank[r]);
      count_max = std::max(count_max, pmy_mesh_->nprtcl_eachrank[r]);
    }
    std::cout << "# snapshot-clump: source=" << path << " source_particles="
              << snapshot.count << " selected=" << target_count << " rank_count_range=["
              << count_min << "," << count_max << "] velocity_mode=" << velocity_mode
              << std::endl;
    std::cout << "# snapshot-clump source selected mean pvel=" << source_mean[0] << " "
              << source_mean[1] << " " << source_mean[2] << std::endl;
  }
}

//----------------------------------------------------------------------------------------
//! \brief Global dust mass, momentum, and directional kinetic energy.  Together with
//! the standard gas history these give action--reaction and temporal-convergence checks.

void DustSnapshotClumpHistory(HistoryData *pdata, Mesh *pm) {
  particles::Particles *ppar = pm->pmb_pack->ppart;
  pdata->nhist = 7;
  pdata->label[0] = "dust_mass";
  pdata->label[1] = "dust_mom1";
  pdata->label[2] = "dust_mom2";
  pdata->label[3] = "dust_mom3";
  pdata->label[4] = "dust_ke1";
  pdata->label[5] = "dust_ke2";
  pdata->label[6] = "dust_ke3";

  auto &pr = ppar->prtcl_rdata;
  int npart = ppar->nprtcl_thispack;
  Real mass = 0.0, mom1 = 0.0, mom2 = 0.0, mom3 = 0.0;
  Real ke1 = 0.0, ke2 = 0.0, ke3 = 0.0;
  Kokkos::parallel_reduce("snapshot_clump_history",
  Kokkos::RangePolicy<>(DevExeSpace(),0,npart),
  KOKKOS_LAMBDA(const int p, Real &mass_, Real &mom1_, Real &mom2_, Real &mom3_,
                Real &ke1_, Real &ke2_, Real &ke3_) {
    Real mp = pr(IPM,p);
    Real vx = pr(IPVX,p), vy = pr(IPVY,p), vz = pr(IPVZ,p);
    mass_ += mp;
    mom1_ += mp*vx;
    mom2_ += mp*vy;
    mom3_ += mp*vz;
    ke1_ += 0.5*mp*vx*vx;
    ke2_ += 0.5*mp*vy*vy;
    ke3_ += 0.5*mp*vz*vz;
  }, Kokkos::Sum<Real>(mass), Kokkos::Sum<Real>(mom1), Kokkos::Sum<Real>(mom2),
     Kokkos::Sum<Real>(mom3), Kokkos::Sum<Real>(ke1), Kokkos::Sum<Real>(ke2),
     Kokkos::Sum<Real>(ke3));
  pdata->hdata[0] = mass;
  pdata->hdata[1] = mom1;
  pdata->hdata[2] = mom2;
  pdata->hdata[3] = mom3;
  pdata->hdata[4] = ke1;
  pdata->hdata[5] = ke2;
  pdata->hdata[6] = ke3;
}
