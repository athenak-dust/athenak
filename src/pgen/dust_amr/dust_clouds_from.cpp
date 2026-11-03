//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_clouds_from.cpp
//! \brief Start a run with 3-D refinable clouds (any mesh whose root grid is that of the file,
//! adaptive or uniform) from the restart file of a uniform run of point particles
//! (<problem>/clouds_from = FILE), e.g. to continue a uniform run on an adaptive mesh.
//!
//! A native restart (-r) keeps the MeshBlocks of the file, which may be too large for an
//! adaptive mesh to refine block by block (e.g. 16 blocks of 128 x 128 x 512 cells).  Here the run
//! starts fresh with the mesh of its input (any block size, <mesh_refinement> as given) at
//! <time>/start_time = the time of the file, and only the gas and the particles are taken from
//! the file.  The file is read by every rank: its input text up to <par_end>, the mesh
//! header, the hydro variables of the MeshBlocks that overlap this rank's blocks and the
//! particle sections (dust_restart.cpp).  It must have the root grid of this run (same box,
//! same cells), all MeshBlocks on the root level, the same hydro variables and passive scalars,
//! the dust particle columns (17 real, 4 or 5 int), only plain level-0 particles (PLEV = 0:
//! point particles, or the clouds of a uniform cloud run) and the dust species of this run
//! (record species_table, when the file has it).  All MeshBlocks of this run must be on the
//! root level when the problem generator runs (no static refinement): the clouds split as
//! the dynamic mesh refines.  A file without particles (all dust gone through the x3 faces)
//! stops the run: there is nothing to convert, and the root cloud mass and the tag counter are
//! taken from the clouds; its gas-only continuation is a native restart (-r) of the file.
//!  - Gas: every cell value copied (the ghost cells with the nearest cell of the mesh; the
//!    boundary exchange of Driver::Initialize sets them).
//!  - Particles: every particle becomes one plain level-0 cloud with its tag, species, mass,
//!    position, velocity and stopping time, in the block that holds it (the leaf owner of the
//!    3-D cloud traversal, as the root parents of a fresh start); the RK registers as at a fresh
//!    start (position and velocity copies, drag rates zero).  A particle beyond an x1 face is
//!    wrapped as bvals_part.cpp wraps a crossing particle (y shifted by the shear offset of the
//!    file's time), y periodically; one beyond an x3 face stops the run (the outflow faces
//!    remove such particles before a file is written).
//!  - Cloud state as at a fresh start: the tag counter from the largest tag of the file and the
//!    tag bound of its records (tag_state), the root cloud mass of G and R (the largest
//!    level-0 cloud of every species), the G and R schedules from the start time, the
//!    split to the leaves (nothing to split on a root-level mesh).
//!  - Checks printed: gas mass and momentum of the file and of this run; count, mass, momentum
//!    and an order-independent 64-bit hash of every particle's tag, species, position,
//!    velocity, stopping time and mass, of the file and of the clouds (equal hashes: every
//!    particle came over bit for bit).
//! The li2018 totals <problem>/vbc_mean0 and vbc_dm come from the input, or else from the
//! input text of the file (written there by the run).  What else the file run kept outside the
//! gas and the particles (the dust records but tag_state and species_table, the x3 outflow
//! ledger, the output numbers) is not read: the run counts from cycle 0.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "particles/particle_counts.hpp"
#include "dust/dust.hpp"
#include "dust/dust_species.hpp"
#include "dust_setup.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust_setup {
namespace {
[[noreturn]] void CFail(const std::string &msg) {
  std::cout << "### FATAL ERROR in <problem>/clouds_from: " << msg << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
// a condition that every rank evaluates alike (from data every rank read): rank 0 tells, all stop
[[noreturn]] void CFailAll(const std::string &msg) {
  if (global_variable::my_rank == 0) {
    std::cout << "### FATAL ERROR in <problem>/clouds_from: " << msg << std::endl;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Barrier(MPI_COMM_WORLD);   // (rank 0 has printed)
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
void ReadAt(std::ifstream &f, std::uint64_t at, void *p, std::uint64_t n, const char *what) {
  f.seekg(static_cast<std::streamoff>(at));
  f.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
  if (!f) CFail(std::string("the file is truncated (") + what + ")");
}
double SumD(double v) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return v;
}
std::uint64_t SumU(std::uint64_t v) {   // modulo 2^64: order-independent
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
  return v;
}
std::uint64_t MaxU(std::uint64_t v) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
#endif
  return v;
}
// splitmix64 finaliser; a particle's hash chains its fields through it
inline std::uint64_t Mix(std::uint64_t h, const std::uint64_t v) {
  std::uint64_t z = h ^ (v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
  z = (z ^ (z >> 30))*0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27))*0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
// compensated (Neumaier) sum, so that the printed totals of the file and of this run differ
// by round-off of the totals, not of the summation order
struct KSum {
  double s = 0.0, c = 0.0;
  void Add(const double x) {
    const double t = s + x;
    c += (std::fabs(s) >= std::fabs(x)) ? (s - t) + x : (x - t) + s;
    s = t;
  }
  double Value() const {return s + c;}
};
inline std::uint64_t Bits(const Real x) {
  std::uint64_t b = 0;
  std::memcpy(&b, &x, sizeof(x) < sizeof(b) ? sizeof(x) : sizeof(b));
  return b;
}
// the value of <block>/key in the input text of a restart file ("" if absent)
std::string FileParam(const std::string &text, const std::string &block, const std::string &key) {
  std::istringstream in(text);
  std::string line, cur;
  auto trim = [](std::string s) {
    const std::size_t a = s.find_first_not_of(" \t\r");
    if (a == std::string::npos) return std::string();
    const std::size_t b = s.find_last_not_of(" \t\r");
    return s.substr(a, b - a + 1);
  };
  while (std::getline(in, line)) {
    const std::size_t hash = line.find('#');
    if (hash != std::string::npos) line = line.substr(0, hash);
    line = trim(line);
    if (line.empty()) continue;
    if (line.front() == '<') {
      const std::size_t e = line.find('>');
      cur = (e == std::string::npos) ? line.substr(1) : line.substr(1, e - 1);
      if (cur == "par_end") break;
      continue;
    }
    if (cur != block) continue;
    const std::size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    if (trim(line.substr(0, eq)) == key) return trim(line.substr(eq + 1));
  }
  return "";
}
}  // namespace

void CloudsFromRestart(ParameterInput *pin, Mesh *pm, const std::string &path) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const int me = global_variable::my_rank;
  dust::DustGasDrag *pd = pmbp->pdust;
  particles::Particles *ppar = pmbp->ppart;
  if (ppar == nullptr || pd == nullptr) CFail("needs dust particles");
  if (pd->smr_transfer_mode != DustSMRTransfer::refinable_cloud || !pd->Clouds3D()) {
    CFail("needs 3-D refinable clouds (<dust>/amr_pic = refinable_cloud on a 3-D mesh)");
  }
  if (pmbp->phydro == nullptr) CFail("needs hydro");
  std::ifstream f(path, std::ios::binary);
  if (!f) CFail("cannot open " + path);

  // ---- the input text and the mesh header (as written by RestartOutput)
  std::uint64_t at = 0;
  std::string text;
  {
    const std::string mark = "<par_end>\n";
    text.assign(std::size_t(1) << 24, '\0');
    f.read(&text[0], static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(f.gcount()));
    f.clear();
    const std::size_t pos = text.find(mark);
    if (pos == std::string::npos) CFail("no <par_end> in the first 16 MiB of " + path);
    at = pos + mark.size();
    text.resize(at);
  }
  int nmb_c = 0, root_c = 0, ncyc_c = 0;
  RegionSize ms_c;
  RegionIndcs mi_c, mb_c;
  Real time_c = 0.0, dt_c = 0.0;
  ReadAt(f, at, &nmb_c, sizeof(int), "header"); at += sizeof(int);
  ReadAt(f, at, &root_c, sizeof(int), "header"); at += sizeof(int);
  ReadAt(f, at, &ms_c, sizeof(RegionSize), "header"); at += sizeof(RegionSize);
  ReadAt(f, at, &mi_c, sizeof(RegionIndcs), "header"); at += sizeof(RegionIndcs);
  ReadAt(f, at, &mb_c, sizeof(RegionIndcs), "header"); at += sizeof(RegionIndcs);
  ReadAt(f, at, &time_c, sizeof(Real), "header"); at += sizeof(Real);
  ReadAt(f, at, &dt_c, sizeof(Real), "header"); at += sizeof(Real);
  ReadAt(f, at, &ncyc_c, sizeof(int), "header"); at += sizeof(int);
  if (nmb_c < 1) CFail("no MeshBlock in the file");
  std::vector<LogicalLocation> ll(nmb_c);
  ReadAt(f, at, ll.data(), nmb_c*sizeof(LogicalLocation), "logical locations");
  at += nmb_c*(sizeof(LogicalLocation) + sizeof(float));   // locations and costs
  std::uint64_t data_size = 0;
  ReadAt(f, at, &data_size, sizeof(std::uint64_t), "data size"); at += sizeof(std::uint64_t);
  const std::uint64_t data0 = at;

  // ---- checks: the same root grid, a uniform file, the same variables, the start time
  const RegionSize &ms = pm->mesh_size;
  const RegionIndcs &mi = pm->mesh_indcs;
  const Real tol = 1.0e-12*std::max(1.0, ms.x3max - ms.x3min);
  if (std::fabs(ms.x1min - ms_c.x1min) > tol || std::fabs(ms.x1max - ms_c.x1max) > tol ||
      std::fabs(ms.x2min - ms_c.x2min) > tol || std::fabs(ms.x2max - ms_c.x2max) > tol ||
      std::fabs(ms.x3min - ms_c.x3min) > tol || std::fabs(ms.x3max - ms_c.x3max) > tol) {
    CFail("the file's box is not the box of this run");
  }
  if (mi.nx1 != mi_c.nx1 || mi.nx2 != mi_c.nx2 || mi.nx3 != mi_c.nx3) {
    CFail("the root grid of this run must be the file's grid (" + std::to_string(mi_c.nx1) + " x " +
          std::to_string(mi_c.nx2) + " x " + std::to_string(mi_c.nx3) + " cells)");
  }
  for (int g=0; g<nmb_c; ++g) {
    if (ll[g].level != root_c) CFail("the file's mesh is refined (only uniform files)");
  }
  const int nmb = pmbp->nmb_thispack;
  {
    auto &lev = pmbp->pmb->mb_lev;
    for (int m=0; m<nmb; ++m) {
      if (lev.h_view(m) != pm->root_level) {
        CFail("every MeshBlock of this run must be on the root level at the start (no static "
              "refinement): the clouds split as the dynamic mesh refines");
      }
    }
  }
  const int nhyd = pmbp->phydro->nhydro, nscal = pmbp->phydro->nscalars;
  const int nvar = nhyd + nscal;
  const int ngc = mb_c.ng;
  const std::uint64_t o1 = mb_c.nx1 + 2*ngc, o2 = mb_c.nx2 + 2*ngc, o3 = mb_c.nx3 + 2*ngc;
  if (data_size != static_cast<std::uint64_t>(nvar)*o1*o2*o3*sizeof(Real)) {
    CFail("the cell data of a MeshBlock of the file is not " + std::to_string(nhyd) +
          " hydro variables and " + std::to_string(nscal) + " passive scalars (data size " +
          std::to_string(data_size) + ", expected " +
          std::to_string(static_cast<std::uint64_t>(nvar)*o1*o2*o3*sizeof(Real)) + ")");
  }
  if (pm->time != time_c) {
    std::ostringstream s;
    s << std::setprecision(17) << "<time>/start_time must be the time of the file, " << time_c
      << " (this run: " << pm->time << ")";
    CFail(s.str());
  }

  // ---- gas: every cell of this rank's blocks from the file block that holds it
  const int NX = mi.nx1, NY = mi.nx2, NZ = mi.nx3;
  KSum fm, fp[3];   // this run's active cells
  KSum cm, cp[3];   // the file's blocks g = me (mod nranks)
  {
    auto &u0 = pmbp->phydro->u0;
    auto u0h = Kokkos::create_mirror_view(u0);
    const auto &ind = pm->mb_indcs;
    const int ng = ind.ng;
    const int n1 = ind.nx1 + 2*ng, n2 = ind.nx2 + 2*ng, n3 = ind.nx3 + 2*ng;
    // the global cell index of the first active cell of each of this rank's blocks
    std::vector<long> I0(nmb), J0(nmb), K0(nmb);
    for (int m=0; m<nmb; ++m) {
      const RegionSize &bs = pmbp->pmb->mb_size.h_view(m);
      I0[m] = std::lround((bs.x1min - ms.x1min)/ms.dx1);
      J0[m] = std::lround((bs.x2min - ms.x2min)/ms.dx2);
      K0[m] = std::lround((bs.x3min - ms.x3min)/ms.dx3);
    }
    std::vector<Real> buf(static_cast<std::size_t>(nvar)*o1*o2*o3);
    std::vector<char> filled(static_cast<std::size_t>(nmb)*n1*n2*n3, 0);
    const double dv = ms.dx1*ms.dx2*ms.dx3;
    for (int g=0; g<nmb_c; ++g) {
      const long gi0 = static_cast<long>(ll[g].lx1)*mb_c.nx1, gj0 = static_cast<long>(ll[g].lx2)*mb_c.nx2;
      const long gk0 = static_cast<long>(ll[g].lx3)*mb_c.nx3;
      // this rank's blocks whose cells (ghosts clamped to the mesh) meet file block g
      auto meets = [&](int m) {
        auto lo = [](long a, long n) {return std::max(0L, std::min(a, n - 1));};
        const long a1 = lo(I0[m] - ng, NX), b1 = lo(I0[m] + ind.nx1 + ng - 1, NX);
        const long a2 = lo(J0[m] - ng, NY), b2 = lo(J0[m] + ind.nx2 + ng - 1, NY);
        const long a3 = lo(K0[m] - ng, NZ), b3 = lo(K0[m] + ind.nx3 + ng - 1, NZ);
        return !(b1 < gi0 || a1 >= gi0 + mb_c.nx1 || b2 < gj0 || a2 >= gj0 + mb_c.nx2 ||
                 b3 < gk0 || a3 >= gk0 + mb_c.nx3);
      };
      std::vector<int> hit;
      for (int m=0; m<nmb; ++m) if (meets(m)) hit.push_back(m);
      const bool audit = (g % global_variable::nranks == me);
      if (hit.empty() && !audit) continue;
      ReadAt(f, data0 + data_size*static_cast<std::uint64_t>(g), buf.data(), data_size, "hydro data");
      auto at_f = [&](int v, long i, long j, long k) -> Real {   // file block g, local active index
        return buf[((static_cast<std::size_t>(v)*o3 + k + ngc)*o2 + j + ngc)*o1 + i + ngc];
      };
      if (audit) {
        for (int k=0; k<mb_c.nx3; ++k) for (int j=0; j<mb_c.nx2; ++j) for (int i=0; i<mb_c.nx1; ++i) {
          cm.Add(at_f(IDN, i, j, k)*dv);
          for (int d=0; d<3; ++d) cp[d].Add(at_f(IM1 + d, i, j, k)*dv);
        }
      }
      for (const int m : hit) {
        for (int k=0; k<n3; ++k) {
          const long K = std::clamp(K0[m] + k - ng, 0L, static_cast<long>(NZ) - 1) - gk0;
          if (K < 0 || K >= mb_c.nx3) continue;
          for (int j=0; j<n2; ++j) {
            const long J = std::clamp(J0[m] + j - ng, 0L, static_cast<long>(NY) - 1) - gj0;
            if (J < 0 || J >= mb_c.nx2) continue;
            for (int i=0; i<n1; ++i) {
              const long I = std::clamp(I0[m] + i - ng, 0L, static_cast<long>(NX) - 1) - gi0;
              if (I < 0 || I >= mb_c.nx1) continue;
              for (int v=0; v<nvar; ++v) u0h(m,v,k,j,i) = at_f(v, I, J, K);
              filled[((static_cast<std::size_t>(m)*n3 + k)*n2 + j)*n1 + i] = 1;
            }
          }
        }
      }
    }
    for (const char c : filled) if (!c) CFail("a cell of this run has no cell in the file");
    for (int m=0; m<nmb; ++m) {
      for (int k=ind.ks; k<=ind.ke; ++k) for (int j=ind.js; j<=ind.je; ++j) for (int i=ind.is; i<=ind.ie; ++i) {
        fm.Add(u0h(m,IDN,k,j,i)*dv);
        for (int d=0; d<3; ++d) fp[d].Add(u0h(m,IM1 + d,k,j,i)*dv);
      }
    }
    Kokkos::deep_copy(u0, u0h);
  }
  const double fmass = SumD(fm.Value()), cmass = SumD(cm.Value());
  double fmom[3], cmom[3];
  for (int d=0; d<3; ++d) {fmom[d] = SumD(fp[d].Value()); cmom[d] = SumD(cp[d].Value());}

  // ---- particles: every section of the file, those in this rank's blocks
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  if (nidata <= PTAGH) CFail("the dust particles of this run have no PTAGH column");
  std::uint64_t secpos = data0 + data_size*static_cast<std::uint64_t>(nmb_c);
  std::vector<std::vector<Real>> rc(nrdata);
  std::vector<std::vector<int>> ic(nidata);
  const Real lx = ms.x1max - ms.x1min, ly = ms.x2max - ms.x2min;
  const Real qshear = pin->GetReal("shearing_box", "qshear");
  const Real omega0 = pin->GetReal("shearing_box", "omega0");
  const Real ysh = std::fmod(qshear*omega0*lx*time_c, ly);
  // file side: every particle (each rank reads all); kept side: this rank's
  KSum pm_s, pp_s[3];
  std::uint64_t phash = 0, nfile = 0, nwrap = 0, ntomb = 0, nlev = 0, nz = 0, tag_bound = 0;
  bool species_seen = false;
  std::string species_diff;
  int nsec = -1;
  const auto lt3 = pd->leaf_traversal3d;
  auto &indcs = pm->mb_indcs;
  const int ncell123 = (indcs.nx1 + 2*indcs.ng)*(indcs.nx2 + 2*indcs.ng)*(indcs.nx3 + 2*indcs.ng);
  std::uint64_t nnoowner = 0;
  Real vmax = 1.0e-30;
  for (int sct=0; ; ++sct) {
    char head[40];
    ReadAt(f, secpos, head, sizeof(head), "particle section head");
    if (std::memcmp(head, "AKPRST01", 8) != 0) CFail("no particle section where one was expected");
    std::int32_t h[6];
    std::memcpy(h, head + 8, sizeof(h));
    std::uint64_t total = 0;
    std::memcpy(&total, head + 32, sizeof(total));
    if (nsec < 0) nsec = h[3];
    if ((h[0] != 1 && h[0] != 2) || h[1] != static_cast<std::int32_t>(sizeof(Real)) ||
        h[2] != static_cast<std::int32_t>(sizeof(int)) || h[3] != nsec || h[4] != nrdata ||
        (h[5] != nidata && h[5] != nidata - 1)) {
      CFail("the particle columns of the file do not match this run");
    }
    const int nif = h[5];
    std::int64_t np = 0;
    std::int32_t nstate = 0;
    ReadAt(f, secpos + 40, &np, sizeof(np), "particle count");
    ReadAt(f, secpos + 40 + sizeof(np) + sizeof(Real), &nstate, sizeof(nstate), "state size");
    if (np < 0 || nstate < 0) CFail("inconsistent particle section");
    const std::uint64_t st0 = secpos + 40 + sizeof(np) + sizeof(Real) + sizeof(nstate);
    const std::uint64_t rows = st0 + static_cast<std::uint64_t>(nstate);
    if (total != rows - secpos + static_cast<std::uint64_t>(np)*(nrdata*sizeof(Real) + nif*sizeof(int)) + 8) {
      CFail("inconsistent particle section size");
    }
    // the dust records: the species table and the tag bound
    {
      std::vector<char> st(static_cast<std::size_t>(nstate));
      if (nstate > 0) ReadAt(f, st0, st.data(), st.size(), "dust records");
      std::size_t a = 0;
      while (a + 20 <= st.size()) {
        char key[17] = {0};
        std::memcpy(key, st.data() + a, 16);
        std::int32_t n = 0;
        std::memcpy(&n, st.data() + a + 16, sizeof(n));
        if (n < 0 || a + 20 + static_cast<std::size_t>(n) > st.size()) CFail("broken dust record");
        const char *d = st.data() + a + 20;
        const std::string name(key);
        if (name == "species_table") {
          species_seen = true;
          const std::string diff = dust::CompareSpeciesRecords(std::string(d, n), pd->species_record);
          if (!diff.empty()) species_diff = diff;
        } else if (name == "tag_state" && n == static_cast<std::int32_t>(3*sizeof(std::uint64_t))) {
          std::uint64_t v[3];
          std::memcpy(v, d, sizeof(v));
          tag_bound = std::max(tag_bound, std::max(v[2], v[0] ? v[1] : 0));
        } else if (name == "next_tag" && n == static_cast<std::int32_t>(sizeof(std::int32_t))) {
          std::int32_t v;
          std::memcpy(&v, d, sizeof(v));
          if (v > 0) tag_bound = std::max(tag_bound, static_cast<std::uint64_t>(v));
        }
        a += 20 + static_cast<std::size_t>(n);
      }
    }
    if (!species_diff.empty()) {
      CFail("the dust species of the input differ from those of the file:\n" + species_diff);
    }
    const std::size_t n = static_cast<std::size_t>(np);
    auto rcol = [&](int c, std::vector<Real> &v) {
      v.resize(n);
      if (n > 0) ReadAt(f, rows + static_cast<std::uint64_t>(c)*n*sizeof(Real), v.data(), n*sizeof(Real), "particle rows");
    };
    auto icol = [&](int c, std::vector<int> &v) {
      v.resize(n);
      if (c >= nif) {std::fill(v.begin(), v.end(), 0); return;}   // PTAGH of version 1: 0
      if (n > 0) {
        ReadAt(f, rows + static_cast<std::uint64_t>(nrdata)*n*sizeof(Real) +
                  static_cast<std::uint64_t>(c)*n*sizeof(int), v.data(), n*sizeof(int), "particle integer rows");
      }
    };
    std::vector<Real> X, Y, Z, M, col;
    std::vector<int> lev, icl;
    rcol(IPX, X); rcol(IPY, Y); rcol(IPZ, Z); rcol(IPM, M);
    icol(PLEV, lev);
    std::vector<std::uint64_t> hp(n, 0);   // the particle hashes (file side)
    std::vector<char> live(n, 1);
    for (std::size_t p=0; p<n; ++p) {
      if (!(M[p] > 0.0)) {live[p] = 0; ++ntomb; continue;}   // tombstones
      if (lev[p] != 0) ++nlev;
      if (X[p] >= ms.x1max) {X[p] -= lx; Y[p] += ysh; ++nwrap;}
      else if (X[p] < ms.x1min) {X[p] += lx; Y[p] -= ysh; ++nwrap;}
      if (Y[p] >= ms.x2max || Y[p] < ms.x2min) {
        ++nwrap;
        while (Y[p] >= ms.x2max) Y[p] -= ly;
        while (Y[p] < ms.x2min) Y[p] += ly;
      }
      if (Z[p] < ms.x3min || Z[p] >= ms.x3max) ++nz;
    }
    if (nlev > 0) CFail(std::to_string(nlev) + " particles of the file are not plain level-0 (PLEV != 0)");
    if (nz > 0) CFail(std::to_string(nz) + " particles of the file lie beyond the x3 faces");
    // the block of every particle: the leaf owner of the 3-D traversal (the root parents of a
    // fresh start, PlaceParticles); slot < nmb: a block of this rank
    std::vector<int> slot(n, -1);
    if (n > 0) {
      DvceArray2D<Real> xd("clouds_from_xyz", 3, n);
      auto xh = Kokkos::create_mirror_view(xd);
      for (std::size_t p=0; p<n; ++p) {xh(0,p) = X[p]; xh(1,p) = Y[p]; xh(2,p) = Z[p];}
      Kokkos::deep_copy(xd, xh);
      DvceArray1D<int> sd("clouds_from_slot", n);
      Kokkos::parallel_for("clouds_from_owner", Kokkos::RangePolicy<>(DevExeSpace(), 0, static_cast<int>(n)),
      KOKKOS_LAMBDA(const int p) {
        const int leaf = lt3.LeafOwner(xd(0,p), xd(1,p), xd(2,p), 0);
        sd(p) = (leaf < 0) ? -1 : leaf/ncell123;
      });
      auto sh = Kokkos::create_mirror_view_and_copy(HostMemSpace(), sd);
      for (std::size_t p=0; p<n; ++p) slot[p] = sh(p);
    }
    std::vector<std::size_t> sel;
    for (std::size_t p=0; p<n; ++p) {
      if (!live[p]) continue;
      if (slot[p] < 0) {++nnoowner; continue;}
      if (slot[p] < nmb) sel.push_back(p);
    }
    if (sct == 0) {   // about np nsec / nranks particles per rank
      const std::size_t est = static_cast<std::size_t>(1.1*np*nsec/global_variable::nranks) + 1024;
      for (auto &c : rc) c.reserve(est);
      for (auto &c : ic) c.reserve(est);
    }
    const std::size_t q0 = rc[0].size();
    for (int c=0; c<nrdata; ++c) rc[c].resize(q0 + sel.size(), 0.0);
    for (int c=0; c<nidata; ++c) ic[c].resize(q0 + sel.size(), 0);
    auto keep_r = [&](int c, const std::vector<Real> &v) {
      for (std::size_t s=0; s<sel.size(); ++s) rc[c][q0 + s] = v[sel[s]];
    };
    auto keep_i = [&](int c, const std::vector<int> &v) {
      for (std::size_t s=0; s<sel.size(); ++s) ic[c][q0 + s] = v[sel[s]];
    };
    // positions (wrapped) and their RK copies, mass
    keep_r(IPX, X); keep_r(IPX1, X); keep_r(IPY, Y); keep_r(IPY1, Y); keep_r(IPZ, Z); keep_r(IPZ1, Z);
    keep_r(IPM, M);
    for (std::size_t p=0; p<n; ++p) {
      if (!live[p]) continue;
      hp[p] = Mix(Mix(Mix(hp[p], Bits(X[p])), Bits(Y[p])), Bits(Z[p]));
      hp[p] = Mix(hp[p], Bits(M[p]));
      pm_s.Add(M[p]);
    }
    X.clear(); X.shrink_to_fit(); Y.clear(); Y.shrink_to_fit(); Z.clear(); Z.shrink_to_fit();
    // velocities and their RK copies, stopping time; the drag rates stay zero
    const int vc[3] = {IPVX, IPVY, IPVZ}, vc1[3] = {IPVX1, IPVY1, IPVZ1};
    for (int d=0; d<3; ++d) {
      rcol(vc[d], col);
      keep_r(vc[d], col); keep_r(vc1[d], col);
      for (std::size_t p=0; p<n; ++p) {
        if (!live[p]) continue;
        hp[p] = Mix(hp[p], Bits(col[p]));
        pp_s[d].Add(M[p]*col[p]);
        vmax = std::max(vmax, std::fabs(col[p]));
      }
    }
    rcol(IPTS, col);
    keep_r(IPTS, col);
    for (std::size_t p=0; p<n; ++p) if (live[p]) hp[p] = Mix(hp[p], Bits(col[p]));
    // tag (both words), species; level 0 (checked); the block of this rank
    {
      std::vector<int> lo, hi;
      icol(PTAG, lo); icol(PTAGH, hi);
      keep_i(PTAG, lo); keep_i(PTAGH, hi);
      std::uint64_t tmax = 0;
      for (std::size_t p=0; p<n; ++p) {
        if (!live[p]) continue;
        const std::uint64_t t = TagFromWords(lo[p], hi[p]);
        tmax = std::max(tmax, t);
        hp[p] = Mix(hp[p], t);
      }
      if (n > 0) tag_bound = std::max(tag_bound, tmax + 1);
    }
    icol(PSP, icl);
    keep_i(PSP, icl);
    for (std::size_t p=0; p<n; ++p) {
      if (!live[p]) continue;
      hp[p] = Mix(hp[p], static_cast<std::uint64_t>(static_cast<std::uint32_t>(icl[p])));
      phash += hp[p];
      ++nfile;
    }
    for (std::size_t s=0; s<sel.size(); ++s) {
      ic[PGID][q0 + s] = pmbp->gids + slot[sel[s]];
      ic[PLEV][q0 + s] = 0;
    }
    secpos += total;
    if (sct + 1 >= nsec) break;
  }
  // a file without particles (all dust left through the x3 faces): nothing to convert.  Rejected
  // here, before any cloud state: the root cloud mass of G and R and the tag counter are taken from
  // the clouds (EnsureRootCloudMass would reduce over none), and a refinable-cloud run without dust
  // has no use.  The gas-only continuation is the native restart of the file (-r), which takes an
  // empty particle population.  (nfile: the live particles of every section, the same on every rank)
  if (nfile == 0) {
    CFailAll("the file has no particles (" + std::to_string(ntomb) + " tombstones): nothing to convert "
             "into clouds; continue the run with a native restart (-r) of the file instead");
  }
  if (nnoowner > 0) CFail(std::to_string(nnoowner) + " particles have no owning block in this mesh");
  const std::int64_t nkeep = rc[0].size();
  if (nkeep > std::numeric_limits<int>::max()) CFail("too many particles on one rank");
  // the clouds of this rank: count, mass, momentum, hash (the same fields as the file side)
  KSum km_s, kp_s[3];
  std::uint64_t khash = 0;
  for (std::int64_t q=0; q<nkeep; ++q) {
    std::uint64_t hq = 0;
    hq = Mix(Mix(Mix(hq, Bits(rc[IPX][q])), Bits(rc[IPY][q])), Bits(rc[IPZ][q]));
    hq = Mix(hq, Bits(rc[IPM][q]));
    hq = Mix(hq, Bits(rc[IPVX][q])); hq = Mix(hq, Bits(rc[IPVY][q])); hq = Mix(hq, Bits(rc[IPVZ][q]));
    hq = Mix(hq, Bits(rc[IPTS][q]));
    hq = Mix(hq, TagFromWords(ic[PTAG][q], ic[PTAGH][q]));
    hq = Mix(hq, static_cast<std::uint64_t>(static_cast<std::uint32_t>(ic[PSP][q])));
    khash += hq;
    km_s.Add(rc[IPM][q]);
    kp_s[0].Add(rc[IPM][q]*rc[IPVX][q]); kp_s[1].Add(rc[IPM][q]*rc[IPVY][q]);
    kp_s[2].Add(rc[IPM][q]*rc[IPVZ][q]);
  }
  // capacity with a quarter of headroom (as a restart of refinable clouds)
  const std::int64_t cap = std::min<std::int64_t>(nkeep + nkeep/4, std::numeric_limits<int>::max());
  Kokkos::realloc(ppar->prtcl_rdata, nrdata, std::max<std::int64_t>(cap, 1));
  Kokkos::realloc(ppar->prtcl_idata, nidata, std::max<std::int64_t>(cap, 1));
  {
    auto prh = Kokkos::create_mirror_view(ppar->prtcl_rdata);
    auto pih = Kokkos::create_mirror_view(ppar->prtcl_idata);
    Kokkos::deep_copy(prh, 0.0);
    Kokkos::deep_copy(pih, 0);
    for (int c=0; c<nrdata; ++c) for (std::int64_t q=0; q<nkeep; ++q) prh(c,q) = rc[c][q];
    for (int c=0; c<nidata; ++c) for (std::int64_t q=0; q<nkeep; ++q) pih(c,q) = ic[c][q];
    Kokkos::deep_copy(ppar->prtcl_rdata, prh);
    Kokkos::deep_copy(ppar->prtcl_idata, pih);
  }
  rc.clear(); ic.clear();
  ppar->nprtcl_thispack = static_cast<int>(nkeep);
  pm->nprtcl_thisrank = static_cast<int>(nkeep);
  for (int r=0; r<global_variable::nranks; ++r) pm->nprtcl_eachrank[r] = 0;
  pm->nprtcl_eachrank[me] = static_cast<int>(nkeep);
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pm->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  // the clouds of all ranks: 64-bit sum of the int counts of the ranks
  const std::int64_t ntot = particles::TotalCount(pm->nprtcl_eachrank, global_variable::nranks);
  if (static_cast<std::uint64_t>(ntot) != nfile) {
    CFailAll("the particles of the file do not fall into the blocks of this run (" + std::to_string(ntot) +
          " kept, " + std::to_string(nfile) + " in the file)");
  }
  pm->nprtcl_total = ntot;
  const double pmass = pm_s.Value(), kmass = SumD(km_s.Value());
  double pmom[3], kmom[3];
  for (int d=0; d<3; ++d) {pmom[d] = pp_s[d].Value(); kmom[d] = SumD(kp_s[d].Value());}
  khash = SumU(khash);
  tag_bound = MaxU(tag_bound);

  // ---- cloud state as at a fresh start: tags, root cloud mass, split to the leaves, dt
  pd->cloud_tag_floor = tag_bound;
  pd->EnsureNextTag();
  pd->EnsureRootCloudMass();
  const long long nsplit = pd->SplitToLeaves();
  SetParticleDt(pm, {}, {}, vmax);

  // ---- li2018: the totals of the input, else those of the file
  for (const char *k : {"vbc_mean0", "vbc_dm"}) {
    if (pin->DoesParameterExist("problem", k)) continue;
    const std::string v = FileParam(text, "problem", k);
    if (!v.empty()) pin->SetString("problem", k, v);
  }

  if (me == 0) {
    std::cout << std::setprecision(17)
              << "# clouds_from " << path << ": t = " << time_c << " (cycle " << ncyc_c << "), "
              << mi_c.nx1 << " x " << mi_c.nx2 << " x " << mi_c.nx3 << " in " << nmb_c << " MeshBlocks of "
              << mb_c.nx1 << " x " << mb_c.nx2 << " x " << mb_c.nx3 << " -> " << pm->nmb_total
              << " root MeshBlocks of " << indcs.nx1 << " x " << indcs.nx2 << " x " << indcs.nx3 << "\n"
              << "#   gas mass " << cmass << " -> " << fmass << " (rel " << (fmass - cmass)/cmass
              << "), momentum " << cmom[0] << " " << cmom[1] << " " << cmom[2] << " -> " << fmom[0] << " "
              << fmom[1] << " " << fmom[2] << "\n"
              << "#   particles " << nfile << " in " << nsec << " section(s) -> " << ntot
              << " level-0 clouds; dust mass " << pmass << " -> " << kmass << " (rel " << (kmass - pmass)/pmass
              << "), momentum " << pmom[0] << " " << pmom[1] << " " << pmom[2] << " -> " << kmom[0] << " "
              << kmom[1] << " " << kmom[2] << "\n"
              << "#   particle hash (tag, species, x, v, tau_s, m) " << phash << " -> " << khash
              << (phash == khash ? " (equal)" : " (DIFFERENT)") << "; wrapped " << nwrap << ", tombstones "
              << ntomb << "; tag bound " << tag_bound << ", next tag " << pd->cloud_next_tag
              << "; root cloud mass";
    for (const Real mr : pd->cloud_merge_mroot) std::cout << " " << mr;
    std::cout << "; split at the start " << nsplit << "\n";
    if (!species_seen) std::cout << "#   (the file has no record species_table: species not checked)\n";
    for (const char *k : {"vbc_mean0", "vbc_dm"}) {
      if (pin->DoesParameterExist("problem", k)) {
        std::cout << "#   <problem>/" << k << " = " << pin->GetString("problem", k) << "\n";
      }
    }
    std::cout << std::flush;
  }
  if (phash != khash) CFail("the clouds are not the particles of the file (hash)");
}

}  // namespace dust_setup
