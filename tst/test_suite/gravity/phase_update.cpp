// Regression for the shear-phase operator updates.  Build the target
// gravity_phase_update_check explicitly; run outside the source tree, optionally under MPI.
// No files are written.
//  1. consecutive fine updates through the public entries (each one queues pinned-stage uploads):
//     after many alternating phases, the fine operator and the V-cycle equal those of a solver
//     updated once to the last phase (bitwise)
//  2. a batch whose coarse matrix, or whose extra (force) operator, is invalid throws on every
//     rank and leaves every operator as it was (bitwise); the solver is then updated validly and
//     equals a fresh solver given the same updates (bitwise)
//  3. ATHENA_SG_PROJECT_FUSE / the Allreduce probe: with EXPECT_FUSION=0|1 in the environment the
//     solver's ProjectionFusion() must match (run once without ATHENA_SG_PROJECT_FUSE, once with
//     it, once with it and ATHENA_SG_PROJECT_FUSE_PROBE_TEST=1, which must refuse); the V-cycle
//     result is printed as a checksum so that runs with and without fusion can be compared
//  4. the completion of the queued uploads (Codex review round 3): every public update that
//     returns normally ends with the fence "gravity_phase_uploads", which may throw (Kokkos throws
//     on a failed CUDA/HIP synchronization); an update that throws ends with the non-throwing
//     fence "gravity_phase_uploads_unwind".  A Kokkos Tools fence callback counts both and, when
//     armed, throws from them as a failed device synchronization would: the error of the normal
//     fence must reach the caller, the error of the unwinding fence must not replace the update's
//     own exception (with ATHENA_SG_PHASE_ALLOC_OLD=1, blocking copies: no fence, none counted)
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gravity/geometric_mg.hpp"
#include "gravity/vacuum_mesh.hpp"
#include <Kokkos_Profiling_ScopedRegion.hpp>   // (Kokkos::Tools::Experimental fence callbacks)

namespace {
using namespace gravity;
using Vector = DvceArray1D<Real>;
void Check(bool condition, const std::string &message) {
  if (!condition) throw std::runtime_error(message);
}
int Min(int value) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&value,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
#endif
  return value;
}
// the bits of a vector on the host
std::vector<std::uint64_t> Bits(const Vector &v) {
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(),v);
  std::vector<std::uint64_t> out(h.extent(0));
  for (size_t i = 0; i < out.size(); ++i) std::memcpy(&out[i],&h(i),sizeof(Real));
  return out;
}
std::uint64_t Hash(const std::vector<std::uint64_t> &bits) {
  std::uint64_t h = 1469598103934665603ull;
  for (auto b : bits) { h ^= b; h *= 1099511628211ull; }
#if MPI_PARALLEL_ENABLED
  std::uint64_t all = 0;
  MPI_Allreduce(&h,&all,1,MPI_UINT64_T,MPI_BXOR,MPI_COMM_WORLD);
  return all;
#else
  return h;
#endif
}

struct Setup {
  CompositeDomain domain;
  GeometricSystem system;
  std::vector<std::vector<int>> rows;
};
std::unique_ptr<Setup> MakeSetup(int rank, int ranks) {
  auto s = std::make_unique<Setup>();
  CompositeDomain domain;
  domain.ndim = 3; domain.root_counts = {16,16,16};
  domain.hi = {1,1,1}; domain.isolated_z = true;
  domain.shear_phase_updates = true;
  std::vector<GravityPatch> patches;
  for (int z = 0; z < 2; ++z) for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
    const int refine = x == 0 && y == 0 && z == 0 ? 2 : 1;
    for (int k = 0; k < refine; ++k) for (int j = 0; j < refine; ++j)
      for (int i = 0; i < refine; ++i) {
        GravityPatch p;
        p.lo = {(x+Real(i)/refine)/2,(y+Real(j)/refine)/2,(z+Real(k)/refine)/2};
        p.hi = {p.lo[0]+Real(1)/(2*refine),p.lo[1]+Real(1)/(2*refine),
                p.lo[2]+Real(1)/(2*refine)};
        p.cells = {8,8,8}; p.gid = patches.size(); p.owner = p.gid%ranks;
        patches.push_back(p);
      }
  }
  auto extended = BuildVacuumMesh(patches,domain,ranks);
  s->domain = extended.domain;
  s->system = BuildGeometricHierarchy(extended.patches,s->domain,rank,ranks,true,true);
  s->rows = {s->system.fine_geometry->PhaseRows()};
  for (auto &g : s->system.coarse_geometry) s->rows.push_back(g->PhaseRows());
  return s;
}
struct Updates {
  std::vector<CsrRowUpdate> a, p;
  CsrRowUpdate force;   // the x1 acceleration rows
};
Updates MakeUpdates(Setup &s, Real phase, int rank) {
  Updates u;
  s.domain.shear_offset = phase;
  auto fine = s.system.fine_geometry->BuildPhaseUpdate(phase,true);
  u.a.push_back(fine.matrix);
  u.force = fine.acceleration[0];
  for (auto &g : s.system.coarse_geometry) u.a.push_back(g->BuildPhaseUpdate(phase,false).matrix);
  for (size_t l = 0; l < s.system.coarse_geometry.size(); ++l) {
    auto &f = l == 0 ? *s.system.fine_geometry : *s.system.coarse_geometry[l-1];
    u.p.push_back(BuildProlongationPhaseUpdate(f,*s.system.coarse_geometry[l],s.domain,rank,
                                               s.system.prolongation_phase_rows[l]));
  }
  return u;
}
struct Solver {
  std::unique_ptr<SparseMultigrid> mg;
  std::unique_ptr<DistributedOperator> force;
  Solver(Setup &s) {
    mg = std::make_unique<SparseMultigrid>(s.system.hierarchy);
    mg->ConfigureHierarchyPhaseRows(s.rows,s.system.prolongation_phase_rows);
    force = std::make_unique<DistributedOperator>(s.system.fine.acceleration[0],
        s.system.fine.row_partition,s.system.fine.row_partition);
    force->ConfigurePhaseRows(s.rows[0]);
  }
  void Batch(Updates &u) {
    std::vector<SparseMultigrid::PhaseExtra> extra{{force.get(),&u.force,nullptr}};
    mg->UpdateHierarchyPhaseRows(u.a,u.p,nullptr,nullptr,&extra);
  }
  // the fine product, the V-cycle and the force product of fixed inputs, as bits
  std::vector<std::uint64_t> State() {
    const int n = mg->LocalSize();
    Vector x("x",n),y("y",n),z("z",n),f("f",n);
    auto h = Kokkos::create_mirror_view(x);
    for (int i = 0; i < n; ++i) h(i) = std::sin(Real(0.37)*(mg->LocalBegin()+i)+Real(0.1));
    Kokkos::deep_copy(x,h);
    mg->ApplyMatrix(0,x,y);
    mg->ApplyVCycle(y,z);
    force->Apply(x,f);
    auto out = Bits(y);
    for (auto b : Bits(z)) out.push_back(b);
    for (auto b : Bits(f)) out.push_back(b);
    return out;
  }
};
void Same(const std::vector<std::uint64_t> &a, const std::vector<std::uint64_t> &b,
          const std::string &what) {
  int same = a == b ? 1 : 0;
  Check(Min(same) == 1,what+": results differ (bitwise)");
}

void ConsecutiveUpdates(Setup &s, int rank) {
  Solver once(s), many(s);
  auto u0 = MakeUpdates(s,Real(0),rank);
  once.Batch(u0); many.Batch(u0);
  const Real phases[] = {Real(-.0625+1.e-9),Real(-.3),Real(-.0625),Real(-1.02),Real(-.71)};
  // public fine entries back to back (the coarse levels stay at phase 0 in both)
  for (int round = 0; round < 4; ++round) {
    for (Real phase : phases) {
      auto u = MakeUpdates(s,phase,rank);
      if (round%2 == 0) many.mg->UpdateFinePhaseRowsKnown(u.a[0],nullptr,nullptr);
      else many.mg->UpdateFinePhaseRows(u.a[0]);
    }
  }
  auto last = MakeUpdates(s,phases[4],rank);
  once.mg->UpdateFinePhaseRowsPlanned(last.a[0],nullptr);
  Same(once.State(),many.State(),"consecutive fine updates");
  // and whole batches back to back
  for (Real phase : phases) { auto u = MakeUpdates(s,phase,rank); many.Batch(u); }
  once.Batch(last);
  Same(once.State(),many.State(),"consecutive batches");
  if (rank == 0) std::cout << "consecutive phase updates PASS\n";
}

void FailedBatch(Setup &s, int rank) {
  Solver used(s), fresh(s);
  auto u0 = MakeUpdates(s,Real(0),rank), u1 = MakeUpdates(s,Real(-.3),rank);
  used.Batch(u0); fresh.Batch(u0);
  const auto before = used.State();
  // an invalid coarse matrix (or prolongation), then an invalid extra operator, each on ONE rank
  // only (the lowest rank that has such phase rows): the reduced flag must stop every rank, and
  // the fine level (updated first) must not have been touched either
  for (int which = 0; which < 2; ++which) {
    auto bad = MakeUpdates(s,Real(-.3),rank);
    std::vector<Real> *target = nullptr;
    if (which == 0) {
      for (size_t l = bad.a.size(); l-- > 1 && target == nullptr;)
        if (!bad.a[l].values.empty()) target = &bad.a[l].values;
      for (size_t l = bad.p.size(); l-- > 0 && target == nullptr;)
        if (!bad.p[l].values.empty()) target = &bad.p[l].values;
    } else if (!bad.force.values.empty()) {
      target = &bad.force.values;
    }
    const int chosen = Min(target != nullptr ? rank : std::numeric_limits<int>::max());
    Check(chosen != std::numeric_limits<int>::max(),"no rank has rows to corrupt");
    if (rank == chosen) (*target)[0] = std::numeric_limits<Real>::quiet_NaN();
    bool threw = false;
    try { used.Batch(bad); } catch (const std::runtime_error &) { threw = true; }
    Check(Min(threw ? 1 : 0) == 1,"an invalid batch did not throw on every rank");
    // (with ATHENA_SG_PHASE_FORCE_OLD=1 the extra operator keeps its own collectives after the
    // hierarchy update, as before the batching: only the throw and the reuse are checked then)
    const char *force_old = std::getenv("ATHENA_SG_PHASE_FORCE_OLD");
    if (which == 0 || force_old == nullptr || force_old[0] != '1') {
      Same(before,used.State(),which == 0 ? "invalid coarse matrix modified the solver"
                                           : "invalid extra operator modified the solver");
    }
  }
  used.Batch(u1); fresh.Batch(u1);
  Same(fresh.State(),used.State(),"reuse after a failed batch");
  if (rank == 0) std::cout << "failed batch and reuse PASS\n";
}

void Fusion(Setup &s, int rank) {
  const char *e = std::getenv("EXPECT_FUSION");
  Solver solver(s);
  auto u = MakeUpdates(s,Real(-.3),rank);
  solver.Batch(u);
  const bool fused = solver.mg->ProjectionFusion();
  if (e != nullptr) Check(fused == (e[0] == '1'),"ProjectionFusion() is not as expected");
  const auto h = Hash(solver.State());
  if (rank == 0) std::cout << "projection fusion " << (fused ? "on" : "off")
                           << " state hash " << std::hex << h << std::dec << "\n";
}

// the completion fences of the public updates, counted (and, when armed, failing) through the
// Kokkos Tools begin-fence callback
struct FenceProbe {
  struct Error : std::runtime_error { using std::runtime_error::runtime_error; };
  static inline int normal = 0, unwind = 0;
  static inline bool arm = false;
  static void Begin(const char *name, const std::uint32_t, std::uint64_t *) {
    const bool n = std::strcmp(name,"gravity_phase_uploads") == 0;
    const bool u = std::strcmp(name,"gravity_phase_uploads_unwind") == 0;
    normal += n ? 1 : 0; unwind += u ? 1 : 0;
    if (arm && (n || u)) throw Error(std::string("simulated device error at fence ")+name);
  }
  static void Reset() { normal = unwind = 0; }
};
void CompletionFence(Setup &s, int rank) {
  const char *old = std::getenv("ATHENA_SG_PHASE_ALLOC_OLD");
  const bool fences = !(old != nullptr && old[0] == '1');
  Solver solver(s);
  auto u0 = MakeUpdates(s,Real(0),rank), u1 = MakeUpdates(s,Real(-.3),rank);
  solver.Batch(u0);
  Kokkos::Tools::Experimental::set_begin_fence_callback(&FenceProbe::Begin);
  // (a) a normal return: one completion fence per public entry, no unwinding fence
  struct Entry { const char *what; std::function<void()> call; };
  const std::vector<Entry> entries{
      {"UpdateFinePhaseRows",[&] { solver.mg->UpdateFinePhaseRows(u1.a[0]); }},
      {"UpdateFinePhaseRowsKnown",
       [&] { solver.mg->UpdateFinePhaseRowsKnown(u0.a[0],nullptr,nullptr); }},
      {"UpdateFinePhaseRowsPlanned",
       [&] { solver.mg->UpdateFinePhaseRowsPlanned(u1.a[0],nullptr); }},
      {"DistributedOperator::UpdatePhaseRows",[&] { solver.force->UpdatePhaseRows(u1.force); }},
      {"UpdateHierarchyPhaseRows",[&] { solver.Batch(u1); }}};
  for (const auto &e : entries) {
    FenceProbe::Reset();
    e.call();
    Check(fences ? FenceProbe::normal >= 1 && FenceProbe::unwind == 0
                 : FenceProbe::normal == 0 && FenceProbe::unwind == 0,
          std::string(e.what)+": unexpected completion fences (normal "
          +std::to_string(FenceProbe::normal)+", unwind "+std::to_string(FenceProbe::unwind)+")");
  }
  // (b) armed: the failure of the normal completion reaches the caller of every entry
  FenceProbe::arm = true;
  for (const auto &e : entries) {
    bool reported = false;
    try { e.call(); } catch (const FenceProbe::Error &) { reported = true; }
    Check(Min(reported == fences ? 1 : 0) == 1,
          std::string(e.what)+(fences ? ": a failed completion fence was not reported"
                                      : ": a fence ran with ATHENA_SG_PHASE_ALLOC_OLD=1"));
  }
  // (c) armed: a batch that throws itself (one rank's coarse matrix invalid) reports its own
  // error; the unwinding fence ran and its (simulated) failure was swallowed
  auto bad = MakeUpdates(s,Real(-.71),rank);
  std::vector<Real> *target = nullptr;
  for (size_t l = bad.a.size(); l-- > 1 && target == nullptr;)
    if (!bad.a[l].values.empty()) target = &bad.a[l].values;
  const int chosen = Min(target != nullptr ? rank : std::numeric_limits<int>::max());
  Check(chosen != std::numeric_limits<int>::max(),"no rank has coarse rows to corrupt");
  if (rank == chosen) (*target)[0] = std::numeric_limits<Real>::quiet_NaN();
  FenceProbe::Reset();
  int own = 0;
  try {
    solver.Batch(bad);
  } catch (const FenceProbe::Error &) {
    own = -1;   // the unwinding fence's error replaced the update's own
  } catch (const std::runtime_error &) {
    own = 1;
  }
  FenceProbe::arm = false;
  Kokkos::Tools::Experimental::set_begin_fence_callback(nullptr);
  Check(Min(own) == 1,"an invalid batch did not report its own error on every rank");
  Check(Min(fences ? (FenceProbe::unwind >= 1 && FenceProbe::normal == 0 ? 1 : 0)
                   : (FenceProbe::unwind == 0 && FenceProbe::normal == 0 ? 1 : 0)) == 1,
        "a throwing batch did not end with (only) the unwinding fence");
  if (rank == 0) std::cout << "completion fences PASS (" << (fences ? "fenced" : "no fences: "
                           "ATHENA_SG_PHASE_ALLOC_OLD") << ")\n";
}
}  // namespace

int main(int argc, char **argv) {
#if MPI_PARALLEL_ENABLED
  int provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
  Check(provided >= MPI_THREAD_FUNNELED,"MPI thread support unavailable");
#endif
  Kokkos::initialize(argc,argv);
  int rank = 0, ranks = 1, status = 0;
#if MPI_PARALLEL_ENABLED
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
#endif
  try {
    auto setup = MakeSetup(rank,ranks);
    ConsecutiveUpdates(*setup,rank);
    FailedBatch(*setup,rank);
    Fusion(*setup,rank);
    CompletionFence(*setup,rank);
    if (rank == 0) std::cout << "gravity phase-update checks PASS\n";
  } catch (const std::exception &error) {
    std::cerr << "rank " << rank << ": " << error.what() << '\n'; status = 1;
#if MPI_PARALLEL_ENABLED
    MPI_Abort(MPI_COMM_WORLD,status);
#endif
  }
  Kokkos::finalize();
#if MPI_PARALLEL_ENABLED
  MPI_Finalize();
#endif
  return status;
}
