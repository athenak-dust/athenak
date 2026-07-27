# Dust GPU Optimization Phase 1: Handoff and Resume Guide

## Snapshot

- Repository: `/work/10737/clolds/vista/athenak_dust/athenak`
- Branch: `dust_optimized`
- Starting commit: `d6de1296`
- Handoff date: 2026-07-27
- Primary target: efficient dust calculations on many GPUs with many particles and
  high-resolution meshes
- Phase requested: easy, low-risk optimizations first; normal floating-point tolerance is
  acceptable

The branch was verified as `dust_optimized` before any source file was changed. No commit was
created by Codex. At the time this document was written, the twelve modified source files were
staged but uncommitted. Always check the current index and working tree rather than assuming that
state is unchanged.

## Phase 1 scope

Phase 1 removes avoidable work, memory, synchronization, and atomics without changing the
high-level dust algorithm or MPI decomposition. It deliberately does not attempt a new
distributed particle-routing design, particle binning, shared-memory deposition, or
communication/computation overlap.

The implementation touches these areas:

1. particle migration task scheduling and storage;
2. particle-mesh deposition and gather kernels;
3. coupled drag-solver workspaces and operator kernels;
4. particle VTK and tracking output.

## Implemented changes

### 1. Particle migration

- Added `ActiveMigrationStage()` in `src/particles/particles_tasks.cpp`.
  For the `imex_dust` pusher with the `imex2+` integrator, particle migration tasks now skip the
  dormant third explicit stage. This avoids a full particle scan and MPI bookkeeping at a stage
  that cannot move particles.
- Added single-rank fast paths. With `nranks == 1`, local particle GIDs and periodic positions are
  still updated, but MPI metadata construction, device append counters, device-to-host counter
  copies, sends, and receives are bypassed.
- Removed the per-stage `MPI_Allgather` of particle counts. Global rank counts are refreshed only
  when particle output actually needs them.
- Removed the unused `dest_gid` member from each particle send-list record.
- Added persistent `ParticleMoveData`/`compactlist` storage.
- Particle arrays now retain capacity and grow geometrically by approximately 1.5 times. They no
  longer shrink to the exact logical particle count after migration.
- Replaced repeated per-hole deep copies during particle compaction with a host-built move list
  followed by one device compaction kernel.

### 2. Particle output

- `src/outputs/vtk_prtcl.cpp` gathers per-rank particle counts only when VTK output is loaded.
- Output copies only the logical particle prefix instead of the entire retained-capacity view.
- `src/outputs/track_prtcl.cpp` now computes `npout_min` from output counts, not unrelated particle
  counts.
- Zero-particle output buffers allocate a safe minimum extent of one.

### 3. Particle-mesh kernels

- Added compile-time particle-mesh scheme helpers in `src/dust/dust.hpp`:
  `PMGridIndex`, `PMStencilWidth`, and `PMCompactWeights`.
- NGP, CIC, and TSC are dispatched on the host to specialized GPU kernels.
- Two-dimensional and three-dimensional paths are also specialized on the host.
- NGP and CIC kernels now visit only their true compact support. They no longer issue reads or
  atomics for stencil entries whose weight is known to be zero.
- Deposition, gather/kick/back-reaction, matrix-free `ApplyA`, and adaptive-solver particle
  operations all use the same compact weight helper.
- Reciprocal volume and reciprocal implicit timestep values are computed once outside particle
  loops and reused.
- The stage-one particle Runge-Kutta register copy was fused into `ExplicitPush`, removing a
  separate particle kernel.

Using the same weight helper for the gather operator and its transpose scatter is important for
momentum conservation and for keeping the coupled operator symmetric positive definite.

### 4. Solver memory and kernels

- Full `qdep` and `dmom` fields and their boundary-exchange objects are allocated only when
  `back_reaction=true`.
- `solver_r` and `solver_p` are allocated only for PCG and adaptive solver modes.
- For back-reacting nonlocal solvers, `solver_ap` is a shallow alias of `dmom` rather than another
  full mesh-sized allocation.
- DC1 and DC2 compute their residual in a local scalar/register instead of requiring
  `solver_r`.
- `ApplyCoupledOperator` now initializes the complete active-plus-ghost output field in one
  kernel:
  active cells receive `rho * field`, and ghost cells receive zero. This replaces a full
  deep-copy zero plus another active-cell kernel.
- Optional deposit boundary objects are null-checked in task cleanup.
- Added `NeedsPostSolveUstarExchange()` so send, receive, and receive initialization make the same
  decision. A redundant post-solve exchange is skipped when the final `ApplyA` already populated
  `ustar` ghosts.

### 5. Files changed

| File | Main responsibility |
| --- | --- |
| `src/bvals/bvals.cpp` | Particle boundary setup and persistent storage initialization |
| `src/bvals/bvals.hpp` | Compact migration record and move-list definitions |
| `src/bvals/bvals_part.cpp` | Single-rank paths, geometric capacity, and one-kernel compaction |
| `src/dust/dust.cpp` | Conditional allocation and solver workspace aliasing |
| `src/dust/dust.hpp` | Compact particle-mesh weight helpers and workspace declarations |
| `src/dust/dust_pm.cpp` | Specialized compact deposit and gather/kick kernels |
| `src/dust/dust_push.cpp` | Fused stage-one particle copy |
| `src/dust/dust_solver.cpp` | Specialized matrix-free kernels and reduced workspace use |
| `src/dust/dust_tasks.cpp` | Matched and conditional `ustar` communication |
| `src/outputs/track_prtcl.cpp` | Tracking-count and empty-buffer fixes |
| `src/outputs/vtk_prtcl.cpp` | Output-time count gathering and logical-prefix copies |
| `src/particles/particles_tasks.cpp` | Dormant migration-stage gating |

The source patch was 796 insertions and 362 deletions when this handoff was written.

## Important invariants

Future changes should preserve the following assumptions.

### Logical particle count versus capacity

`nprtcl_thispack` is the logical number of live particles. A particle view's extent is retained
capacity and may be larger. Every kernel, host copy, output routine, and MPI buffer operation must
use only the valid prefix `[0, nprtcl_thispack)`.

Do not restore code that treats `prtcl.extent(...)` as the live particle count.

### Compaction mapping

The host builds a list of source-to-hole moves from the surviving tail of the array. The device
runs one kernel over that list. A source particle must never also be a destination that is later
read, and all send/receive combinations must preserve exactly the expected live particles.

### `solver_ap` aliases `dmom`

This saves one full mesh-sized vector field but depends on task ordering:

1. stage-two drag history consumes the old `dmom` value before the coupled solve;
2. `ApplyA` may use the aliased storage as scratch;
3. that scratch is dead before `GatherKickPMBR` clears `dmom` for the next feedback deposit.

Do not keep data in `solver_ap` across that boundary without allocating independent storage.
Boundary exchange objects own separate communication buffers, so they do not extend the field
lifetime.

### Matched communication predicates

The condition used to initialize receives must match the conditions used by `SendUstar` and
`RecvUstar`. At this handoff:

- test-particle, local, DC1, and DC2 paths perform the post-solve exchange;
- ApplyA, PCG, and adaptive paths reuse ghosts populated by their final `ApplyA`.

Changing only one side can leave an unmatched MPI receive.

### Particle-mesh transpose consistency

Deposition and gathering must continue to use the same compact weights and cell origins. Divergent
weight calculations can break conservation and the symmetry required by PCG.

## Validation completed

All runtime testing used exactly one process and one NVIDIA GH200 GPU on Vista.

### Builds

- A full Release CUDA build completed successfully with:
  - `Athena_ENABLE_MPI=OFF`
  - `Kokkos_ENABLE_CUDA=ON`
  - `Kokkos_ARCH_HOPPER90=ON`
  - the in-tree `kokkos/bin/nvcc_wrapper`
- A separate serial Release build completed successfully.
- MPI-enabled compilation checks of the modified translation units completed successfully.
- `git diff --cached --check` reported no whitespace errors.

The successful CUDA build was in `/tmp/athenak-opt-gh200.DzYEOX`. This is an ephemeral path and
should not be expected to survive a later session.

Enabling both `Kokkos_ARCH_HOPPER90` and `Kokkos_ARCH_ARMV9_GRACE` encountered an existing Kokkos
CMake feature-check problem. The Hopper-only configuration built and ran correctly.

### One-GPU runtime coverage

Small two-dimensional runs passed for:

- local solver with NGP, CIC, and TSC;
- DC1 with NGP, CIC, and TSC;
- DC2 with TSC;
- ApplyA with TSC;
- PCG with TSC;
- adaptive with TSC;
- `back_reaction=false` while requesting PCG, which follows the test-particle path.

Small three-dimensional runs passed for:

- local/TSC;
- DC1/CIC.

A particle VTK smoke run completed and wrote three files containing the expected 6,400 points.
The tests also exercised solver diagnostics and all completed without a code failure.

Separate exhaustive host-side checks compared the compact NGP/CIC/TSC formulas with the previous
weight formulas over random positions and boundary cases. A combinatorial check of particle
compaction mappings covered all send/receive combinations through eleven particles.

### Validation deliberately not completed

After the user requested that no more tests be run, the full MPI/Serial build in
`/tmp/athenak-opt-mpi.zPiQ0i` was stopped at approximately 52 percent. No further tests were
started.

The following remain unvalidated at runtime:

- a complete MPI-enabled executable;
- the MPI executable launched directly with size one;
- particle migration across MPI rank boundaries;
- two or more GPUs;
- weak or strong scaling;
- performance against the original `dust` branch;
- long-duration conservation and numerical-regression comparisons.

The current results establish a good Phase 1 checkpoint, but they are not evidence of
production-scale multi-GPU performance.

## How to resume

### 1. Verify the branch and state before changing anything

```bash
cd /work/10737/clolds/vista/athenak_dust/athenak
git branch --show-current
git status --short --branch
git rev-parse --short HEAD
```

Stop if the branch is not `dust_optimized`.

The source changes were staged when this guide was created, so inspect both the index and working
tree:

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached
git diff --check
git diff
```

Do not assume a blank `git diff` means there are no changes; staged changes appear under
`git diff --cached`.

### 2. Preserve a checkpoint if one does not already exist

Review the patch first. If it is still the intended Phase 1 patch, add this handoff document and
create a checkpoint commit:

```bash
git add DUST_PHASE1_HANDOFF.md
git commit -m "Optimize dust GPU coupling phase 1"
```

If the source files are no longer staged, stage only the files listed in the table above after
reviewing them. Do not use a broad cleanup or reset because later sessions may contain unrelated
user work.

### 3. Recreate the known-good one-GPU build if needed

Temporary build directories may disappear between sessions. A replacement for the passing GH200
configuration is:

```bash
cmake -S . -B /tmp/athenak-opt-gh200-resume \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/kokkos/bin/nvcc_wrapper" \
  -DAthena_ENABLE_MPI=OFF \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_HOPPER90=ON \
  -DKokkos_ARCH_ARMV9_GRACE=OFF
cmake --build /tmp/athenak-opt-gh200-resume --target athena -j4
```

There is no need to repeat the full one-GPU smoke matrix unless the source has changed.

### 4. Finish the one-rank MPI check

This is the first unfinished validation step. It does not require multiple ranks or GPUs:

```bash
cmake -S . -B /tmp/athenak-opt-mpi-resume \
  -DCMAKE_BUILD_TYPE=Release \
  -DAthena_ENABLE_MPI=ON \
  -DKokkos_ENABLE_SERIAL=ON \
  -DKokkos_ENABLE_CUDA=OFF
cmake --build /tmp/athenak-opt-mpi-resume --target athena -j4
```

Run the resulting executable directly, without `mpirun` or `srun`, for a small
`dust_damping.athinput` case. That initializes MPI with size one and exercises the
`nranks == 1` fast paths. The test input is:

```text
/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/inputs/dust_damping.athinput
```

Use `particles/ppc=3` because that case has three dust species. A value of four is invalid for
the lattice setup.

### 5. Validate rank-boundary migration when resources permit

Start with two ranks on two GPUs before attempting a large job. The initial multi-rank test should:

1. force particles to cross a rank boundary;
2. verify global particle count and unique particle GIDs;
3. compare particle positions and velocities with the `dust` branch within the accepted
   tolerance;
4. exercise empty-send, empty-receive, send-only, receive-only, and simultaneous send/receive
   cases;
5. write particle VTK output once, checking per-rank offsets and the global point count;
6. test periodic boundaries and at least one non-periodic supported boundary configuration.

The existing migration-oriented input is a useful starting point:

```text
/work/10737/clolds/vista/athenak_dust/athenak_dust_tests/inputs/dust_snapshot_migration_tier_s.athinput
```

Do not infer multi-GPU correctness solely from a two-rank run. Once that test is clean, repeat at a
rank count that gives particles multiple possible destination ranks and test sustained migration
over many cycles.

### 6. Measure scaling before starting Phase 2

Collect a baseline from the unoptimized `dust` branch and compare it with `dust_optimized` using
the same executable options, mesh per GPU, particles per cell, rank placement, and output
settings.

Record at least:

- total zone-cycles or particle-updates per second;
- time in deposition, gather/kick, `ApplyA`, particle migration, and output;
- MPI time and collective count;
- host/device synchronization count;
- GPU kernel launch count and duration;
- device memory high-water mark;
- solver iterations, reductions, and `ApplyA` calls;
- particle imbalance per rank.

Run both weak and strong scaling. Repeat enough times to distinguish a real improvement from
launch-to-launch noise, and keep output disabled during timing runs.

## Candidate Phase 2 work

The likely next optimizations, in priority order for many GPUs, are:

1. replace globally replicated migration metadata with targeted rank-to-rank exchange, likely
   count exchange followed by `MPI_Alltoallv` or neighbor-aware communication;
2. bin or sort particles by mesh block and cell so deposition and gather access memory more
   coherently;
3. reduce global-memory atomics with team scratch or hierarchical cell-local accumulation;
4. overlap packing, device/host transfers, and nonblocking MPI with useful GPU work;
5. retain and reuse communication buffers at their high-water capacity;
6. reduce solver reductions or overlap them where the numerical method permits;
7. add repeatable multi-GPU correctness and scaling tests to
   `athenak_dust_tests`.

Choose among these based on profiling. At high particle density, atomic contention and particle
ordering may dominate. At high GPU counts with smaller local domains, migration metadata and MPI
latency may dominate.

## Completion criteria for the next checkpoint

Before calling the optimized branch ready for production multi-GPU use:

- complete the MPI build;
- pass the one-rank MPI fast-path run;
- pass forced two-rank and multi-rank migration tests;
- compare conservation and solution norms against `dust`;
- pass NGP, CIC, and TSC on two-dimensional and three-dimensional decompositions;
- validate particle VTK/tracking output with unequal and zero particle counts on some ranks;
- measure weak and strong scaling against the original branch;
- document the machine, compiler, rank mapping, input files, and measured results.

