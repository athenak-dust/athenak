# Writing an AthenaK Problem Generator

This tutorial explains how to add a custom problem-generator (`pgen`) C++ file to this
AthenaK dust-gas fork. It focuses on Newtonian isothermal hydrodynamics with dust
particles, but the same structure applies to other physics modules.

The most useful examples in the source tree are:

- [`src/pgen/fluids/shu_osher.cpp`](src/pgen/fluids/shu_osher.cpp): a small custom
  hydrodynamic pgen.
- [`src/pgen/tests/dust_damping.cpp`](src/pgen/tests/dust_damping.cpp): a relatively
  simple dust-gas test.
- [`src/pgen/tests/dusty_wave.cpp`](src/pgen/tests/dusty_wave.cpp): gas and dust
  initialization, multiple species, diagnostics, and history output.
- [`src/pgen/tests/dust_nsh.cpp`](src/pgen/tests/dust_nsh.cpp): a more advanced
  multispecies shearing-box equilibrium.

## 1. What a problem generator does

AthenaK constructs the mesh and all requested physics modules before it calls the pgen.
The pgen then:

1. Reads problem-specific parameters.
2. Checks that the required physics modules exist.
3. Initializes the already-allocated gas, magnetic-field, radiation, or particle arrays.
4. Optionally registers boundary, source, history, refinement, or final-diagnostic
   callbacks.

The startup path is approximately:

```text
main.cpp
  -> Mesh::AddCoordinatesAndPhysics()
  -> ProblemGenerator constructor
  -> ProblemGenerator::CallProblemGenerator()
  -> ProblemGenerator::UserProblem()       (custom compile-time pgen)
```

The relevant framework files are [`src/pgen/pgen.hpp`](src/pgen/pgen.hpp),
[`src/pgen/pgen.cpp`](src/pgen/pgen.cpp), and
[`src/CMakeLists.txt`](src/CMakeLists.txt).

## 2. Custom versus built-in pgens

AthenaK has two selection mechanisms.

### Custom compile-time pgen (recommended for a new research problem)

A custom file defines this exact member function:

```cpp
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart)
```

If the file is `src/pgen/dust/my_problem.cpp`, select it while configuring:

```bash
cmake -S . -B build -DPROBLEM=dust/my_problem
cmake --build build -j
```

Do not include the `src/pgen/` prefix or the `.cpp` suffix in `PROBLEM`.

When `PROBLEM` is not `built_in_pgens`, CMake compiles
`src/pgen/${PROBLEM}.cpp`, and `CallProblemGenerator()` calls `UserProblem()` directly.
In this mode, `<problem>/pgen_name` does not select the pgen and may be omitted.

### Built-in pgen

Built-in regression problems are selected at run time:

```ini
<problem>
pgen_name = dusty_wave
```

Adding a new built-in pgen requires all of the following:

1. Add its method declaration to `src/pgen/pgen.hpp`.
2. Add its `.cpp` file to `src/CMakeLists.txt`.
3. Add its name and method call to `ProblemGenerator::CallProblemGenerator()` in
   `src/pgen/pgen.cpp`.

That is useful for a permanent regression test, but it is unnecessary for an ordinary
custom problem.

## 3. A minimal isothermal dust-gas pgen

The following complete example initializes uniform isothermal gas and a lattice of one
dust particle per cell. It assumes a two- or three-dimensional mesh, one dust species,
and `<particles>/ppc = 1`.

Create `src/pgen/dust/uniform_dust.cpp`:

```cpp
// Uniform isothermal gas and a uniform lattice of dust particles.

#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "pgen.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Register callbacks above this test, if the problem has any. See section 8.
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Fail early when the input file does not construct the modules this pgen needs.
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "uniform_dust requires <hydro>, <particles>, and <dust> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pmbp->pdust->nspecies != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "uniform_dust requires <dust>/nspecies = 1"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real ppc = pin->GetOrAddReal("particles", "ppc", 1.0);
  if (ppc != 1.0) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "uniform_dust requires <particles>/ppc = 1"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real vx0  = pin->GetOrAddReal("problem", "vx0", 0.0);
  Real vd0  = pin->GetOrAddReal("problem", "vd0", 0.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;

  auto &u0 = pmbp->phydro->u0;
  par_for("uniform_dust_gas", DevExeSpace(), 0, nmb - 1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = rho0*vx0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
  });

  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  int ncells_per_mb = nx1*nx2*nx3;
  if (npart != nmb*ncells_per_mb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unexpected particle count for ppc = 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &mbsize = pmbp->pmb->mb_size;
  auto gids = pmbp->gids;
  bool three_d = pmy_mesh_->three_d;

  par_for("uniform_dust_particles", DevExeSpace(), 0, npart - 1,
  KOKKOS_LAMBDA(int p) {
    int m = p/ncells_per_mb;
    int c = p - m*ncells_per_mb;
    int i = c % nx1;
    int j = (c/nx1) % nx2;
    int k = c/(nx1*nx2);

    pi(PGID, p) = gids + m;
    pr(IPX, p) = CellCenterX(i, nx1, mbsize.d_view(m).x1min,
                            mbsize.d_view(m).x1max);
    pr(IPY, p) = CellCenterX(j, nx2, mbsize.d_view(m).x2min,
                            mbsize.d_view(m).x2max);
    pr(IPZ, p) = three_d
        ? CellCenterX(k, nx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max)
        : 0.0;

    pr(IPVX, p) = vd0;
    pr(IPVY, p) = 0.0;
    pr(IPVZ, p) = 0.0;
    pr(IPRX, p) = 0.0;
    pr(IPRY, p) = 0.0;
    pr(IPRZ, p) = 0.0;
  });

  // This sets PSP, IPTS, and IPM. It must come after PGID has been initialized because
  // it uses PGID to determine each particle's MeshBlock and cell volume.
  pmbp->pdust->SetDefaultMasses(pin);

  // Supply a finite initial transport timestep. The dust task list recomputes this
  // during evolution.
  ppar->dtnew = std::numeric_limits<float>::max();
}
```

The corresponding input file can start from:

```ini
<job>
basename = uniform_dust

<mesh>
nghost = 2
nx1 = 32
x1min = -0.5
x1max = 0.5
ix1_bc = periodic
ox1_bc = periodic

nx2 = 32
x2min = -0.5
x2max = 0.5
ix2_bc = periodic
ox2_bc = periodic

nx3 = 1
x3min = -0.5
x3max = 0.5
ix3_bc = periodic
ox3_bc = periodic

<meshblock>
nx1 = 16
nx2 = 16
nx3 = 1

<time>
evolution = dynamic
integrator = imex2+
cfl_number = 0.4
tlim = 1.0

<hydro>
eos = isothermal
iso_sound_speed = 1.0
reconstruct = plm
rsolver = hlle

<particles>
particle_type = dust
pusher = imex_dust
ppc = 1.0

<dust>
nspecies = 1
taus_1 = 0.1
dust_to_gas = 0.01
back_reaction = true
deposit = tsc
stopping_time_mode = species_fixed
dt_cfl = 0.5

<problem>
rho0 = 1.0
vx0 = 0.0
vd0 = 0.1

<output1>
file_type = hst
dt = 0.05

<output2>
file_type = bin
variable = cons
dt = 0.1
```

Build and run it from the repository root:

```bash
cmake -S . -B build -DPROBLEM=dust/uniform_dust
cmake --build build -j
./build/src/athena -i inputs/dust/uniform_dust.athinput
```

The executable location can differ with the generator or build environment. If needed,
locate it with `find build -type f -name athena`.

## 4. Gas arrays: conserved variables, not primitive variables

For Newtonian Hydro, initial conditions normally go into:

```cpp
auto &u0 = pmbp->phydro->u0;
```

Its indexing is:

```cpp
u0(meshblock, variable, k, j, i)
```

The principal variable indices are defined in `src/athena.hpp`:

| Index | Meaning in `u0` |
|---|---|
| `IDN` | mass density, rho |
| `IM1` | x1 momentum density, rho vx |
| `IM2` | x2 momentum density, rho vy |
| `IM3` | x3 momentum density, rho vz |
| `IEN` | total energy density for an ideal EOS |

The momentum slots do **not** contain velocity. For example:

```cpp
u0(m, IM1, k, j, i) = rho*vx;
```

For an isothermal EOS, initialize density and all three momenta. For an ideal EOS, also
initialize total energy:

```cpp
Real kinetic = 0.5*rho*(vx*vx + vy*vy + vz*vz);
u0(m, IEN, k, j, i) = pressure/(gamma - 1.0) + kinetic;
```

If magnetic fields are present, the MHD pgens also initialize face-centered magnetic
fields and usually derive cell-centered fields from them. Study
`src/pgen/fluids/field_loop.cpp` or `src/pgen/fluids/blast.cpp` before writing an MHD
pgen; magnetic initialization has additional divergence-free constraints.

## 5. MeshBlock packs, indices, and coordinates

AthenaK groups MeshBlocks into a `MeshBlockPack`. In a kernel,

```cpp
u0(m, IDN, k, j, i)
```

uses:

- `m`: MeshBlock within the local pack.
- `i`, `j`, `k`: cell indices, including the offset caused by ghost storage.

Obtain active-zone bounds from:

```cpp
auto &indcs = pmy_mesh_->mb_indcs;
int is = indcs.is, ie = indcs.ie;
int js = indcs.js, je = indcs.je;
int ks = indcs.ks, ke = indcs.ke;
```

Most pgens initialize only active zones. The boundary machinery later supplies ghost
data.

MeshBlock sizes are stored in a host/device dual view:

```cpp
auto &mbsize = pmbp->pmb->mb_size;
```

Inside a device kernel, use `mbsize.d_view(m)`. On the host, use
`mbsize.h_view(m)`. Cell-center coordinates are calculated with:

```cpp
Real x = CellCenterX(i - is, nx1,
                     mbsize.d_view(m).x1min,
                     mbsize.d_view(m).x1max);
```

Do not use the array index `i` directly as the zero-based cell number; active indices
start at `is`, not necessarily zero.

## 6. Kokkos kernel rules

The common AthenaK wrapper is:

```cpp
par_for("descriptive_kernel_name", DevExeSpace(),
        0, nmb - 1, ks, ke, js, je, is, ie,
KOKKOS_LAMBDA(int m, int k, int j, int i) {
  // Each iteration may execute concurrently.
});
```

`DevExeSpace` is `Kokkos::DefaultExecutionSpace`, selected by the Kokkos build. It may
represent serial CPU execution, OpenMP, CUDA, HIP, or another enabled backend.

Follow these rules inside `KOKKOS_LAMBDA`:

- Use captured scalar values and Kokkos device views.
- Do not read `ParameterInput` inside the kernel; read parameters on the host first.
- Do not access an ordinary `std::vector`, host pointer, or `h_view` in a device kernel.
- Do not print or perform file I/O from a kernel.
- Avoid two iterations writing the same location unless using a suitable atomic or
  reduction.
- Use device-compatible math functions.

For a parameter table constructed on the host, use a `DualArray`:

```cpp
DualArray1D<Real> amplitude("amplitude", nspecies);
for (int s = 0; s < nspecies; ++s) {
  amplitude.h_view(s) = /* host calculation */;
}
amplitude.template modify<HostMemSpace>();
amplitude.template sync<DevExeSpace>();

par_for("use_amplitude", DevExeSpace(), 0, npart - 1,
KOKKOS_LAMBDA(int p) {
  Real value = amplitude.d_view(pi(PSP, p));
});
```

## 7. Dust-particle state

Dust particles have an integer-property view and a real-property view:

```cpp
auto &pi = pmbp->ppart->prtcl_idata;
auto &pr = pmbp->ppart->prtcl_rdata;
```

The indices are defined in `src/athena.hpp`:

| Entry | Meaning |
|---|---|
| `pi(PGID,p)` | global ID of the particle's current MeshBlock |
| `pi(PTAG,p)` | globally assigned particle tag; normally assigned by the framework |
| `pi(PSP,p)` | zero-based dust-species index |
| `pr(IPX,p)`, `pr(IPY,p)`, `pr(IPZ,p)` | position |
| `pr(IPVX,p)`, `pr(IPVY,p)`, `pr(IPVZ,p)` | velocity |
| `pr(IPTS,p)` | stopping time |
| `pr(IPM,p)` | particle mass |
| `pr(IPRX,p)`, `pr(IPRY,p)`, `pr(IPRZ,p)` | drag-rate stage registers |

The framework allocates the particle count before calling the pgen:

```text
npart = <particles>/ppc * number of active cells
```

A pgen arranges and initializes this allocated population; it does not normally append
arbitrary particles.

For simple uniform dust, `DustGasDrag::SetDefaultMasses()` assigns species, stopping
times, and masses:

```cpp
pmbp->pdust->SetDefaultMasses(pin);
```

It stripes species over particle indices and chooses masses so that a uniform gas of
`<problem>/rho0` has the total dust-to-gas ratio specified by
`<dust>/dust_to_gas`. Because it calculates a particle's cell volume from `PGID`, set
`PGID` first.

For nonuniform dust density or unequal species abundances, set `PSP`, `IPTS`, and `IPM`
explicitly. For a uniform lattice, the basic normalization is:

```text
particle mass = desired species density * cell volume
                / particles of that species per cell
```

The dust module subsequently checks that:

- `PSP` lies in `[0, nspecies)`.
- `IPTS` is finite and positive.
- In `species_fixed` mode, `IPTS` equals `taus[PSP]`.

## 8. Restarts and callbacks

The pgen is called for both new runs and restarts. On a restart, the main state has
already been loaded from the restart file, so do not overwrite it:

```cpp
void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  if (restart) return;
  // Initialize state only for a new run.
}
```

Callbacks are not restored merely by reading field data. Register callbacks before the
early return:

```cpp
void MyHistory(HistoryData *pdata, Mesh *pm);
void MyFinalDiagnostic(ParameterInput *pin, Mesh *pm);

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  user_hist_func = MyHistory;
  pgen_final_func = MyFinalDiagnostic;
  if (restart) return;
  // New-run initialization follows.
}
```

Available callback pointers are:

```cpp
user_bcs_func    // physical boundary callback
user_srcs_func   // source term callback
user_ref_func    // refinement callback
user_hist_func   // custom history data
pgen_final_func  // final diagnostic after the main loop
```

Some callbacks must also be enabled by input options. In particular:

```ini
<problem>
user_srcs = true
user_hist = true
```

A boundary face uses a user callback when its mesh boundary flag is `user`. AthenaK
checks that requested callbacks were actually registered and aborts if they are null.
Use an existing pgen with the desired callback type as a template because each callback
has a distinct signature and execution context.

## 9. Extending the minimal example

### Spatially varying gas

Calculate coordinates in the cell kernel and derive local primitive quantities before
converting them to conserved variables:

```cpp
Real x = CellCenterX(i - is, nx1, mbsize.d_view(m).x1min,
                     mbsize.d_view(m).x1max);
Real rho = rho0*(1.0 + amp*cos(2.0*M_PI*x/lx));
Real vx = /* your profile */;

u0(m, IDN, k, j, i) = rho;
u0(m, IM1, k, j, i) = rho*vx;
```

### Multiple dust species

A transparent lattice layout uses an integer number of particles of every species per
cell. Require:

```text
ppc % nspecies == 0
```

Then map each particle to a cell and species explicitly, assign `PSP`, copy the
species stopping time into `IPTS`, and normalize `IPM` using the desired per-species
dust-to-gas ratio. `src/pgen/tests/dusty_wave.cpp` contains this complete pattern.

### Nonuniform dust density

Two common approaches are:

1. Keep a regular particle lattice and vary particle masses.
2. Keep equal masses and vary particle positions or particle number density.

The first is usually simpler for initialization but produces particles of different
mass. The second can better represent equal physical grains, but sampling a target
density field and maintaining load balance are more involved. Whichever representation
you choose, verify the deposited dust density, not merely the particle list.

## 10. Build and debugging workflow

Use a separate debug build while developing a pgen:

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug \
  -DPROBLEM=dust/uniform_dust
cmake --build build-debug -j
```

The debug configuration enables useful Kokkos checks, including bounds and DualView
consistency checks.

Start with a small, single-rank CPU run. Before running a large calculation, check:

- The input contains every required physics block.
- All densities, pressures, stopping times, and masses are finite and positive.
- Gas momentum is `rho*v`, not `v`.
- Ideal-EOS total energy includes internal and kinetic energy.
- Every particle has a valid position, velocity, `PGID`, `PSP`, `IPTS`, and `IPM`.
- Particle positions fall inside the MeshBlock named by `PGID`.
- Particle counts divide into MeshBlocks and cells exactly as assumed by the mapping.
- Species masses sum to the intended dust-to-gas ratio.
- Kernel iterations never write the same output location accidentally.
- Host-only containers are not captured by device kernels.
- Restart runs skip initial-condition writes but still enroll callbacks.
- A short run writes sensible initial output and advances with a finite timestep.

Useful failure symptoms include:

| Symptom | Likely cause |
|---|---|
| Gas moves at the wrong speed | Velocity was stored in `IM1` instead of `rho*vx` |
| GPU compilation fails around a lambda | Host-only object/function used in a kernel |
| Kokkos bounds error | Incorrect active-zone or particle-index mapping |
| Dust stopping-time validation aborts | Invalid `PSP`/`IPTS`, or mismatch in `species_fixed` mode |
| Wrong total dust density | Incorrect particle mass or particles-per-cell normalization |
| Restart returns to initial conditions | State initialization occurred before checking `restart` |
| Requested history/source/BC aborts at startup | Input enabled a callback that the pgen did not register |

## 11. A practical development sequence

For a new physical problem, build it in layers:

1. Initialize uniform gas only and write an initial output.
2. Add the desired spatial gas profile and verify conserved quantities.
3. Place stationary dust particles on a lattice.
4. Verify particle IDs, species, stopping times, masses, and deposited dust density.
5. Add dust velocities and drag evolution.
6. Add multiple species.
7. Add source terms, custom boundaries, or history diagnostics one at a time.
8. Test a restart and compare it with an uninterrupted run.
9. Repeat with multiple MeshBlocks, MPI ranks, and the intended accelerator backend.

This sequence separates initialization mistakes from drag-solver, communication, and
accelerator issues, which makes failures much easier to diagnose.
