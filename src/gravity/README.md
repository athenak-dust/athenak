# Composite self-gravity development

**Production migration is in progress.** Rixin: small functional
validation is not the endpoint. The earlier reference implementation below is not
the production acceptance result. Current work replaces replicated global setup
with distributed geometry/multigrid, refreshes only shear-dependent rows, and
replaces direct isolated-boundary sums with a coarsened vacuum solve. Large WENO-Z
CPU/GPU time, memory, dynamics and restart gates must pass before this work closes.
Progress and fixed acceptance criteria are in
the internal run records.

This experimental module is developed on `selfgrav-amr-shear`, from dust
development commit `6c466d6d`. It is disabled unless `<gravity>/enabled = true`.
It solves on the actual active AMR leaves, including shearing-periodic faces;
it does not substitute a uniform fine-grid volume solve.

The requested endpoint is hydro + multigrid + AMR + shearing box, including dynamic
refine/derefine, restart, MPI and GPU. Two gravity boundary configurations are
required: periodic, and horizontally shearing-periodic/periodic with vertical
isolation. Gas outflow and isolated gravitational potential are separate choices.
Future deposited dust density and particle force gathering can reuse the same
field solver. Current support is 2D/3D Cartesian Newtonian hydro; particles/dust, MHD, radiation
and relativistic modules are rejected when this feature is enabled.

## Running

Initialize the pinned Kokkos submodule and build outside the checkout with the
usual AthenaK CMake options. No additional solver library is required. Examples
are in `inputs/hydro/self_gravity_{poisson,jeans,slab,shear_amr}.athinput`.
All examples select WENO-Z and explicitly select distributed setup; the isolated
example also selects the vacuum boundary backend. The portable slab comparison
uses 16 and 32 root cells with 8-cell blocks so both grids contain an actual
coarse/fine interface.
Run them in an external, project-designated scratch directory. Static Poisson
tests use `evolution=static` and the native `advect` solver; the Jeans and shearing
examples evolve the actual hydro equations.

The portable check runner accepts an existing executable and an external output
directory, without installing packages or modifying the source:

```text
python tst/test_suite/gravity/check_multigrid.py --binary /path/to/athena \
    --scratch /path/to/project-scratch/gravity-check --ranks 2
```

Use the cluster's rank-to-GPU binding when invoking GPU MPI binaries. The broader
hardware validation separately verified distinct GPU assignments.

| `<gravity>` key | Default / meaning |
| --- | --- |
| `enabled` | `false`; absence of the block leaves existing physics unchanged |
| `four_pi_G` | Required positive gravitational constant in code units |
| `solver` | `multigrid` |
| `setup` | `reference` during validation; explicitly select `distributed` for the production candidate |
| `vertical_bc` | `periodic`, or `isolated` in 3D |
| `isolated_method` | `direct` during validation; `vacuum` requires distributed setup |
| `rtol` | `1e-10`, relative linear-system residual |
| `max_iterations`, `gmres_restart` | `150`, `24` |
| `cfl` | `0.25`, safety factor for acceleration and free-fall limits |
| `slab_modes_x`, `slab_modes_y` | `0`: use finest horizontal Nyquist cutoff |
| `report_each_solve` | `false`; final solve summary is always emitted |

Native output choices `gravity_phi`, `gravity_g1/g2/g3`, and `gravity` expose the
potential/acceleration in active cells. Gravity ghost-zone output is rejected.

## Discrete operator

`composite.*` constructs one potential unknown per active leaf cell. Every face
fragment contributes the same integrated normal flux with opposite signs to the
two adjacent cells, including coarse/fine and sheared periodic interfaces.
This is conservation of Poisson/Gauss flux, not a claim of exact total momentum
conservation of the coupled AMR scheme.
Transverse least-squares slopes bring the two potentials to the fragment's
transverse centroid. The slope operator is only for reconstruction; acceleration
uses the area average of shared normal face derivatives. Using the raw slope as
the force had first-order maximum errors in the prototype.

The matrix is the **negative integrated Laplacian**, so the physical right-hand
side is $-4\pi G\rho V$. Periodic solves subtract the volume-weighted density mean
and fix the potential gauge. Vertical isolated solves must retain the mean slab
density. The shear convention is
$\phi(x+L_x,y)=\phi(x,y-s)$, with $s=-q\Omega L_x t$ for the existing hydro
shearing-wave convention.

The reference backend uses dyadic aggregates, smoothed interpolation and Galerkin
coarse matrices. The distributed candidate instead constructs owned leaf rows
from global patch metadata (`mesh_geometry.*`). Its geometric hierarchy
(`geometric_mg.*`) uses tensor linear interpolation, integrated-residual child
sums, and rediscretized coarse operators. No rank expands the global leaf volume.
A V-cycle preconditions restarted flexible GMRES because the composite stencil
is generally nonsymmetric. Bulk matrix, smoother, and vector operations use
Kokkos, with local MPI rows and sparse reusable halo plans. Only the tiny coarsest
problem uses a root CPU LU solve.

At each new shear phase the distributed backend updates surface-dependent rows
of the current matrix and force operators. The preconditioner can remain cached;
convergence is always checked against the current matrix. A failed cached attempt
refreshes the hierarchy once and retries, with both attempts included in timing.
Identical source, topology, phase and gravitational constant can reuse a field;
extra-source callbacks are still evaluated before checking source equality.

The vacuum boundary candidate adds coarsened, empty solver patches outside the
physical x3 interval. It retains the physical density mean and imposes the mean
sheet flux at distant Neumann faces. Exterior resolution follows an exponential
bound based on the slowest horizontal mode. It avoids direct cell-by-mode sums;
the hydro mesh and its boundary conditions are unchanged. Current vacuum geometry
requires power-of-two root counts and block dimensions, with at least eight cells
per active direction and aligned dyadic AMR. Unsupported geometry fails explicitly.

## Coupling and lifecycle

`SelfGravity` owns potential and acceleration in active-cell arrays. Gravity half
kicks enclose hydro; when gravity and 3D shear are both enabled, orbital advection
also uses two halfsteps. Residual hydro uses a fixed midpoint shear phase, and
both orbital halfsteps retain native mixed-level flux correction. The ordinary
gravity-disabled time integrator is unchanged. Each kick keeps density
fixed and preserves ideal-gas internal energy; this is a second-order splitting,
not a claim of exact total gravitational-energy conservation. Density is solved
at the beginning and end physical times, and primitives/ghost values are refreshed
after kicks. Acceleration and free-fall limits participate in the timestep.

Regrid/load-balance invalidates the composite system and solves again from the new
active density. Shared-file restart metadata preserves global block cooldown ages
and whether the checkpoint precedes a pending mesh check. Resume restores/replays
these before the gravity solve and timestep, including changed MPI decompositions.
Potential is reconstructed from density/time; no stale potential is restored.
Per-rank gravity restart files are explicitly unsupported. Legacy checkpoints
without the record remain readable but cannot recover missing AMR ages/phase;
the reader reports that limitation.

`SetExtraDensitySource` is the future source interface: a callback fills an
additional active-cell density at each solve. Particle deposit/gather and its
time coupling remain the responsibility of the future dust implementation.

## Validation and limitations

Whole-AthenaK periodic and mixed 2D/3D manufactured tests show approximately
second-order potential and force $L_2$ convergence. Mixed isolated-slab force
convergence was about 1.73; maximum/interface-force convergence is weaker than
uniform-grid convergence. The isolated slab retains its nonzero mean sheet field.

Physical tests include Jeans growth/oscillation, actual refine/derefine, and
three-dimensional shearing waves with mixed orbital-advection faces. Doubling
the physical shearing AMR resolution reduced ODE projection error by 3.81.
Joint space/time refinement of uniform WENO waves gives full-field error orders
about 1.88 and 1.98, including off-target structure. Fixed-grid timestep tests
are a different statement: PLM tests and the WENO density test did not meet an
expected order 1.8. A donor-cell diagnostic recovered second-order time
convergence. Limiters/characteristic remapping prevent a universal fixed-grid
second-order claim; failed expectations remain recorded.

The integrated `59267024` implementation passed 34 application cases on CPU and
1/2/4 distinct GB200 GPUs, including physical shearing AMR, combined shear/isolated
boundaries, ideal energy, and checkpoints at actual refine/derefine events with
changed decomposition. Active float64 restart differences were at most
$8.9\times10^{-16}$; CPU/GPU differences at most $2.45\times10^{-15}$ in these
tests. Gravity-off fields and history were bitwise unchanged.

The final functional commit `3c46d73f` initializes static-stage counts and skips
unposted orbital-flux correction during initialization. Its two-file host-only
change passed targeted CPU1/4-rank and one-GPU static, physical-shear, and off-path
regressions. The four-distinct-GPU evidence is from `59267024`; it is not relabeled
as a fresh four-GPU execution of the final static fix.

Finite-amplitude AMR tests do not conserve net self-force exactly. For one corner
refinement test the relative imbalance decreased from $1.79\times10^{-5}$ to
$1.07\times10^{-6}$ between 16 and 64 root cells. This is convergent operator/force
coupling error, not an iterative-solver tolerance effect. Total gas plus gravity
energy is also not exactly conserved; ideal-gas SMR tests show convergent drift.
Shared fluxes, small residuals, and weak-wave momentum cancellation alone do not
prove exact conservation of these quantities.

The reference backend has known performance limitations: host setup holds global
geometry and matrices on each rank, and its 3D Galerkin hierarchy has significant
fill. Direct slab boundary sums cost $O(N_{cell}N_{mode})$. They evaluate the
horizontal Fourier projection and free-space vertical Green function on source
cells; this is an accuracy backend, not a claim of production scaling. Two cheaper
coarse-grid prototypes failed standalone multigrid contraction and were not
adopted. They are distinct from the new distributed geometric hierarchy.

The distributed CPU candidate has been tested on CPUs with gcc 13: uniform
$128^3$ hierarchy setup takes 1.07 s on four ranks, compared with 65.45 s for the
reference backend. The complete 4.19-million-physical-cell isolated case adds
29.1% vacuum unknowns; on 16 ranks its initial setup takes 1.21 s and peak RSS is
about 648 MiB per rank. These measurements do not establish GPU production
performance. The required 100-step, matched WENO-Z GPU benchmark remains pending;
defaults will change only after that validation.

Detailed provenance, bounded test batches, and outstanding requirements are in
the internal run records.

References: [Athena++ AMR multigrid](https://arxiv.org/abs/2302.13903),
[its conservative interface formula](https://github.com/PrincetonUniversity/athena/blob/main/src/gravity/mg_gravity.cpp).
The new shearing composite operator and device implementation are separate code;
the Athena++ reference does not establish their correctness.
