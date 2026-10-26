# Hydro self-gravity with AMR and shearing boundaries

The default solver uses distributed geometric multigrid and updates only the
rows affected by the changing shear phase. Vertical isolation uses a coarsened
vacuum exterior, retaining the nonzero mean sheet field. The reference backend
remains available explicitly for accuracy comparisons.

**Post-review fixes accepted:** `2b4adc3c` and `7566e314` make the
optional direct isolated boundary continuous through shear remaps and prevent
extreme-scale FGMRES norms from falsely certifying convergence. CPU 1/2/8-rank
regressions and a run on four distinct GB200 GPUs pass, including
the actual WENO-Z physical SMR and dynamic AMR cases below. The same-job large
SMR baseline/fixed active float64 maximum difference is4.90e-13. Gravity stage
time including setup amortization is73.48/73.57ms per step; this single pair
shows no material regression, not a precise performance difference. The older
30112e27 measurements remain labelled with their original provenance.

**Accepted hydro baseline (2026-10-26):** the numerical implementation at
`30112e27` has passed native WENO-Z physical SMR, dynamic AMR, changed-rank
restart and four-distinct-GB200 checks. The final coarse-grid repair gives
full-period vacuum cells a periodic reconstruction and rejects extra terminal
nullspaces using zero-column and scale-relative LU checks. FGMRES still accepts
only the current true residual. Exact prolongation-query and halo reuse avoid
repeated geometry/communication setup without changing coefficients.

The latest physical gas meshes contain a genuinely localized x/y/z octant:
root64x64x128, blocks32 cubed, locally refined to the spacing of uniform
128x128x256. Static SMR has30 blocks/983040 physical cells; dynamic AMR follows
16->30->16. Four-GB200 runs at CFL0.3 to t0.68 took119.86/89.40 application seconds,
with1452/1110 steps respectively. The maximum-rank gravity-stage sums including
initial topology setup amortization were72.94/69.13ms per step. These different
mesh/time-step histories are not a pure algorithm speedup. CPU2rank/CUDA4rank
active float64 differences on the matched small dynamic case were1.55e-15;
changed-rank2->4 restart differences were below2e-15.

This completes the implemented hydro feature set and its engineering regression
checks. Long nonlinear self-gravitating application validation and deposited
dust gravity/particle gathering remain separate work. Density-weighted total
self-force is convergent, not exactly zero. Rixin: decided not to include the
cuSPARSE experiment in the final module: the adapter, dependency and runtime
selectors are absent from this branch.

The following timings are historical benchmarks with earlier operators and
different meshes/time-step windows, retained for provenance:

On four GB200 GPUs, the 4.19-million-physical-cell WENO-Z disk benchmark measures
about **40.5 ms of gravity work per step at fixed dt=1e-4**. A follow-up using
the same binary and normal CFL=0.3 through 1.02 shear wraps measures **118.4 ms
per step**, including eight hierarchy refreshes. Source, shear updates, failed
attempts, refreshes, accepted solves and force evaluation are included; initial
topology setup is separate. The short fixed-step benchmark does not establish
normal-CFL production cost. Both allocate 18 CPU cores per rank. Exact workloads
and versions are recorded under
the internal run records.

This module is developed on `selfgrav-amr-shear`, originally from dust
development commit `6c466d6d`. Merge `06c8fc23` incorporates the stable fixes
through `530be393`, including ordinary-periodic-x1 mass conservation with AMR;
the original pre-review version remains available at `1661775b`.
It is disabled unless `<gravity>/enabled = true`.
It solves on the actual active AMR leaves, including shearing-periodic faces;
it does not substitute a uniform fine-grid volume solve.

The implemented scope is hydro + multigrid + AMR + shearing box, including dynamic
refine/derefine, restart, MPI and GPU. Gravity boundaries can be periodic, or
horizontally shearing-periodic/periodic with vertical isolation. Gas outflow and
isolated gravitational potential are separate choices.
Future deposited dust density and particle force gathering can reuse the same
field solver. Current support is 2D/3D Cartesian Newtonian hydro with one pack per
rank; particles/dust, MHD, radiation
and relativistic modules are rejected when this feature is enabled. Shearing-
periodic gravity requires 3D hydro; 2D periodic-x1 gravity remains supported.
Distributed setup requires power-of-two active MeshBlock dimensions. The
isolated vacuum backend has additional root-grid restrictions described below.

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
| `setup` | `distributed`; `reference` explicitly selects the replicated accuracy backend |
| `vertical_bc` | `periodic`, or `isolated` in 3D |
| `isolated_method` | `vacuum` for distributed setup; explicit reference setup defaults to `direct` |
| `rtol` | `1e-10`, residual relative to the full linear-system RHS, including the isolated mean-sheet/boundary terms |
| `max_iterations`, `gmres_restart` | `150`, `24` |
| `cfl` | `0.25`, safety factor for acceleration and free-fall limits |
| `slab_modes_x`, `slab_modes_y` | `0`: use finest horizontal Nyquist core/cutoff; direct x spectrum has the one-mode taper described below |
| `report_each_solve` | `false`; final solve summary is always emitted |
| `phase_threads` | `0`: up to eight affinity-limited host workers in GPU builds, one in CPU builds; `1` forces serial assembly |

Native output choices `gravity_phi`, `gravity_g1/g2/g3`, and `gravity` expose the
potential/acceleration in active cells. Gravity ghost-zone output is rejected.

## Discrete operator

`mesh_geometry.*` constructs one point-center potential unknown per active leaf
cell. `composite.*` uses the same finest physical operator and retains a separate
global Galerkin hierarchy. Every face
fragment contributes the same integrated normal flux with opposite signs to the
two adjacent cells, including coarse/fine and sheared periodic interfaces.
This is conservation of Poisson/Gauss flux, not a claim of exact total momentum
conservation of the coupled AMR scheme.
At offset or unequal-width faces, a scaled quadratic least-squares fit on two
face-neighbor rings brings the potentials to the fragment's transverse centroid.
Away from real curvature jumps, the shared center-to-center secant is corrected
by $(h_L-h_R)(\phi_{nn,L}+\phi_{nn,R})/8$.
At a real $C^1$ source/vacuum join it instead uses
$(h_L^2\phi_{nn,L}-h_R^2\phi_{nn,R})/(8d)$, where $d=(h_L+h_R)/2$.
The smooth correction vanishes for equal widths. Applying separate curvatures
only when an equal-width shear face is offset would leave a finite operator
jump as the offset tends to zero; the two formulas avoid that discontinuity.
Both transverse quadratic moments and this normal correction are necessary to
reproduce general quadratic potentials at a coarse/fine face. Aligned equal-width interior faces retain the
seven-point potential stencil. Acceleration is the negative volume gradient of
one shared face-averaged potential per fragment. The face average includes both
normal interpolation curvature and tangential area moments on every face,
including aligned bulk faces; regular bulk Hessians use cheap centered axis
differences. The reconstruction gradient is not used directly as force.
The wider reconstruction affects setup/phase assembly and CSR halo size, while
all force applications remain sparse matrix operations. The potential has point
semantics even though hydro supplies cell-average density; their source
quadrature difference remains a second-order spatial approximation.

The vacuum extension records the physical source-box z walls explicitly as
`CompositeDomain::c1_z_interfaces`. Fits stay on one side of these planes and
expand to at most four face-neighbor rings if necessary. This resolves curvature
jumps at a general $C^1$ interface without subtracting a sheet background.
Finest fits must remain full rank. An explicitly coarse geometry may use a
two-sided fit when a region has too few normal cells; this is a lower-order
preconditioner approximation, counted in `quadratic_interface_fallbacks` and the
setup profile. Constant and shared-flux identities still hold for that fallback.
Phase dependencies include every neighborhood traversed by an expanded fit.

`CompositeDomain::shear_phase_updates` is an explicit capability, defaulting to
true for generic geometry callers that may advance from phase zero. Both fluid
adapters set it from the actual x boundary. With this capability, quadratic fits
first stay on the anchor's physical x side, including at phase zero, so their
coefficients can be reused across phase changes. If that fit remains rank
deficient after four rings, x images are permitted while retaining the fine
source/vacuum z filter. Only accepted x-image samples mark a fit as phase
dependent. Fine x-image fallbacks are counted separately from coarse C1 fallback.
Ordinary-periodic geometry disables the capability, retains centered periodic
Hessians and rejects nonzero shear updates. This stencil choice changes
higher-order errors; it is separate from exact geometry-cache optimization.

The matrix is the **negative integrated Laplacian**, so the physical right-hand
side is $-4\pi G\rho V$. Periodic solves subtract the volume-weighted density mean
and fix the potential gauge. Vertical isolated solves must retain the mean slab
density. The shear convention is
$\phi(x+L_x,y)=\phi(x,y-s)$, with $s=-q\Omega L_x t$ for the existing hydro
shearing-wave convention.

The reference backend uses dyadic aggregates, smoothed interpolation and Galerkin
coarse matrices. The distributed backend instead constructs owned leaf rows
from global patch metadata (`mesh_geometry.*`). Its geometric hierarchy
(`geometric_mg.*`) uses tensor linear interpolation, integrated-residual child
sums, and rediscretized coarse operators. No rank expands the global leaf volume.
A V-cycle preconditions restarted flexible GMRES because the composite stencil
is generally nonsymmetric. Bulk matrix, smoother, and vector operations use
Kokkos, with local MPI rows and sparse reusable halo plans. Only the tiny coarsest
problem uses a root CPU LU solve.

At each new shear phase the distributed backend updates surface-dependent rows
of every multigrid matrix and prolongation operator, together with the finest
force and boundary operators. Prolongation dependencies are traced independently
through recursive shear images. Physical child-sum restriction remains fixed.
Every smoothing diagonal and the tiny terminal factorization use the current
operator. Only the owned part of the at-most-128-row terminal CSR stays on the
host after upload; fine and intermediate host matrices are released.
Convergence is checked against the current true residual. The existing capped
attempt and full-hierarchy refresh/retry remain available, with both attempts
included in timing.
At a restart, a true residual more than 1000 times the larger of the Arnoldi
estimate and requested target marks a preconditioner breakdown. The terminal LU
checks for zero columns and uses a scale-relative pivot floor after removing the
constant mode with an operator-scaled rank-one gauge. Extra coarse nullspaces
therefore fail at setup instead of silently producing huge corrections.
Full-x-period vacuum cells use periodic quadratic samples from the outset;
the one-sided physical-source/vacuum x3 filter remains independent. The explicit
`gravity_coarse_check` target exercises this geometry, phase updates, narrow
domains, matrix scaling and collective rejection of invalid terminal operators.
Identical source, topology, phase and gravitational constant can reuse a field;
extra-source callbacks are still evaluated before checking source equality.
Host phase workers use private gradient caches and concatenate consecutive row
chunks in the original order. They do not call MPI or launch GPU work. MPI builds
request `MPI_THREAD_FUNNELED` for this use; an explicit parallel request fails if
the library cannot provide it, while automatic mode reports a serial fallback.
See the [MPI thread-support contract](https://www.open-mpi.org/doc/v4.1/man3/MPI_Init_thread.3.php).

The vacuum boundary backend adds coarsened, empty solver patches outside the
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

For the orbital split, the last RK stage already supplies midpoint ghosts.
The driver repeats that exchange only if EOS floor counters changed on any rank,
or passive scalars are present (their clipping has no counter). After the second
orbital halfstep, active-cell conserved-to-primitive conversion applies any EOS
floors before gravity samples density; it needs no full boundary exchange there.
The final post-kick exchange remains. In the short two-rank baseline/candidate
tests, including active density floors, histories and all active conserved cells
are bitwise identical. Stored inactive ghosts are not bitwise identical. This
test does not establish a GPU speedup for the reduced refreshes.

Problem-specific loop work, including density renormalization, runs before the
first gravity solve/kick. Gravity-off retains the native orbital splitting,
which differs from the two-halfstep gravity-on path. It is a useful cost control,
but differences from gravity-on cannot be attributed solely to gravity. An
accuracy comparison must also control the time-integration path.

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
convergence was about 1.73 with the earlier linear interface reconstruction;
these historical numbers do not validate the quadratic interface update.
The isolated slab retains its nonzero mean sheet field.

The explicit double-precision `gravity_geometry_check` CMake target checks every quadratic moment
in 2D/3D (including mixed moments), isolated potential/normal-derivative data,
second-order interface force convergence for a cubic potential, periodic Gauss
flux and constant identities, full-wrap phase-update/rebuild equivalence, and
reference row/boundary-sample mapping. Run the executable from an external
scratch directory. This is a direct operator check, separate from convergence
of a physical-density Poisson solve or conservation of hydro momentum.

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

An earlier isolated mean-sheet SMR check found false horizontal acceleration
near vertical coarse/fine faces: its maximum relative to the vertical force
decreased from about 0.41% to 0.20% between 16 and 32 root cells. This spatial
interface error was separate from the iterative residual. Correcting only smooth
quadratic interfaces leaves a curvature-jump error at the physical source/vacuum
join. With the one-sided $C^1$ correction, the matched constant-density vacuum
slab has total force maximum errors $4.2\times10^{-13}$ and $9.1\times10^{-13}$
at 16 and 32 root cells with `rtol=1e-12`. A nonzero slab perturbation retains
spatial error: its interface force maxima decrease by 3.55 under the same mesh
doubling. These are static Poisson checks, not hydro momentum acceptance.

The earlier width-weighted face-normal acceleration average did not preserve
$\sum_i V_i\mathbf{g}_i=0$ after quadratic interface reconstruction and introduced
a self-force linear in a wave's amplitude. The shared potential volume gradient
restores that identity to roundoff for periodic composite meshes. This removes
the constant-density contribution; it does not make
$\sum_i\rho_i V_i\mathbf{g}_i$ exactly zero. The remaining density-weighted
perturbation self-force and nonlinear momentum accuracy require separate checks.

The full-RHS `rtol` does not prescribe relative accuracy of a tiny perturbation.
However, the review's quoted 0.28% potential error was the total-field difference
between tolerances divided by the perturbation amplitude. Subtracting a separately
computed zero-amplitude background at each tolerance reduces that diagnostic to
about 8e-8 (forces about 0.9e-7 to 1.3e-7) against the discrete rtol=1e-13 result.
Most of the quoted difference was common background error. The raw force still
acts on hydro, and this subtraction is not performed at runtime. These static
comparisons neither establish continuum accuracy nor test acceptance of a tiny
changing source in a warm-started solve. The default tolerance is unchanged;
small-signal applications need matched-background and observable convergence
checks rather than an assumed universal perturbation-error bound.

Actual topology changes still rebuild the hierarchy. On the four-GB200 disk
benchmark this setup costs about 4.37 s, separate from the 40.5 ms steady-step
number at fixed small dt. Regridding frequency therefore affects total run cost.
The follow-up normal-CFL GPU run took 1452 steps to t=0.68, with hierarchy refreshes
at steps 172, 336, 496, 681, 839, 1019, 1172 and 1340. Subsequent refresh intervals
were 153--185 steps. Accepted cached solves cost 74.5 ms/step and amortized hierarchy
refresh 22.9 ms/step, the main remaining optimization targets. Every accepted
current-system residual remained <=1e-10. Observed CUDA peak was 2114 MiB/rank
versus 1742 MiB in the matched fixed-step run; this is an observed allocation
window, not a universal peak bound. Normal CFL advances about 4.68 times as much
physical time per step, so its higher per-step cost must not be confused with
a slower advance per unit physical time. These GPU timings use the old frozen
binary from `e1ffa5e8`; they do not include the later boundary-refresh changes.

The reference backend has known performance limitations: host setup holds global
geometry and matrices on each rank, and its 3D Galerkin hierarchy has significant
fill. Direct slab boundary sums cost $O(N_{cell}N_{mode})$. They evaluate the
horizontal Fourier projection and free-space vertical Green function on source
cells; this is an accuracy backend, not a claim of production scaling. Two cheaper
coarse-grid prototypes failed standalone multigrid contraction and were not
adopted. They are distinct from the new distributed geometric hierarchy.

The direct boundary now defines its x cutoff in physical wavenumber:
`a=kx*Lx/(2*pi)` has unit weight through `abs(a)<=slab_modes_x`, with a cosine
taper to zero over the next unit interval. Per-ky integer labels follow the
current shear phase; labels exchanged at a wrap have zero weight and derivative.
The unsheared rectangular finite sum is retained. At nonzero shear this is a
windowed spectral approximation, requiring convergence checks in both cutoffs.
`gravity_slab_window_check` verifies the wrap, cutoff edges, update-history
independence, and the original unsheared sum. The production vacuum path is
unaffected by this reference-boundary change.

FGMRES uses scaled sum-of-squares norms and common extreme-scale normalization
of its RHS and warm guess. Nonfinite inputs and unrepresentable RHS/guess ranges
are rejected. An exactly homogeneous projected system takes the zero gauge-fixed
solution. `gravity_solver_scaling_check` exercises finite scales1e-300..1e308,
warm guesses and collective rejection, including ranks owning no rows. There is
one MPI collective per existing norm, as before, plus three validation norms per
solve and one working solution vector; inner dot-product collective counts are
unchanged. These checks do not promise solvability for every finite-double matrix.

The distributed backend has been tested on CPUs with gcc 13: uniform
$128^3$ hierarchy setup takes 1.07 s on four ranks, compared with 65.45 s for the
reference backend. The complete 4.19-million-physical-cell isolated case adds
29.1% vacuum unknowns; on 16 ranks its initial setup takes 1.21 s and peak RSS is
about 648 MiB per rank. GPU validation includes matched gravity-on/off runs,
actual distinct device assignments, host RSS and allocation/NVML observations.
The initial four-GPU result was 77.9 ms/step and failed the registered 50 ms goal;
parallel host phase assembly brought it to 40.5 ms with the same numerical
operator. CPU coefficients and matrix application, and short WENO histories,
are unchanged by this parallelization. See the frozen experiment records for
the distinction between the complete reference batch and the later thread-count
comparison; results from earlier binaries are not relabeled as later executions.

Detailed provenance, bounded test batches, and outstanding requirements are in
the internal run records.

References: [Athena++ AMR multigrid](https://arxiv.org/abs/2302.13903),
[its conservative interface formula](https://github.com/PrincetonUniversity/athena/blob/main/src/gravity/mg_gravity.cpp).
The new shearing composite operator and device implementation are separate code;
the Athena++ reference does not establish their correctness.
