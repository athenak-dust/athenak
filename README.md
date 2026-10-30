# AthenaK dust: AMR, multi-species input, science outputs and dust self-gravity

This branch (`amrSG_dev`) extends the dust module of the uniform-mesh dust branch [`dust`](https://github.com/athenak-dust/athenak/tree/dust) with

- dust on statically and adaptively refined meshes (refinable dust clouds, or point particles with owner-overlap deposition),
- a multi-species input API (stopping times, abundances and sampling per species, power-law size distributions),
- science outputs for the particles (`pbin`, `phst`) and dust mesh fields in the standard binary output,
- self-gravity of the dust (and, experimentally, of the gas) with a multigrid Poisson solver that works on refined meshes and in the (stratified) shearing box.

The basics of the dust module (particles, `pusher = imex_dust`, the drag couplings `imex` / `pc2`, back-reaction) are as on the `dust` branch; see its [README](https://github.com/athenak-dust/athenak/blob/dust/README.md) and the companion repository [athenak-dust/athenak_dust_tests](https://github.com/athenak-dust/athenak_dust_tests). The upstream AthenaK README is kept in [doc/README_upstream.md](doc/README_upstream.md); build the code as described there (CMake, Kokkos submodule, `-D PROBLEM=built_in_pgens`, with `-D Athena_ENABLE_MPI=ON` for several ranks).

Example inputs for everything below are in [`inputs/dust_amr/`](inputs/dust_amr); each starts with a comment that says what it tests and how long it runs. Every dust problem has its own problem generator (section 3, Problem generators).

## 1. Dust species and outputs

### Species keys

All in `<dust>` unless noted; species are numbered `N = 1 .. nspecies` (the particle's `species_id` is `N - 1`). Two independent groups of keys, each either new or legacy (mixing the new and legacy keys of the same quantity stops the run):

| Quantity | New keys | Legacy keys |
| --- | --- | --- |
| stopping time | `tau_s_N` (code time) or `stokes_N` (`Omega t_s`, Omega from `<shearing_box>/omega0` or `stokes_omega_ref`) | `taus_N` |
| abundance | `epsilon_N` (rho_d/rho_g; the midplane ratio in a stratified box) or `Z_N` (column ratio Sigma_d/Sigma_g, stratified 3-D only) | `<problem>/eps_N`, `dust_to_gas` |
| sampling | `ppc_N`: root particles (clouds) per root cell of species N | `<particles>/ppc` (all species) |
| layer thickness | `<problem>/H_p_N`, `H_p`, or `H_p_min` + `H_p_max` (stratified) | `<problem>/dust_hp` |

- `species_distribution = powerlaw`: `tau_s_min`/`tau_s_max` (or `stokes_min`/`stokes_max`) spaced logarithmically, the total `epsilon` or `Z` split as `t_s^mass_slope`, and `ppc_per_species`.
- `column_normalization` (with `Z_N`): `redraw` (default; the whole column mass is placed inside the box, draws outside are redrawn), `full_profile` (the Gaussian is cut at the box), or `domain` (the dust mass in the box is `Z` times the gas mass in the box).
- With `ppc_N` the key `<particles>/ppc` must not be given. The resolved species table is printed at start-up and stored in the restart files; a restart with a different table stops.
- `dust_nsh`, `dust_si_unstrat`, `dust_si_strat`, `dust_damping_amr`, `dusty_wave_amr`, `dusty_wave` and `dust_damping` take the new keys (`Z_N` only `dust_si_strat`). `streaming_linear`, `dust_freefall`, `dust_jeans` and `dust_snapshot_clump` take only the legacy abundance and sampling keys (the new stopping-time keys work everywhere except in `dust_jeans`).

### Outputs

| Output | Content |
| --- | --- |
| `file_type = pbin` | every active particle: position, velocity, mass, species, stopping time, cloud shape code (level and lobes), 64-bit tag; a per-block table (gid, rank, level, logical location, bounds); the full input in the header (species and stopping time become header constants with one species and fixed stopping times).  One file per output time, written with MPI-IO.  Reader: `vis/python/pbin.py` |
| `file_type = phst` | one text line per output time: number of particles, mass, momentum, kinetic energy, `z_rms`, maximum dust density with its position and level, PC2 stiffness; with G/R also the number of merge groups and removed particles.  Reader: `vis/python/phst.py` (or `athena_read.hst`) |
| `file_type = bin`, `variable = hydro_w` (or `hydro_u`) | adds `dust_total_rho` and `dust_total_mom1..3` automatically when dust is present (`dust_total = false` switches this off) |
| `variable = dust_total` | `dust_total_rho`, `dust_total_mom1..3` alone |
| `variable = dust_species` | `dust_sN_rho`, `dust_sN_mom1..3` for every species (single fields: `dust_sN_rho`, `dust_sN_mom`, `dust_sN_momK`); costs one deposit per species |
| `precision = double` | float64 mesh fields in `bin` output (float32 is the default) |

The dust fields are deposited with the run's own representation (TSC, or the refinable clouds on a refined mesh). `vis/python/bin_convert.py` reads the `bin` files; the dust fields are ordinary variables there.

## 2. Dust self-gravity

Add a `<gravity>` block:

```ini
<gravity>
enabled = true
source = dust            # dust (default with dust), gas, both
kick = dust              # which components feel the field (default = source)
four_pi_G = 0.05
vertical_bc = isolated   # periodic, or isolated in x3 (3-D)
rtol = 1e-10
kdk_reuse = true         # reuse the closing field at the next opening solve when nothing changed
guess = extrapolate      # first guess from the last two potentials
```

- The dust mass is deposited with the same clouds/TSC as the drag; the acceleration is gathered at the particles and applied as a kick-drift-kick around the step.
- Works with refined meshes (static and adaptive), the shearing box (shear-periodic x1) and several MPI ranks. The settings above are the ones used for the stratified SI runs; the self-gravity can be switched on at a restart ([`si3d_strat_amr1280_selfgrav_restart.athinput`](inputs/dust_amr/si3d_strat_amr1280_selfgrav_restart.athinput)).
- Gas as a source (`source = gas | both`) needs `experimental_gas_source = true` (still experimental).
- Validation inputs: [`dust_freefall_u128`](inputs/dust_amr/dust_freefall_u128.athinput), [`dust_freefall_amr`](inputs/dust_amr/dust_freefall_amr.athinput) (homologous collapse of a cold sphere; Lagrangian radii and energy in `freefall_history.jsonl`) and [`dust_jeans_u128`](inputs/dust_amr/dust_jeans_u128.athinput) (linear growth of a dusty Jeans mode).
- Outputs `gravity_phi`, `gravity_g1..3` (or `gravity`) write the potential and acceleration.

## 3. Dust on refined meshes

### Refinable clouds

With `amr_pic = refinable_cloud` every dust particle is a cloud: a P1 (hat-shaped, piecewise linear) mass profile whose width is set by the level of the cloud. Integrated over the cells of its own level the hat gives exactly the TSC weights, so on a uniform mesh the clouds are TSC particles.

- **Split (exact).** When a cloud enters a finer region it is split into the lobed children of the P1 refinement relation: 3 per direction (9 in 2-D, 27 in 3-D) at offsets `0, +-dx/2` (`dx` the parent's cell size) with mass fractions `1/4, 1/2, 1/4` per direction. The sum of the children is identically the parent hat, so the deposited density and the drag on the gas do not change at the split. Children inherit the parent's velocity.
- **X (exact stencil re-merge).** A complete family of children that is still in its split configuration is merged back into its parent (the inverse of the split). Used only in `careful_merge`; off in a 3-D shearing box, where the shear separates the members of a family.
- **G (quantum merge).** Where the mesh is coarse again, fine clouds are merged into clouds of the coarser level, group by group. A group is accepted only if the merged cloud's velocity error moves its mass by less than `alpha` cells within a damping time (`|dv| < alpha dx / t_s`), and if the total change of the deposited density in a pass stays below the budget `beta`.
- **R (resampling).** In regions where many clouds pile up (fine clouds left in a coarse region, dense clumps) the clouds of a small patch are replaced by fewer clouds that keep the patch's mass, momentum and deposited density on the mesh (the constraints are solved exactly), with each new cloud's mass at most `eta` times the nominal cloud mass of its level.

G and R run on a schedule (every `0.25 / Omega` in a shearing box, every N cycles otherwise).

### Switching it on

```ini
<dust>
amr_pic = refinable_cloud       # refinable_cloud (default on a refined mesh) | instant_switch
refinable_mode = default        # default | careful_merge | split_only
```

| `refinable_mode` | Stages | Use |
| --- | --- | --- |
| `default` | split + G + R | production: adaptive meshes, 2-D BA, 3-D stratified SI |
| `careful_merge` | split + X + G + R with quiet-state protections (structure-relative G budget, R skips cold patches); X off in a 3-D shearing box | linear tests and quiet states (LinA) |
| `split_only` | split only, no X / G / R | runs where clouds mostly move into finer regions (e.g. a collapsing cold sphere); clouds that leave a fine region stay fine clouds |

**Quiet linear tests** (LinA and similar) need `careful_merge` or `split_only`. The production `default` (G and R without the quiet gates) destroys a linear mode: on LinA the growth-rate error is +14% to +147% and the eigenfunction is about 100% off-mode. `default` is for nonlinear runs (BA, stratified SI). On LinA `careful_merge` matches `split_only` within 0.01-0.17 percentage points of the growth rate with about 30% fewer particles.

`amr_pic = instant_switch` uses point particles with TSC on the block that owns them (no clouds): a particle's kernel switches to the cell size of a level as soon as it enters a block of that level. It needs no further keys. Any preset key given explicitly (see section 4) overrides the preset's value; the resolved set is printed at start-up (`# DUST_PIC ...`). The old `experimental_*` switches are no longer needed and are ignored with a note.

### Refinement criteria

Dust-density trigger (`<mesh_refinement>/refinement = adaptive`, `<amr_criterion0>/method = user`):

```ini
<dust>
amr_density_trigger = true
amr_rho_units = code            # code: rho_d in code units; mean: rho_d / <rho_d> (default)
amr_rho_refine = 64             # refine a level-0 block where rho_d >= 64 ...
amr_rho_level_factor = 4        # ... and a level-l block where rho_d >= 64 * 4^l
amr_rho_derefine = 32           # a level-l block coarsens below this * 4^(l-1)
amr_rho_buffer = 2              # cells of the block's level around the dense cells
# amr_rho_refine_levels = 64 256 5120   # explicit thresholds per level instead of the factor
```

Scheduled refinement (`amr_sched_n = K` and for `k = 1..K`: `amr_schedk_t0`, `_t1`, `_x1min`, `_x1max`, `_x2min`, `_x2max`, optional `_x3min`, `_x3max`, `_level`) refines boxes in given time windows; it excludes the density trigger. Static meshes use the usual `<refined_regionN>` blocks.

### Examples (`inputs/dust_amr/`)

| Input | Set-up |
| --- | --- |
| [`lina_smr_careful_merge`](inputs/dust_amr/lina_smr_careful_merge.athinput) | LinA linear SI mode on a nested static mesh (64^2 root, central square at 128^2), `careful_merge`, to t = 2 pi; growth-rate error about -0.40% (uniform 64^2: -0.24%, 128^2: -0.059%); with `<mesh_refinement>/prolong_amplitude_eps = 1e-3` (amplitude-aware gas prolongation) about -0.23% |
| [`ba2d_amr`](inputs/dust_amr/ba2d_amr.athinput) | 2-D BA streaming instability (tau_s = 1, eps = 0.2), root 128^2, 5 levels, density trigger 10 * 2^l / 3.3 * 2^(l-1), `default`, to t = 600 |
| [`si3d_strat_amr1280`](inputs/dust_amr/si3d_strat_amr1280.athinput) | 3-D stratified SI, root 320/H, AMR to 1280/H, trigger 64 / 256 (code units), WENO-Z + HLLC, Li & Youdin (2018) vertical boundaries, to 25 orbits |
| [`si3d_strat_amr1280_selfgrav_restart`](inputs/dust_amr/si3d_strat_amr1280_selfgrav_restart.athinput) | restart of the run above with dust self-gravity and a fourth level at rho_d >= 5120 |
| [`damping3d_sched_amr`](inputs/dust_amr/damping3d_sched_amr.athinput) | 3-D drag damping with a refined slab moving through the box |
| [`dustywave2d_sched_amr`](inputs/dust_amr/dustywave2d_sched_amr.athinput) | 2-D damped dusty sound wave with a band refined to level 1, then 2, then coarsened |
| [`nsh2d_sched_amr`](inputs/dust_amr/nsh2d_sched_amr.athinput) | 2-D three-species NSH drift equilibrium with a refined square switched on and off; `dust_species` output |
| [`dust_freefall_u128`](inputs/dust_amr/dust_freefall_u128.athinput), [`dust_freefall_amr`](inputs/dust_amr/dust_freefall_amr.athinput), [`dust_jeans_u128`](inputs/dust_amr/dust_jeans_u128.athinput) | self-gravity validation (section 2) |

### Problem generators

Every dust problem has its own problem generator (`<problem>/pgen_name`). The pieces they share (species, NSH drift equilibrium, particle placement, history columns, refinement requests) are in [`src/pgen/dust_amr/dust_setup.hpp`](src/pgen/dust_amr/dust_setup.hpp).

| `pgen_name` | Problem | Example |
| --- | --- | --- |
| `dust_nsh` | multi-species NSH drift equilibrium in the 2-D r-z or 3-D shearing box, held to round-off (errors in `<basename>-errs.dat`); perturbations for mesh tests: `two_stream_v`, `blob_amp`, `gas_vx_amp`, `dust_ymode_amp` | `nsh2d_sched_amr` |
| `dust_si_unstrat` | streaming instability in the unstratified shearing box (e.g. run BA): the NSH equilibrium with the dust at random positions | `ba2d_amr` |
| `dust_si_strat` | streaming instability in the stratified 3-D shearing box: hydrostatic gas, Gaussian dust layers (`H_p_N` or `dust_hp`, `stratified_dust_sampling = gaussian` or `weight`, `dust_zcut`), the NSH drift of every height (`nsh_vertical`), the vertical boundaries of Li et al. (2018) (`vertical_bc = li2018`) | `si3d_strat_amr1280` |
| `dust_damping_amr` | drag damping of uniform dust moving through gas at rest (`dust_test_v0`), no rotation | `damping3d_sched_amr` |
| `dusty_wave_amr` | damped dusty sound wave (`amp`, `wave_n`), no rotation | `dustywave2d_sched_amr` |
| `dust_freefall` | collapse of a cold dust sphere under its own gravity | `dust_freefall_u128`, `dust_freefall_amr` |
| `dust_jeans` | linear dusty Jeans mode | `dust_jeans_u128` |

`dust_nsh` is in `src/pgen/tests/`, the others in `src/pgen/dust_amr/`. A key that belongs to another of these problem generators (for example `stratified_dust` in `dust_nsh`) stops the run with the name of the right one. On a refined mesh the particles start as root parents: `<problem>/cloud_root_random = true` places `ppc` parents per root cell over the whole box (`root_placement = random`, `lattice` or `generic`), and each parent is split exactly to the leaves it overlaps. On a uniform mesh (or with `amr_pic = instant_switch`) the same keys give the same parents as point particles, so uniform and refined runs start from the same samples. Without `cloud_root_random` the particles are point particles on a lattice or at random positions (`particle_placement`). The `<problem>/ba_*` keys (2-D) add the field snapshots and the particle density distribution of [`dust_snapshots.cpp`](src/pgen/dust_amr/dust_snapshots.cpp).

## 4. Tuning parameters

All in `<dust>` unless noted. The defaults are the validated production values; the preset keys (marked P) take the value of `refinable_mode` unless given.

| Key | Default | Meaning |
| --- | --- | --- |
| `cloud_merge` (P) | on (off in `split_only`) | stage G |
| `cloud_merge_alpha` (P) | 0.05 (2-D), 0.2 (3-D) | G velocity tolerance: cells moved within one damping time |
| `cloud_merge_alpha_leaf` (P) | true | 3-D: measure alpha in cells of the leaf that holds the merge cell |
| `cloud_merge_budget` (P) | 0.3 | G pass budget beta on the change of the deposited density (-1: no budget) |
| `cloud_merge_budget_kappa` (P) | 0 (1 in `careful_merge`) | structure-relative G budget |
| `cloud_merge_schedule`, `cloud_merge_dt_omega` (P) | `time`, 0.25 | G every 0.25/Omega (shearing box) |
| `cloud_merge_interval` (P) | 10 | G every N cycles (`cloud_merge_schedule = cycle`, no omega0) |
| `cloud_merge_stencil` (P) | off (on in `careful_merge`) | stage X |
| `cloud_merge_buffer` | 2 | G merge cells lie at least this many cells (of the target level) from any finer leaf |
| `cloud_resample` (P) | on (off in `split_only`) | stage R |
| `cloud_resample_eta` (P) | 1.5 | R mass cap: a cloud of level L ends at most at eta_L times the level's nominal cloud mass; eta_L = eta on every level unless `cloud_resample_eta_levels` is set |
| `cloud_resample_eta_levels` | (none) | `auto`: eta_L = `min(eta 2^(L/2), max(4, eta))` in 3-D and `min(eta 2^(L/4), max(2, eta))` in 2-D (with eta = 2 in 3-D: 2, 2.83, 4, 4, ...), a cap that grows with the level up to 4 in 3-D and 2 in 2-D (unless eta is larger), below 2^d, at which a level-L cloud would carry the nominal mass of level L-1. A list of numbers gives eta_L per level as given (also above 4; <= 0: no cap on that level), e.g. `2 2.83 4 4` for levels 0-3; the last value holds for deeper levels |
| `cloud_resample_schedule`, `cloud_resample_dt_omega` (P) | `time`, 0.25 | R every 0.25/Omega (shearing box); `cloud_resample_dt` in code time instead |
| `cloud_resample_interval` (P) | 625 | R every N cycles (`cloud_resample_schedule = cycle`) |
| `cloud_resample_after_regrid` (P) | false | also resample right after each regrid |
| `cloud_resample_cold_alpha` (P) | 0 (0.05 in `careful_merge`) | R leaves cold (single-velocity) patches alone |
| `cloud_resample_patch_split` | 1024 | split R patches holding more clouds than this into quadrants |
| `amr_rho_units` | `mean` | trigger thresholds in units of the mean dust density, `code`, or `roche` |
| `amr_rho_refine`, `amr_rho_derefine` | (required) | refine a level-l block at `refine * f^l`, coarsen it below `derefine * f^(l-1)` (`f = amr_rho_level_factor`) |
| `amr_rho_level_factor` | 1 | thresholds multiplied by this per level |
| `amr_rho_refine_levels` | (none) | explicit refinement thresholds per level (starts with `amr_rho_refine`) |
| `amr_rho_buffer` | 2 | buffer cells around dense cells |
| `amr_shear_face_limit` | on | 2:1 level rule across shear-periodic x1 faces |
| `<mesh_refinement>/ncycle_check` | (AthenaK) | cycles between refinement checks (50 is enough with self-gravity) |
| `load_balance_cloud_weight` | 4 | block cost = cells + weight x clouds for the load balance |
| `pc2_feedback_damping` | true | PC2 damping of the feedback stiffness |
| `particle_x3_bc` | outflow (reflect at a reflecting gas wall) | dust at non-periodic x3 faces |
| `<gravity>/kdk_reuse` | false | reuse the closing field at the next opening solve when mesh, phase and dust mass are unchanged |
| `<gravity>/guess` | `previous` | `extrapolate`: linear extrapolation of the last two potentials |
| `<gravity>/rtol`, `max_iterations` | 1e-10, 150 | multigrid convergence |
| `<gravity>/setup` | `distributed` | `distributed`: each rank builds its own rows, rediscretized coarse levels, shear-phase updates of the changed rows only (scales, GPU); `reference`: replicated global matrices for accuracy checks on small problems |
| `<gravity>/solver` | `multigrid` | the only solver |
| `<gravity>/cfl` | 0.25 | time-step safety factor for the acceleration and free-fall limits |

## 5. Known limitations

- MHD in a shearing box with shear-periodic x1 boundaries runs on a uniform mesh only; with SMR/AMR the run stops at start-up. The face-level ghost exchange used by refined shearing boxes exists for cell-centred (hydro, dust) variables only.
- `refinable_mode = default` is meant for nonlinear runs. It damps small linear modes (LinA-like tests); use `careful_merge` or `split_only` there.
