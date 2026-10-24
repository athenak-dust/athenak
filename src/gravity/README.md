# Composite self-gravity development

This module is being implemented on `selfgrav-amr-shear`, from dust development
commit `6c466d6d`. It is disabled unless `<gravity>/enabled = true`. Do not infer
completion from the existence of this directory: whole-code validation and the
isolated slab boundary are still in progress.

The requested endpoint is hydro + multigrid + AMR + shearing box, including dynamic
refine/derefine, restart, MPI and GPU. Two gravity boundary configurations are
required: periodic, and horizontally shearing-periodic/periodic with vertical
isolation. Gas outflow and isolated gravitational potential are separate choices.
Future deposited dust density and particle force gathering must reuse the same
field solver; this change does not enable dust gravity yet.

## Discrete operator

`composite.*` constructs one potential unknown per active leaf cell. Every face
fragment contributes the same integrated normal flux with opposite signs to the
two adjacent cells, including coarse/fine and sheared periodic interfaces.
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

`sparse_mg.*` uses dyadic geometric aggregates, smoothed interpolation and Galerkin
coarse matrices. A V-cycle preconditions restarted flexible GMRES because the
composite stencil is generally nonsymmetric. Bulk matrix, smoother, and vector
operations use Kokkos, with local MPI rows and sparse reusable halo plans. Only
the tiny coarsest problem uses a root CPU LU solve.

## Coupling and lifecycle

`SelfGravity` owns potential and acceleration in active-cell arrays. Gravity half
kicks enclose the existing hydro/orbital-advection update. Each kick keeps density
fixed and preserves ideal-gas internal energy; this is a second-order splitting,
not a claim of exact total gravitational-energy conservation. Density is solved
at the beginning and end physical times, and primitives/ghost values are refreshed
after kicks. Acceleration and free-fall limits participate in the timestep.

Regrid/load-balance invalidates the composite system and solves again from the new
active density. Restart reconstructs the derived potential from restored density
and time. No independent stale potential is read from a restart.

`SetExtraDensitySource` is the future source interface: a callback fills an
additional active-cell density at each solve. Particle deposit/gather and its
time coupling remain the responsibility of the future dust implementation.

## Current evidence and remaining work

The standalone 2D/3D mixed-mesh tests show approximately second-order potential
and acceleration $L_2$ errors. Maximum force convergence near interfaces is
weaker (roughly 1.3--1.6 in tested cases). The C++ matrices match the Python
reference. The engine passes CPU/CUDA and 1/2/4-rank tests, including zero-row
ranks on multigrid levels. These are not whole-AthenaK validation results.

Whole-code manufactured solutions, coupled dynamics, topology changes and
restart checks are in progress. The vertical slab boundary uses source-based
horizontal Fourier sums and a free-space vertical Green function. Its zero mode
must include the full surface density, not periodic mean subtraction.

There are two known performance limitations: host setup currently holds global
geometry and matrices on each rank, and the 3D Galerkin hierarchy has significant
fill. Direct slab boundary sums cost $O(N_{cell}N_{mode})$. They are an accuracy
reference to be measured and improved, not a claim of production scaling.

Detailed provenance, bounded test batches, and outstanding requirements are in
the internal run records.

References: [Athena++ AMR multigrid](https://arxiv.org/abs/2302.13903),
[its conservative interface formula](https://github.com/PrincetonUniversity/athena/blob/main/src/gravity/mg_gravity.cpp).
The new shearing composite operator and device implementation are separate code;
the Athena++ reference does not establish their correctness.
