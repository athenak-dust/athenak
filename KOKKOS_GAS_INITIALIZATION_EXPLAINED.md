# Line-by-Line Explanation of an AthenaK Gas-Initialization Kernel

This document explains every line and important token in this AthenaK problem-generator
fragment:

```cpp
auto &u0 = pmbp->phydro->u0;
par_for("uniform_dust_gas", DevExeSpace(), 0, nmb - 1,
        ks, ke, js, je, is, ie,
KOKKOS_LAMBDA(int m, int k, int j, int i) {
  u0(m, IDN, k, j, i) = rho0;
  u0(m, IM1, k, j, i) = rho0*vx0;
  u0(m, IM2, k, j, i) = 0.0;
  u0(m, IM3, k, j, i) = 0.0;
});
```

The fragment initializes every active gas cell in every MeshBlock in the current
MeshBlock pack. It sets a uniform density `rho0`, a uniform x1 velocity `vx0`, and zero
x2 and x3 velocity.

## The shortest useful translation

Ignoring parallelism for a moment, the code means approximately:

```cpp
for (int m = 0; m <= nmb - 1; ++m) {
  for (int k = ks; k <= ke; ++k) {
    for (int j = js; j <= je; ++j) {
      for (int i = is; i <= ie; ++i) {
        u0(m, IDN, k, j, i) = rho0;
        u0(m, IM1, k, j, i) = rho0*vx0;
        u0(m, IM2, k, j, i) = 0.0;
        u0(m, IM3, k, j, i) = 0.0;
      }
    }
  }
}
```

The important difference is that Kokkos may execute the cell iterations concurrently
on CPU threads or accelerator threads. The nested-loop version is therefore a mental
model, not an exact description of the execution order.

## The objects involved

Before examining individual lines, it helps to know the relevant object hierarchy:

```text
pmbp                         pointer to a MeshBlockPack
  -> phydro                  pointer to its Hydro module
       -> u0                 five-dimensional Kokkos View of conserved gas variables
```

In this fork, `Hydro::u0` is declared in
[`src/hydro/hydro.hpp`](src/hydro/hydro.hpp) as:

```cpp
DvceArray5D<Real> u0;  // conserved variables
```

`DvceArray5D<Real>` is an AthenaK alias defined in
[`src/athena.hpp`](src/athena.hpp):

```cpp
template <typename T>
using DvceArray5D = Kokkos::View<T *****, LayoutWrapper, DevMemSpace>;
```

After substituting `T = Real`, the conceptual type is:

```cpp
Kokkos::View<Real *****, Kokkos::LayoutRight, DevMemSpace>
```

It is a five-dimensional array-like handle whose storage is accessible from AthenaK's
configured device execution space. Here, “device” means the selected computational
backend; it is not necessarily a GPU.

The dimensions of `u0` are allocated in
[`src/hydro/hydro.cpp`](src/hydro/hydro.cpp) in this order:

```cpp
Kokkos::realloc(u0, nmb, nhydro + nscalars, ncells3, ncells2, ncells1);
```

Thus its five indices mean:

```text
u0(meshblock, variable, k, j, i)
```

or, more compactly:

```text
u0(m, n, k, j, i)
```

`LayoutRight` means the rightmost index, `i`, is contiguous and changes fastest in
memory. Adjacent x1 cells are therefore adjacent in storage, which is favorable for the
common inner-x1 traversal pattern.

## Line 1: obtain a local name for the gas array

```cpp
auto &u0 = pmbp->phydro->u0;
```

### `pmbp`

`pmbp` is a pointer to a `MeshBlockPack`:

```cpp
MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;
```

A pointer stores the address of another object. The pointer is expected to be non-null
because the pgen checked it earlier.

### First `->`: `pmbp->phydro`

The arrow operator accesses a member through a pointer. This:

```cpp
pmbp->phydro
```

is shorthand for:

```cpp
(*pmbp).phydro
```

It dereferences `pmbp`, obtaining the `MeshBlockPack`, and then selects its `phydro`
member. `phydro` is itself a pointer, this time to the `hydro::Hydro` object associated
with the pack.

### Second `->`: `pmbp->phydro->u0`

Because `phydro` is also a pointer, another arrow is needed:

```cpp
pmbp->phydro->u0
```

This reaches the `u0` member of the Hydro object. Fully parenthesized, it is equivalent
to:

```cpp
(*((*pmbp).phydro)).u0
```

The arrow-chain version is much easier to read.

### `auto`

`auto` asks the compiler to infer the declared type from the expression on the right.
The underlying type is `DvceArray5D<Real>`.

Without `auto`, the line could be written:

```cpp
DvceArray5D<Real> &u0 = pmbp->phydro->u0;
```

`auto` avoids repeating a long implementation type and remains correct if the exact
view type changes later.

### `&`

In a declaration, `&` makes `u0` a C++ reference. A reference is another name—an
alias—for an existing object. Therefore, this declaration does not replace the Hydro
array and does not allocate a new five-dimensional data array.

After the declaration, the local name:

```cpp
u0
```

refers to the same Kokkos View object as:

```cpp
pmbp->phydro->u0
```

The reference is useful because it makes the kernel shorter and clearer.

There is a subtle Kokkos detail: even copying a `Kokkos::View` normally makes only a
small, reference-counted handle that points to the same allocation; it does not deep-copy
all the field data. Nevertheless, `auto &` avoids even that host-side handle copy at this
line and clearly states that the local name aliases the Hydro member.

### `=` and `;`

Here `=` initializes the reference so that it aliases `pmbp->phydro->u0`. A C++
reference must be bound when declared and cannot later be rebound to a different object.
The semicolon terminates the declaration statement.

## Lines 2–3: describe the parallel iteration space

```cpp
par_for("uniform_dust_gas", DevExeSpace(), 0, nmb - 1,
        ks, ke, js, je, is, ie,
```

### `par_for` is an AthenaK wrapper

`par_for` is not the ordinary spelling of a Kokkos API function. AthenaK defines several
overloaded `par_for` helper templates in [`src/athena.hpp`](src/athena.hpp). They wrap
`Kokkos::parallel_for` and provide convenient one- through five-dimensional index
ranges.

This call supplies eight integer bounds after the execution space, so C++ selects the
four-dimensional overload:

```cpp
template <typename Function>
inline void par_for(const std::string &name, DevExeSpace exec_space,
                    const int &nl, const int &nu,
                    const int &kl, const int &ku,
                    const int &jl, const int &ju,
                    const int &il, const int &iu,
                    const Function &function);
```

In the wrapper the first dimension is named `n`, but in this call the lambda chooses to
name that argument `m`, because it represents a MeshBlock index. Parameter names do not
have to match between a caller and a called function; position determines which value is
which.

### `"uniform_dust_gas"`

This string is the kernel label. It does not change the mathematical result. Kokkos can
use it in profiling, tracing, debugging, and performance tools, so labels should be
descriptive and preferably distinct.

It answers the diagnostic question “which kernel was running?” much better than an
anonymous number would.

### `DevExeSpace()`

`DevExeSpace` is defined in `src/athena.hpp` as:

```cpp
using DevExeSpace = Kokkos::DefaultExecutionSpace;
```

The build configuration determines the concrete default execution space. Depending on
how Kokkos was configured, it can be a serial CPU backend, a threaded CPU backend, CUDA,
HIP, or another supported backend.

The parentheses construct an execution-space instance:

```cpp
DevExeSpace()
```

That instance is passed through AthenaK's wrapper to a Kokkos range policy. The code is
therefore explicitly launching this work in AthenaK's selected device execution space;
Kokkos is not dynamically deciding between CPU and GPU by inspecting the arithmetic.

### `0, nmb - 1`

These are the inclusive lower and upper bounds of the first dimension:

```text
m = 0, 1, 2, ..., nmb - 1
```

`nmb` is the number of MeshBlocks in the current pack. C++ arrays are zero-indexed, so
`nmb` objects occupy indices `0` through `nmb - 1`.

The subtraction matters. Passing `nmb` as the upper bound would request one nonexistent
MeshBlock and cause an out-of-bounds access.

AthenaK's `par_for` bounds are inclusive. This differs from the half-open convention of
many standard C++ loops and from the underlying `Kokkos::RangePolicy`, whose end is
exclusive.

### `ks, ke`

These are the inclusive active-cell bounds in the x3 or `k` direction:

```text
k = ks, ks + 1, ..., ke
```

### `js, je`

These are the inclusive active-cell bounds in the x2 or `j` direction:

```text
j = js, js + 1, ..., je
```

### `is, ie`

These are the inclusive active-cell bounds in the x1 or `i` direction:

```text
i = is, is + 1, ..., ie
```

The active indices do not generally start at zero because the allocated arrays contain
ghost zones around the active domain. A pgen normally initializes the active region;
boundary operations later populate ghost cells.

### Why the line ends with a comma

The `par_for` function call has not finished. Everything shown so far is a sequence of
arguments, and the final argument—the operation to perform at every index—comes next.
The comma separates `ie` from that lambda argument.

## Line 4: define the operation performed at each cell

```cpp
KOKKOS_LAMBDA(int m, int k, int j, int i) {
```

### What a lambda is

A lambda is an unnamed function object written directly at the place where it is used.
The ordinary C++ shape is:

```cpp
[capture-list](parameters) {
  function body
}
```

The lambda is passed to `par_for` as its final argument. AthenaK's wrapper eventually
invokes it once for each `(m,k,j,i)` point in the requested range.

### `KOKKOS_LAMBDA`

`KOKKOS_LAMBDA` is a portability macro. For a non-accelerator build it is essentially:

```cpp
[=]
```

For CUDA or HIP it also adds the compiler annotations needed to call the lambda on the
device. Conceptually, the line resembles:

```cpp
[=](int m, int k, int j, int i) {
```

`[=]` means that referenced variables from the surrounding scope are captured by value.
In this kernel those captured values include the Kokkos View handle `u0` and scalar
parameters such as `rho0` and `vx0`.

Capturing a Kokkos View by value makes a shallow copy of its small handle. Both handles
refer to the same underlying field allocation. This is why assignments through the
captured `u0` change the real Hydro state rather than a private copy of the full gas
array.

Capturing ordinary scalar values such as `rho0` and `vx0` gives the kernel its own copies
of those values, suitable for device execution.

### `int m, int k, int j, int i`

These are the lambda parameters supplied by the AthenaK wrapper:

- `m`: MeshBlock index in this pack.
- `k`: active cell index in x3.
- `j`: active cell index in x2.
- `i`: active cell index in x1.

Each invocation receives one unique index tuple. On a GPU, many invocations may be
executed by different GPU threads. On a threaded CPU backend, they may be distributed
among CPU threads. No ordering between different cells should be assumed.

### `{`

The opening brace begins the body of the lambda—the work to perform for one particular
cell in one particular MeshBlock.

## Line 5: initialize gas density

```cpp
u0(m, IDN, k, j, i) = rho0;
```

### `u0(...)`

A Kokkos View overloads the function-call operator `operator()`, so parentheses are used
for multidimensional indexing:

```cpp
u0(m, variable, k, j, i)
```

This resembles a function call syntactically, but it selects one element of the
five-dimensional field.

### `m`

The first index selects which MeshBlock's data is being initialized.

### `IDN`

The second index selects a physical variable. `IDN` is an enumerator defined in
`src/athena.hpp`:

```cpp
enum VariableIndex {
  IDN = 0,
  IM1 = 1,
  IM2 = 2,
  IM3 = 3,
  IEN = 4,
  // ...
};
```

`IDN` therefore selects variable slot zero, the gas mass density. The name convention is
approximately “index of density.” Using a named enumerator is safer and clearer than
writing the unexplained integer `0`.

### `k, j, i`

These select the spatial cell within MeshBlock `m`.

### `= rho0`

The assignment stores the uniform initial density `rho0` into that cell. Every kernel
invocation receives the same captured scalar `rho0`, which is why the resulting density
field is spatially uniform.

This is conserved mass density, but density is also a primitive variable, so no
conversion is necessary for this component.

## Line 6: initialize x1 momentum density

```cpp
u0(m, IM1, k, j, i) = rho0*vx0;
```

`IM1` selects variable slot one: momentum density in the x1 direction.

The crucial point is that `u0` stores **conserved** variables. Its x1 slot does not store
velocity `vx0`; it stores:

```text
x1 momentum density = mass density * x1 velocity
                    = rho0 * vx0
```

Writing this would generally be wrong:

```cpp
u0(m, IM1, k, j, i) = vx0;  // wrong unless rho0 happens to equal 1
```

The multiplication uses the two scalar values captured from the surrounding pgen. It is
performed independently for every cell. A compiler will likely recognize that the
product is constant and avoid recalculating it unnecessarily, but correctness does not
depend on that optimization.

## Line 7: initialize x2 momentum density

```cpp
u0(m, IM2, k, j, i) = 0.0;
```

`IM2` selects momentum density in the x2 direction. Assigning zero represents zero x2
velocity because:

```text
rho * vy = rho * 0 = 0
```

`0.0` is a floating-point literal. The View element has type `Real`, which is either
`double` or `float` depending on the AthenaK precision build. C++ converts the literal to
the appropriate `Real` type when assigning it.

It is important to initialize this slot explicitly. Newly allocated accelerator memory
should not be assumed to contain zero unless the allocation or code explicitly
guarantees it.

## Line 8: initialize x3 momentum density

```cpp
u0(m, IM3, k, j, i) = 0.0;
```

`IM3` selects momentum density in the x3 direction. This sets the third component of gas
momentum to zero.

Even in a two-dimensional calculation where `nx3 = 1`, AthenaK's Newtonian Hydro state
still carries all three momentum components. A 2-D calculation can therefore represent
an out-of-plane velocity. Setting `IM3` explicitly declares that this problem has none.

## Line 9: close the lambda and launch call

```cpp
});
```

This line contains three separate pieces of punctuation.

### `}`

The brace closes the lambda body begun after `KOKKOS_LAMBDA(...)`.

### `)`

The parenthesis closes the `par_for(` function call. The entire lambda expression is the
last argument passed to `par_for`.

### `;`

The semicolon terminates the C++ expression statement containing the `par_for` call.

Conceptually, the structure is:

```cpp
par_for(
  kernel_name,
  execution_space,
  index_bounds,
  lambda_operation
);
```

## What AthenaK's `par_for` does internally

The exact four-dimensional wrapper is implemented in `src/athena.hpp`. It does not use
a native four-dimensional Kokkos policy. Instead, it calculates the total number of
points:

```cpp
const int nn = nu - nl + 1;
const int nk = ku - kl + 1;
const int nj = ju - jl + 1;
const int ni = iu - il + 1;
const int nnkji = nn*nk*nj*ni;
```

For this call, those quantities correspond to:

```text
nn = nmb
nk = ke - ks + 1
nj = je - js + 1
ni = ie - is + 1
```

The wrapper then launches a one-dimensional Kokkos range:

```text
idx = 0, 1, ..., nnkji - 1
```

For each flat `idx`, it reconstructs four indices with integer division and remainders,
adds the lower-bound offsets, and finally calls your function:

```cpp
function(n, k, j, i);
```

Your lambda names that first argument `m`, so the call becomes conceptually:

```cpp
your_lambda(m, k, j, i);
```

This flattening is an implementation and performance choice. The comment in
`src/athena.hpp` reports that one-dimensional range policies were generally faster than
multidimensional Kokkos range policies in related experiments.

## Inclusive bounds versus the underlying half-open range

Two conventions are present at different layers:

```text
AthenaK par_for:       [lower, upper]       upper is included
Kokkos RangePolicy:   [begin, end)         end is excluded
```

AthenaK converts between them. For example, `m = 0` through `m = nmb - 1` contains
`nmb` values. After multiplying all dimension lengths, it passes Kokkos the half-open
flat range `[0, nnkji)`.

This distinction is a common source of off-by-one bugs. When calling AthenaK's
`par_for`, supply the last valid index as the upper bound.

## Why these writes are safe in parallel

Every lambda invocation receives a unique tuple `(m,k,j,i)`. Each invocation writes
four variable slots at that unique cell:

```text
(m, IDN, k, j, i)
(m, IM1, k, j, i)
(m, IM2, k, j, i)
(m, IM3, k, j, i)
```

No other invocation has the same `(m,k,j,i)` tuple, so no two threads write the same
elements. There is no write-write race, and atomics are unnecessary.

A dangerous pattern would be having every cell update one shared scalar:

```cpp
total_mass += u0(m, IDN, k, j, i);  // race in an ordinary parallel_for
```

That operation requires a Kokkos reduction or a suitable atomic operation. Directly
assigning independent array elements, as the initialization kernel does, is the simplest
and safest parallel pattern.

## What the kernel does not initialize

This example is specifically for an isothermal Hydro configuration. Isothermal Hydro
has four evolved conserved variables: density and three momentum components. Pressure
is determined from density and the configured isothermal sound speed, so there is no
evolved energy slot to initialize.

For an ideal-gas EOS, Hydro has a fifth evolved variable and the pgen must also set:

```cpp
u0(m, IEN, k, j, i) = pressure/(gamma - 1.0)
                      + 0.5*rho*(vx*vx + vy*vy + vz*vz);
```

Omitting `IEN` for an ideal EOS would leave the gas energy invalid or uninitialized.

The kernel also initializes active cells only. It does not initialize ghost cells,
primitive variables `w0`, particle state, or dust deposited fields. Those have separate
initialization or conversion paths.

## Host code and device code in the snippet

The snippet crosses an important boundary:

```cpp
auto &u0 = pmbp->phydro->u0;   // ordinary host-side setup
par_for(...                     // host-side kernel launch
KOKKOS_LAMBDA(...) {            // device-callable kernel body
  ...                           // executes in DevExeSpace
});
```

The host prepares array handles, bounds, and scalar parameters, then requests a kernel
launch. The lambda body must be valid in the selected execution space. This is why one
reads input parameters before the kernel and captures simple scalar values and Kokkos
Views rather than attempting to access `ParameterInput`, `std::cout`, or an ordinary
host-only container from inside the lambda.

Depending on the backend, a Kokkos kernel launch can be asynchronous with respect to the
host. Code should not assume that host-side access to the results is safe until the
appropriate execution-space ordering or fence guarantees completion. Back-to-back
kernels submitted to the same execution-space instance are ordered according to the
backend and Kokkos execution-space semantics, while explicit host inspection commonly
requires a fence or a proper mirror/deep-copy operation. This pgen does not immediately
read `u0` on the host, so it does not need an explicit fence at this point.

## Worked size example

Suppose the pack contains:

```text
nmb = 2 MeshBlocks
nx1 = ie - is + 1 = 16 active x1 cells
nx2 = je - js + 1 = 8 active x2 cells
nx3 = ke - ks + 1 = 1 active x3 cell
```

The kernel has:

```text
2 * 1 * 8 * 16 = 256 independent iterations
```

Each iteration makes four assignments, so the kernel performs:

```text
256 * 4 = 1024 field-element writes
```

It initializes exactly the four isothermal conserved variables in all 256 active cells.
It does not touch ghost cells because the supplied ranges begin at `is/js/ks` and end at
`ie/je/ke`.

## A token-by-token summary

| Token or expression | Meaning |
|---|---|
| `auto` | Infer the type from the initializer |
| `&u0` | Declare `u0` as a reference/alias |
| `pmbp->phydro->u0` | Follow two pointers and access the Hydro conserved-field View |
| `par_for` | AthenaK wrapper around `Kokkos::parallel_for` |
| `"uniform_dust_gas"` | Profiling/debugging label for the kernel |
| `DevExeSpace()` | Instance of the configured default execution space |
| `0, nmb-1` | Inclusive MeshBlock-index range |
| `ks, ke` | Inclusive active x3-index range |
| `js, je` | Inclusive active x2-index range |
| `is, ie` | Inclusive active x1-index range |
| `KOKKOS_LAMBDA` | Portable, device-callable lambda capture macro |
| `(int m, int k, int j, int i)` | Indices supplied for one parallel iteration |
| `u0(...)` | Access one element of the five-dimensional Kokkos View |
| `IDN` | Gas density variable index |
| `IM1` | Gas x1 momentum-density index |
| `IM2` | Gas x2 momentum-density index |
| `IM3` | Gas x3 momentum-density index |
| `rho0*vx0` | Convert uniform x1 velocity to conserved momentum density |
| `}` | End the lambda body |
| `)` | End the `par_for` call |
| `;` | End the C++ statement |

## The main ideas to retain

1. `u0` is a five-dimensional Kokkos View containing conserved gas variables.
2. `auto &u0` gives a short local alias to the Hydro module's View.
3. AthenaK's `par_for` describes an inclusive four-dimensional index box.
4. AthenaK internally flattens that box into a one-dimensional Kokkos range.
5. `KOKKOS_LAMBDA` supplies the operation performed independently at every index tuple.
6. The array order is `(MeshBlock, variable, k, j, i)`.
7. Momentum slots contain `rho*v`, not velocity alone.
8. Independent cells make the assignments safe to execute concurrently.
