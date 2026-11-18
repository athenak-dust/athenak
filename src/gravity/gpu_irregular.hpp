//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_GPU_IRREGULAR_HPP_
#define GRAVITY_GPU_IRREGULAR_HPP_
// Device assembly of the irregular rows of a MeshGeometry system (test switch
// <gravity>/test_gpu_assembly, see mesh_geometry.cpp): level jumps, same-level neighbors whose
// patch widths differ in the last bits, transverse offsets, C1 planes.  The kernels follow
// MeshGeometry::Impl::AssembleRow (Faces, Quad, FacePotential, AddDiagonal) in the same order
// with the same expressions; gpu_irregular.cpp is compiled without floating-point contraction
// and calls fma exactly where the host compiler contracts (sums of products).  A row that needs
// what the kernels do not support (an x boundary or isolated-z wall in any face list, a
// rank-deficient two-ring fit, a fit with repeated cells, full row buffers) gets a nonpositive
// status and is assembled on the host.

#include <array>

#include "gravity/gpu_assembly.hpp"

namespace gravity {
namespace gpu {

// Tables of the irregular rows, in addition to Geometry.
struct Irregular {
  int isolated_z = 0;
  Kokkos::View<double*[3], Kokkos::LayoutRight, DevMemSpace> plo, phi;   // patch boxes
  // the candidate patches of face k of patch p (Faces() tree query of any of its cells, a
  // superset in tree order): cand(cand_off(6 p+k) .. cand_off(6 p+k+1)-1)
  Kokkos::View<int*, DevMemSpace> cand_off, cand;
  Kokkos::View<int*, DevMemSpace> rorder;   // patches in row order (Patch(row))
  Kokkos::View<int*, DevMemSpace> rows;     // the owned local rows of the irregular kernels
};

// The two-ring fits (MeshGeometry::Impl::Quad) the listed rows need, each built once: per
// global real row its fit or -1; per fit its cell, status (0 or why it is left to the host),
// first sample (+ the end) and center coefficient; per sample its cell and coefficient.
struct Quads {
  int count = 0;
  GlobalIndex samples = 0;
  Kokkos::View<int*, DevMemSpace> index;
  Kokkos::View<GlobalIndex*, DevMemSpace> cell;
  Kokkos::View<int*, DevMemSpace> status;
  Kokkos::View<GlobalIndex*, DevMemSpace> off;
  Kokkos::View<GlobalIndex*, DevMemSpace> col;
  Kokkos::View<double*[9], Kokkos::LayoutRight, DevMemSpace> val, center;
  // AddDiagonal's analysis of the cells it is called for, each cell once: per global real row
  // its entry or -1; per entry its status (1 the centered three-point branch, 0 the fit, < 0 why
  // it is left to the host) and face neighbors nb[d][side] at 2 d+side
  int diagonals = 0;
  Kokkos::View<int*, DevMemSpace> dindex, dstatus;
  Kokkos::View<GlobalIndex*, DevMemSpace> dcell;
  Kokkos::View<GlobalIndex*[6], Kokkos::LayoutRight, DevMemSpace> dnb;
};

// The fits of the listed rows, then per listed row its status (1 assembled, 2 assembled and an
// interface row, <= 0 left to the host) and the stored entries of each component (0: A, 1+d:
// acceleration d) of the assembled rows; seconds (optional): marks + list, fit counts, fit
// coefficients, row counts
Quads IrregularCount(const Geometry &g, const Irregular &t, bool forces, int components,
                     const DeviceIndex &count0, const DeviceIndex &count1,
                     const DeviceIndex &count2, const DeviceIndex &count3,
                     const DeviceFlags &status, std::array<double,4> *seconds = nullptr);
// the entries of the assembled rows at their offsets (after IrregularCount and the scan)
void IrregularFill(const Geometry &g, const Irregular &t, const Quads &fits, bool forces,
                   int components, const std::array<DeviceIndex,4> &offset,
                   const std::array<DeviceIndex,4> &col,
                   const std::array<Kokkos::View<double*, DevMemSpace>,4> &val,
                   const DeviceFlags &status);

}  // namespace gpu
}  // namespace gravity
#endif  // GRAVITY_GPU_IRREGULAR_HPP_
