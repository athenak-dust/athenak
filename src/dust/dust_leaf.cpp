//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_leaf.cpp
//! \brief The actual-leaf traversal: the block geometry of
//! LeafTraversal (dust.hpp), rebuilt with the leaf maps after every regrid, and the
//! diagnostic comparison with the virtual-finest-grid traversal (<dust>/cloud_leaf_check).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "mesh/meshblock_pack.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildLeafTraversal
//! \brief Box, cell size and the distinct adjacent blocks (with the periodic shift that
//! places each next to the block) of every MeshBlock of the pack, from mb_size and the
//! AthenaK neighbour table.  One rank: a neighbour must be a block of this pack.

void DustGasDrag::BuildLeafTraversal() {
  Mesh *pm = pmy_pack->pmesh;
  auto &indcs = pm->mb_indcs;
  const int nmb = pmy_pack->nmb_thispack;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &nghbr = pmy_pack->pmb->nghbr;
  const int nslot = pmy_pack->pmb->nnghbr;
  auto &lt = leaf_traversal;
  lt.nmb = nmb;
  lt.gids = pmy_pack->gids;
  lt.nx1 = indcs.nx1; lt.nx2 = indcs.nx2; lt.is = indcs.is; lt.js = indcs.js;
  lt.ncells1 = indcs.nx1 + 2*indcs.ng; lt.ncells2 = indcs.nx2 + 2*indcs.ng;
  lt.x1min = pm->mesh_size.x1min; lt.x1max = pm->mesh_size.x1max;
  lt.x2min = pm->mesh_size.x2min; lt.x2max = pm->mesh_size.x2max;
  lt.lx = lt.x1max - lt.x1min; lt.ly = lt.x2max - lt.x2min;
  lt.root_dx1 = smr_transfer.root_dx1; lt.root_dx2 = smr_transfer.root_dx2;
  lt.lobe3_integral = smr_transfer.lobe3_integral;
  lt.box = DvceArray2D<Real>("dust_leaf_box", nmb, 4);
  lt.nnb = DvceArray1D<int>("dust_leaf_nnb", nmb);
  lt.nb = DvceArray2D<int>("dust_leaf_nb", nmb, LeafTraversal::kMaxNb);
  lt.nbs = DvceArray3D<int>("dust_leaf_nbs", nmb, LeafTraversal::kMaxNb, 2);
  auto box = Kokkos::create_mirror_view(lt.box);
  auto nnb = Kokkos::create_mirror_view(lt.nnb);
  auto nb = Kokkos::create_mirror_view(lt.nb);
  auto nbs = Kokkos::create_mirror_view(lt.nbs);
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    box(m, 0) = sz.x1min; box(m, 1) = sz.x2min; box(m, 2) = sz.dx1; box(m, 3) = sz.dx2;
  }
  for (int m=0; m<nmb; ++m) {
    const auto &sz = mbsize.h_view(m);
    const Real cx = 0.5*(sz.x1min + sz.x1max), cy = 0.5*(sz.x2min + sz.x2max);
    int c = 0;
    for (int n=0; n<nslot; ++n) {
      const int gid = nghbr.h_view(m, n).gid;
      if (gid < 0) continue;
      const int b = gid - lt.gids;
      if (b < 0 || b >= nmb) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "leaf traversal: neighbour " << gid << " of block " << m
                  << " is not in this pack (one MPI rank only)" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      const auto &nz = mbsize.h_view(b);
      const int sx = static_cast<int>(std::lround((0.5*(nz.x1min + nz.x1max) - cx)/lt.lx));
      const int sy = static_cast<int>(std::lround((0.5*(nz.x2min + nz.x2max) - cy)/lt.ly));
      bool seen = (b == m && sx == 0 && sy == 0);
      for (int k=0; k<c && !seen; ++k) seen = (nb(m, k) == b && nbs(m, k, 0) == sx && nbs(m, k, 1) == sy);
      if (seen) continue;
      if (c >= LeafTraversal::kMaxNb) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                  << "leaf traversal: more than " << LeafTraversal::kMaxNb
                  << " distinct neighbours of block " << m << std::endl;
        std::exit(EXIT_FAILURE);
      }
      nb(m, c) = b; nbs(m, c, 0) = sx; nbs(m, c, 1) = sy;
      ++c;
    }
    nnb(m) = c;
  }
  Kokkos::deep_copy(lt.box, box);
  Kokkos::deep_copy(lt.nnb, nnb);
  Kokkos::deep_copy(lt.nb, nb);
  Kokkos::deep_copy(lt.nbs, nbs);
}

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CheckLeafTraversal
//! \brief Diagnostic (<dust>/cloud_leaf_check = N: every N cycles): for every particle, the
//! deposit weights (cloud code) and the gather weights and first moments (gather code) of
//! the actual-leaf traversal against the virtual-finest-grid traversal.  As the native
//! check (CheckNativeTransfer) it compares two pseudo-random linear functionals of the
//! per-leaf weights (equal sums mean equal per-leaf totals, with overwhelming probability),
//! plus the weight sum (1 unless the support leaves the neighbours) and the owning leaf.
//! One line per check in cloud_leaf_check.txt.

void DustGasDrag::CheckLeafTraversal() {
  particles::Particles *ppar = pmy_pack->ppart;
  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  const int npart = ppar->nprtcl_thispack;
  const SMRCompositeTransfer tab = smr_transfer;
  const LeafTraversal lt = leaf_traversal;
  const int tmode = cloud_transition;
  const int gids = pmy_pack->gids;
  const Real inv_dx = 1.0/smr_transfer.root_dx1;
  constexpr Real kTol = 1.0e-12;
  Real ew = 0.0, em = 0.0, es = 0.0;
  int nbadw = 0, nbadm = 0, nbads = 0, nbado = 0;
  Kokkos::parallel_reduce("dust_leaf_check", Kokkos::RangePolicy<DevExeSpace>(0, npart),
  KOKKOS_LAMBDA(const int p, Real &dw, Real &dm, Real &ds, int &bw, int &bm, int &bs, int &bo) {
    auto coef = [](const int target, const Real s) {
      Real v = sin(static_cast<Real>(target)*12.9898 + s)*43758.5453;
      return v - floor(v);
    };
    const Real x1 = pr(IPX,p), x2 = pr(IPY,p);
    const int code = pi(PLEV,p), m = pi(PGID,p) - gids;
    const int gcode = SMRCompositeTransfer::GatherCode(code, tmode);
    // table
    Real tw[2] = {0.0, 0.0}, tm[2] = {0.0, 0.0};
    tab.ForEachCloudWeight(x1, x2, code, [&](const int target, const Real w) {
      tw[0] += w*coef(target, 0.3); tw[1] += w*coef(target, 4.1);
    });
    tab.ForEachCloudMoment(x1, x2, gcode,
    [&](const int target, const Real w, const Real m1x, const Real m1y) {
      tm[0] += w*coef(target, 0.3) + inv_dx*(m1x*coef(target, 1.7) + m1y*coef(target, 2.9));
      tm[1] += w*coef(target, 5.3) + inv_dx*(m1x*coef(target, 6.7) + m1y*coef(target, 7.9));
    });
    // actual leaves
    Real lw[2] = {0.0, 0.0}, lmo[2] = {0.0, 0.0};
    const Real sum = lt.ForEach<false>(x1, x2, code, m, [&](const int target, const Real w) {
      lw[0] += w*coef(target, 0.3); lw[1] += w*coef(target, 4.1);
    });
    lt.ForEach<true>(x1, x2, gcode, m,
    [&](const int target, const Real w, const Real m1x, const Real m1y) {
      lmo[0] += w*coef(target, 0.3) + inv_dx*(m1x*coef(target, 1.7) + m1y*coef(target, 2.9));
      lmo[1] += w*coef(target, 5.3) + inv_dx*(m1x*coef(target, 6.7) + m1y*coef(target, 7.9));
    });
    const Real d1 = fmax(fabs(tw[0] - lw[0]), fabs(tw[1] - lw[1]));
    const Real d2 = fmax(fabs(tm[0] - lmo[0]), fabs(tm[1] - lmo[1]));
    const Real d3 = fabs(sum - 1.0);
    dw = fmax(dw, d1); dm = fmax(dm, d2); ds = fmax(ds, d3);
    if (d1 > kTol) ++bw;
    if (d2 > kTol) ++bm;
    if (d3 > kTol) ++bs;
    if (lt.Owner(x1, x2, m) != tab.LeafOwner(x1, x2)) ++bo;
  }, Kokkos::Max<Real>(ew), Kokkos::Max<Real>(em), Kokkos::Max<Real>(es), Kokkos::Sum<int>(nbadw),
     Kokkos::Sum<int>(nbadm), Kokkos::Sum<int>(nbads), Kokkos::Sum<int>(nbado));
  if (global_variable::my_rank == 0) {
    std::FILE *fp = std::fopen("cloud_leaf_check.txt", "a");
    if (fp != nullptr) {
      std::fseek(fp, 0, SEEK_END);
      if (std::ftell(fp) == 0) {
        std::fprintf(fp, "# ncycle time particles max_dweight max_dmoment max_dsum bad_weight "
                     "bad_moment bad_sum bad_owner (bad: > 1e-12)\n");
      }
      std::fprintf(fp, "%d %.17g %d %.3e %.3e %.3e %d %d %d %d\n", pmy_pack->pmesh->ncycle,
                   pmy_pack->pmesh->time, npart, ew, em, es, nbadw, nbadm, nbads, nbado);
      std::fclose(fp);
    }
    std::cout << "# leaf traversal check cycle " << pmy_pack->pmesh->ncycle << ": "
              << npart << " particles, max |d weights| " << ew << ", max |d moments| " << em
              << ", max |weight sum - 1| " << es << "; over 1e-12: " << nbadw << " / " << nbadm
              << " / " << nbads << ", owner differs " << nbado << std::endl;
  }
}

}  // namespace dust
