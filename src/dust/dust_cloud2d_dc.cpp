//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_cloud2d_dc.cpp
//! \brief 2-D refinable clouds with coupling = imex: the 2-D part of the operator check of
//! drag_solver = dc1 / dc2 (test_cloud_dc1_opcheck, CloudDcOpCheck2D).
//!
//! The rest is written once for both dimensions: the operator of the
//! correction, D M G x with G the kick's gather and D the deposit of the kick's impulse, is
//! CloudCoupledOperatorT (dust_imex3d_dc.cpp; for 2-D clouds the deposit takes the kept gather
//! weights where the shapes agree, else the clouds' own weights), and the consistent gas / dust
//! split of the IMEX residual by the overlap weights (imex_consistent_gas)
//! is CloudImplicitConsistencyOverlapT (dust_imex3d.cpp).  Without imex_consistent_gas the gas
//! takes D_c (the mirror), the behavior of the 2-D clouds in earlier versions.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "driver/driver.hpp"
#include "particles/particles.hpp"
#include "hydro/hydro.hpp"
#include "dust.hpp"
#include "dust_cloud_imex_kernels.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust {

// (the operator of the correction: CloudCoupledOperatorT in dust_imex3d_dc.cpp, both
// dimensions)

// (the 2-D overlap split of the IMEX residual: CloudImplicitConsistencyOverlapT in dust_imex3d.cpp)

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::CloudDcOpCheck2D
//! \brief test_cloud_dc1_opcheck, 2-D clouds (after the two paths of CloudDcOpCheckBefore): the
//! operator on u* with the clouds' own deposit weights against path A (the kept gather weights
//! where the shapes agree), the largest difference to dc_check_[7]; and the number of clouds that
//! take the other deposit branches (same shape but more than kCloudWeightCache leaves; a gather
//! shape other than the deposit's: none since the transition mode 1 retired), summed over the ranks.

void DustGasDrag::CloudDcOpCheck2D(Driver *pdrive, int stage) {
  CloudCoupledOperatorT<LeafTraversal>(pdrive, stage, ustar, dc_res_c, nullptr, true);
  const auto &indcs = pmy_pack->pmesh->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2;
  const int ncells = pmy_pack->nmb_thispack*nx2*nx1;
  auto ra = dc_res_a, rc = dc_res_c;
  Real m7 = 0.0;
  Kokkos::parallel_reduce("dust_cloud2d_dc_check_own", Kokkos::RangePolicy<>(DevExeSpace(), 0, ncells),
  KOKKOS_LAMBDA(const int idx, Real &a) {
    const int i = idx % nx1 + is;
    const int j = (idx/nx1) % nx2 + js;
    const int m = idx/(nx1*nx2);
    for (int c=0; c<3; ++c) {
      const Real d = fabs(ra(m,c,ks,j,i) - rc(m,c,ks,j,i));
      a = fmax(a, (d == d) ? d : 1.0e300);
    }
  }, Kokkos::Max<Real>(m7));
  dc_check_[7] = m7;
  auto pr = pmy_pack->ppart->prtcl_rdata;
  auto pi = pmy_pack->ppart->prtcl_idata;
  const int npart = pmy_pack->ppart->nprtcl_thispack;
  const auto lt = leaf_traversal;
  long long cnt[2] = {0, 0};
  if (npart > 0) {
    int nover = 0, nlobe = 0;
    Kokkos::parallel_reduce("dust_cloud2d_dc_check_branches",
                            Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, int &a, int &b) {
      const Real x0 = pr(IPX,p), y0 = pr(IPY,p);
      const int am = lt.Anchor(pi(PGID,p));
      const int lev = pi(PLEV,p);
      const int glev = CloudGeometry::GatherCode(lev);
      const bool same = (glev & ~CloudGeometry::kCloudGatherPoint) == lev;
      int nt = 0;
      lt.ForEachCloudMoment(x0, y0, glev, am,
      [&](const int, const Real, const Real, const Real) {++nt;});
      if (!same) {
        ++b;
      } else if (nt > CloudGeometry::kCloudWeightCache) {
        ++a;
      }
    }, Kokkos::Sum<int>(nover), Kokkos::Sum<int>(nlobe));
    cnt[0] = nover; cnt[1] = nlobe;
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, cnt, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  dc_cnt2d_[0] = cnt[0]; dc_cnt2d_[1] = cnt[1];
}

}  // namespace dust
