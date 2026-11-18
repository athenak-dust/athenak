//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_dtdiag.cpp
//! \brief Diagnostic (hidden test key <dust>/test_dt_diag = threshold, default 0 = off): in a cycle
//! whose new global dt falls below the threshold, every rank whose own limit is within 25% of the
//! global dt prints one line: its hydro CFL limit with the limiting cell (position, gas state, the
//! dust share of the last IMEX solve there), its particle limit with the limiting cloud, and the
//! gravity limit.  Read-only: it changes no state.

#include <cstdio>
#include <limits>
#include <cmath>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "eos/eos.hpp"
#include "particles/particles.hpp"
#include "dust.hpp"

namespace dust {

void DustGasDrag::DtDiag(const Real dt, const Real dt_hydro, const Real dt_grav,
                         const Real dt_part) {
  Mesh *pm = pmy_pack->pmesh;
  const Real lmin = std::fmin(dt_hydro, std::fmin(dt_grav, dt_part));
  if (!(lmin <= 1.25*dt)) return;
  auto &indcs = pm->mb_indcs;
  const int is = indcs.is, js = indcs.js, ks = indcs.ks;
  const int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  const int nmb = pmy_pack->nmb_thispack;
  const int nkji = nx3*nx2*nx1, nji = nx2*nx1, nmkji = nmb*nkji;
  auto &mbsize = pmy_pack->pmb->mb_size;
  using ML = Kokkos::MinLoc<Real, int>;
  // the hydro limit: the cell with the smallest dx/(|v| + cs) (isothermal)
  typename ML::value_type hres;
  hres.val = std::numeric_limits<Real>::max(); hres.loc = -1;
  const Real cs = pmy_pack->phydro->peos->eos_data.iso_cs;
  auto w0 = pmy_pack->phydro->w0;
  Kokkos::parallel_reduce("dust_dtdiag_hydro", Kokkos::RangePolicy<>(DevExeSpace(), 0, nmkji),
  KOKKOS_LAMBDA(const int idx, typename ML::value_type &v) {
    const int m = idx/nkji, r = idx - m*nkji, k = r/nji + ks, j = (r % nji)/nx1 + js, i = r % nx1 + is;
    Real d = mbsize.d_view(m).dx1/(fabs(w0(m,IVX,k,j,i)) + cs);
    d = fmin(d, mbsize.d_view(m).dx2/(fabs(w0(m,IVY,k,j,i)) + cs));
    d = fmin(d, mbsize.d_view(m).dx3/(fabs(w0(m,IVZ,k,j,i)) + cs));
    if (d < v.val) {v.val = d; v.loc = idx;}
  }, ML(hres));
  // the particle limit: the cloud with the smallest dx/|v_transport|
  particles::Particles *ppar = pmy_pack->ppart;
  const int npart = ppar->nprtcl_thispack;
  auto pr = ppar->prtcl_rdata;
  auto pi = ppar->prtcl_idata;
  const int gids = pmy_pack->gids;
  const bool sh3 = is_shearing_box && pm->three_d;
  const Real qo = qshear*omega0;
  typename ML::value_type pres;
  pres.val = std::numeric_limits<Real>::max(); pres.loc = -1;
  if (npart > 0) {
    Kokkos::parallel_reduce("dust_dtdiag_part", Kokkos::RangePolicy<>(DevExeSpace(), 0, npart),
    KOKKOS_LAMBDA(const int p, typename ML::value_type &v) {
      const int m = pi(PGID,p) - gids;
      Real vy = pr(IPVY,p); if (sh3) vy -= qo*pr(IPX,p);
      Real d = mbsize.d_view(m).dx1/fmax(fabs(pr(IPVX,p)), 1.0e-30);
      d = fmin(d, mbsize.d_view(m).dx2/fmax(fabs(vy), 1.0e-30));
      d = fmin(d, mbsize.d_view(m).dx3/fmax(fabs(pr(IPVZ,p)), 1.0e-30));
      if (d < v.val) {v.val = d; v.loc = p;}
    }, ML(pres));
  }
  // gather the two records into one small device array
  Kokkos::View<Real*, DevMemSpace> rec("dust_dtdiag_rec", 32);
  auto u0 = pmy_pack->phydro->u0;
  auto qp = qdep;
  auto qc = imex_qcen;
  const bool hq = (qp.extent_int(0) >= nmb), hc = (qc.extent_int(0) >= nmb);
  const int hl = hres.loc, pl = pres.loc;
  const auto lt = leaf_traversal3d;
  const bool c3 = clouds3d;
  Kokkos::parallel_for("dust_dtdiag_rec", Kokkos::RangePolicy<>(DevExeSpace(), 0, 1),
  KOKKOS_LAMBDA(const int) {
    for (int q=0; q<32; ++q) rec(q) = 0.0;
    if (hl >= 0) {
      const int m = hl/nkji, r = hl - m*nkji, k = r/nji + ks, j = (r % nji)/nx1 + js, i = r % nx1 + is;
      rec(0) = mbsize.d_view(m).x1min + (i - is + 0.5)*mbsize.d_view(m).dx1;
      rec(1) = mbsize.d_view(m).x2min + (j - js + 0.5)*mbsize.d_view(m).dx2;
      rec(2) = mbsize.d_view(m).x3min + (k - ks + 0.5)*mbsize.d_view(m).dx3;
      rec(3) = mbsize.d_view(m).dx1;
      rec(4) = w0(m,IDN,k,j,i); rec(5) = w0(m,IVX,k,j,i); rec(6) = w0(m,IVY,k,j,i); rec(7) = w0(m,IVZ,k,j,i);
      if (hq) {
        const Real q = qp(m,0,k,j,i);
        rec(8) = q; rec(9) = q/(u0(m,IDN,k,j,i) + q);
        rec(10) = (q > 0.0) ? qp(m,1,k,j,i)/q : 0.0;
        rec(11) = (q > 0.0) ? qp(m,2,k,j,i)/q : 0.0;
        rec(12) = (q > 0.0) ? qp(m,3,k,j,i)/q : 0.0;
      }
      if (hc) rec(13) = qc(m,0,k,j,i);
    }
    if (pl >= 0) {
      const int p = pl, m = pi(PGID,p) - gids;
      rec(16) = pr(IPX,p); rec(17) = pr(IPY,p); rec(18) = pr(IPZ,p);
      rec(19) = pr(IPVX,p); rec(20) = pr(IPVY,p) - (sh3 ? qo*pr(IPX,p) : 0.0); rec(21) = pr(IPVZ,p);
      rec(22) = pr(IPM,p); rec(23) = static_cast<Real>(pi(PLEV,p)); rec(24) = mbsize.d_view(m).dx1;
      if (c3 && hq) {
        const int leaf = lt.LeafOwner(pr(IPX,p), pr(IPY,p), pr(IPZ,p), lt.Anchor(pi(PGID,p)));
        if (leaf >= 0) {
          int b, k, j, i;
          lt.Decode(leaf, b, k, j, i);
          if (b < nmb) {
            const Real q = qp(b,0,k,j,i);
            rec(25) = q; rec(26) = q/(u0(b,IDN,k,j,i) + q); rec(27) = u0(b,IDN,k,j,i);
            rec(28) = (q > 0.0) ? qp(b,2,k,j,i)/q : 0.0;
            if (hc) rec(29) = qc(b,0,k,j,i);
          }
        }
      }
    }
  });
  auto h = Kokkos::create_mirror_view_and_copy(HostMemSpace(), rec);
  std::printf("# DT_DIAG cycle=%d time=%.8e dt=%.4e rank=%d hydro=%.4e grav=%.4e part=%.4e |"
              " hydro_cell x=(%.6f,%.6f,%.6f) dx=%.3e rho=%.4e v=(%.4f,%.4f,%.4f) Q=%.4e f=%.4f"
              " udust=(%.4f,%.4f,%.4f) qcen=%.4e |"
              " part x=(%.6f,%.6f,%.6f) v=(%.4f,%.4f,%.4f) m=%.4e code=%d dx=%.3e cellQ=%.4e cellf=%.4f"
              " cellrho=%.4e cell_udust_y=%.4f cell_qcen=%.4e\n",
              pm->ncycle, pm->time, dt, global_variable::my_rank, dt_hydro, dt_grav, dt_part,
              h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8), h(9), h(10), h(11), h(12), h(13),
              h(16), h(17), h(18), h(19), h(20), h(21), h(22), static_cast<int>(h(23)), h(24), h(25),
              h(26), h(27), h(28), h(29));
  std::fflush(stdout);
}

}  // namespace dust
