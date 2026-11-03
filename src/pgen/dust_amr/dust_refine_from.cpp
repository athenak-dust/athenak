//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_refine_from.cpp
//! \brief Start a uniform run from the restart file of a uniform run with half the resolution
//! (<problem>/refine_from = FILE), e.g. a 2560/H run from the state of a 1280/H run.
//!
//! The coarse file is read by every rank: its input text up to <par_end>, the mesh header, the
//! hydro variables of every MeshBlock and the particle sections (dust_restart.cpp).  It must
//! cover the same x1 and x2 range and the x3 range of this run (equal, or a shorter box: a slab
//! of whole coarse cells, e.g. L_z = 0.2 H cut from 0.4 H), with half the cells per length,
//! all MeshBlocks on the root level, the same hydro variables and passive scalars (the restart
//! writes nhydro + nscalars variables) and the same particle columns; its time must equal
//! <time>/start_time of this run.
//!  - Gas: each coarse cell average is prolongated to its eight fine cells with minmod-limited
//!    slopes, U_f = U_c + (+-dU_x +- dU_y +- dU_z)/4, so the fine cells average to the coarse value
//!    (mass, momentum and the passive scalars rho s conserved exactly).  Slopes are periodic in
//!    x2, zero in the first and last cells along x1 (shear-periodic faces) and x3.
//!  - Particles (<problem>/refine_split = 8, 2 or 1; default 8): with 8, every particle becomes
//!    eight, mass m/8, at the centres of the eight fine cells of a coarse cell centred on it
//!    (offsets +-dx/2 of the fine cell along each direction); with 2, two of them, mass m/2, at
//!    opposite corners (the diagonal chosen by the parent's tag modulo 4); with 1, the particle
//!    as it is.  Particles outside a shorter box are dropped (their count and mass are printed),
//!    velocities and all other columns copied; children beyond an x1 face are wrapped as
//!    bvals_part.cpp wraps a crossing particle (y shifted by the shear offset of the start
//!    time, velocities unchanged, as they are shear-relative); a child beyond an x3 face keeps
//!    the parent's x3.  The RK registers are set as at a fresh start (position and velocity
//!    copies, drag rates zero).  Child tags: refine_split parent_tag + k, checked against the
//!    largest tag the run holds (refine_tags.hpp) before they are formed.
//! What the coarse run kept outside the particles and the hydro variables (dust state records,
//! the li2018 renormalisation totals) is not read: the input carries vbc_mean0 / vbc_dm (for a
//! shorter box dust_si_strat.cpp takes the mean density of the prolongated gas without them).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "dust_setup.hpp"
#include "refine_tags.hpp"

#if MPI_PARALLEL_ENABLED
#include <mpi.h>
#endif

namespace dust_setup {
namespace {
[[noreturn]] void RFail(const std::string &msg) {
  std::cout << "### FATAL ERROR in <problem>/refine_from: " << msg << std::endl;
#if MPI_PARALLEL_ENABLED
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  std::exit(EXIT_FAILURE);
}
void ReadAt(std::ifstream &f, std::uint64_t at, void *p, std::uint64_t n, const char *what) {
  f.seekg(static_cast<std::streamoff>(at));
  f.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
  if (!f) RFail(std::string("the file is truncated (") + what + ")");
}
inline Real MinMod(const Real a, const Real b) {
  if (a*b <= 0.0) return 0.0;
  return (std::fabs(a) < std::fabs(b)) ? a : b;
}
double Sum(double v) {
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
  return v;
}
}  // namespace

void RefineFromRestart(ParameterInput *pin, Mesh *pm, const std::string &path) {
  MeshBlockPack *pmbp = pm->pmb_pack;
  const int me = global_variable::my_rank;
  std::ifstream f(path, std::ios::binary);
  if (!f) RFail("cannot open " + path);
  // ---- the input text and the mesh header
  std::uint64_t at = 0;
  {
    const std::string mark = "<par_end>\n";
    std::string text(std::size_t(1) << 24, '\0');
    f.read(&text[0], static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(f.gcount()));
    f.clear();
    const std::size_t pos = text.find(mark);
    if (pos == std::string::npos) RFail("no <par_end> in the first 16 MiB of " + path);
    at = pos + mark.size();
  }
  int nmb_c = 0, root_c = 0, ncyc_c = 0;
  RegionSize ms_c;
  RegionIndcs mi_c, mb_c;
  Real time_c = 0.0, dt_c = 0.0;
  ReadAt(f, at, &nmb_c, sizeof(int), "header"); at += sizeof(int);
  ReadAt(f, at, &root_c, sizeof(int), "header"); at += sizeof(int);
  ReadAt(f, at, &ms_c, sizeof(RegionSize), "header"); at += sizeof(RegionSize);
  ReadAt(f, at, &mi_c, sizeof(RegionIndcs), "header"); at += sizeof(RegionIndcs);
  ReadAt(f, at, &mb_c, sizeof(RegionIndcs), "header"); at += sizeof(RegionIndcs);
  ReadAt(f, at, &time_c, sizeof(Real), "header"); at += sizeof(Real);
  ReadAt(f, at, &dt_c, sizeof(Real), "header"); at += sizeof(Real);
  ReadAt(f, at, &ncyc_c, sizeof(int), "header"); at += sizeof(int);
  std::vector<LogicalLocation> ll(nmb_c);
  ReadAt(f, at, ll.data(), nmb_c*sizeof(LogicalLocation), "logical locations");
  at += nmb_c*(sizeof(LogicalLocation) + sizeof(float));   // locations and costs
  std::uint64_t data_size = 0;
  ReadAt(f, at, &data_size, sizeof(std::uint64_t), "data size"); at += sizeof(std::uint64_t);
  const std::uint64_t data0 = at;

  // ---- checks: same box, half the cells, all blocks on the root level, same variables
  const RegionSize &ms = pm->mesh_size;
  const RegionIndcs &mi = pm->mesh_indcs;
  const Real tol = 1.0e-12*std::max(1.0, ms_c.x3max - ms_c.x3min);
  if (std::fabs(ms.x1min - ms_c.x1min) > tol || std::fabs(ms.x1max - ms_c.x1max) > tol ||
      std::fabs(ms.x2min - ms_c.x2min) > tol || std::fabs(ms.x2max - ms_c.x2max) > tol ||
      ms.x3min < ms_c.x3min - tol || ms.x3max > ms_c.x3max + tol) {
    RFail("the coarse file does not cover this box (same x1, x2 range; x3 range within)");
  }
  // x3: this box is the slab of coarse cells KC0 .. KC0 + NZS - 1 (all of them unless shorter)
  const Real kc0r = (ms.x3min - ms_c.x3min)/ms_c.dx3, nzsr = (ms.x3max - ms.x3min)/ms_c.dx3;
  const int KC0 = static_cast<int>(std::lround(kc0r)), NZS = static_cast<int>(std::lround(nzsr));
  if (std::fabs(kc0r - KC0) > 1.0e-6 || std::fabs(nzsr - NZS) > 1.0e-6) {
    RFail("the x3 range of this box must be whole coarse cells");
  }
  if (mi.nx1 != 2*mi_c.nx1 || mi.nx2 != 2*mi_c.nx2 || mi.nx3 != 2*NZS) {
    RFail("this mesh must have twice the cells of the coarse one per length along every direction (coarse " +
          std::to_string(mi_c.nx1) + " x " + std::to_string(mi_c.nx2) + " x " + std::to_string(NZS) +
          " in this x3 range)");
  }
  const int nsplit = pin->GetOrAddInteger("problem", "refine_split", 8);
  if (nsplit != 8 && nsplit != 2 && nsplit != 1) RFail("<problem>/refine_split must be 8, 2 or 1");
  for (int g=0; g<nmb_c; ++g) {
    if (ll[g].level != root_c) RFail("the coarse mesh is refined (only uniform coarse runs)");
  }
  if (pm->multilevel) RFail("this run must be uniform (no mesh refinement)");
  if (pmbp->phydro == nullptr) RFail("needs hydro");
  // the hydro variables and the passive scalars (u0 and the restart file: nhydro + nscalars)
  const int nhyd = pmbp->phydro->nhydro, nscal = pmbp->phydro->nscalars;
  const int nvar = nhyd + nscal;
  const int ngc = mb_c.ng;
  const std::uint64_t o1 = mb_c.nx1 + 2*ngc, o2 = mb_c.nx2 + 2*ngc, o3 = mb_c.nx3 + 2*ngc;
  if (data_size != static_cast<std::uint64_t>(nvar)*o1*o2*o3*sizeof(Real)) {
    RFail("the cell data of a coarse MeshBlock is not " + std::to_string(nhyd) +
          " hydro variables and " + std::to_string(nscal) + " passive scalars (data size " +
          std::to_string(data_size) + ", expected " +
          std::to_string(static_cast<std::uint64_t>(nvar)*o1*o2*o3*sizeof(Real)) + ")");
  }
  if (std::fabs(pm->time - time_c) > 1.0e-9*std::max(1.0, std::fabs(time_c))) {
    std::ostringstream s;
    s << std::setprecision(17) << "<time>/start_time must be the time of the coarse file, " << time_c;
    RFail(s.str());
  }

  // ---- gas: the coarse cell averages of the whole mesh, then the fine cells of this rank
  const int NX = mi_c.nx1, NY = mi_c.nx2, NZ = mi_c.nx3;
  const std::size_t ncc = static_cast<std::size_t>(NX)*NY*NZ;
  std::vector<Real> gc(static_cast<std::size_t>(nvar)*ncc);
  double cmass = 0.0, cmom[3] = {0.0, 0.0, 0.0}, cmass_all = 0.0;   // (cmass: the slab of this box)
  std::vector<double> cscal(nscal, 0.0), fscal(nscal, 0.0);         // rho s of each scalar
  {
    std::vector<Real> buf(static_cast<std::size_t>(nvar)*o1*o2*o3);
    for (int g=0; g<nmb_c; ++g) {
      ReadAt(f, data0 + data_size*g, buf.data(), data_size, "hydro data");
      const int i0 = ll[g].lx1*mb_c.nx1, j0 = ll[g].lx2*mb_c.nx2, k0 = ll[g].lx3*mb_c.nx3;
      for (int v=0; v<nvar; ++v) {
        for (int k=0; k<mb_c.nx3; ++k) {
          for (int j=0; j<mb_c.nx2; ++j) {
            for (int i=0; i<mb_c.nx1; ++i) {
              const Real u = buf[((static_cast<std::size_t>(v)*o3 + k + ngc)*o2 + j + ngc)*o1 + i + ngc];
              gc[static_cast<std::size_t>(v)*ncc + (static_cast<std::size_t>(k0 + k)*NY + j0 + j)*NX + i0 + i] = u;
            }
          }
        }
      }
    }
    const double dvc = ms_c.dx1*ms_c.dx2*ms_c.dx3;
    for (std::size_t q=0; q<ncc; ++q) {
      cmass_all += gc[q]*dvc;
      const int kq = static_cast<int>(q/(static_cast<std::size_t>(NX)*NY));
      if (kq < KC0 || kq >= KC0 + NZS) continue;
      cmass += gc[q]*dvc;
      for (int d=0; d<3; ++d) cmom[d] += gc[static_cast<std::size_t>(IM1 + d)*ncc + q]*dvc;
      for (int n=0; n<nscal; ++n) cscal[n] += gc[static_cast<std::size_t>(nhyd + n)*ncc + q]*dvc;
    }
  }
  double fmass = 0.0, fmom[3] = {0.0, 0.0, 0.0};
  {
    auto &u0 = pmbp->phydro->u0;
    auto u0h = Kokkos::create_mirror_view(u0);
    const auto &ind = pm->mb_indcs;
    const int ng = ind.ng;
    const int n1 = ind.nx1 + 2*ng, n2 = ind.nx2 + 2*ng, n3 = ind.nx3 + 2*ng;
    const Real dxf = ms.dx1, dyf = ms.dx2, dzf = ms.dx3;
    auto at_c = [&](int v, int i, int j, int k) -> Real {
      return gc[static_cast<std::size_t>(v)*ncc + (static_cast<std::size_t>(k)*NY + j)*NX + i];
    };
    for (int m=0; m<pmbp->nmb_thispack; ++m) {
      const RegionSize &bs = pmbp->pmb->mb_size.h_view(m);
      const long I0 = std::lround((bs.x1min - ms.x1min)/dxf), J0 = std::lround((bs.x2min - ms.x2min)/dyf);
      const long K0 = std::lround((bs.x3min - ms.x3min)/dzf);
      for (int k=0; k<n3; ++k) {
        for (int j=0; j<n2; ++j) {
          for (int i=0; i<n1; ++i) {
            // global fine index (ghost cells clamped to the mesh: the boundary exchange fills them)
            const long I = std::clamp(I0 + i - ng, 0L, 2L*NX - 1), J = std::clamp(J0 + j - ng, 0L, 2L*NY - 1);
            const long K = std::clamp(K0 + k - ng, 0L, 2L*NZS - 1);
            const int ic = static_cast<int>(I >> 1), jc = static_cast<int>(J >> 1);
            const int kc = KC0 + static_cast<int>(K >> 1);
            const Real sx = (I & 1) ? 0.25 : -0.25, sy = (J & 1) ? 0.25 : -0.25, sz = (K & 1) ? 0.25 : -0.25;
            const bool active = (i >= ng && i < n1 - ng && j >= ng && j < n2 - ng && k >= ng && k < n3 - ng);
            for (int v=0; v<nvar; ++v) {
              const Real uc = at_c(v, ic, jc, kc);
              const Real dux = (ic > 0 && ic < NX - 1) ? MinMod(at_c(v, ic + 1, jc, kc) - uc, uc - at_c(v, ic - 1, jc, kc)) : 0.0;
              const Real duy = MinMod(at_c(v, ic, (jc + 1) % NY, kc) - uc, uc - at_c(v, ic, (jc + NY - 1) % NY, kc));
              const Real duz = (kc > 0 && kc < NZ - 1) ? MinMod(at_c(v, ic, jc, kc + 1) - uc, uc - at_c(v, ic, jc, kc - 1)) : 0.0;
              const Real u = uc + sx*dux + sy*duy + sz*duz;
              u0h(m,v,k,j,i) = u;
              if (active) {
                const double dvf = dxf*dyf*dzf;
                if (v == IDN) fmass += u*dvf;
                if (v >= IM1 && v <= IM3) fmom[v - IM1] += u*dvf;
                if (v >= nhyd) fscal[v - nhyd] += u*dvf;
              }
            }
          }
        }
      }
    }
    Kokkos::deep_copy(u0, u0h);
  }
  fmass = Sum(fmass);
  for (int d=0; d<3; ++d) fmom[d] = Sum(fmom[d]);
  for (int n=0; n<nscal; ++n) fscal[n] = Sum(fscal[n]);

  // ---- particles: every section of the coarse file, eight children each, those of this rank
  particles::Particles *ppar = pmbp->ppart;
  if (ppar == nullptr || pmbp->pdust == nullptr) RFail("needs dust particles");
  const int nrdata = ppar->nrdata, nidata = ppar->nidata;
  // the largest child tag this run can hold (64 bits with the PTAGH word)
  const std::uint64_t tag_max = RefineTagMax(nidata > PTAGH);
  std::uint64_t tag_last_max = 0;
  std::uint64_t secpos = data0 + data_size*static_cast<std::uint64_t>(nmb_c);
  std::vector<std::vector<Real>> rc(nrdata);
  std::vector<std::vector<int>> ic(nidata);
  const Real lx = ms.x1max - ms.x1min, ly = ms.x2max - ms.x2min;
  const Real qshear = pin->GetReal("shearing_box", "qshear"), omega0 = pin->GetReal("shearing_box", "omega0");
  const Real ysh = std::fmod(qshear*omega0*lx*time_c, ly);
  const Real hx = 0.5*ms.dx1, hy = 0.5*ms.dx2, hz = 0.5*ms.dx3;   // half a fine cell
  const int nmb = pmbp->nmb_thispack;
  std::vector<RegionSize> mine(nmb);
  for (int m=0; m<nmb; ++m) mine[m] = pmbp->pmb->mb_size.h_view(m);
  double pmass = 0.0, pmom[3] = {0.0, 0.0, 0.0}, pmz = 0.0;     // parents in this box (every rank: all)
  double dmass = 0.0;                                           // parents outside it (dropped)
  std::int64_t ndrop = 0;
  const Real fsplit = 1.0/static_cast<Real>(nsplit);
  double kmass = 0.0, kmom[3] = {0.0, 0.0, 0.0}, kmz = 0.0;     // children of this rank
  std::int64_t nparent = 0;
  int nsec = -1;
  for (int sct=0; ; ++sct) {
    char head[40];
    ReadAt(f, secpos, head, sizeof(head), "particle section head");
    if (std::memcmp(head, "AKPRST01", 8) != 0) RFail("no particle section where one was expected");
    std::int32_t h[6];
    std::memcpy(h, head + 8, sizeof(h));
    std::uint64_t total = 0;
    std::memcpy(&total, head + 32, sizeof(total));
    if (nsec < 0) nsec = h[3];
    if ((h[0] != 1 && h[0] != 2) || h[1] != static_cast<std::int32_t>(sizeof(Real)) ||
        h[2] != static_cast<std::int32_t>(sizeof(int)) || h[4] != nrdata || (h[5] != nidata && h[5] != nidata - 1)) {
      RFail("the particle columns of the coarse file do not match this run");
    }
    const int nif = h[5];
    std::int64_t np = 0;
    std::int32_t nstate = 0;
    ReadAt(f, secpos + 40, &np, sizeof(np), "particle count");
    ReadAt(f, secpos + 40 + sizeof(np) + sizeof(Real), &nstate, sizeof(nstate), "state size");
    const std::uint64_t rows = secpos + 40 + sizeof(np) + sizeof(Real) + sizeof(nstate) + nstate;
    std::vector<Real> r(static_cast<std::size_t>(np)*nrdata);
    std::vector<int> iv(static_cast<std::size_t>(np)*nif);
    ReadAt(f, rows, r.data(), r.size()*sizeof(Real), "particle rows");
    ReadAt(f, rows + r.size()*sizeof(Real), iv.data(), iv.size()*sizeof(int), "particle integer rows");
    if (sct == 0) {   // about nsplit np nsec / nranks children per rank
      const std::size_t est = static_cast<std::size_t>(1.1*nsplit*np*nsec/global_variable::nranks) + 1024;
      for (auto &c : rc) c.reserve(est);
      for (auto &c : ic) c.reserve(est);
    }
    for (std::int64_t p=0; p<np; ++p) {
      auto R = [&](int c) {return r[static_cast<std::size_t>(c)*np + p];};
      const Real mp = R(IPM);
      if (!(mp > 0.0)) continue;   // tombstones
      if (R(IPZ) < ms.x3min || R(IPZ) >= ms.x3max) {   // outside a shorter box
        ++ndrop; dmass += mp;
        continue;
      }
      ++nparent;
      pmass += mp; pmz += mp*R(IPZ);
      pmom[0] += mp*R(IPVX); pmom[1] += mp*R(IPVY); pmom[2] += mp*R(IPVZ);
      const std::uint64_t tag = TagFromWords(iv[static_cast<std::size_t>(PTAG)*np + p],
                                             (nif > PTAGH) ? iv[static_cast<std::size_t>(PTAGH)*np + p] : 0);
      // the child tags nsplit*tag + k must not wrap around (two parents would share them)
      std::uint64_t tag_last = 0;
      if (!RefineChildTagsFit(tag, nsplit, tag_max, &tag_last)) {
        std::ostringstream msg;
        msg << "the child tags " << nsplit << "*tag + k (k < " << nsplit << ") of the particle "
            << "with tag " << tag << " exceed the largest tag this run can hold, " << tag_max
            << " (refine_split " << nsplit << " needs tags up to " << (tag_max - (nsplit - 1))/nsplit
            << ")";
        RFail(msg.str());
      }
      tag_last_max = std::max(tag_last_max, tag_last);
      // the corners of the children (bit 0: +x, 1: +y, 2: +z); two: one of the four diagonals
      static const int kDiag[4] = {7, 3, 5, 6};
      for (int kid=0; kid<nsplit; ++kid) {
        const int bits = (nsplit == 8) ? kid : (kid == 0 ? kDiag[tag & 3] : 7 ^ kDiag[tag & 3]);
        Real x = R(IPX), y = R(IPY), z = R(IPZ);
        if (nsplit > 1) {
          x += (bits & 1) ? hx : -hx;
          y += (bits & 2) ? hy : -hy;
          z += (bits & 4) ? hz : -hz;
        }
        if (x >= ms.x1max) {x -= lx; y += ysh;}
        else if (x < ms.x1min) {x += lx; y -= ysh;}
        while (y >= ms.x2max) y -= ly;
        while (y < ms.x2min) y += ly;
        if (z < ms.x3min || z >= ms.x3max) z = R(IPZ);
        int owner = -1;
        for (int m=0; m<nmb; ++m) {
          const RegionSize &b = mine[m];
          if (x >= b.x1min && x < b.x1max && y >= b.x2min && y < b.x2max && z >= b.x3min && z < b.x3max) {
            owner = m; break;
          }
        }
        if (owner < 0) continue;
        for (int c=0; c<nrdata; ++c) rc[c].push_back(R(c));
        const std::size_t q = rc[0].size() - 1;
        rc[IPX][q] = x; rc[IPY][q] = y; rc[IPZ][q] = z;
        rc[IPX1][q] = x; rc[IPY1][q] = y; rc[IPZ1][q] = z;
        rc[IPVX1][q] = R(IPVX); rc[IPVY1][q] = R(IPVY); rc[IPVZ1][q] = R(IPVZ);
        rc[IPRX][q] = 0.0; rc[IPRY][q] = 0.0; rc[IPRZ][q] = 0.0;
        rc[IPM][q] = fsplit*mp;
        const std::uint64_t ct = static_cast<std::uint64_t>(nsplit)*tag + static_cast<std::uint64_t>(kid);   // (checked)
        for (int c=0; c<nidata; ++c) ic[c].push_back(0);
        ic[PGID][q] = pmbp->gids + owner;
        ic[PTAG][q] = TagLo(ct);
        ic[PTAGH][q] = TagHi(ct);
        ic[PSP][q] = iv[static_cast<std::size_t>(PSP)*np + p];
        ic[PLEV][q] = iv[static_cast<std::size_t>(PLEV)*np + p];
        kmass += fsplit*mp; kmz += fsplit*mp*z;
        kmom[0] += fsplit*mp*R(IPVX); kmom[1] += fsplit*mp*R(IPVY); kmom[2] += fsplit*mp*R(IPVZ);
      }
    }
    secpos += total;
    if (sct + 1 >= nsec) break;
  }
  const std::int64_t n = rc[0].size();
  if (n > std::numeric_limits<int>::max()) RFail("too many particles on one rank");
  // capacity with a quarter of headroom, as bvals_part.cpp grows the arrays: the first particle
  // exchanges do not resize them (a resize holds the old and the new arrays at once, which set
  // the device memory peak of a large run on the rank with the densest filament)
  const std::int64_t cap = std::min<std::int64_t>(n + n/4, std::numeric_limits<int>::max());
  Kokkos::realloc(ppar->prtcl_rdata, nrdata, std::max<std::int64_t>(cap, 1));
  Kokkos::realloc(ppar->prtcl_idata, nidata, std::max<std::int64_t>(cap, 1));
  {
    auto prh = Kokkos::create_mirror_view(ppar->prtcl_rdata);
    auto pih = Kokkos::create_mirror_view(ppar->prtcl_idata);
    for (int c=0; c<nrdata; ++c) for (std::int64_t q=0; q<n; ++q) prh(c,q) = rc[c][q];
    for (int c=0; c<nidata; ++c) for (std::int64_t q=0; q<n; ++q) pih(c,q) = ic[c][q];
    Kokkos::deep_copy(ppar->prtcl_rdata, prh);
    Kokkos::deep_copy(ppar->prtcl_idata, pih);
  }
  Real vmax = 1.0e-30;
  for (std::int64_t q=0; q<n; ++q) {
    vmax = std::max(vmax, std::max(std::fabs(rc[IPVX][q]), std::max(std::fabs(rc[IPVY][q]), std::fabs(rc[IPVZ][q]))));
  }
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE, &vmax, 1, MPI_ATHENA_REAL, MPI_MAX, MPI_COMM_WORLD);
#endif
  rc.clear(); ic.clear();
  ppar->nprtcl_thispack = static_cast<int>(n);
  pm->nprtcl_thisrank = static_cast<int>(n);
  for (int r=0; r<global_variable::nranks; ++r) pm->nprtcl_eachrank[r] = 0;
  pm->nprtcl_eachrank[me] = static_cast<int>(n);
  long long ntot = n;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, pm->nprtcl_eachrank, 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ntot, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
  pm->nprtcl_total = ntot;
  pmbp->pdust->EnsureNextTag();
  SetParticleDt(pm, {}, {}, vmax);
  kmass = Sum(kmass); kmz = Sum(kmz);
  for (int d=0; d<3; ++d) kmom[d] = Sum(kmom[d]);
  if (me == 0) {
    std::cout << std::setprecision(10)
              << "# refine_from " << path << ": t = " << time_c << " (cycle " << ncyc_c << "), coarse "
              << mi_c.nx1 << " x " << mi_c.nx2 << " x " << mi_c.nx3 << " in " << nmb_c << " MeshBlocks, "
              << nparent << " particles in " << nsec << " section(s) -> " << ntot << " particles (split "
              << nsplit << ")\n";
    if (NZS != mi_c.nx3) {
      std::cout << "#   shorter box: coarse x3 cells " << KC0 << " .. " << KC0 + NZS - 1 << " of " << mi_c.nx3
                << "; gas mass " << cmass << " of " << cmass_all << " (" << cmass/cmass_all << "); "
                << ndrop << " particles of mass " << dmass << " (" << dmass/(pmass + dmass)
                << " of the dust) outside, dropped\n";
    }
    std::cout << "#   gas mass " << cmass << " -> " << fmass << " (rel " << (fmass - cmass)/cmass << "), momentum "
              << cmom[0] << " " << cmom[1] << " " << cmom[2] << " -> " << fmom[0] << " " << fmom[1] << " " << fmom[2] << "\n"
              << "#   dust mass " << pmass << " -> " << kmass << " (rel " << (kmass - pmass)/pmass << "), momentum "
              << pmom[0] << " " << pmom[1] << " " << pmom[2] << " -> " << kmom[0] << " " << kmom[1] << " " << kmom[2]
              << ", <z> " << pmz/pmass << " -> " << kmz/kmass << ", largest child tag " << tag_last_max << "\n";
    const auto prec = std::cout.precision();
    for (int n=0; n<nscal; ++n) {
      std::cout << std::setprecision(17) << "#   passive scalar " << n << " (rho s) " << cscal[n] << " -> "
                << fscal[n] << " (rel " << ((cscal[n] != 0.0) ? (fscal[n] - cscal[n])/cscal[n] : 0.0) << ")\n";
    }
    std::cout << std::setprecision(prec) << std::flush;
  }
}

}  // namespace dust_setup
