// Uniform isothermal gas and a uniform lattice of dust particles.
// Template written by Codex

#include <cstdlib>
#include <iostream>
#include <limits>

#include "athena.hpp"
#include "parameter_input.hpp"
#include "coordinates/cell_locations.hpp"
#include "mesh/mesh.hpp"
#include "hydro/hydro.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "pgen/pgen.hpp"

void ProblemGenerator::UserProblem(ParameterInput *pin, const bool restart) {
  // Register callbacks above this test, if the problem has any. See section 8.
  if (restart) return;

  MeshBlockPack *pmbp = pmy_mesh_->pmb_pack;

  // Fail early when the input file does not construct the modules this pgen needs.
  if (pmbp->phydro == nullptr || pmbp->ppart == nullptr || pmbp->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl
              << "uniform_dust requires <hydro>, <particles>, and <dust> blocks"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (!pin->DoesBlockExist("shearing_box") || !pmy_mesh_->two_d) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
            << std::endl
            << "epicyclic_dust requires a <shearing_box> block and a 2D (r,z) mesh"
            << std::endl;
  std::exit(EXIT_FAILURE);
}

  Real rho0 = pin->GetOrAddReal("problem", "rho0", 1.0);
  Real vx0  = pin->GetOrAddReal("problem", "vx0", 0.0);
  Real vd0  = pin->GetOrAddReal("problem", "vd0", 0.0);

  auto &indcs = pmy_mesh_->mb_indcs;
  int is = indcs.is, ie = indcs.ie;
  int js = indcs.js, je = indcs.je;
  int ks = indcs.ks, ke = indcs.ke;
  int nx1 = indcs.nx1, nx2 = indcs.nx2, nx3 = indcs.nx3;
  int nmb = pmbp->nmb_thispack;

  auto &u0 = pmbp->phydro->u0;
  par_for("uniform_dust_gas", DevExeSpace(), 0, nmb - 1,
          ks, ke, js, je, is, ie,
  KOKKOS_LAMBDA(int m, int k, int j, int i) {
    u0(m, IDN, k, j, i) = rho0;
    u0(m, IM1, k, j, i) = rho0*vx0;
    u0(m, IM2, k, j, i) = 0.0;
    u0(m, IM3, k, j, i) = 0.0;
  });

  particles::Particles *ppar = pmbp->ppart;
  int npart = ppar->nprtcl_thispack;
  int ncells_per_mb = nx1*nx2*nx3;
  if (npart != nmb*ncells_per_mb) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Unexpected particle count for ppc = 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  auto &pr = ppar->prtcl_rdata;
  auto &pi = ppar->prtcl_idata;
  auto &mbsize = pmbp->pmb->mb_size;
  auto gids = pmbp->gids;
  bool three_d = pmy_mesh_->three_d;

  par_for("uniform_dust_particles", DevExeSpace(), 0, npart - 1,
  KOKKOS_LAMBDA(int p) {
    //meshblock id
    int m = p/ncells_per_mb;
    //cell id
    int c = p - m*ncells_per_mb;
    int i = c % nx1;
    int j = (c/nx1) % nx2;
    int k = c/(nx1*nx2);

    pi(PGID, p) = gids + m;
    pr(IPX, p) = CellCenterX(i, nx1, mbsize.d_view(m).x1min,
                            mbsize.d_view(m).x1max);
    pr(IPY, p) = CellCenterX(j, nx2, mbsize.d_view(m).x2min,
                            mbsize.d_view(m).x2max);
    pr(IPZ, p) = three_d
        ? CellCenterX(k, nx3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max)
        : 0.0;

    pr(IPVX, p) = vd0;
    pr(IPVY, p) = 0.0;
    pr(IPVZ, p) = 0.0;
    pr(IPRX, p) = 0.0;
    pr(IPRY, p) = 0.0;
    pr(IPRZ, p) = 0.0;
  });

  // This sets PSP, IPTS, and IPM. It must come after PGID has been initialized because
  // it uses PGID to determine each particle's MeshBlock and cell volume.
  pmbp->pdust->SetDefaultMasses(pin);

  // Supply a finite initial transport timestep. The dust task list recomputes this
  // during evolution.
  ppar->dtnew = std::numeric_limits<float>::max();
}
