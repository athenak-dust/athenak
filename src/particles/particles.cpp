//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particles.cpp
//! \brief implementation of Particles class constructor and assorted other functions

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "bvals/bvals.hpp"
#include "particles.hpp"
#include "particle_counts.hpp"
#include "dust/dust_species.hpp"

namespace particles {
//----------------------------------------------------------------------------------------
// constructor, initializes data structures and parameters

Particles::Particles(MeshBlockPack *ppack, ParameterInput *pin) :
    pmy_pack(ppack) {
  // check this is at least a 2D problem
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle module only works in 2D/3D" <<std::endl;
    std::exit(EXIT_FAILURE);
  }

  // read number of particles per cell, and calculate number of particles this pack
  // Dust: with <dust>/ppc_N (or ppc_per_species) the resolved species table
  // (parsed here, before the allocation) gives the total; <particles>/ppc is then not
  // allowed.  The dust use of <particles>/ppc alone is the legacy input (to be
  // deprecated); other particle types retain this generic input.
  Real ppc = static_cast<Real>(dust::RootPpc(pin));

  // compute number of particles as real number, since ppc can be < 1
  auto &indcs = pmy_pack->pmesh->mb_indcs;
  int ncells = indcs.nx1*indcs.nx2*indcs.nx3;
  Real r_npart = ppc*static_cast<Real>((pmy_pack->nmb_thispack)*ncells);
  // then cast to integer
  nprtcl_thispack = static_cast<int>(r_npart);

  // select particle type
  {
    std::string ptype = pin->GetString("particles","particle_type");
    if (ptype.compare("cosmic_ray") == 0) {
      particle_type = ParticleType::cosmic_ray;
    } else if (ptype.compare("dust") == 0) {
      particle_type = ParticleType::dust;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle type = '" << ptype << "' not recognized"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // select pusher algorithm
  {
    std::string ppush = pin->GetString("particles","pusher");
    if (ppush.compare("drift") == 0) {
      pusher = ParticlesPusher::drift;
    } else if (ppush.compare("imex_dust") == 0) {
      // dust particles are pushed by the DustGasDrag module inside the IMEX stages;
      // Particles::Push is never called for this pusher
      pusher = ParticlesPusher::imex_dust;
    } else {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "Particle pusher must be specified in <particles> block"
                <<std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  // set dimensions of particle arrays. Note particles only work in 2D/3D
  if (pmy_pack->pmesh->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particles only work in 2D/3D, but 1D problem initialized" <<std::endl;
    std::exit(EXIT_FAILURE);
  }
  switch (particle_type) {
    case ParticleType::cosmic_ray:
      {
        int ndim=4;
        if (pmy_pack->pmesh->three_d) {ndim+=2;}
        nrdata = ndim;
        nidata = 2;
        break;
      }
    case ParticleType::dust:
      {
        // dust always carries all three position/velocity components (in 2D r-z the
        // azimuthal velocity lives in IPVZ, matching gas IM3), plus the RK registers
        // (x1,v1), the recorded drag rate R_j, stopping time, and mass
        nrdata = 17;
        // PGID, PTAG, PSP (species index), PLEV (cloud level), PTAGH (high word of the
        // 64-bit tag)
        nidata = 5;
        break;
      }
    default:
      break;
  }
  Kokkos::realloc(prtcl_rdata, nrdata, nprtcl_thispack);
  Kokkos::realloc(prtcl_idata, nidata, nprtcl_thispack);

  // allocate boundary object
  pbval_part = new ParticlesBoundaryValues(this, pin);
}

//----------------------------------------------------------------------------------------
// destructor

Particles::~Particles() {
  delete pbval_part;
}

//----------------------------------------------------------------------------------------
// CreateParticleTags()
// Assigns tags to particles (unique integer).  Note that tracked particles are always
// those with tag numbers less than ntrack.

void Particles::CreateParticleTags(ParameterInput *pin) {
  std::string assign = pin->GetOrAddString("particles","assign_tag","index_order");
  // the tags are 64-bit for dust (words PTAG, PTAGH), the int32 PTAG alone for cosmic rays;
  // all ranks together may hold more than 2^31 particles (the counts of all ranks are known)
  const bool hiword = (nidata > PTAGH);
  const int *count = pmy_pack->pmesh->nprtcl_eachrank;
  const int nranks = global_variable::nranks, myrank = global_variable::my_rank;
  auto int32_tags = [&](const std::int64_t tag_max) {   // the same decision on every rank
    if (!hiword && tag_max > std::numeric_limits<std::int32_t>::max()) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Particle tags are int32 for this particle type: the largest tag "
                << tag_max << " does not fit" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  };

  // tags are assigned sequentially within this rank, starting at 0 with rank=0
  if (assign.compare("index_order") == 0) {
    const std::int64_t tagstart = RankOffset(count, myrank);
    int32_tags(TotalCount(count, nranks) - 1);
    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      const std::uint64_t t = static_cast<std::uint64_t>(tagstart + p);
      pi(PTAG,p) = TagLo(t);
      if (hiword) pi(PTAGH,p) = TagHi(t);
    });

  // tags are assigned sequentially across ranks
  } else if (assign.compare("rank_order") == 0) {
    std::int64_t tag_max = -1;   // the last particle of a rank r has the tag r + nranks (n_r - 1)
    for (int r=0; r<nranks; ++r) {
      if (count[r] > 0) {
        tag_max = std::max(tag_max, r + static_cast<std::int64_t>(nranks)*(count[r] - 1));
      }
    }
    int32_tags(tag_max);
    auto &pi = prtcl_idata;
    par_for("ptags",DevExeSpace(),0,(nprtcl_thispack-1),
    KOKKOS_LAMBDA(const int p) {
      const std::uint64_t t = static_cast<std::uint64_t>(myrank) +
                              static_cast<std::uint64_t>(nranks)*static_cast<std::uint64_t>(p);
      pi(PTAG,p) = TagLo(t);
      if (hiword) pi(PTAGH,p) = TagHi(t);
    });

  // tag algorithm not recognized, so quit with error
  } else {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Particle tag assignment type = '" << assign << "' not recognized"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

} // namespace particles
