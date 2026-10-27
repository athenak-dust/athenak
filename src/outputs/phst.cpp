//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file phst.cpp
//! \brief The dust particle history (file_type = phst), Athena-C style:
//! one text line per output time in <basename>.phst.  16 columns in every run; runs with G/R
//! (refinable clouds with cloud_merge or cloud_resample on a refined mesh) add g_groups and
//! r_removed (18).  The set is fixed at the first start (<outputN>/gr_columns, kept by the
//! restart files), so it never changes within a run.
//!
//!   <output5>
//!   file_type = phst
//!   dt        = 0.1
//!
//! The file is appended to (a restarted run adds a new header block, as for the .hst files).
//! Every header block is a comment block with the definitions, then the two lines of the
//! AthenaK history format ("# Athena++ history data" and "# [1]=time [2]=dt ..."), so that
//! vis/python/athena_read.hst and vis/python/phst.py read the file.  The quantities come from
//! DustGasDrag::ParticleHistory (dust_science.cpp); writing them does not change the run.

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "config.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "dust/dust.hpp"
#include "dust/dust_species.hpp"
#include "outputs.hpp"

namespace {
const char *kLabels[18] = {
  "time", "dt", "n_particles", "mass", "mom1", "mom2", "mom3", "ekin", "z_rms", "rho_max",
  "rho_max_x1", "rho_max_x2", "rho_max_x3", "rho_max_level", "chi_max", "dt_stiff_min",
  "g_groups", "r_removed"};
}  // namespace

//----------------------------------------------------------------------------------------
// ctor: needs the dust module; switches on its interval records

ParticleHistoryOutput::ParticleHistoryOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
    BaseTypeOutput(pin, pm, op) {
  if (pm->pmb_pack->ppart == nullptr || pm->pmb_pack->pdust == nullptr) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Output block '" << op.block_name << "': file_type = phst needs dust "
              << "particles (<particles> and <dust> blocks)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (pm->one_d) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "Output block '" << op.block_name << "': file_type = phst needs a 2-D or "
              << "3-D mesh" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  pm->pmb_pack->pdust->phst_track = true;
  // the column set of the run: from the configuration at the first start; a restart file holds
  // the key, so a restarted run keeps its set whatever its G/R switches
  gr_columns_ = pin->GetOrAddBoolean(op.block_name, "gr_columns",
                                     pm->pmb_pack->pdust->PhstGRColumns());
}

void ParticleHistoryOutput::LoadOutputData(Mesh *pm) {
  // one row per cycle: the final output after an in-loop output of the same cycle (dcycle)
  // would repeat the time with an empty interval
  skip_ = (pm->ncycle == last_cycle_);
  if (skip_) return;
  pm->pmb_pack->pdust->ParticleHistory(row_);
  last_cycle_ = pm->ncycle;
}

//----------------------------------------------------------------------------------------
//! \fn ParticleHistoryOutput::WriteOutputFile
//! \brief Rank 0 appends the row (and, first in this run, the header block).

void ParticleHistoryOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  if (skip_) return;
  if (global_variable::my_rank == 0) {
    const std::string fname = out_params.file_basename + ".phst";
    FILE *fp = std::fopen(fname.c_str(), "a");
    if (fp == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "Output file '" << fname << "' could not be opened" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    if (!header_written_) {
      dust::DustGasDrag *pd = pm->pmb_pack->pdust;
      std::fprintf(fp, "# AthenaK dust particle history (phst) version 1\n");
      std::fprintf(fp, "# source_commit = %s; representation = %s; ndim = %d; nspecies = %d\n",
                   ATHENA_SOURCE_COMMIT, pd->ScienceRepresentationId().c_str(),
                   pm->three_d ? 3 : 2, pd->nspecies);
      {  // the resolved species table
        std::istringstream tl(dust::Species(pin).Describe(""));
        std::string line;
        std::getline(tl, line);   // nspecies is on the line above
        while (std::getline(tl, line)) std::fprintf(fp, "# species: %s\n", line.c_str());
      }
      if (pd->is_shearing_box) {
        std::fprintf(fp, "# velocity_frame = shear_relative: velocities relative to the "
                     "background shear -q Omega x1 (q = %.17g, Omega = %.17g); azimuthal "
                     "component %s, %s_phys = %s - q*Omega*x1\n", pd->qshear, pd->omega0,
                     pm->three_d ? "v2" : "v3", pm->three_d ? "v2" : "v3",
                     pm->three_d ? "v2" : "v3");
      } else {
        std::fprintf(fp, "# velocity_frame = lab\n");
      }
      std::fprintf(fp, "# a row: the state at the end of a cycle, before its mesh check (the "
                   "first row after the problem generator); sums over all ranks, active "
                   "particles only (mass != 0)\n");
      const int ncol = gr_columns_ ? 18 : 16;
      std::fprintf(fp, "# columns: %d (%s)\n", ncol, gr_columns_ ?
                   "the set of runs with G/R: 16 + g_groups, r_removed" :
                   "the set of runs without G/R");
      std::fprintf(fp, "# [1] time, [2] dt (the next step, as in .hst); [3] n_particles; [4] mass, "
                   "[5-7] mom1..3 = sum m v, [8] ekin = sum m |v|^2/2 (stored velocities), [9] "
                   "z_rms = sqrt(sum m z^2/sum m) about 0, z = %s\n",
                   pm->three_d ? "x3" : "x2 (the vertical of the 2-D r-z box)");
      std::fprintf(fp, "# [10] rho_max: the largest dust density (%s) on the active cells, "
                   "deposited at this time with the run's representation (the mesh field "
                   "dust_total_rho); [11-13] the centre of that cell, [14] its level above "
                   "the root (rho_max = 0: centre 0, level -1)\n",
                   pm->three_d ? "mass per volume" : "mass per unit area of the x1-x2 plane");
      std::fprintf(fp, "# [15] chi_max: largest PC2 feedback stiffness chi = dt max(sum w m/(t_s "
                   "V rho_g)) of the cycles since the previous row (dust_stiffness_log.txt); "
                   "[16] dt_stiff_min: smallest stiff_thresh/max_rate of those cycles; both -1 "
                   "when not measured (no back-reaction, no PC2 cycle in the interval, or "
                   "[16] without stiff_thresh)\n");
      if (gr_columns_) {
        std::fprintf(fp, "# [17] g_groups: coarsening events since the previous row (G groups, "
                     "and stage X families where X is on; cloud_merge_log); [18] r_removed: "
                     "particles removed by the resampling since the previous row "
                     "(cloud_resample_log); 0 when off; the interval starts again at a "
                     "restart\n");
      }
      std::fprintf(fp, "# Athena++ history data\n");
      std::fprintf(fp, "#");
      for (int n=0; n<ncol; ++n) std::fprintf(fp, "  [%d]=%s", n + 1, kLabels[n]);
      std::fprintf(fp, "\n");
      header_written_ = true;
    }
    std::fprintf(fp, out_params.data_format.c_str(), static_cast<double>(pm->time));
    std::fprintf(fp, out_params.data_format.c_str(), static_cast<double>(pm->dt));
    for (int n=0; n<(gr_columns_ ? 16 : 14); ++n) {
      std::fprintf(fp, out_params.data_format.c_str(), row_[n]);
    }
    std::fprintf(fp, "\n");
    std::fclose(fp);
  }
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
}
