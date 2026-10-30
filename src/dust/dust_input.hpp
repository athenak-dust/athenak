#ifndef DUST_DUST_INPUT_HPP_
#define DUST_DUST_INPUT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_input.hpp
//! \brief Input keys of the dust module that the mesh and boundary classes read before the
//! dust module exists: <dust>/amr_pic (how the dust particles live on a refined mesh) and the
//! notes for the keys that are no longer read.

#include <cstdlib>
#include <iostream>
#include <string>

#include "globals.hpp"
#include "parameter_input.hpp"

namespace dust {

// <dust>/amr_pic: refinable_cloud (default on a refined mesh) or instant_switch (point particles,
// TSC on the block that owns them, ghost deposits remapped to the leaves).  On a uniform mesh the
// point particles deposit by TSC (none); amr_pic = refinable_cloud selects the cloud path there
// too (exact TSC clouds, a reference with the coupling of the mixed-resolution runs).
enum class AmrPic {none, refinable_cloud, instant_switch};

// The method of <dust>/amr_pic on this mesh, without printing (the dust module prints the
// notes).  The key experimental_smr_transfer of the internal development branches is read as
// amr_pic (smooth_p2 was removed and stops the run).
inline AmrPic ResolveAmrPic(ParameterInput *pin, const bool multilevel) {
  auto fatal = [](const std::string &msg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << msg << std::endl;
    std::exit(EXIT_FAILURE);
  };
  const bool given = pin->DoesParameterExist("dust", "amr_pic");
  const bool legacy = pin->DoesParameterExist("dust", "experimental_smr_transfer");
  std::string v = given ? pin->GetString("dust", "amr_pic") : "";
  // (older input and restart files name instant_switch owner_overlap)
  if (v == "owner_overlap") v = "instant_switch";
  if (given && v != "refinable_cloud" && v != "instant_switch") {
    fatal("<dust>/amr_pic = '" + v + "' not recognized (refinable_cloud or instant_switch)");
  }
  if (legacy) {
    std::string w = pin->GetString("dust", "experimental_smr_transfer");
    if (w == "owner_overlap") w = "instant_switch";
    if (w == "smooth_p2") {
      fatal("<dust>/experimental_smr_transfer = smooth_p2: the smooth_p2 transfer was removed "
            "(use <dust>/amr_pic = refinable_cloud or instant_switch)");
    }
    if (w != "refinable_cloud" && w != "instant_switch" && w != "none") {
      fatal("<dust>/experimental_smr_transfer = '" + w + "' not recognized");
    }
    if (given && w != v && !(w == "none" && !multilevel)) {
      fatal("<dust>/amr_pic = " + v + " and the old key experimental_smr_transfer = " + w +
            " disagree (drop the old key)");
    }
    if (!given) v = w;
  }
  if (v == "refinable_cloud") return AmrPic::refinable_cloud;
  if (v == "instant_switch") return multilevel ? AmrPic::instant_switch : AmrPic::none;
  if (v == "none") return AmrPic::none;
  return multilevel ? AmrPic::refinable_cloud : AmrPic::none;
}

// A key that is no longer read: one NOTE line (rank 0), the run goes on.
inline void NoteDroppedKey(ParameterInput *pin, const char *block, const char *key,
                           const char *why) {
  if (!pin->DoesParameterExist(block, key) || global_variable::my_rank != 0) return;
  std::cout << "# NOTE: <" << block << ">/" << key << " = " << pin->GetString(block, key)
            << " is no longer needed and is ignored (" << why << ")" << std::endl;
}

}  // namespace dust

#endif  // DUST_DUST_INPUT_HPP_
