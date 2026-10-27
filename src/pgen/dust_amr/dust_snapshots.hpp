#ifndef PGEN_DUST_AMR_DUST_SNAPSHOTS_HPP_
#define PGEN_DUST_AMR_DUST_SNAPSHOTS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics and numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_snapshots.hpp
//! \brief the 2-D diagnostic outputs of the dust problem generators, keys <problem>/ba_*
//! (see dust_snapshots.cpp)

#include "athena.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "outputs/outputs.hpp"

namespace dust_snapshots {
// reads and checks the keys (call before the restart branch of the problem generator)
void Setup(ParameterInput *pin, Mesh *pm, const bool dust_active);
// restart: the numbers of the next snapshot and field frame
void Restart(ParameterInput *pin, Mesh *pm);
// from the user history function: the ba_history columns, then the files that are due
void HistoryColumns(HistoryData *pdata, Mesh *pm);
void Write(Mesh *pm);
}  // namespace dust_snapshots

#endif  // PGEN_DUST_AMR_DUST_SNAPSHOTS_HPP_
