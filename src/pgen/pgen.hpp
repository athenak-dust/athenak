#ifndef PGEN_PGEN_HPP_
#define PGEN_PGEN_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file pgen.hpp
//  \brief definitions for ProblemGenerator class

#include <functional>
#include <memory>
#include <vector>

#include "geodesic-grid/spherical_grid.hpp"
#include "parameter_input.hpp"

using ProblemFinalizeFnPtr = void (*)(ParameterInput *pin, Mesh *pm);
using UserBoundaryFnPtr = void (*)(Mesh* pm);
using UserSrctermFnPtr = void (*)(Mesh* pm, const Real bdt);
using UserRefinementFnPtr = void (*)(MeshBlockPack* pmbp);
using UserHistoryFnPtr = void (*)(HistoryData *pdata, Mesh *pm);
// Work at the end of every cycle (Athena's Userwork_in_loop)
using UserWorkInLoopFnPtr = void (*)(Mesh *pm);

//----------------------------------------------------------------------------------------
//! \class ProblemGenerator

class ProblemGenerator {
 public:
  // constructor for new problems
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh);
  // constructor for restarts
  ProblemGenerator(ParameterInput *pin, Mesh *pmesh, IOWrapper resfile,
                   bool single_file_per_rank=false);
  ~ProblemGenerator() = default;

  // true if user BCs are specified on any face
  bool user_bcs;

  // true if user srcterms are specified
  bool user_srcs;

  // true if user history outputs are specified
  bool user_hist;

  // vector of SphericalGrid objects for analysis
  std::vector<std::unique_ptr<SphericalGrid>> spherical_grids;

  // function pointer for final work after main loop (e.g. compute errors).  Called by
  // Driver::Finalize()
  ProblemFinalizeFnPtr pgen_final_func=nullptr;
  // function pointer for user-enrolled BCs.  Called in ApplyPhysicalBCs in task list
  UserBoundaryFnPtr user_bcs_func=nullptr;
  // Optional: user BCs on the coarse arrays (coarse_u0, coarse indices) before the
  // ghost prolongation; without it user faces of the coarse arrays get zero gradient
  UserBoundaryFnPtr user_bcs_coarse_func=nullptr;
  UserSrctermFnPtr user_srcs_func=nullptr;
  UserRefinementFnPtr user_ref_func=nullptr;
  UserHistoryFnPtr user_hist_func=nullptr;
  // Called by the driver at the start of every cycle, before the integrator
  UserWorkInLoopFnPtr user_work_in_loop_func=nullptr;

  // predefined problem generator functions (default test suite)
  void CallProblemGenerator(ParameterInput *pin, bool is_restart);
  void Advection(ParameterInput *pin, const bool restart);
  void AlfvenWave(ParameterInput *pin, const bool restart);
  void BondiAccretion(ParameterInput *pin, const bool restart);
  void CShock(ParameterInput *pin, const bool restart);
  void Diffusion(ParameterInput *pin, const bool restart);
  void DustDamping(ParameterInput *pin, const bool restart);
  void DustNSH(ParameterInput *pin, const bool restart);
  void DustGravityTransfer(ParameterInput *pin, const bool restart);
  void DustSelfForce(ParameterInput *pin, const bool restart);
  void DustFreeFall(ParameterInput *pin, const bool restart);
  void DustJeans(ParameterInput *pin, const bool restart);
  void DustyWave(ParameterInput *pin, const bool restart);
  void LinearWave(ParameterInput *pin, const bool restart);
  void LWImplode(ParameterInput *pin, const bool restart);
  void Monopole(ParameterInput *pin, const bool restart);
  void MRI3d(ParameterInput *pin, const bool restart);
  void OrszagTang(ParameterInput *pin, const bool restart);
  void ObliqueShock(ParameterInput *pin, const bool restart);
  void IsoCollision(ParameterInput *pin, const bool restart);
  void ShockTube(ParameterInput *pin, const bool restart);
  void SimpleWave(ParameterInput *pin, const bool restart);
  void Shwave(ParameterInput *pin, const bool restart);
  void SelfGravityWave(ParameterInput *pin, const bool restart);
  void StreamingLinear(ParameterInput *pin, const bool restart);
  void RadiationLinearWave(ParameterInput *pin, const bool restart);
  void RadiationBeam(ParameterInput *pin, const bool restart);
  void Z4cBoostedPuncture(ParameterInput *pin, const bool restart);
  void Z4cLinearWave(ParameterInput *pin, const bool restart);

  // predefined problem generator functions for unit tests
  void EOSCompose(ParameterInput *pin, const bool restart);
  void GaussLegendre(ParameterInput *pin, const bool restart);
  void ProlongAmpUnit(ParameterInput *pin, const bool restart);

  // Generic error output function (using difference u0-u1)
  void OutputErrors(ParameterInput *pin, Mesh *pm);

  // template for user-specified problem generator
  void UserProblem(ParameterInput *pin, const bool restart);

 private:
  bool single_file_per_rank; // for restart file naming
  Mesh* pmy_mesh_;
};

#endif // PGEN_PGEN_HPP_
