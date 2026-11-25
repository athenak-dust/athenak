#ifndef GRAVITY_SG_TEST_KEYS_HPP_
#define GRAVITY_SG_TEST_KEYS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sg_test_keys.hpp
//! \brief Hidden test keys of <gravity> (diagnostics and A/B switches of the solver and of the
//! dust mass deposit).  Read only when given (not added to the parameters when absent; a given
//! value is in the restart files like every input value); SelfGravity reads
//! them from the input (the same on every rank, so are these switches: several choose
//! collectives) at its construction, before any solver is built.  The defaults hold until then.
//!   test_sg_report = a list of words (spaces or commas), each a diagnostic:
//!       template_check  every build of an operator with template rows compares its product with
//!                       the full CSR product (collective; memory option 8)
//!       mem             per-rank CSR operator statistics at every hierarchy rebuild
//!       level           per-rank rows / nonzeros / longest row / halo of every multigrid level,
//!                       and the phase rows of the first phase update
//!   test_lean_bits = <bit mask>  the memory options (self_gravity.cpp SgLeanBits; 0 = the former
//!                                path), default (or < 0): chosen automatically
//!   test_device_mpi = true       the multigrid halo exchange hands device buffers to MPI (needs
//!                                a GPU-aware MPI; experimental)
//!   test_spmv_rows = <n>         row-team SpMV for operators with >= n nonzeros per row on
//!                                average (0: off), default (or < 0) 40 on GPU backends, 0 on
//!                                host ones
//!   test_project_fuse = true | probe_test   the fused projection of the V-cycle (opt-in, after a
//!                                probe of MPI_Allreduce); probe_test: the probe sees rank 0's
//!                                sums with the last bit flipped (it must then refuse)
//!   test_halo_alltoallv = true   the former halo exchange (one MPI_Alltoallv per product)
//!   test_deposit_atomic = warp | plain | cycle | compare   the atomic adds of the dust gravity
//!                                mass deposit (dust_gravity.cpp), default warp

namespace gravity {

struct SgTestKeys {
  bool template_check = false, mem_stats = false, level_stats = false;
  int lean = -1;                  // -1: automatic
  bool device_mpi = false;
  int spmv_rows = -1;             // -1: the backend default
  bool project_fuse = false, project_fuse_probe_test = false;
  bool halo_alltoallv = false;
  int deposit_atomic = 1;         // 0 plain, 1 warp, 2 cycle, 3 compare (dust_gravity.cpp)
};
SgTestKeys &SgTest();             // (sparse_mg.cpp; set by SelfGravity)

}  // namespace gravity

#endif  // GRAVITY_SG_TEST_KEYS_HPP_
