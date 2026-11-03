#ifndef PARTICLES_PARTICLE_COUNTS_HPP_
#define PARTICLES_PARTICLE_COUNTS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_counts.hpp
//! \brief The number of particles of all ranks, and the index of the first particle of a rank
//! in rank order, from the counts of the ranks (Mesh::nprtcl_eachrank).  A rank holds fewer
//! than 2^31 particles (int counts and int indices into its particle arrays); all ranks together
//! may hold more, so both sums are 64-bit.  Header only, no dependencies
//! (tst/test_suite/particles/particle_counts.cpp checks it).

#include <cstdint>

namespace particles {

// the particles of ranks 0 .. nranks - 1
inline std::int64_t TotalCount(const int *count_eachrank, const int nranks) {
  std::int64_t n = 0;
  for (int r=0; r<nranks; ++r) n += count_eachrank[r];
  return n;
}

// the particles of the ranks before rank: the index of its first particle in rank order
inline std::int64_t RankOffset(const int *count_eachrank, const int rank) {
  return TotalCount(count_eachrank, rank);
}

}  // namespace particles

#endif  // PARTICLES_PARTICLE_COUNTS_HPP_
