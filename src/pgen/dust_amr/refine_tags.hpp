#ifndef PGEN_DUST_AMR_REFINE_TAGS_HPP_
#define PGEN_DUST_AMR_REFINE_TAGS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file refine_tags.hpp
//! \brief <problem>/refine_from: the tags of the children of a particle, nsplit*tag + k
//! (k = 0 .. nsplit - 1), checked against the largest tag the run can hold before they are
//! formed (no silent wrap-around of the 64-bit product).  Header only, no dependencies
//! (tst/test_suite/dust/refine_tags.cpp checks it).

#include <cstdint>
#include <limits>

namespace dust_setup {

// The largest tag a dust particle can carry: two int32 words (PTAG bits 0-31, PTAGH bits
// 32-63) hold 64 bits, of which 2^64 - 1 is never issued (LiveTagBound stops on it); without
// the PTAGH word the int32 PTAG alone (2^31 - 1).
inline std::uint64_t RefineTagMax(const bool has_high_word) {
  return has_high_word ? std::numeric_limits<std::uint64_t>::max() - 1
                       : static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max());
}

// Whether every child tag nsplit*tag + k, k = 0 .. nsplit - 1, is at most tag_max (the test
// is done without forming the product); the largest child tag in *last when it fits.
inline bool RefineChildTagsFit(const std::uint64_t tag, const int nsplit,
                               const std::uint64_t tag_max, std::uint64_t *last = nullptr) {
  if (nsplit < 1) return false;
  const std::uint64_t n = static_cast<std::uint64_t>(nsplit);
  if (tag_max < n - 1) return false;
  if (tag > (tag_max - (n - 1))/n) return false;   // n*tag + n - 1 <= tag_max
  if (last != nullptr) *last = n*tag + (n - 1);
  return true;
}

}  // namespace dust_setup

#endif  // PGEN_DUST_AMR_REFINE_TAGS_HPP_
