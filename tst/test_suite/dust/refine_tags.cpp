//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file refine_tags.cpp
//! \brief Check of the child-tag guard of <problem>/refine_from (src/pgen/dust_amr/
//! refine_tags.hpp): the boundaries for refine_split 8, 2, 1 with 64-bit and int32 tags, the
//! wrap-around example (split 8: tags 1 and 2^61 + 1 both give 8 .. 15 in 64 bits), and the
//! guard against exact 128-bit arithmetic for many tags.  Prints PASS or the failures.
//! Build: the CMake target dust_refine_tag_check (EXCLUDE_FROM_ALL), or any C++17 compiler:
//!   g++ -std=c++17 -I src tst/test_suite/dust/refine_tags.cpp

#include <cstdint>
#include <iostream>
#include <limits>
#include <random>

#include "pgen/dust_amr/refine_tags.hpp"

namespace {
int nfail = 0;
void Expect(const bool ok, const char *what, const std::uint64_t tag, const int n) {
  if (!ok) {
    ++nfail;
    std::cout << "FAIL: " << what << " (tag " << tag << ", split " << n << ")" << std::endl;
  }
}
}  // namespace

int main() {
  using dust_setup::RefineChildTagsFit;
  using dust_setup::RefineTagMax;
  const std::uint64_t one = 1;
  const std::uint64_t m64 = RefineTagMax(true), m32 = RefineTagMax(false);
  Expect(m64 == std::numeric_limits<std::uint64_t>::max() - 1, "64-bit maximum 2^64 - 2", m64, 0);
  Expect(m32 == (one << 31) - 1, "int32 maximum 2^31 - 1", m32, 0);
  // the review example: split 8, tags 1 and 2^61 + 1 (8 (2^61 + 1) wraps to 8)
  std::uint64_t last = 0;
  Expect(RefineChildTagsFit(1, 8, m64, &last) && last == 15, "tag 1, split 8: children 8 .. 15", 1, 8);
  Expect(!RefineChildTagsFit((one << 61) + 1, 8, m64), "tag 2^61 + 1, split 8 rejected", (one << 61) + 1, 8);
  // the boundaries: largest accepted tag, first rejected tag
  struct Case {int n; std::uint64_t tmax, ok, bad;};
  const Case cases[] = {
    {8, m64, (one << 61) - 2, (one << 61) - 1},   // 8 (2^61 - 1) + 7 = 2^64 - 1 is never issued
    {2, m64, (one << 63) - 2, (one << 63) - 1},
    {1, m64, m64, m64 + 1},
    {8, m32, (one << 28) - 1, one << 28},         // 8 (2^28 - 1) + 7 = 2^31 - 1
    {2, m32, (one << 30) - 1, one << 30},
    {1, m32, m32, m32 + 1},
  };
  for (const auto &c : cases) {
    Expect(RefineChildTagsFit(c.ok, c.n, c.tmax, &last) &&
           last == static_cast<std::uint64_t>(c.n)*c.ok + c.n - 1 && last <= c.tmax,
           "largest tag accepted", c.ok, c.n);
    Expect(!RefineChildTagsFit(c.bad, c.n, c.tmax), "next tag rejected", c.bad, c.n);
  }
  Expect(!RefineChildTagsFit(std::numeric_limits<std::uint64_t>::max(), 1, m64), "2^64 - 1 rejected",
         std::numeric_limits<std::uint64_t>::max(), 1);
  Expect(!RefineChildTagsFit(5, 0, m64), "split 0 rejected", 5, 0);
  // against exact arithmetic (unsigned 128-bit), tags spread over all 64 bits
  std::mt19937_64 rng(20261101);
  long long nchecked = 0;
  for (int i=0; i<200000; ++i) {
    const int sh = static_cast<int>(rng() % 64);
    const std::uint64_t tag = rng() >> sh;
    for (const int n : {1, 2, 8}) {
      for (const std::uint64_t tmax : {m64, m32}) {
        const unsigned __int128 lastx = static_cast<unsigned __int128>(n)*tag + (n - 1);
        const bool want = lastx <= tmax;
        std::uint64_t got_last = 0;
        const bool got = RefineChildTagsFit(tag, n, tmax, &got_last);
        Expect(got == want, "agrees with 128-bit arithmetic", tag, n);
        if (got) Expect(got_last == static_cast<std::uint64_t>(lastx), "last child tag", tag, n);
        ++nchecked;
      }
    }
  }
  std::cout << "refine_tags: " << nchecked << " random cases + boundaries, " << nfail << " failures"
            << std::endl;
  std::cout << (nfail == 0 ? "PASS" : "FAIL") << std::endl;
  return nfail == 0 ? 0 : 1;
}
