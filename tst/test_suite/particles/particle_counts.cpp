//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file particle_counts.cpp
//! \brief Check of the 64-bit particle total and rank offset (src/particles/particle_counts.hpp)
//! with per-rank int counts whose sum passes 2^31 and 2^32: ranks holding INT_MAX particles
//! each, the 64-bit results against exact 128-bit sums, and random counts on up to 1024 ranks
//! (Mesh::nprtcl_total, the first tag of a rank, the VTK file offsets).  Prints PASS or the
//! failures.  Build: the CMake target particle_count_check (EXCLUDE_FROM_ALL), or any C++17
//! compiler:  g++ -std=c++17 -I src tst/test_suite/particles/particle_counts.cpp

#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "particles/particle_counts.hpp"

namespace {
int nfail = 0;
void Expect(const bool ok, const std::string &what) {
  if (!ok) {
    ++nfail;
    std::cout << "FAIL: " << what << std::endl;
  }
}
std::string Str(const __int128 v) {return std::to_string(static_cast<long long>(v));}

// the total and every offset of the counts against exact 128-bit prefix sums
void CheckAgainstExact(const std::vector<int> &c, const std::string &name) {
  const int nr = static_cast<int>(c.size());
  __int128 exact = 0;
  for (int r=0; r<nr; ++r) {
    const std::int64_t got = particles::RankOffset(c.data(), r);
    Expect(got == exact, name + ": offset of rank " + std::to_string(r) + " = " +
           std::to_string(got) + ", exact " + Str(exact));
    exact += c[r];
  }
  const std::int64_t tot = particles::TotalCount(c.data(), nr);
  Expect(tot == exact, name + ": total " + std::to_string(tot) + ", exact " + Str(exact));
  Expect(particles::RankOffset(c.data(), nr) == tot, name + ": offset past the last rank = total");
}
}  // namespace

int main() {
  const int imax = std::numeric_limits<int>::max();
  const std::int64_t two31 = std::int64_t(1) << 31, two32 = std::int64_t(1) << 32;
  long long ncases = 0;

  // no rank, one rank, small counts
  Expect(particles::TotalCount(nullptr, 0) == 0, "no rank: total 0");
  CheckAgainstExact({0}, "one empty rank"); ++ncases;
  CheckAgainstExact({7}, "one rank"); ++ncases;
  CheckAgainstExact({3, 0, 5, 1}, "small counts"); ++ncases;

  // the total just above 2^31 (int32 sums wrap to negative) and above 2^32 (uint32 sums wrap)
  {
    const std::vector<int> c = {imax, 1};
    CheckAgainstExact(c, "INT_MAX + 1"); ++ncases;
    Expect(particles::TotalCount(c.data(), 2) == two31, "INT_MAX + 1 = 2^31");
  }
  {
    const std::vector<int> c = {imax, imax, imax, 5};   // 4 ranks
    CheckAgainstExact(c, "3 INT_MAX + 5"); ++ncases;
    Expect(particles::TotalCount(c.data(), 4) == 3*std::int64_t(imax) + 5, "3 INT_MAX + 5");
    Expect(particles::RankOffset(c.data(), 3) == 3*std::int64_t(imax), "rank 3 starts at 3 INT_MAX");
    Expect(particles::RankOffset(c.data(), 3) > two32, "rank 3 starts above 2^32 (tag high word 1)");
    // the 32-bit sums of the old code: wrapped (two's complement), so they differ
    std::uint32_t wrap = 0;
    for (const int n : c) wrap += static_cast<std::uint32_t>(n);
    Expect(static_cast<std::int64_t>(static_cast<std::int32_t>(wrap)) !=
           particles::TotalCount(c.data(), 4), "an int32 sum would have wrapped");
    // VTK offsets: 3 floats per particle before this rank's positions
    const std::uint64_t off = 3*particles::RankOffset(c.data(), 3)*sizeof(float);
    Expect(off == 3ULL*3ULL*static_cast<std::uint64_t>(imax)*4ULL, "VTK position offset of rank 3");
  }
  {
    std::vector<int> c(4096, imax);   // 4096 ranks of INT_MAX: about 8.8e12 in total
    CheckAgainstExact(c, "4096 INT_MAX"); ++ncases;
  }

  // random counts, also with empty ranks and full ranks, on 1 .. 1024 ranks
  std::mt19937_64 rng(20261102);
  for (int i=0; i<1000; ++i) {
    const int nr = 1 + static_cast<int>(rng() % 1024);
    std::vector<int> c(nr);
    for (auto &n : c) {
      const int kind = static_cast<int>(rng() % 4);
      n = (kind == 0) ? 0 : (kind == 1 ? imax : static_cast<int>(rng() % (std::uint64_t(imax) + 1)));
    }
    CheckAgainstExact(c, "random " + std::to_string(i));
    ++ncases;
  }

  std::cout << "particle_counts: " << ncases << " count sets, " << nfail << " failures" << std::endl;
  std::cout << (nfail == 0 ? "PASS" : "FAIL") << std::endl;
  return nfail == 0 ? 0 : 1;
}
