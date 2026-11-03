//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#ifndef GRAVITY_ROW_INDEX_HPP_
#define GRAVITY_ROW_INDEX_HPP_

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gravity/composite.hpp"

namespace gravity {

// Positions in a growing list, by GlobalIndex key (open addressing, linear probing; several
// positions may share a key). Lists longer than kLinear use it instead of a linear scan: a
// coarse cell at a C1 join of the vacuum hierarchy can couple to thousands of small cells, and
// the scans over its rows and fit samples were quadratic (seconds per row on the coarse levels
// of a large box). A lookup visits exactly the stored positions of its key, so callers keep their
// entries, their order and the summation sequence of every coefficient: the results are
// bitwise those of the scan.
class PositionIndex {
 public:
  static constexpr std::size_t kLinear = 32;
  bool Active() const { return !slots_.empty(); }
  void Clear() { slots_.clear(); }
  // index positions 0..n-1 of a list whose key(i) is the key of position i
  template <class KeyOf> void Rebuild(std::size_t n, KeyOf key) {
    std::size_t size = 128;
    while (size < 2*n) size *= 2;
    slots_.assign(size, -1); mask_ = size-1;
    for (std::size_t i = 0; i < n; ++i) Insert(key(i), i);
  }
  // position i was appended (the list now has i+1 entries)
  template <class KeyOf> void Appended(GlobalIndex k, std::size_t i, KeyOf key) {
    if (2*(i+1) > slots_.size()) Rebuild(i+1, key);
    else Insert(k, i);
  }
  // a stored position with key k for which match(position) holds, -1 if none
  template <class KeyOf, class Match>
  std::ptrdiff_t Find(GlobalIndex k, KeyOf key, Match match) const {
    for (std::size_t h = Hash(k)&mask_; slots_[h] >= 0; h = (h+1)&mask_) {
      const auto i = static_cast<std::size_t>(slots_[h]);
      if (key(i) == k && match(i)) return slots_[h];
    }
    return -1;
  }

 private:
  static std::size_t Hash(GlobalIndex k) {
    const std::uint64_t x = static_cast<std::uint64_t>(k)*0x9e3779b97f4a7c15ULL;
    return static_cast<std::size_t>(x^(x >> 29));
  }
  void Insert(GlobalIndex k, std::size_t i) {
    std::size_t h = Hash(k)&mask_;
    while (slots_[h] >= 0) h = (h+1)&mask_;
    slots_[h] = static_cast<std::ptrdiff_t>(i);
  }
  std::vector<std::ptrdiff_t> slots_;
  std::size_t mask_ = 0;
};

// A coefficient row: each column once, in first-contribution order, indexed when long.
// Add gives the entries and sums of the plain scan
//   for (auto &e : v) if (e.first == column) { e.second += value; return; }
//   v.emplace_back(column, value);
// (value 0 skipped).
template <class T>
struct IndexedRow {
  using Entry = std::pair<GlobalIndex, T>;
  std::vector<Entry> v;
  PositionIndex index;
  void clear() { v.clear(); index.Clear(); }
  typename std::vector<Entry>::const_iterator begin() const { return v.begin(); }
  typename std::vector<Entry>::const_iterator end() const { return v.end(); }
  std::size_t size() const { return v.size(); }
  void Add(GlobalIndex column, T value) {
    if (value == T(0)) return;
    auto key = [this](std::size_t i) { return v[i].first; };
    if (!index.Active()) {
      for (auto &e : v) if (e.first == column) { e.second += value; return; }
      v.emplace_back(column, value);
      if (v.size() > PositionIndex::kLinear) index.Rebuild(v.size(), key);
      return;
    }
    const auto i = index.Find(column, key, [](std::size_t) { return true; });
    if (i >= 0) { v[i].second += value; return; }
    v.emplace_back(column, value);
    index.Appended(column, v.size()-1, key);
  }
  // the entries for a caller that reorders them (the index is dropped)
  std::vector<Entry> &Release() { index.Clear(); return v; }
};

}  // namespace gravity
#endif  // GRAVITY_ROW_INDEX_HPP_
