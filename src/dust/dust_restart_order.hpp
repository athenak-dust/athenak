#ifndef DUST_DUST_RESTART_ORDER_HPP_
#define DUST_DUST_RESTART_ORDER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_restart_order.hpp
//! \brief Sizes of the cell-order index records in the dust state of the restart section
//! (dust_restart.cpp).  The dust state of a section has an int32 length, so the optional
//! index records go in only while the whole state stays within kDustStateMaxBytes; the
//! required records come first.  Header only, so that the length rule can be checked
//! without allocating an index of that size.

#include <cstddef>
#include <cstdint>

namespace dust {

// the largest dust state of one section (its length is written as int32)
constexpr std::size_t kDustStateMaxBytes = 0x7fffffff;
// a record: char[16] name, int32 bytes, data
constexpr std::size_t kDustRecordHeadBytes = 16 + sizeof(std::int32_t);
// record cloud_ord_n: int32[3] = {status, n, cycle of the last cell-order sort}
constexpr std::size_t kCloudOrderInfoBytes = kDustRecordHeadBytes + 3*sizeof(std::int32_t);
// status of record cloud_ord_n
constexpr std::int32_t kCloudOrderStored = 1;    // cloud_ord and cloud_ord_tag follow
constexpr std::int32_t kCloudOrderOmitted = 2;   // left out by the writer (state size)
constexpr std::int32_t kCloudOrderNone = 3;      // no index at the time of the file (n = 0)

// records cloud_ord (int32[n]) and cloud_ord_tag (uint64[n])
inline std::size_t CloudOrderArrayBytes(const std::size_t n) {
  return 2*kDustRecordHeadBytes + n*(sizeof(std::int32_t) + sizeof(std::uint64_t));
}

// whether the index of n entries fits after `used` bytes of records (record cloud_ord_n
// included) without the state passing `limit` bytes; no overflow for any n
inline bool CloudOrderFits(const std::size_t used, const std::size_t n, const std::size_t limit) {
  if (used > limit || limit - used < kCloudOrderInfoBytes) return false;
  const std::size_t room = limit - used - kCloudOrderInfoBytes;
  if (room < 2*kDustRecordHeadBytes) return false;
  return n <= (room - 2*kDustRecordHeadBytes)/(sizeof(std::int32_t) + sizeof(std::uint64_t));
}

}  // namespace dust
#endif  // DUST_DUST_RESTART_ORDER_HPP_
