//===-- libkfd/condition.h - Comparison function enum ----------*- C++ -*-===//
//
// Common condition values for waiting on values. These intentionally match the
// hardware bit-patterns in the SDMA and PM4 queue interface.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_CONDITION_H
#define LIBKFD_CONDITION_H

#include <cstdint>

namespace kfd {

enum class Condition : uint8_t {
  LT = 1,
  LTE = 2,
  EQ = 3,
  NE = 4,
  GTE = 5,
  GT = 6
};

} // namespace kfd

#endif // LIBKFD_CONDITION_H
