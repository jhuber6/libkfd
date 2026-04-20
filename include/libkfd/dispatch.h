//===-- libkfd/dispatch.h - Struct definitions for dispatch -----*- C++ -*-===//
//
// Simple struct definitions used to dispatch work to a device. Kept separate
// from the ABI to be exposed to the user.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DISPATCH_H
#define LIBKFD_DISPATCH_H

#include <cstdint>

namespace kfd {

// Three dimensional data for launch configurations.
struct Dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;
};

// User-facing dispatch configuration shared between kernarg setup and dispatch.
struct DispatchConfig {
  Dim3 grid;
  Dim3 block;
  uint32_t dynamic_lds = 0;
  uint32_t private_segment_size = 0;
};

} // namespace kfd

#endif // LIBKFD_DISPATCH_H
