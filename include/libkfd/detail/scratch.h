//===-- libkfd/detail/scratch.h - Scratch memory management -----*- C++ -*-===//
//
// Helpers for computing per-queue scratch (private segment) configurations.
// The Device owns a VRAM-backed pool and each queue sub-allocates from it.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_SCRATCH_H
#define LIBKFD_DETAIL_SCRATCH_H

#include "libkfd/dispatch.h"

#include <cstddef>
#include <cstdint>

namespace kfd {
class Device;
} // namespace kfd

namespace kfd::detail {

// Pessimistic lane count for scratch sizing. Wave32 kernels on RDNA still
// allocate 64-lane slots.
inline constexpr uint32_t SCRATCH_LANES_PER_WAVE = 64;

// COMPUTE_TMPRING_SIZE.WAVESIZE granularity: 1024B pre-GFX11, 256B GFX11+.
// References: LLVM SIDefines.h S_00B860_WAVESIZE_*;
//             ROCr amd_aql_queue.cpp mem_alignment_size
inline uint32_t scratch_alignment_unit(uint32_t gfx_version) {
  return gfx_version / 10000 >= 11 ? 256 : 1024;
}

uint32_t scratch_num_se(const Device &dev);

// The backing buffer is divided into fixed-size wave slots. The SPI assigns
// each wave a slot based on its execution unit (SE, CU, SIMD, slot-in-SIMD).
//
//   ┌──────────────┬──────────────┬─────┬──────────────────────┐
//   │  Wave slot 0 │  Wave slot 1 │ ... │  Wave slot (WAVES-1) │
//   └──────────────┴──────────────┴─────┴──────────────────────┘
//
//   per_wave = align_up(#lanes * per_thread, alignment_unit)
//   total    = WAVES * per_wave
//
// Within each slot, dwords are interleaved across lanes so that all lanes
// accessing the same per-thread offset hit distinct bank-friendly addresses.
//
//   Offset 0x000: [L0 dw0][L1 dw0]...[L63 dw0]  (256 bytes for wave64)
//   Offset 0x100: [L0 dw1][L1 dw1]...[L63 dw1]
//   ...
//
// Returns the packed COMPUTE_TMPRING_SIZE register value:
//   WAVES    [11:0]  - slot count (per-SE on GFX11+, per-XCC on GFX9-10)
//   WAVESIZE [N:12]  - per_wave / alignment_unit
//     N = 24 (GFX9-10, 13 bits), 26 (GFX11, 15 bits), 29 (GFX12, 18 bits)
//     alignment_unit: 1024B (GFX9-10), 256B (GFX11+)
uint32_t compute_tmpring_size(const Device &dev, uint32_t per_thread,
                              size_t region_size);

// Maximum device scratch slot count (CUs * MaxSlotsScratchCU, SE-aligned).
uint32_t scratch_device_slots(const Device &dev);

// Scratch wave slots needed to cover a particular dispatch grid.
uint32_t scratch_dispatch_slots(const Device &dev, Dim3 grid, Dim3 block);

// Backing allocation size for the given per-thread need and wave slot count.
size_t scratch_alloc_size(const Device &dev, uint32_t per_thread,
                          uint32_t slots);

} // namespace kfd::detail

#endif // LIBKFD_DETAIL_SCRATCH_H
