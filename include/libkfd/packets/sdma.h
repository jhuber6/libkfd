//===-- libkfd/packets/sdma.h - Packet encoding for the SDMA CP -*- C++ -*-===//
//
// Header and packed encodings for the SDMA command processor. Packets are
// issued by writing the header with one of the valid packets. Controls data
// movement between valid addresses.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_PACKETS_SDMA_H
#define LIBKFD_PACKETS_SDMA_H

#include "libkfd/abi.h"
#include "libkfd/condition.h"
#include "libkfd/detail/utility.h"

#include <cstdint>

namespace kfd::sdma {

// References: amdgpu/navi10_sdma_pkt_open.h
enum Op : uint8_t {
  NOP = 0,
  COPY = 1,
  FENCE = 5,
  TRAP = 6,
  POLL_REGMEM = 8,
  ATOMIC = 10,
  CONST_FILL = 11,
  TIMESTAMP = 13,
  GCR_REQ = 17,
};

inline constexpr uint32_t COPY_LINEAR_DWORDS = 7;
inline constexpr uint32_t FENCE_DWORDS = 4;
inline constexpr uint32_t TRAP_DWORDS = 2;
inline constexpr uint32_t POLL_REGMEM_DWORDS = 6;
inline constexpr uint32_t ATOMIC_DWORDS = 8;
inline constexpr uint32_t CONST_FILL_DWORDS = 5;
inline constexpr uint32_t TIMESTAMP_DWORDS = 3;
inline constexpr uint32_t GCR_REQ_DWORDS = 5;

// The COPY_LINEAR count field is 22 bits on older hardware (max 0x400000)
// and 30 bits on newer hardware (max 0x40000000). ROCr uses count_ext for
// gfx90a, gfx940+, gfx1030+, and all gfx11/gfx12.
inline constexpr uint32_t MAX_COPY_LINEAR_22BIT = 0x3fffe0;
inline constexpr uint32_t MAX_COPY_LINEAR_30BIT = 0x3ffffffe;

inline uint32_t max_copy_linear_bytes(uint32_t gfx_version) {
  uint32_t major = gfx_version / 10000;
  uint32_t minor = (gfx_version / 100) % 100;
  uint32_t step = gfx_version % 100;
  if (major >= 11)
    return MAX_COPY_LINEAR_30BIT;
  if (major == 10 && minor >= 3)
    return MAX_COPY_LINEAR_30BIT;
  if (major == 9 && (minor >= 4 || step >= 10))
    return MAX_COPY_LINEAR_30BIT;
  return MAX_COPY_LINEAR_22BIT;
}

// NOP - skip dwords. total_dwords includes the header itself.
// Header: op[7:0]=0, sub_op[15:8]=0, count[29:16]=total_dwords-1.
//
// References: SDMA_PKT_NOP_HEADER_* in amdgpu/navi10_sdma_pkt_open.h
inline uint32_t nop(uint32_t *out, uint32_t total_dwords) {
  out[0] = static_cast<uint32_t>(total_dwords - 1) << 16;
  for (uint32_t i = 1; i < total_dwords; ++i)
    out[i] = 0;
  return total_dwords;
}

// COPY_LINEAR - DMA copy between GPU-visible addresses.
//
// Packet layout (7 dwords, single destination):
//   [0] header     - op=1 (COPY), sub_op=0 (LINEAR)
//   [1] count      - byte count minus one (1-based on GFX9+)
//   [2] parameter  - src/dst cache policy (0 = default)
//   [3] src_lo     - source address low 32 bits
//   [4] src_hi     - source address high 32 bits
//   [5] dst_lo     - destination address low 32 bits
//   [6] dst_hi     - destination address high 32 bits
//
// References: SDMA_PKT_COPY_LINEAR_* in amdgpu/navi10_sdma_pkt_open.h;
//             sdma_v6_0_emit_copy_buffer in amdgpu/sdma_v6_0.c
inline uint32_t copy_linear(uint32_t *out, void *dst, const void *src,
                            uint32_t bytes) {
  out[0] = COPY;
  out[1] = bytes > 0 ? bytes - 1 : 0;
  out[2] = 0;
  out[3] = detail::lo(reinterpret_cast<uintptr_t>(src));
  out[4] = detail::hi(reinterpret_cast<uintptr_t>(src));
  out[5] = detail::lo(reinterpret_cast<uintptr_t>(dst));
  out[6] = detail::hi(reinterpret_cast<uintptr_t>(dst));
  return COPY_LINEAR_DWORDS;
}

// FENCE - write a 32-bit value to a GPU-visible address.
//
// Packet layout (4 dwords):
//   [0] header   - op=5 (FENCE); on GFX10+ adds snoop/system/mtype bits
//   [1] addr_lo  - byte address low 32 bits
//   [2] addr_hi  - byte address high 32 bits
//   [3] data     - 32-bit value to write
//
// gfx90a+ / GFX10+ header bits (SDMA 4.4.x+ and all SDMA v5+):
//   mtype[18:16]=3 (uncached), sys[20]=1, snp[22]=1
//
// References: SDMA_PKT_FENCE_HEADER_* in amdgpu/navi10_sdma_pkt_open.h;
//             sdma_v6_0_ring_emit_fence in amdgpu/sdma_v6_0.c
inline uint32_t fence(uint32_t *out, uint32_t gfx_version, void *addr,
                      uint32_t value) {
  if (gfx_version >= abi::GFX_VERSION_GFX9_A)
    out[0] = FENCE | (3u << 16) | (1u << 20) | (1u << 22);
  else
    out[0] = FENCE;
  out[1] = detail::lo(reinterpret_cast<uintptr_t>(addr));
  out[2] = detail::hi(reinterpret_cast<uintptr_t>(addr));
  out[3] = value;
  return FENCE_DWORDS;
}

// TRAP - fire an interrupt. The kernel matches int_context against event_id
// to wake the waiting thread.
//
// Packet layout (2 dwords):
//   [0] header      - op=6 (TRAP)
//   [1] int_context  - 28-bit interrupt context (event trigger data)
//
// References: SDMA_PKT_TRAP_* in amdgpu/navi10_sdma_pkt_open.h
inline uint32_t trap(uint32_t *out, uint32_t int_ctx) {
  out[0] = TRAP;
  out[1] = int_ctx & 0x0FFFFFFFu;
  return TRAP_DWORDS;
}

// CONST_FILL - fill memory with a repeated 32-bit value.
//
// Packet layout (5 dwords):
//   [0] header   - op=11, fillsize[31:30]=2 (DW fill)
//   [1] dst_lo   - destination address low 32 bits
//   [2] dst_hi   - destination address high 32 bits
//   [3] data     - 32-bit fill value
//   [4] count    - byte count minus one (1-based on GFX9+)
//
// Destination address and byte count must be dword-aligned.
//
// On GFX12 (RDNA4), the DW-fill size semantics changed: the HW implicitly
// adds one dword (4 bytes) to the encoded count, so the count must be
// (bytes - 4) rather than the pre-GFX12 (bytes - 1).
//
// References: SDMA_PKT_CONSTANT_FILL_* in amdgpu/navi10_sdma_pkt_open.h
inline uint32_t const_fill(uint32_t *out, uint32_t gfx_version, void *dst,
                           uint32_t value, uint32_t bytes) {
  out[0] = CONST_FILL | (2u << 30);
  out[1] = detail::lo(reinterpret_cast<uintptr_t>(dst));
  out[2] = detail::hi(reinterpret_cast<uintptr_t>(dst));
  out[3] = value;
  out[4] = gfx_version >= abi::GFX_VERSION_GFX12 ? (bytes > 4 ? bytes - 4 : 0)
                                                 : (bytes > 0 ? bytes - 1 : 0);
  return CONST_FILL_DWORDS;
}

// POLL_REGMEM - poll a memory location until a condition is satisfied.
//
// Packet layout (6 dwords):
//   [0] header      - op=8, func[30:28], mem_poll[31]=1
//   [1] addr_lo     - byte address low 32 bits
//   [2] addr_hi     - byte address high 32 bits
//   [3] value       - 32-bit comparison value
//   [4] mask        - applied to memory value before comparison
//   [5] interval[15:0], retry_count[27:16]
//
// The SDMA engine polls: (memory[addr] & mask) <cond> value
//
// References: SDMA_PKT_POLL_REGMEM_* in amdgpu/navi10_sdma_pkt_open.h
inline uint32_t poll_regmem(uint32_t *out, void *addr, Condition cond,
                            uint32_t value, uint32_t mask = 0xFFFFFFFF,
                            uint32_t interval = 0x0A,
                            uint32_t retry_count = 0xFFF) {
  out[0] = POLL_REGMEM | (static_cast<uint32_t>(cond) << 28) | (1u << 31);
  out[1] = detail::lo(reinterpret_cast<uintptr_t>(addr));
  out[2] = detail::hi(reinterpret_cast<uintptr_t>(addr));
  out[3] = value;
  out[4] = mask;
  out[5] = (interval & 0xFFFFu) | ((retry_count & 0xFFFu) << 16);
  return POLL_REGMEM_DWORDS;
}

// ATOMIC - perform an SDMA-side atomic operation on a memory location.
//
// Packet layout (8 dwords):
//   [0] header       - op=10, l[16]=0/1 (32/64-bit), operation[31:25]
//   [1] addr_lo      - byte address low 32 bits
//   [2] addr_hi      - byte address high 32 bits
//   [3] src_data_lo  - operand low 32 bits
//   [4] src_data_hi  - operand high 32 bits
//   [5] cmp_data_lo  - compare value low (for CMPSWAP, zero otherwise)
//   [6] cmp_data_hi  - compare value high
//   [7] loop_interval[12:0]
//
// References: SDMA_PKT_ATOMIC_* in amdgpu/navi10_sdma_pkt_open.h;
//             sdma_registers.h (SDMA_ATOMIC_ADD64 = 0x2F)
enum AtomicOp : uint32_t {
  ATOMIC_SWAP_32 = 0x07,
  ATOMIC_CMPSWAP_32 = 0x08,
  ATOMIC_ADD_32 = 0x0F,
  ATOMIC_SWAP_64 = 0x27,
  ATOMIC_ADD_64 = 0x2F,
};

inline uint32_t atomic_mem(uint32_t *out, AtomicOp op, void *addr,
                           int64_t src_data, int64_t cmp_data = 0) {
  uint32_t is_64_bit = (static_cast<uint32_t>(op) >= 0x20) ? 1u : 0u;
  out[0] = ATOMIC | (is_64_bit << 16) | (static_cast<uint32_t>(op) << 25);
  out[1] = detail::lo(reinterpret_cast<uintptr_t>(addr));
  out[2] = detail::hi(reinterpret_cast<uintptr_t>(addr));
  out[3] = detail::lo(static_cast<uint64_t>(src_data));
  out[4] = detail::hi(static_cast<uint64_t>(src_data));
  out[5] = detail::lo(static_cast<uint64_t>(cmp_data));
  out[6] = detail::hi(static_cast<uint64_t>(cmp_data));
  out[7] = 0;
  return ATOMIC_DWORDS;
}

// TIMESTAMP - write the 64-bit GPU global timestamp to a memory address.
//
// Packet layout (3 dwords):
//   [0] header    - op=13, sub_op=2 (GET_GLOBAL)
//   [1] addr_lo   - destination byte address low 32 bits
//   [2] addr_hi   - destination byte address high 32 bits
//
// The destination address must be 8-byte aligned. Writes an 8-byte
// GPU clock value suitable for measuring SDMA operation latency.
//
// References: SDMA_PKT_TIMESTAMP_* in amdgpu/navi10_sdma_pkt_open.h
inline uint32_t timestamp(uint32_t *out, void *addr) {
  out[0] = TIMESTAMP | (2u << 8); // sub_op = GET_GLOBAL
  out[1] = detail::lo(reinterpret_cast<uintptr_t>(addr));
  out[2] = detail::hi(reinterpret_cast<uintptr_t>(addr));
  return TIMESTAMP_DWORDS;
}

// GCR_REQ - GPU cache flush/invalidate from the SDMA engine (GFX10+).
//
// Packet layout (5 dwords):
//   [0] header      - op=17 (GCR_REQ)
//   [1] payload1    - base_va[31:7] at bits [31:7]
//   [2] payload2    - base_va[47:32] at [15:0], gcr_control[15:0] at [31:16]
//   [3] payload3    - gcr_control[18:16] at [2:0], limit_va[31:7] at [31:7]
//   [4] payload4    - limit_va[47:32] at [15:0], vmid at [27:24]
//
// For a full flush (all addresses), base_va=0 and limit_va=0.
//
// References: SDMA_PKT_GCR_REQ_* in amdgpu/sdma_v6_0_0_pkt_open.h;
//             sdma_v7_0_ring_emit_mem_sync in amdgpu/sdma_v7_0.c
inline constexpr uint32_t GCR_GLI_INV_ALL = 1u << 0;
inline constexpr uint32_t GCR_GLM_WB = 1u << 4;
inline constexpr uint32_t GCR_GLM_INV = 1u << 5;
inline constexpr uint32_t GCR_GLK_INV = 1u << 7;
inline constexpr uint32_t GCR_GLV_INV = 1u << 8;
inline constexpr uint32_t GCR_GL1_INV = 1u << 9;
inline constexpr uint32_t GCR_GL2_INV = 1u << 14;
inline constexpr uint32_t GCR_GL2_WB = 1u << 15;

inline constexpr uint32_t GCR_FLUSH_ALL =
    GCR_GLI_INV_ALL | GCR_GLM_INV | GCR_GLK_INV | GCR_GLV_INV | GCR_GL1_INV |
    GCR_GL2_INV | GCR_GL2_WB;

inline uint32_t gcr_req(uint32_t *out, uint32_t gcr_cntl = GCR_FLUSH_ALL) {
  out[0] = GCR_REQ;
  out[1] = 0;
  out[2] = (gcr_cntl & 0xFFFFu) << 16;
  out[3] = (gcr_cntl >> 16) & 0x7u;
  out[4] = 0;
  return GCR_REQ_DWORDS;
}

} // namespace kfd::sdma

#endif // LIBKFD_PACKETS_SDMA_H
