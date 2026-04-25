//===-- libkfd/packets/pm4.h - Packet encoding for the PM4 CP ---*- C++ -*-===//
//
// Header and packet encodings for the PM4 command processor interface. Packets
// are issued by writing a header that corresponds to one of the valid packet
// types to the queue. Controls memory operations, kernel launches, and events.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_PACKETS_PM4_H
#define LIBKFD_PACKETS_PM4_H

#include "libkfd/abi.h"
#include "libkfd/condition.h"
#include "libkfd/detail/scratch.h"
#include "libkfd/detail/utility.h"

#include <cstdint>

namespace kfd::pm4 {

// PM4 type-3 packet header (GCN / RDNA).
//
// Header layout (32 bits):
//   [0]     predicate   - skip packet when the CP predicate register is clear
//   [1]     shaderType  - 0 = graphics, 1 = compute
//   [7:2]   reserved
//   [15:8]  opcode      - IT_* opcode (see Opcode enum below)
//   [29:16] count       - number of body dwords minus one (14-bit field,
//                          max 0x3FFF, so a packet can span up to 16385 dwords)
//   [31:30] type        - 3 for type-3 packets
//
// Total packet size = count + 2 dwords (1 header + (count + 1) body dwords).
//
// References: PACKET3_COMPUTE in amdgpu/soc15d.h
inline constexpr uint32_t header(uint8_t opcode, uint16_t count) {
  return uint32_t(1) << 1 | static_cast<uint32_t>(opcode) << 8 |
         static_cast<uint32_t>(count) << 16 | uint32_t(3) << 30;
}

// IT_* opcodes consumed by the MEC (Micro Engine Compute) command processor.
//
// References: amdkfd/kfd_pm4_opcodes.h;
//             amdgpu/soc15d.h
enum Opcode : uint8_t {
  NOP = 0x10,
  DISPATCH_DIRECT = 0x15,
  ATOMIC_MEM = 0x1E,
  WRITE_DATA = 0x37,
  WAIT_REG_MEM = 0x3C,
  RELEASE_MEM = 0x49,
  DMA_DATA = 0x50,
  ACQUIRE_MEM = 0x58,
  INDIRECT_BUFFER = 0x3F,
  SET_SH_REG = 0x76,
};

// Compute SH register offsets and dispatch initiator bits.
//
// Bitfield layouts for PGM_RSRC1/2/3 and kernel_code_properties are
// documented in abi.h alongside the KernelDescriptor struct.
//
// References: include/asic_reg/gc/gc_10_3_0_offset.h;
//             amdkfd/kfd_pm4_headers.h;
namespace regs {
inline constexpr uint32_t SH_BASE = 0x2C00;

// 0x2E04-0x2E0B: grid origin, block dims, perf (8 regs).
inline constexpr uint32_t COMPUTE_START_X = 0x2E04;
inline constexpr uint32_t COMPUTE_START_Y = 0x2E05;
inline constexpr uint32_t COMPUTE_START_Z = 0x2E06;
inline constexpr uint32_t COMPUTE_NUM_THREAD_X = 0x2E07;
inline constexpr uint32_t COMPUTE_NUM_THREAD_Y = 0x2E08;
inline constexpr uint32_t COMPUTE_NUM_THREAD_Z = 0x2E09;
inline constexpr uint32_t COMPUTE_PIPELINESTAT_ENABLE = 0x2E0A;
inline constexpr uint32_t COMPUTE_PERFCOUNT_ENABLE = 0x2E0B;

// 0x2E0C-0x2E11: program address, AQL pkt addr, scratch base (6 regs).
// PGM_LO/HI hold the entry VA >> 8. PKT_ADDR is AQL-only (0 for PM4).
inline constexpr uint32_t COMPUTE_PGM_LO = 0x2E0C;
inline constexpr uint32_t COMPUTE_PGM_HI = 0x2E0D;
inline constexpr uint32_t COMPUTE_PKT_ADDR_LO = 0x2E0E;
inline constexpr uint32_t COMPUTE_PKT_ADDR_HI = 0x2E0F;
inline constexpr uint32_t COMPUTE_DISPATCH_SCRATCH_BASE_LO = 0x2E10;
inline constexpr uint32_t COMPUTE_DISPATCH_SCRATCH_BASE_HI = 0x2E11;

// 0x2E12-0x2E13: program resource descriptors (2 regs).
inline constexpr uint32_t COMPUTE_PGM_RSRC1 = 0x2E12;
inline constexpr uint32_t COMPUTE_PGM_RSRC2 = 0x2E13;

// 0x2E15-0x2E1A: occupancy, SIMD masks, scratch ring (6 regs).
// TMPRING_SIZE[11:0] = max scratch waves; [24:12] = per-wave size in alignment
// units (1024B on GFX9-10, 256B on GFX11+).
inline constexpr uint32_t COMPUTE_RESOURCE_LIMITS = 0x2E15;
inline constexpr uint32_t COMPUTE_STATIC_THREAD_MGMT_SE0 = 0x2E16;
inline constexpr uint32_t COMPUTE_STATIC_THREAD_MGMT_SE1 = 0x2E17;
inline constexpr uint32_t COMPUTE_TMPRING_SIZE = 0x2E18;
inline constexpr uint32_t COMPUTE_STATIC_THREAD_MGMT_SE2 = 0x2E19;
inline constexpr uint32_t COMPUTE_STATIC_THREAD_MGMT_SE3 = 0x2E1A;

// 0x2E1B-0x2E1E: preemption resume state (4 regs). Zeroed for fresh dispatches;
// managed by the CWSR trap handler and CP.
inline constexpr uint32_t COMPUTE_RESTART_X = 0x2E1B;
inline constexpr uint32_t COMPUTE_RESTART_Y = 0x2E1C;
inline constexpr uint32_t COMPUTE_RESTART_Z = 0x2E1D;
inline constexpr uint32_t COMPUTE_THREAD_TRACE_ENABLE = 0x2E1E;

// 0x2E28: extended resource descriptor (single reg).
inline constexpr uint32_t COMPUTE_PGM_RSRC3 = 0x2E28;

// 0x2E40-0x2E4F: user data SGPRs (16 regs), loaded into s0-s15.
inline constexpr uint32_t COMPUTE_USER_DATA_0 = 0x2E40;

// DISPATCH_DIRECT compute shader initiator bits.
//
// References: COMPUTE_DISPATCH_INITIATOR in
// include/asic_reg/gc/gc_10_3_0_sh_mask.h
inline constexpr uint32_t DISPATCH_COMPUTE_SHADER_EN = 1u << 0;
// Work-group IDs start at 0, ignoring COMPUTE_START_X/Y/Z.
inline constexpr uint32_t DISPATCH_FORCE_START_AT_000 = 1u << 2;
// Dims are total threads, otherwise dims are group counts.
inline constexpr uint32_t DISPATCH_USE_THREAD_DIMENSIONS = 1u << 5;
// Enable Wave32 mode (GFX10+).
inline constexpr uint32_t DISPATCH_CS_W32_EN = 1u << 15;
} // namespace regs

// NOP fill value for unused ring slots. count=0x3FFF makes the CP skip
// the maximum number of body dwords, so if every slot is filled with this
// value, the CP simply advances until it hits real packets at wptr.
inline constexpr uint32_t CMD_NOP = header(NOP, 0x3FFF);

// Write a properly-sized NOP packet spanning exactly `dwords` ring slots.
// A single NOP header encodes the body count in its COUNT field.
inline void nop_fill(uint32_t *out, uint32_t dwords) {
  if (dwords == 0)
    return;
  if (dwords == 1) {
    out[0] = CMD_NOP;
    return;
  }
  out[0] = header(NOP, static_cast<uint16_t>(dwords - 2));
  for (uint32_t i = 1; i < dwords; ++i)
    out[i] = 0;
}

// L2 cache allocation policy. Shared field encoding at bits [26:25] across
// RELEASE_MEM, ATOMIC_MEM, WRITE_DATA, and WAIT_REG_MEM.
//
// GFX9 (soc15d.h) and GFX10+ (nvd.h) share the same encoding and position.
//
// References: PACKET3_RELEASE_MEM_CACHE_POLICY in amdgpu/nvd.h
enum CachePolicy : uint32_t {
  POLICY_LRU = 0,    // Normal cached (default)
  POLICY_STREAM = 1, // Streaming (evict before others)
  POLICY_NOA = 2,    // No-allocate (write-through, skip L2 allocation)
  POLICY_BYPASS = 3, // Skip L2 entirely, go to memory controller
};

// WRITE_DATA - CP stores one or more 32-bit values to a GPU-visible address.
//
// Packet layout (5 dwords for a single-value write):
//   [0]  header         - IT_WRITE_DATA, count = 3
//   [1]  control        - dst_sel, wr_confirm, cache_policy, engine_sel
//   [2]  dst_addr_lo    - byte address low
//   [3]  dst_addr_hi    - byte address high
//   [4]  data           - 32-bit value
//
// Control word bitfields:
//   [11:8]  dst_sel      - destination target (see DstSel enum)
//   [16]    addr_incr    - 0 = increment address per dword (default)
//   [20]    wr_confirm   - 1 = wait for write to be visible before advancing
//   [26:25] cache_policy - L2 allocation policy
//   [31:30] engine_sel   - 0 = ME (compute), 1 = PFP (graphics), 2 = CE
//
// References: PACKET3_WRITE_DATA in amdgpu/soc15d.h;
//             PACKET3_WRITE_DATA__DST_SEL in amdgpu/nvd.h

// Destination selector for WRITE_DATA.
//
// References: PACKET3_WRITE_DATA__DST_SEL__* in amdgpu/soc15d.h
enum DstSel : uint32_t {
  DST_MEM_REGISTER = 0, // Memory-mapped register
  DST_MEM_GRBM = 1,     // Memory sync via GRBM
  DST_TC_L2 = 2,        // Through TC / L2
  DST_GDS = 3,          // Global Data Share
  DST_MEM_ASYNC = 5,    // Memory async (direct to memory, non-blocking)
};

inline constexpr uint32_t WRITE_DATA_DWORDS = 5;

inline uint32_t write_data(uint32_t *out, void *addr, uint32_t value,
                           DstSel dst = DST_MEM_ASYNC,
                           CachePolicy policy = POLICY_NOA,
                           bool wr_confirm = true) {
  out[0] = header(WRITE_DATA, 3);
  out[1] = (static_cast<uint32_t>(dst) << 8) | (wr_confirm ? (1u << 20) : 0u) |
           (static_cast<uint32_t>(policy) << 25);
  out[2] = detail::lo(reinterpret_cast<uintptr_t>(addr));
  out[3] = detail::hi(reinterpret_cast<uintptr_t>(addr));
  out[4] = value;
  return WRITE_DATA_DWORDS;
}

// DMA_DATA - CP-initiated memory copy or fill.
//
// Packet layout (7 dwords):
//   [0]  header       - IT_DMA_DATA, count = 5
//   [1]  control      - engine_sel[0], src_cache_policy[13],
//                        dst_sel[21:20], dst_cache_policy[25],
//                        src_sel[30:29], cp_sync[31]
//   [2]  src_addr_lo  - source byte address low (or immediate 32-bit fill
//                        value when src_sel = DMA_SRC_DATA)
//   [3]  src_addr_hi  - source byte address high (ignored for DMA_SRC_DATA)
//   [4]  dst_addr_lo  - destination byte address low
//   [5]  dst_addr_hi  - destination byte address high
//   [6]  command/size - SAS[26], DAS[27], SAIC[28], DAIC[29], RAW_WAIT[30]
//                        | byte_count
//
// The CP performs the transfer inline - it does not advance to the next
// packet until complete when CP_SYNC is set. This makes it ideal for small
// inline copies (kernargs, flags, results) without cross-queue sync or
// shader compilation overhead.
//
// BYTE_COUNT field width varies by generation:
//   GFX9:   bits [20:0]  - max 2 MiB per packet
//   GFX10+: bits [25:0]  - max 64 MiB per packet
// Command bits [30:26] are identical across generations.
//
// Control word cache policy fields are 1-bit each (0 = LRU, 1 = Stream),
// unlike the 2-bit cache_policy used by RELEASE_MEM / WRITE_DATA. They
// only affect transfers routed through L2 (src/dst_sel = 3).
//
// GFX10+ also defines DMA_DATA_FILL_MULTI (opcode 0x9A) for multi-
// destination fills; not implemented here.
//
// References: PACKET3_DMA_DATA in amdgpu/soc15d.h, amdgpu/nvd.h;
//             IT_DMA_DATA in amdkfd/kfd_pm4_opcodes.h;
//             gfx_v9_0.c (GDS clear usage on compute ring)

// Source selector for DMA_DATA control[30:29].
//
// References: PACKET3_DMA_DATA_SRC_SEL in amdgpu/soc15d.h
enum DmaSrcSel : uint32_t {
  DMA_SRC_ADDR = 0,    // Memory via source address space (bypasses L2)
  DMA_SRC_GDS = 1,     // Global Data Share
  DMA_SRC_DATA = 2,    // Immediate 32-bit value in DW2 (fill mode)
  DMA_SRC_ADDR_L2 = 3, // Memory via L2 cache
};

// Destination selector for DMA_DATA control[21:20].
//
// References: PACKET3_DMA_DATA_DST_SEL in amdgpu/soc15d.h
enum DmaDstSel : uint32_t {
  DMA_DST_ADDR = 0,    // Memory via dest address space (bypasses L2)
  DMA_DST_GDS = 1,     // Global Data Share
  DMA_DST_ADDR_L2 = 3, // Memory via L2 cache
};

// Command word flags for DMA_DATA DW6[30:26].
//
// References: PACKET3_DMA_DATA_CMD_* in amdgpu/soc15d.h
inline constexpr uint32_t DMA_CMD_SAS = 1u << 26;
inline constexpr uint32_t DMA_CMD_DAS = 1u << 27;
inline constexpr uint32_t DMA_CMD_SAIC = 1u << 28;
inline constexpr uint32_t DMA_CMD_DAIC = 1u << 29;
inline constexpr uint32_t DMA_CMD_RAW_WAIT = 1u << 30;

inline constexpr uint32_t DMA_DATA_DWORDS = 7;

inline constexpr uint32_t DMA_DATA_MAX_BYTES_GFX9 = 1u << 21;  // 2 MiB
inline constexpr uint32_t DMA_DATA_MAX_BYTES_GFX10 = 1u << 26; // 64 MiB

// General DMA_DATA builder with explicit control and command flags.
inline uint32_t dma_data(uint32_t *out, uint32_t control, uint64_t src_addr,
                         uint64_t dst_addr, uint32_t byte_count,
                         uint32_t cmd_flags = 0) {
  out[0] = header(DMA_DATA, 5);
  out[1] = control;
  out[2] = detail::lo(src_addr);
  out[3] = detail::hi(src_addr);
  out[4] = detail::lo(dst_addr);
  out[5] = detail::hi(dst_addr);
  out[6] = cmd_flags | byte_count;
  return DMA_DATA_DWORDS;
}

// Memory-to-memory copy through L2. CP blocks until complete.
inline uint32_t dma_data_copy(uint32_t *out, void *dst, const void *src,
                              uint32_t byte_count) {
  uint32_t control = (static_cast<uint32_t>(DMA_SRC_ADDR_L2) << 29) |
                     (static_cast<uint32_t>(DMA_DST_ADDR_L2) << 20) |
                     (1u << 31); // CP_SYNC
  return dma_data(out, control, reinterpret_cast<uintptr_t>(src),
                  reinterpret_cast<uintptr_t>(dst), byte_count);
}

// Immediate 32-bit fill. Replicates 'value' across 'byte_count' bytes at
// 'dst'. CP blocks until complete.
inline uint32_t dma_data_fill(uint32_t *out, void *dst, uint32_t value,
                              uint32_t byte_count) {
  uint32_t control = (static_cast<uint32_t>(DMA_SRC_DATA) << 29) |
                     (static_cast<uint32_t>(DMA_DST_ADDR_L2) << 20) |
                     (1u << 31); // CP_SYNC
  return dma_data(out, control, static_cast<uint64_t>(value),
                  reinterpret_cast<uintptr_t>(dst), byte_count);
}

// RELEASE_MEM - end-of-pipe fence with cache management.
//
// Packet layout (8 dwords on GFX9+ / AI and RDNA / NV):
//   [0]  header       - IT_RELEASE_MEM, count = 6
//   [1]  ordinal2     - event type, cache actions (ISA-generation-specific)
//   [2]  ordinal3     - dst_sel, int_sel, data_sel
//   [3]  addr_lo      - fence byte address bits [31:2], right-shifted by 2
//   [4]  addr_hi      - high 32 bits of fence address
//   [5]  data_lo      - low 32 bits of fence value (or 0 for timestamps)
//   [6]  data_hi      - high 32 bits
//   [7]  int_ctxid    - interrupt context (0 when polling)
//
// ordinal2 varies by ISA generation:
//
// GFX9 ordinal2 bit layout (amdgpu/soc15d.h):
//   event_type [5:0]          = 0x14
//   event_index [11:8]        = 5
//   EOP_TCL1_VOL_ACTION [12]  - TCL1 volatile writeback
//   EOP_TC_VOL_ACTION [13]    - TC/L2 volatile writeback
//   EOP_TC_WB_ACTION [15]     - L2 writeback
//   EOP_TCL1_ACTION [16]      - L1 invalidate
//   EOP_TC_ACTION [17]        - L2 invalidate
//   EOP_TC_NC_ACTION [19]     - Non-coherent writeback
//   EOP_TC_MD_ACTION [21]     - L2 metadata writeback
//   cache_policy [26:25]      - (not used in kernel fence path)
//   EOP_EXEC [28]             - Trailing fence execute
//
// GFX10+ ordinal2 bit layout (amdgpu/nvd.h):
//   event_type [5:0]          = 0x14
//   event_index [11:8]        = 5
//   GCR_GLM_WB [12]           - GL metadata writeback
//   GCR_GLM_INV [13]          - GL metadata invalidate (must pair with GLM_WB)
//   GCR_GLV_INV [14]          - GL V-cache invalidate
//   GCR_GL1_INV [15]          - GL1 invalidate
//   GCR_GL2_US [16]           - GL2 upstream coherence
//   GCR_GL2_RANGE [17]        - GL2 range mode
//   GCR_GL2_DISCARD [19]      - GL2 discard
//   GCR_GL2_INV [20]          - GL2 invalidate
//   GCR_GL2_WB [21]           - GL2 writeback
//   GCR_SEQ [22]              - Sequential ordering guarantee
//   cache_policy [26:25]      - L2 allocation for the fence write itself
//   EXECUTE [28]              - Trailing fence execute
//
// ordinal3:
//   dst_sel [17:16]  - 0 = memory controller, 1 = TC/L2
//   int_sel [26:24]  - interrupt and write-confirm behavior
//   data_sel [31:29] - what data to write at the fence address
//
// References: amdgpu/soc15d.h; amdgpu/nvd.h (PACKET3_RELEASE_MEM);
//             amdkfd/kfd_pm4_headers_vi.h (ordinal2/ordinal3 bitfields);
//             gfx_v9_0_ring_emit_fence; gfx_v10_0_ring_emit_fence

// Data selection for RELEASE_MEM ordinal3[31:29].
enum DataSel : uint32_t {
  DATA_NONE = 0,      // No data write
  DATA_32 = 1,        // Write low 32 bits of the value field
  DATA_64 = 2,        // Write 64-bit value
  DATA_TIMESTAMP = 3, // Write GPU clock counter (64-bit timestamp)
  DATA_PERFCOUNT = 4, // Write CP performance counter pair
  DATA_GDS = 5,       // Store GDS data to memory
};

// Interrupt selection for RELEASE_MEM ordinal3[26:24].
enum IntSel : uint32_t {
  INT_NONE = 0,         // No interrupt
  INT_ONLY = 1,         // Interrupt only (data_sel must be NONE)
  INT_DATA_CONFIRM = 2, // Interrupt after data write confirmed
  DATA_CONFIRM = 3,     // Data write confirmed, no interrupt
};

inline constexpr uint32_t RELEASE_MEM_DWORDS = 8;

// Standard fence cache flush.
//
// GFX9: writeback + L1/L2 invalidation + metadata.
// GFX10+: writeback + metadata + GCR_SEQ + BYPASS (no L1/L2 invalidation;
//         BYPASS policy ensures the fence write goes directly to the memory
//         controller, so cache invalidation is unnecessary for ordering).
inline uint32_t eop_fence_flush(uint32_t gfx_version) {
  uint32_t dw = 0x14 | (5u << 8);
  if (gfx_version >= abi::GFX_VERSION_GFX10_1) {
    dw |= (1u << 12)    // GCR_GLM_WB
          | (1u << 13)  // GCR_GLM_INV
          | (1u << 21)  // GCR_GL2_WB
          | (1u << 22)  // GCR_SEQ
          | (3u << 25); // CACHE_POLICY = BYPASS
  } else {
    dw |= (1u << 15)    // EOP_TC_WB_ACTION_EN (L2 writeback)
          | (1u << 16)  // EOP_TCL1_ACTION_EN  (L1 invalidate)
          | (1u << 17)  // EOP_TC_ACTION_EN    (L2 invalidate)
          | (1u << 21)  // EOP_TC_MD_ACTION_EN (L2 metadata)
          | (3u << 25); // CACHE_POLICY = BYPASS (Write skips L2)
  }
  return dw;
}

// Writeback-only flush: commit dirty data to memory without invalidating
// cache lines. Caches remain warm for subsequent reads.
//
// GFX9 uses the kernel's AMDGPU_FENCE_FLAG_TC_WB_ONLY path (TC_NC + TC_WB).
// GFX10+ drops GLM_INV since metadata invalidation is unnecessary when only
// flushing dirty data (GLM_WB is still set for metadata writeback).
inline uint32_t eop_wb_flush(uint32_t gfx_version) {
  uint32_t dw = 0x14 | (5u << 8);
  if (gfx_version >= abi::GFX_VERSION_GFX10_1) {
    dw |= (1u << 12)    // GCR_GLM_WB
          | (1u << 21)  // GCR_GL2_WB
          | (1u << 22)  // GCR_SEQ
          | (3u << 25); // CACHE_POLICY = BYPASS
  } else {
    dw |= (1u << 15)    // EOP_TC_WB_ACTION_EN
          | (1u << 19)  // EOP_TC_NC_ACTION_EN (non-coherent writeback)
          | (3u << 25); // CACHE_POLICY = BYPASS
  }
  return dw;
}

// Flexible RELEASE_MEM builder. Takes a pre-built ordinal2 (from
// eop_fence_flush(), eop_wb_flush(), or hand-constructed) and explicit
// data/interrupt selection. All convenience wrappers below delegate here.
inline uint32_t release_mem(uint32_t *out, uint32_t ordinal2, DataSel data,
                            IntSel intr, void *addr = nullptr,
                            uint64_t value = 0, uint32_t int_ctxid = 0) {
  out[0] = header(RELEASE_MEM, 6);
  out[1] = ordinal2;
  out[2] =
      (static_cast<uint32_t>(intr) << 24) | (static_cast<uint32_t>(data) << 29);
  out[3] = detail::lo(reinterpret_cast<uintptr_t>(addr)) & ~0x7u;
  out[4] = detail::hi(reinterpret_cast<uintptr_t>(addr));
  out[5] = detail::lo(value);
  out[6] = detail::hi(value);
  out[7] = int_ctxid;
  return RELEASE_MEM_DWORDS;
}

// Fence write, full flush, no interrupt.
inline uint32_t release_mem(uint32_t *out, uint32_t gfx_version,
                            void *fence_addr, uint64_t fence_value) {
  return release_mem(out, eop_fence_flush(gfx_version), DATA_64, INT_NONE,
                     fence_addr, fence_value);
}

// Fence write + interrupt routed to a KFD event via int_ctxid.
inline uint32_t release_mem(uint32_t *out, uint32_t gfx_version,
                            void *fence_addr, uint32_t fence_value,
                            uint32_t int_ctxid) {
  return release_mem(out, eop_fence_flush(gfx_version), DATA_64,
                     INT_DATA_CONFIRM, fence_addr,
                     static_cast<uint64_t>(fence_value), int_ctxid);
}

// Flush-only, no data write or interrupt.
inline uint32_t release_mem(uint32_t *out, uint32_t gfx_version) {
  return release_mem(out, eop_fence_flush(gfx_version), DATA_NONE, INT_NONE);
}

// ACQUIRE_MEM - invalidate / writeback selected GPU caches.
//
// The packet layout differs between GFX9 (7 dwords, cache actions in
// COHER_CNTL) and GFX10+ (8 dwords, cache actions in GCR_CNTL). Callers
// specify an abstract bitmask of AcquireMemFlags; the function translates
// these into the generation-specific hardware bits.
//
// COHER_SIZE/COHER_BASE are set to the maximum range (full flush).
// Ranged invalidation is possible by setting specific COHER_BASE/SIZE
// and GL2_RANGE modes on GFX10+.
//
// GFX9 COHER_CNTL bits (DW1):
//   [3]   TC_NC_ACTION_ENA          Non-coherent TC action
//   [4]   TC_WC_ACTION_ENA          Write-combine TC action
//   [5]   TC_INV_METADATA_ACTION    TC metadata invalidate
//   [15]  TCL1_VOL_ACTION_ENA       L1 volatile action
//   [18]  TC_WB_ACTION_ENA          L2 writeback
//   [22]  TCL1_ACTION_ENA           L1 invalidate
//   [23]  TC_ACTION_ENA             L2 invalidate
//   [27]  SH_KCACHE_ACTION_ENA      K-cache invalidate
//   [28]  SH_KCACHE_VOL_ACTION_ENA  K-cache volatile action
//   [29]  SH_ICACHE_ACTION_ENA      I-cache invalidate
//   [30]  SH_KCACHE_WB_ACTION_ENA   K-cache writeback
//   [31]  ENGINE_SEL                0 = ME (compute)
//
// GFX10+ GCR_CNTL bits (DW7):
//   [1:0] GLI_INV   GL0 invalidate (0=NOP, 1=ALL, 2=RANGE, 3=FIRST_LAST)
//   [3:2] GL1_RANGE GL1 range mode
//   [4]   GLM_WB    GL metadata writeback
//   [5]   GLM_INV   GL metadata invalidate
//   [6]   GLK_WB    GL K-cache writeback
//   [7]   GLK_INV   GL K-cache invalidate
//   [8]   GLV_INV   GL V-cache invalidate
//   [9]   GL1_INV   GL1 invalidate
//   [10]  GL2_US    GL2 upstream coherence
//   [12:11] GL2_RANGE  GL2 range mode (0=ALL, 1=VOL, 2=RANGE, 3=FIRST_LAST)
//   [13]  GL2_DISCARD GL2 discard
//   [14]  GL2_INV   GL2 invalidate
//   [15]  GL2_WB    GL2 writeback
//   [17:16] SEQ     Ordering (0=PARALLEL, 1=FORWARD, 2=REVERSE)
//   [18]  GCR_RANGE_IS_PA  Range address is physical
//
// POLL_INTERVAL (DW6) controls how often the CP checks for completion;
// the kernel driver uses 0xA (10 cycles) on all generations.
//
// References: amdgpu/soc15d.h; amdgpu/nvd.h;
//             gfx_v9_0_emit_mem_sync; gfx_v10_0_emit_mem_sync
enum AcquireMemFlags : uint32_t {
  ACQ_KCACHE = 1u << 0, // Scalar L1 (K-cache / GLK)
  ACQ_ICACHE = 1u << 1, // Instruction cache
  ACQ_VCACHE = 1u << 2, // Vector L1/L0 (TCL1 / GLV + GL1)
  ACQ_L2_INV = 1u << 3, // L2 / GL2 invalidate
  ACQ_L2_WB = 1u << 4,  // L2 / GL2 writeback
  ACQ_META = 1u << 5,   // Metadata (GL metadata WB + INV)

  ACQ_ALL =
      ACQ_KCACHE | ACQ_ICACHE | ACQ_VCACHE | ACQ_L2_INV | ACQ_L2_WB | ACQ_META,
};

inline constexpr AcquireMemFlags operator|(AcquireMemFlags a,
                                           AcquireMemFlags b) {
  return static_cast<AcquireMemFlags>(static_cast<uint32_t>(a) |
                                      static_cast<uint32_t>(b));
}

inline constexpr uint32_t ACQUIRE_MEM_DWORDS = 8;

inline uint32_t acquire_mem(uint32_t *out, uint32_t gfx_target_version,
                            AcquireMemFlags flags = ACQ_ALL) {
  bool gfx10_plus = gfx_target_version >= abi::GFX_VERSION_GFX10_1;
  uint32_t dwords = gfx10_plus ? 8u : 7u;

  for (uint32_t i = 0; i < dwords; ++i)
    out[i] = 0;

  out[0] = header(ACQUIRE_MEM, static_cast<uint16_t>(dwords - 2));

  if (!gfx10_plus) {
    uint32_t cntl = 0;
    if (flags & ACQ_KCACHE)
      cntl |= (1u << 27); // SH_KCACHE_ACTION_ENA
    if (flags & ACQ_ICACHE)
      cntl |= (1u << 29); // SH_ICACHE_ACTION_ENA
    if (flags & ACQ_VCACHE)
      cntl |= (1u << 22); // TCL1_ACTION_ENA
    if (flags & ACQ_L2_INV)
      cntl |= (1u << 23); // TC_ACTION_ENA
    if (flags & ACQ_L2_WB)
      cntl |= (1u << 18); // TC_WB_ACTION_ENA
    out[1] = cntl;
  } else {
    uint32_t gcr = 0;
    if (flags & ACQ_KCACHE)
      gcr |= (1u << 7); // GLK_INV
    if (flags & ACQ_VCACHE)
      gcr |= (1u << 0)                // GLI_INV = ALL
             | (1u << 8) | (1u << 9); // GLV_INV + GL1_INV
    if (flags & ACQ_L2_INV)
      gcr |= (1u << 14); // GL2_INV
    if (flags & ACQ_L2_WB)
      gcr |= (1u << 15); // GL2_WB
    if (flags & ACQ_META)
      gcr |= (1u << 4) | (1u << 5); // GLM_WB + GLM_INV
    out[7] = gcr;
  }

  out[2] = 0xFFFFFFFF; // COHER_SIZE    (full flush)
  out[3] = 0x00FFFFFF; // COHER_SIZE_HI (full flush)
  out[6] = 0x0000000A; // POLL_INTERVAL

  return dwords;
}

// WAIT_REG_MEM - stall the CP until a memory value satisfies a condition.
//
// Packet layout (7 dwords):
//   [0]  header         - IT_WAIT_REG_MEM, count = 5
//   [1]  control        - function[2:0], mem_space[5:4], operation[7:6],
//                          engine[8] (GFX only, reserved on MEC),
//                          cache_policy[26:25]
//   [2]  addr_lo        - byte address bits [31:2] (dword-aligned)
//   [3]  addr_hi        - byte address bits [63:32]
//   [4]  value          - 32-bit comparison value
//   [5]  mask           - applied to memory value before comparison
//   [6]  poll_interval[15:0], optimize_ace_offload_mode[31]
//
// The CP polls: (memory[addr] & mask) <cond> value
//
// mem_space=1 selects memory (vs register). operation=0 selects wait
// (vs write-wait-write). GFX10+ optimize_ace_offload_mode offloads
// the polling loop from the CP micro-engine, freeing it for other work.
//
// References: IT_WAIT_REG_MEM in amdkfd/kfd_pm4_opcodes.h;
//             PACKET3_WAIT_REG_MEM in amdgpu/soc15d.h;
//             gfx_v10_0_ring_emit_pipeline_sync (poll_interval = 4)
inline constexpr uint32_t WAIT_REG_MEM_DWORDS = 7;

inline uint32_t wait_reg_mem(uint32_t *out, uint32_t gfx_version, void *addr,
                             Condition cond, uint32_t value,
                             uint32_t mask = 0xFFFFFFFF, bool optimize = true) {
  out[0] = header(WAIT_REG_MEM, 5);
  out[1] = static_cast<uint32_t>(cond) | (1u << 4); // mem_space = memory
  out[2] = detail::lo(reinterpret_cast<uintptr_t>(addr)) & ~0x3u;
  out[3] = detail::hi(reinterpret_cast<uintptr_t>(addr));
  out[4] = value;
  out[5] = mask;
  out[6] = 0x04; // poll_interval = 4 cycles (matches kernel)
  if (optimize && gfx_version >= abi::GFX_VERSION_GFX10_1)
    out[6] |= (1u << 31); // optimize_ace_offload_mode
  return WAIT_REG_MEM_DWORDS;
}

// ATOMIC_MEM - perform an atomic operation on a memory location.
//
// Packet layout (9 dwords):
//   [0]  header       - IT_ATOMIC_MEM, count = 7
//   [1]  control      - atomic[6:0] (TC_OP), command[11:8], cache_policy[26:25]
//   [2]  addr_lo      - byte address bits [31:3] (qword-aligned)
//   [3]  addr_hi      - byte address bits [63:32]
//   [4]  src_data_lo  - operand low 32 bits
//   [5]  src_data_hi  - operand high 32 bits
//   [6]  cmp_data_lo  - compare value low (for CMPSWAP operations)
//   [7]  cmp_data_hi  - compare value high (for CMPSWAP operations)
//   [8]  loop_interval[12:0] - cycles between retries (for loop mode)
//
// The atomic[6:0] field is a TC_OP / GL2_OP opcode. The soc15d.h macro masks
// to 6 bits (0x3F) while nvd.h uses the full 7 bits (0x7F); opcodes >= 0x40
// require the wider mask. We always use 0x7F.
//
// This is a mid-pipe operation - it does NOT wait for outstanding shader
// work. Pair with a preceding RELEASE_MEM + WAIT_REG_MEM if ordering after
// shader completion is required.
//
// References: PACKET3_ATOMIC_MEM in amdgpu/soc15d.h, nvd.h;
//             TC_OP enum in rocr-runtime/libhsakmt/include/impl/pm4_cmds.h;
//             amdkfd/kfd_pm4_headers_ai.h

// Atomic operations (TC_OP / GL2_OP values).
//
// RTN variants (0x00-0x3F) return the old value to the CP for consumption by
// COPY_DATA. Non-RTN variants (0x40+) do not return the old value. Prefer RTN
// variants with WAIT_CONFIRM for fence writes, some GFX9 hardware silently
// drops non-RTN atomics targeting uncached system memory.
//
// References: TC_OP enum in pm4_cmds.h; GL2_OP in navi10_enum.h
enum AtomicOp : uint32_t {
  ATOMIC_SWAP_RTN_32 = 0x07,
  ATOMIC_CMPSWAP_RTN_32 = 0x08,
  ATOMIC_ADD_RTN_32 = 0x0F,
  ATOMIC_SWAP_RTN_64 = 0x27,
  ATOMIC_CMPSWAP_RTN_64 = 0x28,
  ATOMIC_ADD_RTN_64 = 0x2F,
  ATOMIC_SWAP_32 = 0x47,
  ATOMIC_CMPSWAP_32 = 0x48,
  ATOMIC_ADD_32 = 0x4F,
  ATOMIC_SWAP_64 = 0x67,
  ATOMIC_ADD_64 = 0x6F,
};

// ATOMIC_MEM command field [11:8].
//
// Only ATOMIC_SINGLE_PASS works reliably across all ISAs. The GFX9 series can
// stall forever on write confirm.
//
// References: MEC_ATOMIC_MEM_command_enum in pm4_cmds.h;
//             PACKET3_ATOMIC_MEM__COMMAND__* in amdgpu/nvd.h
enum AtomicCommand : uint32_t {
  ATOMIC_SINGLE_PASS = 0,   // Single-pass atomic, no retry
  ATOMIC_LOOP = 1,          // Loop until compare satisfied (CMPSWAP)
  ATOMIC_WAIT_CONFIRM = 2,  // Wait for write to reach memory before proceeding
  ATOMIC_SEND_CONTINUE = 3, // Fire and forget (CP does not wait)
};

inline constexpr uint32_t ATOMIC_MEM_DWORDS = 9;

inline uint32_t atomic_mem(uint32_t *out, AtomicOp op, void *addr,
                           int64_t src_data, int64_t cmp_data = 0,
                           AtomicCommand cmd = ATOMIC_WAIT_CONFIRM,
                           CachePolicy policy = POLICY_LRU) {
  out[0] = header(ATOMIC_MEM, 7);
  out[1] = (static_cast<uint32_t>(op) & 0x7Fu) |
           (static_cast<uint32_t>(cmd) << 8) |
           (static_cast<uint32_t>(policy) << 25);
  out[2] = detail::lo(reinterpret_cast<uintptr_t>(addr)) & ~0x7u;
  out[3] = detail::hi(reinterpret_cast<uintptr_t>(addr));
  out[4] = detail::lo(static_cast<uint64_t>(src_data));
  out[5] = detail::hi(static_cast<uint64_t>(src_data));
  out[6] = detail::lo(static_cast<uint64_t>(cmp_data));
  out[7] = detail::hi(static_cast<uint64_t>(cmp_data));
  out[8] = 0;
  return ATOMIC_MEM_DWORDS;
}

// SET_SH_REG - write one or more consecutive SH-mapped registers.
//
// Packet layout (2 + N dwords):
//   [0]  header     - IT_SET_SH_REG, count = 1 + N
//   [1]  reg_offset - (register_addr - SH_BASE)
//   [2..N+1]        - register values
//
// The register offset is relative to the SH base address (0x2C00).
//
// References: IT_SET_SH_REG in amdkfd/kfd_pm4_opcodes.h
inline uint32_t set_sh_reg(uint32_t *out, uint32_t reg_addr,
                           const uint32_t *values, uint32_t count) {
  uint32_t dwords = 2 + count;
  out[0] = header(SET_SH_REG, static_cast<uint16_t>(dwords - 2));
  out[1] = reg_addr - regs::SH_BASE;
  for (uint32_t i = 0; i < count; ++i)
    out[2 + i] = values[i];
  return dwords;
}

inline uint32_t set_sh_reg(uint32_t *out, uint32_t reg_addr, uint32_t value) {
  return set_sh_reg(out, reg_addr, &value, 1);
}

// DISPATCH_DIRECT - launch a compute grid.
//
// Packet layout (5 dwords):
//   [0]  header              - IT_DISPATCH_DIRECT, count = 3
//   [1]  dim_x               - grid dimension X
//   [2]  dim_y               - grid dimension Y
//   [3]  dim_z               - grid dimension Z
//   [4]  dispatch_initiator  - bitfield controlling launch behavior
//
// The meaning of dim_x/y/z depends on DISPATCH_USE_THREAD_DIMENSIONS in the
// initiator. When set, they are total thread counts and the hardware divides
// by COMPUTE_NUM_THREAD_X/Y/Z to derive the block count. When clear (our
// default), they are work-group counts directly.
//
// References: IT_DISPATCH_DIRECT in amdkfd/kfd_pm4_opcodes.h
inline constexpr uint32_t DISPATCH_DIRECT_DWORDS = 5;

inline uint32_t dispatch_direct(uint32_t *out, uint32_t dim_x, uint32_t dim_y,
                                uint32_t dim_z, uint32_t initiator) {
  out[0] = header(DISPATCH_DIRECT, 3);
  out[1] = dim_x;
  out[2] = dim_y;
  out[3] = dim_z;
  out[4] = initiator;
  return DISPATCH_DIRECT_DWORDS;
}

// INDIRECT_BUFFER - redirect the CP to execute packets from a separate buffer.
//
// Packet layout (4 dwords):
//   [0]  header     - IT_INDIRECT_BUFFER, count = 2
//   [1]  ib_base_lo - IB byte address bits [31:0] (must be dword-aligned,
//                      bits [1:0] = 0)
//   [2]  ib_base_hi - IB byte address bits [63:32]
//   [3]  control    - ib_size[19:0], chain[20], offload_polling[21],
//                      valid[23], vmid[27:24], cache_policy[29:28], priv[31]
//
// The CP fetches and executes the packets stored at ib_base for ib_size
// dwords, then returns to the next packet in the ring. The IB buffer must
// reside in GPU-visible memory (mapped via the queue's VMID).
//
// Control word bitfields:
//   [19:0]  ib_size        - number of dwords in the IB (max 0xFFFFF)
//   [20]    chain          - 1 = chain to a subsequent IB (do not return to
//                             the ring after execution; instead jump to the
//                             IB referenced by the last INDIRECT_BUFFER in
//                             the current IB). 0 = return to the ring.
//   [21]    offload_polling - 1 = offload polling from the CP micro-engine
//   [23]    valid          - must be 1 on GFX9-11; marks the IB as valid
//   [27:24] vmid           - virtual memory context ID (0 for KFD userspace
//                             queues; the kernel assigns the VMID at queue
//                             creation)
//   [29:28] cache_policy   - L2 fetch policy for IB reads (see CachePolicy)
//   [31]    priv           - privileged IB (kernel only, must be 0 from
//                             userspace)
//
// IB_SIZE is a 20-bit field, so the maximum indirect buffer is 2^20 - 1 =
// 1,048,575 dwords (~4 MiB). The buffer must contain only complete PM4
// packets; a partial packet at the end is undefined behavior.
//
// References: PACKET3_INDIRECT_BUFFER in amdgpu/soc15d.h, amdgpu/nvd.h;
//             IT_INDIRECT_BUFFER in amdkfd/kfd_pm4_opcodes.h;
//             gfx_v9_0_ring_emit_ib_compute; gfx_v10_0_ring_emit_ib_compute;
//             gfx_v11_0_ring_emit_ib_compute; gfx_v12_0_ring_emit_ib_compute
inline constexpr uint32_t INDIRECT_BUFFER_DWORDS = 4;
inline constexpr uint32_t INDIRECT_BUFFER_MAX_SIZE_DWORDS = 0xFFFFF;

inline uint32_t indirect_buffer(uint32_t *out, const void *ib_addr,
                                uint32_t ib_size_dwords,
                                CachePolicy policy = POLICY_LRU) {
  out[0] = header(INDIRECT_BUFFER, 2);
  out[1] = detail::lo(reinterpret_cast<uintptr_t>(ib_addr));
  out[2] = detail::hi(reinterpret_cast<uintptr_t>(ib_addr));
  out[3] = (ib_size_dwords & 0xFFFFFu) | (1u << 23) // VALID
           | (static_cast<uint32_t>(policy) << 28);
  return INDIRECT_BUFFER_DWORDS;
}

// Build the 4-word V# buffer resource descriptor (SRD) for the private
// segment buffer.
inline void build_scratch_srd(uint32_t *v, uint32_t gfx_version,
                              const void *scratch_base, uint32_t tmpring_size,
                              uint16_t props) {
  uint64_t base = reinterpret_cast<uintptr_t>(scratch_base);
  uint32_t waves = tmpring_size & 0xFFFu;
  uint32_t wavesize = tmpring_size >> 12;
  constexpr uint32_t DST_SEL = (4u << 0) | (5u << 3) | (6u << 6) | (7u << 9);
  v[0] = detail::lo(base);
  v[1] = (detail::hi(base) & 0xFFFFu) | (1u << 31);
  v[2] = waves * wavesize * detail::scratch_alignment_unit(gfx_version);
  if (gfx_version >= abi::GFX_VERSION_GFX10_1) {
    bool wave32 = props & abi::ENABLE_WAVEFRONT_SIZE32;
    v[3] = DST_SEL | (0x14u << 12) | ((wave32 ? 2u : 3u) << 21) | (1u << 23) |
           (1u << 24) | (2u << 28);
  } else {
    v[3] = DST_SEL | (4u << 12) | (4u << 15) | (1u << 19) | (3u << 21) |
           (1u << 23);
  }
}

// Write SET_SH_REG packets for COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI and
// COMPUTE_TMPRING_SIZE. These tell the SPI where the scratch backing store
// lives and how it is partitioned across waves.
inline uint32_t set_scratch_base(uint32_t *out, uint32_t gfx_version,
                                 const void *scratch_base,
                                 uint32_t tmpring_size) {
  uint32_t n = 0;
  if (gfx_version >= abi::GFX_VERSION_GFX9) {
    uint64_t shifted = reinterpret_cast<uintptr_t>(scratch_base) >> 8;
    const uint32_t vals[] = {detail::lo(shifted), detail::hi(shifted)};
    n += set_sh_reg(out + n, regs::COMPUTE_DISPATCH_SCRATCH_BASE_LO, vals, 2);
  }
  n += set_sh_reg(out + n, regs::COMPUTE_TMPRING_SIZE, tmpring_size);
  return n;
}

// Compute the DISPATCH_DIRECT initiator bits from a kernel descriptor.
inline uint32_t dispatch_initiator(const abi::KernelDescriptor &kd,
                                   uint32_t gfx_version) {
  uint32_t init =
      regs::DISPATCH_COMPUTE_SHADER_EN | regs::DISPATCH_FORCE_START_AT_000;
  if ((kd.kernel_code_properties & abi::ENABLE_WAVEFRONT_SIZE32) &&
      gfx_version >= abi::GFX_VERSION_GFX10_1)
    init |= regs::DISPATCH_CS_W32_EN;
  return init;
}

inline constexpr uint32_t MAX_DISPATCH_DWORDS = 140;

// Encode the register-programming sequence for a compute dispatch into 'out'.
// The resulting buffer can be invoked through a DISPATCH_DIRECT packet.
//
// 'grid' is the number of work-groups per dimension; 'block' is threads per
// work-group. DISPATCH_DIRECT receives work-group counts directly (we do not
// set USE_THREAD_DIMENSIONS).
//
// 'dispatch_pkt_addr', when non-zero, is written to the DISPATCH_PTR user
// SGPR pair. It must point to a filled abi::DispatchPacket (see
// abi::fill_implicit_args which places one after the implicit args).
//
// 'scratch_base' is the GPU VA of the scratch buffer (256B-aligned).
// 'tmpring_size' is the packed COMPUTE_TMPRING_SIZE register value
// (WAVES [11:0], WAVESIZE [24:12]+), as returned by compute_tmpring_size().
inline uint32_t build_dispatch_setup(
    uint32_t *out, uint32_t gfx_version, const abi::KernelDescriptor &kd,
    const void *entry_addr, Dim3 grid, Dim3 block,
    const void *kernarg_addr = nullptr, const void *dispatch_pkt_addr = nullptr,
    const void *scratch_base = nullptr, uint32_t tmpring_size = 0,
    uint32_t dynamic_lds = 0, uint32_t private_segment_size = 0) {
  uint32_t written = 0;

  const uint32_t dims[] = {
      0,       0,       0,       // COMPUTE_START_X/Y/Z
      block.x, block.y, block.z, // COMPUTE_NUM_THREAD_X/Y/Z
      0,       0,                // PIPELINESTAT_ENABLE, PERFCOUNT_ENABLE
  };
  written += set_sh_reg(out + written, regs::COMPUTE_START_X, dims, 8);

  // Program address. SCRATCH_BASE_LO/HI is set separately via
  // set_scratch_base so the watcher IB can override it after a resize.
  uint64_t entry_shifted = reinterpret_cast<uintptr_t>(entry_addr) >> 8;
  if (gfx_version >= abi::GFX_VERSION_GFX9) {
    const uint32_t pgm[] = {detail::lo(entry_shifted),
                            detail::hi(entry_shifted), 0, 0};
    written += set_sh_reg(out + written, regs::COMPUTE_PGM_LO, pgm, 4);
  } else {
    const uint32_t pgm[] = {detail::lo(entry_shifted),
                            detail::hi(entry_shifted)};
    written += set_sh_reg(out + written, regs::COMPUTE_PGM_LO, pgm, 2);
  }

  uint32_t lds_bytes = kd.group_segment_fixed_size + dynamic_lds;
  uint32_t lds_blocks = (detail::align_up(lds_bytes, 512u)) / 512u;
  uint32_t rsrc1 = kd.compute_pgm_rsrc1;
  if (abi::needs_cwsr_priv_wa(gfx_version))
    rsrc1 |= abi::COMPUTE_PGM_RSRC1_PRIV;
  uint32_t rsrc2 = kd.compute_pgm_rsrc2 | (lds_blocks << 15);
  const uint32_t rsrc[] = {rsrc1, rsrc2};
  written += set_sh_reg(out + written, regs::COMPUTE_PGM_RSRC1, rsrc, 2);

  written +=
      set_sh_reg(out + written, regs::COMPUTE_PGM_RSRC3, kd.compute_pgm_rsrc3);

  // Occupancy limits and SIMD masks. TMPRING_SIZE is set separately via
  // set_scratch_base so the watcher IB can override it after a resize.
  const uint32_t res1[] = {0, 0xFFFFFFFF, 0xFFFFFFFF};
  written += set_sh_reg(out + written, regs::COMPUTE_RESOURCE_LIMITS, res1, 3);
  const uint32_t res2[] = {0xFFFFFFFF, 0xFFFFFFFF};
  written +=
      set_sh_reg(out + written, regs::COMPUTE_STATIC_THREAD_MGMT_SE2, res2, 2);

  const uint32_t restart[] = {0, 0, 0, 0};
  written += set_sh_reg(out + written, regs::COMPUTE_RESTART_X, restart, 4);

  // User SGPRs are packed in ABI-mandated order from kernel_code_properties.
  // See the ordering table in abi.h for the slot layout.
  uint32_t user_sgpr[16] = {};
  uint32_t i = 0;
  uint16_t props = kd.kernel_code_properties;

  if (props & abi::ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) {
    build_scratch_srd(user_sgpr, gfx_version, scratch_base, tmpring_size,
                      props);
    i += 4;
  }
  if (props & abi::ENABLE_SGPR_DISPATCH_PTR) {
    user_sgpr[i++] = detail::lo(dispatch_pkt_addr);
    user_sgpr[i++] = detail::hi(dispatch_pkt_addr);
  }
  if (props & abi::ENABLE_SGPR_QUEUE_PTR)
    i += 2; // The amd_queue_t pointer, unused for PM4.
  if (props & abi::ENABLE_SGPR_KERNARG_SEGMENT_PTR) {
    user_sgpr[i++] = detail::lo(kernarg_addr);
    user_sgpr[i++] = detail::hi(kernarg_addr);
  }
  if (props & abi::ENABLE_SGPR_DISPATCH_ID)
    i += 2;
  if (props & abi::ENABLE_SGPR_FLAT_SCRATCH_INIT) {
    // GFX9-10 only (GFX11+ uses architected flat scratch and never sets
    // this bit). The shader prologue adds this 64-bit byte address to
    // the per-wave Scratch Wave Offset (system SGPR from SPI) and writes
    // the result to FLAT_SCRATCH: GFX9 writes FLAT_SCR_LO/HI directly,
    // GFX10 uses S_SETREG_B32 into FLAT_SCR HW regs.
    //
    // Note: COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI is a separate register
    // storing the address >> 8. The SPI reads that to compute the
    // per-wave byte offset, but this user SGPR is the unshifted address.
    //
    // References: SIFrameLowering.cpp (flatScratchIsPointer path);
    //             LLVM docs/AMDGPUUsage.rst "Absolute flat scratch"
    user_sgpr[i++] = detail::lo(scratch_base);
    user_sgpr[i++] = detail::hi(scratch_base);
  }
  if (props & abi::ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
    user_sgpr[i++] = private_segment_size ? private_segment_size
                                          : kd.private_segment_fixed_size;

  if (i > 0)
    written +=
        set_sh_reg(out + written, regs::COMPUTE_USER_DATA_0, user_sgpr, i);

  // Scratch base + tmpring via the shared helper. When a scratch resize is
  // pending, the watcher IB overrides these (plus the scratch-dependent
  // SGPRs) with fresh values after the WAIT_REG_MEM stall.
  written +=
      set_scratch_base(out + written, gfx_version, scratch_base, tmpring_size);

  return written;
}

} // namespace kfd::pm4

#endif // LIBKFD_PACKETS_PM4_H
