//===-- libkfd/abi.h - AMDHSA compute ABI types -----------------*- C++ -*-===//
//
// The AMDGPU compute platform uses HSA as its ABI. This header defines the
// kernel descriptor, implicit argument layout, and related constants that
// form the contract between the compiler and the runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_ABI_H
#define LIBKFD_ABI_H

#include "libkfd/detail/utility.h"
#include "libkfd/dispatch.h"
#include "libkfd/topology.h"

#include <cstdint>
#include <cstring>

namespace kfd::abi {

// Constants to determine the current ISA.
inline constexpr uint32_t GFX_VERSION_GFX9 = 90000;
inline constexpr uint32_t GFX_VERSION_GFX9_8 = 90008;
inline constexpr uint32_t GFX_VERSION_GFX9_A = 90010;
inline constexpr uint32_t GFX_VERSION_GFX942 = 90402;
inline constexpr uint32_t GFX_VERSION_GFX950 = 90500;
inline constexpr uint32_t GFX_VERSION_GFX10_1 = 100100;
inline constexpr uint32_t GFX_VERSION_GFX11 = 110000;
inline constexpr uint32_t GFX_VERSION_GFX11_1 = 110001;
inline constexpr uint32_t GFX_VERSION_GFX11_5 = 110005;
inline constexpr uint32_t GFX_VERSION_GFX1151 = 110501;
inline constexpr uint32_t GFX_VERSION_GFX12 = 120000;
inline constexpr uint32_t GFX_VERSION_GFX12_1 = 120001;
inline constexpr uint32_t GFX_VERSION_GFX1250 = 120500;

// Decompose a gfx_target_version into its components.
constexpr uint32_t gfx_version_major(uint32_t v) { return (v / 10000) % 100; }
constexpr uint32_t gfx_version_minor(uint32_t v) { return (v / 100) % 100; }
constexpr uint32_t gfx_version_step(uint32_t v) { return v % 100; }

// COMPUTE_PGM_RSRC1 bit 20: launch waves with STATUS.PRIV=1.
inline constexpr uint32_t COMPUTE_PGM_RSRC1_PRIV = 1u << 20;
// COMPUTE_PGM_RSRC1 bit 29: Pair two CUs into a single work-group-processor.
inline constexpr uint32_t COMPUTE_PGM_RSRC1_WGP = 1u << 29;

// COMPUTE_PGM_RSRC3 bit 13: group launch guarantee (GFX12+).
// When set, the SPI ensures all workgroups in the dispatch are co-resident.
inline constexpr uint32_t COMPUTE_PGM_RSRC3_GLG_EN = 1u << 13;

// GFX11.0.x has a HW defect that suppresses CWSR, under PRIV=1, s_trap 2 is
// interpreted as a NOP on these chips. However LLVM has a software work-around
// so we set this manually.
constexpr bool needs_cwsr_priv_wa(uint32_t gfx) {
  return gfx_version_major(gfx) == 11 && gfx_version_minor(gfx) == 0;
}

// Required alignment for the private scratch region.
constexpr size_t PRIVATE_SEGMENT_ALIGN = 0x10000;

// Kernel code properties bitfield masks.
//
// User SGPR enable bits. Each enabled feature consumes a fixed number of
// SGPR slots, packed contiguously starting at s0 in the order below.
// The runtime writes the corresponding values into COMPUTE_USER_DATA
// registers; the SPI loads them into s0-s15 at wave launch.
//
//   Bit  SGPRs  Contents
//   0    4      Scratch ring V# SRD (GFX6-10 only; must be 0 on GFX11+).
//   1    2      AQL dispatch packet pointer (required for HSA ABI).
//   2    2      amd_queue_t pointer (unused for PM4).
//   3    2      Kernarg segment GPU VA.
//   4    2      Dispatch ID (monotonic counter).
//   5    2      Flat scratch offset + size (GFX9-10 only; must be 0 on GFX11+).
//   6    1      private_segment_fixed_size from the KD.
//
// GFX11+ uses architected flat scratch: the hardware reads the scratch
// base directly from COMPUTE_DISPATCH_SCRATCH_BASE, so bits 0 and 5 are
// retired. The compiler will not set them for GFX11+ targets.
//
// Reference: AMDGPUUsage.rst "Kernel Code Properties"
//            llvm/include/llvm/Support/AMDHSAKernelDescriptor.h
inline constexpr uint16_t ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER = 1u << 0;
inline constexpr uint16_t ENABLE_SGPR_DISPATCH_PTR = 1u << 1;
inline constexpr uint16_t ENABLE_SGPR_QUEUE_PTR = 1u << 2;
inline constexpr uint16_t ENABLE_SGPR_KERNARG_SEGMENT_PTR = 1u << 3;
inline constexpr uint16_t ENABLE_SGPR_DISPATCH_ID = 1u << 4;
inline constexpr uint16_t ENABLE_SGPR_FLAT_SCRATCH_INIT = 1u << 5;
inline constexpr uint16_t ENABLE_SGPR_PRIVATE_SEGMENT_SIZE = 1u << 6;
inline constexpr uint16_t ENABLE_WAVEFRONT_SIZE32 = 1u << 10;
inline constexpr uint16_t USES_DYNAMIC_STACK = 1u << 11;

// Kernarg preload specification bitfield masks.
inline constexpr uint16_t KERNARG_PRELOAD_LENGTH_MASK = 0x007f;
inline constexpr unsigned KERNARG_PRELOAD_OFFSET_SHIFT = 7;
inline constexpr uint16_t KERNARG_PRELOAD_OFFSET_MASK = 0xff80;

// The kernarg preload support places arguments in the previously unused user
// SGPRs during kernel launch. If this feature is enabled then we can simply
// skip over the portion of the kernel call ABI that handled them. This fallback
// prologue was removed on GFX1250+ ISAs.
inline constexpr size_t KERNARG_PRELOAD_PROLOG_SIZE = 256;

// AMDHSA kernel descriptor (64 bytes, 64-byte aligned).
//
// For each kernel, the compiler emits a pair of ELF symbols:
//   <name>.kd  STT_OBJECT in .rodata  -- this descriptor
//   <name>     STT_FUNC   in .text    -- the entry instruction
//
// The runtime reads this structure to populate the AQL dispatch packet or its
// fields to pass directly to a PM4 packet.
//
// Reference: AMDGPUUsage.rst "Kernel Descriptor" Table
//            llvm/include/llvm/Support/AMDHSAKernelDescriptor.h
struct KernelDescriptor {
  // LDS bytes required per work-group, excluding dynamic LDS added at
  // dispatch via the AQL packet's group_segment_size field.
  uint32_t group_segment_fixed_size;

  // Scratch (private) bytes required per work-item. On code object v4 and
  // older the compiler may overestimate when the true minimum cannot be
  // statically determined.
  uint32_t private_segment_fixed_size;

  // Size in bytes of the kernarg segment. Zero means unspecified; non-zero
  // enables CP prefetch optimizations.
  uint32_t kernarg_size;

  uint8_t reserved0[4];

  // Signed byte offset from the start of this descriptor to the kernel
  // entry instruction. Target must be 256-byte aligned. When kernarg
  // preload is active (preload length != 0), the runtime (or CP on AQL)
  // adds KERNARG_PRELOAD_PROLOG_SIZE to skip the compatibility prolog.
  int64_t kernel_code_entry_byte_offset;

  uint8_t reserved1[20];

  // COMPUTE_PGM_RSRC3. Reserved (zero) on GFX6-9 except GFX90A/GFX942.
  //
  // GFX90A/GFX942:
  //   [5:0]   ACCUM_OFFSET    First AccVGPR in unified register file
  //                            (granularity 4; value N maps to offset 4*(N+1))
  //   [16]    TG_SPLIT        1 = waves of a work-group may split across CUs
  //                            (implies no S_BARRIER / no LDS)
  //
  // GFX10-GFX11:
  //   [3:0]   SHARED_VGPR_COUNT  Shared VGPR blocks for subvector mode
  //                               (WF64 only; must be 0 for WF32)
  //   [9:4]   INST_PREF_SIZE     GFX11 only: instruction prefetch hint
  //                               (0-63, 128-byte granularity)
  //   [31]    IMAGE_OP           GFX11 only: image operation hint
  //
  // GFX12:
  //   [11:4]  INST_PREF_SIZE     Instruction prefetch (0-255, 128B granule)
  //   [13]    GLG_EN             Group launch guarantee
  //   [31]    IMAGE_OP           Image operation hint
  uint32_t compute_pgm_rsrc3;

  // COMPUTE_PGM_RSRC1. Controls register allocation granularity,
  // floating-point modes, and per-generation execution flags.
  //
  //   [5:0]   GRANULATED_WORKITEM_VGPR_COUNT
  //             Encodes VGPR allocation; granularity is GPU-specific.
  //             GFX6-9:  max_vgpr = (N+1)*4 - 1
  //             GFX10+:  max_vgpr = (N+1)*8 - 1  (wave32)
  //                      max_vgpr = (N+1)*4 - 1  (wave64)
  //   [9:6]   GRANULATED_WAVEFRONT_SGPR_COUNT
  //             GFX6-9: encodes SGPR blocks. GFX10+: reserved (0).
  //   [11:10] PRIORITY             Must be 0; CP fills.
  //   [13:12] FLOAT_ROUND_MODE_32  IEEE 754 rounding for f32
  //   [15:14] FLOAT_ROUND_MODE_16_64  Rounding for f16/f64
  //   [17:16] FLOAT_DENORM_MODE_32    Denorm handling for f32
  //   [19:18] FLOAT_DENORM_MODE_16_64 Denorm handling for f16/f64
  //   [20]    PRIV                 Launch with STATUS.PRIV=1
  //   [21]    ENABLE_DX10_CLAMP    GFX6-11: DX10 NaN clamp
  //                                GFX12: WG_RR_EN (round-robin scheduling)
  //   [23]    ENABLE_IEEE_MODE     GFX6-11: IEEE mode
  //                                GFX12: reserved
  //   [26]    FP16_OVFL            GFX9+: FP16 overflow mode
  //   [29]    WGP_MODE             GFX10+: 1 = WGP mode, 0 = CU mode
  //   [30]    MEM_ORDERED          GFX10+: memory ordering semantics
  //   [31]    FWD_PROGRESS         GFX10+: forward progress guarantee
  uint32_t compute_pgm_rsrc1;

  // COMPUTE_PGM_RSRC2. Controls scratch, user SGPRs, work-group ID
  // enables, and exception masks.
  //
  //   [0]     ENABLE_PRIVATE_SEGMENT  Scratch enable
  //   [5:1]   USER_SGPR_COUNT         Number of user SGPRs (must be >= the
  //                                    count implied by kernel_code_properties
  //                                    enable bits). GFX125: 6 bits at [6:1].
  //   [6]     ENABLE_TRAP_HANDLER     GFX6-11: must be 0; CP sets at launch.
  //           ENABLE_DYNAMIC_VGPR     GFX12: enables dynamic VGPR allocation.
  //                                   These are DIFFERENT meanings for the same
  //                                   bit - do not set TRAP_PRESENT on GFX12.
  //   [7]     ENABLE_SGPR_WORKGROUP_ID_X
  //   [8]     ENABLE_SGPR_WORKGROUP_ID_Y
  //   [9]     ENABLE_SGPR_WORKGROUP_ID_Z
  //   [10]    ENABLE_SGPR_WORKGROUP_INFO
  //   [12:11] ENABLE_VGPR_WORKITEM_ID
  //             0 = X only, 1 = X+Y, 2 = X+Y+Z
  //   [23:15] GRANULATED_LDS_SIZE     Must be 0 on AQL; CP fills from dispatch.
  //   [30:24] Exception enables       One bit per IEEE 754 exception class
  //             [24] invalid operation
  //             [25] denormal source
  //             [26] division by zero
  //             [27] overflow
  //             [28] underflow
  //             [29] inexact
  //             [30] integer division by zero
  uint32_t compute_pgm_rsrc2;

  // Kernel code properties. Packed enable bits for user SGPR setup, wave
  // size, and dynamic stack.
  //
  //   [0]  ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER
  //          4 SGPRs; must be 0 with architected flat scratch.
  //   [1]  ENABLE_SGPR_DISPATCH_PTR            2 SGPRs
  //   [2]  ENABLE_SGPR_QUEUE_PTR               2 SGPRs
  //   [3]  ENABLE_SGPR_KERNARG_SEGMENT_PTR     2 SGPRs
  //   [4]  ENABLE_SGPR_DISPATCH_ID             2 SGPRs
  //   [5]  ENABLE_SGPR_FLAT_SCRATCH_INIT
  //          2 SGPRs; must be 0 with architected flat scratch.
  //   [6]  ENABLE_SGPR_PRIVATE_SEGMENT_SIZE    1 SGPR
  //   [9:7]   Reserved (0)
  //   [10] ENABLE_WAVEFRONT_SIZE32
  //          GFX10+: 0 = wave64, 1 = wave32. GFX6-9: reserved (0).
  //   [11] USES_DYNAMIC_STACK
  //          Code object v5+: kernel uses a dynamic call stack.
  //   [15:12] Reserved (0)
  //
  // Total user data SGPRs requested by bits [6:0] must be <= 16.
  uint16_t kernel_code_properties;

  // Kernarg preload specification. On when enabled the CP preloads
  // dwords from the kernarg segment into consecutive user SGPRs rather than
  // loading them from the kernel argument pointer. AQL only supports this with
  // compatible firmware, but PM4 can do whatever it wants.
  //
  //   [6:0]  KERNARG_PRELOAD_SPEC_LENGTH   Dwords to preload (0 = disabled)
  //   [15:7] KERNARG_PRELOAD_SPEC_OFFSET   Dword offset into kernarg segment
  uint16_t kernarg_preload;

  uint8_t reserved3[4];
};

static_assert(sizeof(KernelDescriptor) == 64);

// Implicit kernarg block (256 bytes, 8-byte aligned).
//
// The compiler appends this after each kernel's explicit arguments, starting at
// align_up(explicit_kernarg_size, 8) from the kernarg base. The runtime must
// fill these slots before dispatch. the hardware does not populate them and
// they are part of the compute ABI. The vast majority of these are unused.
//
// Reference: clang/lib/Headers/amdhsa_abi.h
struct ImplicitArgs {
  uint32_t block_count_x;
  uint32_t block_count_y;
  uint32_t block_count_z;

  uint16_t group_size_x;
  uint16_t group_size_y;
  uint16_t group_size_z;

  uint16_t remainder_x;
  uint16_t remainder_y;
  uint16_t remainder_z;

  uint64_t reserved0[2];

  uint64_t global_offset_x;
  uint64_t global_offset_y;
  uint64_t global_offset_z;

  uint16_t grid_dims;
  uint8_t reserved1[6];

  uint64_t printf_buffer;
  uint64_t hostcall_buffer;
  uint64_t multigrid_sync_arg;
  uint64_t heap_v1;
  uint64_t default_queue;
  uint64_t completion_action;

  uint32_t dynamic_lds_size;
  uint8_t reserved2[68];

  uint32_t private_base;
  uint32_t shared_base;

  uint64_t queue_ptr;

  uint8_t reserved3[48];
};

static_assert(sizeof(ImplicitArgs) == 256);

// Context save/restore area header (40 bytes).
//
// Placed at the start of the CWSR save area for each XCC. The kernel and trap
// handler read and write this structure during preemption and exception
// delivery. On a shader trap the handler writes EC_QUEUE_WAVE_* flags to
// err_payload_addr via MSG_INTERRUPT and KFD signals err_event_id.
//
// Reference: linux/kfd_ioctl.h (struct kfd_context_save_area_header)
struct CwsrHeader {
  // Offset to last saved control stack top.
  uint32_t control_stack_offset;
  // Saved control stack size in bytes.
  uint32_t control_stack_size;
  // Offset to last saved wave state base.
  uint32_t wave_state_offset;
  // Saved wave state size in bytes.
  uint32_t wave_state_size;
  // Offset to debugger memory (64B aligned).
  uint32_t debug_offset;
  // Debugger memory size in bytes (64B aligned).
  uint32_t debug_size;
  // GPU VA of the exception reason bitmask.
  uint64_t err_payload_addr;
  // KFD event ID signalled on exception.
  uint32_t err_event_id;
  uint32_t reserved1;
};

static_assert(sizeof(CwsrHeader) == 40);

// HSA Architected Queuing Language (AQL) kernel dispatch packet.
//
// In AQL mode the command processor reads this directly from the queue ring
// buffer. For PM4 dispatches the packet is not consumed by hardware, but
// kernels compiled with ENABLE_SGPR_DISPATCH_PTR may still read grid/block
// dimensions from it via the dispatch pointer SGPR pair.  For PM4 we embed the
// packet in the kernarg allocation (same lifetime, same GTT memory type) and
// pass its address through COMPUTE_USER_DATA.
//
// Reference: HSA Platform System Architecture Specification 1.2,
//            Section 2.8 "Kernel Dispatch Packet"
struct alignas(64) DispatchPacket {
  uint16_t header;
  uint16_t setup;
  uint16_t workgroup_size_x;
  uint16_t workgroup_size_y;
  uint16_t workgroup_size_z;
  uint16_t reserved0;
  uint32_t grid_size_x;
  uint32_t grid_size_y;
  uint32_t grid_size_z;
  uint32_t private_segment_size;
  uint32_t group_segment_size;
  uint64_t kernel_object;
  uint64_t kernarg_address;
  uint64_t reserved1;
  uint64_t completion_signal;
};

static_assert(sizeof(DispatchPacket) == 64);

// Fills the dispatch packet as an implicit argument.
inline void fill_dispatch_packet(DispatchPacket &pkt, Dim3 grid, Dim3 block,
                                 uint32_t private_size = 0,
                                 uint32_t group_size = 0) {
  std::memset(&pkt, 0, sizeof(pkt));
  uint16_t dims = static_cast<uint16_t>(1 + (grid.y > 1 || block.y > 1) +
                                        (grid.z > 1 || block.z > 1));
  pkt.setup = dims;
  pkt.workgroup_size_x = static_cast<uint16_t>(block.x);
  pkt.workgroup_size_y = static_cast<uint16_t>(block.y);
  pkt.workgroup_size_z = static_cast<uint16_t>(block.z);
  pkt.grid_size_x =
      static_cast<uint32_t>(static_cast<uint64_t>(block.x) * grid.x);
  pkt.grid_size_y =
      static_cast<uint32_t>(static_cast<uint64_t>(block.y) * grid.y);
  pkt.grid_size_z =
      static_cast<uint32_t>(static_cast<uint64_t>(block.z) * grid.z);
  pkt.private_segment_size = private_size;
  pkt.group_segment_size = group_size;
}

// Maximum concurrent blocks per physical CU for the given kernel and block
// size. Accounts for register usage, wave-slot limits, and LDS.
inline uint32_t blocks_per_cu(const NodeProperties &props, uint32_t gfx_version,
                              const KernelDescriptor &kd, Dim3 block,
                              uint32_t dynamic_lds = 0) {
  uint32_t simd_per_cu = props.simd_per_cu ? props.simd_per_cu : 1;
  uint32_t wave_size =
      (kd.kernel_code_properties & ENABLE_WAVEFRONT_SIZE32) ? 32 : 64;
  uint32_t max_waves = props.max_waves_per_simd ? props.max_waves_per_simd : 8;

  // VGPR occupancy, COMPUTE_PGM_RSRC1[5:0] encodes allocated VGPRs with a
  // generation-dependent granularity. Wave64 on RDNA consumes two physical
  // wave32 slots so the normal numbers are halved.
  uint32_t granulated_vgprs = kd.compute_pgm_rsrc1 & 0x3F;
  uint32_t vgpr_granule, total_vgprs;
  bool wave64_on_rdna = gfx_version >= GFX_VERSION_GFX10_1 && wave_size == 64;
  if (gfx_version >= GFX_VERSION_GFX10_1) {
    vgpr_granule = wave64_on_rdna ? 4 : 8;
    total_vgprs = wave64_on_rdna ? 512 : 1024;
  } else {
    vgpr_granule = 4;
    total_vgprs = 256;
  }
  uint32_t used_vgprs = (granulated_vgprs + 1) * vgpr_granule;
  uint32_t vgpr_waves = used_vgprs > 0 ? total_vgprs / used_vgprs : max_waves;

  // SGPR occupancy, SGPRs do not affect occupancy in GFX10+ so only GFX9
  // needs this limit.
  uint32_t gpr_waves = detail::min(vgpr_waves, max_waves);
  if (gfx_version < GFX_VERSION_GFX10_1) {
    uint32_t granulated_sgprs = (kd.compute_pgm_rsrc1 >> 6) & 0xF;
    uint32_t used_sgprs = (granulated_sgprs + 1) * 8;
    uint32_t sgpr_waves = 800u / detail::align_up(used_sgprs, 16u);
    gpr_waves = detail::min(gpr_waves, sgpr_waves);
  }

  // In WGP mode in GFX10+ pairs two CUs behind one scheduler, doubling the
  // SIMDs available per physical CU.
  bool wgp_mode = gfx_version >= GFX_VERSION_GFX10_1 &&
                  (kd.compute_pgm_rsrc1 & COMPUTE_PGM_RSRC1_WGP);
  uint32_t simds = wgp_mode ? simd_per_cu * 2 : simd_per_cu;
  uint32_t waves_per_cu = simds * detail::min(max_waves, gpr_waves);

  uint32_t block_size = block.x * block.y * block.z;
  uint32_t waves_per_block =
      detail::align_up(block_size, wave_size) / wave_size;
  uint32_t result = waves_per_block > 0 ? waves_per_cu / waves_per_block : 0;

  uint32_t kernel_lds = kd.group_segment_fixed_size + dynamic_lds;
  if (kernel_lds > 0) {
    uint32_t lds_per_cu = props.lds_size_in_kb * 1024;
    result = detail::min(result, lds_per_cu / kernel_lds);
  }

  return result;
}

// Minimum number of CUs needed to complete the kernel launch.
inline uint32_t occupancy(const NodeProperties &props, uint32_t gfx_version,
                          const KernelDescriptor &kd,
                          const DispatchConfig &cfg) {
  uint32_t bpc =
      blocks_per_cu(props, gfx_version, kd, cfg.block, cfg.dynamic_lds);
  uint32_t total_blocks = cfg.grid.x * cfg.grid.y * cfg.grid.z;
  if (bpc == 0)
    return total_blocks > 0 ? static_cast<uint32_t>(-1) : 0;
  return (total_blocks + bpc - 1) / bpc;
}

// Total kernarg allocation size including implicit args and dispatch packet.
// 'kd_kernarg_size' is the kernarg_size field from the kernel descriptor,
// which already covers explicit args + implicit args for COv5/6.
inline size_t kernarg_alloc_size(uint32_t kd_kernarg_size) {
  return detail::align_up(static_cast<size_t>(kd_kernarg_size), size_t(64)) +
         sizeof(DispatchPacket);
}

// Fill the implicit kernarg block and the trailing dispatch packet within a
// kernarg buffer.  'buf' points to the start of the kernarg allocation;
// 'explicit_size' is the byte size of the caller's explicit arg struct.
//
// The compiler may trim unused trailing implicit args by reporting a smaller
// kernarg_size. We only write fields whose offsets fall within the declared
// size.
//
// We need to provide the AQL Dispatch packet because the AMDHSA compute ABI
// reads directly from it even though we aren't using AQL for our compute
// queues.
//
// Layout within the buffer:
//   [0 .. explicit_size)                           explicit args (caller fills)
//   [implicit_offset .. +trimmed_implicit_size)    ImplicitArgs (partial OK)
//   [align_up(kernarg_size, 64) .. +64)            DispatchPacket
inline void fill_implicit_args(void *buf, size_t explicit_size,
                               const KernelDescriptor &kd,
                               const DispatchConfig &cfg) {
  char *base = static_cast<char *>(buf);
  size_t implicit_offset = detail::align_up(explicit_size, size_t(8));
  size_t avail =
      kd.kernarg_size > implicit_offset ? kd.kernarg_size - implicit_offset : 0;

  // The compiler can trim the implicit arguments struct if they are unused.
  auto set = [&](size_t field_offset, const void *src, size_t len) {
    if (field_offset + len <= avail)
      std::memcpy(base + implicit_offset + field_offset, src, len);
  };

  set(offsetof(ImplicitArgs, block_count_x), &cfg.grid.x, sizeof(cfg.grid.x));
  set(offsetof(ImplicitArgs, block_count_y), &cfg.grid.y, sizeof(cfg.grid.y));
  set(offsetof(ImplicitArgs, block_count_z), &cfg.grid.z, sizeof(cfg.grid.z));

  uint16_t gx = static_cast<uint16_t>(cfg.block.x);
  uint16_t gy = static_cast<uint16_t>(cfg.block.y);
  uint16_t gz = static_cast<uint16_t>(cfg.block.z);
  set(offsetof(ImplicitArgs, group_size_x), &gx, sizeof(gx));
  set(offsetof(ImplicitArgs, group_size_y), &gy, sizeof(gy));
  set(offsetof(ImplicitArgs, group_size_z), &gz, sizeof(gz));

  uint16_t dims =
      static_cast<uint16_t>(1 + (cfg.grid.y > 1 || cfg.block.y > 1) +
                            (cfg.grid.z > 1 || cfg.block.z > 1));
  set(offsetof(ImplicitArgs, grid_dims), &dims, sizeof(dims));

  size_t pkt_offset =
      detail::align_up(static_cast<size_t>(kd.kernarg_size), size_t(64));
  DispatchPacket pkt;
  fill_dispatch_packet(pkt, cfg.grid, cfg.block, kd.private_segment_fixed_size,
                       kd.group_segment_fixed_size);
  std::memcpy(base + pkt_offset, &pkt, sizeof(pkt));
}

} // namespace kfd::abi

#endif // LIBKFD_ABI_H
