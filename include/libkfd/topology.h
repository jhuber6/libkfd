//===-- libkfd/topology.h - KFD sysfs topology ------------------*- C++ -*-===//
//
// Snapshot of the KFD topology exported via sysfs. Enumerates CPU and GPU
// nodes visible at /sys/devices/virtual/kfd/kfd/topology and extracts the
// properties required to open render devices and configure queues.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_TOPOLOGY_H
#define LIBKFD_TOPOLOGY_H

#include "libkfd/detail/small_vector.h"
#include "libkfd/error.h"

#include <cstdint>
#include <span>

namespace kfd {

struct NodeProperties {
  // KFD-assigned GPU identifier (0 for CPU-only nodes).
  uint32_t gpu_id;
  // PCI vendor ID.
  uint32_t vendor_id;
  // PCI device ID.
  uint32_t device_id;
  // PCI BDF (bus/device/function) encoding.
  uint32_t location_id;
  // PCI domain number.
  uint32_t domain;
  // ISA version encoded as major/minor/stepping.
  uint32_t gfx_target_version;
  // DRM render node minor number (/dev/dri/renderDN).
  uint32_t drm_render_minor;
  // Number of CPU cores on this node.
  uint32_t cpu_cores_count;
  // Total SIMD units across all CUs.
  uint32_t simd_count;
  // Total shader arrays (SE * SA).
  uint32_t array_count;
  // Shader arrays per shader engine.
  uint32_t simd_arrays_per_engine;
  // Compute units per shader array.
  uint32_t cu_per_simd_array;
  // SIMD units per compute unit.
  uint32_t simd_per_cu;
  // Maximum concurrent wavefronts per SIMD.
  uint32_t max_waves_per_simd;
  // Wavefront width in work-items (32 or 64).
  uint32_t wave_front_size;
  // Scratch memory slots per CU.
  uint32_t max_slots_scratch_cu;
  // Memory bank entries in this node's sysfs.
  uint32_t mem_banks_count;
  // Cache-level entries in this node's sysfs.
  uint32_t caches_count;
  // IO link entries (peer/host connections).
  uint32_t io_links_count;
  // System DMA engines.
  uint32_t num_sdma_engines;
  // XGMI-attached SDMA engines.
  uint32_t num_sdma_xgmi_engines;
  // Hardware queues per SDMA engine.
  uint32_t num_sdma_queues_per_engine;
  // Command processor queues.
  uint32_t num_cp_queues;
  // Accelerated compute complexes (XCCs).
  uint32_t num_xcc;
  // Global wave sync barriers.
  uint32_t num_gws;
  // Max shader engine clock (MHz).
  uint32_t max_engine_clk_fcompute;
  // Max CP/system clock (MHz).
  uint32_t max_engine_clk_ccompute;
  // Local data share size per workgroup in KiB.
  uint32_t lds_size_in_kb;
  // Global data share size in KiB.
  uint32_t gds_size_in_kb;
  // Device-local store in bytes.
  uint64_t local_mem_size;
  // AMDGPU family identifier.
  uint32_t family_id;
  // Graphics microcode firmware version.
  uint32_t fw_version;
  // SDMA engine firmware version.
  uint32_t sdma_fw_version;
  // Bitmask of node capability flags.
  uint32_t capability;
  // Extended capability flags.
  uint32_t capability2;
  // Bitmask of debug/trap support flags.
  uint32_t debug_prop;
  // Per-queue context save/restore area size in bytes.
  uint32_t cwsr_size;
  // Per-queue hardware control stack size in bytes.
  uint32_t ctl_stack_size;
  // Board-unique identifier (serial/ASIC fuse).
  uint64_t unique_id;
  // Shared ID for XGMI-connected GPU hive.
  uint64_t hive_id;

  // Constant capability bits in the node properties.
  static constexpr uint32_t NODE_CAP_SRAM_EDCSUPPORTED = 0x04000000;
};

struct MemoryBank {
  // HSA_HEAPTYPE: 0=system, 1=FB local, 2=LDS, 3=GPU scratch, 4=device SVM.
  uint32_t heap_type;
  // Total size in bytes.
  uint64_t size_in_bytes;
  // HSA_MEMORYPROPERTY flags (hot-pluggable, non-volatile, etc.).
  uint32_t flags;
  // Bus width in bits.
  uint32_t width;
  // Maximum memory clock in MHz.
  uint32_t mem_clk_max;
};

struct CacheInfo {
  // Processor (CU / core) id associated with this cache.
  uint32_t processor_id_low;
  // Cache level (1 = L1, 2 = L2, ...).
  uint32_t level;
  // Cache size in bytes.
  uint32_t size;
  // Cache line size in bytes.
  uint32_t line_size;
  // Cache lines per tag.
  uint32_t lines_per_tag;
  // Set associativity (ways).
  uint32_t associativity;
  // Latency in nanoseconds.
  uint32_t latency;
  // KFD cache type flags (data/instruction/CPU/HSACU/unified).
  uint32_t type;
};

struct IoLink {
  // HSA_IOLINKTYPE: 0=undefined, 1=hypertransport, 2=PCIe, 3=AMBA, 11=XGMI.
  uint32_t type;
  // Source node index in the topology.
  uint32_t node_from;
  // Destination node index in the topology.
  uint32_t node_to;
  // CDIT-style routing weight.
  uint32_t weight;
  // Minimum transfer latency in nanoseconds.
  uint32_t min_latency;
  // Maximum transfer latency in nanoseconds.
  uint32_t max_latency;
  // Minimum bandwidth in MB/s.
  uint32_t min_bandwidth;
  // Maximum bandwidth in MB/s.
  uint32_t max_bandwidth;
  // HSA_LINKPROPERTY flags (coherent, atomics, P2P DMA, etc.).
  uint32_t flags;
};

struct NodeInfo {
  NodeProperties props;
  detail::SmallVector<MemoryBank, 2> memory_banks;
  detail::SmallVector<CacheInfo, 8> caches;
  detail::SmallVector<IoLink, 4> io_links;
};

class Topology {
public:
  static std::expected<Topology, Error> create();

  ~Topology() = default;

  Topology(const Topology &) = delete;
  Topology &operator=(const Topology &) = delete;
  Topology(Topology &&) = default;
  Topology &operator=(Topology &&) = default;

  std::span<const NodeInfo> nodes() const {
    return {storage.data(), storage.size()};
  }
  std::span<NodeInfo> nodes() { return {storage.data(), storage.size()}; }

  uint32_t generation() const { return generation_id; }

private:
  Topology() = default;

  detail::SmallVector<NodeInfo, 4> storage;
  uint32_t generation_id = 0;
};

} // namespace kfd

#endif // LIBKFD_TOPOLOGY_H
