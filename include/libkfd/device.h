//===-- libkfd/device.h - Per-GPU device state ------------------*- C++ -*-===//
//
// Represents a single GPU node. Owns the DRM render fd, the GPUVM aperture,
// the VA allocator, queues, buffer objects, and the scratch memory pool.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DEVICE_H
#define LIBKFD_DEVICE_H

#include "libkfd/detail/elf.h"
#include "libkfd/detail/mutex.h"
#include "libkfd/detail/pool_allocator.h"
#include "libkfd/error.h"
#include "libkfd/memory.h"
#include "libkfd/topology.h"

#include <cstdint>
#include <span>

namespace kfd {

class Context;
class ComputeQueue;

// Device attributes from the DRM render node (AMDGPU_INFO), as distinct from
// NodeProperties which mirrors the KFD sysfs topology.
struct DrmProperties {
  // Reference counter frequency in Hz. Fixed rate timing counter.
  uint64_t timestamp_freq;
  // Peak shader engine clock in Hz.
  uint64_t max_engine_clock;
  // Peak memory clock in Hz.
  uint64_t max_memory_clock;
  // VRAM bus width in bits.
  uint32_t vram_bit_width;
  // VRAM type (AMDGPU_VRAM_TYPE_*).
  uint32_t vram_type;
  // Negotiated PCIe generation.
  uint32_t pcie_gen;
  // Negotiated PCIe lane count.
  uint32_t pcie_num_lanes;
};

class Device {
public:
  // Opens the render node, calls ACQUIRE_VM, queries the GPUVM aperture, and
  // initializes the scratch memory pool.
  static std::expected<Device, Error> create(Context &ctx, NodeInfo info);

  ~Device();

  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&other);
  Device &operator=(Device &&other) = delete;

  const NodeProperties &properties() const { return info.props; }
  const DrmProperties &drm_properties() const { return drm_props; }
  uint32_t gpu_id() const { return info.props.gpu_id; }
  uint32_t gfx_version() const { return info.props.gfx_target_version; }
  int render_fd() const { return drm_fd; }
  Context &context() const { return *ctx; }

  std::string_view name() const {
    return detail::elf::get_name(detail::elf::get_mach(gfx_version()));
  }

  // True if the device has PCI-e large bar enabled and VRAM is host accessible.
  bool vram_host_visible() const {
    for (size_t i = 0; i < info.memory_banks.size(); ++i)
      if (info.memory_banks[i].heap_type == 1)
        return true;
    return false;
  }

  std::span<const MemoryBank> memory_banks() const {
    return {info.memory_banks.data(), info.memory_banks.size()};
  }
  std::span<const CacheInfo> caches() const {
    return {info.caches.data(), info.caches.size()};
  }
  std::span<const IoLink> io_links() const {
    return {info.io_links.data(), info.io_links.size()};
  }

  // Checks if an ELF image can be loaded onto this device.
  bool loadable(std::span<const std::byte> image) const;

  // Lazily mmap the doorbell page on first call, then return a pointer to the
  // specific slot within it. All queues on this device share the same page.
  std::expected<volatile uint64_t *, Error> doorbell(uint64_t raw_offset);

private:
  friend class Context;
  friend class QueueBase;
  friend class ComputeQueue;

  Device(Context &ctx, NodeInfo info);

  Context *ctx;
  NodeInfo info;
  int drm_fd = -1;
  DrmProperties drm_props{};

  // Memory apertures for VRAM and scratch ranges.
  uintptr_t gpuvm_base = 0;
  uintptr_t gpuvm_limit = 0;
  uint64_t scratch_aperture_base = 0;
  uint64_t scratch_aperture_limit = 0;

  // The MMIO region used to signal the command processor.
  Buffer doorbells;
  detail::Mutex doorbell_mtx;

  // Trap handler executable and scratch.
  Buffer trap_tba;
  Buffer trap_tma;

  // Scratch pool management.
  detail::MappedRegion scratch_reservation;
  detail::PoolAllocator scratch_allocator;
};

} // namespace kfd

#endif // LIBKFD_DEVICE_H
