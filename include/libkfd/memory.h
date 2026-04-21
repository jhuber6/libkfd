//===-- libkfd/memory.h - GPU memory buffer types ---------------*- C++ -*-===//
//
// RAII buffer type for GPU memory allocation, mapping, and deallocation through
// KFD. Any memory allocated must be explicitly mapped to the devices intending
// to access it, even if physically allocated on said device.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_MEMORY_H
#define LIBKFD_MEMORY_H

#include "libkfd/detail/box.h"
#include "libkfd/detail/mapped_region.h"
#include "libkfd/detail/mutex.h"
#include "libkfd/detail/small_vector.h"
#include "libkfd/error.h"

#include <cstdint>
#include <span>

namespace kfd {

class Device;

enum class MemType : uint32_t {
  // GPU memory local to the device, can be made host accessible on systems
  // supporting PCI-e large-bar / smart access memory.
  VRAM = /*KFD_IOC_ALLOC_MEM_FLAGS_VRAM=*/0x1,

  // GPU accessible system memory mapped into the GPU's virtual address space.
  // The graphics address remapping table (GART) handles access over PCI-e.
  GTT = /*KFD_IOC_ALLOC_MEM_FLAGS_GTT=*/0x2,
};

enum class MemFlags : uint32_t {
  NONE = 0,

  // GPU page table write permission.
  WRITABLE = /*KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE=*/0x80000000,

  // GPU page table execute permission (instruction fetch).
  EXECUTABLE = /*KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE=*/0x40000000,

  // Request CPU-accessible VRAM via large BAR, for GTT this is implicit.
  HOST_ACCESS = /*KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC=*/0x20000000,

  // Prevent the kernel from falling back to GTT when VRAM is exhausted.
  NO_SUBSTITUTE = /*KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE=*/0x10000000,

  // Fine-grained coherence between CPU and GPU (MTYPE_CC on supported HW).
  COHERENT = /*KFD_IOC_ALLOC_MEM_FLAGS_COHERENT=*/0x4000000,

  // Bypass GPU caches entirely. Useful for doorbells and polling buffers.
  UNCACHED = /*KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED=*/0x2000000,

  // Extended (inter-device) coherence for multi-GPU XGMI topologies.
  EXT_COHERENT = /*KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT=*/0x1000000,
};

constexpr MemFlags operator|(MemFlags a, MemFlags b) {
  return static_cast<MemFlags>(static_cast<uint32_t>(a) |
                               static_cast<uint32_t>(b));
}

constexpr MemFlags operator&(MemFlags a, MemFlags b) {
  return static_cast<MemFlags>(static_cast<uint32_t>(a) &
                               static_cast<uint32_t>(b));
}

class Buffer {
public:
  Buffer() = default;
  ~Buffer();

  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&other);
  Buffer &operator=(Buffer &&other);

  // Allocate memory on the device.
  static std::expected<Buffer, Error>
  allocate(Device &dev, size_t size, MemType type,
           MemFlags flags = MemFlags::WRITABLE, void *addr = nullptr);

  // Pin an existing host memory region for device access.
  static std::expected<Buffer, Error> pin(Device &dev, void *ptr, size_t size);

  // Map this buffer to the device's page tables. Must be done before accessing.
  std::expected<void, Error> map(std::span<Device *const> targets);
  std::expected<void, Error> map(Device &dev);

  // Unmap from one or more GPUs.
  std::expected<void, Error> unmap(std::span<Device *const> targets);

  size_t size() const { return len; }

  void *data() const { return mapping.data(); }

  // Relinquish ownership without freeing. Returns the KFD handle; the caller
  // assumes responsibility for the GPU handle and the underlying mapping.
  uint64_t release() {
    len = 0;
    mapping.release();
    owner = nullptr;
    return std::exchange(handle, 0);
  }

  explicit operator bool() const { return handle != 0; }

private:
  friend class Device;
  friend class QueueBase;
  friend class ComputeQueue;

  Buffer(uint64_t h, size_t sz, detail::MappedRegion mapping,
         detail::SmallVector<uint32_t, 2> mapped_ids, Device *owner,
         detail::Box<detail::Mutex> mtx);

  void destroy();
  void release_device();

  size_t len = 0;
  uint64_t handle = 0;
  detail::MappedRegion mapping;
  detail::Box<detail::Mutex> mtx;
  detail::SmallVector<uint32_t, 2> mapped_ids;
  Device *owner = nullptr;
};

} // namespace kfd

#endif // LIBKFD_MEMORY_H
