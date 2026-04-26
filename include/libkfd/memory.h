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

class Context;
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
  static std::expected<Buffer, Error>
  pin(Device &dev, void *ptr, size_t size,
      MemFlags flags = MemFlags::WRITABLE | MemFlags::EXECUTABLE);

  // Map this buffer to the device's page tables. Must be done before accessing.
  std::expected<void, Error> map(std::span<Device *const> targets);
  std::expected<void, Error> map(Device &device);

  // Unmap from one or more GPUs.
  std::expected<void, Error> unmap(std::span<Device *const> targets);

  size_t size() const { return len; }

  void *data() const { return mapping.data(); }

  Device &owner() const { return *dev; }

  // Relinquish ownership without freeing. Returns the KFD handle; the caller
  // assumes responsibility for the GPU handle and the underlying mapping.
  uint64_t release() {
    len = 0;
    mapping.release();
    dev = nullptr;
    return std::exchange(handle, 0);
  }

  explicit operator bool() const { return handle != 0; }

private:
  friend class Device;
  friend class DMABuffer;
  friend class QueueBase;
  friend class ComputeQueue;

  Buffer(uint64_t h, size_t sz, detail::MappedRegion mapping,
         detail::SmallVector<uint32_t, 2> mapped_ids, Device *dev,
         detail::Box<detail::Mutex> mtx);

  void destroy();
  void release_device();

  size_t len = 0;
  uint64_t handle = 0;
  detail::MappedRegion mapping;
  detail::Box<detail::Mutex> mtx;
  detail::SmallVector<uint32_t, 2> mapped_ids;
  Device *dev = nullptr;
};

// Get a DMA buffer interface to an existing buffer object. Exposes the standard
// POSIX file interface over the device memory.
class DMABuffer {
public:
  DMABuffer() = default;
  ~DMABuffer();

  static std::expected<DMABuffer, Error> create(Buffer &buf,
                                                uint32_t flags = 0);

  DMABuffer(const DMABuffer &) = delete;
  DMABuffer &operator=(const DMABuffer &) = delete;
  DMABuffer(DMABuffer &&other);
  DMABuffer &operator=(DMABuffer &&other);

  int fd() const { return dmabuf_fd; }

  explicit operator bool() const { return dmabuf_fd >= 0; }

private:
  explicit DMABuffer(int fd) : dmabuf_fd(fd) {}

  int dmabuf_fd = -1;
};

// Shared Virtual Memory (SVM) range policy. These annotate process VA ranges
// with migration and access hints consumed by KFD's page fault handler.
enum class SVMFlags : uint32_t {
  NONE = 0,

  // Guarantee host access to memory.
  HOST_ACCESS = /*KFD_IOCTL_SVM_FLAG_HOST_ACCESS=*/0x00000001,

  // Fine grained coherency between all devices with access.
  COHERENT = /*KFD_IOCTL_SVM_FLAG_COHERENT=*/0x00000002,

  // Use any GPU in same hive as preferred device.
  HIVE_LOCAL = /*KFD_IOCTL_SVM_FLAG_HIVE_LOCAL=*/0x00000004,

  // GPUs only read, allows replication.
  GPU_RO = /*KFD_IOCTL_SVM_FLAG_GPU_RO=*/0x00000008,

  // Allow execution on GPU.
  GPU_EXEC = /*KFD_IOCTL_SVM_FLAG_GPU_EXEC=*/0x00000010,

  // GPUs mostly read, may allow similar optimizations as RO.
  GPU_READ_MOSTLY = /*KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY=*/0x00000020,

  // Keep GPU mapping always valid as if XNACK is disabled.
  GPU_ALWAYS_MAPPED = /*KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED=*/0x00000040,

  // Fine grained coherency between all devices using device-scope atomics.
  EXT_COHERENT = /*KFD_IOCTL_SVM_FLAG_EXT_COHERENT=*/0x00000080,
};

constexpr SVMFlags operator|(SVMFlags a, SVMFlags b) {
  return static_cast<SVMFlags>(static_cast<uint32_t>(a) |
                               static_cast<uint32_t>(b));
}

constexpr SVMFlags operator&(SVMFlags a, SVMFlags b) {
  return static_cast<SVMFlags>(static_cast<uint32_t>(a) &
                               static_cast<uint32_t>(b));
}

constexpr SVMFlags operator~(SVMFlags a) {
  return static_cast<SVMFlags>(~static_cast<uint32_t>(a));
}

// Prefetch or migrate a VA range to a device, or to system memory if null.
std::expected<void, Error> svm_prefetch(Context &ctx, void *addr, size_t size,
                                        Device *dev = nullptr);

// Grant one or more devices access to a VA range. When in_place is true
// accesses are remote, otherwise pages may migrate on access.
std::expected<void, Error> svm_set_access(Context &ctx, void *addr, size_t size,
                                          std::span<Device *const> devices,
                                          bool in_place = false);

// Set the preferred migration target for a VA range.
std::expected<void, Error> svm_set_preferred_loc(Context &ctx, void *addr,
                                                 size_t size,
                                                 Device *dev = nullptr);

// Set SVM flags on a VA range. Flags not in the mask are cleared.
std::expected<void, Error> svm_set_flags(Context &ctx, void *addr, size_t size,
                                         SVMFlags flags);

// Set the migration granularity for a VA range (log2 number of pages).
std::expected<void, Error> svm_set_granularity(Context &ctx, void *addr,
                                               size_t size, uint8_t log2_pages);

} // namespace kfd

#endif // LIBKFD_MEMORY_H
