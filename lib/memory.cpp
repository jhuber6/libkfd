//===-- lib/memory.cpp - GPU memory buffer ----------------------*- C++ -*-===//
//
// Buffer methods for GPU memory allocation, mapping, and deallocation via KFD.
//
//===----------------------------------------------------------------------===//

#include "libkfd/memory.h"
#include "ioctl.h"
#include "libkfd/context.h"
#include "libkfd/detail/utility.h"
#include "libkfd/device.h"

#include <cerrno>
#include <sys/mman.h>
#include <unistd.h>

static_assert(static_cast<uint32_t>(kfd::MemType::VRAM) ==
              KFD_IOC_ALLOC_MEM_FLAGS_VRAM);
static_assert(static_cast<uint32_t>(kfd::MemType::GTT) ==
              KFD_IOC_ALLOC_MEM_FLAGS_GTT);
static_assert(static_cast<uint32_t>(kfd::MemFlags::WRITABLE) ==
              static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE));
static_assert(static_cast<uint32_t>(kfd::MemFlags::EXECUTABLE) ==
              static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE));
static_assert(static_cast<uint32_t>(kfd::MemFlags::HOST_ACCESS) ==
              static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC));
static_assert(static_cast<uint32_t>(kfd::MemFlags::NO_SUBSTITUTE) ==
              static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE));
static_assert(static_cast<uint32_t>(kfd::MemFlags::COHERENT) ==
              static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_COHERENT));
static_assert(static_cast<uint32_t>(kfd::MemFlags::UNCACHED) ==
              static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED));
static_assert(static_cast<uint32_t>(kfd::MemFlags::EXT_COHERENT) ==
              static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_EXT_COHERENT));

using namespace kfd::detail;

namespace kfd {

Buffer::Buffer(uint64_t h, size_t sz, MappedRegion mapping,
               SmallVector<uint32_t, 2> mapped_ids, Device *dev, Box<Mutex> mtx)
    : len(sz), handle(h), mapping(std::move(mapping)), mtx(std::move(mtx)),
      mapped_ids(std::move(mapped_ids)), dev(dev) {}

void Buffer::destroy() {
  if (handle == 0)
    return;
  Context &ctx = dev->context();
  LockGuard guard(*mtx);
  if (!mapped_ids.empty()) {
    ioctl::kfd::unmap_memory_from_gpu_args args{
        .handle = handle,
        .device_ids_array_ptr = reinterpret_cast<uintptr_t>(mapped_ids.data()),
        .n_devices = static_cast<uint32_t>(mapped_ids.size()),
    };
    KFD_ASSERT(
        ioctl::call<ioctl::kfd::UNMAP_MEMORY_FROM_GPU>(ctx.kfd_fd(), args));
  }
  ioctl::kfd::free_memory_of_gpu_args args{.handle = handle};
  KFD_ASSERT(ioctl::call<ioctl::kfd::FREE_MEMORY_OF_GPU>(ctx.kfd_fd(), args));
  handle = 0;
}

Buffer::~Buffer() { destroy(); }

void Buffer::release_device() {
  mapping.release();
  destroy();
}

Buffer::Buffer(Buffer &&other)
    : len(std::exchange(other.len, 0)), handle(std::exchange(other.handle, 0)),
      mapping(std::move(other.mapping)), mtx(std::move(other.mtx)),
      mapped_ids(std::move(other.mapped_ids)),
      dev(std::exchange(other.dev, nullptr)) {}

Buffer &Buffer::operator=(Buffer &&other) {
  if (this != &other) {
    destroy();
    handle = std::exchange(other.handle, 0);
    len = std::exchange(other.len, 0);
    mapping = std::move(other.mapping);
    mtx = std::move(other.mtx);
    mapped_ids = std::move(other.mapped_ids);
    dev = std::exchange(other.dev, nullptr);
  }
  return *this;
}

std::expected<Buffer, Error> Buffer::allocate(Device &dev, size_t size,
                                              MemType type, MemFlags flags,
                                              void *addr) {
  if (type == MemType::VRAM &&
      (flags & MemFlags::HOST_ACCESS) != MemFlags::NONE &&
      !dev.vram_host_visible())
    return kfd::unexpected(ENOTSUP,
                           "HOST_ACCESS on VRAM requires large PCI-e BAR");

  Context &ctx = dev.context();
  size = align_up(size, page_size());

  auto reservation = KFD_TRY(MappedRegion::reserve(size, addr));

  void *va = reservation.data();
  uintptr_t va_addr = reinterpret_cast<uintptr_t>(va);

  ioctl::kfd::alloc_memory_of_gpu_args alloc_args{
      .va_addr = va_addr,
      .size = size,
      .gpu_id = dev.gpu_id(),
      .flags = static_cast<uint32_t>(type) | static_cast<uint32_t>(flags),
  };

  KFD_CHECK(
      ioctl::call<ioctl::kfd::ALLOC_MEMORY_OF_GPU>(ctx.kfd_fd(), alloc_args));

  bool host_rw =
      type == MemType::GTT || (flags & MemFlags::HOST_ACCESS) != MemFlags::NONE;
  int prot = host_rw ? (PROT_READ | PROT_WRITE) : PROT_NONE;

  auto rebound = reservation.rebind(dev.render_fd(), prot,
                                    static_cast<off_t>(alloc_args.mmap_offset));
  if (!rebound) {
    ioctl::kfd::free_memory_of_gpu_args free_args{.handle = alloc_args.handle};
    KFD_ASSERT(
        ioctl::call<ioctl::kfd::FREE_MEMORY_OF_GPU>(ctx.kfd_fd(), free_args));
    return kfd::unexpected(rebound.error());
  }

  auto mtx = KFD_TRY(Box<Mutex>::create());
  return Buffer(alloc_args.handle, size, std::move(*rebound), {}, &dev,
                std::move(mtx));
}

std::expected<Buffer, Error> Buffer::pin(Device &dev, void *ptr, size_t size,
                                         MemFlags flags) {
  Context &ctx = dev.context();
  size = align_up(size, page_size());
  uintptr_t va_addr = reinterpret_cast<uintptr_t>(ptr);

  ioctl::kfd::alloc_memory_of_gpu_args alloc_args{
      .va_addr = va_addr,
      .size = size,
      .mmap_offset = va_addr,
      .gpu_id = dev.gpu_id(),
      .flags = static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_USERPTR) |
               static_cast<uint32_t>(flags),
  };

  KFD_CHECK(
      ioctl::call<ioctl::kfd::ALLOC_MEMORY_OF_GPU>(ctx.kfd_fd(), alloc_args));

  auto mtx = KFD_TRY(Box<Mutex>::create());
  return Buffer(alloc_args.handle, size, {}, {}, &dev, std::move(mtx));
}

std::expected<void, Error> Buffer::map(std::span<Device *const> targets) {
  if (!*this)
    return kfd::unexpected(EINVAL, "map called on null buffer");

  Context &ctx = dev->context();

  SmallVector<uint32_t, 8> ids;
  for (auto *d : targets)
    KFD_CHECK(ids.push_back(d->gpu_id()));

  LockGuard guard(*mtx);
  KFD_CHECK(mapped_ids.reserve(mapped_ids.size() + ids.size()));

  ioctl::kfd::map_memory_to_gpu_args args{
      .handle = handle,
      .device_ids_array_ptr = reinterpret_cast<uintptr_t>(ids.data()),
      .n_devices = static_cast<uint32_t>(ids.size()),
  };
  if (auto r = ioctl::call<ioctl::kfd::MAP_MEMORY_TO_GPU>(ctx.kfd_fd(), args);
      !r)
    return r;

  for (auto id : ids) {
    bool found = false;
    for (size_t i = 0; i < mapped_ids.size(); ++i)
      if (mapped_ids[i] == id) {
        found = true;
        break;
      }
    if (!found)
      KFD_ASSERT(mapped_ids.push_back(id));
  }

  return {};
}

std::expected<void, Error> Buffer::map(Device &dev) {
  Device *local = &dev;
  return map(std::span<Device *const>(&local, 1));
}

std::expected<void, Error> Buffer::unmap(std::span<Device *const> targets) {
  if (!*this)
    return kfd::unexpected(EINVAL, "unmap called on null buffer");

  Context &ctx = dev->context();

  SmallVector<uint32_t, 8> ids;
  for (auto *d : targets)
    KFD_CHECK(ids.push_back(d->gpu_id()));

  LockGuard guard(*mtx);

  ioctl::kfd::unmap_memory_from_gpu_args args{
      .handle = handle,
      .device_ids_array_ptr = reinterpret_cast<uintptr_t>(ids.data()),
      .n_devices = static_cast<uint32_t>(ids.size()),
  };
  if (auto r =
          ioctl::call<ioctl::kfd::UNMAP_MEMORY_FROM_GPU>(ctx.kfd_fd(), args);
      !r)
    return r;

  for (auto id : ids) {
    for (size_t i = 0; i < mapped_ids.size(); ++i) {
      if (mapped_ids[i] == id) {
        mapped_ids[i] = mapped_ids.back();
        mapped_ids.pop_back();
        break;
      }
    }
  }

  return {};
}

std::expected<DMABuffer, Error> DMABuffer::create(Buffer &buf, uint32_t flags) {
  if (!buf)
    return kfd::unexpected(EINVAL, "export_dmabuf called on null buffer");

  Context &ctx = buf.dev->context();

  ioctl::kfd::export_dmabuf_args args{
      .handle = buf.handle,
      .flags = flags,
  };
  KFD_CHECK(ioctl::call<ioctl::kfd::EXPORT_DMABUF>(ctx.kfd_fd(), args));

  return DMABuffer(static_cast<int>(args.dmabuf_fd));
}

DMABuffer::~DMABuffer() {
  if (dmabuf_fd >= 0)
    ::close(dmabuf_fd);
}

DMABuffer::DMABuffer(DMABuffer &&other)
    : dmabuf_fd(std::exchange(other.dmabuf_fd, -1)) {}

DMABuffer &DMABuffer::operator=(DMABuffer &&other) {
  if (this != &other) {
    if (dmabuf_fd >= 0)
      ::close(dmabuf_fd);
    dmabuf_fd = std::exchange(other.dmabuf_fd, -1);
  }
  return *this;
}

} // namespace kfd
