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
#include <cstring>
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

static_assert(static_cast<uint32_t>(kfd::SVMFlags::HOST_ACCESS) ==
              KFD_IOCTL_SVM_FLAG_HOST_ACCESS);
static_assert(static_cast<uint32_t>(kfd::SVMFlags::COHERENT) ==
              KFD_IOCTL_SVM_FLAG_COHERENT);
static_assert(static_cast<uint32_t>(kfd::SVMFlags::HIVE_LOCAL) ==
              KFD_IOCTL_SVM_FLAG_HIVE_LOCAL);
static_assert(static_cast<uint32_t>(kfd::SVMFlags::GPU_RO) ==
              KFD_IOCTL_SVM_FLAG_GPU_RO);
static_assert(static_cast<uint32_t>(kfd::SVMFlags::GPU_EXEC) ==
              KFD_IOCTL_SVM_FLAG_GPU_EXEC);
static_assert(static_cast<uint32_t>(kfd::SVMFlags::GPU_READ_MOSTLY) ==
              KFD_IOCTL_SVM_FLAG_GPU_READ_MOSTLY);
static_assert(static_cast<uint32_t>(kfd::SVMFlags::GPU_ALWAYS_MAPPED) ==
              KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED);
static_assert(static_cast<uint32_t>(kfd::SVMFlags::EXT_COHERENT) ==
              KFD_IOCTL_SVM_FLAG_EXT_COHERENT);

using namespace kfd::detail;

namespace kfd {

Buffer::Buffer(uint64_t h, size_t sz, MappedRegion mapping,
               SmallVector<uint32_t, 2> mapped_ids, Device *dev, Mutex mtx)
    : len(sz), handle(h), mapping(std::move(mapping)), mtx(std::move(mtx)),
      mapped_ids(std::move(mapped_ids)), dev(dev) {}

void Buffer::destroy() {
  if (handle == 0)
    return;
  Context &ctx = dev->context();
  LockGuard guard(mtx);
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

  return Buffer(alloc_args.handle, size, std::move(*rebound), {}, &dev, {});
}

std::expected<Buffer, Error> Buffer::pin(Device &dev, const void *ptr,
                                         size_t size, MemFlags flags) {
  size_t aligned = align_up(size, page_size());
  auto region = KFD_TRY(MappedRegion::create(aligned));
  std::memcpy(region.data(), ptr, size);
  return pin_region(dev, std::move(region), flags);
}

std::expected<Buffer, Error>
Buffer::pin_region(Device &dev, MappedRegion region, MemFlags flags) {
  Context &ctx = dev.context();
  size_t size = region.size();
  uintptr_t va_addr = reinterpret_cast<uintptr_t>(region.data());

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

  return Buffer(alloc_args.handle, size, std::move(region), {}, &dev, {});
}

std::expected<void, Error> Buffer::map(std::span<Device *const> targets) {
  if (!*this)
    return kfd::unexpected(EINVAL, "map called on null buffer");

  Context &ctx = dev->context();

  SmallVector<uint32_t, 8> ids;
  for (auto *d : targets)
    KFD_CHECK(ids.push_back(d->gpu_id()));

  LockGuard guard(mtx);
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

std::expected<void, Error> Buffer::map(Device &device) {
  Device *local = &device;
  return map(std::span<Device *const>(&local, 1));
}

std::expected<void, Error> Buffer::unmap(std::span<Device *const> targets) {
  if (!*this)
    return kfd::unexpected(EINVAL, "unmap called on null buffer");

  Context &ctx = dev->context();

  SmallVector<uint32_t, 8> ids;
  for (auto *d : targets)
    KFD_CHECK(ids.push_back(d->gpu_id()));

  LockGuard guard(mtx);

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

namespace {

std::expected<void, kfd::Error>
svm_set_attrs(int fd, void *addr, size_t size,
              std::span<const ioctl::kfd::svm_attribute> attrs) {
  constexpr size_t HDR = sizeof(ioctl::kfd::svm_args);
  size_t extra = attrs.size() * sizeof(ioctl::kfd::svm_attribute);

  alignas(ioctl::kfd::svm_args) char
      buf[HDR + 16 * sizeof(ioctl::kfd::svm_attribute)] = {};
  auto &args = *reinterpret_cast<ioctl::kfd::svm_args *>(buf);
  args.start_addr = reinterpret_cast<uintptr_t>(addr);
  args.size = size;
  args.op = KFD_IOCTL_SVM_OP_SET_ATTR;
  args.nattr = static_cast<uint32_t>(attrs.size());
  std::memcpy(args.attrs, attrs.data(), extra);

  return ioctl::call<ioctl::kfd::SVM>(fd, args, extra);
}

} // namespace

std::expected<void, Error> svm_prefetch(Context &ctx, void *addr, size_t size,
                                        Device *dev) {
  ioctl::kfd::svm_attribute attr{
      .type = KFD_IOCTL_SVM_ATTR_PREFETCH_LOC,
      .value = dev ? dev->gpu_id() : 0,
  };
  return svm_set_attrs(ctx.kfd_fd(), addr, size, {&attr, 1});
}

std::expected<void, Error> svm_set_access(Context &ctx, void *addr, size_t size,
                                          std::span<Device *const> devices,
                                          bool in_place) {
  uint32_t type =
      in_place ? KFD_IOCTL_SVM_ATTR_ACCESS_IN_PLACE : KFD_IOCTL_SVM_ATTR_ACCESS;
  SmallVector<ioctl::kfd::svm_attribute, 8> attrs;
  for (auto *d : devices)
    KFD_CHECK(attrs.push_back({.type = type, .value = d->gpu_id()}));
  return svm_set_attrs(ctx.kfd_fd(), addr, size, {attrs.data(), attrs.size()});
}

std::expected<void, Error> svm_set_preferred_loc(Context &ctx, void *addr,
                                                 size_t size, Device *dev) {
  ioctl::kfd::svm_attribute attr{
      .type = KFD_IOCTL_SVM_ATTR_PREFERRED_LOC,
      .value = dev ? dev->gpu_id() : 0,
  };
  return svm_set_attrs(ctx.kfd_fd(), addr, size, {&attr, 1});
}

std::expected<void, Error> svm_set_flags(Context &ctx, void *addr, size_t size,
                                         SVMFlags flags) {
  uint32_t raw = static_cast<uint32_t>(flags);
  ioctl::kfd::svm_attribute attrs[2] = {
      {.type = KFD_IOCTL_SVM_ATTR_SET_FLAGS, .value = raw},
      {.type = KFD_IOCTL_SVM_ATTR_CLR_FLAGS, .value = ~raw},
  };
  return svm_set_attrs(ctx.kfd_fd(), addr, size, attrs);
}

std::expected<void, Error>
svm_set_granularity(Context &ctx, void *addr, size_t size, uint8_t log2_pages) {
  ioctl::kfd::svm_attribute attr{
      .type = KFD_IOCTL_SVM_ATTR_GRANULARITY,
      .value = log2_pages,
  };
  return svm_set_attrs(ctx.kfd_fd(), addr, size, {&attr, 1});
}

} // namespace kfd
