//===-- lib/device.cpp - Per-GPU device implementation ----------*- C++ -*-===//
//
// Interfaces with the device interface. On initialization we open libdrm to
// bind the driver's virtual address space and set up memory apertures.
//
//===----------------------------------------------------------------------===//

#include "libkfd/device.h"

#include "ioctl.h"
#include "libkfd/abi.h"
#include "libkfd/context.h"
#include "libkfd/detail/elf.h"
#include "libkfd/detail/scratch.h"
#include "libkfd/detail/utility.h"
#include "libkfd/trap_handler.h"

#include <amdgpu.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace kfd::detail;

namespace kfd {
namespace {

// Calling 'ioctl(AMDKFD_IOC_ACQUIRE_VM)' binds the KFD virtual memory to DRM
// for the lifetime of the process. This caches it in the unlikely event that
// someone reopens the context.
struct CachedRenderFD {
  uint32_t gpu_id;
  int fd;
  amdgpu_device_handle amdgpu_dev;
};
constexpr uint32_t MAX_CACHED_FDS = 128;
Mutex render_fd_mtx;
CachedRenderFD render_fd_cache[MAX_CACHED_FDS];
uint32_t render_fd_count;

int find_cached_fd(uint32_t gpu_id) {
  for (uint32_t i = 0; i < render_fd_count; ++i)
    if (render_fd_cache[i].gpu_id == gpu_id)
      return render_fd_cache[i].fd;
  return -1;
}

// Sanity check to ensure this DRM node is actually for AMD compute.
bool is_amdgpu(int fd) {
  char name[16] = {};
  ioctl::drm::version_args ver{
      .name_len = sizeof(name) - 1,
      .name = name,
  };
  if (!ioctl::call<ioctl::drm::GET_VERSION>(fd, ver))
    return false;
  return std::string_view(name, ver.name_len) == "amdgpu";
}

// Opens the libdrm render node to bind the virtual address space to the driver.
std::expected<int, Error> open_render_fd(int kfd_fd,
                                         const NodeProperties &props) {
  LockGuard guard(render_fd_mtx);
  int cached = find_cached_fd(props.gpu_id);
  if (cached >= 0)
    return cached;

  char path[64];
  std::snprintf(path, sizeof(path), "/dev/dri/renderD%u",
                props.drm_render_minor);
  int rfd = ::open(path, O_RDWR | O_CLOEXEC);
  if (rfd < 0) {
    int err = errno;
    return kfd::unexpected(err, "failed to open render node '%s'", path);
  }

  if (!is_amdgpu(rfd)) {
    ::close(rfd);
    return kfd::unexpected(ENODEV, "render node '%s' is not an amdgpu device",
                           path);
  }

  // Initialize the AMDGPU DRM interface once to access memory operations. This
  // gives us access to the device's MMU and page tables.
  amdgpu_device_handle amdgpu_dev = nullptr;
  uint32_t drm_major, drm_minor;
  if (amdgpu_device_initialize(rfd, &drm_major, &drm_minor, &amdgpu_dev)) {
    ::close(rfd);
    return kfd::unexpected(ENODEV, "amdgpu_device_initialize failed for '%s'",
                           path);
  }
  int dfd = amdgpu_device_get_fd(amdgpu_dev);
  ::close(rfd);

  // Bind the DRM virtual address space to the KFD process.
  ioctl::kfd::acquire_vm_args acq{
      .drm_fd = static_cast<uint32_t>(dfd),
      .gpu_id = props.gpu_id,
  };
  if (auto r = ioctl::call<ioctl::kfd::ACQUIRE_VM>(kfd_fd, acq); !r) {
    amdgpu_device_deinitialize(amdgpu_dev);
    return kfd::unexpected(r.error());
  }

  if (render_fd_count >= MAX_CACHED_FDS) {
    amdgpu_device_deinitialize(amdgpu_dev);
    return kfd::unexpected(ENOMEM, "render fd cache full (%u entries)",
                           MAX_CACHED_FDS);
  }
  render_fd_cache[render_fd_count++] = {
      .gpu_id = props.gpu_id,
      .fd = dfd,
      .amdgpu_dev = amdgpu_dev,
  };
  return dfd;
}

// Reserve a VA pool for scratch backing and register it with the kernel.
std::expected<MappedRegion, Error> init_scratch(Device &dev, size_t pool_size) {
  // The backing VA region must be 64 KiB-aligned.
  auto reservation = KFD_TRY(
      MappedRegion::reserve_aligned(pool_size, abi::PRIVATE_SEGMENT_ALIGN));

  ioctl::kfd::set_scratch_backing_va_args va_args{
      .va_addr = reinterpret_cast<uintptr_t>(reservation.data()) >> 16,
      .gpu_id = dev.gpu_id(),
  };
  KFD_CHECK(ioctl::call<ioctl::kfd::SET_SCRATCH_BACKING_VA>(
      dev.context().kfd_fd(), va_args));

  return std::move(reservation);
}

// We compile a trap handler for every supported target.
const TrapHandlerImage *find_trap_image(uint32_t gfx_version) {
  uint32_t target = elf::get_mach(gfx_version);
  uint32_t generic = elf::get_generic_for_gpu(gfx_version);

  const TrapHandlerImage *fallback = nullptr;
  for (unsigned i = 0; i < trap_handler_image_count; ++i) {
    if (trap_handler_images[i].size < sizeof(elf::Elf64_Ehdr))
      continue;
    auto *ehdr =
        reinterpret_cast<const elf::Elf64_Ehdr *>(trap_handler_images[i].data);
    uint32_t mach = ehdr->e_flags & elf::EF_AMDGPU_MACH;
    if (mach == target)
      return &trap_handler_images[i];
    if (generic && mach == generic)
      fallback = &trap_handler_images[i];
  }
  return fallback;
}

// A pair of the executable trap handler code and its scratch memory.
struct TrapHandlerBuffers {
  Buffer tba;
  Buffer tma;
};

// Load and register the trap handler binary with the device.
std::expected<TrapHandlerBuffers, Error> setup_trap_handler(Device &dev) {
  const TrapHandlerImage *img = find_trap_image(dev.gfx_version());
  if (!img)
    return kfd::unexpected(ENOENT, "no trap handler image for device gfx%u%u%x",
                           abi::gfx_version_major(dev.gfx_version()),
                           abi::gfx_version_minor(dev.gfx_version()),
                           abi::gfx_version_step(dev.gfx_version()));

  auto obj = KFD_TRY(elf::ELF64LE::create(
      {reinterpret_cast<const std::byte *>(img->data), img->size}));

  // Get the size in bytes required to store this trap handler on the device.
  auto extent = KFD_TRY(obj.load_extent());
  auto &[lo, hi, align] = extent;

  auto code = KFD_TRY(Buffer::allocate(
      dev, static_cast<size_t>(hi - lo), MemType::GTT,
      MemFlags::WRITABLE | MemFlags::EXECUTABLE | MemFlags::HOST_ACCESS));
  KFD_CHECK(code.map(dev));

  // Copy the program headers to the allocated memory so it can be executed.
  size_t code_size = static_cast<size_t>(hi - lo);
  for (const auto &ph : obj.phdrs()) {
    if (ph.p_type != elf::PT_LOAD || ph.p_filesz == 0)
      continue;
    auto seg = KFD_TRY(obj.segment_data(ph));
    uint64_t offset = ph.p_vaddr - lo;
    if (offset + ph.p_filesz > code_size)
      return kfd::unexpected(ERANGE,
                             "ELF segment at 0x%lx+0x%lx overflows "
                             "code buffer of %zu bytes",
                             static_cast<unsigned long>(offset),
                             static_cast<unsigned long>(ph.p_filesz),
                             code_size);
    std::memcpy(static_cast<char *>(code.data()) + offset, seg.data(),
                static_cast<size_t>(ph.p_filesz));
  }

  auto tma =
      KFD_TRY(Buffer::allocate(dev, page_size(), MemType::GTT,
                               MemFlags::WRITABLE | MemFlags::HOST_ACCESS));
  KFD_CHECK(tma.map(dev));

  // Sets up the trap base address and the trap memory address. This is the
  // loaded ELF and a scratch buffer for use by the trap handler.
  ioctl::kfd::set_trap_handler_args args{
      .tba_addr =
          reinterpret_cast<uint64_t>(code.data()) + (obj.header().e_entry - lo),
      .tma_addr = reinterpret_cast<uint64_t>(tma.data()),
      .gpu_id = dev.gpu_id(),
  };
  KFD_CHECK(
      ioctl::call<ioctl::kfd::SET_TRAP_HANDLER>(dev.context().kfd_fd(), args));

  return TrapHandlerBuffers{std::move(code), std::move(tma)};
}

} // namespace

Device::Device(Context &ctx, NodeInfo info)
    : ctx(&ctx), info(std::move(info)) {}

std::expected<Device, Error> Device::create(Context &ctx, NodeInfo info) {
  auto rfd = KFD_TRY(open_render_fd(ctx.kfd_fd(), info.props));

  // Query the number of device apertures available.
  ioctl::kfd::get_process_apertures_new_args ap_args{};
  ap_args.num_of_nodes = 0;
  ap_args.kfd_process_device_apertures_ptr = 0;
  KFD_CHECK(ioctl::call<ioctl::kfd::GET_PROCESS_APERTURES_NEW>(ctx.kfd_fd(),
                                                               ap_args));

  // Obtain the aperture information for every node in the topology.
  uint32_t total = ap_args.num_of_nodes;
  SmallVector<ioctl::kfd::process_device_apertures, 8> raw;
  KFD_CHECK(raw.resize(total));
  ap_args.kfd_process_device_apertures_ptr =
      reinterpret_cast<uintptr_t>(raw.data());
  ap_args.num_of_nodes = total;
  KFD_CHECK(ioctl::call<ioctl::kfd::GET_PROCESS_APERTURES_NEW>(ctx.kfd_fd(),
                                                               ap_args));

  uint32_t gpu_id = info.props.gpu_id;
  Device dev(ctx, std::move(info));
  dev.doorbell_mtx = KFD_TRY(detail::Box<detail::Mutex>::create());
  for (uint32_t i = 0; i < total; ++i) {
    if (raw[i].gpu_id == gpu_id) {
      dev.gpuvm_base = raw[i].gpuvm_base;
      dev.gpuvm_limit = raw[i].gpuvm_limit;
      dev.scratch_aperture_base = raw[i].scratch_base;
      dev.scratch_aperture_limit = raw[i].scratch_limit;
      dev.drm_fd = rfd;

      ioctl::kfd::set_memory_policy_args pol{
          .alternate_aperture_base = dev.gpuvm_base,
          .alternate_aperture_size = dev.gpuvm_limit - dev.gpuvm_base + 1,
          .gpu_id = gpu_id,
          .default_policy = KFD_IOC_CACHE_POLICY_NONCOHERENT,
          .alternate_policy = KFD_IOC_CACHE_POLICY_COHERENT,
      };
      KFD_CHECK(ioctl::call<ioctl::kfd::SET_MEMORY_POLICY>(ctx.kfd_fd(), pol));

      // Reserve the entire scratch pool VA at device initialization. The
      // allocator hands out slices of it to a dispatch when needed.
      if (dev.scratch_aperture_limit < dev.scratch_aperture_base)
        return kfd::unexpected(
            ERANGE, "scratch aperture limit 0x%lx < base 0x%lx",
            static_cast<unsigned long>(dev.scratch_aperture_limit),
            static_cast<unsigned long>(dev.scratch_aperture_base));
      size_t pool_size =
          dev.scratch_aperture_limit - dev.scratch_aperture_base + 1;
      auto reservation = KFD_TRY(init_scratch(dev, pool_size));

      auto alloc = KFD_TRY(detail::PoolAllocator::create(
          std::span<std::byte>(static_cast<std::byte *>(reservation.data()),
                               reservation.size()),
          abi::PRIVATE_SEGMENT_ALIGN));

      dev.scratch_reservation = std::move(reservation);
      dev.scratch_allocator = std::move(alloc);

      // Register the trap handler executable with the device.
      auto trap = KFD_TRY(setup_trap_handler(dev));
      dev.trap_tba = std::move(trap.tba);
      dev.trap_tma = std::move(trap.tma);

      return dev;
    }
  }

  return kfd::unexpected(ENODEV, "no GPUVM aperture found for gpu_id %u",
                         gpu_id);
}

Device::~Device() = default;

Device::Device(Device &&other)
    : ctx(std::exchange(other.ctx, nullptr)), info(std::move(other.info)),
      drm_fd(std::exchange(other.drm_fd, -1)), gpuvm_base(other.gpuvm_base),
      gpuvm_limit(other.gpuvm_limit),
      scratch_aperture_base(other.scratch_aperture_base),
      scratch_aperture_limit(other.scratch_aperture_limit),
      doorbells(std::move(other.doorbells)),
      doorbell_mtx(std::move(other.doorbell_mtx)),
      trap_tba(std::move(other.trap_tba)), trap_tma(std::move(other.trap_tma)),
      scratch_reservation(std::move(other.scratch_reservation)),
      scratch_allocator(std::move(other.scratch_allocator)) {
  doorbells.owner = this;
  trap_tba.owner = this;
  trap_tma.owner = this;
}

static constexpr size_t DOORBELL_PAGE_SIZE = 8192;

std::expected<volatile uint64_t *, Error>
Device::doorbell(uint64_t raw_offset) {
  detail::LockGuard guard(*doorbell_mtx);

  if (!doorbells) {
    auto reservation = KFD_TRY(MappedRegion::reserve(DOORBELL_PAGE_SIZE));

    void *va = reservation.data();
    uintptr_t va_addr = reinterpret_cast<uintptr_t>(va);
    if (va_addr < gpuvm_base || va_addr + DOORBELL_PAGE_SIZE - 1 > gpuvm_limit)
      return kfd::unexpected(
          EFAULT, "doorbell VA 0x%lx outside GPUVM aperture [0x%lx, 0x%lx]",
          va_addr, gpuvm_base, static_cast<uint64_t>(gpuvm_limit));

    ioctl::kfd::alloc_memory_of_gpu_args alloc_args{
        .va_addr = va_addr,
        .size = DOORBELL_PAGE_SIZE,
        .gpu_id = info.props.gpu_id,
        .flags = static_cast<uint32_t>(KFD_IOC_ALLOC_MEM_FLAGS_DOORBELL |
                                       KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                                       KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                                       KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE),
    };
    KFD_CHECK(ioctl::call<ioctl::kfd::ALLOC_MEMORY_OF_GPU>(ctx->kfd_fd(),
                                                           alloc_args));

    uint64_t page_base = raw_offset & ~(DOORBELL_PAGE_SIZE - 1);
    auto mapping = reservation.rebind(ctx->kfd_fd(), PROT_READ | PROT_WRITE,
                                      static_cast<off_t>(page_base));
    if (!mapping) {
      ioctl::kfd::free_memory_of_gpu_args free_args{.handle =
                                                        alloc_args.handle};
      KFD_ASSERT(ioctl::call<ioctl::kfd::FREE_MEMORY_OF_GPU>(ctx->kfd_fd(),
                                                             free_args));
      return kfd::unexpected(mapping.error());
    }

    SmallVector<uint32_t, 2> ids;
    KFD_CHECK(ids.push_back(info.props.gpu_id));
    ioctl::kfd::map_memory_to_gpu_args map_args{
        .handle = alloc_args.handle,
        .device_ids_array_ptr = reinterpret_cast<uintptr_t>(ids.data()),
        .n_devices = 1,
    };
    if (auto r =
            ioctl::call<ioctl::kfd::MAP_MEMORY_TO_GPU>(ctx->kfd_fd(), map_args);
        !r) {
      ioctl::kfd::free_memory_of_gpu_args free_args{.handle =
                                                        alloc_args.handle};
      KFD_ASSERT(ioctl::call<ioctl::kfd::FREE_MEMORY_OF_GPU>(ctx->kfd_fd(),
                                                             free_args));
      return kfd::unexpected(r.error());
    }

    auto buf_mtx = KFD_TRY(detail::Box<detail::Mutex>::create());
    doorbells =
        Buffer(alloc_args.handle, DOORBELL_PAGE_SIZE, std::move(*mapping),
               std::move(ids), this, std::move(buf_mtx));
  }

  uint32_t slot_off =
      static_cast<uint32_t>(raw_offset & (DOORBELL_PAGE_SIZE - 1));
  return reinterpret_cast<volatile uint64_t *>(
      static_cast<char *>(doorbells.data()) + slot_off);
}

} // namespace kfd
