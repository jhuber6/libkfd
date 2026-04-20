//===-- lib/loader.cpp - Static ELF loader interface ------------*- C++ -*-===//
//
// Loads an AMDHSA ELF code object into GPU memory and provides symbol lookup.
//
//===----------------------------------------------------------------------===//

#include "libkfd/loader.h"

#include "libkfd/context.h"
#include "libkfd/detail/mapped_region.h"
#include "libkfd/detail/utility.h"
#include "libkfd/queue.h"
#include "libkfd/signal.h"

#include <algorithm>
#include <cstring>
#include <utility>

using namespace kfd::detail;

namespace kfd {

std::expected<Buffer, Error>
Kernel::make_kernargs(Device &dev, std::span<const std::byte> explicit_args,
                      const DispatchConfig &cfg) const {
  size_t total = abi::kernarg_alloc_size(descriptor->kernarg_size);
  auto buf = KFD_TRY(Buffer::allocate(
      dev, align_up(total, page_size()), MemType::GTT,
      MemFlags::WRITABLE | MemFlags::HOST_ACCESS | MemFlags::UNCACHED));
  KFD_CHECK(buf.map(dev));

  std::memset(buf.data(), 0, total);
  size_t copy_size = std::min(explicit_args.size(), total);
  std::memcpy(buf.data(), explicit_args.data(), copy_size);
  abi::fill_implicit_args(buf.data(), explicit_args.size(), *descriptor, cfg);

  return buf;
}

std::expected<Executable, Error>
Executable::load(Device &dev, std::span<const std::byte> image, SDMAQueue &sdma,
                 ComputeQueue &compute) {
  auto region = KFD_TRY(MappedRegion::create(image.size_bytes()));
  std::memcpy(region.data(), image.data(), image.size_bytes());

  auto elf = KFD_TRY(elf::ELF64LE::create(region.as_span<const std::byte>()));

  if (elf.header().e_machine != elf::EM_AMDGPU)
    return unexpected(ENOEXEC, "e_machine is not EM_AMDGPU");

  unsigned abi = elf.header().e_ident[elf::EI_ABIVERSION];
  if (abi < elf::ELFABIVERSION_AMDGPU_HSA_V5)
    return unexpected(ENOEXEC, "unsupported HSA ABI %d", abi);

  uint32_t mach = elf.header().e_flags & elf::EF_AMDGPU_MACH;
  bool sramecc =
      dev.properties().capability & NodeProperties::NODE_CAP_SRAM_EDCSUPPORTED;
  bool xnack = dev.context().xnack_enabled();
  if (!elf::is_compatible(elf.header().e_flags, dev.gfx_version(), xnack,
                          sramecc))
    return unexpected(
        ENOEXEC, "ELF e_machine '%s' incompatible with device gfx%u%u%x",
        elf::get_name(mach).data(), elf::gfx_version_major(dev.gfx_version()),
        elf::gfx_version_minor(dev.gfx_version()),
        elf::gfx_version_step(dev.gfx_version()));

  auto extent = KFD_TRY(elf.load_extent());
  auto &[lo, hi, max_align] = extent;

  max_align = std::max(max_align, static_cast<uint64_t>(page_size()));
  uint64_t footprint = align_up(hi - lo, max_align);
  if (footprint > UINT32_MAX)
    return unexpected(EFBIG, "image footprint %lu exceeds 4 GiB",
                      static_cast<unsigned long>(footprint));

  // Allocate GPU VRAM memory for the loaded image to reside in.
  auto img = KFD_TRY(
      Buffer::allocate(dev, static_cast<size_t>(footprint), MemType::VRAM,
                       MemFlags::WRITABLE | MemFlags::EXECUTABLE));
  KFD_CHECK(img.map(dev));

  // Anonymous pages are zero-filled by the kernel, so BSS tails and
  // inter-segment gaps are already correct after the memcpy loop.
  auto staging = KFD_TRY(MappedRegion::create(static_cast<size_t>(footprint)));

  auto pinned =
      KFD_TRY(Buffer::pin(dev, staging.data(), static_cast<size_t>(footprint)));
  KFD_CHECK(pinned.map(dev));

  // Copy the program header values into the staging buffer.
  for (const auto &phdr : elf.phdrs()) {
    if (phdr.p_type != elf::PT_LOAD || phdr.p_filesz == 0)
      continue;
    auto seg = KFD_TRY(elf.segment_data(phdr));
    uint64_t offset = phdr.p_vaddr - lo;
    if (offset + phdr.p_filesz > footprint)
      return unexpected(ERANGE,
                        "ELF segment at 0x%lx+0x%lx overflows "
                        "staging buffer of %lu bytes",
                        static_cast<unsigned long>(offset),
                        static_cast<unsigned long>(phdr.p_filesz),
                        static_cast<unsigned long>(footprint));
    std::memcpy(static_cast<char *>(staging.data()) + offset, seg.data(),
                static_cast<size_t>(phdr.p_filesz));
  }

  // The AMDGPU ABI exposes a single R_AMDGPU_RELATIVE64 dynamic relocation. We
  // need to relocate these by simply adding the VRAM base address to all these
  // symbols listed in the dynamic relocation section.
  uint64_t load_bias = reinterpret_cast<uintptr_t>(img.data()) - lo;
  for (const auto &phdr : elf.phdrs()) {
    if (phdr.p_type != elf::PT_DYNAMIC)
      continue;
    auto dyn_data = elf.segment_data(phdr);
    if (!dyn_data)
      break;
    auto dyns =
        std::span(reinterpret_cast<const elf::Elf64_Dyn *>(dyn_data->data()),
                  dyn_data->size() / sizeof(elf::Elf64_Dyn));

    const elf::Elf64_Rela *rela_table = nullptr;
    uint64_t rela_size = 0;
    for (const auto &dyn : dyns) {
      if (dyn.d_tag == elf::DT_NULL)
        break;
      if (dyn.d_tag == elf::DT_RELA)
        rela_table = reinterpret_cast<const elf::Elf64_Rela *>(
            static_cast<const char *>(staging.data()) + dyn.d_un.d_ptr - lo);
      else if (dyn.d_tag == elf::DT_RELASZ)
        rela_size = dyn.d_un.d_val;
    }

    if (rela_table && rela_size) {
      size_t count = rela_size / sizeof(elf::Elf64_Rela);
      for (size_t i = 0; i < count; ++i) {
        const auto &rel = rela_table[i];
        if (rel.getType() == elf::R_AMDGPU_RELATIVE64) {
          uint64_t value = load_bias + static_cast<uint64_t>(rel.r_addend);
          uint64_t off = rel.r_offset - lo;
          if (off + sizeof(uint64_t) <= footprint)
            std::memcpy(static_cast<char *>(staging.data()) + off, &value,
                        sizeof(uint64_t));
        }
      }
    }
    break;
  }

  auto sig = KFD_TRY(Signal::create(dev.context(), /*initial=*/2));
  // SDMA transfer of the entire footprint from staging into VRAM.
  KFD_CHECK(sdma.copy_linear(img.data(), staging.data(), footprint));
  KFD_CHECK(sdma.signal(sig));

  // Invalidate instruction and data caches so the CUs fetch fresh code
  // from VRAM rather than stale cache lines.
  KFD_CHECK(compute.wait(sig, Condition::EQ, /*value=*/1));
  KFD_CHECK(compute.acquire_mem());
  KFD_CHECK(compute.signal(sig));

  // Wait on all pending operations to complete.
  KFD_CHECK(sig.wait(Condition::EQ, /*value=*/0, UINT64_MAX));

  return Executable(std::move(region), elf, std::move(img), lo);
}

std::expected<std::span<std::byte>, Error>
Executable::symbol(std::string_view name) const {
  auto sym = KFD_TRY(elf.find_symbol(name));
  if (!sym)
    return unexpected(ENOENT, "symbol '%.*s' not found",
                      static_cast<int>(name.size()), name.data());

  auto *ptr =
      static_cast<std::byte *>(image.data()) + sym->st_value - base_vaddr;
  return std::span{ptr, sym->st_size};
}

std::expected<Kernel, Error> Executable::kernel(std::string_view name) const {
  auto sym = KFD_TRY(elf.find_symbol(name));
  if (!sym)
    return unexpected(ENOENT, "symbol '%.*s' not found",
                      static_cast<int>(name.size()), name.data());
  if (sym->st_size < sizeof(abi::KernelDescriptor))
    return unexpected(EINVAL, "symbol '%.*s' too small for type",
                      static_cast<int>(name.size()), name.data());
  if (sym->getType() != elf::STT_OBJECT || !name.ends_with(".kd"))
    return unexpected(EINVAL, "symbol '%.*s' not a kernel descriptor",
                      static_cast<int>(name.size()), name.data());

  void *address =
      static_cast<std::byte *>(image.data()) + sym->st_value - base_vaddr;
  auto descriptor = KFD_TRY(elf.symbol_address(*sym));

  const abi::KernelDescriptor *kd =
      reinterpret_cast<const abi::KernelDescriptor *>(descriptor);
  Kernel kernel{
      .descriptor = kd,
      .address = reinterpret_cast<void *>(kd->kernel_code_entry_byte_offset +
                                          reinterpret_cast<intptr_t>(address)),
  };
  return kernel;
}

} // namespace kfd
