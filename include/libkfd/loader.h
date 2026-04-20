//===-- libkfd/loader.h - GPU code object loader ----------------*- C++ -*-===//
//
// Loads an AMDHSA ELF code object into GPU memory. Iterates the program headers
// to allocate the appropriate VRAM backing and relocate segments. The
// executable is the canonical way to look up symbols and get GPU addresses.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_LOADER_H
#define LIBKFD_LOADER_H

#include "libkfd/abi.h"
#include "libkfd/detail/elf.h"
#include "libkfd/error.h"
#include "libkfd/memory.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace kfd {

class ComputeQueue;
class Device;
class SDMAQueue;

// A representation of an executable kernel on the device. Only valid for the
// lifetime of the Executable it was obtained from.
struct Kernel {
  const abi::KernelDescriptor *descriptor;
  const void *address;

  // Kernargs must outlive the kernel they are attached to, they provide the
  // backing for the AMDGPU compute ABI.
  template <typename T>
  std::expected<Buffer, Error> make_kernargs(Device &dev,
                                             const T &explicit_args,
                                             const DispatchConfig &cfg) const {
    return make_kernargs(dev, std::as_bytes(std::span(&explicit_args, 1)), cfg);
  }
  std::expected<Buffer, Error> make_kernargs(Device &dev,
                                             const DispatchConfig &cfg) const {
    return make_kernargs(dev, {}, cfg);
  }

private:
  std::expected<Buffer, Error>
  make_kernargs(Device &dev, std::span<const std::byte> explicit_args,
                const DispatchConfig &cfg) const;
};

// An owning executable ELF object loaded into GPU VRAM memory.
class Executable {
public:
  using SymbolRange = detail::elf::SymbolRange;

  static std::expected<Executable, Error> load(Device &dev,
                                               std::span<const std::byte> image,
                                               SDMAQueue &sdma,
                                               ComputeQueue &compute);

  ~Executable() = default;

  Executable(const Executable &) = delete;
  Executable &operator=(const Executable &) = delete;
  Executable(Executable &&) = default;
  Executable &operator=(Executable &&) = default;

  std::expected<std::span<std::byte>, Error> symbol(std::string_view sym) const;
  std::expected<Kernel, Error> kernel(std::string_view sym) const;
  SymbolRange symbols() const { return elf.symbols(); }

  void *base() const { return image.data(); }
  size_t size() const { return image.size(); }
  explicit operator bool() const { return static_cast<bool>(image); }

private:
  Executable(detail::MappedRegion &&data, detail::elf::ELF64LE elf,
             Buffer image, uint64_t base_vaddr)
      : data(std::move(data)), elf(elf), image(std::move(image)),
        base_vaddr(base_vaddr) {}

  detail::MappedRegion data;
  detail::elf::ELF64LE elf;
  Buffer image;
  uint64_t base_vaddr = 0;
};

} // namespace kfd

#endif // LIBKFD_LOADER_H
