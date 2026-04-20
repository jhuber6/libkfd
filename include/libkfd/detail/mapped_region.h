//===-- libkfd/detail/mapped_region.h - RAII mmap wrapper -------*- C++ -*-===//
//
// Owning handle for a memory-mapped region. Wraps mmap(2)/munmap(2) with RAII
// semantics and std::expected error reporting.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_MAPPED_REGION_H
#define LIBKFD_DETAIL_MAPPED_REGION_H

#include "libkfd/error.h"

#include <cstddef>
#include <span>
#include <sys/mman.h>
#include <utility>

namespace kfd::detail {

class MappedRegion {
public:
  MappedRegion() = default;

  // Anonymous mapping with explicit mmap protection and flags.
  static std::expected<MappedRegion, Error>
  create(size_t length, int prot = PROT_READ | PROT_WRITE,
         int flags = MAP_PRIVATE | MAP_ANONYMOUS);

  // Reserve a VA range with PROT_NONE. When addr is non-null the reservation
  // is placed at that exact address via MAP_FIXED.
  static std::expected<MappedRegion, Error> reserve(size_t length,
                                                    void *addr = nullptr);

  // Reserve an aligned VA range with PROT_NONE.
  static std::expected<MappedRegion, Error> reserve_aligned(size_t length,
                                                            size_t alignment);

  // Replace this reservation with a file-backed mapping at the same address
  // via MAP_FIXED. Consumes the old region and returns a new one.
  std::expected<MappedRegion, Error> rebind(int fd, int prot, off_t offset);

  ~MappedRegion();

  MappedRegion(const MappedRegion &) = delete;
  MappedRegion &operator=(const MappedRegion &) = delete;

  MappedRegion(MappedRegion &&other)
      : addr(std::exchange(other.addr, nullptr)),
        len(std::exchange(other.len, 0)) {}

  MappedRegion &operator=(MappedRegion &&other);

  bool try_grow(size_t new_size);

  // Relinquish ownership without unmapping. The caller assumes responsibility
  // for the lifetime of the region.
  void *release() {
    len = 0;
    return std::exchange(addr, nullptr);
  }

  void *data() const { return addr; }
  size_t size() const { return len; }

  template <typename T> T *as() const { return static_cast<T *>(addr); }

  template <typename T> std::span<T> as_span() const {
    return {static_cast<T *>(addr), len / sizeof(T)};
  }

  explicit operator bool() const { return addr != nullptr; }

private:
  explicit MappedRegion(void *addr, size_t len) : addr(addr), len(len) {}

  void *addr = nullptr;
  size_t len = 0;
};

} // namespace kfd::detail

#endif // LIBKFD_DETAIL_MAPPED_REGION_H
