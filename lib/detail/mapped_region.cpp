//===-- lib/detail/mapped_region.cpp - mmap/munmap RAII wrapper -*- C++ -*-===//
//
// Simple RAII wrappers around Linux pages.
//
//===----------------------------------------------------------------------===//

#include "libkfd/detail/mapped_region.h"

#include <cerrno>
#include <cstdint>
#include <sys/mman.h>
#include <utility>

namespace kfd::detail {

namespace {

std::expected<void, Error> checked_munmap(void *addr, size_t len) {
  if (::munmap(addr, len) != 0)
    return kfd::unexpected(errno, "munmap failed: %p, %zu", addr, len);
  return {};
}

} // namespace

std::expected<MappedRegion, Error> MappedRegion::create(size_t length, int prot,
                                                        int flags) {
  void *addr = ::mmap(nullptr, length, prot, flags, -1, 0);
  if (addr == MAP_FAILED)
    return kfd::unexpected(errno, "mmap anon %zu bytes (prot=0x%x flags=0x%x)",
                           length, prot, flags);
  return MappedRegion(addr, length);
}

std::expected<MappedRegion, Error> MappedRegion::reserve(size_t length,
                                                         void *addr) {
  int flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE;
  if (addr)
    flags |= MAP_FIXED;
  void *a = ::mmap(addr, length, PROT_NONE, flags, -1, 0);
  if (a == MAP_FAILED)
    return kfd::unexpected(errno, "mmap reserve %zu bytes at %p failed", length,
                           addr);
  return MappedRegion(a, length);
}

std::expected<MappedRegion, Error>
MappedRegion::reserve_aligned(size_t length, size_t alignment) {
  if (alignment == 0 || length > SIZE_MAX - (alignment - 1))
    return kfd::unexpected(
        EINVAL, "reserve_aligned: length %zu + alignment %zu overflows", length,
        alignment);
  auto region = reserve(length + alignment - 1);
  if (!region)
    return region;

  auto raw = reinterpret_cast<uintptr_t>(region->data());
  auto aligned = (raw + alignment - 1) & ~(alignment - 1);
  auto prefix = static_cast<size_t>(aligned - raw);
  auto suffix = region->size() - prefix - length;

  region->release();

  if (prefix)
    KFD_CHECK(checked_munmap(reinterpret_cast<void *>(raw), prefix));
  if (suffix)
    KFD_CHECK(
        checked_munmap(reinterpret_cast<void *>(aligned + length), suffix));

  return MappedRegion(reinterpret_cast<void *>(aligned), length);
}

std::expected<MappedRegion, Error> MappedRegion::rebind(int fd, int prot,
                                                        off_t offset) {
  void *result = ::mmap(addr, len, prot, MAP_SHARED | MAP_FIXED, fd, offset);
  if (result == MAP_FAILED)
    return kfd::unexpected(errno, "mmap fixed %zu bytes fd=%d off=%ld failed",
                           len, fd, offset);
  size_t sz = len;
  addr = nullptr;
  len = 0;
  return MappedRegion(result, sz);
}

MappedRegion::~MappedRegion() {
  if (addr)
    KFD_ASSERT(checked_munmap(addr, len));
}

MappedRegion &MappedRegion::operator=(MappedRegion &&other) {
  if (this != &other) {
    if (addr)
      KFD_ASSERT(checked_munmap(addr, len));
    addr = std::exchange(other.addr, nullptr);
    len = std::exchange(other.len, 0);
  }
  return *this;
}

bool MappedRegion::try_grow(size_t new_size) {
  if (new_size <= len)
    return true;
  void *result = ::mremap(addr, len, new_size, 0);
  if (result == MAP_FAILED)
    return false;
  len = new_size;
  return true;
}

} // namespace kfd::detail
