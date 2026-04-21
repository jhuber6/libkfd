//===-- lib/detail/pool_allocator.cpp ---------------------------*- C++ -*-===//
//
// First-fit free-list allocator with coalescing for managed VA pools.
//
//===----------------------------------------------------------------------===//

#include "libkfd/detail/pool_allocator.h"

#include <bit>
#include <cerrno>

namespace kfd::detail {

std::expected<PoolAllocator, Error>
PoolAllocator::create(std::span<std::byte> region, size_t alignment) {
  if (region.empty() || alignment == 0)
    return unexpected(EINVAL, "invalid region size or alignment");
  if (!std::has_single_bit(alignment))
    return unexpected(EINVAL, "alignment must be a power of two: %zu",
                      alignment);
  if (reinterpret_cast<uintptr_t>(region.data()) % alignment != 0)
    return unexpected(EINVAL, "region base is not aligned to %zu", alignment);

  SmallVector<Block, 8> blocks;
  KFD_CHECK(blocks.push_back({.offset = 0, .size = region.size()}));
  auto mtx = KFD_TRY(Box<Mutex>::create());
  return PoolAllocator(region, alignment, std::move(blocks), std::move(mtx));
}

std::expected<void *, Error> PoolAllocator::allocate(size_t size) {
  if (size == 0)
    return unexpected(EINVAL, "zero-size allocation");

  size_t rounded = (size + align - 1) & ~(align - 1);

  LockGuard lock(*mutex);

  for (size_t i = 0; i < free_list.size(); ++i) {
    Block &blk = free_list[i];
    if (blk.size < rounded)
      continue;

    size_t offset = blk.offset;
    if (blk.size == rounded) {
      for (size_t j = i; j + 1 < free_list.size(); ++j)
        free_list[j] = free_list[j + 1];
      free_list.pop_back();
    } else {
      blk.offset += rounded;
      blk.size -= rounded;
    }

    return static_cast<void *>(managed.data() + offset);
  }

  return unexpected(ENOMEM, "exhausted for size %zu (rounded %zu)", size,
                    rounded);
}

std::expected<void, Error> PoolAllocator::deallocate(void *ptr, size_t size) {
  if (!ptr || size == 0)
    return {};

  auto *base = managed.data();
  auto *p = static_cast<std::byte *>(ptr);
  if (p < base || p >= base + managed.size())
    return unexpected(EINVAL, "deallocate %p outside managed region [%p, %p)",
                      ptr, static_cast<void *>(base),
                      static_cast<void *>(base + managed.size()));

  size_t offset = static_cast<size_t>(p - base);
  size_t rounded = (size + align - 1) & ~(align - 1);

  LockGuard lock(*mutex);

  // Find insertion point to keep the list sorted by offset.
  size_t pos = 0;
  while (pos < free_list.size() && free_list[pos].offset < offset)
    ++pos;

  KFD_CHECK(free_list.emplace_back());
  for (size_t i = free_list.size() - 1; i > pos; --i)
    free_list[i] = free_list[i - 1];
  free_list[pos] = {.offset = offset, .size = rounded};

  // Coalesce with the right neighbor.
  if (pos + 1 < free_list.size() &&
      free_list[pos].offset + free_list[pos].size ==
          free_list[pos + 1].offset) {
    free_list[pos].size += free_list[pos + 1].size;
    for (size_t i = pos + 1; i + 1 < free_list.size(); ++i)
      free_list[i] = free_list[i + 1];
    free_list.pop_back();
  }

  // Coalesce with the left neighbor.
  if (pos > 0 && free_list[pos - 1].offset + free_list[pos - 1].size ==
                     free_list[pos].offset) {
    free_list[pos - 1].size += free_list[pos].size;
    for (size_t i = pos; i + 1 < free_list.size(); ++i)
      free_list[i] = free_list[i + 1];
    free_list.pop_back();
  }

  return {};
}

} // namespace kfd::detail
