//===-- libkfd/detail/pool_allocator.h - VA pool sub-allocator --*- C++ -*-===//
//
// First-fit free-list allocator over an externally-owned VA range. Interface is
// thread-safe.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_POOL_ALLOCATOR_H
#define LIBKFD_DETAIL_POOL_ALLOCATOR_H

#include "libkfd/detail/mutex.h"
#include "libkfd/detail/small_vector.h"
#include "libkfd/error.h"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

namespace kfd::detail {

class PoolAllocator {
public:
  static std::expected<PoolAllocator, Error> create(std::span<std::byte> region,
                                                    size_t alignment);

  std::expected<void *, Error> allocate(size_t size);
  [[nodiscard]] std::expected<void, Error> deallocate(void *ptr, size_t size);

  std::span<std::byte> region() const { return managed; }
  size_t alignment() const { return align; }

  PoolAllocator() = default;
  ~PoolAllocator() = default;

  PoolAllocator(const PoolAllocator &) = delete;
  PoolAllocator &operator=(const PoolAllocator &) = delete;
  PoolAllocator(PoolAllocator &&) = default;
  PoolAllocator &operator=(PoolAllocator &&) = default;

  explicit operator bool() const { return align != 0; }

private:
  struct Block {
    size_t offset;
    size_t size;
  };

  PoolAllocator(std::span<std::byte> region, size_t alignment,
                SmallVector<Block, 8> &&blocks, Mutex mtx)
      : mutex(std::move(mtx)), managed(region), align(alignment),
        free_list(std::move(blocks)) {}

  Mutex mutex;
  std::span<std::byte> managed;
  size_t align = 0;
  SmallVector<Block, 8> free_list;
};

} // namespace kfd::detail

#endif // LIBKFD_DETAIL_POOL_ALLOCATOR_H
