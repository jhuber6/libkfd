//===-- libkfd/detail/box.h - Heap-allocated owning pointer -----*- C++ -*-===//
//
// Move-only owning pointer with RAII semantics. Failable unique-ptr equivalent.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_BOX_H
#define LIBKFD_DETAIL_BOX_H

#include "libkfd/error.h"

#include <cstdlib>
#include <expected>
#include <new>
#include <utility>

namespace kfd::detail {

template <typename T> class Box {
public:
  template <typename... Args>
  static std::expected<Box, Error> create(Args &&...args) {
    void *mem = std::aligned_alloc(alignof(T), sizeof(T));
    if (!mem)
      return unexpected(ENOMEM, "Box allocation of %zu bytes failed",
                        sizeof(T));
    return Box(::new (mem) T(std::forward<Args>(args)...));
  }

  Box() = default;
  ~Box() { reset(); }

  Box(const Box &) = delete;
  Box &operator=(const Box &) = delete;

  Box(Box &&other) : ptr(other.ptr) { other.ptr = nullptr; }
  Box &operator=(Box &&other) {
    if (this != &other) {
      reset();
      ptr = other.ptr;
      other.ptr = nullptr;
    }
    return *this;
  }

  T &operator*() { return *ptr; }
  const T &operator*() const { return *ptr; }
  T *operator->() { return ptr; }
  const T *operator->() const { return ptr; }
  T *get() { return ptr; }
  const T *get() const { return ptr; }
  explicit operator bool() const { return ptr != nullptr; }

private:
  explicit Box(T *p) : ptr(p) {}

  void reset() {
    if (ptr) {
      ptr->~T();
      std::free(ptr);
      ptr = nullptr;
    }
  }

  T *ptr = nullptr;
};

} // namespace kfd::detail

#endif // LIBKFD_DETAIL_BOX_H
