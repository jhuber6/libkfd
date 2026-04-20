//===-- libkfd/detail/small_vector.h ----------------------------*- C++ -*-===//
//
// Minimal vector interface with small storage optimizations. Mimics the LLVM
// ADT interface.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_SMALL_VECTOR_H
#define LIBKFD_DETAIL_SMALL_VECTOR_H

#include "libkfd/error.h"

#include <cstddef>
#include <cstdlib>
#include <new>
#include <utility>

namespace kfd::detail {

template <typename T, size_t N> class SmallVector {
public:
  SmallVector() : ptr(inline_ptr()), count(0), cap(N) {}

  ~SmallVector() {
    destroy_all();
    if (!is_inline())
      std::free(ptr);
  }

  SmallVector(const SmallVector &) = delete;
  SmallVector &operator=(const SmallVector &) = delete;

  SmallVector(SmallVector &&other)
      : count(other.count), cap(other.is_inline() ? N : other.cap) {
    if (other.is_inline()) {
      ptr = inline_ptr();
      for (size_t i = 0; i < count; ++i) {
        ::new (ptr + i) T(std::move(other.ptr[i]));
        other.ptr[i].~T();
      }
    } else {
      ptr = other.ptr;
    }
    other.ptr = other.inline_ptr();
    other.count = 0;
    other.cap = N;
  }

  SmallVector &operator=(SmallVector &&other) {
    if (this == &other)
      return *this;
    destroy_all();
    if (!is_inline())
      std::free(ptr);
    if (other.is_inline()) {
      ptr = inline_ptr();
      for (size_t i = 0; i < other.count; ++i) {
        ::new (ptr + i) T(std::move(other.ptr[i]));
        other.ptr[i].~T();
      }
      cap = N;
    } else {
      ptr = other.ptr;
      cap = other.cap;
    }
    count = other.count;
    other.ptr = other.inline_ptr();
    other.count = 0;
    other.cap = N;
    return *this;
  }

  [[nodiscard]] std::expected<void, Error> push_back(const T &value) {
    if (count == cap) {
      if (auto r = grow(); !r)
        return r;
    }
    ::new (ptr + count) T(value);
    ++count;
    return {};
  }

  [[nodiscard]] std::expected<void, Error> push_back(T &&value) {
    if (count == cap) {
      if (auto r = grow(); !r)
        return r;
    }
    ::new (ptr + count) T(std::move(value));
    ++count;
    return {};
  }

  template <typename... Args>
  [[nodiscard]] std::expected<T *, Error> emplace_back(Args &&...args) {
    if (count == cap)
      KFD_CHECK(grow());
    T *p = ::new (ptr + count) T(std::forward<Args>(args)...);
    ++count;
    return p;
  }

  void pop_back() {
    ptr[count - 1].~T();
    --count;
  }

  [[nodiscard]] std::expected<void, Error> reserve(size_t n) {
    if (n <= cap)
      return {};
    size_t new_cap = n;
    T *buf = static_cast<T *>(std::malloc(new_cap * sizeof(T)));
    if (!buf)
      return unexpected(ENOMEM, "SmallVector reserve of %zu elements failed",
                        n);
    for (size_t i = 0; i < count; ++i) {
      ::new (buf + i) T(std::move(ptr[i]));
      ptr[i].~T();
    }
    if (!is_inline())
      std::free(ptr);
    ptr = buf;
    cap = new_cap;
    return {};
  }

  T &front() { return ptr[0]; }
  const T &front() const { return ptr[0]; }
  T &back() { return ptr[count - 1]; }
  const T &back() const { return ptr[count - 1]; }

  T &operator[](size_t i) { return ptr[i]; }
  const T &operator[](size_t i) const { return ptr[i]; }

  T *data() { return ptr; }
  const T *data() const { return ptr; }

  T *begin() { return ptr; }
  T *end() { return ptr + count; }
  const T *begin() const { return ptr; }
  const T *end() const { return ptr + count; }

  size_t size() const { return count; }
  size_t capacity() const { return cap; }
  bool empty() const { return count == 0; }

  [[nodiscard]] std::expected<void, Error> resize(size_t n) {
    if (n > cap) {
      if (auto r = reserve(n); !r)
        return r;
    }
    for (size_t i = count; i < n; ++i)
      ::new (ptr + i) T{};
    for (size_t i = n; i < count; ++i)
      ptr[i].~T();
    count = n;
    return {};
  }

  void clear() {
    destroy_all();
    count = 0;
  }

private:
  T *ptr;
  size_t count;
  size_t cap;
  alignas(T) unsigned char storage[N * sizeof(T)];

  T *inline_ptr() { return reinterpret_cast<T *>(storage); }
  const T *inline_ptr() const { return reinterpret_cast<const T *>(storage); }
  bool is_inline() const { return ptr == inline_ptr(); }

  void destroy_all() {
    for (size_t i = 0; i < count; ++i)
      ptr[i].~T();
  }

  std::expected<void, Error> grow() {
    size_t new_cap = cap ? cap * 2 : 4;
    T *buf = static_cast<T *>(std::malloc(new_cap * sizeof(T)));
    if (!buf)
      return unexpected(ENOMEM, "SmallVector grow to %zu elements failed",
                        new_cap);
    for (size_t i = 0; i < count; ++i) {
      ::new (buf + i) T(std::move(ptr[i]));
      ptr[i].~T();
    }
    if (!is_inline())
      std::free(ptr);
    ptr = buf;
    cap = new_cap;
    return {};
  }
};

} // namespace kfd::detail

#endif // LIBKFD_DETAIL_SMALL_VECTOR_H
