//===-- libkfd/detail/utility.h - Parsing and string helpers ----*- C++ -*-===//
//
// Low-level helpers for string manipulation and integer parsing. These provide
// consume-style parsing similar to LLVM's StringRef.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_UTILITY_H
#define LIBKFD_DETAIL_UTILITY_H

#include "libkfd/error.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <unistd.h>

namespace kfd::detail {

inline size_t page_size() {
  static uint64_t cached = 0;
  uint64_t val = __atomic_load_n(&cached, __ATOMIC_RELAXED);
  if (__builtin_expect(val == 0, 0)) {
    val = static_cast<uint64_t>(::sysconf(_SC_PAGESIZE));
    __atomic_store_n(&cached, val, __ATOMIC_RELAXED);
  }
  return static_cast<size_t>(val);
}

inline void spin_hint() {
#if __has_builtin(__builtin_ia32_pause)
  __builtin_ia32_pause();
#elif __has_builtin(__builtin_arm_isb)
  __builtin_arm_isb(0xf);
#endif
}

inline void memory_barrier() {
#if __has_builtin(__builtin_ia32_sfence)
  __builtin_ia32_sfence();
  __atomic_thread_fence(__ATOMIC_RELEASE);
#else
  __atomic_thread_fence(__ATOMIC_RELEASE);
#endif
}

constexpr uint32_t lo(uint64_t v) { return static_cast<uint32_t>(v); }
constexpr uint32_t hi(uint64_t v) { return static_cast<uint32_t>(v >> 32); }

inline uint32_t lo(const void *p) { return lo(reinterpret_cast<uintptr_t>(p)); }
inline uint32_t hi(const void *p) { return hi(reinterpret_cast<uintptr_t>(p)); }

template <typename T> constexpr T align_up(T value, T alignment) {
#if __has_builtin(__builtin_align_up)
  return __builtin_align_up(value, alignment);
#else
  return (value + alignment - 1) & ~(alignment - 1);
#endif
}

// Equivalent to s.substr(pos, count) without the out_of_range throw path.
inline std::string_view slice(std::string_view s, size_t pos, size_t count) {
  return {s.data() + pos, count};
}

// Equivalent to s.substr(pos) without the out_of_range throw path.
inline std::string_view slice_from(std::string_view s, size_t pos) {
  return {s.data() + pos, s.size() - pos};
}

inline std::string_view drop_front(std::string_view s, size_t n = 1) {
  return n >= s.size() ? std::string_view{} : slice_from(s, n);
}

inline std::string_view drop_back(std::string_view s, size_t n = 1) {
  return n >= s.size() ? std::string_view{} : slice(s, 0, s.size() - n);
}

template <typename Pred>
inline std::string_view take_while(std::string_view s, Pred p) {
  size_t i = 0;
  while (i < s.size() && p(s[i]))
    ++i;
  return slice(s, 0, i);
}

template <typename Pred>
inline std::string_view take_until(std::string_view s, Pred p) {
  size_t i = 0;
  while (i < s.size() && !p(s[i]))
    ++i;
  return slice(s, 0, i);
}

struct SplitPair {
  std::string_view first;
  std::string_view second;
};

// Split at first occurrence of 'sep'. If not found, first = whole string,
// second = empty.
inline SplitPair split(std::string_view s, char sep) {
  auto pos = s.find(sep);
  if (pos == std::string_view::npos)
    return {.first = s, .second = {}};
  return {.first = slice(s, 0, pos), .second = slice_from(s, pos + 1)};
}

// If 's' starts with 'c', remove it and return true.
inline bool consume_front(std::string_view &s, char c) {
  if (s.empty() || s.front() != c)
    return false;
  s = slice_from(s, 1);
  return true;
}

// Return everything up to the next newline or end, advancing 's' past it.
inline std::string_view consume_line(std::string_view &s) {
  auto pos = s.find('\n');
  if (pos == std::string_view::npos) {
    auto line = s;
    s = {};
    return line;
  }
  auto line = slice(s, 0, pos);
  s = slice_from(s, pos + 1);
  return line;
}

// Consume leading decimal digits from 's', advancing it past them.
inline std::expected<uint64_t, kfd::Error>
consume_integer(std::string_view &s) {
  if (s.empty() || s.front() < '0' || s.front() > '9')
    return kfd::unexpected(EINVAL, "expected digit in '%.*s'",
                           static_cast<int>(s.size()), s.data());
  constexpr uint64_t max_div10 = UINT64_MAX / 10;
  constexpr uint64_t max_mod10 = UINT64_MAX % 10;
  uint64_t val = 0;
  size_t i = 0;
  while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
    uint64_t digit = static_cast<uint64_t>(s[i] - '0');
    if (val > max_div10 || (val == max_div10 && digit > max_mod10))
      return kfd::unexpected(ERANGE, "integer overflow in '%.*s'",
                             static_cast<int>(s.size()), s.data());
    val = val * 10 + digit;
    ++i;
  }
  s = slice_from(s, i);
  return val;
}

} // namespace kfd::detail

#endif // LIBKFD_DETAIL_UTILITY_H
