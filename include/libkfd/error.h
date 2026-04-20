//===-- libkfd/error.h - Rich error type for std::expected ------*- C++ -*-===//
//
// Lightweight error type carrying an errno code and an optional formatted
// diagnostic message.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_ERROR_H
#define LIBKFD_ERROR_H

#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <expected>

namespace kfd {

struct Error {
  int code;
  char msg[128] = {};

  Error(int c) : code(c) {}

  explicit operator int() const { return code; }

private:
  [[gnu::format(printf, 2, 0)]]
  void vformat(const char *fmt, va_list ap) {
    int n = std::vsnprintf(msg, sizeof(msg), fmt, ap);
    if (n > 0 && static_cast<size_t>(n) < sizeof(msg) - 2)
      std::snprintf(msg + n, sizeof(msg) - static_cast<size_t>(n), ": %s",
                    std::strerror(code));
  }

  friend std::unexpected<Error> vunexpected(int, const char *, va_list);
};

inline const char *strerror(const Error &e) {
  return e.msg[0] ? e.msg : std::strerror(e.code);
}
template <typename T>
inline const char *strerror(const std::expected<T, Error> &e) {
  return e.has_value() ? "(success)" : strerror(e.error());
}

inline std::unexpected<Error> unexpected(int code) {
  return std::unexpected<Error>(code);
}

inline std::unexpected<Error> unexpected(Error e) {
  return std::unexpected<Error>(std::move(e));
}

[[gnu::format(printf, 2, 0)]]
inline std::unexpected<Error> vunexpected(int code, const char *fmt,
                                          va_list ap) {
  Error e(code);
  e.vformat(fmt, ap);
  return std::unexpected<Error>(e);
}

[[gnu::format(printf, 2, 3)]]
inline std::unexpected<Error> unexpected(int code, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  auto r = vunexpected(code, fmt, ap);
  va_end(ap);
  return r;
}

} // namespace kfd

// Unwrap the value or propagate the error via early return.
#define KFD_TRY(expr)                                                          \
  ({                                                                           \
    auto &&_kfd_tmp = (expr);                                                  \
    if (!_kfd_tmp)                                                             \
      return ::kfd::unexpected(_kfd_tmp.error());                              \
    *std::move(_kfd_tmp);                                                      \
  })

// Propagate the error on failure, discarding any value.
#define KFD_CHECK(expr)                                                        \
  do {                                                                         \
    if (auto &&_kfd_tmp = (expr); !_kfd_tmp)                                   \
      return ::kfd::unexpected(_kfd_tmp.error());                              \
  } while (0)

// Unwrap the value or print the error and exit. For top-level callers.
#define KFD_EXPECT(expr)                                                       \
  ({                                                                           \
    auto &&_kfd_tmp = (expr);                                                  \
    if (!_kfd_tmp) {                                                           \
      std::fprintf(stderr, "error: %s\n", ::kfd::strerror(_kfd_tmp));          \
      std::exit(1);                                                            \
    }                                                                          \
    *std::move(_kfd_tmp);                                                      \
  })

#endif // LIBKFD_ERROR_H
