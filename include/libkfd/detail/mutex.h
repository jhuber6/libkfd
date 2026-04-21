//===-- libkfd/detail/mutex.h - Futex-based mutex ---------------*- C++ -*-===//
//
// Lightweight non-recursive mutex built on Linux futex(2) and compiler atomic
// builtins. No pthread dependency. Based on Ulrich Drepper's futex method.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_MUTEX_H
#define LIBKFD_DETAIL_MUTEX_H

#include "libkfd/detail/utility.h"

#include <cstdint>

namespace kfd::detail {

class Mutex {
public:
  constexpr Mutex() = default;

  Mutex(const Mutex &) = delete;
  Mutex &operator=(const Mutex &) = delete;
  Mutex(Mutex &&) = delete;
  Mutex &operator=(Mutex &&) = delete;

  void lock() {
    uint32_t expected = UNLOCKED;
    if (__atomic_compare_exchange_n(&state, &expected, LOCKED, false,
                                    __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
      return;

    lock_contended(expected);
  }

  void unlock() {
    if (__atomic_exchange_n(&state, UNLOCKED, __ATOMIC_RELEASE) == CONTENDED)
      unlock_contended();
  }

private:
  static constexpr uint32_t UNLOCKED = 0;
  static constexpr uint32_t LOCKED = 1;
  static constexpr uint32_t CONTENDED = 2;

  [[gnu::cold]] void lock_contended(uint32_t expected);
  [[gnu::cold]] void unlock_contended();

  uint32_t state = UNLOCKED;
};

class LockGuard {
public:
  explicit LockGuard(Mutex &m) : mtx(m) { mtx.lock(); }
  ~LockGuard() { mtx.unlock(); }

  LockGuard(const LockGuard &) = delete;
  LockGuard &operator=(const LockGuard &) = delete;

private:
  Mutex &mtx;
};

} // namespace kfd::detail

#endif // LIBKFD_DETAIL_MUTEX_H
