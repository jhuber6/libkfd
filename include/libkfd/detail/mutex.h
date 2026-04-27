//===-- libkfd/detail/mutex.h - Spinlock mutex ------------------*- C++ -*-===//
//
// Lightweight non-recursive mutex built on a test-and-test-and-set (TTAS) spin
// loop. Contention in this library is short-lived, so a simple mutex is
// sufficient for our purposes.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_MUTEX_H
#define LIBKFD_DETAIL_MUTEX_H

#include "libkfd/detail/utility.h"

#include <cstdint>
#include <utility>

namespace kfd::detail {

class Mutex {
public:
  constexpr Mutex() = default;

  Mutex(const Mutex &) = delete;
  Mutex &operator=(const Mutex &) = delete;

  Mutex(Mutex &&other)
      : state(__atomic_load_n(&other.state, __ATOMIC_RELAXED)) {
    other.state = UNLOCKED;
  }

  Mutex &operator=(Mutex &&other) {
    if (this != &other) {
      state = __atomic_load_n(&other.state, __ATOMIC_RELAXED);
      other.state = UNLOCKED;
    }
    return *this;
  }

  void lock() {
    if (__builtin_expect(__atomic_exchange_n(&state, LOCKED, __ATOMIC_ACQUIRE),
                         0))
      lock_slow();
  }

  void unlock() { __atomic_store_n(&state, UNLOCKED, __ATOMIC_RELEASE); }

private:
  static constexpr uint32_t UNLOCKED = 0;
  static constexpr uint32_t LOCKED = 1;

  [[gnu::noinline]] void lock_slow() {
    do {
      while (__atomic_load_n(&state, __ATOMIC_RELAXED))
        spin_hint();
    } while (__atomic_exchange_n(&state, LOCKED, __ATOMIC_ACQUIRE));
  }

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
