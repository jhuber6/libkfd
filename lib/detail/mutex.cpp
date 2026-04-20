//===-- lib/detail/mutex.cpp - Futex slow paths -----------------*- C++ -*-===//
//
// Out-of-line contended lock/unlock paths. These issue futex(2) syscalls and
// are not expected to be hot.
//
//===----------------------------------------------------------------------===//

#include "libkfd/detail/mutex.h"

#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>

static_assert(FUTEX_WAIT_PRIVATE == 0x80);
static_assert(FUTEX_WAKE_PRIVATE == 0x81);

namespace kfd::detail {

static constexpr unsigned SPIN_LIMIT = 64;

void Mutex::lock_contended(uint32_t expected) {
  for (unsigned i = 0; i < SPIN_LIMIT; ++i) {
    if (expected == CONTENDED)
      break;

    if (expected == UNLOCKED) {
      if (__atomic_compare_exchange_n(&state, &expected, LOCKED, false,
                                      __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
        return;
      continue;
    }

    spin_hint();
    expected = __atomic_load_n(&state, __ATOMIC_RELAXED);
  }

  if (expected != CONTENDED)
    expected = __atomic_exchange_n(&state, CONTENDED, __ATOMIC_ACQUIRE);

  while (expected != UNLOCKED) {
    ::syscall(SYS_futex, &state, FUTEX_WAIT_PRIVATE, CONTENDED, nullptr,
              nullptr, 0);
    expected = __atomic_exchange_n(&state, CONTENDED, __ATOMIC_ACQUIRE);
  }
}

void Mutex::unlock_contended() {
  ::syscall(SYS_futex, &state, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0);
}

} // namespace kfd::detail
