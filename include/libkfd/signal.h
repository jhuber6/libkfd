//===-- libkfd/signal.h - GPU synchronization signal ------------*- C++ -*-===//
//
// A Signal combines a KFD event (for interrupt-driven wakeup) with a fence
// slot in GPU-writable memory. A signal is the canonical way to signal work
// between different devices and queues. The fence value is used to indicate
// when to finish waiting while the event system allows the thread to sleep.
//
// Signal values are intended to atomically decrease as work is completed, so
// most uses will wait until they reach zero, however any atomic modification
// will work.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_SIGNAL_H
#define LIBKFD_SIGNAL_H

#include "libkfd/condition.h"
#include "libkfd/error.h"
#include "libkfd/event.h"

#include <cstdint>
#include <span>

namespace kfd {

class Context;

class Signal {
public:
  static std::expected<Signal, Error> create(Context &ctx,
                                             uint64_t initial = 1);

  ~Signal() = default;

  Signal(const Signal &) = delete;
  Signal &operator=(const Signal &) = delete;
  Signal(Signal &&other);
  Signal &operator=(Signal &&other);

  // GPU-writable fence value decremented by queues on completion.
  uint64_t *fence_addr() const { return fence; }
  // Event page slot address written to trigger the kernel interrupt.
  void *signal_addr() const { return event.signal_addr(); }

  uint32_t event_id() const { return event.event_id(); }
  uint32_t trigger_data() const { return event.trigger_data(); }
  int kfd_fd() const;

  // Clears any pending events and resets the signal value.
  std::expected<void, Error> reset(uint64_t value = 1);

  // Signal the underlying event to wake a pending wait.
  std::expected<void, Error> signal() { return event.signal(); }

  // Block until the fence value satisfies the condition against the compare
  // value, spinning first then falling back to an interrupt-driven wait.
  std::expected<void, Error> wait(Condition cond, uint64_t value,
                                  uint64_t timeout_ns,
                                  uint64_t spin_ns = 1'000'000);

  explicit operator bool() const { return static_cast<bool>(event); }

private:
  Signal(Event &&ev, uint64_t *fence, uint64_t initial)
      : event(std::move(ev)), fence(fence) {
    __atomic_store_n(this->fence, initial, __ATOMIC_RELAXED);
  }
  Event event;
  uint64_t *fence = nullptr;
};

std::expected<void, Error> wait_all(std::span<Signal *> signals, Condition cond,
                                    uint64_t value, uint64_t timeout_ns,
                                    uint64_t spin_ns = 1'000'000);

std::expected<size_t, Error> wait_any(std::span<Signal *> signals,
                                      Condition cond, uint64_t value,
                                      uint64_t timeout_ns,
                                      uint64_t spin_ns = 1'000'000);

} // namespace kfd

#endif // LIBKFD_SIGNAL_H
