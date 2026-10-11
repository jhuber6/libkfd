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
  // ABI compatible with the HSA standard signal as consumed by the AQL packet
  // processor. Other consumers only use the fence and event values.
  //
  // Reference: rocr-runtime inc/amd_hsa_signal.h
  struct alignas(64) Slot {
    int64_t kind = 0;
    uint64_t fence = 0;
    uint64_t event_mailbox = 0;
    uint32_t event_id_word = 0;
    uint32_t reserved = 0;
    uint64_t unused[4] = {};
    // Past the firmware prefix. A move of the owner rewrites it.
    Signal *owner = nullptr;
    Event ev;

    Slot(Event &&ev, uint64_t initial);
  };

  // Slot memory placed into a fence page, indexed according to the event ID.
  Slot *slot = nullptr;

  explicit Signal(Slot *slot) : slot(slot) {
    if (slot)
      __atomic_store_n(&slot->owner, this, __ATOMIC_RELEASE);
  }
  static void *storage();
  static constexpr size_t resident_bytes() { return sizeof(Slot); }
  static bool retire(Slot *slot);

  friend class Context;

public:
  static std::expected<Signal, Error> create(Context &ctx,
                                             uint64_t initial = 1);

  static Signal *from_handle(uint64_t handle);

  ~Signal();

  Signal(const Signal &) = delete;
  Signal &operator=(const Signal &) = delete;
  Signal(Signal &&other) noexcept;
  Signal &operator=(Signal &&other) noexcept;

  // GPU-writable fence value decremented by queues on completion.
  uint64_t *fence_addr() const { return slot ? &slot->fence : nullptr; }

  // Address stored in an AQL packet's completion_signal field. The command
  // processor treats it as the base of the signal and decrements the fence
  // word eight bytes above it.
  uint64_t completion_handle() const {
    return slot ? reinterpret_cast<uintptr_t>(slot) : 0;
  }
  // Event page slot address written to trigger the kernel interrupt.
  void *signal_addr() const { return slot ? slot->ev.signal_addr() : nullptr; }

  uint32_t event_id() const { return slot ? slot->ev.event_id() : 0; }
  uint32_t trigger_data() const { return slot ? slot->ev.trigger_data() : 0; }
  Event &event() { return slot->ev; }

  // Return the current value of the signal.
  uint64_t value() const {
    return slot ? __atomic_load_n(&slot->fence, __ATOMIC_RELAXED) : 0;
  }

  // Resets the signal value.
  std::expected<void, Error> reset(uint64_t value = 1);

  // Signal the underlying event to wake a pending wait.
  std::expected<void, Error> signal() {
    return slot ? slot->ev.signal() : kfd::unexpected(EINVAL, "empty signal");
  }

  // Block until the fence value satisfies the condition against the compare
  // value, spinning first then falling back to an interrupt-driven wait.
  std::expected<void, Error> wait(Condition cond, uint64_t value,
                                  uint64_t timeout_ns,
                                  uint64_t spin_ns = 1'000'000);

  explicit operator bool() const { return slot && static_cast<bool>(slot->ev); }
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
