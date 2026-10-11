//===-- lib/signal.cpp - Signal implementation ------------------*- C++ -*-===//
//
// Signal creation internals, condition, and wait semantics. Signals spin until
// their values satisfy their condition. The fast path performs active polling
// before falling back to sleeping on the event. This is a blocking operation.
//
//===----------------------------------------------------------------------===//

#include "libkfd/signal.h"
#include "ioctl.h"
#include "libkfd/context.h"
#include "libkfd/detail/small_vector.h"
#include "libkfd/detail/utility.h"

#include <cerrno>
#include <cstddef>
#include <ctime>
#include <new>
#include <utility>

namespace kfd {

Signal::Slot::Slot(Event &&incoming, uint64_t initial)
    : ev(std::move(incoming)) {
  __atomic_store_n(&fence, initial, __ATOMIC_RELAXED);
  __atomic_store_n(&event_mailbox,
                   reinterpret_cast<uintptr_t>(ev.signal_addr()),
                   __ATOMIC_RELAXED);
  __atomic_store_n(&event_id_word, ev.event_id(), __ATOMIC_RELAXED);
  __atomic_store_n(&kind, int64_t{1}, __ATOMIC_RELEASE);
}

bool Signal::retire(Slot *resident) {
  if (resident == nullptr)
    return false;
  int64_t live = 1;
  if (!__atomic_compare_exchange_n(&resident->kind, &live, int64_t{0},
                                   /*weak=*/false, __ATOMIC_RELEASE,
                                   __ATOMIC_RELAXED))
    return false;
  // The slot lifetime ends before DESTROY_EVENT. The kernel can recycle the
  // index as soon as that ioctl returns.
  __atomic_store_n(&resident->owner, static_cast<Signal *>(nullptr),
                   __ATOMIC_RELEASE);
  Event ev = std::move(resident->ev);
  resident->~Slot();
  return true;
}

Signal::~Signal() { retire(std::exchange(slot, nullptr)); }

Signal::Signal(Signal &&other) noexcept
    : slot(std::exchange(other.slot, nullptr)) {
  if (slot)
    __atomic_store_n(&slot->owner, this, __ATOMIC_RELEASE);
}

Signal &Signal::operator=(Signal &&other) noexcept {
  if (this != &other) {
    retire(std::exchange(slot, nullptr));
    slot = std::exchange(other.slot, nullptr);
    if (slot)
      __atomic_store_n(&slot->owner, this, __ATOMIC_RELEASE);
  }
  return *this;
}

Signal *Signal::from_handle(uint64_t handle) {
  if (handle == 0)
    return nullptr;
  auto *resident = reinterpret_cast<Slot *>(handle);
  if (__atomic_load_n(&resident->kind, __ATOMIC_ACQUIRE) != 1)
    return nullptr;
  return __atomic_load_n(&resident->owner, __ATOMIC_ACQUIRE);
}

namespace {

bool check_condition(uint64_t current, Condition cond, uint64_t value) {
  switch (cond) {
  case Condition::LT:
    return current < value;
  case Condition::LTE:
    return current <= value;
  case Condition::EQ:
    return current == value;
  case Condition::NE:
    return current != value;
  case Condition::GTE:
    return current >= value;
  case Condition::GT:
    return current > value;
  }
  return false;
}

uint64_t now_ns() {
  struct timespec ts;
  ::clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1'000'000'000 +
         static_cast<uint64_t>(ts.tv_nsec);
}

int check_any(std::span<Signal *> signals, Condition cond, uint64_t value) {
  for (size_t i = 0; i < signals.size(); ++i)
    if (check_condition(
            __atomic_load_n(signals[i]->fence_addr(), __ATOMIC_ACQUIRE), cond,
            value))
      return static_cast<int>(i);
  return -1;
}

bool check_all(std::span<Signal *> signals, Condition cond, uint64_t value) {
  for (auto *sig : signals)
    if (!check_condition(__atomic_load_n(sig->fence_addr(), __ATOMIC_ACQUIRE),
                         cond, value))
      return false;
  return true;
}

std::expected<void, Error> event_wait(std::span<Signal *> signals,
                                      bool wait_for_all, uint64_t timeout_ns) {
  detail::SmallVector<Event *, 8> events;
  for (auto *s : signals)
    KFD_CHECK(events.push_back(&s->event()));
  if (wait_for_all)
    return kfd::wait_all({events.data(), events.size()}, timeout_ns);
  auto r = kfd::wait_any({events.data(), events.size()}, timeout_ns);
  if (!r)
    return kfd::unexpected(r.error());
  return {};
}

uint64_t add_sat(uint64_t a, uint64_t b) {
  uint64_t sum = a + b;
  return sum < a ? UINT64_MAX : sum;
}

} // namespace

std::expected<Signal, Error> Signal::create(Context &ctx, uint64_t initial) {
  auto ev = KFD_TRY(Event::create(ctx));
  auto *base = static_cast<std::byte *>(storage());
  if (base == nullptr)
    return kfd::unexpected(EINVAL, "fence page is not mapped");
  uint32_t index = ev.slot_index();
  if (index >= KFD_SIGNAL_EVENT_LIMIT)
    return kfd::unexpected(EINVAL, "signal id %u >= limit %u", index,
                           static_cast<unsigned>(KFD_SIGNAL_EVENT_LIMIT));
  auto *mem = reinterpret_cast<Slot *>(base + index * sizeof(Slot));
  if (__atomic_load_n(&mem->kind, __ATOMIC_ACQUIRE) != 0)
    return kfd::unexpected(EINVAL, "signal slot %u is still live", index);
  new (mem) Slot(std::move(ev), initial);
  return Signal(mem);
}

std::expected<void, Error> Signal::reset(uint64_t value) {
  __atomic_store_n(fence_addr(), value, __ATOMIC_RELEASE);
  return {};
}

std::expected<void, Error> Signal::wait(Condition cond, uint64_t value,
                                        uint64_t timeout_ns, uint64_t spin_ns) {
  Signal *self = this;
  return wait_all({&self, 1}, cond, value, timeout_ns, spin_ns);
}

std::expected<void, Error> wait_all(std::span<Signal *> signals, Condition cond,
                                    uint64_t value, uint64_t timeout_ns,
                                    uint64_t spin_ns) {
  if (signals.empty())
    return {};

  uint64_t start = now_ns();
  uint64_t spin_deadline = add_sat(start, spin_ns);
  uint64_t abs_deadline = add_sat(start, timeout_ns);

  for (;;) {
    if (check_all(signals, cond, value))
      return {};
    if (timeout_ns < UINT64_MAX && now_ns() >= spin_deadline)
      break;
    if (signals.size() == 1) {
      // MONITORX watches the whole cacheline, so monitoring the low dword of
      // the 64-bit fence still wakes on any write to it.
      auto *fence32 =
          reinterpret_cast<uint32_t *>(signals.front()->fence_addr());
      detail::spin_wait(fence32, __atomic_load_n(fence32, __ATOMIC_RELAXED));
    } else {
      detail::spin_hint();
    }
  }

  for (;;) {
    if (check_all(signals, cond, value))
      return {};
    uint64_t now = now_ns();
    if (now >= abs_deadline)
      return kfd::unexpected(ETIMEDOUT, "signal wait timed out");
    if (auto r = event_wait(signals, true, abs_deadline - now); !r) {
      if (r.error().code == ETIMEDOUT)
        continue;
      return r;
    }
  }
}

std::expected<size_t, Error> wait_any(std::span<Signal *> signals,
                                      Condition cond, uint64_t value,
                                      uint64_t timeout_ns, uint64_t spin_ns) {
  if (signals.empty())
    return kfd::unexpected(EINVAL, "wait_any called with no signals");

  uint64_t start = now_ns();
  uint64_t spin_deadline = add_sat(start, spin_ns);
  uint64_t abs_deadline = add_sat(start, timeout_ns);

  for (;;) {
    int idx = check_any(signals, cond, value);
    if (idx >= 0)
      return static_cast<size_t>(idx);
    if (timeout_ns < UINT64_MAX && now_ns() >= spin_deadline)
      break;
    detail::spin_hint();
  }

  for (;;) {
    int idx = check_any(signals, cond, value);
    if (idx >= 0)
      return static_cast<size_t>(idx);
    uint64_t now = now_ns();
    if (now >= abs_deadline)
      return kfd::unexpected(ETIMEDOUT, "signal wait timed out");
    if (auto r = event_wait(signals, false, abs_deadline - now); !r) {
      if (r.error().code == ETIMEDOUT)
        continue;
      return kfd::unexpected(r.error());
    }
  }
}

} // namespace kfd
