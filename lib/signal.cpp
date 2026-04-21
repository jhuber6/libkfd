//===-- lib/signal.cpp - Signal implementation ------------------*- C++ -*-===//
//
// Signal creation internals, condition, and wait semantics. Signals spin until
// their values satisfy their condition. The fast path performs active polling
// before falling back to sleeping on the event. This is a blocking operation.
// Note that the events are not explicitly cleared on the fast path.
//
//===----------------------------------------------------------------------===//

#include "libkfd/signal.h"
#include "libkfd/context.h"
#include "libkfd/detail/small_vector.h"
#include "libkfd/detail/utility.h"

#include <ctime>
#include <utility>

namespace kfd {

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
                                      bool wait_for_all, uint32_t timeout_ms) {
  detail::SmallVector<Event *, 8> events;
  for (auto *s : signals)
    KFD_CHECK(events.push_back(s->event_ptr()));
  if (wait_for_all)
    return kfd::wait_all({events.data(), events.size()}, timeout_ms);
  auto r = kfd::wait_any({events.data(), events.size()}, timeout_ms);
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
  auto slot = KFD_TRY(ctx.fence_slot(ev.slot_index()));

  Signal sig(std::move(ev), slot, initial);
  return sig;
}

int Signal::kfd_fd() const { return event.kfd_fd(); }

std::expected<void, Error> Signal::reset(uint64_t value) {
  KFD_CHECK(event.reset());
  __atomic_store_n(fence, value, __ATOMIC_RELEASE);
  return {};
}

Signal::Signal(Signal &&other)
    : event(std::move(other.event)),
      fence(std::exchange(other.fence, nullptr)) {}

Signal &Signal::operator=(Signal &&other) {
  if (this != &other) {
    event = std::move(other.event);
    fence = std::exchange(other.fence, nullptr);
  }
  return *this;
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
    if (now_ns() >= spin_deadline)
      break;
    detail::spin_hint();
  }

  for (;;) {
    if (check_all(signals, cond, value))
      return {};
    uint64_t now = now_ns();
    if (now >= abs_deadline)
      return kfd::unexpected(ETIMEDOUT, "signal wait timed out");
    uint64_t remaining_ms = (abs_deadline - now + 999'999) / 1'000'000;
    uint32_t wait_ms = remaining_ms > UINT32_MAX
                           ? UINT32_MAX
                           : static_cast<uint32_t>(remaining_ms);
    if (auto r = event_wait(signals, true, wait_ms); !r) {
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
    if (now_ns() >= spin_deadline)
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
    uint64_t remaining_ms = (abs_deadline - now + 999'999) / 1'000'000;
    uint32_t wait_ms = remaining_ms > UINT32_MAX
                           ? UINT32_MAX
                           : static_cast<uint32_t>(remaining_ms);
    if (auto r = event_wait(signals, false, wait_ms); !r) {
      if (r.error().code == ETIMEDOUT)
        continue;
      return kfd::unexpected(r.error());
    }
  }
}

} // namespace kfd
