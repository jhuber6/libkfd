//===-- lib/event.cpp - KFD signal event implementation ---------*- C++ -*-===//
//
// Calls the event creation ioctls. Each event is an index in the signal page.
// This is the primary mechanism for suspending threads until a device event.
//
//===----------------------------------------------------------------------===//

#include "libkfd/event.h"
#include "ioctl.h"
#include "libkfd/context.h"
#include "libkfd/detail/small_vector.h"
#include "libkfd/detail/utility.h"

#include <cerrno>
#include <cstring>
#include <utility>

static_assert(static_cast<uint32_t>(kfd::EventType::SIGNAL) ==
              KFD_IOC_EVENT_SIGNAL);
static_assert(static_cast<uint32_t>(kfd::EventType::HW_EXCEPTION) ==
              KFD_IOC_EVENT_HW_EXCEPTION);
static_assert(static_cast<uint32_t>(kfd::EventType::MEMORY) ==
              KFD_IOC_EVENT_MEMORY);

namespace kfd {

namespace {

EventData make_event_data(const ioctl::kfd::event_data &ed) {
  return EventData{
      .memory_fault =
          {
              .va = ed.memory_exception_data.va,
              .gpu_id = ed.memory_exception_data.gpu_id,
              .error_type = ed.memory_exception_data.ErrorType,
              .not_present = ed.memory_exception_data.failure.NotPresent,
              .read_only = ed.memory_exception_data.failure.ReadOnly,
              .no_execute = ed.memory_exception_data.failure.NoExecute,
              .imprecise = ed.memory_exception_data.failure.imprecise,
          },
      .hw_exception =
          {
              .gpu_id = ed.hw_exception_data.gpu_id,
              .reset_type = ed.hw_exception_data.reset_type,
              .reset_cause = ed.hw_exception_data.reset_cause,
              .memory_lost = ed.hw_exception_data.memory_lost,
          },
  };
}

} // namespace

std::expected<Event, Error> Event::create(Context &ctx, EventType type) {
  auto raw = static_cast<uint32_t>(type);
  bool is_signal = type == EventType::SIGNAL;
  ioctl::kfd::create_event_args args{
      .event_type = raw,
      .auto_reset = is_signal ? 1u : 0u,
  };
  int kfd_fd = ctx.kfd_fd();
  KFD_CHECK(ioctl::call<ioctl::kfd::CREATE_EVENT>(kfd_fd, args));

  uint64_t *slot = nullptr;
  if (is_signal)
    slot = KFD_TRY(ctx.event_slot(args.event_slot_index));
  return Event(kfd_fd, args.event_id, args.event_trigger_data,
               args.event_slot_index, slot);
}

Event::~Event() {
  if (fd < 0)
    return;

  ioctl::kfd::destroy_event_args args{.event_id = id};
  KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_EVENT>(fd, args));
}

int Event::kfd_fd() const { return fd; }

Event::Event(Event &&other)
    : fd(std::exchange(other.fd, -1)), id(std::exchange(other.id, 0)),
      trigger(std::exchange(other.trigger, 0)),
      slot_idx(std::exchange(other.slot_idx, 0)),
      slot_addr(std::exchange(other.slot_addr, nullptr)) {}

Event &Event::operator=(Event &&other) {
  if (this != &other) {
    if (fd >= 0) {
      ioctl::kfd::destroy_event_args args{.event_id = id};
      KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_EVENT>(fd, args));
    }
    fd = std::exchange(other.fd, -1);
    id = std::exchange(other.id, 0);
    trigger = std::exchange(other.trigger, 0);
    slot_idx = std::exchange(other.slot_idx, 0);
    slot_addr = std::exchange(other.slot_addr, nullptr);
  }
  return *this;
}

std::expected<void, Error> Event::wait(uint64_t timeout_ns) {
  uint32_t timeout_ms = static_cast<uint32_t>(timeout_ns / 1'000'000);
  ioctl::kfd::event_data ed{};
  ed.event_id = id;

  ioctl::kfd::wait_events_args args{
      .events_ptr = reinterpret_cast<uintptr_t>(&ed),
      .num_events = 1,
      .wait_for_all = 1,
      .timeout = timeout_ms,
  };
  KFD_CHECK(ioctl::call<ioctl::kfd::WAIT_EVENTS>(fd, args));
  if (args.wait_result == KFD_IOC_WAIT_RESULT_TIMEOUT)
    return kfd::unexpected(ETIMEDOUT, "event %u wait timed out", id);
  if (args.wait_result != KFD_IOC_WAIT_RESULT_COMPLETE)
    return kfd::unexpected(EIO, "event %u wait failed (wait_result=%u)", id,
                           args.wait_result);
  event_data = make_event_data(ed);
  return {};
}

std::expected<void, Error> Event::reset() {
  ioctl::kfd::reset_event_args args{.event_id = id};
  return ioctl::call<ioctl::kfd::RESET_EVENT>(fd, args);
}

std::expected<void, Error> Event::signal() {
  ioctl::kfd::set_event_args args{.event_id = id};
  return ioctl::call<ioctl::kfd::SET_EVENT>(fd, args);
}

namespace {

std::expected<detail::SmallVector<ioctl::kfd::event_data, 8>, Error>
do_wait_events(std::span<Event *> events, bool wait_for_all,
               uint64_t timeout_ns) {
  uint32_t timeout_ms = static_cast<uint32_t>(timeout_ns / 1'000'000);
  auto n = static_cast<uint32_t>(events.size());
  detail::SmallVector<ioctl::kfd::event_data, 8> eds;
  KFD_CHECK(eds.resize(n));
  for (uint32_t i = 0; i < n; ++i) {
    eds[i] = {};
    eds[i].event_id = events[i]->event_id();
  }

  int fd = events.front()->kfd_fd();
  ioctl::kfd::wait_events_args args{
      .events_ptr = reinterpret_cast<uintptr_t>(eds.data()),
      .num_events = n,
      .wait_for_all = wait_for_all,
      .timeout = timeout_ms,
  };
  KFD_CHECK(ioctl::call<ioctl::kfd::WAIT_EVENTS>(fd, args));
  if (args.wait_result == KFD_IOC_WAIT_RESULT_TIMEOUT)
    return kfd::unexpected(ETIMEDOUT, "event wait timed out after %u ms",
                           timeout_ms);
  if (args.wait_result != KFD_IOC_WAIT_RESULT_COMPLETE)
    return kfd::unexpected(EIO, "event wait failed (wait_result=%u)",
                           args.wait_result);
  return eds;
}

} // namespace

std::expected<void, Error> wait_all(std::span<Event *> events,
                                    uint64_t timeout_ns) {
  if (events.empty())
    return {};
  auto eds = KFD_TRY(do_wait_events(events, true, timeout_ns));
  for (uint32_t i = 0; i < static_cast<uint32_t>(events.size()); ++i)
    events[i]->event_data = make_event_data(eds[i]);
  return {};
}

std::expected<size_t, Error> wait_any(std::span<Event *> events,
                                      uint64_t timeout_ns) {
  if (events.empty())
    return kfd::unexpected(EINVAL, "wait_any called with no events");

  auto eds = KFD_TRY(do_wait_events(events, false, timeout_ns));
  for (uint32_t i = 0; i < static_cast<uint32_t>(events.size()); ++i)
    events[i]->event_data = make_event_data(eds[i]);

  for (size_t i = 0; i < events.size(); ++i) {
    if (eds[i].signal_event_data.last_event_age ||
        eds[i].hw_exception_data.gpu_id)
      return i;
  }
  return 0;
}

} // namespace kfd
