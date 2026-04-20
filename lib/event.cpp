//===-- lib/event.cpp - KFD signal event implementation ---------*- C++ -*-===//
//
// Calls the event creation ioctls. Each event is an index in the signal page.
// This is the primary mechanism for suspending threads until a device event.
//
//===----------------------------------------------------------------------===//

#include "libkfd/event.h"
#include "ioctl.h"
#include "libkfd/context.h"

#include <cerrno>
#include <utility>

static_assert(KFD_IOC_EVENT_SIGNAL == 0);

namespace kfd {

std::expected<Event, Error> Event::create(Context &ctx, uint32_t type) {
  // Standard signals reset themselves upon first being woken, other signals
  // like memory or exceptions need to be persistent so they are not dropped.
  bool is_signal = type == KFD_IOC_EVENT_SIGNAL;
  ioctl::kfd::create_event_args args{
      .event_type = type,
      .auto_reset = is_signal ? 1u : 0u,
  };
  KFD_CHECK(ioctl::call<ioctl::kfd::CREATE_EVENT>(ctx.kfd_fd(), args));

  Event ev(ctx.kfd_fd(), args.event_id, args.event_trigger_data,
           args.event_slot_index,
           is_signal ? ctx.event_slot(args.event_slot_index) : nullptr);
  return ev;
}

Event::~Event() {
  if (fd < 0)
    return;

  ioctl::kfd::destroy_event_args args{.event_id = id};
  ioctl::call<ioctl::kfd::DESTROY_EVENT>(fd, args);
}

Event::Event(Event &&other)
    : fd(std::exchange(other.fd, -1)), id(std::exchange(other.id, 0)),
      trigger(std::exchange(other.trigger, 0)),
      slot_idx(std::exchange(other.slot_idx, 0)),
      slot_addr(std::exchange(other.slot_addr, nullptr)) {}

Event &Event::operator=(Event &&other) {
  if (this != &other) {
    if (fd >= 0) {
      ioctl::kfd::destroy_event_args args{.event_id = id};
      ioctl::call<ioctl::kfd::DESTROY_EVENT>(fd, args);
    }
    fd = std::exchange(other.fd, -1);
    id = std::exchange(other.id, 0);
    trigger = std::exchange(other.trigger, 0);
    slot_idx = std::exchange(other.slot_idx, 0);
    slot_addr = std::exchange(other.slot_addr, nullptr);
  }
  return *this;
}

std::expected<void, Error> Event::wait(uint32_t timeout_ms) {
  ioctl::kfd::event_data ed{};
  ed.event_id = id;

  ioctl::kfd::wait_events_args args{
      .events_ptr = reinterpret_cast<uintptr_t>(&ed),
      .num_events = 1,
      .wait_for_all = 1,
      .timeout = timeout_ms,
  };
  if (auto r = ioctl::call<ioctl::kfd::WAIT_EVENTS>(fd, args); !r)
    return r;
  if (args.wait_result == KFD_IOC_WAIT_RESULT_TIMEOUT)
    return kfd::unexpected(ETIMEDOUT, "event %u wait timed out after %u ms", id,
                           timeout_ms);
  if (args.wait_result != KFD_IOC_WAIT_RESULT_COMPLETE)
    return kfd::unexpected(EIO, "event %u wait failed (wait_result=%u)", id,
                           args.wait_result);
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

} // namespace kfd
