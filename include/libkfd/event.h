//===-- libkfd/event.h - KFD signal events ----------------------*- C++ -*-===//
//
// RAII wrapper around KFD signal events. Events provide interrupt-driven
// GPU-to-CPU notification.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_EVENT_H
#define LIBKFD_EVENT_H

#include "libkfd/error.h"

#include <cstdint>

namespace kfd {

class Context;

// At context initialization we register an event page with the kernel. Each
// event is assigned a slot index. To signal an event, the GPU writes any value
// other than ~0 to that slot and fires an interrupt. The interrupt handler will
// then wake any event waiting on that index (id).
class Event {
public:
  Event() = default;

  static std::expected<Event, Error>
  create(Context &ctx, uint32_t type = /*KFD_IOC_EVENT_SIGNAL=*/0);

  ~Event();

  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  Event(Event &&other);
  Event &operator=(Event &&other);

  // These all identify the event, the kernel uses this ID to quickly find the
  // correct event to check and signal, falling back to an exhaustive search.
  uint32_t trigger_data() const { return trigger; }
  uint32_t event_id() const { return id; }
  uint32_t slot_index() const { return slot_idx; }

  // GPU-visible address of this event's signal page slot.
  void *signal_addr() const { return slot_addr; }

  int kfd_fd() const { return fd; }

  // Block until receiving the signal or timeout expires.
  std::expected<void, Error> wait(uint32_t timeout_ms = UINT32_MAX);

  // Reset to unsignaled state so the event can be waited on again.
  std::expected<void, Error> reset();

  // Signal from the CPU side (via kernel ioctl).
  std::expected<void, Error> signal();

  explicit operator bool() const { return fd >= 0; }

private:
  Event(int fd, uint32_t id, uint32_t trigger, uint32_t slot_idx,
        void *slot_addr)
      : fd(fd), id(id), trigger(trigger), slot_idx(slot_idx),
        slot_addr(slot_addr) {}

  int fd = -1;
  uint32_t id = 0;
  uint32_t trigger = 0;
  uint32_t slot_idx = 0;
  void *slot_addr = nullptr;
};

} // namespace kfd

#endif // LIBKFD_EVENT_H
