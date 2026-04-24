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
#include <span>

namespace kfd {

class Context;

enum class EventType : uint32_t {
  SIGNAL = /*KFD_IOC_EVENT_SIGNAL=*/0,
  HW_EXCEPTION = /*KFD_IOC_EVENT_HW_EXCEPTION=*/3,
  MEMORY = /*KFD_IOC_EVENT_MEMORY=*/8,
};

struct MemoryFaultData {
  uint64_t va;
  uint32_t gpu_id;
  uint32_t error_type;
  uint32_t not_present;
  uint32_t read_only;
  uint32_t no_execute;
  uint32_t imprecise;
};

struct HWExceptionData {
  uint32_t gpu_id;
  uint32_t reset_type;
  uint32_t reset_cause;
  uint32_t memory_lost;
};

struct EventData {
  MemoryFaultData memory_fault;
  HWExceptionData hw_exception;
};

// At context initialization we register an event page with the kernel. Each
// event is assigned a slot index. To signal an event, the GPU writes any value
// other than ~0 to that slot and fires an interrupt. The interrupt handler will
// then wake any event waiting on that index (id).
class Event {
public:
  Event() = default;

  static std::expected<Event, Error> create(Context &ctx,
                                            EventType type = EventType::SIGNAL);

  ~Event();

  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  Event(Event &&other);
  Event &operator=(Event &&other);

  // These all identify the event, the kernel uses this ID to quickly find the
  // correct event to check and signal, falling back to an exhaustive search.
  // Slot index is the logical page index, while event id is internal to kfd.
  uint32_t trigger_data() const { return trigger; }
  uint32_t event_id() const { return id; }
  uint32_t slot_index() const { return slot_idx; }

  // GPU-visible address of this event's signal page slot.
  void *signal_addr() const { return slot_addr; }

  int kfd_fd() const;

  // Block until receiving the signal or timeout expires.
  std::expected<void, Error> wait(uint64_t timeout_ns = UINT64_MAX);

  // Reset to unsignaled state so the event can be waited on again.
  std::expected<void, Error> reset();

  // Signal from the CPU side (via kernel ioctl).
  std::expected<void, Error> signal();

  // Event data populated by the kernel after a wait completes. Contains
  // memory fault or HW exception information depending on the event type.
  const EventData &data() const { return event_data; }

  explicit operator bool() const { return fd >= 0; }

private:
  friend class Context;
  friend std::expected<void, Error> wait_all(std::span<Event *>, uint64_t);
  friend std::expected<size_t, Error> wait_any(std::span<Event *>, uint64_t);

  Event(int fd, uint32_t id, uint32_t trigger, uint32_t slot_idx,
        void *slot_addr)
      : fd(fd), id(id), trigger(trigger), slot_idx(slot_idx),
        slot_addr(slot_addr) {}

  int fd = -1;
  uint32_t id = 0;
  uint32_t trigger = 0;
  uint32_t slot_idx = 0;
  void *slot_addr = nullptr;
  EventData event_data{};
};

std::expected<void, Error> wait_all(std::span<Event *> events,
                                    uint64_t timeout_ns = UINT64_MAX);

std::expected<size_t, Error> wait_any(std::span<Event *> events,
                                      uint64_t timeout_ns = UINT64_MAX);

} // namespace kfd

#endif // LIBKFD_EVENT_H
