//===-- libkfd/context.h - KFD kernel context -------------------*- C++ -*-===//
//
// Interface around the /dev/kfd character device. Performs device discovery and
// necessary initialization steps.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_CONTEXT_H
#define LIBKFD_CONTEXT_H

#include "libkfd/condition.h"
#include "libkfd/detail/box.h"
#include "libkfd/detail/small_vector.h"
#include "libkfd/device.h"
#include "libkfd/error.h"

#include <cstdint>

namespace kfd {

class Event;
class Signal;
struct SignalWatcher;
struct ExceptionWatcher;

struct MemoryFaultInfo {
  // Bitmask of reasons a memory access faulted.
  enum Reason : uint32_t {
    NotPresent = 1u << 0,
    ReadOnly = 1u << 1,
    NoExecute = 1u << 2,
    Imprecise = 1u << 3,
  };
  uint64_t va;
  uint32_t reason;
  uint32_t error_type;
};

struct HardwareExceptionInfo {
  uint32_t reset_type;
  uint32_t reset_cause;
  uint32_t memory_lost;
};

struct FaultInfo {
  enum class Kind : uint32_t {
    MemoryViolation,
    HardwareException,
  };
  Kind kind;
  uint32_t gpu_id;
  union {
    MemoryFaultInfo memory;
    HardwareExceptionInfo hardware;
  };
};

// Invoked by the exception watcher on a GPU memory fault or hardware exception.
using FaultHandler = void (*)(const FaultInfo &fault, void *user_data);

// Invoked by the signal watcher on interrupt. A return value of true re-arms.
using SignalHandler = bool (*)(void *user_data);

struct VersionInfo {
  uint32_t major;
  uint32_t minor;
};

// Minimum kernel interface version we require.
inline constexpr uint32_t MIN_KFD_MAJOR = 1;
inline constexpr uint32_t MIN_KFD_MINOR = 4;

// Owns an open file descriptor to '/dev/kfd' and a Device for every GPU node
// visible in the topology.
class Context {
public:
  static std::expected<Context, Error> create();

  ~Context();

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
  Context(Context &&other);
  Context &operator=(Context &&other);

  std::expected<VersionInfo, Error> version() const;

  int kfd_fd() const { return fd; }
  // Whether GPU page fault retry (SVM) mode is active.
  bool xnack_enabled() const { return xnack; }
  size_t num_devices() const { return nodes.size(); }

  std::span<Device> devices() { return nodes; }
  std::expected<Device *, Error> device(size_t i);

  // Register a handler with the context's watcher threads.
  std::expected<void, Error> register_handler(Signal &sig, Condition cond,
                                              uint64_t value, SignalHandler cb,
                                              void *user_data);
  std::expected<void, Error> register_handler(FaultHandler cb, void *user_data);

private:
  friend class QueueBase;
  friend class Event;
  friend class Signal;

  std::expected<uint64_t *, Error> event_slot(uint32_t id);
  std::expected<uint64_t *, Error> fence_slot(uint32_t id);

  std::expected<void, Error> register_handler(Event &event, SignalHandler cb,
                                              void *user_data = nullptr);
  std::expected<void, Error> unregister_handler(Event &event);

  explicit Context(int fd, bool xnack, detail::SmallVector<Device, 4> devices);
  int fd;
  bool xnack = false;
  detail::SmallVector<Device, 4> nodes;
  detail::Box<SignalWatcher> signal_watcher;
  detail::Box<ExceptionWatcher> exception_watcher;
};

} // namespace kfd

#endif // LIBKFD_CONTEXT_H
