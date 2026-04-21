//===-- libkfd/context.h - KFD kernel context -------------------*- C++ -*-===//
//
// Interface around the /dev/kfd character device. Performs device discovery and
// necessary initialization steps.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_CONTEXT_H
#define LIBKFD_CONTEXT_H

#include "libkfd/detail/small_vector.h"
#include "libkfd/device.h"
#include "libkfd/error.h"

#include <cstdint>

namespace kfd {

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

private:
  friend class QueueBase;
  friend class Event;
  friend class Signal;

  std::expected<uint64_t *, Error> event_slot(uint32_t id);
  std::expected<uint64_t *, Error> fence_slot(uint32_t id);

  void register_queue_error(uint32_t event_id, volatile uint64_t *payload,
                            uint32_t queue_id, uint32_t gpu_id);
  void unregister_queue_error(uint32_t event_id);

  explicit Context(int fd, bool xnack, detail::SmallVector<Device, 4> devices)
      : fd(fd), xnack(xnack), nodes(std::move(devices)) {}
  int fd;
  bool xnack = false;
  detail::SmallVector<Device, 4> nodes;
  void *fault_watcher = nullptr;
};

} // namespace kfd

#endif // LIBKFD_CONTEXT_H
