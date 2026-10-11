//===-- libkfd/cpu.h - Per-node CPU state -----------------------*- C++ -*-===//
//
// A topology node with CPU cores and no GPU id. Owns that node's snapshot.
// A node with a GPU id is a Device, even when it also has cores.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_CPU_H
#define LIBKFD_CPU_H

#include "libkfd/topology.h"

#include <span>

namespace kfd {

class Cpu {
public:
  explicit Cpu(NodeInfo info) : info(std::move(info)) {}

  Cpu(const Cpu &) = delete;
  Cpu &operator=(const Cpu &) = delete;
  Cpu(Cpu &&) = default;
  Cpu &operator=(Cpu &&) = default;

  const NodeProperties &properties() const { return info.props; }

  std::span<const MemoryBank> memory_banks() const {
    return {info.memory_banks.data(), info.memory_banks.size()};
  }
  std::span<const CacheInfo> caches() const {
    return {info.caches.data(), info.caches.size()};
  }
  std::span<const IoLink> io_links() const {
    return {info.io_links.data(), info.io_links.size()};
  }

private:
  NodeInfo info;
};

} // namespace kfd

#endif // LIBKFD_CPU_H
