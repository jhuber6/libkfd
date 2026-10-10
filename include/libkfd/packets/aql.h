//===-- libkfd/packets/aql.h - Packet encoding for the AQL CP ---*- C++ -*-===//
//
// Header and packet encodings for the Architected Queuing Language (AQL)
// packet processor. Packets are fixed 64-byte slots whose header dword is
// published last to hand the slot to the CP. Controls kernel launches and
// barriers.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_PACKETS_AQL_H
#define LIBKFD_PACKETS_AQL_H

#include "libkfd/abi.h"
#include "libkfd/dispatch.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace kfd::aql {

// AQL packet header encoding.
//
//   [7:0]   TYPE                    Packet type
//   [8]     BARRIER                 Wait for all prior packets to complete
//   [10:9]  SCACQUIRE_FENCE_SCOPE   Cache invalidation before launch
//   [12:11] SCRELEASE_FENCE_SCOPE   Cache writeback after completion
//
// Reference: HSA Platform System Architecture Specification 1.2,
//            Section 2.8 "Packet header"
enum PacketType : uint8_t {
  PACKET_VENDOR_SPECIFIC = 0,
  PACKET_INVALID = 1,
  PACKET_KERNEL_DISPATCH = 2,
  PACKET_BARRIER_AND = 3,
  PACKET_AGENT_DISPATCH = 4,
  PACKET_BARRIER_OR = 5,
};

enum FenceScope : uint8_t {
  FENCE_SCOPE_NONE = 0,
  FENCE_SCOPE_AGENT = 1,
  FENCE_SCOPE_SYSTEM = 2,
};

// Ordering carried by each packet. Packets launch in order but may complete
// out of order unless 'barrier' is set.
struct Ordering {
  bool barrier = false;
  FenceScope acquire = FENCE_SCOPE_NONE;
  FenceScope release = FENCE_SCOPE_NONE;
};

constexpr uint16_t header(PacketType type, Ordering order) {
  return static_cast<uint16_t>(static_cast<uint32_t>(type) |
                               (order.barrier ? 1u << 8 : 0u) |
                               (static_cast<uint32_t>(order.acquire) << 9u) |
                               (static_cast<uint32_t>(order.release) << 11u));
}

constexpr PacketType packet_type(uint16_t header) {
  return static_cast<PacketType>(header & 0xFFu);
}

// BARRIER_AND - completes once every non-null signal in 'dep_signal' reaches
// zero. Later packets do not launch until it completes.
//
// Reference: HSA Platform System Architecture Specification 1.2,
//            Section 2.8 "Barrier-AND Packet"
inline constexpr size_t BARRIER_AND_DEPS = 5;

struct alignas(64) BarrierAndPacket {
  uint16_t header;
  uint16_t reserved0;
  uint32_t reserved1;
  uint64_t dep_signal[BARRIER_AND_DEPS];
  uint64_t reserved2;
  uint64_t completion_signal;
};

static_assert(sizeof(BarrierAndPacket) == 64);

// Error codes the CP writes to the queue's inactive signal before stalling.
//
// Reference: ROCr AqlQueue::DynamicQueueEventsHandler
enum QueueInactiveCode : uint32_t {
  INACTIVE_SCRATCH_WAVE64 = 1u << 0,
  INACTIVE_INVALID_DIM = 1u << 1,
  INACTIVE_INVALID_GROUP_MEMORY = 1u << 2,
  INACTIVE_INVALID_CODE_OBJECT = 1u << 3,
  INACTIVE_INVALID_FORMAT = 1u << 5,
  INACTIVE_GROUP_TOO_LARGE = 1u << 6,
  INACTIVE_OUT_OF_VGPRS = 1u << 7,
  INACTIVE_INVALID_VENDOR_FORMAT = 1u << 8,
  INACTIVE_SCRATCH_RECLAIM = 1u << 9,
  INACTIVE_SCRATCH_WAVE32 = 1u << 10,
  INACTIVE_MEMORY_VIOLATION = 1u << 29,
  INACTIVE_ILLEGAL_INSTRUCTION = 1u << 30,
  INACTIVE_DEBUG_TRAP = 1u << 31,
};

inline constexpr uint64_t INACTIVE_SCRATCH =
    INACTIVE_SCRATCH_WAVE64 | INACTIVE_SCRATCH_WAVE32;

// KERNEL_DISPATCH - launch the kernel whose descriptor is at 'kernel_object'.
// The CP reads the descriptor on the device, so this must be the loaded copy.
inline void kernel_dispatch(abi::DispatchPacket &pkt, Dim3 grid, Dim3 block,
                            uint32_t private_size, uint32_t group_size,
                            uint64_t kernel_object, const void *kernarg,
                            uint64_t completion, Ordering order) {
  abi::fill_dispatch_packet(pkt, grid, block, private_size, group_size);
  pkt.kernel_object = kernel_object;
  pkt.kernarg_address = reinterpret_cast<uintptr_t>(kernarg);
  pkt.completion_signal = completion;
  pkt.header = header(PACKET_KERNEL_DISPATCH, order);
}

inline void barrier_and(BarrierAndPacket &pkt, std::span<const uint64_t> deps,
                        uint64_t completion, Ordering order) {
  pkt = {};
  for (size_t i = 0; i < deps.size() && i < BARRIER_AND_DEPS; ++i)
    pkt.dep_signal[i] = deps[i];
  pkt.completion_signal = completion;
  pkt.header = header(PACKET_BARRIER_AND, order);
}

// GFX9+ vendor packet. The CP executes 'ib_jump' (an INDIRECT_BUFFER) and then
// completes the signal. 'dwords_remaining' counts the dwords that follow it,
// eight reserved words plus the signal, and the firmware expects 0xA.
inline constexpr uint16_t VENDOR_PM4_IB = 1;
inline constexpr uint32_t PM4_IB_DWORDS_REMAINING = 0xA;

struct alignas(64) Pm4IndirectPacket {
  uint16_t header;
  uint16_t vendor_format;
  uint32_t ib_jump[4];
  uint32_t dwords_remaining;
  uint32_t reserved[8];
  uint64_t completion_signal;
};

static_assert(sizeof(Pm4IndirectPacket) == 64);

inline void pm4_indirect(Pm4IndirectPacket &pkt, const uint32_t jump[4],
                         uint64_t completion, Ordering order = {}) {
  pkt = {};
  pkt.vendor_format = VENDOR_PM4_IB;
  for (int i = 0; i < 4; ++i)
    pkt.ib_jump[i] = jump[i];
  pkt.dwords_remaining = PM4_IB_DWORDS_REMAINING;
  pkt.completion_signal = completion;
  pkt.header = header(PACKET_VENDOR_SPECIFIC, order);
}

} // namespace kfd::aql

#endif // LIBKFD_PACKETS_AQL_H
