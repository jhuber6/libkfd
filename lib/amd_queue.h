//===-- lib/amd_queue.h - AQL queue descriptor ------------------*- C++ -*-===//
//
// Memory the AQL packet processor reads alongside the ring buffer and when
// completing signals. The layouts are fixed by the CP firmware.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_LIB_AMD_QUEUE_H
#define LIBKFD_LIB_AMD_QUEUE_H

#include "libkfd/event.h"

#include <cstddef>
#include <cstdint>

namespace kfd {

// Signal memory as the AQL CP addresses it. Completing a packet atomically
// decrements 'value', then writes 'event_id' to 'event_mailbox' to raise the
// event. Every Signal's fence and the queue's inactive signal use this layout.
// The CP only writes the timestamps that follow when profiling is enabled.
//
// Reference: rocr-runtime inc/amd_hsa_signal.h
struct alignas(32) CompletionSignal {
  uint64_t reserved0 = 0;
  uint64_t value = 0;
  uint64_t event_mailbox = 0;
  uint32_t event_id = 0;
  uint32_t reserved1 = 0;

  // Slots are recycled with their event IDs, which the kernel hands between
  // threads without a visible happens-before.
  void bind(const Event &event) {
    __atomic_store_n(&event_mailbox,
                     reinterpret_cast<uintptr_t>(event.signal_addr()),
                     __ATOMIC_RELAXED);
    __atomic_store_n(&event_id, event.event_id(), __ATOMIC_RELAXED);
  }
};

static_assert(sizeof(CompletionSignal) == 32);
static_assert(offsetof(CompletionSignal, value) == 8);
static_assert(offsetof(CompletionSignal, event_mailbox) == 16);
static_assert(offsetof(CompletionSignal, event_id) == 24);

// Address the CP takes for a signal whose fence is at 'fence'.
inline uint64_t completion_handle(const uint64_t *fence) {
  return reinterpret_cast<uintptr_t>(fence) - offsetof(CompletionSignal, value);
}

// AMD queue descriptor (amd_queue_v2_t) placed alongside the ring. The CP
// locates it from the read pointer given to KFD and reads its scratch
// configuration on every dispatch. It begins with the public hsa_queue_t.
//
// Reference: rocr-runtime inc/amd_hsa_queue.h
struct alignas(64) AmdQueue {
  AmdQueue(const void *ring, uint32_t packets, const CompletionSignal &inactive)
      : base_address(reinterpret_cast<uintptr_t>(ring)), size(packets),
        queue_inactive_signal(reinterpret_cast<uintptr_t>(&inactive)) {}

  uint32_t type = /*HSA_QUEUE_TYPE_MULTI=*/0;
  uint32_t features = /*HSA_QUEUE_FEATURE_KERNEL_DISPATCH=*/1;
  uint64_t base_address;
  uint64_t doorbell_signal = 0;
  uint32_t size;
  uint32_t reserved0 = 0;
  uint64_t id = UINT64_MAX;
  uint32_t caps = 0;
  uint32_t reserved1[3] = {};
  uint64_t write_dispatch_id = 0;
  // Pre-GFX9 aperture bases, unused here.
  uint32_t unused[16] = {};
  uint64_t read_dispatch_id = 0;
  uint32_t read_dispatch_id_field_base_byte_offset = /*read_dispatch_id=*/128;
  uint32_t compute_tmpring_size = 0;
  uint32_t scratch_resource_descriptor[4] = {};
  uint64_t scratch_backing_memory_location = 0;
  uint64_t scratch_backing_memory_byte_size = 0;
  uint32_t scratch_wave64_lane_byte_size = 0;
  uint32_t queue_properties = /*AMD_QUEUE_PROPERTIES_IS_PTR64=*/1u << 1;
  uint64_t scratch_max_use_index = UINT64_MAX;
  uint64_t queue_inactive_signal;
  uint64_t alt_scratch_max_use_index = UINT64_MAX;
  uint32_t alt_scratch_resource_descriptor[4] = {};
  uint64_t alt_scratch_backing_memory_location = 0;
  uint32_t alt_scratch_dispatch_limit_x = 0;
  uint32_t alt_scratch_dispatch_limit_y = 0;
  uint32_t alt_scratch_dispatch_limit_z = 0;
  uint32_t alt_scratch_wave64_lane_byte_size = 0;
  uint32_t alt_compute_tmpring_size = 0;
  uint32_t reserved3 = 0;
};

static_assert(sizeof(AmdQueue) == 256);
static_assert(offsetof(AmdQueue, write_dispatch_id) == 56);
static_assert(offsetof(AmdQueue, read_dispatch_id) == 128);
static_assert(offsetof(AmdQueue, scratch_resource_descriptor) == 144);
static_assert(offsetof(AmdQueue, queue_inactive_signal) == 192);

} // namespace kfd

#endif // LIBKFD_LIB_AMD_QUEUE_H
