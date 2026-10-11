//===-- lib/queue.cpp - Queue implementation --------------------*- C++ -*-===//
//
// Queue creation, destruction, ring buffer management, and doorbell submission
// for PM4 compute and SDMA queues.
//
//===----------------------------------------------------------------------===//

#include "libkfd/queue.h"
#include "amd_queue.h"
#include "ioctl.h"
#include "libkfd/context.h"
#include "libkfd/detail/scratch.h"
#include "libkfd/detail/small_vector.h"
#include "libkfd/detail/utility.h"

#include <bit>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <new>

static_assert(static_cast<uint8_t>(kfd::QueueType::COMPUTE) ==
              KFD_IOC_QUEUE_TYPE_COMPUTE);
static_assert(static_cast<uint8_t>(kfd::QueueType::SDMA) ==
              KFD_IOC_QUEUE_TYPE_SDMA);
static_assert(static_cast<uint8_t>(kfd::QueueType::COMPUTE_AQL) ==
              KFD_IOC_QUEUE_TYPE_COMPUTE_AQL);
static_assert(static_cast<uint8_t>(kfd::QueueType::SDMA_XGMI) ==
              KFD_IOC_QUEUE_TYPE_SDMA_XGMI);
static_assert(static_cast<uint8_t>(kfd::QueueType::SDMA_BY_ENG_ID) ==
              KFD_IOC_QUEUE_TYPE_SDMA_BY_ENG_ID);

using namespace kfd::detail;

namespace kfd {

struct QueueErrorCtx {
  Context *ctx;
  uint64_t *err_payload;
  uint32_t queue_id;
  uint32_t gpu_id;
  uint64_t error;
};

struct ScratchCtx {
  Device *dev = nullptr;

  // PM4 queues are stalled by construction and make a request.
  uint64_t *scratch_cur_ptr = nullptr;
  uint32_t *ib_buf = nullptr;

  uint32_t *req_perwave = nullptr;
  uint32_t *req_block = nullptr;
  uint32_t *req_props = nullptr;

  void *scratch_va = nullptr;
  size_t scratch_size = 0;
  Buffer scratch_bo;
  uint32_t scratch_perwave = 0;
  uint32_t scratch_tmpring = 0;

  // AQL queues are stalled by the CP and report through the inactive signal.
  AmdQueue *amd_queue = nullptr;
  uint64_t *inactive = nullptr;
  const abi::DispatchPacket *ring = nullptr;
  uint64_t ring_packets = 0;
  QueueErrorCtx *err = nullptr;
  uint32_t queue_id = 0;
};

namespace {

// Number of 32-bit words needed to hold one bit per compute unit on the node.
uint32_t device_cu_words(const Device &dev) {
  const NodeProperties &p = dev.properties();
  uint32_t cu_count = p.simd_count / (p.simd_per_cu ? p.simd_per_cu : 1);
  return align_up(cu_count, 32u) / 32;
}

// Value is hard-coded by the kernel driver.
constexpr size_t EOP_BUFFER_SIZE = 4096;
constexpr MemFlags QUEUE_GTT_FLAGS = MemFlags::WRITABLE | MemFlags::EXECUTABLE |
                                     MemFlags::NO_SUBSTITUTE |
                                     MemFlags::COHERENT | MemFlags::UNCACHED;

// Set the scratch registers in the indirect buffer. This overrides what was
// written in the standard packet with what the watcher thread allocated.
uint32_t set_scratch_sgprs(uint32_t *out, uint32_t gfx_version,
                           const void *scratch_base, uint32_t tmpring_size,
                           uint16_t props) {
  uint32_t n = 0;
  if (props & abi::ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) {
    uint32_t v[4];
    pm4::build_scratch_srd(v, gfx_version, scratch_base, tmpring_size, props);
    n += pm4::set_sh_reg(out + n, pm4::regs::COMPUTE_USER_DATA_0, v, 4);
  }
  if (props & abi::ENABLE_SGPR_FLAT_SCRATCH_INIT) {
    uint32_t pos = 0;
    if (props & abi::ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
      pos += 4;
    if (props & abi::ENABLE_SGPR_DISPATCH_PTR)
      pos += 2;
    if (props & abi::ENABLE_SGPR_QUEUE_PTR)
      pos += 2;
    if (props & abi::ENABLE_SGPR_KERNARG_SEGMENT_PTR)
      pos += 2;
    if (props & abi::ENABLE_SGPR_DISPATCH_ID)
      pos += 2;
    const uint32_t flat[] = {lo(scratch_base), hi(scratch_base)};
    n +=
        pm4::set_sh_reg(out + n, pm4::regs::COMPUTE_USER_DATA_0 + pos, flat, 2);
  }
  return n;
}

QueueError make_queue_error(uint32_t queue_id, uint32_t gpu_id, uint64_t code) {
  using Kind = QueueError::Kind;
  Kind kind = Kind::None;
  if (code & (KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION) |
              KFD_EC_MASK(EC_QUEUE_WAVE_APERTURE_VIOLATION)))
    kind = Kind::MemoryViolation;
  else if (code & KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION))
    kind = Kind::IllegalInstruction;
  else if (code & KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR))
    kind = Kind::MathError;
  else if (code & KFD_EC_MASK(EC_QUEUE_WAVE_ABORT))
    kind = Kind::Abort;
  else if (code) // any other non-zero code is a generic wave trap
    kind = Kind::Trap;
  return QueueError{.queue_id = queue_id, .gpu_id = gpu_id, .kind = kind};
}

// Return the number of bytes to allocate for wave scratch.
uint32_t private_segment_bytes(const abi::KernelDescriptor &kd,
                               uint32_t requested) {
  if (!(kd.kernel_code_properties & abi::USES_DYNAMIC_STACK))
    return kd.private_segment_fixed_size;
  uint32_t extra = requested ? requested : /*8 KiB=*/8 * 1024;
  uint64_t total = uint64_t(kd.private_segment_fixed_size) + extra;
  return total > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(total);
}

QueueError::Kind inactive_error_kind(uint64_t code) {
  using Kind = QueueError::Kind;
  if (code & aql::INACTIVE_SCRATCH)
    return Kind::OutOfResources;
  if (code & aql::INACTIVE_MEMORY_VIOLATION)
    return Kind::MemoryViolation;
  if (code & aql::INACTIVE_ILLEGAL_INSTRUCTION)
    return Kind::IllegalInstruction;
  if (code & aql::INACTIVE_DEBUG_TRAP)
    return Kind::Trap;
  return Kind::InvalidPacket;
}

} // namespace

bool QueueBase::queue_error_handler(void *user_data) {
  auto *ectx = static_cast<QueueErrorCtx *>(user_data);
  uint64_t err = __atomic_load_n(ectx->err_payload, __ATOMIC_ACQUIRE);
  if (!err)
    return true;
  __atomic_store_n(ectx->err_payload, 0, __ATOMIC_RELEASE);

  // Grab the current error from the context payload.
  uint64_t expected = 0;
  __atomic_compare_exchange_n(&ectx->error, &expected, err, /*weak=*/false,
                              __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);

  FaultInfo info{.kind = FaultInfo::Kind::QueueException};
  info.queue = make_queue_error(ectx->queue_id, ectx->gpu_id, err);
  ectx->ctx->notify_fault(info);
  return true;
}

// Grows the backing scratch region to 'needed_perwave' bytes per wave. Returns
// false if no allocation could be made.
bool QueueBase::scratch_grow(ScratchCtx *sctx, uint32_t needed_perwave,
                             uint32_t threads_per_group) {
  void *old_va = sctx->scratch_va;
  if (old_va) {
    sctx->scratch_bo.release_device();
    sctx->scratch_bo = {};
    KFD_ASSERT(
        sctx->dev->scratch_allocator.deallocate(old_va, sctx->scratch_size));
    sctx->scratch_va = nullptr;
    sctx->scratch_size = 0;
  }

  uint32_t gfx = sctx->dev->gfx_version();
  uint32_t lanes = abi::native_wave_size(gfx);
  uint32_t slots = detail::scratch_device_slots(*sctx->dev, 1);
  uint32_t dev_num_xcc =
      sctx->dev->properties().num_xcc ? sctx->dev->properties().num_xcc : 1;
  uint32_t num_se = detail::scratch_num_se(*sctx->dev) / dev_num_xcc;
  if (num_se == 0)
    num_se = 1;
  uint32_t waves_per_group =
      detail::max((threads_per_group + lanes - 1) / lanes, 1u);
  uint32_t min_slots = 2 * num_se;

  // Try to allocate scratch for the requested slots, shrinking the slot count
  // and retrying when storage is short. The size helpers take a per-wave size
  // directly (per_thread=needed_perwave, lanes=1).
  while (slots >= min_slots) {
    size_t alloc_size =
        detail::scratch_alloc_size(gfx, needed_perwave, 1, slots);
    if (alloc_size == 0)
      break;

    auto region = sctx->dev->scratch_allocator.allocate(alloc_size);
    if (!region) {
      slots = ((slots - waves_per_group) / num_se) * num_se;
      continue;
    }

    auto bo = Buffer::allocate(*sctx->dev, alloc_size, MemType::VRAM,
                               MemFlags::WRITABLE, *region);
    if (!bo || !bo->map(*sctx->dev)) {
      if (bo) {
        bo->release_device();
        *bo = {};
      }
      KFD_ASSERT(sctx->dev->scratch_allocator.deallocate(*region, alloc_size));
      slots = ((slots - waves_per_group) / num_se) * num_se;
      continue;
    }

    sctx->scratch_va = *region;
    sctx->scratch_size = alloc_size;
    sctx->scratch_tmpring = detail::compute_tmpring_size(
        *sctx->dev, needed_perwave, 1, alloc_size, 1);
    sctx->scratch_perwave = needed_perwave;
    sctx->scratch_bo = std::move(*bo);
    return true;
  }

  return false;
}

// Woken by the grow handshake. Grows the region if the request exceeds the
// current capacity, rebuilds the sticky IB, then publishes the capacity to
// release the CP's WAIT_REG_MEM.
bool QueueBase::scratch_handler(void *user_data) {
  auto *sctx = static_cast<ScratchCtx *>(user_data);

  uint32_t needed_perwave =
      __atomic_load_n(sctx->req_perwave, __ATOMIC_ACQUIRE);
  uint32_t threads_per_group =
      __atomic_load_n(sctx->req_block, __ATOMIC_RELAXED);
  uint16_t props =
      static_cast<uint16_t>(__atomic_load_n(sctx->req_props, __ATOMIC_RELAXED));
  if (needed_perwave == 0)
    return true;

  if (needed_perwave > sctx->scratch_perwave &&
      !scratch_grow(sctx, needed_perwave, threads_per_group)) {
    // FIXME: Needs a more graceful error handling method for this.
    std::fprintf(stderr, "libkfd: scratch allocation failed\n");
    std::abort();
  }

  // Build the indirect buffer with the current scratch state. The GPU executes
  // it via INDIRECT_BUFFER after its WAIT_REG_MEM clears.
  uint32_t gfx = sctx->dev->gfx_version();
  void *ib_va = sctx->scratch_va;
  uint32_t ib_tmpring = sctx->scratch_tmpring;
  uint32_t ib_n = 0;
  ib_n += pm4::set_scratch_base(sctx->ib_buf + ib_n, gfx, ib_va, ib_tmpring);
  ib_n += set_scratch_sgprs(sctx->ib_buf + ib_n, gfx, ib_va, ib_tmpring, props);
  if (ib_n + 2 <= QueueBase::MAX_SCRATCH_IB_DWORDS) {
    uint32_t pad = QueueBase::MAX_SCRATCH_IB_DWORDS - ib_n;
    sctx->ib_buf[ib_n] =
        pm4::header(pm4::Opcode::NOP, static_cast<uint16_t>(pad - 2));
  }

  // The release store orders the IB writes ahead of the capacity the CP waits
  // on.
  __atomic_store_n(sctx->scratch_cur_ptr, uint64_t(sctx->scratch_perwave),
                   __ATOMIC_RELEASE);
  return true;
}

// Unlike PM4, the AQL queue will stall on its queue with a reason. We can
// detect this, service it, and clear the queue's signal to continue.
bool QueueBase::inactive_handler(void *user_data) {
  auto *sctx = static_cast<ScratchCtx *>(user_data);
  uint64_t code = __atomic_load_n(sctx->inactive, __ATOMIC_ACQUIRE);
  if (code == 0)
    return true;

  if (code & aql::INACTIVE_SCRATCH) {
    AmdQueue *q = sctx->amd_queue;
    uint64_t rd = __atomic_load_n(&q->read_dispatch_id, __ATOMIC_ACQUIRE);
    uint64_t wr = __atomic_load_n(&q->write_dispatch_id, __ATOMIC_ACQUIRE);
    const abi::DispatchPacket *pkt = nullptr;
    for (uint64_t i = rd; i < wr && !pkt; ++i) {
      const abi::DispatchPacket &p = sctx->ring[i & (sctx->ring_packets - 1)];
      uint32_t header = __atomic_load_n(
          reinterpret_cast<const uint32_t *>(&p.header), __ATOMIC_ACQUIRE);
      if (aql::packet_type(static_cast<uint16_t>(header)) ==
              aql::PACKET_KERNEL_DISPATCH &&
          p.private_segment_size)
        pkt = &p;
    }

    uint32_t gfx = sctx->dev->gfx_version();
    uint32_t lanes = (code & aql::INACTIVE_SCRATCH_WAVE32) ? 32 : 64;
    uint32_t per_wave = pkt ? align_up(lanes * pkt->private_segment_size,
                                       detail::scratch_alignment_unit(gfx))
                            : 0;
    // The same size again means the packet does not cover what the command
    // processor measured. Another allocation would free the live region and
    // the queue would stall on every retry.
    if (pkt && per_wave > sctx->scratch_perwave &&
        per_wave <= detail::max_wave_scratch(gfx) &&
        scratch_grow(sctx, per_wave,
                     uint32_t(pkt->workgroup_size_x) * pkt->workgroup_size_y *
                         pkt->workgroup_size_z)) {
      uint32_t srd[4];
      pm4::build_scratch_srd(srd, gfx, sctx->scratch_va, sctx->scratch_tmpring,
                             lanes == 32 ? abi::ENABLE_WAVEFRONT_SIZE32 : 0);
      std::memcpy(q->scratch_resource_descriptor, srd, sizeof(srd));
      q->compute_tmpring_size = sctx->scratch_tmpring;
      q->scratch_backing_memory_location =
          reinterpret_cast<uintptr_t>(sctx->scratch_va);
      q->scratch_backing_memory_byte_size = sctx->scratch_size;
      q->scratch_wave64_lane_byte_size = per_wave / 64;
      q->scratch_max_use_index = UINT64_MAX;
      __atomic_store_n(sctx->inactive, 0, __ATOMIC_RELEASE);
      return true;
    }
  }

  if (QueueErrorCtx *ectx = sctx->err) {
    uint64_t expected = 0;
    __atomic_compare_exchange_n(&ectx->error, &expected, code, /*weak=*/false,
                                __ATOMIC_ACQ_REL, __ATOMIC_RELAXED);
  }

  FaultInfo info{.kind = FaultInfo::Kind::QueueException};
  info.queue = QueueError{.queue_id = sctx->queue_id,
                          .gpu_id = sctx->dev->gpu_id(),
                          .kind = inactive_error_kind(code)};
  sctx->dev->context().notify_fault(info);
  return true;
}

// Create a queue to communicate with the device's command processor. Each queue
// is a ring buffer that consumes variable length packets for the processor.
std::expected<QueueBase, Error>
QueueBase::create(Device &dev, QueueType type, size_t ring_size,
                  uint32_t priority, uint32_t target_xcc,
                  std::optional<uint32_t> engine_id) {
  Context &ctx = dev.context();
  if (engine_id) {
    if (type != QueueType::SDMA && type != QueueType::SDMA_XGMI)
      return kfd::unexpected(EINVAL, "engine id requires an SDMA queue");
    type = QueueType::SDMA_BY_ENG_ID;
  }
  if (!std::has_single_bit(ring_size))
    return kfd::unexpected(EINVAL, "ring size %zu is not a power of two",
                           ring_size);
  if (ring_size > UINT32_MAX)
    return kfd::unexpected(EINVAL, "ring size %zu exceeds uint32_t", ring_size);
  if (priority > 15)
    return kfd::unexpected(
        EINVAL, "invalid queue priority; %u exceeds max of 15", priority);

  const bool is_aql = type == QueueType::COMPUTE_AQL;
  const bool is_compute = type == QueueType::COMPUTE || is_aql;
  // Hardware CP_HQD_PQ_CONTROL.QUEUE_SIZE enforces a minimum of 256 dwords.
  if (is_aql && ring_size < 16 * AQLQueue::PACKET_SIZE)
    return kfd::unexpected(
        EINVAL, "AQL ring of %zu bytes is below the 1024-byte minimum",
        ring_size);

  // Control page holds the read pointer, write pointer, EOP sequence, and
  // exception payload. AQL queues follow it with the CP-visible amd_queue_t,
  // whose dispatch IDs replace the ring pointers, and its inactive signal.
  static_assert(AMD_QUEUE_OFFSET % alignof(AmdQueue) == 0);
  constexpr size_t INACTIVE_OFFSET = AMD_QUEUE_OFFSET + sizeof(AmdQueue);
  // The CP reads this 64-byte record directly.
  constexpr size_t kInactiveBytes = 64;
  static_assert(INACTIVE_OFFSET % kInactiveBytes == 0);
  size_t ctl_size =
      is_aql ? INACTIVE_OFFSET + kInactiveBytes : sizeof(QueueControl);
  auto ctl_buf =
      KFD_TRY(Buffer::allocate(dev, ctl_size, MemType::GTT, QUEUE_GTT_FLAGS));
  KFD_CHECK(ctl_buf.map(dev));
  std::memset(ctl_buf.data(), 0, ctl_buf.size());
  auto *ctl = static_cast<QueueControl *>(ctl_buf.data());

  // CWSR buffer required on GFX9+ for HWS preemption (compute only).
  // The header wires up err_payload_addr and err_event_id so KFD can
  // signal queue exceptions (traps, illegal instructions, etc.).
  MappedRegion cwsr_buf;
  Buffer cwsr_bo_buf;
  detail::Box<Event> err_ev;
  const NodeProperties &props = dev.properties();
  if (is_compute && props.cwsr_size > 0) {
    uint32_t simd_per_cu = props.simd_per_cu ? props.simd_per_cu : 1;
    uint32_t num_xcc = props.num_xcc ? props.num_xcc : 1;
    uint32_t cu_num = props.simd_count / simd_per_cu / num_xcc;
    uint32_t wave_num;
    if (props.gfx_target_version < abi::GFX_VERSION_GFX10_1) {
      uint32_t simd_arrays_per_engine =
          props.simd_arrays_per_engine ? props.simd_arrays_per_engine : 1;
      uint32_t max_waves = props.array_count / simd_arrays_per_engine * 512;
      wave_num = cu_num * 40;
      if (wave_num > max_waves)
        wave_num = max_waves;
    } else {
      wave_num = cu_num * 32;
    }
    uint32_t debug_mem = align_up(wave_num * 32u, 64u);
    size_t total_cwsr =
        align_up(static_cast<size_t>(props.cwsr_size + debug_mem) * num_xcc,
                 page_size());

    // Each queue has an event for the CWSR to handle device-side interrupts.
    auto raw_ev = KFD_TRY(Event::create(ctx));
    err_ev = KFD_TRY(detail::Box<Event>::create(std::move(raw_ev)));

    bool svm_supported =
        (props.capability & NodeProperties::NODE_CAP_SVMAPI_SUPPORTED) != 0;

    // CWSR can fire at any time so we must ensure that the pages are resident.
    // If SVM is supported we use an always-mapped page, otherwise we pin
    // anonymous pages as a USERPTR BO (matching libhsakmt's dGPU fallback).
    auto cwsr_region = KFD_TRY(MappedRegion::create(total_cwsr));
    std::memset(cwsr_region.data(), 0, cwsr_region.size());
    for (uint32_t i = 0; i < num_xcc; ++i) {
      auto *hdr = reinterpret_cast<abi::CwsrHeader *>(
          static_cast<char *>(cwsr_region.data()) +
          static_cast<size_t>(i * props.cwsr_size));
      hdr->debug_offset = (num_xcc - i) * props.cwsr_size;
      hdr->debug_size = debug_mem * num_xcc;
      hdr->err_payload_addr = reinterpret_cast<uint64_t>(&ctl->err_payload);
      hdr->err_event_id = err_ev->event_id();
    }

    if (svm_supported) {
      constexpr SVMFlags CWSR_SVM_FLAGS = SVMFlags::HOST_ACCESS |
                                          SVMFlags::GPU_EXEC |
                                          SVMFlags::GPU_ALWAYS_MAPPED;
      void *svm_addr = cwsr_region.data();
      size_t svm_size = cwsr_region.size();
      KFD_CHECK(svm_set_preferred_loc(ctx, svm_addr, svm_size));
      KFD_CHECK(svm_set_flags(ctx, svm_addr, svm_size, CWSR_SVM_FLAGS));
      Device *dev_ptr = &dev;
      KFD_CHECK(svm_set_access(ctx, svm_addr, svm_size, {&dev_ptr, 1}));
      KFD_CHECK(svm_set_granularity(ctx, svm_addr, svm_size, 0xFF));
      KFD_CHECK(svm_prefetch(ctx, svm_addr, svm_size, &dev));
    } else {
      constexpr MemFlags CWSR_PIN_FLAGS =
          MemFlags::WRITABLE | MemFlags::EXECUTABLE | MemFlags::COHERENT;
      auto bo = KFD_TRY(
          Buffer::pin_region(dev, std::move(cwsr_region), CWSR_PIN_FLAGS));
      KFD_CHECK(bo.map(dev));
      cwsr_bo_buf = std::move(bo);
    }
    if (cwsr_region)
      cwsr_buf = std::move(cwsr_region);
  }

  // An internal end-of-pipe buffer used by the CP to manage EOP events.
  Buffer eop_buf;
  if (is_compute) {
    eop_buf = KFD_TRY(Buffer::allocate(
        dev, EOP_BUFFER_SIZE, MemType::VRAM,
        MemFlags::WRITABLE | MemFlags::EXECUTABLE | MemFlags::NO_SUBSTITUTE));
    KFD_CHECK(eop_buf.map(dev));
  }

  // The ring buffer itself that holds the packets.
  auto ring_buf =
      KFD_TRY(Buffer::allocate(dev, ring_size, MemType::GTT, QUEUE_GTT_FLAGS));
  KFD_CHECK(ring_buf.map(dev));
  std::memset(ring_buf.data(), 0, ring_buf.size());

  // A zeroed AQL slot decodes as a vendor-specific packet, so every header
  // starts out INVALID. The CP restores INVALID after consuming a packet.
  AmdQueue *amdq = nullptr;
  void *inactive = nullptr;
  detail::Box<Event> inactive_ev;
  if (is_aql) {
    auto *slots = static_cast<abi::DispatchPacket *>(ring_buf.data());
    uint32_t packets = static_cast<uint32_t>(ring_size / AQLQueue::PACKET_SIZE);
    for (size_t i = 0; i < packets; ++i)
      slots[i].header = aql::PACKET_INVALID;

    auto raw_ev = KFD_TRY(Event::create(ctx));
    inactive_ev = KFD_TRY(detail::Box<Event>::create(std::move(raw_ev)));
    auto *ctl_bytes = static_cast<std::byte *>(ctl_buf.data());
    inactive = ctl_bytes + INACTIVE_OFFSET;
    __atomic_store_n(
        reinterpret_cast<uint64_t *>(static_cast<std::byte *>(inactive) + 16),
        reinterpret_cast<uintptr_t>(inactive_ev->signal_addr()),
        __ATOMIC_RELAXED);
    __atomic_store_n(
        reinterpret_cast<uint32_t *>(static_cast<std::byte *>(inactive) + 24),
        inactive_ev->event_id(), __ATOMIC_RELAXED);
    amdq = new (ctl_bytes + AMD_QUEUE_OFFSET)
        AmdQueue(ring_buf.data(), packets, inactive);
  }

  uint64_t *write_ptr = is_aql ? &amdq->write_dispatch_id : &ctl->write_ptr;
  uint64_t *read_ptr = is_aql ? &amdq->read_dispatch_id : &ctl->read_ptr;

  // Ensure all the queue memory is committed before submitting it.
  memory_barrier();
  ioctl::kfd::create_queue_args args{
      .ring_base_address = reinterpret_cast<uintptr_t>(ring_buf.data()),
      .write_pointer_address = reinterpret_cast<uintptr_t>(write_ptr),
      .read_pointer_address = reinterpret_cast<uintptr_t>(read_ptr),
      .ring_size = static_cast<uint32_t>(ring_size),
      .gpu_id = dev.gpu_id(),
      .queue_type = static_cast<uint32_t>(type),
      .queue_percentage = (target_xcc << 8) | 100,
      .queue_priority = priority,
      .sdma_engine_id = engine_id.value_or(0),
  };
  if (is_compute) {
    args.eop_buffer_address = reinterpret_cast<uintptr_t>(eop_buf.data());
    args.eop_buffer_size = EOP_BUFFER_SIZE;
  }
  if (cwsr_buf || cwsr_bo_buf) {
    void *cwsr_ptr = cwsr_buf ? cwsr_buf.data() : cwsr_bo_buf.data();
    args.ctx_save_restore_address = reinterpret_cast<uintptr_t>(cwsr_ptr);
    args.ctx_save_restore_size = props.cwsr_size;
    args.ctl_stack_size = props.ctl_stack_size;
  }

  // SDMA queues have limited slots and do not oversubscribe like compute queues
  // do. If the process fails to get one with ENOMEM we retry a few times.
  constexpr int QUEUE_CREATE_RETRIES = 40;
  constexpr long QUEUE_BACKOFF_START_US = 250;
  constexpr long QUEUE_BACKOFF_MAX_US = 50'000;
  long backoff_us = QUEUE_BACKOFF_START_US;
  for (int attempt = 0;; ++attempt) {
    auto r = ioctl::call<ioctl::kfd::CREATE_QUEUE>(ctx.kfd_fd(), args);
    if (r)
      break;
    if (r.error().code != ENOMEM || attempt >= QUEUE_CREATE_RETRIES)
      return kfd::unexpected(r.error());
    struct timespec ts = {.tv_sec = 0, .tv_nsec = backoff_us * 1'000L};
    ::nanosleep(&ts, nullptr);
    backoff_us = detail::min(backoff_us * 2, QUEUE_BACKOFF_MAX_US);
  }

  // Doorbells are MMIO mapped registers that signal the CP to consume packets.
  std::expected<volatile uint64_t *, Error> db_slot =
      dev.doorbell(args.doorbell_offset);
  if (!db_slot) {
    ioctl::kfd::destroy_queue_args dq{.queue_id = args.queue_id};
    KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_QUEUE>(ctx.kfd_fd(), dq));
    return kfd::unexpected(db_slot.error());
  }

  QueueBase q(type, ctx, dev, args.queue_id, priority, target_xcc,
              std::move(ctl_buf), std::move(ring_buf), std::move(eop_buf),
              std::move(cwsr_buf), std::move(cwsr_bo_buf), *db_slot,
              std::move(err_ev), {});

  if (is_compute) {
    uint32_t cu_count =
        props.simd_count / (props.simd_per_cu ? props.simd_per_cu : 1);
    KFD_CHECK(q.cu_mask_words.resize(device_cu_words(dev)));
    for (size_t i = 0; i < q.cu_mask_words.size(); ++i) {
      uint32_t bits = cu_count - static_cast<uint32_t>(i) * 32;
      q.cu_mask_words[i] = bits >= 32 ? 0xFFFFFFFF : (1u << bits) - 1;
    }
  }

  if (q.err_event || is_aql) {
    q.err_watch_ctx = KFD_TRY(detail::Box<QueueErrorCtx>::create(QueueErrorCtx{
        .ctx = &ctx,
        .err_payload = reinterpret_cast<uint64_t *>(&q.ctl()->err_payload),
        .queue_id = q.id,
        .gpu_id = dev.gpu_id(),
        .error = 0}));
  }
  if (q.err_event)
    KFD_CHECK(ctx.register_handler(*q.err_event, queue_error_handler,
                                   q.err_watch_ctx.get()));

  if (is_aql) {
    amdq->id = q.id;

    q.scratch_watch_ctx = KFD_TRY(detail::Box<ScratchCtx>::create());
    auto *sctx = q.scratch_watch_ctx.get();
    sctx->dev = &dev;
    sctx->amd_queue = amdq;
    sctx->inactive =
        reinterpret_cast<uint64_t *>(static_cast<std::byte *>(inactive) + 8);
    sctx->ring = static_cast<const abi::DispatchPacket *>(q.ring.data());
    sctx->ring_packets = ring_size / AQLQueue::PACKET_SIZE;
    sctx->err = q.err_watch_ctx.get();
    sctx->queue_id = q.id;
    q.inactive_event = std::move(inactive_ev);
    KFD_CHECK(ctx.register_handler(*q.inactive_event, inactive_handler, sctx));
  } else if (is_compute) {
    auto raw_ev = KFD_TRY(Event::create(ctx));
    q.scratch_event = KFD_TRY(detail::Box<Event>::create(std::move(raw_ev)));

    q.scratch_watch_ctx = KFD_TRY(detail::Box<ScratchCtx>::create());
    auto *sctx = q.scratch_watch_ctx.get();
    sctx->dev = &dev;
    sctx->scratch_cur_ptr = &ctl->scratch_cur;
    sctx->req_perwave = &ctl->scratch_req_perwave;
    sctx->req_block = &ctl->scratch_req_block;
    sctx->req_props = &ctl->scratch_req_props;
    sctx->ib_buf = ctl->indirect;
    sctx->ib_buf[0] = pm4::header(
        pm4::Opcode::NOP,
        static_cast<uint16_t>(QueueBase::MAX_SCRATCH_IB_DWORDS - 2));
    KFD_CHECK(ctx.register_handler(*q.scratch_event, scratch_handler, sctx));
  }

  return q;
}

QueueBase::QueueBase(QueueType type, Context &ctx, Device &dev, uint32_t id,
                     uint32_t priority, uint32_t target_xcc, Buffer control,
                     Buffer ring, Buffer eop, detail::MappedRegion cwsr,
                     Buffer cwsr_bo, volatile uint64_t *doorbell,
                     detail::Box<Event> err_event, detail::Mutex submit_mtx)
    : type(type), ctx(&ctx), dev(&dev), id(id), queue_priority(priority),
      target_xcc(target_xcc), control(std::move(control)),
      ring(std::move(ring)), eop(std::move(eop)), cwsr(std::move(cwsr)),
      cwsr_bo(std::move(cwsr_bo)), doorbell(doorbell),
      err_event(std::move(err_event)), submit_mtx(std::move(submit_mtx)) {}

QueueBase::~QueueBase() {
  if (!ctx)
    return;

  if (err_event)
    KFD_ASSERT(ctx->unregister_handler(*err_event));
  if (scratch_event)
    KFD_ASSERT(ctx->unregister_handler(*scratch_event));
  if (inactive_event)
    KFD_ASSERT(ctx->unregister_handler(*inactive_event));

  // Destroy the queue before releasing scratch. DESTROY_QUEUE preempts any
  // in-flight waves, so the scratch region is no longer referenced by the GPU
  // once this returns.
  ioctl::kfd::destroy_queue_args args{.queue_id = id};
  KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_QUEUE>(ctx->kfd_fd(), args));

  if (scratch_watch_ctx && scratch_watch_ctx->scratch_va) {
    scratch_watch_ctx->scratch_bo.release_device();
    scratch_watch_ctx->scratch_bo = {};
    KFD_ASSERT(dev->scratch_allocator.deallocate(
        scratch_watch_ctx->scratch_va, scratch_watch_ctx->scratch_size));
  }
}

QueueBase::QueueBase(QueueBase &&other)
    : type(other.type), ctx(std::exchange(other.ctx, nullptr)),
      dev(std::exchange(other.dev, nullptr)), id(std::exchange(other.id, 0)),
      queue_priority(other.queue_priority), target_xcc(other.target_xcc),
      control(std::move(other.control)), ring(std::move(other.ring)),
      eop(std::move(other.eop)), cwsr(std::move(other.cwsr)),
      cwsr_bo(std::move(other.cwsr_bo)),
      doorbell(std::exchange(other.doorbell, nullptr)),
      err_event(std::move(other.err_event)),
      err_watch_ctx(std::move(other.err_watch_ctx)),
      scratch_event(std::move(other.scratch_event)),
      scratch_watch_ctx(std::move(other.scratch_watch_ctx)),
      inactive_event(std::move(other.inactive_event)),
      submit_mtx(std::move(other.submit_mtx)),
      cu_mask_words(std::move(other.cu_mask_words)),
      pending_wptr(std::exchange(other.pending_wptr, 0)) {}

QueueBase &QueueBase::operator=(QueueBase &&other) {
  if (this == &other)
    return *this;

  if (ctx) {
    if (err_event)
      KFD_ASSERT(ctx->unregister_handler(*err_event));
    if (scratch_event)
      KFD_ASSERT(ctx->unregister_handler(*scratch_event));
    if (inactive_event)
      KFD_ASSERT(ctx->unregister_handler(*inactive_event));
    ioctl::kfd::destroy_queue_args dq{.queue_id = id};
    KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_QUEUE>(ctx->kfd_fd(), dq));
    if (scratch_watch_ctx && scratch_watch_ctx->scratch_va) {
      scratch_watch_ctx->scratch_bo.release_device();
      scratch_watch_ctx->scratch_bo = {};
      KFD_ASSERT(dev->scratch_allocator.deallocate(
          scratch_watch_ctx->scratch_va, scratch_watch_ctx->scratch_size));
    }
  }

  type = other.type;
  ctx = std::exchange(other.ctx, nullptr);
  dev = std::exchange(other.dev, nullptr);
  id = std::exchange(other.id, 0);
  queue_priority = other.queue_priority;
  target_xcc = other.target_xcc;
  ring = std::move(other.ring);
  control = std::move(other.control);
  eop = std::move(other.eop);
  cwsr = std::move(other.cwsr);
  cwsr_bo = std::move(other.cwsr_bo);
  doorbell = std::exchange(other.doorbell, nullptr);
  err_event = std::move(other.err_event);
  err_watch_ctx = std::move(other.err_watch_ctx);
  scratch_event = std::move(other.scratch_event);
  scratch_watch_ctx = std::move(other.scratch_watch_ctx);
  inactive_event = std::move(other.inactive_event);
  submit_mtx = std::move(other.submit_mtx);
  cu_mask_words = std::move(other.cu_mask_words);
  pending_wptr = std::exchange(other.pending_wptr, 0);
  return *this;
}

std::expected<void, Error> QueueBase::wait_for_room(uint32_t dwords,
                                                    uint64_t timeout_ns) {
  uint32_t cap = static_cast<uint32_t>(ring_dwords());
  uint32_t pos = static_cast<uint32_t>(pending_wptr & (cap - 1));
  volatile uint32_t *rp_addr =
      reinterpret_cast<volatile uint32_t *>(&ctl()->read_ptr);
  uint32_t last_rp = __atomic_load_n(rp_addr, __ATOMIC_ACQUIRE);
  uint64_t deadline_ns = 0;
  bool armed = false;
  for (;;) {
    uint32_t raw_rp = __atomic_load_n(rp_addr, __ATOMIC_ACQUIRE);
    uint32_t rp = is_sdma() ? raw_rp / sizeof(uint32_t) : raw_rp;
    uint32_t in_flight = (pos + cap - rp) & (cap - 1);
    if (in_flight + dwords < cap)
      return {};
    if (timeout_ns != 0) {
      struct timespec now;
      ::clock_gettime(CLOCK_MONOTONIC, &now);
      uint64_t now_ns = static_cast<uint64_t>(now.tv_sec) * 1'000'000'000 +
                        static_cast<uint64_t>(now.tv_nsec);
      if (!armed || raw_rp != last_rp) {
        last_rp = raw_rp;
        deadline_ns = now_ns + timeout_ns;
        armed = true;
      } else if (now_ns >= deadline_ns) {
        return kfd::unexpected(
            ETIMEDOUT,
            "ring buffer stall; no consumer progress waiting for %u "
            "dwords of room",
            dwords);
      }
    }
    detail::spin_hint();
  }
}

std::expected<void, Error>
QueueBase::submit(const uint32_t *data, size_t n_dwords, uint64_t timeout_ns) {
  detail::LockGuard guard(submit_mtx);
  return submit_impl(data, n_dwords, timeout_ns);
}

std::expected<void, Error> QueueBase::submit_impl(const uint32_t *data,
                                                  size_t n_dwords,
                                                  uint64_t timeout_ns) {
  auto n = static_cast<uint32_t>(n_dwords);
  if (n == 0)
    return {};

  // Errors on the queue stall the CP until cleared, refuse to advance.
  if (QueueErrorCtx *ectx = err_watch_ctx.get()) {
    if (uint64_t err = __atomic_load_n(&ectx->error, __ATOMIC_ACQUIRE))
      return kfd::unexpected(ECANCELED,
                             "queue %u faulted (code 0x%lx); reset required",
                             id, static_cast<unsigned long>(err));
  }

  uint32_t cap = static_cast<uint32_t>(ring_dwords());
  if (n >= cap)
    return kfd::unexpected(
        EMSGSIZE, "submit size %u dwords exceeds ring capacity %u", n, cap);

  uint32_t *base = static_cast<uint32_t *>(ring.data());
  uint32_t mask = cap - 1;
  uint32_t pos = static_cast<uint32_t>(pending_wptr & mask);

  auto to_hw = [this](uint64_t wptr) -> uint64_t {
    return is_sdma() ? wptr * sizeof(uint32_t) : wptr;
  };

  volatile uint64_t *wptr_p = &ctl()->write_ptr;

  if (pos + n > cap) {
    uint32_t pad = cap - pos;
    KFD_CHECK(wait_for_room(pad, timeout_ns));

    if (is_sdma()) {
      base[pos] = (pad - 1) << 16;
      for (uint32_t i = 1; i < pad; ++i)
        base[pos + i] = 0;
    } else {
      pm4::nop_fill(base + pos, pad);
    }

    __atomic_store_n(&pending_wptr, pending_wptr + pad, __ATOMIC_RELEASE);
    uint64_t hw = to_hw(pending_wptr);
    memory_barrier();
    *wptr_p = hw;
    memory_barrier();
    *doorbell = hw;
    pos = 0;
  }

  KFD_CHECK(wait_for_room(n, timeout_ns));

  std::memcpy(base + pos, data, n * sizeof(uint32_t));
  __atomic_store_n(&pending_wptr, pending_wptr + n, __ATOMIC_RELEASE);

  uint64_t hw = to_hw(pending_wptr);
  memory_barrier();
  *wptr_p = hw;
  memory_barrier();
  *doorbell = hw;

  return {};
}

// PM4 queues execute packets in order but do not explicitly wait for
// completion.
std::expected<ComputeQueue, Error> ComputeQueue::create(Device &dev,
                                                        size_t ring_size,
                                                        uint32_t priority,
                                                        uint32_t target_xcc) {
  // Allocate before the queue is live so the VRAM alloc + map cannot trigger
  // a process eviction while the CP is already fetching from the ring.
  auto eop_vram = KFD_TRY(Buffer::allocate(
      dev, sizeof(uint64_t), MemType::VRAM,
      MemFlags::WRITABLE | MemFlags::NO_SUBSTITUTE | MemFlags::UNCACHED));
  KFD_CHECK(eop_vram.map(dev));

  auto base = KFD_TRY(QueueBase::create(dev, QueueType::COMPUTE, ring_size,
                                        priority, target_xcc));

  ComputeQueue q(std::move(base), std::move(eop_vram));

  uint32_t init[2 * pm4::WRITE_DATA_DWORDS];
  auto *eop_u32 = static_cast<uint32_t *>(q.eop_seq.data());
  pm4::write_data(init, eop_u32, 0);
  pm4::write_data(init + pm4::WRITE_DATA_DWORDS, eop_u32 + 1, 0);
  KFD_CHECK(q.base.submit_impl(
      init, static_cast<size_t>(2 * pm4::WRITE_DATA_DWORDS)));
  return q;
}

// Records a kernel launch. If the dispatch needs scratch we emit additional
// packets to drain the queue and update the base scratch registers, publishing
// the request to the watcher thread. Every dispatch shares the programmed base
// registers and are only set on scratch reallocation.
uint32_t ComputeQueue::dispatch_impl(uint32_t *out, const Kernel &kernel,
                                     const DispatchConfig &cfg, void *kernarg) {
  const abi::KernelDescriptor &kd = kernel.descriptor();
  uint32_t gfx = base.dev->gfx_version();

  uint32_t private_segment_size =
      private_segment_bytes(kd, cfg.private_segment_size);

  auto *sctx = base.scratch_watch_ctx.get();
  uint16_t props = kd.kernel_code_properties;

  uint32_t per_wave = static_cast<uint32_t>(detail::align_up(
      size_t(abi::native_wave_size(gfx)) * private_segment_size,
      size_t(detail::scratch_alignment_unit(gfx))));

  bool needs_scratch_allocation =
      private_segment_size > 0 &&
      per_wave > __atomic_load_n(&base.ctl()->scratch_cur, __ATOMIC_ACQUIRE);

  const void *dispatch_pkt_addr = nullptr;
  if (kd.kernel_code_properties & abi::ENABLE_SGPR_DISPATCH_PTR)
    dispatch_pkt_addr =
        static_cast<std::byte *>(kernarg) +
        detail::align_up(static_cast<size_t>(kd.kernarg_size), size_t(64));

  // Invalidate scalar and vector caches (K-cache / GLK V-cache) so the shader
  // reads fresh kernarg data. RELEASE_MEM only flushes L2 and vector L1; the
  // scalar L1 retains stale entries across dispatches to the same kernarg.
  uint32_t n = pm4::acquire_mem(out, gfx, pm4::ACQ_KCACHE | pm4::ACQ_VCACHE);

  // Resolve the launch sub-range for manual launch grid partitioning. A
  // non-zero origin covers [start, start + count) and reports absolute
  // work-group IDs.
  auto launch_count = [](uint32_t count, uint32_t full, uint32_t start) {
    return count ? count : (full > start ? full - start : 0);
  };
  Dim3 start = cfg.grid_start;
  Dim3 count = {launch_count(cfg.grid_count.x, cfg.grid.x, start.x),
                launch_count(cfg.grid_count.y, cfg.grid.y, start.y),
                launch_count(cfg.grid_count.z, cfg.grid.z, start.z)};

  n += pm4::build_dispatch_setup(out + n, gfx, kd, kernel.address(), cfg.grid,
                                 cfg.block, kernarg, dispatch_pkt_addr,
                                 cfg.dynamic_lds, private_segment_size,
                                 cooperative, start);

  // The scratch registers are set once when a reallocation event is triggered.
  // If a scratch allocation is still needed, we publish the request in-band and
  // notify the watcher thread before waiting on its result. The watcher
  // performs the allocation and presents an indirect buffer containing the new
  // scratch base. If the condition was already fulfilled, we skip past this.
  if (needs_scratch_allocation) {
    auto *ctl = base.ctl();
    constexpr uint32_t HANDSHAKE_DWORDS =
        /*in-band request=*/3 * pm4::WRITE_DATA_DWORDS +
        pm4::RELEASE_MEM_DWORDS + pm4::WAIT_REG_MEM_DWORDS +
        pm4::INDIRECT_BUFFER_DWORDS;
    uint64_t *scratch_guard = &ctl->scratch_guard;
    n += pm4::write_data(out + n, scratch_guard, 1);
    n += pm4::cond_write(out + n, Condition::GTE, &ctl->scratch_cur, per_wave,
                         0xFFFFFFFF, scratch_guard, 0);
    n += pm4::cond_exec(out + n, scratch_guard, HANDSHAKE_DWORDS);

    uint32_t threads_per_group = cfg.block.x * cfg.block.y * cfg.block.z;
    n += pm4::write_data(out + n, &ctl->scratch_req_perwave, per_wave);
    n += pm4::write_data(out + n, &ctl->scratch_req_block, threads_per_group);
    n += pm4::write_data(out + n, &ctl->scratch_req_props, props);
    n += pm4::release_mem(out + n, gfx, base.scratch_event->signal_addr(),
                          base.scratch_event->event_id(),
                          base.scratch_event->trigger_data());
    n += pm4::wait_reg_mem(out + n, gfx, &ctl->scratch_cur, Condition::GTE,
                           per_wave);
    n += pm4::indirect_buffer(out + n, sctx->ib_buf,
                              QueueBase::MAX_SCRATCH_IB_DWORDS,
                              pm4::CachePolicy::POLICY_BYPASS);
  }

  // Push the final dispatch with the full configuration.
  n +=
      pm4::dispatch_direct(out + n, start.x + count.x, start.y + count.y,
                           start.z + count.z, pm4::dispatch_initiator(kd, gfx));
  return n;
}

std::expected<void, Error>
ComputeQueue::flush(const uint32_t *data, size_t dwords, uint64_t timeout_ns) {
  return base.submit(data, dwords, timeout_ns);
}

std::expected<void, Error>
QueueBase::set_cu_mask(std::span<const uint32_t> mask) {
  if (mask.empty())
    return kfd::unexpected(EINVAL, "CU mask must not be empty");

  // Trim any tail past the device's CU count; those bits address nothing.
  size_t words = detail::min(mask.size(), size_t{device_cu_words(*dev)});
  ioctl::kfd::set_cu_mask_args args{
      .queue_id = queue_id(),
      .num_cu_mask = static_cast<uint32_t>(words * 32),
      .cu_mask_ptr = reinterpret_cast<uintptr_t>(mask.data()),
  };
  KFD_CHECK(ioctl::call<ioctl::kfd::SET_CU_MASK>(ctx->kfd_fd(), args));

  KFD_CHECK(cu_mask_words.resize(words));
  std::memcpy(cu_mask_words.data(), mask.data(), words * sizeof(uint32_t));
  return {};
}

std::expected<void, Error> QueueBase::set_priority(uint32_t priority) {
  if (priority > 15)
    return kfd::unexpected(
        EINVAL, "invalid queue priority; %u exceeds max of 15", priority);

  ioctl::kfd::update_queue_args args{
      .ring_base_address = reinterpret_cast<uintptr_t>(ring.data()),
      .queue_id = id,
      .ring_size = static_cast<uint32_t>(ring.size()),
      .queue_percentage = (target_xcc << 8) | 100,
      .queue_priority = priority,
  };
  KFD_CHECK(ioctl::call<ioctl::kfd::UPDATE_QUEUE>(ctx->kfd_fd(), args));
  queue_priority = priority;
  return {};
}

std::expected<CooperativeQueue, Error>
CooperativeQueue::create(Device &dev, size_t ring_size, uint32_t priority,
                         uint32_t target_xcc) {
  auto cq = KFD_TRY(ComputeQueue::create(dev, ring_size, priority, target_xcc));

  // GFX12+ uses GLG_EN in the dispatch packet itself, otherwise we need the GWS
  // ioctl to mark the queue as exclusively scheduled (non-preemptible).
  if (dev.gfx_version() < abi::GFX_VERSION_GFX12) {
    if (dev.properties().num_gws == 0)
      return kfd::unexpected(ENOTSUP,
                             "device does not support cooperative launch");

    ioctl::kfd::alloc_queue_gws_args gws{};
    gws.queue_id = cq.queue_id();
    gws.num_gws = 1;
    auto r =
        ioctl::call<ioctl::kfd::ALLOC_QUEUE_GWS>(dev.context().kfd_fd(), gws);
    if (!r)
      return kfd::unexpected(r.error());
  }

  return CooperativeQueue(std::move(cq));
}

AmdQueue *QueueBase::amd_queue() const {
  return reinterpret_cast<AmdQueue *>(static_cast<std::byte *>(control.data()) +
                                      AMD_QUEUE_OFFSET);
}

std::expected<AQLQueue, Error> AQLQueue::create(Device &dev, size_t ring_size,
                                                uint32_t priority,
                                                uint32_t target_xcc) {
  auto base = KFD_TRY(QueueBase::create(dev, QueueType::COMPUTE_AQL, ring_size,
                                        priority, target_xcc));
  return AQLQueue(std::move(base));
}

std::expected<void, Error> AQLQueue::submit(std::span<const std::byte> packets,
                                            uint64_t timeout_ns) {
  if (packets.empty())
    return {};
  if (packets.size() % PACKET_SIZE != 0)
    return kfd::unexpected(EINVAL,
                           "AQL submit of %zu bytes is not a multiple of %zu",
                           packets.size(), PACKET_SIZE);

  QueueErrorCtx *ectx = base.err_watch_ctx.get();
  AmdQueue *q = base.amd_queue();
  const uint64_t cap = q->size;
  const uint64_t count = packets.size() / PACKET_SIZE;
  if (count > cap)
    return kfd::unexpected(
        EMSGSIZE, "AQL submit of %zu packets exceeds ring of %zu",
        static_cast<size_t>(count), static_cast<size_t>(cap));

  // Reserve the whole run in one step, then publish each header in order.
  // The doorbell rings once, with the index of the last packet.
  uint64_t index = __atomic_load_n(&q->write_dispatch_id, __ATOMIC_RELAXED);
  uint64_t last_rd = __atomic_load_n(&q->read_dispatch_id, __ATOMIC_ACQUIRE);
  uint64_t deadline_ns = 0;
  bool armed = false;
  for (;;) {
    if (ectx) {
      if (uint64_t err = __atomic_load_n(&ectx->error, __ATOMIC_ACQUIRE))
        return kfd::unexpected(ECANCELED,
                               "queue %u faulted (code 0x%lx); reset required",
                               base.id, static_cast<unsigned long>(err));
    }

    uint64_t rd = __atomic_load_n(&q->read_dispatch_id, __ATOMIC_ACQUIRE);
    if (index - rd <= cap - count) {
      if (__atomic_compare_exchange_n(&q->write_dispatch_id, &index,
                                      index + count, /*weak=*/true,
                                      __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
        break;
      continue;
    }

    if (timeout_ns != 0) {
      struct timespec now;
      ::clock_gettime(CLOCK_MONOTONIC, &now);
      uint64_t now_ns = static_cast<uint64_t>(now.tv_sec) * 1'000'000'000 +
                        static_cast<uint64_t>(now.tv_nsec);
      if (!armed || rd != last_rd) {
        last_rd = rd;
        deadline_ns = now_ns + timeout_ns;
        armed = true;
      } else if (now_ns >= deadline_ns) {
        return kfd::unexpected(
            ETIMEDOUT, "AQL ring stall; no consumer progress on queue %u",
            base.id);
      }
    }
    detail::spin_hint();
    index = __atomic_load_n(&q->write_dispatch_id, __ATOMIC_RELAXED);
  }

  // A slot is free once the read index passes it, but that does not order
  // this write after the CP's INVALID store or the scratch handler's read.
  __atomic_load_n(base.scratch_watch_ctx->inactive, __ATOMIC_ACQUIRE);
  for (uint64_t n = 0; n < count; ++n) {
    auto *slot = static_cast<std::byte *>(base.ring.data()) +
                 ((index + n) & (cap - 1)) * PACKET_SIZE;
    __atomic_load_n(reinterpret_cast<uint32_t *>(slot), __ATOMIC_ACQUIRE);

    const auto *src = packets.data() + n * PACKET_SIZE;
    std::memcpy(slot + sizeof(uint32_t), src + sizeof(uint32_t),
                PACKET_SIZE - sizeof(uint32_t));
    uint32_t header;
    std::memcpy(&header, src, sizeof(header));
    __atomic_store_n(reinterpret_cast<uint32_t *>(slot), header,
                     __ATOMIC_RELEASE);
  }
  __atomic_store_n(base.doorbell, index + count - 1, __ATOMIC_RELEASE);
  return {};
}

AQLQueue::CommandBuffer &AQLQueue::CommandBuffer::fail(Error error) {
  if (!failed)
    failed = std::move(error);
  return *this;
}

AQLQueue::CommandBuffer &AQLQueue::CommandBuffer::append(const void *packet) {
  if (failed)
    return *this;
  size_t off = packets.size();
  if (auto grown = packets.resize_for_overwrite(off + PACKET_SIZE); !grown)
    return fail(grown.error());
  std::memcpy(packets.data() + off, packet, PACKET_SIZE);
  return *this;
}

AQLQueue::CommandBuffer &AQLQueue::CommandBuffer::dispatch(
    const Kernel &kernel, const DispatchConfig &cfg,
    std::span<std::byte> kernarg, aql::Ordering order, Signal *completion) {
  if (failed)
    return *this;
  if (cfg.grid_start.x || cfg.grid_start.y || cfg.grid_start.z ||
      cfg.grid_count.x || cfg.grid_count.y || cfg.grid_count.z)
    return fail(
        kfd::unexpected(ENOTSUP, "AQL dispatch cannot launch a sub-grid")
            .error());

  auto fits = [](uint32_t block, uint32_t grid) {
    return block <= UINT16_MAX && uint64_t(block) * grid <= UINT32_MAX;
  };
  if (!fits(cfg.block.x, cfg.grid.x) || !fits(cfg.block.y, cfg.grid.y) ||
      !fits(cfg.block.z, cfg.grid.z))
    return fail(
        kfd::unexpected(EINVAL, "launch dimensions exceed AQL limits").error());

  const abi::KernelDescriptor &kd = kernel.descriptor();
  uint32_t private_segment_size =
      private_segment_bytes(kd, cfg.private_segment_size);

  abi::DispatchPacket pkt;
  aql::kernel_dispatch(
      pkt, cfg.grid, cfg.block, private_segment_size,
      kd.group_segment_fixed_size + cfg.dynamic_lds,
      reinterpret_cast<uintptr_t>(kernel.address()) -
          static_cast<uint64_t>(kd.kernel_code_entry_byte_offset),
      kernarg.data(), completion ? completion->completion_handle() : 0, order);
  return append(&pkt);
}

// Each barrier packet holds back the queue until it completes, so a chain of
// them waits on every dependency.
AQLQueue::CommandBuffer &
AQLQueue::CommandBuffer::barrier(std::span<Signal *const> deps,
                                 aql::Ordering order, Signal *completion) {
  if (failed)
    return *this;
  size_t i = 0;
  do {
    uint64_t handles[aql::BARRIER_AND_DEPS] = {};
    size_t n = 0;
    for (; n < aql::BARRIER_AND_DEPS && i < deps.size(); ++n, ++i)
      handles[n] = deps[i]->completion_handle();
    bool last = i == deps.size();
    aql::BarrierAndPacket pkt;
    aql::barrier_and(pkt, {handles, n},
                     last && completion ? completion->completion_handle() : 0,
                     order);
    append(&pkt);
  } while (i < deps.size() && !failed);
  return *this;
}

AQLQueue::CommandBuffer &AQLQueue::CommandBuffer::pm4(const void *ib,
                                                      uint32_t dwords,
                                                      Signal &completion) {
  if (failed)
    return *this;
  uint32_t jump[pm4::INDIRECT_BUFFER_DWORDS];
  pm4::indirect_buffer(jump, ib, dwords);
  // Clear the PM4 packet format header when we send this through AQL, it
  // doesn't like it for some reason.
  jump[0] &= ~(1u << 1);
  aql::Pm4IndirectPacket pkt;
  aql::pm4_indirect(pkt, jump, completion.completion_handle());
  return append(&pkt);
}

std::expected<void, Error>
AQLQueue::CommandBuffer::submit(uint64_t timeout_ns) {
  if (failed)
    return kfd::unexpected(*failed);
  return queue.submit(
      std::span<const std::byte>(packets.data(), packets.size()), timeout_ns);
}

// SDMA queues execute packets in-order to copy between PCI-e or XGMI.
std::expected<SDMAQueue, Error>
SDMAQueue::create(Device &dev, size_t ring_size, uint32_t priority,
                  std::optional<uint32_t> engine_id) {
  uint32_t engines = dev.properties().num_sdma_engines;
  if (engine_id && *engine_id >= engines)
    return kfd::unexpected(EINVAL,
                           "SDMA engine %u is outside the %u PCIe engines",
                           *engine_id, engines);
  auto base = KFD_TRY(QueueBase::create(dev, QueueType::SDMA, ring_size,
                                        priority, /*target_xcc=*/0, engine_id));
  return SDMAQueue(std::move(base));
}

std::expected<XGMIQueue, Error>
XGMIQueue::create(Device &dev, size_t ring_size, uint32_t priority,
                  std::optional<uint32_t> engine_id) {
  uint32_t pcie = dev.properties().num_sdma_engines;
  uint32_t xgmi = dev.properties().num_sdma_xgmi_engines;
  if (engine_id && (*engine_id < pcie || *engine_id >= pcie + xgmi))
    return kfd::unexpected(EINVAL, "XGMI engine %u is outside [%u, %u)",
                           *engine_id, pcie, pcie + xgmi);
  auto base = KFD_TRY(QueueBase::create(dev, QueueType::SDMA_XGMI, ring_size,
                                        priority, /*target_xcc=*/0, engine_id));
  return XGMIQueue(std::move(base));
}

} // namespace kfd
