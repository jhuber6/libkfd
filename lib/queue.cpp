//===-- lib/queue.cpp - Queue implementation --------------------*- C++ -*-===//
//
// Queue creation, destruction, ring buffer management, and doorbell submission
// for PM4 compute and SDMA queues.
//
//===----------------------------------------------------------------------===//

#include "libkfd/queue.h"
#include "ioctl.h"
#include "libkfd/context.h"
#include "libkfd/detail/scratch.h"
#include "libkfd/detail/utility.h"

#include <bit>
#include <cerrno>
#include <cstddef>
#include <cstring>
#include <ctime>

static_assert(static_cast<uint8_t>(kfd::QueueType::COMPUTE) ==
              KFD_IOC_QUEUE_TYPE_COMPUTE);
static_assert(static_cast<uint8_t>(kfd::QueueType::SDMA) ==
              KFD_IOC_QUEUE_TYPE_SDMA);
static_assert(static_cast<uint8_t>(kfd::QueueType::SDMA_XGMI) ==
              KFD_IOC_QUEUE_TYPE_SDMA_XGMI);

using namespace kfd::detail;

namespace kfd {

namespace {

// Value is hard-coded by the kernel driver.
constexpr size_t EOP_BUFFER_SIZE = 4096;
constexpr MemFlags QUEUE_GTT_FLAGS = MemFlags::WRITABLE | MemFlags::EXECUTABLE |
                                     MemFlags::NO_SUBSTITUTE |
                                     MemFlags::COHERENT | MemFlags::UNCACHED;

// Register an anonymous mmap range as SVM with GPU_ALWAYS_MAPPED.
std::expected<void, Error> register_svm(int kfd_fd, void *addr, size_t size,
                                        uint32_t gpu_id) {
  constexpr uint32_t NATTR = 6;
  constexpr size_t ATTR_BYTES = NATTR * sizeof(ioctl::kfd::svm_attribute);
  constexpr size_t TOTAL = sizeof(ioctl::kfd::svm_args) + ATTR_BYTES;

  alignas(ioctl::kfd::svm_args) char buf[TOTAL]{};
  auto *args = reinterpret_cast<ioctl::kfd::svm_args *>(buf);
  ioctl::kfd::svm_attribute *attrs = args->attrs;

  uint32_t flags = KFD_IOCTL_SVM_FLAG_HOST_ACCESS |
                   KFD_IOCTL_SVM_FLAG_GPU_EXEC |
                   KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED;

  args->start_addr = reinterpret_cast<uintptr_t>(addr);
  args->size = size;
  args->op = KFD_IOCTL_SVM_OP_SET_ATTR;
  args->nattr = NATTR;

  attrs[0] = {.type = KFD_IOCTL_SVM_ATTR_PREFETCH_LOC, .value = gpu_id};
  attrs[1] = {.type = KFD_IOCTL_SVM_ATTR_PREFERRED_LOC,
              .value = KFD_IOCTL_SVM_LOCATION_SYSMEM};
  attrs[2] = {.type = KFD_IOCTL_SVM_ATTR_CLR_FLAGS, .value = ~flags};
  attrs[3] = {.type = KFD_IOCTL_SVM_ATTR_SET_FLAGS, .value = flags};
  attrs[4] = {.type = KFD_IOCTL_SVM_ATTR_ACCESS, .value = gpu_id};
  attrs[5] = {.type = KFD_IOCTL_SVM_ATTR_GRANULARITY, .value = 0xFF};

  return ioctl::call<ioctl::kfd::SVM>(kfd_fd, *args, ATTR_BYTES);
}

} // namespace

// Create a queue to communicate with the device's command processor. Each queue
// is a ring buffer that consumes variable length packets for the processor.
std::expected<QueueBase, Error> QueueBase::create(Device &dev, QueueType type,
                                                  size_t ring_size) {
  Context &ctx = dev.context();
  if (!std::has_single_bit(ring_size))
    return kfd::unexpected(EINVAL, "ring size %zu is not a power of two",
                           ring_size);
  if (ring_size > UINT32_MAX)
    return kfd::unexpected(EINVAL, "ring size %zu exceeds uint32_t", ring_size);

  const bool is_compute = type == QueueType::COMPUTE;

  // Control page holds the read pointer, write pointer, EOP sequence, and
  // exception payload.
  auto ctl_buf = KFD_TRY(Buffer::allocate(dev, sizeof(QueueControl),
                                          MemType::GTT, QUEUE_GTT_FLAGS));
  KFD_CHECK(ctl_buf.map(dev));
  std::memset(ctl_buf.data(), 0, ctl_buf.size());
  auto *ctl = static_cast<QueueControl *>(ctl_buf.data());

  // CWSR buffer required on GFX9+ for HWS preemption (compute only).
  // The header wires up err_payload_addr and err_event_id so KFD can
  // signal queue exceptions (traps, illegal instructions, etc.).
  MappedRegion cwsr_buf;
  Event err_ev;
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

    auto cwsr_region = KFD_TRY(MappedRegion::create(total_cwsr));
    std::memset(cwsr_region.data(), 0, cwsr_region.size());

    // Each queue has an event for the CWSR to handle device-side interrupts.
    err_ev = KFD_TRY(Event::create(ctx));
    uint32_t err_event_id = err_ev.event_id();

    for (uint32_t i = 0; i < num_xcc; ++i) {
      auto *hdr = reinterpret_cast<abi::CwsrHeader *>(
          static_cast<char *>(cwsr_region.data()) +
          static_cast<size_t>(i * props.cwsr_size));
      hdr->debug_offset = (num_xcc - i) * props.cwsr_size;
      hdr->debug_size = debug_mem * num_xcc;
      hdr->err_payload_addr = reinterpret_cast<uint64_t>(&ctl->err_payload);
      hdr->err_event_id = err_event_id;
    }

    // CWSR can fire at any time so we must ensure that the pages are resident.
    KFD_CHECK(register_svm(ctx.kfd_fd(), cwsr_region.data(), cwsr_region.size(),
                           dev.gpu_id()));

    cwsr_buf = std::move(cwsr_region);
  }

  // The ring buffer itself that holds the packets.
  auto ring_buf =
      KFD_TRY(Buffer::allocate(dev, ring_size, MemType::GTT, QUEUE_GTT_FLAGS));
  KFD_CHECK(ring_buf.map(dev));
  std::memset(ring_buf.data(), 0, ring_buf.size());

  // An internal end-of-pipe buffer used by the CP to manage EOP events.
  Buffer eop_buf;
  if (is_compute) {
    eop_buf = KFD_TRY(Buffer::allocate(
        dev, EOP_BUFFER_SIZE, MemType::VRAM,
        MemFlags::WRITABLE | MemFlags::EXECUTABLE | MemFlags::NO_SUBSTITUTE));
    KFD_CHECK(eop_buf.map(dev));
  }

  // Ensure all the queue memory is committed before submitting it.
  memory_barrier();
  ioctl::kfd::create_queue_args args{
      .ring_base_address = reinterpret_cast<uintptr_t>(ring_buf.data()),
      .write_pointer_address = reinterpret_cast<uintptr_t>(&ctl->write_ptr),
      .read_pointer_address = reinterpret_cast<uintptr_t>(&ctl->read_ptr),
      .ring_size = static_cast<uint32_t>(ring_size),
      .gpu_id = dev.gpu_id(),
      .queue_type = static_cast<uint32_t>(type),
      .queue_percentage = 100,
      .queue_priority = 7,
  };
  if (is_compute) {
    args.eop_buffer_address = reinterpret_cast<uintptr_t>(eop_buf.data());
    args.eop_buffer_size = EOP_BUFFER_SIZE;
  }
  if (cwsr_buf) {
    args.ctx_save_restore_address =
        reinterpret_cast<uintptr_t>(cwsr_buf.data());
    args.ctx_save_restore_size = props.cwsr_size;
    args.ctl_stack_size = props.ctl_stack_size;
  }

  // Queues are a contended resource, if we fail to get one just retry.
  constexpr int QUEUE_CREATE_RETRIES = 20;
  constexpr long QUEUE_RETRY_MS = 100;
  for (int attempt = 0;; ++attempt) {
    auto r = ioctl::call<ioctl::kfd::CREATE_QUEUE>(ctx.kfd_fd(), args);
    if (r)
      break;
    if (r.error().code != ENOMEM || attempt >= QUEUE_CREATE_RETRIES)
      return kfd::unexpected(r.error());
    struct timespec ts = {.tv_sec = 0, .tv_nsec = QUEUE_RETRY_MS * 1'000'000L};
    ::nanosleep(&ts, nullptr);
  }

  // Doorbells are MMIO mapped registers that signal the CP to consume packets.
  std::expected<volatile uint64_t *, Error> db_slot =
      dev.doorbell(args.doorbell_offset);
  if (!db_slot) {
    ioctl::kfd::destroy_queue_args dq{.queue_id = args.queue_id};
    ioctl::call<ioctl::kfd::DESTROY_QUEUE>(ctx.kfd_fd(), dq);
    return kfd::unexpected(db_slot.error());
  }

  auto mtx = KFD_TRY(detail::Box<detail::Mutex>::create());
  QueueBase q(type, ctx, dev, args.queue_id, std::move(ctl_buf),
              std::move(ring_buf), std::move(eop_buf), std::move(cwsr_buf),
              *db_slot, std::move(err_ev), std::move(mtx));

  if (q.err_event) {
    auto *payload = reinterpret_cast<uint64_t *>(&q.ctl()->err_payload);
    ctx.register_queue_error(q.err_event.event_id(), payload, q.id,
                             dev.gpu_id());
  }

  return q;
}

QueueBase::QueueBase(QueueType type, Context &ctx, Device &dev, uint32_t id,
                     Buffer control, Buffer ring, Buffer eop,
                     detail::MappedRegion cwsr, volatile uint64_t *doorbell,
                     Event err_event, detail::Box<detail::Mutex> submit_mtx)
    : type(type), ctx(&ctx), dev(&dev), id(id), control(std::move(control)),
      ring(std::move(ring)), eop(std::move(eop)), cwsr(std::move(cwsr)),
      doorbell(doorbell), err_event(std::move(err_event)),
      submit_mtx(std::move(submit_mtx)) {}

QueueBase::~QueueBase() {
  if (!ctx)
    return;
  if (err_event)
    ctx->unregister_queue_error(err_event.event_id());
  ioctl::kfd::destroy_queue_args args{.queue_id = id};
  KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_QUEUE>(ctx->kfd_fd(), args));
  if (scratch_va) {
    scratch_bo.release_device();
    scratch_bo = {};
    KFD_ASSERT(dev->scratch_allocator.deallocate(scratch_va, scratch_size));
  }
}

QueueBase::QueueBase(QueueBase &&other)
    : type(other.type), ctx(std::exchange(other.ctx, nullptr)),
      dev(std::exchange(other.dev, nullptr)), id(std::exchange(other.id, 0)),
      control(std::move(other.control)), ring(std::move(other.ring)),
      eop(std::move(other.eop)), cwsr(std::move(other.cwsr)),
      doorbell(std::exchange(other.doorbell, nullptr)),
      err_event(std::move(other.err_event)),
      submit_mtx(std::move(other.submit_mtx)),
      pending_wptr(std::exchange(other.pending_wptr, 0)),
      scratch_bo(std::move(other.scratch_bo)),
      scratch_va(std::exchange(other.scratch_va, nullptr)),
      scratch_size(std::exchange(other.scratch_size, 0)) {}

QueueBase &QueueBase::operator=(QueueBase &&other) {
  if (this == &other)
    return *this;

  if (ctx) {
    if (err_event)
      ctx->unregister_queue_error(err_event.event_id());
    ioctl::kfd::destroy_queue_args dq{.queue_id = id};
    KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_QUEUE>(ctx->kfd_fd(), dq));
    if (scratch_va) {
      scratch_bo.release_device();
      scratch_bo = {};
      KFD_ASSERT(dev->scratch_allocator.deallocate(scratch_va, scratch_size));
    }
  }

  type = other.type;
  ctx = std::exchange(other.ctx, nullptr);
  dev = std::exchange(other.dev, nullptr);
  id = std::exchange(other.id, 0);
  ring = std::move(other.ring);
  control = std::move(other.control);
  eop = std::move(other.eop);
  cwsr = std::move(other.cwsr);
  doorbell = std::exchange(other.doorbell, nullptr);
  err_event = std::move(other.err_event);
  submit_mtx = std::move(other.submit_mtx);
  pending_wptr = std::exchange(other.pending_wptr, 0);
  scratch_bo = std::move(other.scratch_bo);
  scratch_va = std::exchange(other.scratch_va, nullptr);
  scratch_size = std::exchange(other.scratch_size, 0);
  return *this;
}

std::expected<void, Error> QueueBase::wait_for_room(uint32_t dwords) {
  uint32_t cap = static_cast<uint32_t>(ring_dwords());
  uint32_t pos = static_cast<uint32_t>(pending_wptr & (cap - 1));
  volatile uint32_t *rp_addr =
      reinterpret_cast<volatile uint32_t *>(&ctl()->read_ptr);
  constexpr uint64_t TIMEOUT_US = 5'000'000;
  struct timespec now;
  ::clock_gettime(CLOCK_MONOTONIC, &now);
  uint64_t deadline_ns = static_cast<uint64_t>(now.tv_sec) * 1'000'000'000 +
                         static_cast<uint64_t>(now.tv_nsec) +
                         TIMEOUT_US * 1'000;
  for (;;) {
    uint32_t rp = __atomic_load_n(rp_addr, __ATOMIC_ACQUIRE);
    if (type == QueueType::SDMA)
      rp /= sizeof(uint32_t);
    uint32_t in_flight = (pos + cap - rp) % cap;
    if (in_flight + dwords < cap)
      return {};
    ::clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t now_ns = static_cast<uint64_t>(now.tv_sec) * 1'000'000'000 +
                      static_cast<uint64_t>(now.tv_nsec);
    if (now_ns >= deadline_ns)
      return kfd::unexpected(
          ETIMEDOUT, "ring buffer stall waiting for %u dwords of wrap padding",
          dwords);
    detail::spin_hint();
  }
}

std::expected<void, Error> QueueBase::submit(const uint32_t *data,
                                             size_t n_dwords) {
  detail::LockGuard guard(*submit_mtx);
  return submit_impl(data, n_dwords);
}

std::expected<void, Error> QueueBase::submit_impl(const uint32_t *data,
                                                  size_t n_dwords) {
  auto n = static_cast<uint32_t>(n_dwords);
  if (n == 0)
    return {};

  uint32_t cap = static_cast<uint32_t>(ring_dwords());
  if (n >= cap)
    return kfd::unexpected(
        EMSGSIZE, "submit size %u dwords exceeds ring capacity %u", n, cap);

  uint32_t *base = static_cast<uint32_t *>(ring.data());
  uint32_t mask = cap - 1;
  uint32_t pos = static_cast<uint32_t>(pending_wptr & mask);

  auto to_hw = [this](uint64_t wptr) -> uint64_t {
    return type == QueueType::SDMA ? wptr * sizeof(uint32_t) : wptr;
  };

  volatile uint64_t *wptr_p = &ctl()->write_ptr;

  if (pos + n > cap) {
    uint32_t pad = cap - pos;
    KFD_CHECK(wait_for_room(pad));

    if (type == QueueType::SDMA) {
      base[pos] = static_cast<uint32_t>(pad - 1) << 16;
      for (uint32_t i = 1; i < pad; ++i)
        base[pos + i] = 0;
    } else {
      for (uint32_t i = 0; i < pad; ++i)
        base[pos + i] = pm4::CMD_NOP;
    }

    pending_wptr += pad;
    uint64_t hw = to_hw(pending_wptr);
    memory_barrier();
    *wptr_p = hw;
    memory_barrier();
    *doorbell = hw;
    pos = 0;
  }

  if (auto r = wait_for_room(n); !r)
    return r;

  std::memcpy(base + pos, data, n * sizeof(uint32_t));
  pending_wptr += n;

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
                                                        size_t ring_size) {
  auto base = KFD_TRY(QueueBase::create(dev, QueueType::COMPUTE, ring_size));

  // Used to signal the completion of end-of-pipe packets to the queue. The CP
  // will poll this value repeatedly for the finished value so it lives in VRAM.
  auto eop_vram = KFD_TRY(Buffer::allocate(
      dev, sizeof(uint64_t), MemType::VRAM,
      MemFlags::WRITABLE | MemFlags::NO_SUBSTITUTE | MemFlags::UNCACHED));
  KFD_CHECK(eop_vram.map(dev));

  ComputeQueue q(std::move(base), std::move(eop_vram));

  uint32_t init[2 * pm4::WRITE_DATA_DWORDS];
  auto *eop_u32 = static_cast<uint32_t *>(q.eop_seq.data());
  pm4::write_data(init, eop_u32, 0);
  pm4::write_data(init + pm4::WRITE_DATA_DWORDS, eop_u32 + 1, 0);
  KFD_CHECK(q.base.submit_impl(init, 2 * pm4::WRITE_DATA_DWORDS));
  return q;
}

std::expected<void, Error>
ComputeQueue::release_scratch_region(void *va, size_t size, Buffer *bo) {
  if (bo) {
    bo->release_device();
    *bo = {};
  }
  return base.dev->scratch_allocator.deallocate(va, size);
}

std::expected<void, Error> ComputeQueue::try_scratch_alloc(uint32_t per_thread,
                                                           uint32_t slots) {
  size_t alloc_size =
      detail::scratch_alloc_size(base.dev->gfx_version(), per_thread, slots);
  if (alloc_size == 0)
    return {};

  auto region = KFD_TRY(base.dev->scratch_allocator.allocate(alloc_size));

  auto bo = Buffer::allocate(*base.dev, alloc_size, MemType::VRAM,
                             MemFlags::WRITABLE, region);
  if (!bo) {
    KFD_ASSERT(release_scratch_region(region, alloc_size));
    return kfd::unexpected(bo.error());
  }
  if (auto r = bo->map(*base.dev); !r) {
    KFD_ASSERT(release_scratch_region(region, alloc_size, &*bo));
    return kfd::unexpected(r.error());
  }

  base.scratch_va = region;
  base.scratch_size = alloc_size;
  scratch_tmpring =
      detail::compute_tmpring_size(*base.dev, per_thread, alloc_size);
  scratch_per_thread = per_thread;
  scratch_region_size = alloc_size;
  base.scratch_bo = std::move(*bo);
  return {};
}

// Dispatch a complete kernel launch on the compute queue.
std::expected<void, Error> ComputeQueue::dispatch(const Kernel &kernel,
                                                  const DispatchConfig &cfg,
                                                  const Buffer &kernarg) {
  const abi::KernelDescriptor &kd = *kernel.descriptor;
  if (kd.kernarg_preload & abi::KERNARG_PRELOAD_LENGTH_MASK)
    return unexpected(ENOTSUP, "kernarg preload not yet supported");

  uint32_t private_segment_size = cfg.private_segment_size;
  if (!(kd.kernel_code_properties & abi::USES_DYNAMIC_STACK))
    private_segment_size = kd.private_segment_fixed_size;
  else if (!private_segment_size)
    private_segment_size = /*8 KiB=*/8 * 1024;

  detail::LockGuard guard(*base.submit_mtx);

  if (auto r = ensure_scratch(private_segment_size, cfg.block); !r)
    return r;

  const void *dispatch_pkt_addr = nullptr;
  if (kd.kernel_code_properties & abi::ENABLE_SGPR_DISPATCH_PTR)
    dispatch_pkt_addr =
        static_cast<std::byte *>(kernarg.data()) +
        detail::align_up(static_cast<size_t>(kd.kernarg_size), size_t(64));

  uint32_t buf[pm4::MAX_DISPATCH_DWORDS];
  auto n = pm4::build_dispatch(
      buf, base.dev->gfx_version(), kd, kernel.address, cfg.grid, cfg.block,
      kernarg.data(), dispatch_pkt_addr, base.scratch_va, scratch_tmpring,
      cfg.dynamic_lds, private_segment_size);
  return base.submit_impl(buf, n);
}

uint32_t ComputeQueue::build_signal_packet(uint32_t *buf, Signal &sig) {
  // We use a RELEASE_MEM packet with the sequence counter and cache-bypass to
  // wait until the necessary work and cache invalidation has completed. Then
  // we decrement the signal value and fire an event.
  uint32_t seq = __atomic_fetch_add(&next_eop_seq, 1, __ATOMIC_RELAXED) + 1;
  uint32_t n = 0;
  n += pm4::release_mem(buf + n, base.dev->gfx_version(), eop_seq.data(),
                        static_cast<uint64_t>(seq));
  n += pm4::wait_reg_mem(buf + n, base.dev->gfx_version(), eop_seq.data(),
                         Condition::GTE, seq);
  n += pm4::atomic_mem(buf + n, pm4::ATOMIC_ADD_64, sig.fence_addr(),
                       int64_t(-1), 0, pm4::ATOMIC_WAIT_CONFIRM);
  n += pm4::release_mem(buf + n, base.dev->gfx_version(), sig.signal_addr(),
                        sig.event_id(), sig.trigger_data());
  return n;
}

std::expected<void, Error> ComputeQueue::ensure_scratch(uint32_t needed,
                                                        Dim3 block) {
  if (needed <= scratch_per_thread)
    return {};

  size_t per_wave = detail::align_up(
      size_t(detail::SCRATCH_LANES_PER_WAVE) * needed,
      size_t(detail::scratch_alignment_unit(base.dev->gfx_version())));
  if (per_wave > detail::max_wave_scratch(base.dev->gfx_version()))
    return kfd::unexpected(
        ERANGE, "scratch %u B exceeds hardware per-wave limit (%u B / wave)",
        needed, detail::max_wave_scratch(base.dev->gfx_version()));

  // Drain all in-flight work so the current scratch region has no remaining
  // references. Using PM4 does not allow the independent sizing that AQL uses.
  // TODO: Stop this from blocking the user thread using an indirect buffer to
  //       overwrite the scratch base register allocation.
  if (base.scratch_va) {
    auto sig = KFD_TRY(Signal::create(*base.ctx));

    // We are inside the queue mutex.
    uint32_t buf[SIGNAL_DWORDS];
    uint32_t n = build_signal_packet(buf, sig);
    if (auto r = base.submit_impl(buf, n); !r)
      return r;

    if (auto r = sig.wait(Condition::EQ, 0, UINT64_MAX); !r)
      return r;

    KFD_CHECK(release_scratch_region(base.scratch_va, base.scratch_size,
                                     &base.scratch_bo));
    base.scratch_va = nullptr;
    base.scratch_size = 0;
    scratch_tmpring = 0;
    scratch_per_thread = 0;
    scratch_region_size = 0;
  }

  // Size scratch to the full device capacity, If scratch memory is exhausted,
  // progressively reduce the wave count until something fits.
  uint32_t slots = detail::scratch_device_slots(*base.dev);
  uint32_t num_se = detail::scratch_num_se(*base.dev);
  uint32_t waves_per_group =
      detail::max(static_cast<uint32_t>((uint64_t(block.x) * block.y * block.z +
                                         detail::SCRATCH_LANES_PER_WAVE - 1) /
                                        detail::SCRATCH_LANES_PER_WAVE),
                  1u);

  auto retry = [](const std::expected<void, Error> &r) {
    return !r && r.error().code == ENOMEM;
  };
  uint32_t min_slots = 2 * num_se;
  while (slots >= min_slots) {
    if (auto r = try_scratch_alloc(needed, slots); !retry(r))
      return r;
    slots -= waves_per_group;
    slots = (slots / num_se) * num_se;
  }

  return kfd::unexpected(ENOMEM, "scratch allocation failed after reducing "
                                 "occupancy to minimum");
}

// SDMA queues execute packets in-order to copy between PCI-e or XGMI.
std::expected<SDMAQueue, Error> SDMAQueue::create(Device &dev, QueueType type,
                                                  size_t ring_size) {
  if (type == QueueType::COMPUTE)
    return kfd::unexpected(EINVAL, "COMPUTE queue passed to SDMA queue");
  auto base = KFD_TRY(QueueBase::create(dev, type, ring_size));
  return SDMAQueue(std::move(base), type);
}

} // namespace kfd
