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
#include "libkfd/detail/small_vector.h"
#include "libkfd/detail/utility.h"

#include <bit>
#include <cerrno>
#include <csignal>
#include <cstddef>
#include <cstdio>
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

struct QueueErrorCtx {
  uint64_t *err_payload;
  uint32_t queue_id;
  uint32_t gpu_id;
};

struct ScratchRequest {
  uint32_t needed_per_thread;
  Dim3 block;
  uint16_t kernel_code_properties;
};

struct ScratchCtx {
  Device *dev = nullptr;

  uint32_t *scratch_done_ptr = nullptr;
  uint64_t *scratch_ready_ptr = nullptr;
  uint32_t *ib_buf = nullptr;

  void *scratch_va = nullptr;
  size_t scratch_size = 0;
  Buffer scratch_bo;
  uint32_t scratch_tmpring = 0;
  uint32_t scratch_per_thread = 0;

  detail::Mutex mtx;
  detail::SmallVector<ScratchRequest, 8> requests;

  uint32_t seq = 0;
  int error = 0;
};

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

void report_queue_exception(uint32_t queue_id, uint32_t gpu_id, uint64_t code) {
  std::fprintf(stderr, "GPU queue exception (queue %u, gpu_id %u): code 0x%lx",
               queue_id, gpu_id, static_cast<unsigned long>(code));
  if (code & KFD_EC_MASK(EC_QUEUE_WAVE_ABORT))
    std::fprintf(stderr, " wave-abort");
  if (code & KFD_EC_MASK(EC_QUEUE_WAVE_TRAP))
    std::fprintf(stderr, " wave-trap");
  if (code & KFD_EC_MASK(EC_QUEUE_WAVE_MATH_ERROR))
    std::fprintf(stderr, " math-error");
  if (code & KFD_EC_MASK(EC_QUEUE_WAVE_ILLEGAL_INSTRUCTION))
    std::fprintf(stderr, " illegal-inst");
  if (code & KFD_EC_MASK(EC_QUEUE_WAVE_MEMORY_VIOLATION))
    std::fprintf(stderr, " mem-violation");
  if (code & KFD_EC_MASK(EC_QUEUE_WAVE_APERTURE_VIOLATION))
    std::fprintf(stderr, " aperture-violation");
  std::fprintf(stderr, "\n");
}

} // namespace

void QueueBase::queue_error_handler(Event &, void *user_data) {
  auto *ectx = static_cast<QueueErrorCtx *>(user_data);
  uint64_t err = __atomic_load_n(ectx->err_payload, __ATOMIC_ACQUIRE);
  if (err) {
    report_queue_exception(ectx->queue_id, ectx->gpu_id, err);
    __atomic_store_n(ectx->err_payload, 0, __ATOMIC_RELEASE);
    ::raise(SIGABRT);
  }
}

// The GPU stalls for each pending scratch allocation. This function will be
// called once for every scratch request made.
void QueueBase::scratch_handler(Event &, void *user_data) {
  auto *sctx = static_cast<ScratchCtx *>(user_data);
  uint32_t ready = static_cast<uint32_t>(
      __atomic_load_n(sctx->scratch_ready_ptr, __ATOMIC_ACQUIRE));
  uint32_t done = __atomic_load_n(sctx->scratch_done_ptr, __ATOMIC_RELAXED);

  uint32_t next_seq = done + 1;
  if (next_seq > ready)
    return;

  // Fetch the scratch size and parameters the packet is requesting.
  ScratchRequest req;
  {
    detail::LockGuard guard(sctx->mtx);
    if (sctx->requests.empty())
      return;
    req = sctx->requests.front();
    sctx->requests.pop_front();
  }

  // Reallocate if the request exceeds the current scratch size.
  if (req.needed_per_thread >
      __atomic_load_n(&sctx->scratch_per_thread, __ATOMIC_RELAXED)) {
    void *old_va = __atomic_load_n(&sctx->scratch_va, __ATOMIC_RELAXED);
    if (old_va) {
      sctx->scratch_bo.release_device();
      sctx->scratch_bo = {};
      KFD_ASSERT(
          sctx->dev->scratch_allocator.deallocate(old_va, sctx->scratch_size));
      __atomic_store_n(&sctx->scratch_va, (void *)nullptr, __ATOMIC_RELAXED);
      sctx->scratch_size = 0;
      __atomic_store_n(&sctx->scratch_tmpring, 0u, __ATOMIC_RELAXED);
      __atomic_store_n(&sctx->scratch_per_thread, 0u, __ATOMIC_RELAXED);
    }

    // Fetch the number of slots that will be allocating the per-thread scratch.
    uint32_t gfx = sctx->dev->gfx_version();
    Dim3 block = req.block;
    uint32_t slots = detail::scratch_device_slots(*sctx->dev);
    uint32_t num_se = detail::scratch_num_se(*sctx->dev);
    uint32_t waves_per_group =
        detail::max(static_cast<uint32_t>(
                        (static_cast<uint64_t>(block.x) * block.y * block.z +
                         detail::SCRATCH_LANES_PER_WAVE - 1) /
                        detail::SCRATCH_LANES_PER_WAVE),
                    1u);
    uint32_t min_slots = 2 * num_se;
    bool allocated = false;

    // Repeatedly try to allocate scratch for the requested number of slots. If
    // there is insufficient storage we resize the slots and try again.
    while (slots >= min_slots) {
      size_t alloc_size =
          detail::scratch_alloc_size(gfx, req.needed_per_thread, slots);
      if (alloc_size == 0)
        break;

      auto region = sctx->dev->scratch_allocator.allocate(alloc_size);
      if (!region) {
        slots -= waves_per_group;
        slots = (slots / num_se) * num_se;
        continue;
      }

      auto bo = Buffer::allocate(*sctx->dev, alloc_size, MemType::VRAM,
                                 MemFlags::WRITABLE, *region);
      if (!bo) {
        KFD_ASSERT(
            sctx->dev->scratch_allocator.deallocate(*region, alloc_size));
        slots -= waves_per_group;
        slots = (slots / num_se) * num_se;
        continue;
      }

      if (auto r = bo->map(*sctx->dev); !r) {
        bo->release_device();
        *bo = {};
        KFD_ASSERT(
            sctx->dev->scratch_allocator.deallocate(*region, alloc_size));
        slots -= waves_per_group;
        slots = (slots / num_se) * num_se;
        continue;
      }

      __atomic_store_n(&sctx->scratch_va, *region, __ATOMIC_RELAXED);
      sctx->scratch_size = alloc_size;
      __atomic_store_n(&sctx->scratch_tmpring,
                       detail::compute_tmpring_size(
                           *sctx->dev, req.needed_per_thread, alloc_size),
                       __ATOMIC_RELAXED);
      __atomic_store_n(&sctx->scratch_per_thread, req.needed_per_thread,
                       __ATOMIC_RELAXED);
      sctx->scratch_bo = std::move(*bo);
      allocated = true;
      break;
    }

    if (!allocated) {
      sctx->ib_buf[0] = pm4::header(
          pm4::NOP,
          static_cast<uint16_t>(QueueBase::MAX_SCRATCH_IB_DWORDS - 2));
      __atomic_store_n(&sctx->error, ENOMEM, __ATOMIC_RELAXED);
      __atomic_store_n(sctx->scratch_done_ptr, next_seq, __ATOMIC_RELEASE);
      return;
    }
  }

  // Build the indirect buffer with the current scratch state. The GPU
  // executes this via INDIRECT_BUFFER after its WAIT_REG_MEM clears.
  void *ib_va = __atomic_load_n(&sctx->scratch_va, __ATOMIC_RELAXED);
  uint32_t ib_tmpring =
      __atomic_load_n(&sctx->scratch_tmpring, __ATOMIC_RELAXED);
  uint32_t gfx = sctx->dev->gfx_version();
  uint32_t ib_n = 0;
  ib_n += pm4::set_scratch_base(sctx->ib_buf + ib_n, gfx, ib_va, ib_tmpring);
  ib_n += set_scratch_sgprs(sctx->ib_buf + ib_n, gfx, ib_va, ib_tmpring,
                            req.kernel_code_properties);
  if (ib_n + 2 <= QueueBase::MAX_SCRATCH_IB_DWORDS) {
    uint32_t pad = QueueBase::MAX_SCRATCH_IB_DWORDS - ib_n;
    sctx->ib_buf[ib_n] = pm4::header(pm4::NOP, static_cast<uint16_t>(pad - 2));
  }

  // We are done, fire the signal to unblock the WAIT_REG_MEM stalling the CP.
  __atomic_store_n(sctx->scratch_done_ptr, next_seq, __ATOMIC_RELEASE);
}

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

    void *cwsr_data = nullptr;
    bool svm_supported =
        (props.capability & NodeProperties::NODE_CAP_SVMAPI_SUPPORTED) != 0;

    // CWSR can fire at any time so we must ensure that the pages are resident.
    // If SVM is supported we use an always-mapped page, otherwise we pin
    // anonymous pages as a USERPTR BO (matching libhsakmt's dGPU fallback).
    auto cwsr_region = KFD_TRY(MappedRegion::create(total_cwsr));
    std::memset(cwsr_region.data(), 0, cwsr_region.size());
    if (svm_supported) {
      KFD_CHECK(register_svm(ctx.kfd_fd(), cwsr_region.data(),
                             cwsr_region.size(), dev.gpu_id()));
    } else {
      constexpr MemFlags CWSR_PIN_FLAGS =
          MemFlags::WRITABLE | MemFlags::EXECUTABLE | MemFlags::COHERENT;
      auto bo = KFD_TRY(
          Buffer::pin(dev, cwsr_region.data(), total_cwsr, CWSR_PIN_FLAGS));
      KFD_CHECK(bo.map(dev));
      cwsr_bo_buf = std::move(bo);
    }
    cwsr_data = cwsr_region.data();
    cwsr_buf = std::move(cwsr_region);

    for (uint32_t i = 0; i < num_xcc; ++i) {
      auto *hdr = reinterpret_cast<abi::CwsrHeader *>(
          static_cast<char *>(cwsr_data) +
          static_cast<size_t>(i * props.cwsr_size));
      hdr->debug_offset = (num_xcc - i) * props.cwsr_size;
      hdr->debug_size = debug_mem * num_xcc;
      hdr->err_payload_addr = reinterpret_cast<uint64_t>(&ctl->err_payload);
      hdr->err_event_id = err_ev->event_id();
    }
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
  if (cwsr_buf || cwsr_bo_buf) {
    void *cwsr_ptr = cwsr_buf ? cwsr_buf.data() : cwsr_bo_buf.data();
    args.ctx_save_restore_address = reinterpret_cast<uintptr_t>(cwsr_ptr);
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
    KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_QUEUE>(ctx.kfd_fd(), dq));
    return kfd::unexpected(db_slot.error());
  }

  auto mtx = KFD_TRY(detail::Box<detail::Mutex>::create());
  QueueBase q(type, ctx, dev, args.queue_id, std::move(ctl_buf),
              std::move(ring_buf), std::move(eop_buf), std::move(cwsr_buf),
              std::move(cwsr_bo_buf), *db_slot, std::move(err_ev),
              std::move(mtx));

  if (q.err_event) {
    q.err_watch_ctx = KFD_TRY(detail::Box<QueueErrorCtx>::create(
        QueueErrorCtx{reinterpret_cast<uint64_t *>(&q.ctl()->err_payload), q.id,
                      dev.gpu_id()}));
    KFD_CHECK(ctx.watch_event(*q.err_event, queue_error_handler,
                              q.err_watch_ctx.get()));
  }

  if (is_compute) {
    auto raw_ev = KFD_TRY(Event::create(ctx));
    q.scratch_event = KFD_TRY(detail::Box<Event>::create(std::move(raw_ev)));

    q.scratch_watch_ctx = KFD_TRY(detail::Box<ScratchCtx>::create());
    auto *sctx = q.scratch_watch_ctx.get();
    sctx->dev = &dev;
    sctx->scratch_done_ptr = &ctl->scratch_done;
    sctx->scratch_ready_ptr = &ctl->scratch_ready;
    sctx->ib_buf = reinterpret_cast<uint32_t *>(ctl->indirect);
    KFD_CHECK(ctx.watch_event(*q.scratch_event, scratch_handler, sctx));
  }

  return q;
}

QueueBase::QueueBase(QueueType type, Context &ctx, Device &dev, uint32_t id,
                     Buffer control, Buffer ring, Buffer eop,
                     detail::MappedRegion cwsr, Buffer cwsr_bo,
                     volatile uint64_t *doorbell, detail::Box<Event> err_event,
                     detail::Box<detail::Mutex> submit_mtx)
    : type(type), ctx(&ctx), dev(&dev), id(id), control(std::move(control)),
      ring(std::move(ring)), eop(std::move(eop)), cwsr(std::move(cwsr)),
      cwsr_bo(std::move(cwsr_bo)), doorbell(doorbell),
      err_event(std::move(err_event)), submit_mtx(std::move(submit_mtx)) {}

QueueBase::~QueueBase() {
  if (!ctx)
    return;

  if (err_event)
    KFD_ASSERT(ctx->unwatch_event(*err_event));
  if (scratch_event)
    KFD_ASSERT(ctx->unwatch_event(*scratch_event));

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
      control(std::move(other.control)), ring(std::move(other.ring)),
      eop(std::move(other.eop)), cwsr(std::move(other.cwsr)),
      cwsr_bo(std::move(other.cwsr_bo)),
      doorbell(std::exchange(other.doorbell, nullptr)),
      err_event(std::move(other.err_event)),
      err_watch_ctx(std::move(other.err_watch_ctx)),
      scratch_event(std::move(other.scratch_event)),
      scratch_watch_ctx(std::move(other.scratch_watch_ctx)),
      submit_mtx(std::move(other.submit_mtx)),
      pending_wptr(std::exchange(other.pending_wptr, 0)) {}

QueueBase &QueueBase::operator=(QueueBase &&other) {
  if (this == &other)
    return *this;

  if (ctx) {
    if (err_event)
      KFD_ASSERT(ctx->unwatch_event(*err_event));
    if (scratch_event)
      KFD_ASSERT(ctx->unwatch_event(*scratch_event));
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
  submit_mtx = std::move(other.submit_mtx);
  pending_wptr = std::exchange(other.pending_wptr, 0);
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

  KFD_CHECK(wait_for_room(n));

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
  KFD_CHECK(q.base.submit_impl(
      init, static_cast<size_t>(2 * pm4::WRITE_DATA_DWORDS)));
  return q;
}

// Dispatch a complete kernel launch on the compute queue. Scratch handling is
// by far the most complicated piece here. Functionally, it defers allocation to
// the watcher thread to be handled in the background without stalling the user.
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

  auto *sctx = base.scratch_watch_ctx.get();
  if (__atomic_load_n(&sctx->error, __ATOMIC_RELAXED))
    return kfd::unexpected(sctx->error, "scratch allocation failed");

  bool needs_resize = false;
  uint32_t current_seq = sctx->seq;

  // This launch requires more scratch than we currently have allocated. We will
  // need to push a request and stall the CP until the previous work has
  // completed and the new scratch region is allocated by the watcher thread.
  if (private_segment_size >
      __atomic_load_n(&sctx->scratch_per_thread, __ATOMIC_RELAXED)) {
    size_t per_wave = detail::align_up(
        size_t(detail::SCRATCH_LANES_PER_WAVE) * private_segment_size,
        size_t(detail::scratch_alignment_unit(base.dev->gfx_version())));
    if (per_wave > detail::max_wave_scratch(base.dev->gfx_version()))
      return kfd::unexpected(
          ERANGE, "scratch %u B exceeds hardware per-wave limit (%u B / wave)",
          private_segment_size,
          detail::max_wave_scratch(base.dev->gfx_version()));

    ScratchRequest req{private_segment_size, cfg.block,
                       kd.kernel_code_properties};

    detail::LockGuard request_guard(sctx->mtx);
    KFD_CHECK(sctx->requests.push_back(req));

    current_seq = ++sctx->seq;
    needs_resize = true;
  }

  uint32_t done = __atomic_load_n(&base.ctl()->scratch_done, __ATOMIC_ACQUIRE);
  bool needs_ib_stall = private_segment_size > 0 && current_seq > done;

  const void *dispatch_pkt_addr = nullptr;
  if (kd.kernel_code_properties & abi::ENABLE_SGPR_DISPATCH_PTR)
    dispatch_pkt_addr =
        static_cast<std::byte *>(kernarg.data()) +
        detail::align_up(static_cast<size_t>(kd.kernarg_size), size_t(64));

  // Perform the standard register setup for the PM4 compute dispatch.
  uint32_t buf[pm4::MAX_DISPATCH_DWORDS];
  void *va = __atomic_load_n(&sctx->scratch_va, __ATOMIC_RELAXED);
  uint32_t tmpring = __atomic_load_n(&sctx->scratch_tmpring, __ATOMIC_RELAXED);
  auto n = pm4::build_dispatch_setup(
      buf, base.dev->gfx_version(), kd, kernel.address, cfg.grid, cfg.block,
      kernarg.data(), dispatch_pkt_addr, va, tmpring, cfg.dynamic_lds,
      private_segment_size);

  // Here we are either waiting for a scratch allocation or stalled behind
  // another kernel that is. We already encoded the scratch base so we use an
  // indirect buffer to override it dynamically once the memory is allocated.
  if (needs_ib_stall) {
    if (needs_resize) {
      // Wake up the context watcher thread once the previous work has finished.
      n += pm4::release_mem(buf + n, base.dev->gfx_version(),
                            &base.ctl()->scratch_ready, current_seq);
      n += pm4::release_mem(
          buf + n, base.dev->gfx_version(), base.scratch_event->signal_addr(),
          base.scratch_event->event_id(), base.scratch_event->trigger_data());
    }

    // Stall the CP until the watcher thread has finished the allocation.
    n += pm4::wait_reg_mem(buf + n, base.dev->gfx_version(),
                           &base.ctl()->scratch_done, Condition::GTE,
                           current_seq);
    n += pm4::indirect_buffer(buf + n, base.ctl()->indirect,
                              QueueBase::MAX_SCRATCH_IB_DWORDS,
                              pm4::CachePolicy::POLICY_BYPASS);
  }

  // Push the final dispatch with the full configuration.
  n += pm4::dispatch_direct(
      buf + n, cfg.grid.x, cfg.grid.y, cfg.grid.z,
      pm4::dispatch_initiator(kd, base.dev->gfx_version()));

  return base.submit_impl(buf, n);
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
