//===-- libkfd/queue.h - Device command processor queues --------*- C++ -*-===//
//
// Queues are the primary way to interact with the device's command processor.
// These are simple ring buffers that submit packets to the device from user
// space. PM4 compute queues modify memory and register state while SDMA queues
// handle memory movement. Queue submission is handled through ringing MMIO
// doorbell pages.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_QUEUE_H
#define LIBKFD_QUEUE_H

#include "libkfd/abi.h"
#include "libkfd/detail/box.h"
#include "libkfd/detail/mutex.h"
#include "libkfd/detail/utility.h"
#include "libkfd/device.h"
#include "libkfd/loader.h"
#include "libkfd/memory.h"
#include "libkfd/packets/pm4.h"
#include "libkfd/packets/sdma.h"
#include "libkfd/signal.h"

#include "libkfd/error.h"

#include <cstdint>
#include <cstring>

namespace kfd {

class Context;
struct QueueErrorCtx;
struct ScratchCtx;

enum class QueueType : uint8_t {
  COMPUTE = /*KFD_IOC_QUEUE_TYPE_COMPUTE=*/0x0,
  SDMA = /*KFD_IOC_QUEUE_TYPE_SDMA=*/0x1,
  SDMA_XGMI = /*KFD_IOC_QUEUE_TYPE_SDMA_XGMI=*/0x3,
};

// Shared kernel-facing queue state. ComputeQueue and SDMAQueue each hold one
// of these and forward through it; users interact with the concrete wrappers.
class QueueBase {
public:
  ~QueueBase();

  QueueBase(const QueueBase &) = delete;
  QueueBase &operator=(const QueueBase &) = delete;
  QueueBase(QueueBase &&other);
  QueueBase &operator=(QueueBase &&other);

  uint32_t queue_id() const { return id; }
  size_t ring_dwords() const { return ring.size() / sizeof(uint32_t); }
  explicit operator bool() const { return ctx != nullptr; }

private:
  friend class ComputeQueue;
  friend class SDMAQueue;

  // Shared memory written by the kernel/hardware and read by userspace.
  //
  // On GFX9+ both wptr and the doorbell are 64-bit. The read_ptr width
  // depends on the engine:
  //   - PM4 (MEC):  read is 32-bit in dword units (even on GFX9+).
  //   - SDMA:       read is 64-bit in byte units on GFX9+.
  //
  // We declare read_ptr as uint64_t so the field is large enough for both
  // cases and expect that no queue will exceed 4 GiB in size. The EOP sequence
  // is increased monotonically to coordinate end-of-pipe events for signals.
  //
  // Compute queues also use this to signal scratch requests from the handler.
  static constexpr size_t MAX_SCRATCH_IB_DWORDS = 20;
  struct QueueControl {
    uint64_t read_ptr;
    uint64_t write_ptr;
    uint64_t eop_seq;
    uint64_t err_payload;
    uint64_t scratch_ready;
    uint32_t scratch_done;
    uint32_t indirect[MAX_SCRATCH_IB_DWORDS];
  };

  static std::expected<QueueBase, Error> create(Device &dev, QueueType type,
                                                size_t ring_size);

  QueueBase() = default;
  QueueBase(QueueType type, Context &ctx, Device &dev, uint32_t id,
            Buffer control, Buffer ring, Buffer eop, detail::MappedRegion cwsr,
            Buffer cwsr_bo, volatile uint64_t *doorbell,
            detail::Box<Event> err_event,
            detail::Box<detail::Mutex> submit_mtx);

  std::expected<void, Error> submit(const uint32_t *data, size_t dwords);
  std::expected<void, Error> submit_impl(const uint32_t *data, size_t dwords);

  QueueControl *ctl() const {
    return static_cast<QueueControl *>(control.data());
  }
  std::expected<void, Error> wait_for_room(uint32_t dwords);
  static void queue_error_handler(Event &event, void *user_data);
  static void scratch_handler(Event &event, void *user_data);

  QueueType type{};
  Context *ctx = nullptr;
  Device *dev = nullptr;
  uint32_t id = 0;

  Buffer control;
  Buffer ring;
  Buffer eop;
  detail::MappedRegion cwsr;
  Buffer cwsr_bo;

  volatile uint64_t *doorbell = nullptr;
  detail::Box<Event> err_event;
  detail::Box<QueueErrorCtx> err_watch_ctx;
  detail::Box<Event> scratch_event;
  detail::Box<ScratchCtx> scratch_watch_ctx;
  detail::Box<detail::Mutex> submit_mtx;

  uint64_t pending_wptr = 0;
};

class ComputeQueue {
public:
  static std::expected<ComputeQueue, Error>
  create(Device &dev, size_t ring_size = 4ul * detail::page_size());

  ~ComputeQueue() = default;

  ComputeQueue(const ComputeQueue &) = delete;
  ComputeQueue &operator=(const ComputeQueue &) = delete;
  ComputeQueue(ComputeQueue &&) = default;
  ComputeQueue &operator=(ComputeQueue &&) = default;

  std::expected<void, Error> write_data(void *addr, uint32_t value) {
    uint32_t buf[pm4::WRITE_DATA_DWORDS];
    pm4::write_data(buf, addr, value);
    return base.submit(buf, pm4::WRITE_DATA_DWORDS);
  }

  std::expected<void, Error> dma_copy(void *dst, const void *src,
                                      uint32_t byte_count) {
    const uint32_t max = base.dev->gfx_version() >= abi::GFX_VERSION_GFX10_1
                             ? pm4::DMA_DATA_MAX_BYTES_GFX10
                             : pm4::DMA_DATA_MAX_BYTES_GFX9;
    auto *d = static_cast<std::byte *>(dst);
    auto *s = static_cast<const std::byte *>(src);
    uint32_t remaining = byte_count;
    while (remaining) {
      uint32_t chunk = remaining < max ? remaining : max;
      uint32_t buf[pm4::DMA_DATA_DWORDS];
      pm4::dma_data_copy(buf, d, s, chunk);
      KFD_CHECK(base.submit(buf, pm4::DMA_DATA_DWORDS));
      d += chunk;
      s += chunk;
      remaining -= chunk;
    }
    return {};
  }

  std::expected<void, Error> dma_fill(void *dst, uint32_t value,
                                      uint32_t byte_count) {
    const uint32_t max = base.dev->gfx_version() >= abi::GFX_VERSION_GFX10_1
                             ? pm4::DMA_DATA_MAX_BYTES_GFX10
                             : pm4::DMA_DATA_MAX_BYTES_GFX9;
    auto *d = static_cast<std::byte *>(dst);
    uint32_t remaining = byte_count;
    while (remaining) {
      uint32_t chunk = remaining < max ? remaining : max;
      uint32_t buf[pm4::DMA_DATA_DWORDS];
      pm4::dma_data_fill(buf, d, value, chunk);
      KFD_CHECK(base.submit(buf, pm4::DMA_DATA_DWORDS));
      d += chunk;
      remaining -= chunk;
    }
    return {};
  }

  // Decrements the signal's value and fires an event once the preceding work
  // has finished.
  std::expected<void, Error> signal(Signal &sig) {
    uint32_t buf[SIGNAL_DWORDS];
    // We use a RELEASE_MEM packet with the sequence counter and cache-bypass to
    // wait until the necessary work and cache invalidation has completed. Then
    // we decrement the signal value and fire an event.
    uint32_t seq = __atomic_fetch_add(&next_eop_seq, 1, __ATOMIC_RELAXED) + 1;
    uint32_t n = 0;
    n += pm4::release_mem(buf + n, base.dev->gfx_version(), eop_seq.data(),
                          static_cast<uint64_t>(seq));
    n += pm4::wait_reg_mem(buf + n, base.dev->gfx_version(), eop_seq.data(),
                           Condition::GTE, seq);
    n += pm4::atomic_mem(buf + n, pm4::ATOMIC_ADD_RTN_64, sig.fence_addr(),
                         int64_t(-1), 0, pm4::ATOMIC_WAIT_CONFIRM,
                         pm4::POLICY_BYPASS);
    n += pm4::release_mem(buf + n, base.dev->gfx_version(), sig.signal_addr(),
                          sig.event_id(), sig.trigger_data());
    return base.submit(buf, n);
  }

  // Stalls the CP until the signal's value satisfies the condition.
  std::expected<void, Error> wait(Signal &sig, Condition cond, uint32_t value) {
    uint32_t buf[pm4::WAIT_REG_MEM_DWORDS];
    pm4::wait_reg_mem(buf, base.dev->gfx_version(), sig.fence_addr(), cond,
                      value);
    return base.submit(buf, pm4::WAIT_REG_MEM_DWORDS);
  }

  std::expected<void, Error> release_mem(uint32_t ordinal2, pm4::DataSel data,
                                         pm4::IntSel intr, void *addr = nullptr,
                                         uint64_t value = 0,
                                         uint32_t int_ctxid = 0) {
    uint32_t buf[pm4::RELEASE_MEM_DWORDS];
    pm4::release_mem(buf, ordinal2, data, intr, addr, value, int_ctxid);
    return base.submit(buf, pm4::RELEASE_MEM_DWORDS);
  }

  std::expected<void, Error> acquire_mem() {
    uint32_t buf[pm4::ACQUIRE_MEM_DWORDS];
    auto n = pm4::acquire_mem(buf, base.dev->gfx_version());
    return base.submit(buf, n);
  }

  std::expected<void, Error> wait_reg_mem(void *addr, Condition cond,
                                          uint32_t reference,
                                          uint32_t mask = 0xFFFFFFFF) {
    uint32_t buf[pm4::WAIT_REG_MEM_DWORDS];
    pm4::wait_reg_mem(buf, base.dev->gfx_version(), addr, cond, reference,
                      mask);
    return base.submit(buf, pm4::WAIT_REG_MEM_DWORDS);
  }

  std::expected<void, Error> indirect_buffer(const void *ib_addr,
                                             uint32_t ib_size_dwords) {
    uint32_t buf[pm4::INDIRECT_BUFFER_DWORDS];
    pm4::indirect_buffer(buf, ib_addr, ib_size_dwords);
    return base.submit(buf, pm4::INDIRECT_BUFFER_DWORDS);
  }

  std::expected<void, Error>
  atomic_mem(pm4::AtomicOp op, void *addr, int64_t src_data,
             int64_t cmp_data = 0,
             pm4::AtomicCommand cmd = pm4::ATOMIC_SINGLE_PASS) {
    uint32_t buf[pm4::ATOMIC_MEM_DWORDS];
    pm4::atomic_mem(buf, op, addr, src_data, cmp_data, cmd);
    return base.submit(buf, pm4::ATOMIC_MEM_DWORDS);
  }

  // Dispatches a kernel launch onto the queue.
  std::expected<void, Error> dispatch(const Kernel &kernel,
                                      const DispatchConfig &cfg,
                                      const Buffer &kernarg);

  // Variant that submits a completion signal.
  std::expected<void, Error> dispatch(const Kernel &kernel,
                                      const DispatchConfig &cfg,
                                      const Buffer &kernarg,
                                      Signal &completion) {
    KFD_CHECK(dispatch(kernel, cfg, kernarg));
    return signal(completion);
  }

  uint32_t gfx_version() const { return base.dev->gfx_version(); }
  uint32_t queue_id() const { return base.queue_id(); }
  size_t ring_dwords() const { return base.ring_dwords(); }
  explicit operator bool() const { return static_cast<bool>(base); }

private:
  friend class Context;

  static constexpr uint32_t SIGNAL_DWORDS =
      pm4::RELEASE_MEM_DWORDS + pm4::WAIT_REG_MEM_DWORDS +
      pm4::ATOMIC_MEM_DWORDS + pm4::RELEASE_MEM_DWORDS;

  explicit ComputeQueue(QueueBase &&b, Buffer &&vram)
      : eop_seq(std::move(vram)), base(std::move(b)) {}

  Buffer eop_seq;
  QueueBase base;
  uint32_t next_eop_seq = 0;
};

class SDMAQueue {
public:
  static std::expected<SDMAQueue, Error>
  create(Device &dev, QueueType type = QueueType::SDMA,
         size_t ring_size = detail::page_size());

  ~SDMAQueue() = default;

  SDMAQueue(const SDMAQueue &) = delete;
  SDMAQueue &operator=(const SDMAQueue &) = delete;
  SDMAQueue(SDMAQueue &&) = default;
  SDMAQueue &operator=(SDMAQueue &&) = default;

  // Host memory must be in pinned or GTT memory to qualify for DMA transfers.
  std::expected<void, Error> copy_linear(void *dst, const void *src,
                                         size_t bytes) {
    const uint32_t max_bytes =
        sdma::max_copy_linear_bytes(base.dev->gfx_version());
    auto *d = static_cast<char *>(dst);
    auto *s = static_cast<const char *>(src);
    while (bytes) {
      uint32_t chunk =
          static_cast<uint32_t>(bytes < max_bytes ? bytes : max_bytes);
      uint32_t buf[sdma::COPY_LINEAR_DWORDS];
      sdma::copy_linear(buf, d, s, chunk);
      KFD_CHECK(base.submit(buf, sdma::COPY_LINEAR_DWORDS));
      d += chunk;
      s += chunk;
      bytes -= chunk;
    }
    return {};
  }

  std::expected<void, Error> const_fill(void *dst, uint32_t value,
                                        uint32_t bytes) {
    uint32_t buf[sdma::CONST_FILL_DWORDS];
    sdma::const_fill(buf, base.dev->gfx_version(), dst, value, bytes);
    return base.submit(buf, sdma::CONST_FILL_DWORDS);
  }

  std::expected<void, Error> fence(void *addr, uint32_t value) {
    uint32_t buf[sdma::FENCE_DWORDS];
    sdma::fence(buf, base.dev->gfx_version(), addr, value);
    return base.submit(buf, sdma::FENCE_DWORDS);
  }

  std::expected<void, Error>
  flush_caches(uint32_t gcr_cntl = sdma::GCR_FLUSH_ALL) {
    if (base.dev->gfx_version() < abi::GFX_VERSION_GFX10_1)
      return {};
    uint32_t buf[sdma::GCR_REQ_DWORDS];
    sdma::gcr_req(buf, gcr_cntl);
    return base.submit(buf, sdma::GCR_REQ_DWORDS);
  }

  std::expected<void, Error> poll_regmem(void *addr, Condition cond,
                                         uint32_t reference,
                                         uint32_t mask = 0xFFFFFFFF) {
    uint32_t buf[sdma::POLL_REGMEM_DWORDS];
    sdma::poll_regmem(buf, addr, cond, reference, mask);
    return base.submit(buf, sdma::POLL_REGMEM_DWORDS);
  }

  std::expected<void, Error> timestamp(void *addr) {
    uint32_t buf[sdma::TIMESTAMP_DWORDS];
    sdma::timestamp(buf, addr);
    return base.submit(buf, sdma::TIMESTAMP_DWORDS);
  }

  std::expected<void, Error> atomic_mem(sdma::AtomicOp op, void *addr,
                                        int64_t src_data,
                                        int64_t cmp_data = 0) {
    uint32_t buf[sdma::ATOMIC_DWORDS];
    sdma::atomic_mem(buf, op, addr, src_data, cmp_data);
    return base.submit(buf, sdma::ATOMIC_DWORDS);
  }

  // Decrements the signal's value and fires an event once the preceding work
  // has finished. Flush the caches to ensure a coherent state at completion.
  std::expected<void, Error> signal(Signal &sig) {
    uint32_t buf[sdma::GCR_REQ_DWORDS + sdma::ATOMIC_DWORDS +
                 sdma::FENCE_DWORDS + sdma::TRAP_DWORDS];
    uint32_t n = 0;
    if (base.dev->gfx_version() >= abi::GFX_VERSION_GFX10_1)
      n += sdma::gcr_req(buf + n);
    n += sdma::atomic_mem(buf + n, sdma::ATOMIC_ADD_64, sig.fence_addr(), -1);
    n += sdma::fence(buf + n, base.dev->gfx_version(), sig.signal_addr(),
                     sig.event_id());
    n += sdma::trap(buf + n, sig.trigger_data());
    return base.submit(buf, n);
  }

  // Stalls the CP until the signal's value satisfies the condition.
  std::expected<void, Error> wait(Signal &sig, Condition cond, uint32_t value) {
    uint32_t buf[sdma::POLL_REGMEM_DWORDS];
    sdma::poll_regmem(buf, sig.fence_addr(), cond, value);
    return base.submit(buf, sdma::POLL_REGMEM_DWORDS);
  }

  bool is_xgmi() const { return type == QueueType::SDMA_XGMI; }

  uint32_t queue_id() const { return base.queue_id(); }
  size_t ring_dwords() const { return base.ring_dwords(); }
  explicit operator bool() const { return static_cast<bool>(base); }

private:
  explicit SDMAQueue(QueueBase &&b, QueueType type)
      : base(std::move(b)), type(type) {}
  QueueBase base;
  QueueType type;
};

} // namespace kfd

#endif // LIBKFD_QUEUE_H
