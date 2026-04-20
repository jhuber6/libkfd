//===-- lib/context.cpp - KFD context implementation ------------*- C++ -*-===//
//
// Implementation of the Context class. Opens /dev/kfd and manages the interface
// into the kernel.
//
//===----------------------------------------------------------------------===//

#include "libkfd/context.h"
#include "ioctl.h"
#include "libkfd/event.h"
#include "libkfd/memory.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <linux/futex.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

using namespace kfd::detail;

namespace kfd {

namespace {

struct SignalPage {
  void *addr;
  uint64_t handle;
  size_t alloc_size;
};

// Obtain backing memory for one of the pages used to signal work from KFD.
std::expected<SignalPage, Error> alloc_signal_page(SmallVector<Device, 4> &devs,
                                                   size_t size) {
  if (devs.empty())
    return SignalPage{nullptr, 0, 0};

  constexpr MemFlags FLAGS = MemFlags::WRITABLE | MemFlags::EXECUTABLE |
                             MemFlags::HOST_ACCESS | MemFlags::COHERENT |
                             MemFlags::UNCACHED;

  auto bo = KFD_TRY(Buffer::allocate(devs.front(), size, MemType::GTT, FLAGS));

  SmallVector<Device *, 4> ptrs;
  for (auto &d : devs)
    KFD_CHECK(ptrs.push_back(&d));
  KFD_CHECK(bo.map(std::span<Device *const>(ptrs.data(), ptrs.size())));

  void *page = bo.data();
  size_t alloc_size = bo.size();
  uint64_t kfd_handle = bo.release();
  std::memset(page, 0, alloc_size);

  return SignalPage{page, kfd_handle, alloc_size};
}

// Backing memory for the signal system, intentionally leaked once per-process.
uint64_t *event_page = nullptr;
uint64_t *fence_page = nullptr;
Mutex signal_page_mtx{};

constexpr size_t WATCHER_STACK_SIZE = 64ull * 1024;

struct QueueErrorEntry {
  uint32_t event_id;
  volatile uint64_t *err_payload;
  uint32_t queue_id;
  uint32_t gpu_id;
};

struct FaultWatcher {
  FaultWatcher(Event mem, Event hw, Event wake, void *stack)
      : mem_event(std::move(mem)), hw_event(std::move(hw)),
        wake_event(std::move(wake)), stack(stack) {}

  int kfd_fd() const { return mem_event.kfd_fd(); }

  Event mem_event;
  Event hw_event;
  Event wake_event;
  uint32_t exit_flag = 0;
  pid_t tid = 0;
  void *stack;
  Mutex queue_mtx;
  SmallVector<QueueErrorEntry, 8> queue_errors;
};

// Exception bitmask flags written to err_payload_addr by KFD.
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

int fault_watcher_entry(void *arg) {
  auto *w = static_cast<FaultWatcher *>(arg);

  SmallVector<ioctl::kfd::event_data, 16> eds;

  while (!__atomic_load_n(&w->exit_flag, __ATOMIC_ACQUIRE)) {
    // Rebuild the event list each iteration: system events first, then
    // per-queue error events, then the wake event last.
    eds.clear();

    ioctl::kfd::event_data mem_ed{};
    mem_ed.event_id = w->mem_event.event_id();
    if (!eds.push_back(mem_ed))
      continue;

    ioctl::kfd::event_data hw_ed{};
    hw_ed.event_id = w->hw_event.event_id();
    if (!eds.push_back(hw_ed))
      continue;

    {
      LockGuard guard(w->queue_mtx);
      for (auto &qe : w->queue_errors) {
        ioctl::kfd::event_data qed{};
        qed.event_id = qe.event_id;
        if (!eds.push_back(qed))
          break;
      }
    }

    ioctl::kfd::event_data wake_ed{};
    wake_ed.event_id = w->wake_event.event_id();
    if (!eds.push_back(wake_ed))
      continue;

    // Multi-event wait-any with event_data inspection is not covered by
    // Event::wait(), so the raw ioctl is used here.
    ioctl::kfd::wait_events_args wait{
        .events_ptr = reinterpret_cast<uintptr_t>(eds.data()),
        .num_events = static_cast<uint32_t>(eds.size()),
        .wait_for_all = 0,
        .timeout = UINT32_MAX,
    };

    auto r = ioctl::call<ioctl::kfd::WAIT_EVENTS>(w->kfd_fd(), wait);
    if (!r || wait.wait_result != KFD_IOC_WAIT_RESULT_COMPLETE)
      continue;

    if (__atomic_load_n(&w->exit_flag, __ATOMIC_ACQUIRE))
      break;

    if (eds[0].memory_exception_data.gpu_id) {
      auto &d = eds[0].memory_exception_data;
      std::fprintf(stderr,
                   "GPU memory fault at VA 0x%lx (gpu_id %u, error %u):",
                   static_cast<unsigned long>(d.va), d.gpu_id, d.ErrorType);
      if (d.failure.NotPresent)
        std::fprintf(stderr, " page-not-present");
      if (d.failure.ReadOnly)
        std::fprintf(stderr, " read-only");
      if (d.failure.NoExecute)
        std::fprintf(stderr, " no-execute");
      if (d.failure.imprecise)
        std::fprintf(stderr, " imprecise");
      std::fprintf(stderr, "\n");
      std::memset(&eds[0].memory_exception_data, 0,
                  sizeof(eds[0].memory_exception_data));
      ::raise(SIGSEGV);
    }

    if (eds[1].hw_exception_data.gpu_id) {
      auto &d = eds[1].hw_exception_data;
      std::fprintf(stderr,
                   "GPU HW exception (gpu_id %u): reset_type=%u "
                   "reset_cause=%u memory_lost=%u\n",
                   d.gpu_id, d.reset_type, d.reset_cause, d.memory_lost);
      std::memset(&eds[1].hw_exception_data, 0,
                  sizeof(eds[1].hw_exception_data));
      ::raise(SIGABRT);
    }

    LockGuard guard(w->queue_mtx);
    for (auto &qe : w->queue_errors) {
      uint64_t err = __atomic_load_n(qe.err_payload, __ATOMIC_ACQUIRE);
      if (err) {
        report_queue_exception(qe.queue_id, qe.gpu_id, err);
        __atomic_store_n(qe.err_payload, 0, __ATOMIC_RELEASE);
        ::raise(SIGABRT);
      }
    }
  }

  return 0;
}

// Start the dedicated fault watcher background thread. This will sleep until an
// abnormal event from the process or a queue wakes it and handles it.
std::expected<FaultWatcher *, Error> start_fault_watcher(Context &ctx) {
  auto mem_ev = KFD_TRY(Event::create(ctx, KFD_IOC_EVENT_MEMORY));
  auto hw_ev = KFD_TRY(Event::create(ctx, KFD_IOC_EVENT_HW_EXCEPTION));
  auto wake_ev = KFD_TRY(Event::create(ctx, KFD_IOC_EVENT_SIGNAL));

  void *stack = ::mmap(nullptr, WATCHER_STACK_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
  if (stack == MAP_FAILED)
    return kfd::unexpected(errno, "failed to mmap fault watcher stack");

  void *mem = std::malloc(sizeof(FaultWatcher));
  if (!mem) {
    ::munmap(stack, WATCHER_STACK_SIZE);
    return kfd::unexpected(ENOMEM, "failed to allocate FaultWatcher");
  }

  auto *watcher = new (mem) FaultWatcher(std::move(mem_ev), std::move(hw_ev),
                                         std::move(wake_ev), stack);

  // Fork to an independent thread so the KFD process is shared.
  void *stack_top = static_cast<char *>(stack) + WATCHER_STACK_SIZE;
  pid_t child = ::clone(fault_watcher_entry, stack_top,
                        CLONE_VM | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD |
                            CLONE_SYSVSEM | CLONE_CHILD_CLEARTID,
                        watcher,
                        /*parent_tid=*/nullptr,
                        /*tls=*/nullptr,
                        /*child_tid=*/&watcher->tid);
  if (child == -1) {
    int err = errno;
    ::munmap(stack, WATCHER_STACK_SIZE);
    watcher->~FaultWatcher();
    std::free(watcher);
    return kfd::unexpected(err, "failed to clone fault watcher thread");
  }

  watcher->tid = child;
  return watcher;
}

// Stop the fault watcher thread at context destruction by signalling its event.
void stop_fault_watcher(void *ptr) {
  auto *watcher = static_cast<FaultWatcher *>(ptr);
  if (!watcher)
    return;

  __atomic_store_n(&watcher->exit_flag, 1, __ATOMIC_RELEASE);
  (void)watcher->wake_event.signal();

  while (__atomic_load_n(&watcher->tid, __ATOMIC_ACQUIRE) != 0)
    ::syscall(SYS_futex, &watcher->tid, FUTEX_WAIT, watcher->tid, nullptr,
              nullptr, 0);

  void *stack = watcher->stack;
  watcher->~FaultWatcher();
  std::free(watcher);
  ::munmap(stack, WATCHER_STACK_SIZE);
}

void add_queue_error(void *ptr, uint32_t event_id, volatile uint64_t *payload,
                     uint32_t queue_id, uint32_t gpu_id) {
  auto *watcher = static_cast<FaultWatcher *>(ptr);
  if (!watcher)
    return;
  LockGuard guard(watcher->queue_mtx);
  (void)watcher->queue_errors.push_back({event_id, payload, queue_id, gpu_id});
  (void)watcher->wake_event.signal();
}

void remove_queue_error(void *ptr, uint32_t event_id) {
  auto *watcher = static_cast<FaultWatcher *>(ptr);
  if (!watcher)
    return;
  LockGuard guard(watcher->queue_mtx);
  for (size_t i = 0; i < watcher->queue_errors.size(); ++i) {
    if (watcher->queue_errors[i].event_id == event_id) {
      watcher->queue_errors[i] =
          watcher->queue_errors[watcher->queue_errors.size() - 1];
      watcher->queue_errors.pop_back();
      break;
    }
  }
  (void)watcher->wake_event.signal();
}

void free_signal_page(int kfd_fd, SmallVector<Device, 4> &devs,
                      SignalPage &page) {
  if (!page.addr)
    return;
  SmallVector<uint32_t, 4> ids;
  for (auto &d : devs)
    (void)ids.push_back(d.gpu_id());
  ioctl::kfd::unmap_memory_from_gpu_args uargs{
      .handle = page.handle,
      .device_ids_array_ptr = reinterpret_cast<uintptr_t>(ids.data()),
      .n_devices = static_cast<uint32_t>(ids.size()),
  };
  ioctl::call<ioctl::kfd::UNMAP_MEMORY_FROM_GPU>(kfd_fd, uargs);
  ioctl::kfd::free_memory_of_gpu_args fargs{.handle = page.handle};
  ioctl::call<ioctl::kfd::FREE_MEMORY_OF_GPU>(kfd_fd, fargs);
  ::munmap(page.addr, page.alloc_size);
  page.addr = nullptr;
  page.handle = 0;
}

} // namespace

constexpr const char KFD_PATH[] = "/dev/kfd";

std::expected<Context, Error> Context::create() {
  int fd = ::open(KFD_PATH, O_RDWR | O_CLOEXEC);
  if (fd < 0) {
    int err = errno;
    return kfd::unexpected(err, "failed to open '%s'", KFD_PATH);
  }

  ioctl::kfd::version_args ver{};
  if (auto r = ioctl::call<ioctl::kfd::GET_VERSION>(fd, ver); !r) {
    ::close(fd);
    return kfd::unexpected(r.error());
  }
  if (ver.major_version < MIN_KFD_MAJOR ||
      (ver.major_version == MIN_KFD_MAJOR &&
       ver.minor_version < MIN_KFD_MINOR)) {
    ::close(fd);
    return kfd::unexpected(ENOTSUP, "KFD version %u.%u < required %u.%u",
                           ver.major_version, ver.minor_version, MIN_KFD_MAJOR,
                           MIN_KFD_MINOR);
  }

  auto topo = Topology::create();
  if (!topo) {
    ::close(fd);
    return kfd::unexpected(topo.error());
  }

  // Query the process-wide X-NACK mode from the kernel. This allows automatic
  // migration of memory on page faults originating from the device's MMU.
  bool xnack_mode = false;
  ioctl::kfd::set_xnack_mode_args xnack_args{.xnack_enabled = -1};
  if (auto r = ioctl::call<ioctl::kfd::SET_XNACK_MODE>(fd, xnack_args); r)
    xnack_mode = xnack_args.xnack_enabled > 0;

  Context ctx(fd, xnack_mode, {});
  for (auto &node : topo->nodes()) {
    auto dev = KFD_TRY(Device::create(ctx, std::move(node)));
    KFD_CHECK(ctx.nodes.push_back(std::move(dev)));
  }

  // The event page is sticky once initialized so we do this once per process.
  if (!__atomic_load_n(&event_page, __ATOMIC_ACQUIRE)) {
    LockGuard guard(signal_page_mtx);
    if (!__atomic_load_n(&event_page, __ATOMIC_RELAXED)) {
      constexpr size_t EVENT_PAGE_SIZE =
          KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t);
      constexpr size_t FENCE_PAGE_SIZE =
          KFD_SIGNAL_EVENT_LIMIT * sizeof(uint64_t);

      auto ep = KFD_TRY(alloc_signal_page(ctx.nodes, EVENT_PAGE_SIZE));

      // Register the event page with the kernel via a throwaway event.
      ioctl::kfd::create_event_args eargs{
          .event_page_offset = ep.handle,
          .event_type = KFD_IOC_EVENT_SIGNAL,
          .auto_reset = 1,
      };
      if (auto r = ioctl::call<ioctl::kfd::CREATE_EVENT>(ctx.kfd_fd(), eargs);
          !r) {
        free_signal_page(ctx.kfd_fd(), ctx.nodes, ep);
        return kfd::unexpected(r.error());
      }
      ioctl::kfd::destroy_event_args dargs{.event_id = eargs.event_id};
      ioctl::call<ioctl::kfd::DESTROY_EVENT>(ctx.kfd_fd(), dargs);

      auto fp = alloc_signal_page(ctx.nodes, FENCE_PAGE_SIZE);
      if (!fp) {
        free_signal_page(ctx.kfd_fd(), ctx.nodes, ep);
        return kfd::unexpected(fp.error());
      }

      // Indices to these are handled by the KFD event interface.
      __atomic_store_n(&fence_page, static_cast<uint64_t *>(fp->addr),
                       __ATOMIC_RELEASE);
      __atomic_store_n(&event_page, static_cast<uint64_t *>(ep.addr),
                       __ATOMIC_RELEASE);
    }
  }

  // Enable KFD's runtime exception delivery before any queues are created.
  // This activates the mechanism by which the trap handler's MSG_INTERRUPT
  // writes to err_payload_addr and signals err_event_id in the CWSR header.
  ioctl::kfd::runtime_enable_args re{};
  re.r_debug = 0;
  re.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK |
                 KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK;
  KFD_CHECK(ioctl::call<ioctl::kfd::RUNTIME_ENABLE>(ctx.kfd_fd(), re));

  ctx.fault_watcher = KFD_TRY(start_fault_watcher(ctx));

  return ctx;
}

std::expected<Device *, Error> Context::device(size_t i) {
  if (i >= nodes.size())
    return kfd::unexpected(EINVAL, "device index %zu >= device count %zu", i,
                           nodes.size());

  return &nodes[i];
}

Context::~Context() {
  stop_fault_watcher(fault_watcher);
  nodes.clear();
  if (fd >= 0)
    ::close(fd);
}

Context::Context(Context &&other)
    : fd(std::exchange(other.fd, -1)), xnack(other.xnack),
      nodes(std::move(other.nodes)),
      fault_watcher(std::exchange(other.fault_watcher, nullptr)) {
  for (auto &dev : nodes)
    dev.ctx = this;
}

std::expected<VersionInfo, Error> Context::version() const {
  ioctl::kfd::version_args args{};
  KFD_CHECK(ioctl::call<ioctl::kfd::GET_VERSION>(fd, args));
  return VersionInfo{.major = args.major_version, .minor = args.minor_version};
}

uint64_t *Context::event_slot(uint32_t id) {
  return &__atomic_load_n(&event_page, __ATOMIC_ACQUIRE)[id];
}

uint64_t *Context::fence_slot(uint32_t id) {
  return &__atomic_load_n(&fence_page, __ATOMIC_ACQUIRE)[id];
}

void Context::register_queue_error(uint32_t event_id,
                                   volatile uint64_t *payload,
                                   uint32_t queue_id, uint32_t gpu_id) {
  add_queue_error(fault_watcher, event_id, payload, queue_id, gpu_id);
}

void Context::unregister_queue_error(uint32_t event_id) {
  remove_queue_error(fault_watcher, event_id);
}

} // namespace kfd
