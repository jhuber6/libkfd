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

std::expected<void, Error> checked_munmap(void *addr, size_t len) {
  if (::munmap(addr, len) != 0)
    return kfd::unexpected(errno, "munmap failed: %p, %zu", addr, len);
  return {};
}

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

// KFD runtime exception delivery, enabled once per-process.
uint32_t runtime_enabled = 0;
Mutex runtime_mtx{};

} // namespace

constexpr size_t WATCHER_STACK_SIZE = 64ull * 1024;

struct FaultWatcher {
  FaultWatcher(Event mem, Event hw, Event wake, void *stack)
      : mem_event(std::move(mem)), hw_event(std::move(hw)),
        wake_event(std::move(wake)), stack(stack) {}

  ~FaultWatcher() {
    if (tid != 0) {
      __atomic_store_n(&exit_flag, 1, __ATOMIC_RELEASE);
      KFD_ASSERT(wake_event.signal());
      for (pid_t val; (val = __atomic_load_n(&tid, __ATOMIC_ACQUIRE)) != 0;)
        ::syscall(SYS_futex, &tid, FUTEX_WAIT, val, nullptr, nullptr, 0);
    }
    if (stack)
      KFD_ASSERT(checked_munmap(stack, WATCHER_STACK_SIZE));
  }

  struct WatchEntry {
    Event *event;
    WatchCallback callback;
    void *user_data;
    uint32_t event_id;
  };

  Event mem_event;
  Event hw_event;
  Event wake_event;
  uint32_t exit_flag = 0;
  pid_t tid = 0;
  void *stack;
  Mutex watch_mtx;
  SmallVector<WatchEntry, 8> watches;
};

namespace {

int fault_watcher_entry(void *arg) {
  auto *w = static_cast<FaultWatcher *>(arg);

  // We cannot use the raw Event * interface because the user could move the
  // events and the independent watcher thread would not be updated.
  SmallVector<ioctl::kfd::event_data, 16> eds;
  while (!__atomic_load_n(&w->exit_flag, __ATOMIC_ACQUIRE)) {
    eds.clear();

    ioctl::kfd::event_data mem_ed{};
    mem_ed.event_id = w->mem_event.event_id();
    KFD_ASSERT(eds.push_back(mem_ed));

    ioctl::kfd::event_data hw_ed{};
    hw_ed.event_id = w->hw_event.event_id();
    KFD_ASSERT(eds.push_back(hw_ed));

    {
      LockGuard guard(w->watch_mtx);
      for (auto &we : w->watches) {
        ioctl::kfd::event_data ed{};
        ed.event_id = we.event_id;
        KFD_ASSERT(eds.push_back(ed));
      }
    }

    ioctl::kfd::event_data wake_ed{};
    wake_ed.event_id = w->wake_event.event_id();
    KFD_ASSERT(eds.push_back(wake_ed));

    ioctl::kfd::wait_events_args wait{
        .events_ptr = reinterpret_cast<uintptr_t>(eds.data()),
        .num_events = static_cast<uint32_t>(eds.size()),
        .wait_for_all = 0,
        .timeout = UINT32_MAX,
    };
    auto r = ioctl::call<ioctl::kfd::WAIT_EVENTS>(w->mem_event.kfd_fd(), wait);
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
      ::raise(SIGSEGV);
    }

    if (eds[1].hw_exception_data.gpu_id) {
      auto &d = eds[1].hw_exception_data;
      std::fprintf(stderr,
                   "GPU HW exception (gpu_id %u): reset_type=%u "
                   "reset_cause=%u memory_lost=%u\n",
                   d.gpu_id, d.reset_type, d.reset_cause, d.memory_lost);
      ::raise(SIGABRT);
    }

    LockGuard guard(w->watch_mtx);
    for (auto &we : w->watches)
      we.callback(*we.event, we.user_data);
  }

  return 0;
}

// Start the dedicated fault watcher background thread. This will sleep until an
// abnormal event from the process or a queue wakes it and handles it.
std::expected<Box<FaultWatcher>, Error> start_fault_watcher(Context &ctx) {
  auto mem_ev = KFD_TRY(Event::create(ctx, EventType::MEMORY));
  auto hw_ev = KFD_TRY(Event::create(ctx, EventType::HW_EXCEPTION));
  auto wake_ev = KFD_TRY(Event::create(ctx, EventType::SIGNAL));

  void *stack = ::mmap(nullptr, WATCHER_STACK_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_STACK, -1, 0);
  if (stack == MAP_FAILED)
    return kfd::unexpected(errno, "failed to mmap fault watcher stack");

  auto result = Box<FaultWatcher>::create(std::move(mem_ev), std::move(hw_ev),
                                          std::move(wake_ev), stack);
  if (!result) {
    KFD_ASSERT(checked_munmap(stack, WATCHER_STACK_SIZE));
    return kfd::unexpected(result.error());
  }
  auto watcher = std::move(*result);

  // Fork to an independent thread so the KFD process is shared.
  void *stack_top = static_cast<char *>(stack) + WATCHER_STACK_SIZE;
  pid_t child = ::clone(fault_watcher_entry, stack_top,
                        CLONE_VM | CLONE_FILES | CLONE_SIGHAND | CLONE_THREAD |
                            CLONE_SYSVSEM | CLONE_CHILD_CLEARTID,
                        watcher.get(),
                        /*parent_tid=*/nullptr,
                        /*tls=*/nullptr,
                        /*child_tid=*/&watcher->tid);
  if (child == -1)
    return kfd::unexpected(errno, "failed to clone fault watcher thread");

  watcher->tid = child;
  return watcher;
}

std::expected<void, Error> add_watch(FaultWatcher *watcher, Event *event,
                                     WatchCallback cb, void *user_data) {
  if (!watcher)
    return {};
  LockGuard guard(watcher->watch_mtx);
  KFD_CHECK(
      watcher->watches.push_back({event, cb, user_data, event->event_id()}));
  return watcher->wake_event.signal();
}

std::expected<void, Error> remove_watch(FaultWatcher *watcher, Event *event) {
  if (!watcher)
    return {};
  LockGuard guard(watcher->watch_mtx);
  for (size_t i = 0; i < watcher->watches.size(); ++i) {
    if (watcher->watches[i].event == event) {
      watcher->watches[i] = watcher->watches[watcher->watches.size() - 1];
      watcher->watches.pop_back();
      break;
    }
  }
  return watcher->wake_event.signal();
}

void free_signal_page(int kfd_fd, SmallVector<Device, 4> &devs,
                      SignalPage &page) {
  if (!page.addr)
    return;
  SmallVector<uint32_t, 4> ids;
  for (auto &d : devs)
    KFD_ASSERT(ids.push_back(d.gpu_id()));
  ioctl::kfd::unmap_memory_from_gpu_args uargs{
      .handle = page.handle,
      .device_ids_array_ptr = reinterpret_cast<uintptr_t>(ids.data()),
      .n_devices = static_cast<uint32_t>(ids.size()),
  };
  KFD_ASSERT(ioctl::call<ioctl::kfd::UNMAP_MEMORY_FROM_GPU>(kfd_fd, uargs));
  ioctl::kfd::free_memory_of_gpu_args fargs{.handle = page.handle};
  KFD_ASSERT(ioctl::call<ioctl::kfd::FREE_MEMORY_OF_GPU>(kfd_fd, fargs));
  KFD_ASSERT(checked_munmap(page.addr, page.alloc_size));
  page.addr = nullptr;
  page.handle = 0;
}

} // namespace

constexpr const char KFD_PATH[] = "/dev/kfd";

Context::Context(int fd, bool xnack, SmallVector<Device, 4> devices)
    : fd(fd), xnack(xnack), nodes(std::move(devices)) {}

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
          .event_type = static_cast<uint32_t>(EventType::SIGNAL),
          .auto_reset = 1,
      };
      if (auto r = ioctl::call<ioctl::kfd::CREATE_EVENT>(ctx.kfd_fd(), eargs);
          !r) {
        free_signal_page(ctx.kfd_fd(), ctx.nodes, ep);
        return kfd::unexpected(r.error());
      }
      ioctl::kfd::destroy_event_args dargs{.event_id = eargs.event_id};
      KFD_ASSERT(ioctl::call<ioctl::kfd::DESTROY_EVENT>(ctx.kfd_fd(), dargs));

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
  // This is a per-process operation so we only do it once.
  if (!__atomic_load_n(&runtime_enabled, __ATOMIC_ACQUIRE)) {
    LockGuard guard(runtime_mtx);
    if (!__atomic_load_n(&runtime_enabled, __ATOMIC_RELAXED)) {
      ioctl::kfd::runtime_enable_args re{};
      re.r_debug = 0;
      re.mode_mask = KFD_RUNTIME_ENABLE_MODE_ENABLE_MASK |
                     KFD_RUNTIME_ENABLE_MODE_TTMP_SAVE_MASK;
      KFD_CHECK(ioctl::call<ioctl::kfd::RUNTIME_ENABLE>(ctx.kfd_fd(), re));
      __atomic_store_n(&runtime_enabled, 1, __ATOMIC_RELEASE);
    }
  }

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
  fault_watcher = {};
  nodes.clear();
  if (fd >= 0)
    ::close(fd);
}

Context::Context(Context &&other)
    : fd(std::exchange(other.fd, -1)), xnack(other.xnack),
      nodes(std::move(other.nodes)),
      fault_watcher(std::move(other.fault_watcher)) {
  for (auto &dev : nodes)
    dev.ctx = this;
}

Context &Context::operator=(Context &&other) {
  if (this != &other) {
    fault_watcher = {};
    nodes.clear();
    if (fd >= 0)
      ::close(fd);

    fd = std::exchange(other.fd, -1);
    xnack = other.xnack;
    nodes = std::move(other.nodes);
    fault_watcher = std::move(other.fault_watcher);

    for (auto &dev : nodes)
      dev.ctx = this;
  }
  return *this;
}

std::expected<VersionInfo, Error> Context::version() const {
  ioctl::kfd::version_args args{};
  KFD_CHECK(ioctl::call<ioctl::kfd::GET_VERSION>(fd, args));
  return VersionInfo{.major = args.major_version, .minor = args.minor_version};
}

std::expected<uint64_t *, Error> Context::event_slot(uint32_t id) {
  if (id >= KFD_SIGNAL_EVENT_LIMIT)
    return kfd::unexpected(EINVAL, "event id %u >= limit %u\n", id,
                           static_cast<unsigned>(KFD_SIGNAL_EVENT_LIMIT));
  return &__atomic_load_n(&event_page, __ATOMIC_ACQUIRE)[id];
}

std::expected<uint64_t *, Error> Context::fence_slot(uint32_t id) {
  if (id >= KFD_SIGNAL_EVENT_LIMIT)
    return kfd::unexpected(EINVAL, "fence id %u >= limit %u\n", id,
                           static_cast<unsigned>(KFD_SIGNAL_EVENT_LIMIT));
  return &__atomic_load_n(&fence_page, __ATOMIC_ACQUIRE)[id];
}

std::expected<void, Error> Context::watch_event(Event &event, WatchCallback cb,
                                                void *user_data) {
  return add_watch(fault_watcher.get(), &event, cb, user_data);
}

std::expected<void, Error> Context::unwatch_event(Event &event) {
  return remove_watch(fault_watcher.get(), &event);
}

} // namespace kfd
