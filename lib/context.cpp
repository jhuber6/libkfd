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
#include "libkfd/signal.h"

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
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

// Default fault policy describes the fault on stderr and terminates.
void default_fault_handler(const FaultInfo &f, void *) {
  if (f.kind == FaultInfo::Kind::MemoryViolation) {
    std::fprintf(stderr, "GPU memory fault at VA 0x%lx (gpu_id %u, error %u):",
                 static_cast<unsigned long>(f.memory.va), f.gpu_id,
                 f.memory.error_type);
    if (f.memory.reason & MemoryFaultInfo::NotPresent)
      std::fprintf(stderr, " page-not-present");
    if (f.memory.reason & MemoryFaultInfo::ReadOnly)
      std::fprintf(stderr, " read-only");
    if (f.memory.reason & MemoryFaultInfo::NoExecute)
      std::fprintf(stderr, " no-execute");
    if (f.memory.reason & MemoryFaultInfo::Imprecise)
      std::fprintf(stderr, " imprecise");
    std::fprintf(stderr, "\n");
    ::raise(SIGSEGV);
  } else {
    std::fprintf(stderr,
                 "GPU HW exception (gpu_id %u): reset_type=%u reset_cause=%u "
                 "memory_lost=%u\n",
                 f.gpu_id, f.hardware.reset_type, f.hardware.reset_cause,
                 f.hardware.memory_lost);
    ::raise(SIGABRT);
  }
}

} // namespace

// Shared interface for the runtime watcher threads for signals and exceptions.
struct Watcher {
  explicit Watcher(Event wake) : wake_event(std::move(wake)) {}

  Watcher(const Watcher &) = delete;
  Watcher &operator=(const Watcher &) = delete;

  ~Watcher() {
    if (!started)
      return;
    __atomic_store_n(&exit_flag, 1, __ATOMIC_RELEASE);
    KFD_ASSERT(wake_event.signal());
    ::pthread_join(thread, nullptr);
  }

  bool should_exit() const {
    return __atomic_load_n(&exit_flag, __ATOMIC_ACQUIRE);
  }

  std::expected<void, Error> start(void *(*entry)(void *)) {
    int err = ::pthread_create(&thread, nullptr, entry, this);
    if (err != 0)
      return kfd::unexpected(err, "failed to create watcher thread");
    started = true;
    return {};
  }

  Event wake_event;
  uint32_t exit_flag = 0;
  pthread_t thread = {};
  bool started = false;
  Mutex mtx; // Guards derived-class state.
};

// An asynchronous watcher thread that resolves actions to be run on signal
// copmletion.
struct SignalWatcher : Watcher {
  using Watcher::Watcher;

  struct Entry {
    uint32_t event_id;
    uint32_t *fence;
    Condition cond;
    uint32_t value;
    SignalHandler callback;
    void *user_data;
    const void *key;
    uint64_t token;
    uint64_t age;
  };

  SmallVector<Entry, 8> entries;
  uint64_t next_token = 1;
};

// An asynchronous watcher thread that triggers on exception events.
struct ExceptionWatcher : Watcher {
  ExceptionWatcher(Event mem, Event hw, Event wake)
      : Watcher(std::move(wake)), mem_event(std::move(mem)),
        hw_event(std::move(hw)) {}

  Event mem_event;
  Event hw_event;
  FaultHandler fault_handler = default_fault_handler;
  void *fault_user_data = nullptr;
};

namespace {

bool condition_met(Condition cond, uint32_t cur, uint32_t target) {
  switch (cond) {
  case Condition::LT:
    return cur < target;
  case Condition::LTE:
    return cur <= target;
  case Condition::EQ:
    return cur == target;
  case Condition::NE:
    return cur != target;
  case Condition::GTE:
    return cur >= target;
  case Condition::GT:
    return cur > target;
  }
  __builtin_unreachable();
}

FaultInfo make_memory_fault(const ioctl::kfd::event_data &e) {
  const auto &d = e.memory_exception_data;
  uint32_t reason = 0;
  if (d.failure.NotPresent)
    reason |= MemoryFaultInfo::NotPresent;
  if (d.failure.ReadOnly)
    reason |= MemoryFaultInfo::ReadOnly;
  if (d.failure.NoExecute)
    reason |= MemoryFaultInfo::NoExecute;
  if (d.failure.imprecise)
    reason |= MemoryFaultInfo::Imprecise;
  FaultInfo info{.kind = FaultInfo::Kind::MemoryViolation, .gpu_id = d.gpu_id};
  info.memory = {.va = d.va, .reason = reason, .error_type = d.ErrorType};
  return info;
}

FaultInfo make_hw_exception(const ioctl::kfd::event_data &e) {
  const auto &d = e.hw_exception_data;
  FaultInfo info{.kind = FaultInfo::Kind::HardwareException,
                 .gpu_id = d.gpu_id};
  info.hardware = {.reset_type = d.reset_type,
                   .reset_cause = d.reset_cause,
                   .memory_lost = d.memory_lost};
  return info;
}

void dispatch_faults(ExceptionWatcher *w, const ioctl::kfd::event_data &mem,
                     const ioctl::kfd::event_data &hw) {
  if (!mem.memory_exception_data.gpu_id && !hw.hw_exception_data.gpu_id)
    return;
  FaultHandler handler;
  void *user_data;
  {
    LockGuard guard(w->mtx);
    handler = w->fault_handler;
    user_data = w->fault_user_data;
  }
  if (mem.memory_exception_data.gpu_id)
    handler(make_memory_fault(mem), user_data);
  if (hw.hw_exception_data.gpu_id)
    handler(make_hw_exception(hw), user_data);
}

void *signal_watcher_entry(void *arg) {
  auto *w = static_cast<SignalWatcher *>(arg);

  // We cannot use the raw Event * interface because the user could move the
  // events and the independent watcher thread would not be updated. We need to
  // track the age of the event to avoid re-triggering on stale events.
  SmallVector<ioctl::kfd::event_data, 16> eds;
  SmallVector<uint64_t, 16> tokens;
  while (!w->should_exit()) {
    eds.clear();
    tokens.clear();

    ioctl::kfd::event_data wake_ed{};
    wake_ed.event_id = w->wake_event.event_id();
    KFD_ASSERT(eds.push_back(wake_ed));
    KFD_ASSERT(tokens.push_back(0));

    {
      LockGuard guard(w->mtx);
      for (auto &e : w->entries) {
        ioctl::kfd::event_data ed{};
        ed.event_id = e.event_id;
        ed.signal_event_data.last_event_age = e.age;
        KFD_ASSERT(eds.push_back(ed));
        KFD_ASSERT(tokens.push_back(e.token));
      }
    }

    ioctl::kfd::wait_events_args wait{
        .events_ptr = reinterpret_cast<uintptr_t>(eds.data()),
        .num_events = static_cast<uint32_t>(eds.size()),
        .wait_for_all = 0,
        .timeout = UINT32_MAX,
    };
    auto r = ioctl::call<ioctl::kfd::WAIT_EVENTS>(w->wake_event.kfd_fd(), wait);
    if (!r || wait.wait_result != KFD_IOC_WAIT_RESULT_COMPLETE)
      continue;

    if (w->should_exit())
      break;

    // Dispatch under the lock so unregister_handler cannot remove an entry and
    // free its user_data mid-callback. Callbacks must not re-enter the watcher;
    // they return true to stay armed or false to be removed. An entry fires
    // only when KFD reports a new age for it, so a wake for one event never
    // re-fires another and a persistent fence level never re-fires without a
    // new trigger.
    LockGuard guard(w->mtx);
    for (size_t k = 1; k < eds.size(); ++k) {
      uint64_t new_age = eds[k].signal_event_data.last_event_age;
      for (size_t i = 0; i < w->entries.size(); ++i) {
        auto &e = w->entries[i];
        if (e.token != tokens[k])
          continue;
        if (new_age != e.age) {
          e.age = new_age;
          bool met =
              e.fence == nullptr ||
              condition_met(e.cond, __atomic_load_n(e.fence, __ATOMIC_ACQUIRE),
                            e.value);
          if (met && !e.callback(e.user_data)) {
            w->entries[i] = w->entries[w->entries.size() - 1];
            w->entries.pop_back();
          }
        }
        break;
      }
    }
  }

  return nullptr;
}

void *exception_watcher_entry(void *arg) {
  auto *w = static_cast<ExceptionWatcher *>(arg);

  ioctl::kfd::event_data eds[3];
  while (!w->should_exit()) {
    eds[0] = {};
    eds[0].event_id = w->wake_event.event_id();
    eds[1] = {};
    eds[1].event_id = w->mem_event.event_id();
    eds[2] = {};
    eds[2].event_id = w->hw_event.event_id();

    ioctl::kfd::wait_events_args wait{
        .events_ptr = reinterpret_cast<uintptr_t>(eds),
        .num_events = 3,
        .wait_for_all = 0,
        .timeout = UINT32_MAX,
    };
    auto r = ioctl::call<ioctl::kfd::WAIT_EVENTS>(w->wake_event.kfd_fd(), wait);
    if (!r || wait.wait_result != KFD_IOC_WAIT_RESULT_COMPLETE)
      continue;

    if (w->should_exit())
      break;

    dispatch_faults(w, eds[1], eds[2]);
  }

  return nullptr;
}

std::expected<Box<SignalWatcher>, Error> make_signal_watcher(Context &ctx) {
  auto wake = KFD_TRY(Event::create(ctx, EventType::SIGNAL));
  auto box = Box<SignalWatcher>::create(std::move(wake));
  if (!box)
    return kfd::unexpected(box.error());
  KFD_CHECK((*box)->start(signal_watcher_entry));
  return std::move(*box);
}

std::expected<Box<ExceptionWatcher>, Error>
make_exception_watcher(Context &ctx) {
  auto mem = KFD_TRY(Event::create(ctx, EventType::MEMORY));
  auto hw = KFD_TRY(Event::create(ctx, EventType::HW_EXCEPTION));
  auto wake = KFD_TRY(Event::create(ctx, EventType::SIGNAL));
  auto box = Box<ExceptionWatcher>::create(std::move(mem), std::move(hw),
                                           std::move(wake));
  if (!box)
    return kfd::unexpected(box.error());
  KFD_CHECK((*box)->start(exception_watcher_entry));
  return std::move(*box);
}

std::expected<void, Error> add_entry(SignalWatcher *w, uint32_t event_id,
                                     uint32_t *fence, Condition cond,
                                     uint32_t value, SignalHandler cb,
                                     void *user_data, const void *key) {
  if (!w)
    return kfd::unexpected(EINVAL, "register_handler requires a watcher");
  {
    LockGuard guard(w->mtx);
    KFD_CHECK(w->entries.push_back({event_id, fence, cond, value, cb, user_data,
                                    key, w->next_token++, /*age=*/1}));
  }
  return w->wake_event.signal();
}

std::expected<void, Error> remove_entry(SignalWatcher *w, const void *key) {
  if (!w)
    return {};
  LockGuard guard(w->mtx);
  for (size_t i = 0; i < w->entries.size(); ++i) {
    if (w->entries[i].key == key) {
      w->entries[i] = w->entries[w->entries.size() - 1];
      w->entries.pop_back();
      break;
    }
  }
  return w->wake_event.signal();
}

std::expected<void, Error> set_fault_handler(ExceptionWatcher *watcher,
                                             FaultHandler cb, void *user_data) {
  if (!watcher)
    return kfd::unexpected(EINVAL, "set_fault_handler requires a watcher");
  LockGuard guard(watcher->mtx);
  watcher->fault_handler = cb ? cb : default_fault_handler;
  watcher->fault_user_data = cb ? user_data : nullptr;
  return {};
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

  ctx.signal_watcher = KFD_TRY(make_signal_watcher(ctx));
  ctx.exception_watcher = KFD_TRY(make_exception_watcher(ctx));

  return ctx;
}

std::expected<Device *, Error> Context::device(size_t i) {
  if (i >= nodes.size())
    return kfd::unexpected(EINVAL, "device index %zu >= device count %zu", i,
                           nodes.size());

  return &nodes[i];
}

Context::~Context() {
  signal_watcher = {};
  exception_watcher = {};
  nodes.clear();
  if (fd >= 0)
    ::close(fd);
}

Context::Context(Context &&other)
    : fd(std::exchange(other.fd, -1)), xnack(other.xnack),
      nodes(std::move(other.nodes)),
      signal_watcher(std::move(other.signal_watcher)),
      exception_watcher(std::move(other.exception_watcher)) {
  for (auto &dev : nodes)
    dev.ctx = this;
}

Context &Context::operator=(Context &&other) {
  if (this != &other) {
    signal_watcher = {};
    exception_watcher = {};
    nodes.clear();
    if (fd >= 0)
      ::close(fd);

    fd = std::exchange(other.fd, -1);
    xnack = other.xnack;
    nodes = std::move(other.nodes);
    signal_watcher = std::move(other.signal_watcher);
    exception_watcher = std::move(other.exception_watcher);

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
    return kfd::unexpected(EINVAL, "event id %u >= limit %u", id,
                           static_cast<unsigned>(KFD_SIGNAL_EVENT_LIMIT));
  return &__atomic_load_n(&event_page, __ATOMIC_ACQUIRE)[id];
}

std::expected<uint32_t *, Error> Context::fence_slot(uint32_t id) {
  if (id >= KFD_SIGNAL_EVENT_LIMIT)
    return kfd::unexpected(EINVAL, "fence id %u >= limit %u", id,
                           static_cast<unsigned>(KFD_SIGNAL_EVENT_LIMIT));
  return reinterpret_cast<uint32_t *>(
      &__atomic_load_n(&fence_page, __ATOMIC_ACQUIRE)[id]);
}

std::expected<void, Error>
Context::register_handler(Event &event, SignalHandler cb, void *user_data) {
  return add_entry(signal_watcher.get(), event.event_id(), nullptr,
                   Condition::EQ, 0, cb, user_data, /*key=*/&event);
}

std::expected<void, Error> Context::unregister_handler(Event &event) {
  return remove_entry(signal_watcher.get(), /*key=*/&event);
}

std::expected<void, Error>
Context::register_handler(Signal &sig, Condition cond, uint32_t value,
                          SignalHandler cb, void *user_data) {
  return add_entry(signal_watcher.get(), sig.event_id(), sig.fence_addr(), cond,
                   value, cb, user_data, /*key=*/nullptr);
}

std::expected<void, Error> Context::register_handler(FaultHandler cb,
                                                     void *user_data) {
  return set_fault_handler(exception_watcher.get(), cb, user_data);
}

} // namespace kfd
