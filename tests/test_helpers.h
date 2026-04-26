#ifndef LIBKFD_TEST_HELPERS_H
#define LIBKFD_TEST_HELPERS_H

#include "libkfd/context.h"
#include "libkfd/detail/elf.h"
#include "libkfd/detail/utility.h"
#include "libkfd/loader.h"
#include "libkfd/memory.h"
#include "libkfd/queue.h"
#include "libkfd/signal.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

// Like REQUIRE / CHECK, but on failure prints the kfd::Error diagnostic via
// INFO so the Catch2 output includes the actual error string, not just "false".
#define REQUIRE_RESULT(expr)                                                   \
  do {                                                                         \
    auto &&_kfd_r = (expr);                                                    \
    if (!_kfd_r.has_value()) {                                                 \
      INFO(kfd::strerror(_kfd_r));                                             \
      REQUIRE(_kfd_r.has_value());                                             \
    }                                                                          \
  } while (0)

#define CHECK_RESULT(expr)                                                     \
  do {                                                                         \
    auto &&_kfd_r = (expr);                                                    \
    if (!_kfd_r.has_value()) {                                                 \
      INFO(kfd::strerror(_kfd_r));                                             \
      CHECK(_kfd_r.has_value());                                               \
    }                                                                          \
  } while (0)

namespace kfd::test {

// Default timeout for signal waits in tests (5 seconds). If a GPU operation
// hasn't completed in this window, something is probably hung.
inline constexpr uint64_t WAIT_TIMEOUT_NS = 5'000'000'000;

// Lazily creates and caches a single Context for the lifetime of the process.
// Returns nullptr if /dev/kfd is unavailable.
inline kfd::Context *get_ctx() {
  static auto ctx = kfd::Context::create();
  return ctx ? &*ctx : nullptr;
}

// Returns the singleton Context, SKIPping the test if KFD is unavailable or
// no GPUs are present.
inline kfd::Context &require_ctx() {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");
  if (ctx->num_devices() == 0)
    SKIP("No GPUs in topology");
  return *ctx;
}

// Returns device \p i from \p ctx, aborting the current section on failure.
inline kfd::Device &require_gpu(kfd::Context &ctx, size_t i) {
  auto dev = ctx.device(i);
  REQUIRE_RESULT(dev);
  return **dev;
}

// Read an entire binary file into memory.
inline std::vector<std::byte> read_file(const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  REQUIRE(f);
  std::fseek(f, 0, SEEK_END);
  auto sz = std::ftell(f);
  REQUIRE(sz > 0);
  std::fseek(f, 0, SEEK_SET);
  std::vector<std::byte> buf(static_cast<size_t>(sz));
  REQUIRE(std::fread(buf.data(), 1, buf.size(), f) == buf.size());
  std::fclose(f);
  return buf;
}

struct TestBinary {
  const char *path;
  const char *arch;
};

// Find the first test binary whose ELF flags are compatible with \p dev.
// Returns nullptr if nothing matches.
inline const TestBinary *
find_compatible_binary(std::span<const TestBinary> binaries, kfd::Device &dev) {
  namespace elf = kfd::detail::elf;
  for (const auto &bin : binaries) {
    auto buf = read_file(bin.path);
    auto parsed = elf::ELF64LE::create(
        std::span<const std::byte>(buf.data(), buf.size()));
    if (!parsed)
      continue;
    bool sramecc = (dev.properties().capability &
                    kfd::NodeProperties::NODE_CAP_SRAM_EDCSUPPORTED) != 0;
    if (elf::is_compatible(parsed->header().e_flags, dev.gfx_version(),
                           dev.context().xnack_enabled(), sramecc))
      return &bin;
  }
  return nullptr;
}

// Common MemFlags combination for host-accessible GPU buffers used in tests.
inline constexpr MemFlags HOST_GTT_FLAGS =
    MemFlags::WRITABLE | MemFlags::EXECUTABLE | MemFlags::UNCACHED |
    MemFlags::HOST_ACCESS;

// Allocate a host-visible, uncached GTT buffer and map it to the device.
inline kfd::Buffer alloc_host_buffer(kfd::Device &dev,
                                     size_t size = kfd::detail::page_size()) {
  auto buf =
      kfd::Buffer::allocate(dev, size, kfd::MemType::GTT, HOST_GTT_FLAGS);
  REQUIRE_RESULT(buf);
  REQUIRE_RESULT(buf->map(dev));
  return std::move(*buf);
}

// The kernel has a hard cap on SDMA queues (typically 32). When CTest runs
// many device tests in parallel, transient ENOMEM is expected. Retry with
// backoff so tests don't fail spuriously under contention.
template <typename QueueT, typename... Args>
std::expected<QueueT, kfd::Error> create_queue(Args &&...args) {
  constexpr int MAX_RETRIES = 100;
  constexpr long RETRY_NS = 250'000'000L; // 250 ms
  for (int attempt = 0;; ++attempt) {
    auto q = QueueT::create(std::forward<Args>(args)...);
    if (q || q.error().code != ENOMEM || attempt >= MAX_RETRIES)
      return q;
    struct timespec ts = {.tv_sec = 0, .tv_nsec = RETRY_NS};
    ::nanosleep(&ts, nullptr);
  }
}

// Common harness for device tests that need an SDMA queue, a compute queue,
// and a loaded executable built from a table of per-arch test binaries.
struct DeviceFixture {
  kfd::Device *gpu;
  kfd::Executable exe;
  const TestBinary *bin = nullptr;
  kfd::SDMAQueue sdma;
  kfd::ComputeQueue compute;
};

inline std::expected<DeviceFixture, kfd::Error>
make_device_fixture(kfd::Device &dev, std::span<const TestBinary> kernels) {
  auto *bin = find_compatible_binary(kernels, dev);
  if (!bin)
    return kfd::unexpected(ENOEXEC, "no compatible kernel for this GPU");
  auto sdma = KFD_TRY(create_queue<kfd::SDMAQueue>(dev));
  auto compute = KFD_TRY(create_queue<kfd::ComputeQueue>(dev));
  auto buf = read_file(bin->path);
  auto exe = KFD_TRY(kfd::Executable::load(dev, buf, compute));
  return DeviceFixture{&dev, std::move(exe), bin, std::move(sdma),
                       std::move(compute)};
}

} // namespace kfd::test

#endif // LIBKFD_TEST_HELPERS_H
