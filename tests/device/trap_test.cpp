#include "test_helpers.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <thread>

static const kfd::test::TestBinary trap_kernels[] = {
#include "trap_kernels.inc"
};

using kfd::test::make_device_fixture;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

namespace {

std::atomic<int> g_fault_count{0};
std::atomic<kfd::QueueError::Kind> g_fault_kind{kfd::QueueError::Kind::None};

void record_fault(const kfd::FaultInfo &f, void *) {
  if (f.kind == kfd::FaultInfo::Kind::QueueException) {
    g_fault_kind.store(f.queue.kind, std::memory_order_release);
    g_fault_count.fetch_add(1, std::memory_order_acq_rel);
  }
}

} // namespace

TEST_CASE("Trap - wave trap delivers a queue exception", "[device][trap]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, trap_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("wave_trap.kd");
      REQUIRE_RESULT(kernel);

      g_fault_count.store(0, std::memory_order_release);
      g_fault_kind.store(kfd::QueueError::Kind::None,
                         std::memory_order_release);
      REQUIRE_RESULT(ctx.register_handler(record_fault, nullptr));

      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 1}};
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, cfg);

      REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));

      // Poll for the wave trap exception.
      auto deadline = std::chrono::steady_clock::now() +
                      std::chrono::nanoseconds(kfd::test::WAIT_TIMEOUT_NS);
      while (g_fault_count.load(std::memory_order_acquire) == 0 &&
             std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

      REQUIRE_RESULT(ctx.register_handler(nullptr, nullptr));

      CHECK(g_fault_count.load() >= 1);
      CHECK(g_fault_kind.load() == kfd::QueueError::Kind::Abort);
    }
  }
}
