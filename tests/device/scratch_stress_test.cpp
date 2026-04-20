#include "test_helpers.h"

#include "libkfd/abi.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <thread>
#include <vector>

static const kfd::test::TestBinary dispatch_kernels[] = {
#include "dispatch_kernels.inc"
};

using kfd::test::make_device_fixture;
using kfd::test::require_ctx;

namespace {

// Dispatch a kernel that has USES_DYNAMIC_STACK set (due to an unresolvable
// indirect call). The private_segment_size config value is passed through to
// ensure_scratch, exercising progressive scratch growth.
std::expected<void, kfd::Error>
dispatch_with_scratch(kfd::Device &dev, kfd::ComputeQueue &compute,
                      const kfd::Kernel &nop, uint32_t scratch_bytes) {
  kfd::DispatchConfig cfg{
      .grid = {.x = 1},
      .block = {.x = 64},
      .private_segment_size = scratch_bytes,
  };
  auto kernarg = KFD_TRY(nop.make_kernargs(dev, cfg));

  auto sig = KFD_TRY(kfd::Signal::create(dev.context()));

  if (auto r = compute.dispatch(nop, cfg, kernarg); !r)
    return r;
  if (auto r = compute.signal(sig); !r)
    return r;
  return sig.wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS);
}

} // namespace

TEST_CASE("Scratch stress - progressive growth on single queue",
          "[device][scratch]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, dispatch_kernels);
  REQUIRE_RESULT(fix);

  auto nop = fix->exe.kernel("use_scratch.kd");
  REQUIRE_RESULT(nop);

  for (unsigned i = 1; i <= 8; ++i) {
    INFO("private_segment_size = " << i * 16);
    REQUIRE_RESULT(
        dispatch_with_scratch(*fix->gpu, fix->compute, *nop, i * 16));
  }
}

TEST_CASE("Scratch stress - repeated progressive growth", "[device][scratch]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, dispatch_kernels);
  REQUIRE_RESULT(fix);

  auto nop = fix->exe.kernel("use_scratch.kd");
  REQUIRE_RESULT(nop);

  constexpr unsigned ROUNDS = 3;
  for (unsigned r = 0; r < ROUNDS; ++r) {
    INFO("round " << r);
    for (unsigned i = 1; i <= 8; ++i) {
      INFO("private_segment_size = " << i * 16);
      REQUIRE_RESULT(
          dispatch_with_scratch(*fix->gpu, fix->compute, *nop, i * 16));
    }
  }
}

TEST_CASE("Scratch stress - parallel queues with progressive growth",
          "[device][scratch]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, dispatch_kernels);
  REQUIRE_RESULT(fix);

  auto nop = fix->exe.kernel("use_scratch.kd");
  REQUIRE_RESULT(nop);

  constexpr unsigned NUM_THREADS = 2;
  constexpr unsigned ROUNDS = 3;

  std::vector<kfd::ComputeQueue> queues;
  queues.reserve(NUM_THREADS);
  for (unsigned i = 0; i < NUM_THREADS; ++i) {
    auto q = kfd::ComputeQueue::create(*fix->gpu);
    REQUIRE_RESULT(q);
    queues.push_back(std::move(*q));
  }

  struct ThreadResult {
    unsigned errors = 0;
    int first_errno = 0;
  };
  std::vector<ThreadResult> results(NUM_THREADS);

  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);

  for (unsigned t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&, t] {
      for (unsigned r = 0; r < ROUNDS; ++r) {
        for (unsigned i = 1; i <= 8; ++i) {
          auto res = dispatch_with_scratch(*fix->gpu, queues[t], *nop, i * 16);
          if (!res) {
            if (!results[t].first_errno)
              results[t].first_errno = res.error().code;
            ++results[t].errors;
            return;
          }
        }
      }
    });
  }

  for (auto &th : threads)
    th.join();

  for (unsigned t = 0; t < NUM_THREADS; ++t) {
    INFO("thread " << t << " errno=" << results[t].first_errno);
    CHECK(results[t].errors == 0);
  }
}
