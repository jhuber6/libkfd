#include "test_helpers.h"

#include "libkfd/abi.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

static const kfd::test::TestBinary dispatch_kernels[] = {
#include "dispatch_kernels.inc"
};

using kfd::test::alloc_host_buffer;
using kfd::test::make_device_fixture;
using kfd::test::require_ctx;

namespace {

// Dispatch a kernel that has USES_DYNAMIC_STACK set (due to an unresolvable
// indirect call) so we can configure it freely.
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

  KFD_CHECK(compute.dispatch(nop, cfg, kernarg));
  KFD_CHECK(compute.signal(sig));
  return sig.wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS);
}

constexpr uint32_t VERIFY_THREADS = 64;

// verify_scratch.kd: out[tid] = sum(tid+i for i in 0..15) = 16*tid + 120.
void check_verify_scratch_output(const void *data) {
  auto *vals = static_cast<const unsigned *>(data);
  for (uint32_t t = 0; t < VERIFY_THREADS; ++t) {
    unsigned expected = 16 * t + 120;
    INFO("thread " << t);
    CHECK(vals[t] == expected);
  }
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

TEST_CASE("Scratch stress - correctness after progressive resize",
          "[device][scratch]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, dispatch_kernels);
  REQUIRE_RESULT(fix);

  auto kernel = fix->exe.kernel("verify_scratch.kd");
  REQUIRE_RESULT(kernel);

  constexpr size_t out_bytes = VERIFY_THREADS * sizeof(unsigned);
  auto out = alloc_host_buffer(
      *fix->gpu, kfd::detail::align_up(out_bytes, kfd::detail::page_size()));

  struct Args {
    unsigned *out;
  };

  uint32_t sizes[] = {256, 512, 1024, 2048};
  for (uint32_t pss : sizes) {
    INFO("private_segment_size = " << pss);
    std::memset(out.data(), 0xFF, out_bytes);

    Args args{static_cast<unsigned *>(out.data())};
    kfd::DispatchConfig cfg{
        .grid = {.x = 1},
        .block = {.x = VERIFY_THREADS},
        .private_segment_size = pss,
    };
    auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
    REQUIRE_RESULT(kernarg);

    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);

    REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
    REQUIRE_RESULT(fix->compute.signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

    check_verify_scratch_output(out.data());
  }
}

TEST_CASE("Scratch stress - rapid serial resize bursts", "[device][scratch]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, dispatch_kernels);
  REQUIRE_RESULT(fix);

  auto kernel = fix->exe.kernel("verify_scratch.kd");
  REQUIRE_RESULT(kernel);

  constexpr unsigned N = 8;
  uint32_t sizes[N] = {256, 512, 256, 1024, 512, 2048, 1024, 4096};
  constexpr size_t out_bytes = VERIFY_THREADS * sizeof(unsigned);
  size_t aligned_out =
      kfd::detail::align_up(out_bytes, kfd::detail::page_size());

  struct Args {
    unsigned *out;
  };

  std::vector<kfd::Buffer> outputs;
  std::vector<kfd::Buffer> kernargs;
  outputs.reserve(N);
  kernargs.reserve(N);

  for (unsigned i = 0; i < N; ++i) {
    outputs.push_back(alloc_host_buffer(*fix->gpu, aligned_out));
    std::memset(outputs.back().data(), 0xFF, out_bytes);

    Args args{static_cast<unsigned *>(outputs.back().data())};
    kfd::DispatchConfig cfg{
        .grid = {.x = 1},
        .block = {.x = VERIFY_THREADS},
        .private_segment_size = sizes[i],
    };
    auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
    REQUIRE_RESULT(kernarg);
    REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
    kernargs.push_back(std::move(*kernarg));
  }

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);
  REQUIRE_RESULT(fix->compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

  for (unsigned i = 0; i < N; ++i) {
    INFO("dispatch " << i << " (private_segment_size=" << sizes[i] << ")");
    check_verify_scratch_output(outputs[i].data());
  }
}

TEST_CASE("Scratch stress - interleaved scratch and non-scratch",
          "[device][scratch]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, dispatch_kernels);
  REQUIRE_RESULT(fix);

  auto scratch_kernel = fix->exe.kernel("verify_scratch.kd");
  REQUIRE_RESULT(scratch_kernel);
  auto nop_kernel = fix->exe.kernel("nop.kd");
  REQUIRE_RESULT(nop_kernel);

  constexpr size_t out_bytes = VERIFY_THREADS * sizeof(unsigned);
  auto out = alloc_host_buffer(
      *fix->gpu, kfd::detail::align_up(out_bytes, kfd::detail::page_size()));

  struct Args {
    unsigned *out;
  };

  for (unsigned round = 0; round < 4; ++round) {
    INFO("round " << round);
    std::memset(out.data(), 0xFF, out_bytes);

    Args args{static_cast<unsigned *>(out.data())};
    kfd::DispatchConfig scratch_cfg{
        .grid = {.x = 1},
        .block = {.x = VERIFY_THREADS},
        .private_segment_size = 256 + round * 256,
    };
    auto ka_scratch =
        scratch_kernel->make_kernargs(*fix->gpu, args, scratch_cfg);
    REQUIRE_RESULT(ka_scratch);
    REQUIRE_RESULT(
        fix->compute.dispatch(*scratch_kernel, scratch_cfg, *ka_scratch));

    kfd::DispatchConfig nop_cfg{.grid = {.x = 1}, .block = {.x = 1}};
    auto ka_nop = nop_kernel->make_kernargs(*fix->gpu, nop_cfg);
    REQUIRE_RESULT(ka_nop);
    REQUIRE_RESULT(fix->compute.dispatch(*nop_kernel, nop_cfg, *ka_nop));

    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);
    REQUIRE_RESULT(fix->compute.signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

    check_verify_scratch_output(out.data());
  }
}
