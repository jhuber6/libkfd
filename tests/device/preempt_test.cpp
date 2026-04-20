#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <ctime>
#include <vector>

static const kfd::test::TestBinary preempt_kernels[] = {
#include "preempt_kernels.inc"
};

using kfd::test::alloc_host_buffer;
using kfd::test::create_queue;
using kfd::test::make_device_fixture;
using kfd::test::require_ctx;

namespace {

constexpr struct timespec SETTLE = {.tv_sec = 0, .tv_nsec = 100'000'000};

kfd::Buffer dispatch_spinner(kfd::ComputeQueue &queue,
                             const kfd::Kernel &kernel, kfd::Device &gpu,
                             unsigned *flag) {
  struct Args {
    unsigned *flag;
  };
  Args args{flag};
  kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 1}};
  auto kernarg = kernel.make_kernargs(gpu, args, cfg);
  REQUIRE_RESULT(kernarg);
  REQUIRE_RESULT(queue.dispatch(kernel, cfg, *kernarg));
  return std::move(*kernarg);
}

void check_device_alive(kfd::Context &ctx, kfd::ComputeQueue &compute,
                        kfd::Device &gpu) {
  auto out_buf = alloc_host_buffer(gpu);
  auto *out = static_cast<unsigned *>(out_buf.data());
  *out = 0;
  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);
  REQUIRE_RESULT(compute.write_data(out, 0xDEADBEEFu));
  REQUIRE_RESULT(compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  CHECK(*out == 0xDEADBEEFu);
}

} // namespace

TEST_CASE("CWSR - queue destruction preempts spinning kernel",
          "[device][cwsr]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, preempt_kernels);
  REQUIRE_RESULT(fix);

  if (fix->gpu->properties().cwsr_size == 0)
    SKIP("CWSR not supported on this device");

  auto kernel = fix->exe.kernel("spin_on_flag.kd");
  REQUIRE_RESULT(kernel);

  auto flag_buf = alloc_host_buffer(*fix->gpu);
  auto *flag = static_cast<unsigned *>(flag_buf.data());
  __atomic_store_n(flag, 0u, __ATOMIC_RELEASE);

  {
    kfd::Buffer ka;
    auto doomed = create_queue<kfd::ComputeQueue>(*fix->gpu);
    REQUIRE_RESULT(doomed);
    ka = dispatch_spinner(*doomed, *kernel, *fix->gpu, flag);
    ::nanosleep(&SETTLE, nullptr);
  }

  __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);
  check_device_alive(ctx, fix->compute, *fix->gpu);
}

TEST_CASE("CWSR - preemption with active scratch", "[device][cwsr]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, preempt_kernels);
  REQUIRE_RESULT(fix);

  if (fix->gpu->properties().cwsr_size == 0)
    SKIP("CWSR not supported on this device");

  auto kernel = fix->exe.kernel("spin_with_scratch.kd");
  REQUIRE_RESULT(kernel);

  auto flag_buf = alloc_host_buffer(*fix->gpu);
  auto *flag = static_cast<unsigned *>(flag_buf.data());
  __atomic_store_n(flag, 0u, __ATOMIC_RELEASE);

  {
    kfd::Buffer ka;
    auto doomed = create_queue<kfd::ComputeQueue>(*fix->gpu);
    REQUIRE_RESULT(doomed);
    ka = dispatch_spinner(*doomed, *kernel, *fix->gpu, flag);
    ::nanosleep(&SETTLE, nullptr);
  }

  __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);
  check_device_alive(ctx, fix->compute, *fix->gpu);
}

TEST_CASE("CWSR - repeated preemption stress", "[device][cwsr][stress]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, preempt_kernels);
  REQUIRE_RESULT(fix);

  if (fix->gpu->properties().cwsr_size == 0)
    SKIP("CWSR not supported on this device");

  auto kernel = fix->exe.kernel("spin_on_flag.kd");
  REQUIRE_RESULT(kernel);

  auto flag_buf = alloc_host_buffer(*fix->gpu);
  auto *flag = static_cast<unsigned *>(flag_buf.data());

  constexpr int ITERATIONS = 3;
  for (int i = 0; i < ITERATIONS; ++i) {
    INFO("iteration " << i);
    __atomic_store_n(flag, 0u, __ATOMIC_RELEASE);

    {
      kfd::Buffer ka;
      auto doomed = create_queue<kfd::ComputeQueue>(*fix->gpu);
      REQUIRE_RESULT(doomed);
      ka = dispatch_spinner(*doomed, *kernel, *fix->gpu, flag);
      ::nanosleep(&SETTLE, nullptr);
    }

    __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);
  }

  check_device_alive(ctx, fix->compute, *fix->gpu);
}

TEST_CASE("CWSR - concurrent queue preemption", "[device][cwsr]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, preempt_kernels);
  REQUIRE_RESULT(fix);

  if (fix->gpu->properties().cwsr_size == 0)
    SKIP("CWSR not supported on this device");

  auto kernel = fix->exe.kernel("spin_on_flag.kd");
  REQUIRE_RESULT(kernel);

  auto flag_buf = alloc_host_buffer(*fix->gpu);
  auto *flag = static_cast<unsigned *>(flag_buf.data());
  __atomic_store_n(flag, 0u, __ATOMIC_RELEASE);

  {
    constexpr int N = 3;
    std::vector<kfd::Buffer> kernargs;
    std::vector<kfd::ComputeQueue> queues;
    kernargs.reserve(N);
    queues.reserve(N);

    for (int i = 0; i < N; ++i) {
      auto q = create_queue<kfd::ComputeQueue>(*fix->gpu);
      if (!q) {
        if (queues.empty())
          SKIP("Could not create additional compute queues");
        break;
      }
      queues.push_back(std::move(*q));
    }

    for (auto &q : queues)
      kernargs.push_back(dispatch_spinner(q, *kernel, *fix->gpu, flag));

    ::nanosleep(&SETTLE, nullptr);
  }

  __atomic_store_n(flag, 1u, __ATOMIC_RELEASE);
  check_device_alive(ctx, fix->compute, *fix->gpu);
}
