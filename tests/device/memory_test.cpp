#include "test_helpers.h"

#include "libkfd/abi.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>

static const kfd::test::TestBinary memory_kernels[] = {
#include "memory_kernels.inc"
};

using kfd::test::make_device_fixture;
using kfd::test::require_ctx;

TEST_CASE("Memory - scratch_sum writes correct value", "[device][memory]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, memory_kernels);
  REQUIRE_RESULT(fix);

  auto kernel = fix->exe.kernel("scratch_sum.kd");
  REQUIRE_RESULT(kernel);

  auto out = kfd::test::alloc_host_buffer(*fix->gpu);
  std::memset(out.data(), 0, sizeof(uint32_t));

  struct Args {
    unsigned *out;
  };
  Args args{.out = static_cast<unsigned *>(out.data())};

  kfd::DispatchConfig cfg{
      .grid = {.x = 1},
      .block = {.x = 64},
  };
  auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
  REQUIRE_RESULT(kernarg);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
  REQUIRE_RESULT(fix->compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

  // Thread 0: scratch[i] = 0 + i for i in [0,16), sum = 0+1+...+15 = 120.
  unsigned val;
  std::memcpy(&val, out.data(), sizeof(val));
  CHECK(val == 120);
}

TEST_CASE("Memory - lds_reduce produces correct sum", "[device][memory]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, memory_kernels);
  REQUIRE_RESULT(fix);

  auto kernel = fix->exe.kernel("lds_reduce.kd");
  REQUIRE_RESULT(kernel);

  constexpr uint32_t THREADS = 64;
  constexpr uint32_t NUM_WG = 4;
  constexpr size_t out_bytes = NUM_WG * sizeof(unsigned);
  auto out = kfd::test::alloc_host_buffer(
      *fix->gpu, kfd::detail::align_up(out_bytes, kfd::detail::page_size()));
  std::memset(out.data(), 0xFF, out_bytes);

  struct Args {
    unsigned *out;
  };
  Args args{.out = static_cast<unsigned *>(out.data())};

  kfd::DispatchConfig cfg{
      .grid = {.x = NUM_WG},
      .block = {.x = THREADS},
  };
  auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
  REQUIRE_RESULT(kernarg);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
  REQUIRE_RESULT(fix->compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

  // sum(0..63) = 64*63/2 = 2016.
  unsigned expected = THREADS * (THREADS - 1) / 2;
  auto *vals = static_cast<const unsigned *>(out.data());
  for (uint32_t i = 0; i < NUM_WG; ++i) {
    INFO("workgroup " << i);
    CHECK(vals[i] == expected);
  }
}

TEST_CASE("Memory - dynamic_lds_fill with runtime-sized LDS",
          "[device][memory]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, memory_kernels);
  REQUIRE_RESULT(fix);

  auto kernel = fix->exe.kernel("dynamic_lds_fill.kd");
  REQUIRE_RESULT(kernel);

  constexpr uint32_t THREADS = 64;
  constexpr uint32_t NUM_WG = 4;
  constexpr size_t out_bytes = NUM_WG * sizeof(unsigned);
  auto out = kfd::test::alloc_host_buffer(
      *fix->gpu, kfd::detail::align_up(out_bytes, kfd::detail::page_size()));
  std::memset(out.data(), 0xFF, out_bytes);

  struct Args {
    unsigned *out;
  };
  Args args{.out = static_cast<unsigned *>(out.data())};

  kfd::DispatchConfig cfg{
      .grid = {.x = NUM_WG},
      .block = {.x = THREADS},
      .dynamic_lds = THREADS * sizeof(unsigned),
  };
  auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
  REQUIRE_RESULT(kernarg);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
  REQUIRE_RESULT(fix->compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

  // Each thread writes tid*3; sum = 3 * sum(0..63) = 3 * 2016 = 6048
  unsigned expected = 3 * THREADS * (THREADS - 1) / 2;
  auto *vals = static_cast<const unsigned *>(out.data());
  for (uint32_t i = 0; i < NUM_WG; ++i) {
    INFO("workgroup " << i);
    CHECK(vals[i] == expected);
  }
}

TEST_CASE("Memory - dynamic_stack with non-inlined call", "[device][memory]") {
  auto &ctx = require_ctx();
  auto fix = make_device_fixture(ctx, memory_kernels);
  REQUIRE_RESULT(fix);

  auto kernel = fix->exe.kernel("dynamic_stack.kd");
  REQUIRE_RESULT(kernel);

  auto out = kfd::test::alloc_host_buffer(*fix->gpu);
  std::memset(out.data(), 0, sizeof(uint32_t));

  struct Args {
    unsigned *out;
  };
  Args args{.out = static_cast<unsigned *>(out.data())};

  kfd::DispatchConfig cfg{
      .grid = {.x = 1},
      .block = {.x = 64},
  };
  auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
  REQUIRE_RESULT(kernarg);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
  REQUIRE_RESULT(fix->compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

  // Thread 0: compute(0) => buf[i] = i for i in [0,8), sum = 0+1+...+7 = 28.
  unsigned val;
  std::memcpy(&val, out.data(), sizeof(val));
  CHECK(val == 28);
}
