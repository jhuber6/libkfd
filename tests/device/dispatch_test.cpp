#include "test_helpers.h"

#include "libkfd/abi.h"

#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

static const kfd::test::TestBinary dispatch_kernels[] = {
#include "dispatch_kernels.inc"
};

using kfd::test::DeviceFixture;
using kfd::test::make_device_fixture;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

TEST_CASE("Dispatch - NOP kernel completes", "[device][dispatch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("nop.kd");
      REQUIRE_RESULT(kernel);

      kfd::DispatchConfig cfg{
          .grid = {.x = 1},
          .block = {.x = 64},
      };
      auto kernarg = kernel->make_kernargs(*fix->gpu, cfg);
      REQUIRE_RESULT(kernarg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
      REQUIRE_RESULT(fix->compute.signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("Dispatch - single-thread store writes sentinel",
          "[device][dispatch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("store.kd");
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
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      unsigned val;
      std::memcpy(&val, out.data(), sizeof(val));
      CHECK(val == 0xCAFEBABE);
    }
  }
}

TEST_CASE("Dispatch - fill_local_ids writes correct thread IDs",
          "[device][dispatch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("fill_local_ids.kd");
      REQUIRE_RESULT(kernel);

      constexpr uint32_t THREADS = 64;
      constexpr size_t out_bytes = THREADS * sizeof(unsigned);
      auto out = kfd::test::alloc_host_buffer(
          *fix->gpu,
          kfd::detail::align_up(out_bytes, kfd::detail::page_size()));
      std::memset(out.data(), 0xFF, out_bytes);

      struct Args {
        unsigned *out;
      };
      Args args{.out = static_cast<unsigned *>(out.data())};

      kfd::DispatchConfig cfg{
          .grid = {.x = 1},
          .block = {.x = THREADS},
      };
      auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
      REQUIRE_RESULT(kernarg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
      REQUIRE_RESULT(fix->compute.signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *vals = static_cast<const unsigned *>(out.data());
      for (uint32_t i = 0; i < THREADS; ++i) {
        INFO("thread " << i);
        CHECK(vals[i] == i);
      }
    }
  }
}

TEST_CASE("Dispatch - fill_wg_ids across multiple workgroups",
          "[device][dispatch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("fill_wg_ids.kd");
      REQUIRE_RESULT(kernel);

      constexpr uint32_t NUM_WG = 16;
      constexpr uint32_t THREADS = 64;
      constexpr size_t out_bytes = NUM_WG * sizeof(unsigned);
      auto out = kfd::test::alloc_host_buffer(
          *fix->gpu,
          kfd::detail::align_up(out_bytes, kfd::detail::page_size()));
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
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *vals = static_cast<const unsigned *>(out.data());
      for (uint32_t i = 0; i < NUM_WG; ++i) {
        INFO("workgroup " << i);
        CHECK(vals[i] == i);
      }
    }
  }
}

namespace {

void run_check_dims(DeviceFixture &fix, kfd::Dim3 grid, kfd::Dim3 block) {
  auto kernel = fix.exe.kernel("check_dims.kd");
  REQUIRE_RESULT(kernel);

  auto out = kfd::test::alloc_host_buffer(
      *fix.gpu,
      kfd::detail::align_up(6 * sizeof(unsigned), kfd::detail::page_size()));
  std::memset(out.data(), 0xFF, 6 * sizeof(unsigned));

  struct Args {
    unsigned *out;
  };
  Args args{.out = static_cast<unsigned *>(out.data())};

  kfd::DispatchConfig cfg{.grid = grid, .block = block};
  auto kernarg = kernel->make_kernargs(*fix.gpu, args, cfg);
  REQUIRE_RESULT(kernarg);

  auto &ctx = fix.gpu->context();
  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  REQUIRE_RESULT(fix.compute.dispatch(*kernel, cfg, *kernarg));
  REQUIRE_RESULT(fix.compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

  auto *vals = static_cast<const unsigned *>(out.data());
  CHECK(vals[0] == grid.x);
  CHECK(vals[1] == grid.y);
  CHECK(vals[2] == grid.z);
  CHECK(vals[3] == block.x);
  CHECK(vals[4] == block.y);
  CHECK(vals[5] == block.z);
}

} // namespace

TEST_CASE("Dispatch - 1D grid dimensions correct", "[device][dispatch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);
      run_check_dims(*fix, {.x = 8}, {.x = 128});
    }
  }
}

TEST_CASE("Dispatch - 2D grid dimensions correct", "[device][dispatch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);
      run_check_dims(*fix, {.x = 4, .y = 3}, {.x = 32, .y = 2});
    }
  }
}

TEST_CASE("Dispatch - 3D grid dimensions correct", "[device][dispatch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);
      run_check_dims(*fix, {.x = 2, .y = 3, .z = 4}, {.x = 8, .y = 4, .z = 2});
    }
  }
}
