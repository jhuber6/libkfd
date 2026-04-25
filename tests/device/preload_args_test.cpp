#include "test_helpers.h"

#include "libkfd/abi.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

static const kfd::test::TestBinary preload_args_kernels[] = {
#include "preload_args_kernels.inc"
};

using kfd::test::require_ctx;
using kfd::test::require_gpu;

TEST_CASE("Dispatch - kernarg preload verifies 16 scalar arguments",
          "[device][dispatch][preload]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = kfd::test::make_device_fixture(gpu, preload_args_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("verify_preload.kd");
      REQUIRE_RESULT(kernel);

      const auto &kd = kernel->descriptor();
      uint16_t preload_len =
          kd.kernarg_preload & kfd::abi::KERNARG_PRELOAD_LENGTH_MASK;
      REQUIRE(preload_len > 0);

      struct Args {
        uint32_t a[16];
      };
      Args args;
      for (uint32_t i = 0; i < 16; ++i)
        args.a[i] = i + 1;

      kfd::DispatchConfig cfg{
          .grid = {.x = 1},
          .block = {.x = 1},
      };
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg, *sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}
