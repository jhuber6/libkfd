#include "test_helpers.h"

#include "libkfd/abi.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>

static const kfd::test::TestBinary cooperative_kernels[] = {
#include "cooperative_kernels.inc"
};

using kfd::test::require_ctx;
using kfd::test::require_gpu;

TEST_CASE("Cooperative - occupancy calculation", "[cooperative]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto fix = kfd::test::find_compatible_binary(cooperative_kernels, gpu);
      if (!fix)
        SKIP("no compatible kernel for this GPU");

      auto buf = kfd::test::read_file(fix->path);
      auto compute = kfd::test::create_queue<kfd::ComputeQueue>(gpu);
      REQUIRE_RESULT(compute);
      auto exe = kfd::Executable::load(gpu, buf, *compute);
      REQUIRE_RESULT(exe);

      auto kernel = exe->kernel("coop_store.kd");
      REQUIRE_RESULT(kernel);

      const auto &props = gpu.properties();
      uint32_t simd_per_cu = props.simd_per_cu ? props.simd_per_cu : 1;
      uint32_t num_cus = props.simd_count / simd_per_cu;

      kfd::Dim3 block{.x = 64};
      uint32_t bpc = kfd::abi::blocks_per_cu(props, gpu.gfx_version(),
                                             kernel->descriptor(), block);

      INFO("blocks per CU: " << bpc);
      CHECK(bpc > 0);
      CHECK(bpc * num_cus <= num_cus * props.max_waves_per_simd * simd_per_cu);
    }
  }
}

TEST_CASE("Cooperative - queue create/destroy", "[device][cooperative]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      if (gpu.properties().num_gws == 0 &&
          gpu.gfx_version() < kfd::abi::GFX_VERSION_GFX12)
        SKIP("device does not support cooperative (no GWS)");

      auto cq = kfd::CooperativeQueue::create(gpu);
      REQUIRE_RESULT(cq);
      CHECK(static_cast<bool>(*cq));
    }
  }
}

TEST_CASE("Cooperative - basic dispatch", "[device][cooperative]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      if (gpu.properties().num_gws == 0 &&
          gpu.gfx_version() < kfd::abi::GFX_VERSION_GFX12)
        SKIP("device does not support cooperative (no GWS)");

      auto cq = kfd::CooperativeQueue::create(gpu);
      REQUIRE_RESULT(cq);

      auto fix = kfd::test::find_compatible_binary(cooperative_kernels, gpu);
      if (!fix)
        SKIP("no compatible kernel for this GPU");

      auto buf = kfd::test::read_file(fix->path);
      auto exe = kfd::Executable::load(gpu, buf, *cq);
      REQUIRE_RESULT(exe);

      auto kernel = exe->kernel("coop_store.kd");
      REQUIRE_RESULT(kernel);

      constexpr uint32_t NUM_WG = 4;
      constexpr uint32_t THREADS = 64;
      constexpr size_t out_bytes = NUM_WG * sizeof(unsigned);
      auto out = kfd::test::alloc_host_buffer(
          gpu, kfd::detail::align_up(out_bytes, kfd::detail::page_size()));
      std::memset(out.data(), 0xFF, out_bytes);

      struct Args {
        unsigned *out;
      };
      Args args{.out = static_cast<unsigned *>(out.data())};

      kfd::DispatchConfig cfg{
          .grid = {.x = NUM_WG},
          .block = {.x = THREADS},
      };
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(cq->dispatch(*kernel, cfg, *kernarg));
      REQUIRE_RESULT(cq->signal(*sig));
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

TEST_CASE("Cooperative - co-residency probe", "[device][cooperative]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      if (gpu.properties().num_gws == 0 &&
          gpu.gfx_version() < kfd::abi::GFX_VERSION_GFX12)
        SKIP("device does not support cooperative (no GWS)");

      auto cq = kfd::CooperativeQueue::create(gpu);
      REQUIRE_RESULT(cq);

      auto fix = kfd::test::find_compatible_binary(cooperative_kernels, gpu);
      if (!fix)
        SKIP("no compatible kernel for this GPU");

      auto buf = kfd::test::read_file(fix->path);
      auto exe = kfd::Executable::load(gpu, buf, *cq);
      REQUIRE_RESULT(exe);

      auto kernel = exe->kernel("coop_probe.kd");
      REQUIRE_RESULT(kernel);

      const auto &props = gpu.properties();
      uint32_t simd_per_cu = props.simd_per_cu ? props.simd_per_cu : 1;
      uint32_t num_cus = props.simd_count / simd_per_cu;

      const auto &kd = kernel->descriptor();
      uint32_t wave_size =
          (kd.kernel_code_properties & kfd::abi::ENABLE_WAVEFRONT_SIZE32) ? 32
                                                                          : 64;

      kfd::Dim3 block{.x = 64};
      uint32_t waves_per_block = block.x / wave_size;
      uint32_t bpc =
          kfd::abi::blocks_per_cu(props, gpu.gfx_version(), kd, block);
      REQUIRE(bpc > 0);
      uint32_t max_wgs = bpc * num_cus;

      uint32_t num_wgs = max_wgs > 4 ? max_wgs / 2 : max_wgs;
      uint32_t total_waves = num_wgs * waves_per_block;
      INFO("launching " << num_wgs << " workgroups (" << total_waves
                        << " waves) of " << max_wgs << " max workgroups");

      size_t arrived_bytes =
          kfd::detail::align_up(sizeof(unsigned), kfd::detail::page_size());
      size_t results_bytes = kfd::detail::align_up(
          static_cast<size_t>(total_waves) * sizeof(unsigned),
          kfd::detail::page_size());

      auto arrived_buf = kfd::test::alloc_host_buffer(gpu, arrived_bytes);
      auto results_buf = kfd::test::alloc_host_buffer(gpu, results_bytes);
      std::memset(arrived_buf.data(), 0, sizeof(unsigned));
      std::memset(results_buf.data(), 0, total_waves * sizeof(unsigned));

      struct Args {
        unsigned *arrived;
        unsigned *results;
        unsigned total;
      };
      Args args{
          .arrived = static_cast<unsigned *>(arrived_buf.data()),
          .results = static_cast<unsigned *>(results_buf.data()),
          .total = total_waves,
      };

      kfd::DispatchConfig cfg{
          .grid = {.x = num_wgs},
          .block = block,
      };
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(cq->dispatch(*kernel, cfg, *kernarg));
      REQUIRE_RESULT(cq->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *results = static_cast<const unsigned *>(results_buf.data());
      uint32_t failures = 0;
      for (uint32_t i = 0; i < total_waves; ++i) {
        if (results[i] != 1)
          ++failures;
      }
      INFO(failures << " of " << total_waves
                    << " waves did not observe all peers");
      CHECK(failures == 0);
    }
  }
}
