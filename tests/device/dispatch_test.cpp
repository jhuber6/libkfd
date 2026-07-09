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
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, cfg);

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
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

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
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

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
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

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

TEST_CASE("Dispatch - static grid split reconstructs a full launch",
          "[device][dispatch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("grid_memset.kd");
      REQUIRE_RESULT(kernel);

      constexpr uint32_t NUM_WG = 256;
      constexpr uint32_t THREADS = 64;
      constexpr uint32_t CHUNKS = 8;
      constexpr uint32_t PER = NUM_WG / CHUNKS; // work-groups per chunk
      constexpr uint32_t N = NUM_WG * THREADS;
      static_assert(NUM_WG % CHUNKS == 0);

      auto out = kfd::test::alloc_host_buffer(
          *fix->gpu, kfd::detail::align_up(N * sizeof(unsigned),
                                           kfd::detail::page_size()));
      auto *vals = static_cast<unsigned *>(out.data());

      struct Args {
        unsigned *out;
        unsigned tag;
      };
      kfd::DispatchConfig full{.grid = {.x = NUM_WG}, .block = {.x = THREADS}};

      // Sanity: an unmodified full-grid dispatch still stamps everything.
      {
        std::memset(vals, 0, N * sizeof(unsigned));
        auto ka = kernel->alloc();
        REQUIRE_RESULT(ka);
        Args args{.out = vals, .tag = 0xABCDu};
        kernel->fill(*ka, args, full);
        auto sig = kfd::Signal::create(ctx);
        REQUIRE_RESULT(sig);
        REQUIRE_RESULT(fix->compute.dispatch(*kernel, full, *ka));
        REQUIRE_RESULT(fix->compute.signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
        bool ok = true;
        for (uint32_t i = 0; i < N; ++i)
          ok &= vals[i] == 0xABCDu;
        CHECK(ok);
      }

      // Split the single logical grid into CHUNKS disjoint sub-dispatches,
      // each stamping a distinct tag over its own work-group range.
      std::memset(vals, 0, N * sizeof(unsigned));
      std::vector<kfd::Buffer> kernargs;
      for (uint32_t c = 0; c < CHUNKS; ++c) {
        auto ka = kernel->alloc();
        REQUIRE_RESULT(ka);
        Args args{.out = vals, .tag = c + 1};
        kernel->fill(*ka, args, full); // full grid seen by the shader
        kernargs.push_back(std::move(*ka));

        kfd::DispatchConfig sub = full;
        sub.grid_start = {.x = c * PER, .y = 0, .z = 0};
        sub.grid_count = {.x = PER, .y = 0, .z = 0};
        REQUIRE_RESULT(fix->compute.dispatch(*kernel, sub, kernargs.back()));
      }

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);
      REQUIRE_RESULT(fix->compute.signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      // Every slot must carry its own chunk's tag. A dropped COMPUTE_START
      // offset would let the last (largest) chunk overwrite all slots.
      unsigned mismatches = 0;
      for (uint32_t i = 0; i < N; ++i) {
        unsigned expected = i / THREADS / PER + 1;
        if (vals[i] != expected) {
          if (mismatches < 8) {
            INFO("slot " << i << " (block " << i / THREADS << ")");
            CHECK(vals[i] == expected);
          }
          ++mismatches;
        }
      }
      CHECK(mismatches == 0);
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
  auto kernarg = kernel->alloc();
  REQUIRE_RESULT(kernarg);
  kernel->fill(*kernarg, args, cfg);

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
