#include "test_helpers.h"

#include "libkfd/abi.h"

#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstdint>
#include <cstring>

static const kfd::test::TestBinary sync_kernels[] = {
#include "sync_kernels.inc"
};

using kfd::test::alloc_host_buffer;
using kfd::test::create_queue;
using kfd::test::make_device_fixture;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

namespace {

constexpr uint32_t THREADS = 64;
constexpr size_t BUF_BYTES = THREADS * sizeof(unsigned);

size_t buf_alloc_size() {
  return kfd::detail::align_up(BUF_BYTES, kfd::detail::page_size());
}

} // namespace

TEST_CASE("Sync - SDMA fill then compute dispatch", "[device][sync]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, sync_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("increment.kd");
      REQUIRE_RESULT(kernel);

      auto in = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      auto out = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      std::memset(in.data(), 0, buf_alloc_size());
      std::memset(out.data(), 0, buf_alloc_size());

      auto sdma_done = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sdma_done);
      auto done = kfd::Signal::create(ctx);
      REQUIRE_RESULT(done);

      REQUIRE_RESULT(fix->sdma.const_fill(in.data(), 42u, BUF_BYTES));
      REQUIRE_RESULT(fix->sdma.signal(*sdma_done));

      REQUIRE_RESULT(fix->compute.wait(*sdma_done, kfd::Condition::EQ, 0));

      struct Args {
        unsigned *out;
        const unsigned *in;
      };
      Args args{static_cast<unsigned *>(out.data()),
                static_cast<const unsigned *>(in.data())};
      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = THREADS}};
      auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
      REQUIRE_RESULT(kernarg);

      REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
      REQUIRE_RESULT(fix->compute.signal(*done));
      REQUIRE_RESULT(
          done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *vals = static_cast<const unsigned *>(out.data());
      for (uint32_t i = 0; i < THREADS; ++i) {
        INFO("element " << i);
        CHECK(vals[i] == 43u);
      }
    }
  }
}

TEST_CASE("Sync - compute dispatch then SDMA readback", "[device][sync]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, sync_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("increment.kd");
      REQUIRE_RESULT(kernel);

      auto in = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      auto out = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      auto readback = alloc_host_buffer(*fix->gpu, buf_alloc_size());

      auto *src = static_cast<unsigned *>(in.data());
      for (uint32_t i = 0; i < THREADS; ++i)
        src[i] = i * 10;
      std::memset(out.data(), 0, buf_alloc_size());
      std::memset(readback.data(), 0xFF, buf_alloc_size());

      auto compute_done = kfd::Signal::create(ctx);
      REQUIRE_RESULT(compute_done);
      auto done = kfd::Signal::create(ctx);
      REQUIRE_RESULT(done);

      struct Args {
        unsigned *out;
        const unsigned *in;
      };
      Args args{static_cast<unsigned *>(out.data()),
                static_cast<const unsigned *>(in.data())};
      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = THREADS}};
      auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
      REQUIRE_RESULT(kernarg);

      REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
      REQUIRE_RESULT(fix->compute.signal(*compute_done));

      REQUIRE_RESULT(fix->sdma.wait(*compute_done, kfd::Condition::EQ, 0));
      REQUIRE_RESULT(
          fix->sdma.copy_linear(readback.data(), out.data(), BUF_BYTES));
      REQUIRE_RESULT(fix->sdma.signal(*done));
      REQUIRE_RESULT(
          done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *vals = static_cast<const unsigned *>(readback.data());
      for (uint32_t i = 0; i < THREADS; ++i) {
        INFO("element " << i);
        CHECK(vals[i] == i * 10 + 1);
      }
    }
  }
}

TEST_CASE("Sync - SDMA to compute to SDMA pipeline", "[device][sync]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, sync_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("increment.kd");
      REQUIRE_RESULT(kernel);

      auto in = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      auto out = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      auto readback = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      std::memset(in.data(), 0, buf_alloc_size());
      std::memset(out.data(), 0, buf_alloc_size());
      std::memset(readback.data(), 0xFF, buf_alloc_size());

      auto sig = kfd::Signal::create(ctx, /*initial=*/3);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(fix->sdma.const_fill(in.data(), 100u, BUF_BYTES));
      REQUIRE_RESULT(fix->sdma.signal(*sig));
      REQUIRE_RESULT(fix->sdma.wait(*sig, kfd::Condition::EQ, 1));
      REQUIRE_RESULT(
          fix->sdma.copy_linear(readback.data(), out.data(), BUF_BYTES));
      REQUIRE_RESULT(fix->sdma.signal(*sig));

      REQUIRE_RESULT(fix->compute.wait(*sig, kfd::Condition::EQ, 2));
      struct Args {
        unsigned *out;
        const unsigned *in;
      };
      Args args{static_cast<unsigned *>(out.data()),
                static_cast<const unsigned *>(in.data())};
      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = THREADS}};
      auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
      REQUIRE_RESULT(kernarg);
      REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
      REQUIRE_RESULT(fix->compute.signal(*sig));

      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *vals = static_cast<const unsigned *>(readback.data());
      for (uint32_t i = 0; i < THREADS; ++i) {
        INFO("element " << i);
        CHECK(vals[i] == 101u);
      }
    }
  }
}

TEST_CASE("Sync - two SDMA producers join before compute", "[device][sync]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, sync_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("add_buffers.kd");
      REQUIRE_RESULT(kernel);

      auto sdma2 = create_queue<kfd::SDMAQueue>(*fix->gpu);
      if (!sdma2)
        SKIP("Could not create second SDMA queue");

      auto buf_a = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      auto buf_b = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      auto out = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      std::memset(buf_a.data(), 0, buf_alloc_size());
      std::memset(buf_b.data(), 0, buf_alloc_size());
      std::memset(out.data(), 0, buf_alloc_size());

      auto ready = kfd::Signal::create(ctx, /*initial=*/2);
      REQUIRE_RESULT(ready);
      auto done = kfd::Signal::create(ctx);
      REQUIRE_RESULT(done);

      REQUIRE_RESULT(fix->sdma.const_fill(buf_a.data(), 10u, BUF_BYTES));
      REQUIRE_RESULT(fix->sdma.signal(*ready));

      REQUIRE_RESULT(sdma2->const_fill(buf_b.data(), 20u, BUF_BYTES));
      REQUIRE_RESULT(sdma2->signal(*ready));

      REQUIRE_RESULT(fix->compute.wait(*ready, kfd::Condition::EQ, 0));

      struct Args {
        unsigned *out;
        const unsigned *a;
        const unsigned *b;
      };
      Args args{static_cast<unsigned *>(out.data()),
                static_cast<const unsigned *>(buf_a.data()),
                static_cast<const unsigned *>(buf_b.data())};
      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = THREADS}};
      auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
      REQUIRE_RESULT(kernarg);

      REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
      REQUIRE_RESULT(fix->compute.signal(*done));
      REQUIRE_RESULT(
          done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *vals = static_cast<const unsigned *>(out.data());
      for (uint32_t i = 0; i < THREADS; ++i) {
        INFO("element " << i);
        CHECK(vals[i] == 30u);
      }
    }
  }
}

TEST_CASE("Sync - repeated pipeline iterations", "[device][sync][stress]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, sync_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto kernel = fix->exe.kernel("increment.kd");
      REQUIRE_RESULT(kernel);

      auto in = alloc_host_buffer(*fix->gpu, buf_alloc_size());
      auto out = alloc_host_buffer(*fix->gpu, buf_alloc_size());

      auto sdma_done = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sdma_done);
      auto done = kfd::Signal::create(ctx);
      REQUIRE_RESULT(done);

      struct Args {
        unsigned *out;
        const unsigned *in;
      };

      constexpr uint32_t ITERATIONS = 50;
      for (uint32_t iter = 0; iter < ITERATIONS; ++iter) {
        INFO("iteration " << iter);

        if (iter > 0) {
          REQUIRE_RESULT(sdma_done->reset());
          REQUIRE_RESULT(done->reset());
        }

        REQUIRE_RESULT(fix->sdma.const_fill(in.data(), iter, BUF_BYTES));
        REQUIRE_RESULT(fix->sdma.signal(*sdma_done));

        REQUIRE_RESULT(fix->compute.wait(*sdma_done, kfd::Condition::EQ, 0));

        Args args{static_cast<unsigned *>(out.data()),
                  static_cast<const unsigned *>(in.data())};
        kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = THREADS}};
        auto kernarg = kernel->make_kernargs(*fix->gpu, args, cfg);
        REQUIRE_RESULT(kernarg);

        REQUIRE_RESULT(fix->compute.dispatch(*kernel, cfg, *kernarg));
        REQUIRE_RESULT(fix->compute.signal(*done));
        REQUIRE_RESULT(
            done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

        auto *vals = static_cast<const unsigned *>(out.data());
        for (uint32_t i = 0; i < THREADS; ++i) {
          if (vals[i] != iter + 1) {
            INFO("element " << i);
            REQUIRE(vals[i] == iter + 1);
          }
        }
      }
    }
  }
}
