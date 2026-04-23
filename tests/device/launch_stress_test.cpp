#include "test_helpers.h"

#include "libkfd/abi.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

static const kfd::test::TestBinary dispatch_kernels[] = {
#include "dispatch_kernels.inc"
};

using kfd::test::DeviceFixture;
using kfd::test::make_device_fixture;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

namespace {

struct LaunchFixture : DeviceFixture {
  kfd::Kernel nop;
  kfd::Buffer nop_kernarg;
};

std::expected<LaunchFixture, kfd::Error> make_fixture(kfd::Device &dev) {
  auto base = KFD_TRY(make_device_fixture(dev, dispatch_kernels));
  auto nop = KFD_TRY(base.exe.kernel("nop.kd"));
  kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
  auto ka = KFD_TRY(nop.make_kernargs(*base.gpu, cfg));
  return LaunchFixture{std::move(base), std::move(nop), std::move(ka)};
}

void dispatch_nop(LaunchFixture &fix, kfd::Dim3 grid = {.x = 1},
                  kfd::Dim3 block = {.x = 64}) {
  kfd::DispatchConfig cfg{.grid = grid, .block = block};
  bool is_default = grid.x == 1 && grid.y == 1 && grid.z == 1 &&
                    block.x == 64 && block.y == 1 && block.z == 1;
  if (is_default) {
    REQUIRE_RESULT(fix.compute.dispatch(fix.nop, cfg, fix.nop_kernarg));
  } else {
    auto ka = fix.nop.make_kernargs(*fix.gpu, cfg);
    REQUIRE_RESULT(ka);
    REQUIRE_RESULT(fix.compute.dispatch(fix.nop, cfg, *ka));
  }
}

} // namespace

TEST_CASE("Launch - sequential dispatches with trailing fence",
          "[device][launch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_fixture(gpu);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      constexpr unsigned N = 100;
      for (unsigned i = 0; i < N; ++i) {
        INFO("dispatch " << i);
        dispatch_nop(*fix, {.x = 1}, {.x = 64});
      }

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);
      REQUIRE_RESULT(fix->compute.signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("Launch - sequential dispatches with per-dispatch signal",
          "[device][launch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_fixture(gpu);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      constexpr unsigned N = 64;
      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      for (unsigned i = 0; i < N; ++i) {
        INFO("dispatch " << i);
        if (i > 0)
          REQUIRE_RESULT(sig->reset());
        dispatch_nop(*fix, {.x = 1}, {.x = 64});
        REQUIRE_RESULT(fix->compute.signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }
    }
  }
}

TEST_CASE("Launch - block size sweep", "[device][launch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_fixture(gpu);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      static constexpr uint32_t sizes[] = {1,   32,  33,  64,  65,
                                           128, 256, 512, 1024};

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      bool first = true;
      for (uint32_t bs : sizes) {
        INFO("block_size " << bs);
        if (!first)
          REQUIRE_RESULT(sig->reset());
        first = false;
        dispatch_nop(*fix, {.x = 4}, {.x = bs});
        REQUIRE_RESULT(fix->compute.signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }
    }
  }
}

TEST_CASE("Launch - large grid", "[device][launch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_fixture(gpu);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      dispatch_nop(*fix, {.x = 4096}, {.x = 256});

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);
      REQUIRE_RESULT(fix->compute.signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("Launch - two independent compute queues", "[device][launch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_fixture(gpu);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto compute2 = kfd::ComputeQueue::create(*fix->gpu);
      REQUIRE_RESULT(compute2);

      kfd::DispatchConfig cfg{.grid = {.x = 4}, .block = {.x = 64}};
      auto kernarg = fix->nop.make_kernargs(*fix->gpu, cfg);
      REQUIRE_RESULT(kernarg);

      constexpr unsigned N = 32;
      for (unsigned i = 0; i < N; ++i) {
        REQUIRE_RESULT(fix->compute.dispatch(fix->nop, cfg, *kernarg));
        REQUIRE_RESULT(compute2->dispatch(fix->nop, cfg, *kernarg));
      }

      auto sig1 = kfd::Signal::create(ctx);
      auto sig2 = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig1);
      REQUIRE_RESULT(sig2);

      REQUIRE_RESULT(fix->compute.signal(*sig1));
      REQUIRE_RESULT(compute2->signal(*sig2));
      REQUIRE_RESULT(
          sig1->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      REQUIRE_RESULT(
          sig2->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("Launch - shared queue with per-thread signals", "[device][launch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_fixture(gpu);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      constexpr unsigned NUM_THREADS = 4;
      constexpr unsigned DISPATCHES_PER_THREAD = 16;

      std::vector<std::expected<kfd::Signal, kfd::Error>> sigs;
      sigs.reserve(NUM_THREADS);
      for (unsigned i = 0; i < NUM_THREADS; ++i) {
        sigs.push_back(kfd::Signal::create(ctx));
        REQUIRE_RESULT(sigs.back());
      }

      kfd::DispatchConfig cfg{.grid = {.x = 4}, .block = {.x = 64}};
      auto kernarg = fix->nop.make_kernargs(*fix->gpu, cfg);
      REQUIRE_RESULT(kernarg);

      std::vector<std::thread> threads;
      threads.reserve(NUM_THREADS);
      std::atomic<unsigned> failures{0};

      for (unsigned t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t] {
          for (unsigned i = 0; i < DISPATCHES_PER_THREAD; ++i) {
            if (!fix->compute.dispatch(fix->nop, cfg, *kernarg)) {
              failures.fetch_add(1, std::memory_order_relaxed);
              return;
            }
          }
          if (!fix->compute.signal(*sigs[t])) {
            failures.fetch_add(1, std::memory_order_relaxed);
            return;
          }
        });
      }

      for (auto &th : threads)
        th.join();

      REQUIRE(failures.load() == 0);

      for (unsigned t = 0; t < NUM_THREADS; ++t) {
        INFO("waiting on signal " << t);
        REQUIRE_RESULT(
            sigs[t]->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }
    }
  }
}

TEST_CASE("Launch - ring buffer saturation", "[device][launch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_fixture(gpu);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      // Default ring = 4 pages = 4096 dwords. A NOP dispatch is ~50-80 dwords,
      // so 500 dispatches forces the ring to wrap many times.
      constexpr unsigned N = 500;
      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      for (unsigned i = 0; i < N; ++i) {
        dispatch_nop(*fix, {.x = 1}, {.x = 64});

        // Periodic fence so the GPU drains and frees ring space.
        if ((i & 63) == 63) {
          REQUIRE_RESULT(sig->reset());
          REQUIRE_RESULT(fix->compute.signal(*sig));
          REQUIRE_RESULT(
              sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
        }
      }

      REQUIRE_RESULT(sig->reset());
      REQUIRE_RESULT(fix->compute.signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}
