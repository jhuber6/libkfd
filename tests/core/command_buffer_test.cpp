#include "test_helpers.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <thread>
#include <vector>

using kfd::test::alloc_host_buffer;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

TEST_CASE("CommandBuffer - batched writes submit atomically", "[command]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      constexpr uint32_t N = 16;
      auto buf = alloc_host_buffer(gpu, N * sizeof(uint32_t));
      auto *slots = static_cast<uint32_t *>(buf.data());
      std::memset(slots, 0xFF, N * sizeof(uint32_t));

      // One buffer, many packets, a single submit / doorbell ring.
      auto cmd = queue->command();
      for (uint32_t i = 0; i < N; ++i)
        cmd.write_data(slots + i, i + 1);
      cmd.signal(*sig);
      REQUIRE_RESULT(cmd.submit());

      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      for (uint32_t i = 0; i < N; ++i)
        CHECK(slots[i] == i + 1);
    }
  }
}

TEST_CASE("CommandBuffer - replay reissues the recorded packets", "[command]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);
      auto *counter = static_cast<uint32_t *>(buf.data());
      *counter = 0;

      auto cmd = queue->command();
      cmd.atomic_mem(kfd::pm4::ATOMIC_ADD_RTN_32, counter, 1).signal(*sig);

      constexpr uint32_t REPLAYS = 5;
      for (uint32_t i = 0; i < REPLAYS; ++i) {
        REQUIRE_RESULT(sig->reset());
        REQUIRE_RESULT(cmd.submit());
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }

      CHECK(*counter == REPLAYS);
    }
  }
}

TEST_CASE("CommandBuffer - reset allows reuse", "[command]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);
      auto *dst = static_cast<uint32_t *>(buf.data());
      *dst = 0;

      auto cmd = queue->command();
      cmd.write_data(dst, 0xAAAA'AAAA).signal(*sig);
      REQUIRE_RESULT(cmd.submit());
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      CHECK(*dst == 0xAAAA'AAAA);

      cmd.reset();
      CHECK(cmd.empty());
      REQUIRE_RESULT(sig->reset());
      cmd.write_data(dst, 0xBBBB'BBBB).signal(*sig);
      REQUIRE_RESULT(cmd.submit());
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      CHECK(*dst == 0xBBBB'BBBB);
    }
  }
}

TEST_CASE("CommandBuffer - empty submit is a no-op", "[command]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto cmd = queue->command();
      CHECK(cmd.empty());
      REQUIRE_RESULT(cmd.submit());
    }
  }
}

TEST_CASE("CommandBuffer - mixed packet kinds in one batch", "[command]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu, 2 * kfd::detail::page_size());
      auto *src = static_cast<uint32_t *>(buf.data());
      auto *dst = src + (kfd::detail::page_size() / sizeof(uint32_t));
      *src = 0;
      *dst = 0;

      auto cmd = queue->command();
      cmd.write_data(src, 0xC0FFEE00)
          .copy_data(kfd::pm4::COPY_SRC_TC_L2, reinterpret_cast<uintptr_t>(src),
                     kfd::pm4::COPY_DST_MEM, reinterpret_cast<uintptr_t>(dst))
          .signal(*sig);
      REQUIRE_RESULT(cmd.submit());

      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      CHECK(*src == 0xC0FFEE00);
      CHECK(*dst == 0xC0FFEE00);
    }
  }
}

TEST_CASE("CommandBuffer - concurrent recording on a shared queue",
          "[command][stress][mt]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      constexpr size_t N_THREADS = 4;
      constexpr size_t ITERS = 64;
      constexpr size_t PER = 8; // writes batched per submission

      auto buf = alloc_host_buffer(gpu, N_THREADS * PER * sizeof(uint32_t));
      auto *base = static_cast<uint32_t *>(buf.data());
      std::memset(base, 0, N_THREADS * PER * sizeof(uint32_t));

      std::vector<kfd::Signal> signals;
      signals.reserve(N_THREADS);
      for (size_t i = 0; i < N_THREADS; ++i) {
        auto s = kfd::Signal::create(ctx);
        REQUIRE_RESULT(s);
        signals.push_back(std::move(*s));
      }

      std::atomic<int> failures{0};
      auto worker = [&](size_t tid) {
        auto *slots = base + tid * PER;
        for (size_t j = 0; j < ITERS; ++j) {
          uint32_t seq = static_cast<uint32_t>(j + 1);
          if (!signals[tid].reset().has_value()) {
            failures.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          auto cmd = queue->command();
          for (size_t k = 0; k < PER; ++k)
            cmd.write_data(slots + k, seq);
          cmd.signal(signals[tid]);
          if (!cmd.submit().has_value()) {
            failures.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          if (!signals[tid]
                   .wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS)
                   .has_value()) {
            failures.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          for (size_t k = 0; k < PER; ++k)
            if (slots[k] != seq)
              failures.fetch_add(1, std::memory_order_relaxed);
        }
      };

      std::vector<std::thread> threads;
      threads.reserve(N_THREADS);
      for (size_t i = 0; i < N_THREADS; ++i)
        threads.emplace_back(worker, i);
      for (auto &t : threads)
        t.join();

      CHECK(failures.load() == 0);
    }
  }
}
