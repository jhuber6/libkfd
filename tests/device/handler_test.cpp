#include "test_helpers.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <thread>

using kfd::test::require_ctx;
using kfd::test::require_gpu;

namespace {

int wait_for(std::atomic<int> &counter, int target) {
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::nanoseconds(kfd::test::WAIT_TIMEOUT_NS);
  while (counter.load(std::memory_order_acquire) < target &&
         std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  return counter.load(std::memory_order_acquire);
}

bool bump(void *data) {
  static_cast<std::atomic<int> *>(data)->fetch_add(1,
                                                   std::memory_order_acq_rel);
  return false;
}

struct Rearm {
  std::atomic<int> count{0};
  int limit;
};

bool on_complete(void *data) {
  auto *r = static_cast<Rearm *>(data);
  int fired = r->count.fetch_add(1, std::memory_order_acq_rel) + 1;
  return fired < r->limit;
}

} // namespace

TEST_CASE("Handler - one-shot fires once and is removed", "[device][handler]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);
      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      std::atomic<int> count{0};
      REQUIRE_RESULT(
          ctx.register_handler(*sig, kfd::Condition::EQ, 0, bump, &count));

      REQUIRE_RESULT(queue->signal(*sig));
      CHECK(wait_for(count, 1) == 1);

      // A second completion must not re-invoke the one-shot handler.
      REQUIRE_RESULT(sig->reset());
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      CHECK(count.load() == 1);
    }
  }
}

TEST_CASE("Handler - callback stays armed by returning true",
          "[device][handler]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      constexpr int N = 4;
      Rearm r{.limit = N};
      REQUIRE_RESULT(
          ctx.register_handler(*sig, kfd::Condition::EQ, 0, on_complete, &r));

      for (int i = 0; i < N; ++i) {
        INFO("signal " << i);
        REQUIRE_RESULT(queue->signal(*sig));
        CHECK(wait_for(r.count, i + 1) == i + 1);
        REQUIRE_RESULT(sig->reset());
      }
      CHECK(r.count.load() == N);

      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      CHECK(r.count.load() == N);
    }
  }
}
