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

void bump(void *data) {
  static_cast<std::atomic<int> *>(data)->fetch_add(1,
                                                   std::memory_order_acq_rel);
}

struct Rearm {
  kfd::Context *ctx;
  kfd::Signal *sig;
  std::atomic<int> count{0};
  uint64_t target;
};

void on_complete(void *data) {
  auto *r = static_cast<Rearm *>(data);
  r->count.fetch_add(1, std::memory_order_acq_rel);
  if (r->target > 0) {
    r->target -= 1;
    (void)r->ctx->register_handler(*r->sig, kfd::Condition::EQ, r->target,
                                   on_complete, r);
  }
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

TEST_CASE("Handler - callback re-registers itself to re-arm",
          "[device][handler]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      // Decrement once per signal.
      constexpr int N = 4;
      auto sig = kfd::Signal::create(ctx, N);
      REQUIRE_RESULT(sig);

      Rearm r{.ctx = &ctx, .sig = &*sig, .target = N - 1};
      REQUIRE_RESULT(ctx.register_handler(*sig, kfd::Condition::EQ, r.target,
                                          on_complete, &r));

      for (int i = 0; i < N; ++i) {
        INFO("signal " << i);
        REQUIRE_RESULT(queue->signal(*sig));
        CHECK(wait_for(r.count, i + 1) == i + 1);
      }
      CHECK(r.count.load() == N);
    }
  }
}
