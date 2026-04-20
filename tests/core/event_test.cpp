#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>

using kfd::test::require_ctx;

TEST_CASE("Event - creates and destroys cleanly", "[event]") {
  auto &ctx = require_ctx();

  auto ev = kfd::Event::create(ctx);
  REQUIRE_RESULT(ev);
  CHECK(static_cast<bool>(*ev));
  CHECK(ev->trigger_data() != 0);
}

TEST_CASE("Event - CPU signal + wait round-trip", "[event]") {
  auto &ctx = require_ctx();

  auto ev = kfd::Event::create(ctx);
  REQUIRE_RESULT(ev);

  REQUIRE_RESULT(ev->signal());
  REQUIRE_RESULT(ev->wait(1000));
}

TEST_CASE("Event - wait times out when unsignaled", "[event]") {
  auto &ctx = require_ctx();

  auto ev = kfd::Event::create(ctx);
  REQUIRE_RESULT(ev);

  auto r = ev->wait(10);
  CHECK(!r.has_value());
}

TEST_CASE("Event - GPU release_mem signals event via Signal",
          "[event][queue]") {
  auto &ctx = require_ctx();

  auto dev = ctx.device(0);
  REQUIRE_RESULT(dev);

  auto queue = kfd::ComputeQueue::create(**dev);
  REQUIRE_RESULT(queue);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);
  REQUIRE_RESULT(queue->signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}

TEST_CASE("Event - signal wait after write_data", "[event][queue]") {
  auto &gpu = kfd::test::require_gpu();
  auto &ctx = gpu.context();

  auto queue = kfd::ComputeQueue::create(gpu);
  REQUIRE_RESULT(queue);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  auto buf = kfd::test::alloc_host_buffer(gpu, 4096);
  auto *dst = static_cast<volatile uint32_t *>(buf.data());
  *dst = 0;

  REQUIRE_RESULT(queue->write_data(buf.data(), 0xDEADBEEF));
  REQUIRE_RESULT(queue->signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  CHECK(*dst == 0xDEADBEEF);
}

TEST_CASE("Event - multiple sequential submits", "[event][queue]") {
  auto &ctx = require_ctx();

  auto dev = ctx.device(0);
  REQUIRE_RESULT(dev);

  auto queue = kfd::ComputeQueue::create(**dev);
  REQUIRE_RESULT(queue);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  constexpr uint32_t N = 4;
  for (uint32_t i = 0; i < N; ++i) {
    if (i != 0)
      REQUIRE_RESULT(sig->reset());
    REQUIRE_RESULT(queue->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Event - repeated submit+wait cycles", "[event][queue]") {
  auto &ctx = require_ctx();

  auto dev = ctx.device(0);
  REQUIRE_RESULT(dev);

  auto queue = kfd::ComputeQueue::create(**dev);
  REQUIRE_RESULT(queue);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  for (uint32_t i = 1; i <= 8; ++i) {
    if (i != 1)
      REQUIRE_RESULT(sig->reset());
    REQUIRE_RESULT(queue->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Event - rapid GPU submit cycles", "[event][queue][stress]") {
  auto &ctx = require_ctx();

  auto dev = ctx.device(0);
  REQUIRE_RESULT(dev);

  auto queue = kfd::ComputeQueue::create(**dev);
  REQUIRE_RESULT(queue);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  constexpr uint32_t N = 50;
  for (uint32_t i = 0; i < N; ++i) {
    if (i != 0)
      REQUIRE_RESULT(sig->reset());
    REQUIRE_RESULT(queue->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Event - move semantics", "[event]") {
  auto &ctx = require_ctx();

  auto ev = kfd::Event::create(ctx);
  REQUIRE_RESULT(ev);
  uint32_t orig_id = ev->event_id();
  uint32_t orig_trigger = ev->trigger_data();

  kfd::Event moved(std::move(*ev));
  CHECK(!static_cast<bool>(*ev));
  CHECK(static_cast<bool>(moved));
  CHECK(moved.event_id() == orig_id);
  CHECK(moved.trigger_data() == orig_trigger);

  REQUIRE_RESULT(moved.signal());
  REQUIRE_RESULT(moved.wait(1000));
}
