#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <utility>

using kfd::test::require_ctx;

TEST_CASE("Signal - completion handle addresses the fence word",
          "[device][signal]") {
  auto &ctx = require_ctx();

  auto sig = kfd::Signal::create(ctx, 1);
  REQUIRE_RESULT(sig);
  uint64_t handle = sig->completion_handle();

  CHECK(sig->value() == 1);
  CHECK(handle + 8 == reinterpret_cast<uintptr_t>(sig->fence_addr()));
  CHECK(sig->event_id() != 0);
  CHECK(sig->signal_addr() != nullptr);
  CHECK(kfd::Signal::from_handle(handle) == &*sig);

  kfd::Signal moved = std::move(*sig);
  CHECK(moved.completion_handle() == handle);
  CHECK(moved.value() == 1);
  CHECK(!static_cast<bool>(*sig));
  CHECK(kfd::Signal::from_handle(handle) == &moved);
}

TEST_CASE("Signal - destruction clears the signal kind", "[device][signal]") {
  auto &ctx = require_ctx();

  uint64_t handle = 0;
  {
    auto sig = kfd::Signal::create(ctx, 1);
    REQUIRE_RESULT(sig);
    handle = sig->completion_handle();
    CHECK(__atomic_load_n(reinterpret_cast<int64_t *>(handle),
                          __ATOMIC_RELAXED) == 1);
  }
  CHECK(__atomic_load_n(reinterpret_cast<int64_t *>(handle),
                        __ATOMIC_RELAXED) == 0);
  CHECK(kfd::Signal::from_handle(handle) == nullptr);
}
