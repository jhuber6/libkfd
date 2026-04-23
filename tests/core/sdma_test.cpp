#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>

using kfd::test::alloc_host_buffer;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

TEST_CASE("SDMA - queue creates and destroys cleanly", "[sdma]") {
  auto &ctx = require_ctx();

  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto dev = ctx.device(di);
      REQUIRE_RESULT(dev);

      auto queue = kfd::test::create_queue<kfd::SDMAQueue>(**dev);
      REQUIRE_RESULT(queue);
      CHECK(queue->queue_id() >= 0);
      CHECK(queue->ring_dwords() > 0);
    }
  }
}

TEST_CASE("SDMA - simple fence", "[sdma]") {
  auto &ctx = require_ctx();

  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto dev = ctx.device(di);
      REQUIRE_RESULT(dev);

      auto queue = kfd::test::create_queue<kfd::SDMAQueue>(**dev);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("SDMA - const fill", "[sdma]") {
  auto &ctx = require_ctx();

  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::test::create_queue<kfd::SDMAQueue>(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);
      std::memset(buf.data(), 0, buf.size());

      auto *dst = static_cast<volatile uint32_t *>(buf.data());

      REQUIRE_RESULT(
          queue->const_fill(buf.data(), 0xDEADBEEF, 256 * sizeof(uint32_t)));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      for (uint32_t i = 0; i < 256; ++i)
        CHECK(dst[i] == 0xDEADBEEF);
    }
  }
}

TEST_CASE("SDMA - copy linear", "[sdma]") {
  auto &ctx = require_ctx();

  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::test::create_queue<kfd::SDMAQueue>(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      const size_t BUF_BYTES = kfd::detail::page_size();
      const size_t N = BUF_BYTES / sizeof(uint32_t);

      auto src_buf = alloc_host_buffer(gpu, BUF_BYTES);
      auto dst_buf = alloc_host_buffer(gpu, BUF_BYTES);

      auto *src = static_cast<volatile uint32_t *>(src_buf.data());
      auto *dst = static_cast<volatile uint32_t *>(dst_buf.data());
      for (uint32_t i = 0; i < N; ++i)
        src[i] = i;
      std::memset(const_cast<void *>(static_cast<volatile void *>(dst)), 0xFF,
                  BUF_BYTES);

      REQUIRE_RESULT(
          queue->copy_linear(dst_buf.data(), src_buf.data(), BUF_BYTES));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      for (uint32_t i = 0; i < N; ++i)
        CHECK(dst[i] == i);
    }
  }
}

TEST_CASE("SDMA - fill then copy back", "[sdma]") {
  auto &ctx = require_ctx();

  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::test::create_queue<kfd::SDMAQueue>(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      constexpr uint32_t FILL_DWORDS = 128;
      constexpr uint32_t FILL_BYTES = FILL_DWORDS * sizeof(uint32_t);

      auto a = alloc_host_buffer(gpu);
      auto b = alloc_host_buffer(gpu);

      std::memset(a.data(), 0, a.size());
      std::memset(b.data(), 0, b.size());

      {
        REQUIRE_RESULT(queue->const_fill(a.data(), 0xCAFEBABE, FILL_BYTES));
        REQUIRE_RESULT(queue->signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }

      {
        REQUIRE_RESULT(sig->reset());
        REQUIRE_RESULT(queue->copy_linear(b.data(), a.data(), FILL_BYTES));
        REQUIRE_RESULT(queue->signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }

      auto *out = static_cast<volatile uint32_t *>(b.data());
      for (uint32_t i = 0; i < FILL_DWORDS; ++i)
        CHECK(out[i] == 0xCAFEBABE);
    }
  }
}

TEST_CASE("SDMA - multiple submissions across ring wrap", "[sdma][stress]") {
  auto &ctx = require_ctx();

  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::test::create_queue<kfd::SDMAQueue>(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);

      auto *dst = static_cast<volatile uint32_t *>(buf.data());

      constexpr uint32_t pkt_dwords =
          kfd::sdma::CONST_FILL_DWORDS + kfd::sdma::ATOMIC_DWORDS +
          kfd::sdma::FENCE_DWORDS + kfd::sdma::TRAP_DWORDS;
      uint32_t n_pkts =
          static_cast<uint32_t>(queue->ring_dwords() / pkt_dwords) + 1;

      for (uint32_t i = 0; i < n_pkts; ++i) {
        dst[0] = 0;

        if (i != 0)
          REQUIRE_RESULT(sig->reset());

        REQUIRE_RESULT(queue->const_fill(buf.data(), i + 1, sizeof(uint32_t)));
        REQUIRE_RESULT(queue->signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
        REQUIRE(dst[0] == i + 1);
      }
    }
  }
}
