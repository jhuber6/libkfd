#include "test_helpers.h"

#include "libkfd/detail/utility.h"
#include "libkfd/packets/pm4.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <ctime>

using kfd::test::alloc_host_buffer;
using kfd::test::create_queue;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

namespace {

namespace pm4 = kfd::pm4;

constexpr uint32_t SENTINEL = 0xFFFFFFFFu;

enum Slot : uint32_t { GATE = 0, GATE_HI = 1, TARGET = 2 };

inline void sleep_ms(long ms) {
  struct timespec ts = {.tv_sec = ms / 1000,
                        .tv_nsec = (ms % 1000) * 1'000'000};
  ::nanosleep(&ts, nullptr);
}

} // namespace

TEST_CASE("PM4 - INDIRECT_BUFFER executes a staged packet", "[device][pm4]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto q = create_queue<kfd::ComputeQueue>(gpu);
      REQUIRE_RESULT(q);
      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto ctrl = alloc_host_buffer(gpu);
      auto ibbuf = alloc_host_buffer(gpu);
      auto *c = static_cast<volatile uint32_t *>(ctrl.data());
      auto *ib = static_cast<uint32_t *>(ibbuf.data());

      c[TARGET] = 0;
      uint32_t n =
          pm4::write_data(ib, const_cast<uint32_t *>(&c[TARGET]), 0xABCD1234u);

      REQUIRE_RESULT(q->indirect_buffer(ib, n));
      REQUIRE_RESULT(q->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      CHECK(c[TARGET] == 0xABCD1234u);
    }
  }
}

TEST_CASE("PM4 - COND_EXEC gates the following packets", "[device][pm4]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto q = create_queue<kfd::ComputeQueue>(gpu);
      REQUIRE_RESULT(q);
      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto ctrl = alloc_host_buffer(gpu);
      auto ibbuf = alloc_host_buffer(gpu);
      auto *c = static_cast<volatile uint32_t *>(ctrl.data());
      auto *ib = static_cast<uint32_t *>(ibbuf.data());

      // COND_EXEC(gate) followed by a gated write. The write is skipped when
      // the gate reads zero, executed otherwise.
      uint32_t n = 0;
      n += pm4::cond_exec(ib + n, const_cast<uint32_t *>(&c[GATE]),
                          pm4::WRITE_DATA_DWORDS);
      n += pm4::write_data(ib + n, const_cast<uint32_t *>(&c[TARGET]), 0xC0DEu);

      auto run = [&](uint32_t gate) -> bool {
        c[GATE] = gate;
        c[TARGET] = SENTINEL;
        if (auto r = sig->reset(); !r)
          return false;
        if (auto r = q->indirect_buffer(ib, n); !r)
          return false;
        if (auto r = q->signal(*sig); !r)
          return false;
        return sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS)
            .has_value();
      };

      INFO("gate = 0 must skip the write");
      REQUIRE(run(0));
      CHECK(c[TARGET] == SENTINEL);

      INFO("gate = 1 must execute the write");
      REQUIRE(run(1));
      CHECK(c[TARGET] == 0xC0DEu);
    }
  }
}

TEST_CASE("PM4 - COND_WRITE writes only when the compare holds",
          "[device][pm4]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto q = create_queue<kfd::ComputeQueue>(gpu);
      REQUIRE_RESULT(q);
      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto ctrl = alloc_host_buffer(gpu);
      auto ibbuf = alloc_host_buffer(gpu);
      auto *c = static_cast<volatile uint32_t *>(ctrl.data());
      auto *ib = static_cast<uint32_t *>(ibbuf.data());

      uint32_t n = pm4::cond_write(ib, kfd::Condition::EQ,
                                   const_cast<uint32_t *>(&c[GATE]),
                                   /*reference=*/0x55u, /*mask=*/0xFFFFFFFFu,
                                   const_cast<uint32_t *>(&c[TARGET]),
                                   /*write_data=*/0xBEEFu);

      auto run = [&](uint32_t gate) -> bool {
        c[GATE] = gate;
        c[TARGET] = 0;
        if (auto r = sig->reset(); !r)
          return false;
        if (auto r = q->indirect_buffer(ib, n); !r)
          return false;
        if (auto r = q->signal(*sig); !r)
          return false;
        return sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS)
            .has_value();
      };

      INFO("gate == reference must write");
      REQUIRE(run(0x55u));
      CHECK(c[TARGET] == 0xBEEFu);

      INFO("gate != reference must not write");
      REQUIRE(run(0x11u));
      CHECK(c[TARGET] == 0u);
    }
  }
}

TEST_CASE("PM4 - WAIT_REG_MEM64 waits on the full 64-bit value",
          "[device][pm4]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto q = create_queue<kfd::ComputeQueue>(gpu);
      REQUIRE_RESULT(q);
      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto ctrl = alloc_host_buffer(gpu);
      auto *base = static_cast<volatile uint32_t *>(ctrl.data());
      auto *gate = reinterpret_cast<volatile uint64_t *>(ctrl.data());
      volatile uint32_t *started = base + 2;
      volatile uint32_t *done = base + 3;

      constexpr uint32_t MAGIC = 0x5AFE'6402u;
      constexpr uint64_t REFERENCE = uint64_t(1) << 32; // 2^32
      constexpr uint64_t BELOW = REFERENCE - 1;         // 0xFFFF'FFFF

      *gate = BELOW;
      *started = 0;
      *done = 0;
      kfd::detail::memory_barrier();

      auto cmd = q->command();
      cmd.write_data(const_cast<uint32_t *>(started), MAGIC)
          .wait_reg_mem(const_cast<uint64_t *>(gate), kfd::Condition::GTE,
                        REFERENCE)
          .write_data(const_cast<uint32_t *>(done), MAGIC)
          .signal(*sig);
      REQUIRE_RESULT(cmd.submit());

      sleep_ms(200);
      CHECK(*started == MAGIC);
      CHECK(*done == 0u);

      *gate = REFERENCE;
      kfd::detail::memory_barrier();

      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      CHECK(*done == MAGIC);
    }
  }
}
