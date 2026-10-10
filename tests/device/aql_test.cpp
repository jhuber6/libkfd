#include "test_helpers.h"

#include <bit>
#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

static const kfd::test::TestBinary dispatch_kernels[] = {
#include "dispatch_kernels.inc"
};

using kfd::test::alloc_host_buffer;
using kfd::test::create_queue;
using kfd::test::make_device_fixture;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

namespace {

struct Args {
  unsigned *out;
};

// Invalidates caches before the kernel and makes its writes visible to the
// host before signalling completion.
constexpr kfd::aql::Ordering RELEASE = {
    .acquire = kfd::aql::FENCE_SCOPE_SYSTEM,
    .release = kfd::aql::FENCE_SCOPE_SYSTEM,
};

} // namespace

TEST_CASE("AQL - NOP kernel completes", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      auto kernel = fix->exe.kernel("nop.kd");
      REQUIRE_RESULT(kernel);

      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, cfg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(q->dispatch(*kernel, cfg, *kernarg, RELEASE, &*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("AQL - load via PM4 indirect buffer", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      auto buf = kfd::test::read_file(fix->bin->path);
      auto loaded = kfd::Executable::load(gpu, buf, *q, fix->sdma);
      REQUIRE_RESULT(loaded);

      auto kernel = loaded->kernel("nop.kd");
      REQUIRE_RESULT(kernel);

      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, cfg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(q->dispatch(*kernel, cfg, *kernarg, RELEASE, &*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("AQL - store writes sentinel", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      auto kernel = fix->exe.kernel("store.kd");
      REQUIRE_RESULT(kernel);

      auto out = alloc_host_buffer(gpu);
      std::memset(out.data(), 0, sizeof(unsigned));

      Args args{static_cast<unsigned *>(out.data())};
      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(q->dispatch(*kernel, cfg, *kernarg, RELEASE, &*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      unsigned val;
      std::memcpy(&val, out.data(), sizeof(val));
      CHECK(val == 0xCAFEBABE);
    }
  }
}

TEST_CASE("AQL - multi-dimensional launch reports dimensions",
          "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      auto kernel = fix->exe.kernel("check_dims.kd");
      REQUIRE_RESULT(kernel);

      auto out = alloc_host_buffer(gpu);
      std::memset(out.data(), 0, 6 * sizeof(unsigned));

      Args args{static_cast<unsigned *>(out.data())};
      kfd::DispatchConfig cfg{
          .grid = {.x = 3, .y = 5, .z = 7},
          .block = {.x = 4, .y = 2, .z = 8},
      };
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      REQUIRE_RESULT(q->dispatch(*kernel, cfg, *kernarg, RELEASE, &*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      unsigned vals[6];
      std::memcpy(vals, out.data(), sizeof(vals));
      CHECK(vals[0] == 3);
      CHECK(vals[1] == 5);
      CHECK(vals[2] == 7);
      CHECK(vals[3] == 4);
      CHECK(vals[4] == 2);
      CHECK(vals[5] == 8);
    }
  }
}

TEST_CASE("AQL - ring wraps many times", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      constexpr size_t RING_PACKETS = 16;
      auto q = create_queue<kfd::AQLQueue>(gpu, RING_PACKETS *
                                                    kfd::AQLQueue::PACKET_SIZE);
      REQUIRE_RESULT(q);

      auto kernel = fix->exe.kernel("nop.kd");
      REQUIRE_RESULT(kernel);

      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, cfg);

      constexpr unsigned LAUNCHES = 1000;
      for (unsigned i = 0; i < LAUNCHES; ++i)
        REQUIRE_RESULT(q->dispatch(*kernel, cfg, *kernarg, {}, nullptr));

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);
      REQUIRE_RESULT(q->barrier({}, {.barrier = true}, &*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("AQL - a batch rings the doorbell once", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      constexpr size_t RING_PACKETS = 16;
      auto q = create_queue<kfd::AQLQueue>(gpu, RING_PACKETS *
                                                    kfd::AQLQueue::PACKET_SIZE);
      REQUIRE_RESULT(q);

      auto sig = kfd::Signal::create(ctx, 2);
      REQUIRE_RESULT(sig);
      CHECK(sig->completion_handle() + 8 ==
            reinterpret_cast<uintptr_t>(sig->fence_addr()));

      auto cmd = q->command();
      cmd.barrier({}, {}, &*sig);
      cmd.barrier({}, {}, &*sig);
      CHECK(cmd.size() == 2);
      REQUIRE_RESULT(cmd.submit());
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      REQUIRE_RESULT(sig->reset(2));
      REQUIRE_RESULT(cmd.submit());
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      std::byte one[32] = {};
      auto partial = q->submit(std::span<const std::byte>(one, sizeof(one)));
      REQUIRE(!partial);
      CHECK(partial.error().code == EINVAL);

      std::byte overflow[RING_PACKETS * kfd::AQLQueue::PACKET_SIZE +
                         kfd::AQLQueue::PACKET_SIZE] = {};
      auto big = q->submit(overflow);
      REQUIRE(!big);
      CHECK(big.error().code == EMSGSIZE);
    }
  }
}

TEST_CASE("AQL - barrier waits on a host signal", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      auto gate = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(gate);
      auto done = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(done);

      kfd::Signal *deps[] = {&*gate};
      REQUIRE_RESULT(q->barrier(deps, {}, &*done));

      // The queue must remain blocked while the gate is held.
      auto r = done->wait(kfd::Condition::EQ, 0, 50'000'000);
      REQUIRE(!r);
      CHECK(r.error().code == ETIMEDOUT);
      CHECK(done->value() == 1);

      REQUIRE_RESULT(gate->reset(0));
      REQUIRE_RESULT(
          done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("AQL - cross-queue dependency orders work", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto producer = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(producer);
      auto consumer = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(consumer);

      auto kernel = fix->exe.kernel("store.kd");
      REQUIRE_RESULT(kernel);

      auto out = alloc_host_buffer(gpu);
      std::memset(out.data(), 0, sizeof(unsigned));
      Args args{static_cast<unsigned *>(out.data())};
      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

      auto gate = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(gate);
      auto stored = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(stored);
      auto done = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(done);

      kfd::Signal *consumer_deps[] = {&*stored};
      REQUIRE_RESULT(consumer->barrier(consumer_deps, {}, &*done));

      kfd::Signal *producer_deps[] = {&*gate};
      REQUIRE_RESULT(producer->barrier(producer_deps, {}, nullptr));
      REQUIRE_RESULT(
          producer->dispatch(*kernel, cfg, *kernarg, RELEASE, &*stored));

      CHECK(done->value() == 1);
      REQUIRE_RESULT(gate->reset(0));
      REQUIRE_RESULT(
          done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      unsigned val;
      std::memcpy(&val, out.data(), sizeof(val));
      CHECK(val == 0xCAFEBABE);
    }
  }
}

TEST_CASE("AQL - CP-requested scratch grows", "[device][aql][scratch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      auto kernel = fix->exe.kernel("verify_scratch.kd");
      REQUIRE_RESULT(kernel);

      constexpr uint32_t THREADS = 64;
      constexpr size_t out_bytes = THREADS * sizeof(unsigned);
      auto out = alloc_host_buffer(gpu);

      for (uint32_t pss : {256u, 512u, 1024u, 2048u, 512u}) {
        INFO("private_segment_size = " << pss);
        std::memset(out.data(), 0xFF, out_bytes);

        Args args{static_cast<unsigned *>(out.data())};
        kfd::DispatchConfig cfg{
            .grid = {.x = 1},
            .block = {.x = THREADS},
            .private_segment_size = pss,
        };
        auto kernarg = kernel->alloc();
        REQUIRE_RESULT(kernarg);
        kernel->fill(*kernarg, args, cfg);

        auto sig = kfd::Signal::create(ctx);
        REQUIRE_RESULT(sig);
        REQUIRE_RESULT(q->dispatch(*kernel, cfg, *kernarg, RELEASE, &*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

        auto *vals = static_cast<const unsigned *>(out.data());
        for (uint32_t t = 0; t < THREADS; ++t) {
          INFO("thread " << t);
          CHECK(vals[t] == 16 * t + 120);
        }
      }
    }
  }
}

TEST_CASE("AQL - concurrent producers share one queue", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto q =
          create_queue<kfd::AQLQueue>(gpu, 32 * kfd::AQLQueue::PACKET_SIZE);
      REQUIRE_RESULT(q);

      auto kernel = fix->exe.kernel("nop.kd");
      REQUIRE_RESULT(kernel);

      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, cfg);

      constexpr uint64_t THREADS = 4;
      constexpr uint64_t PER_THREAD = 500;
      auto sig = kfd::Signal::create(ctx, THREADS * PER_THREAD);
      REQUIRE_RESULT(sig);

      std::vector<unsigned> failures(THREADS, 0);
      std::vector<std::thread> threads;
      threads.reserve(THREADS);
      for (unsigned t = 0; t < THREADS; ++t) {
        threads.emplace_back([&, t] {
          for (unsigned i = 0; i < PER_THREAD; ++i)
            if (!q->dispatch(*kernel, cfg, *kernarg, RELEASE, &*sig))
              ++failures[t];
        });
      }
      for (auto &th : threads)
        th.join();

      for (unsigned t = 0; t < THREADS; ++t)
        CHECK(failures[t] == 0);
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("AQL - scratch grows under concurrent producers",
          "[device][aql][scratch]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      constexpr size_t QUEUES = 2;
      std::vector<kfd::AQLQueue> queues;
      for (size_t i = 0; i < QUEUES; ++i) {
        auto q =
            create_queue<kfd::AQLQueue>(gpu, 16 * kfd::AQLQueue::PACKET_SIZE);
        REQUIRE_RESULT(q);
        queues.push_back(std::move(*q));
      }

      auto kernel = fix->exe.kernel("verify_scratch.kd");
      REQUIRE_RESULT(kernel);

      constexpr uint32_t THREADS = 8;
      constexpr uint32_t ITERS = 50;
      constexpr uint32_t LANES = 64;
      std::vector<kfd::Buffer> outs;
      for (uint32_t t = 0; t < THREADS; ++t)
        outs.push_back(alloc_host_buffer(gpu));

      std::vector<unsigned> failures(THREADS, 0);
      std::vector<std::thread> threads;
      threads.reserve(THREADS);
      for (uint32_t t = 0; t < THREADS; ++t) {
        threads.emplace_back([&, t] {
          kfd::AQLQueue &q = queues[t % QUEUES];
          auto *vals = static_cast<unsigned *>(outs[t].data());
          auto sig = kfd::Signal::create(ctx, 1);
          auto kernarg = kernel->alloc();
          if (!sig || !kernarg) {
            ++failures[t];
            return;
          }
          for (uint32_t i = 0; i < ITERS; ++i) {
            std::memset(vals, 0xFF, LANES * sizeof(unsigned));
            Args args{vals};
            kfd::DispatchConfig cfg{
                .grid = {.x = 1},
                .block = {.x = LANES},
                .private_segment_size = 128u << ((t + i) % 5),
            };
            kernel->fill(*kernarg, args, cfg);
            if (!q.dispatch(*kernel, cfg, *kernarg, RELEASE, &*sig) ||
                !sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS) ||
                !sig->reset(1)) {
              ++failures[t];
              return;
            }
            for (uint32_t l = 0; l < LANES; ++l)
              if (vals[l] != 16 * l + 120)
                ++failures[t];
          }
        });
      }
      for (auto &th : threads)
        th.join();

      for (uint32_t t = 0; t < THREADS; ++t) {
        INFO("thread " << t);
        CHECK(failures[t] == 0);
      }
    }
  }
}

TEST_CASE("AQL - CU mask restricts dispatch", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      const auto &pr = gpu.properties();
      uint32_t cu_count = pr.simd_count / (pr.simd_per_cu ? pr.simd_per_cu : 1);
      uint32_t set_bits = 0;
      for (uint32_t word : q->cu_mask())
        set_bits += static_cast<uint32_t>(std::popcount(word));
      CHECK(set_bits == cu_count);

      CHECK_FALSE(q->set_cu_mask({}).has_value());
      // GFX10+ requires CUs to be enabled in WGP pairs.
      uint32_t mask = 0x3;
      REQUIRE_RESULT(q->set_cu_mask({&mask, 1}));
      REQUIRE(q->cu_mask().size() == 1);
      CHECK(q->cu_mask()[0] == mask);

      auto kernel = fix->exe.kernel("store.kd");
      REQUIRE_RESULT(kernel);
      auto out = alloc_host_buffer(gpu);
      std::memset(out.data(), 0, sizeof(unsigned));
      Args args{static_cast<unsigned *>(out.data())};
      kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, args, cfg);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);
      REQUIRE_RESULT(q->dispatch(*kernel, cfg, *kernarg, RELEASE, &*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      unsigned val;
      std::memcpy(&val, out.data(), sizeof(val));
      CHECK(val == 0xCAFEBABE);
    }
  }
}

TEST_CASE("AQL - barrier waits on more signals than one packet holds",
          "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      constexpr size_t DEPS = 7;
      std::vector<kfd::Signal> gates;
      std::vector<kfd::Signal *> deps;
      for (size_t i = 0; i < DEPS; ++i) {
        auto g = kfd::Signal::create(ctx, 1);
        REQUIRE_RESULT(g);
        gates.push_back(std::move(*g));
      }
      for (auto &g : gates)
        deps.push_back(&g);
      auto done = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(done);

      REQUIRE_RESULT(q->barrier(deps, {}, &*done));
      for (size_t i = 0; i + 1 < DEPS; ++i)
        REQUIRE_RESULT(gates[i].reset(0));

      auto r = done->wait(kfd::Condition::EQ, 0, 50'000'000);
      REQUIRE(!r);
      CHECK(r.error().code == ETIMEDOUT);

      REQUIRE_RESULT(gates.back().reset(0));
      REQUIRE_RESULT(
          done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("AQL - completion wakes a sleeping waiter", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      auto gate = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(gate);
      auto done = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(done);

      kfd::Signal *deps[] = {&*gate};
      REQUIRE_RESULT(q->barrier(deps, {}, &*done));

      // Without the event, the waiter only notices at its timeout.
      bool ok = false;
      auto elapsed = std::chrono::steady_clock::duration::max();
      std::thread waiter([&] {
        auto start = std::chrono::steady_clock::now();
        ok = done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS,
                        /*spin_ns=*/0)
                 .has_value();
        elapsed = std::chrono::steady_clock::now() - start;
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      REQUIRE_RESULT(gate->reset(0));
      waiter.join();

      CHECK(ok);
      CHECK(elapsed < std::chrono::seconds(1));
    }
  }
}

TEST_CASE("AQL - signals are shared with PM4 queues", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto aql = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(aql);
      auto pm4 = create_queue<kfd::ComputeQueue>(gpu);
      REQUIRE_RESULT(pm4);

      auto gate = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(gate);
      auto to_aql = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(to_aql);
      auto to_pm4 = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(to_pm4);
      auto done = kfd::Signal::create(ctx, 1);
      REQUIRE_RESULT(done);

      REQUIRE_RESULT(pm4->wait(*gate, kfd::Condition::EQ, 0));
      REQUIRE_RESULT(pm4->signal(*to_aql));
      kfd::Signal *deps[] = {&*to_aql};
      REQUIRE_RESULT(aql->barrier(deps, {}, &*to_pm4));
      REQUIRE_RESULT(pm4->wait(*to_pm4, kfd::Condition::EQ, 0));
      REQUIRE_RESULT(pm4->signal(*done));

      CHECK(done->value() == 1);
      REQUIRE_RESULT(gate->reset(0));
      REQUIRE_RESULT(
          done->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      CHECK(to_aql->value() == 0);
      CHECK(to_pm4->value() == 0);
    }
  }
}

TEST_CASE("AQL - sub-grid launches are rejected", "[device][aql]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, dispatch_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto q = create_queue<kfd::AQLQueue>(gpu);
      REQUIRE_RESULT(q);

      auto kernel = fix->exe.kernel("nop.kd");
      REQUIRE_RESULT(kernel);

      kfd::DispatchConfig cfg{
          .grid = {.x = 4},
          .block = {.x = 64},
          .grid_start = {.x = 1},
      };
      auto kernarg = kernel->alloc();
      REQUIRE_RESULT(kernarg);
      kernel->fill(*kernarg, cfg);

      auto r = q->dispatch(*kernel, cfg, *kernarg, {}, nullptr);
      REQUIRE(!r);
      CHECK(r.error().code == ENOTSUP);
    }
  }
}
