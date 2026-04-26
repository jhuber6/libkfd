#include "test_helpers.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <ctime>

#include <thread>
#include <vector>

static bool poll_value(uint32_t *addr, uint32_t expected,
                       uint32_t timeout_us = 5'000'000) {
  struct timespec now;
  ::clock_gettime(CLOCK_MONOTONIC, &now);
  uint64_t deadline_ns = static_cast<uint64_t>(now.tv_sec) * 1'000'000'000 +
                         static_cast<uint64_t>(now.tv_nsec) +
                         static_cast<uint64_t>(timeout_us) * 1'000;
  for (;;) {
    if (__atomic_load_n(addr, __ATOMIC_ACQUIRE) == expected)
      return true;
    ::clock_gettime(CLOCK_MONOTONIC, &now);
    uint64_t now_ns = static_cast<uint64_t>(now.tv_sec) * 1'000'000'000 +
                      static_cast<uint64_t>(now.tv_nsec);
    if (now_ns >= deadline_ns)
      return false;
    kfd::detail::spin_hint();
  }
}

using kfd::test::alloc_host_buffer;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

TEST_CASE("Queue - non-power-of-two ring size is rejected", "[queue]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto q = kfd::ComputeQueue::create(gpu, 3 * kfd::detail::page_size());
      REQUIRE_FALSE(q.has_value());
      CHECK(q.error().code == EINVAL);
    }
  }
}

TEST_CASE("Queue - creates and destroys cleanly", "[queue]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);
      CHECK(queue->queue_id() >= 0);
      CHECK(queue->ring_dwords() > 0);
    }
  }
}

TEST_CASE("Queue - simple fence submit", "[queue]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);
      REQUIRE_RESULT(queue->signal(*sig));
      CHECK_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("Queue - WRITE_DATA with fence", "[queue]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);
      std::memset(buf.data(), 0xFF, buf.size());

      auto *dst = static_cast<volatile uint32_t *>(buf.data());

      REQUIRE_RESULT(queue->write_data(buf.data(), 0xDEADBEEF));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      CHECK(*dst == 0xDEADBEEF);
    }
  }
}

TEST_CASE("Queue - multiple submissions across ring wrap", "[queue]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);
      auto *dst = static_cast<volatile uint32_t *>(buf.data());

      constexpr uint32_t pkt_dwords = kfd::pm4::WRITE_DATA_DWORDS;

      uint32_t n_pkts =
          static_cast<uint32_t>(queue->ring_dwords() / pkt_dwords) + 1;

      for (uint32_t i = 0; i < n_pkts; ++i) {
        *dst = 0xFFFFFFFF;

        REQUIRE_RESULT(sig->reset());
        REQUIRE_RESULT(queue->write_data(buf.data(), i));
        REQUIRE_RESULT(queue->signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
        REQUIRE(*dst == i);
      }
    }
  }
}

TEST_CASE("Queue - multiple queues coexist", "[queue][stress]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      constexpr size_t N = 4;
      std::vector<kfd::ComputeQueue> queues;
      std::vector<kfd::Signal> signals;
      queues.reserve(N);
      signals.reserve(N);

      for (size_t i = 0; i < N; ++i) {
        auto q = kfd::ComputeQueue::create(gpu);
        REQUIRE_RESULT(q);
        queues.push_back(std::move(*q));
        auto sig = kfd::Signal::create(ctx);
        REQUIRE_RESULT(sig);
        signals.push_back(std::move(*sig));
      }

      for (size_t i = 0; i < N; ++i)
        REQUIRE_RESULT(queues[i].signal(signals[i]));

      for (size_t i = 0; i < N; ++i)
        CHECK_RESULT(
            signals[i].wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("Queue - rapid create-destroy cycles", "[queue][stress]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      constexpr int CYCLES = 20;
      for (int i = 0; i < CYCLES; ++i) {
        auto q = kfd::ComputeQueue::create(gpu);
        REQUIRE_RESULT(q);

        auto sig = kfd::Signal::create(ctx);
        REQUIRE_RESULT(sig);
        REQUIRE_RESULT(q->signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }
    }
  }
}

TEST_CASE("Queue - heavy ring traffic with many wraps", "[queue][stress]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);
      auto *dst = static_cast<volatile uint32_t *>(buf.data());

      constexpr uint32_t pkt_dwords = kfd::pm4::WRITE_DATA_DWORDS;

      uint32_t n_pkts =
          static_cast<uint32_t>(queue->ring_dwords() / pkt_dwords) * 10;

      for (uint32_t i = 0; i < n_pkts; ++i) {
        *dst = 0xFFFFFFFF;

        REQUIRE_RESULT(sig->reset());
        REQUIRE_RESULT(queue->write_data(buf.data(), i));
        REQUIRE_RESULT(queue->signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
        REQUIRE(*dst == i);
      }
    }
  }
}

TEST_CASE("Queue - concurrent queues with interleaved submissions",
          "[queue][stress]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      constexpr size_t NQ = 3;
      constexpr size_t ROUNDS = 50;

      std::vector<kfd::ComputeQueue> queues;
      std::vector<kfd::Signal> signals;
      queues.reserve(NQ);
      signals.reserve(NQ);
      for (size_t i = 0; i < NQ; ++i) {
        auto q = kfd::ComputeQueue::create(gpu);
        REQUIRE_RESULT(q);
        queues.push_back(std::move(*q));
        auto sig = kfd::Signal::create(ctx);
        REQUIRE_RESULT(sig);
        signals.push_back(std::move(*sig));
      }

      auto buf = alloc_host_buffer(gpu);
      auto *base = static_cast<volatile uint32_t *>(buf.data());
      std::memset(const_cast<void *>(static_cast<volatile void *>(base)), 0,
                  buf.size());

      for (size_t r = 0; r < ROUNDS; ++r) {
        for (size_t qi = 0; qi < NQ; ++qi) {
          auto &q = queues[qi];
          uint32_t val = static_cast<uint32_t>(r * NQ + qi + 1);
          if (r > 0)
            REQUIRE_RESULT(signals[qi].reset());
          REQUIRE_RESULT(
              q.write_data(static_cast<uint32_t *>(buf.data()) + qi, val));
          REQUIRE_RESULT(q.signal(signals[qi]));
        }

        for (size_t qi = 0; qi < NQ; ++qi) {
          uint32_t val = static_cast<uint32_t>(r * NQ + qi + 1);
          REQUIRE_RESULT(signals[qi].wait(kfd::Condition::EQ, 0,
                                          kfd::test::WAIT_TIMEOUT_NS));
          REQUIRE(*(base + qi) == val);
        }
      }
    }
  }
}

TEST_CASE("Queue - survives allocation pressure", "[queue][stress]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      {
        REQUIRE_RESULT(queue->signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }

      constexpr int N_PRESSURE = 32;
      constexpr size_t PRESSURE_SIZE = 1024 * 1024;
      std::vector<kfd::Buffer> pressure;
      pressure.reserve(N_PRESSURE);
      for (int i = 0; i < N_PRESSURE; ++i) {
        auto a = kfd::Buffer::allocate(gpu, PRESSURE_SIZE, kfd::MemType::GTT,
                                       kfd::MemFlags::WRITABLE |
                                           kfd::MemFlags::COHERENT);
        if (!a)
          break;
        if (!a->map(gpu).has_value())
          break;
        pressure.push_back(std::move(*a));
      }
      REQUIRE(pressure.size() > 0);

      {
        REQUIRE_RESULT(sig->reset());
        REQUIRE_RESULT(queue->signal(*sig));
        CHECK_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      }
    }
  }
}

TEST_CASE("Queue - back-to-back submits without polling", "[queue]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);
      auto *base = static_cast<volatile uint32_t *>(buf.data());
      std::memset(const_cast<void *>(static_cast<volatile void *>(base)), 0xFF,
                  buf.size());

      constexpr uint32_t N = 8;
      for (uint32_t i = 0; i < N; ++i) {
        REQUIRE_RESULT(
            queue->write_data(static_cast<uint32_t *>(buf.data()) + i, i + 1));
      }
      REQUIRE_RESULT(queue->signal(*sig));

      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      for (uint32_t i = 0; i < N; ++i)
        CHECK(*(base + i) == i + 1);
    }
  }
}

TEST_CASE("Queue - mixed operations produce correct results", "[queue]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = alloc_host_buffer(gpu);
      auto *dst = static_cast<volatile uint32_t *>(buf.data());
      *dst = 0xFFFFFFFF;

      REQUIRE_RESULT(queue->write_data(buf.data(), 0xBAADF00D));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
      CHECK(*dst == 0xBAADF00D);
    }
  }
}

TEST_CASE("Queue - multi-threaded shared queue stress", "[queue][stress][mt]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      constexpr size_t N_THREADS = 4;
      constexpr size_t ITERS = 100;
      constexpr size_t TOTAL_SLOTS = N_THREADS * ITERS;

      auto buf = kfd::Buffer::allocate(
          gpu, TOTAL_SLOTS * sizeof(uint32_t), kfd::MemType::GTT,
          kfd::MemFlags::WRITABLE | kfd::MemFlags::EXECUTABLE |
              kfd::MemFlags::UNCACHED | kfd::MemFlags::COHERENT |
              kfd::MemFlags::NO_SUBSTITUTE);
      REQUIRE_RESULT(buf);
      REQUIRE_RESULT(buf->map(gpu));

      auto *base = static_cast<uint32_t *>(buf->data());
      std::memset(base, 0, buf->size());

      std::vector<kfd::Signal> signals;
      signals.reserve(N_THREADS);
      for (size_t i = 0; i < N_THREADS; ++i) {
        auto sig = kfd::Signal::create(ctx);
        REQUIRE_RESULT(sig);
        signals.push_back(std::move(*sig));
      }

      std::atomic<int> failures{0};

      auto worker = [&](size_t tid) {
        for (size_t j = 0; j < ITERS; ++j) {
          auto *slot = base + tid * ITERS + j;
          uint32_t seq = static_cast<uint32_t>(j + 1);
          if (!signals[tid].reset().has_value()) {
            failures.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          if (!queue->write_data(slot, seq).has_value()) {
            failures.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          if (!queue->signal(signals[tid]).has_value()) {
            failures.fetch_add(1, std::memory_order_relaxed);
            continue;
          }
          if (!signals[tid]
                   .wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS)
                   .has_value())
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

      for (size_t i = 0; i < N_THREADS; ++i)
        for (size_t j = 0; j < ITERS; ++j)
          CHECK(*(base + i * ITERS + j) == static_cast<uint32_t>(j + 1));
    }
  }
}

TEST_CASE("Queue - WRITE_DATA into pinned user memory", "[queue][memory]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      uint32_t host_val = 0;
      auto pinned = kfd::Buffer::pin(gpu, &host_val, sizeof(host_val));
      REQUIRE_RESULT(pinned);
      REQUIRE_RESULT(pinned->map(gpu));

      REQUIRE_RESULT(queue->write_data(pinned->data(), 0xDEADBEEF));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("Queue - WRITE_DATA into VRAM", "[queue][memory]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto buf = kfd::Buffer::allocate(
          gpu, kfd::detail::page_size(), kfd::MemType::VRAM,
          kfd::MemFlags::WRITABLE | kfd::MemFlags::EXECUTABLE);
      REQUIRE_RESULT(buf);
      REQUIRE_RESULT(buf->map(gpu));

      REQUIRE_RESULT(queue->write_data(buf->data(), 0xCAFEBABE));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  }
}

TEST_CASE("Queue - DMA_DATA copy GTT to VRAM and back", "[queue][dma]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      if (!gpu.vram_host_visible())
        SKIP("Device does not support host-visible VRAM (large BAR disabled)");

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      constexpr size_t SIZE = 4096;
      constexpr uint32_t PATTERN = 0xCAFEBABE;

      auto gtt_src = kfd::Buffer::allocate(gpu, SIZE, kfd::MemType::GTT,
                                           kfd::MemFlags::WRITABLE |
                                               kfd::MemFlags::HOST_ACCESS |
                                               kfd::MemFlags::UNCACHED);
      REQUIRE_RESULT(gtt_src);
      REQUIRE_RESULT(gtt_src->map(gpu));

      auto vram = kfd::Buffer::allocate(gpu, SIZE, kfd::MemType::VRAM,
                                        kfd::MemFlags::WRITABLE |
                                            kfd::MemFlags::HOST_ACCESS |
                                            kfd::MemFlags::NO_SUBSTITUTE);
      REQUIRE_RESULT(vram);
      REQUIRE_RESULT(vram->map(gpu));

      auto gtt_dst = kfd::Buffer::allocate(gpu, SIZE, kfd::MemType::GTT,
                                           kfd::MemFlags::WRITABLE |
                                               kfd::MemFlags::HOST_ACCESS |
                                               kfd::MemFlags::UNCACHED);
      REQUIRE_RESULT(gtt_dst);
      REQUIRE_RESULT(gtt_dst->map(gpu));

      auto *src = static_cast<uint32_t *>(gtt_src->data());
      auto *dst = static_cast<uint32_t *>(gtt_dst->data());
      for (size_t i = 0; i < SIZE / sizeof(uint32_t); ++i)
        src[i] = PATTERN ^ static_cast<uint32_t>(i);
      std::memset(gtt_dst->data(), 0, SIZE);
      std::memset(vram->data(), 0, SIZE);

      // GTT -> VRAM -> GTT, then verify the round-trip.
      REQUIRE_RESULT(queue->dma_copy(vram->data(), gtt_src->data(),
                                     static_cast<uint32_t>(SIZE)));
      REQUIRE_RESULT(queue->dma_copy(gtt_dst->data(), vram->data(),
                                     static_cast<uint32_t>(SIZE)));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      for (size_t i = 0; i < SIZE / sizeof(uint32_t); ++i)
        CHECK(dst[i] == (PATTERN ^ static_cast<uint32_t>(i)));
    }
  }
}

TEST_CASE("Queue - DMA_DATA fill pattern", "[queue][dma]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      constexpr size_t SIZE = 4096;
      constexpr uint32_t FILL_VAL = 0xDEAD1234;

      auto buf = alloc_host_buffer(gpu, SIZE);
      std::memset(buf.data(), 0, SIZE);

      REQUIRE_RESULT(
          queue->dma_fill(buf.data(), FILL_VAL, static_cast<uint32_t>(SIZE)));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *words = static_cast<uint32_t *>(buf.data());
      for (size_t i = 0; i < SIZE / sizeof(uint32_t); ++i)
        CHECK(words[i] == FILL_VAL);
    }
  }
}

TEST_CASE("Queue - DMA_DATA copy varying sizes", "[queue][dma]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      constexpr size_t MAX_SIZE = 16384;

      auto src_buf = alloc_host_buffer(gpu, MAX_SIZE);
      auto dst_buf = alloc_host_buffer(gpu, MAX_SIZE);

      auto *src = static_cast<uint32_t *>(src_buf.data());
      for (size_t i = 0; i < MAX_SIZE / sizeof(uint32_t); ++i)
        src[i] = static_cast<uint32_t>(i);

      for (uint32_t size : {4u, 64u, 256u, 1024u, 4096u, 16384u}) {
        std::memset(dst_buf.data(), 0xFF, MAX_SIZE);

        REQUIRE_RESULT(sig->reset());
        REQUIRE_RESULT(queue->dma_copy(dst_buf.data(), src_buf.data(), size));
        REQUIRE_RESULT(queue->signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

        auto *dst = static_cast<uint32_t *>(dst_buf.data());
        for (uint32_t i = 0; i < size / sizeof(uint32_t); ++i)
          CHECK(dst[i] == i);
      }
    }
  }
}

TEST_CASE("Queue - INDIRECT_BUFFER dispatches write", "[queue][indirect]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto dst = alloc_host_buffer(gpu);
      auto *fence = static_cast<volatile uint32_t *>(dst.data());
      *fence = 0xFFFFFFFF;

      auto ib = alloc_host_buffer(gpu);
      auto *ib_words = static_cast<uint32_t *>(ib.data());
      uint32_t n = kfd::pm4::write_data(ib_words, dst.data(), 0xDEADBEEF);

      REQUIRE_RESULT(queue->indirect_buffer(ib.data(), n));
      REQUIRE_RESULT(queue->signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      CHECK(*fence == 0xDEADBEEF);
    }
  }
}

TEST_CASE("Queue - custom release_mem with host pointer polling",
          "[queue][custom]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto buf = alloc_host_buffer(gpu);
      auto *fence = static_cast<volatile uint64_t *>(buf.data());
      *fence = 0;

      constexpr uint64_t SENTINEL = 0xDEAD'BEEF'CAFE'BABEull;
      uint32_t ordinal2 = kfd::pm4::eop_fence_flush(gpu.gfx_version());
      REQUIRE_RESULT(queue->release_mem(ordinal2, kfd::pm4::DATA_64,
                                        kfd::pm4::DATA_CONFIRM, buf.data(),
                                        SENTINEL));

      REQUIRE(poll_value(reinterpret_cast<uint32_t *>(buf.data()),
                         static_cast<uint32_t>(SENTINEL)));
      CHECK(*fence == SENTINEL);
    }
  }
}

TEST_CASE("Queue - custom release_mem with manual event", "[queue][custom]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto event = kfd::Event::create(ctx);
      REQUIRE_RESULT(event);

      // First, write a user-visible fence to a host buffer.
      auto buf = alloc_host_buffer(gpu);
      auto *fence = static_cast<volatile uint64_t *>(buf.data());
      *fence = 0;

      constexpr uint64_t DONE = 42;
      uint32_t ordinal2 = kfd::pm4::eop_fence_flush(gpu.gfx_version());
      REQUIRE_RESULT(queue->release_mem(ordinal2, kfd::pm4::DATA_64,
                                        kfd::pm4::DATA_CONFIRM, buf.data(),
                                        DONE));

      // Then fire the event: the kernel requires event_id written to
      // signal_addr.
      REQUIRE_RESULT(queue->release_mem(
          ordinal2, kfd::pm4::DATA_64, kfd::pm4::INT_DATA_CONFIRM,
          event->signal_addr(), static_cast<uint64_t>(event->event_id()),
          event->trigger_data()));

      REQUIRE_RESULT(event->wait(kfd::test::WAIT_TIMEOUT_NS));
      CHECK(*fence == DONE);
    }
  }
}

TEST_CASE("Queue - custom release_mem writeback-only fence",
          "[queue][custom]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto buf = alloc_host_buffer(gpu);
      auto *fence = static_cast<volatile uint64_t *>(buf.data());
      *fence = 0;

      constexpr uint64_t VAL = 0x1234;
      uint32_t ordinal2 = kfd::pm4::eop_wb_flush(gpu.gfx_version());
      REQUIRE_RESULT(queue->release_mem(ordinal2, kfd::pm4::DATA_64,
                                        kfd::pm4::DATA_CONFIRM, buf.data(),
                                        VAL));

      REQUIRE(poll_value(reinterpret_cast<uint32_t *>(buf.data()),
                         static_cast<uint32_t>(VAL)));
      CHECK(*fence == VAL);
    }
  }
}

TEST_CASE("Queue - custom release_mem timestamp", "[queue][custom]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);

      auto queue = kfd::ComputeQueue::create(gpu);
      REQUIRE_RESULT(queue);

      auto buf = alloc_host_buffer(gpu);
      auto *ts = static_cast<volatile uint64_t *>(buf.data());
      *ts = 0;

      uint32_t ordinal2 = kfd::pm4::eop_fence_flush(gpu.gfx_version());
      REQUIRE_RESULT(queue->release_mem(ordinal2, kfd::pm4::DATA_TIMESTAMP,
                                        kfd::pm4::DATA_CONFIRM, buf.data()));

      // Poll the low dword until it becomes nonzero; timestamp writes both
      // halves.
      auto *lo32 = reinterpret_cast<uint32_t *>(buf.data());
      for (int i = 0; i < 5'000'000 && *ts == 0; ++i)
        kfd::detail::spin_hint();
      (void)lo32;
      CHECK(*ts != 0);
    }
  }
}
