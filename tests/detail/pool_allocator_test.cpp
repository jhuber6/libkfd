#include "libkfd/detail/pool_allocator.h"

#include "test_helpers.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <set>
#include <thread>
#include <vector>

using kfd::detail::PoolAllocator;

static auto *const BASE = reinterpret_cast<std::byte *>(uintptr_t{0x1000000});

static std::span<std::byte> region(size_t size) { return {BASE, size}; }

TEST_CASE("PoolAllocator - create valid", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);
  CHECK(a->region().data() == BASE);
  CHECK(a->region().size() == 4096);
  CHECK(a->alignment() == 64);
  CHECK(static_cast<bool>(*a));
}

TEST_CASE("PoolAllocator - create rejects invalid parameters",
          "[pool_allocator]") {
  CHECK_FALSE(PoolAllocator::create(region(0), 64).has_value());
  CHECK_FALSE(PoolAllocator::create(region(4096), 0).has_value());
  CHECK_FALSE(PoolAllocator::create(region(4096), 100).has_value());
}

TEST_CASE("PoolAllocator - default constructed is empty", "[pool_allocator]") {
  PoolAllocator a;
  CHECK_FALSE(static_cast<bool>(a));
}

TEST_CASE("PoolAllocator - single allocation", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);

  auto p = a->allocate(64);
  REQUIRE_RESULT(p);
  CHECK(*p >= BASE);
  CHECK(*p < BASE + 4096);
}

TEST_CASE("PoolAllocator - zero-size allocation fails", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);
  CHECK_FALSE(a->allocate(0).has_value());
}

TEST_CASE("PoolAllocator - oversized allocation fails", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);
  CHECK_FALSE(a->allocate(8192).has_value());
}

TEST_CASE("PoolAllocator - allocate entire pool", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);

  auto p = a->allocate(4096);
  REQUIRE_RESULT(p);
  CHECK(*p == BASE);
  CHECK_FALSE(a->allocate(1).has_value());
}

TEST_CASE("PoolAllocator - exact-fit allocation (no rounding waste)",
          "[pool_allocator]") {
  auto a = PoolAllocator::create(region(1024), 64);
  REQUIRE_RESULT(a);

  // 192 is not a power of two but is a multiple of 64. A buddy allocator
  // would round this to 256, wasting 64 bytes. The pool allocator should
  // allocate exactly 192 and leave 832 free.
  auto p = a->allocate(192);
  REQUIRE_RESULT(p);

  auto q = a->allocate(832);
  REQUIRE_RESULT(q);

  CHECK_FALSE(a->allocate(1).has_value());
}

TEST_CASE("PoolAllocator - rounds up to alignment", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);

  auto p = a->allocate(1);
  REQUIRE_RESULT(p);
  CHECK(reinterpret_cast<uintptr_t>(*p) % 64 == 0);

  // 1 byte rounds to 64, so 4096 - 64 = 4032 remaining.
  auto q = a->allocate(4032);
  REQUIRE_RESULT(q);
  CHECK_FALSE(a->allocate(1).has_value());
}

TEST_CASE("PoolAllocator - fill with min blocks", "[pool_allocator]") {
  constexpr size_t TOTAL = 4096;
  constexpr size_t ALIGN = 64;
  constexpr size_t COUNT = TOTAL / ALIGN;

  auto a = PoolAllocator::create(region(TOTAL), ALIGN);
  REQUIRE_RESULT(a);

  std::set<void *> ptrs;
  for (size_t i = 0; i < COUNT; ++i) {
    auto p = a->allocate(ALIGN);
    REQUIRE_RESULT(p);
    ptrs.insert(*p);
  }
  CHECK(ptrs.size() == COUNT);
  CHECK_FALSE(a->allocate(1).has_value());
}

TEST_CASE("PoolAllocator - deallocate and reuse", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(256), 64);
  REQUIRE_RESULT(a);

  auto p1 = a->allocate(64);
  auto p2 = a->allocate(64);
  auto p3 = a->allocate(64);
  auto p4 = a->allocate(64);
  REQUIRE_RESULT(p1);
  REQUIRE_RESULT(p2);
  REQUIRE_RESULT(p3);
  REQUIRE_RESULT(p4);
  CHECK_FALSE(a->allocate(64).has_value());

  REQUIRE_RESULT(a->deallocate(*p2, 64));
  auto p5 = a->allocate(64);
  REQUIRE_RESULT(p5);
  CHECK(*p5 == *p2);

  REQUIRE_RESULT(a->deallocate(*p1, 64));
  REQUIRE_RESULT(a->deallocate(*p3, 64));
  REQUIRE_RESULT(a->deallocate(*p4, 64));
  REQUIRE_RESULT(a->deallocate(*p5, 64));

  auto whole = a->allocate(256);
  REQUIRE_RESULT(whole);
}

TEST_CASE("PoolAllocator - coalescing adjacent blocks", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(256), 64);
  REQUIRE_RESULT(a);

  auto left = a->allocate(128);
  auto right = a->allocate(128);
  REQUIRE_RESULT(left);
  REQUIRE_RESULT(right);
  CHECK_FALSE(a->allocate(256).has_value());

  REQUIRE_RESULT(a->deallocate(*left, 128));
  REQUIRE_RESULT(a->deallocate(*right, 128));

  auto whole = a->allocate(256);
  REQUIRE_RESULT(whole);
  CHECK(*whole == BASE);
}

TEST_CASE("PoolAllocator - partial coalesce blocked by neighbor",
          "[pool_allocator]") {
  auto a = PoolAllocator::create(region(256), 64);
  REQUIRE_RESULT(a);

  auto p1 = a->allocate(64);
  auto p2 = a->allocate(64);
  auto p3 = a->allocate(64);
  REQUIRE_RESULT(p1);
  REQUIRE_RESULT(p2);
  REQUIRE_RESULT(p3);

  // Free the first; can't coalesce into a 128-byte block because p2 is live.
  REQUIRE_RESULT(a->deallocate(*p1, 64));
  CHECK_FALSE(a->allocate(128).has_value());

  // Free p2; now p1+p2 coalesce into 128 bytes.
  REQUIRE_RESULT(a->deallocate(*p2, 64));
  auto merged = a->allocate(128);
  REQUIRE_RESULT(merged);
  CHECK(*merged == BASE);
}

TEST_CASE("PoolAllocator - deallocate null is safe", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);
  REQUIRE_RESULT(a->deallocate(nullptr, 64));
}

TEST_CASE("PoolAllocator - deallocate out-of-range is safe",
          "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);
  REQUIRE_RESULT(a->deallocate(reinterpret_cast<void *>(uintptr_t{0}), 64));
  REQUIRE_RESULT(a->deallocate(BASE + 8192, 64));
}

TEST_CASE("PoolAllocator - move construction", "[pool_allocator]") {
  auto a = PoolAllocator::create(region(4096), 64);
  REQUIRE_RESULT(a);

  auto p = a->allocate(128);
  REQUIRE_RESULT(p);

  PoolAllocator b(std::move(*a));
  CHECK(static_cast<bool>(b));

  auto q = b.allocate(64);
  REQUIRE_RESULT(q);
}

TEST_CASE("PoolAllocator - stress alloc-free cycle", "[pool_allocator]") {
  constexpr size_t TOTAL = 65536;
  constexpr size_t BLOCK = 64;
  auto a = PoolAllocator::create(region(TOTAL), BLOCK);
  REQUIRE_RESULT(a);

  std::vector<void *> ptrs;
  for (int round = 0; round < 10; ++round) {
    ptrs.clear();
    for (auto p = a->allocate(BLOCK); p; p = a->allocate(BLOCK))
      ptrs.push_back(*p);

    CHECK_FALSE(a->allocate(1).has_value());

    for (void *p : ptrs)
      REQUIRE_RESULT(a->deallocate(p, BLOCK));

    auto whole = a->allocate(TOTAL);
    REQUIRE_RESULT(whole);
    REQUIRE_RESULT(a->deallocate(*whole, TOTAL));
  }
}

TEST_CASE("PoolAllocator - mixed sizes stress", "[pool_allocator]") {
  constexpr size_t TOTAL = 65536;
  constexpr size_t ALIGN = 64;
  auto a = PoolAllocator::create(region(TOTAL), ALIGN);
  REQUIRE_RESULT(a);

  struct Alloc {
    void *ptr;
    size_t size;
  };
  std::vector<Alloc> allocs;
  size_t sizes[] = {64, 128, 192, 320, 1024, 64, 64, 256};

  for (int round = 0; round < 5; ++round) {
    allocs.clear();
    for (size_t sz : sizes) {
      auto p = a->allocate(sz);
      if (p)
        allocs.push_back({*p, sz});
    }

    for (auto &al : allocs)
      REQUIRE_RESULT(a->deallocate(al.ptr, al.size));

    auto whole = a->allocate(TOTAL);
    REQUIRE_RESULT(whole);
    REQUIRE_RESULT(a->deallocate(*whole, TOTAL));
  }
}

TEST_CASE("PoolAllocator - non-power-of-two sizes don't waste space",
          "[pool_allocator]") {
  constexpr size_t TOTAL = 65536;
  constexpr size_t ALIGN = 64;
  auto a = PoolAllocator::create(region(TOTAL), ALIGN);
  REQUIRE_RESULT(a);

  // Allocate 100 blocks of 640 bytes each (non-power-of-two but aligned).
  // 100 * 640 = 64000, leaving 1536 free. A buddy allocator would round 640
  // to 1024, fitting only 64 blocks in the same space.
  constexpr size_t BLOCK_SIZE = 640;
  constexpr size_t COUNT = TOTAL / BLOCK_SIZE;

  struct Alloc {
    void *ptr;
    size_t size;
  };
  std::vector<Alloc> allocs;

  for (size_t i = 0; i < COUNT; ++i) {
    auto p = a->allocate(BLOCK_SIZE);
    REQUIRE_RESULT(p);
    allocs.push_back({*p, BLOCK_SIZE});
  }
  CHECK(allocs.size() == COUNT);

  for (auto &al : allocs)
    REQUIRE_RESULT(a->deallocate(al.ptr, al.size));

  auto whole = a->allocate(TOTAL);
  REQUIRE_RESULT(whole);
  REQUIRE_RESULT(a->deallocate(*whole, TOTAL));
}

TEST_CASE("PoolAllocator - threaded alloc-free", "[pool_allocator]") {
  constexpr size_t TOTAL = 1 << 20; // 1 MiB
  constexpr size_t ALIGN = 64;
  constexpr unsigned NUM_THREADS = 8;
  constexpr unsigned ROUNDS = 500;

  auto a = PoolAllocator::create(region(TOTAL), ALIGN);
  REQUIRE_RESULT(a);

  std::atomic<unsigned> failures{0};
  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);

  for (unsigned t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&, t] {
      for (unsigned r = 0; r < ROUNDS; ++r) {
        size_t sz = ALIGN * (1 + ((t * 7 + r * 3) % 16));
        auto p = a->allocate(sz);
        if (!p) {
          ++failures;
          continue;
        }
        auto addr = reinterpret_cast<uintptr_t>(*p);
        if (addr < reinterpret_cast<uintptr_t>(BASE) ||
            addr >= reinterpret_cast<uintptr_t>(BASE) + TOTAL)
          ++failures;
        if (addr % ALIGN != 0)
          ++failures;
        if (!a->deallocate(*p, sz))
          ++failures;
      }
    });
  }

  for (auto &t : threads)
    t.join();

  CHECK(failures.load() == 0);

  auto whole = a->allocate(TOTAL);
  REQUIRE_RESULT(whole);
  REQUIRE_RESULT(a->deallocate(*whole, TOTAL));
}

TEST_CASE("PoolAllocator - threaded saturation", "[pool_allocator]") {
  constexpr size_t TOTAL = 4096;
  constexpr size_t ALIGN = 64;
  constexpr size_t MAX_BLOCKS = TOTAL / ALIGN;
  constexpr unsigned NUM_THREADS = 8;

  auto a = PoolAllocator::create(region(TOTAL), ALIGN);
  REQUIRE_RESULT(a);

  std::atomic<unsigned> total_allocated{0};
  std::atomic<unsigned> duplicates{0};
  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);

  struct PerThread {
    void *ptrs[MAX_BLOCKS]{};
    unsigned count = 0;
  };
  std::vector<PerThread> per_thread(NUM_THREADS);

  for (unsigned t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&, t] {
      auto &pt = per_thread[t];
      for (auto p = a->allocate(ALIGN); p; p = a->allocate(ALIGN)) {
        pt.ptrs[pt.count++] = *p;
        total_allocated.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &t : threads)
    t.join();

  CHECK(total_allocated.load() == MAX_BLOCKS);

  std::set<void *> all;
  for (unsigned t = 0; t < NUM_THREADS; ++t)
    for (unsigned i = 0; i < per_thread[t].count; ++i)
      if (!all.insert(per_thread[t].ptrs[i]).second)
        ++duplicates;

  CHECK(duplicates.load() == 0);
  CHECK(all.size() == MAX_BLOCKS);

  for (unsigned t = 0; t < NUM_THREADS; ++t)
    for (unsigned i = 0; i < per_thread[t].count; ++i)
      REQUIRE_RESULT(a->deallocate(per_thread[t].ptrs[i], ALIGN));

  auto whole = a->allocate(TOTAL);
  REQUIRE_RESULT(whole);
  REQUIRE_RESULT(a->deallocate(*whole, TOTAL));
}

TEST_CASE("PoolAllocator - threaded mixed sizes", "[pool_allocator]") {
  constexpr size_t TOTAL = 1 << 20;
  constexpr size_t ALIGN = 64;
  constexpr unsigned NUM_THREADS = 8;
  constexpr unsigned ROUNDS = 200;

  auto a = PoolAllocator::create(region(TOTAL), ALIGN);
  REQUIRE_RESULT(a);

  std::atomic<unsigned> bad{0};
  std::vector<std::thread> threads;
  threads.reserve(NUM_THREADS);

  for (unsigned t = 0; t < NUM_THREADS; ++t) {
    threads.emplace_back([&, t] {
      struct Alloc {
        void *ptr;
        size_t size;
      };
      std::vector<Alloc> held;
      held.reserve(32);

      for (unsigned r = 0; r < ROUNDS; ++r) {
        size_t sz = ALIGN * (1 + ((t + r) % 32));
        if (sz > TOTAL)
          sz = TOTAL;
        auto p = a->allocate(sz);
        if (p) {
          auto addr = reinterpret_cast<uintptr_t>(*p);
          if (addr % ALIGN != 0)
            bad.fetch_add(1, std::memory_order_relaxed);
          held.push_back({*p, sz});
        }

        if (held.size() > 16 || r == ROUNDS - 1) {
          for (auto &al : held)
            if (!a->deallocate(al.ptr, al.size))
              bad.fetch_add(1, std::memory_order_relaxed);
          held.clear();
        }
      }
    });
  }

  for (auto &t : threads)
    t.join();

  CHECK(bad.load() == 0);

  // After all threads have freed their allocations the pool must fully
  // coalesce, allowing a single allocation of the entire region.
  auto whole = a->allocate(TOTAL);
  REQUIRE_RESULT(whole);
  REQUIRE_RESULT(a->deallocate(*whole, TOTAL));
}
