#include "libkfd/detail/mutex.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <thread>
#include <vector>

using kfd::detail::LockGuard;
using kfd::detail::Mutex;

TEST_CASE("Mutex - default state is unlocked", "[mutex]") {
  Mutex m;
  m.lock();
  m.unlock();
}

TEST_CASE("Mutex - repeated lock/unlock cycles", "[mutex]") {
  Mutex m;
  for (int i = 0; i < 64; ++i) {
    m.lock();
    m.unlock();
  }
}

TEST_CASE("LockGuard - releases on scope exit", "[mutex]") {
  Mutex m;
  {
    LockGuard g(m);
  }
  m.lock();
  m.unlock();
}

TEST_CASE("Mutex - mutual exclusion under contention", "[mutex]") {
  Mutex m;
  uint64_t counter = 0;
  constexpr unsigned NumThreads = 8;
  constexpr unsigned Increments = 100'000;

  std::vector<std::thread> threads;
  threads.reserve(NumThreads);
  for (unsigned t = 0; t < NumThreads; ++t) {
    threads.emplace_back([&] {
      for (unsigned i = 0; i < Increments; ++i) {
        LockGuard g(m);
        ++counter;
      }
    });
  }
  for (auto &t : threads)
    t.join();

  CHECK(counter == uint64_t{NumThreads} * Increments);
}

TEST_CASE("Mutex - spin path resolves short critical sections", "[mutex]") {
  Mutex m;
  std::atomic<bool> done{false};
  constexpr unsigned Rounds = 4096;

  m.lock();

  std::thread holder([&] {
    for (unsigned i = 0; i < Rounds; ++i) {
      LockGuard g(m);
    }
    done.store(true, std::memory_order_release);
  });

  m.unlock();

  unsigned acquired = 0;
  while (!done.load(std::memory_order_acquire)) {
    LockGuard g(m);
    ++acquired;
  }

  holder.join();
  CHECK(acquired > 0);
}

TEST_CASE("Mutex - high thread count does not corrupt state", "[mutex]") {
  Mutex m;
  uint64_t value = 0;
  constexpr unsigned NumThreads = 32;
  constexpr unsigned Iters = 10'000;

  std::vector<std::thread> threads;
  threads.reserve(NumThreads);
  for (unsigned t = 0; t < NumThreads; ++t) {
    threads.emplace_back([&] {
      for (unsigned i = 0; i < Iters; ++i) {
        LockGuard g(m);
        uint64_t tmp = value;
        value = tmp + 1;
      }
    });
  }
  for (auto &t : threads)
    t.join();

  CHECK(value == uint64_t{NumThreads} * Iters);
}
