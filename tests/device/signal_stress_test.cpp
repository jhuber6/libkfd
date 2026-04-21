#include "test_helpers.h"

#include "libkfd/abi.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

static const kfd::test::TestBinary dispatch_kernels[] = {
#include "dispatch_kernels.inc"
};

using kfd::test::DeviceFixture;
using kfd::test::make_device_fixture;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

namespace {

struct SignalFixture : DeviceFixture {
  kfd::Kernel nop;
  kfd::Buffer nop_kernarg;
  kfd::DispatchConfig nop_cfg;
};

std::expected<SignalFixture, kfd::Error>
make_fixture(kfd::Context &ctx, kfd::Dim3 grid = {.x = 1},
             kfd::Dim3 block = {.x = 64}) {
  auto base = KFD_TRY(make_device_fixture(ctx, dispatch_kernels));
  auto nop = KFD_TRY(base.exe.kernel("nop.kd"));
  kfd::DispatchConfig cfg{.grid = grid, .block = block};
  auto ka = KFD_TRY(nop.make_kernargs(*base.gpu, cfg));
  return SignalFixture{std::move(base), std::move(nop), std::move(ka), cfg};
}

void dispatch_nop(SignalFixture &fix) {
  REQUIRE_RESULT(fix.compute.dispatch(fix.nop, fix.nop_cfg, fix.nop_kernarg));
}

} // namespace

TEST_CASE("Signal stress - reuse across many cycles",
          "[device][signal][stress]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  auto queue = kfd::ComputeQueue::create(gpu);
  REQUIRE_RESULT(queue);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  constexpr unsigned CYCLES = 200;
  for (unsigned i = 0; i < CYCLES; ++i) {
    INFO("cycle " << i);
    if (i > 0)
      REQUIRE_RESULT(sig->reset());
    REQUIRE_RESULT(queue->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Signal stress - counting signal single queue",
          "[device][signal][stress]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  auto queue = kfd::ComputeQueue::create(gpu);
  REQUIRE_RESULT(queue);

  constexpr uint64_t COUNT = 10;
  auto sig = kfd::Signal::create(ctx, COUNT);
  REQUIRE_RESULT(sig);

  for (uint64_t i = 0; i < COUNT; ++i)
    REQUIRE_RESULT(queue->signal(*sig));

  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}

TEST_CASE("Signal stress - large counting signal", "[device][signal][stress]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  auto queue = kfd::ComputeQueue::create(gpu);
  REQUIRE_RESULT(queue);

  constexpr uint64_t COUNT = 100;
  auto sig = kfd::Signal::create(ctx, COUNT);
  REQUIRE_RESULT(sig);

  for (uint64_t i = 0; i < COUNT; ++i)
    REQUIRE_RESULT(queue->signal(*sig));

  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}

TEST_CASE("Signal stress - multiple independent signals",
          "[device][signal][stress]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  auto queue = kfd::ComputeQueue::create(gpu);
  REQUIRE_RESULT(queue);

  constexpr unsigned N = 8;
  std::vector<kfd::Signal> signals;
  signals.reserve(N);
  for (unsigned i = 0; i < N; ++i) {
    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);
    signals.push_back(std::move(*sig));
  }

  for (unsigned i = 0; i < N; ++i)
    REQUIRE_RESULT(queue->signal(signals[i]));

  for (unsigned i = 0; i < N; ++i) {
    INFO("waiting on signal " << i);
    REQUIRE_RESULT(
        signals[i].wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Signal stress - independent signals on separate queues",
          "[device][signal][stress]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  constexpr unsigned NQ = 4;
  std::vector<kfd::ComputeQueue> queues;
  std::vector<kfd::Signal> signals;
  queues.reserve(NQ);
  signals.reserve(NQ);
  for (unsigned i = 0; i < NQ; ++i) {
    auto q = kfd::ComputeQueue::create(gpu);
    REQUIRE_RESULT(q);
    queues.push_back(std::move(*q));
    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);
    signals.push_back(std::move(*sig));
  }

  for (unsigned i = 0; i < NQ; ++i)
    REQUIRE_RESULT(queues[i].signal(signals[i]));

  for (unsigned i = 0; i < NQ; ++i) {
    INFO("waiting on signal " << i);
    REQUIRE_RESULT(
        signals[i].wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Signal stress - shared counting signal from two queues",
          "[device][signal][stress]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  auto q1 = kfd::ComputeQueue::create(gpu);
  auto q2 = kfd::ComputeQueue::create(gpu);
  REQUIRE_RESULT(q1);
  REQUIRE_RESULT(q2);

  auto sig = kfd::Signal::create(ctx, 2);
  REQUIRE_RESULT(sig);

  REQUIRE_RESULT(q1->signal(*sig));
  REQUIRE_RESULT(q2->signal(*sig));

  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}

TEST_CASE("Signal stress - shared counting signal from four queues",
          "[device][signal][stress]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  constexpr unsigned NQ = 4;
  std::vector<kfd::ComputeQueue> queues;
  queues.reserve(NQ);
  for (unsigned i = 0; i < NQ; ++i) {
    auto q = kfd::ComputeQueue::create(gpu);
    REQUIRE_RESULT(q);
    queues.push_back(std::move(*q));
  }

  auto sig = kfd::Signal::create(ctx, NQ);
  REQUIRE_RESULT(sig);

  for (unsigned i = 0; i < NQ; ++i)
    REQUIRE_RESULT(queues[i].signal(*sig));

  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}

TEST_CASE("Signal stress - repeated multi-queue counting signal",
          "[device][signal][stress]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  constexpr unsigned NQ = 3;
  constexpr unsigned ROUNDS = 30;

  std::vector<kfd::ComputeQueue> queues;
  queues.reserve(NQ);
  for (unsigned i = 0; i < NQ; ++i) {
    auto q = kfd::ComputeQueue::create(gpu);
    REQUIRE_RESULT(q);
    queues.push_back(std::move(*q));
  }

  auto sig = kfd::Signal::create(ctx, NQ);
  REQUIRE_RESULT(sig);

  for (unsigned r = 0; r < ROUNDS; ++r) {
    INFO("round " << r);
    if (r > 0)
      REQUIRE_RESULT(sig->reset(NQ));

    for (unsigned i = 0; i < NQ; ++i)
      REQUIRE_RESULT(queues[i].signal(*sig));

    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Signal stress - threaded signals on shared queue",
          "[device][signal][stress][mt]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  auto queue = kfd::ComputeQueue::create(gpu);
  REQUIRE_RESULT(queue);

  constexpr unsigned N_THREADS = 4;
  constexpr unsigned ITERS = 50;

  std::vector<kfd::Signal> signals;
  signals.reserve(N_THREADS);
  for (unsigned i = 0; i < N_THREADS; ++i) {
    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);
    signals.push_back(std::move(*sig));
  }

  auto worker = [&](unsigned tid) {
    for (unsigned i = 0; i < ITERS; ++i) {
      REQUIRE_RESULT(signals[tid].reset());
      REQUIRE_RESULT(queue->signal(signals[tid]));
      REQUIRE_RESULT(
          signals[tid].wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(N_THREADS);
  for (unsigned t = 0; t < N_THREADS; ++t)
    threads.emplace_back(worker, t);
  for (auto &th : threads)
    th.join();
}

TEST_CASE("Signal stress - threaded signals on separate queues",
          "[device][signal][stress][mt]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  constexpr unsigned N_THREADS = 4;
  constexpr unsigned ITERS = 50;

  std::vector<kfd::ComputeQueue> queues;
  std::vector<kfd::Signal> signals;
  queues.reserve(N_THREADS);
  signals.reserve(N_THREADS);
  for (unsigned i = 0; i < N_THREADS; ++i) {
    auto q = kfd::ComputeQueue::create(gpu);
    REQUIRE_RESULT(q);
    queues.push_back(std::move(*q));
    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);
    signals.push_back(std::move(*sig));
  }

  auto worker = [&](unsigned tid) {
    for (unsigned i = 0; i < ITERS; ++i) {
      REQUIRE_RESULT(signals[tid].reset());
      REQUIRE_RESULT(queues[tid].signal(signals[tid]));
      REQUIRE_RESULT(
          signals[tid].wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(N_THREADS);
  for (unsigned t = 0; t < N_THREADS; ++t)
    threads.emplace_back(worker, t);
  for (auto &th : threads)
    th.join();
}

TEST_CASE("Signal - wait_any returns first completed signal",
          "[device][signal]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  auto queue = kfd::ComputeQueue::create(gpu);
  REQUIRE_RESULT(queue);

  constexpr unsigned N = 4;
  std::vector<kfd::Signal> signals;
  std::vector<kfd::Signal *> ptrs;
  signals.reserve(N);
  ptrs.reserve(N);
  for (unsigned i = 0; i < N; ++i) {
    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);
    signals.push_back(std::move(*sig));
  }
  for (unsigned i = 0; i < N; ++i)
    ptrs.push_back(&signals[i]);

  constexpr unsigned TARGET = 2;
  REQUIRE_RESULT(queue->signal(signals[TARGET]));

  auto result =
      kfd::wait_any(ptrs, kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS);
  REQUIRE_RESULT(result);
  CHECK(*result == TARGET);

  for (unsigned i = 0; i < N; ++i) {
    if (i == TARGET)
      continue;
    uint64_t val = __atomic_load_n(signals[i].fence_addr(), __ATOMIC_ACQUIRE);
    CHECK(val != 0);
  }
}

TEST_CASE("Signal - wait_all blocks until all signals complete",
          "[device][signal]") {
  auto &gpu = require_gpu();
  auto &ctx = gpu.context();

  constexpr unsigned N = 4;
  std::vector<kfd::ComputeQueue> queues;
  std::vector<kfd::Signal> signals;
  std::vector<kfd::Signal *> ptrs;
  queues.reserve(N);
  signals.reserve(N);
  ptrs.reserve(N);
  for (unsigned i = 0; i < N; ++i) {
    auto q = kfd::ComputeQueue::create(gpu);
    REQUIRE_RESULT(q);
    queues.push_back(std::move(*q));
    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);
    signals.push_back(std::move(*sig));
  }
  for (unsigned i = 0; i < N; ++i)
    ptrs.push_back(&signals[i]);

  for (unsigned i = 0; i < N; ++i)
    REQUIRE_RESULT(queues[i].signal(signals[i]));

  REQUIRE_RESULT(
      kfd::wait_all(ptrs, kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

  for (unsigned i = 0; i < N; ++i) {
    uint64_t val = __atomic_load_n(signals[i].fence_addr(), __ATOMIC_ACQUIRE);
    CHECK(val == 0);
  }
}

TEST_CASE("Signal stress - rapid dispatch+signal cycles",
          "[device][signal][stress]") {
  auto &ctx = require_ctx();
  auto fix = make_fixture(ctx);
  REQUIRE_RESULT(fix);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);

  constexpr unsigned CYCLES = 200;
  for (unsigned i = 0; i < CYCLES; ++i) {
    INFO("cycle " << i);
    if (i > 0)
      REQUIRE_RESULT(sig->reset());
    dispatch_nop(*fix);
    REQUIRE_RESULT(fix->compute.signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Signal stress - many dispatches single signal",
          "[device][signal][stress]") {
  auto &ctx = require_ctx();
  auto fix = make_fixture(ctx);
  REQUIRE_RESULT(fix);

  constexpr unsigned N = 100;
  for (unsigned i = 0; i < N; ++i)
    dispatch_nop(*fix);

  auto sig = kfd::Signal::create(ctx);
  REQUIRE_RESULT(sig);
  REQUIRE_RESULT(fix->compute.signal(*sig));
  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}

TEST_CASE("Signal stress - counting signal with NOP dispatches",
          "[device][signal][stress]") {
  auto &ctx = require_ctx();
  auto fix = make_fixture(ctx);
  REQUIRE_RESULT(fix);

  constexpr uint64_t COUNT = 20;
  auto sig = kfd::Signal::create(ctx, COUNT);
  REQUIRE_RESULT(sig);

  for (uint64_t i = 0; i < COUNT; ++i) {
    dispatch_nop(*fix);
    REQUIRE_RESULT(fix->compute.signal(*sig));
  }

  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}

TEST_CASE("Signal stress - two queues with NOP dispatches and shared signal",
          "[device][signal][stress]") {
  auto &ctx = require_ctx();
  auto fix = make_fixture(ctx);
  REQUIRE_RESULT(fix);

  auto compute2 = kfd::ComputeQueue::create(*fix->gpu);
  REQUIRE_RESULT(compute2);

  kfd::DispatchConfig cfg{.grid = {.x = 4}, .block = {.x = 64}};
  auto kernarg = fix->nop.make_kernargs(*fix->gpu, cfg);
  REQUIRE_RESULT(kernarg);

  auto sig = kfd::Signal::create(ctx, 2);
  REQUIRE_RESULT(sig);

  REQUIRE_RESULT(fix->compute.dispatch(fix->nop, cfg, *kernarg));
  REQUIRE_RESULT(fix->compute.signal(*sig));

  REQUIRE_RESULT(compute2->dispatch(fix->nop, cfg, *kernarg));
  REQUIRE_RESULT(compute2->signal(*sig));

  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}

TEST_CASE("Signal stress - repeated two-queue shared signal with dispatches",
          "[device][signal][stress]") {
  auto &ctx = require_ctx();
  auto fix = make_fixture(ctx);
  REQUIRE_RESULT(fix);

  auto compute2 = kfd::ComputeQueue::create(*fix->gpu);
  REQUIRE_RESULT(compute2);

  kfd::DispatchConfig cfg{.grid = {.x = 4}, .block = {.x = 64}};
  auto kernarg = fix->nop.make_kernargs(*fix->gpu, cfg);
  REQUIRE_RESULT(kernarg);

  auto sig = kfd::Signal::create(ctx, 2);
  REQUIRE_RESULT(sig);

  constexpr unsigned ROUNDS = 30;
  for (unsigned r = 0; r < ROUNDS; ++r) {
    INFO("round " << r);
    if (r > 0)
      REQUIRE_RESULT(sig->reset(2));

    REQUIRE_RESULT(fix->compute.dispatch(fix->nop, cfg, *kernarg));
    REQUIRE_RESULT(fix->compute.signal(*sig));

    REQUIRE_RESULT(compute2->dispatch(fix->nop, cfg, *kernarg));
    REQUIRE_RESULT(compute2->signal(*sig));

    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
  }
}

TEST_CASE("Signal stress - threaded dispatch+signal on shared queue",
          "[device][signal][stress][mt]") {
  auto &ctx = require_ctx();
  auto fix = make_fixture(ctx);
  REQUIRE_RESULT(fix);

  constexpr unsigned N_THREADS = 4;
  constexpr unsigned ITERS = 30;

  kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
  auto kernarg = fix->nop.make_kernargs(*fix->gpu, cfg);
  REQUIRE_RESULT(kernarg);

  std::vector<kfd::Signal> signals;
  signals.reserve(N_THREADS);
  for (unsigned i = 0; i < N_THREADS; ++i) {
    auto sig = kfd::Signal::create(ctx);
    REQUIRE_RESULT(sig);
    signals.push_back(std::move(*sig));
  }

  auto worker = [&](unsigned tid) {
    for (unsigned i = 0; i < ITERS; ++i) {
      REQUIRE_RESULT(signals[tid].reset());
      REQUIRE_RESULT(fix->compute.dispatch(fix->nop, cfg, *kernarg));
      REQUIRE_RESULT(fix->compute.signal(signals[tid]));
      REQUIRE_RESULT(
          signals[tid].wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    }
  };

  std::vector<std::thread> threads;
  threads.reserve(N_THREADS);
  for (unsigned t = 0; t < N_THREADS; ++t)
    threads.emplace_back(worker, t);
  for (auto &th : threads)
    th.join();
}

TEST_CASE("Signal stress - threaded counting signal from shared queue",
          "[device][signal][stress][mt]") {
  auto &ctx = require_ctx();
  auto fix = make_fixture(ctx);
  REQUIRE_RESULT(fix);

  constexpr unsigned N_THREADS = 4;

  kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 64}};
  auto kernarg = fix->nop.make_kernargs(*fix->gpu, cfg);
  REQUIRE_RESULT(kernarg);

  auto sig = kfd::Signal::create(ctx, N_THREADS);
  REQUIRE_RESULT(sig);

  auto worker = [&](unsigned) {
    REQUIRE_RESULT(fix->compute.dispatch(fix->nop, cfg, *kernarg));
    REQUIRE_RESULT(fix->compute.signal(*sig));
  };

  std::vector<std::thread> threads;
  threads.reserve(N_THREADS);
  for (unsigned t = 0; t < N_THREADS; ++t)
    threads.emplace_back(worker, t);
  for (auto &th : threads)
    th.join();

  REQUIRE_RESULT(sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
}
