#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>

using kfd::test::get_ctx;

namespace {

bool svm_supported(kfd::Device &dev) {
  return (dev.properties().capability &
          kfd::NodeProperties::NODE_CAP_SVMAPI_SUPPORTED) != 0;
}

} // namespace

TEST_CASE("SVM - basic device access", "[svm]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);

    if (!svm_supported(**dev))
      SKIP("SVM not supported on device " << i);
    CAPTURE(i, (*dev)->gpu_id());

    auto region = kfd::detail::MappedRegion::create(kfd::detail::page_size());
    REQUIRE_RESULT(region);
    std::memset(region->data(), 0, region->size());

    kfd::Device *dev_ptr = *dev;
    REQUIRE_RESULT(
        kfd::svm_set_preferred_loc(*ctx, region->data(), region->size()));
    REQUIRE_RESULT(kfd::svm_set_flags(*ctx, region->data(), region->size(),
                                      kfd::SVMFlags::HOST_ACCESS |
                                          kfd::SVMFlags::GPU_ALWAYS_MAPPED));
    REQUIRE_RESULT(kfd::svm_set_access(*ctx, region->data(), region->size(),
                                       {&dev_ptr, 1}));
    REQUIRE_RESULT(
        kfd::svm_set_granularity(*ctx, region->data(), region->size(), 0xFF));
    REQUIRE_RESULT(
        kfd::svm_prefetch(*ctx, region->data(), region->size(), *dev));

    auto compute = kfd::test::create_queue<kfd::ComputeQueue>(**dev);
    REQUIRE_RESULT(compute);

    auto sig = kfd::Signal::create(*ctx);
    REQUIRE_RESULT(sig);

    auto *val = static_cast<uint32_t *>(region->data());
    *val = 0;

    REQUIRE_RESULT(compute->write_data(val, 0xDEADBEEF));
    REQUIRE_RESULT(compute->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

    CHECK(*val == 0xDEADBEEF);
  }
}

TEST_CASE("SVM - atomic increment via PM4", "[svm]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);

    if (!svm_supported(**dev))
      SKIP("SVM not supported on device " << i);
    CAPTURE(i, (*dev)->gpu_id());

    auto region = kfd::detail::MappedRegion::create(kfd::detail::page_size());
    REQUIRE_RESULT(region);
    std::memset(region->data(), 0, region->size());

    kfd::Device *dev_ptr = *dev;
    REQUIRE_RESULT(
        kfd::svm_set_preferred_loc(*ctx, region->data(), region->size()));
    REQUIRE_RESULT(kfd::svm_set_flags(*ctx, region->data(), region->size(),
                                      kfd::SVMFlags::HOST_ACCESS |
                                          kfd::SVMFlags::GPU_ALWAYS_MAPPED));
    REQUIRE_RESULT(kfd::svm_set_access(*ctx, region->data(), region->size(),
                                       {&dev_ptr, 1}));
    REQUIRE_RESULT(
        kfd::svm_set_granularity(*ctx, region->data(), region->size(), 0xFF));
    REQUIRE_RESULT(
        kfd::svm_prefetch(*ctx, region->data(), region->size(), *dev));

    auto compute = kfd::test::create_queue<kfd::ComputeQueue>(**dev);
    REQUIRE_RESULT(compute);

    auto sig = kfd::Signal::create(*ctx);
    REQUIRE_RESULT(sig);

    auto *counter = static_cast<int64_t *>(region->data());
    *counter = 0;

    constexpr int64_t GPU_INCREMENTS = 10;
    for (int64_t j = 0; j < GPU_INCREMENTS; ++j)
      REQUIRE_RESULT(
          compute->atomic_mem(kfd::pm4::ATOMIC_ADD_RTN_64, counter, 1));

    REQUIRE_RESULT(compute->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

    CHECK(*counter == GPU_INCREMENTS);
  }
}

TEST_CASE("SVM - preferred location to system memory", "[svm]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);

    if (!svm_supported(**dev))
      SKIP("SVM not supported on device " << i);
    CAPTURE(i, (*dev)->gpu_id());

    auto region = kfd::detail::MappedRegion::create(kfd::detail::page_size());
    REQUIRE_RESULT(region);
    std::memset(region->data(), 0, region->size());

    kfd::Device *dev_ptr = *dev;
    REQUIRE_RESULT(
        kfd::svm_set_preferred_loc(*ctx, region->data(), region->size()));
    REQUIRE_RESULT(kfd::svm_set_flags(*ctx, region->data(), region->size(),
                                      kfd::SVMFlags::HOST_ACCESS |
                                          kfd::SVMFlags::GPU_ALWAYS_MAPPED));
    REQUIRE_RESULT(kfd::svm_set_access(*ctx, region->data(), region->size(),
                                       {&dev_ptr, 1}));
    REQUIRE_RESULT(
        kfd::svm_prefetch(*ctx, region->data(), region->size(), *dev));

    auto compute = kfd::test::create_queue<kfd::ComputeQueue>(**dev);
    REQUIRE_RESULT(compute);

    auto sig = kfd::Signal::create(*ctx);
    REQUIRE_RESULT(sig);

    auto *val = static_cast<int64_t *>(region->data());
    *val = 42;

    REQUIRE_RESULT(compute->atomic_mem(kfd::pm4::ATOMIC_ADD_RTN_64, val, 1));
    REQUIRE_RESULT(compute->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

    CHECK(*val == 43);
  }
}

TEST_CASE("SVM - bidirectional CPU and GPU atomics", "[svm]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);

    if (!svm_supported(**dev))
      SKIP("SVM not supported on device " << i);
    CAPTURE(i, (*dev)->gpu_id());

    auto region = kfd::detail::MappedRegion::create(kfd::detail::page_size());
    REQUIRE_RESULT(region);
    std::memset(region->data(), 0, region->size());

    kfd::Device *dev_ptr = *dev;
    REQUIRE_RESULT(kfd::svm_set_flags(*ctx, region->data(), region->size(),
                                      kfd::SVMFlags::HOST_ACCESS |
                                          kfd::SVMFlags::GPU_ALWAYS_MAPPED |
                                          kfd::SVMFlags::COHERENT));
    REQUIRE_RESULT(kfd::svm_set_access(*ctx, region->data(), region->size(),
                                       {&dev_ptr, 1}));
    REQUIRE_RESULT(
        kfd::svm_prefetch(*ctx, region->data(), region->size(), *dev));

    auto compute = kfd::test::create_queue<kfd::ComputeQueue>(**dev);
    REQUIRE_RESULT(compute);

    auto sig = kfd::Signal::create(*ctx);
    REQUIRE_RESULT(sig);

    auto *counter = static_cast<int64_t *>(region->data());
    *counter = 0;

    constexpr int64_t CPU_ADD = 5;
    constexpr int64_t GPU_ADD = 7;
    __atomic_fetch_add(counter, CPU_ADD, __ATOMIC_SEQ_CST);

    REQUIRE_RESULT(
        compute->atomic_mem(kfd::pm4::ATOMIC_ADD_RTN_64, counter, GPU_ADD));
    REQUIRE_RESULT(compute->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

    CHECK(*counter == CPU_ADD + GPU_ADD);
  }
}
