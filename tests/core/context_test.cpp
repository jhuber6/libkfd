#include "test_helpers.h"

#include "libkfd/detail/elf.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>

TEST_CASE("Context - opens /dev/kfd and reports version", "[context]") {
  auto ctx = kfd::Context::create();
  if (!ctx)
    SKIP("KFD not available: " << kfd::strerror(ctx.error()));

  CHECK(ctx->kfd_fd() >= 0);

  auto ver = ctx->version();
  REQUIRE_RESULT(ver);
  CHECK(ver->major >= 1);
}

TEST_CASE("Context - discovers at least one GPU", "[context]") {
  auto ctx = kfd::Context::create();
  if (!ctx)
    SKIP("KFD not available");

  CHECK(ctx->num_devices() > 0);
}

TEST_CASE("Context - device initializes lazily on first access", "[context]") {
  auto ctx = kfd::Context::create();
  if (!ctx)
    SKIP("KFD not available");
  if (ctx->num_devices() == 0)
    SKIP("No GPUs in topology");

  auto dev = ctx->device(0);
  REQUIRE_RESULT(dev);
  CHECK((*dev)->render_fd() >= 0);
  CHECK((*dev)->gpu_id() != 0);
}

TEST_CASE("Context - move construction preserves state", "[context]") {
  auto result = kfd::Context::create();
  if (!result)
    SKIP("KFD not available: " << kfd::strerror(result.error()));

  int original_fd = result->kfd_fd();
  size_t original_ndevs = result->num_devices();

  kfd::Context moved(std::move(*result));
  CHECK(moved.kfd_fd() == original_fd);
  CHECK(moved.num_devices() == original_ndevs);

  if (moved.num_devices() > 0) {
    auto dev = moved.device(0);
    REQUIRE_RESULT(dev);
    CHECK(&(*dev)->context() == &moved);
  }
}

TEST_CASE("Context - move assignment preserves state", "[context]") {
  auto first = kfd::Context::create();
  if (!first)
    SKIP("KFD not available: " << kfd::strerror(first.error()));

  auto second = kfd::Context::create();
  if (!second)
    SKIP("Second context failed: " << kfd::strerror(second.error()));

  int second_fd = second->kfd_fd();
  size_t second_ndevs = second->num_devices();

  *first = std::move(*second);
  CHECK(first->kfd_fd() == second_fd);
  CHECK(first->num_devices() == second_ndevs);

  if (first->num_devices() > 0) {
    auto dev = first->device(0);
    REQUIRE_RESULT(dev);
    CHECK(&(*dev)->context() == &*first);
  }
}

TEST_CASE("Context - multiple contexts coexist", "[context]") {
  auto a = kfd::Context::create();
  if (!a)
    SKIP("KFD not available: " << kfd::strerror(a.error()));

  auto b = kfd::Context::create();
  REQUIRE_RESULT(b);

  CHECK(a->kfd_fd() != b->kfd_fd());
  CHECK(a->num_devices() == b->num_devices());

  if (a->num_devices() > 0) {
    auto dev_a = a->device(0);
    auto dev_b = b->device(0);
    REQUIRE_RESULT(dev_a);
    REQUIRE_RESULT(dev_b);
    CHECK(&(*dev_a)->context() == &*a);
    CHECK(&(*dev_b)->context() == &*b);
    CHECK((*dev_a)->gpu_id() == (*dev_b)->gpu_id());

    auto ver_a = a->version();
    auto ver_b = b->version();
    REQUIRE_RESULT(ver_a);
    REQUIRE_RESULT(ver_b);
    CHECK(ver_a->major == ver_b->major);
    CHECK(ver_a->minor == ver_b->minor);
  }
}

TEST_CASE("Context - node properties have reasonable values", "[context]") {
  auto ctx = kfd::Context::create();
  if (!ctx)
    SKIP("KFD not available");
  if (ctx->num_devices() == 0)
    SKIP("No GPUs in topology");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);

    const auto &node = (*dev)->properties();
    CAPTURE(node.gpu_id, node.device_id);
    CHECK(node.gpu_id != 0);
    CHECK(node.vendor_id != 0);
    CHECK(node.device_id != 0);
    CHECK(node.gfx_target_version != 0);
    CHECK(node.drm_render_minor != 0);

    using namespace kfd::detail::elf;
    uint32_t mach = get_mach(node.gfx_target_version);
    CHECK(mach != EF_AMDGPU_MACH_NONE);
    CHECK(get_gfx_version(mach) == node.gfx_target_version);

    char name[16] = {};
    format_gfx_version(name, sizeof(name), node.gfx_target_version);
    CHECK(std::string_view(name) == get_name(mach));
  }
}
