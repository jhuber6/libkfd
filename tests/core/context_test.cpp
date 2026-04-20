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
