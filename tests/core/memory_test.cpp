#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

using kfd::test::get_ctx;

TEST_CASE("Memory - GTT alloc round-trip", "[memory]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);
    CAPTURE(i, (*dev)->gpu_id());

    auto buf = kfd::Buffer::allocate(**dev, 4096, kfd::MemType::GTT,
                                     kfd::MemFlags::WRITABLE |
                                         kfd::MemFlags::HOST_ACCESS);
    REQUIRE_RESULT(buf);
    REQUIRE_RESULT(buf->map(**dev));
    CHECK(buf->data() != nullptr);
    CHECK(buf->size() >= 4096);
    CHECK(static_cast<bool>(*buf));

    auto *ptr = static_cast<uint32_t *>(buf->data());
    constexpr uint32_t count = 4096 / sizeof(uint32_t);
    for (uint32_t j = 0; j < count; ++j)
      ptr[j] = j;
    for (uint32_t j = 0; j < count; ++j)
      CHECK(ptr[j] == j);
  }
}

TEST_CASE("Memory - VRAM host-visible alloc round-trip", "[memory]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);

    if (!(*dev)->vram_host_visible())
      SKIP("Device does not support host-visible VRAM (large BAR disabled)");
    CAPTURE(i, (*dev)->gpu_id());

    auto buf = kfd::Buffer::allocate(**dev, 4096, kfd::MemType::VRAM,
                                     kfd::MemFlags::WRITABLE |
                                         kfd::MemFlags::HOST_ACCESS);
    REQUIRE_RESULT(buf);
    REQUIRE_RESULT(buf->map(**dev));
    CHECK(buf->data() != nullptr);

    auto *ptr = static_cast<uint32_t *>(buf->data());
    ptr[0] = 0xcafebabe;
    CHECK(ptr[0] == 0xcafebabe);
  }
}

TEST_CASE("Memory - large GTT allocation", "[memory]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);
    CAPTURE(i, (*dev)->gpu_id());

    constexpr size_t size = 1u << 20;
    auto buf = kfd::Buffer::allocate(**dev, size, kfd::MemType::GTT,
                                     kfd::MemFlags::WRITABLE |
                                         kfd::MemFlags::HOST_ACCESS);
    REQUIRE_RESULT(buf);
    REQUIRE_RESULT(buf->map(**dev));
    CHECK(buf->size() >= size);

    std::memset(buf->data(), 0xab, buf->size());
    auto *bytes = static_cast<unsigned char *>(buf->data());
    CHECK(bytes[0] == 0xab);
    CHECK(bytes[buf->size() - 1] == 0xab);
  }
}

TEST_CASE("Memory - multiple allocations coexist", "[memory]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);
    CAPTURE(i, (*dev)->gpu_id());

    auto a = kfd::Buffer::allocate(**dev, 4096, kfd::MemType::GTT,
                                   kfd::MemFlags::WRITABLE |
                                       kfd::MemFlags::HOST_ACCESS);
    auto b = kfd::Buffer::allocate(**dev, 4096, kfd::MemType::GTT,
                                   kfd::MemFlags::WRITABLE |
                                       kfd::MemFlags::HOST_ACCESS);
    REQUIRE_RESULT(a);
    REQUIRE_RESULT(b);
    REQUIRE_RESULT(a->map(**dev));
    REQUIRE_RESULT(b->map(**dev));

    CHECK(a->data() != b->data());

    static_cast<uint32_t *>(a->data())[0] = 1;
    static_cast<uint32_t *>(b->data())[0] = 2;
    CHECK(static_cast<uint32_t *>(a->data())[0] == 1);
    CHECK(static_cast<uint32_t *>(b->data())[0] == 2);
  }
}

TEST_CASE("Memory - DMA-BUF export", "[memory][dmabuf]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);
    CAPTURE(i, (*dev)->gpu_id());

    auto buf = kfd::Buffer::allocate(**dev, 4096, kfd::MemType::GTT,
                                     kfd::MemFlags::WRITABLE |
                                         kfd::MemFlags::HOST_ACCESS);
    REQUIRE_RESULT(buf);
    REQUIRE_RESULT(buf->map(**dev));

    auto dmabuf = kfd::DMABuffer::create(*buf);
    REQUIRE_RESULT(dmabuf);
    CHECK(dmabuf->fd() >= 0);

    // DMA-BUF fds support lseek to query the backing size.
    off_t size = ::lseek(dmabuf->fd(), 0, SEEK_END);
    CHECK(size >= 4096);

    // Multiple exports from the same buffer yield independent fds.
    auto second = kfd::DMABuffer::create(*buf);
    REQUIRE_RESULT(second);
    CHECK(second->fd() != dmabuf->fd());

    // Verify the fd is actually closed when the DmaBuf is destroyed.
    int saved_fd = dmabuf->fd();
    dmabuf = {};
    CHECK(::fcntl(saved_fd, F_GETFD) == -1);
  }
}
