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

    // Verify the fd is actually closed when the DMABuffer is destroyed.
    int saved_fd = dmabuf->fd();
    dmabuf = {};
    CHECK(::fcntl(saved_fd, F_GETFD) == -1);
  }
}

TEST_CASE("Memory - DMA-BUF import", "[memory][dmabuf]") {
  auto *ctx = get_ctx();
  if (!ctx)
    SKIP("KFD not available");

  auto bad = kfd::DMABuffer::info(*ctx, -1);
  CHECK(!bad);
  CHECK(bad.error().code == EINVAL);

  for (size_t i = 0; i < ctx->num_devices(); ++i) {
    auto dev = ctx->device(i);
    REQUIRE_RESULT(dev);
    CAPTURE(i, (*dev)->gpu_id());
    kfd::Device &gpu = **dev;

    constexpr uint32_t N = 4096 / sizeof(uint32_t);
    auto src = kfd::Buffer::allocate(gpu, 4096, kfd::MemType::GTT,
                                     kfd::MemFlags::WRITABLE |
                                         kfd::MemFlags::HOST_ACCESS);
    REQUIRE_RESULT(src);
    REQUIRE_RESULT(src->map(gpu));
    auto *words = static_cast<uint32_t *>(src->data());
    for (uint32_t j = 0; j < N; ++j)
      words[j] = j;

    auto exported = kfd::DMABuffer::create(*src);
    REQUIRE_RESULT(exported);
    auto info = kfd::DMABuffer::info(*ctx, exported->fd());
    REQUIRE_RESULT(info);
    CHECK(info->size == src->size());
    CHECK(info->gpu_id == gpu.gpu_id());
    CHECK(info->type == kfd::MemType::GTT);

    auto imported = kfd::Buffer::import(*ctx, exported->fd());
    REQUIRE_RESULT(imported);
    CHECK(::fcntl(exported->fd(), F_GETFD) != -1);
    CHECK(imported->data() != src->data());
    CHECK(imported->size() == src->size());
    CHECK(std::memcmp(imported->data(), src->data(), src->size()) == 0);

    static_cast<uint32_t *>(imported->data())[0] = 0x11111111;
    CHECK(words[0] == 0x11111111);

    REQUIRE_RESULT(imported->map(gpu));
    auto dst = kfd::Buffer::allocate(gpu, src->size(), kfd::MemType::GTT,
                                     kfd::MemFlags::WRITABLE |
                                         kfd::MemFlags::HOST_ACCESS);
    REQUIRE_RESULT(dst);
    REQUIRE_RESULT(dst->map(gpu));
    std::memset(dst->data(), 0, dst->size());

    auto queue = kfd::test::create_queue<kfd::SDMAQueue>(gpu);
    REQUIRE_RESULT(queue);
    auto sig = kfd::Signal::create(*ctx);
    REQUIRE_RESULT(sig);
    REQUIRE_RESULT(
        queue->copy_linear(dst->data(), imported->data(), src->size()));
    REQUIRE_RESULT(queue->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    CHECK(static_cast<uint32_t *>(dst->data())[0] == 0x11111111);
    CHECK(std::memcmp(static_cast<uint32_t *>(dst->data()) + 1, words + 1,
                      (N - 1) * sizeof(uint32_t)) == 0);

    exported = {};
    CHECK(static_cast<uint32_t *>(imported->data())[0] == 0x11111111);

    if (!gpu.vram_host_visible())
      continue;

    auto vram = kfd::Buffer::allocate(gpu, 4096, kfd::MemType::VRAM,
                                      kfd::MemFlags::WRITABLE |
                                          kfd::MemFlags::HOST_ACCESS |
                                          kfd::MemFlags::NO_SUBSTITUTE);
    REQUIRE_RESULT(vram);
    auto vram_fd = kfd::DMABuffer::create(*vram);
    REQUIRE_RESULT(vram_fd);
    auto vram_info = kfd::DMABuffer::info(*ctx, vram_fd->fd());
    REQUIRE_RESULT(vram_info);
    CHECK(vram_info->type == kfd::MemType::VRAM);
    CHECK(vram_info->gpu_id == gpu.gpu_id());
    auto vram_imported = kfd::Buffer::import(*ctx, vram_fd->fd());
    REQUIRE_RESULT(vram_imported);
    static_cast<uint32_t *>(vram->data())[0] = 0x22222222;
    CHECK(static_cast<uint32_t *>(vram_imported->data())[0] == 0x22222222);

    auto hidden = kfd::Buffer::allocate(gpu, 4096, kfd::MemType::VRAM,
                                        kfd::MemFlags::WRITABLE |
                                            kfd::MemFlags::NO_SUBSTITUTE);
    REQUIRE_RESULT(hidden);
    REQUIRE_RESULT(hidden->map(gpu));
    auto hidden_fd = kfd::DMABuffer::create(*hidden);
    REQUIRE_RESULT(hidden_fd);
    auto hidden_info = kfd::DMABuffer::info(*ctx, hidden_fd->fd());
    REQUIRE_RESULT(hidden_info);
    CHECK(hidden_info->type == kfd::MemType::VRAM);
    auto hidden_imported = kfd::Buffer::import(*ctx, hidden_fd->fd());
    REQUIRE_RESULT(hidden_imported);
    REQUIRE_RESULT(hidden_imported->map(gpu));

    auto scratch = kfd::Buffer::allocate(gpu, 4096, kfd::MemType::GTT,
                                         kfd::MemFlags::WRITABLE |
                                             kfd::MemFlags::HOST_ACCESS);
    REQUIRE_RESULT(scratch);
    REQUIRE_RESULT(scratch->map(gpu));
    std::memset(scratch->data(), 0, scratch->size());
    REQUIRE_RESULT(sig->reset(1));
    REQUIRE_RESULT(queue->const_fill(hidden->data(), 0xA5A5A5A5, 4096));
    REQUIRE_RESULT(queue->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    REQUIRE_RESULT(sig->reset(1));
    REQUIRE_RESULT(
        queue->copy_linear(scratch->data(), hidden_imported->data(), 4096));
    REQUIRE_RESULT(queue->signal(*sig));
    REQUIRE_RESULT(
        sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));
    CHECK(static_cast<uint32_t *>(scratch->data())[0] == 0xA5A5A5A5);
  }
}
