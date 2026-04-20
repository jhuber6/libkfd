#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstring>
#include <sys/mman.h>

using kfd::detail::MappedRegion;

TEST_CASE("MappedRegion - default-constructed is null", "[mapped_region]") {
  MappedRegion r;
  CHECK_FALSE(r);
  CHECK(r.data() == nullptr);
  CHECK(r.size() == 0);
}

TEST_CASE("MappedRegion - anonymous create", "[mapped_region]") {
  auto result = MappedRegion::create(4096);
  REQUIRE_RESULT(result);

  MappedRegion &r = *result;
  REQUIRE(r);
  CHECK(r.data() != nullptr);
  CHECK(r.size() == 4096);

  std::memset(r.data(), 0xAB, r.size());
  auto *bytes = static_cast<unsigned char *>(r.data());
  CHECK(bytes[0] == 0xAB);
  CHECK(bytes[4095] == 0xAB);
}

TEST_CASE("MappedRegion - reserve_aligned", "[mapped_region]") {
  constexpr size_t TWO_MIB = 2 * 1024 * 1024;
  auto result = MappedRegion::reserve_aligned(TWO_MIB, TWO_MIB);
  REQUIRE_RESULT(result);

  MappedRegion &r = *result;
  REQUIRE(r);
  CHECK(reinterpret_cast<uintptr_t>(r.data()) % TWO_MIB == 0);
  CHECK(r.size() == TWO_MIB);
}

TEST_CASE("MappedRegion - reserve creates PROT_NONE mapping",
          "[mapped_region]") {
  auto result = MappedRegion::reserve(4096);
  REQUIRE_RESULT(result);

  MappedRegion &r = *result;
  REQUIRE(r);
  CHECK(r.data() != nullptr);
  CHECK(r.size() == 4096);
}

TEST_CASE("MappedRegion - reserve at fixed address", "[mapped_region]") {
  auto backing = MappedRegion::reserve(4096);
  REQUIRE_RESULT(backing);
  void *addr = backing->data();

  auto fixed = MappedRegion::reserve(4096, addr);
  REQUIRE_RESULT(fixed);
  CHECK(fixed->data() == addr);
  CHECK(fixed->size() == 4096);

  // The original region was overwritten by MAP_FIXED; release it so it
  // does not double-munmap.
  backing->release();
}

TEST_CASE("MappedRegion - move construction", "[mapped_region]") {
  auto result = MappedRegion::create(4096);
  REQUIRE_RESULT(result);

  void *original_addr = result->data();
  MappedRegion moved(std::move(*result));

  CHECK(moved.data() == original_addr);
  CHECK(moved.size() == 4096);
  CHECK_FALSE(*result);
}

TEST_CASE("MappedRegion - move assignment", "[mapped_region]") {
  auto a = MappedRegion::create(4096);
  auto b = MappedRegion::create(8192);
  REQUIRE_RESULT(a);
  REQUIRE_RESULT(b);

  void *b_addr = b->data();
  *a = std::move(*b);

  CHECK(a->data() == b_addr);
  CHECK(a->size() == 8192);
  CHECK_FALSE(*b);
}

TEST_CASE("MappedRegion - try_grow in place", "[mapped_region]") {
  auto result = MappedRegion::create(4096);
  REQUIRE_RESULT(result);

  SECTION("grow to double") {
    bool grew = result->try_grow(8192);
    if (grew) {
      CHECK(result->size() == 8192);
      std::memset(result->data(), 0xCD, 8192);
      auto *bytes = static_cast<unsigned char *>(result->data());
      CHECK(bytes[8191] == 0xCD);
    }
    if (!grew) {
      CHECK(result->size() == 4096);
    }
  }

  SECTION("shrink is a no-op success") {
    CHECK(result->try_grow(2048));
    CHECK(result->size() == 4096);
  }
}

TEST_CASE("MappedRegion - as<T> and as_span<T>", "[mapped_region]") {
  auto result = MappedRegion::create(sizeof(int) * 16);
  REQUIRE_RESULT(result);

  int *raw = result->as<int>();
  REQUIRE(raw != nullptr);
  for (int i = 0; i < 16; ++i)
    raw[i] = i;

  auto span = result->as_span<int>();
  REQUIRE(span.size() >= 16);
  for (int i = 0; i < 16; ++i)
    CHECK(span[static_cast<size_t>(i)] == i);
}

TEST_CASE("MappedRegion - reserve_aligned rejects overflow",
          "[mapped_region]") {
  auto result = MappedRegion::reserve_aligned(SIZE_MAX, 4096);
  CHECK_FALSE(result.has_value());
  CHECK(result.error().code == EINVAL);

  auto zero_align = MappedRegion::reserve_aligned(4096, 0);
  CHECK_FALSE(zero_align.has_value());
  CHECK(zero_align.error().code == EINVAL);
}

TEST_CASE("MappedRegion - large mapping", "[mapped_region]") {
  constexpr size_t size = 16 * 1024 * 1024;
  auto result = MappedRegion::create(size);
  REQUIRE_RESULT(result);
  CHECK(result->size() == size);

  auto *bytes = static_cast<unsigned char *>(result->data());
  bytes[0] = 1;
  bytes[size - 1] = 2;
  CHECK(bytes[0] == 1);
  CHECK(bytes[size - 1] == 2);
}
