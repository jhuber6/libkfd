#include "libkfd/detail/small_vector.h"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>

using kfd::detail::SmallVector;

namespace {

struct MoveOnly {
  int value;
  explicit MoveOnly(int v = 0) : value(v) {}
  MoveOnly(MoveOnly &&o) noexcept : value(o.value) { o.value = -1; }
  MoveOnly &operator=(MoveOnly &&o) noexcept {
    value = o.value;
    o.value = -1;
    return *this;
  }
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly &operator=(const MoveOnly &) = delete;
};

} // namespace

TEST_CASE("SmallVector - default construction", "[small_vector]") {
  SmallVector<int, 4> v;
  REQUIRE(v.size() == 0);
  REQUIRE(v.empty());
  REQUIRE(v.begin() == v.end());
}

TEST_CASE("SmallVector - push_back within inline capacity", "[small_vector]") {
  SmallVector<MoveOnly, 4> v;
  for (int i = 0; i < 4; ++i)
    REQUIRE(v.push_back(MoveOnly(i)));

  REQUIRE(v.size() == 4);
  for (int i = 0; i < 4; ++i)
    CHECK(v[static_cast<size_t>(i)].value == i);
}

TEST_CASE("SmallVector - push_back spills to heap", "[small_vector]") {
  SmallVector<MoveOnly, 2> v;
  for (int i = 0; i < 10; ++i)
    REQUIRE(v.push_back(MoveOnly(i)));

  REQUIRE(v.size() == 10);
  for (int i = 0; i < 10; ++i)
    CHECK(v[static_cast<size_t>(i)].value == i);
}

TEST_CASE("SmallVector - emplace_back", "[small_vector]") {
  SmallVector<MoveOnly, 4> v;
  auto ref = v.emplace_back(42);
  REQUIRE(ref.has_value());
  REQUIRE(v.size() == 1);
  CHECK((*ref)->value == 42);
  CHECK(v[0].value == 42);
}

TEST_CASE("SmallVector - iteration", "[small_vector]") {
  SmallVector<MoveOnly, 4> v;
  for (int i = 0; i < 3; ++i)
    REQUIRE(v.push_back(MoveOnly(i * 10)));

  int expected = 0;
  for (auto &elem : v) {
    CHECK(elem.value == expected);
    expected += 10;
  }
}

TEST_CASE("SmallVector - clear", "[small_vector]") {
  SmallVector<MoveOnly, 4> v;
  for (int i = 0; i < 3; ++i)
    REQUIRE(v.push_back(MoveOnly(i)));

  v.clear();
  REQUIRE(v.size() == 0);
  REQUIRE(v.empty());

  REQUIRE(v.push_back(MoveOnly(99)));
  REQUIRE(v.size() == 1);
  CHECK(v[0].value == 99);
}

TEST_CASE("SmallVector - move construction inline", "[small_vector]") {
  SmallVector<MoveOnly, 4> a;
  for (int i = 0; i < 3; ++i)
    REQUIRE(a.push_back(MoveOnly(i)));

  SmallVector<MoveOnly, 4> b(std::move(a));

  REQUIRE(b.size() == 3);
  for (int i = 0; i < 3; ++i)
    CHECK(b[static_cast<size_t>(i)].value == i);

  REQUIRE(a.size() == 0);
  REQUIRE(a.empty());
}

TEST_CASE("SmallVector - move construction heap", "[small_vector]") {
  SmallVector<MoveOnly, 2> a;
  for (int i = 0; i < 8; ++i)
    REQUIRE(a.push_back(MoveOnly(i)));

  SmallVector<MoveOnly, 2> b(std::move(a));

  REQUIRE(b.size() == 8);
  for (int i = 0; i < 8; ++i)
    CHECK(b[static_cast<size_t>(i)].value == i);

  REQUIRE(a.size() == 0);
}

TEST_CASE("SmallVector - move assignment inline to inline", "[small_vector]") {
  SmallVector<MoveOnly, 4> a;
  REQUIRE(a.push_back(MoveOnly(1)));
  REQUIRE(a.push_back(MoveOnly(2)));

  SmallVector<MoveOnly, 4> b;
  REQUIRE(b.push_back(MoveOnly(99)));

  b = std::move(a);

  REQUIRE(b.size() == 2);
  CHECK(b[0].value == 1);
  CHECK(b[1].value == 2);
  REQUIRE(a.size() == 0);
}

TEST_CASE("SmallVector - move assignment heap to empty", "[small_vector]") {
  SmallVector<MoveOnly, 2> a;
  for (int i = 0; i < 6; ++i)
    REQUIRE(a.push_back(MoveOnly(i)));

  SmallVector<MoveOnly, 2> b;
  b = std::move(a);

  REQUIRE(b.size() == 6);
  for (int i = 0; i < 6; ++i)
    CHECK(b[static_cast<size_t>(i)].value == i);
  REQUIRE(a.size() == 0);
}
