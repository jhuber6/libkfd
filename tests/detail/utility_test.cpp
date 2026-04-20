#include "libkfd/detail/utility.h"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <string_view>

using namespace kfd::detail;

TEST_CASE("consume_integer - basic decimal", "[utility]") {
  std::string_view s = "12345";
  auto val = consume_integer(s);
  REQUIRE(val.has_value());
  CHECK(*val == 12345);
  CHECK(s.empty());
}

TEST_CASE("consume_integer - stops at non-digit", "[utility]") {
  std::string_view s = "42abc";
  auto val = consume_integer(s);
  REQUIRE(val.has_value());
  CHECK(*val == 42);
  CHECK(s == "abc");
}

TEST_CASE("consume_integer - trailing newline", "[utility]") {
  std::string_view s = "6530\n";
  auto val = consume_integer(s);
  REQUIRE(val.has_value());
  CHECK(*val == 6530);
  CHECK(s == "\n");
}

TEST_CASE("consume_integer - zero", "[utility]") {
  std::string_view s = "0";
  auto val = consume_integer(s);
  REQUIRE(val.has_value());
  CHECK(*val == 0);
  CHECK(s.empty());
}

TEST_CASE("consume_integer - empty string fails", "[utility]") {
  std::string_view s;
  CHECK_FALSE(consume_integer(s).has_value());
}

TEST_CASE("consume_integer - no leading digits fails", "[utility]") {
  std::string_view s = "abc";
  CHECK_FALSE(consume_integer(s).has_value());
  CHECK(s == "abc");
}

TEST_CASE("consume_integer - large 64-bit value", "[utility]") {
  std::string_view s = "6737903363342137313";
  auto val = consume_integer(s);
  REQUIRE(val.has_value());
  CHECK(*val == 6737903363342137313ULL);
}

TEST_CASE("slice - extracts middle portion", "[utility]") {
  std::string_view s = "hello world";
  CHECK(slice(s, 6, 5) == "world");
}

TEST_CASE("slice_from - extracts tail", "[utility]") {
  std::string_view s = "key value";
  CHECK(slice_from(s, 4) == "value");
}

TEST_CASE("consume_integer - sysfs property line pattern", "[utility]") {
  std::string_view line = "4098";
  auto val = consume_integer(line);
  REQUIRE(val.has_value());
  CHECK(*val == 4098);
}

TEST_CASE("split - key value pair", "[utility]") {
  auto [key, val] = split("vendor_id 4098", ' ');
  CHECK(key == "vendor_id");
  CHECK(val == "4098");
}

TEST_CASE("split - no separator returns whole string", "[utility]") {
  auto [first, second] = split("nosep", ' ');
  CHECK(first == "nosep");
  CHECK(second.empty());
}

TEST_CASE("consume_line - single line", "[utility]") {
  std::string_view text = "hello";
  auto line = consume_line(text);
  CHECK(line == "hello");
  CHECK(text.empty());
}

TEST_CASE("consume_line - multiple lines", "[utility]") {
  std::string_view text = "aaa\nbbb\nccc";
  CHECK(consume_line(text) == "aaa");
  CHECK(consume_line(text) == "bbb");
  CHECK(consume_line(text) == "ccc");
  CHECK(text.empty());
}

TEST_CASE("consume_line - trailing newline", "[utility]") {
  std::string_view text = "line\n";
  CHECK(consume_line(text) == "line");
  CHECK(text.empty());
}

TEST_CASE("consume_front - match", "[utility]") {
  std::string_view s = "/path";
  CHECK(consume_front(s, '/'));
  CHECK(s == "path");
}

TEST_CASE("consume_front - no match", "[utility]") {
  std::string_view s = "path";
  CHECK_FALSE(consume_front(s, '/'));
  CHECK(s == "path");
}

TEST_CASE("take_while - digits", "[utility]") {
  std::string_view s = "123abc";
  auto digits = take_while(s, [](char c) { return c >= '0' && c <= '9'; });
  CHECK(digits == "123");
}

TEST_CASE("take_until - space", "[utility]") {
  std::string_view s = "key value";
  auto word = take_until(s, [](char c) { return c == ' '; });
  CHECK(word == "key");
}

TEST_CASE("drop_front/drop_back - basic trimming", "[utility]") {
  std::string_view s = "hello";
  CHECK(drop_front(s, 2) == "llo");
  CHECK(drop_back(s, 2) == "hel");
  CHECK(drop_front(s, 10).empty());
  CHECK(drop_back(s, 10).empty());
}

TEST_CASE("consume_line + split - sysfs properties pattern", "[utility]") {
  std::string_view text = "vendor_id 4098\ndevice_id 29605\n";

  auto line1 = consume_line(text);
  auto [k1, v1] = split(line1, ' ');
  CHECK(k1 == "vendor_id");
  CHECK(v1 == "4098");

  auto line2 = consume_line(text);
  auto [k2, v2] = split(line2, ' ');
  CHECK(k2 == "device_id");
  CHECK(v2 == "29605");

  CHECK(text.empty());
}
