#include "test_helpers.h"

#include "libkfd/error.h"

#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstring>
#include <string_view>

using namespace kfd;

TEST_CASE("Error - from errno code", "[error]") {
  Error e(ENOENT);
  CHECK(e.code == ENOENT);
  CHECK(static_cast<int>(e) == ENOENT);
  CHECK(e.msg[0] == '\0');
  CHECK(std::string_view(kfd::strerror(e)) == std::strerror(ENOENT));
}

TEST_CASE("Error - unexpected with format appends strerror", "[error]") {
  auto u = kfd::unexpected(EACCES, "open '%s'", "/dev/kfd");
  CHECK(u.error().code == EACCES);
  std::string_view msg(u.error().msg);
  CHECK(msg.find("open '/dev/kfd'") != std::string_view::npos);
  CHECK(msg.find(std::strerror(EACCES)) != std::string_view::npos);
}

TEST_CASE("Error - unexpected from bare errno", "[error]") {
  auto u = kfd::unexpected(ENOMEM);
  CHECK(u.error().code == ENOMEM);
  CHECK(u.error().msg[0] == '\0');
}

TEST_CASE("Error - unexpected from Error object", "[error]") {
  Error e(EPERM);
  auto u = kfd::unexpected(e);
  CHECK(u.error().code == EPERM);
}

TEST_CASE("Error - unexpected with format string", "[error]") {
  auto u = kfd::unexpected(EFAULT, "VA 0x%lx outside aperture", 0xDEADUL);
  CHECK(u.error().code == EFAULT);
  CHECK(std::string_view(u.error().msg).find("0xdead") !=
        std::string_view::npos);
}

TEST_CASE("Error - std::expected success and failure", "[error]") {
  std::expected<int, Error> ok = 42;
  CHECK_RESULT(ok);
  CHECK(std::string_view(kfd::strerror(ok)) == "(success)");

  std::expected<int, Error> err = kfd::unexpected(EINVAL, "bad index %u", 7u);
  CHECK_FALSE(err.has_value());
  CHECK(err.error().code == EINVAL);
  CHECK(std::string_view(kfd::strerror(err)).find("bad index 7") !=
        std::string_view::npos);
}

TEST_CASE("Error - std::expected<void, Error>", "[error]") {
  std::expected<void, Error> ok;
  CHECK_RESULT(ok);

  std::expected<void, Error> err = kfd::unexpected(EIO);
  CHECK_FALSE(err.has_value());
  CHECK(err.error().code == EIO);
}
