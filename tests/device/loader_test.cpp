#include "test_helpers.h"

#include "libkfd/abi.h"
#include "libkfd/loader.h"

#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Auto-generated table of all compiled device test kernels.
static const kfd::test::TestBinary device_kernels[] = {
#include "kernel_kernels.inc"
};

using kfd::test::make_device_fixture;
using kfd::test::read_file;
using kfd::test::require_ctx;
using kfd::test::require_gpu;

TEST_CASE("Loader - load succeeds for compatible arch", "[device][loader]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, device_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      INFO("arch: " << fix->bin->arch);
      auto buf = read_file(fix->bin->path);
      auto loaded = kfd::Executable::load(*fix->gpu, buf, fix->compute);
      REQUIRE_RESULT(loaded);
      CHECK(loaded->base() != nullptr);
      CHECK(loaded->size() > 0);
      CHECK(static_cast<bool>(*loaded));
    }
  }
}

TEST_CASE("Loader - symbol lookup finds known globals", "[device][loader]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, device_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto buf = read_file(fix->bin->path);
      auto loaded = kfd::Executable::load(*fix->gpu, buf, fix->compute);
      REQUIRE_RESULT(loaded);

      for (auto name : {"x", "y", "z", "bss_arr"}) {
        INFO("symbol: " << name);
        auto sym = loaded->symbol(name);
        REQUIRE_RESULT(sym);
      }
    }
  }
}

TEST_CASE("Loader - symbol lookup returns error for missing symbol",
          "[device][loader]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, device_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto buf = read_file(fix->bin->path);
      auto loaded = kfd::Executable::load(*fix->gpu, buf, fix->compute);
      REQUIRE_RESULT(loaded);

      auto sym = loaded->symbol("nonexistent_symbol_12345");
      CHECK_FALSE(sym.has_value());
    }
  }
}

TEST_CASE("Loader - global values readable via DMA", "[device][loader]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, device_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto buf = read_file(fix->bin->path);
      auto loaded = kfd::Executable::load(*fix->gpu, buf, fix->compute);
      REQUIRE_RESULT(loaded);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto readback = kfd::Buffer::allocate(
          *fix->gpu, kfd::detail::page_size(), kfd::MemType::GTT,
          kfd::MemFlags::WRITABLE | kfd::MemFlags::UNCACHED |
              kfd::MemFlags::HOST_ACCESS);
      REQUIRE_RESULT(readback);
      REQUIRE_RESULT(readback->map(*fix->gpu));

      struct {
        const char *name;
        unsigned expected;
      } globals[] = {
          {"x", 0xdeadbeef},
          {"y", 0xfeedface},
          {"z", 0xcafebabe},
      };

      bool first = true;
      for (const auto &[name, expected] : globals) {
        INFO("symbol: " << name);
        if (!first)
          REQUIRE_RESULT(sig->reset());
        first = false;
        auto sym = loaded->symbol(name);
        REQUIRE_RESULT(sym);

        std::memset(readback->data(), 0, sizeof(unsigned));
        REQUIRE_RESULT(fix->sdma.copy_linear(readback->data(), sym->data(),
                                             sizeof(unsigned)));
        REQUIRE_RESULT(fix->sdma.signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

        unsigned val;
        std::memcpy(&val, readback->data(), sizeof(val));
        CHECK(val == expected);
      }
    }
  }
}

TEST_CASE("Loader - BSS region is zeroed", "[device][loader]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, device_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto buf = read_file(fix->bin->path);
      auto loaded = kfd::Executable::load(*fix->gpu, buf, fix->compute);
      REQUIRE_RESULT(loaded);

      auto bss = loaded->symbol("bss_arr");
      REQUIRE_RESULT(bss);
      CHECK(bss->size() == 4096);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      constexpr uint32_t CHECK_BYTES = 256;
      auto readback = kfd::Buffer::allocate(
          *fix->gpu, kfd::detail::page_size(), kfd::MemType::GTT,
          kfd::MemFlags::WRITABLE | kfd::MemFlags::UNCACHED |
              kfd::MemFlags::HOST_ACCESS);
      REQUIRE_RESULT(readback);
      REQUIRE_RESULT(readback->map(*fix->gpu));

      std::memset(readback->data(), 0xFF, CHECK_BYTES);
      REQUIRE_RESULT(
          fix->sdma.copy_linear(readback->data(), bss->data(), CHECK_BYTES));
      REQUIRE_RESULT(fix->sdma.signal(*sig));
      REQUIRE_RESULT(
          sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

      auto *bytes = static_cast<const unsigned char *>(readback->data());
      for (uint32_t i = 0; i < CHECK_BYTES; ++i) {
        INFO("bss_arr[" << i << "]");
        CHECK(bytes[i] == 0);
      }
    }
  }
}

TEST_CASE("Loader - symbols() iterates dynamic symbol names",
          "[device][loader]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, device_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto buf = read_file(fix->bin->path);
      auto loaded = kfd::Executable::load(*fix->gpu, buf, fix->compute);
      REQUIRE_RESULT(loaded);

      auto range = loaded->symbols();
      CHECK_FALSE(range.empty());

      bool found_x = false;
      for (std::string_view name : range) {
        CHECK_FALSE(name.empty());
        if (name == "x")
          found_x = true;
      }
      CHECK(found_x);
    }
  }
}

TEST_CASE("Loader - R_AMDGPU_RELATIVE64 relocations applied",
          "[device][loader]") {
  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, device_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto buf = read_file(fix->bin->path);
      auto loaded = kfd::Executable::load(*fix->gpu, buf, fix->compute);
      REQUIRE_RESULT(loaded);

      auto sig = kfd::Signal::create(ctx);
      REQUIRE_RESULT(sig);

      auto readback = kfd::Buffer::allocate(
          *fix->gpu, kfd::detail::page_size(), kfd::MemType::GTT,
          kfd::MemFlags::WRITABLE | kfd::MemFlags::UNCACHED |
              kfd::MemFlags::HOST_ACCESS);
      REQUIRE_RESULT(readback);
      REQUIRE_RESULT(readback->map(*fix->gpu));

      auto base = reinterpret_cast<uintptr_t>(loaded->base());
      auto limit = base + loaded->size();

      SECTION("data pointer reloc_ptr_a points to reloc_target_a") {
        auto ptr_sym = loaded->symbol("reloc_ptr_a");
        REQUIRE_RESULT(ptr_sym);
        auto target_sym = loaded->symbol("reloc_target_a");
        REQUIRE_RESULT(target_sym);

        REQUIRE_RESULT(fix->sdma.copy_linear(readback->data(), ptr_sym->data(),
                                             sizeof(uint64_t)));
        REQUIRE_RESULT(fix->sdma.signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

        uint64_t val;
        std::memcpy(&val, readback->data(), sizeof(val));
        CHECK(val >= base);
        CHECK(val < limit);
        CHECK(val == reinterpret_cast<uintptr_t>(target_sym->data()));
      }

      SECTION("data pointer reloc_ptr_b points to reloc_target_b") {
        auto ptr_sym = loaded->symbol("reloc_ptr_b");
        REQUIRE_RESULT(ptr_sym);
        auto target_sym = loaded->symbol("reloc_target_b");
        REQUIRE_RESULT(target_sym);

        REQUIRE_RESULT(fix->sdma.copy_linear(readback->data(), ptr_sym->data(),
                                             sizeof(uint64_t)));
        REQUIRE_RESULT(fix->sdma.signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

        uint64_t val;
        std::memcpy(&val, readback->data(), sizeof(val));
        CHECK(val >= base);
        CHECK(val < limit);
        CHECK(val == reinterpret_cast<uintptr_t>(target_sym->data()));
      }

      SECTION("function pointer reloc_fptr points within loaded image") {
        auto ptr_sym = loaded->symbol("reloc_fptr");
        REQUIRE_RESULT(ptr_sym);

        REQUIRE_RESULT(fix->sdma.copy_linear(readback->data(), ptr_sym->data(),
                                             sizeof(uint64_t)));
        REQUIRE_RESULT(fix->sdma.signal(*sig));
        REQUIRE_RESULT(
            sig->wait(kfd::Condition::EQ, 0, kfd::test::WAIT_TIMEOUT_NS));

        uint64_t val;
        std::memcpy(&val, readback->data(), sizeof(val));
        CHECK(val >= base);
        CHECK(val < limit);
        CHECK(val != 0);
      }
    }
  }
}

TEST_CASE("Loader - kernel descriptor readable from host", "[device][loader]") {
  using namespace kfd::detail::elf;
  using namespace kfd::abi;

  auto &ctx = require_ctx();
  for (size_t di = 0; di < ctx.num_devices(); ++di) {
    DYNAMIC_SECTION("device " << di) {
      auto &gpu = require_gpu(ctx, di);
      auto fix = make_device_fixture(gpu, device_kernels);
      if (!fix && fix.error().code == ENOEXEC)
        SKIP(kfd::strerror(fix));
      REQUIRE_RESULT(fix);

      auto buf = read_file(fix->bin->path);
      auto elf =
          ELF64LE::create(std::span<const std::byte>(buf.data(), buf.size()));
      REQUIRE_RESULT(elf);

      auto sym = elf->find_symbol("kernel.kd");
      if (!sym.has_value() || !*sym)
        SKIP("kernel.kd not in dynamic symbol table");

      REQUIRE((*sym)->st_size >= sizeof(KernelDescriptor));

      auto addr = elf->symbol_address(**sym);
      REQUIRE_RESULT(addr);

      auto *kd = reinterpret_cast<const KernelDescriptor *>(*addr);
      CHECK(kd->kernel_code_entry_byte_offset > 0);
    }
  }
}
