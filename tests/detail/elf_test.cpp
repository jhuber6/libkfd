#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

// Auto-generated table of all compiled test kernels.
static const kfd::test::TestBinary test_kernel_binaries[] = {
#include "elf_test_kernel_kernels.inc"
};

static std::span<const kfd::test::TestBinary> test_binaries() {
  return test_kernel_binaries;
}

using namespace kfd::detail::elf;
using kfd::test::read_file;

namespace {

bool has_section(const ELF64LE &elf, std::string_view name) {
  for (const auto &shdr : elf.sections()) {
    auto n = elf.section_name(shdr);
    if (n.has_value() && *n == name)
      return true;
  }
  return false;
}

const Elf64_Shdr *find_section(const ELF64LE &elf, std::string_view name) {
  for (const auto &shdr : elf.sections()) {
    auto n = elf.section_name(shdr);
    if (n.has_value() && *n == name)
      return &shdr;
  }
  return nullptr;
}

template <typename T> T read_le(const void *ptr) {
  T val;
  std::memcpy(&val, ptr, sizeof(T));
  return val;
}

} // namespace

TEST_CASE("ELF - create rejects truncated buffer", "[elf]") {
  std::byte tiny[4] = {};
  auto elf = ELF64LE::create({tiny, sizeof(tiny)});
  CHECK_FALSE(elf.has_value());
}

TEST_CASE("ELF - create rejects bad magic", "[elf]") {
  std::byte garbage[128] = {};
  auto elf = ELF64LE::create({garbage, sizeof(garbage)});
  CHECK_FALSE(elf.has_value());
}

TEST_CASE("ELF - header valid ELF64LE AMDGPU", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      const auto &ehdr = elf->header();
      CHECK(ehdr.e_ident[EI_MAG0] == ELFMAG0);
      CHECK(ehdr.e_ident[EI_MAG1] == ELFMAG1);
      CHECK(ehdr.e_ident[EI_MAG2] == ELFMAG2);
      CHECK(ehdr.e_ident[EI_MAG3] == ELFMAG3);
      CHECK(ehdr.e_ident[EI_CLASS] == ELFCLASS64);
      CHECK(ehdr.e_ident[EI_DATA] == ELFDATA2LSB);
      CHECK(ehdr.e_ident[EI_VERSION] == EV_CURRENT);
      CHECK(ehdr.e_machine == EM_AMDGPU);
      CHECK(ehdr.e_phnum > 0);
      CHECK(ehdr.e_shnum > 0);
    }
  }
}

TEST_CASE("ELF - e_flags mach ID matches requested arch", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      uint32_t mach = elf->header().e_flags & EF_AMDGPU_MACH;
      CHECK(get_name(mach) == bin.arch);
    }
  }
}

TEST_CASE("ELF - get_gfx_version generic targets return 0", "[elf]") {
  for (const auto &bin : test_binaries()) {
    std::string_view arch(bin.arch);
    if (arch.find("generic") == std::string_view::npos)
      continue;
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      uint32_t mach = elf->header().e_flags & EF_AMDGPU_MACH;
      CHECK(get_gfx_version(mach) == 0);
    }
  }
}

TEST_CASE("ELF - get_gfx_version known specific targets", "[elf]") {
  struct {
    uint32_t mach;
    uint32_t version;
  } cases[] = {
      {EF_AMDGPU_MACH_AMDGCN_GFX900, 90000},
      {EF_AMDGPU_MACH_AMDGCN_GFX906, 90006},
      {EF_AMDGPU_MACH_AMDGCN_GFX908, 90008},
      {EF_AMDGPU_MACH_AMDGCN_GFX90A, 90010},
      {EF_AMDGPU_MACH_AMDGCN_GFX90C, 90012},
      {EF_AMDGPU_MACH_AMDGCN_GFX942, 90402},
      {EF_AMDGPU_MACH_AMDGCN_GFX950, 90500},
      {EF_AMDGPU_MACH_AMDGCN_GFX1030, 100300},
      {EF_AMDGPU_MACH_AMDGCN_GFX1100, 110000},
      {EF_AMDGPU_MACH_AMDGCN_GFX1200, 120000},
      {EF_AMDGPU_MACH_AMDGCN_GFX1250, 120500},
  };
  for (const auto &[mach, version] : cases) {
    CAPTURE(mach, version);
    CHECK(get_gfx_version(mach) == version);
  }
}

TEST_CASE("ELF - get_mach round-trips with get_gfx_version", "[elf]") {
  struct {
    uint32_t mach;
    uint32_t version;
  } cases[] = {
      {EF_AMDGPU_MACH_AMDGCN_GFX900, 90000},
      {EF_AMDGPU_MACH_AMDGCN_GFX90A, 90010},
      {EF_AMDGPU_MACH_AMDGCN_GFX942, 90402},
      {EF_AMDGPU_MACH_AMDGCN_GFX1030, 100300},
      {EF_AMDGPU_MACH_AMDGCN_GFX1100, 110000},
      {EF_AMDGPU_MACH_AMDGCN_GFX1200, 120000},
  };
  for (const auto &[mach, version] : cases) {
    CAPTURE(mach, version);
    CHECK(get_mach(version) == mach);
    CHECK(get_gfx_version(get_mach(version)) == version);
    CHECK(get_mach(get_gfx_version(mach)) == mach);
  }
}

TEST_CASE("ELF - get_mach unknown version returns NONE", "[elf]") {
  CHECK(get_mach(0) == EF_AMDGPU_MACH_NONE);
  CHECK(get_mach(99999) == EF_AMDGPU_MACH_NONE);
}

TEST_CASE("ELF - format_gfx_version known strings", "[elf]") {
  struct {
    uint32_t version;
    const char *expected;
  } cases[] = {
      {90000, "gfx900"},   {90010, "gfx90a"},   {90012, "gfx90c"},
      {90402, "gfx942"},   {90500, "gfx950"},   {100300, "gfx1030"},
      {110000, "gfx1100"}, {120000, "gfx1200"}, {120500, "gfx1250"},
  };
  for (const auto &[version, expected] : cases) {
    char buf[16] = {};
    int n = format_gfx_version(buf, sizeof(buf), version);
    CAPTURE(version, expected, buf);
    CHECK(n > 0);
    CHECK(std::string_view(buf) == expected);
  }
}

TEST_CASE("ELF - format_gfx_version matches get_name for specific targets",
          "[elf]") {
  uint32_t specific_machs[] = {
      EF_AMDGPU_MACH_AMDGCN_GFX900,  EF_AMDGPU_MACH_AMDGCN_GFX906,
      EF_AMDGPU_MACH_AMDGCN_GFX90A,  EF_AMDGPU_MACH_AMDGCN_GFX942,
      EF_AMDGPU_MACH_AMDGCN_GFX1030, EF_AMDGPU_MACH_AMDGCN_GFX1100,
      EF_AMDGPU_MACH_AMDGCN_GFX1200, EF_AMDGPU_MACH_AMDGCN_GFX1250,
  };
  for (uint32_t mach : specific_machs) {
    uint32_t ver = get_gfx_version(mach);
    REQUIRE(ver != 0);
    char buf[16] = {};
    format_gfx_version(buf, sizeof(buf), ver);
    CAPTURE(mach, ver, buf);
    CHECK(std::string_view(buf) == get_name(mach));
  }
}

TEST_CASE("ELF - sections expected sections present", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      CHECK(has_section(*elf, ".text"));
      CHECK(has_section(*elf, ".dynsym"));
      CHECK(has_section(*elf, ".dynamic"));
      CHECK(has_section(*elf, ".gnu.hash"));
      CHECK(has_section(*elf, ".custom"));
    }
  }
}

TEST_CASE("ELF - sections .text is executable", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      const auto *text = find_section(*elf, ".text");
      REQUIRE(text);
      CHECK(text->sh_type == SHT_PROGBITS);
      CHECK((text->sh_flags & SHF_ALLOC) != 0);
      CHECK((text->sh_flags & SHF_EXECINSTR) != 0);
      CHECK(text->sh_size > 0);
    }
  }
}

TEST_CASE("ELF - sections .dynsym is valid", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      const auto *dynsym = find_section(*elf, ".dynsym");
      REQUIRE(dynsym);
      CHECK(dynsym->sh_type == SHT_DYNSYM);
      CHECK(dynsym->sh_entsize == sizeof(Elf64_Sym));
      CHECK(dynsym->sh_size >= dynsym->sh_entsize);
    }
  }
}

TEST_CASE("ELF - sections .dynamic is valid", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      const auto *dyn = find_section(*elf, ".dynamic");
      REQUIRE(dyn);
      CHECK(dyn->sh_type == SHT_DYNAMIC);
      CHECK((dyn->sh_flags & SHF_ALLOC) != 0);
      CHECK(dyn->sh_entsize == sizeof(Elf64_Dyn));
    }
  }
}

TEST_CASE("ELF - sections custom '.custom' section has expected data",
          "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      const auto *custom = find_section(*elf, ".custom");
      REQUIRE(custom);
      CHECK(custom->sh_type == SHT_PROGBITS);
      CHECK((custom->sh_flags & SHF_ALLOC) != 0);

      auto data = elf->section_data(*custom);
      REQUIRE_RESULT(data);
      REQUIRE(data->size() >= 4 * sizeof(unsigned));

      unsigned vals[4];
      std::memcpy(vals, data->data(), sizeof(vals));
      CHECK(vals[0] == 1);
      CHECK(vals[1] == 2);
      CHECK(vals[2] == 3);
      CHECK(vals[3] == 4);
    }
  }
}

TEST_CASE("ELF - sections all ALLOC sections have valid data", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      for (const auto &shdr : elf->sections()) {
        if (!(shdr.sh_flags & SHF_ALLOC))
          continue;
        auto data = elf->section_data(shdr);
        CHECK_RESULT(data);
      }
    }
  }
}

TEST_CASE("ELF - phdrs PT_LOAD segments are well-formed", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      unsigned load_count = 0;
      for (const auto &phdr : elf->phdrs()) {
        if (phdr.p_type != PT_LOAD)
          continue;
        ++load_count;

        CHECK(phdr.p_filesz <= phdr.p_memsz);
        CHECK((phdr.p_flags & PF_R) != 0);

        if (phdr.p_align > 1)
          CHECK((phdr.p_align & (phdr.p_align - 1)) == 0);

        auto data = elf->segment_data(phdr);
        REQUIRE_RESULT(data);
        CHECK(data->size() == phdr.p_filesz);
      }
      CHECK(load_count >= 2);
    }
  }
}

TEST_CASE("ELF - phdrs PT_LOAD permissions match content", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      bool found_rx = false;
      bool found_rw = false;
      for (const auto &phdr : elf->phdrs()) {
        if (phdr.p_type != PT_LOAD)
          continue;
        if ((phdr.p_flags & (PF_R | PF_X)) == (PF_R | PF_X))
          found_rx = true;
        if ((phdr.p_flags & (PF_R | PF_W)) == (PF_R | PF_W))
          found_rw = true;
      }
      CHECK(found_rx);
      CHECK(found_rw);
    }
  }
}

TEST_CASE("ELF - phdrs PT_DYNAMIC segment present", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      bool found = false;
      for (const auto &phdr : elf->phdrs()) {
        if (phdr.p_type == PT_DYNAMIC) {
          found = true;
          CHECK(phdr.p_filesz > 0);
          auto data = elf->segment_data(phdr);
          CHECK_RESULT(data);
        }
      }
      CHECK(found);
    }
  }
}

TEST_CASE("ELF - phdrs PT_LOAD covers all ALLOC sections", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      auto phdrs = elf->phdrs();
      for (const auto &shdr : elf->sections()) {
        if (!(shdr.sh_flags & SHF_ALLOC) || shdr.sh_size == 0)
          continue;

        bool covered = false;
        for (const auto &phdr : phdrs) {
          if (phdr.p_type != PT_LOAD)
            continue;
          if (shdr.sh_addr >= phdr.p_vaddr &&
              shdr.sh_addr + shdr.sh_size <= phdr.p_vaddr + phdr.p_memsz) {
            covered = true;
            break;
          }
        }

        auto name = elf->section_name(shdr);
        INFO("section: " << (name.has_value() ? *name : "<unknown>"));
        CHECK(covered);
      }
    }
  }
}

TEST_CASE("ELF - phdrs virtual address range is contiguous", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      uint64_t lo = UINT64_MAX;
      uint64_t hi = 0;
      for (const auto &phdr : elf->phdrs()) {
        if (phdr.p_type != PT_LOAD)
          continue;
        lo = std::min(lo, phdr.p_vaddr);
        hi = std::max(hi, phdr.p_vaddr + phdr.p_memsz);
      }
      REQUIRE(hi > lo);
      uint64_t footprint = hi - lo;
      CHECK(footprint > 0);
      CHECK(footprint < 64 * 1024 * 1024);
    }
  }
}

TEST_CASE("ELF - find_symbol kernel functions exist", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      for (auto name : {"kernel", "use"}) {
        INFO("symbol: " << name);
        auto sym = elf->find_symbol(name);
        REQUIRE_RESULT(sym);
        REQUIRE(*sym != nullptr);
        CHECK((*sym)->getType() == STT_FUNC);
        CHECK((*sym)->st_size > 0);
      }
    }
  }
}

TEST_CASE("ELF - find_symbol globals exist as objects", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      for (auto name : {"x", "y", "z"}) {
        INFO("symbol: " << name);
        auto sym = elf->find_symbol(name);
        REQUIRE_RESULT(sym);
        REQUIRE(*sym != nullptr);
        CHECK((*sym)->getType() == STT_OBJECT);
        CHECK((*sym)->st_size == sizeof(unsigned));
      }
    }
  }
}

TEST_CASE("ELF - find_symbol nonexistent symbol returns null", "[elf]") {
  auto buf = read_file(test_binaries()[0].path);
  auto elf = ELF64LE::create({buf.data(), buf.size()});
  REQUIRE_RESULT(elf);

  auto sym = elf->find_symbol("this_symbol_does_not_exist");
  REQUIRE_RESULT(sym);
  CHECK(*sym == nullptr);
}

TEST_CASE("ELF - find_symbol bss_arr is in a NOBITS section", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      auto sym = elf->find_symbol("bss_arr");
      REQUIRE_RESULT(sym);
      REQUIRE(*sym != nullptr);
      CHECK((*sym)->getType() == STT_OBJECT);
      CHECK((*sym)->st_size == 4096);

      auto shdrs = elf->sections();
      REQUIRE((*sym)->st_shndx < shdrs.size());
      CHECK(shdrs[(*sym)->st_shndx].sh_type == SHT_NOBITS);
    }
  }
}

TEST_CASE("ELF - symbol_address kernel code resolvable", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      auto sym = elf->find_symbol("kernel");
      REQUIRE_RESULT(sym);
      REQUIRE(*sym != nullptr);

      auto addr = elf->symbol_address(**sym);
      REQUIRE_RESULT(addr);
      auto *p = static_cast<const std::byte *>(*addr);
      CHECK(p >= elf->base());
      CHECK(p < elf->base() + elf->size());
    }
  }
}

TEST_CASE("ELF - symbol_address globals have expected values", "[elf]") {
  for (const auto &bin : test_binaries()) {
    DYNAMIC_SECTION("arch: " << bin.arch) {
      auto buf = read_file(bin.path);
      auto elf = ELF64LE::create({buf.data(), buf.size()});
      REQUIRE_RESULT(elf);

      struct {
        const char *name;
        unsigned expected;
      } globals[] = {
          {"x", 0xdeadbeef},
          {"y", 0xfeedface},
          {"z", 0xcafebabe},
      };

      for (const auto &[name, expected] : globals) {
        INFO("global: " << name);
        auto sym = elf->find_symbol(name);
        REQUIRE_RESULT(sym);
        REQUIRE(*sym != nullptr);

        auto addr = elf->symbol_address(**sym);
        REQUIRE_RESULT(addr);
        CHECK(read_le<unsigned>(*addr) == expected);
      }
    }
  }
}
