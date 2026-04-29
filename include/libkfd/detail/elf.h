//===-- libkfd/detail/elf.h - Minimal ELF64LE parser ------------*- C++ -*-===//
//
// Non-owning ELF image wrapper for 64-bit little-endian objects. Provides
// section and segment access plus symbol lookup via hash tables.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_DETAIL_ELF_H
#define LIBKFD_DETAIL_ELF_H

#include "libkfd/error.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <span>
#include <string_view>

namespace kfd::detail::elf {

// ELF64 type definitions.
using Elf64_Addr = uint64_t;
using Elf64_Off = uint64_t;
using Elf64_Half = uint16_t;
using Elf64_Word = uint32_t;
using Elf64_Sword = int32_t;
using Elf64_Xword = uint64_t;
using Elf64_Sxword = int64_t;

struct Elf64_Ehdr {
  uint8_t e_ident[16];
  Elf64_Half e_type;
  Elf64_Half e_machine;
  Elf64_Word e_version;
  Elf64_Addr e_entry;
  Elf64_Off e_phoff;
  Elf64_Off e_shoff;
  Elf64_Word e_flags;
  Elf64_Half e_ehsize;
  Elf64_Half e_phentsize;
  Elf64_Half e_phnum;
  Elf64_Half e_shentsize;
  Elf64_Half e_shnum;
  Elf64_Half e_shstrndx;
};

struct Elf64_Phdr {
  Elf64_Word p_type;
  Elf64_Word p_flags;
  Elf64_Off p_offset;
  Elf64_Addr p_vaddr;
  Elf64_Addr p_paddr;
  Elf64_Xword p_filesz;
  Elf64_Xword p_memsz;
  Elf64_Xword p_align;
};

struct Elf64_Shdr {
  Elf64_Word sh_name;
  Elf64_Word sh_type;
  Elf64_Xword sh_flags;
  Elf64_Addr sh_addr;
  Elf64_Off sh_offset;
  Elf64_Xword sh_size;
  Elf64_Word sh_link;
  Elf64_Word sh_info;
  Elf64_Xword sh_addralign;
  Elf64_Xword sh_entsize;
};

struct Elf64_Sym {
  Elf64_Word st_name;
  uint8_t st_info;
  uint8_t st_other;
  Elf64_Half st_shndx;
  Elf64_Addr st_value;
  Elf64_Xword st_size;

  uint8_t get_binding() const { return st_info >> 4; }
  uint8_t get_type() const { return st_info & 0xf; }
};

struct Elf64_Nhdr {
  Elf64_Word n_namesz;
  Elf64_Word n_descsz;
  Elf64_Word n_type;
};

struct Elf64_Dyn {
  Elf64_Sxword d_tag;
  union {
    Elf64_Xword d_val;
    Elf64_Addr d_ptr;
  } d_un;
};

struct Elf64_Rela {
  Elf64_Addr r_offset;
  Elf64_Xword r_info;
  Elf64_Sxword r_addend;

  uint32_t get_type() const { return static_cast<uint32_t>(r_info); }
  uint32_t get_symbol() const { return static_cast<uint32_t>(r_info >> 32); }
};

// On-disk layout of a GNU hash table (.gnu.hash).
struct Elf64_GnuHash {
  Elf64_Word nbuckets;
  Elf64_Word symndx;
  Elf64_Word maskwords;
  Elf64_Word shift2;

  const uint64_t *filter() const {
    return reinterpret_cast<const uint64_t *>(&shift2 + 1);
  }
  const Elf64_Word *buckets() const {
    return reinterpret_cast<const Elf64_Word *>(filter() + maskwords);
  }
  const Elf64_Word *values() const { return buckets() + nbuckets; }
};

// e_ident indices and values.
inline constexpr unsigned EI_MAG0 = 0;
inline constexpr unsigned EI_MAG1 = 1;
inline constexpr unsigned EI_MAG2 = 2;
inline constexpr unsigned EI_MAG3 = 3;
inline constexpr unsigned EI_CLASS = 4;
inline constexpr unsigned EI_DATA = 5;
inline constexpr unsigned EI_VERSION = 6;
inline constexpr unsigned EI_OSABI = 7;
inline constexpr unsigned EI_ABIVERSION = 8;
inline constexpr unsigned EI_NIDENT = 16;

inline constexpr uint8_t ELFMAG0 = 0x7f;
inline constexpr uint8_t ELFMAG1 = 'E';
inline constexpr uint8_t ELFMAG2 = 'L';
inline constexpr uint8_t ELFMAG3 = 'F';

inline constexpr uint8_t ELFCLASS64 = 2;
inline constexpr uint8_t ELFDATA2LSB = 1;
inline constexpr uint8_t EV_CURRENT = 1;

// Object file type (e_type).
inline constexpr Elf64_Half ET_NONE = 0;
inline constexpr Elf64_Half ET_REL = 1;
inline constexpr Elf64_Half ET_EXEC = 2;
inline constexpr Elf64_Half ET_DYN = 3;
inline constexpr Elf64_Half ET_CORE = 4;

// Machine type (e_machine).
inline constexpr Elf64_Half EM_NONE = 0;
inline constexpr Elf64_Half EM_AMDGPU = 224;

// Section header types (sh_type).
inline constexpr Elf64_Word SHT_NULL = 0;
inline constexpr Elf64_Word SHT_PROGBITS = 1;
inline constexpr Elf64_Word SHT_SYMTAB = 2;
inline constexpr Elf64_Word SHT_STRTAB = 3;
inline constexpr Elf64_Word SHT_RELA = 4;
inline constexpr Elf64_Word SHT_HASH = 5;
inline constexpr Elf64_Word SHT_DYNAMIC = 6;
inline constexpr Elf64_Word SHT_NOTE = 7;
inline constexpr Elf64_Word SHT_NOBITS = 8;
inline constexpr Elf64_Word SHT_REL = 9;
inline constexpr Elf64_Word SHT_DYNSYM = 11;
inline constexpr Elf64_Word SHT_GNU_HASH = 0x6ffffff6;

// Section header flags (sh_flags).
inline constexpr Elf64_Xword SHF_WRITE = 0x1;
inline constexpr Elf64_Xword SHF_ALLOC = 0x2;
inline constexpr Elf64_Xword SHF_EXECINSTR = 0x4;

// Special section indices.
inline constexpr Elf64_Half SHN_UNDEF = 0;
inline constexpr Elf64_Half SHN_ABS = 0xfff1;

// Symbol table.
inline constexpr Elf64_Word STN_UNDEF = 0;

inline constexpr uint8_t STB_LOCAL = 0;
inline constexpr uint8_t STB_GLOBAL = 1;
inline constexpr uint8_t STB_WEAK = 2;

inline constexpr uint8_t STT_NOTYPE = 0;
inline constexpr uint8_t STT_OBJECT = 1;
inline constexpr uint8_t STT_FUNC = 2;
inline constexpr uint8_t STT_SECTION = 3;

// Program header types (p_type).
inline constexpr Elf64_Word PT_NULL = 0;
inline constexpr Elf64_Word PT_LOAD = 1;
inline constexpr Elf64_Word PT_DYNAMIC = 2;
inline constexpr Elf64_Word PT_INTERP = 3;
inline constexpr Elf64_Word PT_NOTE = 4;
inline constexpr Elf64_Word PT_PHDR = 6;
inline constexpr Elf64_Word PT_TLS = 7;
inline constexpr Elf64_Word PT_GNU_RELRO = 0x6474e552;

// Dynamic section tags (d_tag).
inline constexpr Elf64_Sxword DT_NULL = 0;
inline constexpr Elf64_Sxword DT_RELA = 7;
inline constexpr Elf64_Sxword DT_RELASZ = 8;
inline constexpr Elf64_Sxword DT_RELAENT = 9;

// AMDGPU relocation types.
inline constexpr uint32_t R_AMDGPU_NONE = 0;
inline constexpr uint32_t R_AMDGPU_RELATIVE64 = 13;

// Program header flags (p_flags).
inline constexpr Elf64_Word PF_X = 1;
inline constexpr Elf64_Word PF_W = 2;
inline constexpr Elf64_Word PF_R = 4;

// AMDGPU OS/ABI identification.
inline constexpr uint8_t ELFOSABI_AMDGPU_HSA = 64;
inline constexpr uint8_t ELFOSABI_AMDGPU_PAL = 65;
inline constexpr uint8_t ELFOSABI_AMDGPU_MESA3D = 66;

inline constexpr uint8_t ELFABIVERSION_AMDGPU_HSA_V2 = 0;
inline constexpr uint8_t ELFABIVERSION_AMDGPU_HSA_V3 = 1;
inline constexpr uint8_t ELFABIVERSION_AMDGPU_HSA_V4 = 2;
inline constexpr uint8_t ELFABIVERSION_AMDGPU_HSA_V5 = 3;
inline constexpr uint8_t ELFABIVERSION_AMDGPU_HSA_V6 = 4;

// AMDGPU e_flags: machine selection and feature bits.

// clang-format off

// X-macro table of all AMDGPU machine IDs.
#define AMDGPU_MACH_LIST(X)                                                    \
  X(0x20, EF_AMDGPU_MACH_AMDGCN_GFX600, "gfx600", 60000)                       \
  X(0x21, EF_AMDGPU_MACH_AMDGCN_GFX601, "gfx601", 60001)                       \
  X(0x22, EF_AMDGPU_MACH_AMDGCN_GFX700, "gfx700", 70000)                       \
  X(0x23, EF_AMDGPU_MACH_AMDGCN_GFX701, "gfx701", 70001)                       \
  X(0x24, EF_AMDGPU_MACH_AMDGCN_GFX702, "gfx702", 70002)                       \
  X(0x25, EF_AMDGPU_MACH_AMDGCN_GFX703, "gfx703", 70003)                       \
  X(0x26, EF_AMDGPU_MACH_AMDGCN_GFX704, "gfx704", 70004)                       \
  X(0x28, EF_AMDGPU_MACH_AMDGCN_GFX801, "gfx801", 80001)                       \
  X(0x29, EF_AMDGPU_MACH_AMDGCN_GFX802, "gfx802", 80002)                       \
  X(0x2a, EF_AMDGPU_MACH_AMDGCN_GFX803, "gfx803", 80003)                       \
  X(0x2b, EF_AMDGPU_MACH_AMDGCN_GFX810, "gfx810", 80100)                       \
  X(0x2c, EF_AMDGPU_MACH_AMDGCN_GFX900, "gfx900", 90000)                       \
  X(0x2d, EF_AMDGPU_MACH_AMDGCN_GFX902, "gfx902", 90002)                       \
  X(0x2e, EF_AMDGPU_MACH_AMDGCN_GFX904, "gfx904", 90004)                       \
  X(0x2f, EF_AMDGPU_MACH_AMDGCN_GFX906, "gfx906", 90006)                       \
  X(0x30, EF_AMDGPU_MACH_AMDGCN_GFX908, "gfx908", 90008)                       \
  X(0x31, EF_AMDGPU_MACH_AMDGCN_GFX909, "gfx909", 90009)                       \
  X(0x32, EF_AMDGPU_MACH_AMDGCN_GFX90C, "gfx90c", 90012)                       \
  X(0x33, EF_AMDGPU_MACH_AMDGCN_GFX1010, "gfx1010", 100100)                    \
  X(0x34, EF_AMDGPU_MACH_AMDGCN_GFX1011, "gfx1011", 100101)                    \
  X(0x35, EF_AMDGPU_MACH_AMDGCN_GFX1012, "gfx1012", 100102)                    \
  X(0x36, EF_AMDGPU_MACH_AMDGCN_GFX1030, "gfx1030", 100300)                    \
  X(0x37, EF_AMDGPU_MACH_AMDGCN_GFX1031, "gfx1031", 100301)                    \
  X(0x38, EF_AMDGPU_MACH_AMDGCN_GFX1032, "gfx1032", 100302)                    \
  X(0x39, EF_AMDGPU_MACH_AMDGCN_GFX1033, "gfx1033", 100303)                    \
  X(0x3a, EF_AMDGPU_MACH_AMDGCN_GFX602, "gfx602", 60002)                       \
  X(0x3b, EF_AMDGPU_MACH_AMDGCN_GFX705, "gfx705", 70005)                       \
  X(0x3c, EF_AMDGPU_MACH_AMDGCN_GFX805, "gfx805", 80005)                       \
  X(0x3d, EF_AMDGPU_MACH_AMDGCN_GFX1035, "gfx1035", 100305)                    \
  X(0x3e, EF_AMDGPU_MACH_AMDGCN_GFX1034, "gfx1034", 100304)                    \
  X(0x3f, EF_AMDGPU_MACH_AMDGCN_GFX90A, "gfx90a", 90010)                       \
  X(0x41, EF_AMDGPU_MACH_AMDGCN_GFX1100, "gfx1100", 110000)                    \
  X(0x42, EF_AMDGPU_MACH_AMDGCN_GFX1013, "gfx1013", 100103)                    \
  X(0x43, EF_AMDGPU_MACH_AMDGCN_GFX1150, "gfx1150", 110500)                    \
  X(0x44, EF_AMDGPU_MACH_AMDGCN_GFX1103, "gfx1103", 110003)                    \
  X(0x45, EF_AMDGPU_MACH_AMDGCN_GFX1036, "gfx1036", 100306)                    \
  X(0x46, EF_AMDGPU_MACH_AMDGCN_GFX1101, "gfx1101", 110001)                    \
  X(0x47, EF_AMDGPU_MACH_AMDGCN_GFX1102, "gfx1102", 110002)                    \
  X(0x48, EF_AMDGPU_MACH_AMDGCN_GFX1200, "gfx1200", 120000)                    \
  X(0x49, EF_AMDGPU_MACH_AMDGCN_GFX1250, "gfx1250", 120500)                    \
  X(0x4a, EF_AMDGPU_MACH_AMDGCN_GFX1151, "gfx1151", 110501)                    \
  X(0x4c, EF_AMDGPU_MACH_AMDGCN_GFX942, "gfx942", 90402)                       \
  X(0x4e, EF_AMDGPU_MACH_AMDGCN_GFX1201, "gfx1201", 120001)                    \
  X(0x4f, EF_AMDGPU_MACH_AMDGCN_GFX950, "gfx950", 90500)                       \
  X(0x50, EF_AMDGPU_MACH_AMDGCN_GFX1310, "gfx1310", 130100)                    \
  X(0x51, EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC, "gfx9-generic", 0)               \
  X(0x52, EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC, "gfx10-1-generic", 0)         \
  X(0x53, EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC, "gfx10-3-generic", 0)         \
  X(0x54, EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, "gfx11-generic", 0)             \
  X(0x55, EF_AMDGPU_MACH_AMDGCN_GFX1152, "gfx1152", 110502)                    \
  X(0x58, EF_AMDGPU_MACH_AMDGCN_GFX1153, "gfx1153", 110503)                    \
  X(0x59, EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC, "gfx12-generic", 0)             \
  X(0x5a, EF_AMDGPU_MACH_AMDGCN_GFX1251, "gfx1251", 120501)                    \
  X(0x5b, EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC, "gfx12-5-generic", 0)         \
  X(0x5c, EF_AMDGPU_MACH_AMDGCN_GFX1172, "gfx1172", 110702)                    \
  X(0x5d, EF_AMDGPU_MACH_AMDGCN_GFX1170, "gfx1170", 110700)                    \
  X(0x5e, EF_AMDGPU_MACH_AMDGCN_GFX1171, "gfx1171", 110701)                    \
  X(0x5f, EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC, "gfx9-4-generic", 0)

enum : unsigned {
  EF_AMDGPU_MACH = 0x0ff,
  EF_AMDGPU_MACH_NONE = 0x000,

#define X(NUM, ENUM, NAME, VER) ENUM = NUM,
  AMDGPU_MACH_LIST(X)
#undef X

  EF_AMDGPU_MACH_AMDGCN_FIRST = EF_AMDGPU_MACH_AMDGCN_GFX600,
  EF_AMDGPU_MACH_AMDGCN_LAST = EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC,

  // V2 feature flags.
  EF_AMDGPU_FEATURE_XNACK_V2 = 0x01,
  EF_AMDGPU_FEATURE_TRAP_HANDLER_V2 = 0x02,

  // V3 feature flags.
  EF_AMDGPU_FEATURE_XNACK_V3 = 0x100,
  EF_AMDGPU_FEATURE_SRAMECC_V3 = 0x200,

  // V4+ XNACK selection.
  EF_AMDGPU_FEATURE_XNACK_V4 = 0x300,
  EF_AMDGPU_FEATURE_XNACK_UNSUPPORTED_V4 = 0x000,
  EF_AMDGPU_FEATURE_XNACK_ANY_V4 = 0x100,
  EF_AMDGPU_FEATURE_XNACK_OFF_V4 = 0x200,
  EF_AMDGPU_FEATURE_XNACK_ON_V4 = 0x300,

  // V4+ SRAMECC selection.
  EF_AMDGPU_FEATURE_SRAMECC_V4 = 0xc00,
  EF_AMDGPU_FEATURE_SRAMECC_UNSUPPORTED_V4 = 0x000,
  EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 = 0x400,
  EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 = 0x800,
  EF_AMDGPU_FEATURE_SRAMECC_ON_V4 = 0xc00,

  // Generic target versioning (high byte of e_flags).
  EF_AMDGPU_GENERIC_VERSION = 0xff000000,
  EF_AMDGPU_GENERIC_VERSION_OFFSET = 24,
  EF_AMDGPU_GENERIC_VERSION_MIN = 1,
  EF_AMDGPU_GENERIC_VERSION_MAX = 0xff,
};

// clang-format on

// AMDGPU special section indices and note types.
inline constexpr Elf64_Half SHN_AMDGPU_LDS = 0xff00;

enum : unsigned {
  // Code Object V2 notes (owner "AMD").
  NT_AMD_HSA_CODE_OBJECT_VERSION = 1,
  NT_AMD_HSA_HSAIL = 2,
  NT_AMD_HSA_ISA_VERSION = 3,
  NT_AMD_HSA_METADATA = 10,
  NT_AMD_HSA_ISA_NAME = 11,
  NT_AMD_PAL_METADATA = 12,

  // Code Object V3+ notes (owner "AMDGPU").
  NT_AMDGPU_METADATA = 32,
};

// Hash functions for symbol lookup.
inline uint32_t hash_gnu(std::string_view name) {
  uint32_t h = 5381;
  for (char ch : name)
    h = (h << 5) + h + static_cast<uint8_t>(ch);
  return h;
}

// Forward iterator that projects an Elf64_Sym span into symbol names by
// resolving each st_name through the associated string table.
class SymbolRange {
public:
  class iterator {
  public:
    using value_type = std::string_view;
    using difference_type = ptrdiff_t;

    std::string_view operator*() const {
      if (cur->st_name >= strtab_size)
        return {};
      size_t max_len = static_cast<size_t>(strtab_size - cur->st_name);
      const char *start = strtab + cur->st_name;
      const void *nul = std::memchr(start, '\0', max_len);
      if (!nul)
        return {};
      return {start,
              static_cast<size_t>(static_cast<const char *>(nul) - start)};
    }

    iterator &operator++() {
      ++cur;
      return *this;
    }
    iterator operator++(int) {
      auto tmp = *this;
      ++cur;
      return tmp;
    }
    friend bool operator==(const iterator &a, const iterator &b) {
      return a.cur == b.cur;
    }

  private:
    friend class SymbolRange;
    friend class ELF64LE;
    const Elf64_Sym *cur = nullptr;
    const char *strtab = nullptr;
    uint64_t strtab_size = 0;
  };

  iterator begin() const { return first; }
  iterator end() const { return last; }
  bool empty() const { return first.cur == last.cur; }

private:
  friend class ELF64LE;
  iterator first;
  iterator last;
};

// Non-owning, parsed ELF image. Only supports AMDGCN executables.
class ELF64LE {
public:
  static std::expected<ELF64LE, Error> create(std::span<const std::byte> buf);

  const Elf64_Ehdr &header() const {
    return *reinterpret_cast<const Elf64_Ehdr *>(buf.data());
  }

  std::span<const Elf64_Phdr> phdrs() const;
  std::expected<std::span<const std::byte>, Error>
  segment_data(const Elf64_Phdr &phdr) const;

  std::span<const Elf64_Shdr> sections() const;
  std::expected<std::string_view, Error>
  section_name(const Elf64_Shdr &shdr) const;
  std::expected<std::span<const std::byte>, Error>
  section_data(const Elf64_Shdr &shdr) const;

  std::expected<const Elf64_Sym *, Error>
  find_symbol(std::string_view name) const;

  std::expected<const void *, Error> symbol_address(const Elf64_Sym &sym) const;

  SymbolRange symbols() const;

  template <typename T>
  std::expected<const T *, Error> lookup(std::string_view name) const {
    auto sym = KFD_TRY(find_symbol(name));
    if (!sym)
      return unexpected(ENOENT, "symbol '%.*s' not found",
                        static_cast<int>(name.size()), name.data());
    if (sym->st_size < sizeof(T))
      return unexpected(EINVAL, "symbol '%.*s' too small for type",
                        static_cast<int>(name.size()), name.data());
    auto addr = KFD_TRY(symbol_address(*sym));
    return reinterpret_cast<const T *>(addr);
  }

  struct LoadExtent {
    uint64_t lo;
    uint64_t hi;
    uint64_t max_align;
  };

  // Compute the virtual address range [lo, hi) and maximum alignment across
  // all PT_LOAD segments.
  std::expected<LoadExtent, Error> load_extent() const;

  const std::byte *base() const { return buf.data(); }
  size_t size() const { return buf.size(); }

private:
  explicit ELF64LE(std::span<const std::byte> buf) : buf(buf) {}
  std::span<const std::byte> buf;

  const Elf64_Shdr *strtab_for_symtab(const Elf64_Shdr &symtab) const;
  std::span<const Elf64_Sym> symbols_for(const Elf64_Shdr &symtab) const;
};

// Map an EF_AMDGPU_MACH_* value to its canonical string name (e.g. "gfx900").
// Returns an empty view for unknown values.
std::string_view get_name(uint32_t mach);

// Map an EF_AMDGPU_MACH_* value to its decimal-packed gfx_target_version.
// Returns 0 for generic or unknown / values.
uint32_t get_gfx_version(uint32_t mach);

// Map a decimal-packed gfx_target_version to its EF_AMDGPU_MACH_* value.
// Returns EF_AMDGPU_MACH_NONE (0) for unknown versions.
uint32_t get_mach(uint32_t gfx_version);

// Format a gfx_target_version as "gfxNNNX" (e.g. "gfx1030", "gfx90a").
// Returns the number of characters written (excluding null terminator).
int format_gfx_version(char *buf, size_t size, uint32_t version);

// Returns true if the EF_AMDGPU_MACH value denotes a generic target.
bool is_generic_mach(uint32_t mach);

// Returns the generic mach that covers \p gfx_version, or EF_AMDGPU_MACH_NONE
// if the GPU is not part of any generic family.
uint32_t get_generic_for_gpu(uint32_t gfx_version);

// Check whether an AMDGPU ELF is compatible with a device given its
// gfx_target_version and the process-wide xnack / per-device sramecc state.
bool is_compatible(uint32_t e_flags, uint32_t gfx_version,
                   bool xnack_enabled = false, bool sramecc_enabled = false);

} // namespace kfd::detail::elf

#endif // LIBKFD_DETAIL_ELF_H
