//===-- detail/elf.cpp - Minimal ELF64LE parser ---------------------------===//
//
// Implementation of the non-owning ELF64LE image wrapper. Primarily copied from
// the LLVM class but specialized for AMDGPU.
//
//===----------------------------------------------------------------------===//

#include "libkfd/detail/elf.h"

#include <cstdio>

namespace kfd::detail::elf {

// X-macro table mapping each generic target to the processors it supports.
#define AMDGPU_GENERIC_MAP(X)                                                  \
  /* gfx9-generic */                                                           \
  X(EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC, 90000, 1)                              \
  X(EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC, 90002, 1)                              \
  X(EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC, 90004, 1)                              \
  X(EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC, 90006, 1)                              \
  X(EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC, 90009, 1)                              \
  X(EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC, 90012, 1)                              \
  /* gfx9-4-generic */                                                         \
  X(EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC, 90402, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC, 90500, 1)                            \
  /* gfx10-1-generic */                                                        \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC, 100100, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC, 100101, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC, 100102, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC, 100103, 1)                          \
  /* gfx10-3-generic */                                                        \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC, 100300, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC, 100301, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC, 100302, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC, 100303, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC, 100304, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC, 100305, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC, 100306, 1)                          \
  /* gfx11-generic */                                                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, 110000, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, 110001, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, 110002, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, 110003, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, 110500, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, 110501, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, 110502, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC, 110503, 1)                            \
  /* gfx12-generic */                                                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC, 120000, 1)                            \
  X(EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC, 120001, 1)                            \
  /* gfx12-5-generic */                                                        \
  X(EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC, 120500, 1)                          \
  X(EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC, 120501, 1)

// Gets the processor name for the ELF e_machine value.
std::string_view get_name(uint32_t mach) {
  switch (mach) {
#define X(NUM, ENUM, NAME, VER)                                                \
  case NUM:                                                                    \
    return NAME;
    AMDGPU_MACH_LIST(X)
#undef X
  default:
    return {};
  }
}

// Gets the numerical processor value from the ELF e_machine value.
uint32_t get_gfx_version(uint32_t mach) {
  switch (mach) {
#define X(NUM, ENUM, NAME, VER)                                                \
  case NUM:                                                                    \
    return VER;
    AMDGPU_MACH_LIST(X)
#undef X
  default:
    return 0;
  }
}

// Gets the ELF e_machine value from the numerical processor value.
uint32_t get_mach(uint32_t gfx_version) {
  struct Entry {
    uint32_t ver;
    uint32_t mach;
  };
  static constexpr Entry TABLE[] = {
#define X(NUM, ENUM, NAME, VER) {VER, NUM},
      AMDGPU_MACH_LIST(X)
#undef X
  };
  for (const auto &e : TABLE)
    if (e.ver != 0 && e.ver == gfx_version)
      return e.mach;
  return EF_AMDGPU_MACH_NONE;
}

int format_gfx_version(char *buf, size_t size, uint32_t version) {
  return std::snprintf(buf, size, "gfx%u%u%x", gfx_version_major(version),
                       gfx_version_minor(version), gfx_version_step(version));
}

bool is_generic_mach(uint32_t mach) {
  switch (mach) {
  case EF_AMDGPU_MACH_AMDGCN_GFX9_GENERIC:
  case EF_AMDGPU_MACH_AMDGCN_GFX9_4_GENERIC:
  case EF_AMDGPU_MACH_AMDGCN_GFX10_1_GENERIC:
  case EF_AMDGPU_MACH_AMDGCN_GFX10_3_GENERIC:
  case EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC:
  case EF_AMDGPU_MACH_AMDGCN_GFX12_GENERIC:
  case EF_AMDGPU_MACH_AMDGCN_GFX12_5_GENERIC:
    return true;
  default:
    return false;
  }
}

uint32_t get_generic_for_gpu(uint32_t gfx_version) {
  struct Entry {
    uint32_t generic_mach;
    uint32_t gfx_ver;
  };
  static constexpr Entry TABLE[] = {
#define X(GENERIC, VER, MIN_VER) {GENERIC, VER},
      AMDGPU_GENERIC_MAP(X)
#undef X
  };
  for (const auto &e : TABLE)
    if (e.gfx_ver == gfx_version)
      return e.generic_mach;
  return EF_AMDGPU_MACH_NONE;
}

// Checks if an ELF (e_flags) can run on the device (gfx_version).
bool is_compatible(uint32_t e_flags, uint32_t gfx_version, bool xnack_enabled,
                   bool sramecc_enabled) {
  // Check XNACK feature requirement.
  switch (e_flags & EF_AMDGPU_FEATURE_XNACK_V4) {
  case EF_AMDGPU_FEATURE_XNACK_ON_V4:
    if (!xnack_enabled)
      return false;
    break;
  case EF_AMDGPU_FEATURE_XNACK_OFF_V4:
    if (xnack_enabled)
      return false;
    break;
  default:
    break;
  }

  // Check SRAMECC feature requirement.
  switch (e_flags & EF_AMDGPU_FEATURE_SRAMECC_V4) {
  case EF_AMDGPU_FEATURE_SRAMECC_ON_V4:
    if (!sramecc_enabled)
      return false;
    break;
  case EF_AMDGPU_FEATURE_SRAMECC_OFF_V4:
    if (sramecc_enabled)
      return false;
    break;
  default:
    break;
  }

  uint32_t mach = e_flags & EF_AMDGPU_MACH;

  if (!is_generic_mach(mach))
    return get_gfx_version(mach) == gfx_version;

  // Generic target, look up the GPU in this generic's support table and
  // verify the code object's generic version is high enough.
  uint32_t co_version =
      (e_flags & EF_AMDGPU_GENERIC_VERSION) >> EF_AMDGPU_GENERIC_VERSION_OFFSET;

  struct Entry {
    uint32_t generic_mach;
    uint32_t gfx_ver;
    uint32_t min_version;
  };
  static constexpr Entry TABLE[] = {
#define X(GENERIC, VER, MIN_VER) {GENERIC, VER, MIN_VER},
      AMDGPU_GENERIC_MAP(X)
#undef X
  };
  for (const auto &e : TABLE)
    if (e.generic_mach == mach && e.gfx_ver == gfx_version &&
        co_version >= e.min_version)
      return true;

  return false;
}

// ELF64LE construction and validation.
std::expected<ELF64LE, Error> ELF64LE::create(std::span<const std::byte> buf) {
  if (buf.size() < sizeof(Elf64_Ehdr))
    return kfd::unexpected(ENOEXEC, "buffer %zu bytes, need >= %zu", buf.size(),
                           sizeof(Elf64_Ehdr));
  if (reinterpret_cast<uintptr_t>(buf.data()) % alignof(Elf64_Ehdr) > 0)
    return kfd::unexpected(EINVAL, "buffer %p not aligned, need >= %zu",
                           static_cast<const void *>(buf.data()),
                           alignof(Elf64_Ehdr));

  const auto &ehdr = *reinterpret_cast<const Elf64_Ehdr *>(buf.data());

  if (ehdr.e_ident[EI_MAG0] != ELFMAG0 || ehdr.e_ident[EI_MAG1] != ELFMAG1 ||
      ehdr.e_ident[EI_MAG2] != ELFMAG2 || ehdr.e_ident[EI_MAG3] != ELFMAG3)
    return kfd::unexpected(ENOEXEC, "invalid ELF magic");

  if (ehdr.e_ident[EI_CLASS] != ELFCLASS64)
    return kfd::unexpected(ENOEXEC, "not ELF64 (EI_CLASS=%u)",
                           static_cast<unsigned>(ehdr.e_ident[EI_CLASS]));

  if (ehdr.e_ident[EI_DATA] != ELFDATA2LSB)
    return kfd::unexpected(ENOEXEC, "not little-endian (EI_DATA=%u)",
                           static_cast<unsigned>(ehdr.e_ident[EI_DATA]));

  if (ehdr.e_phnum && ehdr.e_phentsize < sizeof(Elf64_Phdr))
    return kfd::unexpected(ENOEXEC, "e_phentsize %u < %zu",
                           static_cast<unsigned>(ehdr.e_phentsize),
                           sizeof(Elf64_Phdr));

  uint64_t ph_table = static_cast<uint64_t>(ehdr.e_phnum) *
                      static_cast<uint64_t>(ehdr.e_phentsize);
  if (ehdr.e_phnum &&
      (ph_table / ehdr.e_phentsize != ehdr.e_phnum || ph_table > buf.size() ||
       ehdr.e_phoff > buf.size() - ph_table))
    return kfd::unexpected(ERANGE, "phdrs end past buffer %zu", buf.size());

  if (ehdr.e_shnum && ehdr.e_shentsize < sizeof(Elf64_Shdr))
    return kfd::unexpected(ENOEXEC, "e_shentsize %u < %zu",
                           static_cast<unsigned>(ehdr.e_shentsize),
                           sizeof(Elf64_Shdr));

  uint64_t sh_table = static_cast<uint64_t>(ehdr.e_shnum) *
                      static_cast<uint64_t>(ehdr.e_shentsize);
  if (ehdr.e_shnum &&
      (sh_table / ehdr.e_shentsize != ehdr.e_shnum || sh_table > buf.size() ||
       ehdr.e_shoff > buf.size() - sh_table))
    return kfd::unexpected(ERANGE, "sections end past buffer %zu", buf.size());

  if (ehdr.e_shnum && ehdr.e_shstrndx >= ehdr.e_shnum)
    return kfd::unexpected(ENOEXEC, "e_shstrndx %u >= e_shnum %u",
                           static_cast<unsigned>(ehdr.e_shstrndx),
                           static_cast<unsigned>(ehdr.e_shnum));

  return ELF64LE(buf);
}

// Segment access.
std::span<const Elf64_Phdr> ELF64LE::phdrs() const {
  const auto &ehdr = header();
  if (!ehdr.e_phnum)
    return {};
  const auto *p =
      reinterpret_cast<const Elf64_Phdr *>(buf.data() + ehdr.e_phoff);
  return {p, ehdr.e_phnum};
}

std::expected<std::span<const std::byte>, Error>
ELF64LE::segment_data(const Elf64_Phdr &phdr) const {
  if (phdr.p_filesz > buf.size() || phdr.p_offset > buf.size() - phdr.p_filesz)
    return kfd::unexpected(
        ERANGE, "segment offset 0x%lx + filesz 0x%lx past buffer %zu",
        static_cast<unsigned long>(phdr.p_offset),
        static_cast<unsigned long>(phdr.p_filesz), buf.size());
  return buf.subspan(phdr.p_offset, phdr.p_filesz);
}

std::expected<ELF64LE::LoadExtent, Error> ELF64LE::load_extent() const {
  uint64_t lo = UINT64_MAX, hi = 0, max_align = 1;
  for (const auto &ph : phdrs()) {
    if (ph.p_type != PT_LOAD)
      continue;
    if (ph.p_vaddr < lo)
      lo = ph.p_vaddr;
    if (ph.p_memsz > UINT64_MAX - ph.p_vaddr)
      return kfd::unexpected(ERANGE,
                             "PT_LOAD vaddr 0x%lx + memsz 0x%lx "
                             "overflows uint64_t",
                             static_cast<unsigned long>(ph.p_vaddr),
                             static_cast<unsigned long>(ph.p_memsz));
    uint64_t end = ph.p_vaddr + ph.p_memsz;
    if (end > hi)
      hi = end;
    if (ph.p_align > max_align)
      max_align = ph.p_align;
  }
  if (lo >= hi)
    return kfd::unexpected(ENOEXEC, "no PT_LOAD segments in ELF");
  return LoadExtent{lo, hi, max_align};
}

// Section access.
std::span<const Elf64_Shdr> ELF64LE::sections() const {
  const auto &ehdr = header();
  if (!ehdr.e_shnum)
    return {};
  const auto *p =
      reinterpret_cast<const Elf64_Shdr *>(buf.data() + ehdr.e_shoff);
  return {p, ehdr.e_shnum};
}

std::expected<std::string_view, Error>
ELF64LE::section_name(const Elf64_Shdr &shdr) const {
  const auto &ehdr = header();
  auto shdrs = sections();
  if (ehdr.e_shstrndx >= shdrs.size())
    return kfd::unexpected(ENOEXEC, "e_shstrndx %u >= section count %zu",
                           static_cast<unsigned>(ehdr.e_shstrndx),
                           shdrs.size());

  const auto &strtab_hdr = shdrs[ehdr.e_shstrndx];
  if (shdr.sh_name >= strtab_hdr.sh_size)
    return kfd::unexpected(ENOEXEC, "sh_name offset %u >= strtab size %lu",
                           static_cast<unsigned>(shdr.sh_name),
                           static_cast<unsigned long>(strtab_hdr.sh_size));
  if (strtab_hdr.sh_size > buf.size() ||
      strtab_hdr.sh_offset > buf.size() - strtab_hdr.sh_size)
    return kfd::unexpected(ERANGE, "strtab end past buffer %zu", buf.size());

  const char *base =
      reinterpret_cast<const char *>(buf.data()) + strtab_hdr.sh_offset;
  size_t max_len = strtab_hdr.sh_size - shdr.sh_name;
  const char *start = base + shdr.sh_name;
  const void *nul = std::memchr(start, '\0', max_len);
  if (!nul)
    return kfd::unexpected(ENOEXEC, "unterminated section name at offset %u",
                           static_cast<unsigned>(shdr.sh_name));
  return std::string_view{
      start, static_cast<size_t>(static_cast<const char *>(nul) - start)};
}

std::expected<std::span<const std::byte>, Error>
ELF64LE::section_data(const Elf64_Shdr &shdr) const {
  if (shdr.sh_type == SHT_NOBITS)
    return std::span<const std::byte>{};
  if (shdr.sh_size > buf.size() || shdr.sh_offset > buf.size() - shdr.sh_size)
    return kfd::unexpected(ERANGE, "section end past buffer %zu", buf.size());
  return buf.subspan(shdr.sh_offset, shdr.sh_size);
}

// Internal helpers for symbol table access.
const Elf64_Shdr *ELF64LE::strtab_for_symtab(const Elf64_Shdr &symtab) const {
  auto shdrs = sections();
  if (symtab.sh_link >= shdrs.size())
    return nullptr;
  return &shdrs[symtab.sh_link];
}

std::span<const Elf64_Sym>
ELF64LE::symbols_for(const Elf64_Shdr &symtab) const {
  if (symtab.sh_entsize != sizeof(Elf64_Sym) || symtab.sh_size > buf.size() ||
      symtab.sh_offset > buf.size() - symtab.sh_size)
    return {};
  auto count = static_cast<size_t>(symtab.sh_size / symtab.sh_entsize);
  const auto *p =
      reinterpret_cast<const Elf64_Sym *>(buf.data() + symtab.sh_offset);
  return {p, count};
}

namespace {

std::string_view sym_name(const Elf64_Sym &sym, const char *strtab,
                          uint64_t strtab_size) {
  if (sym.st_name >= strtab_size)
    return {};
  size_t max_len = static_cast<size_t>(strtab_size - sym.st_name);
  const char *start = strtab + sym.st_name;
  const void *nul = std::memchr(start, '\0', max_len);
  if (!nul)
    return {};
  return {start, static_cast<size_t>(static_cast<const char *>(nul) - start)};
}

const Elf64_Sym *lookup_gnu_hash(std::string_view name, const Elf64_GnuHash &ht,
                                 uint64_t ht_size,
                                 std::span<const Elf64_Sym> syms,
                                 const char *strtab, uint64_t strtab_size) {
  if (ht.nbuckets == 0 || ht.maskwords == 0)
    return nullptr;

  // Validate that filter, buckets, and values fit within the section.
  uint64_t header_bytes = sizeof(Elf64_GnuHash);
  uint64_t filter_bytes =
      static_cast<uint64_t>(ht.maskwords) * sizeof(uint64_t);
  uint64_t bucket_bytes =
      static_cast<uint64_t>(ht.nbuckets) * sizeof(Elf64_Word);
  uint64_t min_size = header_bytes + filter_bytes + bucket_bytes;
  if (min_size < header_bytes || min_size > ht_size)
    return nullptr;

  if (ht.shift2 >= 32)
    return nullptr;

  uint32_t name_hash = hash_gnu(name);

  // Bloom filter check.
  uint64_t word = ht.filter()[(name_hash / 64) % ht.maskwords];
  uint64_t mask =
      (1ull << (name_hash % 64)) | (1ull << ((name_hash >> ht.shift2) % 64));
  if ((word & mask) != mask)
    return nullptr;

  const auto *buckets = ht.buckets();
  const auto *values = ht.values();
  uint64_t values_bytes = ht_size - min_size;
  uint64_t n_values = values_bytes / sizeof(Elf64_Word);

  for (Elf64_Word i = buckets[name_hash % ht.nbuckets];
       i >= ht.symndx && i < syms.size(); ++i) {
    uint32_t vi = i - ht.symndx;
    if (vi >= n_values)
      break;
    uint32_t chain_hash = values[vi];

    if ((name_hash | 1) == (chain_hash | 1) &&
        sym_name(syms[i], strtab, strtab_size) == name)
      return &syms[i];

    if (chain_hash & 1)
      break;
  }
  return nullptr;
}

} // namespace

std::expected<const Elf64_Sym *, Error>
ELF64LE::find_symbol(std::string_view name) const {
  auto shdrs = sections();

  // Walk sections looking for the GNU hash table and symbol table.
  for (const auto &shdr : shdrs) {
    if (shdr.sh_type != SHT_GNU_HASH)
      continue;

    if (shdr.sh_size > buf.size() || shdr.sh_offset > buf.size() - shdr.sh_size)
      return kfd::unexpected(ERANGE, "hash section end past buffer %zu",
                             buf.size());

    if (shdr.sh_size < sizeof(Elf64_GnuHash))
      return kfd::unexpected(ENOEXEC, "hash section too small (%lu < %zu)",
                             static_cast<unsigned long>(shdr.sh_size),
                             sizeof(Elf64_GnuHash));

    if (shdr.sh_link >= shdrs.size())
      return kfd::unexpected(ENOEXEC, "hash sh_link %u >= section count %zu",
                             static_cast<unsigned>(shdr.sh_link), shdrs.size());
    const auto &symtab_hdr = shdrs[shdr.sh_link];

    auto syms = symbols_for(symtab_hdr);
    if (syms.empty())
      continue;

    const auto *str_hdr = strtab_for_symtab(symtab_hdr);
    if (!str_hdr)
      return kfd::unexpected(ENOEXEC, "symtab has no valid strtab (sh_link)");
    if (str_hdr->sh_size > buf.size() ||
        str_hdr->sh_offset > buf.size() - str_hdr->sh_size)
      return kfd::unexpected(ERANGE, "symtab strtab end past buffer %zu",
                             buf.size());

    const char *strtab =
        reinterpret_cast<const char *>(buf.data()) + str_hdr->sh_offset;
    uint64_t strtab_size = str_hdr->sh_size;

    const auto *ht =
        reinterpret_cast<const Elf64_GnuHash *>(buf.data() + shdr.sh_offset);
    if (const Elf64_Sym *result =
            lookup_gnu_hash(name, *ht, shdr.sh_size, syms, strtab, strtab_size))
      return result;
  }

  return nullptr;
}

// Symbol address resolution.
std::expected<const void *, Error>
ELF64LE::symbol_address(const Elf64_Sym &sym) const {
  auto shdrs = sections();
  if (sym.st_shndx >= shdrs.size())
    return kfd::unexpected(ENOEXEC,
                           "symbol st_shndx %u out of range (sections %zu)",
                           static_cast<unsigned>(sym.st_shndx), shdrs.size());

  const auto &sec = shdrs[sym.st_shndx];
  if (sec.sh_type == SHT_NOBITS)
    return kfd::unexpected(ENOEXEC, "symbol in NOBITS section");

  if (sec.sh_size > UINT64_MAX - sec.sh_addr)
    return kfd::unexpected(ERANGE, "section addr 0x%lx + size 0x%lx wraps",
                           static_cast<unsigned long>(sec.sh_addr),
                           static_cast<unsigned long>(sec.sh_size));

  uint64_t sec_rel = sym.st_value - sec.sh_addr;
  if (sym.st_value < sec.sh_addr || sec_rel > sec.sh_size ||
      sym.st_size > sec.sh_size - sec_rel)
    return kfd::unexpected(
        ERANGE,
        "symbol value 0x%lx + size 0x%lx outside section "
        "[0x%lx, 0x%lx)",
        static_cast<unsigned long>(sym.st_value),
        static_cast<unsigned long>(sym.st_size),
        static_cast<unsigned long>(sec.sh_addr),
        static_cast<unsigned long>(sec.sh_addr + sec.sh_size));

  if (sec_rel > UINT64_MAX - sec.sh_offset)
    return kfd::unexpected(ERANGE, "section offset 0x%lx + rel 0x%lx wraps",
                           static_cast<unsigned long>(sec.sh_offset),
                           static_cast<unsigned long>(sec_rel));
  uint64_t offset = sec.sh_offset + sec_rel;
  if (offset >= buf.size() || sym.st_size > buf.size() - offset)
    return kfd::unexpected(ERANGE,
                           "symbol file offset 0x%lx + size 0x%lx past "
                           "buffer of %zu bytes",
                           static_cast<unsigned long>(offset),
                           static_cast<unsigned long>(sym.st_size), buf.size());

  return buf.data() + offset;
}

SymbolRange ELF64LE::symbols() const {
  SymbolRange range;
  for (const auto &shdr : sections()) {
    if (shdr.sh_type != SHT_DYNSYM)
      continue;

    auto shdrs = sections();
    if (shdr.sh_link >= shdrs.size() || shdr.sh_entsize != sizeof(Elf64_Sym))
      break;

    auto sym_data = section_data(shdr);
    if (!sym_data || sym_data->empty())
      break;

    auto str_data = section_data(shdrs[shdr.sh_link]);
    if (!str_data)
      break;

    size_t count = shdr.sh_size / shdr.sh_entsize;
    if (count <= 1)
      break;

    const auto *syms = reinterpret_cast<const Elf64_Sym *>(sym_data->data());
    const char *strtab = reinterpret_cast<const char *>(str_data->data());

    auto make = [&](const Elf64_Sym *s) {
      SymbolRange::iterator it;
      it.cur = s;
      it.strtab = strtab;
      it.strtab_size = str_data->size();
      return it;
    };
    range.first = make(syms + 1);
    range.last = make(syms + count);
    break;
  }
  return range;
}

} // namespace kfd::detail::elf
