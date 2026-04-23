//===-- tools/kfdinfo/main.cpp - GPU topology inspector ---------*- C++ -*-===//
//
// Reads the KFD sysfs topology and prints a comprehensive, human-readable
// summary of every GPU node. Includes identity, compute layout, memory banks,
// cache hierarchy, IO links, firmware, and capabilities.
//
//===----------------------------------------------------------------------===//

#include "libkfd/detail/elf.h"
#include "libkfd/topology.h"

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

namespace {

constexpr const char *BOLD = "\033[1m";
constexpr const char *DIM = "\033[2m";
constexpr const char *CYAN = "\033[36m";
constexpr const char *GREEN = "\033[32m";
constexpr const char *YELLOW = "\033[33m";
constexpr const char *RESET = "\033[0m";

bool use_color = false;

const char *b() { return use_color ? BOLD : ""; }
const char *d() { return use_color ? DIM : ""; }
const char *c() { return use_color ? CYAN : ""; }
const char *g() { return use_color ? GREEN : ""; }
const char *y() { return use_color ? YELLOW : ""; }
const char *r() { return use_color ? RESET : ""; }

const char *heap_type_name(uint32_t type) {
  switch (type) {
  case 0:
    return "System";
  case 1:
    return "VRAM (public)";
  case 2:
    return "VRAM (private)";
  case 3:
    return "GDS";
  case 4:
    return "LDS";
  case 5:
    return "Scratch";
  default:
    return "Unknown";
  }
}

const char *link_type_name(uint32_t type) {
  switch (type) {
  case 0:
    return "Undefined";
  case 1:
    return "HyperTransport";
  case 2:
    return "PCIe";
  case 3:
    return "AMBA";
  case 4:
    return "MIPI";
  case 11:
    return "XGMI";
  default:
    return "Unknown";
  }
}

void format_size(char *buf, size_t len, uint64_t bytes) {
  if (bytes >= (1ULL << 30))
    std::snprintf(buf, len, "%.1f GiB", static_cast<double>(bytes) / (1 << 30));
  else if (bytes >= (1ULL << 20))
    std::snprintf(buf, len, "%.1f MiB", static_cast<double>(bytes) / (1 << 20));
  else if (bytes >= (1ULL << 10))
    std::snprintf(buf, len, "%.1f KiB", static_cast<double>(bytes) / (1 << 10));
  else
    std::snprintf(buf, len, "%lu B", static_cast<unsigned long>(bytes));
}

void format_cache_type(char *buf, size_t len, uint32_t type) {
  bool data = type & 1;
  bool inst = type & 2;
  bool hsacu = type & 8;

  if (data && inst)
    std::snprintf(buf, len, "Unified%s", hsacu ? " (GPU)" : "");
  else if (data)
    std::snprintf(buf, len, "Data%s", hsacu ? " (GPU)" : "");
  else if (inst)
    std::snprintf(buf, len, "Instruction%s", hsacu ? " (GPU)" : "");
  else
    std::snprintf(buf, len, "0x%x", type);
}

void print_separator() {
  std::printf("%s", d());
  for (int i = 0; i < 72; ++i)
    std::putchar('-');
  std::printf("%s\n", r());
}

[[gnu::format(printf, 2, 3)]]
void print_field(const char *label, const char *fmt, ...) {
  std::printf("  %-28s ", label);
  va_list ap;
  va_start(ap, fmt);
  std::vprintf(fmt, ap);
  va_end(ap);
  std::putchar('\n');
}

void print_section(const char *title) {
  std::printf("\n  %s%s%s\n", y(), title, r());
}

void print_node(const kfd::NodeInfo &node, uint32_t index) {
  const auto &p = node.props;
  using namespace kfd::detail;

  char gfx_str[32];
  elf::format_gfx_version(gfx_str, sizeof(gfx_str), p.gfx_target_version);
  auto name = elf::get_name(elf::get_mach(p.gfx_target_version));

  // Decode PCI BDF from location_id.
  uint32_t bus = (p.location_id >> 8) & 0xff;
  uint32_t dev = (p.location_id >> 3) & 0x1f;
  uint32_t func = p.location_id & 0x07;

  // Derive compute unit count.
  uint32_t simd_per_cu = p.simd_per_cu ? p.simd_per_cu : 1;
  uint32_t num_cus = p.simd_count / simd_per_cu;
  uint32_t num_se =
      p.simd_arrays_per_engine ? p.array_count / p.simd_arrays_per_engine : 0;

  std::printf("\n%s%s=== GPU %u ===%s\n", b(), c(), index, r());

  print_section("Identity");
  if (!name.empty())
    print_field("Name:", "%s%.*s%s", g(), static_cast<int>(name.size()),
                name.data(), r());
  print_field("ISA:", "%s%s%s", g(), gfx_str, r());
  print_field("Device ID:", "0x%04x", p.device_id);
  print_field("Vendor ID:", "0x%04x", p.vendor_id);
  if (p.family_id)
    print_field("Family ID:", "%u", p.family_id);
  print_field("PCI Location:", "%s%04x:%02x:%02x.%x%s", g(), p.domain, bus, dev,
              func, r());
  print_field("KFD GPU ID:", "%u", p.gpu_id);
  print_field("Render Node:", "/dev/dri/renderD%u", p.drm_render_minor);
  if (p.unique_id)
    print_field("Unique ID:", "0x%016lx",
                static_cast<unsigned long>(p.unique_id));
  if (p.hive_id)
    print_field("Hive ID:", "0x%016lx", static_cast<unsigned long>(p.hive_id));

  print_section("Compute");
  print_field("Compute Units:", "%s%u%s", g(), num_cus, r());
  if (num_se)
    print_field("Shader Engines:", "%u", num_se);
  if (p.simd_arrays_per_engine)
    print_field("Shader Arrays / SE:", "%u", p.simd_arrays_per_engine);
  print_field("CUs / Shader Array:", "%u", p.cu_per_simd_array);
  print_field("SIMDs / CU:", "%u", p.simd_per_cu);
  print_field("Wavefront Size:", "%u", p.wave_front_size);
  print_field("Max Waves / SIMD:", "%u", p.max_waves_per_simd);
  uint32_t total_waves = p.max_waves_per_simd * p.simd_count;
  print_field("Max Concurrent Waves:", "%s%u%s", g(), total_waves, r());
  if (p.num_xcc > 1)
    print_field("XCCs:", "%u", p.num_xcc);
  if (p.max_engine_clk_fcompute)
    print_field("Max Engine Clock:", "%u MHz", p.max_engine_clk_fcompute);

  print_section("Memory");
  char size_buf[32];
  uint64_t vram_total = p.local_mem_size;
  if (!vram_total) {
    for (size_t i = 0; i < node.memory_banks.size(); ++i)
      if (node.memory_banks[i].heap_type == 1 ||
          node.memory_banks[i].heap_type == 2)
        vram_total += node.memory_banks[i].size_in_bytes;
  }
  if (vram_total) {
    format_size(size_buf, sizeof(size_buf), vram_total);
    print_field("VRAM:", "%s%s%s", g(), size_buf, r());
  }
  if (p.lds_size_in_kb)
    print_field("LDS / Workgroup:", "%u KiB", p.lds_size_in_kb);
  if (p.gds_size_in_kb)
    print_field("GDS:", "%u KiB", p.gds_size_in_kb);

  if (!node.memory_banks.empty()) {
    print_section("Memory Banks");
    for (size_t i = 0; i < node.memory_banks.size(); ++i) {
      const auto &bank = node.memory_banks[i];
      format_size(size_buf, sizeof(size_buf), bank.size_in_bytes);
      std::printf("  %s[%zu]%s %-10s %10s", d(), i, r(),
                  heap_type_name(bank.heap_type), size_buf);
      if (bank.width)
        std::printf("  %u-bit", bank.width);
      if (bank.mem_clk_max)
        std::printf("  %u MHz", bank.mem_clk_max);
      std::putchar('\n');
    }
  }

  if (!node.caches.empty()) {
    print_section("Caches");
    for (size_t i = 0; i < node.caches.size(); ++i) {
      const auto &cache = node.caches[i];
      char type_buf[32];
      format_cache_type(type_buf, sizeof(type_buf), cache.type);
      format_size(size_buf, sizeof(size_buf),
                  static_cast<uint64_t>(cache.size) * 1024);
      std::printf("  %sL%u%s  %-20s %10s", g(), cache.level, r(), type_buf,
                  size_buf);
      if (cache.line_size)
        std::printf("  %3u B/line", cache.line_size);
      if (cache.associativity)
        std::printf("  %u-way", cache.associativity);
      std::putchar('\n');
    }
  }

  print_section("Queues & Engines");
  print_field("CP Queues:", "%u", p.num_cp_queues);
  print_field("SDMA Engines:", "%u", p.num_sdma_engines);
  if (p.num_sdma_xgmi_engines)
    print_field("SDMA XGMI Engines:", "%u", p.num_sdma_xgmi_engines);
  print_field("Queues / SDMA Engine:", "%u", p.num_sdma_queues_per_engine);
  print_field("Scratch Slots / CU:", "%u", p.max_slots_scratch_cu);
  if (p.num_gws)
    print_field("GWS Barriers:", "%u", p.num_gws);

  print_section("Firmware");
  print_field("GFX uCode:", "%u", p.fw_version);
  print_field("SDMA uCode:", "%u", p.sdma_fw_version);

  if (!node.io_links.empty()) {
    print_section("IO Links");
    for (size_t i = 0; i < node.io_links.size(); ++i) {
      const auto &link = node.io_links[i];
      std::printf("  %s[%zu]%s Node %u -> %u  %-6s  weight=%u", d(), i, r(),
                  link.node_from, link.node_to, link_type_name(link.type),
                  link.weight);
      if (link.min_bandwidth || link.max_bandwidth)
        std::printf("  BW=%u-%u MB/s", link.min_bandwidth, link.max_bandwidth);
      if (link.min_latency || link.max_latency)
        std::printf("  lat=%u-%u ns", link.min_latency, link.max_latency);
      std::putchar('\n');
    }
  }

  print_section("Capabilities");
  print_field("CWSR Size:", "%g KiB", p.cwsr_size / 1024.0);
  print_field("Ctl Stack Size:", "%g KiB", p.ctl_stack_size / 1024.0);
  print_field("Capability:", "0x%08x", p.capability);
  if (p.capability2)
    print_field("Capability2:", "0x%08x", p.capability2);
  if (p.debug_prop)
    print_field("Debug Properties:", "0x%08x", p.debug_prop);
  bool sram_ecc =
      p.capability & kfd::NodeProperties::NODE_CAP_SRAM_EDCSUPPORTED;
  print_field("SRAM ECC:", "%s", sram_ecc ? "Yes" : "No");
}

} // namespace

int main(int argc, char **argv) {
  // Detect terminal color support.
  use_color = ::isatty(1);
  for (int i = 1; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--no-color")
      use_color = false;
    else if (std::string_view(argv[i]) == "--color")
      use_color = true;
  }

  auto topo = KFD_EXPECT(kfd::Topology::create());

  auto nodes = topo.nodes();
  std::printf("%s%skfdinfo%s - KFD Topology Inspector\n", b(), c(), r());
  print_separator();
  std::printf("  Topology generation:   %u\n", topo.generation());
  std::printf("  GPU nodes discovered:  %s%zu%s\n", g(), nodes.size(), r());

  for (size_t i = 0; i < nodes.size(); ++i) {
    print_separator();
    print_node(nodes[i], static_cast<uint32_t>(i));
  }

  print_separator();
  std::putchar('\n');

  return 0;
}
