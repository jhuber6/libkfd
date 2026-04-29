//===-- lib/topology.cpp - KFD sysfs topology discovery ---------*- C++ -*-===//
//
// Reads /sys/devices/virtual/kfd/kfd/topology to enumerate nodes and parse
// their properties. The generation_id is checked before and after the scan,
// if the kernel changed the topology mid-read we retry the whole snapshot.
//
//===----------------------------------------------------------------------===//

#include "libkfd/topology.h"
#include "libkfd/abi.h"
#include "libkfd/detail/mapped_region.h"
#include "libkfd/detail/utility.h"

#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <string_view>
#include <unistd.h>
#include <utility>

using namespace kfd::detail;

namespace kfd {

namespace {

constexpr const char SYSFS_TOPOLOGY[] = "/sys/devices/virtual/kfd/kfd/topology";
constexpr int MAX_GENERATION_RETRIES = 3;

// Map the sysfs file into a memory region for parsing.
std::expected<detail::MappedRegion, Error> read_sysfs(const char *path) {
  int fd = ::open(path, O_RDONLY);
  if (fd < 0) {
    int err = errno;
    return kfd::unexpected(err, "failed to open sysfs file '%s'", path);
  }

  auto region = detail::MappedRegion::create(detail::page_size());
  if (!region) {
    ::close(fd);
    return kfd::unexpected(region.error());
  }

  size_t total = 0;
  for (;;) {
    auto *buf = static_cast<char *>(region->data());
    while (total < region->size()) {
      ssize_t n = ::read(fd, buf + total, region->size() - total);
      if (n < 0) {
        if (errno == EINTR)
          continue;
        int err = errno;
        ::close(fd);
        return kfd::unexpected(err, "read error on sysfs file");
      }
      if (n == 0) {
        ::close(fd);
        return region;
      }
      total += static_cast<size_t>(n);
    }
    if (!region->try_grow(region->size() * 2)) {
      ::close(fd);
      return kfd::unexpected(ENOMEM, "sysfs read buffer exceeded limit");
    }
  }
}

// Read a sysfs file containing a single unsigned integer.
std::expected<uint64_t, Error> read_sysfs_uint(const char *path) {
  auto file = KFD_TRY(read_sysfs(path));

  std::string_view text(file.as_span<char>());
  return KFD_TRY(consume_integer(text));
}

// Parse the known strings from the sysfs node properties.
void parse_property(NodeProperties &props, std::string_view key, uint64_t val) {
  auto u32 = static_cast<uint32_t>(val);

  if (key == "vendor_id")
    props.vendor_id = u32;
  else if (key == "device_id")
    props.device_id = u32;
  else if (key == "location_id")
    props.location_id = u32;
  else if (key == "domain")
    props.domain = u32;
  else if (key == "gfx_target_version")
    props.gfx_target_version = u32;
  else if (key == "drm_render_minor")
    props.drm_render_minor = u32;
  else if (key == "cpu_cores_count")
    props.cpu_cores_count = u32;
  else if (key == "simd_count")
    props.simd_count = u32;
  else if (key == "array_count")
    props.array_count = u32;
  else if (key == "simd_arrays_per_engine")
    props.simd_arrays_per_engine = u32;
  else if (key == "cu_per_simd_array")
    props.cu_per_simd_array = u32;
  else if (key == "simd_per_cu")
    props.simd_per_cu = u32;
  else if (key == "max_waves_per_simd")
    props.max_waves_per_simd = u32;
  else if (key == "wave_front_size")
    props.wave_front_size = u32;
  else if (key == "max_slots_scratch_cu")
    props.max_slots_scratch_cu = u32;
  else if (key == "mem_banks_count")
    props.mem_banks_count = u32;
  else if (key == "caches_count")
    props.caches_count = u32;
  else if (key == "io_links_count")
    props.io_links_count = u32;
  else if (key == "num_sdma_engines")
    props.num_sdma_engines = u32;
  else if (key == "num_sdma_xgmi_engines")
    props.num_sdma_xgmi_engines = u32;
  else if (key == "num_sdma_queues_per_engine")
    props.num_sdma_queues_per_engine = u32;
  else if (key == "num_cp_queues")
    props.num_cp_queues = u32;
  else if (key == "num_xcc")
    props.num_xcc = u32;
  else if (key == "num_gws")
    props.num_gws = u32;
  else if (key == "max_engine_clk_fcompute")
    props.max_engine_clk_fcompute = u32;
  else if (key == "max_engine_clk_ccompute")
    props.max_engine_clk_ccompute = u32;
  else if (key == "lds_size_in_kb")
    props.lds_size_in_kb = u32;
  else if (key == "gds_size_in_kb")
    props.gds_size_in_kb = u32;
  else if (key == "local_mem_size")
    props.local_mem_size = val;
  else if (key == "family_id")
    props.family_id = u32;
  else if (key == "fw_version")
    props.fw_version = u32;
  else if (key == "sdma_fw_version")
    props.sdma_fw_version = u32;
  else if (key == "capability")
    props.capability = u32;
  else if (key == "capability2")
    props.capability2 = u32;
  else if (key == "debug_prop")
    props.debug_prop = u32;
  else if (key == "cwsr_size")
    props.cwsr_size = u32;
  else if (key == "ctl_stack_size")
    props.ctl_stack_size = u32;
  else if (key == "unique_id")
    props.unique_id = val;
  else if (key == "hive_id")
    props.hive_id = val;
}

// Each entry in the node properties is a string followed by a number, parse it
// line by line to fill the node properties.
NodeProperties parse_properties(std::string_view text, uint32_t gpu_id) {
  NodeProperties props{};
  props.gpu_id = gpu_id;

  while (!text.empty()) {
    auto line = consume_line(text);
    auto [key, val_str] = split(line, ' ');
    if (key.empty())
      continue;

    auto val = consume_integer(val_str);
    if (!val)
      continue;

    parse_property(props, key, *val);
  }

  return props;
}

MemoryBank parse_memory_bank(std::string_view text) {
  MemoryBank bank{};
  while (!text.empty()) {
    auto line = consume_line(text);
    auto [key, val_str] = split(line, ' ');
    if (key.empty())
      continue;
    auto val = consume_integer(val_str);
    if (!val)
      continue;
    auto u32 = static_cast<uint32_t>(*val);
    if (key == "heap_type")
      bank.heap_type = u32;
    else if (key == "size_in_bytes")
      bank.size_in_bytes = *val;
    else if (key == "flags")
      bank.flags = u32;
    else if (key == "width")
      bank.width = u32;
    else if (key == "mem_clk_max")
      bank.mem_clk_max = u32;
  }
  return bank;
}

CacheInfo parse_cache(std::string_view text) {
  CacheInfo cache{};
  while (!text.empty()) {
    auto line = consume_line(text);
    auto [key, val_str] = split(line, ' ');
    if (key.empty())
      continue;
    auto val = consume_integer(val_str);
    if (!val)
      continue;
    auto u32 = static_cast<uint32_t>(*val);
    if (key == "processor_id_low")
      cache.processor_id_low = u32;
    else if (key == "level")
      cache.level = u32;
    else if (key == "size")
      cache.size = u32;
    else if (key == "cache_line_size")
      cache.line_size = u32;
    else if (key == "cache_lines_per_tag")
      cache.lines_per_tag = u32;
    else if (key == "association")
      cache.associativity = u32;
    else if (key == "latency")
      cache.latency = u32;
    else if (key == "type")
      cache.type = u32;
  }
  return cache;
}

IoLink parse_io_link(std::string_view text) {
  IoLink link{};
  while (!text.empty()) {
    auto line = consume_line(text);
    auto [key, val_str] = split(line, ' ');
    if (key.empty())
      continue;
    auto val = consume_integer(val_str);
    if (!val)
      continue;
    auto u32 = static_cast<uint32_t>(*val);
    if (key == "type")
      link.type = u32;
    else if (key == "node_from")
      link.node_from = u32;
    else if (key == "node_to")
      link.node_to = u32;
    else if (key == "weight")
      link.weight = u32;
    else if (key == "min_latency")
      link.min_latency = u32;
    else if (key == "max_latency")
      link.max_latency = u32;
    else if (key == "min_bandwidth")
      link.min_bandwidth = u32;
    else if (key == "max_bandwidth")
      link.max_bandwidth = u32;
    else if (key == "flags")
      link.flags = u32;
  }
  return link;
}

std::expected<detail::SmallVector<MemoryBank, 2>, Error>
read_memory_banks(uint32_t node_index, uint32_t count) {
  detail::SmallVector<MemoryBank, 2> banks;
  char path[256];
  for (uint32_t i = 0; i < count; ++i) {
    std::snprintf(path, sizeof(path), "%s/nodes/%u/mem_banks/%u/properties",
                  SYSFS_TOPOLOGY, node_index, i);
    auto file = KFD_TRY(read_sysfs(path));
    std::string_view view(file.as_span<const char>());
    KFD_CHECK(banks.push_back(parse_memory_bank(view)));
  }
  return banks;
}

std::expected<detail::SmallVector<IoLink, 4>, Error>
read_io_links(uint32_t node_index, uint32_t count) {
  detail::SmallVector<IoLink, 4> links;
  char path[256];
  for (uint32_t i = 0; i < count; ++i) {
    std::snprintf(path, sizeof(path), "%s/nodes/%u/io_links/%u/properties",
                  SYSFS_TOPOLOGY, node_index, i);
    auto file = KFD_TRY(read_sysfs(path));
    std::string_view view(file.as_span<const char>());
    KFD_CHECK(links.push_back(parse_io_link(view)));
  }
  return links;
}

std::expected<detail::SmallVector<CacheInfo, 8>, Error>
read_caches(uint32_t node_index, uint32_t count) {
  detail::SmallVector<CacheInfo, 8> caches;
  char path[256];

  // Several caches are duplicated across each compute unit.
  auto is_duplicate = [](const CacheInfo &a, const CacheInfo &b) {
    return a.level == b.level && a.size == b.size && a.type == b.type;
  };
  for (uint32_t i = 0; i < count; ++i) {
    std::snprintf(path, sizeof(path), "%s/nodes/%u/caches/%u/properties",
                  SYSFS_TOPOLOGY, node_index, i);
    auto file = KFD_TRY(read_sysfs(path));
    std::string_view view(file.as_span<const char>());
    CacheInfo entry = parse_cache(view);

    bool dup = false;
    for (size_t j = 0; j < caches.size(); ++j) {
      if (is_duplicate(caches[j], entry)) {
        dup = true;
        break;
      }
    }
    if (dup)
      continue;

    KFD_CHECK(caches.push_back(entry));
  }
  return caches;
}

std::expected<NodeProperties, Error> read_properties(uint32_t node_index) {
  char path[256];
  std::snprintf(path, sizeof(path), "%s/nodes/%u/gpu_id", SYSFS_TOPOLOGY,
                node_index);
  auto gpu_id = KFD_TRY(read_sysfs_uint(path));

  std::snprintf(path, sizeof(path), "%s/nodes/%u/properties", SYSFS_TOPOLOGY,
                node_index);
  auto file = KFD_TRY(read_sysfs(path));

  std::string_view view(file.as_span<const char>());
  return parse_properties(view, static_cast<uint32_t>(gpu_id));
}

// Probe sequential node indices until open fails.
uint32_t count_nodes() {
  char path[256];
  uint32_t n = 0;
  for (;;) {
    std::snprintf(path, sizeof(path), "%s/nodes/%u", SYSFS_TOPOLOGY, n);
    int fd = ::open(path, O_RDONLY | O_DIRECTORY);
    if (fd < 0)
      break;
    ::close(fd);
    ++n;
  }
  return n;
}

// Read a single node's properties and subtrees.
std::expected<NodeInfo, Error> read_node(uint32_t index) {
  auto props = KFD_TRY(read_properties(index));

  if (props.gpu_id == 0)
    return NodeInfo{props, {}, {}, {}};

  auto banks = KFD_TRY(read_memory_banks(index, props.mem_banks_count));

  auto caches = KFD_TRY(read_caches(index, props.caches_count));

  auto links = KFD_TRY(read_io_links(index, props.io_links_count));

  return NodeInfo{props, std::move(banks), std::move(caches), std::move(links)};
}

// VGPR file size per CU, keyed by gfx_target_version. Matches the kernel's
// kfd_get_vgpr_size_per_cu().
uint32_t vgpr_size_per_cu(uint32_t gfxv) {
  if (gfxv == abi::GFX_VERSION_GFX942 || gfxv == abi::GFX_VERSION_GFX9_A ||
      gfxv == abi::GFX_VERSION_GFX9_8 || gfxv == abi::GFX_VERSION_GFX950)
    return /*512 KiB=*/0x80000;
  if (gfxv >= abi::GFX_VERSION_GFX11)
    return /*384 KiB=*/0x60000;
  return /*256 KiB=*/0x40000;
}

// Mirrors the kernel's kfd_queue_ctx_save_restore_size().
std::pair<uint32_t, uint32_t> compute_cwsr_sizes(const NodeProperties &props) {
  uint32_t gfxv = props.gfx_target_version;
  constexpr uint32_t SGPR_SIZE_PER_CU = /*16 KiB=*/0x4000;
  constexpr uint32_t LDS_SIZE_PER_CU = /*64 KiB=*/0x10000;
  constexpr uint32_t HWREG_SIZE_PER_CU = /*4 KiB=*/0x1000;
  constexpr uint32_t CWSR_HEADER_SIZE = sizeof(abi::CwsrHeader);

  uint32_t num_xcc = props.num_xcc ? props.num_xcc : 1;
  uint32_t simd_per_cu = props.simd_per_cu ? props.simd_per_cu : 1;
  uint32_t cu_num = props.simd_count / simd_per_cu / num_xcc;

  uint32_t wave_num;
  if (gfxv < abi::GFX_VERSION_GFX10_1) {
    uint32_t simd_arrays_per_engine =
        props.simd_arrays_per_engine ? props.simd_arrays_per_engine : 1;
    uint32_t max_waves = props.array_count / simd_arrays_per_engine * 512;
    wave_num = detail::min(cu_num * 40, max_waves);
  } else {
    wave_num = cu_num * 32;
  }

  uint32_t lds_per_cu = (gfxv == abi::GFX_VERSION_GFX950)
                            ? (props.lds_size_in_kb << 10)
                            : LDS_SIZE_PER_CU;
  uint32_t wg_data_per_cu = vgpr_size_per_cu(gfxv) + SGPR_SIZE_PER_CU +
                            lds_per_cu + HWREG_SIZE_PER_CU;
  uint32_t wg_data_size = static_cast<uint32_t>(
      align_up(size_t{cu_num} * wg_data_per_cu, page_size()));

  uint32_t ctl_stack_bytes = (gfxv >= abi::GFX_VERSION_GFX10_1) ? 12u : 8u;
  uint32_t ctl_stack_size = wave_num * ctl_stack_bytes + 8;
  ctl_stack_size = align_up(CWSR_HEADER_SIZE + ctl_stack_size,
                            static_cast<uint32_t>(page_size()));

  if (abi::gfx_version_major(gfxv) == /*GFX10=*/10)
    ctl_stack_size = detail::min(ctl_stack_size, /*28 KiB=*/0x7000u);

  return {ctl_stack_size + wg_data_size, ctl_stack_size};
}

} // namespace

std::expected<Topology, Error> Topology::create() {
  char path[256];
  std::snprintf(path, sizeof(path), "%s/generation_id", SYSFS_TOPOLOGY);

  for (int attempt = 0; attempt < MAX_GENERATION_RETRIES; ++attempt) {
    auto gen_before = KFD_TRY(read_sysfs_uint(path));

    uint32_t num_nodes = count_nodes();

    Topology topo;
    topo.generation_id = static_cast<uint32_t>(gen_before);

    for (uint32_t i = 0; i < num_nodes; ++i) {
      auto node = KFD_TRY(read_node(i));
      if (node.props.gpu_id == 0)
        continue;

      // Linux 6.19+ added this as a property, calculate it otherwise.
      if (!node.props.cwsr_size || !node.props.ctl_stack_size) {
        auto [cwsr_size, ctl_stack_size] = compute_cwsr_sizes(node.props);
        node.props.cwsr_size = cwsr_size;
        node.props.ctl_stack_size = ctl_stack_size;
      }
      KFD_CHECK(topo.storage.push_back(std::move(node)));
    }

    auto gen_after = KFD_TRY(read_sysfs_uint(path));

    if (gen_before == gen_after)
      return topo;

    // Generation changed while reading, retry.
  }

  return kfd::unexpected(EAGAIN, "topology generation changed during scan");
}

} // namespace kfd
