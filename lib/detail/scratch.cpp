//===-- lib/detail/scratch.cpp - Scratch memory helpers ---------*- C++ -*-===//
//
// Internal helpers for computing scratch configurations from kernel descriptors
// and device topology.
//
//===----------------------------------------------------------------------===//

#include "libkfd/detail/scratch.h"
#include "libkfd/abi.h"
#include "libkfd/device.h"

namespace kfd::detail {

uint32_t scratch_num_se(const Device &dev) {
  const auto &p = dev.properties();
  if (p.simd_arrays_per_engine == 0)
    return 1;
  uint32_t n = p.array_count / p.simd_arrays_per_engine;
  return n ? n : 1;
}

uint32_t compute_tmpring_size(const Device &dev, uint32_t per_thread,
                              size_t region_size) {
  if (per_thread == 0 || region_size == 0)
    return 0;

  const auto &props = dev.properties();
  size_t per_wave = static_cast<size_t>(SCRATCH_LANES_PER_WAVE) * per_thread;

  uint32_t align = scratch_alignment_unit(dev.gfx_version());
  per_wave = align_up(per_wave, static_cast<size_t>(align));

  if (per_wave > max_wave_scratch(dev.gfx_version()))
    return 0;

  uint32_t simd_per_cu = props.simd_per_cu ? props.simd_per_cu : 1;
  uint32_t num_cus = props.simd_count / simd_per_cu;
  uint32_t num_se = scratch_num_se(dev);
  uint32_t num_xcc = props.num_xcc ? props.num_xcc : 1;
  uint32_t se_per_xcc = num_se / num_xcc;
  if (se_per_xcc == 0)
    se_per_xcc = 1;

  uint32_t wavesize = static_cast<uint32_t>(per_wave / align);
  uint32_t waves;

  if (dev.gfx_version() >= abi::GFX_VERSION_GFX11) {
    // GFX11+ - WAVES field is per-SE (not total).
    // References: ROCr FillComputeTmpRingSize_Gfx11 in amd_aql_queue.cpp
    uint32_t cus_per_xcc = num_cus / num_xcc;
    uint32_t device_max = cus_per_xcc * props.max_slots_scratch_cu;
    uint32_t max_waves = static_cast<uint32_t>(region_size / per_wave);
    max_waves /= num_se;
    if (max_waves > device_max)
      max_waves = device_max;
    if (max_waves == 0)
      max_waves = 1;
    waves = max_waves;
  } else {
    // GFX9/10 - WAVES is the total wave count fitting one XCC's portion of
    // the buffer; must be a multiple of SEs-per-XCC.
    // References: ROCr FillComputeTmpRingSize in amd_aql_queue.cpp
    uint32_t device_max = num_cus * props.max_slots_scratch_cu;
    size_t per_xcc_size = region_size / num_xcc;
    uint32_t max_waves = static_cast<uint32_t>(per_xcc_size / per_wave);
    if (max_waves > device_max)
      max_waves = device_max;
    max_waves = (max_waves / se_per_xcc) * se_per_xcc;
    if (max_waves == 0)
      max_waves = se_per_xcc;
    waves = max_waves;
  }

  return (wavesize << 12) | waves;
}

uint32_t scratch_device_slots(const Device &dev) {
  const auto &p = dev.properties();
  uint32_t simd_per_cu = p.simd_per_cu ? p.simd_per_cu : 1;
  uint32_t num_cus = p.simd_count / simd_per_cu;
  uint32_t num_se = scratch_num_se(dev);

  // The Shader Processor Input (SPI) assigns slots per-SE, so we round up to
  // the number of SEs to produce the max slots needed for the device.
  uint32_t slots = align_up(num_cus, num_se) * p.max_slots_scratch_cu;
  return slots ? slots : num_se;
}

size_t scratch_alloc_size(uint32_t gfx_version, uint32_t per_thread,
                          uint32_t slots) {
  if (per_thread == 0 || slots == 0)
    return 0;

  size_t per_wave =
      align_up(static_cast<size_t>(SCRATCH_LANES_PER_WAVE) * per_thread,
               static_cast<size_t>(scratch_alignment_unit(gfx_version)));

  size_t needed = per_wave * slots;
  return align_up(needed, abi::PRIVATE_SEGMENT_ALIGN);
}

} // namespace kfd::detail
