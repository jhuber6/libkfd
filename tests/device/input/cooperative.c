#include <gpuintrin.h>

__gpu_kernel void coop_store(unsigned *out) {
  if (__gpu_thread_id_x() == 0)
    out[__gpu_block_id_x()] = __gpu_block_id_x();
}

// Synchronizes across all active wavefronts. Should be flaky if the launch is
// not truly cooperative.
__gpu_kernel void coop_probe(unsigned *arrived, unsigned *results,
                             unsigned total) {
  if (__gpu_lane_id() != 0)
    return;

  unsigned global_id =
      __gpu_block_id_x() * __gpu_num_threads_x() / __gpu_num_lanes() +
      __gpu_thread_id_x() / __gpu_num_lanes();

  __atomic_fetch_add(arrived, 1, __ATOMIC_RELAXED);
  unsigned success = 0;
  for (unsigned i = 0; i < 0x100000; ++i) {
    if (__atomic_load_n(arrived, __ATOMIC_RELAXED) == total) {
      success = 1;
      break;
    }
    __builtin_amdgcn_s_sleep(1);
  }
  results[global_id] = success;
}
