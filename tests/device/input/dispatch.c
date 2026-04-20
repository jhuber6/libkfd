#include <gpuintrin.h>

__gpu_kernel void nop(void) {}

// Indirect call forces dynamic stack usage.
static void (*volatile indirect_fn)(unsigned *, unsigned);

__gpu_kernel void use_scratch(unsigned *out) {
  void (*fn)(unsigned *, unsigned) = indirect_fn;
  if (fn)
    fn(out, __gpu_thread_id_x());
}

__gpu_kernel void store(unsigned *out) {
  if (__gpu_thread_id_x() == 0 && __gpu_block_id_x() == 0)
    *out = 0xCAFEBABE;
}

__gpu_kernel void fill_local_ids(unsigned *out) {
  out[__gpu_thread_id_x()] = __gpu_thread_id_x();
}

__gpu_kernel void fill_wg_ids(unsigned *out) {
  if (__gpu_thread_id_x() == 0)
    out[__gpu_block_id_x()] = __gpu_block_id_x();
}

__gpu_kernel void check_dims(unsigned *out) {
  if (__gpu_thread_id_x() != 0 || __gpu_thread_id_y() != 0 ||
      __gpu_thread_id_z() != 0)
    return;
  if (__gpu_block_id_x() != 0 || __gpu_block_id_y() != 0 ||
      __gpu_block_id_z() != 0)
    return;
  out[0] = __gpu_num_blocks_x();
  out[1] = __gpu_num_blocks_y();
  out[2] = __gpu_num_blocks_z();
  out[3] = __gpu_num_threads_x();
  out[4] = __gpu_num_threads_y();
  out[5] = __gpu_num_threads_z();
}
