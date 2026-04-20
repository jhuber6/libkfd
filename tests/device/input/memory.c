#include <gpuintrin.h>

__gpu_kernel void scratch_sum(unsigned *out) {
  volatile unsigned scratch[16];
  unsigned tid = __gpu_thread_id_x();
  for (unsigned i = 0; i < 16; ++i)
    scratch[i] = tid + i;
  unsigned sum = 0;
  for (unsigned i = 0; i < 16; ++i)
    sum += scratch[i];
  if (tid == 0 && __gpu_block_id_x() == 0)
    *out = sum;
}

[[clang::loader_uninitialized]]
static __gpu_local unsigned lds_buf[256];

__gpu_kernel void lds_reduce(unsigned *out) {
  unsigned tid = __gpu_thread_id_x();
  unsigned bid = __gpu_block_id_x();
  lds_buf[tid] = tid;
  __gpu_sync_threads();

  for (unsigned stride = __gpu_num_threads_x() / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      lds_buf[tid] += lds_buf[tid + stride];
    __gpu_sync_threads();
  }

  if (tid == 0)
    out[bid] = lds_buf[0];
}

extern __gpu_local unsigned DynamicSharedBuffer[];

__gpu_kernel void dynamic_lds_fill(unsigned *out) {
  unsigned tid = __gpu_thread_id_x();
  unsigned bid = __gpu_block_id_x();
  DynamicSharedBuffer[tid] = tid * 3;
  __gpu_sync_threads();

  if (tid == 0) {
    unsigned sum = 0;
    for (unsigned i = 0; i < __gpu_num_threads_x(); ++i)
      sum += DynamicSharedBuffer[i];
    out[bid] = sum;
  }
}

[[gnu::noinline]] static unsigned compute(unsigned x) {
  volatile unsigned buf[8];
  for (unsigned i = 0; i < 8; ++i)
    buf[i] = x + i;
  unsigned sum = 0;
  for (unsigned i = 0; i < 8; ++i)
    sum += buf[i];
  return sum;
}

__gpu_kernel void dynamic_stack(unsigned *out) {
  unsigned tid = __gpu_thread_id_x();
  unsigned val = compute(tid);
  if (tid == 0 && __gpu_block_id_x() == 0)
    *out = val;
}
