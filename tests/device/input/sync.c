#include <gpuintrin.h>

__gpu_kernel void increment(unsigned *out, const unsigned *in) {
  unsigned tid = __gpu_thread_id_x();
  out[tid] = in[tid] + 1;
}

__gpu_kernel void add_buffers(unsigned *out, const unsigned *a,
                              const unsigned *b) {
  unsigned tid = __gpu_thread_id_x();
  out[tid] = a[tid] + b[tid];
}
