#include <gpuintrin.h>

__gpu_kernel void saxpy(float *y, const float *x, float a, unsigned n) {
  unsigned i = __gpu_thread_id_x() + __gpu_block_id_x() * __gpu_num_threads_x();
  if (i < n)
    y[i] = a * x[i] + y[i];
}
