#include <gpuintrin.h>

__gpu_kernel void spin_on_flag(unsigned *flag) {
  while (__scoped_atomic_load_n(flag, __ATOMIC_ACQUIRE,
                                __MEMORY_SCOPE_SYSTEM) == 0)
    ;
}

__gpu_kernel void spin_with_scratch(unsigned *flag) {
  volatile unsigned buf[32];
  for (unsigned i = 0; i < 32; ++i)
    buf[i] = i * i;
  while (__scoped_atomic_load_n(flag, __ATOMIC_ACQUIRE,
                                __MEMORY_SCOPE_SYSTEM) == 0) {
    buf[0] += 1;
  }
}
