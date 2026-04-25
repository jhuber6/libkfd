#include <gpuintrin.h>

__gpu_kernel void verify_preload(unsigned a1, unsigned a2, unsigned a3,
                                 unsigned a4, unsigned a5, unsigned a6,
                                 unsigned a7, unsigned a8, unsigned a9,
                                 unsigned a10, unsigned a11, unsigned a12,
                                 unsigned a13, unsigned a14, unsigned a15,
                                 unsigned a16) {
  if (a1 != 1 || a2 != 2 || a3 != 3 || a4 != 4 || a5 != 5 || a6 != 6 ||
      a7 != 7 || a8 != 8 || a9 != 9 || a10 != 10 || a11 != 11 || a12 != 12 ||
      a13 != 13 || a14 != 14 || a15 != 15 || a16 != 16)
    __builtin_trap();
}
