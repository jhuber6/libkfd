#include <gpuintrin.h>

// Executes s_trap (via __builtin_trap) to exercise KFD's wave-trap exception
// delivery back to the runtime through the CWSR header.
__gpu_kernel void wave_trap(void) { __builtin_trap(); }
