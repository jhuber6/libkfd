#include <gpuintrin.h>

#define EXTERN_VIS [[gnu::visibility("protected")]]

void foo() {}

__gpu_kernel void kernel() { foo(); }

EXTERN_VIS unsigned x = 0xdeadbeef;
EXTERN_VIS unsigned y = 0xfeedface;
EXTERN_VIS unsigned z = 0xcafebabe;

EXTERN_VIS unsigned char bss_arr[4096] = {};

[[gnu::section(".custom")]] unsigned data[4] = {1, 2, 3, 4};

EXTERN_VIS unsigned reloc_target_a = 0x11111111;
EXTERN_VIS unsigned reloc_target_b = 0x22222222;
EXTERN_VIS unsigned *reloc_ptr_a = &reloc_target_a;
EXTERN_VIS unsigned *reloc_ptr_b = &reloc_target_b;
EXTERN_VIS void (*reloc_fptr)(void) = foo;

__gpu_kernel void use(void **ptr) { *ptr = bss_arr; }
