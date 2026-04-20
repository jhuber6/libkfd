#define EXTERN_VIS [[gnu::visibility("protected")]]
#define KERNEL EXTERN_VIS [[clang::device_kernel]]

void foo() {}

KERNEL void kernel() { foo(); }

EXTERN_VIS unsigned x = 0xdeadbeef;
EXTERN_VIS unsigned y = 0xfeedface;
EXTERN_VIS unsigned z = 0xcafebabe;

EXTERN_VIS unsigned char bss_arr[4096] = {};

[[gnu::section(".custom")]] unsigned data[4] = {1, 2, 3, 4};

KERNEL void use(void **ptr) { *ptr = bss_arr; }
