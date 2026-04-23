//===-- lib/trap_handler.c - Binary trap embedding ----------------*- C -*-===//
//
// Provides the binary blobs for the trap code. Each supported architecture
// gets its trap handler ELF baked in the object.
//
//===----------------------------------------------------------------------===//

#include "libkfd/trap_handler.h"

#ifdef HAS_GFX908
static const unsigned char trap_gfx908[] = {
#embed "trap_handler_gfx908.bin"
};
#endif
#ifdef HAS_GFX90A
static const unsigned char trap_gfx90a[] = {
#embed "trap_handler_gfx90a.bin"
};
#endif
#ifdef HAS_GFX9_GENERIC
static const unsigned char trap_gfx9_generic[] = {
#embed "trap_handler_gfx9-generic.bin"
};
#endif
#ifdef HAS_GFX9_4_GENERIC
static const unsigned char trap_gfx9_4_generic[] = {
#embed "trap_handler_gfx9-4-generic.bin"
};
#endif
#ifdef HAS_GFX10_1_GENERIC
static const unsigned char trap_gfx10_1_generic[] = {
#embed "trap_handler_gfx10-1-generic.bin"
};
#endif
#ifdef HAS_GFX10_3_GENERIC
static const unsigned char trap_gfx10_3_generic[] = {
#embed "trap_handler_gfx10-3-generic.bin"
};
#endif
#ifdef HAS_GFX11_GENERIC
static const unsigned char trap_gfx11_generic[] = {
#embed "trap_handler_gfx11-generic.bin"
};
#endif
#ifdef HAS_GFX1170
static const unsigned char trap_gfx1170[] = {
#embed "trap_handler_gfx1170.bin"
};
#endif
#ifdef HAS_GFX1171
static const unsigned char trap_gfx1171[] = {
#embed "trap_handler_gfx1171.bin"
};
#endif
#ifdef HAS_GFX1172
static const unsigned char trap_gfx1172[] = {
#embed "trap_handler_gfx1172.bin"
};
#endif
#ifdef HAS_GFX12_GENERIC
static const unsigned char trap_gfx12_generic[] = {
#embed "trap_handler_gfx12-generic.bin"
};
#endif
#ifdef HAS_GFX12_5_GENERIC
static const unsigned char trap_gfx12_5_generic[] = {
#embed "trap_handler_gfx12-5-generic.bin"
};
#endif

#define X_ENTRY(id) {trap_##id, sizeof(trap_##id)},

// clang-format off
const struct TrapHandlerImage trap_handler_images[] = {
#ifdef HAS_GFX908
    X_ENTRY(gfx908)
#endif
#ifdef HAS_GFX90A
    X_ENTRY(gfx90a)
#endif
#ifdef HAS_GFX9_GENERIC
    X_ENTRY(gfx9_generic)
#endif
#ifdef HAS_GFX9_4_GENERIC
    X_ENTRY(gfx9_4_generic)
#endif
#ifdef HAS_GFX10_1_GENERIC
    X_ENTRY(gfx10_1_generic)
#endif
#ifdef HAS_GFX10_3_GENERIC
    X_ENTRY(gfx10_3_generic)
#endif
#ifdef HAS_GFX11_GENERIC
    X_ENTRY(gfx11_generic)
#endif
#ifdef HAS_GFX1170
    X_ENTRY(gfx1170)
#endif
#ifdef HAS_GFX1171
    X_ENTRY(gfx1171)
#endif
#ifdef HAS_GFX1172
    X_ENTRY(gfx1172)
#endif
#ifdef HAS_GFX12_GENERIC
    X_ENTRY(gfx12_generic)
#endif
#ifdef HAS_GFX12_5_GENERIC
    X_ENTRY(gfx12_5_generic)
#endif
};
// clang-format on

#undef X_ENTRY

const unsigned trap_handler_image_count =
    sizeof(trap_handler_images) / sizeof(trap_handler_images[0]);
