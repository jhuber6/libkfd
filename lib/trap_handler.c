//===-- lib/trap_handler.c - Binary trap embedding ----------------*- C -*-===//
//
// Provides the binary blobs for the trap code. Each supported architecture
// gets its trap handler ELF baked in the object.
//
//===----------------------------------------------------------------------===//

#include "libkfd/trap_handler.h"

static const unsigned char trap_gfx908[] = {
#embed "trap_handler_gfx908.bin"
};
static const unsigned char trap_gfx90a[] = {
#embed "trap_handler_gfx90a.bin"
};
static const unsigned char trap_gfx9_generic[] = {
#embed "trap_handler_gfx9-generic.bin"
};
static const unsigned char trap_gfx9_4_generic[] = {
#embed "trap_handler_gfx9-4-generic.bin"
};
static const unsigned char trap_gfx10_1_generic[] = {
#embed "trap_handler_gfx10-1-generic.bin"
};
static const unsigned char trap_gfx10_3_generic[] = {
#embed "trap_handler_gfx10-3-generic.bin"
};
static const unsigned char trap_gfx11_generic[] = {
#embed "trap_handler_gfx11-generic.bin"
};
static const unsigned char trap_gfx1170[] = {
#embed "trap_handler_gfx1170.bin"
};
static const unsigned char trap_gfx1171[] = {
#embed "trap_handler_gfx1171.bin"
};
static const unsigned char trap_gfx1172[] = {
#embed "trap_handler_gfx1172.bin"
};
static const unsigned char trap_gfx12_generic[] = {
#embed "trap_handler_gfx12-generic.bin"
};
static const unsigned char trap_gfx12_5_generic[] = {
#embed "trap_handler_gfx12-5-generic.bin"
};

#define TRAP_HANDLER_LIST(X)                                                   \
  X(gfx908)                                                                    \
  X(gfx90a)                                                                    \
  X(gfx9_generic)                                                              \
  X(gfx9_4_generic)                                                            \
  X(gfx10_1_generic)                                                           \
  X(gfx10_3_generic)                                                           \
  X(gfx11_generic)                                                             \
  X(gfx1170)                                                                   \
  X(gfx1171)                                                                   \
  X(gfx1172)                                                                   \
  X(gfx12_generic)                                                             \
  X(gfx12_5_generic)

#define X_ENTRY(id) {trap_##id, sizeof(trap_##id)},
const struct TrapHandlerImage trap_handler_images[] = {
    TRAP_HANDLER_LIST(X_ENTRY)};
#undef X_ENTRY

const unsigned trap_handler_image_count =
    sizeof(trap_handler_images) / sizeof(trap_handler_images[0]);
