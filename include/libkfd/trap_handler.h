//===-- libkfd/trap_handler.h - Embedded trap handler images ------*- C -*-===//
//
// Declares the table of per-architecture trap handler ELF binaries that are
// baked into the library at build time.
//
//===----------------------------------------------------------------------===//

#ifndef LIBKFD_TRAP_HANDLER_H
#define LIBKFD_TRAP_HANDLER_H

#ifdef __cplusplus
extern "C" {
#endif

struct trap_handler_image {
  const unsigned char *data;
  unsigned long size;
};

extern const struct trap_handler_image trap_handler_images[];
extern const unsigned trap_handler_image_count;

#ifdef __cplusplus
}
#endif

#endif // LIBKFD_TRAP_HANDLER_H
