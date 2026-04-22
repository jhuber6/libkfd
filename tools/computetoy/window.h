//===-- tools/computetoy/window.h - DRI3/Present X11 window -----*- C++ -*-===//
//
// Encapsulates X11 window creation, DRI3 pixmap import from DMA buffer file
// descriptors, and Present-based buffer management in an RAII format.
//
//===----------------------------------------------------------------------===//

#ifndef COMPUTETOY_WINDOW_H
#define COMPUTETOY_WINDOW_H

#include "libkfd/error.h"

#include <cstdint>
#include <expected>
#include <memory>

#include <xcb/xcb.h>

class DRI3Window {
public:
  ~DRI3Window();

  DRI3Window(const DRI3Window &) = delete;
  DRI3Window &operator=(const DRI3Window &) = delete;
  DRI3Window(DRI3Window &&other);
  DRI3Window &operator=(DRI3Window &&) = delete;

  // Connect to X, create a window, and initialize DRI3 + Present extensions.
  static std::expected<DRI3Window, kfd::Error> create(uint32_t width,
                                                      uint32_t height,
                                                      uint32_t num_buffers,
                                                      const char *title);

  // Import a DMA buffer file descriptor as a presentable pixmap. The fd is
  // dup'd internally and the caller retains ownership of the original.
  std::expected<void, kfd::Error> import_buffer(uint32_t index, int dmabuf_fd,
                                                size_t size, uint32_t stride);

  // Poll X11 events. Returns false when the window should close (Escape key
  // or WM_DELETE_WINDOW).
  bool poll();

  // Block until the X server is done reading the given buffer.
  void wait_idle(uint32_t index);

  // Present the pixmap for the given buffer and mark it busy.
  void present(uint32_t index);

private:
  DRI3Window(xcb_connection_t *conn, xcb_window_t win, xcb_colormap_t colormap,
             xcb_atom_t wm_delete, xcb_special_event_t *present_special,
             uint8_t depth, uint32_t w, uint32_t h, uint32_t num_buffers);

  void drain_present_events(bool block);

  xcb_connection_t *conn = nullptr;
  xcb_window_t win = 0;
  xcb_colormap_t colormap = 0;
  xcb_atom_t wm_delete = 0;
  xcb_special_event_t *present_special = nullptr;
  uint8_t depth = 0;
  uint32_t w = 0;
  uint32_t h = 0;
  uint32_t num_buffers = 0;

  std::unique_ptr<xcb_pixmap_t[]> pixmaps;
  std::unique_ptr<bool[]> busy;
};

#endif // COMPUTETOY_WINDOW_H
