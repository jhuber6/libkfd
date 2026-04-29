//===-- tools/computetoy/main.cpp - Compute-driven display ------*- C++ -*-===//
//
// Provides a mock-up of a Shadertoy-esque fragment shader using pure compute.
// We launch a compute kernel with a VRAM DMA buffer and write pixels to it.
// These are then presented to the window.
//
// $ computetoy <kernel.elf> [WIDTHxHEIGHT]
//
//===----------------------------------------------------------------------===//

#include "window.h"

#include "libkfd/libkfd.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

namespace {

constexpr uint32_t NUM_BUFFERS = 3;
constexpr uint32_t BLOCK_X = 16;
constexpr uint32_t BLOCK_Y = 16;

struct Framebuffer {
  kfd::Buffer buffer;
  kfd::DMABuffer dmabuf;
  kfd::Buffer kernarg;
  std::unique_ptr<kfd::Signal> signal;
};

// The textures that the DMA-buf gets converted to must be 256-byte strided.
constexpr uint32_t STRIDE_ALIGN = 256;

struct Uniforms {
  void *framebuffer;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;
  float time;
  uint32_t frame;
};

std::vector<std::byte> read_file(const char *path) {
  std::FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::fprintf(stderr, "error: cannot open '%s'\n", path);
    std::exit(1);
  }
  std::fseek(f, 0, SEEK_END);
  auto sz = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<std::byte> buf(static_cast<size_t>(sz));
  if (std::fread(buf.data(), 1, buf.size(), f) != buf.size()) {
    std::fprintf(stderr, "error: short read on '%s'\n", path);
    std::exit(1);
  }
  std::fclose(f);
  return buf;
}

void parse_resolution(const char *str, uint32_t &w, uint32_t &h) {
  if (std::sscanf(str, "%ux%u", &w, &h) != 2) {
    std::fprintf(
        stderr, "error: invalid resolution '%s', expected WIDTHxHEIGHT\n", str);
    std::exit(1);
  }
}

std::expected<kfd::Device *, kfd::Error>
find_device(kfd::Context &ctx, std::span<const std::byte> image) {
  for (size_t i = 0; i < ctx.num_devices(); ++i) {
    kfd::Device &dev = *KFD_EXPECT(ctx.device(i));
    if (dev.loadable(image))
      return &dev;
  }
  return kfd::unexpected(ENOEXEC, "No compatible GPUs found");
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <kernel.elf> [WIDTHxHEIGHT]\n", argv[0]);
    return 1;
  }

  uint32_t width = 1280;
  uint32_t height = 720;
  if (argc > 2)
    parse_resolution(argv[2], width, height);
  auto file = read_file(argv[1]);

  auto ctx = KFD_EXPECT(kfd::Context::create());
  auto &dev = *KFD_EXPECT(find_device(ctx, file));
  std::printf("GPU: %.*s (gfx%u)\n", static_cast<int>(dev.name().size()),
              dev.name().data(), dev.properties().gfx_target_version);

  auto compute = KFD_EXPECT(kfd::ComputeQueue::create(dev));

  auto exe = KFD_EXPECT(kfd::Executable::load(dev, file, compute));
  auto kernel = KFD_EXPECT(exe.kernel("fragment.kd"));

  // The DRM backend will override the resolution to match the native mode.
  auto win = KFD_EXPECT(Window::create(width, height, NUM_BUFFERS,
                                       "libkfd computetoy", dev.render_fd()));
  width = win->width();
  height = win->height();

  kfd::DispatchConfig cfg{
      .grid = {.x = (width + BLOCK_X - 1) / BLOCK_X,
               .y = (height + BLOCK_Y - 1) / BLOCK_Y},
      .block = {.x = BLOCK_X, .y = BLOCK_Y},
  };

  uint32_t stride =
      (width * sizeof(unsigned) + STRIDE_ALIGN - 1) & ~(STRIDE_ALIGN - 1);
  uint32_t pitch = stride / sizeof(unsigned);
  size_t fb_size = static_cast<size_t>(stride) * height;

  Framebuffer fbs[NUM_BUFFERS];
  for (uint32_t i = 0; i < NUM_BUFFERS; ++i) {
    fbs[i].buffer = KFD_EXPECT(kfd::Buffer::allocate(
        dev, fb_size, kfd::MemType::VRAM, kfd::MemFlags::WRITABLE));
    KFD_EXPECT(fbs[i].buffer.map(dev));
    fbs[i].dmabuf = KFD_EXPECT(kfd::DMABuffer::create(fbs[i].buffer));

    Uniforms initial{
        .framebuffer = fbs[i].buffer.data(),
        .width = width,
        .height = height,
        .pitch = pitch,
    };
    fbs[i].kernarg = KFD_EXPECT(kernel.alloc());
    kernel.fill(fbs[i].kernarg, initial, cfg);

    fbs[i].signal =
        std::make_unique<kfd::Signal>(KFD_EXPECT(kfd::Signal::create(ctx)));
  }

  for (uint32_t i = 0; i < NUM_BUFFERS; ++i)
    KFD_EXPECT(win->import_buffer(i, fbs[i].dmabuf.fd(), fbs[i].buffer.size(),
                                  stride));

  auto elf_mtime = std::filesystem::last_write_time(argv[1]);

  uint32_t current = 0;
  uint32_t frame = 0;
  auto start = std::chrono::high_resolution_clock::now();
  auto fps_time = start;
  uint32_t fps_frames = 0;

  std::printf("Entering render loop at %ux%u...\n", width, height);
  std::printf("Press 'q' to quit\n");

  while (win->poll()) {
    win->wait_idle(current);

    auto now = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(now - start).count();

    std::error_code ec;
    if (auto t = std::filesystem::last_write_time(argv[1], ec);
        !ec && t != elf_mtime) {
      elf_mtime = t;
      auto new_file = read_file(argv[1]);
      exe = KFD_EXPECT(kfd::Executable::load(dev, new_file, compute));
      kernel = KFD_EXPECT(exe.kernel("fragment.kd"));
      for (uint32_t i = 0; i < NUM_BUFFERS; ++i) {
        fbs[i].kernarg = KFD_EXPECT(kernel.alloc());
        Uniforms init{
            .framebuffer = fbs[i].buffer.data(),
            .width = width,
            .height = height,
            .pitch = pitch,
        };
        kernel.fill(fbs[i].kernarg, init, cfg);
      }
    }

    Uniforms args{
        .framebuffer = fbs[current].buffer.data(),
        .width = width,
        .height = height,
        .pitch = pitch,
        .time = time,
        .frame = frame,
    };
    std::memcpy(fbs[current].kernarg.data(), &args, sizeof(args));

    KFD_EXPECT(fbs[current].signal->reset());
    KFD_EXPECT(compute.dispatch(kernel, cfg, fbs[current].kernarg,
                                *fbs[current].signal));
    KFD_EXPECT(fbs[current].signal->wait(kfd::Condition::EQ, 0, UINT64_MAX));

    win->present(current);

    current = (current + 1) % NUM_BUFFERS;
    ++frame;
    ++fps_frames;

    auto elapsed = std::chrono::duration<double>(now - fps_time).count();
    if (elapsed >= 2.0) {
      std::printf("%.1f FPS (%u frames in %.1fs)\n",
                  static_cast<double>(fps_frames) / elapsed, fps_frames,
                  elapsed);
      fps_frames = 0;
      fps_time = now;
    }
  }

  std::printf("Exiting after %u frames.\n", frame);
  return 0;
}
