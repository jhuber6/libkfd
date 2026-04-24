//===-- tools/sandbox/main.cpp - SAXPY example ------------------*- C++ -*-===//
//
// Demonstrates the libkfd API by running a SAXPY kernel (y = a*x + y) on the
// GPU. This example does not require RPC support; it only exercises the core
// context, memory, loader, dispatch, and signal interfaces.
//
// $ sandbox <saxpy.elf> [N=1048576]
//
//===----------------------------------------------------------------------===//

#include "libkfd/libkfd.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static std::vector<std::byte> read_file(const char *path) {
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

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <saxpy.elf> [N=1048576]\n", argv[0]);
    return 1;
  }

  size_t n = argc > 2 ? static_cast<size_t>(std::atol(argv[2])) : (1u << 20);
  constexpr float A = 2.0f;

  // Open /dev/kfd and enumerate GPUs.
  auto ctx = KFD_EXPECT(kfd::Context::create());
  if (ctx.num_devices() == 0) {
    std::fprintf(stderr, "error: no GPUs found\n");
    return 1;
  }
  auto &dev = ctx.devices().front();
  std::printf("GPU: device=0x%04x  gfx=%u\n", dev.properties().device_id,
              dev.properties().gfx_target_version);

  // Create a compute queue for dispatch and an SDMA queue for loading.
  auto compute = KFD_EXPECT(kfd::ComputeQueue::create(dev));
  auto sdma = KFD_EXPECT(kfd::SDMAQueue::create(dev));

  // Load the GPU code object and look up the kernel descriptor.
  auto file = read_file(argv[1]);
  auto exe = KFD_EXPECT(kfd::Executable::load(dev, file, sdma, compute));
  auto kernel = KFD_EXPECT(exe.kernel("saxpy.kd"));

  // Allocate host-visible GTT buffers for the x and y arrays.
  size_t buf_bytes = n * sizeof(float);
  auto x_buf = KFD_EXPECT(kfd::Buffer::allocate(
      dev, buf_bytes, kfd::MemType::GTT, kfd::MemFlags::WRITABLE));
  auto y_buf = KFD_EXPECT(kfd::Buffer::allocate(
      dev, buf_bytes, kfd::MemType::GTT, kfd::MemFlags::WRITABLE));
  KFD_EXPECT(x_buf.map(dev));
  KFD_EXPECT(y_buf.map(dev));

  auto *x = static_cast<float *>(x_buf.data());
  auto *y = static_cast<float *>(y_buf.data());
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i);
    y[i] = static_cast<float>(i) * 0.5f;
  }

  // Allocate device VRAM to DMA copy the memory from the GTT buffer.
  auto x_dev = KFD_EXPECT(kfd::Buffer::allocate(
      dev, buf_bytes, kfd::MemType::VRAM, kfd::MemFlags::WRITABLE));
  auto y_dev = KFD_EXPECT(kfd::Buffer::allocate(
      dev, buf_bytes, kfd::MemType::VRAM, kfd::MemFlags::WRITABLE));
  KFD_EXPECT(x_dev.map(dev));
  KFD_EXPECT(y_dev.map(dev));

  // Create a signal to use for SDMA copies, source must be pinned for DMA.
  kfd::Signal mem = KFD_EXPECT(kfd::Signal::create(ctx, /*initial=*/1));
  KFD_EXPECT(sdma.copy_linear(x_dev.data(), x_buf.data(), buf_bytes));
  KFD_EXPECT(sdma.copy_linear(y_dev.data(), y_buf.data(), buf_bytes));
  KFD_EXPECT(sdma.signal(mem));
  KFD_EXPECT(mem.wait(kfd::Condition::EQ, 0, UINT64_MAX));

  // Kernel arguments must match the GPU function signature in layout:
  //   void saxpy(float *y, const float *x, float a, unsigned n)
  struct SaxpyArgs {
    float *y;
    const float *x;
    float a;
    unsigned n;
  };
  SaxpyArgs args{static_cast<float *>(y_dev.data()),
                 static_cast<const float *>(x_dev.data()), A,
                 static_cast<unsigned>(n)};

  constexpr uint32_t BLOCK = 256;
  uint32_t grid = (static_cast<uint32_t>(n) + BLOCK - 1) / BLOCK;
  kfd::DispatchConfig cfg{.grid = {.x = grid}, .block = {.x = BLOCK}};
  auto kernarg = KFD_EXPECT(kernel.alloc());
  kernel.fill(kernarg, args, cfg);

  // Dispatch and wait for completion via a signal.
  auto sig = KFD_EXPECT(kfd::Signal::create(ctx));

  auto t0 = std::chrono::high_resolution_clock::now();
  KFD_EXPECT(compute.dispatch(kernel, cfg, kernarg, sig));
  KFD_EXPECT(sig.wait(kfd::Condition::EQ, 0, UINT64_MAX));
  auto t1 = std::chrono::high_resolution_clock::now();

  double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
  std::printf("SAXPY: N=%zu  a=%.1f  dispatch+wait=%.1f us\n", n,
              static_cast<double>(A), us);

  KFD_EXPECT(mem.reset(/*value=*/1));
  KFD_EXPECT(sdma.copy_linear(x_buf.data(), x_dev.data(), buf_bytes));
  KFD_EXPECT(sdma.copy_linear(y_buf.data(), y_dev.data(), buf_bytes));
  KFD_EXPECT(sdma.signal(mem));
  KFD_EXPECT(mem.wait(kfd::Condition::EQ, 0, UINT64_MAX));

  // y[i] should now equal A * i + i * 0.5.
  unsigned errors = 0;
  for (size_t i = 0; i < n; ++i) {
    float expected = A * static_cast<float>(i) + static_cast<float>(i) * 0.5f;
    if (std::fabs(y[i] - expected) > 1e-2f) {
      if (errors < 5)
        std::fprintf(stderr, "  MISMATCH [%zu]: got %.6f, expected %.6f\n", i,
                     static_cast<double>(y[i]), static_cast<double>(expected));
      ++errors;
    }
  }

  if (errors == 0)
    std::printf("PASS: all %zu elements verified\n", n);
  else
    std::printf("FAIL: %u / %zu mismatches\n", errors, n);

  return errors ? 1 : 0;
}
