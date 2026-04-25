//===-- tools/loader/main.cpp - Mirror for llvm-gpu-loader ------*- C++ -*-===//
//
// This is an implementation of the 'llvm-gpu-loader' utility using the libkfd
// interface. It allows for GPUs to operate like a standard hosted environment
// that launches a 'main' function. This can only be built if the user compiler
// contains the RPC headers from the LLVM libc project.
//
// $ clang main.c --target=amdgcn-amd-amdhsa -flto -startfiles -stdlib -o image
// $ gpu-loader image
//
//===----------------------------------------------------------------------===//

#include "libkfd/libkfd.h"

#include <shared/rpc.h>
#include <shared/rpc_opcodes.h>
#include <shared/rpc_server.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

struct Options {
  unsigned threads_x = 1, threads_y = 1, threads_z = 1;
  unsigned blocks_x = 1, blocks_y = 1, blocks_z = 1;
  int file_idx = 0;
};

[[noreturn]] void usage(const char *prog) {
  std::fprintf(stderr,
               "Usage: %s [options] <gpu-elf> [args...]\n\n"
               "Options:\n"
               "  --threads-x N   Threads in the x dimension (default 1)\n"
               "  --threads-y N   Threads in the y dimension (default 1)\n"
               "  --threads-z N   Threads in the z dimension (default 1)\n"
               "  --threads N     Alias for --threads-x\n"
               "  --blocks-x N    Blocks in the x dimension (default 1)\n"
               "  --blocks-y N    Blocks in the y dimension (default 1)\n"
               "  --blocks-z N    Blocks in the z dimension (default 1)\n"
               "  --blocks N      Alias for --blocks-x\n"
               "  -h, --help      Show this message\n",
               prog);
  std::exit(0);
}

Options parse_args(int argc, const char **argv) {
  Options opts{};
  auto parse_uint = [&](const char *flag, int i) -> unsigned {
    if (i + 1 >= argc) {
      std::fprintf(stderr, "error: %s requires an argument\n", flag);
      std::exit(1);
    }
    char *end;
    unsigned long val = std::strtoul(argv[i + 1], &end, 10);
    if (*end != '\0' || val == 0) {
      std::fprintf(stderr, "error: %s: expected positive integer\n", flag);
      std::exit(1);
    }
    return static_cast<unsigned>(val);
  };

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-h") == 0 ||
        std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
    } else if (std::strcmp(argv[i], "--threads-x") == 0 ||
               std::strcmp(argv[i], "--threads") == 0) {
      opts.threads_x = parse_uint(argv[i], i);
      ++i;
    } else if (std::strcmp(argv[i], "--threads-y") == 0) {
      opts.threads_y = parse_uint(argv[i], i);
      ++i;
    } else if (std::strcmp(argv[i], "--threads-z") == 0) {
      opts.threads_z = parse_uint(argv[i], i);
      ++i;
    } else if (std::strcmp(argv[i], "--blocks-x") == 0 ||
               std::strcmp(argv[i], "--blocks") == 0) {
      opts.blocks_x = parse_uint(argv[i], i);
      ++i;
    } else if (std::strcmp(argv[i], "--blocks-y") == 0) {
      opts.blocks_y = parse_uint(argv[i], i);
      ++i;
    } else if (std::strcmp(argv[i], "--blocks-z") == 0) {
      opts.blocks_z = parse_uint(argv[i], i);
      ++i;
    } else {
      opts.file_idx = i;
      return opts;
    }
  }

  std::fprintf(stderr, "error: no input file\n");
  std::exit(1);
}

struct BeginArgs {
  int argc;
  void *argv;
  void *envp;
};

struct StartArgs {
  int argc;
  void *argv;
  void *envp;
  void *ret;
};

struct EndArgs {};

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

kfd::Buffer copy_arg_vector(kfd::Device &dev, int argc, const char **argv) {
  size_t arg_list = sizeof(std::byte *) * (static_cast<size_t>(argc) + 1);
  size_t len = 0;
  for (int i = 0; i < argc; ++i)
    len += strlen(argv[i]) + 1;

  kfd::Buffer dev_argv = KFD_EXPECT(kfd::Buffer::allocate(
      dev, arg_list + len, kfd::MemType::GTT, kfd::MemFlags::WRITABLE));
  KFD_EXPECT(dev_argv.map(dev));

  void *str = reinterpret_cast<std::byte *>(dev_argv.data()) + arg_list;
  for (int i = 0; i < argc; ++i) {
    size_t size = strlen(argv[i]) + 1;
    std::memcpy(str, argv[i], size);
    static_cast<void **>(dev_argv.data())[i] = str;
    str = reinterpret_cast<std::byte *>(str) + size;
  }
  reinterpret_cast<void **>(dev_argv.data())[argc] = nullptr;
  return dev_argv;
}

kfd::Buffer copy_env_vector(kfd::Device &dev, const char **envp) {
  int envc = 0;
  for (const char **env = envp; *env != 0; ++env)
    ++envc;
  return copy_arg_vector(dev, envc, envp);
}

constexpr size_t RPC_ALLOC_ALIGN = static_cast<const size_t>(2 * 1024 * 1024);

void rpc_server(rpc::Server &server, kfd::Signal &doorbell, kfd::Device &dev,
                uint32_t lane_size, std::atomic<bool> &running) {
  std::unordered_map<void *, kfd::Buffer> allocs;

  while (running.load(std::memory_order_relaxed)) {
    auto port = server.try_open(lane_size);
    if (!port.has_value()) {
      KFD_EXPECT(doorbell.wait(kfd::Condition::NE, 0, UINT64_MAX));
      continue;
    }

    switch (port->get_opcode()) {
    case LIBC_MALLOC: {
      port->recv_and_send([&](rpc::Buffer *buffer, uint32_t) {
        size_t size = kfd::detail::align_up(
            static_cast<size_t>(buffer->data[0]), RPC_ALLOC_ALIGN);
        auto region =
            kfd::detail::MappedRegion::reserve_aligned(size, RPC_ALLOC_ALIGN);
        if (!region) {
          buffer->data[0] = 0;
          return;
        }
        void *addr = region->release();
        auto buf = kfd::Buffer::allocate(dev, size, kfd::MemType::VRAM,
                                         kfd::MemFlags::WRITABLE, addr);
        if (!buf || !buf->map(dev)) {
          ::munmap(addr, size);
          buffer->data[0] = 0;
          return;
        }
        void *ptr = buf->data();
        buffer->data[0] = reinterpret_cast<uintptr_t>(ptr);
        allocs.emplace(ptr, std::move(*buf));
      });
      break;
    }
    case LIBC_FREE: {
      port->recv([&](rpc::Buffer *buffer, uint32_t) {
        allocs.erase(reinterpret_cast<void *>(buffer->data[0]));
      });
      break;
    }
    default:
      if (rpc::handle_libc_opcodes(*port, lane_size) != rpc::RPC_SUCCESS)
        std::fprintf(stderr, "error: unhandled RPC opcode %u\n",
                     port->get_opcode());
      break;
    }
  }
}

void launch_if_present(kfd::Device &dev, kfd::Executable &exe,
                       kfd::ComputeQueue &compute, std::string_view name) {
  std::expected<kfd::Kernel, kfd::Error> kernel = exe.kernel(name);
  if (!kernel)
    return;

  kfd::DispatchConfig cfg{.grid = {.x = 1}, .block = {.x = 1}};
  kfd::Signal sig =
      KFD_EXPECT(kfd::Signal::create(dev.context(), /*initial=*/1));
  kfd::Buffer ka = KFD_EXPECT(kernel->alloc());
  kernel->fill(ka, cfg);
  KFD_EXPECT(compute.dispatch(*kernel, cfg, ka, sig));
  KFD_EXPECT(sig.wait(kfd::Condition::EQ, 0, UINT64_MAX));
}

} // namespace

int main(int argc, const char **argv, const char **envp) {
  Options opts = parse_args(argc, argv);
  int gpu_argc = argc - opts.file_idx;
  const char **gpu_argv = argv + opts.file_idx;

  auto file = read_file(gpu_argv[0]);

  kfd::Context ctx = KFD_EXPECT(kfd::Context::create());
  if (ctx.num_devices() == 0) {
    std::fprintf(stderr, "error: no GPUs found\n");
    return 1;
  }
  kfd::Device &dev = ctx.devices().front();

  kfd::ComputeQueue compute = KFD_EXPECT(kfd::ComputeQueue::create(dev));
  kfd::SDMAQueue sdma = KFD_EXPECT(kfd::SDMAQueue::create(dev));

  kfd::Executable exe =
      KFD_EXPECT(kfd::Executable::load(dev, file, sdma, compute));

  kfd::Kernel begin = KFD_EXPECT(exe.kernel("_begin.kd"));
  kfd::Kernel start = KFD_EXPECT(exe.kernel("_start.kd"));
  kfd::Kernel end = KFD_EXPECT(exe.kernel("_end.kd"));

  const auto &props = dev.properties();
  bool wave32 = start.descriptor().kernel_code_properties &
                kfd::abi::ENABLE_WAVEFRONT_SIZE32;
  uint32_t lane_size = wave32 ? 32 : 64;
  uint32_t simd_per_cu = props.simd_per_cu ? props.simd_per_cu : 1;
  uint32_t port_count = static_cast<uint32_t>(
      std::min(static_cast<uint64_t>(props.simd_count / simd_per_cu *
                                     props.max_waves_per_simd),
               rpc::MAX_PORT_COUNT));

  size_t rpc_size = rpc::Server::allocation_size(lane_size, port_count);
  kfd::Buffer rpc_buf = KFD_EXPECT(
      kfd::Buffer::allocate(dev, rpc_size, kfd::MemType::GTT,
                            kfd::MemFlags::WRITABLE | kfd::MemFlags::COHERENT));
  KFD_EXPECT(rpc_buf.map(dev));
  std::memset(rpc_buf.data(), 0, rpc_size);

  kfd::Signal doorbell = KFD_EXPECT(kfd::Signal::create(ctx, /*initial=*/0));
  rpc::Doorbell db{
      .value = doorbell.fence_addr(),
      .mailbox = static_cast<uint64_t *>(doorbell.signal_addr()),
      .event_id = doorbell.event_id(),
  };
  std::memcpy(static_cast<std::byte *>(rpc_buf.data()) +
                  rpc::Server::doorbell_offset(),
              &db, sizeof(rpc::Doorbell));

  kfd::Buffer client_staging;
  auto client_sym = exe.symbol("__llvm_rpc_client");
  if (client_sym) {
    client_staging = KFD_EXPECT(kfd::Buffer::allocate(
        dev, sizeof(rpc::Client), kfd::MemType::GTT, kfd::MemFlags::WRITABLE));
    KFD_EXPECT(client_staging.map(dev));
    rpc::Client client(port_count, rpc_buf.data());
    std::memcpy(client_staging.data(), &client, sizeof(rpc::Client));
    KFD_EXPECT(compute.dma_copy(client_sym->data(), client_staging.data(),
                                static_cast<uint32_t>(sizeof(rpc::Client))));
  }

  rpc::Server server(port_count, rpc_buf.data());
  std::atomic<bool> rpc_running{true};
  std::thread rpc_thread(rpc_server, std::ref(server), std::ref(doorbell),
                         std::ref(dev), lane_size, std::ref(rpc_running));

  kfd::Buffer dev_argv = copy_arg_vector(dev, gpu_argc, gpu_argv);
  kfd::Buffer dev_envp = copy_env_vector(dev, envp);

  kfd::Buffer dev_ret = KFD_EXPECT(kfd::Buffer::allocate(
      dev, sizeof(int), kfd::MemType::GTT, kfd::MemFlags::WRITABLE));
  KFD_EXPECT(dev_ret.map(dev));

  kfd::Signal sig = KFD_EXPECT(kfd::Signal::create(ctx, 3));

  BeginArgs begin_args{gpu_argc, dev_argv.data(), dev_envp.data()};
  kfd::DispatchConfig begin_cfg{.grid = {.x = 1}, .block = {.x = 1}};
  kfd::Buffer begin_ka = KFD_EXPECT(begin.alloc());
  begin.fill(begin_ka, begin_args, begin_cfg);

  StartArgs start_args{gpu_argc, dev_argv.data(), dev_envp.data(),
                       dev_ret.data()};
  kfd::DispatchConfig start_cfg{
      .grid = {.x = opts.blocks_x, .y = opts.blocks_y, .z = opts.blocks_z},
      .block = {.x = opts.threads_x, .y = opts.threads_y, .z = opts.threads_z},
  };
  kfd::Buffer start_ka = KFD_EXPECT(start.alloc());
  start.fill(start_ka, start_args, start_cfg);

  kfd::DispatchConfig end_cfg{.grid = {.x = 1}, .block = {.x = 1}};
  kfd::Buffer end_ka = KFD_EXPECT(end.alloc());
  end.fill(end_ka, end_cfg);

  launch_if_present(dev, exe, compute, "amdgcn.device.init.kd");

  KFD_EXPECT(compute.dispatch(begin, begin_cfg, begin_ka, sig));
  KFD_EXPECT(compute.dispatch(start, start_cfg, start_ka, sig));
  KFD_EXPECT(compute.dispatch(end, end_cfg, end_ka, sig));
  KFD_EXPECT(sig.wait(kfd::Condition::EQ, 0, UINT64_MAX));

  launch_if_present(dev, exe, compute, "amdgcn.device.fini.kd");

  rpc_running.store(false, std::memory_order_release);
  std::atomic_ref<uint64_t>(*doorbell.fence_addr()).store(1);
  KFD_EXPECT(doorbell.signal());
  rpc_thread.join();

  return *static_cast<int *>(dev_ret.data());
}
