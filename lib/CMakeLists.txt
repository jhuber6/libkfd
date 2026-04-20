target_sources(kfd PRIVATE
  device.cpp
  event.cpp
  context.cpp
  loader.cpp
  memory.cpp
  queue.cpp
  signal.cpp
  topology.cpp
  trap_handler.c

  detail/mutex.cpp
  detail/scratch.cpp
  detail/elf.cpp
  detail/mapped_region.cpp
  detail/pool_allocator.cpp
)

# Build per-architecture AMDGCN trap handler executables. The list here must
# match the #embed entries in trap_handler.c.
set(trap_targets "")
foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
    set(target trap_handler_${arch}.bin)
    add_executable(${target} device/trap_handler.S)
    target_compile_options(${target} PRIVATE
      --target=amdgcn--amdhsa
      -mcpu=${arch}
    )
    target_link_options(${target} PRIVATE
      --target=amdgcn--amdhsa
      -mcpu=${arch}
      -Wl,--entry=trap_entry
    )
    list(APPEND trap_targets ${target})
endforeach()

target_compile_options(kfd PRIVATE --embed-dir=${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(kfd ${trap_targets})
