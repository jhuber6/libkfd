# Compile the SAXPY GPU kernel for every target architecture.
foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
  string(REPLACE "-" "_" suffix "${arch}")
  set(target "saxpy_${suffix}")

  add_executable(${target} saxpy.c)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/kernels")

  target_compile_options(${target} PRIVATE
    ${common_compile_options}
    --target=amdgcn--amdhsa
    -mcpu=${arch}
    -nogpulib
    -nostdlibinc
    -O2
  )

  target_link_options(${target} PRIVATE
    --target=amdgcn--amdhsa
    -mcpu=${arch}
  )
endforeach()

add_executable(sandbox main.cpp)
target_link_libraries(sandbox PRIVATE libkfd::kfd)
target_compile_options(sandbox PRIVATE ${common_compile_options})
