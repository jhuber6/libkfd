include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.13.0
  GIT_SHALLOW    TRUE
  SYSTEM
)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)

set(test_compile_options ${common_compile_options} -Wno-c2y-extensions)
set(test_include_dir ${CMAKE_CURRENT_SOURCE_DIR})

function(kfd_add_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE kfd Catch2::Catch2WithMain)
  target_include_directories(${name} PRIVATE ${test_include_dir})
  target_compile_options(${name} PRIVATE ${test_compile_options})
  catch_discover_tests(${name})
endfunction()

function(kfd_add_device_test name source)
  add_executable(${name} ${source})
  target_link_libraries(${name} PRIVATE kfd Catch2::Catch2WithMain)
  target_include_directories(${name} PRIVATE ${test_include_dir})
  target_compile_options(${name} PRIVATE ${test_compile_options})
  catch_discover_tests(${name} TEST_PREFIX "device/" PROPERTIES LABELS device)
endfunction()

# Compile a GPU kernel source for every architecture in LIBKFD_GPU_ARCHS and
# generate a .inc file containing a C initializer list of {path, arch} pairs.
#
# Usage:  add_gpu_kernel(name source_file)
#   name        - base name for targets and the generated .inc file
#   source_file - C source file in the current source directory
# Produces: ${CMAKE_CURRENT_BINARY_DIR}/${name}_kernels.inc
# Appends targets to the global property LIBKFD_GPU_KERNEL_TARGETS.
function(add_gpu_kernel name source_file)
  set(targets "")
  set(entries "")

  foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
    string(REPLACE "-" "_" suffix "${arch}")
    set(target "${name}_${suffix}")

    add_executable(${target} ${source_file})
    set_target_properties(${target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      AMDGPU_ARCH "${arch}"
    )

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

    list(APPEND targets ${target})
    string(APPEND entries
      "  {\"$<TARGET_FILE:${target}>\", \"${arch}\"},\n")
  endforeach()

  set_property(GLOBAL APPEND PROPERTY LIBKFD_GPU_KERNEL_TARGETS ${targets})

  file(GENERATE
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${name}_kernels.inc"
    CONTENT "${entries}"
  )
endfunction()

add_executable(link_test link_test.cpp)
target_link_libraries(link_test PRIVATE "$<LINK_LIBRARY:WHOLE_ARCHIVE,kfd>")
target_link_options(link_test PRIVATE -nostdlib++)
add_test(NAME link_test COMMAND link_test)

add_subdirectory(detail)
add_subdirectory(core)
add_subdirectory(device)
