find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(XCB xcb xcb-dri3 xcb-present)
endif()
if(NOT XCB_FOUND)
  message(WARNING "computetoy: xcb/xcb-dri3/xcb-present not found, skipping build")
  return()
endif()

# Compile image kernels for every target architecture.
foreach(arch IN LISTS LIBKFD_GPU_ARCHS)
  string(REPLACE "-" "_" suffix "${arch}")
  set(target "plasma_${suffix}")

  add_executable(${target} shaders/plasma.c)
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

add_executable(computetoy main.cpp window.cpp)
target_link_libraries(computetoy PRIVATE libkfd::kfd ${XCB_LIBRARIES})
target_include_directories(computetoy PRIVATE ${XCB_INCLUDE_DIRS})
target_compile_options(computetoy PRIVATE ${common_compile_options})
