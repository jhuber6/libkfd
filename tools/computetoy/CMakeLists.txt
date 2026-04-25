find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(XCB xcb xcb-dri3 xcb-present)
  pkg_check_modules(LIBDRM libdrm)
endif()
if(NOT XCB_FOUND AND NOT LIBDRM_FOUND)
  message(WARNING "computetoy: neither xcb nor libdrm found, skipping build")
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
target_link_libraries(computetoy PRIVATE libkfd::kfd)
target_compile_options(computetoy PRIVATE ${common_compile_options})

if(XCB_FOUND)
  target_sources(computetoy PRIVATE xcb_window.cpp)
  target_compile_definitions(computetoy PRIVATE HAVE_XCB)
  target_link_libraries(computetoy PRIVATE ${XCB_LIBRARIES})
  target_include_directories(computetoy PRIVATE ${XCB_INCLUDE_DIRS})
endif()

if(LIBDRM_FOUND)
  target_sources(computetoy PRIVATE drm_window.cpp)
  target_compile_definitions(computetoy PRIVATE HAVE_LIBDRM)
  target_link_libraries(computetoy PRIVATE ${LIBDRM_LIBRARIES})
  target_include_directories(computetoy PRIVATE ${LIBDRM_INCLUDE_DIRS})
endif()
