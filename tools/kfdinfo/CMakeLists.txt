add_executable(kfdinfo main.cpp)
target_link_libraries(kfdinfo PRIVATE libkfd::kfd)
target_compile_options(kfdinfo PRIVATE ${common_compile_options})
install(TARGETS kfdinfo DESTINATION ${CMAKE_INSTALL_BINDIR})
