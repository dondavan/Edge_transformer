add_subdirectory("executorch")

add_executable(gpt2_cpu gpt2_cpu.cpp)

target_link_libraries(
  gpt2_cpu
  PRIVATE executorch
          extension_module_static
          extension_tensor
          optimized_native_cpu_ops_lib
          xnnpack_backend)