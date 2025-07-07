add_subdirectory("executorch")

target_link_libraries(
  my_target
  PRIVATE executorch
          extension_module_static
          extension_tensor
          optimized_native_cpu_ops_lib
          xnnpack_backend)