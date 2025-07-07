#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/extension/threadpool/threadpool.h>


#include <iostream>
#include <chrono>



int main()
{
  std::cout << "mobilebert" <<std::endl;
  executorch::extension::threadpool::get_threadpool()->_unsafe_reset_threadpool(4);
  // Create a Module.
  executorch::extension::Module module("./models/mobilebert-uncased.pte");

  // Wrap the input data with a Tensor.
  std::vector<int64_t> input_tokens = {101, 5672, 2033, 2011, 2066, 1012,  102};
  auto inputs = executorch::extension::from_blob(input_tokens.data(),
    {1, static_cast<int>(input_tokens.size())},
    executorch::aten::ScalarType::Long);

  auto start_time = std::chrono::high_resolution_clock::now();
  // Perform an inference.
  const auto result = module.forward(inputs);
  // Check for success or failure.
  if (result.ok()) {
    // Retrieve the output data.
    const auto output = result->at(0).toTensor().const_data_ptr<float>();
  }

  auto   end_time  = std::chrono::high_resolution_clock::now();
  double cost_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
  std::cout << "Run cost: " << cost_time << std::endl;
}