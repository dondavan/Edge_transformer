# Note: this script is the same as the script from the "Setting up ExecuTorch"
# page, with one minor addition to lower to the Vulkan backend.
import torch
from torch.export import export
from executorch.exir import to_edge

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

# Start with a PyTorch model that adds two input tensors (matrices)
class Add(torch.nn.Module):
  def __init__(self):
    super(Add, self).__init__()

  def forward(self, x: torch.Tensor, y: torch.Tensor):
      return x + y

# 1. torch.export: Defines the program with the ATen operator set.
aten_dialect = export(Add(), (torch.ones(1), torch.ones(1)))

# 2. to_edge: Make optimizations for Edge devices
edge_program = to_edge(aten_dialect)
# 2.1 Lower to the Vulkan backend
edge_program = edge_program.to_backend(VulkanPartitioner())

# 3. to_executorch: Convert the graph to an ExecuTorch program
executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
with open("vk_add.pte", "wb") as file:
    file.write(executorch_program.buffer)