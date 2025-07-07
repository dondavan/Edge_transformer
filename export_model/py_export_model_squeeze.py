import torch
import torchvision.models as models
from transformers import AutoTokenizer, SqueezeBertModel
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


model_name = "squeezebert-uncased"

tokenizer = AutoTokenizer.from_pretrained("squeezebert/squeezebert-uncased")# Load model directl
model = SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")

model = model.eval()
text = "I am a Robot."
encoded_input = tokenizer(text, return_tensors='pt')
input = (encoded_input['input_ids'],)

print(input)

et_program = to_edge_transform_and_lower(
    torch.export.export(model, input),
    partitioner=[XnnpackPartitioner()]
).to_executorch()


"""
Export
"""

with open(f"models/{model_name}.pte", "wb") as f:
    f.write(et_program.buffer)



"""
Validating
"""
print("\n\n\n\n\n\n\n")
print("Validating")
print("\n\n\n\n\n\n\n")

from typing import List
runtime = Runtime.get()
program = runtime.load_program(f"models/{model_name}.pte")
method = program.load_method("forward")
edge_output = method.execute(input)
print("Run succesfully via executorch")

desktop_output = model(**encoded_input)
print("edge:")
print(edge_output[0])
print("desktop")
print(desktop_output["last_hidden_state"])
print("Comparing against original PyTorch module")
print(torch.allclose(edge_output[0], desktop_output["last_hidden_state"], rtol=1e-3, atol=1e-5))