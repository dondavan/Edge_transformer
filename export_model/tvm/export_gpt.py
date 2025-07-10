import torch
import os
import numpy as np
import torch
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program
from transformers import GPT2Tokenizer, GPT2Model
import tvm
from tvm import relax
from datetime import datetime


from transformers import AutoModelForCausalLM, AutoTokenizer

#model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", torch_dtype=torch.float, attn_implementation="eager",torchscript=True)
#tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

# Export Models
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', torchscript=True)
model = model.to(torch.float).eval()

text = "I am a super good Robot."
encoded_input = tokenizer(text, return_tensors='pt')

input = (encoded_input['input_ids'].to(torch.float),)


exported_program = export(model, args=input)
mod: tvm.IRModule = from_exported_program(exported_program)


mod, params = relax.frontend.detach_params(mod)

### Run
target = tvm.target.arm_cpu()
ex = tvm.compile(mod, target=target)
dev = tvm.cpu()
vm = relax.VirtualMachine(ex, dev)
# Need to allocate data and params on GPU device
gpu_data = tvm.nd.array(encoded_input['input_ids'].to(torch.float).detach(), dev)

start_time = datetime.now()
gpu_out = vm["main"](gpu_data)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

print(gpu_out)