import torch
import torchvision.models as models
import numpy as np
import os


from transformers import GPT2Tokenizer, GPT2Model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

print(model)

save_path = f"./data/{model_name}"
if not os.path.isdir(save_path):
   os.makedirs(save_path)

# Embeddding
layer_name = "token_embedding"
np.save(f"./data/{model_name}/{layer_name}.npy",model.wte.weight.detach().numpy())
layer_name = "positional_embedding"
np.save(f"./data/{model_name}/{layer_name}.npy",model.wpe.weight.detach().numpy())

# Encoder
for layer_num in range (0,12):
    save_path = f"./data/{model_name}/layer_{layer_num}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    layer_name = "attn_proj_weight"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.h[layer_num].attn.c_proj.weight.detach().numpy())
    layer_name = "attn_proj_bias"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.h[layer_num].attn.c_proj.bias.detach().numpy())