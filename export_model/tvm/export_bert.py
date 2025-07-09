import torch
import os
import numpy as np
import torch
from torch.export import export
from transformers import BertTokenizer, BertModel

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program



model = BertModel.from_pretrained("bert-base-uncased")
model.to(torch.float)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "I am a super good Robot."
encoded_input = tokenizer(text, return_tensors='pt')

input = (encoded_input['input_ids'],)

exported_program = export(model, args=input)

# Use the importer to import the ExportedProgram to Relax.
mod: tvm.IRModule = from_exported_program(exported_program)

print(mod)