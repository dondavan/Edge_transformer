import ai_edge_torch
import numpy
import torch
import torchvision
from transformers import BertTokenizer, BertModel

model_name = "bert-base-uncased"
model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
input = (encoded_input['input_ids'], encoded_input['attention_mask'])


desktop_output = model(**encoded_input)

edge_model = ai_edge_torch.convert(model.eval(), input)
edge_output = edge_model(*input)

"""
Validating
"""
print("\n\n\n\n\n\n\n")
print("Validating")
print("\n\n\n\n\n\n\n")


edge_output_tensor = torch.from_numpy(edge_output['last_hidden_state'])
print("Comparing against original PyTorch module")
print(torch.allclose(edge_output_tensor, desktop_output["last_hidden_state"], rtol=1e-3, atol=1e-5))


"""
Export
"""
edge_model.export(f"models/{model_name}.tflite")