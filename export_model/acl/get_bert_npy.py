import torch
import torchvision.models as models
from transformers import BertTokenizer, BertModel
import numpy as np
import os

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased").eval()

save_path = f"./data/{model_name}"
if not os.path.isdir(save_path):
   os.makedirs(save_path)


vocabs = tokenizer.get_vocab()
with open(f'{save_path}/vocab.txt', 'w') as vocab_file:
    for vocab_dict in vocabs:
        vocab_file.write(vocab_dict+" "+str(vocabs[vocab_dict])+"\n")

vocab_file.close()

# Embeddding
layer_name = "token_embedding"
np.save(f"./data/{model_name}/{layer_name}.npy",model.embeddings.word_embeddings.weight.detach().numpy())
layer_name = "segment_embedding"
np.save(f"./data/{model_name}/{layer_name}.npy",model.embeddings.token_type_embeddings.weight.detach().numpy())
layer_name = "positional_embedding"
np.save(f"./data/{model_name}/{layer_name}.npy",model.embeddings.position_embeddings.weight.detach().numpy())

# Pooler
layer_name = "pooler_weight"
np.save(f"./data/{model_name}/{layer_name}.npy",model.pooler.dense.weight.detach().numpy())
layer_name = "pooler_bias"
np.save(f"./data/{model_name}/{layer_name}.npy",model.pooler.dense.bias.detach().numpy())

# Encoder
for layer_num in range (0,12):
    save_path = f"./data/{model_name}/layer_{layer_num}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    layer_name = "query_weight"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.query.weight.detach().numpy())
    layer_name = "query_bias"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.query.bias.detach().numpy())

    layer_name = "key_weight"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.key.weight.detach().numpy())
    layer_name = "key_bias"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.key.bias.detach().numpy())

    layer_name = "value_weight"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.value.weight.detach().numpy())
    layer_name = "value_bias"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.value.bias.detach().numpy())


    layer_name = "ff_weight_0"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].intermediate.dense.weight.detach().numpy())
    layer_name = "ff_bias_0"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].intermediate.dense.bias.detach().numpy())

    layer_name = "ff_weight_1"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].output.dense.weight.detach().numpy())
    layer_name = "ff_bias_1"
    np.save(f"./data/{model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].output.dense.bias.detach().numpy())