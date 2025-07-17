import torch
import torchvision.models as models
from transformers import BertTokenizer, BertModel
import numpy as np
import os

ori_model_name = "bert-clustered-labelled"
ori_model_path = "../models"
ori_dir = f"{ori_model_path }/{ori_model_name}"

data_path = "../data"
save_model_name = "bert-clustered-labelled"
model = BertModel.from_pretrained(ori_dir)
tokenizer = BertTokenizer.from_pretrained(ori_dir)

save_path = f"{data_path}/{save_model_name}"
if not os.path.isdir(save_path):
   os.makedirs(save_path)


vocabs = tokenizer.get_vocab()
with open(f'{save_path}/vocab.txt', 'w') as vocab_file:
    for vocab_dict in vocabs:
        vocab_file.write(vocab_dict+" "+str(vocabs[vocab_dict])+"\n")

vocab_file.close()

# Cluster Center
cluster_center = np.load(f"{ori_model_path}/{ori_model_name}/cluster_center.npy")
np.save(f"{data_path}/{save_model_name}/cluster_center.npy",cluster_center)

# Embeddding
layer_name = "token_embedding"
np.save(f"{data_path}/{save_model_name}/{layer_name}.npy",model.embeddings.word_embeddings.weight.detach().numpy())
layer_name = "segment_embedding"
np.save(f"{data_path}/{save_model_name}/{layer_name}.npy",model.embeddings.token_type_embeddings.weight.detach().numpy())
layer_name = "positional_embedding"
np.save(f"{data_path}/{save_model_name}/{layer_name}.npy",model.embeddings.position_embeddings.weight.detach().numpy())

# Pooler
layer_name = "pooler_weight"
np.save(f"{data_path}/{save_model_name}/{layer_name}.npy",model.pooler.dense.weight.detach().numpy())
layer_name = "pooler_bias"
np.save(f"{data_path}/{save_model_name}/{layer_name}.npy",model.pooler.dense.bias.detach().numpy())

# Encoder
for layer_num in range (0,12):
    save_path = f"{data_path}/{save_model_name}/layer_{layer_num}"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    layer_name = "query_weight"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.query.weight.detach().numpy())
    layer_name = "query_bias"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.query.bias.detach().numpy())

    layer_name = "key_weight"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.key.weight.detach().numpy())
    layer_name = "key_bias"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.key.bias.detach().numpy())

    layer_name = "value_weight"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.value.weight.detach().numpy())
    layer_name = "value_bias"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].attention.self.value.bias.detach().numpy())


    layer_name = "ff_weight_0"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].intermediate.dense.weight.detach().numpy())
    layer_name = "ff_bias_0"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].intermediate.dense.bias.detach().numpy())

    layer_name = "ff_weight_1"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].output.dense.weight.detach().numpy())
    layer_name = "ff_bias_1"
    np.save(f"{data_path}/{save_model_name}/layer_{layer_num}/{layer_name}.npy",model.encoder.layer[layer_num].output.dense.bias.detach().numpy())