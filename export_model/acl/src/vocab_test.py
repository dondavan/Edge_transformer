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