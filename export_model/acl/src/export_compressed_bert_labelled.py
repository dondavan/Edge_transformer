
# pip install --upgrade datasets fsspec

# !pip uninstall -y transformers
# !pip install transformers --upgrade

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW   # Import AdamW from torch.optim (PyTorch 1.2+ has it)
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.cluster import KMeans
import copy
import numpy as np

# Load dataset
dataset = load_dataset("glue", "sst2")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

# Tokenization function
def tokenize_fn(example):
    return tokenizer(example["sentence"], truncation=True, padding="max_length", max_length=128)

train_dataset = train_dataset.map(tokenize_fn, batched=True)
val_dataset = val_dataset.map(tokenize_fn, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

###################################################################################################################################
def train_one_epoch(model, train_loader, optimizer, device):
    model.to(device)
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Average training loss: {avg_loss:.4f}")

def validation(model, val_loader, device):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n✅ Accuracy on SST-2 validation set: {acc:.4f}")

def global_weight_clustering(model,cluster_numebr, B=1):
  model.to('cpu')
  weights=np.array([])
  with torch.no_grad():
    count=0
    for name, params in model.named_parameters():
            w=params.reshape(-1,B).numpy()
            if len(weights)==0:
                    weights=w
            else:
                weights = np.concatenate((weights,w ))
            # print(name)
    print('weight_clustering', weights.shape)
    weights = weights.astype('double')
    kmeans = KMeans(n_clusters=cluster_numebr, init='k-means++', max_iter=5, n_init='auto', random_state=0)
    kmeans.fit(weights)
    cluster_centers=torch.from_numpy(kmeans.cluster_centers_)
    cluster_list=[]
    start_index=0
    for name, params in model.named_parameters():
                param_shape=list(params.size())
                w=params.reshape(-1,B)
                print(name, w.shape)
                for i in range(len(w)):
                    ww=np.array(kmeans.labels_[start_index])
                    w[i]=torch.from_numpy(ww)
                    start_index+=1
                cluster_list=w
                reshape_size_tuple=tuple(param_shape)
                cluster_list=torch.tensor(cluster_list,dtype=torch.float)
                cluster_list=cluster_list.reshape(reshape_size_tuple)
                params.data=cluster_list.data
  return cluster_centers , kmeans.labels_
 
# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)   # from torch.optim
num_epochs = 1
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# for i in range(num_epochs):
#     print(f"Epoch {i+1}/{num_epochs}")
#     train_one_epoch(model, train_loader, optimizer, device)
#     lr_scheduler.step()
#     validation(model, val_loader, device)

#save_directory = "../data/bert-sst2-finetuned"
#model.save_pretrained(save_directory)
#tokenizer.save_pretrained(save_directory)
#print(f"✅ Model and tokenizer saved to {save_directory}")



###################################################################################################################################
#bert = BertForSequenceClassification.from_pretrained(save_directory)
#tokenizer = BertTokenizer.from_pretrained(save_directory)
#bert.to(device)

#print("uncompressed model on SST-2 dataset...")
#validation(bert, val_loader, device)

compressed_bert = copy.deepcopy(model)
number_of_clusters = 128
number_blocks = 1

print("Compressing model with global weight clustering...")
cluster_centers, labels=global_weight_clustering(compressed_bert,number_of_clusters, B=number_blocks)
validation(compressed_bert, val_loader, device)

print(cluster_centers)
print(labels)

save_directory = "../models/bert-clustered-labelled"
tokenizer.save_pretrained(save_directory)
compressed_bert.save_pretrained(save_directory)

cluster_centers_32 =cluster_centers.to(torch.float32)
np.save(f"{save_directory}/cluster_center.npy",cluster_centers_32.detach().numpy())

###################################################################################################################################
# Optimizer and scheduler
optimizer = AdamW(compressed_bert.parameters(), lr=2e-5)   # from torch.optim
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

"""
for i in range(4):
    print('##################################################################################')
    print(f"Retraining compressed model Epoch {i+1}/{num_epochs}", 'number_of_clusters:', number_of_clusters, 'number_of_blocks:', number_blocks)
    train_one_epoch(compressed_bert, train_loader, optimizer, device)
    validation( compressed_bert, val_loader, device)
    lr_scheduler.step()
    cluster_centers, labels=global_weight_clustering(compressed_bert,number_of_clusters, B=number_blocks)
    validation( compressed_bert, val_loader, device)
"""




