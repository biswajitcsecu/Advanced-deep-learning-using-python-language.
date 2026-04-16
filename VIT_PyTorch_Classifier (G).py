#!/usr/bin/env python
# coding: utf-8

# ## **Cats vs Dogs: Binary Classifier with PyTorch CNN**

# In[134]:


#Import Libraries
from __future__ import print_function
import glob2
import glob
from itertools import chain
import os,sys
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from vit_pytorch.efficient import ViT
import warnings

warnings.filterwarnings('ignore')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# In[135]:


print(f"Torch: {torch.__version__}")


# ## **Model hyperparameters**

# In[137]:


# Training settings
batch_size = 32
epochs = 2
lr = 3e-5
gamma = 0.7
seed = 42


# In[138]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)


# ## **Load train and test data**

# In[139]:


## Load Data
train_dir = 'data/train/'
test_dir = 'data/test/'

train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))


# In[140]:


print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")


# In[141]:


labels = [path.split('/')[-1].split('.')[0] for path in train_list]


# In[142]:


## Random Plots

random_idx = np.random.randint(1, len(train_list), size=9)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for idx, ax in enumerate(axes.ravel()):
    img = Image.open(train_list[idx])
    ax.set_title(labels[idx])
    ax.imshow(img)


# In[143]:


## Split
train_list, valid_list = train_test_split(train_list,test_size=0.2,random_state=seed)
print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")


# In[158]:


## Image Augmentation
train_transforms = transforms.Compose(
    [        
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5,), (0.5,))
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ]
)

test_transforms = transforms.Compose(
    [        
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ]
)


# ## **Data and Preprocess Images**

# In[159]:


# Load Datasets
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')       
        img_transformed = self.transform(img)
        label = img_path.split("/")[-1].split(".")[0]
        label = 1 if label == "dog" else 0
        
        return (img_transformed, label)


# In[160]:


def collate_fn(data):
    zipped = zip(data)
    return list(zipped)


# In[161]:


train_data = CatsDogsDataset(train_list, transform=train_transforms)
valid_data = CatsDogsDataset(valid_list, transform=test_transforms)
test_data = CatsDogsDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True, drop_last=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True, drop_last=True )
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True, drop_last=True)

print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))


# ## **Convolutional Architecture**

# In[162]:


## Efficient Attention
efficient_transformer = Linformer(dim=128,seq_len=49+1,depth=12, heads=8, k=64)


# In[163]:


## Visual Transformer
model = ViT(dim=128,image_size=224,patch_size=32,num_classes=2,transformer=efficient_transformer,channels=3,).to(device)


# In[164]:


#loss function
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
#scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


# ## **Model Traning**

# In[165]:


# Train
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")


# ## **Making Predictions**

# In[167]:


test_files = os.listdir('data/test/')
test_files = list(filter(lambda x: x != 'test', test_files))

def test_path(p): 
    return f"data/test/{p}"

test_files = list(map(test_path, test_files))

class TestDataset(Dataset):
    def __init__(self, image_paths, test_transforms):
        super().__init__()
        self.paths = image_paths
        self.len = len(self.paths)
        self.test_transforms = test_transforms

    def __len__(self): return self.len

    def __getitem__(self, index): 
        path = self.paths[index]
        image = Image.open(path).convert('RGB')
        image = self.test_transforms(image)
        filed = path.split('/')[-1].split('.')[0]
        return (image, filed)

test_ds = TestDataset(test_files, test_transforms)
test_dl = DataLoader(test_ds, batch_size=32)
len(test_ds), len(test_dl)


# In[168]:


# Model pediction
dog_probs = []
with torch.no_grad():
    for X, fileid in test_dl:
        preds = model(X)
        preds_list = F.softmax(preds, dim=1)[:, 1].tolist()
        dog_probs += list(zip(list(fileid), preds_list))


# In[169]:


# display some images
for img, probs in zip(test_files[:5], dog_probs[:5]):
    pil_im = Image.open(img, 'r')
    label = "dog" if probs[1] > 0.5 else "cat"
    title = "prob of dog: " + str(probs[1]) + " Classified as: " + label
    plt.figure()
    plt.imshow(pil_im)
    plt.suptitle(title)
    plt.show()


# In[ ]:




