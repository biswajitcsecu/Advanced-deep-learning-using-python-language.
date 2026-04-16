#!/usr/bin/env python
# coding: utf-8

# In[89]:


import os
import cv2 
import glob
import numpy as np 
import pandas as pd 
from tqdm.notebook import tqdm, tnrange,trange
import segmentation_models_pytorch as smp
from torch_snippets import *
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchsummary import summary
import warnings

warnings.filterwarnings('ignore')


# In[3]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ## **Data processing**

# In[4]:


labels=sorted(glob.glob('dataset/train/masks/*.png'))
images=sorted(glob.glob('dataset/train/images/*.png'))


# In[5]:


tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])


# In[6]:


train_images=images[:int(0.8*(len(images)))]
train_labels=labels[:int(0.8*(len(labels)))]
test_images=images[int(0.8*(len(images))):]
test_label=labels[int(0.8*(len(labels))):]


# In[7]:


class SegData(Dataset):
    def __init__(self,images,labels):
        self.images=images
        self.labels=labels
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, ix):
        image = read(self.images[ix], 1)
        image = cv2.resize(image, (128,128))
        mask = read(self.labels[ix],1)
        mask = cv2.resize(mask, (128,128))
        mask=cv2.cvtColor(mask,cv2.COLOR_RGB2GRAY)
        return image, mask
        
    def choose(self): return self[randint(len(self))]
    
    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([tfms(im.copy()/255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.tensor(mask[None]) for mask in masks]).long().to(device)
        return ims, ce_masks



# In[8]:


train_dataset=SegData(train_images,train_labels)
test_dataset=SegData(test_images,test_label)


# In[9]:


trn_dl = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn,drop_last=True)
test_dl=DataLoader(test_dataset,batch_size=8,shuffle=True,collate_fn=test_dataset.collate_fn)


# ## **Data visulization**

# In[10]:


im, mask = next(iter(trn_dl))
subplots([im[6].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=2, sz=6,
          titles=['Original image','Original mask']) 


# In[11]:


im, mask = next(iter(test_dl)) 
subplots([im[4].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=2, sz=6,
         titles=['Original image','Original mask']) 


# ## **CNNModel**

# In[82]:


backbone  = 'efficientnet-b1'
num_classes=12

def build_model():
    model = smp.Unet(encoder_name=backbone, encoder_weights="imagenet",in_channels=3,classes=num_classes,activation=None,)
    model.to(device)
    
    return model



# ## **Hyperparameters**

# In[83]:


model = build_model()


# ##  **Loss Function**

# In[84]:


ce = nn.CrossEntropyLoss()

def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    acc = (torch.argmax(preds,1) == targets).float().mean()
    return ce_loss, acc


# ## **Model summary**

# In[85]:


summary(model,(3,128,128))


# ## **Model setup**

# In[86]:


model=model.to(device)
criterion = UnetLoss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
n_epochs = 50


# ## **Model Train Structure**

# In[87]:


def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss= criterion(_masks, masks)
    
    return loss.item(), acc.item()
    


# ## **Model Train**

# In[88]:


log = Report(n_epochs)
for ex in range(n_epochs):
    N = len(trn_dl)
    for bx, data in enumerate(trn_dl):
        loss, acc = train_batch(model, data, optimizer, criterion)
        log.record(ex+(bx+1)/N, trn_loss=loss, trn_acc=acc, end='\r')
    log.report_avgs(ex+1)


# ## **Log report**

# In[92]:


log.plot_epochs(['trn_loss','trn_acc'])


# ## **ModelEvaluation**

# In[94]:


img,mask=next(iter(test_dl))
_mask=model(img)
_,_mask=torch.max(_mask,dim=1)

for i in tqdm.tqdm(range(4-1)):
    subplots([img[i].permute(1,2,0).detach().cpu()[:,:,i],_mask.permute(1,2,0).detach().cpu()[:,:,i],
              mask.permute(1,2,0).detach().cpu()[:,:,i]] ,nc=3,sz=8,
             titles=['Original image','Original mask','Predicted mask'])    
 


# ## **Model Prediction**

# In[95]:


#Model Prediction
im, mask = next(iter(test_dl))
_mask = model(im)
_, _mask = torch.max(_mask, dim=1)

subplots([im[5].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3,sz=8,
         titles=['Original image','Original mask','Predicted mask'])     


subplots([im[6].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, sz=8,
         titles=['Original image','Original mask','Predicted mask']) 


# ## **Model Evaluation**

# In[96]:


#Model Prediction
im, mask = next(iter(test_dl))
_mask = model(im)
_, _mask = torch.max(_mask, dim=1)


subplots([im[3].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, sz=8,
         titles=['Original image','Original mask','Predicted mask'])     

subplots([im[4].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]], nc=3, sz=8,
         titles=['Original image','Original mask','Predicted mask']) 


# ## **Model Evaluation**

# In[97]:


#Model Prediction
im, mask = next(iter(test_dl))
_mask = model(im)
_, _mask = torch.max(_mask, dim=1)

#Display multiple images 
subplots([im[1].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, sz=8, 
         titles=['Original image','Original mask','Predicted mask'])     

subplots([im[2].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, sz=8,
         titles=['Original image','Original mask','Predicted mask']) 


# In[ ]:




