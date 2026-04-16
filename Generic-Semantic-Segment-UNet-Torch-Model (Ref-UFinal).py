#!/usr/bin/env python
# coding: utf-8

# In[51]:


#Import Required Packages
import os
import math
import random
import numpy as np
from numpy.random import randint
import glob
from math import ceil
import pandas as pd
from numpy import linalg as LA
import scipy.spatial.distance
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, tnrange,trange
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch import nn
from torch_snippets import *
from torch_snippets import subplots
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.models import vgg16_bn
from torchsummary import summary


# In[52]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ## **Data processing**

# In[53]:


H,W=[224,224]
tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])


# In[54]:


#Dataset
class SegData(Dataset):
    def __init__(self, split):
        self.items = stems(f'dataset/images_{split}')
        self.split = split
        
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, ix):
        image = cv2.imread(f'dataset/images_{self.split}/{self.items[ix]}.png', 1)
        image = cv2.resize(image, (H,W))
        mask = cv2.imread(f'dataset/masks_{self.split}/{self.items[ix]}.png', 0)
        mask = cv2.resize(mask, (H,W))
        
        return image, mask
    
    def choose(self): 
        return self[randint(len(self))]
    
    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([tfms(im.copy()/255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        
        return ims, ce_masks    


# In[55]:


#Dataset
trn_ds = SegData('train')
val_ds = SegData('test')
trn_dl = DataLoader(trn_ds, batch_size=16, shuffle=True, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)


# ## **Data visulization**

# In[56]:


show(trn_ds[8][0])   


# ## **CNNModel**

# In[95]:


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)
        
        return F.upsample(final, x.size()[2:], mode='bilinear')


# ##  **Loss Function**

# In[96]:


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

#Loss Functions
ce= CrossEntropyLoss2d()
#ce = nn.CrossEntropyLoss()       
#nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss 

def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    
    return ce_loss, acc


# ## **Model Train**

# In[97]:


#Train------------

def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()

#Train-------------

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)
    return loss.item(), acc.item()     


# ## **Hyperparameters**

# In[99]:


#Train
num_classes=12
model = UNet(num_classes).to(device)
criterion = UnetLoss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 15 

summary(model, (3, H, W))


# ## **Train**

# In[ ]:


#Train
log = Report(n_epochs)

for ex in range(n_epochs):
    N = len(trn_dl)
    for bx, data in enumerate(trn_dl):
        loss, acc = train_batch(model, data, optimizer, criterion)
        log.record(ex+(bx+1)/N, trn_loss=loss, trn_acc=acc, end='\r')

    N = len(val_dl)
    for bx, data in enumerate(val_dl):
        loss, acc = validate_batch(model, data, criterion)
        log.record(ex+(bx+1)/N, val_loss=loss, val_acc=acc, end='\r')
        
    log.report_avgs(ex+1)
    
print('<======================Done========================>')     


# ## **Log report**

# In[ ]:


log.plot_epochs(['trn_loss','val_loss'])


# ## **Model Prediction**

# In[ ]:


#Model Prediction
im, mask = next(iter(val_dl))
_mask = model(im)
_, _mask = torch.max(_mask, dim=1)

subplots([im[0].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, titles=['Original image','Original mask','Predicted mask'])     


subplots([im[0].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, titles=['Original image','Original mask','Predicted mask']) 


# ## **Model Evaluation**

# In[ ]:


#Model Prediction
im, mask = next(iter(val_dl))
_mask = model(im)
_, _mask = torch.max(_mask, dim=1)


subplots([im[0].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, titles=['Original image','Original mask','Predicted mask'])     

subplots([im[0].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, titles=['Original image','Original mask','Predicted mask']) 


# In[ ]:





# In[ ]:




