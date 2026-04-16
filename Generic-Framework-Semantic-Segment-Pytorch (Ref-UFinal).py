#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Required Packages
import os
import math
import random
import numpy as np
from numpy.random import randint
import glob
import pandas as pd
from numpy import linalg as LA
import scipy.spatial.distance
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, tnrange,trange
from torch_snippets import *
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torchvision.models import vgg16_bn
from torchsummary import summary


# In[47]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ## **Data processing**

# In[48]:


H,W=[224,224]
tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])


# In[49]:


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
        mask = cv2.imread(f'dataset/annotations_{self.split}/{self.items[ix]}.png', 0)
        mask = cv2.resize(mask, (H,W))
        
        return image, mask
    
    def choose(self): return self[randint(len(self))]
    
    def collate_fn(self, batch):
        ims, masks = list(zip(*batch))
        ims = torch.cat([tfms(im.copy()/255.)[None] for im in ims]).float().to(device)
        ce_masks = torch.cat([torch.Tensor(mask[None]) for mask in masks]).long().to(device)
        
        return ims, ce_masks
     


# In[50]:


#Dataset
trn_ds = SegData('train')
val_ds = SegData('test')
trn_dl = DataLoader(trn_ds, batch_size=4, shuffle=True, collate_fn=trn_ds.collate_fn)
val_dl = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=val_ds.collate_fn)


# ## **Data visulization**

# In[51]:


show(trn_ds[10][0])   


# ## **CNNModel**

# In[52]:


def conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True)
    )
     

class UNet(nn.Module):
    def __init__(self, pretrained=True, out_channels=12):
        super().__init__()

        self.encoder = vgg16_bn(pretrained=pretrained).features
        self.block1 = nn.Sequential(*self.encoder[:6])
        self.block2 = nn.Sequential(*self.encoder[6:13])
        self.block3 = nn.Sequential(*self.encoder[13:20])
        self.block4 = nn.Sequential(*self.encoder[20:27])
        self.block5 = nn.Sequential(*self.encoder[27:34])

        self.bottleneck = nn.Sequential(*self.encoder[34:])
        self.conv_bottleneck = conv(512, 1024)

        self.up_conv6 = up_conv(1024, 512)
        self.conv6 = conv(512 + 512, 512)
        self.up_conv7 = up_conv(512, 256)
        self.conv7 = conv(256 + 512, 256)
        self.up_conv8 = up_conv(256, 128)
        self.conv8 = conv(128 + 256, 128)
        self.up_conv9 = up_conv(128, 64)
        self.conv9 = conv(64 + 128, 64)
        self.up_conv10 = up_conv(64, 32)
        self.conv10 = conv(32 + 64, 32)
        self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        bottleneck = self.bottleneck(block5)
        x = self.conv_bottleneck(bottleneck)

        x = self.up_conv6(x)
        x = torch.cat([x, block5], dim=1)
        x = self.conv6(x)

        x = self.up_conv7(x)
        x = torch.cat([x, block4], dim=1)
        x = self.conv7(x)

        x = self.up_conv8(x)
        x = torch.cat([x, block3], dim=1)
        x = self.conv8(x)

        x = self.up_conv9(x)
        x = torch.cat([x, block2], dim=1)
        x = self.conv9(x)

        x = self.up_conv10(x)
        x = torch.cat([x, block1], dim=1)
        x = self.conv10(x)

        x = self.conv11(x)

        return x
    


# ##  **Loss Function**

# In[53]:


ce = nn.CrossEntropyLoss()

def UnetLoss(preds, targets):
    ce_loss = ce(preds, targets)
    acc = (torch.max(preds, 1)[1] == targets).float().mean()
    
    return ce_loss, acc


# ## **Model Train**

# In[54]:


#Train
def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    _masks = model(ims)
    optimizer.zero_grad()
    loss, acc = criterion(_masks, ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item(), acc.item()
#Train

@torch.no_grad()
def validate_batch(model, data, criterion):
    model.eval()
    ims, masks = data
    _masks = model(ims)
    loss, acc = criterion(_masks, masks)
    return loss.item(), acc.item()
     


# ## **Hyperparameters**

# In[ ]:


#Train
model = UNet().to(device)
criterion = UnetLoss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 30  

summary(model, (1, H, W))


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
     


# ## **Log report**

# In[ ]:


log.plot_epochs(['trn_loss','val_loss'])


# ## **Model Prediction**

# In[ ]:


#Model Prediction
im, mask = next(iter(val_dl))
_mask = model(im)
_, _mask = torch.max(_mask, dim=1)

figure, axes = plt.subplots(8,2, figsize=(6,6))
subplots([im[0].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, titles=['Original image','Original mask','Predicted mask'])     


figure, axes = plt.subplots(8,2, figsize=(6,6))
subplots([im[1].permute(1,2,0).detach().cpu()[:,:,0], mask.permute(1,2,0).detach().cpu()[:,:,0]
          ,_mask.permute(1,2,0).detach().cpu()[:,:,0]],nc=3, titles=['Original image','Original mask','Predicted mask']) 


# In[ ]:




