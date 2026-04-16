#!/usr/bin/env python
# coding: utf-8

# ## **Endoscop semantic segmentation UNet Pre-trained Model-Torch**

# In[104]:


import sys
import numpy
import random
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import os
import cv2
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch import cat
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import BasicBlock
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from torchvision import models
from torch.nn import Upsample
import torch.optim as optim
from torchsummary import summary
import torchvision.transforms as transforms
from torchvision.models import segmentation 
from torch.nn import Conv2d as Conv2D
import torch.nn.init as init

import warnings
warnings.filterwarnings('ignore')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[52]:


seed = 42
def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
set_seed(seed)


# ## **Loading the Dataset**

# In[53]:


#Data load
class Dataset(torch.utils.data.Dataset):
    def __init__(self, images, image_folder, mask_folder, transform=None):
        self.images = images
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.images[idx])
        mask_path = os.path.join(self.mask_folder, self.images[idx])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask).unsqueeze(0)


# In[54]:


#Data prepaire
base_path = 'Kvasir/train/'
image_folder = os.path.join(base_path, 'images')
mask_folder = os.path.join(base_path, 'masks')

images = os.listdir(image_folder)
masks = os.listdir(mask_folder)

train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)
test_images, val_images = train_test_split(test_images, test_size=0.5, random_state=42)


# In[55]:


#dataset sizes
dataset_sizes = [len(train_images), len(test_images), len(val_images)]
labels = ["Train", "Test", "Val"]

plt.pie(dataset_sizes, labels = labels)
plt.show()


# In[56]:


#Data preparation
H,W=[128, 128]
nbatch_size=16

train_transform = A.Compose([A.Resize(H, W), A.HorizontalFlip(p= 0.5), A.VerticalFlip(p= 0.5), A.RandomRotate90(p= 0.5)])
val_transform = A.Compose([A.Resize(H, W), A.HorizontalFlip(p= 0.5), A.VerticalFlip(p= 0.5), A.RandomRotate90(p= 0.5)])
test_transform = A.Compose([A.Resize(H, W)])

trainset = Dataset(train_images, image_folder, mask_folder, transform= train_transform)
testset = Dataset(test_images, image_folder, mask_folder, transform= test_transform)
valset = Dataset(val_images, image_folder, mask_folder, transform= val_transform)

train_batch_size = nbatch_size
val_batch_size = nbatch_size
test_batch_size = nbatch_size

#Data loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size= train_batch_size, num_workers= 2, shuffle= True)
valloader = torch.utils.data.DataLoader(valset, batch_size= val_batch_size, num_workers= 2, shuffle= True)
testloader = torch.utils.data.DataLoader(testset, batch_size= test_batch_size, num_workers= 2, shuffle= True)


# ## **Data visulization**

# In[57]:


#Display train images and masks
images, masks = next(iter(trainloader))
images = images.permute(0, 2, 3, 1)
masks = masks.permute(0, 2, 3, 1)
plt.figure(figsize=(18, 6))
for i in tqdm(range(6)):
    plt.subplot(2, 6, i+1)
    plt.axis("off")
    plt.imshow(images[i])

    plt.subplot(2, 6, i+7)
    plt.axis("off")
    plt.imshow(masks[i])
plt.tight_layout()
plt.show()


# In[58]:


#Display  test images and masks
images, masks = next(iter(testloader))
images = images.permute(0, 2, 3, 1)
masks = masks.permute(0, 2, 3, 1)
plt.figure(figsize=(18, 6))

for i in tqdm(range(6)):
    plt.subplot(2, 6, i+1)
    plt.axis("off")
    plt.imshow(images[i])

    plt.subplot(2, 6, i+7)
    plt.axis("off")
    plt.imshow(masks[i])
    
plt.tight_layout()
plt.show()


# ## **Designing the U-NET model**
# 

# In[59]:


class Up(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv = nn.Sequential(
            Conv2D(channel_in, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )        
        
    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        difference_in_X = x1.size()[2] - x2.size()[2]
        difference_in_Y = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (difference_in_X // 2, int(difference_in_X / 2),
                        difference_in_Y // 2, int(difference_in_Y / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Down(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(Down, self).__init__()
        self.conv = nn.Sequential(
            Conv2D(channel_in, channel_out, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = F.max_pool2d(x,2)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, channel_in, classes):
        super(UNet, self).__init__()
        self.input_conv = self.conv = nn.Sequential(
            Conv2D(channel_in, 8, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 32)
        self.up1 = Up(64, 16)
        self.up2 = Up(32, 8)
        self.up3 = Up(16, 4)
        self.output_conv = nn.Conv2d(4, classes, kernel_size = 1)
        
    def forward(self, x):
        x1 = self.input_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        output = self.output_conv(x)
        return F.sigmoid(output)
    
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight, gain=numpy.sqrt(2.0))
        init.constant(m.bias, 0.1)
    


# In[61]:


model = UNet(3, 1)

print(model)


# In[62]:


#Model summury
summary(model, input_size=(3, H, W))


# ## **Loss function**

# In[63]:


class DICE_BCE_Loss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        intersection = 2*(logits * targets).sum() + self.smooth
        union = (logits + targets).sum() + self.smooth
        dice_loss = 1. - intersection / union
        loss = nn.BCEWithLogitsLoss()  
        bce_loss = loss(logits, targets)

        return dice_loss + bce_loss
    
def dice_coeff(logits, targets):
    intersection = 2*(logits * targets).sum()
    union = (logits + targets).sum()
    if union == 0:
        return 1
    dice_coeff = intersection / union
    
    return dice_coeff.item()


# ## **Train Function**

# In[64]:


#train
def train(model, trainloader, optimizer, loss, epochs=10):
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0
        train_dice = 0
        for i, (images, masks) in enumerate(trainloader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            l = loss(logits, masks)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_dice += dice_coeff(logits, masks)
        train_loss /= len(trainloader)
        train_dice /= len(trainloader)
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        #Validation
        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for i, (images, masks) in enumerate(valloader):
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                l = loss(logits, masks)
                val_loss += l.item()
                val_dice += dice_coeff(logits, masks)
        val_loss /= len(valloader)
        val_dice /= len(valloader)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        print(f"Epoch:{epoch + 1} Train Loss:{train_loss:.4f}|Train DICE Coeff:{train_dice:.4f}|Val Loss:{val_loss:.4f}|Val DICE Coeff: {val_dice:.4f}")
        
    return train_losses, train_dices, val_losses, val_dices


# ## **Hyperparameters and Training**

# In[65]:


#Train
epochs = 100
loss = DICE_BCE_Loss()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses, train_dices, val_losses, val_dices = train(model, trainloader, optimizer, loss, epochs)


# In[78]:


#Performance plot
plt.figure(figsize= (10, 6))
plt.subplot(1, 2, 1)
plt.plot(np.arange(epochs), train_dices)
plt.plot(np.arange(epochs), val_dices)
plt.xlabel("Epoch")
plt.ylabel("DICE Coeff")
plt.legend(["Train DICE", "Val DICE"])
#Train loss and validation loss
plt.subplot(1, 2, 2)
plt.plot(np.arange(epochs), train_losses)
plt.plot(np.arange(epochs), val_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train Loss", "Val Loss"])
plt.tight_layout()
plt.show()


# ## **Testing Model**

# In[79]:


#Model Evaluation
images, masks = next(iter(testloader))
with torch.no_grad():
    pred = model(images.to(device)).cpu().detach()
    pred = pred > 0.5

def display_batch(images, masks, pred):
    images = images.permute(0, 2, 3, 1)
    masks = masks.permute(0, 2, 3, 1)
    pred = pred.permute(0, 2, 3, 1)

    images = images.numpy()
    masks = masks.numpy()
    pred = pred.numpy()

    images = np.concatenate(images, axis=1)
    masks = np.concatenate(masks, axis=1)
    pred = np.concatenate(pred, axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(20, 8))
    fig.tight_layout()
    ax[0].imshow(images)
    ax[0].set_title('Images')
    ax[1].imshow(masks, cmap= 'gray')
    ax[1].set_title('Masks')
    ax[2].imshow(pred, cmap= 'turbo')
    ax[2].set_title('Predictions')
    
    plt.tight_layout()
    plt.show()

display_batch(images, masks, pred)


# ## **Prediction of model**

# In[80]:


#Predicted masks and image
test_dataloader_iter = iter(testloader)
inputs, labels = next(test_dataloader_iter)
inputs = inputs.to(device)

with torch.no_grad():
    outputs = model(inputs)
    
predicted_masks = outputs.detach().cpu().numpy()

# Plot the images and masks
batch_size=8
fig, axes = plt.subplots(nrows=batch_size, ncols=3, figsize=(8, 18))

for i in tqdm(range(batch_size)):
    axes[i, 0].imshow(images[i].permute(1, 2, 0))
    axes[i, 0].axis('off')
    axes[i, 0].set_title('Input Image')
    
    axes[i, 1].imshow(labels[i].permute(1, 2, 0))
    axes[i, 1].axis('off')
    axes[i, 1].set_title('Mask Image')

    axes[i, 2].imshow(predicted_masks[i].squeeze(), cmap='Dark2')
    axes[i, 2].axis('off')
    axes[i, 2].set_title('Predicted Mask')

plt.tight_layout()
plt.show()


# In[122]:


#Prediction model
test_iterator = iter(testloader)

for batch in test_iterator:
    inputs, masks = batch        
    # Convert predicted masks to numpy arrays
    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(inputs)
        
    predicted_masks = outputs.detach().cpu().numpy()

    # Plot the input image, ground truth mask, and predicted mask   
    for i in tqdm(range(len(inputs))):
        fig = plt.figure(figsize=(6, 14))
        plt.subplot(1, 3, 1)
        plt.imshow(inputs[i].permute(1, 2, 0))
        plt.axis('off')
        plt.title('Input Image')
        
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title('Ground Truth Mask')
        
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(predicted_masks[i].squeeze(), cmap='cividis')
        plt.title('Predicted Mask')
        plt.tight_layout()
        plt.show()
        


# In[126]:


#Model inference
random.seed(10)
samples = random.sample(range(len(testloader)), 5)

cols = ['Image', 'Mask', 'Prediction']
fig, axes = plt.subplots(len(samples), 3, figsize=(9, 12), sharex='row', sharey='row',subplot_kw={'xticks':[], 'yticks':[]})

for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=10) 
i=0
images, masks = next(iter(testloader))    
for i in tqdm(range(len(samples))):
        image, mask = images[i], masks[i]
        pred = model(torch.tensor(image).unsqueeze(0).to(device))
        pred = (pred > 0.5).squeeze(0)  

        mask = mask.squeeze(0)
        pred = np.array(pred.squeeze(0).cpu())
        axes[i, 0].imshow(np.array(image).transpose(1, 2, 0));
        axes[i, 1].imshow(mask,cmap='gray');
        axes[i, 2].imshow(pred,cmap='cividis');
        
plt.tight_layout()
plt.show()        
        


# ## **-----------------END-----------------**
