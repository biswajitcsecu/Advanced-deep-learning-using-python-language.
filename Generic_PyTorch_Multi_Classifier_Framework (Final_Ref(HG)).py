#!/usr/bin/env python
# coding: utf-8

# ## **Rice Images Classify**

# In[227]:


#Import pkg
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.notebook import tqdm, tnrange,trange
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")


# ## **Transforms**

# In[184]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[185]:


# Define the data transformation for preprocessing
transform = transforms.Compose([
    transforms.Resize((60, 60)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ## **Applying Transform on Data**

# In[186]:


# dataset and apply the transformations
dataset = torchvision.datasets.ImageFolder(root='data/train/', transform=transform)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

num_classes = len(dataset.classes)


# ## **Data Plotting**

# In[189]:


# Function to show an image
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize the image
    npimg = img.numpy()
    npimg = (npimg * 255).astype(np.uint8)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


# In[190]:


# Show images along with their labels
images, labels =next(iter(train_loader))
imshow(torchvision.utils.make_grid(images))
print(' '.join(f'{class_labels[labels[j]]:5s}' for j in range(len(labels))))


# In[203]:


#Data Plotting
batch = next(iter(train_loader))
rows = 4
columns = int(batch_size / rows)
class_dict = {0:'Basmati', 1:'Karacadag', 2:'Arborio', 3:'Jasmine', 4:'Ipsala'}

plt.figure(figsize = (18, 5))
for pos in tqdm(range(batch_size)):
    plt.subplot(rows, columns, pos+1)
    img=batch[0][pos].permute(1,2,0)
    img = img / 2 + 0.5 
    npimg = img.numpy()
    npimg = (npimg * 255).astype(np.uint8)
    plt.imshow(npimg)
    plt.title(class_dict[batch[1][pos].item()])
    plt.axis('off')
plt.tight_layout()


# ## **CNN Model**

# In[204]:


#Model
class ConvNet(nn.Module):
    def __init__(self,output):
        super(ConvNet,self).__init__()        
        self.conv1 = nn.Conv2d(3,16,kernel_size = 3)
        self.batch1 = nn.BatchNorm2d(16) 
        self.rel1 = nn.ReLU()        
        self.conv2 = nn.Conv2d(16,32,kernel_size = 3)
        self.batch2 = nn.BatchNorm2d(32)
        self.rel2 = nn.ReLU()        
        self.maxp2 = nn.MaxPool2d(kernel_size = 2)        
        self.conv3 = nn.Conv2d(32,64,kernel_size = 2)
        self.batch3 = nn.BatchNorm2d(64)
        self.rel3 = nn.ReLU()        
        self.drop3 = nn.Dropout(0.2)        
        self.fc = nn.Linear(64*27*27,output)
        
    def forward(self,inputs):        
        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.rel1(x)        
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.rel2(x)
        x = self.maxp2(x)
        x = self.conv3(x)
        x = self.batch3(x)
        x = self.rel3(x)
        x = self.drop3(x)
        x = x.view(-1,64*27*27)        
        x = self.fc(x)
        
        return(x)


# ## **Hyperpara meters**

# In[205]:


# Hyperparameters
epoch = 1
model = ConvNet(num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
criterion = nn.CrossEntropyLoss()
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer,verbose = 1,patience = 6,factor = 0.0001)


# ## **Model Train**

# In[206]:


#Model Train
train_graph = []
test_graph = []
val_graph = []

for epochs in tqdm(range(epoch)):
    model.train()
    train_loss = 0
    test_loss = 0
    acc = 0
    h = 0       
    for img,lab in train_loader:
        img = img.to(device)
        lab = lab.to(device)
        optimizer.zero_grad()
        
        output = model(img)
        loss = criterion(output,lab)
        loss_tr = loss.item()
        loss.backward()
        optimizer.step()
        train_loss += loss_tr / img.size(0)        
        h += 1 
        
    #Model evaluation
    model.eval()
    
    for test_img,test_label in val_loader:
        test_img,test_label = test_img.to(device),test_label.to(device)        
        out_test = model(test_img)
        
        test_loss_counter = criterion(out_test,test_label)        
        test_loss += test_loss_counter.item() / test_img.size(0)
        
        _,pred = torch.max(out_test,1)
        val_val = torch.sum(pred == test_label.data)
        acc += val_val.cpu() 
        
    validation =  acc / len(val_dataset)
    tetsting = test_loss
    training = train_loss
    print(f"Epoch: {epochs + 1}/{epoch} Validation: {validation } Train Loss: {training} Test Loss: {tetsting}") 
    
    test_graph.append(tetsting)
    val_graph.append(validation)
    train_graph.append(training)


# ## **Model performance**

# In[252]:


#Model performance
plt.figure(figsize=(5,4))
plt.plot(train_graph)
plt.plot(test_graph)
plt.plot(val_graph)
plt.legend(["Train Loss","Test Loss","Doğruluk"])


# ## **Model Prediction**

# In[255]:


#Model Prediction
transformer = transforms.Compose([transforms.Resize((60,60)),transforms.ToPILImage(),transforms.ToTensor()])
val_loader = DataLoader(val_dataset,batch_size = 64,shuffle = True)

model = model.cpu()
plt.figure(figsize = (3, 3))
with torch.no_grad():    
    for img,label in val_loader:
        imgs = img[0]
        labels = label[0]
        output_model = imgs.unsqueeze(0)
        output_model = model(output_model)
        index = output_model.argmax()
        imgs = torch.permute(imgs,(1,2,0))
        imgs = np.clip(imgs, 0, 1)
        plt.imshow(imgs)
        plt.title(f"Guess:{dataset.classes[index]} \n Real:{dataset.classes[labels]}")
        plt.axis('off')
        break


# ## **Model Evaluation**

# In[220]:


#Model Evaluation
model.eval()
y_true = list()
y_pred = list()

with torch.no_grad():
    for val_data in val_loader:
        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
        pred = model(val_images.float()).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(val_labels[i].item())
            y_pred.append(pred[i].item())
            


# In[258]:


#classification report
train_dir='data/train/'
class_names = sorted(os.listdir(train_dir))
num_class = len(class_names)
print(class_names)
print('<=---------------------------------------------------------=>')
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
print('<=---------------------------------------------------------=>')


# In[259]:


#test data
class_names = sorted(os.listdir(train_dir))
print(class_names)
num_class = len(t_class_names)
image_files = [[os.path.join(train_dir, class_name, x) 
               for x in os.listdir(os.path.join(train_dir, class_name))] 
               for class_name in class_names]

image_file_list = []
image_label_list = []
for i, class_name in enumerate(class_names):
    image_file_list.extend(image_files[i])
    image_label_list.extend([i] * len(image_files[i]))
    


# In[246]:


#Prediction
y_pred = list()
with torch.no_grad():
    for val_data in val_loader:
        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
        pred = model(val_images.float()).argmax(dim=1)
        for i in range(len(pred)):
            y_pred.append(pred[i].item())

num_total=len(image_file_list)
plt.subplots(3,3, figsize=(5,5))

for i,k in enumerate(np.random.randint(num_total, size=9)):
    im = Image.open(image_file_list[k])
    arr = np.array(im)
    plt.subplot(3,3, i+1)    
    plt.xlabel(class_names[y_pred[i]]) 
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
plt.tight_layout()
plt.show()


# In[ ]:




