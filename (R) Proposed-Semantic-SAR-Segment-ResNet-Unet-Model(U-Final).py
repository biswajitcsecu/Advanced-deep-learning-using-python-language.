#!/usr/bin/env python
# coding: utf-8

# ## **SAR Semantic Segment ResNet Unet-Model (U-Final)**

# In[22]:


#Libraries------------------ 
from __future__ import print_function
get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc
import glob
import cv2
from PIL import Image
import glob2
from pathlib import Path
from tqdm.notebook import trange, tqdm
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D, Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose,SeparableConv2D, MaxPooling2D, Activation
from tensorflow.keras.layers import  LeakyReLU,  MaxPool2D,Flatten
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.utils import plot_model
from tensorflow.keras import models
from tensorflow.python.keras.utils import*
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pickle
import keract
import sklearn
from sklearn.cluster import KMeans
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.callbacks import * 
from tensorflow.keras.applications import ResNet50
from sklearn.utils import shuffle
from tensorflow.keras.metrics import MeanIoU
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


K.clear_session()
warnings.filterwarnings('ignore')
plt.style.use("ggplot")



# ## **Data Loading and Preprocessing** 

# In[23]:


#Load image data-------------------
H,W,CH=[128,128,3]
def cv_load_img(path):
    img= cv2.imread(path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(W,H))
    return img


# In[24]:


#Load data---------------------
BASE_DIR="SAR/train/"
img_path= os.listdir(BASE_DIR+'images')
mask_path= os.listdir(BASE_DIR+'masks')


# In[25]:


#plot sample images--------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in tqdm(range(5)):
    path= BASE_DIR + 'images/'
    ax[i].imshow(load_img(path + img_path[i]))
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()


# In[26]:


#plot sample masks--------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in tqdm(range(5)):
    path= BASE_DIR + 'masks/'
    ax[i].imshow(cv_load_img(path + mask_path[i])[:, :, 0], 'gray')
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()


# In[27]:


#plot sample images--with blended mask ------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in tqdm(range(5)):
    path1= BASE_DIR + 'images/'
    ax[i].imshow((cv_load_img(path1 + img_path[i])/255) * (cv_load_img(path + mask_path[i])/255))
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()


# ## **Augmented Images and Masks**

# In[28]:


#data  preparation
X_train, X_test, y_train, y_test = train_test_split(img_path, mask_path, test_size=0.2, random_state=22)
len(X_train), len(X_test)


# In[29]:


#batch generation-----------------------
def load_data(path_list, gray=False):
    data=[]
    for path in tqdm(path_list):
        img= cv_load_img(path)
        if gray:
            img= img[:, :, 0:1]
        img= cv2.resize(img, (W, H))
        data.append(img)
    return np.array(data)


# In[30]:


#train data generation---------------------
X_train= load_data([BASE_DIR + 'images/' + x for x in X_train])/255.0
X_test= load_data([BASE_DIR + 'images/' + x for x in X_test])/255.0
X_train.shape, X_test.shape


# In[31]:


##test data generation---------------------
Y_train= load_data([BASE_DIR + 'masks/' + x for x in y_train], gray=True)/255.0
Y_test= load_data([BASE_DIR + 'masks/' + x for x in y_test], gray=True)/255.0
Y_train= Y_train.reshape(-1, W, H, 1)
Y_test= Y_test.reshape(-1, W, H, 1)

Y_train.shape, Y_test.shape


# ## **Model UNet-Resnet34**

# In[32]:


BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

# preprocess input
X_train = preprocess_input(X_train)
X_val = preprocess_input(X_test)


# ## **Loss metrics**

# In[33]:


# dice_loss metric
def dice_loss(y_true, y_pred):
    numerator = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true * y_true) + tf.reduce_sum(y_pred * y_pred) - tf.reduce_sum(y_true * y_pred)

    return 1 - numerator / denominator


# In[34]:


#iou metric
smooth =100
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


# In[35]:


#jacard_coef
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)



# ## **Model configure**

# In[36]:


# define model
model = Unet(BACKBONE, encoder_weights='imagenet',classes=1, input_shape=(H, W, CH),encoder_features='default',
             decoder_filters=(256, 128, 64, 32, 16)
            )
#loss = keras.losses.BinaryCrossentropy()
model.compile('Adam', loss=[bce_jaccard_loss,dice_loss], metrics=['accuracy', jacard_coef, iou])
#loss=[bce_jaccard_loss,dice_loss]


# ## **Model Summary**

# In[37]:


#Summary of model------------------
model.summary()


# In[38]:


#Plot of model------------------
dot_img_file = 'model.png'
#plot_model(model, to_file=dot_img_file, show_shapes=True)


# In[ ]:


#Train model---------------------------
nbatch_size=32
nepochs=50
history = model.fit(X_train,Y_train,batch_size=nbatch_size,
                    epochs=nepochs,validation_split=0.2,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,                   
                   )


# ## **Performance Plots**

# In[ ]:


df_result = pd.DataFrame(history.history)
df_result


# ## **Visualize the model predictions**

# In[ ]:


# Plotting loss change over epochs---------------
nrange=nepochs
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['loss'])
plt.title('change in loss over epochs')
plt.legend(['training_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
#plt.axis('off')
plt.grid(None)
plt.show()
plt.tight_layout()

# Plotting accuracy change over epochs---------------------
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['accuracy'])
plt.title('change in training accuracy coefitient over epochs')
plt.legend(['training accuracy'])
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.grid(None)
plt.show()
plt.tight_layout()

# Plotting accuracy change over epochs---------------------
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['jacard_coef'])
plt.title('change in jacard_coef coefitient over epochs')
plt.legend(['jacard_coef'])
plt.xlabel('epochs')
plt.ylabel('jacard_coef')
plt.grid(None)
plt.show()
plt.tight_layout()


# In[ ]:


# Creating predictions on our test set-----------------
predictions = model.predict(X_test)


# In[ ]:


# create predictes mask--------------
def create_mask(predictions,input_shape=(W,H,1)):
    mask = np.zeros(input_shape)
    mask[predictions>0.5] = 1
    return mask


# In[ ]:


# Ploting results for one image----------------

def plot_results_for_one_sample(sample_index):
    
    mask = create_mask(predictions[sample_index])    
    fig = plt.figure(figsize=(20,20))
    #image
    fig.add_subplot(1,4,1)
    plt.title('Input image')
    plt.imshow(X_test[sample_index])
    plt.axis('off')
    plt.grid(None)
    #mask
    fig.add_subplot(1,4,2)
    plt.title('Real mask')
    plt.imshow(Y_test[sample_index],cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Predicted mask
    fig.add_subplot(1,4,3)
    plt.title('Predicted mask')  
    plt.imshow(mask,cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Segment
    fig.add_subplot(1,4,4)
    plt.title("Segment image")
    plt.imshow(X_test[sample_index]*mask)
    plt.grid(None)
    plt.axis('off')  
    fig.tight_layout()
    


# ## **Model prediction**

# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(0)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(2)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(3)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(4)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(5)


# ## **Activation silency map**

# In[ ]:


#Activation silency map
img_path='SAR/data/1.png'
image = load_img(img_path, target_size= (H, W))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
y_hat = model.predict(image)


# In[ ]:


#layers
layers=['conv0']

activations= keract.get_activations(model, image, layer_names= layers, nodes_to_evaluate= None, output_format= 'simple',
                                    auto_compile= True)
keract.display_activations(activations, cmap='viridis', save= False, directory= 'activations')


# ## ---------------------------------**END**-------------------------------

# In[ ]:




