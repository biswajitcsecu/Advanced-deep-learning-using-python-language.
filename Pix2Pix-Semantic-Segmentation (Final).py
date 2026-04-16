#!/usr/bin/env python
# coding: utf-8

# ## **Pix2Pix-Semantic-Segmentation**

# In[48]:


from __future__ import print_function
get_ipython().run_line_magic('env', 'SM_FRAMEWORK=tf.keras')
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import PIL
from IPython.display import clear_output
import glob
import os
import cv2
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import pyramid_reduce, resize

import tensorflow as tf
from tensorflow.keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from tensorflow.keras.layers import Add, LeakyReLU, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models import PSPNet
from segmentation_models import get_preprocessing
from segmentation_models.losses import JaccardLoss,DiceLoss
from segmentation_models.metrics import IOUScore

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
tf.keras.backend.clear_session()


# ## **Load the dataset**
# 

# In[49]:


img_list = sorted(glob.glob('ISIC/Train/images/BCC/*.jpg'))
mask_list = sorted(glob.glob('ISIC/Train/masks/BCC/*.png'))

print(len(img_list), len(mask_list))



# In[50]:


IMG_SIZE = 128

x_data, y_data = np.empty((2, len(img_list), IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)

for i, img_path in enumerate(img_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    x_data[i] = img
    
for i, img_path in enumerate(mask_list):
    img = imread(img_path)
    img = resize(img, output_shape=(IMG_SIZE, IMG_SIZE, 1), preserve_range=True)
    y_data[i] = img
    
y_data /= 255.

fig, ax = plt.subplots(1, 2)
ax[0].imshow(x_data[12].squeeze(), cmap='gray')
ax[1].imshow(y_data[12].squeeze(), cmap='gray')


# In[51]:


#Split data
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1)

print(x_train.shape, y_train.shape)
print(x_val.shape, y_val.shape)


# In[54]:


#Model
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

net = Conv2D(32, kernel_size=3, activation='relu', padding='same')(inputs)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(64, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Conv2D(128, kernel_size=3, activation='relu', padding='same')(net)
net = MaxPooling2D(pool_size=2, padding='same')(net)

net = Dense(128, activation='relu')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(128, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
net = Conv2D(64, kernel_size=3, activation='sigmoid', padding='same')(net)

net = UpSampling2D(size=2)(net)
outputs = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(net)

model = Model(inputs=inputs, outputs=outputs)



# In[56]:


jloss = JaccardLoss()
iou = IOUScore()

model.compile(optimizer='adam', loss=['binary_crossentropy',jloss], metrics=['acc', 'mse',iou])
model.summary()


# In[ ]:


nepochs=100
nbatch_size=64
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=nepochs, batch_size=nbatch_size, 
    callbacks=[ ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)],
                     shuffle=True, max_queue_size=10, workers=1, use_multiprocessing=True,
                   )


# ## **Evaluation**

# In[59]:


fig, ax = plt.subplots(2, 2, figsize=(10, 7))

ax[0, 0].set_title('loss')
ax[0, 0].plot(history.history['loss'], 'r')
ax[0, 1].set_title('acc')
ax[0, 1].plot(history.history['acc'], 'b')

ax[1, 0].set_title('val_loss')
ax[1, 0].plot(history.history['val_loss'], 'r--')
ax[1, 1].set_title('val_acc')
ax[1, 1].plot(history.history['val_acc'], 'b--')


# In[60]:


#Predicted masks
preds = model.predict(x_val)

fig, ax = plt.subplots(len(x_val), 3, figsize=(10, 100))

for i, pred in enumerate(preds):
    ax[i, 0].imshow(x_val[i].squeeze(), cmap='gray')
    ax[i, 1].imshow(y_val[i].squeeze(), cmap='gray')
    ax[i, 2].imshow(pred.squeeze(), cmap='gray')

plt.show()


# In[ ]:




