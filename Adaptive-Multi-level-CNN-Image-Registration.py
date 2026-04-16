#!/usr/bin/env python
# coding: utf-8

# ## **Adaptive multi-level network for deformable registration of 3D brain MR images**

# In[3]:


#Import the necessary libraries:
get_ipython().run_line_magic('matplotlib', 'inline')
import random
import cv2
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

K.clear_session()


# ## **Load Data and Display**

# In[8]:


H,W,CH=[128,128,3]
higher = 'SAR/high/'
lower = 'SAR/low/'

clean = []
for file in tqdm(sorted(os.listdir(higher))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(higher + '/' + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)

clean = np.array(clean)

blurry = []
for file in tqdm(sorted(os.listdir(lower))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(lower + '/' + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)

blurry = np.array(blurry)


# In[9]:


x = clean
y = blurry
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# In[10]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in range(9):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Build the adaptive multi-level network model**

# In[20]:


def AMCNN():
    # Define the input layers
    inputs = layers.Input(shape=(H, W, CH))

    # Define the downsampling path
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Define the upsampling path
    up1 = layers.UpSampling2D(size=(2, 2))(pool2)
    up1 = layers.Conv2D(64, 2, activation='relu', padding='same')(up1)
    concat1 = layers.Concatenate()([conv2, up1])
    conv3 = layers.Conv2D(64, 3, activation='relu', padding='same')(concat1)
    up2 = layers.UpSampling2D(size=(2, 2))(conv3)
    up2 = layers.Conv2D(32, 2, activation='relu', padding='same')(up2)
    concat2 = layers.Concatenate()([conv1, up2])

    # Output layer
    outputs = layers.Conv2D(3, 1, activation='linear')(concat2)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Build the model
model = AMCNN()


# In[21]:


def loss_function(y_true, y_pred):
    # Define the loss function (e.g., mean squared error)
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return loss

# Compile the model
model.compile(optimizer='adam', loss=["mse",loss_function],metrics=["acc"])
model.summary()


# In[22]:


tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)


# ## **Model train**

# In[23]:


nepochs=50
nbatch_size=32
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[27]:


#Plot history loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 101, 25))
plt.show()

#Plot history Accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Performence Evaluation**

# In[28]:


#Model performance evaluation

print("Input----------------Ground Truth-----------------Predicted Value")

for i in (range(6)):    
    r = random.randint(0, len(clean)-1)
    x, y = blurry[r],clean[r]
    x_inp=x.reshape(1,H,W,CH)
    result = model.predict(x_inp)
    result = result.reshape(H,W,CH)

    fig = plt.figure(figsize=(12,10))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x)

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y)
    
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(result)
    
plt.grid('off')    
plt.show()
print("--------------Done!----------------")


# In[ ]:




