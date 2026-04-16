#!/usr/bin/env python
# coding: utf-8

# ## **Similarity attention-based CNN for image super resolution**

# In[ ]:


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
from tensorflow.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

K.clear_session()


# ## **Load Data and Display**

# In[4]:


H,W,CH=[128,128,3]
higher = 'SAR/high/'
lower = 'SAR/low/'


# In[5]:


clean = []
for file in tqdm(sorted(os.listdir(higher))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(good + '/' + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)

clean = np.array(clean)
blurry = []
for file in tqdm(sorted(os.listdir(lower))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(bad + '/' + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)

blurry = np.array(blurry)


# In[6]:


x = clean
y = blurry
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# In[11]:


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


# ## **Similarity Attention-CNN Model Creation**

# In[30]:


# Define the model architecture
def similarity_attention_cnn():
    # Input layer
    input_shape=(H, W, CH)
    inputs = tf.keras.Input(shape=input_shape)

    # Downsampling layer
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(inputs)

    # Residual blocks
    for _ in range(8):
        residual = x
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Add()([x, residual])

    # Upsampling layer
    x = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=2, padding='same', activation='relu')(x)

    # Similarity attention layer
    similarity = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same', activation='sigmoid')(x)
    features = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same')(x)
    x = tf.keras.layers.Multiply()([features, similarity])
    x = tf.keras.layers.Add()([x, inputs])

    # Output layer
    outputs = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same')(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)

    return model


# In[32]:


# Adjust the shape according to your image dimensions
model = similarity_attention_cnn()
model.summary()


# In[40]:


tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)


# In[41]:


# Define the loss function
def loss_function(y_true, y_pred):    
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return loss


# ## **Model compilation and train**

# In[33]:


model.compile(loss='mse', optimizer='adam',metrics=["acc"])


# In[34]:


nepochs=50
nbatch_size=32
#history = model.fit(blurry,clean,validation_data=(blurry, clean),epochs=nepochs,batch_size=nbatch_size,verbose=1,
#,shuffle=True,max_queue_size=10,workers=1,use_multiprocessing=True )

history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[42]:


#Plot history loss
#plt.style.use('seaborn')
plt.figure(figsize=(10,6))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# In[43]:


#Plot history Accuracy
#plt.style.use('seaborn')
plt.figure(figsize=(10,6))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Performance Evaluation**

# In[44]:


print("Input --------------------Ground Truth------------------------- Predicted Value")

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




