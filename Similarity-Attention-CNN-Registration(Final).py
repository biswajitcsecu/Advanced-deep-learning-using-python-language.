#!/usr/bin/env python
# coding: utf-8

# ## **Similarity attention-based CNN for image registration**

# In[56]:


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

# In[ ]:


H,W,CH=[128,128,3]
good = 'SAR/high/'
bad = 'SAR/low/'


# In[ ]:


clean = []
for file in tqdm(sorted(os.listdir(good))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(good + '/' + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)

clean = np.array(clean)
blurry = []
for file in tqdm(sorted(os.listdir(bad))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(bad + '/' + file, target_size=(H,W))
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)

blurry = np.array(blurry)


# In[ ]:


x = clean
y = blurry
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train[0].shape)
print(y_train[0].shape)


# In[ ]:


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

# In[ ]:


def similarity_attention_cnn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    
    # Encoder
    x = layers.Conv2D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)
    encoded = layers.Dense(256, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(8 * 8 * 256, activation='relu')(encoded)
    x = layers.Reshape((8, 8, 256))(x)
    x = layers.Conv2DTranspose(128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=2)(x)
    x = layers.Conv2DTranspose(64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D(size=4)(x)
    x = layers.Conv2DTranspose(3, kernel_size=3, activation='sigmoid', padding='same')(x)
    decoded = layers.UpSampling2D(size=2)(x)
    
    # Attention mechanism
    attention = layers.Attention()([decoded, inputs])
    
    # Concatenate attention and decoded features
    concatenated = layers.Concatenate()([attention, decoded])
    
    # Final prediction
    output = layers.Conv2D(3, kernel_size=3, activation='sigmoid', padding='same')(concatenated)
    # Model creation
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model


# In[ ]:


# Example usage
input_shape = (H, W, CH) 
# Adjust the shape according to your image dimensions
model = similarity_attention_cnn(input_shape)
model.summary()


# ## **Model compilation and train**

# In[43]:


model.compile(loss='mse', optimizer='adam',metrics=["acc"])


# In[46]:


nepochs=50
nbatch_size=32
#history = model.fit(blurry,clean,validation_data=(blurry, clean),epochs=nepochs,batch_size=nbatch_size,verbose=1,
#,shuffle=True,max_queue_size=10,workers=1,use_multiprocessing=True )

history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[52]:


#Plot history loss
plt.style.use('seaborn')
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# In[53]:


#Plot history Accuracy
plt.style.use('seaborn')
plt.figure(figsize=(12,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Evaluation**

# In[61]:


print("\n\n Input --------------------\n Ground Truth\n-------------------------\n Predicted Value")

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




