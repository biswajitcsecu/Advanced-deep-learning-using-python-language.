#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D, Input, UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


# In[29]:


# Load the images
img1 = cv2.imread('image1.png', 0) 
img1 = cv2.resize(img1, (256, 256)) / 255.
img1 = img1.astype('float32')
img2 = cv2.imread('image2.png', 0) / 255.
img2 = img2.astype('float32') 
img2 = cv2.resize(img2, (256, 256))

# Combine the images into a single tensor
img_tensor = np.stack((img1, img2), axis=2)


# In[24]:


def cnn_model():
    inputs = Input(shape=(None, None, 2))
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)

    up4 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    outputs = Conv2D(2, 1, activation='sigmoid')(conv5)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model


# In[25]:


# Define the model architecture
def create_model():
    inputs = Input(shape=(None, None, 2))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    up4 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    merge4 = Concatenate()([conv2, conv4])
    up5 = UpSampling2D(size=(2, 2))(merge4)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    merge5 = Concatenate()([conv1, conv5])
    conv6 = Conv2D(2, (3, 3), activation='relu', padding='same')(merge5)
    model = Model(inputs=inputs, outputs=conv6)
    return model


# In[ ]:


# Define the loss function
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=255))


# In[ ]:


# Define the loss function
def custom_loss(y_true, y_pred):
    mse_loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    ssim_loss = ssim_loss(y_true, y_pred)
    return (mse_loss - ssim_loss)


# In[40]:


# Create the model
#model = create_model()
model = cnn_model()
model.summary()


# In[43]:


# Compile the model
model.compile(optimizer='adam', loss=[ssim_loss],metrics=["accuracy"],)

tf.keras.utils.plot_model(model, show_shapes=True)


# In[60]:


# Train the model
nepochs=200
nbatch_size=1
history=model.fit(np.expand_dims(img_tensor, axis=0), np.expand_dims(img_tensor, axis=0), epochs=nepochs,
                 batch_size=nbatch_size,verbose=1,shuffle=True,max_queue_size=4,workers=1,use_multiprocessing=False,                 
                 )


# In[63]:


nbatch_size=1
def plotLearningCurve(history,epochnum,batchnum):
    epochRange = range(1,epochnum+1) 
    plt.figure(figsize = (5,5))
    plt.plot(epochRange,history.history['loss'],'b',label = 'Training Loss')
    plt.plot(epochRange,history.history['accuracy'],'r',label = 'accuracy')
    plt.xlabel('Epoch', fontsize = 15)
    plt.ylabel('Loss', fontsize = 15)
    plt.grid(color='gray', linestyle='--')
    plt.legend()
    plt.title('LOSS, Epochs={}, Batch={}'.format(epochnum, batchnum))
    plt.axis('off')
    plt.show()

plotLearningCurve(history,nepochs,nbatch_size)


# In[64]:


# Predict the fused image
fused_image = model.predict(np.expand_dims(img_tensor, axis=0))[0, :, :, 0]

# Save the fused image
cv2.imwrite('fused_image.png', fused_image * 255.0)
fused_image =(fused_image * 255.0)


# In[65]:


# Display the results
fig, ax= plt.subplots(nrows=1, ncols=2,figsize=(12, 10))
plt.subplot(131),plt.imshow(img1,cmap = 'gray'),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(img2,cmap = 'gray'),plt.title('Mask')
plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(fused_image,cmap = 'gray'),plt.title('Segmented')
plt.xticks([]), plt.yticks([])
plt.show()


# In[ ]:




