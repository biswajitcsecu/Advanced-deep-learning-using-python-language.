#!/usr/bin/env python
# coding: utf-8

# ## **Similarity attention-based CNN for image registration**

# In[99]:


get_ipython().run_line_magic('matplotlib', 'inline')
import random
import cv2
import os
import numpy as np
import pandas as pd
from numpy.random import randint
from tqdm.notebook import tqdm, tnrange,trange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Add, Activation,add,LayerNormalization,AvgPool2D
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D, Lambda
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D,  Multiply
from tensorflow.keras.layers import Input, Reshape, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback ,ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D,  MaxPool2D, UpSampling2D, concatenate, Activation
from tensorflow.keras.layers import Layer, Reshape, Conv2DTranspose, Multiply, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

K.clear_session()


# ## **Load Data and Display**

# In[73]:


H,W,CH=[128,128,1]
good = 'MSU/train/high/'
bad = 'MSU/train/low/'


# In[62]:


clean = []
for file in tqdm(sorted(os.listdir(good))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(good + '/' + file, target_size=(H,W),color_mode='grayscale')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)

clean = np.array(clean)
blurry = []
for file in tqdm(sorted(os.listdir(bad))):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(bad + '/' + file, target_size=(H,W),color_mode='grayscale')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)

blurry = np.array(blurry)


# In[63]:


x = clean
y = blurry

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(y_train.shape)


# In[64]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[65]:


figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(y_train[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **ImageDataGenerator**

# In[66]:


# Create an instance of the ImageDataGenerator for data augmentation
data_gen = ImageDataGenerator(
    rotation_range=10,   
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    zoom_range=0.1,  
    horizontal_flip=True    
)

data_gen.fit(x_train)
train_generator = data_gen.flow(x_train, y_train, batch_size=32)

# Visualize a few images from the generator
images, labels = train_generator.next()
num_images = len(images)
num_rows = 4
num_cols = num_images // num_rows

# Create a figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
if num_rows > 1:
    axes = axes.flatten()
# Iterate over the images and labels
for i, (image, label) in enumerate(zip(images, labels)):
    ax = axes[i]    
    ax.imshow(image,cmap='gray')
    ax.axis('off')    
    label_idx = label.argmax()
    ax.set_title(f"Label: {label_idx}")

plt.tight_layout()
plt.show()


# ## **CNN Model Creation**

# In[ ]:


def build_unet_model(input_shape):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Decoder
    up4 = UpSampling2D(size=(2, 2))(pool3)
    up4 = Conv2D(128, 2, activation='relu', padding='same')(up4)
    merge4 = concatenate([conv3, up4], axis=3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(merge4)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv4)

    up5 = UpSampling2D(size=(2, 2))(conv4)
    up5 = Conv2D(64, 2, activation='relu', padding='same')(up5)
    merge5 = concatenate([conv2, up5], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(merge5)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(conv5)

    up6 = UpSampling2D(size=(2, 2))(conv5)
    up6 = Conv2D(32, 2, activation='relu', padding='same')(up6)
    merge6 = concatenate([conv1, up6], axis=3)
    conv6 = Conv2D(32, 3, activation='relu', padding='same')(merge6)
    conv6 = Conv2D(32, 3, activation='relu', padding='same')(conv6)

    # Output
    output = Conv2D(1, 1, activation='sigmoid')(conv6)

    model = Model(inputs=inputs, outputs=output)
    return model


# In[75]:


input_shape = (H, W, CH) 
model = build_unet_model(input_shape)


# ## **Loss function**

# In[ ]:


# Define loss function
def pixel_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# ## **Model compilation and train**

# In[76]:


K.clear_session()
model.compile(optimizer='adam',loss=[pixel_loss,'mse'], metrics=["acc"])
model.summary()


# In[77]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True,dpi=75)


# ## **Model Callback**

# In[87]:


#Specify Callback
def show_image(image, title=None):
    plt.imshow(image,cmap='gray')
    plt.title(title)
    plt.axis('off')

class ShowProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(x_train))
        himage = x_train[id]
        limage = y_train[id]
        pred_img = self.model(tf.expand_dims(himage,axis=0))[0]
        
        plt.figure(figsize=(10,8))
        plt.subplot(1,3,1)
        show_image(himage, title="High Image")
        
        plt.subplot(1,3,2)
        show_image(limage, title="Low Image")
        
        plt.subplot(1,3,3)
        show_image(pred_img, title="Predicted Image")
            
        plt.tight_layout()
        plt.show()


# ## **Model Train**

# In[80]:


nepochs=1
nbatch_size=16
cbs = [ShowProgress()]
SPE=len(x_train)//nbatch_size

history = model.fit(train_generator, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True, steps_per_epoch=SPE,callbacks=cbs,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# ## **Performance evaluation**

# In[92]:


#Plot history loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# In[93]:


#Plot history Accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['Train', 'Test'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.xticks(np.arange(0, 101, 25))
plt.show()


# ## **Model Evaluation**

# In[94]:


print("\n\n Input --------------------\n Ground Truth\n-------------------------\n Predicted Value")

for i in tqdm(range(6)):    
    r = random.randint(0, len(clean)-1)
    x, y = blurry[r],clean[r]
    x_inp=x.reshape(1,H,W,CH)
    result = model.predict(x_inp)
    result = result.reshape(H,W,CH)

    fig = plt.figure(figsize=(12,10))
    fig.subplots_adjust(hspace=0.1, wspace=0.2)
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x,cmap='gray')

    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y,cmap='gray')
    
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(result,cmap='gray')
    
plt.grid('off')    
plt.show()
print("--------------Done!----------------")


# ## **Model predictions**

# In[95]:


figure, axes = plt.subplots(3,3, figsize=(20,20))
for i in tqdm(range(0,3)):
    rand_num = random.randint(0,50)
    original_img = x_test[rand_num]
    axes[i,0].imshow(original_img)
    axes[i,0].title.set_text('Image')
    
    original_mask = y_test[rand_num]
    axes[i,1].imshow(original_mask)
    axes[i,1].title.set_text('Image')
    
    original_img = np.expand_dims(original_img, axis=0)
    predicted_mask = model.predict(original_img).reshape(H,W)
    axes[i,2].imshow(predicted_mask)
    axes[i,2].title.set_text('Predicted Image')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Model Evaluation**

# In[96]:


# predict test images
predict_y = model.predict(x_test)

plt.figure(figsize=(15,15))
for i in  tqdm(range(0,9,3)):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.title('High image')
    
    plt.subplot(4,3,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_test[i])
    plt.title('Lowimage')
    
    plt.subplot(4,3,i+3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i])
    plt.title('Output by model')

plt.show()


# ## **Model Evaluation**

# In[98]:


#Display
def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

for i in tqdm(range(20)):
    idx = randint(len(x_test))
    image = x_test[idx]
    mask = y_test[idx]
    pred_mask = model.predict(tf.expand_dims(image,axis=0))[0]
        
    plt.figure(figsize=(10,8))
    plt.subplot(1,4,1)
    show_image(image, title="Higher Image")
        
    plt.subplot(1,4,2)
    show_image(mask, title="Lower Image")
        
    plt.subplot(1,4,3)
    show_image(pred_mask, title="Predicted Image")  
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  
    



# ## **------------------------------------Done----------------------------------------**

# In[ ]:




