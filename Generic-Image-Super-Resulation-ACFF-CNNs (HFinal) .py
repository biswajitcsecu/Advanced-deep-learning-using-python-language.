#!/usr/bin/env python
# coding: utf-8

# In[22]:


#Import Required Packages
import numpy as np
import pandas as pd
import imageio
import random
import os
import cv2
from glob import glob
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Input, Conv2D, MaxPooling2D,Activation, Add, UpSampling2D 
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, concatenate, Multiply, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import *
from tensorflow.keras.losses import *
from keras import backend as keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.densenet import DenseNet201, DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import warnings


K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data Preparation**

# In[4]:


#path of higher  and lower image directories
high_path='LOL/train/high/'
low_path='LOL/train/low/'

#load paths of images and masks files
high_paths=[i for i in glob(high_path+'*.png')]
low_paths =[i for i in glob(low_path+'*.png')]

H,W,CH=[128,128,3]
#Create lists of image and mask paths
no_of_high_img, no_of_low_img = len(high_paths), len(low_paths)

print(f"2. An example of a high image path is: \n {high_paths[0]}")
print(f"3. An example of a low path is: \n {low_paths[0]}")


# ## **Visulization of images**

# In[5]:


#display image and  mask
nsamples = len(high_paths)

for i in tqdm(range(3)):
    N = random.randint(0, nsamples - 1)

    imgh = imageio.imread(high_paths[N])
    imgl = imageio.imread(low_paths[N])
    fig, arr = plt.subplots(1, 2, figsize=(12, 8))
    arr[0].imshow(imgh)
    arr[0].set_title('Higher Image')
    arr[0].axis("off")
    arr[1].imshow(imgl)
    arr[1].set_title('Lower Image')
    arr[1].axis("off")        
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Spliting images and masks into training, validation, and test sets**

# In[6]:


# First split the image paths into training and validation sets
train_himage_paths, val_himage_paths, train_limage_paths, val_limage_paths = train_test_split(high_paths, low_paths,
                                                                                        train_size=0.8, random_state=0)

# Keep part of the validation set as test set
validation_himage_paths, test_himage_paths, validation_limage_paths, test_limage_paths = train_test_split(val_himage_paths, 
                           val_limage_paths, train_size = 0.80, random_state=0)

print(f'There are {len(train_himage_paths)} images in the Training Set')
print(f'There are {len(validation_himage_paths)} images in the Validation Set')
print(f'There are {len(test_himage_paths)} images in the Test Set')
print(f'-------------------done-------------------------------')


# ## **Create a data pipeline**

# In[7]:


#Read imagand mask
def read_image(high_path, low_path):    
    himage = tf.io.read_file(high_path)
    himage = tf.image.decode_png(himage, channels=3)
    himage = tf.image.resize(himage, (H, W), method='nearest')
    himage = tf.image.convert_image_dtype(himage, tf.float32)
    
    limage = tf.io.read_file(low_path)
    limage = tf.image.decode_png(limage, channels=3)
    limage = tf.image.resize(limage, (H, W), method='nearest')
    limage = tf.image.convert_image_dtype(limage,  tf.float32)

    
    return himage, limage


# In[8]:


#Create a data generator function to read and load images and masks in batches

def data_generator(high_paths, low_paths, buffer_size, batch_size):    
    high_list = tf.constant(high_paths) 
    low_list = tf.constant(low_paths)
    dataset = tf.data.Dataset.from_tensor_slices((high_list, low_list))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)
    
    return dataset


# In[9]:


#Create data pipelines for the training, validation and test sets using both functions
nbatch_size = 16
buffer_size = 500

train_dataset = data_generator(train_himage_paths, train_limage_paths, buffer_size, nbatch_size)
validation_dataset = data_generator(validation_himage_paths, validation_limage_paths, buffer_size, nbatch_size)
test_dataset = data_generator(test_himage_paths, test_limage_paths, buffer_size, nbatch_size)


# ## **Model Architecture**

# In[25]:


SIZE=H
def down(filters , kernel_size, apply_batch_normalization = True):
    downsample = tf.keras.models.Sequential()
    downsample.add(layers.Conv2D(filters,kernel_size,padding = 'same', strides = 2))
    if apply_batch_normalization:
        downsample.add(layers.BatchNormalization())
    downsample.add( tf.keras.layers.LeakyReLU())
    return downsample


def up(filters, kernel_size, dropout = False):
    upsample = tf.keras.models.Sequential()
    upsample.add(layers.Conv2DTranspose(filters, kernel_size,padding = 'same', strides = 2))
    if dropout:
        upsample.dropout(0.2)
    upsample.add( tf.keras.layers.LeakyReLU())
    return upsample

def model():
    inputs = layers.Input(shape= [SIZE,SIZE,3])
    d1 = down(128,(3,3),False)(inputs)
    d2 = down(128,(3,3),False)(d1)
    d3 = down(256,(3,3),True)(d2)
    d4 = down(512,(3,3),True)(d3)
    
    d5 = down(512,(3,3),True)(d4)
    #upsampling
    u1 = up(512,(3,3),False)(d5)
    u1 = layers.concatenate([u1,d4])
    u2 = up(256,(3,3),False)(u1)
    u2 = layers.concatenate([u2,d3])
    u3 = up(128,(3,3),False)(u2)
    u3 = layers.concatenate([u3,d2])
    u4 = up(128,(3,3),False)(u3)
    u4 = layers.concatenate([u4,d1])
    u5 = up(3,(3,3),False)(u4)
    u5 = layers.concatenate([u5,inputs])
    output = layers.Conv2D(3,(2,2),strides = 1, padding = 'same')(u5)
    
    return tf.keras.Model(inputs=inputs, outputs=output)


# ## **Buildup Model**

# In[26]:


H, W, CH = [128,128,3]
input_shape=(H, W, CH)
nClasses=3
model = model()


# ## **Loss functions**

# In[27]:


# Define the perception loss
def perception_loss(y_true, y_pred):
    return MeanSquaredError()(tf.image.sobel_edges(y_true), tf.image.sobel_edges(y_pred))

def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


# ## **Metrics**

# In[28]:


#iou metric
smooth =100
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

#jacard_coef
def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


# ## **Model Compilation**

# In[31]:


opt = Adam(learning_rate=0.0001)
loss1=[perception_loss,'sparse_categorical_crossentropy',dice_loss]
metric1 = ['accuracy',iou,jacard_coef]
model.compile(optimizer = opt, loss = 'mse', metrics = ['accuracy'])

#binary_crossentropy,sparse_categorical_crossentropy,categorical_crossentropy
model.summary()


# In[32]:


tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)


# ## **Train Model**

# In[35]:


#train parameters
callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=1e-1, patience=5, verbose=1, min_lr = 2e-6)
nepochs = 25
nbatch_size =48


# In[36]:


#train
history = model.fit(train_dataset,validation_data = validation_dataset, 
                    epochs = nepochs,verbose=1,callbacks = [callback, reduce_lr], 
                    batch_size = nbatch_size,shuffle = True,
                    max_queue_size=10,workers=1,use_multiprocessing=True,
                   )


# # **Model performances**

# In[38]:


#Model performances
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']
#-----------------plots----------------------#
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()


# ## **Compute Model Accuracy**

# In[39]:


train_loss, train_accuracy = model.evaluate(train_dataset, batch_size = 32)
validation_loss, validation_accuracy = model.evaluate(validation_dataset, batch_size = 32)
test_loss, test_accuracy = model.evaluate(test_dataset, batch_size = 32)

#-------------------------------------------------------------------------------# 
print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')
print(f'-------------------------------Done-------------------------------------')


# ## **Model Evaluation**

# In[61]:


#display: an input image, its true low, and its predicted image

def display(display_list):
    plt.figure(figsize=(16, 16))
    title = ['High Image', 'Low Image', 'Predicted Image']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# In[73]:


#Evaluate Predicted enhance
def create_pred_image(pred_image):
        predicted[0] = list(predicted).pop(0)    
        return pred_image[0]


# In[74]:


#show predictions results
def show_predictions(dataset, num):
    if dataset:
        for himage, limage in dataset.take(num):
            pred_image = np.clip(model.predict(himage) ,0.0,1.0)
            display([himage[0], limage[0], create_pred_image(pred_image)])
    else:
        display([sample_himage, sample_limage,create_pred_image(model.predict(sample_himage[tf.newaxis, ...]))])
        


# In[75]:


#Predict and compare low of images in the training set
show_predictions(train_dataset, 8)


# In[76]:


#Predict and compare low of images in the validation set
show_predictions(validation_dataset,9)


# In[77]:


#Predict and compare low of images in the test set
show_predictions(test_dataset, 26)


# ## **Model evaluation**

# In[78]:


print("Input ----------------------------Ground Truth-------------------------------------Predicted Value")
for idx in tqdm(range(6)): 
    for himage, limage in test_dataset.take(idx):
        x,y =himage[0], limage[0],
        pred_image = model.predict(himage)
        result =  create_pred_image(pred_image)        
        #Plot result
        fig = plt.figure(figsize=(12,10))
        fig.subplots_adjust(hspace=0.1, wspace=0.2)
        #Image
        ax = fig.add_subplot(1, 3, 1)
        plt.axis("off")
        ax.set_title('Higher Image')
        ax.imshow(x)
        #Mask
        ax = fig.add_subplot(1, 3, 2)
        plt.axis("off")
        ax.set_title('Lower Image')
        ax.imshow(y)
        #result
        ax = fig.add_subplot(1, 3, 3)
        plt.axis("off")
        plt.imshow(result) 
        ax.set_title('Predicted Image')
        plt.grid('off')    
        plt.show()

print("--------------Done!----------------")


# ## **Prediction Visualization**

# In[83]:


def plot_images(high,low,predicted):
    plt.figure(figsize=(15,15))
    plt.subplot(1,3,1)
    plt.title('High Image', color = 'green', fontsize = 12)
    plt.imshow(high)
    plt.subplot(1,3,2)
    plt.title('Low Image ', color = 'black', fontsize = 12)
    plt.imshow(low)
    plt.subplot(1,3,3)
    plt.title('Predicted Image ', color = 'Red', fontsize = 12)
    plt.imshow(predicted)  
    plt.show()

for idx in tqdm(range(1,10)):
    for himage, limage in test_dataset.take(idx):
        X,Y = himage[0], limage[0],
        pred_image = model.predict(himage)
        result =  create_pred_image(pred_image)
        plot_images(X,Y,result)
        


# In[ ]:





# In[ ]:




