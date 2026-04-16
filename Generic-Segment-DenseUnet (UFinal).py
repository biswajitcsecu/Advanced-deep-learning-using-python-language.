#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


#path of images and mask directories
image_path='Kvasir/train/images/'
mask_path='Kvasir/train/masks/'

#load paths of images and masks files
image_paths=[i for i in glob(image_path+'*.jpg')]
mask_paths =[i for i in glob(mask_path+'*.jpg')]

H,W,CH=[128,128,3]
#Create lists of image and mask paths
no_of_images, no_of_masks = len(image_paths), len(mask_paths)

print(f"1. There are {no_of_images} images and {no_of_masks} masks in our dataset")
print(f"2. An example of an image path is: \n {image_paths[0]}")
print(f"3. An example of a mask path is: \n {mask_paths[0]}")


# ## **Visulization of  masks and images**

# In[3]:


#display image and  mask
no_of_samples = len(image_paths)

for i in tqdm(range(3)):
    N = random.randint(0, no_of_samples - 1)

    img = imageio.imread(image_paths[N])
    mask = imageio.imread(mask_paths[N])
    mask=np.array([max(mask[i,j])for i in range(mask.shape[0])for j in range(mask.shape[1])]).reshape(img.shape[0],img.shape[1])
    fig, arr = plt.subplots(1, 3, figsize=(20, 8))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[0].axis("off")
    arr[1].imshow(mask)
    arr[1].set_title('Mask')
    arr[1].axis("off")    
    arr[2].imshow(mask, cmap='Paired')
    arr[2].set_title('Image Overlay')
    arr[2].axis("off")
    
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Spliting images and masks into training, validation, and test sets**

# In[4]:


# First split the image paths into training and validation sets
train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(image_paths, mask_paths,
                                                                                        train_size=0.8, random_state=0)

# Keep part of the validation set as test set
validation_image_paths, test_image_paths, validation_mask_paths, test_mask_paths = train_test_split(val_image_paths, 
                           val_mask_paths, train_size = 0.80, random_state=0)

print(f'There are {len(train_image_paths)} images in the Training Set')
print(f'There are {len(validation_image_paths)} images in the Validation Set')
print(f'There are {len(test_image_paths)} images in the Test Set')
print(f'-------------------done-------------------------------')


# ## **Create a data pipeline**

# In[5]:


#Read imagand mask
def read_image(image_path, mask_path):    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (H, W), method='nearest')

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.convert_image_dtype(mask,  tf.float32)
    mask = tf.image.resize(mask, (H, W), method='nearest')
    
    return image, mask


# In[6]:


#Create a data generator function to read and load images and masks in batches

def data_generator(image_paths, mask_paths, buffer_size, batch_size):    
    image_list = tf.constant(image_paths) 
    mask_list = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)
    
    return dataset


# In[7]:


#Create data pipelines for the training, validation and test sets using both functions
nbatch_size = 16
buffer_size = 500

train_dataset = data_generator(train_image_paths, train_mask_paths, buffer_size, nbatch_size)
validation_dataset = data_generator(validation_image_paths, validation_mask_paths, buffer_size, nbatch_size)
test_dataset = data_generator(test_image_paths, test_mask_paths, buffer_size, nbatch_size)


# ## **Model Architecture**

# In[9]:


#DenseUnet

def DenseBlock(channels,inputs):
    conv1_1 = Conv2D(channels, (1, 1),activation=None, padding='same')(inputs)
    conv1_1=BatchActivate(conv1_1)
    conv1_2 = Conv2D(channels//4, (3, 3), activation=None, padding='same')(conv1_1)
    conv1_2 = BatchActivate(conv1_2)

    conv2=concatenate([inputs,conv1_2])
    conv2_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv2)
    conv2_1 = BatchActivate(conv2_1)
    conv2_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv2_1)
    conv2_2 = BatchActivate(conv2_2)

    conv3 = concatenate([inputs, conv1_2,conv2_2])
    conv3_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv3)
    conv3_1 = BatchActivate(conv3_1)
    conv3_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv3_1)
    conv3_2 = BatchActivate(conv3_2)

    conv4 = concatenate([inputs, conv1_2, conv2_2,conv3_2])
    conv4_1 = Conv2D(channels, (1, 1), activation=None, padding='same')(conv4)
    conv4_1 = BatchActivate(conv4_1)
    conv4_2 = Conv2D(channels // 4, (3, 3), activation=None, padding='same')(conv4_1)
    conv4_2 = BatchActivate(conv4_2)
    result=concatenate([inputs,conv1_2, conv2_2,conv3_2,conv4_2])
    return result

def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def DenseUNet(input_shape):
    filters=16
    keep_prob=0.9
    block_size=7

    inputs = Input(shape=input_shape)

    conv1 = Conv2D(filters * 1, (3, 3), activation=None, padding="same")(inputs)
    conv1 = BatchActivate(conv1)
    conv1 = DenseBlock(filters * 1, conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = DenseBlock(filters * 2, pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = DenseBlock(filters * 4, pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)

    convm = DenseBlock(filters * 8, pool3)

    deconv3 = Conv2DTranspose(filters * 4, (3, 3), strides=(2, 2), padding="same")(convm)
    uconv3 = concatenate([deconv3, conv3])
    uconv3 = Conv2D(filters * 4, (1, 1), activation=None, padding="same")(uconv3)
    uconv3 = BatchActivate(uconv3)
    uconv3 = DenseBlock(filters * 4, uconv3)

    deconv2 = Conv2DTranspose(filters * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = concatenate([deconv2, conv2])
    uconv2 = Conv2D(filters * 2, (1, 1), activation=None, padding="same")(uconv2)
    uconv2 = BatchActivate(uconv2)
    uconv2 = DenseBlock(filters * 2, uconv2)

    deconv1 = Conv2DTranspose(filters * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = concatenate([deconv1, conv1])
    uconv1 = Conv2D(filters * 1, (1, 1), activation=None, padding="same")(uconv1)
    uconv1 = BatchActivate(uconv1)
    uconv1 = DenseBlock(filters * 1, uconv1)

    outputs = Conv2D(4, (1, 1), padding="same", activation=None)(uconv1)
    outputs = Activation('sigmoid')(outputs)

    model = Model(inputs=inputs, outputs=outputs)
    
    return model


# ## **Buildup Model**

# In[10]:


H, W, CH = [128,128,3]
input_shape=(H, W, CH)
nClasses=6
model = DenseUNet(input_shape)


# ## **Loss functions**

# In[11]:


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

# In[12]:


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

# In[13]:


opt = Adam(learning_rate=0.0001)
loss1=[perception_loss,'sparse_categorical_crossentropy',dice_loss]
metric1 = ['accuracy',iou,jacard_coef]
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#binary_crossentropy,sparse_categorical_crossentropy,categorical_crossentropy
model.summary()


# In[14]:


tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)


# ## **Train Model**

# In[17]:


#train parameters
callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=1e-1, patience=5, verbose=1, min_lr = 2e-6)
nepochs = 20
nbatch_size = 8


# In[18]:


#train
history = model.fit(train_dataset,validation_data = validation_dataset, 
                    epochs = nepochs,verbose=1,callbacks = [callback, reduce_lr], 
                    batch_size = nbatch_size,shuffle = True,
                    max_queue_size=10,workers=1,use_multiprocessing=True,
                   )


# # **Model performances**

# In[20]:


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

# In[22]:


train_loss, train_accuracy = model.evaluate(train_dataset, batch_size = 32)
validation_loss, validation_accuracy = model.evaluate(validation_dataset, batch_size = 32)
test_loss, test_accuracy = model.evaluate(test_dataset, batch_size = 32)

#-------------------------------------------------------------------------------# 
print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')
print(f'-------------------------------Done-------------------------------------')


# ## **Model Evaluation**

# In[23]:


#Evaluate Predicted Segmentations
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)    
    return pred_mask[0]


# In[29]:


#display: an input image, its true mask, and its predicted mask

def display(display_list):
    plt.figure(figsize=(16, 16))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='Paired')
        plt.axis('off')
    plt.show()


# In[30]:


#show predictions results

def show_predictions(dataset, num):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,create_mask(model.predict(sample_image[tf.newaxis, ...]))])
        


# In[31]:


#Predict and compare masks of images in the training set
show_predictions(train_dataset, 8)


# In[32]:


#Predict and compare masks of images in the validation set
show_predictions(validation_dataset,9)


# In[33]:


#Predict and compare masks of images in the test set
show_predictions(test_dataset, 26)


# ## **Model evaluation**

# In[35]:


print("Input ----------------------------Ground Truth-------------------------------------Predicted Value")
for idx in tqdm(range(6)): 
    for image, mask in test_dataset.take(idx):
        pred_mask = model.predict(image)
        x,y =image[0], mask[0],
        pred_mask = model.predict(image)
        result =  create_mask(pred_mask)        
        #Plot result
        fig = plt.figure(figsize=(12,10))
        fig.subplots_adjust(hspace=0.1, wspace=0.2)
        #Image
        ax = fig.add_subplot(1, 3, 1)
        plt.axis("off")
        ax.imshow(x, cmap='Paired')
        #Mask
        ax = fig.add_subplot(1, 3, 2)
        plt.axis("off")
        ax.imshow(y, cmap='Paired')
        #result
        ax = fig.add_subplot(1, 3, 3)
        plt.axis("off")
        plt.imshow(result, cmap='Paired') 
        plt.grid('off')    
        plt.show()

print("--------------Done!----------------")


# In[ ]:




