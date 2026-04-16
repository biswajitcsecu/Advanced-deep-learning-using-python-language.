#!/usr/bin/env python
# coding: utf-8

# In[200]:


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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Activation, Input, Conv2D, MaxPooling2D 
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model, load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
import warnings


K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data Preparation**

# In[203]:


#path of images and mask directories
image_path='Kvasir/train/images/'
mask_path='Kvasir/train/masks/'

#load paths of images and masks files
image_paths=[i for i in glob(image_path+'*.jpg')]
mask_paths =[i for i in glob(mask_path+'*.jpg')]

#Create lists of image and mask paths
number_of_images, number_of_masks = len(image_paths), len(mask_paths)

print(f"1. There are {number_of_images} images and {number_of_masks} masks in our dataset")
print(f"2. An example of an image path is: \n {image_paths[0]}")
print(f"3. An example of a mask path is: \n {mask_paths[0]}")


# ## **Preview random masked and unmasked images**

# In[204]:


#display image and  mask
number_of_samples = len(image_paths)

for i in tqdm(range(3)):
    N = random.randint(0, number_of_samples - 1)

    img = imageio.imread(image_paths[N])
    mask = imageio.imread(mask_paths[N])
    mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])
    fig, arr = plt.subplots(1, 3, figsize=(20, 8))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[0].axis("off")
    arr[1].imshow(mask)
    arr[1].set_title('Segmentation')
    arr[1].axis("off")    
    arr[2].imshow(mask, cmap='Paired')
    arr[2].set_title('Image Overlay')
    arr[2].axis("off")
plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Split the images and masks into training, validation, and test sets**

# In[205]:


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


# ## **Create a data pipeline to read and preprocess data**

# In[206]:


def read_image(image_path, mask_path):
    
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    #image = tf.image.convert_image_dtype(image, tf.float32)/255.
    image = tf.image.resize(image, (128, 128), method='nearest')

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.image.convert_image_dtype(mask,  tf.float32)
    mask = tf.image.resize(mask, (128, 128), method='nearest')
    
    return image, mask


# In[207]:


#Create a data generator function to read and load images and masks in batches
def data_generator(image_paths, mask_paths, buffer_size, batch_size):
    
    image_list = tf.constant(image_paths) 
    mask_list = tf.constant(mask_paths)
    dataset = tf.data.Dataset.from_tensor_slices((image_list, mask_list))
    dataset = dataset.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache().shuffle(buffer_size).batch(batch_size)
    
    return dataset


# In[208]:


#Create data pipelines for the training, validation and test sets using both functions

nbatch_size = 32
buffer_size = 500

train_dataset = data_generator(train_image_paths, train_mask_paths, buffer_size, nbatch_size)
validation_dataset = data_generator(validation_image_paths, validation_mask_paths, buffer_size, nbatch_size)
test_dataset = data_generator(test_image_paths, test_mask_paths, buffer_size, nbatch_size)


# ## **Model Architecture and Training**

# In[229]:


#encoding blocks
def encoding_block(inputs, filters, dropout_probability=0, max_pooling=True):    
    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(inputs)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)

    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(C)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)

    skip_connection = C  
    
    if max_pooling:
        next_layer = MaxPooling2D(pool_size=(2, 2))(C)        
    else:
        next_layer = C
            
    return next_layer, skip_connection


# In[230]:


#decoding blocks
def decoding_block(inputs, skip_connection_input, filters):
    CT = Conv2DTranspose(filters, 3, strides=(2,2), padding="same", kernel_initializer="he_normal")(inputs)
    
    residual_connection = concatenate([CT, skip_connection_input], axis=3)

    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(residual_connection)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)
    
    C = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(C)
    C = BatchNormalization()(C)
    C = Activation("relu")(C)
    
    return C


# In[231]:


#Unet model

def unet_model(input_size, filters, n_classes):
    inputs = Input(input_size)
        
    # Contracting Path (encoding)
    C1, S1 = encoding_block(inputs, filters, max_pooling=True)
    C2, S2 = encoding_block(C1, filters * 2, max_pooling=True)
    C3, S3 = encoding_block(C2, filters * 4, max_pooling=True)
    C4, S4 = encoding_block(C3, filters * 8, max_pooling=True)
    
    C5, S5 = encoding_block(C4, filters * 16, max_pooling=False)
    
    # Expanding Path (decoding)
    U6 = decoding_block(C5, S4, filters * 8)
    U7 = decoding_block(U6, S3,  filters * 4)
    U8 = decoding_block(U7, S2,  filters = filters * 2)
    U9 = decoding_block(U8, S1,  filters = filters)

    C10 = Conv2D(filters,3,activation='relu',padding='same',kernel_initializer='he_normal')(U9)

    C11 = Conv2D(filters = n_classes, kernel_size = (1,1), activation='softmax', padding='same')(C10) #sigmoid,softmax
    
    model = Model(inputs=inputs, outputs=C11)

    return model


# In[232]:


img_height = 128
img_width = 128
num_channels = 3
filters = 32
n_classes = 10

model = unet_model((img_height, img_width, num_channels), filters=32, n_classes=10)
model.summary()


# In[233]:


tf.keras.utils.plot_model(model, 'model.png', show_shapes=True)


# ## **Model Compilation**

# In[234]:


opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) #binary_crossentropy
callback = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=1e-1, patience=5, verbose=1, min_lr = 2e-6)
nepochs = 3


# In[235]:


#train
history = model.fit(train_dataset,validation_data = validation_dataset, 
                    epochs = nepochs,verbose=1, 
                    callbacks = [callback, reduce_lr], 
                    batch_size = nbatch_size,shuffle = True)


# # **Model performances**

# In[245]:


#Model performances
acc = [0.] + history.history['accuracy']
val_acc = [0.] + history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

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

# In[246]:


train_loss, train_accuracy = model.evaluate(train_dataset, batch_size = 32)
validation_loss, validation_accuracy = model.evaluate(validation_dataset, batch_size = 32)
test_loss, test_accuracy = model.evaluate(test_dataset, batch_size = 32)


# In[267]:


print(f'Model Accuracy on the Training Dataset: {round(train_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Validation Dataset: {round(validation_accuracy * 100, 2)}%')
print(f'Model Accuracy on the Test Dataset: {round(test_accuracy * 100, 2)}%')


# ## **Model Evaluation**

# In[261]:


#Evaluate Predicted Segmentations
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)    
    return pred_mask[0]


# In[262]:


#display: an input image, its true mask, and its predicted mask
def display(display_list):
    plt.figure(figsize=(16, 16))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


# In[263]:


def show_predictions(dataset, num):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,create_mask(model.predict(sample_image[tf.newaxis, ...]))])


# In[268]:


#Predict and compare masks of images in the training set

show_predictions(train_dataset, 8)


# In[269]:


#Predict and compare masks of images in the validation set

show_predictions(validation_dataset,9)


# In[270]:


#Predict and compare masks of images in the test set

show_predictions(test_dataset, 26)


# In[ ]:





# In[ ]:




