#!/usr/bin/env python
# coding: utf-8

# ## **Infrared Image SR with GANs**

# In[2]:


import random
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
from numpy import linalg as LA
from numpy.random import randint
from tqdm.notebook import tqdm, tnrange,trange
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keract
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.mobilenet import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, Activation, UpSampling2D, Concatenate, Rescaling, GlobalAvgPool2D
from tensorflow.keras.layers import Dense, Input, Add, add, Activation,add,LayerNormalization,AvgPool2D
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D, Lambda
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D,  Multiply
from tensorflow.keras.layers import Input, Reshape, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda, Flatten,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback ,ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D,  MaxPool2D, UpSampling2D, concatenate, Activation
from tensorflow.keras.layers import Layer, Reshape, Conv2DTranspose, Multiply, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K

import warnings

warnings.filterwarnings('ignore')
K.clear_session()


# ## **Data load and splitting**

# In[3]:


batch_size = 12
H,W,CH=[128,128,3]
good = 'IRRS/train/high/'
bad = 'IRRS/train/low/'
dataset_split =2000


# In[4]:


clean = []
for file in tqdm(sorted(os.listdir(good)[0 : dataset_split])):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(good + '/' + file, target_size=(64,64),color_mode='rgb')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)
clean = np.array(clean)

blurry = []
for file in tqdm(sorted(os.listdir(bad)[0 : dataset_split])):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(bad + '/' + file, target_size=(256,256),color_mode='rgb')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)
blurry = np.array(blurry)


# In[5]:


#Slice datasets
x = clean
y = blurry
train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.2)

# Construct tf.data.Dataset object
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(batch_size)

print(train_x.shape)
print(train_y.shape)


#  ## **Visualization the rgb image and gray**

# In[6]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(8,8))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(train_x[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[7]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(8,8))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(train_y[index])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **GAN: Generator**

# In[8]:


def build_generator(scaling_factor=2, feature_maps=64, residual_blocks=16, leaky_alpha=0.2,residual_scalar=0.2):
    input_layer = Input((None, None, 3))
    xIn = Rescaling(scale=1.0/255, offset=0.0)(input_layer)
    
    xIn = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(xIn)
    xIn = LeakyReLU(leaky_alpha)(xIn)
    
    # construct the residual in residual block
    
    x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(xIn)
    x1 = LeakyReLU(leaky_alpha)(x)
    x1 = Add()([xIn, x1])
    
    x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x1)
    x2 = LeakyReLU(leaky_alpha)(x)
    x2 = Add()([x1, x2])
    
    x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x2)
    x3 = LeakyReLU(leaky_alpha)(x)
    x3 = Add()([x2, x3])
    
    x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x3)
    x4 = LeakyReLU(leaky_alpha)(x)
    x4 = Add()([x3, x4])
    x4 = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x4)    
    xSkip = Add()([xIn, x4])
    
    # scale the residual outputs with a scalar between [0,1]
    xSkip = Lambda(lambda x: x * residual_scalar)(xSkip)
    
    # create a number of residual in residual blocks
    for blockId in tqdm(range(residual_blocks-1)):
        x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(xSkip)
        x1 = LeakyReLU(leaky_alpha)(x)
        x1 = Add()([xSkip, x1])
        
        x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x1)
        x2 = LeakyReLU(leaky_alpha)(x)
        x2 = Add()([x1, x2])
        
        x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x2)
        x3 = LeakyReLU(leaky_alpha)(x)
        x3 = Add()([x2, x3])
        
        x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x3)
        x4 = LeakyReLU(leaky_alpha)(x)
        x4 = Add()([x3, x4])
        x4 = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x4)
        
        xSkip = Add()([xSkip, x4])
        xSkip = Lambda(lambda x: x * residual_scalar)(xSkip)
        
    x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(xSkip)
    x = Add()([xIn, x])
    
    # upscale the image with pixel shuffle
    x = Conv2D(filters=feature_maps * (scaling_factor // 2), kernel_size=3,padding="same")(x)
    x = tf.nn.depth_to_space(x, 2)
    x = LeakyReLU(leaky_alpha)(x)
    
    # upscale the image with pixel shuffle
    x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x)
    x = tf.nn.depth_to_space(x, 2)
    x = LeakyReLU(leaky_alpha)(x)
    
    x = Conv2D(filters=3, kernel_size=3, padding="same", activation="tanh")(x)
    output_layer = Rescaling(scale=127.5, offset=127.5)(x)
    generator = Model(inputs=input_layer, outputs=output_layer)
    
    return generator


# ## **Discriminator**

# In[31]:


def build_discriminator(feature_maps= 64, leaky_alpha = 0.2, disc_blocks = 4):
    input_layer = Input((None, None, 3))
    x = Rescaling(scale=1.0/127.5, offset=-1)(input_layer)
    x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x)
    x = LeakyReLU(leaky_alpha)(x)
    x = Conv2D(filters=feature_maps, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(leaky_alpha)(x)
    
    # create a downsample conv kernel config
    downConvConf = {"strides": (2,2),"padding": "same",}
    
    # create a number of discriminator blocks
    for i in tqdm(range(1, disc_blocks)):
        x = Conv2D(filters=feature_maps * (2 ** i), kernel_size=3, **downConvConf)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leaky_alpha)(x)
        
        x = Conv2D(filters=feature_maps * (2 ** i), kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leaky_alpha)(x)

    x = GlobalAvgPool2D()(x)
    x = LeakyReLU(leaky_alpha)(x)
    output_layer = Dense(3, activation="sigmoid")(x)
    discriminator = Model(inputs=input_layer, outputs=output_layer)
    
    return discriminator


# ## **Build gan**

# In[32]:


def build_gan(generator, discriminator):
    input_img = Input(shape=(64, 64, 3))
    generated_img = generator(input_img)
    gan_output = discriminator(generated_img)
    gan = Model(input_img, [generated_img, gan_output])
    return gan


# In[33]:


# Build generator, discriminator, and GAN
generator = build_generator()
discriminator = build_discriminator()
gan = build_gan(generator, discriminator)


# ## **Model summary**

# In[34]:


generator.summary()
tf.keras.utils.plot_model(generator, 'GEN_Model.png', show_shapes=True, dpi=75)


# In[35]:


discriminator.summary()
tf.keras.utils.plot_model(discriminator, 'Dis_Model.png', show_shapes=True, dpi=75)


# In[36]:


gan.summary()
tf.keras.utils.plot_model(gan, 'GAN_Model.png', show_shapes=True, dpi=75)


# ## **Model loss**

# In[37]:


cross_entropy = tf.keras.losses.BinaryCrossentropy()
mse = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) - tf.random.uniform(shape=real_output.shape, maxval=0.1), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output) + tf.random.uniform(shape=fake_output.shape, maxval=0.1), fake_output)
    total_loss = real_loss + fake_loss
    
    return total_loss

def generator_loss(fake_output, real_y):
    real_y = tf.cast(real_y, 'float32')
    
    return mse(fake_output, real_y)


# ## **Compile model**

# In[38]:


# Compile discriminator
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
# Compile GAN
gan.compile(optimizer=Adam(learning_rate=0.0002,beta_1=0.5),loss=['mean_squared_error','binary_crossentropy'],
            loss_weights=[1, 1], metrics=['accuracy'])


# ## **Training**

# In[ ]:


# Training loop
batch_size = 8
epochs = 400

for epoch in tqdm(range(epochs)):
    print('Epoch:', epoch+1)

    # Generate random indices for batch sampling
    idx = np.random.randint(0, train_x.shape[0], batch_size)
    blurred_images = train_x[idx]
    clear_images = train_y[idx]

    # Generate deblurred images
    generated_images = generator.predict(blurred_images)

    # Train discriminator
    discriminator_loss_real = discriminator.train_on_batch(clear_images, np.ones((batch_size,) + discriminator.output_shape[1:]))
    discriminator_loss_generated=discriminator.train_on_batch(generated_images,np.zeros((batch_size,)
                                                                                        +discriminator.output_shape[1:]))
    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_generated)

    # Train generator (GAN)
    generator_loss = gan.train_on_batch(blurred_images, [clear_images, np.ones((batch_size,) + discriminator.output_shape[1:])])

    # Print losses
    print('Discriminator Loss:', discriminator_loss)
    print('Generator Loss:', generator_loss)
    print('<=------------------------------------Done-------------------------------------=>')


# ## **Model predictions**

# In[ ]:


#Results---------
y = generator(test_x).numpy()

for i in tqdm(range(len(test_x))):
    plt.figure(figsize=(8,8))
    or_image = plt.subplot(3,3,1)
    or_image.set_title('Grayscale Input', color='red', fontsize=10)
    plt.axis('off')
    plt.imshow(np.clip( test_x[i],0,1))
    
    in_image = plt.subplot(3,3,2)    
    image = Image.fromarray((y[i] * 255).astype('uint8'))
    image = np.asarray( image)
    in_image.set_title('Colorized Output',  color='green', fontsize=10)
    plt.axis('off')
    plt.imshow(image)
    
    out_image = plt.subplot(3,3,3)
    image = Image.fromarray((test_y[i] * 255).astype('uint8'))
    plt.axis('off')
    out_image.set_title('Ground Truth',  color='blue', fontsize=10)
    plt.imshow(image)    
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  
    


# ## **Model predictions**

# In[ ]:


# Creating predictions on our test set-----------------

predictions = generator(test_x).numpy()


# In[ ]:


# Ploting results for one image----------------

def plot_results_for_one_sample(sample_index):    
    pdimg =predictions[sample_index] 
    fig = plt.figure(figsize=(10,10))
    # Gray image-------------------
    fig.add_subplot(1,3,1)
    plt.title('Low image')
    org=test_x[sample_index]
    np.clip((org),0,1 ).astype('uint8')
    plt.imshow(org, cmap='gray')
    plt.axis('off')
    plt.grid(None)
    
    #RGB image----------
    fig.add_subplot(1,3,2)
    plt.title('High image')
    rgborg= test_y[sample_index]
    np.clip((rgborg),0,1 ).astype('uint8').resize((H, W))
    plt.imshow(rgborg)
    plt.axis('off')
    plt.grid(None)
    
    #Predicted image------------
    fig.add_subplot(1,3,3)
    plt.title('Predicted image')  
    plt.imshow(np.clip(pdimg,0,1))
    plt.axis('off')
    plt.grid(None)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show() 


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(5)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(15)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# ## **Model predictions**

# In[ ]:


# Generate deblurred images from test set
test_images = generator.predict(test_x)

# Plotting example results
n = 10
plt.figure(figsize=(20, 4))
for i in tqdm(range(n)):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(np.clip((test_x[i]),0,1 ))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(np.clip((test_images[i]),0,1 ))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()     


# In[ ]:




