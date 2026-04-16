#!/usr/bin/env python
# coding: utf-8

# ## **Image SR with GANs**

# In[168]:


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

# In[235]:


batch_size = 12
H,W,CH=[128,128,3]
good = 'CUFED/train/high/'
bad = 'CUFED/train/low/'
dataset_split =700


# In[236]:


clean = []
for file in tqdm(sorted(os.listdir(good)[0 : dataset_split])):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(good + '/' + file, target_size=(96,96),color_mode='rgb')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        clean.append(image)
clean = np.array(clean)

blurry = []
for file in tqdm(sorted(os.listdir(bad)[0 : dataset_split])):
    if any(extension in file for extension in ['.jpg', 'jpeg', '.png']):
        image = tf.keras.preprocessing.image.load_img(bad + '/' + file, target_size=(384,384),color_mode='rgb')
        image = tf.keras.preprocessing.image.img_to_array(image).astype('float32') / 255.
        blurry.append(image)
blurry = np.array(blurry)


# In[237]:


#Slice datasets
x = clean
y = blurry
train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.1)

# Construct tf.data.Dataset object
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(batch_size)

print(train_x.shape)
print(train_y.shape)


#  ## **Visualization the rgb image and gray**

# In[238]:


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


# In[239]:


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

# In[308]:


class SubpixelConv2D(Layer):
    def __init__(self, upsampling_factor=2, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % (factor) != 0:
            raise ValueError('Channel ' + str(last_dim) + ' upsampling_factor^2: ' +str(factor) + '.')

    def call(self, inputs, **kwargs):
        return tf.nn.depth_to_space( inputs, self.upsampling_factor )

    def get_config(self):
        config = { 'upsampling_factor': self.upsampling_factor, }
        base_config = super(SubpixelConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [ input_shape[0],input_shape_1,input_shape_2,int(input_shape[3]/factor)]
        
        return tuple( dims )


# In[318]:


def build_generator(input_shape):
    # w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)
    relu= Activation('relu')

    nin= Input(shape= input_shape)
    n= Conv2D(64, (3,3), padding='SAME', activation= 'relu',
                        kernel_initializer='HeNormal')(nin)
    temp= n


    # B residual blocks
    for i in range(3):
        nn= Conv2D(64, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
        nn= BatchNormalization(gamma_initializer= g_init)(nn)
        nn= relu(nn)
        nn= Conv2D(64, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
        nn= BatchNormalization(gamma_initializer= g_init)(nn)

        nn= add([n, nn])
        n= nn

    n= Conv2D(64, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
    n= BatchNormalization(gamma_initializer= g_init)(n)
    n= add([n, temp])
    # B residual blacks end

    n= Conv2D(256, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
    n= SubpixelConv2D(upsampling_factor=2)(n)
    n= relu(n)

    n= Conv2D(256, (3,3), padding='SAME', kernel_initializer='HeNormal')(n)
    n= SubpixelConv2D(upsampling_factor=2)(n)
    n= relu(n)

    nn= Conv2D(3, (1,1), padding='SAME', kernel_initializer='HeNormal', activation= 'tanh')(n)


    generator = Model(inputs=nin, outputs=nn, name="generator")
    
    return generator


# ## **Discriminator**

# In[319]:


def build_discriminator(input_shape):

    g_init= tf.random_normal_initializer(1., 0.02)
    ly_relu= LeakyReLU(alpha= 0.2)
    df_dim = 16

    nin = Input(input_shape)
    n = Conv2D(64, (4, 4), (2, 2), padding='SAME', kernel_initializer='HeNormal')(nin)
    n= ly_relu(n)

    for i in range(2, 6):
        n = Conv2D(df_dim*(2**i),(4, 4), (2, 2), padding='SAME', kernel_initializer='HeNormal')(n)
        n= ly_relu(n)
        n= BatchNormalization(gamma_initializer= g_init)(n)

    n= Conv2D(df_dim*16, (1, 1), (1, 1), padding='SAME', kernel_initializer='HeNormal')(n)
    n= ly_relu(n)
    n= BatchNormalization(gamma_initializer= g_init)(n)

    n= Conv2D(df_dim*8, (1, 1), (1, 1), padding='SAME', kernel_initializer='HeNormal')(n)
    n= BatchNormalization(gamma_initializer= g_init)(n)
    temp= n

    n= Conv2D(df_dim*4, (3, 3), (1, 1), padding='SAME', kernel_initializer='HeNormal')(n)
    n= ly_relu(n)
    n= BatchNormalization(gamma_initializer= g_init)(n)

    n= Conv2D(df_dim*8, (3, 3), (1, 1), padding='SAME', kernel_initializer='HeNormal')(n)
    n= BatchNormalization(gamma_initializer= g_init)(n)

    n= add([n, temp])

    n= Flatten()(n)
    no= Dense(units=1, kernel_initializer='HeNormal', activation= 'sigmoid')(n)
    discriminator= Model(inputs=nin, outputs=no, name="discriminator")
    return discriminator


# ## **Build gan**

# In[322]:


def build_gan(generator, discriminator):
    input_img = Input(shape=(96, 96, 3))
    generated_img = generator(input_img)
    gan_output = discriminator(generated_img)
    gan = Model(input_img, [generated_img, gan_output])
    return gan


# In[323]:


# Build generator, discriminator, and GAN
generator = build_generator((96, 96, 3))
discriminator = build_discriminator((384, 384, 3))
gan = build_gan(generator, discriminator)


# ## **Model summary**

# In[324]:


generator.summary()
tf.keras.utils.plot_model(generator, 'Model.png', show_shapes=True, dpi=75)


# In[325]:


discriminator.summary()
tf.keras.utils.plot_model(discriminator, 'Model.png', show_shapes=True, dpi=75)


# In[326]:


gan.summary()
tf.keras.utils.plot_model(gan, 'Model.png', show_shapes=True, dpi=75)


# ## **Model loss**

# In[327]:


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

# In[328]:


# Compile discriminator
discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='binary_crossentropy', metrics=['accuracy'])
# Compile GAN
gan.compile(optimizer=Adam(learning_rate=0.0002,beta_1=0.5),loss=['mean_squared_error','binary_crossentropy'],
            loss_weights=[1, 1], metrics=['accuracy'])


# ## **Training**

# In[ ]:


# Training loop
batch_size = 24
epochs = 1000

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

# In[363]:


#Results---------
y = generator(test_x).numpy()

for i in tqdm(range(len(test_x))):
    plt.figure(figsize=(6,6))
    or_image = plt.subplot(3,3,1)
    or_image.set_title('Grayscale Input', color='red', fontsize=10)
    plt.axis('off')
    plt.imshow(np.clip( test_x[i],0,1), aspect='auto')
    
    in_image = plt.subplot(3,3,2)    
    image = Image.fromarray((y[i] * 255).astype('uint8'))
    image = np.asarray( image)
    in_image.set_title('Colorized Output',  color='green', fontsize=10)
    plt.axis('off')
    plt.imshow(image, aspect='auto')
    
    out_image = plt.subplot(3,3,3)
    image = Image.fromarray((test_x[i] * 255).astype('uint8'))
    plt.axis('off')
    out_image.set_title('Ground Truth',  color='blue', fontsize=10)
    plt.imshow(image, aspect='auto')    
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  
    


# ## **Model predictions**

# In[345]:


# Creating predictions on our test set-----------------

predictions = generator(test_x).numpy()


# In[355]:


# Ploting results for one image----------------

def plot_results_for_one_sample(sample_index):    
    pdimg =predictions[sample_index] 
    fig = plt.figure(figsize=(10,10))
    # Gray image-------------------
    fig.add_subplot(1,3,1)
    plt.title('Gray image')
    org=test_x[sample_index]
    np.clip((org),0,1 ).astype('uint8')
    plt.imshow(org, cmap='gray')
    plt.axis('off')
    plt.grid(None)
    
    #RGB image----------
    fig.add_subplot(1,3,2)
    plt.title('RGB image')
    rgborg= test_x[sample_index]
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


# In[356]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[357]:


#Show predicted result---------------
plot_results_for_one_sample(5)


# In[358]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[359]:


#Show predicted result---------------
plot_results_for_one_sample(15)


# In[360]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[361]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# ## **Model predictions**

# In[362]:


# Generate deblurred images from test set
test_images = generator.predict(test_x)

# Plotting example results
n = 10
plt.figure(figsize=(20, 4))
for i in tqdm(range(n)):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(np.clip(test_x[i],0,1 ), aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(np.clip(test_images[i],0,1 ), aspect='auto')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()     


# In[ ]:




