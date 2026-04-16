#!/usr/bin/env python
# coding: utf-8

# ## **Image Colorization with GANs**

# In[2]:


import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import plot_model
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, UpSampling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.layers import  Concatenate, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_squared_error, MeanSquaredError
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tqdm.notebook import tqdm,tnrange,trange
from tensorflow.keras import backend as K

import warnings

K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data load and splitting**

# In[24]:


batch_size = 12
img_size = 128
dataset_split =300
master_dir = 'DIV2K/train/high/'


# In[25]:


#Data process
x = []
y = []

for image_file in os.listdir(master_dir)[0 : dataset_split]:
    rgb_image = Image.open(os.path.join(master_dir, image_file)).resize((img_size, img_size))
    rgb_img_array = (np.asarray(rgb_image))/ 255.
    gray_image = rgb_image.convert('L')
    gray_img_array = (np.asarray(gray_image).reshape((img_size, img_size, 1)))/ 255.
    x.append(gray_img_array)
    y.append(rgb_img_array)
    


# In[26]:


# Train-test splitting
train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.1)

# Construct tf.data.Dataset object
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(batch_size)


#  ## **Visualization the rgb image and gray**

# In[27]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(8,8))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(train_x[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[28]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(8,8))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(train_y[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **GAN: Generator**

# In[29]:


def get_generator_model():
    input_shape = (128, 128, 1)  # Input grayscale image shape

    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(inputs)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=2)(x)
    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=4)(x)
    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

    model = Model(inputs=inputs, outputs=x)
    return model


# ## **GAN:Discriminator**

# In[30]:


def get_discriminator_model():
    
    input_shape = (128, 128, 3)  # Input color image shape

    inputs = Input(shape=input_shape)

    x = Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = LeakyReLU()(x)

    x = Conv2D(128, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Conv2D(1, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU()(x)
    x = BatchNormalization()(x)

    x = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    
    return model


# ### **Loss Functions**

# In[31]:


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


# ## **Build model**

# In[32]:


generator_optimizer = tf.keras.optimizers.Adam(0.0005)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0005)


# In[33]:


generator = get_generator_model()
generator.summary()

tf.keras.utils.plot_model(generator, 'Model.png', show_shapes=True, dpi=70)


# In[34]:


discriminator = get_discriminator_model()
discriminator.summary()
tf.keras.utils.plot_model(discriminator, 'Model.png', show_shapes=True, dpi=70)


# ## **Training GAN**

# In[35]:


@tf.function
def train_step(input_x, real_y):    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(input_x, training=True)
        real_output = discriminator(real_y, training=True)
        generated_output = discriminator(generated_images, training=True)
        
        # L2 Loss 
        gen_loss = generator_loss( generated_images, real_y )
        disc_loss = discriminator_loss( real_output, generated_output )

    # Compute the gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Optimize with Adam
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


# ## **Model Training**

# In[ ]:


#Train loop
num_epochs = 5

for idx in tqdm(range(num_epochs)):
    print('epochs:', idx)
    for (x_input, y_target) in dataset:        
        train_step(x_input, y_target)
        print( '<=----------------------------------!done!----------------------------------=>')


# ## **Model predictions**

# In[ ]:


#Results---------
y = generator(test_x[0:]).numpy()

H,W=[128,128]

for i in tqdm(range(len(test_x))):
    plt.figure(figsize=(8,8))
    or_image = plt.subplot(3,3,1)
    or_image.set_title('Grayscale Input', color='red', fontsize=10)
    plt.axis('off')
    plt.imshow( test_x[i].reshape((H,W)), cmap='gray')
    
    in_image = plt.subplot(3,3,2)    
    image = Image.fromarray((y[i] * 255).astype('uint8')).resize((H, W))
    image = np.asarray( image )
    in_image.set_title('Colorized Output',  color='green', fontsize=10)
    plt.axis('off')
    plt.imshow( image )
    
    out_image = plt.subplot(3,3,3)
    image = Image.fromarray((test_y[i] * 255).astype('uint8')).resize((H, W))
    plt.axis('off')
    out_image.set_title('Ground Truth',  color='blue', fontsize=10)
    plt.imshow( image )    
    
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  
    


# ## **Model predictions**

# In[ ]:


# Creating predictions on our test set-----------------

predictions = generator(test_x[0:]).numpy()


# In[ ]:


# Ploting results for one image----------------

def plot_results_for_one_sample(sample_index):    
    pdimg =predictions[sample_index] 
    fig = plt.figure(figsize=(10,10))
    # Gray image-------------------
    fig.add_subplot(1,3,1)
    plt.title('Gray image')
    org=test_x[sample_index]
    np.clip((org),0,1 ).astype('uint8').resize((H, W))
    plt.imshow(org, cmap='gray')
    plt.axis('off')
    plt.grid(None)
    
    #RGB image----------
    fig.add_subplot(1,3,2)
    plt.title('RGB image')
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


# In[ ]:




