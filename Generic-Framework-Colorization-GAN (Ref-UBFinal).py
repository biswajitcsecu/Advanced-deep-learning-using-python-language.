#!/usr/bin/env python
# coding: utf-8

# ## **Image Colorization with GANs**

# In[72]:


import os
import random
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
from tensorflow import keras
from tqdm.notebook import tqdm,tnrange,trange
from tensorflow.keras import backend as K

import warnings

K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data load and splitting**

# In[68]:


batch_size = 12
img_size = 128
dataset_split =300
master_dir = 'DIV2K/train/high/'


# In[69]:


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
    


# In[70]:


# Train-test splitting
train_x, test_x, train_y, test_y = train_test_split(np.array(x), np.array(y), test_size=0.1)

# Construct tf.data.Dataset object
dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
dataset = dataset.batch(batch_size)


#  ## **Visualization the rgb image and gray**

# In[73]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(train_x[index],cmap='gray')
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[74]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
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

# In[75]:


def get_generator_model():
    inputs = tf.keras.layers.Input(shape=(img_size, img_size, 1))
    conv1 = tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=1)(inputs)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1)(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=1)(conv1)
    conv1 = tf.keras.layers.LeakyReLU()(conv1)

    conv2 = tf.keras.layers.Conv2D(32, kernel_size=(5, 5), strides=1)(conv1)
    conv2 = tf.keras.layers.LeakyReLU()( conv2 )
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=1)(conv2)
    conv2 = tf.keras.layers.LeakyReLU()(conv2)
    conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3 , 3), strides=1)( conv2 )
    conv2 = tf.keras.layers.LeakyReLU()(conv2)

    conv3 = tf.keras.layers.Conv2D( 64, kernel_size=(5, 5), strides=1)(conv2)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1)(conv3)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)
    conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1)(conv3)
    conv3 = tf.keras.layers.LeakyReLU()(conv3)

    bottleneck = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='tanh', padding='same')(conv3)

    concat_1 = tf.keras.layers.Concatenate()([bottleneck, conv3])
    conv_up_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu')(concat_1)
    conv_up_3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_3)
    conv_up_3 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_3)

    concat_2 = tf.keras.layers.Concatenate()([conv_up_3, conv2])
    conv_up_2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu')(concat_2)
    conv_up_2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_2)
    conv_up_2 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_2)

    concat_3 = tf.keras.layers.Concatenate()([conv_up_2, conv1])
    conv_up_1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(concat_3)
    conv_up_1 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=1, activation='relu')(conv_up_1)
    conv_up_1 = tf.keras.layers.Conv2DTranspose(3, kernel_size=(5, 5), strides=1, activation='relu')(conv_up_1)

    model = tf.keras.models.Model(inputs, conv_up_1)
    
    return model


# ## **GAN:Discriminator**

# In[76]:


def get_discriminator_model():
    layers = [
        tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=1, activation='relu', input_shape=(img_size, img_size,3)),
        tf.keras.layers.Conv2D(32, kernel_size=(7, 7), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(64, kernel_size=(5, 5), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.Conv2D(256, kernel_size=(3, 3), strides=1, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu' ),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid') 
    ]
    model = tf.keras.models.Sequential(layers)
    
    return model


# ### **Loss Functions**

# In[ ]:


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

# In[ ]:


generator_optimizer = tf.keras.optimizers.Adam(0.0005)
discriminator_optimizer = tf.keras.optimizers.Adam(0.0005)


# In[80]:


generator = get_generator_model()
generator.summary()

tf.keras.utils.plot_model(generator, 'Model.png', show_shapes=True, dpi=75)


# In[81]:


discriminator = get_discriminator_model()
discriminator.summary()
tf.keras.utils.plot_model(discriminator, 'Model.png', show_shapes=True, dpi=75)


# ## **Training GAN**

# In[82]:


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

# In[121]:


#Train loop
num_epochs = 5

for idx in tqdm(range(num_epochs)):
    print('epochs:', idx)
    for (x_input, y_target) in dataset:        
        train_step(inputs, targets)
        print( '<=----------------------------------!done!----------------------------------=>')


# ## **Model predictions**

# In[129]:


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

# In[130]:


# Creating predictions on our test set-----------------

predictions = generator(test_x[0:]).numpy()


# In[131]:


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


# In[132]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[133]:


#Show predicted result---------------
plot_results_for_one_sample(5)


# In[134]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[135]:


#Show predicted result---------------
plot_results_for_one_sample(15)


# In[136]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[137]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# In[ ]:




