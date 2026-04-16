#!/usr/bin/env python
# coding: utf-8

# In[64]:


#Libraries------------------ 
import os
import random
import glob
import cv2
import gc
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
from itertools import chain
from skimage.transform import resize
from skimage.morphology import label
from skimage.io import imread, imshow, concatenate_images
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.layers import Activation, Multiply, Add, GlobalAveragePooling2D, Reshape
from tensorflow.keras.models import load_model, Model,Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, multiply
from tensorflow.keras.layers import SeparableConv2D, Flatten, Dropout, Dense
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPool2D,Conv2DTranspose
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K

K.clear_session()
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## **Load Data**

# In[65]:


#Load image data-------------------
w,h,ch=128,128,3
def load_img(path):
    img= cv2.imread(path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(w,h), interpolation=cv2.INTER_LANCZOS4)
    return img

#Load data---------------------
BASE_DIR="SkinCan/train/"
img_path= os.listdir(BASE_DIR+'images')
mask_path= os.listdir(BASE_DIR+'masks')


# ## **Visulization image and mask**

# In[66]:


#plot sample images--------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in range(5):
    path= BASE_DIR + 'images/'
    ax[i].imshow(load_img(path + img_path[i]))
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()


# In[67]:


#plot sample masks--------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in range(5):
    path= BASE_DIR + 'masks/'
    ax[i].imshow(load_img(path + mask_path[i])[:, :, 0], 'gray')
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()


# In[68]:


#plot sample images--with blended mask ------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in range(5):
    path1= BASE_DIR + 'images/'
    ax[i].imshow((load_img(path1 + img_path[i])/255) * (load_img(path + mask_path[i])/255))
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()



# ## **Preprocessing dataset**

# In[122]:


#batch generation-----------------------
def load_data(path_list, gray=False):
    data=[]
    for path in tqdm(path_list):
        img= load_img(path)
        if gray:
            img= img[:, :, 0:1]
        img= cv2.resize(img, (w, h),interpolation=cv2.INTER_AREA)
        data.append(img)
    return np.array(data)

#data  preparation
X_train, X_test, y_train, y_test = train_test_split(img_path, mask_path, test_size=0.2, random_state=22)


# In[123]:


#train data generation---------------------
X_train= load_data([BASE_DIR + 'images/' + x for x in X_train])/255.0
X_test= load_data([BASE_DIR + 'images/' + x for x in X_test])/255.0

X_train.shape, X_test.shape


# In[124]:


##test data generation---------------------
Y_train= load_data([BASE_DIR + 'masks/' + x for x in y_train], gray=True)/255.0
Y_test= load_data([BASE_DIR + 'masks/' + x for x in y_test], gray=True)/255.0
Y_train= Y_train.reshape(-1, w, h, 1)
Y_test= Y_test.reshape(-1, w, h, 1)

Y_train.shape, Y_test.shape


# ## **CNN model architecture**

# In[125]:


#co-segmentation using Squeeze-and-Excitation (SE) block
def co_attention_block(input_1, input_2, filters):
    # Encoding branch
    conv1_1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_1)
    conv2_1 = Conv2D(filters * 2, (3, 3), activation='relu', padding='same')(conv1_1)
    conv3_1 = Conv2D(filters * 4, (3, 3), activation='relu', padding='same')(conv2_1)

    # Decoding branch
    conv1_2 = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_2)
    conv2_2 = Conv2D(filters * 2, (3, 3), activation='relu', padding='same')(conv1_2)
    conv3_2 = Conv2D(filters * 4, (3, 3), activation='relu', padding='same')(conv2_2)

    # Co-attention mechanism
    attention_1 = GlobalAveragePooling2D()(conv3_1)
    attention_1 = Reshape((1, 1, filters * 4))(attention_1)
    attention_1 = Conv2D(filters * 4, (1, 1), activation='sigmoid')(attention_1)

    attention_2 = GlobalAveragePooling2D()(conv3_2)
    attention_2 = Reshape((1, 1, filters * 4))(attention_2)
    attention_2 = Conv2D(filters * 4, (1, 1), activation='sigmoid')(attention_2)

    # Apply attention
    attention_branch_1 = Multiply()([conv3_1, attention_2])
    attention_branch_2 = Multiply()([conv3_2, attention_1])

    # Concatenate features
    concat = Add()([attention_branch_1, attention_branch_2])

    # Additional convolutional layers
    conv4 = Conv2D(filters * 4, (3, 3), activation='relu', padding='same')(concat)
    conv5 = Conv2D(filters * 2, (3, 3), activation='relu', padding='same')(conv4)
    conv6 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv5)

    # Output segmentation mask
    output = Conv2D(1, (1, 1), activation='sigmoid')(conv6)

    return output

# co-attention computation block
input_shape = (h, w, 3) 
filters = 32 

input_1 = Input(shape=input_shape)
input_2 = Input(shape=input_shape)

output = co_attention_block(input_1, input_2, filters)

model = Model(inputs=[input_1, input_2], outputs=output)


# ## **Create model**

# In[ ]:





# ## **Loss functions**

# In[126]:


# IoU loss-------------

def jaccard_coef(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection+smooth)

# Jaccard loss-----------------
def jaccard_loss(y_true,y_pred,smooth=1):

    return -jaccard_coef(y_true,y_pred,smooth)


# ## **Model compilation**

# In[129]:


#compile model--------------------
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=['binary_crossentropy','mse'],metrics=['accuracy']) 

#'sparse_categorical_crossentropy','categorical_crossentropy'=class>2
#'binary_crossentropy' class=2
model.summary()


# In[130]:


# Plotting  model---------------------------
tf.keras.utils.plot_model(model,to_file="model.png",show_shapes=True,show_layer_names=True,dpi=60)


# ## **Model Training**

# In[ ]:


#Train model---------------------------
nbatch_size=32
nepochs=20
SPE = len(X_train)//nbatch_size
history = model.fit([X_train,X_train],Y_train,
                    batch_size=nbatch_size,epochs=nepochs,
                    validation_split=0.2,steps_per_epoch=SPE,verbose=1,
                   shuffle=True,max_queue_size=8,workers=1,use_multiprocessing=True,
                   )



# ## **Performance evaluation**

# In[ ]:


df_result = pd.DataFrame(history.history)
df_result


# In[ ]:


# Plotting loss change over epochs---------------
nrange=nepochs
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['loss'])
plt.title('change in loss over epochs')
plt.legend(['training_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
plt.tight_layout()

# Plotting accuracy change over epochs---------------------
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['accuracy'])
plt.title('change in training accuracy coefitient over epochs')
plt.legend(['training accuracy'])
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.show()
plt.tight_layout()


# ## **Model predictions**

# In[ ]:


# predict test images
predict_y = model.predict([X_test,X_test])

plt.figure(figsize=(15,15))
for i in tqdm(range(0,9,3)):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_train[i])
    plt.title('High image')
    
    plt.subplot(4,3,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i])
    plt.title('Low image')
    
    plt.subplot(4,3,i+3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i])
    plt.title('Output by model')

plt.show()


# ## **Model predictions**

# In[ ]:


# Creating predictions on our test set-----------------
predictions = model.predict([X_test,X_test])

# create predictes mask--------------
def create_mask(predictions,input_shape=(w,h,1)):
    mask = np.zeros(input_shape)
    mask[predictions>0.5] = 1
    return mask


# In[ ]:


# Ploting results for one image----------------

def plot_results_for_one_sample(idx):    
    mask =predictions[idx] #create_mask(predictions[sample_index])   for gray-scale
    fig = plt.figure(figsize=(20,20))
    #image-------------------
    fig.add_subplot(1,4,1)
    plt.title('Input image')
    plt.imshow(X_test[idx])
    plt.axis('off')
    plt.grid(None)
    #mask-----------
    fig.add_subplot(1,4,2)
    plt.title('Real mask')
    plt.imshow(Y_test[idx],cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Predicted mask------------
    fig.add_subplot(1,4,3)
    plt.title('Predicted mask')  
    plt.imshow(mask,cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Segment---------------
    fig.add_subplot(1,4,4)
    plt.title("Segment image")
    plt.imshow(X_test[idx]*mask)
    plt.grid(None)
    plt.axis('off')  
    fig.tight_layout()    
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


#Show predicted result---------------
plot_results_for_one_sample(30)


# In[ ]:




