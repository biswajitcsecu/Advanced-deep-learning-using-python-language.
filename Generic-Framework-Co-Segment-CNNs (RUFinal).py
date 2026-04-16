#!/usr/bin/env python
# coding: utf-8

# In[50]:


#Libraries------------------ 
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import gc
from tqdm.notebook import trange, tqdm
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
import glob
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, multiply
from tensorflow.keras.layers import (BatchNormalization, Conv2DTranspose, 
                                     SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense)
from tensorflow.keras.preprocessing.image import load_img, array_to_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, MaxPool2D,Conv2DTranspose, concatenate,Input
from tensorflow.keras.callbacks import CSVLogger
K.clear_session()
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')



# ## **Load Data**

# In[51]:


#Load image data-------------------
w,h,ch=128,128,3
def load_img(path):
    img= cv2.imread(path)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(w,h))
    return img

#Load data---------------------
BASE_DIR="SkinCan/train/"
img_path= os.listdir(BASE_DIR+'images')
mask_path= os.listdir(BASE_DIR+'masks')


# ## **Visulization image and mask**

# In[52]:


#plot sample images--------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in range(5):
    path= BASE_DIR + 'images/'
    ax[i].imshow(load_img(path + img_path[i]))
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()


# In[53]:


#plot sample masks--------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in range(5):
    path= BASE_DIR + 'masks/'
    ax[i].imshow(load_img(path + mask_path[i])[:, :, 0], 'gray')
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()


# In[54]:


#plot sample images--with blended mask ------------
fig, ax= plt.subplots(1,5, figsize=(20, 10))
for i in range(5):
    path1= BASE_DIR + 'images/'
    ax[i].imshow((load_img(path1 + img_path[i])/255) * (load_img(path + mask_path[i])/255))
    ax[i].set_xticks([]); ax[i].set_yticks([])

fig.tight_layout()
plt.show()



# ## **Preprocessing dataset**

# In[55]:


#batch generation-----------------------
def load_data(path_list, gray=False):
    data=[]
    for path in tqdm(path_list):
        img= load_img(path)
        if gray:
            img= img[:, :, 0:1]
        img= cv2.resize(img, (w, h))
        data.append(img)
    return np.array(data)

#data  preparation
X_train, X_test, y_train, y_test = train_test_split(img_path, mask_path, test_size=0.2, random_state=22)
len(X_train), len(X_test)



# In[56]:


#train data generation---------------------
X_train= load_data([BASE_DIR + 'images/' + x for x in X_train])/255.0
X_test= load_data([BASE_DIR + 'images/' + x for x in X_test])/255.0

X_train.shape, X_test.shape


# In[57]:


##test data generation---------------------
Y_train= load_data([BASE_DIR + 'masks/' + x for x in y_train], gray=True)/255.0
Y_test= load_data([BASE_DIR + 'masks/' + x for x in y_test], gray=True)/255.0
Y_train= Y_train.reshape(-1, w, h, 1)
Y_test= Y_test.reshape(-1, w, h, 1)

Y_train.shape, Y_test.shape


# ## **CNN model architecture**

# In[58]:


# Convolutional blocks-----------------
def co_attention_block(input_1, input_2, filters):
    # Encoding branch
    conv1_1 = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_1)
    conv2_1 = Conv2D(filters * 2, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)
    conv3_1 = Conv2D(filters * 4, (3, 3), activation='relu', padding='same')(pool2)

    # Decoding branch
    conv1_2 = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_2)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)
    conv2_2 = Conv2D(filters * 2, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)
    conv3_2 = Conv2D(filters * 4, (3, 3), activation='relu', padding='same')(pool2)

    # Co-attention mechanism
    attention = multiply([conv3_1, conv3_2])
    attention = Conv2D(filters * 4, (1, 1), activation='relu', padding='same')(attention)

    # Concatenate features
    concat = Concatenate()([conv3_1, conv3_2, attention])

    # Additional convolutional layers
    conv4 = Conv2D(filters * 4, (3, 3), activation='relu', padding='same')(concat)
    conv5 = Conv2D(filters * 2, (3, 3), activation='relu', padding='same')(conv4)
    conv6 = Conv2D(filters, (3, 3), activation='relu', padding='same')(conv5)

    # Upsampling
    upsample1 = UpSampling2D(size=(2, 2))(conv6)
    upsample2 = UpSampling2D(size=(2, 2))(upsample1)

    # Output segmentation mask
    output = Conv2D(1, (1, 1), activation='sigmoid')(upsample2)

    return output

# Example usage of the co-attention computation block
input_shape = (h, w, 3)  # Input image shape
filters = 32  # Number of filters in the convolutional layers

input_1 = Input(shape=input_shape)
input_2 = Input(shape=input_shape)

output = co_attention_block(input_1, input_2, filters)

model = Model(inputs=[input_1, input_2], outputs=output)


# ## **Create model**

# In[ ]:





# ## **Loss functions**

# In[59]:


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

# In[60]:


#compile model--------------------
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),loss=['binary_crossentropy'],metrics=['accuracy', jaccard_loss]) 
#'sparse_categorical_crossentropy'=class>2
#'binary_crossentropy' class=2
model.summary()


# In[61]:


# Plotting  model---------------------------
tf.keras.utils.plot_model(model,to_file="model.png",show_shapes=True,show_layer_names=True,dpi=60)


# ## **Model Training**

# In[ ]:


#Train model---------------------------
nbatch_size=64
nepochs=100
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

# Plotting jacard  accuracy change over epochs---------------------
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['jaccard_loss'])
plt.title('change in jacard_coef coefitient over epochs')
plt.legend(['jacard_coef'])
plt.xlabel('epochs')
plt.ylabel('jacard_coef')
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

def plot_results_for_one_sample(sample_index):    
    mask =predictions[sample_index] #create_mask(predictions[sample_index])   for gray-scale
    fig = plt.figure(figsize=(20,20))
    #image-------------------
    fig.add_subplot(1,4,1)
    plt.title('Input image')
    plt.imshow(X_test[sample_index])
    plt.axis('off')
    plt.grid(None)
    #mask-----------
    fig.add_subplot(1,4,2)
    plt.title('Real mask')
    plt.imshow(Y_test[sample_index],cmap='gray')
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
    plt.imshow(X_test[sample_index]*mask)
    plt.grid(None)
    plt.axis('off')  
    fig.tight_layout()    
plt.show()


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(6)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(14)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# In[ ]:


#Show predicted result---------------
plot_results_for_one_sample(30)


# ## **Model Evaluation**

# In[ ]:


# Predict on train, val and test
preds_train = model.predict([X_train[:int(X_train.shape[0]*0.9)],X_train[:int(X_train.shape[0]*0.9)]], verbose=1)
preds_val = model.predict([X_train[int(X_train.shape[0]*0.9):],X_train[int(X_train.shape[0]*0.9):]], verbose=1)
preds_test = model.predict([X_test,X_test], verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)


# In[ ]:


# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
ax[0].imshow(X_train[ix])
ax[0].axis('off')
ax[0].set_title('Image')

ax[1].imshow(Y_train[ix])
ax[1].axis('off')
ax[1].set_title('Mask')

ax[2].imshow(np.squeeze(preds_train_t[ix]))
ax[2].axis('off')
ax[2].set_title('Predicted mask')

plt.show()


# In[ ]:





# In[ ]:




