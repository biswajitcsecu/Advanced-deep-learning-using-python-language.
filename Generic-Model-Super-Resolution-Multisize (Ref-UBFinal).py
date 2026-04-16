#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import Required Packages
import os
import math
import random
import numpy as np
from numpy.random import randint
import glob
import pandas as pd
from numpy import linalg as LA
import scipy.spatial.distance
import matplotlib.pyplot as plt
from skimage.io import imshow
from tqdm.notebook import tqdm, tnrange,trange
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D,MaxPooling2D,UpSampling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback ,ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import  BatchNormalization, Activation, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201, DenseNet121
from tensorflow.keras import models
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input, Add,  Activation,add,LayerNormalization,AvgPool2D
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D,  Multiply
from tensorflow.keras.layers import Input, Reshape, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback ,ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D,  MaxPool2D, UpSampling2D, concatenate, Activation
from tensorflow.keras.layers import Layer, Input, Add, Multiply, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import backend as K

np.random.seed(1337)
import warnings
K.clear_session()
warnings.filterwarnings("ignore")


# ## **Load Data**

# In[5]:


# Set some parameters
in_height = 64
in_width = 64
out_height = 256
out_width = 256
color_dim = 3
path_in =  'ISRU/train/high/'
path_out = 'ISRU/train/low/'


# In[6]:


#Getting the image path 
def load_images(path_in, path_out, test_size, valid_size):
    
    images = next(os.walk(path_in))[2] 
    num_images = len(images)
    
    testSize  = int(test_size * num_images)
    validSize = int(valid_size * num_images)
    trainSize = int(math.ceil((1-(test_size+valid_size)) * num_images))
    
    X_train = np.zeros((trainSize,in_height,in_width,color_dim), dtype=np.float32)
    X_test = np.zeros((testSize,in_height,in_width,color_dim), dtype=np.float32)
    X_valid = np.zeros((validSize,in_height,in_width,color_dim), dtype=np.float32)

    y_train = np.zeros((trainSize,out_height,out_width,color_dim), dtype=np.float32)
    y_test  = np.zeros((testSize,out_height,out_width,color_dim), dtype=np.float32)
    y_valid  = np.zeros((validSize,out_height,out_width,color_dim), dtype=np.float32)
    
    trainIdx = 0
    testIdx = 0
    validIdx = 0
    
    for idx, image in enumerate(images):
        image_in_path = os.path.join(path_in,image)
        image_out_path = os.path.join(path_out,image)

        image_in_raw =  load_img(image_in_path, target_size=(in_height,in_width),color_mode='rgb')
        image_out_raw = load_img(image_out_path, target_size=(out_height,out_width),color_mode='rgb')

        image_in = (img_to_array(image_in_raw)).squeeze() / 255 
        image_out = (img_to_array(image_out_raw)).squeeze() / 255          
        
        try:
            if (idx % 100 == 0):
                print("Stage " + str(idx))
        
            if (idx < testSize):
                X_test[testIdx] = image_in
                y_test[testIdx] = image_out
                testIdx+=1
                
            elif (idx < testSize + validSize):
                X_valid[validIdx] = image_in
                y_valid[validIdx] = image_out
                validIdx+=1
                
            else:
                X_train[trainIdx] = image_in
                y_train[trainIdx] = image_out
                trainIdx+=1
                
        except Exception as e: 
            print("\nERROR!!!!")
            print(image_in.shape)
            print(trainIdx)
            
            print(image_out.shape)
            print(testIdx)
            
            print("id "+str(idx))
            print(image_in_path)
            print(e)          
            print("\n")
            
    return X_train,X_valid,X_test,y_train,y_valid,y_test


# ## **Slicing and Reshaping Images Pipeline**

# In[ ]:


X_train,X_valid,X_test,y_train,y_valid,y_test = load_images(path_in, path_out, .2, .2)
print("Shape of training images:", X_train.shape, y_train.shape)
print("Shape of test images:",X_test.shape,y_test.shape)


# ## **Data Visualization**

# In[8]:


#Low Resolution Image
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[index],cmap=None)
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[9]:


#High Resolution Imge
figure, axes = plt.subplots(8,2, figsize=(10,10))
indexes=[]
for i in tqdm(range(9)):
    index=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(y_valid[index],cmap=None)
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Model Architecture**

# In[17]:


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # second layer
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def create_network(input_img, n_filters=16, dropout=0.5, batchnorm=True):
    # expansive path
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(4, 4), padding='same')(input_img)
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), padding='same') (c6)
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), padding='same') (c7)
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    
    u9 = Conv2DTranspose(n_filters*1, (3, 3), padding='same') (c8)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(color_dim, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    
    return model


# ## **Model creation**

# In[18]:


input_img = Input((in_height, in_width, color_dim), name='img')
model = create_network(input_img, n_filters=32, dropout=0.05, batchnorm=False)


# ## **Loss functions**

# In[ ]:


#iou metric
smooth =100
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac


# ## **Model compilation**

# In[ ]:


#Compile model
K.clear_session()
model.compile(loss='mean_squared_error', optimizer = RMSprop(),metrics=['accuracy'])
#model.compile(optimizer=Adam(learning_rate=1e-4),loss=['mean_squared_error'],metrics=['accuracy',iou]) #
#model.compile( optimizer='adam',loss=['dice_loss,binary_crossentropy'],metrics=['accuracy']) 
#'sparse_categorical_crossentropy','categorical_crossentropy'=class>2
#'binary_crossentropy' class=2
#mean_squared_error
model.summary() 


# In[ ]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True,dpi=60)


# ## **Callback**

# In[20]:


#Specify Callback
def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

class ShowProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(X_train))
        himage = X_train[id]
        limage = X_valid[id]
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
        


# In[ ]:


cbs = [ShowProgress(),ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1)]


# ## **Model Training**

# In[ ]:


#Fit Model
nbatch_size = 10
nepochs = 50
SPE=len(X_train)//nbatch_size
History = model.fit(X_train, y_train, verbose = 1, 
                         batch_size=nbatch_size,epochs=nepochs,
                         validation_data=(X_valid, y_valid),
                         shuffle=True, steps_per_epoch=SPE,callbacks=cbs,
                         max_queue_size=8,workers=16,use_multiprocessing=True,
                        )


# In[ ]:


df_result = pd.DataFrame(History.history)
df_result.sort_values('val_loss', ascending=True, inplace = True)
df_result


# ## **Visualization of metrics**

# In[ ]:


figure, axes = plt.subplots(8,2, figsize=(10,10))
loss = History.history['loss']
val_loss = History.history['val_loss']
epochsRange = range(len(History.history['loss']))
plt.figure()
plt.plot(epochsRange, loss, label='Training loss',color='g')
plt.plot(epochsRange, val_loss, label='Validation loss',color='b')
plt.title('Training and Validation loss')
plt.legend()
plt.show()


# In[ ]:


figure, axes = plt.subplots(8,2, figsize=(10,10))
plt.plot(History.history['acc'])
plt.plot(History.history['val_acc'])
plt.title('Training and Validation Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# ## **Inference on validation dataset**

# In[ ]:


pred = model.predict(X_test)

def CalcL1(obj1,obj2):
    return np.absolute(obj1-obj2).sum()

def CalcL2(obj1,obj2):
    return LA.norm(obj1-obj2)

for i in tqdm(range(40)):
    fig = plt.figure(figsize=(16,16))

    ax = fig.add_subplot(1, 3, 1)
    plt.imshow(X_test[i]) 
    ax.title.set_text('Input: 64X64 low')
    ax.set_ylabel('ylabel')
    
    ax = fig.add_subplot(1, 3, 2)
    plt.imshow(pred[i])
    ax.title.set_text('Output: 256X256 rebuilt')
    ax.set_ylabel('ylabel')
    
    ax = fig.add_subplot(1, 3, 3)
    plt.imshow(y_test[i])
    ax.title.set_text('Orig: 256X256')
    l1Dist = CalcL1(pred[i],y_test[i])
    l2Dist = CalcL2(pred[i],y_test[i])
    ax.set_xlabel('L1: ' + str(l1Dist) + '  L2: ' + str(l2Dist))


# ## **Model predictions**

# In[ ]:


# predict test images
predict_y = model.predict(X_test)

plt.figure(figsize=(15,15))
for i in range(0,9,3):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(X_test[i])
    plt.title('Higer Image')
    
    plt.subplot(4,3,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_test[i])
    plt.title('Lower Image')
    
    plt.subplot(4,3,i+3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i])
    plt.title('Output by model')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()



# ## **Model predictions**

# In[ ]:


figure, axes = plt.subplots(3,3, figsize=(20,20))
for i in tqdm(range(0,3)):
    rand_num = random.randint(0,50)
    original_img = X_test[rand_num]
    axes[i,0].imshow(original_img)
    axes[i,0].title.set_text('Higer Image')
    
    original = y_test[rand_num]
    axes[i,1].imshow(original)
    axes[i,1].title.set_text('Lower Image')
    
    original_img = np.expand_dims(original_img, axis=0)
    predicted = model.predict(original_img).reshape(H,W,CH)
    axes[i,2].imshow(predicted)
    axes[i,2].title.set_text('Predicted Image')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Model Evaluation**

# In[ ]:


#Evaluation
def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

for i in tqdm(range(20)):
    idx = randint(len(X_test))
    image = X_test[idx]
    limage = y_test[idx]
    pred = model.predict(tf.expand_dims(image,axis=0))[0]
        
    plt.figure(figsize=(10,8))
    plt.subplot(1,4,1)
    show_image(image, title="Higher Image")
        
    plt.subplot(1,4,2)
    show_image(limage, title="Lower Image")
        
    plt.subplot(1,4,3)
    show_image(pred, title="Predicted Image")  
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  
    


# In[ ]:


def PSNR(y_true,y_pred):
    mse=tf.reduce_mean( (y_true - y_pred) ** 2 )
    return 20 * np.log10(1 / (mse ** 0.5))

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def pixel_MSE(y_true,y_pred):
    return tf.reduce_mean( (y_true - y_pred) ** 2 )

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

for i in tqdm(range(5,15)):    
    himage = X_test[i]
    limage = y_test[i]
    predicted = np.clip(model.predict(tf.expand_dims(himage,axis=0))[0],0.0,1.0)
    plot_images(X_test[i],y_test[i],predicted)
    print('PSNR', PSNR(X_test[i],predicted))


