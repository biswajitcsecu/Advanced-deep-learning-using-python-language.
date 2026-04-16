#!/usr/bin/env python
# coding: utf-8

# ## **Saliency Guided Dense Network with Attention Learning**

# In[253]:


#Import Required Packages
import cv2
import os
import random
import numpy as np
from numpy.random import randint
import glob
import pandas as pd
from tqdm.notebook import tqdm, tnrange
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet201, DenseNet121
from tensorflow.keras import models
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, multiply
from tensorflow.keras.layers import Dense, Input, Add,  Activation,add,LayerNormalization,AvgPool2D
from tensorflow.keras.layers import Conv2D, Flatten, BatchNormalization,Concatenate,SeparableConv2D
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, UpSampling2D,  Multiply
from tensorflow.keras.layers import Input, Reshape, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D,GlobalMaxPooling2D
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback ,ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D,  MaxPool2D, UpSampling2D, concatenate, Activation
from tensorflow.keras.layers import Layer, Input, Add, Multiply, Dropout, BatchNormalization
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras import backend as K

import warnings

K.clear_session()
warnings.filterwarnings("ignore")
random.seed(23)


# ## **Data Loading**

# In[254]:


#Getting the image path 
high_image_path = "LLVIP/train/VIS/*.jpg"
low_image_path = "LLVIP/train/IR/*.jpg"

# size images 
H,W,CH=(128,128,3)

image_high = sorted(glob.glob(high_image_path), key=lambda x: x.split('.')[0])
image_low = sorted(glob.glob(low_image_path), key=lambda x: x.split('.')[0])


# ## **Data Preprocessing**

# In[255]:


#appending then into the list 
himages =[]
limages = []
#listed higher images
for image in image_high:
    img = cv2.imread(image, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (H, W), interpolation = cv2.INTER_AREA)
    himages.append(img)
himages = np.array(himages)/255.
#listed lower images
for limg in image_low:
    img = cv2.imread(limg, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (H, W), interpolation=cv2.INTER_NEAREST)
    limages.append(img)
limages = np.array(limages)/255.


# ## **splitting the image into train and test** 

# In[256]:


# splitting the image into train and test 
X=himages
Y=limages
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)


# ## **Train, Test split size**

# In[257]:


x_train = np.reshape(x_train,(len(x_train),H,W,3))
y_train = np.reshape(y_train,(len(y_train),H,W,3))
print("Shape of training images:", x_train.shape, y_train.shape)

x_test = np.reshape(x_test,(len(x_test),H,W,3))
y_test = np.reshape(y_test,(len(y_test),H,W,3))
print("Shape of test images:",x_test.shape,y_test.shape)


# ## **Data Visualization**

# In[258]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
idx=[]
plt.suptitle('High resolution')
for i in tqdm(range(9)):
    idx=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[idx])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[259]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
idx=[]
plt.suptitle('Low resolution')
for i in tqdm(range(9)):
    idx=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(y_train[idx])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Visualization the higher image and  lower images**

# In[260]:


#show image
def show_image(image, title=None, alpha=1):
    plt.imshow(image,  alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    
plt.figure(figsize=(8,15))
for i in tqdm(range(8)):
    plt.subplot(4,2,i+1)
    if (i+1)%2!=0:
        idx = np.random.randint(len(x_train))
        image = x_train[idx]
        mask = y_train[idx]
        show_image(image)
    elif (i+1)%2==0:
        show_image(mask)        


#  ## **Data Augmentation**

# In[261]:


def data_generator(x_train, y_train, batch_size):
    data_gen_args = dict(
        width_shift_range = 0.1,
            height_shift_range = 0.1,
            rotation_range = 10,
            zoom_range = 0.1)
    
    himage_generator = ImageDataGenerator(**data_gen_args).flow(x_train, x_train, batch_size, seed = 42)
    limage_generator = ImageDataGenerator(**data_gen_args).flow(y_train, y_train, batch_size, seed = 42)
    while True:
        x_batch, _ = himage_generator.next()
        y_batch, _ = limage_generator.next()
        yield x_batch, y_batch
        
def augmentedImageChecker(x_train, y_train, seed):
    hraw_batch, lraw_batch = next(data_generator(x_train, y_train, seed))
    fix, ax = plt.subplots(5,2, figsize=(8,20))    
    for i in tqdm(range(5)):
        ax[i,0].set_title('Train High Image')
        ax[i,0].axis('off') 
        ax[i,0].imshow(hraw_batch[i,:,:,0])
        ax[i,1].set_title('Train Low Image')
        ax[i,1].imshow(lraw_batch[i,:,:,0])
        ax[i,1].axis('off')    
    plt.show()        


# In[262]:


augmentedImageChecker(x_train, y_train, 5)


# ## **CNN Model**

# In[263]:


# Define the Saliency Guided Dense Network with Attention Learning
def saliency_guided_dense_network(input_shape):
    inputs = keras.Input(shape=input_shape)
    
    # Load the pre-trained VGG16 model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the weights of the pre-trained model
    for layer in vgg16.layers:
        layer.trainable = False

    # Get the output feature map from the pre-trained model
    base_model_output = vgg16.get_layer('block4_conv3').output

    # Feature extraction layers
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same')(base_model_output)
    conv2 = layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv4 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv3)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv6 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv6)

    # Saliency branch
    saliency_conv = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    saliency_attention = layers.Conv2D(1, 1, activation='sigmoid')(saliency_conv)
    saliency_output = layers.Multiply()([pool3, saliency_attention])

    # Fusion branch
    fusion_conv = layers.Conv2D(256, 3, activation='relu', padding='same')(pool3)
    fusion_attention = layers.Conv2D(256, 1, activation='sigmoid')(fusion_conv)
    fusion_output = layers.Multiply()([pool3, fusion_attention])

    # Concatenate the fusion and saliency outputs
    concat = layers.Concatenate()([saliency_output, fusion_output])

    # Fusion and reconstruction layers
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same')(concat)
    conv8 = layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)
    up1 = layers.UpSampling2D(size=(8, 8))(conv8)

    conv9 = layers.Conv2D(128, 3, activation='relu', padding='same')(up1)
    conv10 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv9)
    up2 = layers.UpSampling2D(size=(8, 8))(conv10)

    conv11 = layers.Conv2D(128, 3, activation='relu', padding='same')(up2)
    conv12 = layers.Conv2D(128, 3, activation='relu', padding='same')(conv11)

    # Output layer
    output = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(conv12)

    model = keras.Model(inputs=vgg16.input, outputs=output)
    
    return model


# ## **Create model**

# In[264]:


K.clear_session()
# Define the input shape
input_shape = (128, 128, 3)

# Create the ARDNet model
model = saliency_guided_dense_network(input_shape)


# ## **Loss functions**

# In[265]:


# Perceptual loss
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(H, W, CH))
feature_extractor = tf.keras.Model(inputs=vgg_model.input,outputs=vgg_model.get_layer('block4_conv2').output)

def perceptual_loss(y_true, y_pred):
    target_features = feature_extractor(y_true)
    generated_features = feature_extractor(y_pred)    
    loss = tf.keras.losses.MeanSquaredError()(target_features, generated_features)    
    return loss

#iou metric
smooth =100
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

#PSNR
def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return (10.0 * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true), axis=-1)))) / 2.303

#SSIM
def SSIM(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))


# ## **Model compilation**

# In[269]:


#Model compile-------
#loss = 'perceptual_loss,MSE', metrics = [PSNR, 'accuracy', SSIM]
model.compile(optimizer=Adam(learning_rate=1e-4),loss=[perceptual_loss,'MSE'],metrics=['accuracy',SSIM]) 
#model.compile( optimizer='adam',loss=['dice_loss,binary_crossentropy'],metrics=['accuracy']) 
#'sparse_categorical_crossentropy'=class>2
#'binary_crossentropy' class=2
model.summary()


# In[270]:


#Plot model
tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True,dpi=75)


# ## **Callback**

# In[271]:


#Specify Callback
def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

class ShowProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(x_train))
        himage = x_train[id]
        limage = y_train[id]
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



# ## **Model Training**

# In[ ]:


nepochs=25
nbatch_size=8
cbs = [ShowProgress()]
SPE=len(x_train)//nbatch_size


#Fit Model
history = model.fit(data_generator(x_train, y_train,nbatch_size), epochs=nepochs, validation_data=(x_test, y_test),
                    verbose=1,shuffle=True, steps_per_epoch=SPE,callbacks=cbs,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )#batch_size=nbatch_size,


# In[ ]:


df_result = pd.DataFrame(history.history)
df_result.sort_values('val_loss', ascending=True, inplace = True)
df_result


# ## **Performance evaluation**

# In[ ]:


#Performance
plt.figure(figsize = (12,8))
plt.title("Learning curve")
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend();

# Accuracy
plt.figure(figsize = (12,8))
plt.title("Learning curve")
plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["val_accuracy"], label="val_Accuracy")
plt.plot(np.argmax(history.history["val_accuracy"]), np.max(history.history["val_accuracy"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend();

# SSIM
plt.figure(figsize = (12,8))
plt.title("Learning curve")
plt.plot(history.history["SSIM"], label="SSIM")
plt.plot(history.history["val_SSIM"], label="val_SSIM")
plt.plot(np.argmax(history.history["val_SSIM"]), np.max(history.history["val_SSIM"]), marker="x", color="r",
         label="best model")
plt.xlabel("Epochs")
plt.ylabel("SSIM")
plt.legend();


# ## **Model Evaluation**

# In[ ]:


# predict test images
predict_y = model.predict(x_test)

plt.figure(figsize=(15,15))
for i in tqdm(range(0,9,3)):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
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
    original_img = x_test[rand_num]
    axes[i,0].imshow(original_img)
    axes[i,0].title.set_text('Higer Image')
    
    original_mask = y_test[rand_num]
    axes[i,1].imshow(original_mask)
    axes[i,1].title.set_text('Lower Image')
    
    original_img = np.expand_dims(original_img, axis=0)
    predicted_mask = model.predict(original_img).reshape(H,W,CH)
    axes[i,2].imshow(predicted_mask)
    axes[i,2].title.set_text('Predicted Image')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Predict & Evaluate Model**

# In[ ]:


# Predict on train, val and test
preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(x_test, verbose=1)

# Threshold predictions
preds_train_t = preds_train.astype(np.uint8)
preds_val_t = preds_val.astype(np.uint8)
preds_test_t = preds_test.astype(np.uint8)

# Perform a sanity check on some random training samples
ix = random.randint(0, len(x_train))
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
ax[0].imshow(x_train[ix])
ax[0].axis('off')
ax[0].set_title('Higher Image')

ax[1].imshow(y_train[ix])
ax[1].axis('off')
ax[1].set_title('Lower Image')

ax[2].imshow(np.squeeze(preds_train_t[ix]))
ax[2].axis('off')
ax[2].set_title('Predicted Image')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()



# ## **Evaluation**

# In[ ]:


def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

for i in tqdm(range(20)):
    idx = randint(len(x_test))
    image = x_test[idx]
    mask = y_test[idx]
    pred_mask = model.predict(tf.expand_dims(image,axis=0))[0]
        
    plt.figure(figsize=(10,8))
    plt.subplot(1,4,1)
    show_image(image, title="Higher Image")
        
    plt.subplot(1,4,2)
    show_image(mask, title="Lower Image")
        
    plt.subplot(1,4,3)
    show_image(pred_mask, title="Predicted Image")  
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  
    


# ## **Model inference on datasets**

# In[ ]:


def PSNR(y_true,y_pred):
    mse=tf.reduce_mean( (y_true - y_pred) ** 2 )
    return 20 * log10(1 / (mse ** 0.5))

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
    himage = x_test[i]
    limage = y_test[i]
    predicted = np.clip(model.predict(tf.expand_dims(himage,axis=0))[0],0.0,1.0)
    plot_images(x_test[i],y_test[i],predicted)
    print('PSNR', PSNR(x_test[i],predicted))
    


# ## **---------------------------------END------------------------------------------**

# In[ ]:




