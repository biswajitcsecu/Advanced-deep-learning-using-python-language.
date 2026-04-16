#!/usr/bin/env python
# coding: utf-8

# In[31]:


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
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,Callback 
from tensorflow.keras import backend as K
random.seed(23)

import warnings

K.clear_session()
warnings.filterwarnings("ignore")


# ## **Data Loading**

# In[4]:


#Getting the image path 
image_path = "Melanoma/train/images/*.jpg"
mask_path = "Melanoma/train/masks/*.png"

## size images 
H,W,CH=(128,128,3)

image_names = sorted(glob.glob(image_path), key=lambda x: x.split('.')[0])
mask_names = sorted(glob.glob(mask_path), key=lambda x: x.split('.')[0])


# ## **Data Preprocessing**

# In[5]:


#appending then into the list 
images =[]
masks = []

for image in image_names:
    img = cv2.imread(image, 1)
    img = cv2.resize(img, (H, W))
    images.append(img)
    
images = np.array(images)/255.

for mask in mask_names:
    msk = cv2.imread(mask, 0)
    msk = cv2.resize(msk, (H, W))
    masks.append(msk)
    
masks = np.array(masks)/255.


# In[6]:


## splitting the image into train and test 
X=images
Y=masks
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.1, random_state=23)


# ## **Data Visualization**

# In[8]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
idx=[]
for i in tqdm(range(9)):
    idx=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(x_train[idx])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[9]:


#Display---------------------
figure, axes = plt.subplots(8,2, figsize=(10,10))
idx=[]
for i in tqdm(range(9)):
    idx=random.randint(0,10)    
    plt.subplot(3,3,i+1)
    plt.imshow(y_train[idx])
    plt.axis("off")

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# In[10]:


def show_image(image, title=None, cmap='gray', alpha=1):
    plt.imshow(image, cmap=cmap, alpha=alpha)
    if title is not None:
        plt.title(title)
    plt.axis('off')
    
plt.figure(figsize=(8,15))
for i in range(8):
    plt.subplot(4,2,i+1)
    if (i+1)%2!=0:
        idx = np.random.randint(len(x_train))
        image = x_train[idx]
        mask = y_train[idx]
        show_image(image)
    elif (i+1)%2==0:
        show_image(mask)
        


# ## **Create model**

# In[11]:


def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
    
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x =tf.keras.layers.Activation('relu')(x)
    
    #Second Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
        
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

# Now defining Unet 
def CNNUnet(inputImage, numFilters = 16, droupouts = 0.1, doBatchNorm = True):
    # defining encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(droupouts)(p1)
    
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(droupouts)(p2)
    
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(droupouts)(p3)
    
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    p4 = tf.keras.layers.Dropout(droupouts)(p4)
    
    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    u6 = tf.keras.layers.Dropout(droupouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(droupouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(droupouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(droupouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    
    output = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c9)
    model = tf.keras.Model(inputs = [inputImage], outputs = [output])
    
    return model


# In[ ]:


#DSNet
def DSARNet(nClasses, input_height, input_width):    
    img_input = Input(shape=(input_height, input_width, 3))    
    Encoder_Dense = DenseNet121( weights = 'imagenet',include_top = False,input_tensor = img_input) 
    Encoder = SeparableConv2D(filters = 1024,kernel_size = (3, 3),activation = 'relu',kernel_initializer='he_normal',
                              padding="same")(Encoder_Dense.output)
    
    Encoder = BatchNormalization()(Encoder)
    
    # Decoding
    Decoder = UpSampling2D(size = (2, 2))(Encoder)
    Decoder = concatenate([Encoder_Dense.get_layer(name="pool3_pool").output, Decoder], axis=-1)
    Decoder = SeparableConv2D(filters = 1024,kernel_size = (3, 3),activation = 'relu',kernel_initializer='he_normal',
                              padding = "same")(Decoder)
    Decoder = BatchNormalization()(Decoder)
    
    Decoder = UpSampling2D(size = (2, 2))(Decoder)
    Decoder = concatenate([Encoder_Dense.get_layer(name="pool2_pool").output, Decoder], axis=-1)
    Decoder = SeparableConv2D(filters = 512,kernel_size = (3, 3),activation = 'relu',kernel_initializer='he_normal',
                              padding = "same")(Decoder)
    Decoder = BatchNormalization()(Decoder)

    Decoder = UpSampling2D(size = (2, 2))(Decoder)
    Decoder = concatenate([Encoder_Dense.get_layer(name="pool1").output, Decoder], axis=-1)
    Decoder = SeparableConv2D(filters = 256,kernel_size = (3, 3),activation = 'relu',kernel_initializer='he_normal',
                              padding = "same")(Decoder)
    Decoder = BatchNormalization()(Decoder)

    Decoder = UpSampling2D( size = (2, 2))(Decoder)
    Decoder = concatenate([Encoder_Dense.get_layer(name="conv1/bn").output, Decoder], axis=-1)
    Decoder = SeparableConv2D(filters = 128,kernel_size = (3, 3),activation = 'relu',kernel_initializer='he_normal',
                              padding = "same")(Decoder)
    Decoder = BatchNormalization()(Decoder)

    Decoder = UpSampling2D(size = (2, 2))(Decoder)    
    Decoder = SeparableConv2D(filters = 64,kernel_size = (3, 3),activation = 'relu',kernel_initializer='he_normal',
                              padding = "same")(Decoder)
    Decoder = BatchNormalization()(Decoder)

    Decoder = SeparableConv2D(filters = nClasses,kernel_size = (1, 1),activation = 'relu',kernel_initializer='he_normal',
                              padding = "same")(Decoder)
    Decoder = BatchNormalization()(Decoder)    

    Predicted_Mask = Conv2D(filters = 1,kernel_size = 1,activation = 'sigmoid')(Decoder)        
    DSARNet_model = Model(inputs = img_input, outputs = Predicted_Mask)

    return DSARNet_model


# In[12]:


## instanctiating model
inputs = tf.keras.layers.Input((H, W, CH))
#model = CNNUnet(inputs, droupouts= 0.07)

num_classes =1
model=DSARNet(num_classes, H, W)


# ## **Loss functions**

# In[13]:


#dice_loss
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

#iou metric
smooth =100
def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

#jacard_coef
def jaccard_coef(y_true,y_pred,smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f*y_pred_f)
    return (intersection+smooth)/(K.sum(y_true_f)+K.sum(y_pred_f)-intersection+smooth)

def jaccard_loss(y_true,y_pred,smooth=1):
    return -jaccard_coef(y_true,y_pred,smooth)


# ## **Model compilation**

# In[ ]:


#Model compile-------
model.compile( optimizer='adam',loss=[dice_loss,'binary_crossentropy'],metrics=['accuracy', jaccard_loss, iou]) 
#model.compile( optimizer='adam',loss=['binary_crossentropy'],metrics=['accuracy']) 
#'sparse_categorical_crossentropy'=class>2
#'binary_crossentropy' class=2
model.summary()


# In[15]:


tf.keras.utils.plot_model(model, 'Model.png', show_shapes=True,dpi=60)


# ## **Callback**

# In[16]:


#Specify Callback
def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

class ShowProgress(Callback):
    def on_epoch_end(self, epoch, logs=None):
        id = np.random.randint(len(x_train))
        image = x_train[id]
        mask = y_train[id]
        pred_mask = self.model(tf.expand_dims(image,axis=0))[0]
        
        plt.figure(figsize=(10,8))
        plt.subplot(1,3,1)
        show_image(image, title="Image")
        
        plt.subplot(1,3,2)
        show_image(mask, title="Mask")
        
        plt.subplot(1,3,3)
        show_image(pred_mask, title="Predicted Mask")
            
        plt.tight_layout()
        plt.show()


# ## **Model Training**

# In[17]:


nepochs=25
nbatch_size=32
cbs = [ShowProgress()]
SPE=len(x_train)//nbatch_size


# In[18]:


#Fit Model
history = model.fit(x_train, y_train, epochs=nepochs, validation_data=(x_test, y_test),
                    batch_size=nbatch_size,verbose=1,shuffle=True, steps_per_epoch=SPE,callbacks=cbs,
                    max_queue_size=8,workers=1,use_multiprocessing=True,
                   )


# In[25]:


df_result = pd.DataFrame(history.history)
df_result


# ## **Performance evaluation**

# In[26]:


# Plotting loss change over epochs----
nrange=nepochs
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['loss'])
plt.title('change in loss over epochs')
plt.legend(['training_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
plt.tight_layout()

# Plotting accuracy change over epochs---
x = [i for i in tqdm(range(nrange))]
plt.plot(x,history.history['accuracy'])
plt.title('change in training accuracy coefitient over epochs')
plt.legend(['training accuracy'])
plt.xlabel('epochs')
plt.ylabel('training accuracy')
plt.show()
plt.tight_layout()


# ## **Model Evaluation**

# In[27]:


# predict test images
predict_y = model.predict(x_test)

plt.figure(figsize=(15,15))
for i in range(0,9,3):
    plt.subplot(4,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i])
    plt.title('Image')
    
    plt.subplot(4,3,i+2)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i])
    plt.title('Mask')
    
    plt.subplot(4,3,i+3)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(predict_y[i], cmap='Paired')
    plt.title('Output by model')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Model predictions**

# In[28]:


figure, axes = plt.subplots(3,3, figsize=(20,20))
for i in tqdm(range(0,3)):
    rand_num = random.randint(0,50)
    original_img = x_test[rand_num]
    axes[i,0].imshow(original_img)
    axes[i,0].title.set_text('Original Image')
    
    original_mask = y_test[rand_num]
    axes[i,1].imshow(original_mask)
    axes[i,1].title.set_text('Original Mask')
    
    original_img = np.expand_dims(original_img, axis=0)
    predicted_mask = model.predict(original_img).reshape(H,W)
    axes[i,2].imshow(predicted_mask, cmap='Paired')
    axes[i,2].title.set_text('Predicted Mask')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# ## **Predict & Evaluate Model**

# In[29]:


# Predict on train, val and test
preds_train = model.predict(x_train[:int(x_train.shape[0]*0.9)], verbose=1)
preds_val = model.predict(x_train[int(x_train.shape[0]*0.9):], verbose=1)
preds_test = model.predict(x_test, verbose=1)

# Threshold predictions
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
ix = random.randint(0, len(preds_train_t))
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
ax[0].imshow(x_train[ix])
ax[0].axis('off')
ax[0].set_title('Image')

ax[1].imshow(y_train[ix])
ax[1].axis('off')
ax[1].set_title('Mask')

ax[2].imshow(np.squeeze(preds_train_t[ix]), cmap='Paired')
ax[2].axis('off')
ax[2].set_title('Predicted mask')

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()


# ## **Evaluation**

# In[34]:


def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

for i in tqdm(range(20)):
    idx = randint(len(x_test))
    image = x_test[idx]
    mask = y_test[idx]
    pred_mask = model.predict(tf.expand_dims(image,axis=0))[0]
    post_process = (pred_mask[:,:,0] > 0.5).astype('int')
        
    plt.figure(figsize=(10,8))
    plt.subplot(1,4,1)
    show_image(image, title="Original Image")
        
    plt.subplot(1,4,2)
    show_image(mask, title="Original Mask")
        
    plt.subplot(1,4,3)
    show_image(pred_mask, title="Predicted Mask")
    
    plt.subplot(1,4,4)
    show_image(post_process, title="Post=Processed Mask")      
    
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()  


# ## **Segmentation predictions**

# In[35]:


# Creating predictions on our test set-----------------
predictions = model.predict(x_test)
# create predictes mask--------------
def create_mask(predictions,input_shape=(W,H,1)):
    mask = np.zeros(input_shape)
    mask[predictions>0.5] = 1
    return mask


# In[36]:


# Ploting results for one image----------------
def plot_results_for_one_sample(sample_index):    
    mask =create_mask(predictions[sample_index]) #create_mask(predictions[sample_index])   for gray-scale
    fig = plt.figure(figsize=(20,20))
    #image-------------------
    fig.add_subplot(1,4,1)
    plt.title('Input image')
    plt.imshow(x_test[sample_index])
    plt.axis('off')
    plt.grid(None)
    #mask-----------
    fig.add_subplot(1,4,2)
    plt.title('Real mask')
    plt.imshow(y_test[sample_index],cmap='gray')
    plt.axis('off')
    plt.grid(None)
    #Predicted mask------------
    fig.add_subplot(1,4,3)
    plt.title('Predicted mask')  
    plt.imshow(mask, cmap='Paired')
    plt.axis('off')
    plt.grid(None)
    #Segment---------------
    fig.add_subplot(1,4,4)
    plt.title("Segment image")
    plt.imshow(x_test[sample_index]*mask)
    plt.grid(None)
    plt.axis('off')  
    fig.tight_layout()    

plt.subplots_adjust(wspace=0, hspace=0)
plt.tight_layout()
plt.show()    


# In[37]:


#Show predicted result---------------
plot_results_for_one_sample(1)


# In[38]:


#Show predicted result---------------
plot_results_for_one_sample(6)


# In[39]:


#Show predicted result---------------
plot_results_for_one_sample(10)


# In[40]:


#Show predicted result---------------
plot_results_for_one_sample(14)


# In[41]:


#Show predicted result---------------
plot_results_for_one_sample(20)


# In[42]:


#Show predicted result---------------
plot_results_for_one_sample(25)


# In[43]:


#Show predicted result---------------
plot_results_for_one_sample(30)


# In[ ]:





# In[ ]:




