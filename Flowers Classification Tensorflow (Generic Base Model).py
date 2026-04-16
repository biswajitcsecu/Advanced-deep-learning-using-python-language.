#!/usr/bin/env python
# coding: utf-8

# ## **Import the Libraries**

# In[21]:


from __future__ import print_function
import os
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tqdm.notebook import tqdm


# ## **Set Global Constants**

# In[22]:


IMG_H = 128
IMG_W = 128
BATCH_SIZE = 64
EPOCHS = 25
LEARNING_RATE = 0.001
IMAGE_SHAPE = (IMG_H, IMG_W, 3)
FINE_TUNING_START = 75


# ## **Load the Train, Test and Validation Datasets**

# In[23]:


train_dataset = tf.keras.utils.image_dataset_from_directory(
    "daisy/train",
    shuffle=True,
    image_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    "daisy/valid",
    shuffle=True,
    image_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    "daisy/test",
    shuffle=True,
    image_size=(IMG_H, IMG_W),
    batch_size=BATCH_SIZE
)


# ## **Plot a Small Batch of the Train Dataset**

# In[24]:


class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
plt.subplots_adjust(hspace=0.5)
for image_batch, label_batch in train_dataset.take(1):
    for i in range(24):
        plt.subplot(6, 4, i+1)
        plt.imshow(image_batch[i] / 255)
        plt.title(class_names[label_batch[i]])
        plt.axis("off")
plt.show()       


# ## **Set Performance Helpers**

# In[25]:


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)


# ## **Load the Base Model**

# In[26]:


base_model = tf.keras.applications.InceptionV3( include_top = False, input_shape=IMAGE_SHAPE,  weights="imagenet")
base_model.trainable = False
base_model.summary()


# ## **Build the Classification Model**

# In[27]:


model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SHAPE, batch_size=BATCH_SIZE),
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.Rescaling(1./127.5, offset=-1), base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


# In[28]:


tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True,dpi=70)


# ## **Compile the Classification Model**

# In[29]:


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"]
)


# ## **Fit the Classification Model on the Training Data**

# In[30]:


history = model.fit( train_dataset, validation_data=validation_dataset, epochs=EPOCHS, verbose=1, shuffle=True,)


# ## **Evaluate the Classification Model**

# In[31]:


loss, acc = model.evaluate(validation_dataset)

print("Accuracy: {:.2f}".format(acc))
print("Loss: {:.2f}".format(loss))


# ## **Model performance**

# In[32]:


accuracy = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(10,10))
ax = plt.subplot(2,1,1)
plt.plot(accuracy)
plt.plot(val_acc)
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.set_ylim([0, 1])
ax.set_title("Training and Validation Accuracy")

ax = plt.subplot(2,1,2)
plt.plot(loss)
plt.plot(val_loss)
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training and Validation Loss")


# ## **Evaluate the Predictions on the Test Dataset**

# In[33]:


predictions = model.predict(test_dataset)


# In[34]:


predicted_batch_labels = [(1 if prediction >= 0.5 else 0) for prediction in predictions]
predicted_batch_labels = np.asarray(predicted_batch_labels)[:64]

test_batch_labels = [label for image, label in test_dataset]
test_batch_labels = test_batch_labels[0]
test_batch_labels = np.asarray(test_batch_labels)


# In[35]:


plt.figure(figsize=(10,10))
plt.subplots_adjust(hspace=0.5)
for image_batch, _ in test_dataset.take(1):
    for i in range(24):
        plt.subplot(6, 4, i+1)
        plt.imshow(image_batch[i] / 255)
        plt.title(class_names[predicted_batch_labels[i]])
        plt.axis("off")
plt.show()


# ## **Classification Report**

# In[36]:


#Classification Report
conf_matrix = confusion_matrix(y_true=test_batch_labels, y_pred=predicted_batch_labels)

fig, ax = plt.subplots(figsize=(5.5, 5.5))
ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=14)
plt.ylabel('Actuals', fontsize=14)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[39]:


#Classification aqqurecy
tn, fp, fn, tp = conf_matrix.ravel()
print("<<<-----------------Classification aqqurecy!------------------->>> ")
print("True Positive (TP): ", tp)
print("True Negative (TN): ", tn)
print("False Positive (FP): ", fp)
print("False Negative (FN): ", fn)

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print("\n\nMetrics:")
print("Accuracy: ", round(accuracy, 2))
print("Precision: ", round(precision, 2))
print("Recall: ", round(recall, 2))
print("F1-score: ", round(f1_score, 2))
print("<<<-------------------------------Done!-------------------------->>> ")


# In[ ]:




