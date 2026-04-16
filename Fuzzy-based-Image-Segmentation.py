#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm,tnrange,trange
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')


# In[29]:


#Rough Possibilistic Type-2 Fuzzy C-Means clustering 


# In[43]:


def rp_fcm(image, num_clusters, fuzziness,max_iter):
    # Convert the image to a 2D array
    pixels = image.reshape(-1, 3).astype(float)

    # Normalize the pixel values
    pixels /= 255.0

    # Initialize the membership matrix
    membership = np.random.rand(len(pixels), num_clusters)

    # Normalize the membership matrix
    membership_sum = np.sum(membership, axis=1)
    membership /= membership_sum[:, np.newaxis]

    # Initialize the prototype matrix
    prototypes = np.random.rand(num_clusters, 3)

    # Fuzzy C-Means iteration    
    for _ in tqdm(range(max_iter)):
        # Calculate the distances between pixels and prototypes
        distances = np.linalg.norm(pixels[:, np.newaxis] - prototypes, axis=2)

        # Update the membership values
        membership_new = 1 / np.power(distances, (2 / (fuzziness - 1)))
        membership_sum = np.sum(membership_new, axis=1)
        membership_new /= membership_sum[:, np.newaxis]

        # Check convergence
        if np.linalg.norm(membership - membership_new) < 1e-10:
            break

        membership = membership_new

        # Update the prototypes
        prototypes = np.dot(membership.T, pixels) / np.sum(membership, axis=0)[:, np.newaxis]

    # Perform hard clustering using K-Means
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(pixels)

    # Get the labels assigned by K-Means
    kmeans_labels = kmeans.labels_

    # Convert the labels to the same shape as the input image
    kmeans_labels = kmeans_labels.reshape(image.shape[:2])

    return kmeans_labels


# In[44]:


# Load the image
image_path = 'image3.jpg'
image = cv2.imread(image_path)

# Apply RP-FCM clustering for image segmentation
num_clusters = 3
fuzziness = 1.5
max_iter = 100
segmented_image = rp_fcm(image, num_clusters, fuzziness, max_iter)


# In[45]:


fig, axs = plt.subplots(1, 2, figsize=(10, 8))
axs[0].imshow(image)
axs[0].set_title('Image')
axs[0].axis('off')
axs[1].imshow(segmented_image)
axs[1].set_title('segmented_image')
axs[1].axis('off')
plt.show()


# In[ ]:




