#!/usr/bin/env python
# coding: utf-8

# In[28]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# ## **Method-I**

# In[29]:


# Function to initialize the level set function
def initialize_level_set(image_shape, box):
    level_set = np.ones(image_shape[:2])
    level_set[box[0]:box[2], box[1]:box[3]] = -1
    return level_set

# Function to evolve the level set function
def evolve_level_set(image, level_set, num_iterations, alpha, beta):
    for _ in range(num_iterations):
        level_set = cv2.dilate(level_set, None)
        level_set = cv2.erode(level_set, None)

        # Calculate the mean intensity inside and outside the contour
        inside_mean = np.mean(image[level_set > 0])
        outside_mean = np.mean(image[level_set < 0])

        # Update the level set function
        level_set[image - outside_mean < inside_mean - image] = -1
        level_set[image - outside_mean > inside_mean - image] = 1

        # Smooth the level set function
        level_set = cv2.GaussianBlur(level_set.astype(np.float32), (5, 5), 0)

    return level_set


# In[30]:


# Read the input images
image1 = cv2.imread('image1.jpg', 0)
image2 = cv2.imread('image2.jpg', 0)

# Initialize the level set function for the first image
box = (50, 50, 150, 150)  # Example bounding box coordinates
level_set1 = initialize_level_set(image1.shape, box)

# Initialize the level set function for the second image
level_set2 = initialize_level_set(image2.shape, box)

# Evolve the level set functions
num_iterations = 200
alpha = 0.75
beta = 1.0
level_set1 = evolve_level_set(image1, level_set1, num_iterations, alpha, beta)
level_set2 = evolve_level_set(image2, level_set2, num_iterations, alpha, beta)

# Perform co-segmentation
segmented_image1 = np.zeros_like(image1)
segmented_image1[level_set1 > 0] = 255

segmented_image2 = np.zeros_like(image2)
segmented_image2[level_set2 > 0] = 255


# In[31]:


# Plot the original image and the edge map
fig, axs = plt.subplots(1, 4, figsize=(16, 10))
axs[0].imshow(image1, cmap='gray')
axs[0].set_title('Image')
axs[0].axis('off')
axs[1].imshow(image2, cmap='gray')
axs[1].set_title('image')
axs[1].axis('off')
axs[2].imshow(segmented_image1, cmap='gray')
axs[2].set_title('segmented_image1')
axs[2].axis('off')
axs[3].imshow(segmented_image2, cmap='gray')
axs[3].set_title('segmented_image2')
axs[3].axis('off')
plt.show()


# ## **Method-II**

# In[32]:


# Function to initialize the level set function
def initialize_level_set(image_shape, box):
    level_set = np.ones(image_shape[:2])
    level_set[box[0]:box[2], box[1]:box[3]] = -1
    return level_set

# Function to calculate edge-based terms
def calculate_edge_term(image):
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    edge_term = np.sqrt(gradient_x**2 + gradient_y**2)
    return edge_term


# In[33]:


def evolve_level_set(image, level_set, num_iterations, alpha, beta, gamma):
    edge_term = calculate_edge_term(image)
    for _ in range(num_iterations):
        level_set = cv2.dilate(level_set, None)
        level_set = cv2.erode(level_set, None)

        # Calculate the mean intensity inside and outside the contour
        inside_mean = np.mean(image[level_set > 0])
        outside_mean = np.mean(image[level_set < 0])

        # Calculate the mean edge term inside and outside the contour
        inside_edge_mean = np.mean(edge_term[level_set > 0])
        outside_edge_mean = np.mean(edge_term[level_set < 0])

        # Update the level set function with edge-based terms
        level_set[image - outside_mean < inside_mean - image + alpha * (edge_term - inside_edge_mean)] = -1
        level_set[image - outside_mean > inside_mean - image - beta * (edge_term - outside_edge_mean)] = 1

        # Regularization term to maintain contour smoothness
        curvature = calculate_edge_term(level_set)
        level_set[curvature < gamma] = 0

    return level_set


# In[34]:


# Initialize the level set function for the first image
box = (50, 50, 150, 150)  # Example bounding box coordinates
level_set1 = initialize_level_set(image1.shape, box)

# Initialize the level set function for the second image
level_set2 = initialize_level_set(image2.shape, box)

# Evolve the level set functions
num_iterations = 100
alpha = 1.0
beta = 1.0
gamma = 0.2
level_set1 = evolve_level_set(image1, level_set1, num_iterations, alpha, beta, gamma)
level_set2 = evolve_level_set(image2, level_set2, num_iterations, alpha, beta, gamma)

# Perform co-segmentation
segmented_image1 = np.zeros_like(image1)
segmented_image1[level_set1 > 0] = 255

segmented_image2 = np.zeros_like(image2)
segmented_image2[level_set2 > 0] = 255


# In[35]:


# Plot the original image and the edge map
fig, axs = plt.subplots(1, 4, figsize=(16, 10))
axs[0].imshow(image1, cmap='gray')
axs[0].set_title('Image')
axs[0].axis('off')
axs[1].imshow(image2, cmap='gray')
axs[1].set_title('image')
axs[1].axis('off')
axs[2].imshow(segmented_image1, cmap='gray')
axs[2].set_title('segmented_image1')
axs[2].axis('off')
axs[3].imshow(segmented_image2, cmap='gray')
axs[3].set_title('segmented_image2')
axs[3].axis('off')
plt.show()


# In[ ]:





# In[ ]:




