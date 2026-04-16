#!/usr/bin/env python
# coding: utf-8

# ## **Level Set algorithm for Image Segmentation**

# In[7]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


# ## **Define the initial level set function**

# In[8]:


# Define the initial level set function
def initial_level_set(img_shape, center, radius):
    X, Y = np.meshgrid(np.arange(img_shape[1]), np.arange(img_shape[0]))
    phi = np.sqrt((X - center[0])**2 + (Y - center[1])**2) - radius
    return phi


# In[9]:


# Define the Heaviside function
def heaviside(x, epsilon=1e-5):
    return 0.5 * (1 + (2/np.pi)*np.arctan(x/epsilon))

# Define the Dirac delta function
def dirac(x, epsilon=1e-5):
    return (epsilon/np.pi) / (epsilon**2 + x**2)


# ## **Evolution of the level set function**

# In[10]:


# Define the evolution equation for the level set function
def level_set_evolution(img, phi, timestep, mu, nu, lambda1, lambda2):
    # Calculate the gradient of the level set function
    grad_phi = np.gradient(phi)
    grad_phi_norm = np.sqrt(grad_phi[0]**2 + grad_phi[1]**2)
    nx = grad_phi[0] / (grad_phi_norm + 1e-8)
    ny = grad_phi[1] / (grad_phi_norm + 1e-8)

    # Calculate the curvature of the level set function
    grad2_phi = np.gradient(grad_phi_norm)
    curvature = grad2_phi[0] * ny - grad2_phi[1] * nx

    # Calculate the region-based term
    c1 = np.mean(img * heaviside(phi))
    c2 = np.mean(img * (1 - heaviside(phi)))
    region_term = lambda1 * (img - c1)**2 - lambda2 * (img - c2)**2

    # Update the level set function
    phi += timestep * (mu * curvature - nu + region_term)

    return phi


# ## **Load the input image**

# In[17]:


# Load the input image
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the level set function
phi = initial_level_set(img.shape, center=[img.shape[1]//2, img.shape[0]//2], radius=50)

# Set the parameters for the level set algorithm
timestep = 0.01
mu = 0.1
nu = 0
lambda1 = 0.04
lambda2 = 0.025
iters=500

# Iterate the level set function for a certain number of iterations
for i in tqdm(range(iters)):
    phi = level_set_evolution(img, phi, timestep, mu, nu, lambda1, lambda2)

# Threshold the level set function to obtain the segmentation result
seg = phi > 0


# ## **Display segmentation result**

# In[19]:


# Display the input image and segmentation result
# Plot the original image and the edge map
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[1].imshow((seg.astype(np.uint8)), cmap='gray')
axs[1].set_title('Segmentation Result')
axs[1].axis('off')
plt.show()


# In[ ]:




