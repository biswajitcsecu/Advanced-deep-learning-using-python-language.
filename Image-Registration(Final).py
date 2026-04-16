#!/usr/bin/env python
# coding: utf-8

# ## **Mutual-Information-Image-Registration**

# In[2]:


from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage import io
from PIL import Image, ImageEnhance
from skimage.exposure import match_histograms
import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## **Reading images**

# In[4]:


I = io.imread('Registration/images/img1.bmp').astype('uint8')
J = io.imread('Registration/images/img2.bmp').astype('uint8')
original_I = I.copy()
original_J = J.copy()

h1, w1 = I.shape
h2, w2 = J.shape


# ## **Visualizing images**

# In[7]:


#plot---

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(2,2,1)
plt.axis('off')
plt.imshow(I, cmap='gray')
plt.title("Reference Image")
fig.add_subplot(2,2,2)
plt.imshow(J, cmap='gray') 
plt.title("Real Image")
plt.axis('off')
plt.show()


# In[10]:


#Histogram matching
J = match_histograms(original_J, I, multichannel=False).astype('uint8')

fig=plt.figure(figsize=(10, 10))
fig.add_subplot(2,2,1)
plt.imshow(I, cmap='gray')
plt.title("Reference Image")
fig.add_subplot(2,2,2)
plt.imshow(J, cmap='gray') 
plt.title("Real Image")
plt.show()


# In[11]:


#Image enhancement
J = ImageEnhance.Brightness(Image.fromarray(original_J)).enhance(1.8)
J = np.asarray(J)

fig = plt.figure(figsize=(10, 10))
fig.add_subplot(2,2,1)
plt.imshow(I, cmap='gray')
plt.title("Reference Image")
fig.add_subplot(2,2,2)
plt.imshow(J, cmap='gray') 
plt.title("Real Image")
plt.show()


# ## **Image Registration Helper Functions**

# In[12]:


def ImageRegistration(I, J, dx, dy):   
    h1, w1 = I.shape
    h2, w2 = J.shape
    img = I.copy()
    img[dx: min(dx+h2, h1), dy:min(dy+w2, w1)] = J[:min(h2, h1-dx), :min(w2, w1-dy)]
    return img

def plot(I, J, dx, dy): 
    h1, w1 = I.shape
    h2, w2 = J.shape
    img = ImageRegistration(I, J, dx, dy)
    fig = plt.figure(figsize=(30,30))
    fig.add_subplot(1,4,1)
    plt.imshow(I, cmap='gray')
    plt.title("Reference Image", fontsize=30)
    fig.add_subplot(1,4,2)
    plt.imshow(J, cmap='gray') 
    plt.title("Real Image", fontsize=30)
    fig.add_subplot(1,4,3)
    plt.imshow(I[dx:dx+h2, dy:dy+w2], cmap='gray')
    plt.title("Selected Part of Reference Image", fontsize=30)
    fig.add_subplot(1,4,4)
    plt.imshow(img, cmap='gray') 
    plt.title("Registered Image", fontsize=30)
    plt.tight_layout()


# ## **Mutual Information Entropy (MI)**

# In[13]:


EPS = np.finfo(float).eps

def mutual_information_2d(x, y, sigma=1, normalized=False):
    jh = np.histogram2d(x, y, bins=(256, 256))[0]
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, 256))
    s2 = np.sum(jh, axis=1).reshape((256, -1))

    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(jh * np.log(jh))) - 1
    else:
        mi = np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(s2 * np.log(s2))

    return mi


# ## **Calculate mutual information entropy**

# In[14]:


max_mi = 0.
for i in range(h1-h2):
    for j in range(w1-w2):
        K = I[i:i+h2, j:j+w2]
        mi = mutual_information_2d(J.ravel(), K.ravel())
        if mi > max_mi:
            dx = i
            dy = j
            max_mi = mi


# ## **Visualize the registration results**

# In[16]:


dx=1
dy=1
plot(I, original_J, dx, dy)


# ## **Registration according to Euclidean distance**

# In[17]:


max_D = 0.
for i in range(h1-h2):
    for j in range(w1-w2):
        K = I[i:i+h2, j:j+w2]
        D = np.square(J - K).sum()
        if D > max_D:
            dx = i
            dy = j
            max_D = D


# In[18]:


plot(I, original_J, dx, dy)


# ## **SIFT**

# In[20]:


# SIFT 

sift = cv2.ORB_create()
keypoints_I, descriptors_I= sift.detectAndCompute(I, None)
keypoints_J, descriptors_J = sift.detectAndCompute(J, None)



# In[21]:


fig = plt.figure(figsize=(10,10))
fig.add_subplot(1,2,1)
img = cv2.drawKeypoints(I, keypoints_I, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#绘制关键点
plt.imshow(img, 'gray')
fig.add_subplot(1,2,2)
img = cv2.drawKeypoints(J, keypoints_J, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)#绘制关键点
plt.imshow(img, 'gray')


# ## **Brute Force Search and Nearest Neighbor Matching (K=2)**

# In[22]:


bf = cv2.BFMatcher(crossCheck=False)
matches = bf.knnMatch(descriptors_I, descriptors_J, k=2)

good = []
for m, n in matches:
    if m.distance < 0.825*n.distance:
        good.append([m])


# ## **Match point visualization**

# In[24]:


imMatches = cv2.drawMatchesKnn(I, keypoints_I, J, keypoints_J, good, None, flags=2)

plt.figure(figsize = (10, 10))
ax = plt.gca()
ax.xaxis.set_ticks_position('top')
ax.set_xlim(1, imMatches.shape[1])
ax.yaxis.set_ticks_position('left')  
ax.set_ylim(imMatches.shape[0],1) 
ax.imshow(imMatches)
plt.show()


# ## **Calculate the offset**

# In[25]:


points1 = np.zeros((len(good), 2), dtype=np.float32)
points2 = np.zeros((len(good), 2), dtype=np.float32)

for i, match in enumerate(good):
    points1[i, :] = keypoints_I[match[0].queryIdx].pt
    points2[i, :] = keypoints_J[match[0].trainIdx].pt
    

dx, dy = (points1-points2).mean(0).astype('uint8')
print('offset calculated using the mean', dy, dx)

x, y = (points1-points2).T
x.sort(), y.sort()
k1, k2 = 1, 1
dxx, dyy = x[k1:-k1].mean().astype('int'),  y[k2:-k2].mean().astype('int')
print('Offset computed using debiased mean', dyy, dxx)


# ## **Image Registration Visualization**
# 

# In[31]:


dx,dy=[1,1]
plot(I, original_J, dx, dy)


# In[32]:


plot(I, original_J, dxx, dyy)


# In[ ]:




