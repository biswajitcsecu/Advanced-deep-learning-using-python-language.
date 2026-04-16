#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[3]:


#Method 1


# In[4]:


def extract_features(image):
    #  Scale-Invariant Feature Transform (SIFT) algorithm
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()
    
    # Detect and compute keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    return keypoints, descriptors

def match_features(desc1, desc2):
    # Perform feature matching between two sets of descriptors  
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    # Perform matching
    matches = bf.match(desc1, desc2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    return matches

def block_based_registration(image1, image2, block_size):
    # Extract features from the first image
    keypoints1, descriptors1 = extract_features(image1)
    
    # Extract features from the second image
    keypoints2, descriptors2 = extract_features(image2)
    
    # Perform feature matching
    matches = match_features(descriptors1, descriptors2)
    
    # Initialize arrays to store matched keypoints
    src_pts = np.zeros((len(matches), 2), dtype=np.float32)
    dst_pts = np.zeros((len(matches), 2), dtype=np.float32)
    
    # Extract matched keypoints
    for i, match in enumerate(matches):
        src_pts[i, :] = keypoints1[match.queryIdx].pt
        dst_pts[i, :] = keypoints2[match.trainIdx].pt
    
    # Perform block-based filtering
    filtered_matches = []
    
    for i in range(0, len(matches), block_size):
        block_matches = matches[i:i+block_size]
        
        if len(block_matches) > 0:
            filtered_matches.append(block_matches[0])
    
    # Initialize arrays to store filtered matched keypoints
    filtered_src_pts = np.zeros((len(filtered_matches), 2), dtype=np.float32)
    filtered_dst_pts = np.zeros((len(filtered_matches), 2), dtype=np.float32)
    
    # Extract filtered matched keypoints
    for i, match in enumerate(filtered_matches):
        filtered_src_pts[i, :] = keypoints1[match.queryIdx].pt
        filtered_dst_pts[i, :] = keypoints2[match.trainIdx].pt
    
    # Perform image registration using the filtered matched keypoints
    homography, _ = cv2.findHomography(filtered_src_pts, filtered_dst_pts, cv2.RANSAC)
    
    return homography


# In[5]:


# Example usage
image1 = cv2.imread('image1.png')
image2 = cv2.imread('image2.png')
block_size = 10

homography = block_based_registration(image1, image2, block_size)
print("Homography matrix:")
print(homography)


# In[6]:


#Method 2


# In[7]:


# Function to extract features from a block
def extract_features(block):
    # Apply feature extraction algorithms (e.g., SIFT, SURF, ORB, etc.)
    # Here, we use ORB as an example
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(block, None)
    return keypoints, descriptors

# Function to match features between two sets of blocks
def match_features(descriptors1, descriptors2):
    # Create a BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


# In[8]:


# Load the reference and sensed SAR images
ref_image = cv2.imread('image1.png', cv2.IMREAD_GRAYSCALE)
sensed_image = cv2.imread('image2.png', cv2.IMREAD_GRAYSCALE)

# Set block size and stride
block_size = 16  # Size of each block
stride = 8  # Distance between consecutive blocks


# In[9]:


# Divide the images into blocks and extract features
ref_features = []
sensed_features = []

for i in range(0, ref_image.shape[0] - block_size + 1, stride):
    for j in range(0, ref_image.shape[1] - block_size + 1, stride):
        # Extract blocks from reference image
        ref_block = ref_image[i:i + block_size, j:j + block_size]
        ref_keypoints, ref_descriptors = extract_features(ref_block)
        ref_features.append((ref_keypoints, ref_descriptors))

        # Extract blocks from sensed image
        sensed_block = sensed_image[i:i + block_size, j:j + block_size]
        sensed_keypoints, sensed_descriptors = extract_features(sensed_block)
        sensed_features.append((sensed_keypoints, sensed_descriptors))

# Match features between reference and sensed blocks
matches = []

for i in range(len(ref_features)):
    ref_keypoints, ref_descriptors = ref_features[i]
    sensed_keypoints, sensed_descriptors = sensed_features[i]

    block_matches = match_features(ref_descriptors, sensed_descriptors)
    matches.append(block_matches)

# Perform further processing (e.g., filtering, RANSAC, etc.) on the matches

# Example: Draw matches between two blocks
block1_matches = matches[0]
block2_matches = matches[1]

# Create an output image to display matches
output_image = np.zeros((max(ref_image.shape[0], sensed_image.shape[0]),
                         ref_image.shape[1] + sensed_image.shape[1], 3), dtype=np.uint8)
output_image[:ref_image.shape[0], :ref_image.shape[1]] = cv2.cvtColor(ref_image, cv2.COLOR_GRAY2BGR)
output_image[:sensed_image.shape[0], ref_image.shape[1]:] = cv2.cvtColor(sensed_image, cv2.COLOR_GRAY2BGR)

# Draw matches
for match in block1_matches:
    ref_idx = match.queryIdx
    sensed_idx = match.trainIdx
    pt1 = ref_keypoints[ref_idx].pt
    pt2 = sensed_keypoints[sensed_idx].pt
    pt2 = (int(pt2[0]) + ref_image.shape[1], int(pt2[1]))
    cv2.line(output_image, (int(pt1[0]), int(pt1[1])), pt2, (0, 255, 0), 1)


# In[10]:


# Show the results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 10))
plt.gray()

ax[0].imshow(image1)
ax[0].axis('off')
ax[0].set_title('Reference')

ax[1].imshow(image2)
ax[1].axis('off')
ax[1].set_title('Target')

ax[2].imshow(output_image)
ax[2].axis('off')
ax[2].set_title('Registration')

plt.show()


# In[ ]:




