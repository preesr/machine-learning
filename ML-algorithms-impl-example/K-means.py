#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Preethi Srinivasan


K-means clustering algorithm and apply it to compress an image.
K-means algorithm for image compression by reducing the number of colors that occur in an image to only those that are most common in that image

References: Andrew Ng, github jdwittenauer,scipy.org
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.linalg import norm
import sys
import matplotlib.image as mpimg


def findClosestCentroids(X, centroids):
    m = X.shape[0]
    k = centroids.shape[0]
    idx = np.zeros(m)

    for i in range(m):
        # Max int in python
        min_distance = sys.maxsize
        # j is index of the centroid in k which is 3 for the initial example
        for j in range(k):
            # norm - distance in Euclidean space. Length of the projection
            distance = norm((X[i,:] - centroids[j,:]))**2
            if distance < min_distance:
                min_distance = distance
                idx[i] = j
    return idx;

raw_data = loadmat('data/ex7_data/ex7data2.mat')
X = raw_data['X']
#print(raw_data)
#O/P: 'X'. X has m x 2.  Unsupervised Learning. No y data
# Given in ex7.pdf
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

#idx = findClosestCentroids(X, initial_centroids)
#print("idx first three values:",idx[0:3])
# idx first three values: [ 0.  2.  1.] - pdf ex7.pdf ans: 1 3 2. That is because Ocatave calculates index from 1



# Formula in ex7.pdf Page 4. where Ck is the set of examples that are assigned to centroid k. Concretely, if two examples say x(3) and x(5) are assigned to centroid k = 2, then you should update 2 = 1/2 (x(3) + x(5))
def computeMeans(X, idx, k):
    # computes Centroids by calculating the means of the points
    n = X.shape[1]
    mu = np.zeros((k, n))
    for i in range(k):
        # np.where returns a tuple of size 1
        assigned_to_centroid = np.where(idx == i)
        #print("indices", assigned_to_centroid)
        #print("len(indices[0])",len(assigned_to_centroid[0]))
        #print("len(indices)",len(assigned_to_centroid))
#==============================================================================
# indices (array([  0,   3,   4, ..., 273, 277, 299]),)
# len(indices[0]) 191
# len(indices) 1
# indices (array([  2, 104, 109, ..., 296, 297, 298]),)
# len(indices[0]) 103
# len(indices) 1
# indices (array([  1, 210, 218, 243, 245, 295]),)
# len(indices[0]) 6
# len(indices) 1
#==============================================================================
        # len(assigned_to_centroid) returns 1 for all iterations. As per the tuple, it has just one element. So len(assigned_to_centroid[0]) returns correct length of the element in index 0 which is where all elements are present. As this is not intuitive I can convert it to an array if needed: len(np.array(np.where(idx == 1)).flatten()) np.array(np.where(idx == 1)) shape:(1,103)  np.array(np.where(idx == 1)).flatten() shape: (103,)

        mu[i,:] = (1/len(assigned_to_centroid[0])) * np.sum(X[assigned_to_centroid,:], axis=1)

    return mu
#centroids = computeMeans(X, idx, K)
#print("computeMeans(X, idx, K)",centroids)

    # Implementing the K-means algorithm in ex7.pdf page 3
max_iters = 10;

def runkMeans(x, initial_centroids, max_iters):
    m = X.shape[0]
    k = initial_centroids.shape[0]
    ids = np.zeros(m)
    centroids = initial_centroids

    for iter in range(max_iters):
        ids = findClosestCentroids(x, centroids)
        centroids = computeMeans(x, ids, k)

    return ids, centroids


def initCentroids(X, k):
    m, n = X.shape
    initial_centroids = np.zeros((k,n))
    # Choose random k indices between range of 0 to m values
    idx = np.random.randint(0, m, k)
    for i in range(k):
        initial_centroids[i,:] = X[idx[i],:]

    return initial_centroids

# No. of centroids
# Here the initial_centroids is the existing manually given initial_centroids
K = initial_centroids.shape[0]
# Here initial_centroids is randomly generated initial_centroids
#initial_centroids = initCentroids(X,K)



idx, centroids = runkMeans(X, initial_centroids, max_iters)
# My own implementation of plotKMeans which can be used to test variable amount of clusters and not just a fixed one
def plotKMeans(idx):
    k = centroids.shape[0]
    fig, ax = plt.subplots(figsize=(12,8))
    for i in range(k):
        cluster = X[np.array(np.where(idx == i)).flatten()]
        ax.scatter(cluster[:,0], cluster[:,1], s=30, color='r')
    return;

#plotKMeans(idx)


def plotFixedKMeans():
    cluster1 = X[np.where(idx == 0)[0],:]
    cluster2 = X[np.where(idx == 1)[0],:]
    cluster3 = X[np.where(idx == 2)[0],:]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
    ax.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
    ax.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
    ax.legend()
    return;

#plotFixedKMeans()

#fig, ax = plt.subplots(figsize=(12,8))
#ax.scatter(X[:,0], X[:,1], s=30, color='r', label='Cluster 1')

#print(initCentroids(X, 3))

#===========================Part 2 Image Compression==========================
# In a straightforward 24-bit color representation of an image , each pixel is repre-
# sented as three 8-bit unsigned integers (ranging from 0 to 255) that specify
# the red, green and blue intensity values. This encoding is often refered to as
# the RGB encoding. Our image contains thousands of colors, and in this part
# of the exercise, you will reduce the number of colors to 16 colors.
# By making this reduction, it is possible to represent (compress) the photo
# in an efficient way. Specically, you only need to store the RGB values of
# the 16 selected colors, and for each pixel in the image you now need to only
# store the index of the color at that location (where only 4 bits are necessary
# to represent 16 possibilities).
# In this exercise, you will use the K-means algorithm to select the 16 colors
# that will be used to represent the compressed image. Concretely, you will
# treat every pixel in the original image as a data example and use the K-means
# algorithm to find the 16 colors that best group (cluster) the pixels in the 3-
# dimensional RGB space. Once you have computed the cluster centroids on
# the image, you will then use the 16 colors to replace the pixels in the original
# image.

#==============================================================================
# each pixel is a data sample X m = 16384 = 128 X 128
# Clusters K 16


# Reading and showing an image by plotting in Python
img=mpimg.imread('data/ex7_data/bird_small.png')
#==============================================================================
# plt.figure()
# plt.imshow(img)
#==============================================================================


#print(img.shape)
# (128, 128, 3)
#print(img)
#==============================================================================
# float 32 array of size (128, 128, 3)
# [[[ 0.85882354  0.70588237  0.40392157]
#   [ 0.90196079  0.72549021  0.45490196]
#   [ 0.88627452  0.72941178  0.43137255]....
#==============================================================================

image_data = loadmat('data/ex7_data/bird_small.mat')
#print(image_data)
#==============================================================================
# Has A which is an array of the following:
# [[219, 180, 103],
#         [230, 185, 116],
#         [226, 186, 110],....
# What I see here is 128 x 3 data. Each col data here is ranging between 0 to 255
#==============================================================================
A = image_data['A']
print(A.shape) # A.shape (128, 128, 3)

#==============================================================================
# This creates a three-dimensional matrix A whose first two indices identify
# a pixel position and whose last index represents red, green, or blue. For
# example, A(50, 33, 3) gives the blue intensity of the pixel at row 50 and
# column 33.
#print(A[51, 33, 2]) # O/P 40
#==============================================================================



A = A / 255; #Divide by 255 so that all values are in the range 0 - 1. Feature scaling as feature scaling is dividing by the range which is 255 here?
# Reshape the image into an Nx3 matrix where N = number of pixels.
X_img = np.reshape(A, (A.shape[0] * A.shape[1], A.shape[2]))
print(X_img.shape)
#(16384, 3)
#==============================================================================
# print(X)
# [ 0.85882353  0.70588235  0.40392157]
#  [ 0.90196078  0.7254902   0.45490196]
#  [ 0.88627451  0.72941176  0.43137255]...
#  which is the value I got when I had previously printed img directly
#==============================================================================
K_img = 16
max_iters_img = 10;
# Calculating K means. i.e grouping pixels into 16 clusters.

# When using K-Means, it is important the initialize the centroids randomly.
initial_centroids_img = initCentroids(X_img,K_img)
idx_img, centroids_img = runkMeans(X_img, initial_centroids_img, max_iters_img)
# idx_img shape: (16384,) which is not used.  centroids_img shape: (16,3) which is used to get the 16 colors combination of 3 - R,G,B
#==============================================================================
# After finding the top K = 16 colors (centroids_img) to represent the image, you can now
# assign each pixel position to its closest centroid using the findClosestCentroids
# function. This allows you to represent the original image using the centroid
# assignments of each pixel. Notice that you have signicantly reduced the number of bits that are #required to describe the image. The original image
# required 24 bits for each one of the 128128 pixel locations, resulting in total
# size of 128x128x24 = 393216 bits. The new representation requires some
# overhead storage in form of a dictionary of 16 colors, each of which require
# 24 bits, but the image itself then only requires 4 bits per pixel location. The final number of bits used is therefore 16 x 24 + 128 x 128 x 4 = 65,920 bits
#==============================================================================
#assign each pixel position to its closest centroid using the findClosestCentroidsfunction. This is how you are reducing the image to 16 colors
idx_img_final = findClosestCentroids(X_img, centroids_img) # shape: (16384,)


# Converting idx_img_final values which hold indices as int eg 3 13 13.... (Currently they are float 3.000) so that they can be used as indices to retrieve data from an array
# Now assign centroid[idx_img in int] to 16384 rows of pixels with 3 columns corresponding to that centroid. By centroids_img[idx_img_final.astype(int),:] you just gave it a shape of 16384 X whatever columns (which is 3). Index is not out of bound as all these are ranging from 0 to 15 values
X_recovered = centroids_img[idx_img_final.astype(int),:] # X_recovered before reshape (16384, 3)

# Reshaping X to the 128 X 128 X 3 image
X_recovered = np.reshape(X_recovered, (A.shape[0], A.shape[1], A.shape[2]))
print("X_recovered.shape",X_recovered.shape) # O/P :  (128, 128, 3)
plt.figure()
plt.imshow(X_recovered)

