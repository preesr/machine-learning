#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Preethi Srinivasan

principal component analysis (PCA) to perform
dimensionality reduction.
experiment with an example 2D. reduce the data from 2D to 1D.
dataset to get intuition on how PCA works, and then use it on a bigger
dataset of 5000 face image dataset.
Reduce dimensions from 1024 pixels to 100 pixels per image

References: Andrew Ng, github jdwittenauer,rragundez, scipy.org
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat



raw_data = loadmat('data/ex7_data/ex7data1.mat')
X = raw_data['X']
#print(raw_data) # Has X whixh has 2 columns. 50 X 2

def plotInitialDataSet(x):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(x[:,0], x[:,1], s=30, color='r', label='Dataset 1')
    return

#plotInitialDataSet(X)

# ex7.pdf page 11 for steps 1) Feature normalize 2) Find covariance matrix 3) Calculate svd - Singular Value Decomposition.

def featureNormalize(X):
    X_normalized = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    return X_normalized;

# Covariance matrix is also known as Sigma
def calculateCovariance(X_normalized):
    m = X_normalized.shape[0]
    X_normalized = np.matrix(X_normalized)
    X_covariance = (X_normalized.T * X_normalized)/m
    return X_covariance

# Calculate eigen vector. From U you get Ureduce. From S you can use to choose the right k-dimension to which you can reduce the data
def calculateSVD(X_covariance):
    U, S, V = np.linalg.svd(X_covariance)

    return  U, S, V

def pca(x):
    X_normalized = featureNormalize(x)
    X_covariance = calculateCovariance(X_normalized)
    U, S, V = calculateSVD(X_covariance)
    return U,S,V

U, S, V = pca(X)
#print("U, S, V",U, S, V) # U shape 2x2 ; V shape: 2x2  ; S shape (2,)
# You must expect to see U[1,1], U[1,2] as  -0.707107 -0.707107 as per ex7_pca.m. Which is exactly what I get here

#========# Dimensionality Reduction with PCA===================================
# After computing the principal components, you can use them to reduce the
# feature dimension of your dataset by projecting each example onto a lower
# dimensional space, x(i) -> z(i) (e.g., projecting the data from 2D to 1D). In
# this part of the exercise, you will use the eigenvectors returned by PCA and
# project the example dataset into a 1-dimensional space.
# In practice, if you were using a learning algorithm such as linear regression
# or perhaps neural networks, you could now use the projected data instead
# of the original data. By using the projected data, you can train your model
# faster as there are less dimensions in the input.
# ex7.pdf page 12.
# U_reduce = U(:, 1:K)
# From Lecture 14 slides:
# z = U_reduce.T * x
#==============================================================================
K = 1

def projectData(X,U,k):
    U_reduce = U[:,:k]
    #print("U_reduce",U_reduce) # U_reduce shape (2,1)
    z = np.dot(featureNormalize(X),U_reduce)
    return z

Z = projectData(X,U,K)
# Z(1) should be 1.481274 as per ex7_pca.m. Mine is 1.49 which is good
#print("Z:",Z) # Z shape : 50 x 1

def recover_data(Z, U, k):
    U_reduce = U[:,:k]
    return np.dot(Z, U_reduce.T)

X_recovered = recover_data(Z, U, K)
#  X_rec(1, 1), X_rec(1, 2) should be -1.047419 -1.047419 as per ex7_pca.m. Mine are -1.058	and -1.0580 which seems okay
#print ("X_recovered",X_recovered) # X_recovered shape: 50 x 2
#X_recovered is a matrix. Plot doesn't work for a matrix and it should be an array
# Plot recovered data. When we reduced the data to one dimension, we lost the variations around that diagonal line, so in our reproduction everything falls along that diagonal.
#plotInitialDataSet(np.array(X_recovered))

#============Face Image Dataset==============================
# Run PCA on face images to see how it
# can be used in practice for dimension reduction. The dataset ex7faces.mat
# contains a dataset3 X of face images, each 32 X 32 in grayscale. Each row of X corresponds to one face image (a row vector of length 1024).
#==============================================================================
image_data = loadmat('data/ex7_data/ex7faces.mat')
X_img = image_data['X']
print("X_img.shape",X_img.shape) # X_img.shape (5000, 1024) 5000 images with each 1024 pixels which is 32x32 image each
#face = np.reshape(X_img[1,:], (32, 32))
#plt.imshow(face)

def showfaces(x,no_of_images):
    imv = np.empty((32,0))
    imag = []
    for i in range(no_of_images): # to show 100 rows. i.e 100 images
        # Without the .T transpose, the image of each number was rotated
        # reshapes the array of 1024 elements in given (32,32)
        im = np.reshape(x[i],(32,32)).T
        # I do not follow the below lines on how image is rendered but it works. From github rragundez
        imv = np.append(imv,im,axis=1)
        if (i+1) % 32 == 0:
            imag.append(imv)
            imv = np.empty((32,0))
    image = np.concatenate((imag[:]),axis = 0)
    plt.imshow(image)

no_of_images = 100
#showfaces(X_img,no_of_images)
# Reduced dimension to be 100
K_img = 100
U_img, S_img, V_img = pca(X_img) # U_img shape = (1024, 1024)

Z_img = projectData(X_img,U_img,K_img)
print("Z_img.shape",Z_img.shape) # Z_img.shape (5000, 100)
X_recovered_img = recover_data(Z_img, U_img, K_img)
print("X_recovered_img.shape",X_recovered_img.shape) # X_recovered_img.shape (5000, 1024)
X_normalized_img = featureNormalize(X_img)
#showfaces(X_normalized_img,no_of_images)
showfaces(X_recovered_img,no_of_images)

# You can look at the difference between faces_normalized.png and faces_recovered.png in figures_generated folder