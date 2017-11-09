#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MultiClass Logistic Regression To recognize hand written digits 0-9.

@author: Preethi

References: Andrew Ng,  github jdwittenauer,rragundez, scipy.org, scikit-learn.org


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model
# Input file in MATLAB's native format, so to load it in Python we need to use a SciPy utility.
from scipy.io import loadmat
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import cm
import random
import os
from scipy.optimize import minimize

# Returns a dict called data
data = loadmat('data/ex3data1.mat')
#dataFormatted = np.append(data['X'],data['y'],axis=1)
print(data)
X = data['X']
#The result categories in y have values ranging from 1- 10.
y = data['y']


cols = X.shape
lambda_val = 0.1
num_of_y_labels=10

print ("X, y shape:",X.shape, y.shape)
# O/P (5000, 400) (5000, 1)

#function that takes an example array and converts it into an image in gray scale
#it shoes 200 examples
def showNumbers(x):
    #im_number = 4000
    imv = np.empty((20,0))
    imag = []
    for i in range(3800,4000):
        # Without the .T transpose, the image of each number was rotated
        # reshapes the array of 400 elements in given (20,20)
        im = np.reshape(x[i],(20,20)).T
        #I do not follow the below lines on how image is rendered but it works. From github rragundez
        imv = np.append(imv,im,axis=1)
        if (i+1) % 20 == 0:
            imag.append(imv)
            imv = np.empty((20,0))
    image = np.concatenate((imag[:]),axis = 0)
    plt.imshow(image, cmap = cm.Greys_r)

#==============================================================================
# I set this and changed the range from 3800 to 4000 for im_number=4000,
# I saw other numbers as well apart from 0
# random.shuffle(dataFormatted)
# Looks like -1 omits a column
# showNumbers(dataFormatted[:,:-1])
#==============================================================================
#showNumbers(X)

print ('This are some 200 examples of the images that serve as the training set')

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Implementing the formula in pdf ex2.pdf page 9 for cost function
def costReg(theta, X, y, lambda_val):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    # excluding theta0 from the formula like in page 9
    reg = (lambda_val/(2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg




# Implementing the formula in pdf ex3.pdf page 6 for gradient.
#Note: This is only gradient, not gradient descent. You will be using optimal functions
# to calculate gradient descent instead of iteratively manually converging
# The reason you don't use a learning rate alpha here directly is because I am not calculating the
#Gradient descent. I am just calculating the gradient. Hence no need of learning rate in my implementation
# Nevertheless, my understanding here is the learning rate needed for logistic regression and that is
# determined by the fmin_tnc and scikit learn algorithms

# Using the same Gradient calculation like RegularizedLogisticRegression
# but eliminating for loop for better vectorization. USing formula in ex3.pdf page 6
# (1/m) * X.T * (h(x) - y)  + reggularization  :
def gradientReg(theta, X, y, lambda_val):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
# (h(x) - y)
    error = sigmoid(X * theta.T) - y
    #(1/m) * X.T * (h(x) - y) + reggularization
    grad = ((X.T*error) / len (X)).T + ((lambda_val/len(X)) * theta)
        # because we are not penalizing Oo with regularization as per the videos. Hence, update term
        # without regularization
    grad[0,0] = np.sum(np.multiply(error, X[:,0])) / len(X)
    return np.array(grad).ravel()

# testing

theta_t = ([-2,-1,1,2])
y_t = ([1],[0],[1],[0],[1])
X_t = ([1,0.1,0.6,1.1],
       [1,0.2,0.7,1.2],
       [1,0.3,0.8,1.3],
       [1,0.4,0.9,1.4],
       [1,0.5,1.0,1.5]);

lambda_t = 3;
cost = costReg(theta_t, X_t, y_t, lambda_t)
print ("test data cost:",cost)
grad = gradientReg(theta_t, X_t, y_t, lambda_t)
print ("test data gradient:",grad)
#cost and gradient worked correctly for the test data.
# cost: 2.53481939611
#gradient: [ 0.14656137 -0.54855841  0.72472227  1.39800296]



# Insert into array X values of at Col 0
X = np.append(np.ones((X.shape[0],1)),X,axis=1)
cols = X.shape[1]
m = X.shape[0]
print("X array shape after appending X0:",np.shape(X))
initial_theta= np.zeros(cols)
all_theta_for_labels = np.zeros((num_of_y_labels, cols))

# Not sure why the opt.fmin_cg is not working. Accuracy comes only to 10%
#==============================================================================
# def oneVsAll(theta,x,y,lamda,C,miter):
#     #Y = (y == C).astype(int)
#    # you will be using fmin_cg for this exercise (instead of fmin_tnc).
# #fmin_cg works similarly to fminunc, but is more more ecient for dealing with a large number of parameters.
# #fmin_cg is like fmin_tnc - finding the gradient descent using your cost and gradient implementation
#     theta = opt.fmin_cg(costReg,theta,fprime = gradientReg,args=(X,y,lamda), maxiter = miter,disp = 0)
#     return theta
#==============================================================================

def one_vs_all(X, y, num_labels, lamda_value):
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, num_labels + 1):
        theta = np.zeros(cols)

        y_i = np.array([1 if label == i else 0 for label in y])
        y_i = np.reshape(y_i, (m, 1))
        # y_i shape before reshape: (5000,)
        #y_i shape after reshape: (5000, 1)
        # Reshapes a given array  into the dimension you ask for (Eg: 400 pixels reshaped as 20x20)

        # minimize the objective function
#we're transforming y from a class label to a binary value for each classifier
#(either is class i or is not class i). Finally, we're using SciPy's newer optimization API to
# minimize the cost function for each classifier. The API takes an objective function,
# an initial set of parameters, an optimization method, and a jacobian (gradient) function if specified.
# The parameters found by the optimization routine are then assigned to the parameter array.
# minimize - Minimization of scalar function of one or more variables - here it refers to costReg
# TNC - truncated Newton (TNC) algorithm. Directly used in the OnlineLogisticRegression example
# fmin.x = The optimization result represented as a OptimizeResult object.
#  Important attributes: x the solution array



        fmin = minimize(fun=costReg, x0=theta, args=(X, y_i, lamda_value), method='TNC', jac=gradientReg)
        # fmin.x shape: (401,)
        # Thetas for label 1 is saved in 0, for label 10 is saved in row 9
        all_theta_for_labels[i-1,:] = fmin.x
        #print("fmin.x shape:", np.shape(fmin.x))

    #print(np.shape(all_theta_for_labels))
    # (10, 401)
    return all_theta_for_labels


#==============================================================================


def predict(theta, X):
    probability = sigmoid(np.dot(X,theta.T))
    #return probability an array of 5000x10
    #argmax- Returns the indices of the maximum values along an axis.
        # create array of the index with the maximum probability.
        #i.e out of the 10 label probabilities calculated
        # which label(index) does it seem to be the most likely
        #On a new input,to make a prediction, pick the class that maximizes h(x)
    h_argmax = np.argmax(probability, axis=1)
    #print("shape of h_argmax",np.shape(h_argmax))
    # shape of h_argmax (5000,)
    # because our array was zero-indexed we need to add one for the true label prediction
    # Thetas for label 1 was saved in 0, for label 10 was saved in row 9
    h_argmax = h_argmax + 1
    #print("h_argmax",h_argmax)
    return h_argmax

def accuracyOfHypothesis(y_val, x_val,theta):
    predictions = predict(theta,x_val)
    # zip - Make an iterator that aggregates elements from each of the iterables.
    correct = [1 if a == b else 0
           for (a, b) in zip(predictions, y_val)]
    accuracy = (sum(correct) / float(len(correct))) * 100
    return accuracy


all_theta_for_labels = one_vs_all(X, y, num_of_y_labels, lambda_val)
#==============================================================================
# max_iterations=20
# for C in range(1,(num_of_y_labels+1)):
#     #  the best min cost calculated theta for label 10 is at all_theta_for_labels[0,:] row
#     #  the best min cost calculated theta for label 1 is at all_theta_for_labels[1,:] row and so on...
#     #for all 10 classifiers
#     all_theta_for_labels[C % 10,:] = oneVsAll(initial_theta,X,y,lambda_val,num_of_y_labels,max_iterations)
#     print ('Finished oneVsAll checking number:', C)
# print ('All the numbers have been checked:')
#==============================================================================


accuracy=accuracyOfHypothesis(y, X,all_theta_for_labels)
print ('accuracy = {0}%'.format(accuracy))
