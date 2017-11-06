#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Regression with one variable to predict profitts for a food truck.
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new
outlet. The chain already has trucks in various cities and you have data for
profits and populations from the cities.You would like to use this data to help you select which city to expand to next.

@author: Preethi Srinivasan
References: Andrew Ng,  github jdwittenauer, scipy.org, scikit-learn.org
"""

import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

#==============================================================================
# \generating an identity matrix
# def warmUpExercise():
#     return (numpy.identity(5));
#
# print (warmUpExercise())
#==============================================================================

# plotting graph from a given data set
#==============================================================================
# def plotData():
#     path = os.getcwd() + '/data/ex1data1.txt'
#     data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
#     # To get only a column you want
#     #print(data.Population)
#     data.head
#     data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#     return;
#print (plotData())
#==============================================================================



alpha = 0.01
num_iters=1500

# Convert array of zeros to matrix
theta = numpy.matrix(numpy.zeros((2,1)))
#print (theta.T)
path = os.getcwd() + '/data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# insert Xo a column of 1s
data.insert(0,'Ones', 1)
# Size of the training set
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
#Convert DataFrame to Matrix
X = numpy.matrix(X.values)
y = numpy.matrix(y.values)
m = len(X) # No. of training samples
j_history = numpy.zeros((num_iters,1))


def computeCost(X,y,theta):
    # ((Oo + O1x) - y) ^ 2
    inner_equation =  numpy.power(((X * theta) - y),2)
    # the mean sum
    cost = numpy.sum(inner_equation)/(2*len(X))
    return cost

print(computeCost(X,y,theta))

def gradientDescent(X, y, theta, alpha, num_iters):
    temp = theta;
    for i in range(num_iters):
        error_diff = (X * theta) - y
        # For each theta which is 2 elements here 2 X 1 matrix
        for j in range(len(theta)):
            # numpy.multiply is not matrix multiplication (*). It is element-wise multiplication
            # theta[j] is the jth row
            temp[j]    = theta[j] - (alpha/len(X)) * (numpy.sum(numpy.multiply(error_diff, X[:,j])))

        theta=temp
        j_history[i] = computeCost(X,y,theta)
    return theta, j_history

theta_prediction, j_history = gradientDescent(X, y, theta, alpha, num_iters)
# value of theta after gradient decent. The best values for theta
print ("value of theta after gradient decent. The best values for theta:")
print(theta_prediction)
# value of the cost function with the new theta. The lowest difference between the hypothesis and the actual y
print ("value of the cost function with the new theta. The lowest difference between the hypothesis and the actual y")
print(computeCost(X,y,theta_prediction))
#print(j_history)

# Plot a subgraph with Population vs Profit for the prediction on the training data
def plotPopulationVsProfitPrediction():

    #Return evenly spaced numbers over a specified interval.
    x = numpy.linspace(data.Population.min(), data.Population.max(), 100)
    # f(x) = Oo + O1x
    function = theta_prediction[0,0] + theta_prediction[1,0]*x
    # Sub plot returns figure object fig, and ax : Axes object or array of Axes objects.
    fig, ax = plt.subplots(figsize=(12,8))

    # plot x-axis, y-axis, color, series of attributes. Here we use label
    ax.plot(x, function, 'r', label='Prediction')
    # scatter : x-axis, y-axis, series of attributes. Here we use label
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')
    return;

# plotPopulationVsProfitPrediction()

def plotIterationEpoch():


    fig, ax = plt.subplots(figsize=(12,8))
    #print(j_history)
    #Return evenly spaced values within a given interval. Not sure why linspace didn't work here
    ax.plot(numpy.arange(num_iters), j_history, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    return;
# plotIterationEpoch()


def contourGraph():

         #Grid over which we will calculate J. Calculate J for a fixed set of
         #theta from the below specified range. This range was given in the Andrew Ng course Exercise pre-definitions
         # file name ex1.m
         # Create a matrix of result generated with the X and Y (here Oo and O1. J is size of Oo X Size O1)
     theta0_vals = numpy.linspace(-10, 10, 100)
     theta1_vals = numpy.linspace(-1, 4, 100)
     J_vals = numpy.zeros(shape=(theta0_vals.size, theta1_vals.size))
     #Fill out J_vals
     for t1, element in enumerate(theta0_vals):
         for t2, element2 in enumerate(theta1_vals):
             thetaT = numpy.zeros(shape=(2, 1))
             thetaT[0][0] = element
             thetaT[1][0] = element2
             J_vals[t1, t2] = computeCost(X, y, thetaT)


     #theta0_history_matrix = numpy.matrix(theta_history[:,0])
     #theta1_history_matrix = numpy.matrix(theta_history[:,1])
     #plt.contour(theta0_history_matrix, theta1_history_matrix, J_vals, numpy.logspace(-2, 3, 20))
     plt.contour(theta0_vals, theta1_vals, J_vals, numpy.logspace(-2, 3, 20))
     plt.title('Contour Plot')
     plt.xlabel('theta_0')
     plt.ylabel('theta_1')
     plt.scatter(theta[0,0], theta[1,0])
     plt. show()
     return;

#contourGraph()

# Using a small implementation of SciKit to the above Linear Regression, Job and gradient descent
# calculation using scikit libraries

def sciKitLinearRegression():

     #
    model = linear_model.LinearRegression()
    # Fits a linear Regression Model. Without the fit() you will get the following error
    # "NotFittedError: This LinearRegression instance is not fitted yet.
    #Call 'fit' with appropriate arguments before using this method."
    model.fit(X, y)
    # A1 returns flattened array.
    #Below, it means take the 2nd column (x values) and flatten to one row
    #print(numpy.array(X[:, 1].A1))
    x=numpy.array(X[:, 1].A1)
    prediction=model.predict(X)
    f = prediction.flatten()
    # Interestingly if you divide mean squared error by 2, you get the
    # The lowest difference between the hypothesis and the actual y
    # which 4.48 calculated by manually implementing gradient descent in the above functions
    print("Mean squared error: %.2f"
      % mean_squared_error(y, prediction))
    # The coefficients which is theta1 in Oo + O1x
    print('Coefficients: \n', model.coef_)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y, prediction))
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data.Population, data.Profit, label='Traning Data')
    ax.legend(loc=2)
    ax.set_xlabel('Population')
    ax.set_ylabel('Profit')
    ax.set_title('Predicted Profit vs. Population Size')

sciKitLinearRegression()
