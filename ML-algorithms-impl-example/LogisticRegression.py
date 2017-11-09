"""
build a logistic regression model to predict whether a student gets admitted into a university.


@author: Preethi Srinivasan
References: Andrew Ng,  github jdwittenauer,scipy.org
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt

import os
path = os.getcwd() + '/data/ex2data1.txt'
data = pd.read_csv(path, header=None, names=['Exam 1', 'Exam 2', 'Admitted'])
data.head()



def plotData():
    # Returns all the rows in data with Admitted as 1. Read as 'is in'
    positive = data[data['Admitted'].isin([1])]
    negative = data[data['Admitted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    # scatter is taking the following parameters x-axis, y-axis, size of the dots, marker
    # and label whichis the data that later comes up in the legend when legend is invoked
    ax.scatter(positive['Exam 1'], positive['Exam 2'], s=50, c='b', marker='o', label='Admitted')
    ax.scatter(negative['Exam 1'], negative['Exam 2'], s=50, c='r', marker='x', label='Not Admitted')
    ax.legend()
    ax.set_xlabel('Exam 1 Score')
    ax.set_ylabel('Exam 2 Score')
    return;

#plotData()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def checkSigmoid():
    nums = np.arange(-10, 10, step=1)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(nums, sigmoid(nums), 'r')
    return;
#checkSigmoid()

# Implementing the formula in pdf ex2.pdf page 4 for cost function
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))


# add a ones column . this makes the matrix multiplication work out easier
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)
initial_theta = np.zeros(cols-1)

cost(initial_theta, X, y)


      # Implementing the formula in pdf ex2.pdf page 5 for gradient.
    #Note: This is only gradient, not gradient descent. You will be using optimal functions
    # to calculate gradient descent instead of iteratively manually converging

    # The reason we use a for loop hear is the formula is using the ith feature unlike the previous cost function
    # Take a look at the formula in the above mentioned PDF file
def gradient(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    #Ravel Return a contiguous flattened array.
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:,i])
        grad[i] = np.sum(term) / len(X)

    return grad
gradient(initial_theta, X, y)

# Usingn optimal algorithm with parameters as follows
## func = function that calculates the job and gradient descent.
#           Return f and g, where f is the value of the function and g its gradient (a list of floats).
## x0= Initial estimate of minimum.
## fprime = Gradient of func
## args = X and y values here
#Function invocation in my LogisticRegression based on Matrices instead of Array
#throwing : ValueError: shapes (3,) and (100,1) not aligned: 3 (dim 0) != 100 (dim 0)
#I used all implementation via array and did conversion to matrix in the function
# That seems to help the optimization scipy function to work
#Returns:
#x : ndarray
#The solution.
#nfeval : int
#The number of function evaluations.
#rc : int
#Return code as defined in the RCSTRINGS dict.
# No need to pick alpha learning rate as by definition optimization algorithms do it for you
result = opt.fmin_tnc(func=cost, x0=initial_theta, fprime=gradient, args=(X, y))
# O/P: (array([-25.16131863,   0.20623159,   0.20147149]), 36, 0)
print(result)
print(result[0])

def predict(testX, theta):
    #probability = np.matrix(np.zeros(testX.shape[0]))
    probability=sigmoid(testX * theta.T)
    return probability

theta_predicted = np.matrix(np.asarray(result[0]))
# You have found out the best values for theta for this problem. Hence, the application is here.
# Given a new data, you are now predicting the outcome
probability= predict(np.matrix([[1, 45, 85]]), theta_predicted)
print("probability:")
print(probability)
# TO DO: Revisit building decision boundary if you have time

