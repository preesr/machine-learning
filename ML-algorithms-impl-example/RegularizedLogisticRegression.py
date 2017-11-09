"""
implement regularized logistic regression
to predict whether microchips from a fabrication plant passes quality assurance (QA)


@author: Preethi Srinivasan
References: Andrew Ng,  github jdwittenauer, Quora, scipy.org, scikit-learn.org
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import os
path = os.getcwd() + '/data/ex2data2.txt'
data = pd.read_csv(path, header=None, names=['Microchip Test 1', 'Microchip Test 2', 'Accepted'])
print(data.head())

def plotData():
    # Returns all the rows in data with Admitted as 1. Read as 'is in'
    positive = data[data['Accepted'].isin([1])]
    negative = data[data['Accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    # scatter is taking the following parameters x-axis, y-axis, size of the dots, marker
    # and label whichis the data that later comes up in the legend when legend is invoked
    ax.scatter(positive['Microchip Test 1'], positive['Microchip Test 2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Microchip Test 1'], negative['Microchip Test 2'], s=50, c='r', marker='x', label='Not Accepted')
    ax.legend()
    ax.set_xlabel('Microchip Test 1')
    ax.set_ylabel('Microchip Test 2')
    return;

#plotData()

degree = 6
x1 = data['Microchip Test 1']
x2 = data['Microchip Test 2']


#==============================================================================
# Figure shows that our dataset cannot be separated into positive and

# negative examples by a straight-line through the plot. Therefore, a straight-

# forward application of logistic regression will not perform well on this dataset

# since logistic regression will only be able draw linear decision boundary.

# One way is data better is to create more features from each data

# point. In the provided function mapFeature.m, we will map the features into

# all polynomial terms of x1 and x2 up to the sixth power.

# Here Ones are inserted at the end of data unlike others because we are

#computing the polynomial terms of higher degree which itself will become the data

# and appended after the ones and we will remove plain vanilla existing 2 features

# from the front, so 'Ones' will automatically come to the first col after y

# is removed from the front. F10 is Microchip Test 1 data i.e x1*0 where 0 is x2

# No no loss of data. Only adding new polynomial data

# By adding all polynomial combinations of x1 and x2 as features, We are basically telling
# that they I don't know what the best equation is,so you adjust the theta for the feature
#such that if it is not worth it, you assign say 0 to that polynomial masked as a feature
#==============================================================================
data.insert(3, 'Ones', 1)
#print(data.head())

# Correct way of calculating polynomial features

for i in range(1, degree+1):
    for j in range(0, i+1):
        #print(i-j,":",j)
        data['F' + str(i-j) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
#Return new object with labels in requested axis removed.
#axis=1 denotes that we are referring to a column, not a row
data.drop('Microchip Test 1', axis=1, inplace=True)
data.drop('Microchip Test 2', axis=1, inplace=True)

print(data.head())



def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def checkSigmoid():
    nums = np.arange(-10, 10, step=1)

    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(nums, sigmoid(nums), 'r')
    return;
#checkSigmoid()

# Implementing the formula in pdf ex2.pdf page 9 for cost function
def costReg(theta, X, y, lambda_val):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    # excluding theta0 from the formula like in page 9
    reg = (lambda_val/(2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return (np.sum(first - second) / (len(X)) + reg)




      # Implementing the formula in pdf ex2.pdf page 9 and 10 for gradient.
    #Note: This is only gradient, not gradient descent. You will be using optimal functions
    # to calculate gradient descent instead of iteratively manually converging

    # The reason we use a for loop hear is the formula is using the ith feature unlike the
    # previous cost function
    # Take a look at the formula in the above mentioned PDF file

# The reason you don't use a learning rate alpha here directly is because I am not calculating the
#Gradient descent. I am just calculating the gradient. Hence no need of learning rate in my implementation
# Nevertheless,  learning rate is needed for logistic regression and that is
# determined by the fmin_tnc and scikit learn algorithms as per definition of the optimization algorithms mentioned in the videos
def gradientReg(theta, X, y, lambda_val):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    #Ravel Return a contiguous flattened array.

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:,i])
    # Conditional check for 0 because we are not penalizing Oo as per the videos
    # and the pdf ex2.pdf page 9
        if (i ==0 ):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((lambda_val/len(X)) * theta[:,i])

    return grad

# set X and y (remember from above that we moved the label to column 0)
cols = data.shape[1]
X = data.iloc[:,1:cols]
y = data.iloc[:,0:1]


# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
y = np.array(y.values)

theta = np.zeros(cols-1)
lambda_default = 1
# O/P comes correctly Cost: 0.69314718056
#gradient: [  8.47457627e-03   1.87880932e-02 ...
# Manual calculation accuracy = 83%
# Testing
#theta = np.ones(cols - 1)
#lambda_default = 10
# O/P comes correctly : Cost: 3.16450933162
#gradient: [ 0.34604507  0.16135192  0.19479576....

cost_val = costReg(theta, X, y,lambda_default)
print("Cost:", cost_val)


gradient_val = gradientReg(theta, X, y, lambda_default)
print ("gradient:", gradient_val)


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

result = opt.fmin_tnc(func=costReg, x0=theta,
                      fprime=gradientReg, args=(X, y, lambda_default))

print(result)

def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]

def calculateProbability(theta, X):
    probability = sigmoid(X * theta.T)
    return probability

def accuracyOfTheta(theta, X):
    theta_min = np.matrix(result[0])
    predictions = predict(theta_min, X)
    # zip - Make an iterator that aggregates elements from each of the iterables.
    correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0
           for (a, b) in zip(predictions, y)]
    # map here is for converting correct to int.
    #accuracy = (sum(map(int, correct)) % len(correct))
    # total (correct/total) * 100 gives accuracy.
   #  accuracy = (sum(correct) / len(correct)) * 100
    accuracy = (sum(correct) / len(correct)) * 100
    return accuracy

accuracy = accuracyOfTheta(theta, X)
print ('Manual calculation accuracy = {0}%'.format(round(accuracy)))

# Instead of doing all this computation, use SciKit
#==============================================================================
# penalty : str, ‘l1’ or ‘l2’, default: ‘l2’
# Used to specify the norm used in the penalization.
# The ‘newton-cg’, ‘sag’ and ‘lbfgs’ solvers support only l2 penalties.
# Like in support vector machines, smaller values specify stronger regularization.
# https://www.quora.com/In-scikit-learn-logistic-regression-what-are-l1-and-l2-values
# mean square is l2. So Typical linear regression (L^2) minimizes the sum of squared errors,
# so being off by +4 is 16 times worse than being off by +1.
# L2 is Ridge : Ridge Regression is a technique for analyzing multiple regression
# data that suffer from multicollinearity. When multicollinearity occurs,
# least squares estimates are unbiased,
# but their variances are large so they may be far from the true value.

# L1 is Lasso: In statistics and machine learning, lasso (least absolute shrinkage
# and selection operator) (also Lasso or LASSO) is a regression analysis method that performs
# both variable selection and regularization in order
# to enhance the prediction accuracy and interpretability of the statistical model it produces.

# C : float, default: 1.0
# Inverse of regularization strength; must be a positive float.
# My understanding of C is it is the Lambda that regularizes the logistic regression


#==============================================================================

model = linear_model.LogisticRegression(penalty='l2', C=1)
# Fits a linear Regression Model. Without the fit() you will get the following error
# "NotFittedError: This LinearRegression instance is not fitted yet.
#Call 'fit' with appropriate arguments before using this method."
model.fit(X, y.ravel())
print("Scikit Accuracy score:",model.score(X, y))
# When I increased C to do parameter tuning in power of 10, accuracy largely increased
# Andrew's course says large lambda causes
#undefitting as it heavily penalizes the theta making them almost zero.
# Very low lambda causes overfitting as it does not penalize theta enough to regularize
# Predict class labels for samples in X.
prediction=model.predict(X)
# The below f is used for plotting functions if you like
f = prediction.flatten()
    # The difference between the hypothesis and the actual y
print("Mean squared error: %.2f"
      % mean_squared_error(y, prediction))
    # The coefficients which is O1,O2 in Oo + O1x1, O2x2..
coefficients = model.coef_
print('Coefficients: \n', coefficients)
    # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, prediction))

# print(result[0])
#
# def predict(testX, theta):
#     #probability = np.matrix(np.zeros(testX.shape[0]))
#     probability=sigmoid(testX * theta.T)
#     return probability
#
# theta_predicted = np.matrix(np.asarray(result[0]))
# probability= predict(np.matrix([[1, 45, 85]]), theta_predicted)
# print("probability:")
# print(probability)
#==============================================================================


def plotDecisionBoundary():
    # Returns all the rows in data with Admitted as 1. Read as 'is in'
    positive = data[data['Accepted'].isin([1])]
    negative = data[data['Accepted'].isin([0])]

    # scatter is taking the following parameters x-axis, y-axis, size of the dots, marker
    # and label whichis the data that later comes up in the legend when legend is invoked
    plt.scatter(positive['F10'], positive['F01'], s=50, c='b', marker='o', label='Accepted')
    plt.scatter(negative['F10'], negative['F01'], s=50, c='r', marker='x', label='Not Accepted')
    plt.legend()
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')

# # this plot is for the decision boundary line
    x1 = np.linspace(-1,1,100)
    x2 = np.linspace(-1,1,100)

    Jv = np.empty((100,100))

    for i,v1 in enumerate(x1):
        for j,v2 in enumerate(x2):
            v = np.array([v1,v2])
            # the dimensions don't match in calculating the grid.
            #Hence this function is not working
            Jv[i,j] = calculateProbability(result[0],v)
    plt.contour(x1,x2,Jv,[0.5])
    plt.plot(x1, x2, 'r', label='Decision Boundary')

    return;
#plotDecisionBoundary()
#==============================================================================


