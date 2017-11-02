"""
Implement linear regression with multiple variables to predict the prices of houses.

@author: Preethi Srinivasan
References: Andrew Ng,  github jdwittenauer, scipy.org, scikit-learn.org
"""
import numpy
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


alpha = 0.01
num_iters=1500

# Convert array of zeros to matrix

#print (theta.T)
path = os.getcwd() + '/data/ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size','Bedrooms', 'Price'])


# # insert Xo a column of 1s

# # Size of the training set


#==============================================================================

def featureNormalize():
    # Gets the mean and Standard deviation among all the data row and cols in the data frame
    data_normalized = (data2 - data2.mean()) / data2.std()
    return data_normalized;

data_normalized=featureNormalize()

print(data_normalized.head())
data_normalized.insert(0,'Ones', 1)
cols = data_normalized.shape[1]
X = data_normalized.iloc[:,0:cols-1]
print(X.head())
y = data_normalized.iloc[:,cols-1:cols]
 #Convert DataFrame to Matrix
X = numpy.matrix(X.values)
y = numpy.matrix(y.values)
theta = numpy.matrix(numpy.zeros((cols-1,1)))
print("theta")
print(theta)
# m = len(X) # No. of training samples
j_history = numpy.zeros((num_iters,1))

#==============================================================================
def computeCost(X,y,theta):
     # ((Oo + O1x.....for multivariable) - y) ^ 2
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
# # value of theta after gradient decent. The best values for theta
print ("value of theta after gradient decent. The best values for theta:")
print(theta_prediction)
# # value of the cost function with the new theta. The lowest difference between the hypothesis and the actual y
print ("value of the cost function with the new theta. The lowest difference between the hypothesis and the actual y")
print(computeCost(X,y,theta_prediction))


def plotIterationEpoch():
    fig, ax = plt.subplots(figsize=(12,8))
    #Return evenly spaced values within a given interval. Not sure why linspace didn't work here
    ax.plot(numpy.arange(num_iters), j_history, 'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.set_title('Error vs. Training Epoch')
    return;
#plotIterationEpoch()


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
    # The coefficients which is O1,O2 in Oo + O1x1, O2x2..
    print('Coefficients: \n', model.coef_)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y, prediction))
    # Technically you need a 3D graph. For simplicity plotting only Size and Price
    # to see if the prediction fits
    fig, ax = plt.subplots(figsize=(12,8))
    ax.plot(x, f, 'r', label='Prediction')
    ax.scatter(data_normalized.Size, data_normalized.Price, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Size')
    ax.set_ylabel('Price')
    ax.set_title('Size vs. Price')

#sciKitLinearRegression()

# Find Value of the Thetas
def normalEquation():
    print("Theta values by Normal Equation:")
    print ((X.T * X).I * X.T * y)
    return;
normalEquation()

