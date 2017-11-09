"""
@author: Preethi Srinivasan

implement regularized linear regression and use it to
study models with different bias-variance properties.

implement regularized linear regression to predict the amount of water flowing out of a dam using the change of water level in a reservoir

Also plot training and test errors on a
learning curve to diagnose bias-variance problems.

Use scikit learn as shown below for polynomiallinearRegression

References: Andrew Ng, github JWarmenhoven, scipy.org, scikit-learn.org

"""
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
#from scipy.optimize import minimize
from scipy.optimize import fmin_cg

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

# Returns a dict called data
data = loadmat('data/ex5data1.mat')

print(data.keys())
# 'X', 'y', 'Xtest', 'ytest', 'Xval', 'yval'

X_train = data['X']

y_train = data['y']

m_train = X_train.shape[0]
print("m_train",m_train)
# 12
X_crossval = data['Xval']
y_crossval = data['yval']
m_crossval = X_crossval.shape[0]
print("m_crossval:",m_crossval)
# 21
m_test = data['Xtest'].shape[0]
print("m_test:",m_test)
#21

y_test = data['ytest']
# Plot a subgraph with Population vs Profit for the prediction on the training data
def plotWaterLevelChangeVsWaterFlow():

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(X_train, y_train, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Change in Water Level ')
    ax.set_ylabel('Water Flowing out of the dam')
    ax.set_title('Water Level vs. Water Flow')
    return;

#plotWaterLevelChangeVsWaterFlow()


def computeCost(theta, X, y, lambda_val):
    m = y.shape[0]
        #ex5.pdf page 4
    # h(x) = ((Oo + O1x) - y) ^ 2
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    inner_equation =  np.power(((X * theta.T) - y),2)
    # the mean sum
    # Excluding theta0 for calculating weight
    cost = (np.sum(inner_equation)/(2*m)) + ((lambda_val/2 * m) * np.sum(np.power(theta[1:],2)))
    return cost


def computeGradient(theta, X, y, lambda_val):
    m = y.shape[0]
     #ex5.pdf page 4
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
 # (h(x) - y)
    error = (X * theta.T) - y
     #(1/m) * X.T * (h(x) - y) + reggularization
    grad = ((X.T*error) / m).T + ((lambda_val/m) * theta)
         # because we are not penalizing Oo with regularization as per the videos. Hence, update term
         # without regularization
    grad[0,0] = np.sum(np.multiply(error, X[:,0])) / m
    return np.array(grad).ravel()



lamda = 0
# insert Xo a column of 1s
X_train = np.insert(X_train, 0, values=np.ones(m_train), axis=1)
initial_theta= np.ones(X_train.shape[1])
print("initial_theta.shape",initial_theta.shape)
#initial_theta.shape (2,)
print("X_train.shape",X_train.shape)
# X_train.shape (12, 2)
print(computeCost(initial_theta,X_train,y_train,lamda))
# 303.951525554
print(computeGradient(initial_theta,X_train,y_train,lamda))
# [ -15.30301567  598.16741084]

# full fill the array with the number you want. Here I have specified 0
#print(np.full((X_train.shape[1]), 0))
#[0 0]
#print(np.zeros(X_train.shape[1]))
#[ 0.  0.]

def linearRegression(X,y,lambda_val):
    #initial_theta = np.zeros(X.shape[1])
    initial_theta=np.full((X.shape[1]), 0)
    #initial_theta= np.ones(X.shape[1])
    # Using initial theta of ones doesn't converge while calculating the learning curve, I get "Warning: Desired error not necessarily achieved due to precision loss." Using initial theta of zeros doesn't throw such errors. I tried with initial theta being 2 etc. it works correctly. But just not for value 1
    result = fmin_cg(computeCost,initial_theta,fprime = computeGradient,args=(X,y,lambda_val), maxiter = 200,disp = 1)
    return result
    #result = minimize(fun=computeCost, x0=initial_theta, args=(X, y, lambda_val), method='TNC', jac=computeGradient,options={'maxiter': 200})
    # print 'train_linear_reg:', result.success

    #return result.x

result = linearRegression(X_train,y_train,lamda)
print("result:",result)
#result [ 13.08790734   0.36777925]

def plotWaterFlowTrainPrediction():

    fig, ax = plt.subplots(figsize=(12,8))
    x = np.linspace(data['X'].min(), data['X'].max(), 100)
    # f(x) = Oo + O1x
    function = result[0] + result[1]*x
    # Sub plot returns figure object fig, and ax : Axes object or array of Axes objects.


    # plot x-axis, y-axis, color, series of attributes. Here we use label attribute
    ax.plot(x, function, 'r', label='Prediction')
    ax.scatter(data['X'], y_train, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Change in Water Level ')
    ax.set_ylabel('Water Flowing out of the dam')
    ax.set_title('Water Level vs. Water Flow')
    return;

#plotWaterFlowTrainPrediction()
#a learning curve plots training and cross validation error as a function of training set size.
# note that the training error does not include the regular-ization term. Hence pass 0 for lambda

def learningCurveErrorCalc(x_train, y_train, x_crossval, y_crossval, lambda_val):
    m = y_train.shape[0]
    error_train = np.zeros((m, 1))
    error_crossval = np.zeros((m, 1))
    for i in range(m):

        # You are calculating error and result for a training set of different sizes i.e m
        # i+1 denotes till which row you are calculating.i+1 is 1 then 1st row is returned
        # when I did just i infinity was thrown as m is 0 and it tries to divide the cost and gradient functions by 0

#==============================================================================
# A = np.matrix( [[1,2,3],[11,12,13],[21,22,23]])
#
# print(A[:3])
# [[ 1  2  3]
#  [11 12 13]
#  [21 22 23]]
#
# print(A[:2])
# [[ 1  2  3]
#  [11 12 13]]
#
# print(A[:1])
# [[1 2 3]]
#
# print(A[:0])
#==============================================================================
        resultval = linearRegression(x_train[:i+1],y_train[:i+1],lambda_val)
        error_train[i] = computeCost(resultval, x_train[:i+1], y_train[:i+1], lambda_val)
        # Here you just use the theta you got using training set to calculate error for the whole crossval data set unlike the training set ex5.pdf page 6
#==============================================================================
#         #When you are computing the training set error,
# make sure you compute it on the training subset (i.e., X(1:n,:) and y(1:n))
# (instead of the entire training set). However, for the cross validation error,
# you should compute it over the entire cross validation set.
#==============================================================================
        error_crossval[i] = computeCost(resultval, x_crossval, y_crossval, lambda_val)
    return (error_train, error_crossval)

X_crossval = np.insert(X_crossval, 0, values=np.ones(m_crossval), axis=1)
#t_error, v_error = learningCurveErrorCalc(X_train, y_train, X_crossval, y_crossval, 0)

#==============================================================================
# Understanding high bias and high variance:
#     https://www.coursera.org/learn/machine-learning/supplement/79woL/learning-curves
#     Low training set size: causes Jtrain(Θ) to be low and JCV(Θ) to be high.
#
# Large training set size: causes both Jtrain(Θ) and JCV(Θ) to be high with Jtrain(Θ)≈JCV(Θ).
#
# If a learning algorithm is suffering from high bias, getting more training data will not (by itself) help much.
#
# Experiencing high variance:
#
# Low training set size: Jtrain(Θ) will be low and JCV(Θ) will be high.
#
# Large training set size: Jtrain(Θ) increases with training set size and JCV(Θ) continues to decrease without leveling off. Also, Jtrain(Θ) < JCV(Θ) but the difference between them remains significant.
#
# If a learning algorithm is suffering from high variance, getting more training data is likely to help.

# In the below example it is high bias as error is high and Jtrain(Θ)≈JCV(Θ)
# If I increase lamda to 10, the error got even worse, i.e adding even more bias to a high bias model
#==============================================================================

def plotLearningCurve(t_error,v_error):
    plt.plot(np.arange(1,13), t_error, label='Training error')
    plt.plot(np.arange(1,13), v_error, label='Validation error')
    plt.title('Learning curve for linear regression')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend();
    return;

#plotLearningCurve(t_error,v_error)

#==============================================================================
# Polynomial features is correct. checked with the auto generated one with sci-kit
def polynomialFeatures(X, degree):
#     # array of m rows and degree cols
    X_poly = np.zeros((len(X),degree))
#     # Without flatten it gave me the error : ValueError: could not broadcast input array from shape (12,1) into shape (12)
    X = X.flatten()
    for i in range(1,degree+1):
        # The entire ith column is X to the power of i where X is always the initial set of given inputs - water level. So 0th column here is x, 1st column is x^2
#         # Note that you don't have to account for the zero-eth power in this function. ex5.pdf page 8
        X_poly[:,i-1] = np.power(X,i)
    return X_poly

def featureNormalize(X):
    # Gets the mean and Standard deviation among all the data row and cols in X
#==============================================================================
#     >>> a = np.array([[1, 2], [3, 4]])
# >>> np.mean(a)
# 2.5
# >>> np.mean(a, axis=0)
# array([ 2.,  3.])
# >>> np.mean(a, axis=1)
# array([ 1.5,  3.5])
#==============================================================================
# feature normalize without axis=0 in mean and standard deviation is throwing way too much error in learning curves
    X_normalized = (X - np.mean(X,axis=0)) / np.std(X,axis=0)
    return X_normalized;


degree = 8


# # Reading train, cross, test freshly from data itself to avoid confusion in already inserted X0 for previous variables. I will add X0 below for these X again
X_train1 = data['X']
X_polytrain = np.insert(featureNormalize(polynomialFeatures(X_train1,degree)), 0, values=np.ones(m_train), axis=1)
#
X_test1 = data['Xtest']
X_polytest = np.insert(featureNormalize(polynomialFeatures(X_test1,degree)), 0, values=np.ones(m_test), axis=1)
#
X_val1 = data['Xval']
X_polyval = np.insert(featureNormalize(polynomialFeatures(X_val1,degree)), 0, values=np.ones(m_crossval), axis=1)
lamda_poly = 3
theta_result_poly = linearRegression(X_polytrain,y_train,lamda_poly)
print("result_poly:",theta_result_poly)

# Test set evaluation for the theta_result_poly
print("test set error:",computeCost(theta_result_poly, X_polytest, y_test,lamda_poly))
# test set error:  5.69781967317 should be 3.8599 for  lambda = 3. page 13 ex5.pdf

# My own brute force implementation of graph prediction. Similar to plotFit
def plotWaterFlowTrainPredictionPoly():

    fig, ax = plt.subplots(figsize=(12,8))
    x = np.linspace(data['X'].min(), data['X'].max(), 100)
    # f(x) = Oo + O1x
    function = theta_result_poly[0] + theta_result_poly[1]*np.power(x,1) + theta_result_poly[2]*np.power(x,2) + theta_result_poly[3]*np.power(x,3) + theta_result_poly[4]*np.power(x,4) + theta_result_poly[5]*np.power(x,5) + theta_result_poly[6]*np.power(x,6) + theta_result_poly[7]*np.power(x,7) + theta_result_poly[8]*np.power(x,8)

    # Sub plot returns figure object fig, and ax : Axes object or array of Axes objects.


    # plot x-axis, y-axis, color, series of attributes. Here we use label
    ax.plot(x, function, 'r', label='Prediction')
    ax.scatter(data['X'], y_train, label='Training Data')
    ax.legend(loc=2)
    ax.set_xlabel('Change in Water Level ')
    ax.set_ylabel('Water Flowing out of the dam')
    ax.set_title('Water Level vs. Water Flow')
    return;

#plotWaterFlowTrainPredictionPoly()
#==============================================================================

def plotPolyFit(min_x, max_x, mu, sigma, theta, degree):
    theta = theta.flatten()
    fig1, ax1 = plt.subplots(figsize=(12,8))
#==============================================================================
#  reshape(-1,1)  The criterion to satisfy for providing the new shape is that 'The new shape should be compatible with the original shape' numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1)). It simply means that it is an unknown dimension and we want numpy to figure it out.
#==============================================================================
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape((-1,1))
    #print("x shape", x.shape)
    #x shape (2512, 1)
    # for the given x, xreating the polynomial
    X_poly = np.insert(featureNormalize(polynomialFeatures(x,degree)), 0, values=np.ones(x.shape[0]), axis=1)
    # calculating the function using the predicted theta in the equation. As this is calculated for the polynomial variable, using vectorization to calculate the prediction.it is the h(x) in ex5.pdf page 7 with the predicted theta

    y_function = np.dot(X_poly,theta)
    #print("X_poly.shape, theta.shape",X_poly.shape, theta.shape)
    #X_poly.shape, theta.shape (2512, 9) (9,)
    #y_function = X_poly * theta
    #print("y_function dot product shape",y_function.shape)
    # y_function dot product shape (2512,)
    #print("y_function * shape",y_fun.shape)
    # y_function * shape (2512, 9) hence I was getting 9 prediction lines in the graph
    ax1.plot(x,y_function,'r--',linewidth=2)
    #plt.plot(x,np.dot(X_poly,theta),'b--',linewidth=2)
    ax1.scatter(data['X'], y_train, label='Training Data')
    ax1.set_xlabel('Change in water level (x)')
    ax1.set_ylabel('Water flowing out of the dam (y)')

    return

#plotPolyFit(np.min(X_train), np.max(X_train), np.mean(X_train), np.std(X_train), theta_result_poly, degree)
# The prediction kind of fits. Not a huge disjoint of the pattern of the data and the line
t_error_poly,v_error_poly = learningCurveErrorCalc(X_polytrain, y_train, X_polyval, y_crossval, lamda_poly)
plotLearningCurve(t_error_poly,v_error_poly)
# the learning curve looks a lot like the Fig 5 page 9 ex5.pdf for lambda = 0. for lambda = 1 it doesn't look very much live the one given ex5.pdf. Not sure what the error is.

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape((-1,1))
    error_train = np.zeros(len(lambda_vec))
    error_val = np.zeros(len(lambda_vec))

    for i in range(len(lambda_vec)):
        reg_param = lambda_vec[i]
        est_theta = linearRegression(X,y,reg_param)
#==============================================================================
# 		#error_train[i] = (np.sum((np.dot(X,est_theta)-y)**2))/(2.0*m_train)
# 		#error_val[i] = (np.sum((np.dot(Xval,est_theta)-yval)**2))/(2.0*m_val)
#==============================================================================
        error_train[i] = computeCost(est_theta, X, y, reg_param)
        error_val[i] = computeCost(est_theta, Xval, yval, reg_param)
    plt.plot(lambda_vec, error_train, lambda_vec, error_val);
    plt.title('Selecting lambda using a cross validation set')
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('lambda')
    plt.ylabel('Error')
    plt.show()
    return

#validationCurve(X_polytrain,y_train,X_polyval,y_crossval)
# For this it shows best value of lambda being around 3 or 1. But the correct answer is 3
# My implementation of polynomial is not working perfectly well. But in real life, I will be using sci kit learn libraries below for computing linear regression polynomial model fitting


# Using sci-kit learn for Polynomial Linear Regression

def polynomialLinearRegressionSciKit(X_train):
    poly = PolynomialFeatures(degree=8)
    X_train_poly = poly.fit_transform(X_train[:,1].reshape(-1,1))

    regr2 = LinearRegression()
    regr2.fit(X_train_poly, y_train)

# alpha - Regularization strength. seems to be the lambda value
    regr3 = Ridge(alpha=20)
    regr3.fit(X_train_poly, y_train)
# Advantage of ridge regression over linear regressions:
#==============================================================================
#     it chooses the feature's estimates (ββ) to penalize in such a way that less influential features (Some features cause very small influence on dependent variable) undergo more penalization. In some domains, the number of independent variables is many, as well as we are not sure which of the independent variables influences dependent variable. In this kind of scenario, ridge regression plays a better role than linear regression.
#==============================================================================
# plot range for x
    plot_x = np.linspace(-60,45)
# using coefficients to calculate y
# intercept_ Independent term in the linear model. Looks like theta0 to me. Rest of the equation is Theta*X for the plots = sum of those. It is the equation in ex5.pdf page 4
    plot_y = regr2.intercept_+ np.sum(regr2.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)
    plot_y2 = regr3.intercept_ + np.sum(regr3.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)

    plt.plot(plot_x, plot_y, label='Scikit-learn LinearRegression')
    plt.plot(plot_x, plot_y2, label='Scikit-learn Ridge (alpha={})'.format(regr3.alpha))
    plt.scatter(X_train[:,1], y_train, s=50, c='r', marker='x', linewidths=1)
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial regression degree 8')
    plt.legend(loc=4);
    print("scikit theta values:",regr2.coef_)

polynomialLinearRegressionSciKit(X_train)

