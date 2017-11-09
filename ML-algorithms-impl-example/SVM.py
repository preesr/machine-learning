
"""
@author: Preethi Srinivasan


Support vector machines

Experimenting with data sets
Gaussian kernels with SVMs

References: Andrew Ng, github jdwittenauer, scipy.org, scikit-learn.org
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from sklearn.svm import SVC

raw_data = loadmat('data/ex6_data/ex6data1.mat')

#print(raw_data.keys())
#'X', 'y'. X has 2 columns of data
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])

data['y'] = raw_data['y']
print(data.head())

def plotInitialDataSet():
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive['X1'], positive['X2'], s=50,c='g', marker='x', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s=50,c='r', marker='o', label='Negative')
    ax.legend()
    return

#plotInitialDataSet()

def scikitSVCPlotAndAnalyse():
    #==============================================================================
    # LinearSVC - Linear Support Vector Classification.
    # Penalty parameter C
    # loss : string, ‘hinge’ or ‘squared_hinge’ (default=’squared_hinge’)
    # Specifies the loss function. ‘hinge’ is the standard SVM loss (used e.g. by the SVC class) while ‘squared_hinge’ is the square of the hinge loss.
    #==============================================================================
    svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
    #Fit the model according to the given training data.
    svc.fit(data[['X1', 'X2']], data['y'])
    #Returns the mean accuracy on the given test data and labels.

    print("Scikit Linear SVC accuracy score:",svc.score(data[['X1', 'X2']], data['y']))
    #0.98 for C=1. This loss of 2% is due to one sample point in X which is an outlier if you see in the graph
    # C=100000000 1.0. But this is not how we want to fit the data. It has high variance

    #decision_function(X) - Predict confidence scores for samples.
    data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
    ax.set_title('SVM Decision Confidence')
    # C = 1 outlier classified incorrectly which is a natural fit. C = 100000 outlier classified correctly but a higher variance fit
    return;

#scikitSVCPlotAndAnalyse()
# implement formula in ex6.pdf page 6
def gaussianKernel(x1,x2,sigma):
    return np.exp(-(np.sum((x1-x2)**2)/(2*(sigma**2))))

x1 = np.array([1.0, 2.0, 1.0])
x2 = np.array([0.0, 4.0, -1.0])
sigma = 2

print("Gaussian kernel for test data",gaussianKernel(x1, x2, sigma))
# Gaussian kernel for test data 0.324652467358

raw_data2 = loadmat('data/ex6_data/ex6data2.mat')

print(raw_data2.keys())
#'X', 'y'. X has 2 columns of data
data2 = pd.DataFrame(raw_data2['X'], columns=['X1', 'X2'])

data2['y'] = raw_data2['y']

print(data2.head())

def plotNonLinearDataSet2():
    positive2 = data2[data2['y'].isin([1])]
    negative2 = data2[data2['y'].isin([0])]

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(positive2['X1'], positive2['X2'], s=50,c='g', marker='x', label='Positive')
    ax.scatter(negative2['X1'], negative2['X2'], s=50,c='r', marker='o', label='Negative')
    ax.legend()
    return
#plotNonLinearDataSet2()

# Gaussian Kernel is also known as RBF kernel : the (Gaussian) radial basis function kernel, or RBF kernel, is a popular kernel function used in various kernelized learning algorithms.
# An equivalent, but simpler, definition involves a parameter gamma = 1/2* (sigma^2)
def scikitSVCPlotAndAnalyseNonLinear():
    #==============================================================================
    # SVC - Support Vector Classification.
    # Penalty parameter C
    #==============================================================================
    svc = svm.SVC(C=100, gamma=10,probability=True)
    #Fit the model according to the given training data.
    svc.fit(data2[['X1', 'X2']], data2['y'])
    #Returns the mean accuracy on the given test data and labels.

    print("Scikit Non Linear RBF SVC accuracy score:",svc.score(data2[['X1', 'X2']], data2['y']))

#==============================================================================
# predict_proba
# Compute probabilities of possible outcomes for samples in X.
# The model need to have probability information computed at training time: fit with attribute probability set to True.
#Returns the probability of the sample for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes_.
#The probability model is created using cross validation, so the results can be slightly different than those obtained by predict. Also, it will produce meaningless results on very small datasets
#==============================================================================
    #decision_function(X) - Predict confidence scores for samples.
    #data2['SVM 2 Confidence'] = svc.decision_function(data2[['X1', 'X2']])
    # predict_proba return n x 2. Has 2 columns. Hence get only the first column [:,0]
    data2['Probability'] = svc.predict_proba(data2[['X1', 'X2']])[:,0]
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(data2['X1'], data2['X2'], s=30, c=data2['Probability'], cmap='Reds')
    #ax.scatter(data2['X1'], data2['X2'], s=50, c=data2['SVM 2 Confidence'], cmap='seismic')
    ax.set_title('SVM Probability')
    # C = 1 outlier classified incorrectly which is a natural fit. C = 100000 outlier classified correctly but a higher variance fit
    return;

scikitSVCPlotAndAnalyseNonLinear()

# To find out optimal parameters for C and Gamma, use sci kit's model_selection.GridSearchCV http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
# Find the best C and Gamma values for DataSet 3 using the cross validation data

raw_data3 = loadmat('data/ex6_data/ex6data3.mat')

print(raw_data3.keys())
#'X', 'y', 'yval', 'Xval'
data3_X = pd.DataFrame(raw_data3['X'], columns=['X1', 'X2'])
# Without using ravel it gives an error message A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
data3_y = raw_data3['y'].ravel()

data3_valX = pd.DataFrame(raw_data3['Xval'], columns=['X1', 'X2'])
data3_valy = raw_data3['yval'].ravel()
#print(data3_valX.head())

def findBestParamDataSet3():
    C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]

    best_score = 0
    best_params = {'C': None, 'gamma': None}
    for c in C_values:
        for gamma in gamma_values:
            svc = svm.SVC(C=c, gamma=gamma)
            svc.fit(data3_X, data3_y)
            score = svc.score(data3_valX, data3_valy)

            if score > best_score:
                best_score = score
                best_params['C'] = c
                best_params['gamma'] = gamma
    return best_score, best_params

print("best_score, best_params",findBestParamDataSet3())
# best_score, best_params (0.96499999999999997, {'C': 0.3, 'gamma': 100})






