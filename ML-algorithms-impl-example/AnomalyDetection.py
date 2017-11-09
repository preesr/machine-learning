#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Preethi Srinivasan

Implement an anomaly detection algorithm to detect anomalous behavior in server computers.The features measure the throughput (mb/s) and latency (ms) of response of each server.

References: Andrew Ng, github jdwittenauer, aqibsaeed, scipy.org, scikit-learn.org

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
#from scipy.stats import multivariate_normal
from sklearn import svm

raw_data = loadmat('data/ex8_data/ex8data1.mat')
X = raw_data['X']
m = X.shape[0] # 307 examples of throughput (mb/s) and latency (ms)

Xval = raw_data['Xval'] # (307L, 2L),
yval = raw_data['yval'] #(307L, 1L)
#print("Xval,yval", Xval.shape, yval.shape)

#print(raw_data.keys()) # Contains 'X' , Xval which is m x 2 and 'yval'.

#==============================================================================
# covariance = np.cov(X.T)
#
# print ("covariance of X",covariance , covariance.shape)
#==============================================================================
def plotInitialDataSet(x):
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(x[:,0], x[:,1], s=30, color='b', label='Dataset 1')
    return

#plotInitialDataSet(X)

def estimate_gaussian(X):
    mu = np.mean(X,axis=0)
    sigma = np.var(X,axis=0)# variance is square of Standard Deviation

    return mu, sigma

mu, sigma = estimate_gaussian(X) # mu (2,) sigma (2,)
#print("mu, sigma",mu, sigma)

# You can calculate the Probability Density Function Using SciKit.
# stats.norm gives the distribution: A normal continuous random variable. .pdf() gives the probability density function. So we now have the probability value of each value in dataset Xval.

pval = np.zeros((Xval.shape[0], Xval.shape[1]))
#pval = multivariate_normal(mean=mu).pdf(Xval) f1 score is worse with this

for i in range(Xval.shape[1]):
    pval[:,i] = stats.norm(mu[i], sigma[i]).pdf(Xval[:,i]) # PDF for feature i


# pval shape : (307L, 2L)

# Select the best threshhold epsilon. If p is below epsilon then it is an anomaly.select the best epsilon based on the F1 score.

def selectThreshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        prediction = (pval < epsilon) #* 1 # Without * 1 it was showing True False. when you do boolean * 1 it will give you 0 or 1
        #print("prediction",prediction, prediction.shape) # (307, 2)
#==============================================================================
#         prediction [[ True False]
#         [ True  True]....
#==============================================================================
#==============================================================================
#         tp is the number of true positives: the ground truth label says it's an
# anomaly and our algorithm correctly classified it as an anomaly.
# fp is the number of false positives: the ground truth label says it's not
# an anomaly, but our algorithm incorrectly classified it as an anomaly.
# fn is the number of false negatives: the ground truth label says it's an
# anomaly, but our algorithm incorrectly classified it as not being anoma-
# lous.
#==============================================================================
        tp = np.sum(np.logical_and(prediction == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(prediction == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(prediction == 0, yval == 1)).astype(float)
        #print("tp,fp,fn",tp,fp,fn)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)
        # RuntimeWarning: invalid value encountered in double_scalars. Somewhere it is divided by 0 causing this runtime error
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1



epsilon, f1= selectThreshold(pval,yval)
print("epsilon, f1",epsilon, f1)
# (0.0095667060059568421, 0.7142857142857143)
# you should see a value epsilon of about 8.99e-05
# you should see a Best F1 value of  0.875000

# Get the prediction for data set X
p = np.zeros((X.shape[0], X.shape[1]))
for i in range(X.shape[1]):
    p[:,i] = stats.norm(mu[i], sigma[i]).pdf(X[:,i]) # PDF for feature i
    # p shape is 307,2

def plotOutliers(p,epsilon):
    # indexes of the values considered to be outliers
    outliers = np.where(p < epsilon) #size: 2 tuple
    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(X[:,0], X[:,1])
    # because outlier is tuple of size 2
    print("outliers[0]",outliers[0])
    ax.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')
    return;

plotOutliers(p,epsilon)

raw_data2 = loadmat('data/ex8_data/ex8data2.mat')
#print("raw_data2",raw_data2.keys()) 'X', 'Xval', 'yval'

X2 = raw_data2['X']
m2 = X2.shape[0] # 1000

Xval2 = raw_data2['Xval'] # (100L, 11L),
yval2 = raw_data2['yval'] #(100, 1)

mu2, sigma2 = estimate_gaussian(X2)
print("mu2, sigma2",mu2, sigma2)


pval2 = np.zeros((Xval2.shape[0], Xval2.shape[1]))


for i in range(Xval2.shape[1]):
    pval2[:,i] = stats.norm(mu2[i], sigma2[i]).pdf(Xval2[:,i]) # PDF for feature i

epsilon2, f12= selectThreshold(pval2,yval2)
print("epsilon2, f12",epsilon2, f12)

p2 = np.zeros((X2.shape[0], X2.shape[1]))
for i in range(X2.shape[1]):
    p2[:,i] = stats.norm(mu2[i], sigma2[i]).pdf(X2[:,i]) # PDF for feature i

outliers2 = np.where(p2 < epsilon2)
print("No. of anomalies", len(outliers2[0]))
#==============================================================================
# epsilon2, f12 0.012420382026 0.1875
# No. of anomalies 10026
# You should see a value epsilon of about 1.38e-18, and 117 anomalies found.

#==============================================================================
# OneClassSVM Unsupervised Outlier Detection.
# Estimate the support of a high-dimensional distribution.
# nu - An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors. Should be in the interval (0, 1]. By default 0.5 will be taken.
#==============================================================================

clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf.fit(X)
pred = clf.predict(X)
normal = X[pred == 1]
abnormal = X[pred == -1]

def plotsciKitAnomalyDetection():
    plt.figure()
    plt.plot(normal[:,0],normal[:,1],'bo')
    plt.plot(abnormal[:,0],abnormal[:,1],'ro')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()
    return;

plotsciKitAnomalyDetection()


clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
clf.fit(X2)
pred = clf.predict(X2)
normal = X2[pred == 1]
abnormal = X2[pred == -1]

print("Large dataset anomalies using scikit",len(abnormal))

