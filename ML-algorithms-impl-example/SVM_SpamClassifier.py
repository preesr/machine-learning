#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Support vector machines

Build Spam Classifier using SVMs

@author: Preethi

References: Andrew Ng,  github jdwittenauer,rragundez, scipy.org, scikit-learn.org
Refer ex6.pdf Part 2 from Page 10 to understand how to create a feature set by converting each email into a row in x feature set
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
from sklearn.svm import SVC

spam_train = loadmat('data/ex6_data/spamTrain.mat')
spam_test = loadmat('data/ex6_data/spamTest.mat')

X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y'].ravel()
ytest = spam_test['ytest'].ravel()

print("Dimensions of X.shape, y.shape, Xtest.shape, ytest.shape:",X.shape, y.shape, Xtest.shape, ytest.shape)
# (4000, 1899) (4000,) (1000, 1899) (1000,)
# Which means 4000 emails were used as samples for training. Each email was checked for 1899 stemmed words in vocab.txt and marked 0 or 1 if the word was present in vocab.txt or not respectively. vocabulary list was selected by choosing all words which occur at least a 100 times in the spam corpus, resulting in a list of 1899 words

svc = svm.SVC()
#Fit the model according to the given training data. It uses default parameters SVC()
svc.fit(X,y)

print('Training accuracy = {0}%'.format(np.round(svc.score(X, y) * 100, 2)))

print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100, 2)))
#Training accuracy = 94.4%
#Test accuracy = 95.3%

# If I give C =100 and gamma as 10 it overfits and testing does not have a natural fit
#Training accuracy = 100.0%
#Test accuracy = 79.8%