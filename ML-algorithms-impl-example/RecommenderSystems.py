#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Preethi Srinivasan

Implement collaborative
filtering learning algorithm and apply it to a dataset of movie ratings
enter your own movie preferences, you can get your own movie recommendations!

References: Andrew Ng, github jdwittenauer, scipy.org
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
#from scipy.optimize import fmin_cg
from scipy.optimize import minimize


#==============================================================================
#  This dataset consists of ratings on a scale of 1 to 5. The dataset has nu = 943 users, and nm = 1682 movies.
#==============================================================================
raw_data = loadmat('data/ex8_data/ex8_movies.mat')
Y = raw_data['Y']
R = raw_data['R']
#print(Y[1])

#print(raw_data.keys()) # 'Y', 'R'

#==============================================================================
# The matrix Y (a num movies x num users matrix) stores the ratings y(i;j)
# (from 1 to 5).0s in Y denotes not rated. The matrix R is an binary-valued indicator matrix, where
# R(i; j) = 1 if user j gave a rating to movie i, and R(i; j) = 0 otherwise. The
# objective of collaborative filtering is to predict movie ratings for the movies that users have not yet rated, that is, the entries with R(i; j) = 0. This will  allow us to recommend the movies with the highest predicted ratings to the user.
# Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 943 users
# R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a rating to movie i
# X is a nm x 100 matrix and Theta is a nu x 100 matrix.


#     X - num_movies  x num_features matrix of movie features
#     Theta - num_users  x num_features matrix of user features
#     Y - num_movies x num_users matrix of user ratings of movies
#     R - num_movies x num_users matrix, where R(i, j) = 1 if the
#             i-th movie was rated by the j-th user
#==============================================================================
#==============================================================================

# Showing data as an image to visualize this large data
def plotHeatMap():
    fig, ax = plt.subplots(figsize=(12,12))
    ax.imshow(Y)
    ax.set_xlabel('Users')
    ax.set_ylabel('Movies')
    fig.tight_layout()

#plotHeatMap()

def cofiCostFunc(params, Y, R, num_features,lambdaval):
    Y = np.matrix(Y)  # (1682, 943)
    R = np.matrix(R)  # (1682, 943)
    num_movies = Y.shape[0]
    num_users = Y.shape[1]
    J = 0;

#==============================================================================
#     X = reshape(params(1:num_movies*num_features), num_movies, num_features);
# Theta = reshape(params(num_movies*num_features+1:end), ...
#                 num_users, num_features);
#
#==============================================================================

    X = np.matrix(np.reshape(params[:num_movies * num_features], (num_movies, num_features)))
    theta = np.matrix(np.reshape(params[num_movies * num_features:], (num_users, num_features)))
    X_grad = np.zeros(X.shape);
    theta_grad = np.zeros(theta.shape);

    # Using formulae in ex8.pdf page 9,10
    error = np.multiply(((X * theta.T) - Y),R) # num_movies x num_users shape
    J = (1/2) * np.sum(np.power(error,2)) + ((lambdaval/2) * np.sum(np.power(theta,2))) + ((lambdaval/2) *
         np.sum(np.power(X,2)))
    X_grad = (error * theta) + (lambdaval * X)
    theta_grad = (error.T * X) + (lambdaval * theta)
    grad = np.concatenate((np.ravel(X_grad), np.ravel(theta_grad)))
#==============================================================================
# No need of theta 0. We are learning both the Theta weights and the features themselves. If the data set needs a bias unit feature for the lowest cost fit, then it will learn it. There is no reason to force adding one.
#==============================================================================
    return J, grad

def testDataSetUp(Y,R):
    params_data = loadmat('data/ex8_data/ex8_movieParams.mat')
    X = params_data['X']
    #print("X.shape", X.shape) # X.shape (1682, 10)
    Theta = params_data['Theta']
    #print("Theta.shape",Theta.shape) # Theta.shape (943, 10)

    num_users = 4
    num_movies = 5
    num_features = 3
    X = X[:num_movies,:num_features]
    Theta = Theta[:num_users, :num_features]
    Y = Y[:num_movies,:num_users]
    R = R[:num_movies, :num_users]
    params = np.concatenate((np.ravel(X), np.ravel(Theta)))

    print("test data cost function",cofiCostFunc(params, Y, R, num_features,1.5))
    #test data cost function 22.2246037257
    return;

testDataSetUp(Y,R)

# There was a weird error in reading the given data file: "UnicodeDecodeError: 'ascii' codec can't decode byte". I just stripped the text file off the rest of the name and kept only id and the first word. I comma seperated all the values, imported to a sheet, deleted rest of the rows. Downloaded the csv file and saved it as txt. The newfile is movie_ids_edited.txt
movie_list_file = open("data/ex8_data/movie_ids_edited.txt","r", encoding="utf-8")
n = 1682;
movie_idx = {}
for line in movie_list_file:
    idx, movieName = line.split(',',1)
    #print(idx)
    #print(movieName)
    # .rstrip() to remove the trailing nextline
    movie_idx[int(idx) - 1] = movieName.rstrip()

#print(movie_idx[1500])

ratings = np.zeros((n, 1))
# I assign ratings to movies in index 0,6,11 etc.
ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

print('Rated {0} with {1} stars.'.format(movie_idx[0], str(int(ratings[0]))))
print('Rated {0} with {1} stars.'.format(movie_idx[6], str(int(ratings[6]))))
print('Rated {0} with {1} stars.'.format(movie_idx[11], str(int(ratings[11]))))
print('Rated {0} with {1} stars.'.format(movie_idx[53], str(int(ratings[53]))))
print('Rated {0} with {1} stars.'.format(movie_idx[63], str(int(ratings[63]))))
print('Rated {0} with {1} stars.'.format(movie_idx[65], str(int(ratings[65]))))
print('Rated {0} with {1} stars.'.format(movie_idx[68], str(int(ratings[68]))))
print('Rated {0} with {1} stars.'.format(movie_idx[97], str(int(ratings[97]))))
print('Rated {0} with {1} stars.'.format(movie_idx[182], str(int(ratings[182]))))
print('Rated {0} with {1} stars.'.format(movie_idx[225], str(int(ratings[225]))))
print('Rated {0} with {1} stars.'.format(movie_idx[354], str(int(ratings[354]))))

Y = np.append(Y, ratings, axis=1)
# ratings! = 0 returns boolean 0 or 1. Hence non-zero ratings have 1 appended and zero ratings have 0 appended
R = np.append(R, ratings != 0, axis=1)

# print(Y.shape, R.shape, ratings.shape) # (1682, 944) (1682, 944) (1682, 1)

m, n = Y.shape;
Ymean = np.zeros((m, 1))
Ynorm = np.zeros(Y.shape)

for i in range(m):
    idx = np.where(R[i, :] == 1);
    Ymean[i] = np.mean(Y[i, idx]);
    Ynorm[i, idx] = Y[i, idx] - Ymean[i];

print("Ynorm.mean()",Ynorm.mean())


movies = Y.shape[0]  # 1682
users = Y.shape[1]  # 944
features = 10
learning_rate = 10

X = np.random.random(size=(movies, features))
Theta = np.random.random(size=(users, features))
params = np.concatenate((np.ravel(X), np.ravel(Theta)))

fmin = minimize(fun=cofiCostFunc, x0=params, args=(Ynorm, R, features, learning_rate),
                method='CG', jac=True, options={'maxiter': 100})

#==============================================================================
# Features in X The "romance vs action"feature discussion that Prof Ng uses is just to illustrate the process. In practice we have no idea what the learned features represent. They are just numbers in an algorithm
#==============================================================================
X = np.matrix(np.reshape(fmin.x[:movies * features], (movies, features))) # 1682 x 10
Theta = np.matrix(np.reshape(fmin.x[movies * features:], (users, features))) # 944 x 10

predictions = X * Theta.T
# predictions[:, -1] is giving the value of the last column.
my_preds = predictions[:, -1] + Ymean
#my_preds = predictions[:, -1]

# [::-1] sorts reverse
sorted_preds = np.sort(my_preds, axis=0)[::-1] # 1682 x 1

# sorted_preds does not give the index while sorting. We need the index in my_preds to get the movie name. argsort gives the index. We reverse sort my_preds and get the index using argsort
idx = np.argsort(my_preds, axis=0)[::-1]

for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {0} for movie {1}.'.format(str(float(my_preds[j])), movie_idx[j]))

#==============================================================================
# The values show in ex8.pdf are incorrect. https://www.coursera.org/learn/machine-learning/discussions/weeks/9/threads/nINPO9bYEeak2Q7ZSVnQwg
# My result is correct:
#     Predicted rating of 5.000000028770556 for movie Prefontaine.
# Predicted rating of 5.000000017584485 for movie Santa.
# Predicted rating of 4.999999996912166 for movie Entertaining.
# Predicted rating of 4.999999983576859 for movie Someone.
# Predicted rating of 4.999999978889374 for movie They.
# Predicted rating of 4.99999997659324 for movie Saint.
# Predicted rating of 4.999999939534285 for movie Star.
# Predicted rating of 4.999999935777307 for movie Marlene.
# Predicted rating of 4.999999812289167 for movie Great.
#
#==============================================================================
