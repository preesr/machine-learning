# Below implementation works
# fmincg seems to take a lot of time try to use another algorithm if you like. I reduced max iterations from 250 to 50. That definitely makes it faster and the accuracy is at 95.94%. ex4.pdf says 95.3% as the correct correct +-1.
# Theta assignment and fwd propagation should also be inside the 2 functions both cost and grad. Otherwise it doesn't work correctly
# issue was not calling fwd propagation everytime within cost function and gradient method. I had previously kept it globally and not inside the methods. Looks like the fwd propagation and theta assigning gets passed for each iteration iteratively so not having them inside the method makes fwd prop and theta assignment load the same initial parameters and hence the accuracy was at ~10%.
# I had implemented sigmoid gradient incorrectly. Corrected it

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural Network Implementation To recognize hand written digits 0-9 using backpropagation.

@author: Preethi
References: Andrew Ng,  github jdwittenauer,rragundez, scipy.org, scikit-learn.org
"""

import numpy as np
import matplotlib.pyplot as plt
#import scipy.optimize as opt
# Input file in MATLAB's native format, so to load it in Python we need to use a SciPy utility.
from scipy.io import loadmat
from matplotlib import cm
import random
from scipy.optimize import fmin_cg
#from scipy.optimize import minimize
from sklearn.preprocessing import OneHotEncoder
from numpy.random import rand

# Returns a dict called data
data = loadmat('ex4data1.mat')

X = data['X']
#The result categories in y have values ranging from 1- 10.
y = data['y']
print ("X, y shape:",X.shape, y.shape)
# O/P (5000, 400) (5000, 1)



#function that takes an example array and converts it into an image in gray scale
#it shoes 200 examples
def showNumbers(x):
    #im_number = 4000
    imv = np.empty((20,0))
    imag = []
    for i in range(2300,2500):
        # Without the .T transpose, the image of each number was rotated
        # reshapes the array of 400 elements in given (20,20)
        im = np.reshape(x[i],(20,20)).T
        # I do not follow the below lines on how image is rendered but it works. From github rragundez
        imv = np.append(imv,im,axis=1)
        if (i+1) % 20 == 0:
            imag.append(imv)
            imv = np.empty((20,0))
    image = np.concatenate((imag[:]),axis = 0)
    plt.imshow(image, cmap = cm.Greys_r)

#==============================================================================
# I set this and changed the range from 3800 to 4000 for im_number=4000,
# I saw different numbers. So looks like I am reading the data correctly
#==============================================================================
#showNumbers(X)
#==============================================================================
# one-hot encode our y labels. One-hot encoding turns a class label n (out of k classes) into a vector of length k where index n is "hot" (1) while the rest are zero. Scikit-learn has a built in utility for this.
#==============================================================================
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
#print("y_onehot.shape:", y_onehot.shape)
#print(y_onehot[0,:])
#O/P: y_onehot.shape: (5000, 10)

#print(y[0,:])
# (5000, 1)
#==============================================================================
# The neural network we're going to build for this exercise has an input layer matching the size of our instance data (400 + the bias unit), a hidden layer with 25 units (26 with the bias unit), and an output layer with 10 units corresponding to our one-hot encoding for the class labels
#==============================================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm
# Forward propagation screenshot implementation
def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    #print("a1.shape",a1.shape)
    #a1.shape (5000, 401)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T

    h = sigmoid(z3)
    #print("h shape:",h.shape)
    return a1, z2, a2, z3, h
# theta1.shape, theta2.shape ((25L, 401L), (10L, 26L))
# a1.shape, z2.shape, a2.shape, z3.shape, h.shape
# (5000L, 401L), (5000L, 25L), (5000L, 26L), (5000L, 10L), (5000L, 10L)



# Remember: The optimization functions want the data as a vector (array) and for vectorization matrices are the best. Hence, roll and unroll as needed. reshape() is useful to make parameter array into parameter matrices

# initial setup
input_size = X.shape[1]
hidden_size = 25
num_labels = 10
lamda = 1

def initializeTheta():
    eps = 0.12
    # Theta1 shape: ((25L, 401L) (All the arrows from 401 input features to 25 hidden layers)
    theta1 = rand(hidden_size,(input_size + 1))*2*eps-eps
    # Theta 2 shape: From 26 hidden layers including the bias layer to 10 outputs (10L, 26L))
    theta2 = rand(num_labels,(hidden_size + 1))*2*eps-eps
    theta = np.append(theta1.flatten(),theta2.flatten())
    return theta

# randomly initialized parameter array of the size of the full network's parameters.
params = initializeTheta()

m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)

# unravel the parameter array into parameter matrices for each layer
   # # Theta1 shape: ((25L, 401L) (All the arrows from 401 input features to 25 hidden layers)
# # reshape the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
# # Theta 2 shape: From 26 hidden layers including the bias layer to 10 outputs (10L, 26L))
# #From 25*401 to last entries
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
print ("theta1, theta2 shapes",theta1.shape, theta2.shape)


#print("a1.shape, z2.shape, a2.shape, z3.shape, h.shape:",a1.shape, z2.shape, a2.shape, z3.shape, h.shape)

#==============================================================================
# Starting Backpropagation implementation
# https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm
#==============================================================================

# Calculate g prime g' called as sigmoid gradient. Page 7 ex4.pdf
# Remember: if the pdf say xz then it means matrix multiplication of x * z also known as dot product or np.dot or * in python
# if the Octave pdf saya x .* z that means it is a element-wise product which is np.multiply in python
def sigmoidGradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

# https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm
def costReg(params, input_size, hidden_size, num_labels, X, y, lambda_val):
    # # Theta1 shape: ((25L, 401L) (All the arrows from 401 input features to 25 hidden layers)
# # reshape the parameter array into parameter matrices for each layer
# # From beginning to 25*401 entries

    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    # # Theta 2 shape: From 26 hidden layers including the bias layer to 10 outputs (10L, 26L))
# #From 25*401 to last entries
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
        # run the feed-forward pass
    # Feedforward the neural network and return the cost in the variable J as per nnCostFunction.m
    # Without this, accuracy fails to 9.6%
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # Implementing the ex4.pdf page 5.
    J = 0
    first = np.multiply(-y, np.log(h))
    second = np.multiply((1 - y), np.log(1 - h))
    J = np.sum(first - second)/m

    # Now the regularization as per page 6 in ex4.pdf . The sum of k 1 to no.of features and j 1 to no. of hidden layers are already part of theta1 and theta2 and hence it is vectorized
    # [:,1:] is excluding the 1st entry of theta which is the bias and it should not be regularized. By regularizing the bias you are penalizing it making it close to zero and that will not work . Eg: −30+20x1+20x2 is an AND function. for x1,x2 element of 0,1. If you make -30 (the bias) to 0 by regularization, then this will not be an AND function anymore
    J+= (float(lambda_val)/(2 * m)) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2)))


    return J

def gradReg(params, input_size, hidden_size, num_labels, X, y, lambda_val):
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
          # perform backpropagation
    # Back prop algorithm
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)
            # getting the t of the input set (5000) from each of the return values in forward prop function. Page 9 step 1
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    for t in range(m):
        # getting the t of the input set (5000) from each of the return values in forward prop function. Page 9 step 1
        a1t = a1[t,:]  # (1, 401)
        z2t = z2[t,:]  # (1, 25)
        a2t = a2[t,:]  # (1, 26)
        ht = h[t,:]  # (1, 10)
        yt = y[t,:]  # (1, 10)

        # calculate small delta. Here is the where the back prop starts from the output layer. Page 9 step 2 ex4.pdf
        #https://www.coursera.org/learn/machine-learning/supplement/pjdBA/backpropagation-algorithm: 3. Using y(t), compute δ(L)=a(L)−y(t)
                #Where L is our total number of layers and a(L) is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y.
        d3t = ht - yt  # (1, 10)

        # Inserting bias at the 2nd layer of the network
        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        # small delta2 is (theta2).T * d3 * g-prime(z2) as per page 9 item 3. ex4.pdf
        #From theta2 shape: (10, 26) d3t shape:(1, 10) : Doing transpose for both and multiplying gives 26 , 1. z2t shape (1, 26). Hence do a transpose of the previous result gives 1 , 26. Then element-wise multiply with z2t 1 , 26.
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoidGradient(z2t))

        # Item 4 page 9 ex4.pdf. We should not include small delta0. Hence [:,1:]
        delta1 = delta1 + (d2t[:,1:]).T * a1t # dimensions: (25, 401) + (25 , 1) * (1 , 401)
        delta2 = delta2 + d3t.T * a2t # dimensions: (10, 26) + (10 , 1) * (1 , 26)
# Step 5 in page 9 ex4.pdf
    delta1 = delta1 / m
    delta2 = delta2 / m

    # add the gradient regularization term excluding the bias 0th element
    delta1[:,1:] = delta1[:,1:] + (theta1[:,1:] * lambda_val) / m
    delta2[:,1:] = delta2[:,1:] + (theta2[:,1:] * lambda_val) / m
    #print("delta1 shape:",delta1.shape)
    #(25 * 401)
    #print("delta1 shape:",delta2.shape)
    # 10 * 26
# unravel the gradient matrices into a single array
# total number of elements = (25 * 401) + (10 * 26) = 10285
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    #print("grad shape:",grad.shape)
    # grad shape: (10285,)

    return grad

#cost = costReg(params, input_size, hidden_size, num_labels, X, y_onehot, lamda)
#grad=gradReg(params, input_size, hidden_size, num_labels, X, y_onehot, lamda)
#print(cost, grad.shape)




# minimize the the cost function. i.e gradient descent. This gave me only 87.46% accuracy
#==============================================================================
#fmin = minimize(fun=costReg, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, lamda), method='TNC', jac=gradReg, options={'maxiter': 50})
#==============================================================================
#fmin_cg Minimize a function using a nonlinear conjugate gradient algorithm. Takes more time but gives better accuracy
min_theta = fmin_cg(costReg,params,fprime = gradReg,args=(input_size, hidden_size, num_labels, X, y_onehot, lamda), maxiter = 50,disp = 1)
#print(theta)

X = np.matrix(X)
min_theta1 = np.matrix(np.reshape(min_theta[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
min_theta2 = np.matrix(np.reshape(min_theta[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, min_theta1, min_theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)
print("y_pred:",y_pred)


# zip - Make an iterator that aggregates elements from each of the iterables.
correct = [1 if a == b else 0
       for (a, b) in zip(y_pred, y)]
accuracy = (sum(correct) / float(len(correct))) * 100

print ('accuracy = {0}%'.format(accuracy))
# TO DO: Implement Gradient checking, visualization of data if you have time
