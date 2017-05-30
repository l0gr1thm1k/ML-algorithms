# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:37:20 2017

@author: Daniel
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
# from sklearn.linear_model import LogisticRegressionCV

# generate a dataset
np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.2)
plt.figure(figsize=(11, 8.5))
plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral, edgecolor='black')


num_examples = len(X)
nn_input_dim = 2
nn_output_dim = 2

# hand picked hyperparameters
epsilon = 0.01 # the learning rate
reg_lambda = 0.01 # regurlarization coefficient

def calculate_loss(model):
    # this is cross-entropy
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # forward propagation to calculate prediction
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # calculate the losss
    correct_logprobs =  -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs)
    # add regularization term to loss
    data_loss += reg_lambda/2 * (np.sum(np.square(W1) + np.sum(np.square(W2))))
    return 1./ num_examples * data_loss


def predict(model, x):
    # helper function to predict 0 or 1
    W1, b1, W2, b2 = model['W1'], model['b2'], model['W2'], model['b2']
    # forward propogation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def build_model(nn_hdim, num_passes=20000, print_loss=False):
    
    # Initialize to random values
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.rand(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))
    
    # the model returned
    model = {}
    for i in range(num_passes):
        
        # forward propogation
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        a2 = np.tanh(z2)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # backpropogation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = (X.T).dot(delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)
        
        # add regularization terms
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1
        
        # gradient descent parameter update
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2
        
        # assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
        # optionally print loss
        if print_loss and i % 1000 == 0:
            print("loss after iteration %i: %f" % (i, calculate_loss(model)))
    
    return model

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


model = build_model(3, print_loss=True)

plot_decision_boundary(lambda x: predict(model, x))
plt.title("Decision Boundary for hidden layer size 3")