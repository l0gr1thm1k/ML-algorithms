# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:39:34 2017

@author: Daniel
"""

import pandas as pd
import numpy as np
import random
import network_architecture as na
from preprocess import preprocess


def sigmoid(x, theta):
    """
    @desc   - The sigmoid activation function
    @param  - x: a np.array object of input values
    @param  - theta: an np.array object of weight values
    @return - activation: a float representing the value of the activation
              function given the inputs and the weighrs.
    """
    exponent = np.dot(x, theta)
    activation = 1 / ( 1 + np.e ** (-exponent) )
    return activation

def initialize_weights(n):
    """
    @desc   - initialize a weight vector of n elements. Could initialize the 
              weights to be:
                  
                  (-1/sqrt(n), 1/sqrt(n))
           
    @param  - n: an integer defining the length of the weight vector
    @return - theta: a vector of random weight values
    """
    low = -(1 / np.sqrt(n))
    high = (1 / np.sqrt(n))
    theta = [random.uniform(low, high) for i in range(n)]
    # theta = [0 for i in range(n)]
    return theta
   
    
def create_network():
    # define the network architecture
    network = na.Network()
    
    # make the input layer
    input_layer = na.Layer()
    input_layer.nodes = []
    for i in range(X.shape[1]):
        node = na.Node()
        node.weights = initialize_weights(X.shape[1])
        input_layer.nodes.append(node)
    network.layers.append(input_layer)        

    # make a hidden layer
    for h in range(0):
        hidden_layer = na.Layer()
        hidden_layer.nodes = [] 
        for i in range(5):
            node = na.Node()
            node.weights = initialize_weights(len(network.layers[-1].nodes))
            hidden_layer.nodes.append(node)
        network.layers.append(hidden_layer)

    # make an output layer
    output_layer = na.Layer()
    output_layer.nodes = []
    # for i in range(2):
    node = na.Node()
    node.weights = initialize_weights(len(network.layers[-1].nodes))
    output_layer.nodes.append(node)
    network.layers.append(output_layer)

    return network


def predict(network, training_example):
    """
    @desc   - feed information forward through the network and make a prediction
    @param  - network: the network defined
    @param  - training_example: an input instance to the network
    @return - activation: a float representing the prediction of the netowrk
              over this particular example.
    """
    x = training_example
    for layer in network.layers:
        new_x = []
        for node in layer.nodes:
            activation = sigmoid(x, node.weights)
            new_x.append(activation)
        x = new_x
    return activation


def cross_entropy(y_pred, y):
    """
    @desc   - calculate the error of the model using cross-entropy. 
    @param  - y_pred: the vector of predicted outputs
    @param  - y: the vector of correct outputs
    @return - loss: a float representing the loss in the model
    """       
    pass
 
if __name__ == '__main__':
    X, y = preprocess('tennis.csv')
    y_pred = []
    network = create_network()       

    # calculate output for each training example
    for training_example in X:
        print(predict(network, training_example))
    np.exp()