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
    for h in range(1):
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


if __name__ == '__main__':
    X, y = preprocess('tennis.csv')
    y_pred = []
    network = create_network()       

    # calculate output for each training example
    
    first_run = False
    for training_example in X:
        x = training_example
        n = len(x)

        for layer in network.layers:
            new_x = []
            for node in layer.nodes:
                activation = sigmoid(x, node.weights)
                new_x.append(sigmoid(x, node.weights))
            if first_run is True:
                print("This layer received %d inputs" % len(x))
            x = new_x
        first_run = False
            
        # only one output in the output layer
        prediction = x # [0]
        print(prediction)
    '''
        if prediction >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
        
    accuracy = 0
    for i, j in zip(y, y_pred):
        if i == j:
            accuracy += 1
    accuracy = accuracy / len(y)
    print(accuracy)
    y_pred = pd.Series(y_pred)
    '''