# -*- coding: utf-8 -*-
"""
Created on Thu May 25 08:39:34 2017

@author: Daniel
"""

import numpy as np
import random

from preprocess import preprocess

class Network:
    
    def __init__(self):
        self.layers = None
        self.input_layer = None
        self.output_layer = None
        

class Node:

    def __init__(self):
        self.inbound_connections = []
        self.activation = 0.0
        self.inputs = None
        self.weights = None
        
class Layer:
    
    def __init__(self):
        self.activation = None

def sigmoid(theta):
    """
    @desc   - The sigmoid activation function
    @param  - x: a np.array object of input values
    @param  - theta: an np.array object of weight values
    @return - activation: a float representing the value of the activation
              function given the inputs and the weighrs.
    """
    #exponent = np.dot(x, theta)
    exponent = sum(theta)
    activation = 1 / ( 1 + np.e ** (-exponent) )
    return activation

def initialize_weights(n):
    sign_changes = np.random.choice(n)
    indicies = np.random.choice([i for i in range(n)], sign_changes, replace=False)
    theta = [random.random() for i in range(n)]
    for i in indicies:
        theta[i] = -1 * theta[i]
    # theta = [np.random.choice(n) * np.sqrt(2.0/n) for i in range(n)]
    return theta
   
    
def create_network():
    # define the network architecture
    network = Network()
    
    # make the input layer
    network.input_layer = Layer()
    network.input_layer.nodes = []
    weights = initialize_weights(X.shape[1])
    for i in range(X.shape[1]):
        node = Node()
        node.weight = weights[i]
        network.input_layer.nodes.append(node)
        
    # make a hidden layer
    network.hidden_layer = Layer()
    bias_unit = Node()
    bias_unit.activation = 1.0
    weights = initialize_weights(6)
    bias_unit.weight = weights[0]
    network.hidden_layer.nodes = []
    network.hidden_layer.nodes.append(bias_unit)
    for i in range(1, 6):
        node = Node()
        node.inputs = network.input_layer
        node.weight = weights[i]
        network.hidden_layer.nodes.append(node)
        
    # calculate the output of the network
    network.output = 0
    return network


if __name__ == '__main__':
    X, y = preprocess('tennis.csv')
    y_pred = []
    network = create_network()       

    # calculate output for each training example
    accuracy = 0
    for i, training_example in enumerate(X):
        x = training_example
        n = len(x)
        print("%" * 80)
        
        # calculate activation values for each node in input layer
        for feature, input_node in zip(x, network.input_layer.nodes):
            input_node.activation = feature * input_node.weight
        
        # calucate input layer activation with sigmoid
        # sigmoid(x, input_node.weights)
        activations = np.array([x.activation for x in network.input_layer.nodes])
        network.input_layer.activation = sigmoid(activations)
        print("input layer activation value: %.6f" %network.input_layer.activation)
        
        inputs = [x.activation for x in network.input_layer.nodes]
        for input_node in network.hidden_layer.nodes:
            node_activations = activations * input_node.weight
            input_node.activation = sigmoid(node_activations)
        
        # hidden layer activation
        network.hidden_layer.activation = sum([x.activation for x in network.hidden_layer.nodes]) / len(network.hidden_layer.nodes)
        print("hidden layer activation value: %.6f" % network.hidden_layer.activation)        
    

        if network.hidden_layer.activation >= 0.5:
            if y[i] == 1:
                accuracy += 1
            # print("Neuron activated with charge %.6f" % test)
        else:
            # print("Neuron not activated")
            if y[i] == 0:
                accuracy += 1
    accuracy = accuracy / X.shape[0]
    print("\nAfter one feedforward pass, the system accuracy is: %.6f" % accuracy)