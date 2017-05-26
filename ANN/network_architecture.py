# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:03:45 2017

@author: Daniel
"""


class Network:
    
    def __init__(self):
        self.layers = []
        # self.input_layer = None
        # self.output_layer = None

        
class Layer:
    
    def __init__(self):
        # self.activation = None
        self.nodes = []

class Node:

    def __init__(self):
        self.inbound_connections = []
        self.outbound_connections = []
        self.activation = 0.0
        self.inputs = None
        self.weights = None
        
