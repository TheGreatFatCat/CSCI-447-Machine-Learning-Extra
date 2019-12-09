"""
Justin Keeling
Alex Harry
John Lambrecht
Andrew Smith

Institution: Montana State University
Course: CSCI-447 Machine Learning
Instructor: John Shepherd

File: contains classes and functions that are used primarily for the purpose of creating an AutoEncoder
"""

import random
import math
import numpy as np

# TODO: Remove when read...
""" 
________________________________________________________________________________________________________________________
READ THIS
------------------------------------------------------------------------------------------------------------------------
An auto encoder is a neural network! 
- neural_net.fit(X, Y)  where X is the vector of data and Y is the labels

*** The difference from a neural network and auto encoder is the following ***
- The labels are the input vector
- neural_net.fit(X, X)  where X is the vector of data and X is the labels
------------------------------------------------------------------------------------------------------------------------
"""


class AutoEncoder:
    """
    An AutoEncoder is an unsupervised algorithm and is also a neural network. The difference between the neural network
    and the auto encoder is that the target function is the input.

    Three layers: Input, hidden, and output
    Structure: Encoder and Decoder
    Encoder: Compresses data. or maps input data into a hidden representation.
    Decoder: Reconstructs the data back to the input.

    Specifics: uses back propagation to tune and train. Utilizes Feed forward.

    This class contains functions to structure, fit, and  predict auto encoders.
    """

    def __init__(self, data_instance, input_output_size, num_layers, is_stacked):
        """
        Initializes an AutoEncoder object.
        :param data_instance: the data object of the data to train and test upon
        :param input_output_size: arbitrarily choose size of input
        :param num_layers: arbitrarily set the number of hidden layers
        :param is_stacked: boolean to determine if the encoder is should feeding another encoder
        """
        self.data_obj = data_instance
        self.df = self.data_obj.df  # data frame of preprocessed data
        self.io_size = input_output_size
        self.num_layers = num_layers
        self.is_stacked = is_stacked

    def make_layers(self):
        pass

    def vectorize(self):
        pass

    def networkize(self):
        pass

    def sigmoid_function(self):
        pass

    def predict(self):
        pass

    def cost(self):
        pass

    def set_output(self):
        pass

    def back_propagation(self):
        pass

    def gradient_descent(self):
        # TODO: not sure if needed for back prop... not mentioned in assignment
        pass

# TODO: determine if class NetworkClient is needed... See program 4


class Layer:
    """
    Creates the layers in the network of the encoder.
    """
    def __init__(self, no_of_nodes):
        self.no_of_nodes = no_of_nodes
        self.nodes = []

    def make_nodes(self):
        for nodes in range(self.no_of_nodes):
            self.nodes.append(Neuron(float(random.randint(-1, 1))/100))

    def make_input_layer(self, inputs):
        i = 0
        for input in inputs:
            self.nodes[i].value = input
            i += 1

    def next_layer(self):
        pass

    def 


class Neuron:
    """
    creates the neurons within a layer.
    """
    def __init__(self, bias, value=None):
        self.bias = bias
        self.prev_bias_change = 0
        self.bias_change = 0
        self.is_sigmoidal = None
        self.is_linear = None
        self.incoming_weights = []
        self.outgoing_weights = []
        self.value = value
        self.delta = 0


class Weight:
    """
    creates the weights associated to a neuron within a layer
    """
    def __init__(self, L_neuron, R_neuron):
        self.L_neuron = L_neuron
        self.R_neuron = R_neuron
        self.weight = float(random.randint(-1, 1)) / 100
        self.weight_change = 0
        self.prev_change = 0
        self.momentum_cof = .5
        self.eta = .1

    def set_weight(self, weight):
        self.weight = weight
