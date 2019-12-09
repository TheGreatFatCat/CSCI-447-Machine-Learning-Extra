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
    def __init__(self, num_nodes, is_input_layer, is_output_layer, input):
        """
        Initialize a layer in the Neural Network.
        :param num_nodes: the number of nodes in the network
        :param is_input_layer: boolean var to specify if it is the input layer
        :param is_output_layer: boolean var to specify if it is the output layer
        """
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer

        self.no_of_nodes = num_nodes

        self.nodes = None

        # TODO: determine if needed. Should consider iff we can do matrix multiplications...
        # self.weight_matrix = None  # a row is all the nodes in previous layer connected to a node in this layer
        # self.bias_vector = []  # contains the bias values with respect to the nodes

        self._previous_layer = None
        self._next_layer = None

        self._initialize_layer(input)  # creates the layer


    @staticmethod
    def _initialize_hidden_nodes(weight_matrix, bias_vector):
        """
        initializes nodes within the layer.
        :param bias_vector: the bias for all the nodes in a list
        :param weight_matrix: a matrix of weights associated to the previous layer
        :return: nodes list of Neurons
        """
        input = [0]*len(bias_vector)  # input is 0 for non input layer nodes; sigmoid not calculated
        nodes = []  # list to hold nodes
        for i, b, w in zip(input, bias_vector, weight_matrix):
            nodes.append(Neuron(i, b, w))  # create a Neuron and append it to list
        return nodes

    def _initialize_weights(self):
        """
        Randomly initialize weights and bias vector entries.
        :return: weight matrix and vector
        """
        weight_matrix = []
        size = self.get_previous_layer().no_of_nodes
        for i in range(size):  # iterate through number of nodes in previous layer
            weight_vector = []
            for j in range(self.no_of_nodes):  # iterate through number of nodes in current layer
                weight_vector.append(random.uniform(-1, 1))  # random value inserted into vector
            weight_matrix.append(weight_vector)  # append the vector into the matrix
            bias_vector.append(float(random.randint(-1, 1))/100)  # random initialized bias
        return weight_matrix, bias_vector

    def _initialize_layer(self, input):
        """
        initializes the layer using the other initializing functions.
        :return: None
        """
        if self.is_input_layer:  # initialize input layer
            for i in input:
                self.nodes.append(Neuron(i, None, None))
        else:  # initialize hidden_layer and output
            weight_matrix, bias_vector = self.initialize_weights()
            self.nodes = self._initialize_hidden_nodes(None, weight_matrix, bias_vector)


    def get_next_layer(self):
        """
        Getter function for the previous layer
        :return: next layer ; type Layer
        """
        return self._next_layer

    def get_previous_layer(self):
        """
        Getter function for previous layer
        :return: previous layer ; type Layer
        """
        return self._previous_layer



class Neuron:
    """
    creates the neurons within a layer.
    """
    def __init__(self, value, bias, weight_vector):
        """
        Initialize a neuron in a layer
        :param value: the value it currently contains
        :param bias: the bias associated to the node
        :param weight_vector: the weights associated to the node with respect to the previous layers nodes
        """
        self._value = value
        self.bias = bias
        self.weight_vector = weight_vector
        self.delta = 0

    def adjust_bias(self, amount):
        """
        PLAN to give the location of the weight to adjust and a positive or negative value to adjust by.
        :param amount: amount to adjust by
        :return: None
        """
        pass

    def adjust_weight(self, location, amount):
        """
        PLAN to give the location of the weight to adjust and a positive or negative value to adjust by.
        :param location: location of the weight to change
        :param amount: amount to adjust by
        :return: None
        """
        pass


    def change_value(self, new_val):
        """
        overwrite old value with new calculated value
        :param new_val:
        :return: None
        """
        self._value = new_val

    def get_value(self):
        """
        getter function for value, either a sigmoidal value or a feature in the example
        :return: value ; type float
        """
        return self._value


# TODO: not being used, at least not yet... using vectors and matrices currently...
# class Weight:
#     """
#     creates the weights associated to a neuron within a layer
#     """
#     def __init__(self, L_neuron, R_neuron):
#         self.L_neuron = L_neuron
#         self.R_neuron = R_neuron
#         self.weight = float(random.randint(-1, 1)) / 100
#         self.weight_change = 0
#         self.prev_change = 0
#         self.momentum_cof = .5
#         self.eta = .1
#
#     def set_weight(self, weight):
#         self.weight = weight
