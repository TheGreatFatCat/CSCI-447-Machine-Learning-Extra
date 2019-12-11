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
    static_output_layer = None

    def __init__(self, num_layers, is_stacked, num_hidden_layers, io_size, eta, alpha):
        """
        Initializes an AutoEncoder object.
        :param num_layers: int ; arbitrarily set the number of hidden layers
        :param is_stacked: boolean ; to determine if the encoder is feeding another encoder
        :param num_hidden_layers: list ; number of  hidden layer nodes
        :param io_size: int ; input and output layer node size
        """
        self.is_stacked = is_stacked  # controls whether two or more encoders involved
        self.input_size = io_size
        self.output_size = io_size
        self.hidden_node_sizes = num_hidden_layers  # vector of hidden layer node size
        self.num_hidden_layers = num_layers
        self.input_layer = None
        self.current_layer = None
        self.output_layer = None
        self.eta = eta
        self.alpha = alpha
        self.inner_encoder = None
        print(self)

    def fit_auto_encoder(self, data_obj):
        """
        Create the structure of the neural network
        :return: None
        """
        df = data_obj.train_df
        first_iter = True  # create structure first iteration
        j = 0
        n = 0
        num_of_iterations = 0
        while num_of_iterations < 100:
            for row in df.iterrows():  # iterate through each example
                if first_iter:  # first iteration sets up structure
                    self.input_layer = Layer(self.input_size, True, False, row, None)  # create hidden layer
                    self.current_layer = self.input_layer
                    self.create_hidden_layer(self.num_hidden_layers,
                                             self.hidden_node_sizes)  # create layers in hidden layer
                    self.output_layer = Layer(self.output_size, False, True, None, self.current_layer)
                    self.current_layer.set_next_layer(self.output_layer)  # connect last hidden to output
                    first_iter = False  # does not let a new structure to overwrite existing
                n += 1
                self.training(self, row[1])
            num_of_iterations +=1

    def fit_stacked_auto_encoder(self, data_obj):  # , #num_layers, num_hidden_layer):
        """
        Initializes another auto encoder such that the current auto encoder is
        :param data_obj:
        :return:
        """
        df = data_obj.train_df
        first_iter = True  # create structure first iteration
        num_of_iterations = 0
        n = 0
        while num_of_iterations <= 50:
            for row in df.iterrows():  # iterate through each example
                if first_iter:  # first iteration sets up structure
                    self.input_layer = Layer(self.input_size, True, False, row, None)  # create hidden layer
                    # print("Create Input Layer")
                    self.current_layer = self.input_layer
                    self.inner_encoder = AutoEncoder(self.num_hidden_layers, True, [3],
                                                     self.hidden_node_sizes[0], 0.2, 0.45)  # int, bool, list, int
                    # print("Create AutoEncoder as Hidden Layer")
                    self.inner_encoder.input_layer = Layer(self.inner_encoder.input_size, True, False, None,
                                                           self.input_layer)
                    # print("Create Inner AutoEncoder's Input Layer")
                    self.inner_encoder.current_layer = self.inner_encoder.input_layer
                    self.current_layer.set_next_layer(self.inner_encoder.current_layer)
                    self.inner_encoder.create_hidden_layer(self.inner_encoder.num_hidden_layers,
                                                           self.inner_encoder.hidden_node_sizes)
                    # print("Create AutoEncoder's Hidden Layer; next line shows hidden layer creation")
                    self.inner_encoder.output_layer = Layer(self.inner_encoder.output_size, False, True, None,
                                                            self.inner_encoder.current_layer)
                    # print("Create AutoEncoder's Output Layer")
                    self.inner_encoder.current_layer.set_next_layer(self.inner_encoder.output_layer)
                    self.current_layer = self.inner_encoder.output_layer
                    self.output_layer = Layer(self.output_size, False, True, None, self.current_layer)
                    # print("Create Output Layer")
                    self.current_layer.set_next_layer(self.output_layer)  # connect last hidden to output
                    first_iter = False
                n += 1
                self.training(self, row[1])
                inp_v = []
                for node in self.input_layer.get_next_layer().nodes:
                    inp_v.append(node.get_value())
                self.training(self.inner_encoder, inp_v)
            num_of_iterations += 1

    def create_hidden_layer(self, num_layers, num_nodes):
        """
        Create the hidden layer's layers
        :param num_layers: number of hidden layers
        :param num_nodes: list; number of nodes in hidden layer
        :return: None
        """
        if len(num_nodes) is 1:
            new_layer = Layer(num_nodes[0], False, False, None, self.current_layer)
            self.current_layer.set_next_layer(new_layer)
            temp = self.current_layer
            self.current_layer = temp.get_next_layer()
        else:
            for i in range(num_layers):  # create the user-defined number of layers
                new_layer = Layer(num_nodes[i], False, False, None, self.current_layer)
                self.current_layer.set_next_layer(new_layer)  # link from current to next layer
                temp = self.current_layer
                self.current_layer = temp.get_next_layer()

    def activation_function_process(self, current, output):
        """
        Given the current layer (can be input), get the next layer. For each node in next layer, calculate the Wa+b
        :param current:
        :return: None
        """
        current_layer = current
        next_layer = current_layer.get_next_layer()  # next layer to calculate Z and sigmoid for.
        if next_layer is output:
            self.linear_activation(current, next_layer)
        else:
            for target_node in next_layer.nodes:  # for each node in the next layer, calculate activation function
                sum_value = 0
                for node, weight in zip(current_layer.nodes,
                                        target_node.weight_vector):  # iterate through current layer nodes in parallel to the weights of the target node (target node has previous layer size weights).
                    sum_value += node.get_value() * weight  # multiply weights and value
                sum_value += target_node.bias  # add in bias last
                target_node.z_value = sum_value  # value to sigmoid
                target_node.sigmoid_function()  # creates the a range between [0, 1] from the value z.
            current = current_layer

    def linear_activation(self, current, next_layer):
        """
        Performs linear activation function
        :param current: the current layer ; layer
        :param next_layer: the next layer to current, layer
        :return: None
        """
        l = []
        current_layer = current
        for target_node in next_layer.nodes:  # for each node in the next layer, calculate activation function
            sum_value = 0
            for node, weight in zip(current_layer.nodes,
                                    target_node.weight_vector):  # iterate through current layer nodes in parallel to the weights of the target node (target node has previous layer size weights).
                sum_value += node.get_value() * weight  # multiply weights and value
            sum_value += target_node.bias  # add in bias last
            target_node.set_value(sum_value)  # value to sigmoid
            l.append(sum_value)

    def test(self, df):
        """
        Given a test set, it runs the examples through the network and finds the means squared error
        :param df: dataframe; the test data frame
        :return: None
        """
        print("_____________________________")
        print("-----------Testing-----------")
        print("Size of Test Set: ", len(df))
        test_data = df
        actual = []
        predicted = []
        for row in test_data.iterrows():  # iterate through the data
            self.current_layer = self.input_layer
            for node, new_input in zip(self.input_layer.nodes, row[1]):  # set the input layer values
                node.set_value(new_input)
            self.feed_forward_process()  # feed forward to get values
            actual, predicted = self.print_output()  # get the predicted values and the actual
        squared = 0
        for i, p in zip(actual, predicted):
            squared += (i - p) ** 2
        ave = squared / len(predicted)
        print("Means Squared Error: ", ave, "\n\n")

    def back_propagation_process(self):
        while self.current_layer.get_previous_layer() != None:  # goes backwards through code
            if self.current_layer is self.output_layer:  # different for output layer than rest
                j = 0
                for node in self.current_layer.nodes:  # goes through all nodes
                    node.delta = -(self.input_layer.nodes[
                                       j].get_value() - node.get_value())  # activation function is linear
                    node.bias_change += node.delta  # for batch change in training function
                    j += 1
                    i = 0
                    for weight in node.weight_vector:  # for changing weight
                        node.weight_change_vector[i] = node.delta * self.current_layer.get_previous_layer().nodes[
                            i].get_value()  # adjusts values for weights
            else:  # works different in all other layers
                f = 0
                for node in self.current_layer.nodes:  # goes through nodes
                    summer = 0
                    for noder in self.current_layer.get_next_layer().nodes:  # summing from next layer
                        summer += noder.delta * noder.weight_vector[f]
                    node.delta = node.get_value() * (1 - node.get_value()) * summer  # setting delta for current node
                    node.bias_change += node.delta  # for batch change in training function
                    u = 0
                    for weight in node.weight_vector:  # iterates through incoming weights
                        node.weight_change_vector[u] += node.delta * self.current_layer.get_previous_layer().nodes[
                            u].get_value()  # for batch change in training function
                    f += 1
            self.current_layer = self.current_layer.get_previous_layer()  # goes to previous layer
        return

    def training(self, encoder, input):
        j = 0
        for node in encoder.input_layer.nodes:  # resetting the input nodes
            node.set_value(input[j])
            j += 1
        encoder.feed_forward_process()  # updating values
        encoder.back_propagation_process()  # determining changes
        encoder.current_layer = encoder.output_layer  # setting to end layer
        while encoder.current_layer.get_previous_layer() != None:  # goes through all the layers
            for node in encoder.current_layer.nodes:  # goes through each node in layer
                node.previous_bias_change = -self.eta * node.bias_change + self.alpha * node.previous_bias_change  # adding momentum and reducing step
                node.bias += node.previous_bias_change  # moving bias
                node.bias_change = 0  # resetting for batch
                i = 0
                for weight in node.weight_vector:  # going through incoming weights
                    node.previous_weight_change[i] = -self.eta * node.weight_change_vector[i] + self.alpha * \
                                                     node.previous_weight_change[
                                                         i]  # adding momenturm and reducing step
                    weight += node.previous_weight_change[i]  # moving weight
                    node.weight_change_vector[i] = 0  # resetting for batch
                    i += 1
            encoder.current_layer = encoder.current_layer.get_previous_layer()  # going to next layer
        return

    def feed_forward_process(self):
        """
        Goes through network nodes and finds the sigmoid value for each node.
        :return: None
        """
        self.current_layer = self.input_layer
        while self.current_layer is not self.output_layer:  # once it is the output layer, no sigmoid value to compute
            self.activation_function_process(self.current_layer,
                                             self.output_layer)  # performs sigmoid functions for layer
            self.current_layer = self.current_layer.get_next_layer()

    def print_layer_neuron_data(self):
        """
        prints the neuron data and the encoders data
        :return:
        """
        self.current_layer = self.input_layer
        while True:
            self.current_layer.print_layer_data()
            for node in self.current_layer.nodes:
                node.print_neuron_data()
                pass
            if self.current_layer is self.output_layer:
                break
            else:
                self.current_layer = self.current_layer.get_next_layer()

    def print_output(self):
        """
        prints the output and the input from testing. Also returns the actual and predicted values
        :return:
        """
        values2 = []
        values1 = []
        for node1, node2 in zip(self.output_layer.nodes, self.input_layer.nodes):
            values1.append(node1.get_value())
            values2.append(node2.get_value())
        # print("Output: ", values1, "\nInput: ", values2)
        return values2, values1


class Layer:
    """
    Creates the layers in the network of the encoder.
    Structure as a linked list essentially.
    """

    def __init__(self, num_nodes, is_input_layer, is_output_layer, input, prev=None):
        """
        Initialize a layer in the Neural Network.
        :param num_nodes: the number of nodes in the network
        :param is_input_layer: boolean var to specify if it is the input layer
        :param is_output_layer: boolean var to specify if it is the output layer
        """
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        self.no_of_nodes = num_nodes
        self.nodes = []
        self._previous_layer = prev
        self._next_layer = None
        self._initialize_layer(input)  # creates the layer

    def _initialize_hidden_nodes(self, weight_matrix, bias_vector):
        """
        initializes nodes within the layer.
        :param bias_vector: the bias for all the nodes in a list
        :param weight_matrix: a matrix of weights associated to the previous layer
        :return: nodes list of Neurons
        """
        input = [0] * self.no_of_nodes  # input is 0 for non input layer nodes; sigmoid not calculated
        nodes = []  # list to hold nodes
        for neuron in range(self.no_of_nodes):
            n = Neuron(input[neuron], bias_vector[neuron], [], len(self._previous_layer.nodes))
            for w in weight_matrix:
                n.weight_vector.append(w[neuron])
            nodes.append(n)
        return nodes

    def _initialize_weights(self):
        """
        Randomly initialize weights and bias vector entries.
        :return: weight matrix and vector
        """
        weight_matrix = []
        bias_vector = []
        size = self.get_previous_layer().no_of_nodes
        for i in range(size):  # iterate through number of nodes in previous layer
            weight_vector = []
            for j in range(self.no_of_nodes):  # iterate through number of nodes in current layer
                weight_vector.append(random.uniform(-1, 1))  # random value inserted into vector
            weight_matrix.append(weight_vector)  # append the vector into the matrix
        for i in range(self.no_of_nodes):
            bias_vector.append(float(random.uniform(-1, 1)) / 100)  # random initialized bias
        return weight_matrix, bias_vector

    def _initialize_layer(self, input):
        """
        initializes the layer using the other initializing functions.
        :return: None
        """
        if self.get_previous_layer() is None:  # initialize input layer
            for i in input[1]:
                self.nodes.append(Neuron(i, None, None, 0))
        else:  # initialize hidden_layer and output
            weight_matrix, bias_vector = self._initialize_weights()
            self.nodes = self._initialize_hidden_nodes(weight_matrix, bias_vector)

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

    def set_next_layer(self, next_layer):
        """
        Link to the next layer
        :param next_layer:
        :return: None
        """
        self._next_layer = next_layer

    def set_previous_layer(self, prev_layer):
        """
        Link for to the previous layer
        :param prev_layer:
        :return: None
        """
        self._previous_layer = prev_layer

    def print_layer_data(self):
        """
        prints the layer data within an encoder
        :return:
        """
        if self.is_input_layer:
            string = "_________________________________________________________________"
            center = "Input Layer"
            print(string)
            print(center)
            print(string)
            print("Next Layer Nodes = %s" % self.get_next_layer().nodes)
        if self.is_output_layer:
            string = "_________________________________________________________________"
            center = "Output Layer"
            print(string)
            print(center)
            print(string)
            print("Previous Layer Node= %s" % self.get_previous_layer().nodes)
        if self.is_output_layer == False and self.is_input_layer == False:
            string = "_________________________________________________________________"
            center = "Hidden Layer"
            print(string)
            print(center)
            print(string)
            print("Next Layer Nodes = %s" % self.get_next_layer().nodes)
            if not self.get_previous_layer().is_input_layer:
                print("Previous Layer Node= %s" % self.get_previous_layer().nodes)
            else:
                print("Previous Layer Node= Input Layer")

        print("Number of Nodes = %s" % self.no_of_nodes)
        print("Nodes = %s" % self.nodes)


class Neuron:
    """
    creates the neurons within a layer.
    """

    def __init__(self, value, bias, weight_vector, prev_node_nums):
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
        self.z_value = 0
        self.weight_change_vector = [0] * prev_node_nums
        self.previous_weight_change = [0] * prev_node_nums
        self.bias_change = 0
        self.previous_bias_change = 0

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

    def set_value(self, val):
        self._value = val

    def sigmoid_function(self):
        """
        activation function for node
        :return: None
        """
        self._value = 1 / (1 + np.exp(-self.z_value))

    def print_neuron_data(self):
        print("\t----------Neuron------------")
        print("\t value = %s" % self.get_value())
        print("\t bias = %s" % self.bias)
        if self.weight_vector is not None:
            print("\t weight_vector = ", list(self.weight_vector), "\n")
        else:
            print("\t weight_vector = ", None, "\n")
