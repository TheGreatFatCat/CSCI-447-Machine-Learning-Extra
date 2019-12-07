import random
import math
import numpy as np


class NeuralNetwork:
    def __init__(self, data_instance): #, edited_data, compressed_data, centroids_cluster, medoids_cluster):
        self.data_instance = data_instance
        # self.edited_data = edited_data
        # self.compressed_data = compressed_data
        # self.centroids_cluster = centroids_cluster
        # self.medoids_cluster = medoids_cluster

    def make_layers(self, no_of_layers, no_of_nodes):
        """
        :param no_of_layers: sets up the number of hidden layers for the network
        :param no_of_nodes: sets up the number of nodes in the hidden layer
        :return:
        """
        # row = self.data_instance.df.shape[0]-1
        first_layer_size = self.data_instance.df.shape[1]-1
        layers = []
        layers.append(Layer(first_layer_size))
        layers[0].make_nodes()
        for i in range(no_of_layers):
            layers.append(Layer(no_of_nodes))
            layers[i+1].make_nodes()
            if i+1 == no_of_layers:
                outputs = self.set_output()
                layers.append(Layer(len(outputs)))
                layers[i+2].make_nodes()
            #     print("sup")
            # print(i)
            # print(layers[1].no_of_nodes)
            for j in range(layers[i].no_of_nodes):
                for f in range(len(layers[i+1].nodes)):
                    layers[i].nodes[j].outgoing_weights.append(Weight(layers[i].nodes[j], layers[i+1].nodes[f]))
                    layers[i + 1].nodes[f].incoming_weights = layers[i].nodes[j].outgoing_weights
        for j in range(len(layers[-2].nodes)):
            for f in range(len(layers[-1].nodes)):
                layers[-2].nodes[j].outgoing_weights.append(Weight(layers[-2].nodes[j], layers[-1].nodes[f]))
                layers[-1].nodes[f].incoming_weights = layers[-2].nodes[j].outgoing_weights
        return layers, outputs



        # point_nets = []
        # for index, row in self.data_instance.train_df.iterrows():
        #     layers = []
        #     layers.append(Layer(len(row.drop(columns=self.data_instance.label_col))))
        #     layers[0].make_input_layer(row.drop(columns=self.data_instance.label_col))
        #     for i in range(no_of_layers):
        #         layers.append(Layer(no_of_nodes))
        #         layers[i+1].make_nodes()
        #         if i+1 == no_of_layers:
        #             layers.append(Layer(len(self.set_output())))
        #             layers[i+2].make_nodes()
        #         #     print("sup")
        #         # print(i)
        #         # print(layers[1].no_of_nodes)
        #         for j in range(layers[i].no_of_nodes):
        #             for f in range(len(layers[i+1].nodes)):
        #                 layers[i].nodes[j].outgoing_weights.append(Weight(layers[i].nodes[j], layers[i+1].nodes[f]))
        #                 layers[i + 1].nodes[f].incoming_weights = layers[i].nodes[j].outgoing_weights
        #     for j in range(len(layers[-2].nodes)):
        #         for f in range(len(layers[-1].nodes)):
        #             layers[-2].nodes[j].outgoing_weights.append(Weight(layers[-2].nodes[j], layers[-1].nodes[f]))
        #             layers[-1].nodes[f].incoming_weights = layers[-2].nodes[j].outgoing_weights
        #     point_nets.append(layers)

    def sigmoid(self, layers, input):
        # i = 0
        layers[0].make_input_layer(input)
        for layer in layers[1:]:
            for node in layer.nodes:
                sigmoid_total = 0
                # print(node.bias_change)
                for weight in node.incoming_weights:
                    # print(weight.weight_change)
                    # print(len(layers))
                    # print(len(layers[1:]))
                    # print(weight.get_weight())
                    sigmoid_total += weight.get_weight() * weight.L_neuron.value
                sigmoid_total += node.bias
                # print(sigmoid_total)
                # try:
                node.value = 1/(1 + np.exp(-sigmoid_total))
                # except:

        output = []
        for node in layers[-1].nodes:
            output.append(node.value)
        return output

    def prediction(self, outputs, output_values):
        guess = 0
        for i in range(len(outputs)):
            if output_values[i] > guess:
                guess = i
        return outputs[guess]

    def cost(self, output_values, outputs, expected):
        high_value = 0
        for i in range(len(outputs)):
            if outputs[i] == expected:
                high_value = i
        compare = []
        for j in range(len(output_values)):
            if j != high_value:
                compare.append(float(0))
            else:
                compare.append(float(1))
        cost = 0
        for f in range(len(output_values)):
            cost += (output_values[f]-compare[f]) ** 2
        return cost, compare



    def set_output(self):
        output = []
        label = self.data_instance.label_col
        if not self.data_instance.regression:
            for index, row in self.data_instance.train_df.iterrows():
                if row[label] not in output:
                    output.append(row[label])
        else:
            for index, row in self.data_instance.train_df.iterrows():
                if row[label] not in output:
                    output.append(row[label])
        return sorted(output)

    def back_prop(self, layers, compare):  # oh boy here we go wish me luck
        # for i in range(len(layers)-1, 0, -1):
        #     for node in layers[i].nodes:
        #         node.bias_change += (-(1-node.value)*node.value*(1-node.value))
        #         for weight in node.incoming_weights:
        #             weight.weight_change += (-(1-cost)*node.value*(1-node.value)*weight.L_neuron.value)
        for i in range(len(layers)-1, 0, -1):
            j = 0
            for node in layers[i].nodes:
                if i == len(layers)-1:
                    # print(compare)
                    node.delta = (node.value-compare[j])*node.value*(1-node.value)
                    # print(type(-(compare[j]-node.value)*node.value*(1-node.value)))
                    j += 1
                else:
                    start = 0
                    for weight in node.outgoing_weights:
                        start += weight.R_neuron.delta * weight.weight
                    node.delta = node.value*(1-node.value)*start
                node.bias_change += node.delta
                # print(node.bias_change)
                for weight in node.incoming_weights:
                    weight.weight_change += weight.L_neuron.value*node.delta
                    # print(weight.weight_change)
        return

    def gradient_descent(self, layers, eta, alpha, compare, row):
        for i in range(len(layers)-1, 0, -1):
            for node in layers[i].nodes:
                node.prev_bias_change = node.bias_change
                node.bias_change = 0
                for weight in node.incoming_weights:
                    weight.prev_change = weight.weight_change
                    weight.weight_change = 0
        for i in range(len(row)):
            self.sigmoid(layers, row[i])
            print("adjusting values for different input")  # TODO: remove while recording video
            self.back_prop(layers, compare[i])
            print("backpropping")  # TODO: remove while recording video
        changes = []
        print("adjusting biases and weights for the last %d inputs" % (len(row)))  # TODO: remove while recording video
        for j in range(len(layers) - 1, 0, -1):
            # print(j)
            for node in layers[j].nodes:
                # node.prev_bias_change = node.bias_change
                # print(node.bias_change)
                # print(len(row))
                node.bias_change = -eta * node.bias_change/len(row)
                node.bias += node.bias_change# +alpha*node.prev_bias_change
                changes.append(node.bias_change)
                # print(node.bias_change)
                # print(node.bias_change)
                for weight in node.incoming_weights:
                    # weight.prev_change = weight.weight_change
                    weight.weight_change = -eta*weight.weight_change/len(row)
                    weight.weight += weight.weight_change# +alpha*weight.prev_change
                    changes.append(weight.weight_change)
                    # print(weight.weight_change)
        return changes, len(changes)

    # def run_it(self, train, hidden_layers, hidden_nodes, eta, alpha):
    #     network =


class NetworkClient:
    def __init__(self, data_instance):
        self.data_instance = data_instance

    def prepare(self):
        self.data_instance.split_data()
        if self.data_instance.regression:
            self.data_instance.regression_data_bins(9, True)

    def train_it(self, hidden_layers, hidden_nodes, eta, alpha, stoch):
        saved = None
        network = NeuralNetwork(self.data_instance)
        layers, output_layer = network.make_layers(hidden_layers, hidden_nodes)
        output_predictions = []
        costs = []
        compare = []
        for index, row in self.data_instance.train_df.iterrows():
            output_predictions.append(network.sigmoid(layers, row.drop(self.data_instance.label_col)))
            cos, comp = network.cost(output_predictions[-1], output_layer, row[self.data_instance.label_col])
            costs.append(cos)
            compare.append(comp)
        tries = 0
        while True:
            tries += 1
            # print(tries)
            # f=0
            group = []
            comp_group = []
            check_group = []
            j = 0
            for index, row in self.data_instance.train_df.iterrows():
                j += 1
                if j % stoch == 0:
                    changes, length = network.gradient_descent(layers, eta, alpha, comp_group, group)
                    # print(check_group)
                    # print(output_layer)
                    # print(comp_group)
                    group = []
                    comp_group = []
                group.append(row.drop(self.data_instance.label_col))
                check_group.append(row[self.data_instance.label_col])
                cos, comp = network.cost(output_predictions[-1], output_layer, row[self.data_instance.label_col])
                comp_group.append(comp)
                # for j in range(stoch):
                #     group.append(costs[f+j])
                #     comp_group.append(compare[f+j])
                # f+=1
            # for i in range(0, len(costs), stoch):
            #     group = []
            #     comp_group = []
            #     for j in range(stoch):
            #         if i+j < len(costs):
            #             # print(len(costs))
            #             # print(i+j)
            #             group.append(costs[i+j])
            #             comp_group.append(compare[i+j])
            #
            #     changes, length = network.gradient_descent(layers, group, eta, alpha, comp_group)
            #     # print(changes)
            output_predictions = []
            costs = []
            for index, row in self.data_instance.train_df.iterrows():
                output_predictions.append(network.sigmoid(layers, row.drop(self.data_instance.label_col)))
                costs.append(network.cost(output_predictions[-1], output_layer, row[self.data_instance.label_col])[0])
            # print(changes)
            if saved is None:
                saved = costs
            else:
                if tries % 100 == 0:
                    print("Summed Saved: ", sum(saved), " Summed Costs: ", sum(costs))
                if sum(saved) > sum(costs):
                    saved = costs

                else:
                    break

            checker = .0005
            test = 0
            # for i in range(len(changes)):
            #     my_break = False
            #     test += abs(changes[i])
            #  TODO: put in if statement checking cost
            if all(abs(x) <= checker for x in changes):  # changes.all() <= checker:  # abs(changes[i])
                # my_break = True
                break
        # print(my_break)
        # if my_break:
        #     break
        return layers, output_layer, network

    def testing(self, layers, output_set, network):
        correct = 0
        total = 0
        for index, row in self.data_instance.test_df.iterrows():
            output_prediction = network.sigmoid(layers, row.drop(self.data_instance.label_col))
            if network.prediction(output_set, output_prediction) == row[self.data_instance.label_col]:
                correct += 1
            total += 1
        return (correct/total)


class Layer:
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


class Neuron:
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
    def __init__(self, L_neuron, R_neuron):
        self.L_neuron = L_neuron
        self.R_neuron = R_neuron
        self.weight = float(random.randint(-1, 1))/100
        self.weight_change = 0
        self.prev_change = 0
        self.momentum_cof = .5
        self.eta = .1

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight
