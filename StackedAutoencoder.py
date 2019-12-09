"""
Justin Keeling
Alex Harry
John Lambrecht
Andrew Smith

Institution: Montana State University
Course: CSCI-447 Machine Learning
Instructor: John Shepherd

File: Contains the classes and function to create a stacked auto encoder. Structure associated to AutoEncoder.py
"""

import random
import math
import numpy as np


# TODO: arbitrary number of auto encoder layers (hidden layer); in our case, hidden layer is new auto encoder for stack,

class SAE:
    """
    SAE: Stacked Auto Encoder
    A SAE is an auto encoders whose N hidden layers are also N auto encoders.

    Three layers: Input, hidden, and output
    Structure: Encoder and Decoder
    Encoder: Compresses data. or maps input data into a hidden representation.
    Decoder: Reconstructs the data back to the input.

    Specifics: uses back propagation to tune and train. Utilizes Feed forward. Uses Neural Network to predict.

    This class contains functions to structure, fit, and  predict auto encoders
    """

    def __init__(self, data_instance, input_output_size, num_layers):

        self.data_obj = data_instance
        self.io_size = input_output_size
        self.num_layers = num_layers
        self.neural_net = None



