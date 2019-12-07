"""
Justin Keeling
Alex Harry
John Lambrecht
Andrew Smith

Instituiton: Montana State University
Course: CSCI-447 Machine Learning
Instructor: John Shepherd

File:
"""

import random
import math
import numpy as np


class SAE:
    """
    SAE: Stacked Auto Encoder
    A SAE is an auto encoders whose N hidden layers are also N auto encoders.


    """
    def __init__(self, data_instance, input_output_size, num_layers):

        self.data_obj = data_instance
        self.io_size = input_output_size
        self.num_layers = num_layers
        self.neural_net = None



