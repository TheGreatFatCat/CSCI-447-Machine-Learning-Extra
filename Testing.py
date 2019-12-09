import unittest
from Autoencoder import AutoEncoder, Layer, Neuron
from Data import Data, DataConverter
import pandas as pd
import numpy as np

import collections


class MyTestCase(unittest.TestCase):

    def test_auto_encoder_structure(self):
        """
        Test the layers in the auto encoder
        :return:
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)  # load data
        df = data.df.sample(n=10)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        auto = AutoEncoder(data, 3, False, [3, 2, 3])
        auto.print_layer_neuron_data()
        """Structure is good to go"""


if __name__ == '__main__':
    unittest.main()
