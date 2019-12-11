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
        Test format of structures by visual as well
        Test traversing forwards and backwards. Visual check as well
        :return: None
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)  # load data
        df = data.df.sample(n=100)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        auto = AutoEncoder(1, False, [3, 2, 3], df.shape[1], 0.2, 0.45)
        auto.fit_auto_encoder(data_obj=data)
        auto.print_layer_neuron_data()
        auto.test(data.test_df)
        """Structure is good to go"""

        current = auto.output_layer
        while True:
            print(current.no_of_nodes)
            if current is auto.input_layer:
                break
            current = current.get_previous_layer()
        """traversing is good; from printing above (forwards) and right above (backwards)"""


    def test_stack_encoder_structure(self):
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)  # load data
        # data.df = data.df.sample(n= 500)  # minimal data frame
        data.df = (data.df - data.df.mean()) / (data.df.max() - data.df.min())
        data.split_data(data_frame=data.df)  # sets test and train data
        auto = AutoEncoder(3, False, [7, 5, 7], data.train_df.shape[1], 0.03, 0.45)
        auto.fit_stacked_auto_encoder(data.train_df)
        auto.print_layer_neuron_data()
        auto.test(data.test_df)


    def test_check(self):
        a = 1
        b = a
        print(a, b)
        a = 2
        print(a, b)


if __name__ == '__main__':
    unittest.main()
