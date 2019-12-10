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
        :return:
        """
        data = Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False)  # load data
        df = data.df.sample(n=10)  # minimal data frame
        data.split_data(data_frame=df)  # sets test and train data
        auto = AutoEncoder(3, False, [3, 2, 3], df.shape[1])
        auto.initialize_auto_encoder(data_obj=data)
        auto.print_layer_neuron_data()
        """Structure is good to go"""

        current = auto.output_layer
        while True:
            print(current.no_of_nodes)
            if current is auto.input_layer:
                break
            current = current.get_previous_layer()
        """traversing is good; from printing above (forwards) and right above (backwards)"""



if __name__ == '__main__':
    unittest.main()
