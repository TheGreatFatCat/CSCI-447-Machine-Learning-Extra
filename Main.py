"""
Justin Keeling
Alex Harry
John Lambrecht
Andrew Smith

Institution: Montana State University
Course: CSCI-447 Machine Learning
Instructor: John Shepherd

File: Controls the flow of the code by calling functions and implementing objects in other files to create a stacked
auto encoder.
"""

from Data import Data
import pandas as pd
import csv
from Autoencoder import AutoEncoder


def load_data():
    """
    loads the data (csv) files
    :return: list of Data instances
    """
    with open('data/segmentation.data') as fd:
        reader = csv.reader(fd)
    data_list = [Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False),
                 Data('car', pd.read_csv(r'data/car.data', header=None), 5, False),
                 Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None, skiprows=5), 0, False),
                 Data('machine', pd.read_csv(r'data/machine.data', header=None), 0, True),
                 Data('forest_fires', pd.read_csv(r'data/forestfires.data', header=None), 12, True),
                 Data('wine', pd.read_csv(r'data/wine.data', header=None), 0, True)]
    return data_list


class Main:
    def __init__(self):
        self.data_list = load_data()

    def run_data_stacked_autoencoder(self):
        """
        Run all the data and create an auto encoder
        :return:
        """
        show_struct = True

        for data in self.data_list:
            print("----------------------------------All Data: %s----------------------------------" % data.name)
            print("___________________________________\nData Set: ", data.name)
            data.df = (data.df - data.df.mean()) / (data.df.max() - data.df.min())  # normalize
            data.split_data(data_frame=data.df)  # sets test and train data
            if data.name is "Segmentation":
                auto = AutoEncoder(3, False, [data.df.shape[1] - 1, data.df.shape[1] - 3, data.df.shape[1] - 1],
                                   data.train_df.shape[1], 0.01, 0.45)  # set hyperparameters
            else:
                auto = AutoEncoder(3, False, [data.df.shape[1] - 1, data.df.shape[1] - 3, data.df.shape[1] - 1],
                                   data.train_df.shape[1], 0.01, 0.45)  # set hyperparameters
            auto.fit_stacked_auto_encoder(data.train_df)  # run with stack
            if show_struct:
                auto.print_layer_neuron_data()  # print the network
                show_struct = False
            auto.test(data.test_df)  # test network

    def run_data_autoencoder(self):
        """
        Run all the data and create an auto encoder
        :return:
        """
        show_struct = True

        for data in self.data_list:
            print("----------------------------------All Data: %s----------------------------------" % data.name)
            print("___________________________________\nData Set: ", data.name)
            data.df = (data.df - data.df.mean()) / (data.df.max() - data.df.min())  # normalize
            data.split_data(data_frame=data.df)  # sets test and train data
            if data.name is "Segmentation":
                auto = AutoEncoder(3, False, [data.df.shape[1] - 1, data.df.shape[1] - 3, data.df.shape[1] - 1],
                                   data.train_df.shape[1], 0.01, 0.45)  # set hyperparameters
            else:
                auto = AutoEncoder(3, False, [data.df.shape[1] - 1, data.df.shape[1] - 3, data.df.shape[1] - 1],
                                   data.train_df.shape[1], 0.01, 0.45)  # set hyperparameters
            auto.fit_auto_encoder(data.train_df)  # run with stack
            if show_struct:
                auto.print_layer_neuron_data()  # print the network
                show_struct = False
            auto.test(data.test_df)  # test network

if __name__ == '__main__':
    Main().run_data_stacked_autoencoder()
    # Main().run_data_autoencoder()
