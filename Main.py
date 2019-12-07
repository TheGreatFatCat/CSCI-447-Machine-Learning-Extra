from Data import Data
import pandas as pd
import csv

def load_data():
    """
    loads the data (csv) files
    :return: list of Data instances
    """
    with open('data/segmentation.data') as fd:
        reader = csv.reader(fd)
    data_list = [Data('abalone', pd.read_csv(r'data/abalone.data', header=None), 8, False),
                 Data('car', pd.read_csv(r'data/car.data', header=None), 5, False),
                 Data('segmentation', pd.read_csv(r'data/segmentation.data', header=None, skiprows=4), 0, False),
                 Data('machine', pd.read_csv(r'data/machine.data', header=None), 0, True),
                 Data('forest_fires', pd.read_csv(r'data/forestfires.data', header=None), 12, True),
                 Data('wine', pd.read_csv(r'data/wine.data', header=None), 0, True)]
    return data_list
    # cat: abalone, car, segmentation
    # reg: machine, forestfires, wine


class Main:
    def __init__(self):
        self.data_list = load_data()

    # def perform_KNN(self, k_val, query_point, train_data):

