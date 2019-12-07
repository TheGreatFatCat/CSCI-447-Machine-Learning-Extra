import pandas as pd
import numpy as np

CATEGORICAL_DICTIONARY = {}
ALLOWED_DICTIONARY = {}


class Data:
    def __init__(self, name, df, label_col, regression):
        self.name = name
        CATEGORICAL_DICTIONARY = {}
        ALLOWED_DICTIONARY = {}
        data_converter = DataConverter()
        self.set_allowed_dictionary(df.copy())
        self.df = data_converter.convert_to_numerical(df.copy())
        self.test_df = None
        self.train_df = None
        self.label_col = label_col
        self.k_dfs = None
        self.regression = regression


    @staticmethod
    def get_row_size(df):
        """
        call to get any df row size
        """
        return df.shape[0]

    @staticmethod
    def get_col_size(df):
        """
        call to get any df column size
        """
        return df.shape[1]

    def split_data(self, data_frame, train_percent=.8):
        """
        splits the data according to the train percent.
        :return:
        """
        # TODO:if dataframe or train_percent are empty, use if statement to split data in a universal way
        # use numpys split with pandas sample to randomly split the data
        # self.train_df = temp_df.sample(frac=0.75, random_state=0)
        # self.test_df= temp_df.split(self.train_df)
        self.train_df, self.test_df = np.split(data_frame.sample(frac=1), [int(.8 * len(data_frame))])
        # print("Train ", self.train_df.shape)
        # print("Test ", self.test_df.shape)

    def split_k_fold(self, k_val, dataset):
        """
        Split data into list of K different parts
        :param k_val: k value to set size of folds.
        :return: list of lists where arranged as follows [[train,test], [train, test]] repeated k times
        where train is traing data (index 0) and test is testing data (index 1)
        """
        k__split_data = np.array_split(dataset, k_val)  # splits dataset into k parts
        # now we need to split up data into 1 list and k others combined into 1 list for test/train
        test_train_sets = []
        temp_list = [None] * 2
        length = len(k__split_data)
        # create these new lists and add them to test_train_sets
        for i in range(length):  # go through every split list
            # APPARENTLY PYTHON DEVS THOUGHT IT WAS A GOOD FUCKING IDEA TO MAKE LISTS THAT HAVE DIFFERENT NAMES BOTH
            # REMOVE VALS WHEN THE REMOVE FUNCTION IS APPLIED TO ONE OF THEM.   WHY GOD WHY
            data_to_combine = np.array_split(dataset, k_val)
            temp_list[0] = k__split_data[i]
            del data_to_combine[i]
            temp_list[1] = pd.concat(data_to_combine)
            test_train_sets.append(temp_list)

            # TODO: I don't think we need  i +=1, but we can check

    def k_fold(self, df, k_val):
        # TODO: Test this and see if the data is split accordingly and no elements in a df have are equal to another df
        column_size = self.get_col_size(df)  # get column size
        group = int(column_size / k_val)  # get number of data points per group
        grouped_data_frames = []
        for g, k_df in df.groupby(np.arange(len(column_size)) // group):
            grouped_data_frames.append(k_df)
        return grouped_data_frames

    def quick_setup(self):
        self.split_data(train_percent=0.8)

    def set_allowed_dictionary(self,
                               df):  # Get the unique values of strings per column.  This is for conversion in DataConverter
        for allowed_keys in range(df.shape[1]):
            unique_values = df[
                allowed_keys].unique()  # Unique values in dataframe source: https://chrisalbon.com/python/data_wrangling/pandas_list_unique_values_in_column/, : Chris Albon, Bob Haffner
            for item in unique_values:
                if allowed_keys not in ALLOWED_DICTIONARY.keys():  # Append all items allowed in a column to a dictionary for later converting

                    ALLOWED_DICTIONARY[allowed_keys] = [item]
                else:
                    ALLOWED_DICTIONARY[allowed_keys].append(item)

    def regression_data_bins(self, bin_size, quartile):
        """
        Used for the regression data sets(ONLY the label columns)!
        :param quartile: boolean value that determines if quartile based binning
        :param bin_size: number of bins to cut data into
        :return: data frame with regression 'label' column in bins.
        """
        bin_df = self.df.copy()  # do not overwrite
        if quartile:  # use standard deviation approach
            bin_df[self.label_col] = pd.qcut(self.df[self.label_col], retbins=False, q=bin_size)
        else:  # cut into bin_size number of equal bins
            bin_df[self.label_col] = pd.cut(self.df[self.label_col], bin_size)
        # TODO remove print statement... it is here for understanding... run with test_discretize in testing file to see output
        print("\nentire dataframe\n", bin_df)
        print("\nLabel column specifically... \n", bin_df[self.label_col])
        print(type(bin_df.iloc[0][8]))
        return bin_df


class DataConverter:

    def convert_to_numerical(self, data):  # Convert data to all numerical data frame

        counter = 1
        temp_data_frame_list = []  # Temp list to be returned converted as a dataframe
        for row in data.iterrows():  # Loop through the rows of the dataframe
            temp_row_list = []
            allowed_keys = 0
            for item in row[1]:  # Loop through each item in the row
                if item not in CATEGORICAL_DICTIONARY.keys():  # Checks if item is in dictionary
                    if type(item) is str:
                        CATEGORICAL_DICTIONARY[item] = [type(item),
                                                        counter]  # Add item in the dictionary and assign a value
                        counter += 1
                    else:
                        CATEGORICAL_DICTIONARY[item] = [type(item),
                                                        item]  # Add item in the dictionary and assign a value
                    allowed_keys += 1
                temp_row_list.append(CATEGORICAL_DICTIONARY[item][1])  # Append to a tem list
            temp_data_frame_list.append(temp_row_list)  # Append the list to the temp_data_frame list
        return pd.DataFrame(temp_data_frame_list)  # Return dataframe

    def convert_data_to_original(self, data):  # Convert data back to categorical
        min_found = False
        temp_data_frame_list = []  # Temp list to return as a dataframe later
        for row in data.iterrows():  # Iterate the rows of the data set
            temp_row_list = []  # Temp list for row to be placed in dataframe
            allowed_key = 0
            for item in row[1]:  # Loop through each item in the list
                closest_point = [None, float(
                    'inf')]  # Maximum float value initializer. Source: https://stackoverflow.com/questions/10576548/python-usable-max-and-min-values, user: user648852
                for key, val in CATEGORICAL_DICTIONARY.items():  # Loop through the keys in the list and find the closest distance to a point.
                    min_found = False
                    value = val[1]  # Gets the numerical value of the dictionary
                    difference = float(item) - float(value)  # Gets the difference between the values
                    if difference < 0.0:  # Converts to a positive number if needed
                        difference *= -1
                    if difference < closest_point[
                        1]:  # Checks if the point is closer than the previous closest data point
                        if key in ALLOWED_DICTIONARY[allowed_key]:
                            closest_point = [key, difference]
                            if difference == 0.0:  # Breaks if the difference is zero, as no point will be closer.
                                temp_row_list.append(closest_point[0])  # Append to the row list
                                min_found = True
                                break

                allowed_key += 1
                if not min_found:
                    temp_row_list.append(closest_point[0])  # Append to the row list

            temp_data_frame_list.append(temp_row_list)  # Append the row to the new dataset list
        return pd.DataFrame(temp_data_frame_list)  # Returns a dataframe
