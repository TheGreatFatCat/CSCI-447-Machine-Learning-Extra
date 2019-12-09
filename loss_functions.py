import data as data
import pandas as pd
from PAM import KNN
import math


class LF:

    def zero_one_loss(self, predicted, labels):

            """
                 Condense the data set by instantiating a Z = None. Add x_initial to Z if initial class(x_initial) != class(x)
                 where x is an example in Z.
                 So: Eliminates redundant data.
                 :param predicted: Predicted labels
                 :param labels: actual labels
                 :return: the zero one less type float
             """
            for iterator in range(len(predicted)):
                predicted[iterator] = round(predicted[iterator])
            length_of_labels = len(labels) # Length of entire predicted dataset
            zero_count = 0  # count of correctly classified examples
            one_count = 0  # count of incorrectly classified examples
            # Following code for using zip to loop along two lists source: https://stackoverflow.com/questions/1919044/is-there-a-better-way-to-iterate-over-two-lists-getting-one-element-from-each-l, user: sateesh, Agostino
            for (predict, label) in zip(predicted, labels):  # go through all the points at the same time
                if predict == label:
                    zero_count += 1
                else:
                    one_count += 1
            # print("\n--- KNN Classified", zero_count, "Examples Correctly and", one_count, "Incorrectly---")
            print("\n--- Zero One Loss:", (one_count/length_of_labels), "---")
            return one_count/length_of_labels  # Return zero one loss




    def mean_squared_error(self, predicted_data, actual_data):
        """
        :param predicted_data:  list of predicted values for datapoints (assume same order)
        :param actual_data: actual values for those said data points  (assume same order)
        :return MSE from the predicted data
         """
        n = len(actual_data)  # get out n for MSE
        sum = 0  # sum of the MSE squared values

        # Following code for using zip to loop along two lists source: https://stackoverflow.com/questions/1919044/is-there-a-better-way-to-iterate-over-two-lists-getting-one-element-from-each-l, user: sateesh, Agostino
        for (predict, true) in zip(predicted_data, actual_data): # go through all the points at the same time
            currentSum = (true - predict) ** 2  # square it
            sum += currentSum # add current to total sum

        # divide by n
        sum = sum/n
        print("\n--- Mean Squared Error:", sum,  "---\n")

        return sum # done, return su