'''
Created on April 8th, 2018
author: Julian Weisbord
sources:
description: Apply linear regression
'''

import random
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DATA = "./housing_train.txt"
TEST_DATA = "./housing_test.txt"
D_FEATURES = [2, 4, 6, 8, 10]
PLOT = False

def calculate_weight(dataset, add_column=True, num_new_features=None, std_norm_vals=None):
    '''
    Description: Calculate weights using Linear Regression
    Input: add_column <Boolean> Add a bias column if True
           num_new_features <int> Number of random features to add
               to input data
    Return: w <numpy matrix> weight matrix
            X <numpy matrix> usps data with different features
            Y <numpy matrix> label for each data example
            row_count <float64> number of rows in data set
    '''
    X = []
    Y = []
    num_columns = 13  # Keep track of number of columns to insert more
    with open(dataset, 'r') as data:
        # Everything must be float64's or there are weird issues resulting in infinity
        row_count = np.float64(0.0)
        if add_column:
            num_columns += 1
        for row in data:
            # Load training features into X
            X.append(np.float64(row.split()[:-1]))
            if add_column:
                X[-1] = np.insert(X[-1], 0, np.float64(1.0))  # Make first column 1

            # Add additional features to input data if specified as an arg.
            if num_new_features:
                for column in range(num_new_features):
                    # print("column: ", column)
                    # print("num_new_features: ", num_new_features)
                    num_columns += 1
                    # print("num_columns: ", num_columns)
                    X[-1] = np.insert(X[-1], num_columns - 1, np.float64(std_norm_vals[column]))
                num_columns -= num_new_features
            # Load output features into Y
            Y.append(np.float64(row.split()[-1]))
            row_count += np.float64(1.0)
        X = np.array(X, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)

        X_t = np.transpose(X)

        X_t_X = np.matmul(X_t, X)
        if num_new_features:
            print("X_t_X", X_t_X)
            print("-----------------")

        inv_eigen_X = np.divide(np.float64(1.0), X_t_X, where=X_t_X!=0.0)
        # Above statement doesn't catch all divide by zero errors some of
            # these result in nan or infinity and should be made zero
        for row in inv_eigen_X:
            row[np.isnan(row)] = 0
            row[np.isinf(row)] = 0


        w = np.matmul(inv_eigen_X, np.matmul(X_t, Y))
        return w, X, Y, row_count

def calculate_ase(w, X, Y, num_examples):
    '''
    Description: Calculate weights using Linear Regression
    Input: num_examples <int> Total amount of usps data
           to input data
            w <numpy matrix> weight matrix
            X <numpy matrix> usps data with different features
            Y <numpy matrix> label for each data example
    Return: Average Square Error
    '''
    ase_c1 = np.transpose(np.subtract(Y, np.matmul(X, w)))
    ase_c2 = np.subtract(Y, np.matmul(X, w))
    print("ase_c1 ", ase_c1)
    print("ase_c2 ", ase_c2)

    ase = np.divide(np.matmul(ase_c1, ase_c2), num_examples)
    print("ase: ", ase)
    return ase

def main():
    # Q1 Load the training data into the corresponding X and Y matrices,
        # where X stores the features and Y stores the desired outputs.
        # The rows of X and Y correspond to the examples and the columns
        # of X correspond to the features. Introduce the dummy variable to
        # X by adding an extra column of ones to X (You can make this extra
        # column to be the first column. Changing the position of the added
        # column will only change the order of the learned weight and does not matter
        # in practice. Compute the optimal weight vector w.
        # Feel free to use existing numerical packages to perform the computation.
        # Report the learned weight vector.

    w_train, X_train, Y_train, num_examples_train = calculate_weight(TRAIN_DATA)
    print("w_train_1: ", w_train)
    w_test, X_test, Y_test, num_examples_test = calculate_weight(TEST_DATA)
    print("w_test_1: ", w_test)
    # Q2 Apply the learned weight vector to the training data and testing data respectively
        # and compute for each case the average squared error(ASE),
        # which is the SSE normalized by the total number of examples in the data.
        # Report the training and testing ASEs.

    train_ase = calculate_ase(w_train, X_train, Y_train, num_examples_train)
    test_ase = calculate_ase(w_test, X_test, Y_test, num_examples_test)
    print("train_ase_Q2: ", train_ase)
    print("test_ase_Q2: ", test_ase)

    #Q3 Remove the dummy variable (the column of ones) from X, repeat 1 and 2.
        # How does this change influence the ASE on the training and testing data?
        # Provide an explanation for this influence.

    w_train, X_train, Y_train, num_examples_train = calculate_weight(TRAIN_DATA, add_column=False)
    w_test, X_test, Y_test, num_examples_test = calculate_weight(TEST_DATA, add_column=False)

    train_ase = calculate_ase(w_train, X_train, Y_train, num_examples_train)
    test_ase = calculate_ase(w_test, X_test, Y_test, num_examples_test)
    print("train_ase w/out column of 1's: ", train_ase)
    print("test_ase w/out column of 1's: ", test_ase)

    # Q4 Modify the data by adding additional random features. You will do this to both
        # training and testing data. In particular, for each instance, generate d (consider d = 2, 4, 6, ...10,
        # feel free to explore more values) random features, by sampling from a standard normal distribution.
        # For each d value, apply linear regression to find the optimal weight vector and compute its
        # resulting training and testing ASEs. Plot the training and testing ASEs as a function of d.
        # What trends do you observe for training data and test data respectively?
        # Do more features lead to better prediction performance at testing stage?
        # Provide an explanation to your observations.
    ase_train_scores = []
    ase_test_scores = []

    for num_new_features in D_FEATURES:
        # std_norm_val = 1
        std_norm_vals = []
        for num in range(num_new_features):
            std_norm_vals.append(np.float64(num))
            std_norm_vals.append(np.float64(-num))
            # std_norm_val += 1


        w_train, X_train, Y_train, num_examples_train = calculate_weight(TRAIN_DATA,
                                                                         add_column=True,
                                                                         num_new_features=num_new_features,
                                                                         std_norm_vals=std_norm_vals)

        w_test, X_test, Y_test, num_examples_test = calculate_weight(TEST_DATA,
                                                                     add_column=True,
                                                                     num_new_features=num_new_features,
                                                                     std_norm_vals=std_norm_vals)
        print("w_train: ", w_train)
        print("w_test: ", w_test)
        train_ase = calculate_ase(w_train, X_train, Y_train, num_examples_train)
        test_ase = calculate_ase(w_test, X_test, Y_test, num_examples_test)

        ase_train_scores.append(train_ase)
        ase_test_scores.append(test_ase)

        print("train_ase:", train_ase)
        print("test_ase:", test_ase)

    print("ase_train_scores: ", ase_train_scores)
    print("ase_test_scores: ", ase_test_scores)
    # Plot ase's as a function of d
    if PLOT:
        plt.plot(D_FEATURES, ase_train_scores)
        plt.xlabel('D Values')
        plt.ylabel('Train Average Squared Error')
        plt.show()

        plt.plot(D_FEATURES, ase_test_scores)
        plt.xlabel('D Values')
        plt.ylabel('Test Average Squared Error')
        plt.show()

if __name__ == "__main__":
    main()
