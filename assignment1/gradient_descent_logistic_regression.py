'''
Created on April 10th, 2018
author: Julian Weisbord
sources:
description:
'''
import csv
import numpy as np
import matplotlib.pyplot as plt

N_EPOCHS = 10
LEARNING_RATE = .1
PLOT = False
LAMBDA = .00001

def load_data(filename):
    X = []
    Y = []

    with open(filename, 'r') as csvfile:
        X = [[int(x) for x in line] for line in csv.reader(csvfile, delimiter=',')]

        for line in X:
            Y.append([line[-1]])
            line.pop()

        print(X[0])
        print(Y)

    X = np.array(X, dtype=np.longdouble)
    Y = np.array(Y, dtype=np.longdouble)
    return X, Y

def sigmoid(weight_param, x_param):
    denom_sigmoid = np.longdouble(1 + np.exp(np.dot(-weight_param, x_param)))
    sig = np.longdouble(np.divide(1, denom_sigmoid, where=denom_sigmoid!=0.0))
    return sig

def gradient_descent(X, Y, L2_Regularization=False):
    example_accuracy = []
    X = np.c_[np.ones((X.shape[0])), X]  # Add bias of 1 to each example
    feature_len = X.shape[1]
    example_count = np.longdouble(X.shape[0])
    print("X.shape ", X.shape)
    # Random weight vector with shape equal to number of features
    w = np.zeros(feature_len)
    k = np.zeros(feature_len)
    step = 0
    correct_count = 0
    while(step < N_EPOCHS):
        print("Iteration: ", step)
        grad = np.zeros(feature_len, dtype=np.longdouble)
        for example in range(example_count):
            param_accuracy = []
            # y_hat is the predicted output
            y_hat = sigmoid(w.T, X[example])
            if L2_Regularization:
                reg = .5 * LAMBDA * np.linalg.norm(k, 2)
                y_hat +=reg
                # print(w)
                print(reg)
            if y_hat >= .5:
                y_hat = 1
            # y_hat[y_hat >= .5] = 1  # Replace all values greater than .5 with 1
            # print("y_hat : ", y_hat)
            loss = y_hat - Y[example]
            if loss[0] == 0:
                correct_count += 1
                print(correct_count)
            grad += loss[0] * X[example]


        w += -LEARNING_RATE * grad
        # example_accuracy.append(np.sum(param_accuracy))
        step += 1
        example_accuracy.append(np.float(correct_count / example_count))
        correct_count = 0


    print(" Accuracy per Epoch: ", example_accuracy)
    #
    # print("Total Accuracy: ", total_accuracy)
    # print("correct count: ", correct_count)
    return w, example_accuracy

def main():

    X, Y = load_data("./usps-4-9/usps-4-9-train.csv")
    X_test, Y_test = load_data("./usps-4-9/usps-4-9-test.csv")

    # Question 1
    w, example_accuracy = gradient_descent(X, Y)
    epoch_list = [epoch for epoch in range(N_EPOCHS)]
    if PLOT:
        plt.plot(epoch_list, example_accuracy)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.show()

    # Question 3
    w_L2, example_accuracy_L2 = gradient_descent(X, Y, L2_Regularization=True)
    w_L2, example_accuracy_L2 = gradient_descent(X_test, Y_test, L2_Regularization=True)
if __name__ == "__main__":
    main()
