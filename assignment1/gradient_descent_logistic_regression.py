'''
Created on April 10th, 2018
author: Julian Weisbord
description: Gradient descent for logistic regression with L2 Regularization
'''
import csv
import numpy as np
import matplotlib.pyplot as plt

N_EPOCHS = 10
LEARNING_RATE = .1
PLOT = False
LAMBDA = .00000001

def load_data(filename):
    '''
    Description: Load a csv file of data and features
    Input: filename <String> path of input csv file
    Return: X <numpy array> usps data with different features
            Y <numpy array> label for each data example
    '''

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
    '''
    Description: Apply the sigmoid function to the data.
    Input: weight_param <numpy matrix>
           x_param <numpy matrix>
    Return: sig <int> The output prediction label
    '''

    denom_sigmoid = np.longdouble(1 + np.exp(np.dot(-weight_param, x_param)))
    sig = np.longdouble(np.divide(1, denom_sigmoid, where=denom_sigmoid!=0.0))
    return sig

def gradient_descent(X, Y, L2_Regularization=False):
    '''
    Description: Update the weights with gradient descent
    Input: X <numpy matrix> usps data with different features
           Y <numpy array> label for each data example
           L2_Regularization <Boolean> True if using L2 Regularization
               with logistic regression
    Return: sig <int> The output prediction label
    '''
    example_accuracy = []
    X = np.c_[np.ones((X.shape[0])), X]  # Add bias of 1 to each example
    feature_len = X.shape[1]
    example_count = np.longdouble(X.shape[0])
    print("X.shape ", X.shape)
    # Random weight vector with shape equal to number of features
    w = np.zeros(feature_len)
    l2_reg = 0
    step = 0
    correct_count = 0
    while(step < N_EPOCHS):
        print("Iteration: ", step)
        grad = np.zeros(feature_len, dtype=np.longdouble)
        for example in range(example_count):
            # y_hat is the predicted output
            y_hat = sigmoid(w.T, X[example])
            if L2_Regularization:
                l2_reg = LAMBDA * w  # = d/dw(.5*lambda*||w^2||)

            if y_hat >= .5:
                y_hat = 1
            loss = y_hat - Y[example]
            if loss[0] == 0:
                correct_count += 1
                print(correct_count)
            grad += loss[0] * X[example] + l2_reg

        w += -LEARNING_RATE * grad

        step += 1
        example_accuracy.append(np.float(correct_count / example_count))
        correct_count = 0


    print(" Accuracy per Epoch: ", example_accuracy)

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
    w_L2_train, example_accuracy_L2_train = gradient_descent(X, Y, L2_Regularization=True)
    w_L2_test, example_accuracy_L2_test = gradient_descent(X_test, Y_test, L2_Regularization=True)
    print("Example accuracy no L2_Regularization: ", example_accuracy)
    print("example_accuracy_L2_train: ", example_accuracy_L2_train)
    print("example_accuracy_L2_test: ", example_accuracy_L2_test)


if __name__ == "__main__":
    main()
