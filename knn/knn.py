
import math
import operator
# import matplotlib.pyplot as plt
import numpy as np


def gen_k_values():
    k = []
    for i in range(1, 52, 2):
        k.append(i)
    return k

def knn(train, test):
    distances = []

    ex_count = 0
    # Iterate through each row not including the last element
        # which is the label
    # print("train: ", train)
    # print("test: ", test)
    # exit()
    for row in train[:, 0:-1]:
        # print("row: ", row)
        # print("ex_count: ", ex_count)
        # print("test[ex_count]", test[ex_count])
        # print("test[ex_count][0:-1] ", test[ex_count][0:-1])
        distance = euclidean_dist(test[0:-1], row)
        distances.append((train[ex_count][-1], distance))
        ex_count += 1

    distances.sort(key=operator.itemgetter(1))
    # print("distances", distances)
    return distances

def load_data():
    train_data = np.genfromtxt("./knn_data/knn_train.csv", delimiter=",")
    val_data = np.genfromtxt("./knn_data/knn_test.csv", delimiter=",")
    X_train = train_data[:, 1:31]
    Y_train = train_data[:, 0:1]
    X_test = val_data[:, 1:31]
    Y_test = val_data[:, 0:1]
    return X_train, Y_train, X_test, Y_test


def euclidean_dist(test_row, train_row):
    distance = 0
    num_features = 0
    for feature in test_row:
        # print("train_row[num_features]", train_row[num_features])
        distance += pow((feature - train_row[num_features]), 2)
        num_features += 1
    return math.sqrt(distance)


def accuracy(train, test, K, leave_out=False):
    knn_alg = []
    error = []

    if leave_out:
        knn_alg = knn(train, test)
        error = []
        for i, k in enumerate(K):
            incorrect_num = 0
            err = sum(pair[0] for pair in knn_alg[:k])
            if err > 0 and test[-1] == -1 or err < 0 and test[-1] == 1:
                incorrect_num += 1
            error.append(incorrect_num)

        return error

    for example in test:
        knn_alg.append(knn(train, example))

    for k_val in K:
        incorrect_count = np.float32(0.0)
        row_num = 0
        for row in knn_alg:
            row_error = 0
            for dist in row[:k_val]:
                row_error += dist[0]
            if(test[row_num][-1] == 1 and row_error < 0):
                incorrect_count += 1
            if(test[row_num][-1] == -1 and row_error > 0):
                incorrect_count += 1
            row_num += 1
            # print("row_num",  row_num)
        err = incorrect_count / len(train)
        error.append(err)

    return error





def main():
    pass

if __name__ == "__main__":
    main()
