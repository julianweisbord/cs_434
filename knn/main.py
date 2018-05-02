
import numpy as np
# import matplotlib.pyplot as plt
import knn
from dec_tree import DecTree

PLOT = False
D = [1, 2, 3, 4, 5, 6]

def main():

    X_train, Y_train, X_test, Y_test = knn.load_data()
    print("X_train[0]", X_train[0])
    print(X_test)
    # Normalize, each feature will have a value between 0 and 1
    new_dim = np.newaxis
    X_train = X_train / X_train.sum(axis=1)[:, new_dim]
    # count = 0
    # for row in X_train:
    #     for feature in row:
    #         if float(feature) > 0.5:
    #             count +=1
    # print("Num train features greater than .5", count)
    X_test = X_test / X_test.sum(axis=1)[:, new_dim]
    print("X_train after normalize: ", X_train[0])
    train_normal = np.append(X_train, Y_train, axis=1)
    print("train[0]: ", train_normal[0])
    test_normal = np.append(X_test, Y_test, axis=1)
    print("test[0]: ", test_normal[0])
    #
    # # Implement the K-nearest neighbor algorithm, where K is a parameter.
    #
    K = knn.gen_k_values()
    train_err = knn.accuracy(train_normal, train_normal, K)
    print(train_err)

    # Part 2 of Question 1:
    train_error = knn.accuracy(train_normal, train_normal, K)

    temp = [[] for _ in range(len(test_normal))]
    cross_error = []

    for i, example in enumerate(train_normal):

        train_normal = np.delete(train_normal, i, 0)
        temp[i] = knn.accuracy(train_normal, example, K, leave_out=True)
        train_normal = np.insert(train_normal, i, example, 0)

    for total in np.sum(temp, axis=0):
        cross_error.append(np.float32(total) / (len(test_normal) -1))

    print("!!!!!!Cross results", cross_error)

    test_error = knn.accuracy(train_normal, test_normal, K)

    print("!!!!!!training results", train_error)
    print("!!!!!!testing results", test_error)
    print(len(train_error), len(test_error))


    if PLOT:
        plt.plot(K, train_error)
        plt.plot(K, test_error)
        print("len cross: ", len(cross_error))
        plt.plot(K, cross_error)
        plt.xlabel('K values')
        plt.ylabel('Error')
        plt.show()

    # Question 2, Decision Trees:
    # Create Decision Stump:
    # train = np.genfromtxt("knn_data/knn_train.csv", delimiter=",")
    # print("new train", train)
    # stump = DecTree(train_normal, test_normal, 1, depth_limit=1, stump=True)
    # for d in D:
    #     tree = DecTree(train_normal, test_normal, 10, d)


if __name__ == "__main__":
    main()
