'''
References: https://www.youtube.com/watch?v=QHOazyP-YlM
            https://www.youtube.com/watch?v=LDRbO9a6XPU
            https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
'''
from __future__ import print_function
import math
import numpy as np
import numbers



CLASS_VALUES = [-1.0, 1.0]
FEATURES = 30
MIN_ROWS = 11
VERBOSE = 0

class Leaf():
    def __init__(self, data_sub_tree):
        self.class_prediction = label_count(data_sub_tree)

class Node():
    def __init__(self, best_gain, greater_branch, lesser_branch, col_index, val):
        self.greater_branch = greater_branch  # list of greater rows
        self.lesser_branch = lesser_branch  # list of lesser rows
        self.best_gain = best_gain
        self.question_feature_index = col_index
        self.val = val
        self.left = None
        self.right = None
        self.leaf = False
        self.split = None


class DecTree():
    def __init__(self, data, test_data, depth, depth_limit=10000, stump=False):
        self.gain = 0
        self.build_dec_tree(data, test_data, depth, depth_limit,  stump=stump)

    def build_dec_tree(self, data, test_data, depth, depth_limit, stump=False):

        root_node = get_best_q(data)
        print("Root Node gain: ", root_node.best_gain)
        if stump:
            assert depth == 1
            self.build_tree(root_node, 1, depth_limit=1)
            # print("Type of root_node!!!!!: ", type(root_node))
            # if isinstance(root_node, Node):
            #     exit()
            self.print_tree(root_node, depth_limit=1)
            print("\n Information Gain: ", self.get_gain(root_node))
            print("Train Error", self.error(data, root_node))
            print("Test Error", self.error(test_data, root_node))
        else:
            self.build_tree(root_node, 1, depth_limit)
            self.print_tree(root_node, depth_limit=50)
            print("\n Information Gain: ", self.get_gain(root_node))
            print("Train Error", self.error(data, root_node))
            print("Test Error", self.error(test_data, root_node))

        return root_node

    def build_tree(self, node, depth_current, depth_limit):

        # Recursively create tree graph
        greater_branch = node.greater_branch
        lesser_branch = node.lesser_branch
        print("In build_tree!!!! Current depth: ", depth_current)
        if not greater_branch or not lesser_branch:
            # If the data isn't split
            max, class_count = self.get_class_majority(lesser_branch + greater_branch)
            node.right = node.left = max
            node.split = class_count
            # print("node.right and node.left get populated: ", node.left, node.right)
            return
        # If depth_limit is not yet reached:
        if depth_limit <= depth_current:
            node.left, node.split = self.get_class_majority(lesser_branch)
            node.right, node.split = self.get_class_majority(greater_branch)
            return
        if len(lesser_branch) <= MIN_ROWS:
            node.left = self.get_class_majority(lesser_branch)
        else:
            node.left = get_best_q(lesser_branch)
            self.build_tree(node.left, depth_current + 1, depth_limit)

        if len(greater_branch) <= MIN_ROWS:
            node.right, node.split = self.get_class_majority(greater_branch)
        else:
            node.right = get_best_q(greater_branch)
            self.build_tree(node.right, depth_current + 1, depth_limit)


    def get_class_majority(self, data_sub_tree):
        class_count = label_count(data_sub_tree)
        # print("class_count", class_count)
        if CLASS_VALUES[0] not in class_count:
            return class_count[CLASS_VALUES[1]], class_count
        elif CLASS_VALUES[1] not in class_count:
            return class_count[CLASS_VALUES[0]], class_count

        max_class = max(class_count[CLASS_VALUES[0]], class_count[CLASS_VALUES[1]])
        return max_class, class_count

    def print_tree(self, node, depth=1, depth_limit=10000):
    	# If the node is a leaf, meaning all of the data to the left and
            # right of it are split perfectly into seperate classes
            # then print " Predict" + prediction
        print("depth", depth)
        if depth_limit < depth:
            return

        if isinstance(node, Node):
            # print("IS INSTANCE")
            print("\n[greater or less than: {} ?]".format(node.val))

        else:
            # print("NOT INSTANCE")
            # print(node)
            return

        print("Example count left {} Example Count right {}: ".format(len(node.lesser_branch), len(node.greater_branch)))
        if isinstance(node.right, int):
            # print("int instance")
            print([node.split.keys()[0]])
        if isinstance(node.left, int):
            # print("int instance")
            print([node.split.keys()[1]])
        if len(node.lesser_branch) == 0 or len(node.greater_branch) == 0:
            print("Leaf Node")
            print([node.split.keys()[0]])
            print([node.split.keys()[1]])
            return

        if depth_limit > depth:
            # Recursively call print_tree for the greater branches
            print(" " + "----> Greater:")
            self.print_tree(node.right, depth + 1)

            # Recursively call print_tree for the lesser branches
            print(" " + "----> Lesser:")
            self.print_tree(node.left, depth + 1)


    def get_gain(self, node):
        return node.best_gain

    def error(self, dataset, tree):
        incorrect = 0
        # print("example: ", example)
        if isinstance(dataset, int):
            print("dataset not list", dataset)
            return

        for example in dataset:
            pred = self.gen_prediction(tree, example)
            if pred < example[-1]:
                incorrect += 1
        error_rate = np.float32(incorrect) / len(dataset)
        return error_rate

    def gen_prediction(self, node, example):
        # if node:
        #     print("node: ", node)
        #     print("Node was not None, node.val: ", node.val)
        #     # return
        # else:
        #     print("Node was None")
        if node.val > example[node.question_feature_index]:
            if isinstance(node.right, Node):
                return self.gen_prediction(node.right, example)
            else:
                return node.val
        else:
            if isinstance(node.left, Node):
                return self.gen_prediction(node.left, example)
            else:
                return node.val


def get_best_q(data):
    best_gain = 0
    start_entropy = entropy(data)
    # print("start_entropy gain: ", start_entropy)
    # if start_entropy != 0.0 and start_entropy != 1.0:
    #     print("Holy start_entropy: ", start_entropy)
    #     exit()
    if start_entropy == 0:
        start_entropy = 1
    for col_index in range(FEATURES - 1):
        # print("col_index: ", col_index)
        for example in data:
            greater_rows, lesser_rows = partition(col_index, data[col_index], example[col_index], data)
            # print("Length of rows: ", len(greater_rows), len(lesser_rows))
            if len(greater_rows) == 0 or len(lesser_rows) == 0:
                # print("continuing from get_best_q because greater or lesser rows were 0")
                continue
            gain = start_entropy - information_gain(data, (greater_rows, lesser_rows))
            # if gain != 0.0 and gain != 1.0:
            #     print("Holy Gain: ", gain)
            #     exit()
            if gain > best_gain:
                print("New best gain: ", gain)
                best_gain = gain
    node = Node(best_gain, greater_rows, lesser_rows, col_index, example[col_index])
    return node


def information_gain(data, partitions):

    # calculate uncertainty of left, right child nodes
    lesser_data, greater_data = partitions
    lesser_ratio = np.float32(len(lesser_data)) / np.float32(len(data))
    greater_ratio = np.float32(len(greater_data)) / np.float32(len(data))
    # start_entropy = entropy(data)  # initial entropy
    if len(lesser_data) == 0 or len(greater_data) == 0:
        return 0
    info_gain = (lesser_ratio * entropy(lesser_data)) - (greater_ratio * entropy(greater_data))
    # print("Info Gain: ", info_gain)
    return info_gain


def entropy(data_sub_tree):
    # for x, class_val in enumerate(CLASS_VALUES):
    #     if class_val > 0:
    #         pos_index = x
    #     if class_val < 0:
    #         neg_index = x
    # count all vals in tree with one and divide by length of data
    class_count = label_count(data_sub_tree)
    # if not class_count:
    #     entropy = 0
    #     return entropy
    # print(" class_count: ", class_count, "len(data_sub_tree): ", len(data_sub_tree))
    p_plus = class_count[1.0] / len(data_sub_tree)
    # count all vals in tree with negative one and divide by length of data

    p_min = np.float32(class_count[-1.0]) / np.float32(len(data_sub_tree))
    if p_plus == 0 and p_min == 0:
        # print("Data split perfectly")
        return 0
    elif p_plus != 0:
        return -p_plus * math.log(p_plus, 2)
    elif p_min != 0:
        return -p_min * math.log(p_min, 2)

    print("p_plus: ", p_plus, "p_min: ", p_min)
    entropy_calculation = - p_plus * math.log(p_plus, 2) - p_min * math.log(p_min, 2)
    print("entropy calculation: ", entropy_calculation)
    return entropy_calculation

def label_count(data_sub_tree):

    class_count = {}
    for val in CLASS_VALUES:
        # initialize classes in count dict
        class_count[val] = 0
    for example in data_sub_tree:
        label = example[-1]
        if label not in class_count:
            class_count[label] = 0
        class_count[label] += 1
    return class_count



def partition(col_index, feature, part_threshold, data):
    greater_rows = []
    lesser_rows = []
    for example in data:
        assert col_index != 30  # this is the label
        if VERBOSE > 2:
            print("Part Threshold: ", part_threshold)
            print("example[col_index]", example[col_index])

        if example[col_index] > part_threshold:
            # print("example[col_index] > part_threshold")
            greater_rows.append(example)
        if example[col_index] <= part_threshold:
            # print("example[col_index] <= part_threshold")
            # exit()
            lesser_rows.append(example)
    return greater_rows, lesser_rows




def opt_bin_split(data):
    pass
