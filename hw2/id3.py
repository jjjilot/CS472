#!/usr/bin/python3
#
# CIS 472/572 -- Programming Homework #1
#
# Starter code provided by Daniel Lowd, 1/25/2018
#
#
import sys
import re
# Node class for the decision tree
import node

# my imports
import math

train = None
varnames = None
test = None
testvarnames = None
root = None


# Helper function computes entropy of Bernoulli distribution with
# parameter p
def entropy(p):
    # >>>> YOUR CODE GOES HERE <<<<
    if p == 0 or p == 1:
        return 0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


# Compute information gain for a particular split, given the counts
# py_pxi : number of occurences of y=1 with x_i=1 for all i=1 to n
# pxi : number of occurrences of x_i=1
# py : number of ocurrences of y=1
# total : total length of the data
def infogain(py_pxi, pxi, py, total):
    # >>>> YOUR CODE GOES HERE <<<<
    if total == 0:
        return 0

    # Entropy of Y overall
    p_y = py / total
    h_y = entropy(p_y)

    # If the variable doesn't divide data don't split
    if pxi == 0 or pxi == total:
        return 0

    # Conditional probabilities
    p_y_given_x1 = py_pxi / pxi if pxi > 0 else 0
    p_y_given_x0 = (py - py_pxi) / (total - pxi) if (total - pxi) > 0 else 0

    # Entropy of Y conditioned on X_i
    h_y_given_x1 = entropy(p_y_given_x1)
    h_y_given_x0 = entropy(p_y_given_x0)

    # Weighted conditional entropy
    cond_entropy = (pxi / total) * h_y_given_x1 + ((total - pxi) / total) * h_y_given_x0

    # Information gain
    gain = h_y - cond_entropy
    return gain


# OTHER SUGGESTED HELPER FUNCTIONS:
# - collect counts for each variable value with each class label
# - find the best variable to split on, according to mutual information
# - partition data based on a given variable
def compute_counts(data):
    total = len(data)
    num_vars = len(data[0]) - 1  # all except the class label
    py = sum([x[-1] for x in data])  # total y = 1

    stats = []  # list of (py_pxi, pxi)

    for i in range(num_vars):
        pxi = sum([x[i] for x in data])
        py_pxi = sum([x[i] and x[-1] for x in data])  # Only count when both are 1
        stats.append((py_pxi, pxi))

    return stats, py, total


def choose_best_variable(data):
    stats, py, total = compute_counts(data)
    best_gain = -1
    best_var = None

    for i, (py_pxi, pxi) in enumerate(stats):
        gain = infogain(py_pxi, pxi, py, total)
        if gain > best_gain:
            best_gain = gain
            best_var = i

    return best_var


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        data.append([int(x) for x in p.split(l.strip())])
    return (data, varnames)


# Saves the model to a file.  Most of the work here is done in the
# node class.  This should work as-is with no changes needed.
def print_model(root, modelfile):
    f = open(modelfile, 'w+')
    root.write(f, 0)


# Build tree in a top-down manner, selecting splits until we hit a
# pure leaf or all splits look bad.
def build_tree(data, varnames):
    # >>>> YOUR CODE GOES HERE <<<<
    # Base case 1: empty
    if not data:
        return node.Leaf(varnames, 0)

    labels = [row[-1] for row in data]

    # Base case 2: all same label
    if all(label == labels[0] for label in labels):
        return node.Leaf(varnames, labels[0])

    best_var = choose_best_variable(data)

    # Partition data
    left = [row for row in data if row[best_var] == 0]
    right = [row for row in data if row[best_var] == 1]

    # Split on most common variable once data is stagnant (labels don't matter)
    if not left or not right:
        majority_label = max(set(labels), key=labels.count)
        return node.Leaf(varnames, majority_label)

    # print(f"Splitting on variable {best_var}")
    # print(f"Left size: {len(left)}, Right size: {len(right)}")
    
    # Again!
    left_branch = build_tree(left, varnames)
    right_branch = build_tree(right, varnames)

    return node.Split(varnames, best_var, left_branch, right_branch)


# "varnames" is a list of names, one for each variable
# "train" and "test" are lists of examples.
# Each example is a list of attribute values, where the last element in
# the list is the class value.
def loadAndTrain(trainS, testS, modelS):
    global train
    global varnames
    global test
    global testvarnames
    global root
    (train, varnames) = read_data(trainS)
    (test, testvarnames) = read_data(testS)
    modelfile = modelS

    # build_tree is the main function you'll have to implement, along with
    # any helper functions needed.  It should return the root node of the
    # decision tree.
    root = build_tree(train, varnames)
    print_model(root, modelfile)


def runTest():
    correct = 0
    # The position of the class label is the last element in the list.
    yi = len(test[0]) - 1
    for x in test:
        # Classification is done recursively by the node class.
        # This should work as-is.
        pred = root.classify(x)
        if pred == x[yi]:
            correct += 1
    acc = float(correct) / len(test)
    return acc


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 3):
        print('Usage: python3 id3.py <train> <test> <model>')
        sys.exit(2)
    loadAndTrain(argv[0], argv[1], argv[2])

    acc = runTest()
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
