#!/usr/bin/python
#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt
import numpy as np # to prevent big numbers destroying everything : )

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    #MY CODE HERE
    
    # Extract features and labels as NumPy arrays
    X = np.array([x for x, _ in data])
    y = np.array([1 if label == 1 else 0 for _, label in data])

    # Initialize weights and bias
    numvars = X.shape[1]
    w = np.zeros(numvars)
    b = 0.0

    for _ in range(MAX_ITERS):
        # Linear combination (z) and sigmoid prediction (p)
        z = np.dot(X, w) + b
        p = 1 / (1 + np.exp(-z))

        # Calculate gradients
        errors = p - y
        grad_w = np.dot(errors, X) + l2_reg_weight * w
        grad_b = np.sum(errors)

        # Update weights and bias
        w -= eta * grad_w
        b -= eta * grad_b

        # Convergence check
        grad_magnitude = sqrt(np.sum(grad_w ** 2) + grad_b ** 2)
        if grad_magnitude < 0.0001:
            print("Converged")
            break

    return w, b


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model
    # YOUR CODE HERE
    
    # Vectorized linear combination and sigmoid
    z = np.dot(w, x) + b
    return 1 / (1 + np.exp(-z))


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
