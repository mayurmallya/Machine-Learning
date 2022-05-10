#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import function_set2 as a2

# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001

# Step size for gradient descent.
#eta = 0.1
etas = [0.5, 0.3, 0.1, 0.05, 0.01]

# Load data.
data = np.genfromtxt('data.txt')
np.random.seed(12)
np.random.shuffle(data)

# Data matrix, with column of ones at end.
X = data[:, 0:3]

# Target values, 0 for class 1, 1 for class 2.
t = data[:, 3]

# For plotting data
class1 = np.where(t == 0)
X1 = X[class1]
class2 = np.where(t == 1)
X2 = X[class2]

# Initialize w.
w = np.array([0.1, 0, 0])

# Error values over all iterations.
e_all = []

DATA_FIG = 1

# Set up the slope-intercept figure
#SI_FIG = 2
#plt.figure(SI_FIG, figsize=(8.5, 6))
#plt.rcParams.update({'font.size': 15})
#plt.title('Separator in slope-intercept space')
#plt.xlabel('slope')
#plt.ylabel('intercept')
#plt.axis([-5, 5, -10, 0])


for i in range(0, 5):
    e_all = []
    w = np.array([0.1, 0, 0])
    for epoch in range(0, max_iter):

        for j in range(0, 200):
            # Compute output using current w on all data X.
            y = sps.expit(np.dot(X, w))

            # Gradient of the error, using Eqn 4.91
            grad_e = np.multiply((y[j] - t[j]), X.T[:, j])

            # Update w, *subtracting* a step in the error derivative since we're minimizing
            w_old = w
            w = w - etas[i] * grad_e
        y = sps.expit(np.dot(X, w))
        # e is the error, negative log-likelihood (Eqn 4.90)
        e = -np.mean(np.multiply(t, np.log(y + 0.001)) + np.multiply((1 - t), np.log(1 - y + 0.001)))

        # Add this error to the end of error vector.
        e_all.append(e)
        print('epoch {0:d}, negative log-likelihood {1:.4f}, w={2}'.format(epoch, e, w.T))

        if epoch > 0:
            if np.absolute(e - e_all[epoch - 1]) < tol:
                break

        #DATA_FIG = 1
        #plt.figure(DATA_FIG)
        #plt.clf()
        #plt.plot(X1[:, 0], X1[:, 1], 'b.')
        #plt.plot(X2[:, 0], X2[:, 1], 'g.')
        #a2.draw_sep(w)
        #plt.axis([-5, 15, -10, 10])

    DATA_FIG = 1
    plt.figure(DATA_FIG, figsize=(8.5, 6))
    #plt.clf()
    plt.plot(X1[:, 0], X1[:, 1], 'b.')
    plt.plot(X2[:, 0], X2[:, 1], 'g.')
    plt.title('Learned Separator')
    a2.draw_sep(w)
    plt.axis([-5, 15, -10, 10])

    #eta = etas[i]
    #plt.title('Separator with eta = %d' % eta)
    #plt.show()
    TRAIN_FIG = 3
    plt.figure(TRAIN_FIG, figsize=(8.5, 6))
    plt.plot(e_all)
    plt.ylabel('Negative log likelihood')
    plt.title('Training logistic regression using GSD')
    plt.xlabel('Epoch')
    ##plt.show()

plt.legend(['eta = 0.5', 'eta = 0.3', 'eta = 0.1', 'eta = 0.05', 'eta = 0.01'])
plt.show()
plt.figure(DATA_FIG, figsize=(8.5, 6))
plt.draw()
end = 1

