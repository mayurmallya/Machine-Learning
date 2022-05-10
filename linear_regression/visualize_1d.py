#!/usr/bin/env python

import function_set1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]
#x = values[:, 7:]
#x = a1.normalize_data(x)


for f in range(10, 13):
    N_TRAIN = 100
    x_train = values[0:N_TRAIN, f]
    x_test = values[N_TRAIN:, f]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    #x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    x_ev = np.linspace(np.asscalar(min(min(x_train), min(x_test))),
                   np.asscalar(max(max(x_train), max(x_test))), num=500)
    #x1_ev = np.linspace(0, 10, num=500)
    #x2_ev = np.linspace(0, 10, num=50)

    # TO DO::
    # Perform regression on the linspace samples.
    # Put your regression estimate here in place of y_ev.
    basis = 'polynomial'
    degree = 3

    x_ev = np.asmatrix(x_ev)
    x_ev = np.transpose(x_ev)
    #print(x_ev.shape)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_ev, np.zeros((500, 1), dtype=int), w, degree, basis, 0)

    #y1_ev = np.random.random_sample(x1_ev.shape)
    #y2_ev = np.random.random_sample(x2_ev.shape)
    #y1_ev = 100*np.sin(x1_ev)
    #y2_ev = 100*np.sin(x2_ev)
    q = f+1
    plt.plot(x_ev, t_est, 'g--')
    plt.plot(x_train, t_train, 'r.')
    plt.plot(x_test, t_test, 'b.')
    plt.title('Visualization of a function (without bias term) and data points for feature %d' %q)
    plt.legend(['Learned Polynomial', 'Training datapoints', 'Testing datapoints'])
    plt.show()

for f in range(10, 13):
    N_TRAIN = 100
    x_train = values[0:N_TRAIN, f]
    x_test = values[N_TRAIN:, f]
    t_train = targets[0:N_TRAIN]
    t_test = targets[N_TRAIN:]

    # Plot a curve showing learned function.
    # Use linspace to get a set of samples on which to evaluate
    #x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
    x_ev = np.linspace(np.asscalar(min(min(x_train), min(x_test))),
                   np.asscalar(max(max(x_train), max(x_test))), num=500)
    #x1_ev = np.linspace(0, 10, num=500)
    #x2_ev = np.linspace(0, 10, num=50)

    # TO DO::
    # Perform regression on the linspace samples.
    # Put your regression estimate here in place of y_ev.
    basis = 'polynomial'
    degree = 3

    x_ev = np.asmatrix(x_ev)
    x_ev = np.transpose(x_ev)
    #print(x_ev.shape)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, degree, 0, 0, 1)
    (t_est, te_err) = a1.evaluate_regression(x_ev, np.zeros((500, 1), dtype=int), w, degree, basis, 1)

    #y1_ev = np.random.random_sample(x1_ev.shape)
    #y2_ev = np.random.random_sample(x2_ev.shape)
    #y1_ev = 100*np.sin(x1_ev)
    #y2_ev = 100*np.sin(x2_ev)
    q = f+1
    plt.plot(x_ev, t_est, 'g--')
    plt.plot(x_train, t_train, 'r.')
    plt.plot(x_test, t_test, 'b.')
    plt.title('Visualization of a function (with bias term) and data points for feature %d' %q)
    plt.legend(['Learned Polynomial', 'Training datapoints', 'Testing datapoints'])
    plt.show()

N_TRAIN = 100
x_train = values[0:N_TRAIN, 10]
x_test = values[N_TRAIN:, 10]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)

basis = 'polynomial'
degree = 3

x_ev = np.asmatrix(x_ev)
x_ev = np.transpose(x_ev)
#print(x_ev.shape)
(w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, degree, 0, 0, 0)
(t_est, te_err) = a1.evaluate_regression(x_ev, np.zeros((500, 1), dtype=int), w, degree, basis, 0)

plt.plot(x_ev, t_est, 'g--')
plt.plot(x_train, t_train, 'r.')
plt.plot(x_test, t_test, 'b.')
plt.title('Visualization of a function (without bias term) and data points for feature 11')
plt.legend(['Learned Polynomial', 'Training datapoints', 'Testing datapoints'])
plt.show()

N_TRAIN = 100
x_train = values[0:N_TRAIN, 10]
x_test = values[N_TRAIN:, 10]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)

basis = 'polynomial'
degree = 3

x_ev = np.asmatrix(x_ev)
x_ev = np.transpose(x_ev)
#print(x_ev.shape)
(w, tr_err2) = a1.linear_regression(x_train, t_train, basis, 0, degree, 0, 0, 1)
(t_est, te_err2) = a1.evaluate_regression(x_ev, np.zeros((500, 1), dtype=int), w, degree, basis, 1)

plt.plot(x_ev, t_est, 'g--.')
plt.plot(x_train, t_train, 'r.')
plt.plot(x_test, t_test, 'b.')
plt.title('Visualization of a function (with bias term) and data points for feature 11')
plt.legend(['Learned Polynomial', 'Training datapoints', 'Testing datapoints'])
plt.show()
