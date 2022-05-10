#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

highest_child_mortality_rate_1990 = np.amax(values[:,0])
print('Highest child mortality rate in 1990 is ',highest_child_mortality_rate_1990)
highest_child_mortality_country_1990 = np.argmax(values[:,0])
#print(highest_child_mortality_country_1990)
print('Country with highest child mortality rate in 1990 is ', countries[highest_child_mortality_country_1990])


highest_child_mortality_rate_2011 = np.amax(values[:,1])
print('Highest child mortality rate in 2011 is ',highest_child_mortality_rate_2011)
highest_child_mortality_country_2011 = np.argmax(values[:,1])
#print(highest_child_mortality_country_2011)
print('Country with highest child mortality rate in 2011 is ', countries[highest_child_mortality_country_2011])


targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]




# Pass the required parameters to these functions
RMS_train_errors = [0, 0, 0, 0, 0, 0]
RMS_test_errors = [0, 0, 0, 0, 0, 0]
#degree = 6
basis = 'polynomial'
for degree in range(1, 7):
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, degree, basis, 0)
    RMS_train_errors[degree-1] = tr_err
    RMS_test_errors[degree-1] = te_err
degree = [1, 2, 3, 4, 5, 6]
print(RMS_train_errors)
print(RMS_test_errors)






# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.plot(degree, RMS_train_errors)
plt.plot(degree, RMS_test_errors)
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Training error', 'Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()

x = a1.normalize_data(x)
N_TRAIN = 100
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]



# Pass the required parameters to these functions
RMS_train_errors = [0, 0, 0, 0, 0, 0]
RMS_test_errors = [0, 0, 0, 0, 0, 0]
#degree = 6
basis = 'polynomial'
for degree in range(1, 7):
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, degree, basis, 0)
    RMS_train_errors[degree-1] = tr_err
    RMS_test_errors[degree-1] = te_err
degree = [1, 2, 3, 4, 5, 6]
print(RMS_train_errors)
print(RMS_test_errors)






# Produce a plot of results.
plt.rcParams.update({'font.size': 15})
plt.plot(degree, RMS_train_errors)
plt.plot(degree, RMS_test_errors)
#plt.plot(test_err.keys(), test_err.values())
plt.ylabel('RMS')
plt.legend(['Training error','Testing error'])
plt.title('Fit with polynomials, no regularization')
plt.xlabel('Polynomial degree')
plt.show()
