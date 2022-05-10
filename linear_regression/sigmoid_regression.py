import function_set1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]
N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

x = values[:, 10]
x_train = x[0:N_TRAIN]
x_test = x[N_TRAIN:]

basis = 'sigmoid'
#x_ev = np.linspace(np.asscalar(min(x_train)), np.asscalar(max(x_train)), num=500)
x_ev = np.linspace(np.asscalar(min(min(x_train), min(x_test))),
                  np.asscalar(max(max(x_train), max(x_test))), num=500)
x_ev = np.asmatrix(x_ev)
x_ev = np.transpose(x_ev)

(w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, 0, 0, 0, 0)
(t_est1, te_err1) = a1.evaluate_regression(x_test, t_test, w, 0, basis, 0)
(t_est, te_err) = a1.evaluate_regression(x_ev, np.zeros((500, 1), dtype=int), w, 0, basis, 0)

plt.plot(x_ev, t_est, 'g--')
plt.plot(x_train, t_train, 'r.')
plt.plot(x_test, t_test, 'b.')
plt.title('Using Sigmoid basis function (with bias) for feature 11')
plt.legend(['Learned Polynomial', 'Training datapoints', 'Testing datapoints'])
plt.show()

plt.plot(x_ev, t_est, 'g--')
#plt.plot(x_train, t_train, 'r.')
#plt.plot(x_test, t_test, 'b.')
plt.title('Using Sigmoid basis function (with bias) for feature 11')
plt.legend(['Learned Polynomial'])
plt.show()

print('Training error = ', tr_err)
print('Testing error = ', te_err1)
