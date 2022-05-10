import function_set1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]
N_TRAIN = 100
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

RMS_training_error2 = [0, 0, 0, 0, 0, 0, 0, 0]
RMS_testing_error2 = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range(7, 15):
    x = values[:, i]
    x_train = x[0:N_TRAIN]
    x_test = x[N_TRAIN:]

    basis = 'polynomial'
    degree = 3

    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, degree, basis, 0)
    RMS_training_error2[i-7] = tr_err
    RMS_testing_error2[i-7] = te_err

print(RMS_training_error2)
print(RMS_testing_error2)

#code borrowed from https://pythonspot.com/matplotlib-bar-chart/
fig, ax = plt.subplots()
index = np.arange(8)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, RMS_training_error2, bar_width,
alpha=opacity,
color='b',
label='Training error')

rects2 = plt.bar(index + bar_width, RMS_testing_error2, bar_width,
alpha=opacity,
color='g',
label='Testing error')

plt.xlabel('Features')
plt.ylabel('Error')
plt.title('Errors in unregularized regression with 3 degree polynomials with bias')
plt.xticks(index + bar_width/2, ('8', '9', '10', '11', '12', '13', '14', '15'))
plt.legend()

plt.tight_layout()
plt.show()
#borrowed code ends here, thanks.

RMS_training_error2 = [0, 0, 0, 0, 0, 0, 0, 0]
RMS_testing_error2 = [0, 0, 0, 0, 0, 0, 0, 0]

for i in range(7, 15):
    x = values[:, i]
    x_train = x[0:N_TRAIN]
    x_test = x[N_TRAIN:]

    basis = 'polynomial'
    degree = 3

    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, 0, degree, 0, 0, 1)
    (t_est, te_err) = a1.evaluate_regression(x_test, t_test, w, degree, basis, 1)
    RMS_training_error2[i-7] = tr_err
    RMS_testing_error2[i-7] = te_err

print(RMS_training_error2)
print(RMS_testing_error2)

#code borrowed from https://pythonspot.com/matplotlib-bar-chart/
fig, ax = plt.subplots()
index = np.arange(8)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, RMS_training_error2, bar_width,
alpha=opacity,
color='b',
label='Training error')

rects2 = plt.bar(index + bar_width, RMS_testing_error2, bar_width,
alpha=opacity,
color='g',
label='Testing error')

plt.xlabel('Features')
plt.ylabel('Error')
plt.title('Errors in unregularized regression with 3 degree polynomials without bias')
plt.xticks(index + bar_width/2, ('8', '9', '10', '11', '12', '13', '14', '15'))
plt.legend()

plt.tight_layout()
plt.show()
#borrowed code ends here, thanks.

