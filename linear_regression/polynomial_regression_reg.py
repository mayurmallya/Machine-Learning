import function_set1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:, 1]
x = values[:, 7:]
x = a1.normalize_data(x)

N_TOTAL = 100
x_total = x[0:N_TOTAL, :]
t_total = targets[0:N_TOTAL]

basis = 'polynomial'
degree = 2

lmbda = [0, 0.01, 0.1, 1, 10, 100, 1000, 10000]
avg_val_error = [0, 0, 0, 0, 0, 0, 0, 0]
for i in range(0, 8):

    x_val = x_total[0:10, :]
    x_train = x_total[10:100, :]
    t_val = t_total[0:10, :]
    t_train = t_total[10:100, :]
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = te_err

    x_val = x_total[10:20, :]
    x_train = np.concatenate((x_total[0:10, :], x_total[20:100, :]), axis=0)
    t_val = t_total[10:20, :]
    t_train = np.concatenate((t_total[0:10, :], t_total[20:100, :]), axis=0)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    x_val = x_total[20:30, :]
    x_train = np.concatenate((x_total[0:20, :], x_total[30:100, :]), axis=0)
    t_val = t_total[20:30, :]
    t_train = np.concatenate((t_total[0:20, :], t_total[30:100, :]), axis=0)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    x_val = x_total[30:40, :]
    x_train = np.concatenate((x_total[0:30, :], x_total[40:100, :]), axis=0)
    t_val = t_total[30:40, :]
    t_train = np.concatenate((t_total[0:30, :], t_total[40:100, :]), axis=0)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    x_val = x_total[40:50, :]
    x_train = np.concatenate((x_total[0:40, :], x_total[50:100, :]), axis=0)
    t_val = t_total[40:50, :]
    t_train = np.concatenate((t_total[0:40, :], t_total[50:100, :]), axis=0)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    x_val = x_total[50:60, :]
    x_train = np.concatenate((x_total[0:50, :], x_total[60:100, :]), axis=0)
    t_val = t_total[50:60, :]
    t_train = np.concatenate((t_total[0:50, :], t_total[60:100, :]), axis=0)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    x_val = x_total[60:70, :]
    x_train = np.concatenate((x_total[0:60, :], x_total[70:100, :]), axis=0)
    t_val = t_total[60:70, :]
    t_train = np.concatenate((t_total[0:60, :], t_total[70:100, :]), axis=0)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    x_val = x_total[70:80, :]
    x_train = np.concatenate((x_total[0:70, :], x_total[80:100, :]), axis=0)
    t_val = t_total[70:80, :]
    t_train = np.concatenate((t_total[0:70, :], t_total[80:100, :]), axis=0)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    x_val = x_total[80:90, :]
    x_train = np.concatenate((x_total[0:80, :], x_total[90:100, :]), axis=0)
    t_val = t_total[80:90, :]
    t_train = np.concatenate((t_total[0:80, :], t_total[90:100, :]), axis=0)
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    x_val = x_total[90:100, :]
    x_train = x_total[0:90, :]
    t_val = t_total[90:100, :]
    t_train = t_total[0:90, :]
    (w, tr_err) = a1.linear_regression(x_train, t_train, basis, lmbda[i], degree, 0, 0, 0)
    (t_est, te_err) = a1.evaluate_regression(x_val, t_val, w, degree, basis, 0)
    total_val_error = total_val_error + te_err

    avg_val_error[i] = total_val_error/10
    print('Average validation error for lambda =', lmbda[i], 'is : ', avg_val_error[i])

plt.semilogx(lmbda, avg_val_error)
plt.hlines(avg_val_error[0], 0, 10000, colors='r', label='For lambda = 0')
plt.xlabel('Lambda')
plt.ylabel('Average validation error')
plt.title('Average validation error v/s Lambda')
plt.legend(['Lambda[0]', 'Other lambda'])
plt.show()

print('Lambda = 1000, I choose you!')
