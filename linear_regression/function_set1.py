import numpy as np
import pandas as pd
from scipy import nanmean

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_', encoding='latin1')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    #print(features)
    # Separate country names from feature values.
    countries = data.values[:,0]
    #print(countries)
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')
    #print(values)

    # Modify NaN values (missing values).
    mean_vals = nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, reg_lambda=0, degree=0, mu=0, s=1, bias=0):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      mu,s are parameters of Gaussian basis

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # Construct the design matrix.
    # Pass the required parameters to this function
    phi = design_matrix(x, degree, basis, bias)
    #print(phi)
    # Learning Coefficients
    if reg_lambda > 0:
        # regularized regression
        t1 = reg_lambda * np.identity(67)
        t2 = np.matmul(np.transpose(phi), phi)
        t3 = t1 + t2
        t4 = np.linalg.inv(t3)
        t5 = np.matmul(np.transpose(phi), t)
        w = np.matmul(t4, t5)
    else:
        # no regularization
        temp = np.linalg.pinv(phi)
        #print(temp)
        w = np.matmul(temp, t)

    # Measure root mean squared error on training data.
    t_rain = np.matmul(phi, w)
    tr_err = t - t_rain
    train_err = np.sqrt(np.sum(np.power(tr_err, 2))/len(tr_err))
    #print('RMS train error for degree', degree, 'polynomial is ', train_err)

    return (w, train_err)



def design_matrix(x, degree, basis=None, bias=0):
    """ Compute a design matrix Phi from given input datapoints and basis.

    Args:
        ?????

    Returns:
      phi design matrix
    """
    phi = []
    x_temp = x
    if basis == 'polynomial':
        for i in range(2, degree + 1):
            x_temp = np.concatenate((x_temp, np.power(x, i)), axis=1)
            #print(np.shape(x_temp))
        phi = x_temp
        if bias == 0:
            bias1 = np.ones((len(x), 1))
            phi = np.concatenate((bias1, x_temp), axis=1)
    elif basis == 'sigmoid':
        phi1 = 1 / (1 + np.exp((100 - x_temp)/2000))
        phi2 = 1 / (1 + np.exp((10000 - x_temp)/2000))
        x_new = np.concatenate((phi1, phi2), axis=1)
        if bias == 0:
            bias1 = np.ones((len(x), 1))
            phi = np.concatenate((bias1, x_new), axis=1)
    else: 
        assert(False), 'Unknown basis %s' % basis

    return phi


def evaluate_regression(x_test, t_test, w, degree, basis, bias):
    """Evaluate linear regression on a dataset.

    Args:
      ?????

    Returns:
      t_est values of regression on inputs
      err RMS error on training set if t is not None
      """
    phi = design_matrix(x_test, degree, basis, bias)
    t_est = np.matmul(phi, w)
    te_err1 = t_test - t_est
    te_err = np.sqrt(np.sum(np.power(te_err1, 2))/len(te_err1))
    #print('RMS test error for degree', degree, 'polynomial is', te_err)

    return (t_est, te_err)
