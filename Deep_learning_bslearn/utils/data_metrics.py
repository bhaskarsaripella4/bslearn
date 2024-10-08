from __future__ import division
import numpy as np
import sys
import math




def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true==y_pred, axis=1)/len(y_true)
    return accuracy

def calculate_variance(X):
    "variance is the squared diff of all values with its mean"

    mean = np.ones(np.shape(X))*X.mean(0)

    variance = (1/np.shape(X)[0])*np.diag((X-mean).T.dot(X-mean))

    return variance


def calculate_std_dev(X):
    "calculate the standard deviation"
    std_dev = np.sqrt(calculate_variance(X))

    return std_dev


def calculate_covariance_matrix(X, Y = None):
    if Y is None:
        Y = X

    n_samples = np.shape(X)[0]
    covariance_matrix = (1/(n_samples-1))*(X-X.mean(axis = 0)).T.dot(Y-Y.mean(axis = 0))

    return np.array(covariance_matrix, dtype=float)


def calculate_correlation_matrix(X, Y = None):
    if Y is None:
        Y = X

    n_samples = np.shape(X)[0]
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)

    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype= float)




