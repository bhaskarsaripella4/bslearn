from __future__ import division
from itertools import combinations_with_replacement
import numpy as np
import math
import sys


def make_diagonal(X):
    """ converts a vector into a diagonal matrix"""
    n = len(X)
    M = np.zeros((n,n))

    for i in range(0,n):
        M[i,i] = X[i]

    return M

def Normalize(X, axis=-1, order = 2):
    """ Normalize X normA = A - min(A(:))
        normA = normA ./ max(normA(:))
        count axis from the last to the first axis axis =-1."""

    l2_normX = np.atleast_1d(np.linalg.norm(X,axis = axis, order = order))
    # if 0 is norm, make them 1 to avoid infinites.
    l2_normX[l2_normX == 0] = 1
    return X/np.expand_dims(l2_normX,axis = axis)


def Standardize(X):
    "standardize = X - mean/std"

    std_dev = X.std(axis = 0)
    mean = X.mean(axis = 0)
    standardized = np.empty(np.shape(X))
    for col in range(np.shape(X)[1]):
        if std_dev[col]:
            standardized[:,col] = (X[:,col]-mean[:,col])/std_dev[col]

    return standardized


