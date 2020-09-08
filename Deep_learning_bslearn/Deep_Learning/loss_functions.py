from __future__ import division
import numpy as np
from From_Scratch_sample_implementations.Deep_learning_bslearn.utils import accuracy_score


class Loss(object):
    def loss(self, y_true, y_pred):
        raise NotImplementedError

    def gradient(self, y, y_pred):
        raise NotImplementedError

    def acc(self, y , y_pred):
        return 0


class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y_true, y_pred):
        return 0.5*np.power((y_true-y_pred),2)

    def gradient(self, y, y_pred):
        return -(y-y_pred)



class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y_true , y_pred):
        # Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15,1-1e-15)
        return -y_true*np.log(y_pred) - (1-y_true)*np.log(1-y_pred)


    def gradient(self, y_true, y_pred):
        #Avoid division by zero
        y_pred = np.clip(y_pred, 1e-15,1-1e-15)
        return -(1/y_true) + (1-y_true)/(1-y_pred)


    def acc(self, y , y_pred):
        return accuracy_score(np.argmax(y, axis=1),np.argmax(y_pred, axis=1))

