#Activation functions reference from https://en.wikipedia.org/wiki/Activation_function

import numpy as np


class Sigmoid():
    def __call__(self, x):
        return 1/(1+np.exp(-x))

    def gradient(self,x):
        return self.__call__(x)*(1-self.__call__(x))



class Softmax():
    def __call__(self, x):
        e_x = np.exp(x-np.max(x, axis=-1, keepdims= True))
        return e_x/np.sum(e_x, axis=-1, keepdims=True)

    def gradient(self,x):
        return self.__call__(x)*(1-self.__call__(x))


class Tanh():
    def __call__(self, x):
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def gradient(self,x):
        return 1-np.power(self.__call__(x),2)


class ReLu():
    def __call__(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self,x):
        return np.where(x >= 0, 1, 0)

class ELU():
    def __init__(self, alpha = 0.1):
        self.alpha = alpha

    def __call__(self, x):
        return np.where(x > 0, self.alpha*(np.exp(x)-1), x)

    def gradient(self,x):
        return np.where(x > 0, 1, self.__call__(x) + self.alpha)


class SELU():
    def __init__(self, Lambda = 1.0507009873554804934193349852946, alpha = 1.6732632423543772848170429916717):
        self.Lambda = Lambda
        self.alpha = alpha

    def __call__(self,x ):
        return self.Lambda*np.where(x >= 0, self.alpha*(np.exp(x)-1), x)

    def gradient(self,x):
        return self.Lambda*np.where(x >= 0, 1, self.alpha*np.exp(x))


class LeakyReLu():
    def __call__(self, x):
        return np.where(x >= 0, x, 0.01*x)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0.01)


class Softplus():
    def __call__(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return 1/(1+np.exp(x))