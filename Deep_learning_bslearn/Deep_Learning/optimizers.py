from From_Scratch_sample_implementations.Deep_learning_bslearn.utils import make_diagonal, Normalize
import numpy as np
#https://ruder.io/optimizing-gradient-descent/index.html





class Stochastic_Gradient_Descent():
    def __init__(self, learning_rate = 0.001, momentum = 0):
        self.lr = learning_rate
        self.beta = momentum
        self.w_updated = None


    def update(self, w, grad_w):

        if self.w_updated is None:
            self.w_updated = np.zeros(np.shape(w))


        # use momentum if passed.
        self.w_updated = self.beta*self.w_updated + (1-self.beta)*grad_w

        # reduce w using gradient
        return w - self.w_updated*self.lr


class Nesterov_Accelerated_Gradient():
    def __init__(self, learning_rate = 0.001, momentum = 0.4):
        self.lr = learning_rate
        self.beta = momentum
        self.w_updated = np.array([])

    def update(self, w, grad_w_func):

        # the gradient in the updation term has a mometum*previous weight state. Thus its similar to the
        # already pointing gradient direction which is used as starting point.
        approx_next_grad = np.clip(grad_w_func(w - self.beta*self.w_updated),-1,1)

        if not self.w_updated.any():
            self.w_updated = np.zeros(np.shape(w))

        self.w_updated = self.beta*self.w_updated + self.lr*approx_next_grad

        return w - self.w_updated


class Adam():
    def __init__(self, learning_rate = 0.001, b1 = 0.9, b2 = 0.99):
        self.lr = learning_rate
        self.b1 = b1    #decay rate
        self.b2 = b2    #decay rate
        self.eps = 1e-08
        self.m = None
        self.v = None

    def update(self, w, grad_w):
        #If not initialized
        if self.m is None:
            m = np.zeros(np.shape(grad_w))
            v = np.zeros(np.shape(grad_w))

        self.m = self.b1*self.m + (1 - self.b1)*grad_w
        self.v = self.b2*self.v + (1 - self.b2)*np.power(grad_w,2)


        m_hat = self.m/(1-self.b1)
        v_hat = self.v/(1-self.b2)

        self.w_updated = self.lr*m_hat/(np.sqrt(v_hat) + self.eps)
        return w - self.w_updated



class RMSprop():
    def __init__(self, learning_rate = 0.01, beta = 0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.Eg = None #Moving average of the squared gradients of w
        self.eps = 1e-8


    def update(self, w, grad_w): # gradient with respect to w
        # moving averages of squared gradients
        if self.Eg is None:
            self.Eg = np.zeros(np.shape(grad_w))


        self.Eg = self.Eg*self.beta + (1-self.beta)*np.power(grad_w,2)

        # Divide the learning rate by the magnitude moving averages of the recent gradients

        return w - self.learning_rate*grad_w/np.sqrt(self.Eg+self.eps)




        











