import numpy as np
from model.templates import Activation


class ReLU(Activation):

    def __call__(self, x):
        self.memorized_input = x.copy()
        return np.clip(x, 0, x.max())

    def backward(self, grad):
        x = self.memorized_input
        calc = np.array(grad, copy=True)
        calc[x <= 0] = 0
        return calc, {}


class Softmax(Activation):

    def __call__(self, x):
        self.memorized_input = x.copy()
        return np.exp(x) / (np.sum(np.exp(x), axis=1).reshape((-1, 1)))

    def backward(self, grad):
        x = self.memorized_input
        reshape = x.reshape(-1, 1)
        return np.diagflat(reshape) - np.dot(reshape, reshape.T), {}


class Tanh(Activation):

    def __call__(self, x):
        self.memorized_input = x.copy()
        return np.tanh(x)

    def backward(self, grad):
        x = self.memorized_input
        return (1.0 - (np.tanh(x) ** 2)) * grad, {}
