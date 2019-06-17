import numpy as np
from model.templates import Activation


class ReLU(Activation):

    def __call__(self, x):
        return np.clip(x, 0, x.max())

    def backward(self, x):
        calc = np.zeros_like(x)
        calc[x > 0] = 1
        return calc, {}


class Softmax(Activation):

    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape((-1, 1))

    def backward(self, x):
        reshape = x.reshape(-1, 1)
        return np.diagflat(reshape) - np.dot(reshape, reshape.T), {}


class Tanh(Activation):

    def __call__(self, x):
        return np.tanh(x)

    def backward(self, x):
        return 1.0 - np.tanh(x)**2, {}
