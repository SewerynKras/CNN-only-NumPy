import numpy as np


def mean_squared_error(real, predicted):
    return np.mean((real - predicted) ** 2)


def mean_absolute_error(real, predicted):
    return np.mean(np.abs(real - predicted))


def cross_entropy(real, predicted, epsilon=1e-8):
    return - np.sum(real * np.log(predicted + epsilon)) / real.shape[0]
