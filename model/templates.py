"""
Templates implemented in this file ensure compatibility
between all components
"""
import numpy as np


class Layer:

    def __init__(self, *args, **kwargs):
        self.initialized = True
        self.memorized_input = None
        self.variables = []

    def __call__(*args, **kwargs):
        raise NotImplementedError()

    def backward(*args, **kwargs):
        raise NotImplementedError()

    def _create_weights(*args, **kwargs):
        raise NotImplementedError()

    def calculate_size(self, *args, **kwargs):
        size = 0
        for var in self.variables:
            size += np.product(var.value.shape)
        return size


class Activation(Layer):
    pass


class Optimizer:

    def __call__(*args, **kwargs):
        raise NotImplementedError()

    def __enter__(*args, **kwargs):
        pass

    def register_variables(*args, **kwargs):
        pass

    def __exit__(*args, **kwargs):
        pass


class Variable:

    def __init__(self, value):
        self.value = value.astype("float32")
