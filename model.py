import numpy as np


class InvalidModeException(Exception):
    pass


class Conv2D:

    def __init__(self, size, strides):
        ...


class Padding:

    def __init__(self, size, mode='zero', **kwargs):
        if mode not in ['zero', 'average', 'value']:
            raise InvalidModeException(f"'{self.mode}' is not a valid mode")

        self.size = size
        self.mode = mode
        if mode == 'value':
            if 'value' not in kwargs.keys():
                raise ValueError("Mode set to 'value' but no value provided")
            self.value = kwargs['value']

    def __call__(self, x):
        if self.mode == 'zero':
            value = 0
        elif self.mode == 'average':
            value = np.mean(x)
        elif self.mode == 'value':
            value = self.value
        else:
            raise InvalidModeException(f"'{self.mode}' is not a valid mode")

        return np.pad(array=x,
                      pad_width=self.size,
                      mode='constant',
                      constant_values=value)


class Pooling:

    def __init__(self, size, mode='max'):
        self.size = size
        self.mode = mode

    def __call__(self, x):

        size = self.size
        mode = self.mode
        
        batch_size, height, width, channels = x.shape
        #FIXME:

        if self.mode.lower() == "max":
            
class Model:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, input):
        x = input
        for layer in self.layers:
            x = layer(x)
        return x
