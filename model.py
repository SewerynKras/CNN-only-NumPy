import numpy as np


class InvalidModeException(Exception):
    pass


class Conv2D:

    def __init__(self, size, step, filters):
        self.size = size
        self.step = step
        self.filters = filters
        self.initialized = False

    def __call__(self, x):

        if not self.initialized:
            self._create_weights(x.shape)

        size_h, size_w = self.size
        step = self.step
        filters = self.filters
        batch_size, height, width, channels = x.shape

        new_height = int((height - size_h) / step + 1)
        new_width = int((width - size_w) / step + 1)

        new_array = np.zeros((batch_size, new_height, new_width, filters))

        h_idx = 0
        for h in range(0, new_height):
            w_idx = 0
            for w in range(0, new_width):
                for c in range(filters):
                    fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                    weights = self.weights[:, :, :, c]
                    bias = self.biases[:, :, :, c]
                    calc = fragment * weights
                    calc += bias
                    calc = np.sum(calc, axis=(1, 2, 3))
                    new_array[:, h, w, c] = calc
                w_idx += step
            h_idx += step

        return new_array

    def _create_weights(self, inp_shape):
        size_h, size_w = self.size
        weights = np.random.standard_normal((size_h, size_w, inp_shape[-1], self.filters))
        weights *= 0.01  # now weights will be close but not equal to 0
        biases = np.zeros((size_h, size_w, inp_shape[-1], self.filters))
        self.biases = biases.astype("float32")
        self.weights = weights.astype("float32")
        self.initialized = True


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

    def __init__(self, size, step, mode='max'):
        self.size = size
        self.step = step
        if mode.lower() not in ['max', 'mean']:
            raise InvalidModeException(f"'{self.mode}' is not a valid mode")
        self.mode = mode.lower()

    def __call__(self, x):

        size_h, size_w = self.size
        step = self.step
        mode = self.mode
        batch_size, height, width, channels = x.shape

        new_height = int((height - size_h) / step + 1)
        new_width = int((width - size_w) / step + 1)

        new_array = np.zeros((batch_size, new_height, new_width, channels))

        h_idx = 0
        for h in range(0, new_height):
            w_idx = 0
            for w in range(0, new_width):
                for c in range(channels):
                    fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, c]
                    if mode == "max":
                        new_array[:, h, w, c] = np.max(fragment,
                                                       axis=(1, 2))
                    if mode == "mean":
                        new_array[:, h, w, c] = np.mean(fragment,
                                                        axis=(1, 2))

                w_idx += step
            h_idx += step

        return new_array


class Flat:

    def __init__(self, units):
        self.units = units
        self.weights = None
        self.biases = np.zeros(units).astype("float32")
        self.initialized = False

    def __call__(self, x):
        if not self.initialized:
            self._create_weights(x.shape)
        return np.dot(x, self.weights) + self.biases

    def _create_weights(self, inp_shape):
        weights = np.random.standard_normal((inp_shape[-1], self.units))
        weights *= 0.01  # now weights will be close but not equal to 0
        self.weights = weights.astype("float32")
        self.initialized = True


class ReLU:

    def __call__(self, x):
        return np.clip(x, 0, x.max())


class Softmax:

    def __call__(self, x):
        return np.exp(x) / np.sum(np.exp(x))


class Tanh:

    def __call__(self, x):
        return np.tanh(x)


class Model:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
