import numpy as np
from model.utils import InvalidModeException


class Conv2D:

    def __init__(self, size, step, filters):
        self.size = size
        self.step = step
        self.filters = filters
        self.initialized = False
        self.memorized_input = None

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
        for h in range(new_height):
            w_idx = 0
            for w in range(new_width):
                for c in range(filters):
                    fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                    weights = self.weights[:, :, :, c]
                    bias = self.biases[c]
                    calc = fragment * weights
                    calc = np.sum(calc, axis=(1, 2, 3))
                    calc += bias.flatten()
                    new_array[:, h, w, c] = calc
                w_idx += step
            h_idx += step

        self.memorized_input = x
        return new_array

    def _create_weights(self, inp_shape):
        size_h, size_w = self.size
        weights = np.random.standard_normal((size_h, size_w, inp_shape[-1], self.filters))
        weights *= 0.01  # now weights will be close but not equal to 0
        biases = np.zeros((self.filters,))
        self.biases = biases.astype("float32")
        self.weights = weights.astype("float32")
        self.initialized = True

    def backward(self, grad):

        size_h, size_w = self.size
        step = self.step
        batch_size, height, width, channels = self.memorized_input.shape
        batch_size, new_height, new_width, new_channels = grad.shape
        x = self.memorized_input

        input_grad = np.zeros_like(self.memorized_input)
        weights_grad = np.zeros_like(self.weights)
        bias_grad = np.zeros_like(self.biases)

        h_idx = 0
        for h in range(new_height):
            w_idx = 0
            for w in range(new_width):
                for c in range(new_channels):
                    grad_fragment = input_grad[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                    input_fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]

                    grad_fragment += self.weights[:, :, :, c] * grad[:, h, w, c]
                    weights_grad[:, :, :, c] += np.sum(input_fragment * grad[:, h, w, c])
                    bias_grad[c] += np.sum(grad[:, h, w, c])

                w_idx += step
            h_idx += step
        return input_grad, weights_grad, bias_grad


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

    def backward(self, grad):
        size_h, size_w = self.size
        return grad[:, size_h:-size_h, size_w:-size_w, :]


class Pooling:

    def __init__(self, size, step, mode='max'):
        self.size = size
        self.step = step
        if mode.lower() not in ['max', 'mean']:
            raise InvalidModeException(f"'{mode}' is not a valid mode")
        self.mode = mode.lower()
        self.memorized_input = None

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

        self.memorized_input = x
        return new_array

    def backward(self, grad):

        size_h, size_w = self.size
        step = self.step
        batch_size, height, width, channels = self.memorized_input.shape
        batch_size, new_height, new_width, new_channels = grad.shape
        x = self.memorized_input

        gradient = np.zeros_like(self.memorized_input)

        h_idx = 0
        for h in range(new_height):
            w_idx = 0
            for w in range(new_width):
                for c in range(new_channels):
                    fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, c]
                    grad_fragment = gradient[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, c]
                    if self.mode == "max":
                        max_mask = np.max(fragment, axis=(1, 2)).reshape((-1, 1, 1))
                        max_mask = np.array(fragment == max_mask)
                        grad_fragment += max_mask * grad[:, h, w, c]
                    elif self.mode == "mean":
                        mean_val = grad[:, h, w, c] / (size_h * size_w)
                        grad_fragment += np.full((batch_size, size_h, size_w), mean_val)
                w_idx += step
            h_idx += step
        return gradient


class Flat:

    def __init__(self, units):
        self.units = units
        self.weights = None
        self.biases = np.zeros(units).astype("float32")
        self.initialized = False
        self.memorized_input = None

    def __call__(self, x):
        if not self.initialized:
            self._create_weights(x.shape)
        self.memorized_input = x
        return np.dot(x, self.weights) + self.biases

    def _create_weights(self, inp_shape):
        weights = np.random.standard_normal((inp_shape[-1], self.units))
        weights *= 0.01  # now weights will be close but not equal to 0
        self.weights = weights.astype("float32")
        self.initialized = True
