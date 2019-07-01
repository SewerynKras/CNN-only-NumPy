import numpy as np
from model.utils import InvalidModeException
from model.templates import Layer, Variable


class Conv2D(Layer):

    def __init__(self, size, step, filters):
        """
        Performs a 2D convolution on an array

        Arguments:
            size {float (int, int)}
            step {int}
            filters {int}
        """
        self.size = size
        self.step = step
        self.filters = filters
        self.initialized = False

    def __call__(self, x):

        if not self.initialized:
            self._create_weights(x.shape)

        weights = self.weights.value
        biases = self.biases.value
        size_h, size_w = self.size
        step = self.step
        filters = self.filters
        batch_size, height, width, channels = x.shape

        new_height = int((height - size_h) / step + 1)
        new_width = int((width - size_w) / step + 1)

        new_array = np.zeros((batch_size, new_height, new_width, filters))

        # h_idx and w_idx keep track of the fragment index
        # to select the correct fragment simply select [idx:idx+size]
        h_idx = 0
        for h in range(new_height):
            w_idx = 0
            for w in range(new_width):
                inp_frag = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                # this way the entire filters channel gets evaluated at once
                # so there's no need to iterate over it
                # (this greately speeds up the calculations)
                calc = np.expand_dims(inp_frag, axis=-1) * weights
                calc = np.sum(calc, axis=(1, 2, 3))
                calc += biases
                new_array[:, h, w, :] = calc

                w_idx += step
            h_idx += step

        self.memorized_input = x
        return new_array

    def _create_weights(self, inp_shape):
        """
        Creates the initial weights based on the given input shape

        Arguments:
            inp_shape {tuple}
        """

        size_h, size_w = self.size
        weights = np.random.standard_normal((size_h, size_w, inp_shape[-1], self.filters))
        weights *= 0.01  # now weights will be close but not equal to 0
        biases = np.zeros((self.filters,))

        self.biases = Variable(biases)
        self.weights = Variable(weights)
        self.variables = [self.weights, self.biases]
        self.initialized = True

    def backward(self, grad):
        """
        Performs the backpropagation calculations

        Arguments:
            grad {np.Array} -- output of the previous calculation

        Returns:
            (input_grad, {variable: variable_grad})
        """
        weights = self.weights.value
        biases = self.biases.value
        size_h, size_w = self.size
        step = self.step
        batch_size, height, width, channels = self.memorized_input.shape
        batch_size, new_height, new_width, new_channels = grad.shape
        x = self.memorized_input

        input_grad = np.zeros_like(self.memorized_input)
        weights_grad = np.zeros_like(weights)
        bias_grad = np.zeros_like(biases)

        # h_idx and w_idx keep track of the fragment index
        # to select the correct fragment simply select [idx:idx+size]
        h_idx = 0
        for h in range(new_height):
            w_idx = 0
            for w in range(new_width):
                grad_fragment = input_grad[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                input_fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]

                # expand/reshape these fragment to go over all channel/filters
                # dimensions (this speeds up the calculations)
                weights_expanded = np.expand_dims(weights, axis=0)
                input_expanded = np.expand_dims(input_fragment, axis=-1)
                grad_reshaped = grad[:, h, w, :].reshape((x.shape[0], 1, 1, 1, -1))

                grad_fragment += np.sum(weights_expanded * grad_reshaped, axis=-1)
                weights_grad += np.sum(input_expanded * grad_reshaped, axis=0)
                bias_grad[:] += np.sum(grad[:, h, w, :])

                w_idx += step
            h_idx += step

        return input_grad, {self.weights: weights_grad,
                            self.biases: bias_grad}


class Padding(Layer):

    def __init__(self, size, mode='zero', **kwargs):
        """
        Pads an array with the given value

        Arguments:
            size {tuple} -- (int, int) -- vertical and horizontal pad

        Keyword Arguments:
            mode {str} -- aviable modes: ['zero', 'value', 'mean']
            value {float} -- when the selected mode is "value" this argument
                should be provided
        """
        if mode not in ['zero', 'mean', 'value']:
            raise InvalidModeException(f"'{mode}' is not a valid mode")

        self.size = size
        self.mode = mode
        self.variables = []
        if mode == 'value':
            if 'value' not in kwargs.keys():
                raise ValueError("Mode set to 'value' but no value provided")
            self.value = kwargs['value']

    def __call__(self, x):
        if self.mode == 'zero':
            value = 0
        elif self.mode == 'mean':
            value = np.mean(x)
        elif self.mode == 'value':
            value = self.value

        s = self.size
        return np.pad(array=x,
                      pad_width=[(0, 0), (s, s), (s, s), (0, 0)],
                      mode='constant',
                      constant_values=value)

    def backward(self, grad):
        """
        Performs the backpropagation calculation

        Arguments:
            grad {np.Array}
        """
        s = self.size
        return grad[:, s:-s, s:-s, :], {}


class Pooling(Layer):

    def __init__(self, size, step, mode='max'):
        """
        Applies pooling to an array

        Arguments:
            size {tuple (int, int)}
            step {int}
            mode {str} -- aviable modes: ["max", "mean"] (default: {'max'})
        """
        self.size = size
        self.step = step
        if mode.lower() not in ['max', 'mean']:
            raise InvalidModeException(f"'{mode}' is not a valid mode")
        self.mode = mode.lower()
        self.variables = []

    def __call__(self, x):

        size_h, size_w = self.size
        step = self.step
        mode = self.mode
        batch_size, height, width, channels = x.shape

        new_height = int((height - size_h) / step + 1)
        new_width = int((width - size_w) / step + 1)

        new_array = np.zeros((batch_size, new_height, new_width, channels))

        # h_idx and w_idx keep track of the fragment index
        # to select the correct fragment simply select [idx:idx+size]
        h_idx = 0
        for h in range(0, new_height):
            w_idx = 0
            for w in range(0, new_width):
                fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                if mode == "max":
                    new_array[:, h, w, :] = np.max(fragment,
                                                   axis=(1, 2))
                if mode == "mean":
                    new_array[:, h, w, :] = np.mean(fragment,
                                                    axis=(1, 2))

                w_idx += step
            h_idx += step

        self.memorized_input = x
        return new_array

    def backward(self, grad):
        """
        Performs the backpropagation calculation

        Arguments:
            grad {np.Array}
        """
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
                fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                grad_fragment = gradient[:, h_idx:h_idx + size_h, w_idx:w_idx + size_w,:]

                if self.mode == "max":
                    # only add the max values of each batch and channel
                    # based on the created mask
                    max_mask = np.max(fragment, axis=(1, 2))
                    max_mask = np.array(fragment == max_mask)
                    grad_fragment += max_mask * grad[:, h, w, :].reshape((batch_size, 1, 1, -1))

                elif self.mode == "mean":
                    # add a mean of each batch and channel
                    size = size_h * size_w
                    mean_val = grad[:, h, w, :] / size
                    stacked = np.stack([mean_val.T] * size).T
                    stacked = stacked.reshape((batch_size, size_h, size_w, channels))
                    grad_fragment += stacked

                w_idx += step
            h_idx += step
        return gradient, {}


class FullyConnected(Layer):

    def __init__(self, units):
        """
        Multiplies an array by weights and adds bias

        Arguments:
            units {int}
        """
        self.units = units
        self.weights = None
        self.biases = None
        self.initialized = False

    def __call__(self, x):
        if not self.initialized:
            self._create_weights(x.shape)

        weights = self.weights.value
        biases = self.biases.value

        self.memorized_input = x
        return np.dot(x, weights) + biases

    def _create_weights(self, inp_shape):
        """
        Creates the initial weights based on the given input shape

        Arguments:
            inp_shape {tuple}
        """
        weights = np.random.standard_normal((inp_shape[-1], self.units))
        weights *= 0.01  # now weights will be close but not equal to 0
        biases = np.zeros((self.units,))
        self.weights = Variable(weights)
        self.biases = Variable(biases)
        self.variables = [self.weights, self.biases]
        self.initialized = True

    def backward(self, grad):
        """
        Performs the backpropagation calculation

        Arguments:
            grad {np.Array}
        """
        weights = self.weights.value

        input_grad = np.dot(grad, weights.T)
        weights_grad = np.dot(grad.T, self.memorized_input).T
        bias_grad = np.sum(grad, axis=0)

        return input_grad, {self.weights: weights_grad,
                            self.biases: bias_grad}


class Flatten(Layer):

    def __init__(self):
        """
        Flattens an array
        """
        self.memorized_input = None
        self.variables = []

    def __call__(self, x):
        self.memorized_input = x
        return x.reshape((x.shape[0], -1))

    def backward(self, grad):
        """
        Performs the backpropagation calculation

        Arguments:
            grad {np.Array}
        """
        shape = self.memorized_input.shape
        return grad.reshape(shape), {}
