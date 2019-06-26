import numpy as np
from model.utils import InvalidModeException
from model.templates import Layer, Variable


class Conv2D(Layer):

    def __init__(self, size, step, filters):
        """
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

        h_idx = 0
        for h in range(new_height):
            w_idx = 0
            for w in range(new_width):
                inp_frag = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                calc = np.expand_dims(inp_frag, axis=-1) * weights
                calc = np.sum(calc, axis=(1, 2, 3))
                calc += biases
                new_array[:, h, w, :] = calc
                w_idx += step
            h_idx += step

        self.memorized_input = x.copy()
        return new_array

    def _create_weights(self, inp_shape):
        size_h, size_w = self.size
        weights = np.random.standard_normal((size_h, size_w, inp_shape[-1], self.filters))
        weights *= 0.01  # now weights will be close but not equal to 0
        biases = np.zeros((self.filters,))
        self.biases = Variable(biases)
        self.weights = Variable(weights)
        self.variables = [self.weights, self.biases]
        self.initialized = True

    def backward(self, grad):

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

        h_idx = 0
        for h in range(new_height):
            w_idx = 0
            for w in range(new_width):
                grad_fragment = input_grad[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]
                input_fragment = x[:, h_idx:h_idx+size_h, w_idx:w_idx+size_w, :]

                weights_reshaped = np.expand_dims(weights, axis=0)
                input_reshaped = np.expand_dims(input_fragment, axis=-1)
                grad_reshaped = grad[:, h, w, :].reshape((x.shape[0], 1, 1, 1, -1))

                grad_fragment += np.sum(weights_reshaped * grad_reshaped, axis=-1)
                weights_grad += np.sum(input_reshaped * grad_reshaped, axis=0)
                bias_grad[:] += np.sum(grad[:, h, w, :])

                w_idx += step
            h_idx += step
        return input_grad, {self.weights: weights_grad,
                            self.biases: bias_grad}


class Padding(Layer):

    def __init__(self, size, mode='zero', **kwargs):
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
        s = self.size
        return grad[:, s:-s, s:-s, :], {}


class Pooling(Layer):

    def __init__(self, size, step, mode='max'):
        """
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

        self.memorized_input = x.copy()
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
                        grad_fragment += max_mask * grad[:, h, w, c].reshape((-1, 1, 1))
                    elif self.mode == "mean":
                        mean_val = grad[:, h, w, c] / (size_h * size_w)
                        grad_fragment += np.full((batch_size, size_h, size_w), mean_val)
                w_idx += step
            h_idx += step
        return gradient, {}


class FullyConnected(Layer):

    def __init__(self, units):
        """
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

        self.memorized_input = x.copy()
        return np.dot(x, weights) + biases

    def _create_weights(self, inp_shape):
        weights = np.random.standard_normal((inp_shape[-1], self.units))
        weights *= 0.01  # now weights will be close but not equal to 0
        biases = np.zeros((self.units,))
        self.weights = Variable(weights)
        self.biases = Variable(biases)
        self.variables = [self.weights, self.biases]
        self.initialized = True

    def backward(self, grad):

        weights = self.weights.value

        input_grad = np.dot(grad, weights.T)
        weights_grad = np.dot(grad.T, self.memorized_input).T
        bias_grad = np.sum(grad, axis=0)

        return input_grad, {self.weights: weights_grad,
                            self.biases: bias_grad}


class Flatten(Layer):

    def __init__(self):
        self.memorized_input = None
        self.variables = []

    def __call__(self, x):
        self.memorized_input = x.copy()
        return x.reshape((x.shape[0], -1))

    def backward(self, grad):
        shape = self.memorized_input.shape
        return grad.reshape(shape), {}


class BatchNorm(Layer):

    def __init__(self, epsilon=1e-8):
        self.epsilon=epsilon
        self.initialized = False

    def __call__(self, x):

        if not self.initialized:
            self._create_weights(x.shape)

        batch_size, height, width, channels = x.shape
        epsilon = self.epsilon
        gamma = self.gamma.value
        beta = self.beta.value

        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        norm = (x - mean) / np.sqrt(var + epsilon)
        calc = gamma * norm + beta

        self.memorized_input = x.copy()
        return calc

    def _create_weights(self, inp_shape):
        gamma = np.ones((1,))
        beta = np.zeros((1,))
        self.gamma = Variable(gamma)
        self.beta = Variable(beta)
        self.variables = [self.gamma, self.beta]
        self.initialized = True

    def backward(self, grad):
        x = self.memorized_input
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        gamma = self.gamma.value

        epsilon = self.epsilon

        x_min_mean = x - mean
        std = np.sqrt(var + epsilon)

        inp_grad = grad * gamma
        var_grad = np.sum(x_min_mean * inp_grad, axis=0)
        var_grad *= (-0.5 * ((1.0 / std) ** 3))
        mean_grad = np.sum(inp_grad * (1.0 / std), axis=0)
        mean_grad += var_grad * np.mean(-2.0 * x_min_mean, axis=0)

        inp_grad *= (1.0 / std)
        inp_grad += (var_grad * 2 * (x_min_mean / x.shape[0])) + (mean_grad / x.shape[0])

        gamma_grad = np.sum(grad * (x_min_mean / std))
        beta_grad = np.sum(grad)

        return inp_grad, {self.gamma: gamma_grad,
                          self.beta: beta_grad}

if __name__ == "__main__":
    lay = Conv2D((2, 2), 1, 16)
    inp = np.random.random((12, 28, 28, 3))
    lay.backward(lay(inp))
