from model.templates import Activation


class Model:

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, real, predicted, optimizer):

        input_grad = predicted - real

        with optimizer:

            # Don't calculate the initial derivative if the
            # last layer is an activation function
            if isinstance(self.layers[-1], Activation):
                layers = self.layers[:-1]
            else:
                layers = self.layers

            for layer in reversed(layers):
                input_grad, var_grads = layer.backward(input_grad)
                for var, grad in var_grads.items():
                    optimizer.apply_grad(grad=grad, variable=var)

    def calculate_size(self):
        """
        Returns the total number of trainable variables
        """
        size = 0
        for lay in self.layers:
            size += lay.calculate_size()
        return size
