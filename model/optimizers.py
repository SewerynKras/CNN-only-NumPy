import numpy as np
from model.templates import Optimizer


class Adam(Optimizer):

    def __init__(self,
                 learning_rate=0.01,
                 beta_1=0.9,
                 beta_2=0.99,
                 epsilon=1e-8):
        """
        Implementation of Adam (https://arxiv.org/abs/1412.6980)
        Default values are equal to those proposed in the paper

        Keyword Arguments:
            learning_rate {float} -- (default: {0.01})
            beta_1 {float} --  (default: {0.9})
            beta_2 {float} --  (default: {0.99})
            epsilon {float} -- (default: {1e-8})
        """
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.timestep = 1
        self.variables = {}

    def __exit__(self, *args):
        self.timestep += 1

    def register_variables(self, model):
        """
        Creates internal variables for the given model,
        This function should be called before any training begins

        Arguments:
            model_layers {list}
        """
        for layer in model.layers:
            for variable in layer.variables:
                self.variables[variable] = {"m": 0.0, "v": 0.0}

    def apply_grad(self, grad, variable):
        """
        Calculates and applies the given gradient to the layer

        Arguments:
            grad {np.array}
            layer {Layer}

        Raises:
            ValueError: if the given layer was not registered before
        """

        if variable not in self.variables.keys():
            raise ValueError(f"Variable '{variable}' not registered")

        m = self.variables[variable]['m']
        v = self.variables[variable]['v']
        timestep = self.timestep
        lr = self.learning_rate
        beta_1 = self.beta_1
        beta_2 = self.beta_2
        epsilon = self.epsilon

        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * (grad ** 2)

        # Bias-correction
        m_corr = m / (1 - np.power(beta_1, timestep))
        v_corr = v / (1 - np.power(beta_2, timestep))

        variable.value -= lr * m_corr / (np.sqrt(v_corr) + epsilon)


class SGD(Optimizer):

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def apply_grad(self, grad, variable):
        variable.value -= self.learning_rate * grad
