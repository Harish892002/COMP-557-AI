import numpy as np
import nn


class SGD:

    def __init__(self, parameters, learning_rate=0.01):
        """
        Stochastic Gradient Descent (SGD) optimizer.

        Args:
            parameters (list): List of Parameter nodes to optimize.
            learning_rate (float): Step size.
        """
        self.parameters = parameters
        self.lr = learning_rate

    def step(self, loss):
        """
        Perform one optimization step.

        Args:
            loss: The loss node (SquareLoss or SoftmaxLoss).
        """
        grads = nn.gradients(loss, self.parameters)  # List of Constant nodes

        for param, grad in zip(self.parameters, grads):
            param.update(grad, -self.lr)


class Adam:

    def __init__(self,
                 parameters,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8):
        """
        Adam optimizer.

        Args:
            parameters (list): List of Parameter nodes to optimize.
            learning_rate (float): Step size.
            beta1 (float): Exponential decay rate for the first moment estimates.
            beta2 (float): Exponential decay rate for the second moment estimates.
            epsilon (float): Small constant for numerical stability.
        """
        self.parameters = parameters
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Initialize moment estimates
        self.m = [np.zeros_like(param.data) for param in parameters]
        self.v = [np.zeros_like(param.data) for param in parameters]
        self.t = 0  # Time step

    def step(self, loss):
        """
        Perform one optimization step.

        Args:
            loss: The loss node (SquareLoss or SoftmaxLoss).
        """
        grads = nn.gradients(loss, self.parameters)  # List of Constant nodes
        self.t += 1

        for i, (param, grad) in enumerate(zip(self.parameters, grads)):
            g = grad.data  # Extract numpy array from Constant

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g**2)

            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Update parameters
            param.update(nn.Constant(m_hat / (np.sqrt(v_hat) + self.epsilon)),
                         -self.lr)
