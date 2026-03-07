"""
Neural Layer Implementation
Handles weight initialization, forward pass, and gradient computation
"""

import numpy as np


class NeuralLayer:
    def __init__(self, in_dim, out_dim, activation=None, weight_init="random"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation

        # weight initialization
        if weight_init == "xavier":
            limit = np.sqrt(6 / (in_dim + out_dim))
            self.W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        else:  # random
            self.W = np.random.randn(in_dim, out_dim) * 0.01

        self.b = np.zeros((1, out_dim))

        # placeholders for backward
        self.grad_W = None
        self.grad_b = None

    def forward(self, X):
        """
        X shape: (b, in_dim)
        """
        self.X = X

        self.Z = X @ self.W + self.b

        if self.activation is None:
            self.A = self.Z
        else:
            self.A = self.activation.forward(self.Z)

        return self.A

    def backward(self, dA):
        """
        dA shape: (b, out_dim)
        returns gradient wrt input (dX)
        """

        if self.activation is not None:
            dZ = self.activation.backward(dA)
        else:
            dZ = dA

        b = self.X.shape[0]

        ############### Can normalize the gradients by batch size here if needed (divide both rhs by b) ###############

        self.grad_W = self.X.T @ dZ
        self.grad_b = np.sum(dZ, axis=0, keepdims=True)

        dX = dZ @ self.W.T

        return dX