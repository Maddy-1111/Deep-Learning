"""
Optimization Algorithms
Implements: SGD, Momentum, NAG, RMSProp
"""

import numpy as np


class SGD:
    def __init__(self, lr=0.01, weight_decay=0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, layers):
        for layer in layers:
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b


class Momentum:
    def __init__(self, layers, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay

        self.vW = [np.zeros_like(layer.W) for layer in layers]
        self.vb = [np.zeros_like(layer.b) for layer in layers]

    def step(self, layers):
        for i, layer in enumerate(layers):

            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.vW[i] = self.beta * self.vW[i] + (1 - self.beta) * grad_W
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * grad_b

            layer.W -= self.lr * self.vW[i]
            layer.b -= self.lr * self.vb[i]


class NAG:
    def __init__(self, layers, lr=0.01, beta=0.9, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay

        self.vW = [np.zeros_like(layer.W) for layer in layers]
        self.vb = [np.zeros_like(layer.b) for layer in layers]

    def step(self, layers):
        for i, layer in enumerate(layers):

            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            v_prev_W = self.vW[i].copy()
            v_prev_b = self.vb[i].copy()

            self.vW[i] = self.beta * self.vW[i] + grad_W
            self.vb[i] = self.beta * self.vb[i] + grad_b

            layer.W -= self.lr * (self.beta * v_prev_W + (1 - self.beta) * grad_W)
            layer.b -= self.lr * (self.beta * v_prev_b + (1 - self.beta) * grad_b)


class RMSProp:
    def __init__(self, layers, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay

        self.sW = [np.zeros_like(layer.W) for layer in layers]
        self.sb = [np.zeros_like(layer.b) for layer in layers]

    def step(self, layers):
        for i, layer in enumerate(layers):

            grad_W = layer.grad_W + self.weight_decay * layer.W
            grad_b = layer.grad_b

            self.sW[i] = self.beta * self.sW[i] + (1 - self.beta) * (grad_W ** 2)
            self.sb[i] = self.beta * self.sb[i] + (1 - self.beta) * (grad_b ** 2)

            layer.W -= self.lr * grad_W / (np.sqrt(self.sW[i]) + self.eps)
            layer.b -= self.lr * grad_b / (np.sqrt(self.sb[i]) + self.eps)