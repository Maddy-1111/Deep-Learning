"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np


class CrossEntropyLoss:

    def forward(self, logits, y_true):

        logits_shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shift)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.y_true = y_true
        b = logits.shape[0]

        loss = -np.sum(y_true * np.log(self.probs + 1e-12)) / b
        return loss

    def backward(self, logits, y_true):

        b = logits.shape[0]
        logits_shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shift)

        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return (probs - y_true) / b


class MSELoss:

    def forward(self, y_pred, y_true):

        self.y_pred = y_pred
        self.y_true = y_true

        b = y_pred.shape[0]
        loss = np.sum((y_pred - y_true) ** 2) / b
        return loss

    def backward(self, y_pred, y_true):
        
        b = y_true.shape[0]
        return 2 * (y_pred - y_true) / b