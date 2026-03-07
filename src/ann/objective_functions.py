"""
Loss/Objective Functions and Their Derivatives
Implements: Cross-Entropy, Mean Squared Error (MSE)
"""

import numpy as np


class CrossEntropyLoss:
    """
    Softmax + Cross Entropy combined (numerically stable)
    """

    def forward(self, logits, y_true):
        """
        logits: (b, C)
        y_true: (b, C) one-hot
        """

        ################ stable softmax (subtract max logit to prevent overflow) ################
        logits_shift = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shift)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        self.y_true = y_true
        b = logits.shape[0]

        loss = -np.sum(y_true * np.log(self.probs + 1e-12)) / b
        return loss

    def backward(self):
        """
        gradient w.r.t logits
        """
        b = self.y_true.shape[0]
        return (self.probs - self.y_true) / b


class MSELoss:
    """
    Mean Squared Error
    """

    def forward(self, y_pred, y_true):
        """
        y_pred: (b, C)
        y_true: (b, C)
        """

        self.y_pred = y_pred
        self.y_true = y_true

        b = y_pred.shape[0]
        loss = np.sum((y_pred - y_true) ** 2) / b
        return loss

    def backward(self):
        """
        gradient w.r.t predictions
        """
        b = self.y_true.shape[0]
        return 2 * (self.y_pred - self.y_true) / b