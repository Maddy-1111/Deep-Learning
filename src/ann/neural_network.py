"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""
import numpy as np
from ann.neural_layer import NeuralLayer
from ann.activations import ReLU, Sigmoid, Tanh
from ann.objective_functions import CrossEntropyLoss, MSELoss
from ann.optimizers import SGD, Momentum, NAG, RMSProp



class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):

        self.lr = cli_args.learning_rate
        self.weight_decay = cli_args.weight_decay

        # choose activation
        if cli_args.activation == "relu":
            act = ReLU
        elif cli_args.activation == "sigmoid":
            act = Sigmoid
        elif cli_args.activation == "tanh":
            act = Tanh
        else:
            raise ValueError("Unsupported activation")

        self.loss_name = cli_args.loss
        # choose loss
        if cli_args.loss == "cross_entropy":
            self.loss_fn = CrossEntropyLoss()
        else:
            self.loss_fn = MSELoss()

        # build architecture
        input_dim = 784
        output_dim = 10

        sizes = [input_dim] + cli_args.hidden_size + [output_dim]

        self.layers = []

        for i in range(len(sizes) - 1):

            activation = None
            if i < len(sizes) - 2:
                activation = act()

            layer = NeuralLayer(
                sizes[i],
                sizes[i+1],
                activation=activation,
                weight_init=cli_args.weight_init
            )

            self.layers.append(layer)

        #choose optimizer
        if hasattr(cli_args, "optimizer"):
            if cli_args.optimizer == "sgd":
                self.optimizer = SGD(cli_args.learning_rate, cli_args.weight_decay)

            elif cli_args.optimizer == "momentum":
                self.optimizer = Momentum(self.layers, cli_args.learning_rate)

            elif cli_args.optimizer == "nag":
                self.optimizer = NAG(self.layers, cli_args.learning_rate)

            elif cli_args.optimizer == "rmsprop":
                self.optimizer = RMSProp(self.layers, cli_args.learning_rate)


    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits
        """

        A = X

        for layer in self.layers:
            A = layer.forward(A)

        return A


    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        """

        grad_W_list = []
        grad_b_list = []

        m = y_true.shape[0]

        # gradient of loss w.r.t logits
        if self.loss_name == "cross_entropy":

            # compute softmax
            logits_shift = y_pred - np.max(y_pred, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shift)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            dA = (probs - y_true) / m

        elif self.loss_name == "mse":

            dA = 2 * (y_pred - y_true) / m

        else:
            raise ValueError("Unsupported loss")

        # backprop through layers
        for layer in reversed(self.layers):

            dA = layer.backward(dA)

            grad_W_list.append(layer.grad_W)
            grad_b_list.append(layer.grad_b)

        # convert to numpy object arrays
        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)

        for i, (gw, gb) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[i] = gw
            self.grad_b[i] = gb

        return self.grad_W, self.grad_b  


    def update_weights(self):
        self.optimizer.step(self.layers)


    def train(self, X_train, y_train, epochs=1, batch_size=32):

        n = X_train.shape[0]

        for epoch in range(epochs):

            indices = np.random.permutation(n)

            for start in range(0, n, batch_size):

                batch_idx = indices[start:start+batch_size]

                Xb = X_train[batch_idx]
                yb = y_train[batch_idx]

                logits = self.forward(Xb)

                loss = self.loss_fn.forward(logits, yb)

                self.backward(yb, logits)

                self.update_weights()


    def evaluate(self, X, y):

        logits = self.forward(X)

        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)

        accuracy = np.mean(preds == labels)

        return accuracy


    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f"W{i}"] = layer.W.copy()
            d[f"b{i}"] = layer.b.copy()
        return d


    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f"W{i}"
            b_key = f"b{i}"
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()