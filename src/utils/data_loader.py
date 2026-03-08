"""
Data Loading and Preprocessing
Handles MNIST and Fashion-MNIST datasets
"""

import numpy as np
from keras.datasets import mnist, fashion_mnist


def one_hot_encode(y, num_classes=10):
    y_onehot = np.zeros((y.shape[0], num_classes))
    y_onehot[np.arange(y.shape[0]), y] = 1
    return y_onehot


def normalize(X):
    return X.astype(np.float32) / 255.0


def flatten(X):
    return X.reshape(X.shape[0], -1)


def load_dataset(dataset_name="mnist"):
    """
    Loads MNIST or Fashion-MNIST and returns
    X_train, y_train, X_test, y_test

    X shape: (N, 784)
    y shape: (N, 10) one-hot
    """

    if dataset_name.lower() == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    elif dataset_name.lower() == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    else:
        raise ValueError("dataset must be 'mnist' or 'fashion_mnist'")

    # preprocessing
    X_train = normalize(flatten(X_train))
    X_test = normalize(flatten(X_test))

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    return X_train, y_train, X_test, y_test


def create_batches(X, y, batch_size, shuffle=True):

    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]