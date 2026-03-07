"""
Inference Script
Evaluate trained models on test sets
"""

import argparse
import numpy as np
from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from ann.objective_functions import CrossEntropyLoss, MSELoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def parse_arguments():
    """
    Parse command-line arguments for inference.
    """

    parser = argparse.ArgumentParser(description="Run inference on test set")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Relative path to saved model weights (.npy)")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--hidden_size", type=int, nargs="+", default=[128,128])

    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu","sigmoid","tanh"])

    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy","mse"])

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--weight_init", type=str, default="xavier")

    return parser.parse_args()


def load_model(model_path):
    """
    Load trained model weights.
    """
    data = np.load(model_path, allow_pickle=True).item()
    return data


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test data.
    """

    logits = model.forward(X_test)

    # loss
    if isinstance(model.loss_fn, CrossEntropyLoss):
        loss = model.loss_fn.forward(logits, y_test)
    else:
        loss = model.loss_fn.forward(logits, y_test)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")

    return {
        "logits": logits,
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():

    args = parse_arguments()

    # load dataset
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    # rebuild model
    model = NeuralNetwork(args)

    # load weights
    weights = load_model(args.model_path)
    model.set_weights(weights)

    results = evaluate_model(model, X_test, y_test)

    print("Loss:", results["loss"])
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision"])
    print("Recall:", results["recall"])
    print("F1:", results["f1"])

    print("Evaluation complete!")

    return results


if __name__ == "__main__":
    main()