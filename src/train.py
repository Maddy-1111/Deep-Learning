"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop"])

    parser.add_argument("--hidden_size", type=int, nargs="+", default=[128,128])

    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu","sigmoid","tanh"])

    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy","mse"])

    parser.add_argument("--weight_init", type=str, default="xavier",
                        choices=["random","xavier"])

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--wandb_project", type=str, default="da6401_assignment")

    parser.add_argument("--model_save_path", type=str, default="best_model.npy")

    return parser.parse_args()


def main():
    """
    Main training function.
    """

    args = parse_arguments()

    # initialize wandb
    wandb.init(project=args.wandb_project, config=vars(args))

    # load dataset
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    # build model
    model = NeuralNetwork(args)

    best_f1 = 0
    best_weights = None

    for epoch in range(args.epochs):

        model.train(X_train, y_train, epochs=1, batch_size=args.batch_size)

        acc = model.evaluate(X_test, y_test)

        wandb.log({
            "epoch": epoch,
            "test_accuracy": acc
        })

        print(f"Epoch {epoch+1} test accuracy: {acc}")

        # save best model
        if acc > best_f1:
            best_f1 = acc
            best_weights = model.get_weights()

    # save model weights
    if best_weights is not None:
        np.save(args.model_save_path, best_weights)

    print("Training complete!")


if __name__ == "__main__":
    main()