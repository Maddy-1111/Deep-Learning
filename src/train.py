"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import numpy as np
import wandb

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset
from sklearn.model_selection import train_test_split



def parse_arguments():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(description="Train a neural network")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--epochs", type=int, default=5)
    
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--learning_rate", type=float, default=0.001)

    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "momentum", "nag", "rmsprop"])
    
    parser.add_argument("--num_layers", type=int, default=None)

    parser.add_argument("--hidden_size", type=str, nargs="+", default=[128, 128])
    
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu","sigmoid","tanh"])

    parser.add_argument("--loss", type=str, default="cross_entropy",
                        choices=["cross_entropy","mse"])

    parser.add_argument("--weight_init", type=str, default="xavier",
                        choices=["random","xavier"])

    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--wandb_project", type=str, default="da6401_assignment_1")

    parser.add_argument("--model_save_path", type=str, default="best_model.npy")

    return parser.parse_args()


def main():

    args = parse_arguments()

    try:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            group=getattr(args, "group", None)
        )
    except:
        wandb = None

    config = wandb.config if wandb else vars(args)

    args.learning_rate = config.get("learning_rate", args.learning_rate)
    args.batch_size = config.get("batch_size", args.batch_size)
    args.optimizer = config.get("optimizer", args.optimizer)
    args.activation = config.get("activation", args.activation)
    args.weight_init = config.get("weight_init", args.weight_init)

    args.hidden_size = config.get("hidden_size", args.hidden_size)
    args.hidden_size = list(map(int, " ".join(str(x) for x in args.hidden_size).split()))

    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42
    )

    # build model
    model = NeuralNetwork(args)

    best_acc = 0
    best_weights = None

    for epoch in range(args.epochs):

        model.train(X_train, y_train, epochs=1, batch_size=args.batch_size)

        idx_train = np.random.choice(len(X_train), 6000, replace=False)
        idx_test  = np.random.choice(len(X_test), 6000, replace=False)

        train_acc = model.evaluate(X_train[idx_train], y_train[idx_train])
        val_acc   = model.evaluate(X_val, y_val)
        test_acc  = model.evaluate(X_test[idx_test], y_test[idx_test])

        print(f"Epoch {epoch+1} val accuracy: {val_acc}")

        if wandb:
            wandb.log({
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "test_accuracy": test_acc
            }, step=epoch)

            # Only using validation accuracy for model selection. Test accuracy is only for final evaluation after training is complete.
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = model.get_weights()

    if best_weights is not None:
        np.save(args.model_save_path, best_weights)

    print("Training complete!")


if __name__ == "__main__":
    main()