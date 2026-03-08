import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required="best_model.npy")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist"])

    parser.add_argument("--hidden_size", type=int, nargs="+", default=[128,128,128])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--loss", type=str, default="cross_entropy")

    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--weight_init", type=str, default="xavier")

    return parser.parse_args()


def plot_confusion_matrix(labels, preds):

    cm = confusion_matrix(labels, preds)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix (Test Set)")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.show()

    return cm


def plot_misclassified_images(X_test, labels, preds, max_images=25):

    mis_idx = np.where(preds != labels)[0]

    plt.figure(figsize=(10,10))

    for i in range(min(max_images, len(mis_idx))):

        idx = mis_idx[i]

        plt.subplot(5,5,i+1)
        plt.imshow(X_test[idx].reshape(28,28), cmap="gray")

        plt.title(f"T:{labels[idx]} P:{preds[idx]}")
        plt.axis("off")

    plt.suptitle("Misclassified Test Images")

    plt.savefig("misclassified_examples.png", dpi=300)
    plt.show()


def most_confused_pairs(cm):

    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    flat_idx = np.argsort(cm_no_diag.ravel())[::-1]

    print("\nMost confused class pairs:\n")

    for i in range(5):

        r, c = np.unravel_index(flat_idx[i], cm.shape)

        print(f"True {r} predicted as {c} : {cm[r,c]} times")


def main():

    args = parse_args()

    # load dataset
    X_train, y_train, X_test, y_test = load_dataset(args.dataset)

    # rebuild model
    model = NeuralNetwork(args)

    # load weights
    weights = np.load(args.model_path, allow_pickle=True).item()
    model.set_weights(weights)

    # predictions
    logits = model.forward(X_test)

    preds = np.argmax(logits, axis=1)
    labels = np.argmax(y_test, axis=1)

    # confusion matrix
    cm = plot_confusion_matrix(labels, preds)

    # creative visualization
    plot_misclassified_images(X_test, labels, preds)

    # most confused pairs
    most_confused_pairs(cm)


if __name__ == "__main__":
    main()