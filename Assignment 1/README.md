# Assignment 1: Multi-Layer Perceptron for Image Classification

Implementation of a multi-layer perceptron (MLP) to classify the MNIST dataset.

**Links:**
- [GitHub Repository](https://github.com/Maddy-1111/Deep-Learning)
- [WandB Project](https://api.wandb.ai/links/ee23b040-indian-institute-of-technology-madras/jco8gz27)

## Project Structure

```
src/
├── ann/                          # Neural network implementation
│   ├── activations.py           # Activation functions
│   ├── neural_layer.py          # Single layer implementation
│   ├── neural_network.py        # Full network
│   ├── objective_functions.py   # Loss functions
│   └── optimizers.py            # Optimization algorithms
├── utils/
│   └── data_loader.py           # Data loading utilities
├── train.py                      # Training script
├── inference.py                  # Inference script
├── analysis.py                   # Model analysis
├── sweep.yaml                    # Hyperparameter sweep config
└── best_model.npy               # Trained model weights

notebooks/
└── wandb_demo.ipynb             # WandB integration demo
```
