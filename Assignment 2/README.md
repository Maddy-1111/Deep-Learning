# Assignment 2: Multi-Task Learning on Oxford-IIIT Pets

Implementation of classification, localization, and segmentation pipelines for the Oxford-IIIT Pets dataset.

**Links:**
- [GitHub Repository](https://github.com/Maddy-1111/Deep-Learning)
- [WandB Project](# Assignment 1: Multi-Layer Perceptron for Image Classification

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
)

## Project Structure

```
models/                 # Classification, localization, segmentation, and multitask models
losses/                 # Task-specific loss functions (including IoU loss)
data/                   # Dataset loader scripts
checkpoints/            # Saved model checkpoints
dataset/                # Oxford-IIIT Pets dataset files
google-images/          # Real-world images for inference
outputs/                # Generated predictions/outputs
train.py                # Training entry point
inference.py            # Evaluation/inference entry point
wandb_scripts.py        # Logging/visualization scripts for WandB
```

## Training

to train, run:

```bash
python train.py --task classification --epochs 20 --batch-size 64
python train.py --task localization --epochs 30 --pretrained-classifier ./checkpoints/classification.pth
python train.py --task segmentation --epochs 20 --pretrained-classifier ./checkpoints/classification.pth
```

## Validation

to validate scores run:

```bash
python inference.py --task classification --checkpoint ./checkpoints/classification.pth
python inference.py --task localization --checkpoint ./checkpoints/localization.pth
python inference.py --task segmentation --checkpoint ./checkpoints/segmentation.pth
```

## Real Image Inference

to run on real images, you can run:

```bash
python wandb_scripts.py --image-dir google-images --project DA6401_Assignment_2 --run-name google_images_predictions
```