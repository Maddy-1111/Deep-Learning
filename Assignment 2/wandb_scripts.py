import argparse
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import wandb
from albumentations.pytorch import ToTensorV2
from PIL import Image

import models


def normalize_feature_map(x: torch.Tensor) -> torch.Tensor:
    x = x - x.min()
    return x / (x.max() + 1e-6)


def make_grid(channel_maps: torch.Tensor, title: str, max_maps: int = 16) -> plt.Figure:
    num_maps = min(max_maps, channel_maps.shape[0])
    cols = 4
    rows = int(np.ceil(num_maps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for idx in range(rows * cols):
        ax = axes[idx]
        ax.axis("off")
        if idx < num_maps:
            ax.imshow(channel_maps[idx].cpu().numpy(), cmap="viridis")
            ax.set_title(f"ch {idx}", fontsize=9)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize first/last conv feature maps for Task 1 classifier")
    parser.add_argument("--image-path", type=str, default="dataset/images/leonberger_178.jpg")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/classification.pth")
    parser.add_argument("--num-maps", type=int, default=16)
    parser.add_argument("--project", type=str, default="DA6401_Assignment_2")
    parser.add_argument("--run-name", type=str, default="feature_map_visualization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=args.project, name=args.run_name)

    model = models.VGG11Classifier(num_classes=37).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    image = Image.open(args.image_path).convert("RGB")
    image_np = np.array(image)
    image_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)

    features = {}

    def first_hook(_m, _i, o):
        features["first"] = o.detach()

    def last_hook(_m, _i, o):
        features["last"] = o.detach()

    conv_layers = [module for module in model.encoder.modules() if isinstance(module, nn.Conv2d)]
    if len(conv_layers) < 2:
        raise RuntimeError("Expected at least two Conv2d layers in the encoder.")

    first_handle = conv_layers[0].register_forward_hook(first_hook)
    last_handle = conv_layers[-1].register_forward_hook(last_hook)

    with torch.no_grad():
        _ = model(image_tensor)

    first_handle.remove()
    last_handle.remove()

    first_maps = normalize_feature_map(features["first"][0].cpu())
    last_maps = normalize_feature_map(features["last"][0].cpu())

    first_fig = make_grid(first_maps, "First Convolution Layer Feature Maps", max_maps=args.num_maps)
    last_fig = make_grid(last_maps, "Last Convolution Layer Feature Maps", max_maps=args.num_maps)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    first_path = output_dir / "first_conv_feature_maps.png"
    last_path = output_dir / "last_conv_feature_maps.png"
    first_fig.savefig(first_path, dpi=180)
    last_fig.savefig(last_path, dpi=180)
    plt.close(first_fig)
    plt.close(last_fig)

    wandb.log(
        {
            "input_image": wandb.Image(image_np),
            "first_conv_feature_maps": wandb.Image(str(first_path)),
            "last_conv_feature_maps": wandb.Image(str(last_path)),
        }
    )
    wandb.finish()

    print(f"Saved first-layer feature-map grid to: {first_path}")
    print(f"Saved last-layer feature-map grid to: {last_path}")


if __name__ == "__main__":
    main()