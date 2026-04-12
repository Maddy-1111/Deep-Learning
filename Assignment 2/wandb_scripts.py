import argparse
import io
from pathlib import Path

import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn as nn
import wandb
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader

import models
from data.pets_dataset import OxfordIIITPetDataset


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


def calculate_iou(pred_box: np.ndarray, target_box: np.ndarray) -> float:
    """Calculate IoU between two boxes in [xc, yc, w, h] format.
    
    Args:
        pred_box: Predicted box [xc, yc, w, h]
        target_box: Ground truth box [xc, yc, w, h]
    
    Returns:
        IoU score between 0 and 1
    """
    eps = 1e-6
    
    # Convert from [xc, yc, w, h] to [x1, y1, x2, y2]
    def to_corners(box):
        xc, yc, w, h = box
        return [xc - w/2, yc - h/2, xc + w/2, yc + h/2]
    
    p_box = to_corners(pred_box)
    t_box = to_corners(target_box)
    
    # Intersection
    inter_x1 = max(p_box[0], t_box[0])
    inter_y1 = max(p_box[1], t_box[1])
    inter_x2 = min(p_box[2], t_box[2])
    inter_y2 = min(p_box[3], t_box[3])
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    
    # Union
    pred_area = pred_box[2] * pred_box[3]
    target_area = target_box[2] * target_box[3]
    union = pred_area + target_area - intersection
    
    iou = intersection / (union + eps)
    return float(iou)


def draw_boxes_on_image(image_np: np.ndarray, pred_box: np.ndarray, target_box: np.ndarray) -> plt.Figure:
    """Draw prediction and ground truth boxes on image.
    
    Args:
        image_np: Image array (H, W, 3)
        pred_box: Predicted box [xc, yc, w, h]
        target_box: Ground truth box [xc, yc, w, h]
    
    Returns:
        Figure with boxes drawn
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image_np)
    ax.axis("off")
    
    def draw_box(ax, box, color, label):
        xc, yc, w, h = box
        x1 = xc - w/2
        y1 = yc - h/2
        rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none", label=label)
        ax.add_patch(rect)
    
    # Draw ground truth (green) and prediction (red)
    draw_box(ax, target_box, "green", "Ground Truth")
    draw_box(ax, pred_box, "red", "Prediction")
    
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    return fig


def fig_to_image(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf)
    img.load()
    plt.close(fig)
    return img


def visualize_bbox_predictions(
    image: Image.Image,
    pred_box: np.ndarray,
    target_box: np.ndarray,
    iou: float
) -> np.ndarray:
    """Create visualization with bounding boxes overlaid on image.
    
    Returns:
        Image array suitable for wandb.Image()
    """
    image_np = np.array(image)
    
    fig = draw_boxes_on_image(image_np, pred_box, target_box)
    img_with_boxes = fig_to_image(fig)
    
    return np.array(img_with_boxes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Log bounding box predictions with W&B table")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/localization.pth")
    parser.add_argument("--classifier-checkpoint", type=str, default="checkpoints/classification.pth")
    parser.add_argument("--dataset", type=str, default="dataset")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--project", type=str, default="DA6401_Assignment_2")
    parser.add_argument("--run-name", type=str, default="bbox_predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(project=args.project, name=args.run_name)

    # Load localization model
    model = models.VGG11Localizer().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    # Load classifier model for confidence scores
    classifier = models.VGG11Classifier(num_classes=37).to(device)
    classifier_checkpoint = torch.load(args.classifier_checkpoint, map_location="cpu")

    if isinstance(classifier_checkpoint, dict) and "model_state_dict" in classifier_checkpoint:
        classifier_state_dict = classifier_checkpoint["model_state_dict"]
    elif isinstance(classifier_checkpoint, dict) and "state_dict" in classifier_checkpoint:
        classifier_state_dict = classifier_checkpoint["state_dict"]
    else:
        classifier_state_dict = classifier_checkpoint

    classifier.load_state_dict(classifier_state_dict)
    classifier.eval()

    # Load test dataset
    transform = A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="albumentations", label_fields=["class_labels"]),
    )

    test_ds = OxfordIIITPetDataset(
        root_dir=args.dataset,
        split="test",
        tasks=["localization"],
        transform=transform
    )

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Create W&B table
    table = wandb.Table(columns=[
        "Image",
        "Image ID",
        "Ground Truth Box\n(xc, yc, w, h)",
        "Predicted Box\n(xc, yc, w, h)",
        "Confidence Score",
        "Intersection over Union (IoU)"
    ])

    # Process test samples
    num_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if num_processed >= args.num_samples:
                break

            images = batch["image"].to(device)
            target_boxes = batch["bbox"].to(device)  # [1, 4]
            orig_sizes = batch["orig_size"].to(device)  # [1, 2]

            # Get localization prediction
            pred_coords_normalized = model(images, orig_sizes)  # [1, 4] in absolute coords

            # Get classifier logits for confidence score
            classifier_logits = classifier(images)  # [1, 37]
            # Use softmax to get probabilities and take the max as confidence
            confidences = torch.softmax(classifier_logits, dim=1)
            confidence = confidences[0].max().cpu().item()

            # Since model returns absolute coordinates when image_size is provided,
            # we need to normalize for comparison
            orig_w, orig_h = orig_sizes[0].cpu().numpy()
            pred_box_np = pred_coords_normalized[0].cpu().numpy()
            target_box_np = target_boxes[0].cpu().numpy()

            # Normalize predictions to [xc, yc, w, h] format with original image dimensions
            # The model output is already in this format when image_size is provided
            iou = calculate_iou(pred_box_np, target_box_np)

            # Load original image for visualization
            img_id = test_ds.image_ids[batch_idx]
            image_path = Path(args.dataset) / "images" / f"{img_id}.jpg"
            image_orig = Image.open(image_path).convert("RGB")

            # Create visualization
            vis_image = visualize_bbox_predictions(image_orig, pred_box_np, target_box_np, iou)

            # Format box strings
            gt_box_str = f"({target_box_np[0]:.1f}, {target_box_np[1]:.1f}, {target_box_np[2]:.1f}, {target_box_np[3]:.1f})"
            pred_box_str = f"({pred_box_np[0]:.1f}, {pred_box_np[1]:.1f}, {pred_box_np[2]:.1f}, {pred_box_np[3]:.1f})"

            # Add to table
            table.add_data(
                wandb.Image(vis_image),
                img_id,
                gt_box_str,
                pred_box_str,
                f"{confidence:.4f}",
                f"{iou:.4f}"
            )

            num_processed += 1

    # Log table to W&B
    wandb.log({"bbox_predictions_table": table})
    wandb.finish()

    print(f"Logged {num_processed} test samples with bounding box predictions to W&B")


if __name__ == "__main__":
    main()