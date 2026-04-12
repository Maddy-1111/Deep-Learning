import argparse
import re
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import wandb
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont

import models


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_breed_names(list_path: Path) -> list[str]:
    breed_names = {}
    with open(list_path, "r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            breed_name = re.sub(r"[_-]?\d+$", "", parts[0]).strip()
            breed_names[int(parts[1]) - 1] = breed_name or parts[0]

    if not breed_names:
        raise ValueError(f"Could not read breed labels from {list_path}")

    return [breed_names[index] for index in sorted(breed_names)]


def trimap_to_rgb(mask: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [255, 140, 0],
            [30, 144, 255],
            [255, 255, 255],
        ],
        dtype=np.uint8,
    )
    mask = np.clip(mask.astype(np.int64), 0, 2)
    return palette[mask]


def build_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=tuple(IMAGENET_MEAN.tolist()), std=tuple(IMAGENET_STD.tolist())),
            ToTensorV2(),
        ]
    )


def load_state_dict(checkpoint_path: str) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def build_model(task: str, checkpoint_path: str, device: torch.device):
    if task == "classification":
        model = models.VGG11Classifier(num_classes=37)
    elif task == "localization":
        model = models.VGG11Localizer()
    elif task == "segmentation":
        model = models.VGG11UNet(num_classes=3)
    else:
        raise ValueError(f"Unsupported task: {task}")

    model.load_state_dict(load_state_dict(checkpoint_path))
    model.to(device)
    model.eval()
    return model


def list_image_files(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS]
    if not image_paths:
        raise ValueError(f"No supported images found in {image_dir}")
    return image_paths


def draw_bbox_overlay(image_np: np.ndarray, bbox_xywh: np.ndarray, label_text: str) -> np.ndarray:
    image = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x_center, y_center, width, height = [float(value) for value in bbox_xywh]
    x1 = max(0.0, x_center - width / 2.0)
    y1 = max(0.0, y_center - height / 2.0)
    x2 = min(float(image.width - 1), x_center + width / 2.0)
    y2 = min(float(image.height - 1), y_center + height / 2.0)

    draw.rectangle([x1, y1, x2, y2], outline=(255, 69, 0), width=4)
    if label_text:
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_y1 = max(0.0, y1 - text_height - 6.0)
        draw.rectangle([x1, text_y1, x1 + text_width + 8.0, text_y1 + text_height + 6.0], fill=(255, 69, 0))
        draw.text((x1 + 4.0, text_y1 + 2.0), label_text, fill=(255, 255, 255), font=font)

    return np.asarray(image)


def resize_rgb_image(image_np: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(image_np).resize(size_xy, Image.NEAREST))


def main() -> None:
    parser = argparse.ArgumentParser(description="Log predictions for images in a folder with W&B")
    parser.add_argument("--image-dir", type=str, default="google-images")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--classification-checkpoint", type=str, default="checkpoints/classification.pth")
    parser.add_argument("--localization-checkpoint", type=str, default="checkpoints/localization.pth")
    parser.add_argument("--segmentation-checkpoint", type=str, default="checkpoints/segmentation.pth")
    parser.add_argument("--labels-file", type=str, default="dataset/annotations/list.txt")
    parser.add_argument("--project", type=str, default="DA6401_Assignment_2")
    parser.add_argument("--run-name", type=str, default="google_images_multitask_predictions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = build_transform()
    breed_names = load_breed_names(Path(args.labels_file))

    run = wandb.init(project=args.project, name=args.run_name)

    classification_model = build_model("classification", args.classification_checkpoint, device)
    localization_model = build_model("localization", args.localization_checkpoint, device)
    segmentation_model = build_model("segmentation", args.segmentation_checkpoint, device)

    image_paths = list_image_files(Path(args.image_dir))
    if args.max_images is not None:
        image_paths = image_paths[: max(0, args.max_images)]

    if not image_paths:
        raise ValueError("No images selected for logging.")

    table = wandb.Table(columns=[
        "Image Name",
        "BBox Overlay",
        "Classification",
        "Trimap",
    ])

    num_processed = 0
    with torch.no_grad():
        for image_path in image_paths:
            original_image = Image.open(image_path).convert("RGB")
            original_np = np.asarray(original_image)
            orig_size = torch.tensor([[original_image.width, original_image.height]], dtype=torch.float32, device=device)

            transformed = transform(image=original_np)
            image_tensor = transformed["image"].unsqueeze(0).to(device)

            cls_logits = classification_model(image_tensor)
            cls_probs = torch.softmax(cls_logits, dim=1)
            cls_index = int(torch.argmax(cls_probs, dim=1).item())
            cls_confidence = float(cls_probs[0, cls_index].item())
            cls_name = breed_names[cls_index] if 0 <= cls_index < len(breed_names) else f"class_{cls_index}"
            cls_text = f"{cls_name} ({cls_confidence:.2%})"

            bbox_xywh = localization_model(image_tensor, orig_size)[0].detach().cpu().numpy()
            overlay_np = draw_bbox_overlay(original_np, bbox_xywh, cls_text)

            seg_logits = segmentation_model(image_tensor)
            seg_mask = torch.argmax(seg_logits, dim=1)[0].detach().cpu().numpy()
            seg_trimap = trimap_to_rgb(seg_mask)
            seg_trimap = resize_rgb_image(seg_trimap, (original_image.width, original_image.height))

            table.add_data(
                image_path.name,
                wandb.Image(overlay_np),
                cls_text,
                wandb.Image(seg_trimap),
            )

            wandb.log(
                {
                    f"samples/{image_path.stem}_classification": cls_text,
                    f"samples/{image_path.stem}_trimap": wandb.Image(seg_trimap),
                },
                step=num_processed,
            )
            num_processed += 1

    wandb.log({"google_images_predictions_table": table})

    if run is not None:
        run.summary["images_processed"] = num_processed
        run.summary["image_dir"] = str(Path(args.image_dir))
        run.summary["classification_checkpoint"] = args.classification_checkpoint
        run.summary["localization_checkpoint"] = args.localization_checkpoint
        run.summary["segmentation_checkpoint"] = args.segmentation_checkpoint

    wandb.finish()

    print(f"Logged {num_processed} images from {args.image_dir} to W&B.")


if __name__ == "__main__":
    main()