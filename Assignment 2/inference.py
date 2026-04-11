"""Inference and evaluation for Assignment 2."""

import argparse

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.multitask import MultiTaskPerceptionModel
from models.segmentation import VGG11UNet


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Oxford-IIIT Pet tasks")
    parser.add_argument("--task", type=str, default="classification", choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--dataset", type=str, default="./dataset")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    return parser.parse_args()


def build_transform(include_bbox: bool = False):
    bbox_params = A.BboxParams(format="albumentations", label_fields=["class_labels"]) if include_bbox else None
    return A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=bbox_params,
    )


def build_model(task: str, checkpoint: str | None, device: torch.device):
	if task == "classification":
		model = VGG11Classifier(num_classes=37)
		ckpt = checkpoint or "checkpoints/classification.pth"
		model.load_state_dict(torch.load(ckpt, map_location="cpu"))
	elif task == "localization":
		model = VGG11Localizer()
		ckpt = checkpoint or "checkpoints/localization.pth"
		model.load_state_dict(torch.load(ckpt, map_location="cpu"))
	elif task == "segmentation":
		model = VGG11UNet(num_classes=3)
		ckpt = checkpoint or "checkpoints/segmentation.pth"
		model.load_state_dict(torch.load(ckpt, map_location="cpu"))
	else:
		model = MultiTaskPerceptionModel(num_breeds=37, seg_classes=3)
	model = model.to(device)
	model.eval()
	return model


def _xywh_iou(pred_xywh: torch.Tensor, target_xywh: torch.Tensor) -> torch.Tensor:
	p_x1 = pred_xywh[:, 0] - pred_xywh[:, 2] / 2
	p_y1 = pred_xywh[:, 1] - pred_xywh[:, 3] / 2
	p_x2 = pred_xywh[:, 0] + pred_xywh[:, 2] / 2
	p_y2 = pred_xywh[:, 1] + pred_xywh[:, 3] / 2

	t_x1 = target_xywh[:, 0] - target_xywh[:, 2] / 2
	t_y1 = target_xywh[:, 1] - target_xywh[:, 3] / 2
	t_x2 = target_xywh[:, 0] + target_xywh[:, 2] / 2
	t_y2 = target_xywh[:, 1] + target_xywh[:, 3] / 2

	inter_x1 = torch.maximum(p_x1, t_x1)
	inter_y1 = torch.maximum(p_y1, t_y1)
	inter_x2 = torch.minimum(p_x2, t_x2)
	inter_y2 = torch.minimum(p_y2, t_y2)

	inter_w = (inter_x2 - inter_x1).clamp(min=0)
	inter_h = (inter_y2 - inter_y1).clamp(min=0)
	intersection = inter_w * inter_h

	area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
	area_t = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
	union = area_p + area_t - intersection + 1e-8

	return intersection / union


def _average_precision(scores: np.ndarray, ious: np.ndarray, iou_threshold: float) -> float:
	if scores.size == 0:
		return 0.0

	order = np.argsort(-scores)
	sorted_ious = ious[order]
	tp = (sorted_ious >= iou_threshold).astype(np.float64)
	fp = 1.0 - tp

	tp_cum = np.cumsum(tp)
	fp_cum = np.cumsum(fp)
	recalls = tp_cum / max(float(len(ious)), 1.0)
	precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

	mrec = np.concatenate(([0.0], recalls, [1.0]))
	mpre = np.concatenate(([0.0], precisions, [0.0]))

	for i in range(mpre.size - 1, 0, -1):
		mpre[i - 1] = max(mpre[i - 1], mpre[i])

	idx = np.where(mrec[1:] != mrec[:-1])[0]
	return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _compute_map_from_ious(ious: np.ndarray):
	if ious.size == 0:
		return 0.0, 0.0

	# No objectness head in this assignment; IoU acts as ranking confidence.
	scores = ious.copy()
	ap_50 = _average_precision(scores, ious, iou_threshold=0.50)
	thresholds = np.arange(0.50, 0.96, 0.05)
	ap_all = [_average_precision(scores, ious, iou_threshold=float(t)) for t in thresholds]
	return ap_50, float(np.mean(ap_all))


def _dice_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
	pred = torch.argmax(logits, dim=1)
	inter = torch.zeros(num_classes, dtype=torch.float64, device=logits.device)
	pred_count = torch.zeros(num_classes, dtype=torch.float64, device=logits.device)
	tgt_count = torch.zeros(num_classes, dtype=torch.float64, device=logits.device)

	for c in range(num_classes):
		pred_c = (pred == c)
		tgt_c = (target == c)
		inter[c] = (pred_c & tgt_c).sum()
		pred_count[c] = pred_c.sum()
		tgt_count[c] = tgt_c.sum()

	return inter, pred_count, tgt_count


def evaluate_classification(model, loader, device, pin_memory):
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device, non_blocking=pin_memory)

            logits = model(images)
            preds_all.append(torch.argmax(logits, dim=1).cpu())
            labels_all.append(labels.cpu())

    y_pred = torch.cat(preds_all).numpy()
    y_true = torch.cat(labels_all).numpy()
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def evaluate_localization(model, loader, device, pin_memory):
    ious_all = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=pin_memory)
            targets = batch["bbox"].to(device, non_blocking=pin_memory).float()
            sizes = batch["orig_size"].to(device, non_blocking=pin_memory)

            preds = model(images, sizes)
            ious = _xywh_iou(preds, targets)
            ious_all.append(ious.detach().cpu())

    ious_np = torch.cat(ious_all).numpy() if ious_all else np.array([], dtype=np.float32)
    map_50, map_50_95 = _compute_map_from_ious(ious_np)
    return {
        "mAP@50": map_50,
        "mAP@50:95": map_50_95,
    }


def evaluate_segmentation(model, loader, device, pin_memory):
    num_classes = 3
    total_inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_pred = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_tgt = torch.zeros(num_classes, dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=pin_memory)
            masks = batch["mask"].to(device, non_blocking=pin_memory)

            logits = model(images)
            inter, pred_count, tgt_count = _dice_from_logits(logits, masks, num_classes=num_classes)
            total_inter += inter
            total_pred += pred_count
            total_tgt += tgt_count

    dice_per_class = (2.0 * total_inter + 1e-8) / (total_pred + total_tgt + 1e-8)
    return {
        "dice": float(dice_per_class.mean().item()),
    }


def evaluate_multitask(model, loader, device, pin_memory):
    cls_preds = []
    cls_labels = []
    ious_all = []

    num_classes = 3
    total_inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_pred = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_tgt = torch.zeros(num_classes, dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=pin_memory)
            labels = batch["label"].to(device, non_blocking=pin_memory)
            masks = batch["mask"].to(device, non_blocking=pin_memory)
            boxes = batch["bbox"].to(device, non_blocking=pin_memory).float()

            out = model(images)
            out_cls = out["classification"]
            out_loc = out["localization"]
            out_seg = out["segmentation"]

            cls_preds.append(torch.argmax(out_cls, dim=1).cpu())
            cls_labels.append(labels.cpu())
            ious_all.append(_xywh_iou(out_loc, boxes).detach().cpu())

            inter, pred_count, tgt_count = _dice_from_logits(out_seg, masks, num_classes=num_classes)
            total_inter += inter
            total_pred += pred_count
            total_tgt += tgt_count

    y_pred = torch.cat(cls_preds).numpy()
    y_true = torch.cat(cls_labels).numpy()
    ious_np = torch.cat(ious_all).numpy() if ious_all else np.array([], dtype=np.float32)
    map_50, map_50_95 = _compute_map_from_ious(ious_np)

    dice_per_class = (2.0 * total_inter + 1e-8) / (total_pred + total_tgt + 1e-8)
    return {
        "classification_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "localization_mAP@50": map_50,
        "localization_mAP@50:95": map_50_95,
        "segmentation_dice": float(dice_per_class.mean().item()),
    }


def print_results(task: str, metrics: dict):
    print(f"Task: {task}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda") if args.pin_memory is None else args.pin_memory

    if args.task == "multitask":
        task_list = ["classification", "localization", "segmentation"]
        transform = build_transform(include_bbox=True)
    else:
        task_list = [args.task]
        transform = build_transform(include_bbox=(args.task == "localization"))

    dataset = OxfordIIITPetDataset(
        root_dir=args.dataset,
        split="test",
        tasks=task_list,
        transform=transform,
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    model = build_model(args.task, args.checkpoint, device)

    if args.task == "classification":
        metrics = evaluate_classification(model, loader, device, pin_memory)
    elif args.task == "localization":
        metrics = evaluate_localization(model, loader, device, pin_memory)
    elif args.task == "segmentation":
        metrics = evaluate_segmentation(model, loader, device, pin_memory)
    else:
        metrics = evaluate_multitask(model, loader, device, pin_memory)

    print_results(args.task, metrics)
    return metrics


if __name__ == "__main__":
    main()