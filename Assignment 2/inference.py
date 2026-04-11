"""Inference and evaluation for Assignment 2."""

import argparse

import albumentations as A
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from losses import IoULoss
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.multitask import MultiTaskPerceptionModel
from models.segmentation import VGG11UNet


def parse_args():
	parser = argparse.ArgumentParser(description="Inference for Oxford-IIIT Pet tasks")
	parser.add_argument(
		"--task",
		type=str,
		default="classification",
		choices=["classification", "localization", "segmentation", "multitask"],
	)
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


def evaluate_classification(model, loader, device, pin_memory):
	criterion = nn.CrossEntropyLoss()
	total_loss = 0.0
	total_samples = 0
	preds_all = []
	labels_all = []

	with torch.no_grad():
		for batch in loader:
			images = batch["image"].to(device, non_blocking=pin_memory)
			labels = batch["label"].to(device, non_blocking=pin_memory)

			logits = model(images)
			loss = criterion(logits, labels)

			bs = labels.size(0)
			total_loss += loss.item() * bs
			total_samples += bs
			preds_all.append(torch.argmax(logits, dim=1).cpu())
			labels_all.append(labels.cpu())

	y_pred = torch.cat(preds_all).numpy()
	y_true = torch.cat(labels_all).numpy()
	return {
		"loss": total_loss / max(total_samples, 1),
		"accuracy": accuracy_score(y_true, y_pred),
	}


def evaluate_localization(model, loader, device, pin_memory):
	mse_criterion = nn.MSELoss()
	iou_criterion = IoULoss(reduction="mean")
	total_mse = 0.0
	total_iou_loss = 0.0
	total_samples = 0

	with torch.no_grad():
		for batch in loader:
			images = batch["image"].to(device, non_blocking=pin_memory)
			targets = batch["bbox"].to(device, non_blocking=pin_memory).float()
			sizes = batch["orig_size"].to(device, non_blocking=pin_memory)

			preds = model(images, sizes)

			bs = targets.size(0)
			total_mse += mse_criterion(preds, targets).item() * bs
			total_iou_loss += iou_criterion(preds, targets).item() * bs
			total_samples += bs

	mean_mse = total_mse / max(total_samples, 1)
	mean_iou_loss = total_iou_loss / max(total_samples, 1)
	return {
		"mse": mean_mse,
		"iou_loss": mean_iou_loss,
		"iou": 1.0 - mean_iou_loss,
	}


def evaluate_segmentation(model, loader, device, pin_memory):
	criterion = nn.CrossEntropyLoss()
	total_loss = 0.0
	total_correct = 0
	total_pixels = 0

	with torch.no_grad():
		for batch in loader:
			images = batch["image"].to(device, non_blocking=pin_memory)
			masks = batch["mask"].to(device, non_blocking=pin_memory)

			logits = model(images)
			loss = criterion(logits, masks)

			pred = torch.argmax(logits, dim=1)
			total_correct += (pred == masks).sum().item()
			total_pixels += masks.numel()

			bs = masks.size(0)
			total_loss += loss.item() * bs

	return {
		"loss": total_loss / max(len(loader.dataset), 1),
		"pixel_accuracy": total_correct / max(total_pixels, 1),
	}


def evaluate_multitask(model, loader, device, pin_memory):
	cls_criterion = nn.CrossEntropyLoss()
	loc_mse_criterion = nn.MSELoss()
	loc_iou_criterion = IoULoss(reduction="mean")
	seg_criterion = nn.CrossEntropyLoss()

	cls_loss = 0.0
	loc_mse = 0.0
	loc_iou_loss = 0.0
	seg_loss = 0.0

	cls_preds = []
	cls_labels = []
	seg_correct = 0
	seg_pixels = 0
	total_samples = 0

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

			bs = labels.size(0)
			total_samples += bs

			cls_loss += cls_criterion(out_cls, labels).item() * bs
			loc_mse += loc_mse_criterion(out_loc, boxes).item() * bs
			loc_iou_loss += loc_iou_criterion(out_loc, boxes).item() * bs
			seg_loss += seg_criterion(out_seg, masks).item() * bs

			cls_preds.append(torch.argmax(out_cls, dim=1).cpu())
			cls_labels.append(labels.cpu())

			seg_pred = torch.argmax(out_seg, dim=1)
			seg_correct += (seg_pred == masks).sum().item()
			seg_pixels += masks.numel()

	y_pred = torch.cat(cls_preds).numpy()
	y_true = torch.cat(cls_labels).numpy()
	mean_iou_loss = loc_iou_loss / max(total_samples, 1)

	return {
		"classification_loss": cls_loss / max(total_samples, 1),
		"classification_accuracy": accuracy_score(y_true, y_pred),
		"localization_mse": loc_mse / max(total_samples, 1),
		"localization_iou_loss": mean_iou_loss,
		"localization_iou": 1.0 - mean_iou_loss,
		"segmentation_loss": seg_loss / max(total_samples, 1),
		"segmentation_pixel_accuracy": seg_correct / max(seg_pixels, 1),
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