import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses import IoULoss


def evaluate_segmentation(model, loader, criterion, device, pin_memory, num_classes=3):
    model.eval()
    running_val_loss = 0.0
    total_correct = 0
    total_pixels = 0

    total_inter = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_pred = torch.zeros(num_classes, dtype=torch.float64, device=device)
    total_tgt = torch.zeros(num_classes, dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=pin_memory)
            masks = batch['mask'].to(device, non_blocking=pin_memory)

            logits = model(images)
            running_val_loss += criterion(logits, masks).item()

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == masks).sum().item()
            total_pixels += masks.numel()

            for c in range(num_classes):
                pred_c = (preds == c)
                tgt_c = (masks == c)
                total_inter[c] += (pred_c & tgt_c).sum()
                total_pred[c] += pred_c.sum()
                total_tgt[c] += tgt_c.sum()

    val_loss = running_val_loss / len(loader)
    pixel_accuracy = total_correct / max(total_pixels, 1)
    dice_per_class = (2.0 * total_inter + 1e-8) / (total_pred + total_tgt + 1e-8)
    macro_dice = float(dice_per_class.mean().item())
    return val_loss, pixel_accuracy, macro_dice


def evaluate_localization_loss(model, loader, criterion, device, pin_memory):
    model.eval()
    running_test_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device, non_blocking=pin_memory)
            targets = batch['bbox'].to(device, non_blocking=pin_memory).float()
            sizes = batch['orig_size'].to(device, non_blocking=pin_memory)

            outputs = model(images, sizes)
            running_test_loss += criterion(outputs, targets).item()

    return running_test_loss / len(loader)

def train():
    parser = argparse.ArgumentParser(description="Train VGG11 for different tasks")
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'localization', 'segmentation'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='./dataset')
    parser.add_argument('--pretrained-classifier', type=str, default=None)
    parser.add_argument('--resume-checkpoint', type=str, default=None)
    parser.add_argument('--fine-tune', type=str, default='full', choices=['strict', 'partial', 'full'])
    parser.add_argument('--batchnorm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--wandb-project', type=str, default='DA6401_Assignment_2')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    parser.add_argument('--wandb', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    pin_memory = (device.type == 'cuda') if args.pin_memory is None else args.pin_memory

    # 1. Setup Task-Specific Logic
    if args.task == 'classification':
        model = VGG11Classifier(num_classes=37, dropout_p=args.dropout, use_batchnorm=args.batchnorm, fine_tune=args.fine_tune).to(device)
        criterion = nn.CrossEntropyLoss()
        data_key = 'label'
    elif args.task == 'localization':
        model = VGG11Localizer(pretrained_path=args.pretrained_classifier, fine_tune=args.fine_tune, dropout_p=args.dropout).to(device)
        criterion = IoULoss() 
        data_key = 'bbox'
    elif args.task == 'segmentation':
        model = VGG11UNet(pretrained_path=args.pretrained_classifier, fine_tune=args.fine_tune, dropout_p=args.dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        data_key = 'mask'

    # Optional resume from a task checkpoint.
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    # 2. Transforms
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']) if args.task == 'localization' else None)

    # 3. Data Loaders
    train_ds = OxfordIIITPetDataset(root_dir=args.dataset, split='train', tasks=[args.task], transform=transform)
    test_ds = OxfordIIITPetDataset(root_dir=args.dataset, split='test', tasks=[args.task], transform=transform)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': args.num_workers, 'pin_memory': pin_memory}
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = args.prefetch_factor
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    run = None
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    os.makedirs('checkpoints', exist_ok=True)


    # 4. Training Loop
    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device, non_blocking=pin_memory)
            targets = batch[data_key].to(device, non_blocking=pin_memory)

            # For localization, ensure targets are float
            if args.task == 'localization':
                targets = targets.float()
                sizes = batch['orig_size'].to(device, non_blocking=pin_memory)

            optimizer.zero_grad()
            outputs = model(images, sizes) if args.task == 'localization' else model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_time = time.perf_counter() - epoch_start
        epoch_metrics = {'train/loss': avg_loss, 'epoch/time_sec': epoch_time}

        if args.task == 'localization':
            test_loss = evaluate_localization_loss(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                pin_memory=pin_memory,
            )
            print(
                f"Epoch [{epoch+1}/{args.epochs}], Task: {args.task}, "
                f"Train Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Time: {epoch_time:.2f}s"
            )
            epoch_metrics = {
                'train/loss': avg_loss,
                'test/loss': test_loss,
            }
        elif args.task == 'segmentation':
            val_loss, pixel_acc, macro_dice = evaluate_segmentation(
                model=model,
                loader=test_loader,
                criterion=criterion,
                device=device,
                pin_memory=pin_memory,
                num_classes=3,
            )
            epoch_metrics['val/loss'] = val_loss
            epoch_metrics['val/pixel_accuracy'] = pixel_acc
            epoch_metrics['val/dice_macro'] = macro_dice

            print(
                f"Epoch [{epoch+1}/{args.epochs}], Task: {args.task}, "
                f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Pixel Acc: {pixel_acc:.4f}, Val Dice: {macro_dice:.4f}, Time: {epoch_time:.2f}s"
            )
        else:
            print(f"Epoch [{epoch+1}/{args.epochs}], Task: {args.task}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        if run is not None:
            wandb.log(epoch_metrics, step=epoch + 1)

        # Save Checkpoint
        torch.save(model.state_dict(), f"checkpoints/{args.task}.pth")

    if run is not None:
        wandb.finish()

if __name__ == "__main__":
    train()