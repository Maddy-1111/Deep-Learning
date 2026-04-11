import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from data.pets_dataset import OxfordIIITPetDataset
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses import IoULoss

def train():
    parser = argparse.ArgumentParser(description="Train VGG11 for different tasks")
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'localization', 'segmentation'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='./dataset')
    parser.add_argument('--pretrained-classifier', type=str, default=None)
    parser.add_argument('--fine-tune', type=str, default='full', choices=['strict', 'partial', 'full'])
    parser.add_argument('--batchnorm', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--prefetch-factor', type=int, default=2)
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
        model = VGG11Localizer(pretrained_path=args.pretrained_classifier, fine_tune=args.fine_tune).to(device)
        criterion = IoULoss() 
        data_key = 'bbox'
    elif args.task == 'segmentation':
        model = VGG11UNet(pretrained_path=args.pretrained_classifier, fine_tune=args.fine_tune).to(device)
        criterion = nn.CrossEntropyLoss()
        data_key = 'mask'

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

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        epoch_time = time.perf_counter() - epoch_start
        print(f"Epoch [{epoch+1}/{args.epochs}], Task: {args.task}, Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")

        # Save Checkpoint
        torch.save(model.state_dict(), f"checkpoints/{args.task}.pth")

if __name__ == "__main__":
    train()