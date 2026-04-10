import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import TO_TENSOR_V2
import wandb

from pets_dataset import OxfordIIITPetDataset
from classification import VGG11Classifier

def train():
    # Initialize W&B
    # wandb.init(project="da6401_assignment_2", name="task1_classification")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define Transforms
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        TO_TENSOR_V2(),
    ])

    # Data Loaders
    train_dataset = OxfordIIITPetDataset(root_dir='./data', split='trainval', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Model, Loss, Optimizer
    model = VGG11Classifier(num_classes=37, dropout_p=0.5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training Loop
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
        wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc})

    wandb.finish()

if __name__ == "__main__":
    train()