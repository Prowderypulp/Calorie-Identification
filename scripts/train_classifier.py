"""
Training script for the EfficientNet-B0 food classifier.

Fine-tunes a pre-trained EfficientNet-B0 on the Food-101 dataset
(or a custom dataset) with transfer learning.

Usage:
    python -m scripts.train_classifier --data_dir ./data/food-101 --epochs 20
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def get_transforms(image_size: int = 224):
    """Training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, val_tf = get_transforms()

    if args.use_torchvision_dataset:
        print("Using torchvision Food101 dataset...")
        train_dataset = datasets.Food101(root=args.data_dir, split="train", transform=train_tf, download=False)
        val_dataset = datasets.Food101(root=args.data_dir, split="test", transform=val_tf, download=False)
        num_classes = 101
        class_names = train_dataset.classes
    else:
        train_dataset = datasets.ImageFolder(Path(args.data_dir) / "train", train_tf)
        val_dataset = datasets.ImageFolder(Path(args.data_dir) / "test", val_tf)
        num_classes = len(train_dataset.classes)
        class_names = train_dataset.classes

    print(f"Found {num_classes} classes, {len(train_dataset)} train / {len(val_dataset)} val images")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model = model.to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    best_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        if epoch == args.unfreeze_epoch:
            print(f"Epoch {epoch}: Unfreezing backbone")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}/{args.epochs} — Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "classifier.pth")
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    import json
    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"Training complete. Best val acc: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 food classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--use_torchvision_dataset", action="store_true", help="Use torchvision.datasets.Food101")
    parser.add_argument("--output_dir", type=str, default="./app/models", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze_epoch", type=int, default=3, help="Epoch to unfreeze backbone")
    args = parser.parse_args()
    train(args)
