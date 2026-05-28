"""
Training script for the EfficientNet-B0 portion estimator.

Fine-tunes a pre-trained EfficientNet-B0 (or uses the classifier's backbone)
on a dataset like Nutrition5k to predict food weight in grams.

Usage:
    python -m scripts.train_portion --train_csv ./data/train.csv --val_csv ./data/val.csv --img_dir ./data/images
"""

import argparse
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


class PortionDataset(Dataset):
    def __init__(self, csv_path: str, img_dir: str, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Expected columns: image_name, weight_g
        if "image_name" not in self.data.columns or "weight_g" not in self.data.columns:
            raise ValueError("CSV must contain 'image_name' and 'weight_g' columns")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = self.img_dir / row["image_name"]
        
        # Convert to RGB in case of grayscale or RGBA
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        weight = torch.tensor([float(row["weight_g"])], dtype=torch.float32)
        return image, weight


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
    
    train_dataset = PortionDataset(args.train_csv, args.img_dir, train_tf)
    val_dataset = PortionDataset(args.val_csv, args.img_dir, val_tf)

    print(f"Loaded {len(train_dataset)} train / {len(val_dataset)} val images")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Replace classifier with a regression head
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1280, 512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )
    
    model = model.to(device)

    # Freeze backbone for first few epochs
    for param in model.features.parameters():
        param.requires_grad = False

    # L1 Loss (MAE) is typically better for weight estimation than MSE
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Unfreeze backbone after warmup
        if epoch == args.unfreeze_epoch:
            print(f"Epoch {epoch}: Unfreezing backbone")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=args.lr * 0.1)

        # Train
        model.train()
        running_loss, total = 0.0, 0
        for images, weights in train_loader:
            images, weights = images.to(device), weights.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, weights)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, weights)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            total += images.size(0)

        train_loss = running_loss / total
        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for images, weights in val_loader:
                images, weights = images.to(device), weights.to(device)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                loss = criterion(outputs, weights)
                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)

        val_loss = val_loss / val_total
        print(f"Epoch {epoch+1}/{args.epochs} — Train MAE: {train_loss:.2f}g, Val MAE: {val_loss:.2f}g")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), output_dir / "portion_estimator.pth")
            print(f"  Saved best model (val_loss={val_loss:.2f}g)")

    print(f"Training complete. Best val MAE: {best_loss:.2f}g")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 portion estimator")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training metadata CSV")
    parser.add_argument("--val_csv", type=str, required=True, help="Path to validation metadata CSV")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--output_dir", type=str, default="./app/models", help="Where to save model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--unfreeze_epoch", type=int, default=3, help="Epoch to unfreeze backbone")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (use 0-2 on Colab)")
    args = parser.parse_args()
    train(args)
