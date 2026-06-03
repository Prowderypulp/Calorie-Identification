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
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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

    # DataLoader with persistent workers and prefetching for speed
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

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

    # Compile model for optimized GPU kernels (20-50% speedup)
    if device.type == "cuda":
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    # Freeze backbone for first few epochs
    for param in model.features.parameters():
        param.requires_grad = False

    # L1 Loss (MAE) is typically better for weight estimation than MSE
    criterion = nn.L1Loss()
    # AdamW decouples weight decay from gradients — standard for fine-tuning
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing with linear warmup during frozen-backbone phase
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.unfreeze_epoch)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.unfreeze_epoch)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.unfreeze_epoch])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_loss = float('inf')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grad_accum = args.grad_accum_steps

    for epoch in range(args.epochs):
        # Unfreeze backbone after warmup
        if epoch == args.unfreeze_epoch:
            print(f"Epoch {epoch}: Unfreezing backbone")
            for param in model.features.parameters():
                param.requires_grad = True
            # Rebuild optimizer with all params and lower LR for backbone
            optimizer = optim.AdamW([
                {"params": model.features.parameters(), "lr": args.lr * 0.1},
                {"params": model.classifier.parameters(), "lr": args.lr},
            ], weight_decay=args.weight_decay)
            # Fresh cosine schedule for remaining epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.unfreeze_epoch)

        # Train
        model.train()
        running_loss, total = 0.0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, weights) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            weights = weights.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, weights) / grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(images)
                loss = criterion(outputs, weights) / grad_accum
                loss.backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum * images.size(0)
            total += images.size(0)

        train_loss = running_loss / total
        scheduler.step()

        # Validate
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for images, weights in val_loader:
                images = images.to(device, non_blocking=True)
                weights = weights.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                loss = criterion(outputs, weights)
                val_loss += loss.item() * images.size(0)
                val_total += images.size(0)

        val_loss = val_loss / val_total
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} — Train MAE: {train_loss:.2f}g, Val MAE: {val_loss:.2f}g, LR: {lr_current:.2e}")

        if val_loss < best_loss:
            best_loss = val_loss
            state_dict = getattr(model, "_orig_mod", model).state_dict()
            torch.save(state_dict, output_dir / "portion_estimator.pth")
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
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--unfreeze_epoch", type=int, default=3, help="Epoch to unfreeze backbone")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (use 0-2 on Colab)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()
    train(args)
