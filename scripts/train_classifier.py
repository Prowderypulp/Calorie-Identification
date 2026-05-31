"""
Training script for the EfficientNet-B0 food classifier.

Fine-tunes a pre-trained EfficientNet-B0 on the Food-101 dataset
(or a custom dataset) with transfer learning.

Usage:
    python -m scripts.train_classifier --data_dir ./data/food-101 --epochs 20
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
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

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model = model.to(device)

    # Compile model for optimized GPU kernels (20-50% speedup)
    if device.type == "cuda":
        model = torch.compile(model)
        print("Model compiled with torch.compile")

    for param in model.features.parameters():
        param.requires_grad = False

    # Label smoothing prevents overconfidence on noisy 101-class data
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # AdamW decouples weight decay from gradients — standard for fine-tuning
    optimizer = optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine annealing with linear warmup during frozen-backbone phase
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=args.unfreeze_epoch)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.unfreeze_epoch)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[args.unfreeze_epoch])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    best_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grad_accum = args.grad_accum_steps

    for epoch in range(args.epochs):
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

        model.train()
        running_loss, correct, total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)

        for step, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels) / grad_accum
                scaler.scale(loss).backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels) / grad_accum
                loss.backward()
                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * grad_accum * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                else:
                    outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        lr_current = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} — Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}, LR: {lr_current:.2e}")

        if val_acc > best_acc:
            best_acc = val_acc
            # Unwrap torch.compile so the checkpoint loads into a plain efficientnet_b0
            state_dict = getattr(model, "_orig_mod", model).state_dict()
            torch.save(state_dict, output_dir / "classifier.pth")
            print(f"  Saved best model (val_acc={val_acc:.4f})")

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
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--unfreeze_epoch", type=int, default=3, help="Epoch to unfreeze backbone")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (use 0-2 on Colab)")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()
    train(args)
