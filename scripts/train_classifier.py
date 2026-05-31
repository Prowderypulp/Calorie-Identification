"""
Training script for the Food-101 classifier.

Uses ResNet-50 with torchvision IMAGENET1K_V2 weights (80.86% ImageNet top-1).
Tuned for Google Colab T4: fp16 AMP + channels_last + RandAugment.

Usage:
    python scripts/train_classifier.py --data_dir ./data --use_torchvision_dataset \
        --output_dir ./app/models --epochs 10 --batch_size 128 --num_workers 4
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
from torchvision.transforms import InterpolationMode


def get_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(232, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = build_model(num_classes).to(device, memory_format=torch.channels_last)

    # Discriminative LRs: head learns fast, backbone fine-tunes slowly. No frozen phase —
    # V2 weights are already strong; full end-to-end with proper LR converges faster.
    backbone_params = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("fc.")]
    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr * 0.1},
            {"params": head_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    warmup_iters = max(1, args.warmup_epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_iters))
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_iters])

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_acc = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

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
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = model(images)
                else:
                    outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        head_lr = optimizer.param_groups[1]["lr"]
        backbone_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{args.epochs} — Train acc: {train_acc:.4f}, "
            f"Val acc: {val_acc:.4f}, LR head/bb: {head_lr:.2e}/{backbone_lr:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_dir / "classifier.pth")
            print(f"  Saved best model (val_acc={val_acc:.4f})")

    with open(output_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)
    print(f"Training complete. Best val acc: {best_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-50 food classifier")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--use_torchvision_dataset", action="store_true")
    parser.add_argument("--output_dir", type=str, default="./app/models")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="Head LR; backbone is 0.1×")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    train(args)
