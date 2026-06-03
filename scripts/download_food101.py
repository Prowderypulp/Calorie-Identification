"""
Download and extract the Food-101 dataset.
"""

import torchvision.datasets as datasets
from pathlib import Path

def download():
    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading Food-101 dataset (this may take a while)...")
    
    # Download training and test splits
    train_dataset = datasets.Food101(root=str(data_dir), split="train", download=True)
    test_dataset = datasets.Food101(root=str(data_dir), split="test", download=True)
    
    print(f"Food-101 downloaded successfully to {data_dir / 'food-101'}!")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

if __name__ == "__main__":
    download()
