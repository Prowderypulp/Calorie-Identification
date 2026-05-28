"""
Export a trained PyTorch classifier or portion estimator to ONNX format.

Usage:
    python -m scripts.export_onnx --model_path ./app/models/classifier.pth --model_type classifier --num_classes 101
    python -m scripts.export_onnx --model_path ./app/models/portion_estimator.pth --model_type portion
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def export(args):
    model = models.efficientnet_b0(weights=None)
    
    if args.model_type == "classifier":
        model.classifier[1] = nn.Linear(1280, args.num_classes)
    elif args.model_type == "portion":
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX model to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", choices=["classifier", "portion"], default="classifier")
    parser.add_argument("--output_path", default="./app/models/model.onnx")
    parser.add_argument("--num_classes", type=int, default=101)
    export(parser.parse_args())

