"""
Export a trained PyTorch classifier or portion estimator to ONNX format.

Usage:
    python -m scripts.export_onnx --model_path ./app/models/classifier.pth --model_type classifier --num_classes 101
    python -m scripts.export_onnx --model_path ./app/models/portion_estimator.pth --model_type portion
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def export(args):
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Skipping export: {model_path} does not exist (train this model first).")
        sys.exit(0)

    if args.model_type == "classifier":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    elif args.model_type == "portion":
        model = models.efficientnet_b0(weights=None)
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

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # dynamo=False forces the legacy TorchScript-based exporter. The newer dynamo
    # exporter (default in torch >=2.5) silently produces broken graphs for some
    # torchvision models — symptom is a sub-MB output file with no weights.
    torch.onnx.export(
        model, dummy, str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"Exported ONNX model to {output_path}")
    print(f"File size: {size_mb:.1f} MB")
    if size_mb < 1.0:
        print("WARNING: ONNX file is suspiciously small — weights may not have been embedded.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_type", choices=["classifier", "portion"], default="classifier")
    parser.add_argument("--output_path", default="./app/models/model.onnx")
    parser.add_argument("--num_classes", type=int, default=101)
    export(parser.parse_args())

