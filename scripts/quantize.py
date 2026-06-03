"""
INT8 static quantization for ONNX models.

Usage:
    python -m scripts.quantize --model_path ./app/models/classifier.onnx --calib_dir ./data/food-101/test
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType


class FoodCalibrationReader(CalibrationDataReader):
    """Feeds calibration images to the quantizer."""

    def __init__(self, calib_dir: str, limit: int = 300):
        self.image_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            self.image_paths.extend(Path(calib_dir).rglob(ext))
        self.image_paths = self.image_paths[:limit]
        self.index = 0
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        img = Image.open(self.image_paths[self.index]).convert("RGB")
        img = img.resize((256, 256)).crop((16, 16, 240, 240))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        self.index += 1
        return {"input": arr}


def quantize(args):
    output_path = args.model_path.replace(".onnx", "_int8.onnx")
    reader = FoodCalibrationReader(args.calib_dir, limit=args.num_samples)
    quantize_static(
        args.model_path, output_path, reader,
        quant_format=QuantType.QInt8,
        per_channel=True,
    )
    orig = Path(args.model_path).stat().st_size / 1024 / 1024
    quant = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Quantized: {orig:.1f} MB → {quant:.1f} MB ({quant/orig*100:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./app/models/classifier.onnx")
    parser.add_argument("--calib_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=300)
    quantize(parser.parse_args())
