"""
Image preprocessing utilities.

Handles loading raw bytes into a normalized tensor suitable for
EfficientNet-B0 inference (224×224, ImageNet normalization).
"""

import io

import numpy as np
from PIL import Image

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TARGET_SIZE = 224


def load_and_preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw image bytes to a preprocessed numpy tensor.

    Steps:
        1. Decode image bytes → PIL Image (RGB).
        2. Resize to 256×256 (preserving aspect is not needed since
           EfficientNet expects a square crop).
        3. Center-crop to 224×224.
        4. Convert to float32 [0, 1].
        5. Normalize with ImageNet mean/std.
        6. Transpose to CHW and add batch dimension → (1, 3, 224, 224).

    Args:
        image_bytes: Raw bytes of a JPEG/PNG/WebP image.

    Returns:
        Preprocessed tensor of shape (1, 3, 224, 224) as np.float32.
    """
    # Decode
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Match the training val transform: Resize(232) short-side, then CenterCrop(224).
    # This is the torchvision ResNet50_Weights.IMAGENET1K_V2 recipe.
    resize_short = 232
    w, h = img.size
    if w < h:
        new_w, new_h = resize_short, int(round(h * resize_short / w))
    else:
        new_h, new_w = resize_short, int(round(w * resize_short / h))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    left = (new_w - TARGET_SIZE) // 2
    top = (new_h - TARGET_SIZE) // 2
    img = img.crop((left, top, left + TARGET_SIZE, top + TARGET_SIZE))

    # To numpy float32 [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Normalize (per-channel)
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD

    # HWC → CHW, add batch dimension
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr
