"""
Food image classifier using EfficientNet-B0.

Supports two inference backends:
  1. ONNX Runtime (production) — fast, quantized, CPU-optimized.
  2. PyTorch (fallback/dev) — used when ONNX model is not yet available.

The module loads the appropriate backend based on what model files exist.
"""

import json
from pathlib import Path

import numpy as np
import structlog

from app.config import settings

logger = structlog.get_logger()


class FoodClassifier:
    """
    Classifies a food image into one of the known food categories.

    On initialization, attempts to load an ONNX model. If not found,
    falls back to a PyTorch model. If neither exists, operates in
    stub mode (returns a placeholder prediction for development).
    """

    def __init__(self) -> None:
        self.class_names: list[str] = []
        self._session = None
        self._model = None
        self._backend = "stub"

        # Load class names from mapping
        self._load_class_names()

        # Try ONNX first, then PyTorch, then stub
        if settings.classifier_path.exists():
            self._load_onnx()
        else:
            self._try_load_pytorch()

        logger.info(
            "classifier_init",
            backend=self._backend,
            num_classes=len(self.class_names),
        )

    def _load_class_names(self) -> None:
        """Load class names from the class_mapping.json file."""
        mapping_path = settings.class_mapping_path
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            self.class_names = list(mapping.keys())
        else:
            logger.warning(
                "class_mapping_missing",
                path=str(mapping_path),
                msg="Using empty class list. Create class_mapping.json.",
            )

    def _load_onnx(self) -> None:
        """Load the ONNX model for inference."""
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = settings.ONNX_THREADS
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            self._session = ort.InferenceSession(
                str(settings.classifier_path),
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._backend = "onnx"
            logger.info("classifier_onnx_loaded", path=str(settings.classifier_path))
        except Exception as e:
            logger.error("classifier_onnx_failed", error=str(e))
            self._try_load_pytorch()

    def _try_load_pytorch(self) -> None:
        """Try to load a PyTorch model as fallback."""
        pth_path = settings.MODEL_DIR / "classifier.pth"
        if pth_path.exists():
            try:
                import torch
                import torchvision.models as models

                model = models.efficientnet_b0(weights=None)
                # Replace classifier head for our number of classes
                if self.class_names:
                    import torch.nn as nn

                    model.classifier[1] = nn.Linear(1280, len(self.class_names))

                model.load_state_dict(torch.load(pth_path, map_location="cpu"))
                model.eval()
                self._model = model
                self._backend = "pytorch"
                logger.info("classifier_pytorch_loaded", path=str(pth_path))
            except Exception as e:
                logger.error("classifier_pytorch_failed", error=str(e))
                self._backend = "stub"
        else:
            logger.warning(
                "classifier_no_model",
                msg="No model found. Running in stub mode.",
            )
            self._backend = "stub"

    def predict(self, tensor: np.ndarray) -> tuple[str, float]:
        """
        Classify a preprocessed food image tensor.

        Args:
            tensor: Preprocessed image tensor of shape (1, 3, 224, 224).

        Returns:
            Tuple of (predicted_class_name, confidence_score).
        """
        if self._backend == "onnx":
            return self._predict_onnx(tensor)
        elif self._backend == "pytorch":
            return self._predict_pytorch(tensor)
        else:
            return self._predict_stub(tensor)

    def _predict_onnx(self, tensor: np.ndarray) -> tuple[str, float]:
        """Run inference via ONNX Runtime."""
        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: tensor})
        logits = outputs[0][0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()

        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        class_name = self.class_names[idx] if idx < len(self.class_names) else "unknown"
        return class_name, confidence

    def _predict_pytorch(self, tensor: np.ndarray) -> tuple[str, float]:
        """Run inference via PyTorch."""
        import torch

        with torch.no_grad():
            t = torch.from_numpy(tensor)
            logits = self._model(t)
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, idx = torch.max(probs, dim=1)

        idx = int(idx.item())
        class_name = self.class_names[idx] if idx < len(self.class_names) else "unknown"
        return class_name, float(confidence.item())

    def _predict_stub(self, tensor: np.ndarray) -> tuple[str, float]:
        """Return a placeholder prediction for development."""
        logger.warning("classifier_stub", msg="Using stub prediction.")
        return "unknown_food", 0.0
