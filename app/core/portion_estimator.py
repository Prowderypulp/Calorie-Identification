"""
Portion size estimator.

Predicts the weight (in grams) of the detected food item from the
image features and the predicted class label.

When trained models are available, uses a regression head on top of
the shared EfficientNet feature extractor. Falls back to class-specific
default portions from the class_mapping.json.
"""

import json

import numpy as np
import structlog

from app.config import settings

logger = structlog.get_logger()

# Fallback default portion if class is unknown
DEFAULT_PORTION_G = 200.0


class PortionEstimator:
    """
    Estimates the portion weight (grams) of a food item.

    In production, this runs a small regression model.
    During development, it returns class-specific default portions
    from the class_mapping.json file.
    """

    def __init__(self) -> None:
        self._session = None
        self._defaults: dict[str, float] = {}
        self._backend = "defaults"

        self._load_defaults()

        if settings.portion_model_path.exists():
            self._load_onnx()

        logger.info("portion_estimator_init", backend=self._backend)

    def _load_defaults(self) -> None:
        """Load default portion sizes from class_mapping.json."""
        mapping_path = settings.class_mapping_path
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            for food, info in mapping.items():
                if isinstance(info, dict) and "default_portion_g" in info:
                    self._defaults[food] = info["default_portion_g"]

    def _load_onnx(self) -> None:
        """Load the ONNX regression model."""
        try:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = settings.ONNX_THREADS

            self._session = ort.InferenceSession(
                str(settings.portion_model_path),
                sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._backend = "onnx"
            logger.info(
                "portion_onnx_loaded", path=str(settings.portion_model_path)
            )
        except Exception as e:
            logger.error("portion_onnx_failed", error=str(e))

    def predict(self, tensor: np.ndarray, food_name: str) -> float:
        """
        Estimate portion weight for the given food image.

        Args:
            tensor: Preprocessed image tensor (1, 3, 224, 224).
            food_name: The predicted food class label.

        Returns:
            Estimated weight in grams.
        """
        if self._backend == "onnx" and self._session is not None:
            try:
                input_name = self._session.get_inputs()[0].name
                outputs = self._session.run(None, {input_name: tensor})
                weight = float(outputs[0][0])
                # Clamp to reasonable range
                return max(10.0, min(weight, 2000.0))
            except Exception as e:
                logger.error("portion_inference_failed", error=str(e))

        # Fallback: use class-specific default
        return self._defaults.get(food_name, DEFAULT_PORTION_G)
