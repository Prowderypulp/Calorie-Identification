"""
Application configuration.

Uses pydantic-settings to load from environment variables with sensible defaults.
All paths are relative to the project root unless overridden.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the food analysis service."""

    model_config = SettingsConfigDict(
        env_prefix="NUTRIVISION_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    APP_NAME: str = "NutriVision"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # --- Paths ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_DIR: Path = BASE_DIR / "app" / "models"
    DATA_DIR: Path = BASE_DIR / "app" / "data"

    # --- Model files ---
    CLASSIFIER_MODEL: str = "classifier.onnx"
    PORTION_MODEL: str = "portion_estimator.onnx"

    # --- Database ---
    NUTRITION_DB: str = "nutrition.db"
    CLASS_MAPPING: str = "class_mapping.json"

    # --- Inference ---
    IMAGE_SIZE: int = 224
    CONFIDENCE_THRESHOLD: float = 0.5
    ONNX_THREADS: int = 4

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000


    @property
    def classifier_path(self) -> Path:
        return self.MODEL_DIR / self.CLASSIFIER_MODEL

    @property
    def portion_model_path(self) -> Path:
        return self.MODEL_DIR / self.PORTION_MODEL

    @property
    def nutrition_db_path(self) -> Path:
        return self.DATA_DIR / self.NUTRITION_DB

    @property
    def class_mapping_path(self) -> Path:
        return self.DATA_DIR / self.CLASS_MAPPING


settings = Settings()
