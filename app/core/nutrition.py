"""
Nutrition calculator.

Looks up per-100g nutrient profiles from the local USDA SQLite database,
scales by the estimated portion weight, and returns the nutritional breakdown.
"""

import json
import sqlite3
from pathlib import Path

import structlog

from app.config import settings

logger = structlog.get_logger()


class NutritionCalculator:
    """
    Calculates nutritional values for a food item given its class name
    and estimated portion weight.

    The lookup strategy is:
    1. Check class_mapping.json for a direct USDA food ID mapping.
    2. Query the SQLite database for the nutrient profile.
    3. Scale nutrients by (portion_weight / 100).

    Falls back to a hardcoded approximate profile if the food is not
    found in the database.
    """

    def __init__(self) -> None:
        self._db_path = settings.nutrition_db_path
        self._class_mapping: dict = {}
        self._db_available = False

        # Load class → USDA mapping
        if settings.class_mapping_path.exists():
            with open(settings.class_mapping_path) as f:
                self._class_mapping = json.load(f)

        # Verify DB exists
        if self._db_path.exists():
            self._db_available = True
            logger.info("nutrition_db_loaded", path=str(self._db_path))
        else:
            logger.warning(
                "nutrition_db_missing",
                path=str(self._db_path),
                msg="Run scripts/build_nutrition_db.py to create the database.",
            )

    def _get_connection(self) -> sqlite3.Connection:
        """Get a read-only SQLite connection."""
        conn = sqlite3.connect(
            f"file:{self._db_path}?mode=ro", uri=True, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row
        return conn

    def calculate(self, food_name: str, weight_g: float) -> dict:
        """
        Calculate nutritional values for a food item.

        Args:
            food_name: The classified food label.
            weight_g: Estimated portion weight in grams.

        Returns:
            Dict with keys: calories, protein_g, carbs_g, fat_g, fiber_g.
        """
        # Get per-100g profile
        per_100g = self._lookup(food_name)

        # Scale by actual portion
        scale = weight_g / 100.0
        return {
            "calories": round(per_100g["calories"] * scale, 1),
            "protein_g": round(per_100g["protein_g"] * scale, 1),
            "carbs_g": round(per_100g["carbs_g"] * scale, 1),
            "fat_g": round(per_100g["fat_g"] * scale, 1),
            "fiber_g": round(per_100g["fiber_g"] * scale, 1),
        }

    def _lookup(self, food_name: str) -> dict:
        """
        Look up the per-100g nutrient profile for a food.

        Tries USDA ID from mapping first, then name search in DB,
        then falls back to the mapping's inline nutrients, and
        finally to a generic default.
        """
        mapping_entry = self._class_mapping.get(food_name, {})

        # Strategy 1: USDA ID lookup
        if self._db_available and "usda_id" in mapping_entry:
            result = self._query_by_id(mapping_entry["usda_id"])
            if result:
                return result

        # Strategy 2: Name search in DB
        if self._db_available:
            result = self._query_by_name(food_name)
            if result:
                return result

        # Strategy 3: Inline nutrients from mapping
        if "nutrients_per_100g" in mapping_entry:
            return mapping_entry["nutrients_per_100g"]

        # Strategy 4: Generic fallback
        logger.warning(
            "nutrition_fallback",
            food=food_name,
            msg="No nutrition data found. Using generic estimate.",
        )
        return {
            "calories": 150.0,
            "protein_g": 5.0,
            "carbs_g": 20.0,
            "fat_g": 6.0,
            "fiber_g": 2.0,
        }

    def _query_by_id(self, usda_id: int) -> dict | None:
        """Query the nutrition DB by USDA food ID."""
        try:
            conn = self._get_connection()
            cursor = conn.execute(
                "SELECT calories, protein_g, carbs_g, fat_g, fiber_g "
                "FROM foods WHERE id = ?",
                (usda_id,),
            )
            row = cursor.fetchone()
            conn.close()
            if row:
                return dict(row)
        except Exception as e:
            logger.error("nutrition_query_failed", error=str(e))
        return None

    def _query_by_name(self, food_name: str) -> dict | None:
        """Query the nutrition DB by food name (fuzzy LIKE match)."""
        try:
            conn = self._get_connection()
            # Try exact match first, then LIKE
            cursor = conn.execute(
                "SELECT calories, protein_g, carbs_g, fat_g, fiber_g "
                "FROM foods WHERE LOWER(name) = LOWER(?) LIMIT 1",
                (food_name,),
            )
            row = cursor.fetchone()
            if not row:
                cursor = conn.execute(
                    "SELECT calories, protein_g, carbs_g, fat_g, fiber_g "
                    "FROM foods WHERE LOWER(name) LIKE ? LIMIT 1",
                    (f"%{food_name.lower()}%",),
                )
                row = cursor.fetchone()
            conn.close()
            if row:
                return dict(row)
        except Exception as e:
            logger.error("nutrition_name_query_failed", error=str(e))
        return None
