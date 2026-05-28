"""
Dietary recommendation engine.

Provides rule-based + content-based suggestions to help users make
healthier food choices based on the nutritional analysis of their meal.
"""

import structlog

logger = structlog.get_logger()

# Thresholds for triggering suggestions (per meal, rough guidelines)
HIGH_CALORIE_THRESHOLD = 600  # kcal
HIGH_FAT_THRESHOLD = 25  # grams
LOW_PROTEIN_THRESHOLD = 10  # grams
HIGH_CARB_THRESHOLD = 80  # grams

# Pre-defined healthier alternatives organized by concern
ALTERNATIVES = {
    "high_calorie": [
        {
            "food": "Grilled chicken salad",
            "calories": 250,
            "protein_g": 30,
        },
        {
            "food": "Steamed vegetables with rice",
            "calories": 280,
            "protein_g": 8,
        },
        {
            "food": "Lentil soup (dal)",
            "calories": 180,
            "protein_g": 12,
        },
    ],
    "high_fat": [
        {
            "food": "Grilled fish",
            "calories": 200,
            "protein_g": 25,
        },
        {
            "food": "Tandoori chicken (skinless)",
            "calories": 220,
            "protein_g": 28,
        },
    ],
    "low_protein": [
        {
            "food": "Paneer tikka",
            "calories": 260,
            "protein_g": 20,
        },
        {
            "food": "Egg white omelette",
            "calories": 120,
            "protein_g": 18,
        },
        {
            "food": "Greek yogurt",
            "calories": 100,
            "protein_g": 17,
        },
    ],
    "high_carb": [
        {
            "food": "Cauliflower rice bowl",
            "calories": 150,
            "protein_g": 5,
        },
        {
            "food": "Zucchini noodles with pesto",
            "calories": 200,
            "protein_g": 6,
        },
    ],
}


class Recommender:
    """
    Generates dietary suggestions based on the nutritional profile
    of the analyzed meal.

    Strategy:
    1. Rule-based: Check if any macro exceeds health thresholds.
    2. For each triggered rule, suggest a healthier alternative
       from the pre-defined alternatives database.
    """

    def suggest(
        self, food_name: str, nutrients: dict, max_suggestions: int = 3
    ) -> list[dict]:
        """
        Generate dietary suggestions.

        Args:
            food_name: The detected food class label.
            nutrients: Dict with calories, protein_g, carbs_g, fat_g, fiber_g.
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            List of suggestion dicts with keys: reason, food, calories, protein_g.
        """
        suggestions = []

        calories = nutrients.get("calories", 0)
        fat = nutrients.get("fat_g", 0)
        protein = nutrients.get("protein_g", 0)
        carbs = nutrients.get("carbs_g", 0)

        # Rule 1: High calorie meal
        if calories > HIGH_CALORIE_THRESHOLD:
            for alt in ALTERNATIVES["high_calorie"]:
                suggestions.append(
                    {
                        "reason": f"Your meal has {calories:.0f} kcal, which is high for a single meal. Consider a lighter option:",
                        **alt,
                    }
                )

        # Rule 2: High fat
        if fat > HIGH_FAT_THRESHOLD:
            for alt in ALTERNATIVES["high_fat"]:
                suggestions.append(
                    {
                        "reason": f"High fat content detected ({fat:.0f}g). Try a leaner alternative:",
                        **alt,
                    }
                )

        # Rule 3: Low protein
        if protein < LOW_PROTEIN_THRESHOLD and calories > 200:
            for alt in ALTERNATIVES["low_protein"]:
                suggestions.append(
                    {
                        "reason": f"Low protein ({protein:.0f}g) relative to calories. Boost protein with:",
                        **alt,
                    }
                )

        # Rule 4: High carbs
        if carbs > HIGH_CARB_THRESHOLD:
            for alt in ALTERNATIVES["high_carb"]:
                suggestions.append(
                    {
                        "reason": f"High carbohydrate content ({carbs:.0f}g). Consider a lower-carb option:",
                        **alt,
                    }
                )

        # Deduplicate and limit
        seen = set()
        unique = []
        for s in suggestions:
            if s["food"] not in seen:
                seen.add(s["food"])
                unique.append(s)

        return unique[:max_suggestions]
