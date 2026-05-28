"""Tests for the nutrition calculation module."""

import pytest
from app.core.nutrition import NutritionCalculator


@pytest.fixture
def calc():
    return NutritionCalculator()


def test_calculate_known_food(calc):
    """Nutrition lookup should return scaled values for a known food."""
    result = calc.calculate("pizza", 150)
    assert result["calories"] > 0
    assert "protein_g" in result
    assert "carbs_g" in result
    assert "fat_g" in result
    assert "fiber_g" in result


def test_calculate_unknown_food_returns_fallback(calc):
    """Unknown food should return the generic fallback profile."""
    result = calc.calculate("alien_food_xyz", 100)
    assert result["calories"] == 150.0  # generic fallback


def test_scaling(calc):
    """200g should give exactly 2x the nutrients of 100g."""
    r100 = calc.calculate("pizza", 100)
    r200 = calc.calculate("pizza", 200)
    assert abs(r200["calories"] - r100["calories"] * 2) < 0.2


def test_zero_weight(calc):
    """Zero weight should return zero nutrients."""
    result = calc.calculate("pizza", 0)
    assert result["calories"] == 0.0
