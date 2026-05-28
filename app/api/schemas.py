"""
Pydantic schemas for API request/response validation.

All API contracts are defined here so the Flutter team can auto-generate
client code from the OpenAPI spec.
"""

from pydantic import BaseModel, Field


# ---------- Response models ----------


class NutrientInfo(BaseModel):
    """Nutritional breakdown for a single food item."""

    calories: float = Field(..., description="Energy in kcal")
    protein_g: float = Field(..., description="Protein in grams")
    carbs_g: float = Field(..., description="Carbohydrates in grams")
    fat_g: float = Field(..., description="Total fat in grams")
    fiber_g: float = Field(..., description="Dietary fiber in grams")


class FoodItem(BaseModel):
    """A single detected food item with its nutrition."""

    food_name: str = Field(..., description="Predicted food class label")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Classification confidence score"
    )
    estimated_weight_g: float = Field(
        ..., description="Estimated portion weight in grams"
    )
    nutrients: NutrientInfo


class Suggestion(BaseModel):
    """A dietary suggestion/recommendation."""

    reason: str = Field(..., description="Why this suggestion is made")
    food: str = Field(..., description="Suggested food item")
    calories: float
    protein_g: float


class AnalysisResponse(BaseModel):
    """Full response from the /analyze endpoint."""

    success: bool = True
    food_items: list[FoodItem] = Field(
        ..., description="Detected food items with nutritional info"
    )
    total_nutrients: NutrientInfo = Field(
        ..., description="Summed nutrients across all detected items"
    )
    suggestions: list[Suggestion] = Field(
        default_factory=list, description="Dietary recommendations"
    )
    processing_time_ms: float = Field(
        ..., description="Total processing time in milliseconds"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""

    success: bool = False
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str
    models_loaded: bool
