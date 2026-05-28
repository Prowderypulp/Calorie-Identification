"""
Main analysis endpoint.

Accepts a food image, runs it through the full pipeline
(classify → estimate portion → lookup nutrition → recommend),
and returns the complete nutritional analysis.
"""

import time

import structlog
from fastapi import APIRouter, File, Request, UploadFile, HTTPException

from app.api.schemas import (
    AnalysisResponse,
    ErrorResponse,
    FoodItem,
    NutrientInfo,
    Suggestion,
)
from app.utils.image_utils import load_and_preprocess

logger = structlog.get_logger()

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
)
async def analyze_food(
    request: Request,
    file: UploadFile = File(..., description="Food image (JPEG, PNG, or WebP)"),
) -> AnalysisResponse:
    """
    Analyze a food image and return nutritional information.

    Pipeline:
    1. Validate and preprocess the uploaded image.
    2. Classify the food using EfficientNet-B0.
    3. Estimate portion size (weight in grams).
    4. Look up nutrition from the USDA database and scale by weight.
    5. Generate dietary suggestions.
    """
    start = time.perf_counter()

    # --- Input validation ---
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type: {file.content_type}. "
            f"Allowed: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({len(image_bytes)} bytes). Max: {MAX_FILE_SIZE}.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # --- Preprocess ---
    try:
        tensor = load_and_preprocess(image_bytes)
    except Exception as e:
        logger.error("preprocess_failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    # --- Classify ---
    classifier = request.app.state.classifier
    food_name, confidence = classifier.predict(tensor)
    logger.info("classified", food=food_name, confidence=round(confidence, 3))

    # --- Estimate portion ---
    portion_estimator = request.app.state.portion_estimator
    weight_g = portion_estimator.predict(tensor, food_name)
    logger.info("portion_estimated", food=food_name, weight_g=round(weight_g, 1))

    # --- Nutrition lookup ---
    nutrition_calc = request.app.state.nutrition_calculator
    nutrients = nutrition_calc.calculate(food_name, weight_g)

    food_item = FoodItem(
        food_name=food_name,
        confidence=round(confidence, 4),
        estimated_weight_g=round(weight_g, 1),
        nutrients=NutrientInfo(**nutrients),
    )

    # --- Recommendations ---
    recommender = request.app.state.recommender
    suggestions_raw = recommender.suggest(food_name, nutrients)
    suggestions = [Suggestion(**s) for s in suggestions_raw]

    elapsed_ms = (time.perf_counter() - start) * 1000

    return AnalysisResponse(
        food_items=[food_item],
        total_nutrients=food_item.nutrients,
        suggestions=suggestions,
        processing_time_ms=round(elapsed_ms, 1),
    )
