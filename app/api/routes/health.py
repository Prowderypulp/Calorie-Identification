"""
Health check endpoint.

Used by the Flutter app and infrastructure to verify the service is alive
and models are loaded.
"""

from fastapi import APIRouter, Request

from app.api.schemas import HealthResponse
from app.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return service health status and whether models are loaded."""
    models_loaded = (
        hasattr(request.app.state, "classifier")
        and request.app.state.classifier is not None
    )
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        models_loaded=models_loaded,
    )
