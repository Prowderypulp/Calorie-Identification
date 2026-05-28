"""
FastAPI application entry point.

Creates the app, registers routers, and configures startup/shutdown events
for loading ML models into memory once.
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.api.routes import analyze, health
from app.core.classifier import FoodClassifier
from app.core.portion_estimator import PortionEstimator
from app.core.nutrition import NutritionCalculator
from app.core.recommender import Recommender

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    logger.info("startup", msg="Loading models...")

    app.state.classifier = FoodClassifier()
    app.state.portion_estimator = PortionEstimator()
    app.state.nutrition_calculator = NutritionCalculator()
    app.state.recommender = Recommender()

    logger.info("startup", msg="All models loaded successfully.")
    yield

    logger.info("shutdown", msg="Releasing resources.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered food image analysis and nutrition estimation API.",
    lifespan=lifespan,
)

# --- CORS (allow Flutter app from any origin during development) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request timing middleware ---
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        elapsed_ms=round(elapsed_ms, 1),
    )
    return response


# --- Register routers ---
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(analyze.router, prefix="/api/v1", tags=["analysis"])
