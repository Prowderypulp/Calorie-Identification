# NutriVision

AI-powered food image analysis and nutrition estimation API. Upload a photo of a
meal and NutriVision classifies the food, estimates the portion weight, looks up
its nutritional profile, and returns dietary suggestions — all over a single REST
endpoint.

It is built with **FastAPI** and runs vision models through **ONNX Runtime** on
CPU, with PyTorch fallbacks for development.

## Pipeline

Each uploaded image flows through four stages:

1. **Classify** — a ResNet-50 classifier predicts the food category
   (trained on Food-101, 101 classes).
2. **Estimate portion** — a regression model predicts the portion weight in
   grams (trained on Nutrition5k), falling back to class-specific defaults.
3. **Look up nutrition** — per-100g nutrient profiles are pulled from a local
   USDA SQLite database and scaled by the estimated weight.
4. **Recommend** — a rule-based engine flags high-calorie / high-fat /
   low-protein / high-carb meals and suggests healthier alternatives.

Each stage degrades gracefully: if a trained model or the database is missing,
the service falls back to defaults or stub predictions so the API stays up.

## Project layout

```
app/
  main.py              # FastAPI app: model loading, CORS, request timing
  config.py            # Settings (env-driven, NUTRIVISION_ prefix)
  api/
    schemas.py         # Pydantic request/response contracts
    routes/
      analyze.py       # POST /api/v1/analyze — the main pipeline
      health.py        # GET  /api/v1/health
  core/
    classifier.py      # Food classification (ONNX / PyTorch / stub)
    portion_estimator.py  # Portion weight regression
    nutrition.py       # USDA SQLite lookup + scaling
    recommender.py     # Rule-based dietary suggestions
  utils/               # Image preprocessing, logging
  models/              # Model weights (.onnx / .pth), class names
  data/                # nutrition.db, class_mapping.json
scripts/               # Dataset download, training, ONNX export, DB build
tests/                 # API and nutrition unit tests
```

## Getting started

### Requirements

- Python 3.11+

### Install

```bash
pip install -r requirements.txt
```

### Build the nutrition database

```bash
python -m scripts.build_nutrition_db
```

### Run the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive API docs are then available at `http://localhost:8000/docs`.

## API

Base path: `/api/v1`

### `GET /health`

Returns service status and whether the models are loaded.

```json
{ "status": "ok", "version": "0.1.0", "models_loaded": true }
```

### `POST /analyze`

Multipart upload of a single food image.

- **Field:** `file` — JPEG, PNG, or WebP, max 10 MB.

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "file=@meal.jpg"
```

Example response:

```json
{
  "success": true,
  "food_items": [
    {
      "food_name": "pizza",
      "confidence": 0.93,
      "estimated_weight_g": 280.0,
      "nutrients": {
        "calories": 745.0,
        "protein_g": 30.8,
        "carbs_g": 92.4,
        "fat_g": 28.0,
        "fiber_g": 5.6
      }
    }
  ],
  "total_nutrients": { "calories": 745.0, "protein_g": 30.8, "carbs_g": 92.4, "fat_g": 28.0, "fiber_g": 5.6 },
  "suggestions": [
    {
      "reason": "Your meal has 745 kcal, which is high for a single meal. Consider a lighter option:",
      "food": "Grilled chicken salad",
      "calories": 250,
      "protein_g": 30
    }
  ],
  "processing_time_ms": 84.2
}
```

Every response also includes an `X-Process-Time-Ms` header.

## Configuration

Settings are read from environment variables (prefix `NUTRIVISION_`) or a `.env`
file. Common options:

| Variable | Default | Description |
| --- | --- | --- |
| `NUTRIVISION_DEBUG` | `false` | Enable debug mode |
| `NUTRIVISION_CONFIDENCE_THRESHOLD` | `0.5` | Minimum classification confidence |
| `NUTRIVISION_IMAGE_SIZE` | `224` | Model input size |
| `NUTRIVISION_ONNX_THREADS` | `4` | ONNX Runtime intra-op threads |
| `NUTRIVISION_HOST` | `0.0.0.0` | Bind host |
| `NUTRIVISION_PORT` | `8000` | Bind port |

See `app/config.py` for the full list.

## Training

Model training and export are driven by the scripts in `scripts/`. The full
pipeline (train classifier → train portion estimator → export to ONNX) is wired
up in `run_training.sh`:

```bash
./run_training.sh
```

Individual steps:

- `scripts/download_food101.py` / `scripts/download_nutrition5k.py` — fetch datasets
- `scripts/train_classifier.py` — train the food classifier on Food-101
- `scripts/train_portion.py` — train the portion estimator on Nutrition5k
- `scripts/export_onnx.py` — export PyTorch checkpoints to ONNX
- `scripts/quantize.py` — quantize ONNX models for faster CPU inference
- `scripts/build_nutrition_db.py` — build the USDA nutrition SQLite database

## Docker

```bash
docker build -t nutrivision .
docker run -p 8000:8000 nutrivision
```

The image installs dependencies, builds the nutrition database, and serves the
API on port 8000.

## Testing

```bash
pytest
```
