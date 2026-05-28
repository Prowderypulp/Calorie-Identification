"""
Build the local USDA nutrition SQLite database from the class_mapping.json.

This script creates a lightweight nutrition.db from the inline nutrient data
in class_mapping.json. When the full USDA FoodData Central CSV is available,
this script can be extended to ingest it.

Usage:
    python -m scripts.build_nutrition_db
"""

import json
import sqlite3
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "app" / "data"
MAPPING_FILE = DATA_DIR / "class_mapping.json"
DB_FILE = DATA_DIR / "nutrition.db"


def build_db() -> None:
    """Create the nutrition SQLite database from class_mapping.json."""

    if DB_FILE.exists():
        DB_FILE.unlink()
        print(f"Removed existing {DB_FILE}")

    conn = sqlite3.connect(str(DB_FILE))
    cursor = conn.cursor()

    # Create schema
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS foods (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            usda_id     INTEGER,
            calories    REAL,
            protein_g   REAL,
            carbs_g     REAL,
            fat_g       REAL,
            fiber_g     REAL
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_foods_name ON foods(name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_foods_usda ON foods(usda_id)")

    # Load mapping
    with open(MAPPING_FILE) as f:
        mapping = json.load(f)

    # Insert foods
    count = 0
    for food_name, info in mapping.items():
        nutrients = info.get("nutrients_per_100g", {})
        cursor.execute(
            "INSERT INTO foods (name, usda_id, calories, protein_g, carbs_g, fat_g, fiber_g) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                food_name,
                info.get("usda_id"),
                nutrients.get("calories", 0),
                nutrients.get("protein_g", 0),
                nutrients.get("carbs_g", 0),
                nutrients.get("fat_g", 0),
                nutrients.get("fiber_g", 0),
            ),
        )
        count += 1

    conn.commit()
    conn.close()
    print(f"Created {DB_FILE} with {count} food entries.")


if __name__ == "__main__":
    build_db()
