from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_ATHLETES_CSV = RAW_DIR / "athlete_events.csv"
RAW_NOC_CSV = RAW_DIR / "noc_regions.csv"  # optional for nicer country names later

PROCESSED_CSV = PROCESSED_DIR / "clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
