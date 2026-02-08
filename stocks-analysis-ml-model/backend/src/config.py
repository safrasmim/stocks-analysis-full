"""
Configuration Module

Central configuration for the Tadawul Stock Prediction System.

Defines:
- Directory paths
- Stock tickers
- Data schema
- Model and feature settings
- API metadata
"""

from pathlib import Path
from typing import Dict, Optional
from pydantic_settings import BaseSettings

# ─────────────────────────────────────────
# Directory structure
# ─────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, MODELS_DIR, SCRIPTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    log_level: str = "INFO"

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"


settings = Settings()

API_TITLE = "Tadawul Stock Movement Prediction API"
API_VERSION = "1.0.0"
API_DESCRIPTION = (
    "News-driven stock movement prediction for Tadawul using "
    "FinBERT sentiment and basic text features."
)

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# ─────────────────────────────────────────
# Tadawul tickers
# ─────────────────────────────────────────

TICKERS: Dict[str, Dict[str, str]] = {
    "1120": {"name": "Al Rajhi Bank", "sector": "Banking"},
    "2010": {"name": "SABIC", "sector": "Petrochemicals"},
    "7010": {"name": "STC", "sector": "Telecom"},
    "1150": {"name": "Alinma Bank", "sector": "Banking"},
    "4325": {"name": "MASAR", "sector": "Financial Services"},
}

TICKER_NAMES = {code: info["name"] for code, info in TICKERS.items()}

# ─────────────────────────────────────────
# Data schema
# ─────────────────────────────────────────

# Data schema
DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"
HEADLINE_COLUMN = "headline"
TEXT_COLUMN = "article_text"
LABEL_COLUMN = "target_direction"  # 0 = Down, 1 = Up
RETURN_COLUMN = "actual_return"
SOURCE_COLUMN = "source"
URL_COLUMN = "url"  # ← ADD THIS LINE


REQUIRED_COLUMNS = [DATE_COLUMN, TICKER_COLUMN, TEXT_COLUMN, LABEL_COLUMN]

# ─────────────────────────────────────────
# Model paths
# ─────────────────────────────────────────

RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"

# ─────────────────────────────────────────
# NLP / FinBERT configuration
# ─────────────────────────────────────────

FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
FINBERT_LABELS = ["negative", "neutral", "positive"]
FINBERT_MAX_LENGTH = 128
FINBERT_BATCH_SIZE = 16

# ─────────────────────────────────────────
# Feature names
# ─────────────────────────────────────────

TEXT_FEATURES = [
    "feat_length",
    "feat_word_count",
]

SENTIMENT_FEATURES = [
    "sent_negative",
    "sent_neutral",
    "sent_positive",
    "sent_compound",
]

ALL_FEATURE_COLUMNS = TEXT_FEATURES + SENTIMENT_FEATURES

# ─────────────────────────────────────────
# Train/validation/test split
# ─────────────────────────────────────────

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─────────────────────────────────────────
# Logging
# ─────────────────────────────────────────

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "app.log"
LOG_LEVEL = settings.log_level


# ─────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────

def get_data_path(filename: str) -> Path:
    """Return full path for a data file."""
    return DATA_DIR / filename


def validate_ticker(ticker: str) -> bool:
    """Return True if ticker is known."""
    return ticker in TICKERS


def get_ticker_info(ticker: str) -> Optional[Dict]:
    """Return descriptive info for a ticker (or None)."""
    return TICKERS.get(ticker)
