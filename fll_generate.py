#!/usr/bin/env python3
"""
TADAWUL STOCK PREDICTION SYSTEM - COMPLETE PROJECT GENERATOR

MSc Thesis Project: News-Driven Stock Movement Prediction
Target Market: Tadawul (Saudi Stock Exchange)
Stocks: 1120 (Al Rajhi), 2010 (SABIC), 7010 (STC), 1150 (Alinma), 4325 (MASAR)

This script generates the COMPLETE working project with:
- Backend (FastAPI + ML/DL models)
- Sample data generator
- Model training script
- README

USAGE:
1. Save this file as: generate_project.py
2. Open PowerShell, cd to the folder containing it
3. Run: python generate_project.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime


def print_header(text: str) -> None:
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_section(text: str) -> None:
    """Print section header"""
    print(f"\n{'─' * 80}")
    print(f"📌 {text}")
    print("─" * 80)


def create_file(path: str | Path, content: str) -> None:
    """Create a file with content, creating parent directories as needed"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ {path}")


def main() -> None:
    start_time = datetime.now()

    print_header("TADAWUL STOCK PREDICTION SYSTEM")
    print(f"🕐 Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Working directory: {Path.cwd()}")

    # Create base project directory
    base_dir = Path.cwd() / "stocks-analysis-ml-model"
    if base_dir.exists():
        response = input(f"\n⚠️  Directory '{base_dir.name}' already exists. Overwrite? (y/n): ")
        if response.lower() != "y":
            print("❌ Cancelled by user")
            sys.exit(0)

    base_dir.mkdir(exist_ok=True)
    os.chdir(base_dir)
    print(f"\n✅ Created project directory: {base_dir}")

    # ======================================================================
    # BACKEND CONFIG & CORE
    # ======================================================================
    print_section("BACKEND CONFIGURATION FILES")

    # requirements.txt
    create_file(
        "backend/requirements.txt",
        """# Core Framework
fastapi>=0.100.0
uvicorn[standard]>=0.23.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
python-dateutil>=2.8.2
openpyxl>=3.1.0

# Machine Learning
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0

# Deep Learning
torch>=2.0.0

# NLP
transformers>=4.30.0
sentencepiece>=0.1.99
gensim>=4.3.0
nltk>=3.8.1

# Web Scraping
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=4.9.0

# Financial Data
yfinance>=0.2.28

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Utilities
tqdm>=4.65.0
"""
    )

    # .env template
    create_file(
        "backend/.env.example",
        """# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Model Configuration
MODEL_TYPE=random_forest
USE_PRETRAINED=True

# Data Configuration
DATA_SOURCE=local
AUTO_UPDATE_DATA=False

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
"""
    )

    # config.py
    create_file(
        "backend/src/config.py",
        '''"""
Configuration Module

Central configuration for the Tadawul Stock Prediction System.
"""

from pathlib import Path
from typing import Dict, Optional
from pydantic_settings import BaseSettings

# Directory structure
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"

for directory in [DATA_DIR, MODELS_DIR, SCRIPTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
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
API_DESCRIPTION = "News-driven stock movement prediction for Tadawul."

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

TICKERS: Dict[str, Dict[str, str]] = {
    "1120": {"name": "Al Rajhi Bank", "sector": "Banking"},
    "2010": {"name": "SABIC", "sector": "Petrochemicals"},
    "7010": {"name": "STC", "sector": "Telecom"},
    "1150": {"name": "Alinma Bank", "sector": "Banking"},
    "4325": {"name": "MASAR", "sector": "Financial Services"},
}

TICKER_NAMES = {code: info["name"] for code, info in TICKERS.items()}

DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"
HEADLINE_COLUMN = "headline"
TEXT_COLUMN = "article_text"
LABEL_COLUMN = "target_direction"
RETURN_COLUMN = "actual_return"

RF_MODEL_PATH = MODELS_DIR / "random_forest.joblib"
SCALER_PATH = MODELS_DIR / "feature_scaler.joblib"

FINBERT_MODEL_NAME = "yiyanghkust/finbert-tone"
FINBERT_LABELS = ["negative", "neutral", "positive"]

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "app.log"
LOG_LEVEL = settings.log_level


def get_data_path(filename: str) -> Path:
    return DATA_DIR / filename


def validate_ticker(ticker: str) -> bool:
    return ticker in TICKERS


def get_ticker_info(ticker: str) -> Optional[Dict]:
    return TICKERS.get(ticker)
'''
    )

    # ======================================================================
    # DATA LOADER, PREPROCESSING, FEATURES, TRAINER, PREDICTOR, APP
    # ======================================================================

    print_section("BACKEND CORE MODULES")

    # data_loader.py
    create_file(
        "backend/src/data_loader.py",
        '''"""
Data Loader

Loads news data from CSV for the Tadawul prediction system.
"""
import pandas as pd
from pathlib import Path
from .config import DATA_DIR, DATE_COLUMN, TICKER_COLUMN

class DataLoader:
    def __init__(self, filename: str = "raw_news_sample.csv") -> None:
        self.path = DATA_DIR / filename

    def load_raw_news(self) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")
        df = pd.read_csv(self.path)
        if DATE_COLUMN in df.columns:
            df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        return df

    def filter_by_ticker(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        return df[df[TICKER_COLUMN] == ticker].copy()
'''
    )

    # preprocessing.py
    create_file(
        "backend/src/preprocessing.py",
        '''"""
Preprocessing

Cleans article text: lowercasing, basic character filtering.
"""
import re
import pandas as pd
from .config import TEXT_COLUMN

CLEAN_RE = re.compile(r"[^\\w\\s.,!?]+")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = CLEAN_RE.sub(" ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TEXT_COLUMN in df.columns:
        df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(clean_text)
    return df
'''
    )

    # features.py
    create_file(
        "backend/src/features.py",
        '''"""
Feature Extraction

For demo: simple text-length based features (no heavy FinBERT).
"""
import numpy as np
import pandas as pd
from .config import TEXT_COLUMN

FEATURE_COLS = [
    "feat_length",
    "feat_word_count",
]

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    texts = df[TEXT_COLUMN].fillna("").astype(str)
    df["feat_length"] = texts.apply(len)
    df["feat_word_count"] = texts.apply(lambda t: len(t.split()))
    return df

def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLS].values
'''
    )

    # model_trainer.py
    create_file(
        "backend/src/model_trainer.py",
        '''"""
Model Trainer

Trains a simple RandomForestClassifier on extracted features.
"""
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from .config import (
    LABEL_COLUMN,
    RF_MODEL_PATH,
    SCALER_PATH,
    TRAIN_RATIO,
    VAL_RATIO,
)

from .features import get_feature_matrix

class ModelTrainer:
    def __init__(self) -> None:
        self.model_path: Path = RF_MODEL_PATH
        self.scaler_path: Path = SCALER_PATH

    def _split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        test_size = 1.0 - TRAIN_RATIO
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        return X_train, X_test, y_train, y_test

    def train_all(self, df: pd.DataFrame) -> None:
        df = df.dropna(subset=[LABEL_COLUMN])
        X = get_feature_matrix(df)
        y = df[LABEL_COLUMN].astype(int).values

        X_train, X_test, y_train, y_test = self._split_data(X, y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_scaled, y_train)

        acc = clf.score(X_test_scaled, y_test)
        print(f"Validation accuracy: {acc:.3f}")

        joblib.dump(clf, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        print(f"Saved model to: {self.model_path}")
        print(f"Saved scaler to: {self.scaler_path}")
'''
    )

    # predictor.py
    create_file(
        "backend/src/predictor.py",
        '''"""
Predictor

Loads trained model and makes predictions from raw texts.
"""
from pathlib import Path
from typing import List, Dict
import numpy as np
import joblib
import pandas as pd

from .config import TEXT_COLUMN, RF_MODEL_PATH, SCALER_PATH
from .preprocessing import preprocess_dataframe
from .features import extract_features, get_feature_matrix

class Predictor:
    def __init__(
        self,
        model_path: Path | str = RF_MODEL_PATH,
        scaler_path: Path | str = SCALER_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict_from_texts(self, texts: List[str]) -> Dict:
        df = pd.DataFrame({TEXT_COLUMN: texts})
        df = preprocess_dataframe(df)
        df = extract_features(df)
        X = get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            # assume class 1 is "Up"
            classes = list(self.model.classes_)
            if 1 in classes:
                up_idx = classes.index(1)
                p_up = probs[:, up_idx]
            else:
                p_up = probs[:, 0]
        else:
            p_up = np.zeros(len(texts))

        preds = (p_up >= 0.5).astype(int)
        labels = ["Up" if p == 1 else "Down" for p in preds]
        return {
            "predictions": preds.tolist(),
            "labels": labels,
            "probabilities_up": p_up.tolist(),
        }
'''
    )

    # app.py (FastAPI)
    create_file(
        "backend/src/app.py",
        '''"""
FastAPI Application

Main API for Tadawul news-based stock movement prediction.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from .config import (
    API_TITLE,
    API_VERSION,
    API_DESCRIPTION,
    CORS_ORIGINS,
    TICKERS,
)
from .predictor import Predictor

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: Predictor | None = None

class PredictRequest(BaseModel):
    ticker: str
    texts: List[str]

class PredictResponse(BaseModel):
    ticker: str
    labels: List[str]
    predictions: List[int]
    probabilities_up: List[float]


@app.on_event("startup")
async def startup_event():
    global predictor
    predictor = Predictor()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/tickers")
async def list_tickers():
    return TICKERS


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if req.ticker not in TICKERS:
        raise HTTPException(status_code=400, detail="Invalid ticker")
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    result = predictor.predict_from_texts(req.texts)
    return PredictResponse(
        ticker=req.ticker,
        labels=result["labels"],
        predictions=result["predictions"],
        probabilities_up=result["probabilities_up"],
    )
'''
    )


    # ======================================================================
    # SCRIPTS & README
    # ======================================================================

    print_section("UTILITY SCRIPTS")

    # generate_sample_data.py
    create_file(
        "backend/scripts/generate_sample_data.py",
        '''"""
Sample Data Generator

Generates synthetic financial news data for testing and demonstration.
Run: python scripts/generate_sample_data.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_DIR, TICKERS, TICKER_NAMES,
    DATE_COLUMN, TICKER_COLUMN, HEADLINE_COLUMN,
    TEXT_COLUMN, LABEL_COLUMN, RETURN_COLUMN,
)

SAMPLE_DATA_CONFIG = {
    "num_articles_per_stock": 100,
    "date_range_days": 120,
    "positive_ratio": 0.55,
    "start_date": "2024-08-01",
    "include_weekends": False,
    "add_noise": True,
}

POSITIVE_HEADLINES = [
    "{company} reports strong quarterly earnings beating expectations",
    "{company} announces major expansion plans in Saudi market",
    "{company} stock rises on positive investor sentiment",
]

NEGATIVE_HEADLINES = [
    "{company} faces challenges amid market volatility",
    "{company} reports lower than expected quarterly results",
    "{company} stock declines on profit warning",
]


def generate_article_text(headline: str, label: int) -> str:
    intro = f"{headline}. "
    if label == 1:
        body = (
            "The company demonstrated strong performance across key metrics. "
            "Analysts view this development positively and expect continued growth."
        )
    else:
        body = (
            "The company faces several headwinds, and analysts remain cautious "
            "about short-term performance and profitability."
        )
    return intro + body


def generate_sample_data() -> pd.DataFrame:
    print("=" * 80)
    print("GENERATING SAMPLE DATA")
    print("=" * 80)

    cfg = SAMPLE_DATA_CONFIG
    start_date = datetime.strptime(cfg["start_date"], "%Y-%m-%d")
    end_date = start_date + timedelta(days=cfg["date_range_days"])

    all_rows = []

    for ticker, info in TICKERS.items():
        company = info["name"]
        print(f"\nGenerating data for {ticker} - {company}...")

        dates = pd.date_range(
            start=start_date,
            end=end_date,
            periods=cfg["num_articles_per_stock"],
        )

        for date in dates:
            if not cfg["include_weekends"] and date.weekday() >= 5:
                continue

            is_pos = np.random.rand() < cfg["positive_ratio"]
            label = 1 if is_pos else 0
            template = (
                np.random.choice(POSITIVE_HEADLINES)
                if is_pos
                else np.random.choice(NEGATIVE_HEADLINES)
            )
            headline = template.format(company=company)
            text = generate_article_text(headline, label)

            if is_pos:
                ret = np.random.uniform(0.5, 3.0)
            else:
                ret = np.random.uniform(-3.0, -0.5)
            if cfg["add_noise"]:
                ret += np.random.normal(0, 0.5)

            all_rows.append(
                {
                    DATE_COLUMN: date.strftime("%Y-%m-%d"),
                    TICKER_COLUMN: ticker,
                    HEADLINE_COLUMN: headline,
                    TEXT_COLUMN: text,
                    LABEL_COLUMN: label,
                    RETURN_COLUMN: round(ret, 2),
                }
            )

    df = pd.DataFrame(all_rows).sort_values(DATE_COLUMN)
    out_path = DATA_DIR / "raw_news_sample.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nDATA GENERATION COMPLETE")
    print(f"Saved to: {out_path}")
    print(f"Total rows: {len(df)}")
    return df


if __name__ == "__main__":
    generate_sample_data()
'''
    )

    # train_models.py
    create_file(
        "backend/scripts/train_models.py",
        '''"""
Training Pipeline

Run: python scripts/train_models.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import preprocess_dataframe
from src.features import extract_features
from src.model_trainer import ModelTrainer


def main() -> None:
    print("=" * 80)
    print("TADAWUL STOCK PREDICTION - TRAINING PIPELINE")
    print("=" * 80)

    print("\n[1/4] Loading data...")
    loader = DataLoader()
    df = loader.load_raw_news()

    print("\n[2/4] Preprocessing text...")
    df = preprocess_dataframe(df)

    print("\n[3/4] Extracting features...")
    df = extract_features(df)

    print("\n[4/4] Training model...")
    trainer = ModelTrainer()
    trainer.train_all(df)


if __name__ == "__main__":
    main()
'''
    )

    print_section("DOCUMENTATION")

    # README.md
    create_file(
        "README.md",
        """# Tadawul Stock Movement Prediction System

MSc Thesis Project – News-Driven Stock Price Movement Prediction

This project predicts Up/Down movements for selected Tadawul stocks based on
synthetic financial news data.

## Tech Stack

- Backend: Python, FastAPI, scikit-learn
- Models: Random Forest (classification)
- Data: CSV (generated synthetic news)
- Target stocks: 1120, 2010, 7010, 1150, 4325

## Project Structure

stocks-analysis-ml-model/
├── backend/
│   ├── src/
│   │   ├── config.py
│   │   ├── data_loader.py
│   │   ├── preprocessing.py
│   │   ├── features.py
│   │   ├── model_trainer.py
│   │   ├── predictor.py
│   │   └── app.py
│   ├── scripts/
│   │   ├── generate_sample_data.py
│   │   └── train_models.py
│   ├── data/
│   ├── models/
│   └── requirements.txt

## How to Run (Windows)

1. Open PowerShell and go to the project folder:

   cd stocks-analysis-ml-model\\backend

2. Create and activate virtual environment:

   python -m venv venv
   .\\venv\\Scripts\\activate

3. Install dependencies:

   pip install -r requirements.txt

4. Generate sample data:

   python scripts\\generate_sample_data.py

5. Train model:

   python scripts\\train_models.py

6. Start API:

   uvicorn src.app:app --reload --port 8000

7. Open in browser:

   http://localhost:8000/docs

Use the `/predict` endpoint with a ticker and list of news texts to get
predicted Up/Down labels and probabilities.
"""
    )

    # Finish
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print_header("PROJECT GENERATION COMPLETE")
    print(f"📂 Project location: {base_dir}")
    print(f"⏱ Duration: {duration:.1f} seconds")
    print("\nNext steps:")
    print("  1) cd stocks-analysis-ml-model/backend")
    print("  2) python -m venv venv")
    print("  3) .\\venv\\Scripts\\activate")
    print("  4) pip install -r requirements.txt")
    print("  5) python scripts\\generate_sample_data.py")
    print("  6) python scripts\\train_models.py")
    print("  7) uvicorn src.app:app --reload --port 8000")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
