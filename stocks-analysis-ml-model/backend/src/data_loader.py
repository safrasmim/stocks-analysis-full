"""
Data Loader

Loads news data from CSV for the Tadawul prediction system.
"""

from pathlib import Path
import pandas as pd

from .config import DATA_DIR, DATE_COLUMN, TICKER_COLUMN


class DataLoader:
    """Utility class to load and filter news data."""

    def __init__(self, filename: str = "raw_news_sample.csv") -> None:
        self.path: Path = DATA_DIR / filename

    def load_raw_news(self) -> pd.DataFrame:
        """Load raw news CSV into a DataFrame."""
        if not self.path.exists():
            raise FileNotFoundError(f"Data file not found: {self.path}")
        df = pd.read_csv(self.path)
        if DATE_COLUMN in df.columns:
            df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        return df

    def filter_by_ticker(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Return subset of DataFrame for a given ticker."""
        return df[df[TICKER_COLUMN] == ticker].copy()
