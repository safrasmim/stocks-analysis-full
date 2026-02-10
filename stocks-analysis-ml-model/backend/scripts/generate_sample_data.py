"""
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
    DATA_DIR,
    TICKERS,
    TICKER_NAMES,
    DATE_COLUMN,
    TICKER_COLUMN,
    HEADLINE_COLUMN,
    TEXT_COLUMN,
    LABEL_COLUMN,
    RETURN_COLUMN,
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
    """Generate article body conditioned on sentiment label."""
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
    """Create a synthetic dataset of news articles and returns."""
    print("=" * 80)
    print("GENERATING SAMPLE DATA")
    print("=" * 80)

    cfg = SAMPLE_DATA_CONFIG
    start_date = datetime.strptime(cfg["start_date"], "%Y-%m-%d")
    end_date = start_date + timedelta(days=cfg["date_range_days"])

    print("\nConfiguration:")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Articles per stock: {cfg['num_articles_per_stock']}")
    print(f"  Positive ratio: {cfg['positive_ratio']:.0%}")

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
