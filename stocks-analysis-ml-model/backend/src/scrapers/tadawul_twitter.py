"""
Tadawul Twitter Scraper (placeholder)

Real Twitter scraping requires API keys or external tools.
This reads pre-exported CSV tweets.
"""

import pandas as pd
from pathlib import Path


def load_tadawul_tweets(csv_path: str = "data/tadawul_tweets.csv") -> pd.DataFrame:
    """
    Load pre-exported Tadawul Twitter data.
    
    Expected CSV columns: created_at, text, tweet_id
    Export via tools like TwitData or browser extensions.
    """
    path = Path(csv_path)
    if not path.exists():
        print(f"Tweet CSV not found: {path}")
        print("Export from https://twitdata.com or similar tool first.")
        return pd.DataFrame()
    
    df = pd.read_csv(path)
    
    # Standardize schema
    df["date"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d")
    df["headline"] = df["text"].str[:100] + "..."
    df["article_text"] = df["text"]
    df["url"] = "https://twitter.com/TadawulFeed/status/" + df["tweet_id"].astype(str)
    df["source"] = "tadawul_twitter"
    df["language"] = "ar"
    df["original_text"] = df["text"]
    
    # TODO: Translate article_text to English using API
    
    return df.head(100)  # Limit for demo
