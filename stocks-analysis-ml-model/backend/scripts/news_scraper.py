"""
Real News Scraper

Scrapes Argaam and Alarabiya RSS feeds for Tadawul news.
Run: python scripts/news_scraper.py
"""

import sys
from pathlib import Path
import requests
from datetime import datetime
import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_DIR,
    TICKER_COLUMN,
    TEXT_COLUMN,
    DATE_COLUMN,
    HEADLINE_COLUMN,
    URL_COLUMN,
)


def scrape_argaam(days: int = 30) -> pd.DataFrame:
    """Scrape Argaam market news RSS."""
    try:
        response = requests.get(
            "https://www.argaam.com/en/rss/market/market-news.rss", timeout=15
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")

        articles = []
        for item in soup.find_all("item")[:100]:  # More articles
            title = item.title.text.strip() if item.title else ""
            link = item.link.text.strip() if item.link else ""
            description = item.description.text.strip() if item.description else ""
            pub_date = item.pubDate.text.strip() if item.pubDate else ""

            # Parse date or use now
            try:
                pub_dt = pd.to_datetime(pub_date)
            except:
                pub_dt = datetime.now()

            articles.append(
                {
                    DATE_COLUMN: pub_dt.strftime("%Y-%m-%d"),
                    HEADLINE_COLUMN: title,
                    TEXT_COLUMN: description,
                    URL_COLUMN: link,
                }
            )

        df = pd.DataFrame(articles)
        if not df.empty:
            df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        print(f"Argaam: scraped {len(df)} articles")
        return df
    except Exception as e:
        print(f"Argaam scraper failed: {e}")
        return pd.DataFrame()


def scrape_alarabiya(days: int = 30) -> pd.DataFrame:
    """Scrape Alarabiya business RSS."""
    try:
        response = requests.get(
            "https://english.alarabiya.net/rss/business", timeout=15
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "xml")

        articles = []
        for item in soup.find_all("item")[:100]:
            title = item.title.text.strip() if item.title else ""
            link = item.link.text.strip() if item.link else ""
            description = item.description.text.strip() if item.description else ""
            pub_date = item.pubDate.text.strip() if item.pubDate else ""

            try:
                pub_dt = pd.to_datetime(pub_date)
            except:
                pub_dt = datetime.now()

            articles.append(
                {
                    DATE_COLUMN: pub_dt.strftime("%Y-%m-%d"),
                    HEADLINE_COLUMN: title,
                    TEXT_COLUMN: description,
                    URL_COLUMN: link,
                }
            )

        df = pd.DataFrame(articles)
        if not df.empty:
            df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        print(f"Alarabiya: scraped {len(df)} articles")
        return df
    except Exception as e:
        print(f"Alarabiya scraper failed: {e}")
        return pd.DataFrame()


def main():
    print("Scraping real Tadawul news...")
    df1 = scrape_argaam()
    df2 = scrape_alarabiya()

    df = pd.concat([df1, df2], ignore_index=True)

    if df.empty:
        print("No articles scraped. Check internet connection and RSS feeds.")
        return

    # Add dummy ticker (you'd filter by keyword in production)
    df[TICKER_COLUMN] = "2010"  # Default to SABIC

    # Basic filtering
    df = df[df[TEXT_COLUMN].str.len() > 20]  # Remove very short texts

    out_path = DATA_DIR / "real_news_scraped.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✅ Saved {len(df)} real articles to {out_path}")
    print("\nSample headlines:")
    print(df[HEADLINE_COLUMN].head())


if __name__ == "__main__":
    main()
