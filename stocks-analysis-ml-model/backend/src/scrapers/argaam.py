"""
Argaam RSS Scraper (English)
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from pathlib import Path

RSS_URL = "https://www.argaam.com/en/rss"


def scrape_argaam(max_articles: int = 100) -> pd.DataFrame:
    """Scrape Argaam RSS feed."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(RSS_URL, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "xml")
        articles = []
        
        for item in soup.find_all("item")[:max_articles]:
            title = item.find("title").text.strip() if item.find("title") else ""
            link = item.find("link").text.strip() if item.find("link") else ""
            description = item.find("description").text.strip() if item.find("description") else ""
            pub_date = item.find("pubDate").text.strip() if item.find("pubDate") else ""
            
            # Parse date
            try:
                pub_dt = pd.to_datetime(pub_date)
            except:
                pub_dt = datetime.now()
            
            articles.append({
                "date": pub_dt.strftime("%Y-%m-%d"),
                "headline": title,
                "article_text": description,
                "url": link,
                "source": "argaam_rss",
                "language": "en",
            })
        
        df = pd.DataFrame(articles)
        print(f"Argaam: {len(df)} articles")
        return df
        
    except Exception as e:
        print(f"Argaam scraper failed: {e}")
        return pd.DataFrame()
