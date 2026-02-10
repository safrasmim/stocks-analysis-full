"""
TradeArabia RSS Scraper (MENA Business News)
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd


def scrape_tradearabia(max_articles: int = 50) -> pd.DataFrame:
    """Scrape TradeArabia RSS feed."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(
            "https://www.tradearabia.com/rss.xml",
            headers=headers,
            timeout=15
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "xml")
        articles = []
        
        for item in soup.find_all("item")[:max_articles]:
            title = item.find("title")
            title_text = title.text.strip() if title else ""
            
            link = item.find("link")
            link_text = link.text.strip() if link else ""
            
            description = item.find("description")
            desc_text = description.text.strip() if description else ""
            
            pub_date = item.find("pubDate")
            pub_text = pub_date.text.strip() if pub_date else ""
            
            # Safe date parsing
            try:
                pub_dt = pd.to_datetime(pub_text)
            except:
                pub_dt = datetime.now()
            
            articles.append({
                "date": pub_dt.strftime("%Y-%m-%d"),
                "headline": title_text,
                "article_text": desc_text,
                "url": link_text,
                "source": "tradearabia_rss",
                "language": "en",
            })
        
        df = pd.DataFrame(articles)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        print(f"TradeArabia: scraped {len(df)} articles")
        return df
        
    except Exception as e:
        print(f"TradeArabia scraper failed: {e}")
        return pd.DataFrame()
