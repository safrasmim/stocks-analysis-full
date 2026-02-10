"""
Arab News RSS Scraper
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd


def scrape_arabnews(max_articles: int = 100) -> pd.DataFrame:
    """Scrape Arab News RSS feed."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(
            "https://www.arabnews.com/cat/4/rss.xml", 
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
            
            # Parse date
            try:
                pub_dt = pd.to_datetime(pub_text)
            except:
                pub_dt = datetime.now()
            
            articles.append({
                "date": pub_dt.strftime("%Y-%m-%d"),
                "headline": title_text,
                "article_text": desc_text,
                "url": link_text,
                "source": "arabnews_rss",
                "language": "en",
            })
        
        df = pd.DataFrame(articles)
        if not df.empty:
            df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
        print(f"Arab News: scraped {len(df)} articles")
        return df
        
    except Exception as e:
        print(f"Arab News scraper failed: {e}")
        return pd.DataFrame()
