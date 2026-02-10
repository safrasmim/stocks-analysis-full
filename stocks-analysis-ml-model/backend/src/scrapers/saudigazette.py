"""
Saudi Gazette RSS Scraper
Business: https://saudigazette.com.sa/rssFeed/73
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd


def scrape_saudigazette_business(max_articles: int = 100) -> pd.DataFrame:
    """Scrape Saudi Gazette business RSS."""
    RSS_URLS = [
        "https://saudigazette.com.sa/rssFeed/73",  # Business
        "https://saudigazette.com.sa/rssFeed/74",  # General
    ]
    
    all_articles = []
    
    for url in RSS_URLS:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "xml")
            articles = []
            
            for item in soup.find_all("item")[:max_articles//len(RSS_URLS)]:
                title = item.find("title").text.strip() if item.find("title") else ""
                link = item.find("link").text.strip() if item.find("link") else ""
                description = item.find("description").text.strip() if item.find("description") else ""
                pub_date = item.find("pubDate").text.strip() if item.find("pubDate") else ""
                
                try:
                    pub_dt = pd.to_datetime(pub_date)
                except:
                    pub_dt = datetime.now()
                
                articles.append({
                    "date": pub_dt.strftime("%Y-%m-%d"),
                    "headline": title,
                    "article_text": description,
                    "url": link,
                    "source": f"saudigazette_{url.split('/')[-1]}",
                    "language": "en",
                })
            
            all_articles.extend(articles)
            print(f"Saudi Gazette ({url.split('/')[-1]}): {len(articles)} articles")
            
        except Exception as e:
            print(f"Saudi Gazette ({url}) failed: {e}")
    
    df = pd.DataFrame(all_articles)
    print(f"Saudi Gazette total: {len(df)} articles")
    return df
