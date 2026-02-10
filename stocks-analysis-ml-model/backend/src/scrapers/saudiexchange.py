"""
Saudi Exchange Issuer News Scraper (HTML)
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import time


def scrape_saudiexchange(max_pages: int = 3) -> pd.DataFrame:
    """Scrape Saudi Exchange issuer news pages."""
    base_url = "https://www.saudiexchange.sa/wps/portal/saudiexchange/newsandreports/issuer-news?locale=en&page="
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    
    all_articles = []
    
    for page in range(1, max_pages + 1):
        try:
            url = f"{base_url}{page}"
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Look for news cards/table rows
            news_items = soup.find_all(["div", "tr", "article"], class_=lambda x: x and ("news" in x.lower() or "item" in x.lower()))
            
            for item in news_items[:20]:  # Max 20 per page
                # Try multiple possible selectors
                title_elem = item.find(["a", "h3", "h4", "span"], string=True)
                date_elem = item.find(["time", "span"], class_=lambda x: x and ("date" in x.lower()))
                link_elem = item.find("a", href=True)
                
                title = title_elem.get_text().strip() if title_elem else ""
                date_text = date_elem.get_text().strip() if date_elem else ""
                link = link_elem.get("href") if link_elem else ""
                
                if link.startswith("/"):
                    link = "https://www.saudiexchange.sa" + link
                
                # NEW SAFE CODE:
                try:
                    if date_text and date_text.strip():
                        pub_dt = pd.to_datetime(date_text.strip())
                    else:
                        pub_dt = datetime.now()
                except:
                    pub_dt = datetime.now()
                    
                
                if title and len(title) > 10:
                    all_articles.append({
                        "date": pub_dt.strftime("%Y-%m-%d"),
                        "headline": title,
                        "article_text": title[:300] + "..." if len(title) > 300 else title,
                        "url": link,
                        "source": "saudiexchange_html",
                        "language": "en",
                    })
            
            print(f"Saudi Exchange page {page}: {len(all_articles)} total articles")
            time.sleep(1)  # Be nice
            
        except Exception as e:
            print(f"Saudi Exchange page {page} failed: {e}")
    
    df = pd.DataFrame(all_articles)
    print(f"Saudi Exchange total: {len(df)} articles")
    return df
