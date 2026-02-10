"""
Master News Collector - Multi-Source Pipeline

Sources:
- Argaam RSS
- Saudi Gazette RSS (Business + General)
- Arab News RSS
- Saudi Exchange HTML
- TradeArabia RSS ← NEW
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_DIR,
    TICKER_COLUMN,
    DATE_COLUMN,
    HEADLINE_COLUMN,
    TEXT_COLUMN,
    URL_COLUMN,
)

# Import all scrapers
from src.scrapers.argaam import scrape_argaam
from src.scrapers.saudigazette import scrape_saudigazette_business
from src.scrapers.arabnews import scrape_arabnews
from src.scrapers.saudiexchange import scrape_saudiexchange
from src.scrapers.tradearabia import scrape_tradearabia  # ← NEW


def collect_all_news(max_articles_per_source: int = 50) -> pd.DataFrame:
    """Run all scrapers and combine into unified dataset."""
    print("📰 Collecting news from 5 sources...")
    
    dfs = []
    
    # RSS Sources (English)
    print("\n1. Argaam RSS...")
    df1 = scrape_argaam(max_articles_per_source)
    dfs.append(df1)
    
    print("\n2. Saudi Gazette Business RSS...")
    df2 = scrape_saudigazette_business(max_articles_per_source)
    dfs.append(df2)
    
    print("\n3. Arab News RSS...")
    df3 = scrape_arabnews(max_articles_per_source)
    dfs.append(df3)
    
    print("\n4. TradeArabia RSS...")
    df4 = scrape_tradearabia(max_articles_per_source)  # ← NEW SOURCE
    dfs.append(df4)
    
    print("\n5. Saudi Exchange HTML (3 pages)...")
    df5 = scrape_saudiexchange(3)
    dfs.append(df5)
    
    # Combine all sources
    df_all = pd.concat(dfs, ignore_index=True)
    
    if df_all.empty:
        print("❌ No articles collected from any source!")
        return pd.DataFrame()
    
    # Standardize schema
    df_all[TICKER_COLUMN] = "2010"  # Default: SABIC
    
    # Deduplicate by headline + date
    before_dup = len(df_all)
    df_all = df_all.drop_duplicates(subset=[HEADLINE_COLUMN, DATE_COLUMN])
    after_dup = len(df_all)
    
    # Filter short texts
    before_filter = len(df_all)
    df_all = df_all[df_all[TEXT_COLUMN].str.len() > 30]
    after_filter = len(df_all)
    
    # Save
    out_path = DATA_DIR / "combined_news_raw.csv"
    df_all.to_csv(out_path, index=False)
    
    print(f"\n✅ Pipeline complete:")
    print(f"  Raw articles:           {before_dup}")
    print(f"  After deduplication:    {after_dup}")
    print(f"  After length filter:    {after_filter}")
    print(f"\n📊 Source breakdown:")
    print(df_all["source"].value_counts())
    
    print(f"\n💾 Saved to: {out_path}")
    
    # Show quality preview (fix date sorting)
    print("\n📄 Sample (5 newest):")
    df_all[DATE_COLUMN] = pd.to_datetime(df_all[DATE_COLUMN])
    preview = df_all.nlargest(5, DATE_COLUMN)[[DATE_COLUMN, HEADLINE_COLUMN, "source"]]
    print(preview.to_string(index=False))
    
    return df_all


if __name__ == "__main__":
    collect_all_news()
