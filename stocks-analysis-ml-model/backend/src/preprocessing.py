"""
Preprocessing

Cleans article text: lowercasing, removing special characters,
and normalizing whitespace.
"""

import re
import pandas as pd

from .config import TEXT_COLUMN

CLEAN_RE = re.compile(r"[^\w\s.,!?]+")


def clean_text(text: str) -> str:
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = CLEAN_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to the article_text column."""
    df = df.copy()
    if TEXT_COLUMN in df.columns:
        df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).apply(clean_text)
    return df
