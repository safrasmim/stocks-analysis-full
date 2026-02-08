"""
Feature Extraction

Extracts:
- Basic text features (length, word count)
- FinBERT sentiment features (negative/neutral/positive/compound)
"""

from typing import List

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import (
    TEXT_COLUMN,
    FINBERT_MODEL_NAME,
    FINBERT_MAX_LENGTH,
    FINBERT_BATCH_SIZE,
    ALL_FEATURE_COLUMNS,
)

# Global FinBERT objects, lazy-loaded
_tokenizer = None
_model = None


def _load_finbert() -> None:
    """Load FinBERT tokenizer and model (once)."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
        _model.eval()


def _finbert_sentiment(texts: List[str]) -> np.ndarray:
    """Return FinBERT probabilities for each text (N, 3)."""
    _load_finbert()
    with torch.no_grad():
        inputs = _tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=FINBERT_MAX_LENGTH,
            return_tensors="pt",
        )
        outputs = _model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return probs.cpu().numpy()  # shape: (N, 3)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add feature columns to DataFrame:
    - feat_length, feat_word_count
    - sent_negative, sent_neutral, sent_positive, sent_compound
    """
    df = df.copy()
    texts = df[TEXT_COLUMN].fillna("").astype(str)

    # Basic length features
    df["feat_length"] = texts.apply(len)
    df["feat_word_count"] = texts.apply(lambda t: len(t.split()))

    # FinBERT sentiment (batched)
    text_list = texts.tolist()
    all_probs = []

    for i in range(0, len(text_list), FINBERT_BATCH_SIZE):
        batch = text_list[i : i + FINBERT_BATCH_SIZE]
        probs = _finbert_sentiment(batch)  # (batch_size, 3)
        all_probs.append(probs)

    if all_probs:
        probs_all = np.vstack(all_probs)  # (N, 3)
        df["sent_negative"] = probs_all[:, 0]
        df["sent_neutral"] = probs_all[:, 1]
        df["sent_positive"] = probs_all[:, 2]
        df["sent_compound"] = df["sent_positive"] - df["sent_negative"]
    else:
        df["sent_negative"] = 0.0
        df["sent_neutral"] = 0.0
        df["sent_positive"] = 0.0
        df["sent_compound"] = 0.0

    return df


def get_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return feature matrix used by models."""
    return df[ALL_FEATURE_COLUMNS].values
