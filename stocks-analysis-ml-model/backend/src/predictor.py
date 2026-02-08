"""
Predictor

Loads the trained model and scaler, and makes predictions from raw texts.
"""

from pathlib import Path
from typing import List, Dict

import joblib
import numpy as np
import pandas as pd

from .config import TEXT_COLUMN, RF_MODEL_PATH, SCALER_PATH
from .preprocessing import preprocess_dataframe
from .features import extract_features, get_feature_matrix


class Predictor:
    """Inference wrapper around the trained Random Forest model."""

    def __init__(
        self,
        model_path: Path | str = RF_MODEL_PATH,
        scaler_path: Path | str = SCALER_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict_from_texts(self, texts: List[str]) -> Dict:
        """Return predictions and probabilities for a list of texts."""
        df = pd.DataFrame({TEXT_COLUMN: texts})
        df = preprocess_dataframe(df)
        df = extract_features(df)
        X = get_feature_matrix(df)
        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            classes = list(self.model.classes_)
            if 1 in classes:
                up_idx = classes.index(1)
                p_up = probs[:, up_idx]
            else:
                p_up = probs[:, 0]
        else:
            p_up = np.zeros(len(texts))

        preds = (p_up >= 0.5).astype(int)
        labels = ["Up" if p == 1 else "Down" for p in preds]
        return {
            "predictions": preds.tolist(),
            "labels": labels,
            "probabilities_up": p_up.tolist(),
        }
