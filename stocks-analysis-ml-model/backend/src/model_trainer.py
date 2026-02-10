"""
Model Trainer

Trains a RandomForestClassifier on extracted features and saves:
- random_forest.joblib
- feature_scaler.joblib
"""

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    LABEL_COLUMN,
    RF_MODEL_PATH,
    SCALER_PATH,
    TRAIN_RATIO,
)
from .features import get_feature_matrix


class ModelTrainer:
    """Wrapper around training a Random Forest model."""

    def __init__(self) -> None:
        self.model_path: Path = RF_MODEL_PATH
        self.scaler_path: Path = SCALER_PATH

    def _split_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Train-test split (time-aware by disabling shuffle)."""
        test_size = 1.0 - TRAIN_RATIO
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=False,
        )
        return X_train, X_test, y_train, y_test

    def train_all(self, df: pd.DataFrame) -> None:
        """Run full training: preprocess, split, scale, train, save."""
        df = df.dropna(subset=[LABEL_COLUMN])
        if df.empty:
            raise ValueError("No data available for training after filtering labels.")

        X = get_feature_matrix(df)
        y = df[LABEL_COLUMN].astype(int).values

        X_train, X_test, y_train, y_test = self._split_data(X, y)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(X_train_scaled, y_train)

        acc = clf.score(X_test_scaled, y_test)
        print(f"Validation accuracy: {acc:.3f}")

        joblib.dump(clf, self.model_path)
        joblib.dump(scaler, self.scaler_path)
        print(f"Saved model to: {self.model_path}")
        print(f"Saved scaler to: {self.scaler_path}")
