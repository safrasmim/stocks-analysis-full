"""
LSTM Model for Time-Series Stock Movement Prediction

Predicts tomorrow's movement based on past N days of news features.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import joblib

from .config import (
    DATE_COLUMN,
    TICKER_COLUMN,
    LABEL_COLUMN,
    LSTM_MODEL_PATH,
    ALL_FEATURE_COLUMNS,
    LSTM_CONFIG,
)


class LSTMTrainer:
    def __init__(self) -> None:
        self.model_path = LSTM_MODEL_PATH

    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int = 10) -> tuple:
        """Convert data to (X, y) sequences."""
        df = df.sort_values(DATE_COLUMN)
        X, y = [], []
        for _, group in df.groupby(TICKER_COLUMN):
            features = group[ALL_FEATURE_COLUMNS].values
            labels = group[LABEL_COLUMN].values

            for i in range(sequence_length, len(features)):
                X.append(features[i - sequence_length : i])
                y.append(labels[i])

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame) -> None:
        """Train LSTM model."""
        X, y = self.prepare_sequences(df)

        model = Sequential([
            LSTM(LSTM_CONFIG["units"], return_sequences=True),
            Dropout(LSTM_CONFIG["dropout"]),
            LSTM(LSTM_CONFIG["units"]),
            Dropout(LSTM_CONFIG["dropout"]),
            Dense(1, activation="sigmoid"),
        ])

        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=LSTM_CONFIG["early_stopping_patience"],
            restore_best_weights=True,
        )

        history = model.fit(
            X,
            y,
            epochs=LSTM_CONFIG["epochs"],
            batch_size=LSTM_CONFIG["batch_size"],
            validation_split=LSTM_CONFIG["validation_split"],
            callbacks=[early_stop],
            verbose=1,
        )

        model.save(self.model_path)
        print(f"Saved LSTM model to: {self.model_path}")
