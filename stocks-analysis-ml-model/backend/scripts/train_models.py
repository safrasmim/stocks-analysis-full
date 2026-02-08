"""
Training Pipeline

End-to-end: load data → preprocess → extract features → train model.
Run: python scripts/train_models.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.preprocessing import preprocess_dataframe
from src.features import extract_features
from src.model_trainer import ModelTrainer
from src.lstm_trainer import LSTMTrainer

def main() -> None:
    print("=" * 80)
    print("TADAWUL STOCK PREDICTION - TRAINING PIPELINE")
    print("=" * 80)

    print("\n[1/5] Loading data...")
    loader = DataLoader()
    df = loader.load_raw_news()

    print("\n[2/5] Preprocessing text...")
    df = preprocess_dataframe(df)

    print("\n[3/5] Extracting features (length + FinBERT sentiment)...")
    df = extract_features(df)

    print("\n[4/5] Training model...")
    trainer = ModelTrainer()
    trainer.train_all(df)

    
    print("\n[5/5] Training LSTM...")
    lstm = LSTMTrainer()
    lstm.train(df)



if __name__ == "__main__":
    main()
