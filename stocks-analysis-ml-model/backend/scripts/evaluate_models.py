"""
Evaluation Script

Computes accuracy, confusion matrix, feature importance, and saves plots.
Run: python scripts/evaluate_models.py
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    LABEL_COLUMN,
    RF_MODEL_PATH,
    SCALER_PATH,
    ALL_FEATURE_COLUMNS,
)
from src.data_loader import DataLoader
from src.preprocessing import preprocess_dataframe
from src.features import extract_features, get_feature_matrix


def evaluate_model() -> None:
    """Full evaluation pipeline with metrics and plots."""
    print("=" * 80)
    print("TADAWUL STOCK PREDICTION - EVALUATION")
    print("=" * 80)

    # Load model and scaler
    model = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Load and prepare data
    loader = DataLoader()
    df = loader.load_raw_news()
    df = preprocess_dataframe(df)
    df = extract_features(df)

    X = get_feature_matrix(df)
    y_true = df[LABEL_COLUMN].astype(int).values
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]  # prob of Up (class 1)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
    print("\n📊 Classification Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric:12}: {value:.3f}")

    # Feature importance
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {
            "feature": ALL_FEATURE_COLUMNS,
            "importance": feature_importance,
        }
    ).sort_values("importance", ascending=False)

    print("\n📈 Top 5 Feature Importances:")
    for _, row in importance_df.head().iterrows():
        print(f"  {row['feature']:15}: {row['importance']:.3f}")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(DATA_DIR / "evaluation_metrics.csv", index=False)
    print(f"\n💾 Saved metrics to: {DATA_DIR / 'evaluation_metrics.csv'}")

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Down", "Up"],
        yticklabels=["Down", "Up"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Feature Importance
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 3: ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(DATA_DIR / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n📊 Plots saved to data/:")
    print("  - confusion_matrix.png")
    print("  - feature_importance.png")
    print("  - roc_curve.png")


if __name__ == "__main__":
    evaluate_model()
