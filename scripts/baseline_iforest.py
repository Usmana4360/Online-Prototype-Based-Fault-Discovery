# scripts/baseline_iforest.py
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef,
    precision_score, recall_score
)
import json
from src.config import FEATURE_COLS, CLIP_LEN, STRIDE

CSV_PATH  = "data/raw/motor_data_200_drifts_labeled.csv"

def main():
    df     = pd.read_csv(CSV_PATH)
    labels = np.load("data/labels/test_labels.npy")

    feature_data  = df[FEATURE_COLS].values
    window_starts = list(range(0, len(df) - CLIP_LEN + 1, STRIDE))

    # Flatten each window to a feature vector
    windows = np.array([
        feature_data[s:s + CLIP_LEN].flatten()
        for s in window_starts
    ])

    # Train on first 70% of windows
    n_train      = int(0.70 * len(windows))
    train_windows = windows[:n_train]

    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )
    iso.fit(train_windows)

    # Anomaly score: higher = more anomalous
    scores = -iso.score_samples(windows)

    # Align lengths
    min_len = min(len(scores), len(labels))
    scores  = scores[:min_len]
    labels  = labels[:min_len]
    np.save("results/scores_iforest.npy", scores)

    # Threshold at 95th percentile of training scores
    train_scores = scores[:n_train]
    threshold    = np.percentile(train_scores, 95)
    preds        = (scores > threshold).astype(int)

    results = {
        "method"   : "Isolation Forest",
        "roc_auc"  : round(roc_auc_score(labels, scores), 4),
        "pr_auc"   : round(average_precision_score(labels, scores), 4),
        "f1"       : round(f1_score(labels, preds, zero_division=0), 4),
        "mcc"      : round(matthews_corrcoef(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall"   : round(recall_score(labels, preds, zero_division=0), 4),
    }

    print("\nIsolation Forest Baseline Results:")
    for k, v in results.items():
        print(f"  {k:<12}: {v}")

    with open("results/baseline_iforest.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/baseline_iforest.json")

if __name__ == "__main__":
    main()