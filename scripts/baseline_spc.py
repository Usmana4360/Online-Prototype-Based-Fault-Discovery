# scripts/baseline_spc.py
"""
Simplest industrial baseline: flag windows where any feature
exceeds mu ± 3*sigma computed on training data.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef,
    precision_score, recall_score
)
import json
from src.config import FEATURE_COLS, CLIP_LEN, STRIDE

CSV_PATH   = "data/raw/motor_data_200_drifts_labeled.csv"
LABEL_COL  = "Label"

def main():
    df     = pd.read_csv(CSV_PATH)
    labels = df[LABEL_COL].values

    # Fit stats on training rows only (first 70%)
    feature_data = df[FEATURE_COLS].values
    n_train      = int(0.70 * len(feature_data))
    train_data   = feature_data[:n_train]

    mu    = train_data.mean(axis=0)
    sigma = train_data.std(axis=0) + 1e-8

    # Score each window: max z-score across features and timesteps
    window_starts = list(range(0, len(df) - CLIP_LEN + 1, STRIDE))
    scores = np.zeros(len(window_starts))

    for i, start in enumerate(window_starts):
        window = feature_data[start:start + CLIP_LEN]
        z      = np.abs((window - mu) / sigma)
        scores[i] = z.max()  # most extreme deviation in window

    # Align lengths
    min_len  = min(len(scores), len(labels))
    scores   = scores[:min_len]
    labels   = labels[:min_len]

    # Threshold at 3-sigma (z > 3 = anomaly)
    preds = (scores > 3.0).astype(int)

    results = {
        "method"   : "3-Sigma SPC",
        "roc_auc"  : round(roc_auc_score(labels, scores), 4),
        "pr_auc"   : round(average_precision_score(labels, scores), 4),
        "f1"       : round(f1_score(labels, preds, zero_division=0), 4),
        "mcc"      : round(matthews_corrcoef(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall"   : round(recall_score(labels, preds, zero_division=0), 4),
    }

    print("\n3-Sigma SPC Baseline Results:")
    for k, v in results.items():
        print(f"  {k:<12}: {v}")

    with open("results/baseline_spc.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/baseline_spc.json")

if __name__ == "__main__":
    main()