# scripts/baseline_iforest.py

import numpy as np
import pandas as pd
import joblib
import json

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    confusion_matrix
)

from src.config import FEATURE_COLS, CLIP_LEN, STRIDE

CSV_PATH = "data/raw/motor_data_200_drifts_labeled.csv"


# =====================================
# MAIN
# =====================================
def main():

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(CSV_PATH)
    labels = np.load("data/labels/test_labels.npy")

    feature_data = df[FEATURE_COLS].values
    window_starts = list(range(0, len(df) - CLIP_LEN + 1, STRIDE))

    # -----------------------------
    # Build sliding windows
    # -----------------------------
    windows = np.array([
        feature_data[s:s + CLIP_LEN].flatten()
        for s in window_starts
    ])

    # -----------------------------
    # Train split
    # -----------------------------
    n_train = int(0.70 * len(windows))
    train_windows = windows[:n_train]

    # -----------------------------
    # Model
    # -----------------------------
    iso = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42
    )

    iso.fit(train_windows)

    # -----------------------------
    # Anomaly scores
    # -----------------------------
    scores = -iso.score_samples(windows)  # higher = more anomalous

    # Align
    min_len = min(len(scores), len(labels))
<<<<<<< HEAD
    scores = scores[:min_len]
    labels = labels[:min_len]
=======
    scores  = scores[:min_len]
    labels  = labels[:min_len]
    np.save("results/scores_iforest.npy", scores)
>>>>>>> 4d70ebf3cee358a6add21cb775943ad730662ff2

    np.save("results/scores_iforest.npy", scores)

    # -----------------------------
    # Threshold (95th percentile)
    # -----------------------------
    train_scores = scores[:n_train]
    threshold = np.percentile(train_scores, 95)

    preds = (scores > threshold).astype(int)

    # =============================
    # CONFUSION MATRIX METRICS
    # =============================
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    # -----------------------------
    # Core metrics
    # -----------------------------
    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    # -----------------------------
    # NEW: Industrial metrics
    # -----------------------------
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    mdr = fn / (fn + tp) if (fn + tp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # -----------------------------
    # Final results dictionary
    # -----------------------------
    results = {
        "method": "Isolation Forest",

        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),

        "f1": round(f1, 4),
        "mcc": round(mcc, 4),

        "precision": round(precision, 4),
        "recall": round(recall, 4),

        # NEW METRICS
        "fpr": round(fpr, 4),
        "mdr": round(mdr, 4),
        "specificity": round(specificity, 4),

        "best_threshold": round(threshold, 4)
    }

    # -----------------------------
    # Print results
    # -----------------------------
    print("\nIsolation Forest Baseline Results:")
    print("=" * 50)
    for k, v in results.items():
        print(f"{k:<15}: {v}")

    # -----------------------------
    # Save JSON
    # -----------------------------
    with open("results/baseline_iforest.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: results/baseline_iforest.json")


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    main()