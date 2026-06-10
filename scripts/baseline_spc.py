# scripts/baseline_spc.py

"""
Industrial baseline: 3-Sigma SPC anomaly detection

Fix summary:
- Uses WINDOW-level labels (not raw row labels)
- Ensures fair comparison with IF / OCSVM / GCL
- Adds industrial metrics: FPR, MDR, Specificity
"""

import numpy as np
import pandas as pd
import json

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
LABEL_PATH = "data/labels/test_labels.npy"


# =====================================
# MAIN
# =====================================
def main():

    # -----------------------------
    # Load data
    # -----------------------------
    df = pd.read_csv(CSV_PATH)
    labels = np.load(LABEL_PATH)  # window-level labels

    feature_data = df[FEATURE_COLS].values

    # -----------------------------
    # Training stats (normal behavior)
    # -----------------------------
    n_train = int(0.70 * len(feature_data))
    train_data = feature_data[:n_train]

    mu = train_data.mean(axis=0)
    sigma = train_data.std(axis=0) + 1e-8

    # -----------------------------
    # Window creation
    # -----------------------------
    window_starts = list(range(0, len(df) - CLIP_LEN + 1, STRIDE))

    scores = np.zeros(len(window_starts))

    # -----------------------------
    # SPC scoring
    # -----------------------------
    for i, start in enumerate(window_starts):
        window = feature_data[start:start + CLIP_LEN]

<<<<<<< HEAD
        z = np.abs((window - mu) / sigma)
        scores[i] = z.max()  # worst deviation in window
=======
    # Align lengths
    min_len  = min(len(scores), len(labels))
    scores   = scores[:min_len]
    labels   = labels[:min_len]
    np.save("results/scores_spc.npy", scores)

>>>>>>> 4d70ebf3cee358a6add21cb775943ad730662ff2

    # -----------------------------
    # Alignment check
    # -----------------------------
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]

    assert len(scores) == len(labels), "Mismatch between scores and labels"

    np.save("results/scores_spc.npy", scores)

    # -----------------------------
    # Threshold (3-sigma rule)
    # -----------------------------
    preds = (scores > 3.0).astype(int)

    # =============================
    # CONFUSION MATRIX
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
    # INDUSTRIAL METRICS (NEW)
    # -----------------------------
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    mdr = fn / (fn + tp) if (fn + tp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # -----------------------------
    # Results dictionary
    # -----------------------------
    results = {
        "method": "3-Sigma SPC",

        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),

        "f1": round(f1, 4),
        "mcc": round(mcc, 4),

        "precision": round(precision, 4),
        "recall": round(recall, 4),

        # NEW METRICS
        "fpr": round(fpr, 4),
        "mdr": round(mdr, 4),
        "specificity": round(specificity, 4)
    }

    # -----------------------------
    # Print results
    # -----------------------------
    print("\n3-Sigma SPC Baseline Results:")
    print("=" * 50)
    print(f"Evaluated on {len(labels)} windows")
    print(f"Anomalies: {int(labels.sum())} | Normal: {int((labels==0).sum())}")

    for k, v in results.items():
        print(f"{k:<15}: {v}")

    # -----------------------------
    # Save JSON
    # -----------------------------
    with open("results/baseline_spc.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved: results/baseline_spc.json")


# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    main()