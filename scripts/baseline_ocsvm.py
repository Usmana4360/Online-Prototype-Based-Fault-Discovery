# scripts/baseline_ocsvm.py
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef,
    precision_score, recall_score
)
from sklearn.decomposition import PCA
import json
from src.config import FEATURE_COLS, CLIP_LEN, STRIDE

CSV_PATH = "data/raw/motor_data_200_drifts_labeled.csv"

def main():
    df     = pd.read_csv(CSV_PATH)
    labels = np.load("data/labels/test_labels.npy")

    feature_data  = df[FEATURE_COLS].values
    window_starts = list(range(0, len(df) - CLIP_LEN + 1, STRIDE))

    # Flatten windows
    windows  = np.array([
        feature_data[s:s + CLIP_LEN].flatten()
        for s in window_starts
    ])

    n_train       = int(0.70 * len(windows))
    train_windows = windows[:n_train]

    # PCA first — OC-SVM is slow on high-dim data (100*9=900 dims)
    print("Fitting PCA...")
    pca = PCA(n_components=50, random_state=42)
    train_pca = pca.fit_transform(train_windows)
    all_pca   = pca.transform(windows)

    # Scale
    scaler    = StandardScaler()
    train_pca = scaler.fit_transform(train_pca)
    all_pca   = scaler.transform(all_pca)

    print("Fitting One-Class SVM (may take 1-2 minutes)...")
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
    ocsvm.fit(train_pca)

    scores = -ocsvm.score_samples(all_pca)

    min_len = min(len(scores), len(labels))
    scores  = scores[:min_len]
    labels  = labels[:min_len]

    threshold = np.percentile(scores[:n_train], 95)
    preds     = (scores > threshold).astype(int)

    results = {
        "method"   : "One-Class SVM",
        "roc_auc"  : round(roc_auc_score(labels, scores), 4),
        "pr_auc"   : round(average_precision_score(labels, scores), 4),
        "f1"       : round(f1_score(labels, preds, zero_division=0), 4),
        "mcc"      : round(matthews_corrcoef(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall"   : round(recall_score(labels, preds, zero_division=0), 4),
    }

    print("\nOne-Class SVM Baseline Results:")
    for k, v in results.items():
        print(f"  {k:<12}: {v}")

    with open("results/baseline_ocsvm.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved: results/baseline_ocsvm.json")

if __name__ == "__main__":
    main()