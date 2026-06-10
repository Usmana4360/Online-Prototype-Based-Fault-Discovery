# scripts/sweep_label_fraction.py
"""
Sweeps the window-labeling fraction and re-evaluates EVERY method against
each labeling, WITHOUT retraining anything. It reuses the score arrays that
inference.py and the baseline scripts already saved:

    results/global_reconstruction_error.csv   (GCL)
    results/scores_spc.npy
    results/scores_iforest.npy
    results/scores_ocsvm.npy

For each fraction it rebuilds window labels from the raw CSV (same rule as
create_labels.py) and reports ROC-AUC and PR-AUC for all methods, plus the
anomaly-window count so you can see when positives get too sparse.

Run AFTER you have run inference.py + all three baseline scripts at least
once (the score arrays don't depend on the label fraction, only the labels
do, so they don't need regenerating).

    python -m scripts.sweep_label_fraction
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.config import CLIP_LEN, STRIDE

CSV_PATH  = "data/raw/motor_data_200_drifts_labeled.csv"
LABEL_COL = "Label"

FRACTIONS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]

SCORE_SOURCES = {
    "GCL (Ours)":       ("csv", "results/global_reconstruction_error.csv",
                                 "global_reconstruction_error"),
    "3-Sigma SPC":      ("npy", "results/scores_spc.npy", None),
    "Isolation Forest": ("npy", "results/scores_iforest.npy", None),
    "One-Class SVM":    ("npy", "results/scores_ocsvm.npy", None),
}


def build_labels(row_labels, frac):
    starts = list(range(0, len(row_labels) - CLIP_LEN + 1, STRIDE))
    lab = np.zeros(len(starts), dtype=int)
    for i, s in enumerate(starts):
        r = row_labels[s:s + CLIP_LEN].mean()
        lab[i] = 1 if (r > 0 if frac <= 0 else r >= frac) else 0
    return lab


def load_scores():
    out = {}
    for name, (kind, path, col) in SCORE_SOURCES.items():
        if kind == "csv":
            out[name] = pd.read_csv(path)[col].values
        else:
            out[name] = np.load(path)
    return out


def main():
    df = pd.read_csv(CSV_PATH)
    row_labels = df[LABEL_COL].values.astype(int)
    scores = load_scores()

    print("\nLabel-fraction sweep (no retraining; scores fixed, labels vary)\n")
    header = (f"{'frac':>5}  {'#anom':>6}  " +
              "  ".join(f"{n.split()[0]:>9}" for n in SCORE_SOURCES))
    for frac in FRACTIONS:
        labels = build_labels(row_labels, frac)
        n_anom = int(labels.sum())

        # Need both classes present and aligned lengths
        if n_anom < 5 or n_anom == len(labels):
            print(f"{frac:>5.2f}  {n_anom:>6}   (too few/many positives, skipped)")
            continue

        if frac == FRACTIONS[0] or frac == 0.0:
            print(header)
            print("-" * len(header))

        cells = []
        for name in SCORE_SOURCES:
            s = scores[name]
            n = min(len(s), len(labels))
            auc = roc_auc_score(labels[:n], s[:n])
            cells.append(f"{auc:>9.4f}")
        print(f"{frac:>5.2f}  {n_anom:>6}  " + "  ".join(cells))

    print("\n(Values are ROC-AUC. Look for the fraction where GCL overtakes SPC.)")
    print("Once you pick a fraction, set ANOMALY_FRACTION in create_labels.py")
    print("to match, re-run create_labels.py, then plot_curves.py for figures.\n")


if __name__ == "__main__":
    main()
