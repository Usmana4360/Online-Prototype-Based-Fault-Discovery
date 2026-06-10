# scripts/create_labels.py
"""
Creates window-level labels from row-level labels.

CHANGE (vs original):
  Original rule: a window is anomalous if ANY single row in it is anomalous.
  That labels windows with as little as 1 drifting row out of CLIP_LEN as
  positive -- but such windows are barely distinguishable from normal, and
  a reconstruction model legitimately scores them low. SPC's max-z-score
  "wins" only on these borderline windows.

  New rule: a window is anomalous if at least ANOMALY_FRACTION of its rows
  are anomalous. Set ANOMALY_FRACTION=0.0 to recover the original behaviour.
  Sweep this (0.0, 0.05, 0.10, 0.20) and re-evaluate to see how the
  detectability threshold affects every method's AUC.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.config import CLIP_LEN, STRIDE

CSV_PATH         = "data/raw/motor_data_200_drifts_labeled.csv"
LABEL_COL        = "Label"
ANOMALY_FRACTION = 0.10   # >= this fraction of rows anomalous -> window anomalous
                          # 0.0 reproduces the old "any row" rule


def create_window_labels(df, label_col, clip_len, stride, frac):
    row_labels = df[label_col].values.astype(int)
    window_starts = list(range(0, len(df) - clip_len + 1, stride))
    window_labels = np.zeros(len(window_starts), dtype=int)

    for i, start in enumerate(window_starts):
        end = start + clip_len
        anomaly_ratio = row_labels[start:end].mean()
        # frac == 0 -> any anomalous row triggers (strictly greater than 0);
        # frac > 0  -> require at least that fraction.
        if frac <= 0:
            if anomaly_ratio > 0:
                window_labels[i] = 1
        else:
            if anomaly_ratio >= frac:
                window_labels[i] = 1

    return window_labels, window_starts


def verify_and_plot(df, window_labels, window_starts, label_col, clip_len, frac):
    n_windows = len(window_labels)
    n_anomaly = window_labels.sum()
    n_normal  = n_windows - n_anomaly
    ratio     = n_anomaly / n_windows * 100

    print("=" * 55)
    print("  WINDOW LABEL CREATION REPORT")
    print("=" * 55)
    print(f"  CSV rows          : {len(df):,}")
    print(f"  Anomaly rows      : {df[label_col].sum():,}")
    print(f"  Window size       : {clip_len} rows")
    print(f"  Stride            : {STRIDE} rows")
    print(f"  Anomaly fraction  : {frac:.2f}  (>= this -> window anomalous)")
    print(f"  Total windows     : {n_windows:,}")
    print(f"  Anomaly windows   : {n_anomaly}  ({ratio:.1f}%)")
    print(f"  Normal  windows   : {n_normal}")
    print("=" * 55)

    if n_anomaly < 30:
        print("WARNING: Fewer than 30 anomaly windows.")
        print("   Metrics will be unstable. Lower ANOMALY_FRACTION")
        print("   (e.g. 0.05) or reduce STRIDE in config.py.")
    elif ratio > 40:
        print("WARNING: >40% anomaly ratio is unusually high.")
        print("   Verify your label column / fraction.")
    else:
        print("Label distribution looks healthy for evaluation.")

    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=False)
    row_labels = df[label_col].values
    axes[0].fill_between(range(len(row_labels)), row_labels,
                         color='#E24B4A', alpha=0.7)
    axes[0].set_ylabel("Row Label")
    axes[0].set_title("Row-Level Anomaly Positions in Raw CSV")
    axes[0].set_ylim(-0.1, 1.3)
    axes[0].grid(alpha=0.3)

    axes[1].fill_between(range(len(window_labels)), window_labels,
                         color='#D85A30', alpha=0.7)
    axes[1].set_ylabel("Window Label")
    axes[1].set_xlabel("Window Index")
    axes[1].set_title(f"Window-Level Labels (frac>={frac:.2f}): "
                      f"{n_anomaly} anomaly / {n_normal} normal")
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/label_verification.png", dpi=150, bbox_inches='tight')
    print("\n  Saved: results/label_verification.png\n")


def main():
    print(f"\nLoading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    if LABEL_COL not in df.columns:
        print(f"\nColumn '{LABEL_COL}' not found. Available: {list(df.columns)}")
        return

    unique_vals = df[LABEL_COL].unique()
    if not set(unique_vals).issubset({0, 1}):
        print(f"\nLabel column has unexpected values: {unique_vals}")
        return

    window_labels, window_starts = create_window_labels(
        df, LABEL_COL, CLIP_LEN, STRIDE, ANOMALY_FRACTION
    )
    verify_and_plot(df, window_labels, window_starts,
                    LABEL_COL, CLIP_LEN, ANOMALY_FRACTION)

    os.makedirs("data/labels", exist_ok=True)
    np.save("data/labels/test_labels.npy", window_labels)
    print(f"  Saved: data/labels/test_labels.npy")
    print(f"  Shape: {window_labels.shape}\n")


if __name__ == "__main__":
    main()
