# scripts/create_labels.py
"""
Creates window-level labels from row-level labels in your CSV.

Your data:  20,000 rows, ~200 anomaly rows (drift type, gradual, random spread)
Strategy:   A window is labeled ANOMALY if ANY row inside it is anomalous.
            This is correct for drift — you want to catch the window the
            moment drift begins, not wait until majority of window is drifted.
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.config import CLIP_LEN, STRIDE

# ============================================================
# CONFIGURE THESE TWO LINES FOR YOUR FILE
# ============================================================
CSV_PATH   = "data/raw/motor_data_200_drifts_labeled.csv"         # your labeled CSV path
LABEL_COL  = "Label"                      # column name: 0=normal, 1=anomaly
# ============================================================

def create_window_labels(df, label_col, clip_len, stride):
    """
    Maps row-level binary labels to window-level binary labels.
    
    Rule: window label = 1 if ANY row in that window is anomalous.
    This is the correct strategy for drift-type anomalies where
    early detection is the goal.
    """
    row_labels = df[label_col].values.astype(int)
    
    window_starts = list(range(0, len(df) - clip_len + 1, stride))
    window_labels = np.zeros(len(window_starts), dtype=int)
    
    for i, start in enumerate(window_starts):
        end = start + clip_len
        # ANY anomalous row inside window → window is anomalous
        if row_labels[start:end].sum() > 0:
            window_labels[i] = 1
    
    return window_labels, window_starts


def verify_and_plot(df, window_labels, window_starts, label_col, clip_len):
    """
    Prints a verification report and saves a visual sanity check.
    Always run this before proceeding to evaluate.py.
    """
    n_windows  = len(window_labels)
    n_anomaly  = window_labels.sum()
    n_normal   = n_windows - n_anomaly
    ratio      = n_anomaly / n_windows * 100

    print("=" * 55)
    print("  WINDOW LABEL CREATION REPORT")
    print("=" * 55)
    print(f"  CSV rows          : {len(df):,}")
    print(f"  Anomaly rows      : {df[label_col].sum():,}")
    print(f"  Window size       : {clip_len} rows")
    print(f"  Stride            : {STRIDE} rows")
    print(f"  Total windows     : {n_windows:,}")
    print(f"  Anomaly windows   : {n_anomaly}  ({ratio:.1f}%)")
    print(f"  Normal  windows   : {n_normal}")
    print("=" * 55)

    # Health checks
    if n_anomaly < 30:
        print("⚠  WARNING: Fewer than 30 anomaly windows.")
        print("   Metrics will be unstable. Consider reducing STRIDE")
        print("   in config.py (e.g. STRIDE=5) to get more windows.")
    elif ratio > 40:
        print("⚠  WARNING: >40% anomaly ratio is unusually high.")
        print("   Verify your label column is correct.")
    else:
        print(f"✅ Label distribution looks healthy for evaluation.")

    # Visual: row-level anomaly positions + window-level labels side by side
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=False)

    # Top: raw row-level labels
    row_labels = df[label_col].values
    axes[0].fill_between(range(len(row_labels)), row_labels,
                         color='#E24B4A', alpha=0.7)
    axes[0].set_ylabel("Row Label\n(0=Normal, 1=Anomaly)")
    axes[0].set_title("Row-Level Anomaly Positions in Raw CSV")
    axes[0].set_ylim(-0.1, 1.3)
    axes[0].grid(alpha=0.3)

    # Bottom: window-level labels
    axes[1].fill_between(range(len(window_labels)), window_labels,
                         color='#D85A30', alpha=0.7)
    axes[1].set_ylabel("Window Label\n(0=Normal, 1=Anomaly)")
    axes[1].set_xlabel("Window Index")
    axes[1].set_title(f"Window-Level Labels  ({n_anomaly} anomaly / {n_normal} normal)")
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/label_verification.png", dpi=150, bbox_inches='tight')
    print("\n  Saved: results/label_verification.png")
    print("  ← Open this image and visually confirm anomaly positions look correct")
    print("    before running evaluate.py\n")


def main():
    print(f"\nLoading: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # Validate label column exists
    if LABEL_COL not in df.columns:
        print(f"\n❌ Column '{LABEL_COL}' not found in CSV.")
        print(f"   Available columns: {list(df.columns)}")
        print(f"   Edit LABEL_COL at the top of this script.")
        return

    # Validate binary labels
    unique_vals = df[LABEL_COL].unique()
    if not set(unique_vals).issubset({0, 1}):
        print(f"\n❌ Label column has unexpected values: {unique_vals}")
        print(f"   Expected only 0 (normal) and 1 (anomaly).")
        return

    # Create window labels
    window_labels, window_starts = create_window_labels(
        df, LABEL_COL, CLIP_LEN, STRIDE
    )

    # Verify and plot
    verify_and_plot(df, window_labels, window_starts, LABEL_COL, CLIP_LEN)

    # Save
    os.makedirs("data/labels", exist_ok=True)
    np.save("data/labels/test_labels.npy", window_labels)
    print(f"  Saved: data/labels/test_labels.npy")
    print(f"  Shape: {window_labels.shape}\n")


if __name__ == "__main__":
    main()