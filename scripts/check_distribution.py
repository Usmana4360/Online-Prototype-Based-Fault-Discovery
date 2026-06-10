# scripts/check_distribution.py
"""
Checks whether the TRAIN file and the EVAL file share the same feature
distribution. If they don't, the GCL reconstruction error reflects
domain shift (different operating point / scaling), not anomalies --
which would explain why a trivial z-score baseline beats it.

Run: python -m scripts.check_distribution
"""
import numpy as np
import pandas as pd
from src.config import FEATURE_COLS

TRAIN_CSV = "data/raw/55kw_motor_data.csv"
EVAL_CSV  = "data/raw/motor_data_200_drifts_labeled.csv"

def main():
    tr = pd.read_csv(TRAIN_CSV)
    ev = pd.read_csv(EVAL_CSV)

    print(f"Train rows: {len(tr):,}   Eval rows: {len(ev):,}\n")
    print(f"{'feature':<16}{'train_mean':>12}{'eval_mean':>12}"
          f"{'train_std':>12}{'eval_std':>12}{'mean_shift_sigma':>18}")
    print("-" * 82)

    big_shift = []
    for col in FEATURE_COLS:
        if col not in tr.columns or col not in ev.columns:
            print(f"{col:<16}  MISSING in one of the files")
            continue
        tm, em = tr[col].mean(), ev[col].mean()
        ts, es = tr[col].std() + 1e-9, ev[col].std()
        # how far the eval mean sits from train mean in train-std units
        shift = abs(em - tm) / ts
        flag = "  <-- shifted" if shift > 1.0 else ""
        if shift > 1.0:
            big_shift.append(col)
        print(f"{col:<16}{tm:>12.4g}{em:>12.4g}{ts:>12.4g}{es:>12.4g}"
              f"{shift:>18.2f}{flag}")

    print("-" * 82)
    if big_shift:
        print(f"\n>> {len(big_shift)} feature(s) shifted by >1 train-sigma: "
              f"{big_shift}")
        print(">> The scaler fit on the TRAIN file will not center the EVAL")
        print("   data, so reconstruction error is partly measuring this shift,")
        print("   not real anomalies. Consider: (a) train+eval from the same")
        print("   distribution, or (b) re-fit the scaler on the eval file's")
        print("   normal region, or (c) confirm both files are the same sensor.")
    else:
        print("\n>> Distributions look comparable. Train/eval mismatch is "
              "NOT the cause; look at the bottleneck size and GAN ablation.")

if __name__ == "__main__":
    main()
