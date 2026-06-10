# scripts/baseline_paper_ae.py
"""
SOTA comparison baseline.

Faithful reimplementation of the method from:
  Ferreira et al., "Unsupervised Autoencoder-Based Anomaly Detection Under
  Limited Failure Data for Oil Industry," IEEE Open J. Instrum. Meas., 2026.

What is reproduced from the paper (NOT the paper's repo, which is Keras and
hardwired to their pump data):
  - Per-TIMESTAMP fully-connected symmetric autoencoder (input dim = n_features),
    NOT a windowed/conv model. (Paper Sec. III-C, Fig. 2, Table 4.)
  - Anomaly score  AS(x_i) = || x_i - x_hat_i ||^2          (Eq. 3)
  - Moving-minimum anomaly score over last n=10 instances    (Eq. 4)
        MMAS(x_i) = min{ AS(x_i), ..., AS(x_{i-9}) }
  - Acceptance threshold  tau = 1.5 * max(MMAS_train)        (Eq. 5)
  - Training hyperparameters from Table 5:
        optimizer=Adam, loss=MSE, ReLU hidden, linear output,
        epochs=200, batch=512, val_split=20%, shuffle=True
  - z-score normalization fit on the training rows only      (Eq. 2)

What is intentionally NOT carried over (pump-specific, no analog in motor data):
  - pressure-differential feature (no suction/discharge pair)
  - idle-state filtering (motor data is not on/off scheduled like the pumps)

How it is made comparable to your SPC / IsolationForest / OneClassSVM rows:
  - Trains on the SAME first-70%-of-windows region those baselines use
    (unsupervised, no label leakage).
  - The paper is per-row; your labels are per-window. We compute MMAS per row,
    then take the MAX MMAS inside each window as that window's score. This
    matches your window-labeling rule ("window = anomaly if ANY row is anomaly").
  - Metrics reported at BOTH the best-F1 threshold (apples-to-apples with the
    other baselines) and the paper's own tau rule (faithful to the paper).

Output (mirrors baseline_iforest.py):
  results/scores_paper_ae.npy        # per-window MMAS score, for plot_curves.py
  results/baseline_paper_ae.json     # metrics
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, matthews_corrcoef,
    precision_score, recall_score,
)

from src.config import FEATURE_COLS, CLIP_LEN, STRIDE

CSV_PATH   = "data/raw/motor_data_200_drifts_labeled.csv"
LABEL_PATH = "data/labels/test_labels.npy"

# Paper-style hyperparameters (Table 5)
EPOCHS       = 200
BATCH_SIZE   = 512
LR           = 1e-3            # Adam default; paper does not specify, 1e-3 is standard
VAL_SPLIT    = 0.20
N_MMAS       = 10             # moving-minimum window length (paper used n=10)
TRAIN_FRAC   = 0.70          # same split as baseline_iforest / baseline_ocsvm
SEED         = 42

# Symmetric dense AE. Same hidden topology as the paper's Unit-1 net
# (7/6/5/6/7); only the input/output width changes to your 9 features.
HIDDEN_DIMS  = [7, 6, 5, 6, 7]


class DenseAutoencoder(nn.Module):
    """Fully-connected symmetric AE, ReLU hidden, linear output (paper Table 5)."""
    def __init__(self, n_features, hidden_dims):
        super().__init__()
        dims = [n_features] + list(hidden_dims)
        enc = []
        for i in range(len(dims) - 1):
            enc += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        # decoder mirrors encoder; final layer is linear (no activation)
        dec_dims = list(reversed(dims))
        dec = []
        for i in range(len(dec_dims) - 1):
            dec.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec.append(nn.ReLU())
        self.net = nn.Sequential(*enc, *dec)

    def forward(self, x):
        return self.net(x)


def moving_minimum(arr, n):
    """MMAS: for each i, min over [i-n+1 .. i] (clipped at the start)."""
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - n + 1)
        out[i] = arr[lo:i + 1].min()
    return out


def metrics_at_threshold(scores, labels, preds, name):
    return {
        "f1":        round(f1_score(labels, preds, zero_division=0), 4),
        "mcc":       round(matthews_corrcoef(labels, preds), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds, zero_division=0), 4),
        "n_alarms":  int(preds.sum()),
        "_at":       name,
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df     = pd.read_csv(CSV_PATH)
    labels = np.load(LABEL_PATH)
    X_all  = df[FEATURE_COLS].values.astype(np.float32)

    # --- window bookkeeping (identical to your other baselines) ---
    window_starts   = list(range(0, len(df) - CLIP_LEN + 1, STRIDE))
    n_train_windows = int(TRAIN_FRAC * len(window_starts))
    # rows covered by the training windows -> the per-row training region
    train_end_row   = window_starts[n_train_windows - 1] + CLIP_LEN
    X_train_rows    = X_all[:train_end_row]

    # --- z-score normalization, fit on training rows only (Eq. 2) ---
    mu    = X_train_rows.mean(axis=0)
    sigma = X_train_rows.std(axis=0) + 1e-8
    X_norm       = (X_all - mu) / sigma
    X_train_norm = (X_train_rows - mu) / sigma

    # --- train the dense AE on normal-region rows (Table 5) ---
    Xt = torch.tensor(X_train_norm, device=device)
    n  = len(Xt)
    perm    = torch.randperm(n)
    n_val   = int(VAL_SPLIT * n)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    model = DenseAutoencoder(len(FEATURE_COLS), HIDDEN_DIMS).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    mse   = nn.MSELoss()

    print(f"Training dense AE on {len(tr_idx)} rows "
          f"(val {len(val_idx)}), {EPOCHS} epochs, batch {BATCH_SIZE}...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_idx = tr_idx[torch.randperm(len(tr_idx))]
        for b in range(0, len(epoch_idx), BATCH_SIZE):
            batch = Xt[epoch_idx[b:b + BATCH_SIZE]]
            opt.zero_grad()
            loss = mse(model(batch), batch)
            loss.backward()
            opt.step()
        if (epoch + 1) % 50 == 0:
            model.eval()
            with torch.no_grad():
                vloss = mse(model(Xt[val_idx]), Xt[val_idx]).item()
            print(f"  epoch {epoch+1:3d} | val MSE {vloss:.6f}")

    # --- per-row anomaly score AS = ||x - x_hat||^2  (Eq. 3) ---
    model.eval()
    with torch.no_grad():
        X_t   = torch.tensor(X_norm, device=device)
        recon = model(X_t).cpu().numpy()
    AS_row = ((X_norm - recon) ** 2).sum(axis=1)          # squared L2 per row

    # --- MMAS per row (Eq. 4) ---
    MMAS_row = moving_minimum(AS_row, N_MMAS)

    # --- paper threshold tau = 1.5 * max(MMAS_train)  (Eq. 5) ---
    tau = 1.5 * MMAS_row[:train_end_row].max()

    # --- aggregate per-row MMAS -> per-window score (max inside each window) ---
    win_score = np.array([
        MMAS_row[s:s + CLIP_LEN].max() for s in window_starts
    ])

    # align to labels
    m = min(len(win_score), len(labels))
    win_score, y = win_score[:m], labels[:m]
    np.save("results/scores_paper_ae.npy", win_score)

    # --- metrics ---
    roc = round(roc_auc_score(y, win_score), 4)
    pr  = round(average_precision_score(y, win_score), 4)

    # (a) best-F1 threshold, like your other baselines
    cand   = np.percentile(win_score, np.arange(80, 100, 0.5))
    f1s    = [f1_score(y, (win_score > t).astype(int), zero_division=0) for t in cand]
    best_t = cand[int(np.argmax(f1s))]
    best   = metrics_at_threshold(win_score, y,
                                  (win_score > best_t).astype(int), "best_f1")

    # (b) paper's tau rule (window flagged if its max MMAS exceeds tau)
    paper  = metrics_at_threshold(win_score, y,
                                  (win_score > tau).astype(int), "paper_tau")

    results = {
        "method": "Paper Dense-AE + MMAS (Ferreira et al., 2026)",
        "roc_auc": roc,
        "pr_auc":  pr,
        "tau":     float(tau),
        "best_f1_threshold": float(best_t),
        "metrics_best_f1":  best,
        "metrics_paper_tau": paper,
    }

    print("\nPaper Dense-AE + MMAS results:")
    print(f"  roc_auc      : {roc}")
    print(f"  pr_auc       : {pr}")
    print(f"  --- at best-F1 threshold (comparable to other baselines) ---")
    for k in ("f1", "mcc", "precision", "recall"):
        print(f"  {k:<12}: {best[k]}")
    print(f"  --- at paper tau = 1.5*max(MMAS_train) ---")
    for k in ("f1", "mcc", "precision", "recall"):
        print(f"  {k:<12}: {paper[k]}")

    with open("results/baseline_paper_ae.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: results/scores_paper_ae.npy")
    print("Saved: results/baseline_paper_ae.json")


if __name__ == "__main__":
    main()
