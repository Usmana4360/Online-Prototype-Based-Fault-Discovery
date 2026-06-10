# scripts/diagnose.py
"""
Diagnostic tool: why is GCL losing to the 3-Sigma SPC baseline?

Produces:
  results/diagnose_error_vs_labels.png
      Top    : GCL reconstruction error per window, with anomaly windows shaded
               and the mu+3sigma threshold drawn. If the curve does NOT rise
               inside the shaded regions, the model is not separating drift
               from normal -> representation problem, not a thresholding one.
      Bottom : All four methods' (min-max normalized) scores overlaid on the
               same shaded anomaly windows for visual comparison.

  results/diagnose_separation.json
      For each method: mean score on normal windows vs mean score on anomaly
      windows, and a simple separation ratio. A good detector has a large gap.

Run AFTER:
  python -m scripts.create_labels
  python -m scripts.inference          (writes global_reconstruction_error.csv)
  python -m scripts.baseline_spc
  python -m scripts.baseline_iforest
  python -m scripts.baseline_ocsvm
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

LABEL_PATH = "data/labels/test_labels.npy"
GCL_PATH   = "results/global_reconstruction_error.csv"
GCL_COL    = "global_reconstruction_error"

METHODS = [
    {"name": "GCL (Ours)",       "path": None,                          "color": "#0D9488", "gcl": True},
    {"name": "3-Sigma SPC",      "path": "results/scores_spc.npy",      "color": "#DC2626"},
    {"name": "Isolation Forest", "path": "results/scores_iforest.npy",  "color": "#2563EB"},
    {"name": "One-Class SVM",    "path": "results/scores_ocsvm.npy",    "color": "#7C3AED"},
]


def minmax(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-12)


def shade_anomalies(ax, labels):
    """Shade contiguous runs of anomaly windows."""
    labels = np.asarray(labels).astype(int)
    in_run = False
    start = 0
    first = True
    for i, v in enumerate(labels):
        if v == 1 and not in_run:
            in_run, start = True, i
        elif v == 0 and in_run:
            ax.axvspan(start, i, color="#F59E0B", alpha=0.18,
                       label="Anomaly window" if first else None)
            in_run, first = False, False
    if in_run:
        ax.axvspan(start, len(labels), color="#F59E0B", alpha=0.18,
                   label="Anomaly window" if first else None)


def load_all():
    labels = np.load(LABEL_PATH)
    gcl = pd.read_csv(GCL_PATH)[GCL_COL].values

    loaded = []
    for m in METHODS:
        scores = gcl if m.get("gcl") else np.load(m["path"])
        n = min(len(scores), len(labels))
        if len(scores) != len(labels):
            print(f"  NOTE: {m['name']} has {len(scores)} scores vs "
                  f"{len(labels)} labels -> truncating to {n}")
        loaded.append({**m, "scores": scores[:n]})
    labels = labels[:min(len(labels), min(len(d['scores']) for d in loaded))]
    # re-truncate every method to the final common length
    for d in loaded:
        d["scores"] = d["scores"][:len(labels)]
    return labels, loaded


def main():
    os.makedirs("results", exist_ok=True)
    labels, methods = load_all()
    print(f"\nDiagnosing on {len(labels)} windows "
          f"({int(labels.sum())} anomaly / {int((labels==0).sum())} normal)\n")

    # ---------- Separation report ----------
    sep = {}
    norm_mask = labels == 0
    anom_mask = labels == 1
    for d in methods:
        s = d["scores"]
        mean_norm = float(np.mean(s[norm_mask]))
        mean_anom = float(np.mean(s[anom_mask]))
        std_norm  = float(np.std(s[norm_mask]) + 1e-12)
        sep[d["name"]] = {
            "roc_auc":            round(float(roc_auc_score(labels, s)), 4),
            "mean_score_normal":  round(mean_norm, 6),
            "mean_score_anomaly": round(mean_anom, 6),
            # how many normal-std's apart are the two class means:
            "separation_sigmas":  round((mean_anom - mean_norm) / std_norm, 3),
        }
        print(f"  {d['name']:<18}  AUC={sep[d['name']]['roc_auc']:.4f}  "
              f"sep={sep[d['name']]['separation_sigmas']:+.2f} sigma  "
              f"(normal mu={mean_norm:.4g}, anomaly mu={mean_anom:.4g})")

    with open("results/diagnose_separation.json", "w") as f:
        json.dump(sep, f, indent=2)
    print("\nSaved: results/diagnose_separation.json")

    # ---------- Figure ----------
    gcl = next(d for d in methods if d.get("gcl"))
    gcl_scores = gcl["scores"]
    mu, sigma = gcl_scores.mean(), gcl_scores.std()
    thr = mu + 3 * sigma

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Top: raw GCL error + threshold + shaded anomalies
    shade_anomalies(ax_top, labels)
    ax_top.plot(gcl_scores, color="#0D9488", linewidth=1.2,
                label="GCL reconstruction error")
    ax_top.axhline(thr, color="#0D9488", linestyle="--", linewidth=1.2,
                   label=f"mu+3sigma threshold = {thr:.3f}")
    ax_top.axhline(mu, color="#94A3B8", linestyle=":", linewidth=1,
                   label=f"mean = {mu:.3f}")
    ax_top.set_ylabel("Reconstruction MSE")
    ax_top.set_title("GCL Reconstruction Error vs Ground-Truth Anomaly Windows\n"
                     "(error should spike inside the shaded regions)",
                     fontweight="bold")
    ax_top.legend(loc="upper left", framealpha=0.9)
    ax_top.grid(alpha=0.3)

    # Bottom: all methods normalized for shape comparison
    shade_anomalies(ax_bot, labels)
    for d in methods:
        ax_bot.plot(minmax(d["scores"]), color=d["color"], linewidth=1.0,
                    alpha=0.85, label=d["name"])
    ax_bot.set_ylabel("Score (min-max normalized)")
    ax_bot.set_xlabel("Window index")
    ax_bot.set_title("All Methods (normalized) vs Anomaly Windows",
                     fontweight="bold")
    ax_bot.legend(loc="upper left", framealpha=0.9, ncol=2)
    ax_bot.grid(alpha=0.3)

    plt.tight_layout()
    out = "results/diagnose_error_vs_labels.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")

    # ---------- Verdict ----------
    print("\n" + "-" * 60)
    gcl_sep = sep["GCL (Ours)"]["separation_sigmas"]
    spc_sep = sep["3-Sigma SPC"]["separation_sigmas"]
    print("INTERPRETATION:")
    if gcl_sep < spc_sep:
        print(f"  GCL separates anomalies by {gcl_sep:+.2f} sigma, but SPC does")
        print(f"  better at {spc_sep:+.2f} sigma. The autoencoder is")
        print("  reconstructing anomaly windows almost as well as normal ones")
        print("  -> bottleneck too generous OR adversarial term adding noise OR")
        print("     train/eval distribution mismatch. See suggestions in chat.")
    else:
        print(f"  GCL separation ({gcl_sep:+.2f} sigma) >= SPC ({spc_sep:+.2f}).")
        print("  If AUC is still lower, the issue is threshold/operating-point,")
        print("  not representation.")
    print("-" * 60)


if __name__ == "__main__":
    main()
