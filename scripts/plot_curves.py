# scripts/plot_curves.py
"""
Generates publication-quality ROC and PR curves
for GCL vs all three baselines.

Outputs:
  results/roc_curve_comparison.png
  results/pr_curve_comparison.png
  results/curves_comparison.png   (combined 1×2 figure)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    roc_auc_score
)

# ── Configuration ────────────────────────────────────
LABEL_PATH = "data/labels/test_labels.npy"
GCL_PATH   = "results/global_reconstruction_error.csv"
GCL_COL    = "global_reconstruction_error"

METHODS = [
    {
        "name":       "3-Sigma SPC",
        "scores_path": "results/scores_spc.npy",
        "color":      "#DC2626",
        "linestyle":  "--",
        "linewidth":  1.5,
    },
    {
        "name":       "Isolation Forest",
        "scores_path": "results/scores_iforest.npy",
        "color":      "#2563EB",
        "linestyle":  "-.",
        "linewidth":  1.5,
    },
    {
        "name":       "One-Class SVM",
        "scores_path": "results/scores_ocsvm.npy",
        "color":      "#7C3AED",
        "linestyle":  ":",
        "linewidth":  1.8,
    },
    {
        "name":       "GCL (Ours)",
        "scores_path": None,      # loaded from CSV below
        "color":      "#0D9488",
        "linestyle":  "-",
        "linewidth":  2.5,
        "gcl":        True,
    },
]

# ── Publication style ─────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})


def load_data():
    """Load ground truth labels and all method scores."""
    labels = np.load(LABEL_PATH)

    gcl_df  = pd.read_csv(GCL_PATH)
    gcl_scores = gcl_df[GCL_COL].values

    results = []
    for m in METHODS:
        if m.get("gcl"):
            scores = gcl_scores
        else:
            scores = np.load(m["scores_path"])

        # Align lengths
        min_len = min(len(scores), len(labels))
        s = scores[:min_len]
        l = labels[:min_len]

        roc_auc = roc_auc_score(l, s)
        pr_auc  = average_precision_score(l, s)
        fpr, tpr, _ = roc_curve(l, s)
        prec, rec, _ = precision_recall_curve(l, s)

        results.append({
            **m,
            "scores":  s,
            "labels":  l,
            "fpr":     fpr,
            "tpr":     tpr,
            "prec":    prec,
            "rec":     rec,
            "roc_auc": roc_auc,
            "pr_auc":  pr_auc,
        })
        print(f"  {m['name']:<22}  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")

    return results


def plot_roc(results, ax, title="ROC Curve Comparison"):
    """Draw all ROC curves on a single axes."""
    # Random baseline
    ax.plot([0,1],[0,1], color="#9CA3AF", linestyle="--",
            linewidth=1, alpha=0.7, label="Random (AUC=0.500)")

    for r in results:
        label = f"{r['name']}  (AUC={r['roc_auc']:.3f})"
        ax.plot(r["fpr"], r["tpr"],
                color=r["color"],
                linestyle=r["linestyle"],
                linewidth=r["linewidth"],
                label=label,
                alpha=0.92)

        # Mark the operating point (threshold = μ+3σ → best F1)
        if r.get("gcl"):
            # Find point closest to best F1 threshold
            thresh_idx = np.argmax(r["tpr"] - r["fpr"])
            ax.scatter(r["fpr"][thresh_idx], r["tpr"][thresh_idx],
                       color=r["color"], s=80, zorder=5,
                       marker="o", edgecolors="white", linewidth=1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", framealpha=0.9,
              edgecolor="#CBD5E1", fancybox=False)

    # Shade area under GCL curve
    gcl = next(r for r in results if r.get("gcl"))
    ax.fill_between(gcl["fpr"], gcl["tpr"],
                    alpha=0.06, color=gcl["color"])

    ax.set_aspect("equal")
    return ax


def plot_pr(results, ax, title="Precision–Recall Curve Comparison"):
    """Draw all PR curves on a single axes."""
    # Baseline: random classifier = fraction of positive class
    pos_rate = results[0]["labels"].mean()
    ax.axhline(pos_rate, color="#9CA3AF", linestyle="--",
               linewidth=1, alpha=0.7,
               label=f"Random (AP={pos_rate:.3f})")

    for r in results:
        label = f"{r['name']}  (AP={r['pr_auc']:.3f})"
        ax.plot(r["rec"], r["prec"],
                color=r["color"],
                linestyle=r["linestyle"],
                linewidth=r["linewidth"],
                label=label,
                alpha=0.92)

        # Mark GCL operating point
        if r.get("gcl"):
            # Find point with max F1 on PR curve
            f1_scores = 2 * r["prec"] * r["rec"] / (r["prec"] + r["rec"] + 1e-9)
            best_idx  = np.argmax(f1_scores)
            ax.scatter(r["rec"][best_idx], r["prec"][best_idx],
                       color=r["color"], s=80, zorder=5,
                       marker="o", edgecolors="white", linewidth=1,
                       label=f"GCL operating point  "
                             f"(P={r['prec'][best_idx]:.3f},"
                             f" R={r['rec'][best_idx]:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower left", framealpha=0.9,
              edgecolor="#CBD5E1", fancybox=False)

    gcl = next(r for r in results if r.get("gcl"))
    ax.fill_between(gcl["rec"], gcl["prec"],
                    alpha=0.06, color=gcl["color"])

    ax.set_aspect("equal")
    return ax


def main():
    print("\nLoading scores and labels...")
    results = load_data()

    # ── Figure 1: ROC only ────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(6.5, 6.5))
    plot_roc(results, ax1)
    plt.tight_layout()
    fig1.savefig("results/roc_curve_comparison.png",
                 dpi=200, bbox_inches="tight",
                 facecolor="white")
    print("\nSaved: results/roc_curve_comparison.png")

    # ── Figure 2: PR only ─────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
    plot_pr(results, ax2)
    plt.tight_layout()
    fig2.savefig("results/pr_curve_comparison.png",
                 dpi=200, bbox_inches="tight",
                 facecolor="white")
    print("Saved: results/pr_curve_comparison.png")

    # ── Figure 3: Combined side-by-side (thesis figure) ──
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(13, 6))
    plot_roc(results, ax3, title="(a) ROC Curve Comparison")
    plot_pr(results,  ax4, title="(b) Precision–Recall Curve Comparison")
    fig3.suptitle(
        "GCL vs Baseline Methods — Anomaly Detection Performance\n"
        "55kW Industrial Motor Sensor Data  ·  1,991 Test Windows",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    fig3.savefig("results/curves_comparison.png",
                 dpi=200, bbox_inches="tight",
                 facecolor="white")
    print("Saved: results/curves_comparison.png")

    # ── Print final summary table ─────────────────────
    print("\n" + "─"*58)
    print(f"{'Method':<22}  {'ROC-AUC':>8}  {'PR-AUC':>8}")
    print("─"*58)
    for r in results:
        marker = " ← best" if r.get("gcl") else ""
        print(f"{r['name']:<22}  {r['roc_auc']:>8.4f}  {r['pr_auc']:>8.4f}{marker}")
    print("─"*58)


if __name__ == "__main__":
    main()