# scripts/evaluate.py
"""
Complete evaluation pipeline for GCL anomaly detection system.
Run AFTER inference.py has generated results/*.csv files.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score
)
from scipy import stats
import json

def compute_threshold_metrics(global_errors, ground_truth_labels,
                              thresholds=None):
    """
    Compute F1, MCC, Precision, Recall across multiple thresholds.
    
    Args:
        global_errors: np.array of reconstruction errors
        ground_truth_labels: np.array of binary labels (0=normal, 1=anomaly)
        thresholds: if None, uses percentiles 80-99
    
    Returns:
        dict with metrics at each threshold + best threshold
    """
    if thresholds is None:
        thresholds = np.percentile(global_errors, np.arange(80, 100, 1))
    
    results = []
    for thresh in thresholds:
        preds = (global_errors > thresh).astype(int)
        f1  = f1_score(ground_truth_labels, preds, zero_division=0)
        mcc = matthews_corrcoef(ground_truth_labels, preds)
        prec = precision_score(ground_truth_labels, preds, zero_division=0)
        rec  = recall_score(ground_truth_labels, preds, zero_division=0)
        results.append({
            "threshold": thresh, "f1": f1, "mcc": mcc,
            "precision": prec, "recall": rec,
            "n_alarms": preds.sum()
        })
    
    df = pd.DataFrame(results)
    best_idx = df["f1"].idxmax()
    return df, df.iloc[best_idx]

def compute_roc_pr(global_errors, ground_truth_labels):
    """Compute ROC-AUC and Precision-Recall AUC."""
    roc_auc = roc_auc_score(ground_truth_labels, global_errors)
    pr_auc  = average_precision_score(ground_truth_labels, global_errors)
    
    fpr, tpr, _ = roc_curve(ground_truth_labels, global_errors)
    prec, rec, _ = precision_recall_curve(ground_truth_labels, global_errors)
    
    return {
        "roc_auc": roc_auc, "pr_auc": pr_auc,
        "fpr": fpr, "tpr": tpr,
        "precision_curve": prec, "recall_curve": rec
    }

def bootstrap_ci(global_errors, ground_truth_labels, metric_fn,
                 n_bootstrap=1000, alpha=0.05, seed=42):
    """
    Compute 95% bootstrap confidence interval for any metric.
    
    Proper Reasoning: Reviewers require CI to verify results are not
    due to chance. A single evaluation number is insufficient for publication.
    """
    rng = np.random.RandomState(seed)
    n = len(global_errors)
    scores = []
    
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        score = metric_fn(global_errors[idx], ground_truth_labels[idx])
        scores.append(score)
    
    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))
    mean  = np.mean(scores)
    return mean, lower, upper

def wilcoxon_significance_test(your_errors, baseline_errors, ground_truth):
    """
    Wilcoxon signed-rank test: Are your anomaly scores significantly better
    than the baseline? Use this when comparing your AUC to baseline AUC
    across multiple test folds.
    
    Proper Reasoning: IEEE/Springer reviewers expect p < 0.05 for all
    comparative claims. Without this, "our method outperforms X" is
    an unverifiable claim.
    """
    your_scores    = [roc_auc_score(ground_truth, e) for e in your_errors]
    baseline_scores = [roc_auc_score(ground_truth, e) for e in baseline_errors]
    
    stat, p_value = stats.wilcoxon(your_scores, baseline_scores)
    return stat, p_value

def plot_confusion_matrix(ground_truth, predictions, save_path="results/confusion_matrix.png"):
    cm = confusion_matrix(ground_truth, predictions)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im)
    
    classes = ["Normal", "Anomaly"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes)
    
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix — GCL Anomaly Detector')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")

def plot_roc_pr_curves(curves_dict_list, labels, 
                       save_path="results/roc_pr_curves.png"):
    """
    Plot ROC and PR curves for your model AND all baselines side-by-side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(curves_dict_list)))
    
    for (curves, label, c) in zip(curves_dict_list, labels, colors):
        axes[0].plot(curves["fpr"], curves["tpr"], color=c,
                     label=f"{label} (AUC={curves['roc_auc']:.3f})", linewidth=2)
        axes[1].plot(curves["recall_curve"], curves["precision_curve"], color=c,
                     label=f"{label} (AP={curves['pr_auc']:.3f})", linewidth=2)
    
    axes[0].plot([0,1],[0,1],'k--', alpha=0.5, label='Random')
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves — All Methods"); axes[0].legend()
    
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves — All Methods"); axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")

def main():
    # Load your model's errors
    errors = pd.read_csv("results/test_errors.csv")
    global_errors = errors["errors"].values
    
    # *** YOU MUST PROVIDE GROUND TRUTH LABELS ***
    # If you don't have labeled test data, use fault injection timestamps
    # or a labeled hold-out dataset from the motor manufacturer
    ground_truth = np.load("data/labels/test_labels.npy")
    ground_truth = np.load("data/labels/test_labels.npy")

    # Fix off-by-one — trim to same length
    min_len = min(len(global_errors), len(ground_truth))
    global_errors  = global_errors[:min_len]
    ground_truth   = ground_truth[:min_len]

    print(f"Evaluation on {min_len} windows "
        f"({ground_truth.sum()} anomaly / {(ground_truth==0).sum()} normal)")
    
    print("=" * 60)
    print("GCL ANOMALY DETECTOR — FORMAL EVALUATION REPORT")
    print("=" * 60)
    
    # 1. Threshold metrics
    metrics_df, best = compute_threshold_metrics(global_errors, ground_truth)
    print(f"\nBest threshold: {best['threshold']:.5f}")
    print(f"  F1-Score:           {best['f1']:.4f}")
    print(f"  MCC:                {best['mcc']:.4f}")
    print(f"  Precision:          {best['precision']:.4f}")
    print(f"  Recall:             {best['recall']:.4f}")
    
    # 2. ROC-AUC and PR-AUC
    curves = compute_roc_pr(global_errors, ground_truth)
    print(f"\n  ROC-AUC:            {curves['roc_auc']:.4f}")
    print(f"  PR-AUC:             {curves['pr_auc']:.4f}")
    
    # 3. Bootstrap CI for ROC-AUC
    auc_mean, auc_lo, auc_hi = bootstrap_ci(
        global_errors, ground_truth,
        lambda e, y: roc_auc_score(y, e),
        n_bootstrap=2000
    )
    print(f"\n  ROC-AUC Bootstrap CI (95%): {auc_mean:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
    
    # 4. Plots
    threshold = best['threshold']
    predictions = (global_errors > threshold).astype(int)
    plot_confusion_matrix(ground_truth, predictions)
    
    # Save numeric results
    report = {
        "f1": float(best['f1']),
        "mcc": float(best['mcc']),
        "precision": float(best['precision']),
        "recall": float(best['recall']),
        "roc_auc": float(curves['roc_auc']),
        "pr_auc": float(curves['pr_auc']),
        "roc_auc_ci_95": [float(auc_lo), float(auc_hi)],
        "best_threshold": float(threshold)
    }
    with open("results/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved: results/evaluation_report.json")

if __name__ == "__main__":
    main()