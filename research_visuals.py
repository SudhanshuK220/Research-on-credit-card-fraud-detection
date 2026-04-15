import os
import json
import warnings
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
 
warnings.filterwarnings("ignore")
 
# ─────────────────────────────────────────────────────────────────────────────
# SHARED STYLE
# ─────────────────────────────────────────────────────────────────────────────
 
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.linestyle":   "--",
    "grid.alpha":       0.45,
    "figure.dpi":       150,
})
 
# Display names used consistently on every axis
MODEL_DISPLAY = {
    "Support Vector Machine":             "SVM",
    "Support Vector Machine (LinearSVC)": "SVM",
    "K-Nearest Neighbors":                "KNN",
    "Decision Tree":                      "Decision Tree",
    "Random Forest":                      "Random Forest",
    "XGBoost":                            "XGBoost",
}
 
# One colour per model – used identically in every figure
MODEL_COLOURS = {
    "SVM":           "#3498db",
    "KNN":           "#9b59b6",
    "Decision Tree": "#2ecc71",
    "Random Forest": "#e67e22",
    "XGBoost":       "#e74c3c",
}
 
METRIC_COLOURS = {
    "Accuracy":          "#3498db",
    "Precision (Fraud)": "#9b59b6",
    "Recall (Fraud)":    "#2ecc71",
    "F1-Score (Fraud)":  "#e74c3c",
    "ROC-AUC":           "#f39c12",
    "PR-AUC":            "#8B4513",
    "MCC":               "#320453",
}
 
 
def _short(name: str) -> str:
    """Return a short display name for a model."""
    return MODEL_DISPLAY.get(name, name)
 
 
def _colour(short_name: str) -> str:
    return MODEL_COLOURS.get(short_name, "#7f8c8d")
 
 
def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        print(f"  [SKIP] '{path}' not found – run the pipeline first.")
        return {}
    with open(path, "r") as f:
        return json.load(f)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  ORIGINAL FUNCTIONS  ════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
 
def create_comparison_chart():
    """Original: individual bar chart per metric from comparative_evaluation_table.csv"""
    csv_file = "comparative_evaluation_table.csv"
    if not os.path.exists(csv_file):
        print(f"  [SKIP] '{csv_file}' not found.")
        return
 
    print(f"\n  Loading '{csv_file}'...")
    df = pd.read_csv(csv_file)
    model_col = "Model" if "Model" in df.columns else df.columns[0]
    df.set_index(model_col, inplace=True)
 
    desired_metrics = [
        "Accuracy", "Precision (Fraud)", "Recall (Fraud)",
        "F1-Score (Fraud)", "ROC-AUC", "PR-AUC", "MCC",
    ]
    actual_columns = [c for c in desired_metrics if c in df.columns]
 
    short_names = {k: MODEL_DISPLAY.get(k, k) for k in df.index}
 
    for metric in actual_columns:
        fig, ax = plt.subplots(figsize=(12, 7))
        color = METRIC_COLOURS.get(metric, "#1abc9c")
        df[metric].plot(kind="bar", ax=ax, color=color, width=0.5)
        ax.set_xticklabels(
            [short_names.get(l, l) for l in df.index], rotation=30, ha="right", fontsize=12
        )
        clean = metric.replace(" (Fraud)", "")
        ax.set_title(f"{clean} Comparison Across Models", fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Machine Learning Models", fontsize=14, fontweight="bold")
        ax.set_ylabel(f"{clean} Score (0.0 – 1.0)", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.25)
        for p in ax.patches:
            h = p.get_height()
            if h > 0:
                ax.annotate(
                    f"{h:.3f}",
                    (p.get_x() + p.get_width() / 2.0, h),
                    ha="center", va="bottom",
                    xytext=(0, 8), textcoords="offset points",
                    fontsize=11, fontweight="bold",
                )
        plt.tight_layout()
        safe = clean.replace(" ", "_").replace("-", "_")
        path = f"{safe}_chart.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")
 
 
def generate_confusion_matrices():
    """Original: side-by-side confusion-matrix heatmaps from model_performance_results.json"""
    data = _load_json("model_performance_results.json")
    if not data:
        return
 
    models = list(data.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
 
    for ax, model_name in zip(axes, models):
        cm = np.array(data[model_name]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    cbar=False, annot_kws={"size": 14})
        ax.set_title(_short(model_name), fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Predicted Label\n(0: Normal, 1: Fraud)", fontsize=12)
        ax.set_ylabel("True Label\n(0: Normal, 1: Fraud)", fontsize=12)
 
    plt.tight_layout()
    path = "paper_confusion_matrices.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ═══════════════  NEW FIGURE 1 – Wilcoxon p-value heatmap  ═══════════════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_wilcoxon_pvalue_heatmap():
    """
    Figure W1 – Pairwise Wilcoxon signed-rank p-value heatmap.
 
    Source : kfold_results.json  (per-fold scores written by
             kfold_cross_validation.py)
 
    One 5×5 heatmap per metric (F1, MCC, PR-AUC).
    Green cell  → p < 0.05 (statistically significant difference)
    Red cell    → p ≥ 0.05 (not significant)
    Diagonal    → grey (self-comparison, not applicable)
    """
    data = _load_json("kfold_results.json")
    if not data:
        return
 
    # Drop the 'statistical_tests' key that kfold_cross_validation.py appends
    model_data = {k: v for k, v in data.items() if not k.startswith("statistical_tests")}
    if not model_data:
        print("  [SKIP] No model entries in kfold_results.json")
        return
 
    models = list(model_data.keys())
    short_labels = [_short(m) for m in models]
    metrics_to_plot = ["f1", "mcc", "pr_auc"]
    metric_titles  = {"f1": "F1-Score", "mcc": "MCC", "pr_auc": "PR-AUC"}
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Pairwise Wilcoxon Signed-Rank Test – p-value Heatmap\n"
        "(Green: p < 0.05 significant | Red: p ≥ 0.05 not significant)",
        fontsize=14, fontweight="bold", y=1.02,
    )
 
    for ax, metric in zip(axes, metrics_to_plot):
        n = len(models)
        pmat = np.full((n, n), np.nan)
 
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s1 = model_data[models[i]][metric]["per_fold"]
                s2 = model_data[models[j]][metric]["per_fold"]
                try:
                    _, p = wilcoxon(s1, s2)
                except ValueError:
                    p = 1.0
                pmat[i, j] = p
 
        # Build colour matrix: green (<0.05) / red (≥0.05) / grey (diagonal)
        cmap = plt.cm.RdYlGn_r   # low p → green end after we invert
        # We'll draw manually for full control
        ax.set_facecolor("#f5f5f5")
        for i in range(n):
            for j in range(n):
                if i == j:
                    rect = plt.Rectangle((j, n - 1 - i), 1, 1,
                                         color="#cccccc", zorder=1)
                    ax.add_patch(rect)
                    ax.text(j + 0.5, n - 0.5 - i, "—",
                            ha="center", va="center", fontsize=11, color="#888888")
                else:
                    p = pmat[i, j]
                    color = "#27ae60" if p < 0.05 else "#e74c3c"
                    rect = plt.Rectangle((j, n - 1 - i), 1, 1,
                                         color=color, alpha=0.75, zorder=1)
                    ax.add_patch(rect)
                    ax.text(j + 0.5, n - 0.5 - i, f"{p:.3f}",
                            ha="center", va="center", fontsize=9,
                            fontweight="bold", color="white", zorder=2)
 
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(short_labels, rotation=35, ha="right", fontsize=10)
        ax.set_yticklabels(short_labels[::-1], fontsize=10)
        ax.set_title(metric_titles[metric], fontsize=13, fontweight="bold", pad=10)
        ax.grid(False)
        ax.tick_params(length=0)
 
    # Legend patches
    sig_patch   = mpatches.Patch(color="#27ae60", alpha=0.75, label="p < 0.05 (significant)")
    nsig_patch  = mpatches.Patch(color="#e74c3c", alpha=0.75, label="p ≥ 0.05 (not significant)")
    fig.legend(handles=[sig_patch, nsig_patch],
               loc="lower center", ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, -0.07), frameon=False)
 
    plt.tight_layout()
    path = "wilcoxon_pvalue_heatmap.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ═══════════  NEW FIGURE 2 – Per-fold score distribution box plots  ══════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_kfold_boxplots():
    """
    Figure W2 – Per-fold score distribution box plots.
 
    Source : kfold_results.json
 
    One panel per metric (MCC, PR-AUC, F1).  Whiskers = min/max,
    box = IQR, orange line = median.  Colour-coded by model.
    """
    data = _load_json("kfold_results.json")
    if not data:
        return
 
    model_data = {k: v for k, v in data.items() if not k.startswith("statistical_tests")}
    if not model_data:
        return

    models = list(model_data.keys())
    short_labels = [_short(m) for m in models]
    colours = [_colour(s) for s in short_labels]
    metrics = [("mcc", "MCC"), ("pr_auc", "PR-AUC"), ("f1", "F1-Score")]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
    fig.suptitle(
        "10-Fold Cross-Validation – Per-Fold Score Distribution",
        fontsize=15, fontweight="bold", y=1.01,
    )

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        fold_scores = [model_data[m][metric_key]["per_fold"] for m in models]
 
        bp = ax.boxplot(
            fold_scores,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
            widths=0.55,
        )
        for patch, c in zip(bp["boxes"], colours):
            patch.set_facecolor(c)
            patch.set_alpha(0.72)
 
        ax.set_xticks(range(1, len(short_labels) + 1))
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=11)
        ax.set_title(metric_label, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel(f"{metric_label} score", fontsize=11)
        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--")
 
    plt.tight_layout()
    path = "kfold_boxplots_mcc_prauc_f1.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ═══════  NEW FIGURE 3 – PR curves with optimal-threshold markers  ════════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_pr_curves_with_threshold():
    """
    Figure T1 – Precision-Recall curves (all 5 models overlaid) with a
    coloured dot marking the optimal F1 threshold on each curve.
 
    Source : model_performance_results.json
             The JSON must contain 'pr_curve_precisions', 'pr_curve_recalls',
             'pr_curve_thresholds', and 'optimal_threshold'.
 
    NOTE:  If the JSON was written by evaluate_model() in
           credit_card_preprocessing.py WITHOUT storing the full curve arrays,
           this function re-derives them from the stored threshold + metrics.
           For the full curve you need to save precisions/recalls in
           evaluate_model() – see the comment inside the function.
    """
    data = _load_json("model_performance_results.json")
    if not data:
        return
 
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_title(
        "Precision–Recall Curves with Optimal F1 Threshold Markers",
        fontsize=14, fontweight="bold", pad=12,
    )
 
    plotted = 0
    for model_name, metrics in data.items():
        short = _short(model_name)
        colour = _colour(short)
 
        # ── If full PR curve arrays were saved by evaluate_model() ──
        if ("pr_curve_precisions" in metrics and
                "pr_curve_recalls" in metrics):
            precisions = np.array(metrics["pr_curve_precisions"])
            recalls    = np.array(metrics["pr_curve_recalls"])
            thresholds = np.array(metrics.get("pr_curve_thresholds", []))
            pr_auc     = metrics.get("pr_auc_score", metrics.get("pr_auc", None))
            opt_thresh = metrics.get("optimal_threshold", None)
 
        # ── Fallback: only summary scalars available ─────────────────
        else:
            pr_auc     = metrics.get("pr_auc_score", metrics.get("pr_auc", None))
            opt_thresh = metrics.get("optimal_threshold", None)
            precisions = recalls = thresholds = None
 
            if pr_auc is None:
                print(f"  [WARN] No PR-AUC data for {short}, skipping curve.")
                continue
 
            # Draw a single marker + a short annotation instead of a full curve
            # so the figure is still useful even without curve arrays
            label = f"{short}  (PR-AUC={pr_auc:.3f})"
            ax.scatter([], [], color=colour, s=60, label=label)
            if opt_thresh is not None:
                opt_f1 = metrics.get("optimal_f1", None)
                note = f"τ={opt_thresh:.3f}"
                if opt_f1:
                    note += f"  F1={opt_f1:.3f}"
                ax.annotate(f"{short}: {note}", xy=(0.02, 0.5 - plotted * 0.07),
                            xycoords="axes fraction", fontsize=9, color=colour)
            plotted += 1
            continue
 
        label = f"{short}  (PR-AUC={pr_auc:.3f})" if pr_auc else short
        ax.plot(recalls, precisions, color=colour, linewidth=2, label=label)
 
        # Mark optimal threshold point
        if opt_thresh is not None and len(thresholds) > 0:
            f1s = (2 * precisions[:-1] * recalls[:-1]) / (
                precisions[:-1] + recalls[:-1] + 1e-10)
            idx = np.argmax(f1s)
            ax.scatter(recalls[idx], precisions[idx],
                       s=90, color=colour, zorder=5,
                       edgecolors="black", linewidths=0.8)
            ax.annotate(
                f"τ={opt_thresh:.3f}",
                xy=(recalls[idx], precisions[idx]),
                xytext=(recalls[idx] - 0.08, precisions[idx] + 0.04),
                fontsize=8, color=colour,
                arrowprops=dict(arrowstyle="->", color=colour, lw=0.8),
            )
        plotted += 1
 
    ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
 
    # Baseline (random classifier for imbalanced dataset ≈ fraud rate)
    ax.axhline(0.5, color="#aaaaaa", linestyle="--", linewidth=0.8,
               label="Baseline (random)")
 
    plt.tight_layout()
    path = "pr_curves_with_threshold_markers.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    if plotted == 0:
        print("  [INFO] No full PR curve arrays found in model_performance_results.json.")
        print("         Add 'pr_curve_precisions', 'pr_curve_recalls', "
              "'pr_curve_thresholds' to evaluate_model() to get full curves.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ════════  NEW FIGURE 4 – Threshold vs F1-score (5-panel)  ════════════════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_threshold_vs_f1():
    """
    Figure T2 – Threshold vs F1-score curves (5-panel, one per model).
 
    Source : model_performance_results.json
             Needs 'pr_curve_precisions', 'pr_curve_recalls',
             'pr_curve_thresholds' and 'optimal_threshold'.
 
    If the full arrays are not stored, a note is printed and the figure
    is skipped so it doesn't crash the rest of the pipeline.
    """
    data = _load_json("model_performance_results.json")
    if not data:
        return
 
    models = list(data.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
 
    fig.suptitle(
        "F1-Score vs Decision Threshold (per model)\n"
        "Vertical dashed line = optimal F1 threshold",
        fontsize=14, fontweight="bold", y=1.02,
    )
 
    any_plotted = False
    for ax, model_name in zip(axes, models):
        short  = _short(model_name)
        colour = _colour(short)
        metrics = data[model_name]
 
        if ("pr_curve_precisions" not in metrics or
                "pr_curve_recalls" not in metrics or
                "pr_curve_thresholds" not in metrics):
            ax.text(0.5, 0.5, "Full PR curve\nnot stored",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#aaaaaa")
            ax.set_title(short, fontsize=12, fontweight="bold")
            continue
 
        precisions = np.array(metrics["pr_curve_precisions"])
        recalls    = np.array(metrics["pr_curve_recalls"])
        thresholds = np.array(metrics["pr_curve_thresholds"])
        opt_thresh = metrics.get("optimal_threshold", None)
 
        # F1 at each threshold (exclude last precision/recall which has no threshold)
        f1s = (2 * precisions[:-1] * recalls[:-1]) / (
            precisions[:-1] + recalls[:-1] + 1e-10)
 
        ax.plot(thresholds, f1s, color=colour, linewidth=2)
        ax.fill_between(thresholds, f1s, alpha=0.12, color=colour)
 
        if opt_thresh is not None:
            ax.axvline(opt_thresh, color=colour, linestyle="--",
                       linewidth=1.5, label=f"τ*={opt_thresh:.3f}")
            ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
 
        ax.set_xlabel("Decision threshold (τ)", fontsize=10)
        ax.set_title(short, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        any_plotted = True
 
    axes[0].set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = "threshold_vs_f1_curves.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    if not any_plotted:
        print("  [INFO] Add 'pr_curve_precisions', 'pr_curve_recalls', "
              "'pr_curve_thresholds' to evaluate_model() for full curves.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ═══  NEW FIGURE 5 – Default vs optimal threshold metric comparison  ══════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_default_vs_optimal_threshold():
    """
    Figure T3 – Grouped bar chart: metrics at threshold=0.5 vs optimal threshold.
 
    Source : model_performance_results.json
             Uses classification_report (default 0.5) and optimal_f1,
             plus the confusion matrix at the optimal threshold which is
             derived here from stored values.
 
    Metrics compared: Precision (fraud class), Recall (fraud class), F1
    """
    data = _load_json("model_performance_results.json")
    if not data:
        return
 
    rows = []
    for model_name, metrics in data.items():
        report = metrics.get("classification_report", {})
        fraud  = report.get("1", {})
        prec_def  = fraud.get("precision", None)
        rec_def   = fraud.get("recall",    None)
        f1_def    = fraud.get("f1-score",  None)
        f1_opt    = metrics.get("optimal_f1", None)
 
        if any(v is None for v in [prec_def, rec_def, f1_def, f1_opt]):
            continue
 
        # We don't have prec/rec at opt threshold unless the training scripts
        # saved them.  We extract what we can: F1 comparison is always possible.
        rows.append({
            "model":    _short(model_name),
            "F1 @ 0.5": f1_def,
            "F1 @ τ*":  f1_opt,
            "Precision @ 0.5": prec_def,
            "Recall @ 0.5":    rec_def,
        })
 
    if not rows:
        print("  [SKIP] No suitable data for default-vs-optimal chart.")
        return
 
    df = pd.DataFrame(rows).set_index("model")
 
    # Plot F1 comparison (always available) + Precision/Recall at default
    metrics_to_show = ["Precision @ 0.5", "Recall @ 0.5", "F1 @ 0.5", "F1 @ τ*"]
    available = [m for m in metrics_to_show if m in df.columns]
 
    x    = np.arange(len(df))
    width = 0.18
    colours_bars = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12"]
 
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (col, col_colour) in enumerate(zip(available, colours_bars)):
        offset = (i - len(available) / 2 + 0.5) * width
        bars = ax.bar(x + offset, df[col], width, label=col,
                      color=col_colour, alpha=0.82, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=8,
                    fontweight="bold")
 
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Default Threshold (0.5) vs Optimal F1 Threshold (τ*) – Metric Comparison",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)
    ax.axhline(1.0, color="#cccccc", linewidth=0.8, linestyle="--")
 
    plt.tight_layout()
    path = "default_vs_optimal_threshold_comparison.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ══════  NEW FIGURE 6 – Expected cost per model (horizontal bar)  ═════════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_expected_cost_bar():
    """
    Figure C1 – Horizontal bar chart of expected cost per model.
 
    Expected cost = FN_count × 10  +  FP_count × 1
    (stored in model_performance_results.json as 'expected_cost')
 
    Bars are ranked lowest→highest cost (lowest = best for deployment).
    Colour: green (lowest cost) → red (highest cost).
    """
    data = _load_json("model_performance_results.json")
    if not data:
        return
 
    costs = {}
    for model_name, metrics in data.items():
        ec = metrics.get("expected_cost", None)
        if ec is not None:
            costs[_short(model_name)] = ec
 
    if not costs:
        print("  [SKIP] No 'expected_cost' found in model_performance_results.json.")
        return
 
    # Sort ascending (lowest cost first → best at top in horizontal bar)
    sorted_items = sorted(costs.items(), key=lambda x: x[1])
    labels  = [item[0] for item in sorted_items]
    values  = [item[1] for item in sorted_items]
    n       = len(labels)
 
    # Colour: gradient green → red proportional to rank
    palette = plt.cm.RdYlGn(np.linspace(0.85, 0.15, n))
 
    fig, ax = plt.subplots(figsize=(10, 0.9 * n + 2.5))
    bars = ax.barh(labels, values, color=palette, edgecolor="white",
                   height=0.55, alpha=0.88)
 
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(val):,}", va="center", ha="left",
                fontsize=11, fontweight="bold")
 
    ax.set_xlabel("Expected Cost  (FN × 10 + FP × 1)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Expected Misclassification Cost per Model\n"
        "(lower is better – ranked best to worst)",
        fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlim(0, max(values) * 1.18)
    ax.invert_yaxis()   # best (lowest cost) at top
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)
 
    plt.tight_layout()
    path = "expected_cost_per_model.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ══════  NEW FIGURE 7 – FP vs FN scatter with cost iso-lines  ════════════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_fp_fn_scatter_isolines():
    """
    Figure C2 – Scatter plot: FP count (x) vs FN count (y), one point per model.
    Dashed iso-cost lines show equal-cost frontiers (FN×10 + FP×1 = const).
    """
    data = _load_json("model_performance_results.json")
    if not data:
        return
 
    points = {}
    for model_name, metrics in data.items():
        cm = metrics.get("confusion_matrix", None)
        if cm is None:
            continue
        tn, fp, fn, tp = (np.array(cm)).ravel()
        points[_short(model_name)] = (int(fp), int(fn))
 
    if not points:
        print("  [SKIP] No confusion matrix data found.")
        return
 
    fps = [v[0] for v in points.values()]
    fns = [v[1] for v in points.values()]
 
    x_max = max(fps) * 1.35 + 10
    y_max = max(fns) * 1.35 + 5
 
    fig, ax = plt.subplots(figsize=(9, 7))
 
    # Iso-cost lines: FN*10 + FP*1 = C  →  FN = (C - FP) / 10
    all_costs = [int(v * 10 + u * 1) for u, v in zip(fps, fns)]
    c_min, c_max = min(all_costs), max(all_costs)
    iso_levels = np.linspace(c_min * 0.6, c_max * 1.3, 6)
 
    fp_range = np.linspace(0, x_max, 300)
    for c_level in iso_levels:
        fn_line = (c_level - fp_range) / 10.0
        mask = (fn_line >= 0) & (fn_line <= y_max)
        ax.plot(fp_range[mask], fn_line[mask],
                color="#aaaaaa", linestyle="--", linewidth=0.9, alpha=0.7,
                zorder=1)
        # Label the iso-line at the right edge
        if mask.sum() > 0:
            idx = np.where(mask)[0][-1]
            ax.text(fp_range[idx] + x_max * 0.005, fn_line[idx],
                    f"cost={int(c_level):,}",
                    fontsize=7.5, color="#888888", va="center")
 
    # Plot model points
    for model_name, (fp_val, fn_val) in points.items():
        colour = _colour(model_name)
        ax.scatter(fp_val, fn_val, s=120, color=colour,
                   edgecolors="black", linewidths=0.8, zorder=4)
        ax.annotate(
            model_name,
            xy=(fp_val, fn_val),
            xytext=(fp_val + x_max * 0.02, fn_val + y_max * 0.02),
            fontsize=10, fontweight="bold", color=colour,
            arrowprops=dict(arrowstyle="-", color=colour, lw=0.8),
            zorder=5,
        )
 
    ax.set_xlabel("False Positives (FP)", fontsize=13, fontweight="bold")
    ax.set_ylabel("False Negatives (FN)", fontsize=13, fontweight="bold")
    ax.set_title(
        "FP vs FN Trade-off with Cost Iso-lines\n"
        "(dashed lines = equal expected cost; FN cost = 10 × FP cost)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
 
    plt.tight_layout()
    path = "fp_fn_scatter_isolines.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ══════  NEW FIGURE 8 – Cost breakdown stacked bar (FP + FN cost)  ═══════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_cost_breakdown_stacked_bar():
    """
    Figure C3 – Stacked bar chart: FP cost (FP×1) vs FN cost (FN×10) per model.
    Makes the 10× asymmetry immediately visible.
    """
    data = _load_json("model_performance_results.json")
    if not data:
        return
 
    rows = []
    for model_name, metrics in data.items():
        cm = metrics.get("confusion_matrix", None)
        if cm is None:
            continue
        tn, fp, fn, tp = np.array(cm).ravel()
        rows.append({
            "model":    _short(model_name),
            "FP cost (× 1)":   int(fp * 1),
            "FN cost (× 10)":  int(fn * 10),
        })
 
    if not rows:
        print("  [SKIP] No confusion matrix data found for cost breakdown.")
        return
 
    df = pd.DataFrame(rows).set_index("model")
    # Sort by total cost descending so worst model is on the left
    df["total"] = df["FP cost (× 1)"] + df["FN cost (× 10)"]
    df = df.sort_values("total", ascending=False).drop(columns="total")
 
    fig, ax = plt.subplots(figsize=(11, 6))
    df.plot(kind="bar", stacked=True, ax=ax,
            color=["#3498db", "#e74c3c"],
            width=0.55, edgecolor="white", alpha=0.85)
 
    # Annotate total cost on top of each bar
    totals = df.sum(axis=1)
    for i, (model_name, total) in enumerate(totals.items()):
        ax.text(i, total + totals.max() * 0.01, f"{int(total):,}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
 
    ax.set_xticklabels(df.index, rotation=25, ha="right", fontsize=12)
    ax.set_ylabel("Cost (arbitrary units)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Cost Breakdown: FP Cost vs FN Cost per Model\n"
        "(FN cost = 10 × FP cost — FN dominates due to asymmetric penalty)",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
 
    plt.tight_layout()
    path = "cost_breakdown_stacked_bar.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ═══  NEW FIGURE 9 – CV MCC & PR-AUC grouped bar with ±std error bars  ════════
# ─────────────────────────────────────────────────────────────────────────────
 
def plot_cv_mcc_prauc_errorbars():
    """
    Figure CV1 – MCC and PR-AUC mean ± 1 std from 10-fold CV.
 
    Source : kfold_results.json  (written by kfold_cross_validation.py)
 
    Grouped bar chart: two groups (MCC, PR-AUC), one bar per model,
    with black error caps showing ±1 std.
    """
    data = _load_json("kfold_results.json")
    if not data:
        return
 
    model_data = {k: v for k, v in data.items() if not k.startswith("statistical_tests")}
    if not model_data:
        return

    models = list(model_data.keys())
    short_labels = [_short(m) for m in models]
    colours = [_colour(s) for s in short_labels]

    mcc_means   = [model_data[m]["mcc"]["mean"]    for m in models]
    mcc_stds    = [model_data[m]["mcc"]["std"]     for m in models]
    prauc_means = [model_data[m]["pr_auc"]["mean"] for m in models]
    prauc_stds  = [model_data[m]["pr_auc"]["std"]  for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width / 2, mcc_means, width,
                   yerr=mcc_stds, capsize=5,
                   color=colours, alpha=0.75,
                   error_kw=dict(elinewidth=1.4, ecolor="black", capthick=1.4),
                   label="MCC", edgecolor="white")
 
    bars2 = ax.bar(x + width / 2, prauc_means, width,
                   yerr=prauc_stds, capsize=5,
                   color=colours, alpha=0.40,
                   hatch="///", edgecolor="white",
                   error_kw=dict(elinewidth=1.4, ecolor="black", capthick=1.4),
                   label="PR-AUC")
 
    # Value annotations
    for bar, mean, std in zip(bars1, mcc_means, mcc_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.008,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
 
    for bar, mean, std in zip(bars2, prauc_means, prauc_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.008,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
 
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Mean Score ± 1 Std (10-fold CV)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "10-Fold Cross-Validation: MCC and PR-AUC (Mean ± 1 Std)\n"
        "Solid bars = MCC  |  Hatched bars = PR-AUC",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
 
    plt.tight_layout()
    path = "cv_mcc_prauc_errorbars.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# ════════════════════════  MAIN ENTRY POINT  ══════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 65)
    print("  Generating ALL paper-ready visualisations")
    print("=" * 65)
 
    print("\n── ORIGINAL VISUALS ─────────────────────────────────────────")
    create_comparison_chart()
    generate_confusion_matrices()
 
    print("\n── WILCOXON STATISTICAL TESTING ─────────────────────────────")
    plot_wilcoxon_pvalue_heatmap()        # Figure W1
    plot_kfold_boxplots()                 # Figure W2
 
    print("\n── OPTIMAL F1 THRESHOLD ─────────────────────────────────────")
    plot_pr_curves_with_threshold()       # Figure T1
    plot_threshold_vs_f1()               # Figure T2
    plot_default_vs_optimal_threshold()  # Figure T3
 
    print("\n── COST-SENSITIVE EVALUATION ────────────────────────────────")
    plot_expected_cost_bar()              # Figure C1
    plot_fp_fn_scatter_isolines()         # Figure C2
    plot_cost_breakdown_stacked_bar()     # Figure C3
 
    print("\n── CROSS-VALIDATION SUMMARY ENHANCEMENT ────────────────────")
    plot_cv_mcc_prauc_errorbars()         # Figure CV1
 
    print("\n" + "=" * 65)
    print("  All visualisations complete!")
    print("=" * 65)
 
    plt.show()