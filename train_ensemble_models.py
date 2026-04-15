import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score, matthews_corrcoef
)
import xgboost as xgb
import json
import os
import warnings
warnings.filterwarnings('ignore')

from credit_card_preprocessing import preprocess_data, evaluate_model


def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Trains Random Forest with class_weight='balanced'.
    Saves results and generates feature importance chart.
    """
    print("\n" + "-"*50)
    print("  Training Random Forest...")
    print("-"*50)

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    print("  Evaluating Random Forest on the testing set...")
    y_pred       = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc  = average_precision_score(y_test, y_pred_proba)
    mcc     = matthews_corrcoef(y_test, y_pred)

    print(f"\n  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  MCC     : {mcc:.4f}")

    evaluate_model("Random Forest", y_test, y_pred, roc_auc=roc_auc, y_scores=y_pred_proba)
    _append_extra_metrics("Random Forest", mcc, pr_auc)
    _plot_feature_importance(rf_model.feature_importances_,
                             X_train.columns.tolist(),
                             "Random Forest",
                             "RF_Feature_Importance.png")

    return rf_model, y_pred, y_pred_proba


def train_xgboost(X_train, y_train, X_test, y_test):
    """
    Trains XGBoost with scale_pos_weight and early stopping.
    Uses PR-AUC as eval metric (better for imbalanced fraud data).
    """
    print("\n" + "-"*50)
    print("  Training XGBoost...")
    print("-"*50)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  scale_pos_weight = {scale_pos_weight:.2f}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='aucpr',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1
    )

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    print(f"  Best iteration: {xgb_model.best_iteration}")

    print("  Evaluating XGBoost on the testing set...")
    y_pred       = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc  = average_precision_score(y_test, y_pred_proba)
    mcc     = matthews_corrcoef(y_test, y_pred)

    print(f"\n  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  MCC     : {mcc:.4f}")

    evaluate_model("XGBoost", y_test, y_pred, roc_auc=roc_auc, y_scores=y_pred_proba)
    _append_extra_metrics("XGBoost", mcc, pr_auc)
    _plot_feature_importance(xgb_model.feature_importances_,
                             X_train.columns.tolist(),
                             "XGBoost",
                             "XGB_Feature_Importance.png")

    return xgb_model, y_pred, y_pred_proba


def _append_extra_metrics(model_name, mcc, pr_auc):
    """Appends MCC and PR-AUC to the shared model_performance_results.json."""
    json_file = "model_performance_results.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = {}

    if model_name not in all_results:
        all_results[model_name] = {}

    all_results[model_name]["mcc"]         = mcc
    all_results[model_name]["pr_auc_score"] = pr_auc

    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=4)


def _plot_feature_importance(importances, feature_names, model_name, save_path, top_n=15):
    """Plots top-N feature importances as a horizontal bar chart."""
    indices     = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors  = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    ax.barh(range(top_n), top_values[::-1], color=colors[::-1])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=11)
    ax.set_xlabel("Feature Importance Score", fontsize=13, fontweight='bold')
    ax.set_title(f"{model_name} — Top {top_n} Feature Importances",
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_full_comparison_table():
    """Reads JSON and produces a 5-model CSV + F1 bar chart including MCC & PR-AUC."""
    import pandas as pd

    json_file = "model_performance_results.json"
    if not os.path.exists(json_file):
        print("  JSON not found. Run training scripts first.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)

    rows = []
    for model_name, metrics in data.items():
        cr    = metrics.get('classification_report', {})
        fraud = cr.get('1', {})
        
        cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        rows.append({
            "Model":             model_name,
            "Accuracy":          round(cr.get('accuracy', 0), 4),
            "Precision (Fraud)": round(fraud.get('precision', 0), 4),
            "Recall (Fraud)":    round(fraud.get('recall', 0), 4),
            "F1-Score (Fraud)":  round(fraud.get('f1-score', 0), 4),
            "FP":                fp,
            "FN":                fn,
            "FPR":               round(fpr, 4),
            "FNR":               round(fnr, 4),
            "ROC-AUC":           round(metrics.get('roc_auc_score', 0), 4),
            "PR-AUC":            round(metrics.get('pr_auc_score', metrics.get('pr_auc', 0)), 4),
            "MCC":               round(metrics.get('mcc', 0), 4),
            "Opt. Threshold":    round(metrics.get('optimal_threshold', 0), 4),
            "Expected Cost":     metrics.get('expected_cost', 0)
        })

    df = pd.DataFrame(rows)
    print("\n\n" + "="*60)
    print("  MODEL COMPARISON TABLE")
    print("="*60)
    print(df.to_string(index=False))
    df.to_csv("comparative_evaluation_table.csv", index=False)
    print("\n  Saved: comparative_evaluation_table.csv")

    # F1-Score bar chart — all 5 models
    fig, ax = plt.subplots(figsize=(12, 6))
    colors     = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
    short_names = [n.replace(" (LinearSVC)", "\n(LinearSVC)") for n in df["Model"]]
    bars = ax.bar(short_names, df["F1-Score (Fraud)"], color=colors[:len(df)], width=0.5)

    for bar, val in zip(bars, df["F1-Score (Fraud)"]):
        ax.annotate(f"{val:.4f}",
                    (bar.get_x() + bar.get_width() / 2., bar.get_height()),
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    xytext=(0, 5), textcoords='offset points')

    ax.set_title("F1-Score (Fraud) — All 5 Models Comparison",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Machine Learning Models", fontsize=13, fontweight='bold')
    ax.set_ylabel("F1-Score (Fraud Class)", fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(df["F1-Score (Fraud)"]) * 1.25)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("All5_F1Score_Comparison.png", dpi=300)
    plt.close()
    print("  Saved: All5_F1Score_Comparison.png")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("   ENSEMBLE MODELS: RANDOM FOREST + XGBOOST")
    print("="*60)

    print("\nLoading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data()

    if X_train is None:
        print("Error: Preprocessing failed.")
        exit()

    train_random_forest(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)
    generate_full_comparison_table()

    print("\n" + "="*60)
    print("   Ensemble Training Complete!")
    print("="*60)
