"""
shap_explainability.py
──────────────────────
Generates SHAP-based explainability visualisations for all 5 models.
Saves high-resolution plots suitable for an IEEE research paper.

Plots generated:
  1. shap_beeswarm_<model>.png     – feature importance + direction
  2. shap_bar_<model>.png          – mean |SHAP| bar chart
  3. shap_waterfall_<model>.png    – single-prediction explanation
  4. shap_heatmap_all_models.png   – top-10 features across all models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from credit_card_preprocessing import preprocess_data


# ─────────────────────────────────────────────
#  TRAIN ALL MODELS
# ─────────────────────────────────────────────
def train_all_models(X_train, y_train):
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    models = {
        "Decision_Tree": DecisionTreeClassifier(random_state=42, max_leaf_nodes=1500),
        "K_Nearest_Neighbors": KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        "Support_Vector_Machine": CalibratedClassifierCV(
            LinearSVC(random_state=42, dual=False, max_iter=1500), cv=3),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=42,
            n_jobs=-1, class_weight='balanced'),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=neg / pos,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1)
    }

    print("\nTraining all models for SHAP analysis...")
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
    print("All models trained!\n")
    return models


# ─────────────────────────────────────────────
#  GET SHAP EXPLAINER (correct type per model)
# ─────────────────────────────────────────────
def get_explainer(model_name, model, X_background):
    if "XGBoost" in model_name or "Random_Forest" in model_name or "Decision_Tree" in model_name:
        return shap.TreeExplainer(model)
    else:
       # KernelExplainer for SVM and KNN — use K-Means background for massive speedup
        background = shap.kmeans(X_background, 50)
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        return shap.KernelExplainer(predict_fn, background)


def normalize_shap_values(shap_values, n_features):
    # Handle list output from binary classification explainers first
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

    shap_values = np.array(shap_values)

    if shap_values.ndim == 3:
        # Some explainers return (n_samples, n_features, n_classes)
        if shap_values.shape[1] == n_features:
            shap_values = shap_values[:, :, -1]
        elif shap_values.shape[2] == n_features:
            shap_values = shap_values[:, :, -1]
        elif shap_values.shape[0] == n_features:
            shap_values = shap_values[0]
        else:
            shap_values = shap_values.reshape(shap_values.shape[0], -1)

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    return shap_values


# ─────────────────────────────────────────────
#  PLOT: BEESWARM SUMMARY
# ─────────────────────────────────────────────
def plot_beeswarm(shap_values, X_sample, feature_names, model_name):
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title(f"SHAP Summary (Beeswarm) – {model_name.replace('_', ' ')}",
              fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    path = f"shap_beeswarm_{model_name}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  PLOT: BAR CHART (Mean |SHAP|)
# ─────────────────────────────────────────────
def plot_bar(shap_values, X_sample, feature_names, model_name):
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP Feature Importance – {model_name.replace('_', ' ')}",
              fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    path = f"shap_bar_{model_name}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  PLOT: WATERFALL (Single high-risk prediction)
# ─────────────────────────────────────────────
def plot_waterfall(explainer, X_sample, feature_names, model_name):
    try:
        sv = explainer(X_sample[:50])
        fraud_idx = int(np.argmax(np.abs(sv.values).sum(axis=1)))
        plt.figure(figsize=(12, 7))
        shap.waterfall_plot(sv[fraud_idx], max_display=15, show=False)
        plt.title(f"SHAP Waterfall – {model_name.replace('_', ' ')} (High-Risk Prediction)",
                  fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = f"shap_waterfall_{model_name}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    except Exception as e:
        print(f"  Waterfall skipped for {model_name}: {e}")


# ─────────────────────────────────────────────
#  PLOT: CROSS-MODEL HEATMAP
# ─────────────────────────────────────────────
def plot_cross_model_heatmap(mean_shap_dict, feature_names, top_n=10):
    df = pd.DataFrame.from_dict(mean_shap_dict, orient='index', columns=feature_names)
    df = df.T
    df['avg'] = df.mean(axis=1)
    top_features = df.nlargest(top_n, 'avg').drop('avg', axis=1)

    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(top_features.values.T, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(top_features.index)))
    ax.set_xticklabels(top_features.index, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(top_features.columns)))
    ax.set_yticklabels([c.replace('_', ' ') for c in top_features.columns], fontsize=12)
    plt.colorbar(im, ax=ax, label="Mean |SHAP| Value")
    plt.title(f"Cross-Model SHAP Feature Importance Heatmap (Top {top_n} Features)",
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    path = "shap_heatmap_all_models.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def run_shap_analysis():
    print("=" * 60)
    print("  SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 60)

    X_train, X_test, y_train, y_test = preprocess_data()
    if X_train is None:
        return

    feature_names = list(X_train.columns)
    models = train_all_models(X_train, y_train)

    # Use up to 100 test samples (KernelExplainer is slow for SVM/KNN)
    sample_size = min(100, len(X_test))
    X_sample = X_test.iloc[:sample_size]
    X_background = X_train.iloc[:500]

    mean_shap_dict = {}

    for model_name, model in models.items():
        print(f"\n Generating SHAP plots for {model_name.replace('_', ' ')}...")
        explainer = get_explainer(model_name, model, X_background)

        # 1. Isolate KNN to use the minimum statistical sample (30 rows, 50 permutations)
        if model_name == "K_Nearest_Neighbors":
            print("  Applying extreme sample reduction for KNN distance calculations...")
            X_current = X_sample.iloc[:30] 
            shap_values = explainer.shap_values(X_current, nsamples=50)
        else:
            X_current = X_sample
            shap_values = explainer.shap_values(X_current)

        # 2. Normalize the values using your existing function
        shap_values = normalize_shap_values(shap_values, len(feature_names))

        # 3. Pass 'X_current' into the plot functions instead of 'X_sample'
        plot_beeswarm(shap_values, X_current, feature_names, model_name)
        plot_bar(shap_values, X_current, feature_names, model_name)
        plot_waterfall(explainer, X_current, feature_names, model_name)

        mean_shap_dict[model_name] = np.abs(shap_values).mean(axis=0)

    print("\n Generating cross-model heatmap...")
    plot_cross_model_heatmap(mean_shap_dict, feature_names, top_n=10)

    print("\n All SHAP visualisations complete — ready for your IEEE paper!")


if __name__ == "__main__":
    run_shap_analysis()
