import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score,
                             matthews_corrcoef, accuracy_score)
from imblearn.over_sampling import SMOTE
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings('ignore')



#  DATA LOADING (without SMOTE — applied inside fold)
def load_data():
    import pandas as pd_inner
    df = pd_inner.read_csv('creditcard.csv')
    df['Hour'] = df['Time'].apply(lambda x: np.ceil(float(x) / 3600) % 24)
    df = df.drop(['Time'], axis=1)
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    feature_names = list(df.drop('Class', axis=1).columns)
    return X, y, feature_names


#  MODEL DEFINITIONS

def get_models():
    return {
        "Support Vector Machine": CalibratedClassifierCV(
            LinearSVC(random_state=42, dual=False, max_iter=1500), cv=3),
        "Decision Tree":          DecisionTreeClassifier(random_state=42, max_leaf_nodes=1500),
        "K-Nearest Neighbors":    KNeighborsClassifier(n_neighbors=3, n_jobs=-1),
        "Random Forest":          RandomForestClassifier(n_estimators=100, max_depth=20,
                                                         random_state=42, n_jobs=-1,
                                                         class_weight='balanced'),
        "XGBoost": __import__('xgboost').XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, n_jobs=-1)
    }



#  10-FOLD CROSS-VALIDATION

def run_kfold_cv(n_splits=10):
    print(f"\n{'='*60}")
    print(f"  RUNNING {n_splits}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"{'='*60}")

    X, y, _ = load_data()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = get_models()
    smote = SMOTE(random_state=42)
    scaler = RobustScaler()

    # Store per-fold scores for each model
    all_scores = {name: {"f1": [], "roc_auc": [], "pr_auc": [], "mcc": [], "accuracy": []}
                  for name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold_idx + 1}/{n_splits}...")
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        # Scale Amount and Hour (columns -2 and -1 after dropping Time)
        amount_hour_idx = [-2, -1]
        X_train_fold[:, amount_hour_idx] = scaler.fit_transform(X_train_fold[:, amount_hour_idx])
        X_test_fold[:, amount_hour_idx]  = scaler.transform(X_test_fold[:, amount_hour_idx])

        # Apply SMOTE only on training fold
        X_res, y_res = smote.fit_resample(X_train_fold, y_train_fold)

        neg = (y_res == 0).sum()
        pos = (y_res == 1).sum()

        for model_name, model in models.items():
            # XGBoost needs scale_pos_weight set per fold
            if "XGBoost" in model_name:
                model.set_params(scale_pos_weight=neg / pos)

            model.fit(X_res, y_res)
            y_pred  = model.predict(X_test_fold)
            y_score = model.predict_proba(X_test_fold)[:, 1]

            all_scores[model_name]["f1"].append(
                f1_score(y_test_fold, y_pred, zero_division=0))
            all_scores[model_name]["roc_auc"].append(
                roc_auc_score(y_test_fold, y_score))
            all_scores[model_name]["pr_auc"].append(
                average_precision_score(y_test_fold, y_score))
            all_scores[model_name]["mcc"].append(
                matthews_corrcoef(y_test_fold, y_pred))
            all_scores[model_name]["accuracy"].append(
                accuracy_score(y_test_fold, y_pred))

    return all_scores



#  SUMMARY TABLE

def print_summary(all_scores):
    print(f"\n{'='*80}")
    print("  K-FOLD CROSS-VALIDATION RESULTS  (Mean ± Std)")
    print(f"{'='*80}")

    rows = []
    for model_name, scores in all_scores.items():
        rows.append({
            "Model":    model_name,
            "Accuracy": f"{np.mean(scores['accuracy']):.4f} ± {np.std(scores['accuracy']):.4f}",
            "F1-Score": f"{np.mean(scores['f1']):.4f} ± {np.std(scores['f1']):.4f}",
            "ROC-AUC":  f"{np.mean(scores['roc_auc']):.4f} ± {np.std(scores['roc_auc']):.4f}",
            "PR-AUC":   f"{np.mean(scores['pr_auc']):.4f} ± {np.std(scores['pr_auc']):.4f}",
            "MCC":      f"{np.mean(scores['mcc']):.4f} ± {np.std(scores['mcc']):.4f}",
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))



#  WILCOXON SIGNED-RANK TEST

def run_statistical_tests(all_scores, metric='f1'):
    """
    Compares every pair of models using the Wilcoxon signed-rank test.
    p < 0.05 means the difference is statistically significant.
    """
    print(f"\n{'='*60}")
    print(f"  WILCOXON SIGNED-RANK TEST  (metric: {metric.upper()})")
    print(f"  p < 0.05 → statistically significant difference")
    print(f"{'='*60}")

    model_names = list(all_scores.keys())
    results = {}

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            scores1 = all_scores[m1][metric]
            scores2 = all_scores[m2][metric]

            try:
                stat, p_val = wilcoxon(scores1, scores2)
                significant = " YES" if p_val < 0.05 else " NO"
            except ValueError:
                stat, p_val, significant = 0, 1.0, " Identical (cannot test)"

            key = f"{m1} vs {m2}"
            results[key] = {"statistic": float(stat), "p_value": float(round(p_val, 6)),
                            "significant": bool(p_val < 0.05)}
            print(f"  {m1:30s} vs {m2:30s}  |  p={p_val:.4f}  |  Significant: {significant}")

    return results



#  SAVE RESULTS

def save_results(all_scores, stat_results):
    output = {}
    for model_name, scores in all_scores.items():
        output[model_name] = {
            metric: {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
                "per_fold": [float(v) for v in vals]
            }
            for metric, vals in scores.items()
        }
    output["statistical_tests"] = stat_results

    with open("kfold_results.json", "w") as f:
        json.dump(output, f, indent=4)
    print("\n K-Fold results saved to 'kfold_results.json'!")

    # export a clean CSV summary
    rows = []
    for model_name, scores in all_scores.items():
        rows.append({
            "Model":        model_name,
            "Accuracy_Mean": round(np.mean(scores['accuracy']), 4),
            "Accuracy_Std":  round(np.std(scores['accuracy']), 4),
            "F1_Mean":       round(np.mean(scores['f1']), 4),
            "F1_Std":        round(np.std(scores['f1']), 4),
            "ROC_AUC_Mean":  round(np.mean(scores['roc_auc']), 4),
            "ROC_AUC_Std":   round(np.std(scores['roc_auc']), 4),
            "PR_AUC_Mean":   round(np.mean(scores['pr_auc']), 4),
            "PR_AUC_Std":    round(np.std(scores['pr_auc']), 4),
            "MCC_Mean":      round(np.mean(scores['mcc']), 4),
            "MCC_Std":       round(np.std(scores['mcc']), 4),
        })
    pd.DataFrame(rows).to_csv("kfold_summary.csv", index=False)
    print(" K-Fold summary saved to 'kfold_summary.csv'!")




if __name__ == "__main__":
    all_scores   = run_kfold_cv(n_splits=10)
    print_summary(all_scores)
    stat_results = run_statistical_tests(all_scores, metric='f1')
    save_results(all_scores, stat_results)
    print("\n K-Fold Cross Validation + Statistical Tests complete!")
