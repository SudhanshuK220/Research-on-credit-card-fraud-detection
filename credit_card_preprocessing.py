import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (classification_report, confusion_matrix,
                             matthews_corrcoef, average_precision_score,
                             precision_recall_curve)
import os
import json

def preprocess_data(test_size=0.20):
    
    file_path = 'creditcard.csv'

    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return None, None, None, None

    print("Loading dataset...")
    df = pd.read_csv(file_path)

    print("\n--- Step 1: Feature Engineering ---")
    df['Hour'] = df['Time'].apply(lambda x: np.ceil(float(x) / 3600) % 24)
    df = df.drop(['Time'], axis=1)

    X = df.drop('Class', axis=1)
    y = df['Class']

    print(f"\n--- Step 2: Stratified Train-Test Split ({int((1-test_size)*100)}/{int(test_size*100)}) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )

    print("\n--- Step 3: Feature Scaling ---")
    scaler = RobustScaler()
    cols_to_scale = ['Amount', 'Hour']
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    print("\n--- Step 4: Handling Class Imbalance (SMOTE) ---")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("\n Preprocessing Complete! Ready for modeling.")
    return X_train_resampled, X_test, y_train_resampled, y_test


def evaluate_model(model_name, y_true, y_pred, roc_auc=None, y_scores=None):
    """Prints evaluation report AND saves it to a JSON document
    """
    print(f"\n--- {model_name} Evaluation ---")

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    # --- NEW: Matthews Correlation Coefficient ---
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f" MCC (Matthews Correlation Coefficient): {mcc:.4f}")

    # --- NEW: PR-AUC (Average Precision Score) ---
    pr_auc = None
    if y_scores is not None:
        pr_auc = average_precision_score(y_true, y_scores)
        print(f" PR-AUC (Average Precision Score)     : {pr_auc:.4f}")

    if roc_auc is not None:
        print(f" ROC-AUC                              : {roc_auc:.4f}")

    # Save to JSON
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    results = {
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "mcc": mcc
    }

    if roc_auc is not None:
        results["roc_auc_score"] = roc_auc
    if pr_auc is not None:
        results["pr_auc_score"] = pr_auc

    if y_scores is not None:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        optimal_f1 = f1_scores[optimal_idx]
        
        y_pred_opt = (y_scores >= optimal_threshold).astype(int)
        cm_opt = confusion_matrix(y_true, y_pred_opt)
        
        tn_opt, fp_opt, fn_opt, tp_opt = cm_opt.ravel()
        expected_cost = int((fn_opt * 10) + (fp_opt * 1))
        
        print(f"\n--- Optimal F1 Threshold Tuning ---")
        print(f" Optimal F1 Threshold : {optimal_threshold:.4f}")
        print(f" Optimal F1 Score     : {optimal_f1:.4f}")
        print(f" Optimal Confusion Matrix:\n{cm_opt}")
        print(f" Expected Cost (FN=10, FP=1): {expected_cost}")
        
        results["optimal_threshold"] = float(optimal_threshold)
        results["optimal_f1"] = float(optimal_f1)
        results["expected_cost"] = expected_cost
        
        results["pr_curve_precisions"] = precisions.tolist()
        results["pr_curve_recalls"] = recalls.tolist()
        results["pr_curve_thresholds"] = thresholds.tolist()

    file_name = "model_performance_results.json"

    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            try:
                all_results = json.load(f)
            except json.JSONDecodeError:
                all_results = {}
    else:
        all_results = {}

    all_results[model_name] = results

    with open(file_name, 'w') as f:
        json.dump(all_results, f, indent=4)

    print(f"\n Results for '{model_name}' successfully saved to '{file_name}'!")
