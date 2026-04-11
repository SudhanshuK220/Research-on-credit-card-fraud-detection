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

def preprocess_data(test_size=0.30):
    """Handles data loading, engineering, splitting, scaling, and SMOTE.
       test_size: fraction for test set (default 0.30 = 70/30 split)
    """
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
    """Prints evaluation report AND saves it to a JSON document.
       Now includes MCC and PR-AUC (Average Precision Score).
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
