"""
evaluation_table.py  (updated)
──────────────────────────────
Generates a comparative evaluation table from model_performance_results.json.
Now includes MCC and PR-AUC columns for IEEE-quality reporting.
"""

import json
import pandas as pd
import os

def generate_comparative_table():
    file_path = "model_performance_results.json"

    if not os.path.exists(file_path):
        print(f"Error: '{file_path}' not found.")
        print("Please ensure you have run your training scripts first.")
        return

    with open(file_path, 'r') as f:
        data = json.load(f)

    table_data = []

    for model_name, metrics in data.items():
        class_1_metrics = metrics['classification_report']['1']

        accuracy  = metrics['classification_report']['accuracy']
        precision = class_1_metrics['precision']
        recall    = class_1_metrics['recall']
        f1_score  = class_1_metrics['f1-score']
        roc_auc   = metrics.get('roc_auc_score', None)

        # NEW metrics
        mcc    = metrics.get('mcc', None)
        pr_auc = metrics.get('pr_auc_score', None)

        table_data.append({
            "Models":             model_name,
            "Accuracy":           accuracy,
            "Precision (Fraud)":  precision,
            "Recall (Fraud)":     recall,
            "F1-Score (Fraud)":   f1_score,
            "ROC-AUC":            roc_auc,
            "PR-AUC":             pr_auc,   # NEW
            "MCC":                mcc       # NEW
        })

    df = pd.DataFrame(table_data)
    df = df.round(4)

    print("\n" + "=" * 90)
    print("                MODEL COMPARATIVE EVALUATION TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90)

    csv_filename = "comparative_evaluation_table.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n Table successfully saved as '{csv_filename}'!")


if __name__ == "__main__":
    generate_comparative_table()
