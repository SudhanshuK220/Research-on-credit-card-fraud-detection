import os
import sys

#  CHECK DATASET 
def _check_dataset():
    if not os.path.exists('creditcard.csv'):
        print("\n ERROR: 'creditcard.csv' not found in this folder.")
        print("   Download it from: https://www.kaggle.com/mlg-ulb/creditcardfraud")
        sys.exit(1)


def main():
    print("\n" + "=" * 60)
    print("   CREDIT CARD FRAUD DETECTION — FULL RESEARCH PIPELINE")
    print("=" * 60)

    _check_dataset()

    # Train All Models
    print("\n[STEP 1/5] Training all 5 models...")
    from credit_card_preprocessing import preprocess_data, evaluate_model
    from train_SVM import train_svm
    from train_KNN import train_knn
    from train_DecisionTree import train_decision_tree
    from train_ensemble_models import train_random_forest, train_xgboost

    X_train, X_test, y_train, y_test = preprocess_data(test_size=0.30) 

    if X_train is not None:
        train_svm(X_train, y_train, X_test, y_test)
        train_knn(X_train, y_train, X_test, y_test)
        train_decision_tree(X_train, y_train, X_test, y_test)
        train_random_forest(X_train, y_train, X_test, y_test)
        train_xgboost(X_train, y_train, X_test, y_test)
        print("\n All 5 models trained and saved to model_performance_results.json")

    # GENERATE EVALUATION TABLE
    print("\n[STEP 2/5] Generating comparative evaluation table...")
    from evaluation_table import generate_comparative_table
    generate_comparative_table()

    # K-FOLD CROSS-VALIDATION + STATISTICAL TESTS
    print("\n[STEP 3/5] Running 10-Fold CV + Wilcoxon tests (takes around 10-20 min)...")
    from kfold_cross_validation import run_kfold_cv, print_summary, run_statistical_tests, save_results
    all_scores   = run_kfold_cv(n_splits=10)
    print_summary(all_scores)
    stat_results_f1 = run_statistical_tests(all_scores, metric='f1')
    stat_results_mcc = run_statistical_tests(all_scores, metric='mcc')
    stat_results_pr_auc = run_statistical_tests(all_scores, metric='pr_auc')
    save_results(all_scores, stat_results_f1, stat_results_mcc, stat_results_pr_auc)

    # SHAP ANALYSIS
    print("\n[STEP 4/5] Running SHAP Explainability Analysis (SVM/KNN may take around 5-10 min)...")
    from shap_explainability import run_shap_analysis
    run_shap_analysis()

    # RESEARCH VISUALS
    print("\n[STEP 5/5] Generating comparison bar charts...")
    from research_visuals import create_comparison_chart, generate_confusion_matrices
    create_comparison_chart()
    generate_confusion_matrices()

    print("\n" + "=" * 60)
    print("  ALL STEPS COMPLETE!")
    print("=" * 60)
    print("""
Files generated:
  model_performance_results.json   — Raw results (all 5 models)
  comparative_evaluation_table.csv — Full metrics table (incl. MCC, PR-AUC)
  kfold_results.json               — K-Fold CV results per fold
  kfold_summary.csv                — K-Fold Mean ± Std summary table
  shap_beeswarm.png                — SHAP beeswarm plots (per model)
  shap_bar.png                     — SHAP bar charts (per model)
  shap_waterfall.png               — SHAP waterfall plots (per model)
  shap_heatmap_all_models.png      — Cross-model SHAP heatmap
  chart.png                        — Metric comparison bar charts
  paper_confusion_matrices.png     — Confusion matrix heatmaps
""")


if __name__ == "__main__":
    main()
