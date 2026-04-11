# Credit Card Fraud Detection — Research Project
### Comparative Analysis of ML Models | IEEE Publication Ready

---

## Project Structure

```
fraud_detection_project/
│
├── creditcard.csv                    ← Kaggle dataset (you must add this)
│
├── credit_card_preprocessing.py      ← Data loading, SMOTE, scaling, evaluate_model()
├── train_SVM.py                      ← Train Support Vector Machine
├── train_KNN.py                      ← Train K-Nearest Neighbors
├── train_DecisionTree.py             ← Train Decision Tree
├── train_ensemble_models.py          ← Train Random Forest + XGBoost (NEW)
│
├── kfold_cross_validation.py         ← 10-Fold CV + Wilcoxon tests (NEW)
├── shap_explainability.py            ← SHAP plots for all 5 models (NEW)
├── evaluation_table.py               ← Comparative table (now incl. MCC + PR-AUC)
├── research_visuals.py               ← Bar charts + confusion matrix heatmaps
│
├── run_all.py                        ← Master script — runs everything at once
└── requirements.txt                  ← All dependencies
```

---

## Output Files Generated After Running

| File | Description |
|------|-------------|
| `model_performance_results.json` | Raw results for all 5 models |
| `comparative_evaluation_table.csv` | Full metrics table (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, MCC) |
| `kfold_results.json` | Per-fold scores for all models |
| `kfold_summary.csv` | K-Fold Mean ± Std table (use in paper) |
| `shap_beeswarm_*.png` | SHAP beeswarm plots (1 per model) |
| `shap_bar_*.png` | SHAP feature importance bars (1 per model) |
| `shap_waterfall_*.png` | SHAP waterfall — single prediction explanation |
| `shap_heatmap_all_models.png` | Cross-model top-10 features heatmap |
| `Accuracy_chart.png` etc. | Metric bar charts for paper |
| `paper_confusion_matrices.png` | All confusion matrices side by side |

---

## Step-by-Step Setup in VS Code

### STEP 1 — Install Python
Make sure Python 3.9 or higher is installed.
Check by opening VS Code Terminal and running:
```
python --version
```
If not installed, download from: https://www.python.org/downloads/

---

### STEP 2 — Open the Project Folder in VS Code
1. Extract the ZIP file to a folder (e.g., `Desktop/fraud_detection_project`)
2. Open VS Code
3. Click `File` → `Open Folder`
4. Select the extracted `fraud_detection_project` folder
5. Click `Select Folder`

---

### STEP 3 — Add the Dataset
1. Download `creditcard.csv` from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Place `creditcard.csv` directly inside the `fraud_detection_project` folder
   (same level as all the `.py` files)

---

### STEP 4 — Open the Terminal in VS Code
Press:  `` Ctrl + ` ``  (backtick key, top-left of keyboard)
Or go to:  `View` → `Terminal`

---

### STEP 5 — Create a Virtual Environment (Recommended)
In the terminal, run these commands one by one:

**Windows:**
```
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line.

---

### STEP 6 — Install All Dependencies
```
pip install -r requirements.txt
```
This installs everything automatically. It may take 3-5 minutes.

---

### STEP 7 — Select the Virtual Environment in VS Code
1. Press `Ctrl + Shift + P`
2. Type: `Python: Select Interpreter`
3. Choose the one that shows `venv` in the path

---

### STEP 8 — Run the Full Pipeline
To run EVERYTHING at once (recommended):
```
python run_all.py
```

**Expected runtime:**
| Step | Approximate Time |
|------|-----------------|
| Training 5 models | 5–10 minutes |
| 10-Fold Cross Validation | 15–25 minutes |
| SHAP Analysis | 10–20 minutes (SVM/KNN are slow) |
| Charts & Visuals | 1–2 minutes |
| **Total** | **~35–60 minutes** |

---

### STEP 9 — Run Individual Scripts (Optional)
If you want to run just one part:

```bash
# Train individual models only
python train_SVM.py
python train_KNN.py
python train_DecisionTree.py
python train_ensemble_models.py     # Random Forest + XGBoost

# Generate evaluation table
python evaluation_table.py

# Run K-Fold CV + Statistical Tests only
python kfold_cross_validation.py

# Run SHAP analysis only
python shap_explainability.py

# Generate charts only
python research_visuals.py
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'xgboost'"**
→ Run: `pip install xgboost`

**"ModuleNotFoundError: No module named 'shap'"**
→ Run: `pip install shap`

**"creditcard.csv not found"**
→ Make sure the CSV is in the same folder as your `.py` files

**SHAP is very slow for KNN/SVM**
→ This is normal. KernelExplainer is model-agnostic and slower.
  You can reduce sample size in `shap_explainability.py` line:
  `sample_size = min(100, len(X_test))`  ← change 300 to 100

**VS Code shows red underlines on imports**
→ Make sure you selected the correct Python interpreter (Step 7)

---

## New Additions Summary (vs Original Project)

| Addition | File | Purpose |
|----------|------|---------|
| Random Forest | `train_ensemble_models.py` | Ensemble baseline comparison |
| XGBoost | `train_ensemble_models.py` | State-of-the-art ensemble model |
| MCC Metric | `credit_card_preprocessing.py` | Better metric for imbalanced data |
| PR-AUC Metric | `credit_card_preprocessing.py` | More reliable than ROC-AUC for fraud |
| 10-Fold CV | `kfold_cross_validation.py` | Statistical rigour (Mean ± Std) |
| Wilcoxon Test | `kfold_cross_validation.py` | Proves significance of results |
| SHAP Analysis | `shap_explainability.py` | Model explainability (IEEE requirement) |
| Master Script | `run_all.py` | Run entire pipeline with one command |
