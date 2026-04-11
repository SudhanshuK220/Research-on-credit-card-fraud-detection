from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from credit_card_preprocessing import preprocess_data, evaluate_model

def train_svm(X_train, y_train, X_test, y_test):
    print("\nTraining SVM (LinearSVC)... This might take a minute or two.")

    # Wrap with CalibratedClassifierCV to get probability scores for PR-AUC
    base_svm = LinearSVC(random_state=42, dual=False, max_iter=1500)
    svm_model = CalibratedClassifierCV(base_svm, cv=3)
    svm_model.fit(X_train, y_train)

    print("Evaluating SVM on the testing set...")
    y_pred = svm_model.predict(X_test)

    # Get probability scores (needed for ROC-AUC and PR-AUC)
    y_scores = svm_model.predict_proba(X_test)[:, 1]

    print("Calculating ROC-AUC Score...")
    roc_auc = roc_auc_score(y_test, y_scores)

    # Pass y_scores for PR-AUC calculation inside evaluate_model
    evaluate_model("Support Vector Machine (LinearSVC)", y_test, y_pred, roc_auc, y_scores)

if __name__ == "__main__":
    print(" Starting Support Vector Machine (SVM) Pipeline ")
    X_train, X_test, y_train, y_test = preprocess_data()
    if X_train is not None:
        train_svm(X_train, y_train, X_test, y_test)
