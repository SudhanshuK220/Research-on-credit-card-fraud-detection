from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from credit_card_preprocessing import preprocess_data, evaluate_model

def train_decision_tree(X_train, y_train, X_test, y_test):
    print("\nTraining Decision Tree...")
    dt_model = DecisionTreeClassifier(random_state=42, max_leaf_nodes=1500)
    dt_model.fit(X_train, y_train)

    print("Evaluating Decision Tree on the testing set...")
    y_pred = dt_model.predict(X_test)

    # Probability scores for ROC-AUC and PR-AUC
    y_scores = dt_model.predict_proba(X_test)[:, 1]

    print("Calculating ROC-AUC Score...")
    roc_auc = roc_auc_score(y_test, y_scores)

    evaluate_model("Decision Tree", y_test, y_pred, roc_auc, y_scores)

if __name__ == "__main__":
    print(" Starting Decision Tree Pipeline ")
    X_train, X_test, y_train, y_test = preprocess_data()
    if X_train is not None:
        train_decision_tree(X_train, y_train, X_test, y_test)
