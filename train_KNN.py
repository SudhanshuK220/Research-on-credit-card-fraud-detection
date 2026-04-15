from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from credit_card_preprocessing import preprocess_data, evaluate_model

def train_knn(X_train, y_train, X_test, y_test):
    print("\nTraining K-Nearest Neighbors (KNN)...")
    knn_model = KNeighborsClassifier(
    n_neighbors=3,
    weights='distance',   #  Added (cost-sensitive alternative)
    n_jobs=1
)
    knn_model.fit(X_train, y_train)

    eval_size = 5000
    if len(X_test) > eval_size:
        print(f"Evaluating KNN on a random {eval_size}-sample subset of the testing set...")
        X_eval = X_test.sample(n=eval_size, random_state=42)
        y_eval = y_test.loc[X_eval.index]
    else:
        print("Evaluating KNN on the full testing set...")
        X_eval = X_test
        y_eval = y_test

    y_pred = knn_model.predict(X_eval)

    # Probability scores for ROC-AUC and PR-AUC
    y_scores = knn_model.predict_proba(X_eval)[:, 1]

    print("Calculating ROC-AUC Score...")
    roc_auc = roc_auc_score(y_eval, y_scores)

    evaluate_model(
        "K-Nearest Neighbors",
        y_eval,
        y_pred,
        roc_auc=roc_auc,
        y_scores=y_scores
    )

if __name__ == "__main__":
    print(" Starting K-Nearest Neighbors (KNN) Pipeline ")
    X_train, X_test, y_train, y_test = preprocess_data()
    if X_train is not None:
        train_knn(X_train, y_train, X_test, y_test)
