import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Load data
X_train = pd.read_csv("heart_disease_preprocessing/X_train.csv")
X_test = pd.read_csv("heart_disease_preprocessing/X_test.csv")
y_train = pd.read_csv("heart_disease_preprocessing/y_train.csv")
y_test = pd.read_csv("heart_disease_preprocessing/y_test.csv")

# Hyperparameter tuning (manual - contoh saja)
n_estimators = 100
max_depth = 5

with mlflow.start_run():
    # Logging parameter
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Predict
    preds = model.predict(X_test)

    # Metric
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    # Logging metric
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)

    # Logging model
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Akurasi: {acc}")
