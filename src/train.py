import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import mlflow
import joblib
from pathlib import Path


mlflow.set_experiment("Customer Churn Prediction")

DATA_PATH = "../data/processed/cleaned_data.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)

FEATURES = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "contract_type",
    "usage_hours"
]

TARGET = "churn"
X = df[FEATURES]
y = df[TARGET]
print("Features shape:", X.shape)
print("Target shape:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

numeric_features = [
    "age",
    "tenure_months",
    "monthly_charges",
    "support_calls",
    "usage_hours"
]

categorical_features = [
    "contract_type"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                random_state=42
            )
        )
    ]
)

with mlflow.start_run():

    # Parameters
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)

    # Training
    model.fit(X_train, y_train)

    print("Model training completed!")

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    model_info=mlflow.sklearn.log_model(
        model,
        name="churn_model"
    )

    mlflow.register_model(
        model_uri=model_info.model_uri,
        name="CustomerChurnModel"
    )




print("\nModel Evaluation")
print("----------------")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

model_dir = Path("../models")
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "churn_logistic_regression.joblib"

joblib.dump(model, model_path)

print(f"\nModel saved to: {model_path}")