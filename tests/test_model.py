import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def test_model_prediction():

    training_data = pd.DataFrame([
        {
            "age": 25,
            "tenure_months": 5,
            "monthly_charges": 100.0,
            "support_calls": 8,
            "contract_type": "Monthly",
            "usage_hours": 50.0,
            "churn": 1
        },
        {
            "age": 45,
            "tenure_months": 40,
            "monthly_charges": 60.0,
            "support_calls": 1,
            "contract_type": "Yearly",
            "usage_hours": 150.0,
            "churn": 0
        },
        {
            "age": 30,
            "tenure_months": 10,
            "monthly_charges": 90.0,
            "support_calls": 6,
            "contract_type": "Monthly",
            "usage_hours": 70.0,
            "churn": 1
        },
        {
            "age": 50,
            "tenure_months": 50,
            "monthly_charges": 50.0,
            "support_calls": 0,
            "contract_type": "Two-Year",
            "usage_hours": 180.0,
            "churn": 0
        }
    ])

    features = [
        "age",
        "tenure_months",
        "monthly_charges",
        "support_calls",
        "contract_type",
        "usage_hours"
    ]

    X_train = training_data[features]
    y_train = training_data["churn"]

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
            ("classifier", LogisticRegression())
        ]
    )

    model.fit(X_train, y_train)

    test_input = pd.DataFrame([{
        "age": 30,
        "tenure_months": 12,
        "monthly_charges": 70.0,
        "support_calls": 3,
        "contract_type": "Monthly",
        "usage_hours": 100.0
    }])

    prediction = model.predict(test_input)

    assert prediction[0] in [0, 1]