from fastapi import FastAPI
from pathlib import Path
import joblib
import pandas as pd
from pydantic import BaseModel

app=FastAPI(
    title="Customer Churn Prediction",
    description="API for CCP",
    version="1.0.0"
)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_logistic_regression.joblib"
model = joblib.load(MODEL_PATH)
print("Model loaded successfully!")

class CustomerInput(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    support_calls: int
    contract_type: str
    usage_hours: float


@app.post("/predict")
def predict(customer: CustomerInput):

    input_data = pd.DataFrame([customer.model_dump()])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }


@app.get("/")
def home():
    return{
        "message":"API is running"
    }
