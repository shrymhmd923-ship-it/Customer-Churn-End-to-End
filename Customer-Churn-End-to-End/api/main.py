from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    raise RuntimeError(
        "Trained model files were not found. Run `python -m src.train` first."
    )

model = joblib.load(MODEL_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

THRESHOLD = float(metadata.get("threshold", 0.5))

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict the probability that a telecom customer will churn.",
    version="1.0.0"
)

class CustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0, le=1000)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

@app.get("/")
def home():
    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "model": metadata["model_name"]
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        customer_df = pd.DataFrame([customer.model_dump()])
        expected_features = metadata["features"]
        customer_df = customer_df[expected_features]

        probability = float(model.predict_proba(customer_df)[0, 1])
        prediction = int(probability >= THRESHOLD)

        if probability >= 0.70:
            risk_level = "High"
        elif probability >= 0.40:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            "prediction": prediction,
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "churn_probability": round(probability, 4),
            "threshold": THRESHOLD,
            "risk_level": risk_level
        }
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error))
