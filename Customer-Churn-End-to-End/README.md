# Customer Churn Prediction — End-to-End Machine Learning

A binary classification project that estimates the probability that a telecom customer will leave a service provider.

## Project structure

```text
Customer-Churn-End-to-End/
├── data/                 # Place the CSV dataset here
├── models/               # Generated model and metadata
├── reports/              # Generated charts
├── src/
│   ├── data_exploration.py
│   └── train.py
├── api/
│   └── main.py
├── requirements.txt
└── Dockerfile
```

## Dataset

Download the Telco Customer Churn dataset and place
`WA_Fn-UseC_-Telco-Customer-Churn.csv` inside `data/`.

Dataset source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Check the dataset's license and terms before redistributing it.

## Setup (Windows PowerShell)

Use your Python executable:
```powershell
& "C:\Users\DT\Desktop\python313\python.exe" -m pip install -r requirements.txt
```

Run exploratory analysis:

```powershell
& "C:\Users\DT\Desktop\python313\python.exe" -m src.data_exploration
```

Train and evaluate models:

```powershell
& "C:\Users\DT\Desktop\python313\python.exe" -m src.train
```

The training script compares Logistic Regression, Random Forest, and Gradient Boosting using stratified 5-fold cross-validation on the training split. It selects the model by mean CV F1, fits it on the complete training split, and evaluates it once on the held-out test split.

Generated files:
- `models/churn_pipeline.joblib`
- `models/model_metadata.json`
- `reports/model_evaluation.png`
- `reports/eda_visualization.png`

## Run the API

After training:

```powershell
& "C:\Users\DT\Desktop\python313\python.exe" -m uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs

Endpoints:
- `GET /` — API information
- `GET /health` — health check
- `POST /predict` — churn prediction

## Docker

Build:

```powershell
docker build -t customer-churn-api .
```

Run:

```powershell
docker run --rm -p 8000:8000 customer-churn-api
```

## Important notes

- The model's probability is an estimate, not a guarantee that a customer will churn.
- The default classification threshold is 0.50. It should be selected based on business costs and validation data before production use.
- The project uses a preprocessing pipeline to fit imputers, scaling, and one-hot encoding only on training folds, reducing data leakage.
- This is a portfolio project, not a production-ready system for real customer decisions. Add tests, monitoring, access controls, privacy safeguards, and model governance before real deployment.
- Only send synthetic or properly authorized customer data to the API.
