# Customer Churn Prediction – End-to-End Machine Learning

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave a service (Customer Churn), using Python, Scikit-learn, FastAPI, and Docker.

## Project Overview

Customer churn is a major challenge for telecommunications companies. This project builds a machine learning pipeline to analyze customer data, predict churn probability, and classify customers based on their churn risk.

## Key Features

- Exploratory Data Analysis (EDA)
- Data Cleaning and Preprocessing
- Feature Engineering
- Machine Learning Model Training
- Model Evaluation and Cross-Validation
- Churn Probability Prediction
- Risk Level Classification (Low, Medium, High)
- REST API using FastAPI
- Docker Containerization

## Technologies Used

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Docker

## Project Structure

```text
Customer-Churn-End-to-End/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── PUT_DATASET_HERE.txt
│
├── models/
│
├── reports/
│
├── src/
│   ├── __init__.py
│   ├── data_exploration.py
│   ├── preprocessing.py
│   └── train.py
│
├── Dockerfile
├── requirements.txt
└── run_all.py
```

## Dataset

This project uses the Telco Customer Churn dataset.

The CSV file is not included in this repository.

Download the dataset and place it inside the `data/` directory using the expected filename:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Customer-Churn-End-to-End
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

After placing the dataset in the `data/` directory, run:

```bash
python run_all.py
```

This script runs the following steps:

1. Data Exploration
2. Model Training
3. FastAPI Server

The trained model and metadata are generated inside the `models/` directory.

## Run the API Separately

```bash
python -m uvicorn api.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Health check |
| POST | `/predict` | Predict customer churn |

## Docker

Build the Docker image:

```bash
docker build -t customer-churn-api .
```

Run the container:

```bash
docker run --name churn-container -p 8000:8000 customer-churn-api
```

Access the API documentation:

http://127.0.0.1:8000/docs

## Prediction Output

The prediction endpoint returns:

- Prediction (Churn / No Churn)
- Churn Probability
- Decision Threshold
- Risk Level (Low / Medium / High)

## Project Goal

The goal of this project is to demonstrate a complete machine learning workflow, from data exploration and preprocessing to model training, evaluation, API development, and Docker deployment.

---

**Developed as a Machine Learning portfolio project.**
