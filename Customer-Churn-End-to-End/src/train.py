from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.50

def load_and_clean_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the data folder."
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = {"customerID", "Churn", "TotalCharges"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    # Convert whitespace/invalid values in TotalCharges into NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove exact duplicate rows, then remove identifier.
    df = df.drop_duplicates().copy()
    df = df.drop(columns=["customerID"])

    # Target encoding.
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
    df = df.dropna(subset=["Churn"]).copy()
    df["Churn"] = df["Churn"].astype(int)

    return df

def build_preprocessor(X):
    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return preprocessor, numerical_features, categorical_features

def main():
    df = load_and_clean_data()
    print(f"Dataset after basic cleaning: {df.shape}")

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    preprocessor, numerical_features, categorical_features = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )
    scoring = {
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "roc_auc": "roc_auc"
    }

    cv_results = {}
    pipelines = {}

    print("\n===== CROSS-VALIDATION RESULTS (TRAINING DATA ONLY) =====")
    for name, estimator in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            error_score="raise"
        )
        cv_results[name] = {
            metric: float(scores[f"test_{metric}"].mean())
            for metric in scoring
        }
        pipelines[name] = pipeline
        print(f"\n{name}")
        for metric, value in cv_results[name].items():
            print(f"{metric}: {value:.4f}")

    # Model selection is based on mean CV F1, not test-set performance.
    best_model_name = max(cv_results, key=lambda name: cv_results[name]["f1"])
    final_pipeline = pipelines[best_model_name]
    final_pipeline.fit(X_train, y_train)

    y_probability = final_pipeline.predict_proba(X_test)[:, 1]
    y_prediction = (y_probability >= THRESHOLD).astype(int)

    test_metrics = {
        "accuracy": float(accuracy_score(y_test, y_prediction)),
        "precision": float(precision_score(y_test, y_prediction, zero_division=0)),
        "recall": float(recall_score(y_test, y_prediction, zero_division=0)),
        "f1": float(f1_score(y_test, y_prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_probability))
    }

    print("\n===== FINAL HELD-OUT TEST RESULTS =====")
    print(f"Selected model: {best_model_name}")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(y_test, y_prediction, zero_division=0))

    # Save the complete preprocessing + model pipeline.
    model_path = MODEL_DIR / "churn_pipeline.joblib"
    joblib.dump(final_pipeline, model_path)

    metadata = {
        "model_name": best_model_name,
        "threshold": THRESHOLD,
        "target": "Churn",
        "positive_class": 1,
        "features": X.columns.tolist(),
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "cross_validation": cv_results,
        "test_metrics": test_metrics,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE
    }
    with open(MODEL_DIR / "model_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    # Save evaluation plots.
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm = confusion_matrix(y_test, y_prediction)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Predicted No", "Predicted Yes"],
        yticklabels=["Actual No", "Actual Yes"],
        ax=axes[0]
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_xlabel("Prediction")
    axes[0].set_ylabel("Actual")

    RocCurveDisplay.from_predictions(y_test, y_probability, ax=axes[1])
    axes[1].set_title("ROC Curve")

    plt.tight_layout()
    plot_path = REPORT_DIR / "model_evaluation.png"
    plt.savefig(plot_path, dpi=150)
    plt.show()

    print(f"\nSaved model: {model_path}")
    print(f"Saved metadata: {MODEL_DIR / 'model_metadata.json'}")
    print(f"Saved evaluation plot: {plot_path}")
    print("\nTraining completed successfully.")

if __name__ == "__main__":
    main()
