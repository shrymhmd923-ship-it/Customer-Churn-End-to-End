from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = BASE_DIR / "reports"
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download the Telco Customer Churn CSV and place it in the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== FIRST FIVE ROWS =====")
    print(df.head())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== DATASET INFO =====")
    df.info()

    print("\n===== MISSING VALUES =====")
    print(df.isna().sum())

    print("\n===== DUPLICATE ROWS =====")
    print(df.duplicated().sum())

    print("\n===== TARGET DISTRIBUTION =====")
    print(df["Churn"].value_counts(dropna=False))
    print("\nTarget percentages:")
    print((df["Churn"].value_counts(normalize=True, dropna=False) * 100).round(2))

    # Convert TotalCharges for correct numeric summary only.
    analysis_df = df.copy()
    analysis_df["TotalCharges"] = pd.to_numeric(
        analysis_df["TotalCharges"], errors="coerce"
    )
    print("\n===== NUMERICAL SUMMARY =====")
    print(analysis_df.describe())

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.countplot(data=df, x="Churn", ax=axes[0])
    axes[0].set_title("Customer Churn Distribution")
    axes[0].set_xlabel("Churn")
    axes[0].set_ylabel("Number of Customers")

    sns.boxplot(data=analysis_df, x="Churn", y="MonthlyCharges", ax=axes[1])
    axes[1].set_title("Monthly Charges by Churn")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Monthly Charges")

    plt.tight_layout()
    output_path = OUTPUT_DIR / "eda_visualization.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nEDA chart saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
