
from pathlib import Path
import sys
import uvicorn

# Project root directory
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Import project modules
from src.data_exploration import main as run_eda
from src.train import main as run_training


def main():
    print("\n" + "=" * 50)
    print("CUSTOMER CHURN PREDICTION PROJECT")
    print("=" * 50)

    # Step 1: Data Exploration
    print("\n[1/3] Starting Data Exploration...")
    run_eda()

    print("\nData Exploration completed successfully.")

    # Step 2: Model Training
    print("\n[2/3] Starting Model Training...")
    run_training()

    print("\nModel Training completed successfully.")

    # Step 3: Start FastAPI
    print("\n[3/3] Starting FastAPI...")
    print("API Documentation: http://127.0.0.1:8000/docs")

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False
    )


if __name__ == "__main__":
    main()
