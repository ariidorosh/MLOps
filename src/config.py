from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data (DVC tracks this file via .dvc)
DATA_RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"

# Lab 2: prepared data (output of src/prepare.py)
DATA_PREPARED_DIR = ROOT / "data" / "prepared"
PREPARED_TRAIN_PATH = DATA_PREPARED_DIR / "train.csv"
PREPARED_TEST_PATH = DATA_PREPARED_DIR / "test.csv"

# (Optional / legacy from Lab 1) processed data dir, якщо десь у твоєму коді ще використовується
DATA_PROCESSED_DIR = ROOT / "data" / "processed"

# Outputs
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Dataset columns / split
TARGET_COL = "Churn"
ID_COLS = ["customerID"]
RANDOM_STATE = 42
TEST_SIZE = 0.2

# MLflow
MLFLOW_EXPERIMENT_NAME = "telco-churn"
MLFLOW_DB_PATH = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = ROOT / "mlruns"

# UI
MLFLOW_UI_HOST = "127.0.0.1"
MLFLOW_UI_PORT = 5000
AUTO_START_MLFLOW_UI = True