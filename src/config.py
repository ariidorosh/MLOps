# src/config.py
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_PATH = ROOT / "data" / "raw" / "telco_churn.csv"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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