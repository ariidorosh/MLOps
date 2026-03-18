from pathlib import Path
import pandas as pd

TRAIN_PATH = Path("data/prepared/train.csv")
TEST_PATH = Path("data/prepared/test.csv")
TARGET_COL = "Churn"


def test_prepared_files_exist():
    assert TRAIN_PATH.exists(), f"Missing file: {TRAIN_PATH}"
    assert TEST_PATH.exists(), f"Missing file: {TEST_PATH}"


def test_prepared_data_not_empty():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    assert not train_df.empty, "train.csv is empty"
    assert not test_df.empty, "test.csv is empty"


def test_target_column_exists():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    assert TARGET_COL in train_df.columns, f"Missing target column '{TARGET_COL}' in train.csv"
    assert TARGET_COL in test_df.columns, f"Missing target column '{TARGET_COL}' in test.csv"


def test_target_has_at_least_two_classes():
    train_df = pd.read_csv(TRAIN_PATH)
    assert train_df[TARGET_COL].nunique() >= 2, "Target must have at least two classes"