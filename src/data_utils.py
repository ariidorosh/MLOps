# src/data_utils.py
from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import TARGET_COL, ID_COLS, RANDOM_STATE, TEST_SIZE, DATA_PROCESSED_DIR


def load_raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_telco_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # прибираємо id-колонки якщо є
    for col in ID_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # TotalCharges у Telco часто строка + пробіли -> робимо числом
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Чистимо порожні таргети (на всяк випадок)
    df = df.dropna(subset=[TARGET_COL])

    return df


def save_processed_df(df: pd.DataFrame) -> Path:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED_DIR / "telco_churn_clean.csv"
    df.to_csv(out_path, index=False)
    return out_path


def split_xy(df: pd.DataFrame):
    if TARGET_COL not in df.columns:
        raise ValueError(f"Не знайшов колонку таргету '{TARGET_COL}'. Є колонки: {list(df.columns)}")

    y_raw = df[TARGET_COL]

    # мапимо Yes/No -> 1/0, якщо вже 0/1 - теж ок
    if y_raw.dtype == object:
        y = y_raw.map({"Yes": 1, "No": 0})
    else:
        y = y_raw

    if y.isna().any():
        raise ValueError(
            "Не зміг привести таргет до 0/1. Перевір значення в колонці Churn (має бути Yes/No або 0/1)."
        )

    y = y.astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )


def make_train_test(df: pd.DataFrame):
    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)
    return preprocessor, X_train, X_test, y_train, y_test
