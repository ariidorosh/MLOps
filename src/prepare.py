# src/prepare.py
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def _clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """
    Мінімальна чистка Telco Customer Churn:
    - прибрати customerID (якщо є)
    - TotalCharges привести до числа (там часто пробіли/порожні рядки)
    - обрізати пробіли в строкових полях
    """
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


def run_prepare(
    input_path: str,
    out_dir: str,
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[str, str]:
    """
    Готує датасет і зберігає:
      - {out_dir}/train.csv
      - {out_dir}/test.csv
    Повертає (train_path, test_path) як рядки.

    Важливо: таргет (Churn) НЕ перетворюємо в 0/1 тут спеціально,
    щоб не зламати твій поточний train.py, який може робити мапінг сам.
    """
    input_path_p = Path(input_path)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path_p)
    df = _clean_telco(df)

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")

    # прибрати рядки без таргету
    df = df.dropna(subset=[target])

    # якщо TotalCharges має NaN після to_numeric — заповнити медіаною
    if "TotalCharges" in df.columns and df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y = df[target]
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    train_path = out_dir_p / "train.csv"
    test_path = out_dir_p / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[prepare] Saved: {train_path}")
    print(f"[prepare] Saved: {test_path}")
    print(f"[prepare] Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    return str(train_path), str(test_path)


def main():
    parser = argparse.ArgumentParser(description="Prepare Telco Churn dataset for training.")
    parser.add_argument("--input", required=True, help="Path to raw CSV (e.g., data/raw/telco_churn.csv)")
    parser.add_argument("--out-dir", required=True, help="Output dir (e.g., data/prepared)")
    parser.add_argument("--target", default="Churn", help="Target column name (default: Churn)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size (default: 0.2)")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    run_prepare(
        input_path=args.input,
        out_dir=args.out_dir,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()