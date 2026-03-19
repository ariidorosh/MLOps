from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Iterable

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import (
    ROOT,
    PREPARED_TRAIN_PATH,
    PREPARED_TEST_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    TARGET_COL,
    ID_COLS,
    RANDOM_STATE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_DB_PATH,
    MLFLOW_ARTIFACTS_DIR,
)
from .evaluate import (
    compute_metrics,
    save_metrics_json,
    save_confusion_matrix,
    make_classification_report_text,
    save_feature_importance,
)

AUTHOR = "ariidorosh"
DATASET_NAME = "telco-churn"
DATASET_VERSION = "v2_prepared_split"
REPORTS_DIR = ROOT / "reports"


# -------------------------
# MLflow utils
# -------------------------
def _as_sqlite_uri(db_path: Path) -> str:
    return f"sqlite:///{db_path.resolve().as_posix()}"


def _ensure_experiment() -> None:
    artifact_root_uri = MLFLOW_ARTIFACTS_DIR.resolve().as_uri()
    exp = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if exp is None:
        mlflow.create_experiment(
            name=MLFLOW_EXPERIMENT_NAME,
            artifact_location=artifact_root_uri,
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _set_mlflow() -> None:
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(_as_sqlite_uri(MLFLOW_DB_PATH))
    _ensure_experiment()


def _read_requirements_txt() -> list[str] | None:
    req_path = ROOT / "requirements.txt"
    if not req_path.exists():
        return None

    lines: list[str] = []
    for raw in req_path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines or None


def _mlflow_log_model(pipeline: Pipeline) -> None:
    sig = inspect.signature(mlflow.sklearn.log_model)

    kwargs = {}
    if "serialization_format" in sig.parameters:
        kwargs["serialization_format"] = "cloudpickle"

    reqs = _read_requirements_txt()
    if reqs is not None and "pip_requirements" in sig.parameters:
        kwargs["pip_requirements"] = reqs

    if "name" in sig.parameters:
        mlflow.sklearn.log_model(pipeline, name="model", **kwargs)
    else:
        mlflow.sklearn.log_model(pipeline, artifact_path="model", **kwargs)


def _set_common_tags(
    model_name: str, run_group: str | None = None, tuning_param: str | None = None
) -> None:
    mlflow.set_tag("author", AUTHOR)
    mlflow.set_tag("dataset", DATASET_NAME)
    mlflow.set_tag("dataset_version", DATASET_VERSION)
    mlflow.set_tag("model_type", model_name)
    mlflow.set_tag("python", sys.version.split()[0])

    if run_group:
        mlflow.set_tag("run_group", run_group)

    if tuning_param:
        mlflow.set_tag("tuning_param", tuning_param)
    else:
        if model_name == "logreg":
            mlflow.set_tag("tuning_param", "C")
        elif model_name == "rf":
            mlflow.set_tag("tuning_param", "n_estimators/max_depth")


# -------------------------
# Data loading (prepared)
# -------------------------
def _normalize_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    ys = y.astype(str).str.strip()
    uniq = set(ys.unique())

    if uniq.issubset({"Yes", "No"}):
        return ys.map({"No": 0, "Yes": 1}).astype(int)

    if uniq.issubset({"0", "1"}):
        return ys.astype(int)

    return ys


def _load_prepared(
    train_path: Path,
    test_path: Path,
    target_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not train_path.exists():
        raise FileNotFoundError(f"Prepared train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Prepared test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' must exist in both train.csv and test.csv"
        )

    drop_cols = [c for c in ID_COLS if c in train_df.columns]
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols)
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    y_train = _normalize_target(train_df[target_col])
    y_test = _normalize_target(test_df[target_col])

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    if "TotalCharges" in X_train.columns:
        X_train["TotalCharges"] = pd.to_numeric(
            X_train["TotalCharges"], errors="coerce"
        )
        X_test["TotalCharges"] = pd.to_numeric(X_test["TotalCharges"], errors="coerce")

    return X_train, X_test, y_train, y_test


# -------------------------
# Preprocessor
# -------------------------
def _make_onehot() -> OneHotEncoder:
    sig = inspect.signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _build_preprocessor() -> ColumnTransformer:
    numeric_selector = make_column_selector(dtype_include=["number"])
    categorical_selector = make_column_selector(dtype_exclude=["number"])

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_onehot()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_selector),
            ("cat", cat_pipe, categorical_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# -------------------------
# Models
# -------------------------
def _build_model(model_name: str, params: dict):
    if model_name == "logreg":
        C = float(params.get("C", 1.0))
        solver = str(params.get("solver", "liblinear"))
        return LogisticRegression(
            C=C,
            solver=solver,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

    if model_name == "rf":
        n_estimators = int(params.get("n_estimators", 300))
        max_depth = params.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    raise ValueError(f"Unknown model_name: {model_name}")


def _format_c_for_name(c: float) -> str:
    return f"{c:g}"


def _save_final_reports(pipeline: Pipeline, y_test, X_test) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    y_pred_test = pipeline.predict(X_test)
    y_proba_test = (
        pipeline.predict_proba(X_test)[:, 1]
        if hasattr(pipeline, "predict_proba")
        else None
    )
    metrics = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test)

    save_metrics_json(metrics, REPORTS_DIR / "metrics.json")
    save_confusion_matrix(y_test, y_pred_test, REPORTS_DIR / "confusion_matrix.png")

    (REPORTS_DIR / "classification_report.txt").write_text(
        make_classification_report_text(y_test, y_pred_test),
        encoding="utf-8",
    )

    try:
        save_feature_importance(
            pipeline, REPORTS_DIR / "feature_importance.png", top_k=20
        )
    except Exception:
        pass

    return metrics


# -------------------------
# Training routines
# -------------------------
def run_logreg_c_sweep(
    c_values: Iterable[float],
    solver: str = "liblinear",
    run_prefix: str = "lab2_logreg_C",
    run_group: str = "lab2_c_sweep",
    train_path: Path = PREPARED_TRAIN_PATH,
    test_path: Path = PREPARED_TEST_PATH,
) -> None:
    _set_mlflow()

    X_train, X_test, y_train, y_test = _load_prepared(train_path, test_path, TARGET_COL)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_pipeline: Pipeline | None = None
    best_run_name: str | None = None

    base_preprocessor = _build_preprocessor()

    for c in c_values:
        c_val = float(c)
        run_name = f"{run_prefix}_{_format_c_for_name(c_val)}"

        model = _build_model("logreg", {"C": c_val, "solver": solver})
        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(base_preprocessor)),
                ("model", model),
            ]
        )

        run_art_dir = FIGURES_DIR / run_name
        run_art_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", "logreg")
            mlflow.log_param("C", c_val)
            mlflow.log_param("solver", solver)
            _set_common_tags("logreg", run_group=run_group, tuning_param="C")

            pipeline.fit(X_train, y_train)

            y_pred_train = pipeline.predict(X_train)
            y_proba_train = (
                pipeline.predict_proba(X_train)[:, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )
            m_train = compute_metrics(y_train, y_pred_train, y_proba=y_proba_train)
            mlflow.log_metrics(
                {f"train_{k}": v for k, v in m_train.items() if v is not None}
            )

            y_pred_test = pipeline.predict(X_test)
            y_proba_test = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )
            m_test = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test)
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in m_test.items() if v is not None}
            )

            cm_path = run_art_dir / "confusion_matrix.png"
            save_confusion_matrix(y_test, y_pred_test, cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="figures")

            report_path = run_art_dir / "classification_report.txt"
            report_path.write_text(
                make_classification_report_text(y_test, y_pred_test), encoding="utf-8"
            )
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            fi_path = run_art_dir / "feature_importance.png"
            try:
                if save_feature_importance(pipeline, fi_path, top_k=20):
                    mlflow.log_artifact(str(fi_path), artifact_path="figures")
            except Exception:
                pass

            _mlflow_log_model(pipeline)

            score = m_test.get("roc_auc")
            if score is None:
                score = m_test.get("f1", 0.0)

            if score is not None and float(score) > best_score:
                best_score = float(score)
                best_pipeline = pipeline
                best_run_name = run_name

    if best_pipeline is not None:
        out_path = MODELS_DIR / "best_model.joblib"
        joblib.dump(best_pipeline, out_path)

        final_metrics = _save_final_reports(best_pipeline, y_test, X_test)

        print(f"Best model saved to: {out_path}")
        print(f"Best run: {best_run_name}, score={best_score:.4f}")
        print(f"Reports saved to: {REPORTS_DIR}")
        print(f"Final metrics: {final_metrics}")
    else:
        print("Не вийшло натренувати жодної моделі (перевір prepared дані).")


def run_grid_experiments() -> None:
    _set_mlflow()
    X_train, X_test, y_train, y_test = _load_prepared(
        PREPARED_TRAIN_PATH, PREPARED_TEST_PATH, TARGET_COL
    )
    base_preprocessor = _build_preprocessor()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("logreg", {"C": 0.2, "solver": "liblinear"}),
        ("logreg", {"C": 1.0, "solver": "liblinear"}),
        ("logreg", {"C": 5.0, "solver": "liblinear"}),
        ("rf", {"n_estimators": 200, "max_depth": None}),
        ("rf", {"n_estimators": 400, "max_depth": None}),
        ("rf", {"n_estimators": 400, "max_depth": 10}),
    ]

    best_score = -1.0
    best_pipeline: Pipeline | None = None
    best_run_name: str | None = None

    for idx, (model_name, params) in enumerate(experiments, start=1):
        run_name = f"{idx:02d}_{model_name}"
        model = _build_model(model_name, params)

        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(base_preprocessor)),
                ("model", model),
            ]
        )

        run_art_dir = FIGURES_DIR / run_name
        run_art_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_name", model_name)
            for k, v in params.items():
                mlflow.log_param(k, v)
            _set_common_tags(model_name)

            pipeline.fit(X_train, y_train)

            y_pred_train = pipeline.predict(X_train)
            y_proba_train = (
                pipeline.predict_proba(X_train)[:, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )
            m_train = compute_metrics(y_train, y_pred_train, y_proba=y_proba_train)
            mlflow.log_metrics(
                {f"train_{k}": v for k, v in m_train.items() if v is not None}
            )

            y_pred_test = pipeline.predict(X_test)
            y_proba_test = (
                pipeline.predict_proba(X_test)[:, 1]
                if hasattr(pipeline, "predict_proba")
                else None
            )
            m_test = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test)
            mlflow.log_metrics(
                {f"test_{k}": v for k, v in m_test.items() if v is not None}
            )

            cm_path = run_art_dir / "confusion_matrix.png"
            save_confusion_matrix(y_test, y_pred_test, cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="figures")

            _mlflow_log_model(pipeline)

            score = m_test.get("roc_auc")
            if score is None:
                score = m_test.get("f1", 0.0)

            if score is not None and float(score) > best_score:
                best_score = float(score)
                best_pipeline = pipeline
                best_run_name = run_name

    if best_pipeline is not None:
        out_path = MODELS_DIR / "best_model.joblib"
        joblib.dump(best_pipeline, out_path)

        final_metrics = _save_final_reports(best_pipeline, y_test, X_test)

        print(f"Best model saved to: {out_path}")
        print(f"Best run: {best_run_name}, score={best_score:.4f}")
        print(f"Reports saved to: {REPORTS_DIR}")
        print(f"Final metrics: {final_metrics}")


def run_single_experiment(
    model_name: str, params: dict, run_name: str | None = None
) -> None:
    _set_mlflow()
    X_train, X_test, y_train, y_test = _load_prepared(
        PREPARED_TRAIN_PATH, PREPARED_TEST_PATH, TARGET_COL
    )
    base_preprocessor = _build_preprocessor()

    model = _build_model(model_name, params)
    pipeline = Pipeline(
        steps=[
            ("preprocess", clone(base_preprocessor)),
            ("model", model),
        ]
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rn = run_name or f"single_{model_name}"
    run_art_dir = FIGURES_DIR / rn
    run_art_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=rn):
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)
        _set_common_tags(model_name)

        pipeline.fit(X_train, y_train)

        y_pred_test = pipeline.predict(X_test)
        y_proba_test = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else None
        )
        m_test = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in m_test.items() if v is not None})

        cm_path = run_art_dir / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred_test, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="figures")

        _mlflow_log_model(pipeline)

    out_path = MODELS_DIR / "last_model.joblib"
    joblib.dump(pipeline, out_path)

    final_metrics = _save_final_reports(pipeline, y_test, X_test)

    print(f"Saved: {out_path}")
    print(f"Reports saved to: {REPORTS_DIR}")
    print(f"Final metrics: {final_metrics}")


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", choices=["grid", "single", "logreg_c_sweep"], default="grid"
    )

    p.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--solver", type=str, default="liblinear")

    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=0, help="0 означає None")

    p.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.05, 0.2, 0.85, 1.0, 1.15, 5.0, 20.0],
    )
    p.add_argument("--run-prefix", type=str, default="lab2_logreg_C")
    p.add_argument("--run-group", type=str, default="lab2_c_sweep")

    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "logreg_c_sweep":
        run_logreg_c_sweep(
            c_values=args.c_values,
            solver=args.solver,
            run_prefix=args.run_prefix,
            run_group=args.run_group,
        )
        return

    if args.mode == "grid":
        run_grid_experiments()
        return

    if args.model == "logreg":
        run_single_experiment(
            "logreg",
            {"C": args.C, "solver": args.solver},
            run_name=args.run_name,
        )
    else:
        md = None if args.max_depth == 0 else args.max_depth
        run_single_experiment(
            "rf",
            {"n_estimators": args.n_estimators, "max_depth": md},
            run_name=args.run_name,
        )


if __name__ == "__main__":
    main()
