# src/train.py
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import (
    ROOT,
    DATA_RAW_PATH,
    MODELS_DIR,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_DB_PATH,
    MLFLOW_ARTIFACTS_DIR,
)
from .data_utils import load_raw_csv, clean_telco_df, make_train_test, save_processed_df
from .evaluate import (
    compute_metrics,
    save_confusion_matrix,
    make_classification_report_text,
    save_feature_importance,
)

# Метадані для MLflow tags (щоб було гарно в UI)
AUTHOR = "ariidorosh"
DATASET_NAME = "telco-churn"
DATASET_VERSION = "v1"


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


def _build_model(model_name: str, params: dict):
    if model_name == "logreg":
        C = float(params.get("C", 1.0))
        solver = str(params.get("solver", "liblinear"))
        return LogisticRegression(
            C=C,
            solver=solver,
            max_iter=2000,
            random_state=42,
        )

    if model_name == "rf":
        n_estimators = int(params.get("n_estimators", 300))
        max_depth = params.get("max_depth", None)
        if max_depth is not None:
            max_depth = int(max_depth)

        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Невідомий model_name={model_name}")


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


def _mlflow_log_model(pipeline) -> None:
    """
    skops у тебе падав (untrusted numpy.dtype), тому тут тільки cloudpickle.
    Також:
    - якщо MLflow підтримує name -> не буде deprecated warning про artifact_path
    - якщо MLflow підтримує pip_requirements -> підкинемо requirements.txt (менше спаму і краще для відтворюваності)
    """
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


def _set_common_tags(model_name: str, run_group: str | None = None, tuning_param: str | None = None) -> None:
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


def _format_c_for_name(c: float) -> str:
    # для run_name: акуратно і стабільно
    s = f"{c:g}"
    return s


def run_logreg_c_sweep(
    c_values: list[float] | tuple[float, ...],
    solver: str = "liblinear",
    run_prefix: str = "lab1_logreg_C",
    run_group: str = "lab1_c_sweep",
) -> None:
    """
    Лабораторний сценарій:
    - 5+ запусків
    - один ключовий гіперпараметр: C
    - логування train_* і test_* метрик (щоб видно було overfitting)
    - логування артефактів і моделі
    """
    _set_mlflow()

    df = load_raw_csv(DATA_RAW_PATH)
    df = clean_telco_df(df)
    save_processed_df(df)

    preprocessor, X_train, X_test, y_train, y_test = make_train_test(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_pipeline = None
    best_run_name = None

    for idx, c in enumerate(c_values, start=1):
        c_val = float(c)
        run_name = f"{run_prefix}_{_format_c_for_name(c_val)}"

        model = _build_model("logreg", {"C": c_val, "solver": solver})
        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
                ("model", model),
            ]
        )

        # локально зручно мати копію артефактів по рану, але в MLflow шлях буде однаковий:
        # figures/confusion_matrix.png, figures/feature_importance.png, reports/classification_report.txt
        run_art_dir = FIGURES_DIR / run_name
        run_art_dir.mkdir(parents=True, exist_ok=True)

        with mlflow.start_run(run_name=run_name):
            # params
            mlflow.log_param("model_name", "logreg")
            mlflow.log_param("C", c_val)
            mlflow.log_param("solver", solver)

            # tags
            _set_common_tags("logreg", run_group=run_group, tuning_param="C")

            # train
            pipeline.fit(X_train, y_train)

            y_pred_train = pipeline.predict(X_train)
            y_proba_train = (
                pipeline.predict_proba(X_train)[:, 1] if hasattr(pipeline, "predict_proba") else None
            )
            metrics_train = compute_metrics(y_train, y_pred_train, y_proba=y_proba_train)
            mlflow.log_metrics({f"train_{k}": v for k, v in metrics_train.items() if v is not None})

            # test
            y_pred_test = pipeline.predict(X_test)
            y_proba_test = (
                pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
            )
            metrics_test = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics_test.items() if v is not None})

            # artifacts (однакові імена -> "common artifacts" в Compare буде працювати)
            cm_path = run_art_dir / "confusion_matrix.png"
            save_confusion_matrix(y_test, y_pred_test, cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="figures")

            report_path = run_art_dir / "classification_report.txt"
            report_text = make_classification_report_text(y_test, y_pred_test)
            report_path.write_text(report_text, encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            fi_path = run_art_dir / "feature_importance.png"
            if save_feature_importance(pipeline, fi_path, top_k=20):
                mlflow.log_artifact(str(fi_path), artifact_path="figures")

            # model
            _mlflow_log_model(pipeline)

            # best by test_roc_auc else test_f1
            score = metrics_test.get("roc_auc")
            if score is None:
                score = metrics_test.get("f1", 0.0)

            if score is not None and float(score) > best_score:
                best_score = float(score)
                best_pipeline = pipeline
                best_run_name = run_name

    if best_pipeline is not None:
        out_path = MODELS_DIR / "best_model.joblib"
        joblib.dump(best_pipeline, out_path)
        print(f"Best model saved to: {out_path}")
        print(f"Best run: {best_run_name}, score={best_score:.4f}")
    else:
        print("Не вийшло натренувати жодної моделі (перевір дані).")


def run_grid_experiments() -> None:
    """
    Якщо захочеш лишити старий режим (логрег + rf) — він тут.
    Але для методички “один гіперпараметр, 5 запусків” краще використовувати run_logreg_c_sweep().
    """
    _set_mlflow()

    df = load_raw_csv(DATA_RAW_PATH)
    df = clean_telco_df(df)
    save_processed_df(df)

    preprocessor, X_train, X_test, y_train, y_test = make_train_test(df)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        ("logreg", {"C": 0.2, "solver": "liblinear"}),
        ("logreg", {"C": 1.0, "solver": "liblinear"}),
        ("logreg", {"C": 5.0, "solver": "liblinear"}),
        ("rf", {"n_estimators": 200, "max_depth": None}),
        ("rf", {"n_estimators": 400, "max_depth": None}),
        ("rf", {"n_estimators": 400, "max_depth": 10}),
    ]

    best_score = -1.0
    best_pipeline = None
    best_run_name = None

    for idx, (model_name, params) in enumerate(experiments, start=1):
        run_name = f"{idx:02d}_{model_name}"

        model = _build_model(model_name, params)
        pipeline = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),
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
                pipeline.predict_proba(X_train)[:, 1] if hasattr(pipeline, "predict_proba") else None
            )
            metrics_train = compute_metrics(y_train, y_pred_train, y_proba=y_proba_train)
            mlflow.log_metrics({f"train_{k}": v for k, v in metrics_train.items() if v is not None})

            y_pred_test = pipeline.predict(X_test)
            y_proba_test = (
                pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
            )
            metrics_test = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test)
            mlflow.log_metrics({f"test_{k}": v for k, v in metrics_test.items() if v is not None})

            cm_path = run_art_dir / "confusion_matrix.png"
            save_confusion_matrix(y_test, y_pred_test, cm_path)
            mlflow.log_artifact(str(cm_path), artifact_path="figures")

            report_path = run_art_dir / "classification_report.txt"
            report_text = make_classification_report_text(y_test, y_pred_test)
            report_path.write_text(report_text, encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            fi_path = run_art_dir / "feature_importance.png"
            if save_feature_importance(pipeline, fi_path, top_k=20):
                mlflow.log_artifact(str(fi_path), artifact_path="figures")

            _mlflow_log_model(pipeline)

            score = metrics_test.get("roc_auc")
            if score is None:
                score = metrics_test.get("f1", 0.0)

            if score is not None and float(score) > best_score:
                best_score = float(score)
                best_pipeline = pipeline
                best_run_name = run_name

    if best_pipeline is not None:
        out_path = MODELS_DIR / "best_model.joblib"
        joblib.dump(best_pipeline, out_path)
        print(f"Best model saved to: {out_path}")
        print(f"Best run: {best_run_name}, score={best_score:.4f}")
    else:
        print("Не вийшло натренувати жодної моделі (перевір дані).")


def run_single_experiment(model_name: str, params: dict, run_name: str | None = None) -> None:
    _set_mlflow()

    df = load_raw_csv(DATA_RAW_PATH)
    df = clean_telco_df(df)
    save_processed_df(df)

    preprocessor, X_train, X_test, y_train, y_test = make_train_test(df)

    model = _build_model(model_name, params)
    pipeline = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            ("model", model),
        ]
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rn = run_name or f"single_{model_name}"
    run_art_dir = FIGURES_DIR / rn
    run_art_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=rn):
        mlflow.log_param("model_name", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        _set_common_tags(model_name)

        pipeline.fit(X_train, y_train)

        y_pred_train = pipeline.predict(X_train)
        y_proba_train = (
            pipeline.predict_proba(X_train)[:, 1] if hasattr(pipeline, "predict_proba") else None
        )
        metrics_train = compute_metrics(y_train, y_pred_train, y_proba=y_proba_train)
        mlflow.log_metrics({f"train_{k}": v for k, v in metrics_train.items() if v is not None})

        y_pred_test = pipeline.predict(X_test)
        y_proba_test = (
            pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
        )
        metrics_test = compute_metrics(y_test, y_pred_test, y_proba=y_proba_test)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics_test.items() if v is not None})

        cm_path = run_art_dir / "confusion_matrix.png"
        save_confusion_matrix(y_test, y_pred_test, cm_path)
        mlflow.log_artifact(str(cm_path), artifact_path="figures")

        fi_path = run_art_dir / "feature_importance.png"
        if save_feature_importance(pipeline, fi_path, top_k=20):
            mlflow.log_artifact(str(fi_path), artifact_path="figures")

        _mlflow_log_model(pipeline)

    out_path = MODELS_DIR / "last_model.joblib"
    joblib.dump(pipeline, out_path)
    print(f"Saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["grid", "single"], default="grid")

    p.add_argument("--model", choices=["logreg", "rf"], default="logreg")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--solver", type=str, default="liblinear")

    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=0, help="0 означає None")

    p.add_argument("--run_name", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
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