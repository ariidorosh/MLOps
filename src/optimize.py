from __future__ import annotations

import json
from pathlib import Path

import hydra
import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def normalize_target(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    ys = y.astype(str).str.strip()
    uniq = set(ys.unique())

    if uniq.issubset({"Yes", "No"}):
        return ys.map({"No": 0, "Yes": 1}).astype(int)

    if uniq.issubset({"0", "1"}):
        return ys.astype(int)

    raise ValueError(f"Unexpected target values: {sorted(uniq)}")


def load_data(train_path: str, test_path: str, target_col: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = normalize_target(train_df[target_col])
    y_test = normalize_target(test_df[target_col])

    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])

    if "TotalCharges" in X_train.columns:
        X_train["TotalCharges"] = pd.to_numeric(X_train["TotalCharges"], errors="coerce")
        X_test["TotalCharges"] = pd.to_numeric(X_test["TotalCharges"], errors="coerce")

    return X_train, X_test, y_train, y_test


def make_onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_selector = make_column_selector(dtype_include=["number"])
    categorical_selector = make_column_selector(dtype_exclude=["number"])

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_onehot()),
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


def build_model(model_cfg: DictConfig, trial: optuna.Trial, seed: int):
    model_type = model_cfg.type

    if model_type == "logreg":
        C = trial.suggest_float(
            "C",
            float(model_cfg.search_space.C.low),
            float(model_cfg.search_space.C.high),
            log=bool(model_cfg.search_space.C.log),
        )
        solver = str(model_cfg.params.solver)
        max_iter = int(model_cfg.params.max_iter)

        model = LogisticRegression(
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=seed,
        )
        params = {
            "C": C,
            "solver": solver,
            "max_iter": max_iter,
        }
        return model, params

    if model_type == "rf":
        n_estimators = trial.suggest_int(
            "n_estimators",
            int(model_cfg.search_space.n_estimators.low),
            int(model_cfg.search_space.n_estimators.high),
            step=int(model_cfg.search_space.n_estimators.step),
        )
        max_depth = trial.suggest_categorical(
            "max_depth",
            list(model_cfg.search_space.max_depth.choices),
        )
        min_samples_split = trial.suggest_int(
            "min_samples_split",
            int(model_cfg.search_space.min_samples_split.low),
            int(model_cfg.search_space.min_samples_split.high),
        )
        min_samples_leaf = trial.suggest_int(
            "min_samples_leaf",
            int(model_cfg.search_space.min_samples_leaf.low),
            int(model_cfg.search_space.min_samples_leaf.high),
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=int(model_cfg.params.n_jobs),
        )
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }
        return model, params

    raise ValueError(f"Unknown model type: {model_type}")


def evaluate_metric(metric_name: str, y_true, y_pred, y_proba):
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, y_proba))
    if metric_name == "f1":
        return float(f1_score(y_true, y_pred))
    raise ValueError(f"Unsupported metric: {metric_name}")


def make_sampler(name: str, seed: int):
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "grid":
        raise ValueError("Grid sampler is not implemented in this version.")
    raise ValueError(f"Unknown sampler: {name}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train_full, X_test, y_train_full, y_test = load_data(
        cfg.paths.train_csv,
        cfg.paths.test_csv,
        cfg.target_col,
    )

    sampler = make_sampler(cfg.hpo.sampler, int(cfg.seed))
    study = optuna.create_study(
        direction=str(cfg.metric.direction),
        sampler=sampler,
    )

    with mlflow.start_run(run_name=f"hpo_{cfg.model.type}_{cfg.hpo.sampler}") as parent_run:
        mlflow.log_params(
            {
                "model_type": cfg.model.type,
                "sampler": cfg.hpo.sampler,
                "n_trials": int(cfg.hpo.n_trials),
                "metric_name": cfg.metric.name,
                "seed": int(cfg.seed),
            }
        )

        resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
        Path("reports").mkdir(parents=True, exist_ok=True)
        config_dump_path = Path("reports") / f"resolved_config_lab3_{cfg.hpo.sampler}.json"
        config_dump_path.write_text(
            json.dumps(resolved_cfg, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        mlflow.log_artifact(str(config_dump_path), artifact_path="config")

        def objective(trial: optuna.Trial):
            model, params = build_model(cfg.model, trial, int(cfg.seed))
            preprocessor = build_preprocessor()

            pipeline = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", model),
                ]
            )

            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                nested=True,
            ):
                mlflow.log_params(params)
                mlflow.set_tags(
                    {
                        "trial_number": trial.number,
                        "sampler": cfg.hpo.sampler,
                        "model_type": cfg.model.type,
                    }
                )

                if bool(cfg.hpo.use_cv):
                    cv = StratifiedKFold(
                        n_splits=int(cfg.hpo.cv_folds),
                        shuffle=True,
                        random_state=int(cfg.seed),
                    )

                    scoring = "roc_auc" if cfg.metric.name == "roc_auc" else "f1"
                    scores = cross_val_score(
                        pipeline,
                        X_train_full,
                        y_train_full,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=1,
                    )
                    score = float(scores.mean())
                    mlflow.log_metric("cv_score", score)
                    return score

                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full,
                    y_train_full,
                    test_size=float(cfg.split.val_size),
                    random_state=int(cfg.seed),
                    stratify=y_train_full if bool(cfg.split.stratify) else None,
                )

                pipeline.fit(X_train, y_train)

                y_pred = pipeline.predict(X_val)
                y_proba = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else None

                score = evaluate_metric(cfg.metric.name, y_val, y_pred, y_proba)
                mlflow.log_metric("val_score", score)
                return score

        study.optimize(objective, n_trials=int(cfg.hpo.n_trials))

        best_params = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metric("best_value", float(study.best_value))

        final_model, _ = build_model(
            OmegaConf.create(
                {
                    "type": cfg.model.type,
                    "params": cfg.model.params,
                    "search_space": cfg.model.search_space,
                }
            ),
            optuna.trial.FixedTrial(best_params),
            int(cfg.seed),
        )

        final_pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", final_model),
            ]
        )

        final_pipeline.fit(X_train_full, y_train_full)

        y_test_pred = final_pipeline.predict(X_test)
        y_test_proba = final_pipeline.predict_proba(X_test)[:, 1] if hasattr(final_pipeline, "predict_proba") else None
        test_score = evaluate_metric(cfg.metric.name, y_test, y_test_pred, y_test_proba)

        mlflow.log_metric("final_test_score", float(test_score))

        best_params_path = Path("models") / f"best_params_lab3_{cfg.hpo.sampler}.json"
        best_model_path = Path("models") / f"best_model_lab3_{cfg.hpo.sampler}.joblib"

        best_params_path.parent.mkdir(parents=True, exist_ok=True)
        best_model_path.parent.mkdir(parents=True, exist_ok=True)

        best_params_path.write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding="utf-8")
        joblib.dump(final_pipeline, best_model_path)

        mlflow.log_artifact(str(best_params_path), artifact_path="artifacts")
        mlflow.log_artifact(str(best_model_path), artifact_path="artifacts")
        mlflow.sklearn.log_model(final_pipeline, name="model")

        print("Best params:", best_params)
        print("Best value:", study.best_value)
        print("Final test score:", test_score)


if __name__ == "__main__":
    main()