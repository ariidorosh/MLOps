# MLOps Lab 1 — Telco Customer Churn (MLflow Tracking)

## Goal
Build a clean ML project structure and run reproducible training experiments with MLflow tracking:
- data loading and preprocessing
- baseline model training
- multiple experiments with different hyperparameters
- logging parameters, metrics, and artifacts to MLflow

## Dataset
**Telco Customer Churn** (classification)

Place the dataset file here:
- `data/raw/telco_churn.csv`

> Note: `data/` is ignored by git on purpose (MLOps practice).  
> Download the dataset from Kaggle and save it with the exact name above.

## Project structure
- `data/raw/` — raw dataset (not tracked by git)
- `data/processed/` — processed data (not tracked by git)
- `notebooks/` — EDA notebook (`01_eda.ipynb`)
- `src/` — training pipeline code
- `models/` — saved models (not tracked by git)
- `mlruns/` — MLflow local tracking folder (created automatically, not tracked by git)
- `reports/figures/` — generated plots/artifacts (optional)

## Setup (Windows / PyCharm)
Create and activate a virtual environment (already done if you use PyCharm venv), then install dependencies:

```bash
pip install -r requirements.txt


````markdown
## Run training
```bash
python -m src.train
````

## Run MLflow UI

After running at least one experiment:

```bash
mlflow ui
```

Open in browser:

* [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Experiments

At least 5 runs will be logged to MLflow with different model parameters.
Use the MLflow UI to compare runs and select the best configuration.

## Notes

This repository is the base for the next labs (data versioning, pipelines, hyperparameter tuning, CI/CD).
