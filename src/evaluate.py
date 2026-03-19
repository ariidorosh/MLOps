from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = None

    return metrics


def save_metrics_json(metrics: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    clean_metrics = {}
    for key, value in metrics.items():
        if value is None:
            clean_metrics[key] = None
        else:
            clean_metrics[key] = float(value)

    out_path.write_text(
        json.dumps(clean_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots()
    disp.plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_classification_report_text(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, digits=4, zero_division=0)


def _get_feature_names(preprocessor) -> np.ndarray:
    try:
        names = preprocessor.get_feature_names_out()
        return np.array(names, dtype=str)
    except Exception:
        return np.array([], dtype=str)


def save_feature_importance(pipeline, out_path: Path, top_k: int = 20) -> bool:
    """
    Логуємо важливість ознак:
    - RandomForest: feature_importances_
    - LogisticRegression: abs(coef_)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = pipeline.named_steps.get("model")
    preprocessor = pipeline.named_steps.get("preprocess")
    if model is None or preprocessor is None:
        return False

    feat_names = _get_feature_names(preprocessor)
    if feat_names.size == 0:
        return False

    importances = None

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
        title = "Feature Importance (RandomForest)"
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float).ravel()
        importances = np.abs(coef)
        title = "Feature Importance (LogReg |coef|)"

    if importances is None or importances.size != feat_names.size:
        return False

    idx = np.argsort(importances)[::-1][:top_k]
    top_names = feat_names[idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names[::-1], top_vals[::-1])
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True
